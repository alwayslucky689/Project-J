import threading
from dataclasses import dataclass
from memory import topics as topics_mod
from tts.tts_manager import speak_async
from config import settings
from core.personality import PersonalityManager
from core.prompts import build_tool_schema, COMMAND_JSON_RULES
from core.llm import ask_ollama, ask_ollama_streaming, route_to_tool
from core.registry import get_tool
from tools import youtube, spotify, discord, ollama, ookla, personality_tools, tts_tools
from memory.fact_memory import get_facts_context, save_fact
from audio_pipeline import AudioPipeline
from config.paths import STT_VENV_PYTHON, STT_SERVICE_SCRIPT
import re
import time
from config import settings as _settings
from tts.tts_manager import generate_chunk_for_streaming, play_chunk_blocking
from core.llm import chat_streaming
import argparse
from core.conversation import get_manager
from memory.fact_memory import get_facts_context
from memory.sessions import (
    start_session, close_session, append_user_turn, append_assistant_turn,
)
from memory.worker import start_worker, stop_worker, notify_activity

def _build_chat_system_prompt(query: str) -> str:
    """
    System prompt for the chat path.

    Uses only the persona prompt. The chat_prompt YAML field is legacy —
    it was designed for the old /api/generate flow where the whole
    conversation was one flat string. In the /api/chat flow, the user's
    message is a separate turn, not embedded in the system prompt.
    """
    return personality_mgr.get_persona_prompt()

conversation = get_manager(
    system_prompt_fn=_build_chat_system_prompt,
    max_turns=10,
)

#flags to run
def parse_args():
    p = argparse.ArgumentParser(
        description="Project-J voice assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python assistant.py                     # full voice mode\n"
            "  python assistant.py --no-stt            # text input only, skip Parakeet\n"
            "  python assistant.py --no-tts            # voice input, silent responses\n"
            "  python assistant.py --no-stt --no-tts   # pure text REPL\n"
        ),
    )
    p.add_argument(
        "--no-stt", "--stt-off",
        dest="stt_off",
        action="store_true",
        help="Skip starting the STT service (no wake word or voice input)",
    )
    p.add_argument(
        "--no-tts", "--tts-off",
        dest="tts_off",
        action="store_true",
        help="Start with TTS disabled (silent responses)",
    )
    return p.parse_args()

def _ts(label: str, t0: float):
    """Print a labeled timestamp delta if TIMING is enabled."""
    if getattr(_settings, "TIMING", False):
        print(f"⏱️  [{label}] {(time.perf_counter() - t0) * 1000:.0f}ms")

_YOUTUBE_SEARCH = re.compile(
    r"\bsearch\s+(?:for\s+)?(.+?)"
    r"(?:\s+and\s+|\s+then\s+|\s+on\s+youtube\s+and\s+|$)",
    re.I,
)
_YOUTUBE_REF = re.compile(
    r"\b(?:play|show)\s+(?:the\s+)?(?:video\s+|result\s+)?"
    r"(\d+|first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\b",
    re.I,
)

_WORD_NUM = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
}

def _try_youtube_ref(query: str):
    """
    Short-circuit YouTube search + play commands.

    Handles three shapes:
      "play video 1"                       → play_youtube_video
      "search X"                           → search_youtube
      "search X and play video N"          → [search_youtube, play_youtube_video]

    Returns a dict, a list of dicts, or None.
    """
    actions = []

    ref_m = _YOUTUBE_REF.search(query)
    search_m = _YOUTUBE_SEARCH.search(query)

    # Only fire the search branch for bare "search X" queries (no play word)
    if search_m and not ref_m:
        term = search_m.group(1).strip()
        if term:
            return {"tool": "search_youtube", "query": term}

    # Compound: search + play in the same input
    if search_m and ref_m:
        term = search_m.group(1).strip()
        if term:
            actions.append({"tool": "search_youtube", "query": term})

    # Play branch (single or as second step of compound)
    if ref_m:
        tok = ref_m.group(1).lower()
        if tok.isdigit():
            idx = int(tok)
        elif tok in _WORD_NUM:
            idx = _WORD_NUM[tok]
        else:
            idx = None
        if idx is not None:
            actions.append({"tool": "play_youtube_video", "index": idx})

    if not actions:
        return None
    return actions[0] if len(actions) == 1 else actions

# ===== Response contract =====
@dataclass
class AssistantResponse:
    text: str = ""
    was_streamed: bool = False   # True if already printed during streaming
    speak: bool = True           # Whether the caller should TTS this
    was_spoken: bool = False

import re as _re

_TOPIC_CMD = _re.compile(r"^\s*topic\b\s*(.*)$", _re.IGNORECASE)

def _try_topic_command(user_input: str):
    """
    Outer wrapper. Matches the 'topic' prefix, passes the rest to the
    dispatcher, and handles persistence + buffer reload after a switch.
    Returns an AssistantResponse if handled, else None.
    """
    m = _TOPIC_CMD.match(user_input)
    if not m:
        return None

    response = _dispatch_topic_command(m.group(1).strip())
    if response is None:
        return None

    _persist_turn("user", user_input)
    _persist_turn("assistant", response.text)

    from memory.sessions import get_current_session_id
    sid = get_current_session_id()
    if sid is not None:
        _reload_buffer_for_session(sid)

    return response


def _dispatch_topic_command(arg: str):
    """
    arg is everything after 'topic', already stripped.
    Examples: '', 'list', 'off', 'new spanish', 'delete project-j', 'spanish'
    """
    arg_lower = arg.lower()

    # Bare "topic" or "topic list"
    if not arg or arg_lower == "list":
        return _render_topic_list()

    if arg_lower in ("off", "global", "none"):
        sid, _ = topics_mod.leave_topic()
        return AssistantResponse(
            text=f"Returned to global context (session {sid})."
        )

    if arg_lower.startswith("new "):
        name = arg[4:].strip()
        if not name:
            return AssistantResponse(text="Usage: topic new <name>")
        return _create_and_switch(name)

    if arg_lower.startswith("delete "):
        name = arg[7:].strip()
        if not name:
            return AssistantResponse(text="Usage: topic delete <name>")
        return AssistantResponse(text=topics_mod.delete_topic(name))

    return _switch_to_topic(arg)


def _create_and_switch(name: str) -> AssistantResponse:
    try:
        topic = topics_mod.create_topic(name)
    except ValueError as e:
        return AssistantResponse(text=f"❌ {e}")

    topic_dict, sid, is_new = topics_mod.activate_topic(topic["id"])
    verb = "Started" if is_new else "Switched to"
    return AssistantResponse(
        text=f"📂 {verb} topic '{topic['name']}' (session {sid})."
    )


def _switch_to_topic(name: str) -> AssistantResponse:
    topic, sid, is_new = topics_mod.activate_topic(name)
    if topic is None:
        # Offer to create it
        return AssistantResponse(
            text=(f"No topic named '{name}'. "
                  f"Create it with: topic new {name}")
        )
    verb = "Started" if is_new else "Switched to"
    return AssistantResponse(
        text=f"📂 {verb} topic '{topic['name']}' (session {sid})."
    )


def _render_topic_list() -> AssistantResponse:
    topics = topics_mod.list_topics()
    current = topics_mod.current_topic()

    lines = []
    if current:
        lines.append(f"Active: {current['name']}")
    else:
        lines.append("Active: (global)")

    if not topics:
        lines.append("")
        lines.append("No topics yet. Create one with: topic new <name>")
    else:
        lines.append("")
        lines.append(f"{len(topics)} topic(s):")
        for t in topics:
            marker = "▶" if current and t["id"] == current["id"] else " "
            alias_str = ""
            if t.get("aliases"):
                alias_str = f"  (aliases: {', '.join(t['aliases'])})"
            lines.append(f"  {marker} [{t['id']}] {t['name']}{alias_str}")

    return AssistantResponse(text="\n".join(lines))

# ===== Initialization =====
personality_mgr = PersonalityManager()


# system_prompt_fn is called fresh on every turn so personality switches
# and (future) memory injections take effect immediately.
conversation = get_manager(
    system_prompt_fn=lambda q: personality_mgr.get_persona_prompt(),
    max_turns=10,
)
personality_tools.init_personality_tools(personality_mgr)

audio_pipeline = None  # type: AudioPipeline | None

def _reload_buffer_for_session(sid: int):
    """Ensure the in-memory buffer for `sid` matches the DB.
    Called after any context switch so the assistant has the session's
    recent turns available."""
    loaded = conversation.load_from_db(sid, limit=20)
    if loaded:
        print(f"   ↻ Loaded {loaded} turns from DB for session {sid}")
# ===== System Prompt Builder =====
def build_system_prompt(user_input: str) -> str:
    """Full prompt for the fast model on the tool/command path."""
    persona = personality_mgr.get_persona_prompt()
    facts = get_facts_context(user_input)   # ← now takes the query
    tool_schema = build_tool_schema()
    rules = COMMAND_JSON_RULES.replace("{user_input}", user_input)
    parts = [persona, facts, tool_schema, rules]
    return "\n\n".join(p for p in parts if p)

# ===== TTS (blocking) =====
def _speak_safe(text):
    if not text or not settings.ENABLE_TTS:
        return
    t0 = time.perf_counter()
    done = threading.Event()

    def _on_complete():
        done.set()

    try:
        speak_async(text, on_complete=_on_complete)
        done.wait(timeout=60)
        if getattr(settings, "TIMING", False):
            print(f"⏱️  [tts total] {(time.perf_counter() - t0) * 1000:.0f}ms")
    except Exception as e:
        print(f"⚠️ TTS error: {e}")

from memory.sessions import get_or_create_session
# ===== Command Router =====
def route_request(user_input: str) -> AssistantResponse:
    t0 = time.perf_counter()
    topic_response = _try_topic_command(user_input)
    if topic_response is not None:
        return topic_response
    # --- Personality wake-word check ---
    detected = personality_mgr.detect_personality(user_input)
    if detected and personality_mgr.switch_to(detected):
        for word in personality_mgr.get_wake_words(detected):
            user_input = user_input.lower().replace(word.lower(), "").strip()
    if detected != "None" and detected != None:
        print(f" Switched personality to: {detected}")
    if not user_input:
        resp = f"{personality_mgr.get_display_name()} active. How can I help?"
        _persist_turn("assistant", resp,sid)
        return AssistantResponse(text=resp)
    sid = get_or_create_session()
    ref = _try_youtube_ref(user_input)
    if ref is not None:
        result = execute_tool(ref)
        _persist_turn("user", user_input,sid)
        _persist_turn("assistant", result,sid)
        return AssistantResponse(text=result, was_streamed=False)
    elif settings.USE_NEEDLE_ROUTER:
        t_route = time.perf_counter()
        decision = route_to_tool(user_input)
        _ts("needle", t_route)
    else:
        t_route = time.perf_counter()
        router_prompt = build_system_prompt(user_input)
        decision = ask_ollama(router_prompt, is_json=True, model=settings.FAST_MODEL)
        _ts("llama router", t_route)

    if not isinstance(decision, dict) or decision.get("tool") == "chat":
        question = user_input
        if isinstance(decision, dict):
            question = decision.get("message") or user_input
        response, spoken = _handle_chat_streaming(question)
        return AssistantResponse(
            text=response,
            was_streamed=True,
            was_spoken=spoken,
        )

    t_tool = time.perf_counter()
    result = execute_tool(decision)
    _persist_turn("user", user_input,sid)
    _persist_turn("assistant", result,sid)
    return AssistantResponse(text=result, was_streamed=False)
def _build_chat_prompt(user_input: str) -> str:
    """Persona + per-personality chat template, with user input filled in."""
    persona = personality_mgr.get_persona_prompt()
    chat = personality_mgr.get_chat_prompt().replace("{user_input}", user_input)
    return f"{persona}\n\n{chat}"
# ===== Tool Executor =====
def execute_tool(decision):
    if isinstance(decision, list):
        results = []
        for action in decision:
            result = execute_single_action(action)
            if result is not None:
                results.append(str(result))
        return "\n".join(results) if results else ""
    result = execute_single_action(decision)
    return str(result) if result is not None else ""


def execute_single_action(action):
    tool_name = action.get("tool")

    if tool_name == "error":
        return f"⚠️ AI Error: {action.get('message', 'unknown')}"

    tool = get_tool(tool_name)
    if tool is None:
        fallback = personality_mgr.render("unknown_tool")
        if fallback:
            return fallback
        return f"⚠️ Unknown tool: {tool_name}. AI said: {action}"

    kwargs = {k: v for k, v in action.items()
              if k != "tool" and not k.startswith("_")}

    try:
        result = tool.handler(**kwargs)
    except TypeError as e:
        return f"❌ Argument error for {tool_name}: {e}"
    except Exception as e:
        return f"❌ Error executing {tool_name}: {e}"

    # Failure check: if the handler returned False or None, use the failure
    # template instead of the success template. This is why "Now playing"
    # appeared after a Spotify connection error.
    if result is False or result is None:
        fail = personality_mgr.render("tool_failure")
        if fail:
            return fail
        return tool.format(result)

    rendered = personality_mgr.render(tool.response_key, **kwargs)
    if rendered is not None:
        return rendered

    return tool.format(result)

# ===== Response handler (shared by voice + text) =====
def handle_response(response: AssistantResponse):
    """Single place that prints (if needed) and speaks a response."""
    if response.text and not response.was_streamed:
        print(f"🤖 {response.text}")
    if (
        settings.ENABLE_TTS
        and response.speak
        and response.text
        and not response.was_spoken
    ):
        _speak_safe(response.text)

_EXIT_PHRASES = {
    "quit", "exit", "shut down", "shutdown",
    "goodbye", "good bye", "bye", "shut up"
}
# ===== Voice Callback =====
def on_transcription(text):
    """Called by audio_pipeline with recognized speech."""
    print(f"\n  Processing: {text}")

    if text.strip().lower().rstrip(".!?") in _EXIT_PHRASES:
        print("👋 Goodbye!")
        if settings.ENABLE_TTS:
            _speak_safe("Goodbye, Sir.")
        import os
        os._exit(0)

    if audio_pipeline:
        audio_pipeline.set_speaking(True)
    try:
        response = route_request(text)
        handle_response(response)
    finally:
        if audio_pipeline:
            audio_pipeline.set_speaking(False)

def _handle_chat_streaming(question: str) -> tuple[str, bool]:
    from memory.sessions import get_or_create_session
    from tts.tts_manager import is_muted

    sid = get_or_create_session()
    tts_active = settings.ENABLE_TTS and not is_muted()

    messages = conversation.get_messages(sid, current_user_input=question)

    if not tts_active:
        full = chat_streaming(messages, model=settings.REASONING_MODEL)
        conversation.add_user(sid, question)
        conversation.add_assistant(sid, full)
        _persist_turn("user", question,sid)
        _persist_turn("assistant", full,sid)
        return full, False

    from tts.streaming_tts import StreamingTTS
    streamer = StreamingTTS()
    streamer.start()

    try:
        full = chat_streaming(
            messages,
            model=settings.REASONING_MODEL,
            on_token=streamer.push_token,
        )
    finally:
        streamer.finish()

    conversation.add_user(sid, question)
    conversation.add_assistant(sid, full)
    _persist_turn("user", question,sid)
    _persist_turn("assistant", full,sid)
    return full, True

def _persist_turn(role: str, text: str, session_id = None):
    if not text:
        return
    try:
        from memory.sessions import append_user_turn, append_assistant_turn
        from memory.worker import notify_activity
        if role == "user":
            append_user_turn(text, session_id=session_id)
        else:
            append_assistant_turn(text, session_id=session_id)
        notify_activity()
    except Exception as e:
        print(f"⚠️ Could not persist turn: {e}")
# ===== Main =====
def main():
    global audio_pipeline
    args = parse_args()

    # Apply TTS override before anything reads settings.ENABLE_TTS
    if args.tts_off:
        settings.ENABLE_TTS = False
        # Start a memory session and the background extraction worker
    from memory.sessions import get_or_create_session, close_session
    sid = get_or_create_session()
    loaded = conversation.load_from_db(sid, limit=20)
    if loaded:
        print(f"📖 Resumed session {sid} with {loaded} recent turns.")
    start_worker()
    print("=" * 60)
    print("🤖 Project-J AI Assistant")
    print("=" * 60)
    print(f"Personality: {personality_mgr.get_current_name()}")
    current = topics_mod.current_topic()
    if current:
        print(f"Topic:       {current['name']}")
    else:
        print("Topic:       (global)")
    if args.stt_off:
        print("⌨️  STT disabled (--no-stt): text input only")
    if args.tts_off:
        print("🔇 TTS disabled (--no-tts): silent responses")

    if args.stt_off:
        print("⏭️  Skipping STT/Parakeet startup.")
        audio_pipeline = None
    else:
        try:
            audio_pipeline = AudioPipeline(
                on_transcription=on_transcription,
                stt_venv_python=STT_VENV_PYTHON,
                stt_service_script=STT_SERVICE_SCRIPT,
            )
            audio_pipeline.start()
        except Exception as e:
            print(f"⚠️ Could not start voice pipeline: {e}")
            import traceback
            traceback.print_exc()
            print("💡 Continuing with text input only.")
            audio_pipeline = None

    print("\n" + "-" * 60)
    print("💬 Text mode available — type commands directly.")
    if not args.stt_off:
        print("🎤 Voice mode — say 'Jarvis' to wake.")
    print("Type 'topic' for topics, 'topic new <name>' to create.")
    print("Type 'quit' to exit.")
    print("-" * 60 + "\n")
    from memory.store import backfill_fact_embeddings
    backfill_fact_embeddings()
    while True:
        try:
            user_input = input("⌨️  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n Goodbye")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "bye"]:
            break
        if user_input.lower() in ["new session", "clear", "reset conversation"]:
            conversation.clear()
            print(" Conversation cleared.")
            continue
        try:
            response = route_request(user_input)
            handle_response(response)
        except Exception as e:
            print(f"❌ Error: {e}")
    stop_worker()
    from memory.sessions import close_session as _close, get_current_session_id
    sid = get_current_session_id()
    _close(sid)
    if sid is not None:
        conversation.discard(sid)
    if audio_pipeline:
        audio_pipeline.stop()
    print(" Goodbye!")


if __name__ == "__main__":
    main()