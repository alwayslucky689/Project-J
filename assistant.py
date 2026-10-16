import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import threading
from debug import debug_log as dbg
from memory.classifier import needs_memory_lookup
from tools.memory_tools import SEARCH_MEMORY_SCHEMA, search_memory
from dataclasses import dataclass
from memory import topics as topics_mod
from tts.tts_manager import speak_async
from config import settings
from core.personality import PersonalityManager
from core.prompts import build_tool_schema, COMMAND_JSON_RULES
from core.llm import ask_ollama, ask_ollama_streaming, route_to_tool
from core.registry import get_tool
from tools import personality_tools
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
from tools import (youtube, spotify, discord, ollama, ookla,
                   personality_tools, tts_tools, memory_tools)
from core.cancel import (
    CancelToken, set_current, get_current,
    start_listener as start_cancel_listener,
    stop_listener as stop_cancel_listener,
)
# Explicit "save this fact" phrasing. Matches only when a save verb is
# followed by a clause-introducer ("that", "this", "to", "I", ...).
# Bare "remember X" is ambiguous ("remember the pizza thing" could be
# a query) and falls through to the normal router.
_EXPLICIT_SAVE = re.compile(
    r"^\s*(?:remember|note|save|keep\s+in\s+mind|"
    r"don'?t\s+forget)\s+"
    r"(?:that\s+|this\s+|to\s+|i\s+|i'm\s+|i\s+am\s+)"
    r"(.+)$",
    re.IGNORECASE,
)

_TOOL_PATH_HINT = (
    "The user's query has been flagged as likely relevant to their stored "
    "memory. You have a `search_memory` tool available. Call it BEFORE "
    "asking for clarification, before saying you don't know, and before "
    "answering from training data. Memory is more likely to contain the "
    "answer than your general knowledge. If the tool returns nothing "
    "relevant, then ask for clarification. Do not ask the user to "
    "elaborate until you have checked memory first."
)
def _try_explicit_save(query: str):
    """
    Short-circuit explicit save phrases straight to remember_fact.
    Needle is unreliable when the candidate set is two tools; don't
    route an unambiguous save through it.
    """
    m = _EXPLICIT_SAVE.match(query)
    if not m:
        return None
    fact = m.group(1).strip().rstrip(".!?")
    if not fact:
        return None
    result = execute_single_action({"tool": "remember_fact", "fact": fact})
    return AssistantResponse(text=result, was_streamed=False)

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
            "  python assistant.py --backend bonsai    # force Bonsai backend for this run\n"
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
    p.add_argument(
        "--backend",
        choices=["ollama", "bonsai"],
        help="Override CHAT_BACKEND for this run",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Echo debug log lines to console as they're written",
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

    if search_m and not ref_m:
        term = search_m.group(1).strip()
        if term:
            return {"tool": "search_youtube", "query": term}

    if search_m and ref_m:
        term = search_m.group(1).strip()
        if term:
            actions.append({"tool": "search_youtube", "query": term})

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
    was_streamed: bool = False
    speak: bool = True
    was_spoken: bool = False


import re as _re

_TOPIC_CMD = _re.compile(r"^\s*topic\b\s*(.*)$", _re.IGNORECASE)


def _try_topic_command(user_input: str):
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
    arg_lower = arg.lower()

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

    if arg_lower.startswith("suggest "):
        name = arg[8:].strip()
        if not name:
            return AssistantResponse(text="Usage: topic suggest <string>")
        return _render_suggestions(name)

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
        return AssistantResponse(
            text=(f"No topic named '{name}'. "
                  f"Create it with: topic new {name}")
        )
    verb = "Started" if is_new else "Switched to"
    return AssistantResponse(
        text=f"📂 {verb} topic '{topic['name']}' (session {sid})."
    )


def _render_suggestions(query: str) -> AssistantResponse:
    from memory.activation import suggest, FUZZY_THRESHOLD
    scored = suggest(query)
    if not scored:
        return AssistantResponse(text="No topics exist yet.")
    lines = [f"Suggestions for {query!r}:"]
    for topic, score in scored[:5]:
        marker = "▶" if score >= FUZZY_THRESHOLD else " "
        aliases = topic.get("aliases") or []
        alias_str = f"  (aliases: {', '.join(aliases)})" if aliases else ""
        lines.append(f"  {marker} {score:.2f}  {topic['name']}{alias_str}")
    return AssistantResponse(text="\n".join(lines))


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


conversation = get_manager(
    system_prompt_fn=lambda q: personality_mgr.get_persona_prompt(),
    max_turns=10,
)
personality_tools.init_personality_tools(personality_mgr)
from tts.tts_manager import set_voice_provider
set_voice_provider(lambda: personality_mgr.get_voice_sample())
audio_pipeline = None  # type: AudioPipeline | None


def _reload_buffer_for_session(sid: int):
    loaded = conversation.load_from_db(sid, limit=20)
    if loaded:
        print(f"   ↻ Loaded {loaded} turns from DB for session {sid}")


# ===== System Prompt Builder =====
def build_system_prompt(user_input: str) -> str:
    """Full prompt for the fast model on the tool/command path."""
    persona = personality_mgr.get_persona_prompt()
    facts = get_facts_context(user_input)
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


def _try_nl_topic_activation(user_input: str):
    from memory import activation as act_mod

    outcome = act_mod.try_activate(user_input)
    if outcome is None:
        return None

    if outcome["action"] == "activated":
        topic = outcome["topic"]
        verb = "Started" if outcome["is_new"] else "Switched to"
        response = AssistantResponse(
            text=f"📂 {verb} topic '{topic['name']}' "
                 f"(session {outcome['session_id']})."
        )
        _persist_turn("user", user_input)
        _persist_turn("assistant", response.text)

        from memory.sessions import get_current_session_id
        sid = get_current_session_id()
        if sid is not None:
            _reload_buffer_for_session(sid)
        return response

    names = [t["name"] for t in outcome["options"]]
    if len(names) == 2:
        options_str = f"{names[0]} or {names[1]}"
    else:
        options_str = ", ".join(names[:-1]) + f", or {names[-1]}"
    response = AssistantResponse(
        text=f"Which did you mean: {options_str}?"
    )
    _persist_turn("user", user_input)
    _persist_turn("assistant", response.text)
    return response



from memory.sessions import get_or_create_session


# ===== Command Router =====
def route_request(user_input: str) -> AssistantResponse:
    stripped = user_input.strip().lower()
    if stripped in ("stop", "cancel", "shut up"):
        tok = get_current()
        if tok is not None and not tok.is_cancelled():
            tok.cancel()
            return AssistantResponse(text="Cancelled.", speak=False)
        return AssistantResponse(text="Nothing to cancel.", speak=False)

    set_current(None)

    t0 = time.perf_counter()
    sid = get_or_create_session()
    dbg.turn_start(user_input)

    topic_response = _try_topic_command(user_input)
    if topic_response is not None:
        dbg.route("topic_command", (time.perf_counter() - t0) * 1000)
        return topic_response

    nl_topic = _try_nl_topic_activation(user_input)
    if nl_topic is not None:
        dbg.route("nl_topic", (time.perf_counter() - t0) * 1000)
        return nl_topic

    detected = personality_mgr.detect_personality(user_input)
    if detected and personality_mgr.switch_to(detected):
        for word in personality_mgr.get_wake_words(detected):
            user_input = user_input.lower().replace(word.lower(), "").strip()
    if detected != "None" and detected != None:
        print(f" Switched personality to: {detected}")

    if not user_input:
        resp = f"{personality_mgr.get_display_name()} active. How can I help?"
        _persist_turn("assistant", resp, sid)
        return AssistantResponse(text=resp)

    ref = _try_youtube_ref(user_input)
    if ref is not None:
        result = execute_tool(ref)
        dbg.route("youtube_short_circuit", (time.perf_counter() - t0) * 1000)
        _persist_turn("user", user_input, sid)
        _persist_turn("assistant", result, sid)
        dbg.assistant(result)
        dbg.turn_end((time.perf_counter() - t0) * 1000)
        return AssistantResponse(text=result, was_streamed=False)
    save_response = _try_explicit_save(user_input)
    if save_response is not None:
        dbg.route("explicit_save", (time.perf_counter() - t0) * 1000)
        _persist_turn("user", user_input, sid)
        _persist_turn("assistant", save_response.text, sid)
        dbg.assistant(save_response.text)
        dbg.turn_end((time.perf_counter() - t0) * 1000)
        return save_response
    if settings.USE_NEEDLE_ROUTER:
        t_route = time.perf_counter()
        decision = route_to_tool(user_input)
        route_ms = (time.perf_counter() - t_route) * 1000
        _ts("needle", t_route)
    else:
        t_route = time.perf_counter()
        router_prompt = build_system_prompt(user_input)
        decision = ask_ollama(router_prompt, is_json=True, model=settings.FAST_MODEL)
        route_ms = (time.perf_counter() - t_route) * 1000
        _ts("llama router", t_route)

    tool_name = decision.get("tool") if isinstance(decision, dict) else None
    conf = decision.get("_confidence") if isinstance(decision, dict) else None
    candidates = decision.get("_candidates") if isinstance(decision, dict) else None
    dbg.route(tool_name or "?", route_ms, conf, candidates)

    if not isinstance(decision, dict) or decision.get("tool") == "chat":
        dbg.event("ROUTE", f"chat: user={user_input!r} "
                           f"router_msg={(decision.get('message') if isinstance(decision, dict) else None)!r}")
        response, was_streamed = _handle_chat_streaming(user_input)
        return AssistantResponse(
            text=response,
            was_streamed=was_streamed,
        )

    t_tool = time.perf_counter()
    result = execute_tool(decision)
    _persist_turn("user", user_input, sid)
    _persist_turn("assistant", result, sid)
    dbg.assistant(result)
    dbg.turn_end((time.perf_counter() - t0) * 1000)
    return AssistantResponse(text=result, was_streamed=False)


def _build_chat_prompt(user_input: str) -> str:
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
    from core.llm import chat_stream_tokens, chat_stream_tokens_openai
    from memory.retrieval import assemble_memory

    sid = get_or_create_session()
    tts_active = settings.ENABLE_TTS and not is_muted()

    t_turn = time.perf_counter()

    # Time retrieval, then log it
    t_mem = time.perf_counter()
    messages = conversation.get_messages(sid, current_user_input=question)
    try:
        from memory.retrieval import baseline_memory
        dbg.memory(baseline_memory(sid))
    except Exception as e:
        dbg.error(f"baseline load failed: {e}")
        dbg.memory([])

    # Memory tool path
    if needs_memory_lookup(question):
        memory_answer = _try_memory_tool_path(messages, sid)
        if memory_answer is not None:
            conversation.add_user(sid, question)
            conversation.add_assistant(sid, memory_answer)
            _persist_turn("user", question, sid)
            _persist_turn("assistant", memory_answer, sid)
            dbg.assistant(memory_answer)
            dbg.turn_end((time.perf_counter() - t_turn) * 1000)
            return memory_answer, False

    # Streaming path
    token = CancelToken()
    set_current(token)

    streamer = None
    if tts_active:
        from tts.streaming_tts import StreamingTTS
        streamer = StreamingTTS()
        streamer.start()
        token.register_streamer(streamer)

    backend = getattr(settings, "CHAT_BACKEND", "ollama")
    model_name = (getattr(settings, "BONSAI_MODEL", "?")
                  if backend == "bonsai"
                  else getattr(settings, "REASONING_MODEL", "?"))
    dbg.llm_start(backend, model_name)

    if backend == "bonsai":
        gen = chat_stream_tokens_openai(messages, model=settings.REASONING_MODEL)
    else:
        gen = chat_stream_tokens(messages, model=settings.REASONING_MODEL)

    print("🤖 ", end="", flush=True)
    full = ""
    token_count = 0
    cancelled = False
    t_llm = time.perf_counter()
    t_first = None

    try:
        for tok in gen:
            if token.is_cancelled():
                cancelled = True
                break

            if t_first is None:
                t_first = time.perf_counter()
                dbg.llm_first_token((t_first - t_llm) * 1000)

            print(tok, end="", flush=True)
            if streamer is not None:
                streamer.push_token(tok)
            full += tok
            token_count += 1
    finally:
        if full:
            print()
        t_llm_done = time.perf_counter()
        dbg.llm_done(token_count, (t_llm_done - t_llm) * 1000)

        if streamer is not None:
            t_tts = time.perf_counter()
            dbg.tts_start()
            try:
                streamer.finish()
            except Exception:
                pass
            dbg.tts_done((time.perf_counter() - t_tts) * 1000)
        set_current(None)

    conversation.add_user(sid, question)
    if full:
        conversation.add_assistant(sid, full)
    _persist_turn("user", question, sid)
    if full:
        _persist_turn("assistant", full, sid, excluded=cancelled)

    dbg.assistant(full)
    dbg.turn_end((time.perf_counter() - t_turn) * 1000)
    return full, True


def _try_memory_tool_path(messages: list, sid: int) -> str | None:
    from core.llm import chat_with_tools, append_tool_exchange
    import json as _json

    backend = getattr(settings, "CHAT_BACKEND", "ollama")
    model = (getattr(settings, "BONSAI_MODEL", None) if backend == "bonsai"
             else settings.REASONING_MODEL)

    dbg.event("7A", f"enter: backend={backend} model={model} "
                    f"msgs={len(messages)}")

    # First call: model decides whether to use the tool. Longer timeout
    # because the model may emit a short preamble before the tool call.
    first_call_messages = messages
    if messages and messages[0].get("role") == "system":
        sys_with_hint = {
            "role": "system",
            "content": messages[0]["content"] + "\n\n" + _TOOL_PATH_HINT,
        }
        first_call_messages = [sys_with_hint] + messages[1:]
    else:
        first_call_messages = (
            [{"role": "system", "content": _TOOL_PATH_HINT}] + messages
        )

    result = chat_with_tools(
        first_call_messages,
        tools=[SEARCH_MEMORY_SCHEMA],
        model=model,
        timeout=(10, 45),
    )

    if result.get("raw", {}).get("error"):
        dbg.event("7A", f"first call failed: {result['raw']['error']}")
        return None

    tool_calls = result.get("tool_calls") or []
    content = result.get("content", "")
    dbg.event("7A", f"first call: tool_calls={len(tool_calls)} "
                    f"content_len={len(content)}")

    if not tool_calls:
        # Model answered directly. Use it.
        return content.strip() or None

    # Execute each tool call and append the exchange.
    messages_with_tool = messages
    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("arguments") or {}
        dbg.event("7A", f"call: {name}({args})")

        if name != "search_memory":
            tool_result = f"(unknown tool: {name})"
        else:
            query = args.get("query") or ""
            try:
                tool_result = search_memory(query)
            except Exception as e:
                tool_result = f"(tool error: {e})"

        preview = tool_result[:120].replace("\n", " ")
        dbg.event("7A", f"result: {preview}")

        messages_with_tool = append_tool_exchange(
            messages_with_tool, tc, tool_result
        )

    # Second call: model answers using the tool results.
    #
    # IMPORTANT: do NOT send `tools` here. The tool result is in the
    # message stream; offering the tool again invites the model to call
    # it a second time, which produces an empty `content` and hangs the
    # assistant. Also bound the timeout tightly — a stuck follow-up
    # blocks the CLI.
    follow_up = chat_with_tools(
        messages_with_tool,
        tools=[],
        model=model,
        timeout=(10, 30),
    )

    if follow_up.get("raw", {}).get("error"):
        dbg.event("7A", f"follow-up failed: {follow_up['raw']['error']}")
        return None

    final = (follow_up.get("content") or "").strip()
    dbg.event("7A", f"follow-up: content_len={len(final)}")

    if not final:
        return None

    messages[:] = messages_with_tool
    return final

def _persist_turn(role: str, text: str, session_id=None, excluded: bool = False):
    """Write a turn. If excluded=True, mark it to skip future extraction."""
    if not text:
        return None
    try:
        from memory.sessions import append_user_turn, append_assistant_turn
        from memory.store import write_transaction
        from memory.worker import notify_activity
        if role == "user":
            turn_id = append_user_turn(text, session_id=session_id)
        else:
            turn_id = append_assistant_turn(text, session_id=session_id)
        if excluded and turn_id is not None:
            with write_transaction() as conn:
                conn.execute(
                    "UPDATE turns SET excluded_from_extraction = 1 WHERE id = ?",
                    (turn_id,),
                )
        notify_activity()
        return turn_id
    except Exception as e:
        print(f"⚠️ Could not persist turn: {e}")
        return None


# ===== Main =====
def main():
    global audio_pipeline
    args = parse_args()

    if args.tts_off:
        settings.ENABLE_TTS = False

    from memory.sessions import get_or_create_session, close_session
    sid = get_or_create_session()
    loaded = conversation.load_from_db(sid, limit=20)
    if loaded:
        print(f"📖 Resumed session {sid} with {loaded} recent turns.")
    start_worker()
    dbg.init(session_id=sid, debug=getattr(args, "debug", False))
    if (getattr(settings, "CHAT_BACKEND", "ollama") == "bonsai"
            and getattr(settings, "BONSAI_AUTOSTART", True)):
        from bonsai.bonsai_server import start as _start_bonsai
        _start_bonsai()
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
    print("Press Esc during a response to cancel it.")
    print("Type 'quit' to exit.")
    print("-" * 60 + "\n")
    from memory.store import backfill_fact_embeddings
    backfill_fact_embeddings()
    start_cancel_listener()
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
        if user_input.lower() in ("stop", "cancel", "shut up"):
            # Between turns, no streaming is active, so this is basically
            # a no-op acknowledgement. Still useful as a reset point.
            tok = get_current()
            if tok is not None and not tok.is_cancelled():
                tok.cancel()
                print("⏹️  Cancelled.")
            else:
                print("Nothing to cancel.")
            continue
        try:
            response = route_request(user_input)
            handle_response(response)
        except Exception as e:
            print(f"❌ Error: {e}")

    stop_worker()
    stop_cancel_listener()
    from memory.sessions import close_session as _close, get_current_session_id
    sid = get_current_session_id()
    _close(sid)
    if sid is not None:
        conversation.discard(sid)
    from bonsai.bonsai_server import stop as _stop_bonsai
    _stop_bonsai()
    if audio_pipeline:
        audio_pipeline.stop()
    dbg.shutdown()
    print(" Goodbye!")


if __name__ == "__main__":
    main()