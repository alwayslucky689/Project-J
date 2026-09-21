# assistant.py - Merged: wakeword + parakeet STT + blocking TTS
# Phase 0 cleanup: single return contract, blocking TTS, voice pipeline
import threading
from dataclasses import dataclass

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


# ===== Initialization =====
personality_mgr = PersonalityManager()
personality_tools.init_personality_tools(personality_mgr)

audio_pipeline = None  # type: AudioPipeline | None


# ===== System Prompt Builder =====
def build_system_prompt(user_input: str) -> str:
    """Full prompt for the fast model on the tool/command path."""
    persona = personality_mgr.get_persona_prompt()
    facts = get_facts_context()
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


# ===== Command Router =====
def route_request(user_input: str) -> AssistantResponse:
    t0 = time.perf_counter()

    # --- Personality wake-word check ---
    detected = personality_mgr.detect_personality(user_input)
    if detected and personality_mgr.switch_to(detected):
        ...
    _ts("routing", t0)

    ref = _try_youtube_ref(user_input)
    if ref is not None:
        result = execute_tool(ref)
        _ts("short-circuit + tool", t0)
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
    _ts("tool exec", t_tool)
    _ts("total", t0)
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
    """
    Chat path with streaming TTS.

    Streams LLM tokens into a StreamingTTS instance so playback starts at
    the first sentence boundary. Returns (full_text, was_spoken).

    Falls back to plain streaming (no TTS) when TTS is muted or disabled.
    """
    prompt = _build_chat_prompt(question)

    # Skip streaming TTS entirely if it wouldn't speak anyway.
    from tts.tts_manager import is_muted
    tts_active = settings.ENABLE_TTS and not is_muted()

    if not tts_active:
        full = ask_ollama_streaming(prompt, model=settings.REASONING_MODEL)
        return full, False

    from tts.streaming_tts import StreamingTTS

    streamer = StreamingTTS()
    streamer.start()

    try:
        full = ask_ollama_streaming(
            prompt,
            model=settings.REASONING_MODEL,
            on_token=streamer.push_token,
        )
    finally:
        # Blocks until every queued sentence has been synthesized AND played.
        streamer.finish()

    return full, True
# ===== Main =====
def main():
    global audio_pipeline

    print("=" * 60)
    print("🤖 Project-J AI Assistant")
    print("=" * 60)
    print(f"Personality: {personality_mgr.get_current_name()}")

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
    print("🎤 Voice mode — say 'Jarvis' to wake.")
    print("Type 'quit' to exit.")
    print("-" * 60 + "\n")

    while True:
        try:
            user_input = input("⌨️  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "bye"]:
            break

        try:
            response = route_request(user_input)
            handle_response(response)
        except Exception as e:
            print(f"❌ Error: {e}")

    if audio_pipeline:
        audio_pipeline.stop()
    print("👋 Goodbye!")


if __name__ == "__main__":
    main()