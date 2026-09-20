# assistant.py - Merged: wakeword + parakeet STT + blocking TTS
# Phase 0 cleanup: single return contract, blocking TTS, voice pipeline
import threading
from dataclasses import dataclass

from tts.tts_manager import speak_async
from config import settings
from core.personality import PersonalityManager
from core.prompts import build_tool_schema, COMMAND_JSON_RULES
from core.llm import ask_ollama, ask_ollama_streaming
from core.registry import get_tool
from tools import youtube, spotify, discord, ollama, ookla, personality_tools, tts_tools
from memory.fact_memory import get_facts_context, save_fact
from audio_pipeline import AudioPipeline
from config.paths import STT_VENV_PYTHON, STT_SERVICE_SCRIPT

# ===== Response contract =====
@dataclass
class AssistantResponse:
    text: str = ""
    was_streamed: bool = False   # True if already printed during streaming
    speak: bool = True           # Whether the caller should TTS this


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
    """
    Speak and block until playback finishes.
    Does NOT toggle audio_pipeline.set_speaking — the caller owns that
    so we have a single toggle point around the whole request lifecycle.
    """
    if not text or not settings.ENABLE_TTS:
        return

    done = threading.Event()

    def _on_complete():
        done.set()

    try:
        speak_async(text, on_complete=_on_complete)
        done.wait(timeout=60)
    except Exception as e:
        print(f"⚠️ TTS error: {e}")


# ===== Command Router =====
def route_request(user_input: str) -> AssistantResponse:
    # --- Personality wake-word check ---
    detected = personality_mgr.detect_personality(user_input)
    if detected and personality_mgr.switch_to(detected):
        for word in personality_mgr.get_wake_words(detected):
            user_input = user_input.lower().replace(word.lower(), "").strip()
        print(f"🧠 Switched personality to: {detected}")
        if not user_input:
            return AssistantResponse(
                text=f"{personality_mgr.get_display_name()} active. How can I help?"
            )

    # --- Single routing pass: tool or chat? ---
    router_prompt = build_system_prompt(user_input)
    decision = ask_ollama(router_prompt, is_json=True, model=settings.FAST_MODEL)

    # The router signals "not a tool call" by returning ask_question.
    # In that case, route to the chat path with the reasoning model.
    if not isinstance(decision, dict) or decision.get("tool") == "ask_question":
        question = user_input
        if isinstance(decision, dict):
            question = decision.get("question") or user_input
        prompt = _build_chat_prompt(question)
        response = ask_ollama_streaming(prompt, model=settings.REASONING_MODEL)
        return AssistantResponse(text=response, was_streamed=True)

    # --- Tool path ---
    result = execute_tool(decision)
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
        # Personality-aware unknown-tool message
        fallback = personality_mgr.render("unknown_tool")
        if fallback:
            return fallback
        return f"⚠️ Unknown tool: {tool_name}. AI said: {action}"

    kwargs = {k: v for k, v in action.items() if k != "tool"}

    try:
        result = tool.handler(**kwargs)
    except TypeError as e:
        return f"❌ Argument error for {tool_name}: {e}"
    except Exception as e:
        return f"❌ Error executing {tool_name}: {e}"

    # Try personality template first — falls through to the tool's formatter
    # if the personality doesn't define this response_key.
    rendered = personality_mgr.render(tool.response_key, **kwargs)
    if rendered is not None:
        return rendered

    return tool.format(result)

# ===== Response handler (shared by voice + text) =====
def handle_response(response: AssistantResponse):
    """Single place that prints (if needed) and speaks a response."""
    if response.text and not response.was_streamed:
        print(f"🤖 {response.text}")
    if settings.ENABLE_TTS and response.speak and response.text:
        _speak_safe(response.text)


# ===== Voice Callback =====
def on_transcription(text):
    """Called by audio_pipeline with recognized speech."""
    print(f"\n🗣️  Processing: {text}")

    # Block the mic for the entire request + TTS lifecycle
    if audio_pipeline:
        audio_pipeline.set_speaking(True)
    try:
        response = route_request(text)
        handle_response(response)
    finally:
        if audio_pipeline:
            audio_pipeline.set_speaking(False)


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