# assistant.py - Merged: wakeword + parakeet STT + blocking TTS
# Phase 0 cleanup: single return contract, blocking TTS, voice pipeline

import re
import subprocess
import json
import requests
import os
import threading
from dataclasses import dataclass

from tts.tts_manager import speak_async
from config import settings
from config.personality_manager import PersonalityManager
from tools import youtube, spotify, discord, ollama, ookla, personality_tools, tts_tools
from memory.fact_memory import get_facts_context, save_fact
from audio_pipeline import AudioPipeline
from config.paths import OLLAMA_EXE, STT_VENV_PYTHON, STT_SERVICE_SCRIPT
from core.registry import get_tool

if not OLLAMA_EXE:
    print("⚠️ Ollama not found. Please add it to your PATH.")
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
def build_system_prompt():
    base_system = personality_mgr.get_prompt("system")
    facts = get_facts_context()
    if facts:
        return base_system.replace("{user_facts}", facts)
    return base_system.replace("{user_facts}", "")


# ===== Ollama Calls =====
def ask_ollama(prompt_text, is_json=False, model=None):
    model = model or settings.FAST_MODEL

    try:
        response = requests.post(
            url=settings.OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt_text,
                "stream": False,
                "temperature": 0.1 if is_json else 0.7,
            },
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return {"tool": "error", "message": f"Connection error: {e}"} if is_json else f"Error: {e}"

    if response.status_code != 200:
        return {"tool": "error", "message": f"Ollama error: {response.status_code}"} if is_json else f"Error: {response.status_code}"

    try:
        data = response.json()
        ai_output = data.get("response", "").strip()
    except json.JSONDecodeError:
        return {"tool": "error", "message": "Invalid JSON response"} if is_json else "Error: Invalid response"

    if not ai_output:
        return {"tool": "error", "message": "Empty response"} if is_json else "I didn't get a response."

    if is_json:
        try:
            json_match = re.search(r"\{.*\}", ai_output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return {"tool": "error", "message": f"No JSON found in: {ai_output[:200]}"}
        except json.JSONDecodeError:
            return {"tool": "error", "message": f"Invalid JSON: {ai_output[:200]}"}
    else:
        return ai_output


def ask_ollama_streaming(prompt_text, model=None):
    """
    Streams tokens to the terminal. Returns the full response text.
    Streamed output is already visible to the user, so callers should
    NOT reprint — but they SHOULD still synthesize via _speak_safe().
    """
    model = model or settings.FAST_MODEL

    if not OLLAMA_EXE:
        print("❌ Ollama not found! Using HTTP API instead.")
        result = ask_ollama(prompt_text, is_json=False, model=model)
        print(f"🤖 {result}")
        return result

    print("🤖 ", end="", flush=True)
    try:
        process = subprocess.Popen(
            [OLLAMA_EXE, "run", model, prompt_text],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        full_response = ""
        char_count = 0
        while True:
            char = process.stdout.read(1)
            if not char:
                break
            print(char, end="", flush=True)
            full_response += char
            char_count += 1

        process.wait()
        if char_count > 0:
            print()
        return full_response.strip()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return ""


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
                text=f"{detected.capitalize()} personality active. How can I help?"
            )

    # --- Question path (streamed to terminal, then spoken) ---
    question_keywords = [
        "what", "why", "how", "when", "where", "who", "which",
        "does", "do", "is", "are", "did", "could", "would",
        "should", "will", "can", "tell me", "explain", "describe",
    ]
    is_question = (
        user_input.strip().endswith("?")
        or any(user_input.lower().startswith(w) for w in question_keywords)
    )

    if is_question:
        prompt = f"""Answer the user's question naturally, conversationally, and accurately.
Be concise but helpful. Don't mention that you're an AI.

User: {user_input}
Assistant:"""
        response = ask_ollama_streaming(prompt, model=settings.REASONING_MODEL)
        return AssistantResponse(text=response, was_streamed=True)

    # --- Command path (JSON tool call, non-streamed) ---
    action_keywords = [
        "open", "play", "search", "start", "run", "remember", "switch",
        "change", "test", "pause", "resume", "next", "previous", "mute",
        "unmute", "set", "lower", "raise", "clear", "list", "queue",
    ]
    is_command = any(word in user_input.lower() for word in action_keywords)

    if is_command:
        system_context = build_system_prompt()
        command_template = personality_mgr.get_prompt("command")
        prompt = command_template.replace("{user_input}", user_input)
        full_prompt = f"{system_context}\n\n{prompt}"

        decision = ask_ollama(full_prompt, is_json=True, model=settings.FAST_MODEL)
        result = execute_tool(decision)
        return AssistantResponse(text=result, was_streamed=False)

    # --- Default chat (streamed to terminal, then spoken) ---
    default_prompt = f"""The user said: {user_input}. Respond naturally and helpfully.
If they're asking for something, answer directly. If it's a command, tell them clearly.

Your response (natural language):"""
    response = ask_ollama_streaming(default_prompt, model=settings.REASONING_MODEL)
    return AssistantResponse(text=response, was_streamed=True)


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

    # Special case: the LLM signalling it errored out
    if tool_name == "error":
        return f"⚠️ AI Error: {action.get('message', 'unknown')}"

    tool = get_tool(tool_name)
    if tool is None:
        return f"⚠️ Unknown tool: {tool_name}. AI said: {action}"

    # Strip the "tool" key; everything else is a kwarg for the handler
    kwargs = {k: v for k, v in action.items() if k != "tool"}

    try:
        result = tool.handler(**kwargs)
        return tool.format(result)
    except TypeError as e:
        # Wrong argument names — LLM emitted something the handler doesn't accept
        return f"❌ Argument error for {tool_name}: {e}"
    except Exception as e:
        return f"❌ Error executing {tool_name}: {e}"

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