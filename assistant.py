# assistant.py - Complete with voice pipeline and blocking TTS

import re
import subprocess
import json
import requests
import os
import shutil
import threading
import time
from tts_manager import speak_async
from config import settings
from config.personality_manager import PersonalityManager
from tools import youtube, spotify, discord, ollama, ookla, personality_tools
from memory.fact_memory import get_facts_context, save_fact
from audio_pipeline import AudioPipeline


# ===== Paths =====
STT_VENV_PYTHON = r"venv_stt\Scripts\python.exe"
STT_SERVICE_SCRIPT = r"stt\stt_service.py"


def get_ollama_path():
    possible_paths = [
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Users\pstef\AppData\Local\Programs\Ollama\ollama.exe",
        shutil.which("ollama")
    ]
    for path in possible_paths:
        if path and os.path.exists(path):
            return path
    return None


OLLAMA_EXE = get_ollama_path()
if not OLLAMA_EXE:
    print("⚠️ Ollama not found. Please add it to your PATH.")

# Initialize Personality Manager
personality_mgr = PersonalityManager()
personality_tools.init_personality_tools(personality_mgr)

# Global references
audio_pipeline = None


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
                "temperature": 0.1 if is_json else 0.7
            },
            timeout=30
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
            json_match = re.search(r'\{.*\}', ai_output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {"tool": "error", "message": f"No JSON found in: {ai_output[:200]}"}
        except json.JSONDecodeError:
            return {"tool": "error", "message": f"Invalid JSON: {ai_output[:200]}"}
    else:
        return ai_output


def ask_ollama_streaming(prompt_text, model=None):
    model = model or settings.FAST_MODEL

    if not OLLAMA_EXE:
        print("❌ Ollama not found! Using HTTP API instead.")
        return ask_ollama(prompt_text, is_json=False, model=model)

    print("🤖 ", end="", flush=True)

    try:
        process = subprocess.Popen(
            [OLLAMA_EXE, "run", model, prompt_text],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
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


# ===== TTS Wrapper — blocks until playback finishes =====
def _speak_safe(text):
    """
    Speak and block until playback finishes.
    While TTS is playing, the audio pipeline ignores all input.
    """
    if not text:
        return
    if not settings.ENABLE_TTS:
        return

    if audio_pipeline:
        audio_pipeline.set_speaking(True)

    done = threading.Event()

    def _on_complete():
        done.set()

    try:
        speak_async(text, on_complete=_on_complete)
        # Wait up to 60s for long responses
        done.wait(timeout=60)
    except Exception as e:
        print(f"⚠️ TTS error: {e}")
    finally:
        if audio_pipeline:
            audio_pipeline.set_speaking(False)


# ===== Command Router =====
def route_request(user_input):
    detected = personality_mgr.detect_personality(user_input)
    if detected and personality_mgr.switch_to(detected):
        for word in personality_mgr.get_wake_words(detected):
            user_input = user_input.lower().replace(word.lower(), "").strip()
        print(f"🧠 Switched personality to: {detected}")
        if not user_input:
            response = f"💬 {detected.capitalize()} personality active. How can I help?"
            _speak_safe(response)
            return {"response": response, "was_streamed": False}

    question_keywords = ["what", "why", "how", "when", "where", "who", "which",
                         "does", "do", "is", "are", "did", "could", "would",
                         "should", "will", "can", "tell me", "explain", "describe"]
    is_question = user_input.strip().endswith("?") or any(user_input.lower().startswith(w) for w in question_keywords)

    if is_question:
        prompt = f"""Answer the user's question naturally, conversationally, and accurately.
Be concise but helpful. Don't mention that you're an AI.

User: {user_input}
Assistant:"""
        response = ask_ollama_streaming(prompt, model=settings.REASONING_MODEL)
        if response:
            _speak_safe(response)
        return {"response": response, "was_streamed": True}

    action_keywords = ["open", "play", "search", "start", "run", "remember", "switch",
                       "change", "test", "pause", "resume", "next", "previous", "mute",
                       "unmute", "set", "lower", "raise", "clear", "list", "queue"]
    is_command = any(word in user_input.lower() for word in action_keywords)

    if is_command:
        system_context = build_system_prompt()
        command_template = personality_mgr.get_prompt("command")
        prompt = command_template.replace("{user_input}", user_input)
        full_prompt = f"{system_context}\n\n{prompt}"

        decision = ask_ollama(full_prompt, is_json=True, model=settings.FAST_MODEL)
        result = execute_tool(decision)
        return {"response": result, "was_streamed": False}

    default_prompt = f"""The user said: {user_input}. Respond naturally and helpfully.
If they're asking for something, answer directly. If it's a command, tell them clearly.

Your response (natural language):"""
    response = ask_ollama_streaming(default_prompt, model=settings.REASONING_MODEL)
    if response:
        _speak_safe(response)
    return {"response": response, "was_streamed": True}


# ===== Tool Executor =====
def execute_tool(decision):
    if isinstance(decision, list):
        results = []
        for action in decision:
            result = execute_single_action(action)
            if result is not None:
                results.append(str(result))
        return "\n".join(results) if results else ""
    else:
        result = execute_single_action(decision)
        return str(result) if result is not None else ""


def execute_single_action(action):
    tool_name = action.get("tool")
    try:
        if tool_name == "open_youtube":
            result = youtube.open_youtube(action.get("search_query"))
            return result if result is not None else "✅ Opened YouTube"
        elif tool_name == "search_youtube":
            results = youtube.search_youtube(action.get("query"), action.get("max_results", 5))
            if results:
                return f"✅ Found {len(results)} videos. You can say 'play video 1' to play the first one."
            return "❌ No videos found."
        elif tool_name == "play_youtube_video":
            index = action.get("index", 1)
            success = youtube.play_youtube_video(index)
            return "▶️ Playing video" if success else "❌ Could not play video."
        elif tool_name == "open_spotify":
            result = spotify.open_spotify()
            return "✅ Opened Spotify" if result else "❌ Failed to open Spotify"
        elif tool_name == "play_spotify_song":
            result = spotify.play_spotify_song(action.get("song"), action.get("artist"))
            return result if result is not None else "▶️ Playing song"
        elif tool_name == "queue_spotify_song":
            result = spotify.queue_spotify_song(action.get("song"), action.get("artist"))
            return result if result is not None else "🎵 Queued song"
        elif tool_name == "play_spotify_playlist":
            result = spotify.play_spotify_playlist(action.get("playlist"))
            return result if result is not None else "▶️ Playing playlist"
        elif tool_name == "play_my_playlist":
            result = spotify.play_my_playlist(action.get("playlist"))
            return result if result is not None else "▶️ Playing your playlist"
        elif tool_name == "list_playlists":
            spotify.list_playlists()
            return "📋 Listed playlists in the console."
        elif tool_name == "pause_spotify":
            result = spotify.pause_spotify()
            return "⏸️ Paused" if result else "❌ Failed to pause"
        elif tool_name == "resume_spotify":
            result = spotify.resume_spotify()
            return "▶️ Resumed" if result else "❌ Failed to resume"
        elif tool_name == "next_track":
            result = spotify.next_track()
            return "⏭️ Next track" if result else "❌ Failed to skip"
        elif tool_name == "previous_track":
            result = spotify.previous_track()
            return "⏮️ Previous track" if result else "❌ Failed to go back"
        elif tool_name == "set_volume":
            result = spotify.set_volume(action.get("volume", 50))
            return f"🔊 Volume set to {action.get('volume', 50)}%" if result else "❌ Failed to set volume"
        elif tool_name == "raise_volume":
            amount = action.get("amount", 10)
            result = spotify.raise_volume(amount)
            return f"🔊 Volume increased by {amount}%" if result else "❌ Failed to raise volume"
        elif tool_name == "lower_volume":
            amount = action.get("amount", 10)
            result = spotify.lower_volume(amount)
            return f"🔊 Volume decreased by {amount}%" if result else "❌ Failed to lower volume"
        elif tool_name == "clear_queue":
            result = spotify.clear_queue()
            return "🎵 Queue cleared" if result else "❌ Failed to clear queue"
        elif tool_name == "open_discord":
            result = discord.open_discord()
            return "✅ Opened Discord" if result else "❌ Failed to open Discord"
        elif tool_name == "run_speed_test":
            results = ookla.run_speed_test(background=action.get("background", True))
            return ookla.get_formatted_results(results)
        elif tool_name == "quick_speed_test":
            return ookla.quick_speed_test()
        elif tool_name == "mute_tts":
            from tts_manager import mute_tts
            mute_tts()
            return "🔇 Voice output muted. I'll only respond in text."
        elif tool_name == "unmute_tts":
            from tts_manager import unmute_tts
            unmute_tts()
            return "🔊 Voice output enabled."
        elif tool_name == "toggle_tts":
            from tts_manager import toggle_mute, is_muted
            toggle_mute()
            current = "muted" if is_muted() else "enabled"
            return f"🔊 Voice output {current}."
        elif tool_name == "set_tts_volume":
            from tts_manager import set_tts_volume
            volume = action.get("volume", 70)
            set_tts_volume(volume)
            return f"🔊 TTS volume set to {volume}%"
        elif tool_name == "raise_tts_volume":
            from tts_manager import get_tts_volume, set_tts_volume
            amount = action.get("amount", 10)
            current = get_tts_volume()
            new_vol = min(100, current + amount)
            set_tts_volume(new_vol)
            return f"🔊 TTS volume increased to {new_vol}%"
        elif tool_name == "lower_tts_volume":
            from tts_manager import get_tts_volume, set_tts_volume
            amount = action.get("amount", 10)
            current = get_tts_volume()
            new_vol = max(0, current - amount)
            set_tts_volume(new_vol)
            return f"🔊 TTS volume decreased to {new_vol}%"
        elif tool_name == "remember_fact":
            fact = action.get("fact")
            if fact:
                return save_fact(fact)
            return "❌ No fact provided to remember."
        elif tool_name == "change_personality":
            return personality_tools.change_personality(action.get("name"))
        elif tool_name == "ask_question":
            return ollama.ask_question(action.get("question"))
        elif tool_name == "error":
            return f"⚠️ AI Error: {action.get('message')}"
        else:
            return f"⚠️ Unknown tool: {tool_name}. AI said: {action}"
    except Exception as e:
        return f"❌ Error executing {tool_name}: {str(e)}"


# ===== Voice Callback =====
def on_transcription(text):
    """Called by audio_pipeline with recognized speech."""
    print(f"\n🗣️  Processing: {text}")

    # Block audio for the entire duration of processing + TTS
    if audio_pipeline:
        audio_pipeline.set_speaking(True)

    try:
        result = route_request(text)

        if result is not None:
            if isinstance(result, dict):
                response_text = result.get("response", "")
                if not result.get("was_streamed", False) and response_text:
                    print(f"🤖 {response_text}")
            elif isinstance(result, str) and result.strip():
                print(f"🤖 {result}")
    finally:
        # If TTS was used, _speak_safe already released the flag.
        # If TTS was skipped (muted / disabled), release it here.
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
            user_input = input("⌨️  You: ")
            if user_input.lower() in ["quit", "exit", "bye"]:
                break

            result = route_request(user_input)

            if result is not None:
                if isinstance(result, dict):
                    response_text = result.get("response", "")
                    if not result.get("was_streamed", False) and response_text:
                        print(f"🤖 {response_text}")
                elif isinstance(result, str) and result.strip():
                    print(f"🤖 {result}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    if audio_pipeline:
        audio_pipeline.stop()
    print("👋 Goodbye!")


if __name__ == "__main__":
    main()