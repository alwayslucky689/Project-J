# assistant.py - COMPLETE FIXED VERSION
import re
import json
import requests
import os
import sys
from tts_manager import speak_async
from config import settings
from config.personality_manager import PersonalityManager
from tools import youtube, spotify, discord, ollama, ookla, personality_tools
from memory.fact_memory import get_facts_context, save_fact

# 1. Initialize Personality Manager
personality_mgr = PersonalityManager()
personality_tools.init_personality_tools(personality_mgr)

# 2. Helper: Build the full system prompt with memory
def build_system_prompt():
    base_system = personality_mgr.get_prompt("system")
    facts = get_facts_context()
    if facts:
        return base_system.replace("{user_facts}", facts)
    return base_system.replace("{user_facts}", "")

# 3. Core AI Interaction Function (FIXED)
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
    
    ai_output = response.json().get("response", "").strip()
    
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

# 4. Command Router (FIXED)
def route_request(user_input):
    # Check for personality switch
    detected = personality_mgr.detect_personality(user_input)
    if detected and personality_mgr.switch_to(detected):
        for word in personality_mgr.get_wake_words(detected):
            user_input = user_input.lower().replace(word.lower(), "").strip()
        print(f"🧠 Switched personality to: {detected}")
        if not user_input:
            return f"💬 {detected.capitalize()} personality active. How can I help?"
    
    # Step 1: Check if it's a question (bypass JSON)
    question_keywords = ["what", "why", "how", "when", "where", "who", "which", 
                         "does", "do", "is", "are", "did", "could", "would", 
                         "should", "will", "can", "tell me", "explain", "describe"]
    is_question = user_input.strip().endswith("?") or any(user_input.lower().startswith(w) for w in question_keywords)
    
    if is_question:
        prompt = f"""Answer the user's question naturally, conversationally, and accurately. 
Be concise but helpful. Don't mention that you're an AI.

User: {user_input}
Assistant:"""
        response = ask_ollama(prompt, is_json=False, model=settings.REASONING_MODEL)
        return response
    
    # Step 2: Check for command keywords
    action_keywords = ["open", "play", "search", "start", "run", "remember", "switch", "change", "test", "pause", "resume", "next", "previous", "mute", "unmute", "clear", "list", "queue"]
    is_command = any(word in user_input.lower() for word in action_keywords)
    
    if is_command:
        system_context = build_system_prompt()
        command_template = personality_mgr.get_prompt("command")
        prompt = command_template.replace("{user_input}", user_input)
        full_prompt = f"{system_context}\n\n{prompt}"
        
        decision = ask_ollama(full_prompt, is_json=True, model=settings.FAST_MODEL)
        return execute_tool(decision)
    
    # Step 3: Default natural conversation
    default_prompt = f"""The user said: {user_input}. Respond naturally and helpfully.
If they're asking for something, answer directly. If it's a command, tell them clearly.

Your response (natural language):"""
    return ask_ollama(default_prompt, is_json=False, model=settings.REASONING_MODEL)

# 5. Tool Executor (unchanged - works fine)
def execute_tool(decision):
    if isinstance(decision, list):
        results = []
        for action in decision:
            results.append(execute_single_action(action))
        return "\n".join(results)
    else:
        return execute_single_action(decision)

def execute_single_action(action):
    tool_name = action.get("tool")
    try:
        if tool_name == "open_youtube":
            return youtube.open_youtube(action.get("search_query"))
        elif tool_name == "search_youtube":
            results = youtube.search_youtube(action.get("query"), action.get("max_results", 5))
            if results:
                return f"✅ Found {len(results)} videos. You can say 'play video 1' to play the first one."
            else:
                return "❌ No videos found."
        elif tool_name == "play_youtube_video":
            index = action.get("index", 1)
            success = youtube.play_youtube_video(index)
            return "▶️ Playing video" if success else "❌ Could not play video."
        elif tool_name == "open_spotify":
            return spotify.open_spotify()
        elif tool_name == "play_spotify_song":
            return spotify.play_spotify_song(action.get("song"), action.get("artist"))
        elif tool_name == "queue_spotify_song":
            return spotify.queue_spotify_song(action.get("song"), action.get("artist"))
        elif tool_name == "play_spotify_playlist":
            return spotify.play_spotify_playlist(action.get("playlist"))
        elif tool_name == "play_my_playlist":
            return spotify.play_my_playlist(action.get("playlist"))
        elif tool_name == "list_playlists":
            playlists = spotify.list_playlists()
            return "📋 Listed playlists in the console."
        elif tool_name == "pause_spotify":
            return spotify.pause_spotify()
        elif tool_name == "resume_spotify":
            return spotify.resume_spotify()
        elif tool_name == "next_track":
            return spotify.next_track()
        elif tool_name == "previous_track":
            return spotify.previous_track()
        elif tool_name == "set_volume":
            return spotify.set_volume(action.get("volume", 50))
        elif tool_name == "raise_volume":
            return spotify.raise_volume(action.get("amount", 10))
        elif tool_name == "lower_volume":
            return spotify.lower_volume(action.get("amount", 10))
        elif tool_name == "clear_queue":
            return spotify.clear_queue()
        elif tool_name == "open_discord":
            return discord.open_discord()
        elif tool_name == "run_speed_test":
            results = ookla.run_speed_test(background=action.get("background", True))
            return ookla.get_formatted_results(results)
        elif tool_name == "quick_speed_test":
            return ookla.quick_speed_test()
        elif tool_name == "remember_fact":
            fact = action.get("fact")
            if fact:
                return save_fact(fact)
            else:
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

# 6. The Main Loop
def main():
    print("🤖 Project-J AI Assistant Ready!")
    print(f"Current personality: {personality_mgr.get_current_name()}")
    print(f"Wake words: {personality_mgr.get_wake_words()}")
    print("Type 'quit' to exit.")
    print("-" * 40)
    
    while True:
        user_input = input("\n🎤 You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("👋 Goodbye!")
            break
        
        result = route_request(user_input)
        print(f"🤖 {result}")
        if settings.ENABLE_TTS and result and isinstance(result, str):
            speak_async(result)

if __name__ == "__main__":
    main()