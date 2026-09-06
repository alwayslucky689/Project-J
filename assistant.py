# assistant.py

import json
import requests
import os
import sys

# Import your settings and modules
# Import TTS manager
from tts_manager import speak_async
from config import settings
from config.personality_manager import PersonalityManager
# Import all your actual tools
from tools import youtube, spotify, discord, ollama, ookla, personality_tools
from memory.fact_memory import get_facts_context, save_fact 

# 1. Initialize Personality Manager
personality_mgr = PersonalityManager()
personality_tools.init_personality_tools(personality_mgr)

# 3. Helper: Build the full system prompt with memory
def build_system_prompt():
    """Combines the personality system prompt with stored facts."""
    base_system = personality_mgr.get_prompt("system")
    facts = get_facts_context()
    if facts:
        # Inject facts into the prompt
        return base_system.replace("{user_facts}", facts)
    return base_system.replace("{user_facts}", "")

# 4. Core AI Interaction Functions (Your existing ask_ollama remains the same)
def ask_ollama(prompt_text, is_json=False):
    """
    Sends a prompt to Ollama and returns the response.
    If is_json is True, it attempts to parse the response as JSON.
    """
    full_prompt = prompt_text
    if is_json:
        # Ensure the model knows to output JSON
        full_prompt = f"{full_prompt}\n\nYou MUST respond with ONLY valid JSON. Do not include any other text."

    response = requests.post(
        url=settings.OLLAMA_URL,  # From your settings.py
        json={
            "model": settings.OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False
        }
    )
    
    if response.status_code != 200:
        return {"error": f"Ollama error: {response.status_code}"}
    
    ai_output = response.json().get("response", "").strip()
    
    if is_json:
        try:
            return json.loads(ai_output)
        except json.JSONDecodeError:
            return {"tool": "error", "message": f"Invalid JSON: {ai_output}"}
    
    return ai_output

# 5. Command Router (Updated to use your tools)
def route_request(user_input):
    """
    Handles the user input: checks for wake word, then routes to command or question.
    Returns a string (for questions) or executes the tool (for commands).
    """
    # Step A: Check for personality switch (wake word)
    detected = personality_mgr.detect_personality(user_input)
    if detected and personality_mgr.switch_to(detected):
        # Remove the wake word from the input
        for word in personality_mgr.get_wake_words(detected):
            user_input = user_input.lower().replace(word.lower(), "").strip()
        print(f"🧠 Switched personality to: {detected}")
        # If only the wake word was said, return.
        if not user_input:
            return f"💬 {detected.capitalize()} personality active. How can I help?"

    # Step B: Determine if it's a command or question
    action_keywords = ["open", "play", "search", "start", "run", "remember", "switch", "change", "test", "pause", "resume", "next", "previous", "mute", "unmute", "clear", "list", "queue"]
    is_command = any(word in user_input.lower() for word in action_keywords)
    
    if is_command:
        # Get the command prompt for the current personality
        command_template = personality_mgr.get_prompt("command")
        prompt = command_template.replace("{user_input}", user_input)
        
        # Add the system prompt context to guide the AI
        system_context = build_system_prompt()
        full_prompt = f"{system_context}\n\n{prompt}"
        
        decision = ask_ollama(full_prompt, is_json=True)
        
        # Execute the tool(s)
        return execute_tool(decision)
    else:
        # It's a question - use your existing ollama.ask_question function
        return ollama.ask_question(user_input)

# 6. Tool Executor (CORRECTED - now maps to YOUR actual functions)
def execute_tool(decision):
    """
    Executes the tool specified in the AI's JSON decision.
    Supports both single actions and lists of actions.
    """
    if isinstance(decision, list):
        results = []
        for action in decision:
            results.append(execute_single_action(action))
        return "\n".join(results)
    else:
        return execute_single_action(decision)

def execute_single_action(action):
    tool_name = action.get("tool")
    
   
    # Route to your ACTUAL functions
    try:
        if tool_name == "open_youtube":
            # Call your youtube.open_youtube function
            return youtube.open_youtube(action.get("search_query"))
        
        elif tool_name == "search_youtube":
            results = youtube.search_youtube(action.get("query"), action.get("max_results", 5))
            if results:
                return f"✅ Found {len(results)} videos. You can say 'play video 1' to play the first one."
            else:
                return "❌ No videos found."
        
        elif tool_name == "play_youtube_video":
            index = action.get("index", 1)
            # Default to first video if index is not provided or invalid
            if index is None:
                index = 1
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
            # Use your ookla module
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
            # Use your ollama module to answer a question
            return ollama.ask_question(action.get("question"))
        
        elif tool_name == "error":
            return f"⚠️ AI Error: {action.get('message')}"
        
        else:
            return f"⚠️ Unknown tool: {tool_name}. AI said: {action}"
    
    except Exception as e:
        return f"❌ Error executing {tool_name}: {str(e)}"

# 7. The Main Loop
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