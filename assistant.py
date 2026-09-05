# assistant.py
"""
Main AI Assistant script
"""

import json
import re
import requests
from config import settings
from tools import (
    youtube,
    spotify,
    discord,
    ollama,
    ookla
)

# ===== HELPER FUNCTIONS =====

def is_question(text):
    """Check if text is a question"""
    text = text.strip()
    if text.endswith("?"):
        return True
    
    question_words = ["what", "why", "how", "when", "where", "who", "which", 
                      "does", "do", "is", "are", "did", "could", "would", 
                      "should", "will", "can", "tell me", "explain", 
                      "describe", "define", "meaning of", "definition of"]
    
    lower_text = text.lower()
    for word in question_words:
        if lower_text.startswith(word) or f" {word} " in lower_text:
            return True
    
    return False

def ask_ai_for_command(user_input):
    """Let the AI handle complex/multi commands"""
    prompt = f"""You are an AI that controls the user's computer.
You MUST respond with ONLY valid JSON array.

CRITICAL RULES:
- If the user says "play [something]" and it's NOT "play the [number] video", it's probably music → use play_spotify_song
- If the user says "play [playlist name] playlist" → use play_spotify_playlist
- Only use YouTube if the user specifically says "youtube" or "video"

Available tools:
- open_discord: Opens Discord
- open_youtube: Opens YouTube
- search_youtube: Searches YouTube. Requires "search_query"
- play_youtube_video: Plays video. Takes optional "index"
- show_youtube_links: Shows video links
- open_spotify: Opens Spotify in browser
- play_spotify_song: Plays a song. Requires "song_name"
- queue_spotify_song: Queues a song. Requires "song_name"
- play_spotify_playlist: Plays playlist. Requires "playlist_name"
- raise_volume: Raises volume. Takes optional "amount"
- lower_volume: Lowers volume. Takes optional "amount"
- set_volume: Sets volume. Requires "volume"
- pause_spotify: Pauses playback
- resume_spotify: Resumes playback
- next_track: Skips track
- previous_track: Goes back
- ask_question: Answers question. Requires "question"

EXAMPLES:
User: "Play Bohemian Rhapsody"
Response: [{{"tool": "play_spotify_song", "song_name": "Bohemian Rhapsody"}}]

User: "Play night of fire"
Response: [{{"tool": "play_spotify_song", "song_name": "Night of Fire"}}]

User: "Play my chill playlist"
Response: [{{"tool": "play_spotify_playlist", "playlist_name": "chill"}}]

User: "Search for cats on YouTube"
Response: [{{"tool": "search_youtube", "search_query": "cats"}}]

User: "Open YouTube"
Response: [{{"tool": "open_youtube"}}]

Respond to: "{user_input}"
Assistant:"""
    
    try:
        response = requests.post(
            url=settings.OLLAMA_URL,
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1
            },
            timeout=30
        )
        
        ai_output = response.json()["response"].strip()
        
        if settings.DEBUG:
            print(f"🔍 DEBUG: AI Response: {ai_output}")
        
        # Parse JSON
        try:
            json_match = re.search(r'\[.*\]', ai_output, re.DOTALL)
            if json_match:
                actions = json.loads(json_match.group())
            else:
                actions = json.loads(ai_output)
            
            if isinstance(actions, dict):
                actions = [actions]
            elif not isinstance(actions, list):
                actions = [{"tool": "error", "message": "Invalid response"}]
            
            return actions
            
        except json.JSONDecodeError:
            return [{"tool": "error", "message": f"Could not parse: {ai_output}"}]
            
    except Exception as e:
        return [{"tool": "error", "message": f"Error: {str(e)}"}]

def execute_single_tool(action):
    """Executes a single tool action"""
    tool_name = action.get("tool")
    
    # ===== YOUTUBE =====
    if tool_name == "open_youtube":
        youtube.open_youtube(action.get("search_query"))
    
    elif tool_name == "search_youtube":
        query = action.get("search_query")
        if query:
            youtube.search_youtube(query)
    
    elif tool_name == "play_youtube_video":
        index = action.get("index")
        youtube.play_youtube_video(index if index else None)
    
    elif tool_name == "show_youtube_links":
        youtube.show_youtube_links()
    
    # ===== SPOTIFY =====
    elif tool_name == "open_spotify":
        spotify.open_spotify()
    
    elif tool_name == "play_spotify_song":
        song = action.get("song_name")
        if song:
            spotify.play_spotify_song(song)
    
    elif tool_name == "queue_spotify_song":
        song = action.get("song_name")
        if song:
            spotify.queue_spotify_song(song)
    
    elif tool_name == "play_spotify_playlist":
        playlist = action.get("playlist_name")
        if playlist:
            spotify.play_spotify_playlist(playlist)
    
    elif tool_name == "raise_volume":
        amount = action.get("amount", 10)
        spotify.raise_volume(amount)
    
    elif tool_name == "lower_volume":
        amount = action.get("amount", 10)
        spotify.lower_volume(amount)
    
    elif tool_name == "set_volume":
        volume = action.get("volume")
        if volume is not None:
            spotify.set_volume(volume)
    
    elif tool_name == "pause_spotify":
        spotify.pause_spotify()
    
    elif tool_name == "resume_spotify":
        spotify.resume_spotify()
    
    elif tool_name == "next_track":
        spotify.next_track()
    
    elif tool_name == "previous_track":
        spotify.previous_track()
    
    # ===== DISCORD =====
    elif tool_name == "open_discord":
        discord.open_discord()
    
    # ===== QUESTIONS =====
    elif tool_name == "ask_question":
        question = action.get("question")
        if question:
            answer = ollama.ask_question(question)
            print(f"🤖 {answer}\n")
    
    elif tool_name == "error":
        print(f"❌ Error: {action.get('message', 'Unknown error')}")
    
    else:
        print(f"❌ Unknown tool: {tool_name}")

def execute_tool(actions):
    """Executes one or more actions"""
    if isinstance(actions, list):
        for action in actions:
            execute_single_tool(action)
    else:
        execute_single_tool(actions)
    HELP_TEXT = """
📋 AVAILABLE COMMANDS:

🎵 SPOTIFY:
  - play [song name]              - Play a song
  - queue [song name]             - Add song to queue
  - play [playlist name] playlist - Play a playlist (searches your library + Spotify)
  - play my [playlist name]       - Play from YOUR playlists only
  - list playlists                - Show all your playlists
  - pause / resume / play         - Control playback
  - next / previous               - Skip tracks
  - volume up [amount]            - Increase volume
  - volume down [amount]          - Decrease volume
  - set volume [amount]           - Set exact volume
  - mute / unmute                 - Mute/unmute

📺 YOUTUBE:
  - open youtube                  - Open YouTube
  - search for [query]            - Search YouTube
  - play the [number] video       - Play a video from search
  - show me the links             - Show video URLs

💬 OTHER:
  - open discord                  - Open Discord
  - [any question]                - Ask anything!
  - help                          - Show this menu
  - quit / bye / exit             - Exit the assistant
"""

# In main() function:

def main():
    """Main program loop"""
    print("=" * 50)
    print("🤖 Started")
    print("Type 'help' to see all commands")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nWhat do you want? > ").strip()
            
            if user_input.lower() in ["quit", "bye", "exit", "cao", "kys"]:
                print("Goodbye! 👋")
                break
            
            if not user_input:
                continue
            
            lower_input = user_input.lower()
            
            # ============================================
            # 0. HELP COMMAND
            # ============================================
            if lower_input in ["help", "?"]:
                print(HELP_TEXT)
                continue
            
            # ============================================
            # 1. DISCORD COMMANDS
            # ============================================
            if "discord" in lower_input:
                discord.open_discord()
                continue
            # ============================================
            # 1.5 SPEED TEST COMMANDS
            # ============================================
            if "speed test" in lower_input or "speedtest" in lower_input or "internet speed" in lower_input or "check internet" in lower_input or "how fast is my internet" in lower_input:
                print("🌐 Running internet speed test...")
                print("⏳ Please wait, this takes about 30-60 seconds...")
                
                # Run in background
                results = ookla.run_speed_test(background=True)
                
                # Display results
                formatted = ookla.get_formatted_results(results)
                print(formatted)
                continue

            # Also add a "test internet" alias
            if "test internet" in lower_input:
                print("🌐 Running internet speed test...")
                print("⏳ Please wait, this takes about 30-60 seconds...")
                results = ookla.run_speed_test(background=True)
                formatted = ookla.get_formatted_results(results)
                print(formatted)
                continue            
            # ============================================
            # 2. YOUTUBE COMMANDS
            # ============================================
            if "youtube" in lower_input or "video" in lower_input:
                if "search" in lower_input or "find" in lower_input:
                    query = re.sub(r'(search|find|for|on|youtube|video)\s*', '', user_input, flags=re.IGNORECASE).strip()
                    if query:
                        youtube.search_youtube(query)
                    continue
                elif "play" in lower_input:
                    numbers = re.findall(r'\d+', lower_input)
                    if numbers:
                        youtube.play_youtube_video(int(numbers[0]))
                    elif "first" in lower_input:
                        youtube.play_youtube_video(1)
                    else:
                        youtube.play_youtube_video()
                    continue
                elif "link" in lower_input or "url" in lower_input:
                    youtube.show_youtube_links()
                    continue
                else:
                    youtube.open_youtube()
                    continue
            # ============================================
            # 3. SPOTIFY COMMANDS (ALL)
            # ============================================

            # --- LIST PLAYLISTS ---
            if "list playlists" in lower_input or "show playlists" in lower_input:
                spotify.list_playlists()
                continue

            # --- PAUSE ---
            if lower_input == "pause" or "pause spotify" in lower_input:
                spotify.pause_spotify()
                continue

            # --- RESUME / PLAY / UNPAUSE ---
            if lower_input in ["play", "resume", "unpause"] or "resume spotify" in lower_input or "unpause spotify" in lower_input:
                spotify.resume_spotify()
                continue

            # --- NEXT / PREVIOUS ---
            if "next" in lower_input or "skip" in lower_input:
                spotify.next_track()
                continue

            if "previous" in lower_input or "back" in lower_input:
                spotify.previous_track()
                continue

            # --- CLEAR QUEUE (MUST BE BEFORE "queue" check) ---
            if "clear queue" in lower_input or "clear the queue" in lower_input or "empty queue" in lower_input:
                spotify.clear_queue()
                continue

            # --- VOLUME COMMANDS ---
            numbers = re.findall(r'\d+', lower_input)

            if "volume" in lower_input or "louder" in lower_input or "quieter" in lower_input:
                if "up" in lower_input or "louder" in lower_input or "increase" in lower_input:
                    amount = int(numbers[0]) if numbers else 10
                    spotify.raise_volume(amount)
                    continue
                elif "down" in lower_input or "quieter" in lower_input or "decrease" in lower_input:
                    amount = int(numbers[0]) if numbers else 10
                    spotify.lower_volume(amount)
                    continue
                elif "set" in lower_input:
                    amount = int(numbers[0]) if numbers else 50
                    spotify.set_volume(amount)
                    continue
                elif "mute" in lower_input:
                    spotify.mute_volume()
                    continue
                elif "unmute" in lower_input:
                    spotify.unmute_volume()
                    continue

            # --- PLAY MY PLAYLIST ---
            if "play my" in lower_input and "playlist" in lower_input:
                playlist = re.sub(r'(play|my|playlist|on|spotify)\s*', '', user_input, flags=re.IGNORECASE).strip()
                if playlist:
                    spotify.play_spotify_playlist(playlist)
                else:
                    print("❌ Which playlist do you want to play?")
                continue

            # --- QUEUE COMMAND ---
            if "queue" in lower_input:
                # Extract song name
                song = re.sub(r'(queue|add|song|track|to|on|spotify|the|playlist)\s*', '', user_input, flags=re.IGNORECASE).strip()
                if song:
                    spotify.queue_spotify_song(song)
                else:
                    print("❌ What song do you want to queue?")
                continue

            # --- PLAY SONG ---
            if "play" in lower_input and not any(word in lower_input for word in ["youtube", "video", "the first", "the second", "the third", "playlist", "my"]):
                # Extract song name
                song = re.sub(r'(play|on|spotify|song|track|music)\s*', '', user_input, flags=re.IGNORECASE).strip()
                if song:
                    spotify.play_spotify_song(song)
                else:
                    print("❌ What song do you want to play?")
                continue

            # --- PLAY PLAYLIST ---
            if "playlist" in lower_input and "play" in lower_input:
                # Extract playlist name
                playlist = re.sub(r'(play|playlist|on|spotify|the)\s*', '', user_input, flags=re.IGNORECASE).strip()
                if playlist:
                    spotify.play_spotify_playlist(playlist)
                else:
                    print("❌ What playlist do you want to play?")
                continue

            # --- OPEN SPOTIFY ---
            if "open spotify" in lower_input or "spotify" in lower_input:
                spotify.open_spotify()
                continue
            # ============================================
            # 4. QUESTIONS (Now after all commands)
            # ============================================
            if is_question(user_input):
                answer = ollama.ask_question(user_input)
                print(f"🤖 {answer}\n")
                continue
            
            # ============================================
            # 5. FALLBACK TO AI
            # ============================================
            actions = ask_ai_for_command(user_input)
            execute_tool(actions)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()