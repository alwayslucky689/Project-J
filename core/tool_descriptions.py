"""
Central registry of tool descriptions.

These strings are the *interface* the router sees. They determine whether
llama3.2:3b or Needle picks the right tool. Every description follows the
same shape:

  1. What the tool does (one sentence)
  2. WHEN to use it (trigger phrases)
  3. WHEN NOT to use it, and which sibling tool to use instead
  4. The argument signature

Rules 2 and 3 are what small models key on. One-word distinctions don't
survive; explicit "NOT for X — use Y" instruction does.
"""

DESCRIPTIONS = {

    # ===== YouTube =====

    "open_youtube": (
        "Opens YouTube in the browser. Use for 'open youtube', 'go to "
        "youtube'. Optional 'search_query' pre-fills a search box only if "
        "the user provides a term with the open command ('open youtube for "
        "lofi beats'). If they want to see results, use search_youtube. "
        'Takes optional "search_query" (string).'
    ),

    "search_youtube": (
        "Searches YouTube and lists results on screen. Use for 'search "
        "youtube for X', 'find videos of X', 'look up X on youtube'. NOT "
        "for opening the site — use open_youtube for that. "
        'Takes "query" (string) and optional "max_results" (integer, default 5).'
    ),

    "play_youtube_video": (
        "Plays a specific video from the most recent YouTube search results, "
        "by 1-based index. Use only after a search_youtube call, for 'play "
        "video 1', 'play the third result'. NOT for Spotify. "
        'Takes "index" (integer, 1-based).'
    ),

    # ===== Spotify =====

    "open_spotify": (
        "Opens Spotify in the browser. Use for 'open spotify', 'launch "
        "spotify'. NOT for playing anything — use the play_* tools for that. "
        "Takes no arguments."
    ),

    "play_spotify_song": (
        "Plays a SPECIFIC TRACK on Spotify, immediately replacing whatever "
        "is playing. Use for song titles: 'play bohemian rhapsody', 'put on "
        "stairway to heaven'. If the user names an artist WITHOUT a track "
        "('play some queen', 'put on the beatles'), pass that name as the "
        "'artist' argument, NOT as 'song'. NOT for playlists — use "
        "play_spotify_playlist or play_my_playlist. NOT for queuing — use "
        'queue_spotify_song. Takes "song" (string) and optional "artist" (string).'
    ),

    "queue_spotify_song": (
        "Adds a song to the Spotify queue WITHOUT interrupting current "
        "playback. Use for 'queue up X', 'add X to the queue', 'play X "
        "next'. NOT for immediate playback — use play_spotify_song. "
        'Takes "song" (string) and optional "artist" (string).'
    ),

    "play_spotify_playlist": (
        "Plays a playlist from Spotify's PUBLIC CATALOG — curated lists "
        "like 'Discover Weekly', 'Today's Top Hits', genre or mood mixes. "
        "Use when the user names a public playlist. NOT for the user's "
        "personal library — use play_my_playlist for 'my playlist X'. "
        'Takes "playlist" (string).'
    ),

    "play_my_playlist": (
        "Plays a playlist from the USER'S PERSONAL LIBRARY (playlists they "
        "created or saved to their account). Triggered by 'my playlist X', "
        "'play my focus playlist', or a playlist name the user owns. NOT "
        "for Spotify's public catalog — use play_spotify_playlist. "
        'Takes "playlist" (string).'
    ),

    "list_playlists": (
        "Lists the user's saved Spotify playlists in the console. Use for "
        "'list my playlists', 'what playlists do I have', 'show my "
        "playlists'. Takes no arguments."
    ),

    "pause_spotify": (
        "Pauses Spotify playback. Use for 'pause', 'pause the music', "
        "'stop the music'. NOT for stopping a specific app. "
        "Takes no arguments."
    ),

    "resume_spotify": (
        "Resumes paused Spotify playback. Use for 'resume', 'continue', "
        "'unpause', 'play' when nothing specific is named. NOT for starting "
        "a new song — use play_spotify_song. Takes no arguments."
    ),

    "next_track": (
        "Skips to the NEXT track on Spotify. Use for 'next', 'skip', 'skip "
        "this song', 'next track'. NOT for previous — use previous_track. "
        "Takes no arguments."
    ),

    "previous_track": (
        "Goes back to the PREVIOUS track on Spotify. Use for 'previous', "
        "'go back', 'last song', 'rewind'. NOT for next — use next_track. "
        "Takes no arguments."
    ),

    "set_volume": (
        "Sets Spotify volume to an EXACT level (0-100). Use for 'set volume "
        "to N', 'volume N', 'make it N percent'. Requires an explicit "
        "number. NOT for relative changes — use raise_volume or lower_volume. "
        'Takes "volume" (integer 0-100).'
    ),

    "raise_volume": (
        "Increases Spotify volume by a RELATIVE amount. Use for 'turn it "
        "up', 'louder', 'volume up', 'raise volume', 'raise volume by N'. "
        "NOT for absolute values — use set_volume. "
        'Takes optional "amount" (integer, default 10).'
    ),

    "lower_volume": (
        "Decreases Spotify volume by a RELATIVE amount. Use for 'turn it "
        "down', 'quieter', 'volume down', 'lower volume'. NOT for absolute "
        "values — use set_volume. "
        'Takes optional "amount" (integer, default 10).'
    ),

    "clear_queue": (
        "Clears the Spotify queue. Use for 'clear the queue', 'empty the "
        "queue', 'remove everything from the queue'. Takes no arguments."
    ),

    # ===== Discord =====

    "open_discord": (
        "Opens the Discord desktop application. Use for 'open discord', "
        "'fire up discord', 'launch discord'. Takes no arguments."
    ),

    # ===== Speed test =====

    "run_speed_test": (
        "Runs a full internet speed test (download, upload, ping). Takes "
        "30-60 seconds. Use for 'run a speed test', 'how fast is my "
        "internet', 'test my connection', 'check my internet speed'. "
        'Takes optional "background" (boolean, default true).'
    ),

    "quick_speed_test": (
        "Runs a FAST internet speed test with simplified output. Use ONLY "
        "for explicit 'quick speed test' or 'fast speed test'. Default to "
        "run_speed_test for anything else. Takes no arguments."
    ),

    # ===== Chat fallback =====

    "chat": (
        "Handles any input that is NOT a command for another tool. Use for "
        "questions ('what is 2+2', 'what is the capital of France'), "
        "conversation ('how are you', 'tell me a joke'), opinions, math, "
        "definitions, small talk, or nonsense input ('xyzzy'). This is the "
        "DEFAULT when nothing else clearly fits. Do NOT force another tool "
        "when the input is a general question. "
        'Takes "message" (string) — the user\'s original message.'
    ),

    # ===== Personality =====

    "change_personality": (
        "Changes the active AI personality. Use for 'switch to X', 'change "
        "to X', 'become X', 'activate X'. Valid names: jarvis, faye, "
        'computah. Takes "name" (string).'
    ),

    # ===== TTS voice control =====

    "mute_tts": (
        "Mutes the assistant's voice output — it stops speaking but still "
        "prints text. Use for 'mute', 'voice off', 'silence', 'quiet "
        "mode', 'be quiet', 'stop talking'. NOT for toggling — use "
        "toggle_tts. Takes no arguments."
    ),

    "unmute_tts": (
        "Unmutes the assistant's voice output — it resumes speaking. Use "
        "for 'unmute', 'voice on', 'speak again', 'unmute yourself', "
        "'start talking'. NOT for toggling — use toggle_tts. "
        "Takes no arguments."
    ),

    "toggle_tts": (
        "Toggles the assistant's voice output on/off. Use ONLY for explicit "
        "'toggle voice', 'toggle mute', 'flip voice', or when the user asks "
        "to switch without specifying on or off. If they say 'voice on' or "
        "'voice off', use unmute_tts or mute_tts instead. Takes no arguments."
    ),

    "set_tts_volume": (
        "Sets the assistant's TEXT-TO-SPEECH volume to an exact level "
        "(0-100). Use for 'set TTS volume to N', 'TTS volume N'. NOT for "
        "Spotify volume — use set_volume for music. "
        'Takes "volume" (integer 0-100).'
    ),

    "raise_tts_volume": (
        "Increases the assistant's TEXT-TO-SPEECH volume by a relative "
        "amount. Use for 'turn up TTS', 'raise TTS volume', 'louder TTS'. "
        "NOT for Spotify — use raise_volume for music. "
        'Takes optional "amount" (integer, default 10).'
    ),

    "lower_tts_volume": (
        "Decreases the assistant's TEXT-TO-SPEECH volume by a relative "
        "amount. Use for 'turn down TTS', 'lower TTS volume', 'quieter "
        "TTS'. NOT for Spotify — use lower_volume for music. "
        'Takes optional "amount" (integer, default 10).'
    ),

    # ===== Memory =====

    "remember_fact": (
        "Saves a personal fact about the user for later recall. Use for "
        "requests containing verbs like 'remember', 'note that', 'keep in "
        "mind', 'don't forget', 'save this'. NOT for searching the web — "
        "use search_youtube for video search, chat for general questions. "
        'Takes "fact" (string).'
    ),
}