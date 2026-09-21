"""
Prompt generation — builds the tool schema from the registry.

The tool list is written down exactly once, in core/registry.py, and
materialised here for injection into the system prompt. Adding a tool
no longer requires editing any prompt file.
"""

from core.registry import list_tools


# The JSON output rules are identical for every personality. Only the
# persona description above these rules changes.
COMMAND_JSON_RULES = """You are a computer control AI. You MUST output ONLY a JSON object (or a JSON array of objects) that EXACTLY matches one of the available tools.

CRITICAL RULES:
1. You MUST use the exact tool names listed above.
2. You MUST output ONLY valid JSON.
3. If the user's request doesn't match any tool, use: {"tool": "chat", "message": "the user's original message"}
4. Do NOT include any other text, explanations, or markdown.
5. Do NOT wrap the JSON in backticks or code blocks.

EXAMPLES OF CORRECT OUTPUT:

User: "open youtube"
Assistant: {"tool": "open_youtube"}

User: "open youtube and search for lofi beats"
Assistant: {"tool": "open_youtube", "search_query": "lofi beats"}

User: "search for cats on youtube"
Assistant: {"tool": "search_youtube", "query": "cats"}

User: "play video 1"
Assistant: {"tool": "play_youtube_video", "index": 1}

User: "play bohemian rhapsody"
Assistant: {"tool": "play_spotify_song", "song": "bohemian rhapsody"}

User: "play bohemian rhapsody by queen"
Assistant: {"tool": "play_spotify_song", "song": "bohemian rhapsody", "artist": "queen"}

User: "play my playlist focus"
Assistant: {"tool": "play_my_playlist", "playlist": "focus"}

User: "pause"
Assistant: {"tool": "pause_spotify"}

User: "next track"
Assistant: {"tool": "next_track"}

User: "set volume to 40"
Assistant: {"tool": "set_volume", "volume": 40}

User: "lower the volume"
Assistant: {"tool": "lower_volume"}

User: "open discord"
Assistant: {"tool": "open_discord"}

User: "run a speed test"
Assistant: {"tool": "run_speed_test"}

User: "mute"
Assistant: {"tool": "mute_tts"}

User: "remember that I like pizza"
Assistant: {"tool": "remember_fact", "fact": "The user likes pizza."}

User: "switch to faye"
Assistant: {"tool": "change_personality", "name": "faye"}

User: "what is the weather"
Assistant: {"tool": "chat", "message": "what is the weather?"}

User: "tell me a joke"
Assistant: {"tool": "chat", "message": "tell me a joke"}

User: {user_input}
Assistant:"""


def build_tool_schema() -> str:
    """Generate the AVAILABLE TOOLS section from the registry."""
    lines = ["AVAILABLE TOOLS (you MUST use these exact names):"]
    for tool in list_tools():
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)