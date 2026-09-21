"""
Keyword-based tool prefilter.

Needle 2's base checkpoint has an untrained retrieval head. It works
near-perfectly with ≤5 candidate tools and collapses above ~8. This
module narrows the candidate list to a small, query-relevant set.

Design rules:
  - Each category exposes at most 4 tools (plus chat, always).
  - Narrow signals: "play my playlist" fires a different category than
    "play X". Overlap between categories is minimized.
  - TTS-specific signals suppress music-volume categories.
  - FALLBACK_TOOLS is minimal — just chat — so off-topic queries get a
    clean chat route instead of a random tool.

This is a workaround for the current base checkpoint. If a future Needle
release ships trained retrieval weights, this becomes a no-op.
"""

from __future__ import annotations


# Each category: {name: {"signals": [...], "tools": {...}}}
# Signals are matched case-insensitively as substrings.
# Keep tool sets ≤ 4 wherever possible.
CATEGORIES = {

    # ===== YouTube =====

    "youtube_open": {
        "signals": ["open youtube", "go to youtube", "launch youtube"],
        "tools": {"open_youtube"},
    },
    "youtube_search": {
        "signals": [
            "search youtube", "search for", "search ", "find video",
            "find me video", "find videos", "videos of", "videos on youtube",
            " on youtube", "look up on youtube",
        ],
        "tools": {"search_youtube", "open_youtube"},
    },
    "youtube_play": {
        "signals": [
            "play video", "play the video", "play result", "play the result",
            "play the first", "play the second", "play the third",
            "play the fourth", "play the fifth",
            "first result", "second result", "third result",
            "fourth result", "fifth result",
        ],
        "tools": {"play_youtube_video"},
    },

    # ===== Spotify =====

    "spotify_open": {
        "signals": ["open spotify", "launch spotify"],
        "tools": {"open_spotify"},
    },
    "spotify_play_my_playlist": {
        "signals": ["my playlist", "play my "],
        "tools": {"play_my_playlist"},
    },
    "spotify_play_playlist": {
        "signals": ["playlist"],
        "tools": {"play_spotify_playlist"},
    },
    "spotify_play_song": {
        "signals": ["play ", "put on ", "play some ", "play that "],
        "tools": {"play_spotify_song", "queue_spotify_song"},
        # Suppressed if "playlist" is present — see exclusion logic below
    },
    "spotify_queue": {
    "signals": ["queue"],
    "tools": {"queue_spotify_song", "play_spotify_song"},
    },
    "spotify_queue": {
        "signals": ["queue up", "add to the queue", "add to queue",
                    "queue this", "queue that"],
        "tools": {"queue_spotify_song"},
    },
    "spotify_list_playlists": {
        "signals": ["list my playlists", "list playlists",
                    "my playlists", "what playlists"],
        "tools": {"list_playlists"},
    },
    "spotify_playback": {
        "signals": [
            "pause", "resume", "unpause", "next track", "next song",
            "skip", "previous track", "previous song", "go back",
            "rewind", "stop the music", "stop playing",
        ],
        "tools": {
            "pause_spotify", "resume_spotify",
            "next_track", "previous_track",
        },
    },
    "spotify_clear_queue": {
        "signals": ["clear the queue", "empty the queue",
                    "clear queue", "remove from queue"],
        "tools": {"clear_queue"},
    },

    # ===== Volume =====

    "music_volume_set": {
        "signals": ["volume to", "volume "],  # catches "volume 40" and "set volume to 40"
        "tools": {"set_volume"},
    },
    "music_volume_up": {
        "signals": ["turn it up", "louder", "raise volume", "raise the volume", "volume up",
                    "turn up the volume", "crank it up"],
        "tools": {"raise_volume"},
    },
    "music_volume_down": {
    "signals": ["turn it down", "quieter", "lower volume", "volume down",
                "turn down", "lower the", "softer"],
    "tools": {"lower_volume"},
    },
    "tts_volume_set": {
        "signals": ["tts volume to", "set tts", "voice volume to"],
        "tools": {"set_tts_volume"},
    },
    "tts_volume_up": {
        "signals": ["raise tts", "tts volume up", "tts louder", "turn up tts"],
        "tools": {"raise_tts_volume"},
    },
    "tts_volume_down": {
        "signals": ["lower tts", "tts volume down", "tts quieter", "turn down tts"],
        "tools": {"lower_tts_volume"},
    },

    # ===== Voice control =====
    # Split so each branch exposes at most 2 tools.

    "voice_mute": {
        "signals": ["mute", "voice off", "be quiet", "silence",
                    "stop talking", "shut up"],
        "tools": {"mute_tts"},
    },
    "voice_unmute": {
        "signals": ["unmute", "voice on", "speak again",
                    "start talking", "unmute yourself"],
        "tools": {"unmute_tts"},
    },
    "voice_toggle": {
        "signals": ["toggle voice", "toggle mute", "flip voice"],
        "tools": {"toggle_tts"},
    },

    # ===== Apps =====

    "discord_open": {
        "signals": ["discord"],
        "tools": {"open_discord"},
    },

    # ===== Speed test =====

    "speedtest": {
        "signals": ["speed test", "speedtest", "internet speed",
                    "how fast is my internet", "bandwidth",
                    "test my connection", "test connection"],
        "tools": {"run_speed_test", "quick_speed_test"},
    },

    # ===== Memory =====

    "memory_save": {
        "signals": ["remember", "note that", "keep in mind",
                    "don't forget", "dont forget", "save this"],
        "tools": {"remember_fact"},
    },

    # ===== Personality =====

    "personality": {
        "signals": ["switch to", "change to", "become ", "activate ",
                    "switch personality", "change personality",
                    "jarvis", "faye", "computah"],
        "tools": {"change_personality"},
    },
}


# ===== Assembly rules =====

ALWAYS_INCLUDE = {"chat"}

# When no category fires, expose only chat. Small and deterministic.
FALLBACK_TOOLS = {"chat"}


def prefilter_tool_names(query: str) -> set[str]:
    q = query.lower()
    keep: set[str] = set(ALWAYS_INCLUDE)

    is_tts = any(s in q for s in ("tts", "voice volume", "speech volume"))
    has_playlist = "playlist" in q
    has_my_playlist = "my playlist" in q
    has_queue = "queue" in q
    is_search_playlist = "search" in q and ("playlist" in q or "playlists" in q)

    for name, category in CATEGORIES.items():
        if is_tts and name == "music_volume":
            continue
        if is_search_playlist and name == "youtube_search":
            continue
        if has_playlist and name == "spotify_play_song":
            continue
        if has_my_playlist and name == "spotify_play_playlist":
            continue
        if has_queue and name == "spotify_play_song":
            continue
        if any(signal in q for signal in category["signals"]):
            keep |= category["tools"]

    if keep == ALWAYS_INCLUDE:
        keep = set(FALLBACK_TOOLS)
    return keep
def prefilter_schemas(query: str, all_schemas: list[dict]) -> list[dict]:
    """Filter a list of tool schemas down to the query-relevant subset."""
    names = prefilter_tool_names(query)
    return [s for s in all_schemas if s["name"] in names]