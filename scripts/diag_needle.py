"""
Diagnose Needle's failure mode: is it the retrieval head, the selector
head, or the descriptions?
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import needle
import tools  # noqa: F401
from core.registry import list_tools
from core.tool_descriptions import DESCRIPTIONS


def build_all_schemas():
    return [
        {
            "name": t.name,
            "description": DESCRIPTIONS.get(t.name, t.description),
            "parameters": {"type": "object", "properties": {}},
        }
        for t in list_tools()
    ]


def subset(names):
    all_schemas = build_all_schemas()
    return [s for s in all_schemas if s["name"] in names]


def try_query(query, tool_names):
    agent = needle.Needle(tools=subset(tool_names))
    r = agent.complete(query)
    calls = r.get("function_calls") or []
    if not calls:
        return None, r.get("confidence", 0.0)
    return calls[0].get("name"), r.get("confidence", 0.0)


# ---- Test 1: full tool list ----
print("=== FULL TOOL LIST ===")
print(f"  {len(list_tools())} tools loaded")
for q in ["play bohemian rhapsody", "what is 2+2", "open discord"]:
    name, conf = try_query(q, [t.name for t in list_tools()])
    print(f"  {q!r:40} -> {name} ({conf:.3f})")


# ---- Test 2: YouTube only ----
print("\n=== YOUTUBE + CHAT ONLY ===")
yt_tools = ["open_youtube", "search_youtube", "play_youtube_video", "chat"]
for q in ["find me videos of puppies on youtube",
          "play video 1",
          "what is 2+2",
          "play bohemian rhapsody"]:
    name, conf = try_query(q, yt_tools)
    print(f"  {q!r:40} -> {name} ({conf:.3f})")


# ---- Test 3: Spotify only ----
print("\n=== SPOTIFY + CHAT ONLY ===")
sp_tools = ["play_spotify_song", "queue_spotify_song",
            "play_spotify_playlist", "play_my_playlist", "chat"]
for q in ["play bohemian rhapsody",
          "queue up stairway to heaven",
          "play my playlist focus",
          "what is 2+2"]:
    name, conf = try_query(q, sp_tools)
    print(f"  {q!r:40} -> {name} ({conf:.3f})")


# ---- Test 4: order matters? ----
print("\n=== ORDER SENSITIVITY (Spotify tools, reversed order) ===")
sp_reversed = list(reversed(sp_tools))
for q in ["play bohemian rhapsody", "play my playlist focus"]:
    name, conf = try_query(q, sp_reversed)
    print(f"  {q!r:40} -> {name} ({conf:.3f})")
# ---- Test 5: full list + prefilter ----
print("\n=== FULL LIST + KEYWORD PREFILTER ===")
from core.routing import prefilter_schemas as pf
all_schemas = build_all_schemas()

test_queries = [
    "play bohemian rhapsody",
    "what is 2+2",
    "open discord",
    "find me videos of puppies on youtube",
    "play video 1",
    "remember that I like pizza",
    "what is the capital of France",
    "turn it up",
    "voice off",
]
for q in test_queries:
    filtered = pf(q, all_schemas)
    names = [s["name"] for s in filtered]
    agent = needle.Needle(tools=filtered)
    r = agent.complete(q)
    calls = r.get("function_calls") or []
    pick = calls[0].get("name") if calls else "chat"
    conf = r.get("confidence", 0.0)
    print(f"  {q!r:42}")
    print(f"    candidates: {names}")
    print(f"    -> {pick} ({conf:.3f})")