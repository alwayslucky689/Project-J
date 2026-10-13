# tools/memory_tools.py - Memory search exposed as a tool for the chat path
#
# Two consumers:
#   1. The tool router (register() below) — for explicit "search my memory"
#   2. The chat path in assistant.py — passes SEARCH_MEMORY_SCHEMA to
#      chat_with_tools() so the model can call it natively.

from core.registry import register, Tool, Permission


SEARCH_MEMORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_memory",
        "description": (
            "Search the user's long-term memory for facts and past "
            "conversation summaries relevant to a query. Use this when "
            "the user asks about something they told you before, references "
            "a prior conversation, asks what you know about them, or asks "
            "you to recall a decision, preference, or event. Returns the "
            "most relevant records first. Do NOT use for general knowledge "
            "questions you can answer from training (e.g. 'what is 2+2', "
            "'who was Ada Lovelace')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Prefer the user's own phrasing "
                        "over a rewrite — it's what the embedding model "
                        "was tuned against."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


def search_memory(query: str) -> str:
    """
    Retrieve memory records for a query and format them as a safe block.
    Returns a human-readable string, suitable for a tool result message.
    """
    if not query or not query.strip():
        return "(no query provided)"

    try:
        from memory.retrieval import assemble_memory
        from memory.inject import format_memory_block
        from memory.sessions import get_current_session_id
    except ImportError as e:
        return f"(memory retrieval unavailable: {e})"

    sid = get_current_session_id()
    try:
        records = assemble_memory(query, sid)
    except Exception as e:
        return f"(memory retrieval failed: {e})"

    if not records:
        return "(no relevant memories found)"

    block = format_memory_block(records)
    return block or "(no relevant memories found)"


register(Tool(
    name="search_memory",
    description=(
        'Searches the user\'s long-term memory for facts and past '
        'conversations. Use when the user asks about something they told '
        'you before, or references a prior conversation. '
        'Takes "query" (string).'
    ),
    handler=search_memory,
    formatter=lambda s: s,
    permission=Permission.SAFE,
    response_key="memory_searched",
))