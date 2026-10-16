# memory/classifier.py - Cheap prefilter: does this query need memory lookup?
#
# Union of two signals:
#   1. Explicit recall phrasing ("do you remember", "what did I say")
#   2. Decision-shaped or preference-shaped questions ("should we",
#      "which one do I", "do I prefer") — these plausibly have a stored
#      answer even without a recall verb
#
# No semantic gate. BGE-small clusters short sentences tightly; with a
# small fact corpus every query is within any reasonable distance
# threshold, so the gate fires on everything. Revisit only with a
# larger corpus and a re-tuned threshold, or replace with an LLM
# classifier that decides on semantics instead of distance.

import re

_RECALL_PATTERNS = [
    r"\bdo you remember\b",
    r"\bdo you know\b",
    r"\bdid i (tell|mention|say|ever)\b",
    r"\bhave i (told|mentioned|said)\b",
    r"\bremind me\b",
    r"\bwhat did i\b",
    r"\bwhat did we\b",
    r"\bwhat have i\b",
    r"\bwhat have we\b",
    r"\bwhen did i\b",
    r"\bwhen did we\b",
    r"\bwho is my\b",
    r"\bwho's my\b",
    r"\bwhat('s| is| are) my\b",
    r"\blast time\b",
    r"\bpreviously\b",
    r"\bbefore i\b",
    r"\byou know about me\b",
    r"\bknow about me\b",
    r"\btell me about my\b",
    r"\bwhat do you know\b",
    r"\bremember that\b",
    r"\bnote that\b",
    r"\bkeep in mind\b",
    r"\bdon't forget\b",
    r"\bdont forget\b",
    # Decision-shaped. The user is asking for advice or choosing between
    # options — likely referencing a stated preference or constraint.
    r"\bshould (i|we)\b",
    r"\bwhich (one|do i|should i|would i)\b",
    r"\bdo i (prefer|like|want|need|hate)\b",
    r"\bmy (preference|opinion|stance)\b",
    r"\bdo you think i\b",
    r"\bwould i (prefer|like|want)\b",
]

_RECALL_RE = re.compile("|".join(_RECALL_PATTERNS), re.IGNORECASE)


def needs_memory_lookup(query: str,
                        topic_name: str | None = None,
                        topic_aliases: list[str] | None = None) -> bool:
    """
    Return True if the chat path should expose search_memory as a tool.

    This is a GATE, not a decision. The model still chooses whether to
    call the tool. We deliberately bias toward False here because every
    True costs an extra non-streaming round-trip. False negatives mean
    the user has to phrase a recall question more explicitly; True
    positives cost ~1.5–2s of latency on every turn.

    When a natural query that should have hit memory doesn't, add a
    pattern. That's the tuning loop.
    """
    if not query or not query.strip():
        return False

    if _RECALL_RE.search(query):
        return True

    if topic_name and re.search(rf"\b{re.escape(topic_name)}\b",
                                query, re.IGNORECASE):
        return True
    for alias in (topic_aliases or []):
        if alias and re.search(rf"\b{re.escape(alias)}\b",
                               query, re.IGNORECASE):
            return True

    return False