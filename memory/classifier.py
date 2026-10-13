# memory/classifier.py - Cheap prefilter: does this query need memory lookup?
#
# Determines whether the chat path should expose `search_memory` as a tool.
# This is a GATE, not a decision — the LLM still decides whether to call
# the tool. Bias toward True: a false positive costs one extra LLM
# round-trip; a false negative means memory is never consulted.
#
# This is not the tool router. The handoff's "no hardcoded phrases in
# routing" rule applies to routing decisions, not to a keyword gate whose
# only job is to enable an optional capability.

import re


# Phrases that mean "the user is asking me to recall something."
_RECALL_PATTERNS = [
    # Explicit probes
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
    # Possessive recall: "what's my X", "what is my X", "what are my X"
    r"\bwhat('s| is| are) my\b",
    # Prior-reference
    r"\blast time\b",
    r"\bpreviously\b",
    r"\bbefore i\b",
    r"\byou know about me\b",
    r"\bknow about me\b",
    r"\btell me about my\b",
    r"\bwhat do you know\b",
    # Explicit save (still routes through the model, which may just
    # acknowledge — but it needs the tool available to write)
    r"\bremember that\b",
    r"\bnote that\b",
    r"\bkeep in mind\b",
    r"\bdon't forget\b",
    r"\bdont forget\b",
]

_RECALL_RE = re.compile("|".join(_RECALL_PATTERNS), re.IGNORECASE)


def needs_memory_lookup(query: str,
                        topic_name: str | None = None,
                        topic_aliases: list[str] | None = None) -> bool:
    """
    Return True if the query plausibly needs memory retrieval.

    topic_name / topic_aliases are optional context: if the current topic
    is mentioned by name, that's a strong signal.
    """
    if not query or not query.strip():
        return False

    if _RECALL_RE.search(query):
        return True

    # Topic name or alias mentioned in the query
    if topic_name:
        if re.search(rf"\b{re.escape(topic_name)}\b", query, re.IGNORECASE):
            return True
    for alias in (topic_aliases or []):
        if alias and re.search(rf"\b{re.escape(alias)}\b", query, re.IGNORECASE):
            return True

    return False