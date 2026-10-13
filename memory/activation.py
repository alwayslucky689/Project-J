# memory/activation.py - Natural-language topic activation (Phase 6n)
#
# Detects explicit activation intent ("switch to X", "bring up X") and
# switches the topic. Bare mentions never activate — the intent phrase is
# required.
#
# Ambiguity: when a candidate matches multiple topics, the module stores
# the options and returns an "ambiguous" outcome. The next turn either
# resolves it (bare name/alias/index, or cancel) or drops it and lets
# normal routing proceed.

import re
import time
from difflib import SequenceMatcher
from typing import Optional

from memory.topics import list_topics, activate_topic


# ===== Intent phrases =====

INTENT_PHRASES = [
    "let's work on",
    "lets work on",
    "open the topic",
    "switch to",
    "bring up",
    "pull up",
    "jump to",
    "open topic",
]

# Longer phrases first so "let's work on" wins over bare "work on".
_INTENT_SORTED = sorted(INTENT_PHRASES, key=len, reverse=True)


# ===== Tunables =====

FUZZY_THRESHOLD = 0.75    # below this, no match → normal routing
AMBIGUITY_TIE_GAP = 0.02  # topics within this of the top score are "tied"
PENDING_TTL_SECONDS = 90  # how long an ambiguity prompt stays live


# ===== Pending ambiguity state =====

_pending: Optional[dict] = None
# Shape: {"options": [topic_dict, ...], "asked_at": float}


def clear_pending():
    global _pending
    _pending = None


# ===== Helpers =====

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _topic_strings(topic: dict) -> list[str]:
    """Every string a topic can be referred to by: name + aliases."""
    strings = [topic["name"]]
    aliases = topic.get("aliases") or []
    strings.extend(aliases)
    return strings


def _best_score(norm: str, topic: dict) -> float:
    return max(
        SequenceMatcher(None, norm, _normalize(s)).ratio()
        for s in _topic_strings(topic)
    )


# ===== Intent detection + name extraction =====

def _extract_topic_name(query: str) -> Optional[tuple[str, str]]:
    """
    Find an intent phrase and return (intent, candidate_name), or None.

    The intent phrase must be word-bounded so "ego to" doesn't match
    "go to". The candidate is the tail of the string, stripped of
    punctuation and a leading "the".
    """
    lower = query.lower()
    for phrase in _INTENT_SORTED:
        start = 0
        while True:
            idx = lower.find(phrase, start)
            if idx == -1:
                break
            # Word-boundary check: char before must not be alphanumeric,
            # char after must not be alphanumeric.
            before_ok = idx == 0 or not lower[idx - 1].isalnum()
            end = idx + len(phrase)
            after_ok = end == len(lower) or not lower[end].isalnum()

            if before_ok and after_ok:
                tail = query[end:].strip()
                tail = tail.strip(".,!?;:'\"")
                if tail.lower().startswith("the "):
                    tail = tail[4:].strip()
                if tail:
                    return phrase, tail

            start = idx + 1
    return None


# ===== Matching =====

def _find_matches(candidate: str) -> list[dict]:
    """
    Exact name → exact alias → fuzzy. Ambiguity is only possible at the
    fuzzy tier, since create_topic prevents duplicate names/aliases.
    """
    topics = list_topics()
    norm = _normalize(candidate)

    exact_name = [t for t in topics if _normalize(t["name"]) == norm]
    if exact_name:
        return exact_name[:1]

    exact_alias = [
        t for t in topics
        if any(_normalize(a) == norm for a in (t.get("aliases") or []))
    ]
    if exact_alias:
        return exact_alias[:1]

    scored = [(t, _best_score(norm, t)) for t in topics]
    scored = [(t, s) for t, s in scored if s >= FUZZY_THRESHOLD]
    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)
    top_score = scored[0][1]
    return [t for t, s in scored if s >= top_score - AMBIGUITY_TIE_GAP]


# ===== Ambiguity resolution =====

def _check_pending_resolution(query: str) -> Optional[dict]:
    """
    If there's a pending ambiguity prompt and the query resolves it,
    return the chosen topic. Otherwise clear pending and return None.
    """
    global _pending
    if _pending is None:
        return None
    if time.time() - _pending["asked_at"] > PENDING_TTL_SECONDS:
        _pending = None
        return None

    norm = _normalize(query.strip().strip(".,!?"))
    options = _pending["options"]

    # Cancel words drop the pending state.
    if norm in ("cancel", "nevermind", "never mind", "no", "nope",
                "forget it", "none"):
        _pending = None
        return None

    # Numeric selection: "1", "2", ...
    if norm.isdigit():
        idx = int(norm) - 1
        if 0 <= idx < len(options):
            chosen = options[idx]
            _pending = None
            return chosen
        _pending = None
        return None

    # Name or alias selection
    for t in options:
        if any(_normalize(s) == norm for s in _topic_strings(t)):
            _pending = None
            return t

    # Anything else — drop and let normal routing handle it.
    _pending = None
    return None


# ===== Public API =====

def try_activate(query: str) -> Optional[dict]:
    """
    Attempt topic activation from a natural-language input.

    Returns:
      {"action": "activated", "topic": dict, "session_id": int, "is_new": bool}
      {"action": "ambiguous", "options": [dict, ...]}
      None if no activation was attempted (caller should route normally).
    """
    global _pending

    # 1. Resolve a live ambiguity prompt first.
    resolved = _check_pending_resolution(query)
    if resolved is not None:
        topic, sid, is_new = activate_topic(resolved["id"])
        if topic is None:
            return None
        return {"action": "activated", "topic": topic,
                "session_id": sid, "is_new": is_new}

    # 2. Detect intent + candidate name.
    extracted = _extract_topic_name(query)
    if extracted is None:
        return None
    _intent, candidate = extracted

    # 3. Match against known topics.
    matches = _find_matches(candidate)
    if not matches:
        return None

    if len(matches) == 1:
        topic, sid, is_new = activate_topic(matches[0]["id"])
        if topic is None:
            return None
        return {"action": "activated", "topic": topic,
                "session_id": sid, "is_new": is_new}

    # 4. Ambiguous — store and ask.
    _pending = {"options": matches, "asked_at": time.time()}
    return {"action": "ambiguous", "options": matches}


def suggest(query: str) -> list[tuple[dict, float]]:
    """
    For the `topic suggest` debug command. Returns (topic, score) pairs
    sorted by descending score. Score is fuzzy-only, regardless of the
    activation threshold.
    """
    norm = _normalize(query)
    scored = [(t, _best_score(norm, t)) for t in list_topics()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored