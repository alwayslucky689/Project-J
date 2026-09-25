# memory/extract.py - Extract candidate facts from conversation turns
#
# Pipeline per user turn:
#   1. Cheap pre-filter   — skip noise, questions, hypotheticals
#   2. spaCy dep parse    — extract subject/verb/object triples
#   4. LLM fallback       — only if GLiNER finds entities spaCy missed
#
# Assistant turns are passed as context but not extracted from.

import re
import threading
from typing import Optional
import threading


_nlp = None
_nlp_lock = threading.Lock()
_nlp_failed = False

import re

# Properties that are almost always personal facts about the user,
# independent of what project or topic is active.
_PERSONAL_PROPERTIES = {
    # identity
    "name", "age", "birthday", "timezone", "language", "location",
    "lives_in", "lives_at", "is_from", "moved_to",
    # work / study
    "works_at", "works_for", "job_title", "employer",
    "studies", "learns", "is_learning", "knows",
    # relations
    "has_sister", "has_brother", "has_parent", "has_partner",
    "has_child", "has_pet", "has",
    # preferences
    "prefers", "likes", "loves", "hates", "dislikes",
    "enjoys", "favorite",
    # possession
    "owns",
}

# Explicit phrases that force topic scope regardless of property.
_TOPIC_QUALIFIERS = (
    "for this project", "for the project", "in this project",
    "for this codebase", "in this codebase", "for the code",
    "for this app", "for this assistant",
    "on this machine", "on this pc", "on this computer",
    "in this case", "in this context",
)


def _is_personal_property(property_: str) -> bool:
    prop = (property_ or "").lower()
    return any(
        prop == p or prop.startswith(p + "_")
        for p in _PERSONAL_PROPERTIES
    )


def _infer_scope_hint(property_: str, subject: str,
                      source_text: str, has_active_topic: bool) -> str:
    """
    Return 'global', 'topic', or 'unscoped'.

    Rules (in priority order):
      1. Explicit qualifier phrase → topic if active, else unscoped
      2. Personal property → global
      3. Third-party subject → global
      4. Otherwise → topic if a topic is active, else unscoped
    """
    lower = source_text.lower()
    if any(q in lower for q in _TOPIC_QUALIFIERS):
        return "topic" if has_active_topic else "unscoped"

    if _is_personal_property(property_):
        return "global"

    subj = (subject or "").lower()
    if subj.startswith("user's "):
        return "global"

    return "topic" if has_active_topic else "unscoped"


def _context_topic_id(context: str):
    """Extract the topic id from 'topic:<id>' or None for 'global'."""
    if not context or not context.startswith("topic:"):
        return None
    try:
        return int(context.split(":", 1)[1])
    except (ValueError, IndexError):
        return None


def _get_spacy():
    global _nlp, _nlp_failed
    if _nlp is not None or _nlp_failed:
        return _nlp
    with _nlp_lock:
        if _nlp is not None or _nlp_failed:
            return _nlp
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy loaded.")
        except Exception as e:
            print(f"⚠️ spaCy unavailable: {e}")
            _nlp_failed = True
    return _nlp





# ===== Public API =====

def extract_candidates(turns: list[dict]) -> list[dict]:
    user_turns = [t for t in turns if t["role"] == "user"]
    candidates: list[dict] = []

    for turn in user_turns:
        text = turn["text"].strip()
        context = turn.get("context", "global")
        active_topic_id = _context_topic_id(context)
        has_active_topic = active_topic_id is not None

        if not _looks_memory_worthy(text):
            continue

        # Stage 2: spaCy extraction
        spacy_results = _extract_spacy(text)
        if spacy_results:
            for c in spacy_results:
                c["evidence_turn_ids"] = [turn["id"]]
                c["source_text"] = text
                c["session_context"] = context
                c["scope_hint"] = _infer_scope_hint(
                    c.get("property"), c.get("subject"),
                    text, has_active_topic,
                )
            candidates.extend(spacy_results)
            continue

        # Stage 3: LLM fallback
        llm_results = _extract_llm(text)
        for c in llm_results:
            c["evidence_turn_ids"] = [turn["id"]]
            c["source_text"] = text
            c["session_context"] = context
            c["scope_hint"] = _infer_scope_hint(
                c.get("property"), c.get("subject"),
                text, has_active_topic,
            )
        candidates.extend(llm_results)

    return candidates

# ===== Pre-filter =====

_SIGNAL_WORDS = (
    "i ", "i'm ", "i am ", "i've ", "i have ", "i'd ", "i'll ",
    "my ", "we ", "we're ", "our ",
    "prefer", "like", "love", "hate", "dislike", "enjoy",
    "work at", "work for", "live in", "live at", "study", "learning",
    "remember", "note that",
)


def _looks_memory_worthy(text: str) -> bool:
    """Cheap gate: does this sentence plausibly contain a fact about the user?"""
    if len(text) < 8:
        return False
    if text.rstrip().endswith("?"):
        return False
    lower = text.lower()
    if any(lower.startswith(w) for w in ("what ", "why ", "how ", "when ", "where ", "who ")):
        return False
    # Hypotheticals
    if re.search(r"\b(if i|if we|if you|would have|could have|might have been)\b", lower):
        return False
    return any(s in lower for s in _SIGNAL_WORDS)


# ===== Stage 2: spaCy extraction =====

_PRONOUN_USER = {"i", "me", "we", "us"}


def _extract_spacy(text: str) -> list[dict]:
    nlp = _get_spacy()
    if nlp is None:
        return []

    doc = nlp(text)
    results = []

    for sent in doc.sents:
        candidate = _extract_from_sentence(sent)
        if candidate:
            results.append(candidate)

    return results


def _extract_from_sentence(sent) -> Optional[dict]:
    # Find root verb
    root = next((t for t in sent if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ != "VERB":
        return None

    # Find subject
    subj = next(
        (t for t in root.children if t.dep_ in ("nsubj", "nsubjpass")),
        None,
    )
    if subj is None:
        return None

    subject = _resolve_subject(subj)
    if subject is None:
        return None

    # Find object: direct, prepositional, or attribute
    verb_lemma = root.lemma_.lower()
    prep = None
    obj = None

    for child in root.children:
        if child.dep_ in ("dobj", "obj"):
            obj = child
            break
        if child.dep_ == "prep":
            pobj = next(
                (t for t in child.children if t.dep_ == "pobj"),
                None,
            )
            if pobj is not None:
                prep = child.lemma_.lower()
                obj = pobj
                break

    if obj is None:
        for child in root.children:
            if child.dep_ in ("attr", "acomp", "oprd"):
                obj = child
                break

    if obj is None:
        return None

    value = _span_with_modifiers(obj)
    if not value or len(value) < 2:
        return None

    # Compose property name from verb + prep
    prop = f"{verb_lemma}_{prep}" if prep else verb_lemma

    # Natural-language representation
    fact_text = _render_fact_text(subject, root.lemma_, value, prep)

    return {
        "text": fact_text,
        "subject": subject,
        "property": prop,
        "value": value,
        "source": "inferred",
        "epistemic_status": _classify_epistemic(sent.text),
        "extraction_confidence": 0.8,
        "scope_type": "unscoped",
    }


def _resolve_subject(token) -> Optional[str]:
    """Return 'user', 'user's <relation>', or None if not about the user."""
    text = token.text.lower()
    if text in _PRONOUN_USER:
        return "user"

    # Possessive relations: "my brother", "our team"
    for child in token.children:
        if child.dep_ == "poss" and child.text.lower() in ("my", "our"):
            return f"user's {token.lemma_}"

    # Check if preceded by possessive determiner (some spaCy configs)
    if token.i > 0:
        prev = token.doc[token.i - 1].text.lower()
        if prev in ("my", "our"):
            return f"user's {token.lemma_}"

    return None


def _span_with_modifiers(token) -> str:
    """Get a token's text span including determiner and modifiers."""
    parts = []
    for child in token.children:
        if child.dep_ in ("det", "amod", "compound", "nummod", "quantmod"):
            parts.append((child.i, child.text))
    parts.append((token.i, token.text))
    parts.sort()
    text = " ".join(p[1] for p in parts)
    # Drop leading determiners for cleanliness
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


def _render_fact_text(subject, verb_lemma, value, prep) -> str:
    if subject == "user":
        subj_phrase = "The user"
    else:
        subj_phrase = f"The user's {subject.split(' ', 1)[1]}" \
            if " " in subject else subject

    if prep:
        verb_phrase = f"{verb_lemma}s {prep}"
    else:
        verb_phrase = f"{verb_lemma}s"

    return f"{subj_phrase} {verb_phrase} {value}."



# ===== Stage 4: LLM fallback =====

def _extract_llm(text: str) -> list[dict]:
    """Structured extraction via the fast model. Only called for hard cases."""
    try:
        from config import settings
        from core.llm import ask_ollama
    except ImportError:
        return []

    prompt = f"""Extract facts about the speaker from the text below.

Return a JSON array. Each fact must have exactly these fields:
  "subject": "user" or "user's <relation>" (e.g. "user's brother")
  "property": short snake_case identifier (e.g. "employer", "prefers", "lives_in")
  "value": the object of the fact (e.g. "Acme", "dark mode", "Berlin")
  "text": natural-language rendering (e.g. "The user works at Acme.")
  "epistemic_status": one of "assertion" | "plan" | "observation"

Rules:
- Only facts about the speaker or the speaker's relations.
- Skip hypotheticals ("if I...", "I might..." with no commitment).
- Skip questions.
- Preserve uncertainty: "might move" → "plan" not "assertion".
- Return [] if nothing to extract.

Text: {text}

JSON array:"""

    try:
        result = ask_ollama(prompt, is_json=False, model=settings.FAST_MODEL)
    except Exception:
        return []

    # The model may return prose; extract the first JSON array
    import json
    m = re.search(r"\[.*\]", result, re.DOTALL)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []

    candidates = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if not all(k in item for k in ("subject", "property", "value", "text")):
            continue
        candidates.append({
            "text": str(item["text"]),
            "subject": str(item["subject"]),
            "property": str(item["property"]),
            "value": str(item["value"]),
            "source": "inferred",
            "epistemic_status": item.get("epistemic_status", "assertion"),
            "extraction_confidence": 0.6,
            "scope_type": "unscoped",
        })
    return candidates


# ===== Epistemic classification =====

_PLAN_MARKERS = (
    "might ", "may ", "maybe ", "could ", "perhaps ",
    "plan to", "planning to", "going to", "intend to",
    "will ", "i'll ", "thinking about", "considering",
)
_HYPOTHETICAL_MARKERS = (
    "if i ", "if we ", "if you ",
    "would ", "could have", "might have",
)


def _classify_epistemic(sentence: str) -> str:
    lower = sentence.lower()
    if any(m in lower for m in _HYPOTHETICAL_MARKERS):
        return "hypothetical"
    if any(m in lower for m in _PLAN_MARKERS):
        return "plan"
    return "assertion"
