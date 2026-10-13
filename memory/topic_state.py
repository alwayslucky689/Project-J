# memory/topic_state.py - Structured topic state and its merge path
#
# Phase 6k. A topic's durable memory is a structured JSON object:
#
#   {
#     "objective":         str,
#     "active_decisions":  [str, ...],
#     "open_questions":    [str, ...],
#     "updated_at":        ISO8601
#   }
#
# It replaces free-text topics.summary as the primary topic memory. The
# summary column stays for display but is no longer written here.
#
# One LLM call at session close merges the session summary into the topic
# state. The merge is incremental — the LLM sees the existing state plus
# the new summary and produces an updated state. The content is
# *rebuildable* from facts + summaries; a rebuild path is sketched at the
# bottom for future use.

import json
import re
from typing import Optional

from memory.store import get_conn, write_transaction, now_iso


EMPTY_STATE = {
    "objective": "",
    "active_decisions": [],
    "open_questions": [],
    "updated_at": None,
}

MAX_ITEMS_PER_LIST = 10


# ===== Parse / serialize =====

def parse_state(raw) -> dict:
    """Parse a topics.state value. Always returns a full dict."""
    if not raw:
        return dict(EMPTY_STATE)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return dict(EMPTY_STATE)
    if not isinstance(data, dict):
        return dict(EMPTY_STATE)

    return {
        "objective": str(data.get("objective") or ""),
        "active_decisions": _coerce_str_list(data.get("active_decisions")),
        "open_questions": _coerce_str_list(data.get("open_questions")),
        "updated_at": data.get("updated_at"),
    }


def _coerce_str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out[:MAX_ITEMS_PER_LIST]


def serialize_state(state: dict) -> str:
    return json.dumps(state, ensure_ascii=False)


# ===== Read / write =====

def get_state(topic_id: int) -> dict:
    row = get_conn().execute(
        "SELECT state FROM topics WHERE id = ?", (topic_id,)
    ).fetchone()
    if not row:
        return dict(EMPTY_STATE)
    return parse_state(row["state"])


def set_state(topic_id: int, state: dict) -> None:
    """Write state directly, stamping updated_at. No LLM."""
    state = dict(state)
    state["updated_at"] = now_iso()
    with write_transaction() as conn:
        conn.execute(
            "UPDATE topics SET state = ?, summary_updated_at = ? WHERE id = ?",
            (serialize_state(state), now_iso(), topic_id),
        )


# ===== Merge =====

def merge_session_into_state(topic_id: int,
                             session_summary: str) -> Optional[dict]:
    """Merge one session's summary into a topic's structured state.

    Returns the new state dict on success, or None if the merge failed
    (in which case the existing state is left untouched).
    """
    if not session_summary or not session_summary.strip():
        return None

    row = get_conn().execute(
        "SELECT name, description FROM topics WHERE id = ?", (topic_id,)
    ).fetchone()
    if not row:
        return None

    current = get_state(topic_id)
    prompt = _build_merge_prompt(
        topic_name=row["name"],
        topic_description=row["description"] or "",
        current_state=current,
        session_summary=session_summary.strip(),
    )

    new_state = _call_merge_llm(prompt)
    if new_state is None:
        return None

    # Safety net: never let a dropped objective wipe a real one.
    if not new_state["objective"] and current["objective"]:
        new_state["objective"] = current["objective"]

    set_state(topic_id, new_state)
    return new_state


def _call_merge_llm(prompt: str) -> Optional[dict]:
    try:
        from config import settings
        from core.llm import ask_ollama
    except ImportError:
        return None

    try:
        result = ask_ollama(prompt, is_json=False, model=settings.FAST_MODEL)
    except Exception as e:
        print(f"⚠️ Topic state merge LLM call failed: {e}")
        return None

    parsed = _extract_json_object(result)
    if parsed is None:
        return None

    return {
        "objective": str(parsed.get("objective") or "").strip(),
        "active_decisions": _coerce_str_list(parsed.get("active_decisions")),
        "open_questions": _coerce_str_list(parsed.get("open_questions")),
        "updated_at": None,  # stamped by set_state
    }


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _build_merge_prompt(topic_name: str, topic_description: str,
                        current_state: dict, session_summary: str) -> str:
    current_json = json.dumps({
        "objective": current_state.get("objective", ""),
        "active_decisions": current_state.get("active_decisions", []),
        "open_questions": current_state.get("open_questions", []),
    }, ensure_ascii=False, indent=2)

    desc_line = f"\nDescription: {topic_description}\n" if topic_description else ""

    return f"""You maintain a structured state document for a long-running topic.

Topic: {topic_name}
{desc_line}
Current state:
{current_json}

New session summary:
{session_summary}

Produce the updated state as a JSON object with exactly these three fields:
  "objective":        one sentence describing what this topic is about.
  "active_decisions": array of decisions currently in effect.
  "open_questions":   array of questions not yet resolved.

Rules:
- Keep the objective stable unless the session clearly changes it.
- Add decisions introduced in this session.
- Remove decisions that were reversed or explicitly abandoned.
- Move questions that got answered out of open_questions.
- Cap each list at {MAX_ITEMS_PER_LIST} items. Prefer the most recent and
  most important.
- Every item is a short standalone sentence.
- Return ONLY the JSON object. No prose, no markdown, no code fences.

JSON:"""


# ===== Formatting for injection =====

def format_state_block(state: dict) -> Optional[str]:
    """Render a state dict for retrieval injection. None if empty."""
    if not state:
        return None
    objective = (state.get("objective") or "").strip()
    decisions = state.get("active_decisions") or []
    questions = state.get("open_questions") or []
    if not objective and not decisions and not questions:
        return None

    lines = []
    if objective:
        lines.append(f"Objective: {objective}")
    if decisions:
        lines.append("Decisions:")
        for d in decisions:
            lines.append(f"  - {d}")
    if questions:
        lines.append("Open questions:")
        for q in questions:
            lines.append(f"  - {q}")
    return "\n".join(lines)


# ===== Rebuild (sketch, not wired up yet) =====
#
# def rebuild_state(topic_id: int) -> Optional[dict]:
#     """Regenerate state from scratch across all of a topic's session
#     summaries. Expensive (one LLM call per summary, or one batched call).
#     Kept as a stub so 6l / a future repair path has an anchor."""
#     raise NotImplementedError