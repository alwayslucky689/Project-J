# memory/resolve.py - Conflict resolution against existing facts
#
# Given a validated candidate, decide what to do:
#   - inserted: new fact, no conflict
#   - skipped: exact duplicate of an existing active fact
#   - superseded: correction of an existing fact
#   - candidate: conflicts with an active fact but no correction marker;
#                inserted as 'candidate' status so the user can review
from debug import debug_log as dbg
from memory.store import get_conn, write_transaction, now_iso, json_or_none
from typing import Optional

# Properties with exactly one current value. A new value for one of these
# is a genuine conflict: either a correction (supersede) or an ambiguity
# (candidate for review). Everything else accumulates — "likes pizza"
# and "likes noodles" coexist.
_SINGLE_VALUED_PROPERTIES = {
    "lives_in", "lives_at", "is_from", "moved_to",
    "works_at", "works_for", "employer", "job_title",
    "name", "age", "birthday", "timezone",
    "is_learning", "diet", "language",
    "has_partner", "partner_name",
}


def _is_single_valued(property_: str) -> bool:
    return (property_ or "").lower() in _SINGLE_VALUED_PROPERTIES
def _resolve_scope(candidate: dict) -> tuple[str, Optional[int]]:
    """
    Turn a scope_hint + session_context into the final
    (scope_type, scope_topic_id) pair to store.
    """
    hint = candidate.get("scope_hint", "unscoped")
    context = candidate.get("session_context", "global")

    topic_id = None
    if context and context.startswith("topic:"):
        try:
            topic_id = int(context.split(":", 1)[1])
        except (ValueError, IndexError):
            topic_id = None

    if hint == "global":
        return "global", None
    if hint == "topic":
        if topic_id is not None:
            return "topic", topic_id
        # No active topic — degrade gracefully
        return "unscoped", None
    return "unscoped", None
from debug import debug_log as dbg

def resolve_candidate(candidate: dict) -> str:
    scope_type, scope_topic_id = _resolve_scope(candidate)
    candidate["_scope_type"] = scope_type
    candidate["_scope_topic_id"] = scope_topic_id

    subject = candidate["subject"]
    property_ = candidate["property"]
    value = candidate["value"]
    is_correction = candidate.get("is_correction", False)

    conn = get_conn()

    # Debug line placed AFTER all variables are bound.
    dbg.event("RESOLVE", f"start: {subject}/{property_} = {value!r} "
                         f"scope={scope_type}/{scope_topic_id} "
                         f"correction={is_correction} "
                         f"single={_is_single_valued(property_)}")

    # 1. Exact duplicate check — against BOTH active and candidate rows.
    duplicate = conn.execute(
        "SELECT id, status FROM facts "
        "WHERE user_id='default' AND status IN ('active', 'candidate') "
        "AND subject = ? AND property = ? AND value = ? "
        "AND scope_type = ? AND (scope_topic_id IS ?)",
        (subject, property_, value, scope_type, scope_topic_id),
    ).fetchone()
    if duplicate:
        dbg.event("RESOLVE", f"-> skipped (duplicate of #{duplicate['id']} "
                             f"status={duplicate['status']})")
        return "skipped"

    # 2. Slot-conflict check — only for single-valued properties.
    if _is_single_valued(property_):
        existing = conn.execute(
            "SELECT id, value, text FROM facts "
            "WHERE user_id='default' AND status='active' "
            "AND subject = ? AND property = ? "
            "AND scope_type = ? AND (scope_topic_id IS ?)",
            (subject, property_, scope_type, scope_topic_id),
        ).fetchall()
        if existing:
            if is_correction:
                dbg.event("RESOLVE", f"-> superseded ({len(existing)} prior)")
                _supersede(existing, candidate)
                return "superseded"
            dbg.event("RESOLVE",
                      f"-> candidate (conflict with "
                      f"{[r['value'] for r in existing]})")
            _insert(candidate, status="candidate")
            return "candidate"

    # 3. Multi-valued prop or no conflict — insert active.
    dbg.event("RESOLVE", "-> inserted (active)")
    _insert(candidate, status=candidate.get("status", "active"))
    return "inserted"
def _insert(candidate: dict, status: str = "active") -> int:
    now = now_iso()
    scope_type = candidate.get("_scope_type", "unscoped")
    scope_topic_id = candidate.get("_scope_topic_id")
    with write_transaction() as conn:
        cur = conn.execute(
            """
            INSERT INTO facts (
                user_id, subject, property, value, text,
                source, epistemic_status, extraction_confidence,
                scope_type, scope_topic_id,
                valid_from, valid_until,
                status, evidence_turn_ids,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                candidate["subject"],
                candidate["property"],
                candidate["value"],
                candidate["text"],
                candidate.get("source", "inferred"),
                candidate.get("epistemic_status", "assertion"),
                candidate.get("extraction_confidence"),
                scope_type,
                scope_topic_id,
                None, None,
                status,
                json_or_none(candidate.get("evidence_turn_ids")),
                now, now,
            ),
        )
        fact_id = cur.lastrowid

    try:
        from memory.fact_memory import _embed_fact
        _embed_fact(fact_id, candidate["text"])
    except Exception:
        pass

    return fact_id
def _supersede(old_rows, candidate: dict):
    now = now_iso()
    scope_type = candidate.get("_scope_type", "unscoped")
    scope_topic_id = candidate.get("_scope_topic_id")
    with write_transaction() as conn:
        old_id = old_rows[0]["id"]
        cur = conn.execute(
            """
            INSERT INTO facts (
                user_id, subject, property, value, text,
                source, epistemic_status, extraction_confidence,
                scope_type, scope_topic_id,
                valid_from, valid_until,
                status, supersedes, evidence_turn_ids,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                candidate["subject"],
                candidate["property"],
                candidate["value"],
                candidate["text"],
                candidate.get("source", "inferred"),
                candidate.get("epistemic_status", "assertion"),
                candidate.get("extraction_confidence"),
                scope_type,
                scope_topic_id,
                None, None,
                "active",
                old_id,
                json_or_none(candidate.get("evidence_turn_ids")),
                now, now,
            ),
        )
        new_id = cur.lastrowid

        for row in old_rows:
            conn.execute(
                "UPDATE facts SET status='superseded', superseded_by=?, "
                "updated_at=? WHERE id=?",
                (new_id, now, row["id"]),
            )

    try:
        from memory.fact_memory import _embed_fact
        _embed_fact(new_id, candidate["text"])
    except Exception:
        pass