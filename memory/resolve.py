# memory/resolve.py - Conflict resolution against existing facts
#
# Given a validated candidate, decide what to do:
#   - inserted: new fact, no conflict
#   - skipped: exact duplicate of an existing active fact
#   - superseded: correction of an existing fact
#   - candidate: conflicts with an active fact but no correction marker;
#                inserted as 'candidate' status so the user can review

from memory.store import get_conn, write_transaction, now_iso, json_or_none
from typing import Optional


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


def resolve_candidate(candidate: dict) -> str:
    scope_type, scope_topic_id = _resolve_scope(candidate)
    candidate["_scope_type"] = scope_type
    candidate["_scope_topic_id"] = scope_topic_id

    subject = candidate["subject"]
    property_ = candidate["property"]
    value = candidate["value"]
    is_correction = candidate.get("is_correction", False)

    conn = get_conn()

    # Existing active facts with same (subject, property, scope_type, scope_topic_id)
    existing = conn.execute(
        "SELECT id, value, text FROM facts "
        "WHERE user_id='default' AND status='active' "
        "AND subject = ? AND property = ? "
        "AND scope_type = ? "
        "AND (scope_topic_id IS ?)",
        (subject, property_, scope_type, scope_topic_id),
    ).fetchall()

    if not existing:
        _insert(candidate, status=candidate.get("status", "active"))
        return "inserted"

    for row in existing:
        if row["value"] == value:
            return "skipped"

    if is_correction:
        _supersede(existing, candidate)
        return "superseded"

    _insert(candidate, status="candidate")
    return "candidate"


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