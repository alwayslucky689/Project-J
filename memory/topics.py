# memory/topics.py - Topic CRUD and activation
#
# Topics are explicit named contexts. Users switch into one with the
# `topic <name>` command; the assistant switches its session and buffer
# to that topic's context. Sessions belong to exactly one context
# permanently (established in 6g).
#
# For 6i: retrieval is still global. Topic scoping of facts and sessions
# lands in 6j.

import json
import re
from datetime import datetime
from typing import Optional
import sqlite3
from memory.store import get_conn, write_transaction, now_iso
from typing import Optional

# Context strings used to identify sessions by owner.
def topic_context(topic_id: int) -> str:
    return f"topic:{topic_id}"


GLOBAL_CONTEXT = "global"


# ===== Lookup helpers =====

def _normalize(name: str) -> str:
    """Case-insensitive normalization for name/alias matching."""
    return re.sub(r"\s+", " ", name.strip().lower())


# ===== CRUD =====

def create_topic(name: str,
                 description: Optional[str] = None,
                 aliases: Optional[list[str]] = None,
                 user_id: str = "default") -> dict:
    """
    Create a new topic. Raises ValueError if the name already exists.

    Name is stored as given (preserves casing for display); matching
    against existing topics uses case-insensitive normalization.
    """
    name = name.strip()
    if not name:
        raise ValueError("Topic name cannot be empty")

    normalized = _normalize(name)

    # Check for existing name or alias collision
    existing = _find_by_normalized(normalized, user_id)
    if existing:
        raise ValueError(
            f"A topic already matches '{name}' (existing: '{existing['name']}')"
        )

    now = now_iso()
    aliases_json = json.dumps(aliases) if aliases else None

    with write_transaction() as conn:
        cur = conn.execute(
            """
            INSERT INTO topics (
                user_id, name, aliases, description,
                status, created_at, last_active_at, created_from
            ) VALUES (?, ?, ?, ?, 'active', ?, ?, 'explicit')
            """,
            (user_id, name, aliases_json, description, now, now),
        )
        tid = cur.lastrowid

    return get_topic(tid)


def get_topic(identifier) -> Optional[dict]:
    """
    Retrieve a topic by id (int), name (str), or alias (str).
    Case-insensitive for strings.
    """
    conn = get_conn()
    if isinstance(identifier, int):
        row = conn.execute(
            "SELECT * FROM topics WHERE id = ?", (identifier,)
        ).fetchone()
    else:
        row = _find_by_normalized(_normalize(str(identifier)))
    if not row:
        return None
    return _row_to_dict(row)


def list_topics(include_archived: bool = False,
                user_id: str = "default") -> list[dict]:
    conn = get_conn()
    where = "" if include_archived else "AND status = 'active'"
    rows = conn.execute(
        f"""
        SELECT * FROM topics
        WHERE user_id = ? {where}
        ORDER BY last_active_at DESC
        """,
        (user_id,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def delete_topic(identifier, user_id: str = "default") -> str:
    """
    Archive a topic. Does not delete sessions, turns, or facts yet —
    cascading deletion lands in 6l. For now, marking status='archived'
    removes it from the active list.

    Refuses to archive a topic with an open session.
    """
    topic = get_topic(identifier)
    if not topic:
        return f"❌ No topic matching {identifier!r}"

    # Check for open sessions
    conn = get_conn()
    open_session = conn.execute(
        "SELECT id FROM sessions "
        "WHERE context = ? AND ended_at IS NULL LIMIT 1",
        (topic_context(topic["id"]),),
    ).fetchone()
    if open_session:
        return (f"❌ Topic '{topic['name']}' has an open session "
                f"({open_session['id']}). Switch away from it first.")

    with write_transaction() as conn:
        conn.execute(
            "UPDATE topics SET status = 'archived' WHERE id = ?",
            (topic["id"],),
        )
    return f"🗄️  Archived topic: {topic['name']}"


# ===== Activation =====

def activate_topic(identifier, user_id: str = "default"
                   ) -> tuple[Optional[dict], Optional[int], bool]:
    """
    Switch to the topic matching `identifier`.

    Returns (topic_dict, session_id, is_new_session).
    Returns (None, None, False) if no topic matches.

    This is the entry point used by the `topic <name>` command. It calls
    switch_context from memory.sessions, which suspends the current
    context's session and resumes or creates one for the new context.
    """
    from memory.sessions import switch_context

    topic = get_topic(identifier)
    if not topic:
        return None, None, False

    # Touch last_active_at
    with write_transaction() as conn:
        conn.execute(
            "UPDATE topics SET last_active_at = ? WHERE id = ?",
            (now_iso(), topic["id"]),
        )

    ctx = topic_context(topic["id"])
    sid, is_new = switch_context(ctx)

    return topic, sid, is_new


def leave_topic() -> tuple[int, bool]:
    """
    Return to the global context. Returns (session_id, is_new_session).
    """
    from memory.sessions import switch_context
    return switch_context(GLOBAL_CONTEXT)


def current_topic() -> Optional[dict]:
    """Return the topic dict for the current context, or None if global."""
    from memory.sessions import get_current_context
    ctx = get_current_context()
    if not ctx.startswith("topic:"):
        return None
    try:
        tid = int(ctx.split(":", 1)[1])
    except (ValueError, IndexError):
        return None
    return get_topic(tid)


# ===== Description / summary =====

def set_topic_summary(topic_id: int, summary: str):
    with write_transaction() as conn:
        conn.execute(
            "UPDATE topics SET summary = ?, summary_updated_at = ? "
            "WHERE id = ?",
            (summary, now_iso(), topic_id),
        )


# ===== Internal =====
# sqlite3.Row is the row factory, but we don't import sqlite3 directly.
# The type annotation is only for IDE support.
sqlite3_Row = object
def _find_by_normalized(normalized: str,
                        user_id: str = "default") -> Optional[sqlite3_Row]:
    """
    Match a normalized string against topics by name (case-insensitive)
    or any alias. Only matches active topics.
    """
    conn = get_conn()

    # Try exact name match first
    row = conn.execute(
        """
        SELECT * FROM topics
        WHERE user_id = ? AND status = 'active' AND LOWER(name) = ?
        LIMIT 1
        """,
        (user_id, normalized),
    ).fetchone()
    if row:
        return row

    # Fall through to alias scan (aliases stored as JSON array)
    rows = conn.execute(
        "SELECT * FROM topics "
        "WHERE user_id = ? AND status = 'active' AND aliases IS NOT NULL",
        (user_id,),
    ).fetchall()
    for r in rows:
        try:
            aliases = json.loads(r["aliases"] or "[]")
        except (json.JSONDecodeError, TypeError):
            continue
        for alias in aliases:
            if _normalize(str(alias)) == normalized:
                return r
    return None


def _row_to_dict(row) -> dict:
    d = dict(row)
    # Parse aliases into a list for convenience
    if d.get("aliases"):
        try:
            d["aliases"] = json.loads(d["aliases"])
        except (json.JSONDecodeError, TypeError):
            d["aliases"] = []
    else:
        d["aliases"] = []
    return d


