# memory/sessions.py - Session lifecycle and turn persistence
#
# A session belongs to exactly one context ('global' for now, 'topic:<id>'
# when topics land). The active session per context is tracked in memory;
# ownership never changes after creation.
#
# On startup, if the most recent session for the current context is still
# fresh (under SESSION_IDLE_SECONDS), it is resumed. Otherwise it is closed
# and a new one is started.
#
# Turns are always persisted immediately; the in-memory conversation buffer
# is a cache that can be reconstructed from the turns table.

import json
import threading
from datetime import datetime
from typing import Optional

from memory.store import get_conn, write_transaction, now_iso


# Session closes if there's no activity for this long. Applies when
# resuming on startup and when switching contexts.
SESSION_IDLE_SECONDS = 1800  # 30 minutes


# ===== Module state =====

_current_context: str = "global"
_context_sessions: dict[str, int] = {}    # context → active session id
_context_lock = threading.Lock()


# ===== Session lifecycle =====

def start_session(context: str = "global",
                  user_id: str = "default",
                  private: bool = False) -> int:
    """Create a new session for the given context."""
    now = now_iso()
    with write_transaction() as conn:
        cur = conn.execute(
            "INSERT INTO sessions "
            "(user_id, context, started_at, private_session) "
            "VALUES (?, ?, ?, ?)",
            (user_id, context, now, 1 if private else 0),
        )
        sid = cur.lastrowid

    with _context_lock:
        _context_sessions[context] = sid

    print(f"📖 Session {sid} started (context={context}).")
    return sid


def get_current_context() -> str:
    return _current_context


def get_current_session_id() -> Optional[int]:
    """Return the active session id for the current context, or None."""
    with _context_lock:
        return _context_sessions.get(_current_context)


def get_or_create_session(context: Optional[str] = None) -> int:
    """
    Return the active session for the given (or current) context.
    If none exists, tries to resume a recent one; otherwise creates new.
    """
    if context is None:
        context = _current_context

    # Already have an active session for this context?
    with _context_lock:
        existing = _context_sessions.get(context)
    if existing is not None:
        if not _session_is_stale(existing):
            return existing
        # Stale — close and fall through to creating a new one
        close_session(existing)

    # Look for a resumable open session in the DB
    resumable = _find_resumable_session(context)
    if resumable is not None:
        with _context_lock:
            _context_sessions[context] = resumable
        print(f"📖 Resumed session {resumable} (context={context}).")
        return resumable

    # Nothing to resume — start fresh
    return start_session(context)


def _find_resumable_session(context: str) -> Optional[int]:
    """
    Return the most recent open session for a context that's still fresh,
    or None. Closes stale candidates it encounters.
    """
    conn = get_conn()
    row = conn.execute(
        """
        SELECT s.id,
               COALESCE(
                   (SELECT MAX(ts) FROM turns WHERE session_id = s.id),
                   s.started_at
               ) AS last_activity
        FROM sessions s
        WHERE s.context = ? AND s.ended_at IS NULL
        ORDER BY last_activity DESC
        LIMIT 1
        """,
        (context,),
    ).fetchone()

    if not row:
        return None

    if _timestamp_is_stale(row["last_activity"]):
        close_session(row["id"])
        return None

    return row["id"]


def _session_is_stale(session_id: int) -> bool:
    """True if the session hasn't seen activity within SESSION_IDLE_SECONDS."""
    conn = get_conn()
    row = conn.execute(
        """
        SELECT COALESCE(
            (SELECT MAX(ts) FROM turns WHERE session_id = ?),
            started_at
        ) AS last_activity
        FROM sessions WHERE id = ?
        """,
        (session_id, session_id),
    ).fetchone()
    if not row:
        return True
    return _timestamp_is_stale(row["last_activity"])


def _timestamp_is_stale(ts: Optional[str]) -> bool:
    if not ts:
        return True
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return False  # unparseable — safer to keep than drop
    return (datetime.now() - dt).total_seconds() > SESSION_IDLE_SECONDS


def switch_context(new_context: str) -> tuple[int, bool]:
    """
    Suspend the current context's session and resume or create one for
    new_context. Returns (session_id, is_new).

    Not called by any code path in 6g — reserved for topic activation
    in 6i. The mechanism is here so it can be tested in isolation.
    """
    global _current_context

    if new_context == _current_context:
        sid = get_or_create_session(new_context)
        return sid, False

    # Do NOT reassign the previous session's context — it stays owned
    # by the old context. Just move the pointer.
    with _context_lock:
        old_context = _current_context
        _current_context = new_context

    # Check whether this is a resume or a fresh session
    with _context_lock:
        had_existing = new_context in _context_sessions
    sid = get_or_create_session(new_context)
    is_new = not had_existing and _session_turn_count(sid) == 0

    print(f"🔀 Context: {old_context} → {new_context} (session {sid})")
    return sid, is_new


def close_session(session_id: Optional[int] = None):
    """Mark a session as ended and queue it for summarization."""
    if session_id is None:
        session_id = get_current_session_id()
    if session_id is None:
        return

    with write_transaction() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ? "
            "WHERE id = ? AND ended_at IS NULL",
            (now_iso(), session_id),
        )

    # Clear the active-session pointer if it's this one
    with _context_lock:
        for ctx, sid in list(_context_sessions.items()):
            if sid == session_id:
                del _context_sessions[ctx]

    # Queue for summary if it has enough turns
    if _session_turn_count(session_id) >= 2:
        _enqueue_summary(session_id)

    print(f"📕 Session {session_id} closed.")


def _session_turn_count(session_id: int) -> int:
    conn = get_conn()
    return conn.execute(
        "SELECT COUNT(*) c FROM turns WHERE session_id = ?",
        (session_id,),
    ).fetchone()["c"]


# ===== Turn persistence =====

def append_turn(role: str, text: str,
                session_id: Optional[int] = None) -> int:
    if session_id is None:
        session_id = get_or_create_session()
    with write_transaction() as conn:
        cur = conn.execute(
            "INSERT INTO turns (session_id, role, text, ts) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, text, now_iso()),
        )
        return cur.lastrowid


def append_user_turn(text: str, session_id: Optional[int] = None) -> int:
    return append_turn("user", text, session_id)


def append_assistant_turn(text: str, session_id: Optional[int] = None) -> int:
    return append_turn("assistant", text, session_id)


def load_recent_turns(session_id: int, limit: int = 20) -> list[dict]:
    """Return the last `limit` turns for a session, in chronological order."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT role, text FROM turns
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    return [dict(r) for r in reversed(rows)]


# ===== Worker queries =====

def get_unprocessed_turns(limit: int = 50) -> list[dict]:
    checkpoint = _load_checkpoint()
    last_id = checkpoint.get("last_processed_turn_id", 0)

    conn = get_conn()
    rows = conn.execute(
        """
        SELECT t.id, t.session_id, t.role, t.text, t.ts
        FROM turns t
        JOIN sessions s ON s.id = t.session_id
        WHERE t.id > ?
          AND t.excluded_from_extraction = 0
          AND s.private_session = 0
        ORDER BY t.id
        LIMIT ?
        """,
        (last_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def turn_count() -> int:
    return get_conn().execute(
        "SELECT COUNT(*) c FROM turns"
    ).fetchone()["c"]


# ===== Summary queue (durable) =====

def _enqueue_summary(session_id: int):
    with write_transaction() as conn:
        row = conn.execute(
            "SELECT value FROM extraction_state "
            "WHERE key = 'pending_summaries'"
        ).fetchone()
        queue = json.loads(row["value"]) if row and row["value"] else []
        if session_id not in queue:
            queue.append(session_id)
        conn.execute(
            "INSERT OR REPLACE INTO extraction_state "
            "(key, value, updated_at) VALUES ('pending_summaries', ?, ?)",
            (json.dumps(queue), now_iso()),
        )


def pop_pending_summary() -> Optional[int]:
    with write_transaction() as conn:
        row = conn.execute(
            "SELECT value FROM extraction_state "
            "WHERE key = 'pending_summaries'"
        ).fetchone()
        if not row or not row["value"]:
            return None
        queue = json.loads(row["value"])
        if not queue:
            return None
        sid = queue.pop(0)
        conn.execute(
            "INSERT OR REPLACE INTO extraction_state "
            "(key, value, updated_at) VALUES ('pending_summaries', ?, ?)",
            (json.dumps(queue), now_iso()),
        )
    return sid


def pending_summary_count() -> int:
    conn = get_conn()
    row = conn.execute(
        "SELECT value FROM extraction_state WHERE key = 'pending_summaries'"
    ).fetchone()
    if not row or not row["value"]:
        return 0
    return len(json.loads(row["value"]))


# ===== Worker checkpoint =====

def _load_checkpoint() -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT value FROM extraction_state WHERE key = 'worker_checkpoint'"
    ).fetchone()
    if not row or not row["value"]:
        return {"last_processed_turn_id": 0, "consecutive_failures": 0}
    try:
        return json.loads(row["value"])
    except Exception:
        return {"last_processed_turn_id": 0, "consecutive_failures": 0}


def save_checkpoint(data: dict):
    with write_transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO extraction_state (key, value, updated_at) "
            "VALUES ('worker_checkpoint', ?, ?)",
            (json.dumps(data), now_iso()),
        )
def get_unprocessed_turns(limit: int = 50) -> list[dict]:
    checkpoint = _load_checkpoint()
    last_id = checkpoint.get("last_processed_turn_id", 0)

    conn = get_conn()
    rows = conn.execute(
        """
        SELECT t.id, t.session_id, t.role, t.text, t.ts, s.context
        FROM turns t
        JOIN sessions s ON s.id = t.session_id
        WHERE t.id > ?
          AND t.excluded_from_extraction = 0
          AND s.private_session = 0
        ORDER BY t.id
        LIMIT ?
        """,
        (last_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]