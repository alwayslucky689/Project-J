# memory/fact_memory.py - Public fact memory API
#
# Public functions (kept stable across Phase 6):
#   save_fact(fact: str) -> str
#   get_facts_context() -> str
#
# This file is the boundary between the rest of the assistant and the
# storage layer. Nothing outside memory/ should import store.py directly.

from datetime import datetime
from typing import Optional

from core.registry import register, Tool, Permission
from memory.store import get_conn, write_transaction, now_iso, json_or_none


# ===== Write path =====

def save_fact(fact: str,
              subject: str = "user",
              property_: Optional[str] = None,
              value: Optional[str] = None,
              source: str = "user_explicit",
              epistemic_status: str = "assertion",
              scope_type: str = "global",
              scope_topic_id: Optional[int] = None) -> str:
    if not fact or not fact.strip():
        return "❌ Nothing to remember."

    fact = fact.strip()
    now = now_iso()
    new_id = None

    with write_transaction() as conn:
        existing = conn.execute(
            "SELECT id FROM facts "
            "WHERE user_id='default' AND status='active' AND text = ?",
            (fact,),
        ).fetchone()
        if existing:
            return f"✓ Already remembered: {fact}"

        cur = conn.execute(
            """
            INSERT INTO facts (
                user_id, subject, property, value, text,
                source, epistemic_status, extraction_confidence,
                scope_type, scope_topic_id,
                valid_from, valid_until,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default", subject, property_, value, fact,
                source, epistemic_status, None,
                scope_type, scope_topic_id,
                now, None,
                "active", now, now,
            ),
        )
        new_id = cur.lastrowid

    # Embed outside the write lock — the model call can take ~50ms and
    # we don't want to hold up other writers. If it fails, the fact still
    # exists; backfill on next startup will catch it.
    _embed_fact(new_id, fact)

    return f"✓ Remembered: {fact}"


def _embed_fact(fact_id: int, text: str):
    try:
        from memory.embed import embed
        from memory.store import _vec_to_blob
    except ImportError:
        return

    vec = embed(text)
    if vec is None:
        return

    try:
        # vec0 virtual tables don't accept the write_transaction BEGIN
        # pattern cleanly; use a direct insert with retry.
        conn = get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO fact_embeddings (fact_id, embedding) "
            "VALUES (?, ?)",
            (fact_id, _vec_to_blob(vec)),
        )
        conn.commit()
    except Exception as e:
        # Non-fatal. Backfill on next startup will try again.
        print(f"⚠️ Could not embed fact {fact_id}: {e}")


def forget_fact(fact_id: Optional[int] = None,
                search: Optional[str] = None) -> str:
    """
    Mark a fact as deleted and tombstone it so re-extraction can't recreate it.
    Provide either fact_id (precise) or search (case-insensitive substring).
    """
    if fact_id is None and not search:
        return "❌ Specify a fact_id or a search term."

    conn = get_conn()

    if fact_id is not None:
        rows = conn.execute(
            "SELECT id, subject, property, scope_type, scope_topic_id, text "
            "FROM facts WHERE id = ? AND status != 'deleted'",
            (fact_id,),
        ).fetchall()
    else:
        like = f"%{search}%"
        rows = conn.execute(
            "SELECT id, subject, property, scope_type, scope_topic_id, text "
            "FROM facts WHERE status = 'active' AND text LIKE ? "
            "ORDER BY created_at DESC LIMIT 5",
            (like,),
        ).fetchall()

    if not rows:
        return "❌ No matching fact."

    if len(rows) > 1:
        lines = ["Multiple matches — specify by id:"]
        for r in rows:
            lines.append(f"  [{r['id']}] {r['text']}")
        return "\n".join(lines)

    row = rows[0]
    now = now_iso()

    with write_transaction() as tx:
        tx.execute(
            "UPDATE facts SET status='deleted', updated_at=? WHERE id=?",
            (now, row["id"]),
        )
        tx.execute(
            "INSERT INTO tombstones "
            "(user_id, subject, property, scope_type, scope_topic_id, deleted_at, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "default", row["subject"], row["property"],
                row["scope_type"], row["scope_topic_id"],
                now, "user requested forget",
            ),
        )

    return f"🗑️  Forgotten: {row['text']}"
def retrieve_facts(query: str, max_facts: int = 5,
                   max_sessions: int = 2) -> list[dict]:
    """
    Retrieve relevant active facts AND relevant past-session summaries.

    Session summaries are returned with kind='session' and subject='__session__'
    so memory/inject.py can format them distinctly.

    Returns an empty list if nothing relevant. Never raises.
    """
    if not query:
        return _recent_fact_records(max_facts)

    try:
        from memory.embed import embed, is_available
        from memory.store import _vec_to_blob
        if not is_available():
            return _recent_fact_records(max_facts)

        qvec = embed(query)
        if qvec is None:
            return _recent_fact_records(max_facts)

        conn = get_conn()

        # --- Fact retrieval ---
        fact_records = []
        try:
            knn_rows = conn.execute(
                "SELECT fact_id, distance FROM fact_embeddings "
                "WHERE embedding MATCH ? AND k = 20",
                (_vec_to_blob(qvec),),
            ).fetchall()
            if knn_rows:
                id_to_dist = {r["fact_id"]: r["distance"] for r in knn_rows}
                placeholders = ",".join("?" * len(id_to_dist))
                fact_rows = conn.execute(
                    f"SELECT id, subject, property, value, text, source, "
                    f"epistemic_status, scope_type, created_at "
                    f"FROM facts "
                    f"WHERE id IN ({placeholders}) AND status = 'active'",
                    tuple(id_to_dist.keys()),
                ).fetchall()

                DISTANCE_THRESHOLD = 0.45
                for fr in fact_rows:
                    d = id_to_dist[fr["id"]]
                    if d <= DISTANCE_THRESHOLD:
                        fact_records.append({
                            "id": fr["id"],
                            "text": fr["text"],
                            "subject": fr["subject"],
                            "property": fr["property"],
                            "value": fr["value"],
                            "source": fr["source"],
                            "epistemic_status": fr["epistemic_status"],
                            "scope_type": fr["scope_type"],
                            "created_at": fr["created_at"],
                            "distance": d,
                            "kind": "fact",
                        })
                fact_records.sort(key=lambda r: r["distance"])
                fact_records = fact_records[:max_facts]
        except Exception as e:
            print(f"⚠️ Fact retrieval failed: {e}")

        # --- Session summary retrieval ---
        session_records = []
        try:
            sess_knn = conn.execute(
                "SELECT session_id, distance FROM session_embeddings "
                "WHERE embedding MATCH ? AND k = 5",
                (_vec_to_blob(qvec),),
            ).fetchall()
            if sess_knn:
                sid_to_dist = {r["session_id"]: r["distance"] for r in sess_knn}
                sids = tuple(sid_to_dist.keys())
                placeholders = ",".join("?" * len(sids))
                sess_rows = conn.execute(
                    f"SELECT id, summary, started_at FROM sessions "
                    f"WHERE id IN ({placeholders}) AND summary IS NOT NULL",
                    sids,
                ).fetchall()

                # Looser threshold than facts — summaries are longer
                # and their embeddings are naturally more diffuse.
                SESSION_THRESHOLD = 0.55
                for sr in sess_rows:
                    d = sid_to_dist[sr["id"]]
                    if d <= SESSION_THRESHOLD:
                        session_records.append({
                            "id": sr["id"],
                            "text": sr["summary"],
                            "created_at": sr["started_at"],
                            "distance": d,
                            "kind": "session",
                            "subject": "__session__",
                            "source": "session_summary",
                            "scope_type": "session",
                        })
                session_records.sort(key=lambda r: r["distance"])
                session_records = session_records[:max_sessions]
        except Exception as e:
            # Session retrieval is best-effort; don't fail the whole call
            print(f"⚠️ Session retrieval failed: {e}")

        return fact_records + session_records

    except Exception as e:
        print(f"⚠️ Retrieval failed: {e}")
        return _recent_fact_records(max_facts)


def _recent_fact_records(max_facts: int) -> list[dict]:
    """Fallback: most recent active global facts as records."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT id, subject, property, value, text, source,
               epistemic_status, scope_type, created_at
        FROM facts
        WHERE user_id='default'
          AND status='active'
          AND scope_type='global'
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (max_facts,),
    ).fetchall()
    return [dict(r) for r in rows]

# ===== Read path (temporary, upgraded in 6c) =====

def get_facts_context(query: str = "", max_facts: int = 5) -> str:
    """
    Deprecated. Kept for backward compatibility. New code should use
    retrieve_facts() + memory.inject.format_memory_block().

    This returns a plain "Known facts" list for legacy callers. It does NOT
    provide the safe injection formatting.
    """
    records = retrieve_facts(query, max_facts)
    if not records:
        return ""
    lines = ["Known facts about the user:"]
    for r in records:
        lines.append(f"- {r['text']}")
    return "\n".join(lines)


# ===== Tools (registered with the same names as before) =====
def pin_fact(fact_id: int = None, search: str = None) -> str:
    """Pin a fact so it's always in the prompt profile."""
    return _set_pinned(True, fact_id, search)


def unpin_fact(fact_id: int = None, search: str = None) -> str:
    return _set_pinned(False, fact_id, search)


def _set_pinned(pinned: bool, fact_id, search) -> str:
    if fact_id is None and not search:
        return "❌ Specify a fact_id or a search term."
    conn = get_conn()
    if fact_id is not None:
        rows = conn.execute(
            "SELECT id, text, pinned FROM facts "
            "WHERE id=? AND status='active'",
            (fact_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, text, pinned FROM facts "
            "WHERE status='active' AND text LIKE ? "
            "ORDER BY created_at DESC LIMIT 5",
            (f"%{search}%",),
        ).fetchall()

    if not rows:
        return "❌ No matching fact."
    if len(rows) > 1:
        lines = ["Multiple matches — specify by id:"]
        for r in rows:
            tag = "📌" if r["pinned"] else "  "
            lines.append(f"  {tag} [{r['id']}] {r['text']}")
        return "\n".join(lines)

    r = rows[0]
    with write_transaction() as tx:
        tx.execute(
            "UPDATE facts SET pinned=?, updated_at=? WHERE id=?",
            (1 if pinned else 0, now_iso(), r["id"]),
        )
    verb = "Pinned" if pinned else "Unpinned"
    return f"📌 {verb}: {r['text']}"


def _pin_fact(fact_id=None, search=None):
    return pin_fact(fact_id=fact_id, search=search)


def _unpin_fact(fact_id=None, search=None):
    return unpin_fact(fact_id=fact_id, search=search)


register(Tool(
    name="pin_fact",
    description=(
        'Pins a fact to always appear in every prompt. Use sparingly — '
        'reserve for identity, language, hard constraints, and a small '
        'set of durable preferences. Takes "fact_id" (integer) or '
        '"search" (string).'
    ),
    handler=_pin_fact,
    formatter=lambda s: s,
    permission=Permission.SAFE,
    response_key="fact_pinned",
))

register(Tool(
    name="unpin_fact",
    description=(
        'Removes a fact from the pinned profile so it is only retrieved '
        'when relevant. Takes "fact_id" (integer) or "search" (string).'
    ),
    handler=_unpin_fact,
    formatter=lambda s: s,
    permission=Permission.SAFE,
    response_key="fact_unpinned",
))

def _remember_fact(fact):
    return save_fact(fact)


def _forget_fact(fact_id=None, search=None):
    return forget_fact(fact_id=fact_id, search=search)


def _list_facts():
    conn = get_conn()
    rows = conn.execute(
            """
            SELECT fe.fact_id, fe.distance, f.text
            FROM fact_embeddings fe
            JOIN facts f ON f.id = fe.fact_id
            WHERE f.status='active'
              AND fe.embedding MATCH ?
            ORDER BY fe.distance
            LIMIT ?
            """,
            (_vec_to_blob(qvec), max_facts * 2),
        ).fetchall()
    if not rows:
        return "No facts stored yet."
    lines = [f"📋 {len(rows)} facts:"]
    for r in rows:
        lines.append(f"  [{r['id']}] {r['text']}")
    return "\n".join(lines)


register(Tool(
    name="remember_fact",
    description='Stores a fact about the user. Takes "fact" (string).',
    handler=_remember_fact,
    formatter=lambda s: s,
    permission=Permission.SAFE,
    response_key="fact_saved",
))

register(Tool(
    name="forget_fact",
    description=(
        'Deletes a stored fact. Use "search" (string) to match by substring '
        'or "fact_id" (integer) for a specific record. '
        'NOT for editing — use remember_fact to add a corrected version.'
    ),
    handler=_forget_fact,
    formatter=lambda s: s,
    permission=Permission.ACTION,
    response_key="fact_deleted",
))

register(Tool(
    name="list_facts",
    description=(
        "Lists the user's stored facts. Takes no arguments. "
        "Useful when the user asks 'what do you know about me'."
    ),
    handler=_list_facts,
    formatter=lambda s: s,
    permission=Permission.SAFE,
    response_key="facts_listed",
))
def _get_recent_facts(max_facts: int) -> str:
    """
    Fallback retrieval: return the most recent active global facts.

    Used when:
      - No query was provided
      - The embedding model is unavailable
      - Semantic retrieval failed

    This is the 6b behavior, kept as a safety net so memory never
    disappears entirely if the vector path breaks.
    """
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT text FROM facts
        WHERE user_id='default'
          AND status='active'
          AND scope_type='global'
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (max_facts,),
    ).fetchall()
    if not rows:
        return ""
    lines = ["Known facts about the user:"]
    for r in rows:
        lines.append(f"- {r['text']}")
    return "\n".join(lines)