# memory/retrieval.py - Layered memory assembly with a token budget
#
# Produces a prioritized list of memory records for injection into the
# prompt. Higher-priority layers are added first; assembly stops adding
# from a layer when either that layer's cap or the global budget would
# be exceeded. Zero results is valid — no forcing minimums.

from typing import Optional

from memory.store import get_conn, _vec_to_blob

# Soft cap per layer (approximate tokens). Layers are processed in order;
# if a layer fills early, later layers still get their chance with the
# remaining global budget.
LAYER_CAPS = {
    "pinned": 300,
    "global_facts": 700,
    "global_sessions": 400,
}

# Total approximate token budget for memory injection. Does NOT include
# the system prompt, the buffer, or the current query.
MAX_MEMORY_TOKENS = 2000

# Similarity thresholds (cosine distance)
FACT_THRESHOLD = 0.45
SESSION_THRESHOLD = 0.55
# memory/retrieval.py - Layered memory assembly with topic scoping

from typing import Optional

from memory.store import get_conn, _vec_to_blob

LAYER_CAPS = {
    "pinned": 300,
    "topic_facts": 500,
    "global_facts": 500,
    "topic_sessions": 300,
    "global_sessions": 300,
}

MAX_MEMORY_TOKENS = 2000
FACT_THRESHOLD = 0.45
SESSION_THRESHOLD = 0.55


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ===== Context lookup =====

def _get_session_context(session_id: Optional[int]) -> str:
    if session_id is None:
        return "global"
    conn = get_conn()
    row = conn.execute(
        "SELECT context FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return row["context"] if row else "global"


def _extract_topic_id(context: str) -> Optional[int]:
    if not context or not context.startswith("topic:"):
        return None
    try:
        return int(context.split(":", 1)[1])
    except (ValueError, IndexError):
        return None


# ===== Public API =====

def assemble_memory(query: str,
                    session_id: Optional[int] = None) -> list[dict]:
    """
    Assemble memory records for the current query, scoped by the session's
    context. In a topic context, retrieves topic-scoped facts and topic
    session summaries as priority layers, plus global facts and sessions.
    In global context, only global and unscoped content is retrieved.
    """
    context = _get_session_context(session_id)
    topic_id = _extract_topic_id(context)

    layers: list[tuple[str, list[dict]]] = []

    # Pinned: always included, no query needed
    try:
        layers.append(("pinned", _load_pinned()))
    except Exception as e:
        print(f"⚠️ Pinned load failed: {e}")

    if query:
        try:
            topic_facts, global_facts = _retrieve_facts_split(
                query, topic_id, limit_per_bucket=5,
            )
        except Exception as e:
            print(f"⚠️ Fact retrieval failed: {e}")
            topic_facts, global_facts = [], []

        if topic_id is not None and topic_facts:
            layers.append(("topic_facts", topic_facts))
        if global_facts:
            layers.append(("global_facts", global_facts))

        try:
            if topic_id is not None:
                topic_sessions = _retrieve_sessions(
                    query, f"topic:{topic_id}", limit=2,
                )
                if topic_sessions:
                    layers.append(("topic_sessions", topic_sessions))
                global_sessions = _retrieve_sessions(
                    query, "global", limit=1,
                )
            else:
                global_sessions = _retrieve_sessions(
                    query, "global", limit=2,
                )
            if global_sessions:
                layers.append(("global_sessions", global_sessions))
        except Exception as e:
            print(f"⚠️ Session retrieval failed: {e}")

    # Budget-bounded assembly
    result: list[dict] = []
    total_used = 0
    seen_ids: set[tuple[str, int]] = set()

    for layer_name, records in layers:
        layer_cap = LAYER_CAPS.get(layer_name, 400)
        layer_used = 0
        for r in records:
            key = (r.get("kind", "fact"), r.get("id", -1))
            if key in seen_ids:
                continue
            cost = _estimate_tokens(r.get("text", ""))
            if layer_used + cost > layer_cap:
                break
            if total_used + cost > MAX_MEMORY_TOKENS:
                break
            r["layer"] = layer_name
            result.append(r)
            seen_ids.add(key)
            layer_used += cost
            total_used += cost

    return result


# ===== Layer loaders =====

def _load_pinned() -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT id, subject, property, value, text, source, created_at
        FROM facts
        WHERE user_id='default'
          AND status='active'
          AND pinned=1
        ORDER BY created_at
        LIMIT 20
        """
    ).fetchall()
    return [{
        "id": r["id"],
        "kind": "fact",
        "text": r["text"],
        "subject": r["subject"],
        "property": r["property"],
        "value": r["value"],
        "source": r["source"],
        "created_at": r["created_at"],
    } for r in rows]


def _retrieve_facts_split(query: str, topic_id: Optional[int],
                          limit_per_bucket: int = 5
                          ) -> tuple[list[dict], list[dict]]:
    """
    Return (topic_scoped, non_topic_scoped) records in one pass.

    Non-topic-scoped = rows where scope_topic_id IS NULL, which covers
    both 'global' and 'unscoped' facts.
    """
    from memory.embed import embed, is_available
    if not is_available():
        return [], _recent_global_facts(limit_per_bucket)

    qvec = embed(query)
    if qvec is None:
        return [], _recent_global_facts(limit_per_bucket)

    conn = get_conn()
    knn = conn.execute(
        "SELECT fact_id, distance FROM fact_embeddings "
        "WHERE embedding MATCH ? AND k = 30",
        (_vec_to_blob(qvec),),
    ).fetchall()
    if not knn:
        return [], []

    id_to_dist = {r["fact_id"]: r["distance"] for r in knn}
    placeholders = ",".join("?" * len(id_to_dist))
    rows = conn.execute(
        f"SELECT id, subject, property, value, text, source, "
        f"epistemic_status, scope_type, scope_topic_id, created_at "
        f"FROM facts "
        f"WHERE id IN ({placeholders}) AND status='active' AND pinned=0",
        tuple(id_to_dist.keys()),
    ).fetchall()

    topic_records: list[dict] = []
    global_records: list[dict] = []

    for r in rows:
        d = id_to_dist[r["id"]]
        if d > FACT_THRESHOLD:
            continue
        rec = {
            "id": r["id"],
            "kind": "fact",
            "text": r["text"],
            "subject": r["subject"],
            "property": r["property"],
            "value": r["value"],
            "source": r["source"],
            "epistemic_status": r["epistemic_status"],
            "scope_type": r["scope_type"],
            "scope_topic_id": r["scope_topic_id"],
            "created_at": r["created_at"],
            "distance": d,
        }
        if topic_id is not None and r["scope_topic_id"] == topic_id:
            topic_records.append(rec)
        elif r["scope_topic_id"] is None:
            global_records.append(rec)

    topic_records.sort(key=lambda r: r["distance"])
    global_records.sort(key=lambda r: r["distance"])
    return topic_records[:limit_per_bucket], global_records[:limit_per_bucket]


def _retrieve_sessions(query: str, context_filter: str,
                       limit: int = 2) -> list[dict]:
    from memory.embed import embed, is_available
    if not is_available():
        return []

    qvec = embed(query)
    if qvec is None:
        return []

    conn = get_conn()
    knn = conn.execute(
        "SELECT session_id, distance FROM session_embeddings "
        "WHERE embedding MATCH ? AND k = 8",
        (_vec_to_blob(qvec),),
    ).fetchall()
    if not knn:
        return []

    sid_to_dist = {r["session_id"]: r["distance"] for r in knn}
    placeholders = ",".join("?" * len(sid_to_dist))
    rows = conn.execute(
        f"SELECT id, summary, started_at FROM sessions "
        f"WHERE id IN ({placeholders}) "
        f"AND summary IS NOT NULL "
        f"AND context = ?",
        (*sid_to_dist.keys(), context_filter),
    ).fetchall()

    records = []
    for r in rows:
        d = sid_to_dist[r["id"]]
        if d <= SESSION_THRESHOLD:
            records.append({
                "id": r["id"],
                "kind": "session",
                "text": r["summary"],
                "created_at": r["started_at"],
                "distance": d,
            })
    records.sort(key=lambda r: r["distance"])
    return records[:limit]


def _recent_global_facts(limit: int) -> list[dict]:
    """Fallback when embeddings are unavailable — only non-topic facts."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, subject, property, value, text, source, created_at "
        "FROM facts "
        "WHERE user_id='default' AND status='active' AND pinned=0 "
        "AND scope_topic_id IS NULL "
        "ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [{
        "id": r["id"], "kind": "fact", "text": r["text"],
        "subject": r["subject"], "property": r["property"],
        "value": r["value"], "source": r["source"],
        "created_at": r["created_at"],
    } for r in rows]





