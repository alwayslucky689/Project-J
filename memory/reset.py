# memory/reset.py - Bulk reset and deletion tooling (Phase 6l)
#
# Destructive operations for wiping memory during development. All
# operations act on the SQLite DB directly. Regular tables get
# write_transaction; vec0 virtual tables get direct commits because
# they don't cooperate with the BEGIN/COMMIT pattern (same reason
# fact_memory._embed_fact commits directly).
#
# Cascade rules:
#   --all           : every row in facts, fact_topics, turns, sessions,
#                     topics, tombstones, habits, extraction_state, and
#                     all four vec tables. Resets worker checkpoint and
#                     drains the pending-summaries queue.
#   --topic <name>  : deletes the topic, its topic_embedding, every
#                     session with context='topic:<id>' and those
#                     sessions' turns + session_embeddings, and every
#                     fact with scope_topic_id=<id> + its fact_embedding.
#   --facts-only    : facts + fact_embeddings. Sessions, turns, topics
#                     survive. No tombstones — intended for re-extraction.
#   --inferred-only : facts where source='inferred' + their embeddings.
#   --since <date>  : facts created on/after <date> + their embeddings.
#
# No tombstones are written by this module. Tombstones are for
# individual forget_fact calls; bulk wipes are a clean slate.
#
# IMPORTANT: If the assistant is running when you invoke this, its
# in-memory session pointers will reference rows that no longer exist.
# Restart the assistant afterward.
#
# Usage:
#   python -m memory.reset --all
#   python -m memory.reset --all --yes
#   python -m memory.reset --topic project-j
#   python -m memory.reset --facts-only
#   python -m memory.reset --inferred-only
#   python -m memory.reset --since 2026-01-01

import argparse
import json
import sqlite3
import sys
from typing import Optional

from memory.store import get_conn, write_transaction, now_iso


# ===== Vec table helpers =====

def _clear_vec_table(conn: sqlite3.Connection, table: str) -> int:
    """Delete every row from a vec0 table. Returns rows removed (0 on error)."""
    try:
        before = conn.execute(
            f"SELECT COUNT(*) c FROM {table}"
        ).fetchone()["c"]
        # WHERE 1=1 satisfies vec0's requirement that DELETE have a predicate.
        conn.execute(f"DELETE FROM {table} WHERE 1=1")
        conn.commit()
        return before
    except sqlite3.OperationalError as e:
        print(f"⚠️  Could not clear {table}: {e}")
        return 0


def _delete_vec_rows(conn: sqlite3.Connection, table: str,
                     key_col: str, keys: list) -> int:
    """Delete specific rows from a vec0 table by key. Returns the number
    of rows we attempted to delete (not the number that existed)."""
    if not keys:
        return 0
    deleted = 0
    for k in keys:
        try:
            conn.execute(
                f"DELETE FROM {table} WHERE {key_col} = ?", (k,)
            )
            deleted += 1
        except sqlite3.OperationalError as e:
            print(f"⚠️  Could not delete {table} row {k}: {e}")
    conn.commit()
    return deleted


# ===== Pending summaries =====

def _prune_pending_summaries(session_ids: list):
    """Remove given session ids from the extraction_state pending queue.
    Safe to call with an empty list or a missing queue."""
    if not session_ids:
        return
    with write_transaction() as tx:
        row = tx.execute(
            "SELECT value FROM extraction_state "
            "WHERE key = 'pending_summaries'"
        ).fetchone()
        if not row or not row["value"]:
            return
        try:
            queue = json.loads(row["value"])
        except json.JSONDecodeError:
            return
        if not isinstance(queue, list):
            return
        drop = set(session_ids)
        new_queue = [s for s in queue if s not in drop]
        if len(new_queue) == len(queue):
            return
        tx.execute(
            "INSERT OR REPLACE INTO extraction_state "
            "(key, value, updated_at) VALUES ('pending_summaries', ?, ?)",
            (json.dumps(new_queue), now_iso()),
        )


# ===== Public operations =====

def reset_all() -> dict:
    """
    Wipe every memory table. Returns a dict of {table_name: rows_removed}.
    """
    conn = get_conn()
    counts: dict = {}

    with write_transaction() as tx:
        # Children before parents. FK cascades would handle it anyway, but
        # explicit deletes let us report real counts.
        counts["fact_topics"] = tx.execute(
            "DELETE FROM fact_topics"
        ).rowcount
        counts["facts"] = tx.execute("DELETE FROM facts").rowcount
        counts["turns"] = tx.execute("DELETE FROM turns").rowcount
        counts["sessions"] = tx.execute("DELETE FROM sessions").rowcount
        counts["topics"] = tx.execute("DELETE FROM topics").rowcount
        counts["tombstones"] = tx.execute("DELETE FROM tombstones").rowcount
        counts["habits"] = tx.execute("DELETE FROM habits").rowcount
        counts["extraction_state"] = tx.execute(
            "DELETE FROM extraction_state"
        ).rowcount

    # Vec tables: direct commits
    for table in (
        "fact_embeddings",
        "turn_embeddings",
        "topic_embeddings",
        "session_embeddings",
    ):
        counts[table] = _clear_vec_table(conn, table)

    return counts


def reset_topic(identifier) -> Optional[dict]:
    """
    Delete a topic and everything scoped to it. Returns None if no topic
    matched. Otherwise a count dict including 'topic' and 'topic_id' for
    reporting.
    """
    from memory.topics import get_topic, topic_context

    topic = get_topic(identifier)
    if not topic:
        return None

    tid = topic["id"]
    ctx = topic_context(tid)
    conn = get_conn()

    # Gather ids before deleting anything.
    session_ids = [r["id"] for r in conn.execute(
        "SELECT id FROM sessions WHERE context = ?", (ctx,)
    ).fetchall()]

    counts = {
        "topic":         topic["name"],
        "topic_id":      tid,
        "facts":         0,
        "fact_embeddings": 0,
        "sessions":      0,
        "turns":         0,
        "session_embeddings": 0,
        "topic_embeddings":   0,
    }

    fact_ids: list[int] = []
    with write_transaction() as tx:
        # Sessions + turns
        if session_ids:
            ph = ",".join("?" * len(session_ids))
            counts["turns"] = tx.execute(
                f"DELETE FROM turns WHERE session_id IN ({ph})",
                session_ids,
            ).rowcount
            counts["sessions"] = tx.execute(
                f"DELETE FROM sessions WHERE id IN ({ph})",
                session_ids,
            ).rowcount

        # Facts scoped to this topic
        fact_ids = [r["id"] for r in tx.execute(
            "SELECT id FROM facts WHERE scope_topic_id = ?", (tid,)
        ).fetchall()]
        if fact_ids:
            ph = ",".join("?" * len(fact_ids))
            tx.execute(
                f"DELETE FROM fact_topics WHERE fact_id IN ({ph})",
                fact_ids,
            )
            counts["facts"] = tx.execute(
                f"DELETE FROM facts WHERE id IN ({ph})",
                fact_ids,
            ).rowcount

        # Any remaining fact_topics linking this topic (cross-links from
        # other facts that have scope_topic_id elsewhere).
        tx.execute("DELETE FROM fact_topics WHERE topic_id = ?", (tid,))

        # The topic itself
        tx.execute("DELETE FROM topics WHERE id = ?", (tid,))

    # Vec cleanup, direct commits
    counts["fact_embeddings"] = _delete_vec_rows(
        conn, "fact_embeddings", "fact_id", fact_ids
    )
    counts["session_embeddings"] = _delete_vec_rows(
        conn, "session_embeddings", "session_id", session_ids
    )
    counts["topic_embeddings"] = _delete_vec_rows(
        conn, "topic_embeddings", "topic_id", [tid]
    )

    # Drop any pending summaries for the sessions we just deleted.
    _prune_pending_summaries(session_ids)

    return counts


def reset_facts(where_clause: str = "", params: tuple = ()) -> dict:
    """
    Delete facts matching the given WHERE clause (empty = all facts).
    Cascades to fact_embeddings and fact_topics. No tombstones.
    """
    conn = get_conn()
    counts = {"facts": 0, "fact_embeddings": 0}

    with write_transaction() as tx:
        rows = tx.execute(
            f"SELECT id FROM facts {where_clause}", params
        ).fetchall()
        fact_ids = [r["id"] for r in rows]
        if not fact_ids:
            return counts
        ph = ",".join("?" * len(fact_ids))
        tx.execute(
            f"DELETE FROM fact_topics WHERE fact_id IN ({ph})",
            fact_ids,
        )
        counts["facts"] = tx.execute(
            f"DELETE FROM facts WHERE id IN ({ph})",
            fact_ids,
        ).rowcount

    counts["fact_embeddings"] = _delete_vec_rows(
        conn, "fact_embeddings", "fact_id", fact_ids
    )
    return counts


# ===== CLI =====

def _confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def _report(counts: dict):
    if not counts:
        print("Nothing to do.")
        return
    print()
    print("✅ Reset complete.")
    width = max(len(str(k)) for k in counts)
    for k, v in counts.items():
        print(f"   {k:<{width}}  {v}")
    print()
    print("⚠️  If the assistant is running, restart it. Its in-memory")
    print("    session pointers reference rows that no longer exist.")


def main():
    p = argparse.ArgumentParser(
        prog="python -m memory.reset",
        description=(
            "Destructive memory reset. All operations are unrecoverable; "
            "the SQLite DB is the source of truth."
        ),
    )
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip the confirmation prompt.")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true",
                   help="Wipe every memory table.")
    g.add_argument("--topic", metavar="NAME",
                   help="Delete a topic and everything scoped to it.")
    g.add_argument("--facts-only", action="store_true",
                   help="Wipe all facts, keep sessions and turns.")
    g.add_argument("--inferred-only", action="store_true",
                   help="Wipe only facts with source='inferred'.")
    g.add_argument("--since", metavar="DATE",
                   help="Wipe facts created on/after DATE. "
                        "ISO 8601; lexicographic comparison.")

    args = p.parse_args()

    # --- --all ---
    if args.all:
        if not _confirm(
            "This will wipe EVERY memory table "
            "(facts, sessions, turns, topics, tombstones, habits).",
            args.yes,
        ):
            print("Aborted.")
            sys.exit(1)
        _report(reset_all())
        return

    # --- --topic ---
    if args.topic:
        from memory.topics import get_topic
        topic = get_topic(args.topic)
        if not topic:
            print(f"❌ No topic matching {args.topic!r}")
            sys.exit(1)
        if not _confirm(
            f"This will delete topic '{topic['name']}' and all its "
            f"sessions, turns, and scoped facts.",
            args.yes,
        ):
            print("Aborted.")
            sys.exit(1)
        _report(reset_topic(topic["id"]))
        return

    # --- --facts-only ---
    if args.facts_only:
        if not _confirm(
            "This will wipe ALL facts. Sessions and turns are kept.",
            args.yes,
        ):
            print("Aborted.")
            sys.exit(1)
        _report(reset_facts())
        return

    # --- --inferred-only ---
    if args.inferred_only:
        if not _confirm(
            "This will wipe every fact with source='inferred'.",
            args.yes,
        ):
            print("Aborted.")
            sys.exit(1)
        _report(reset_facts(where_clause="WHERE source = 'inferred'"))
        return

    # --- --since ---
    if args.since:
        if not _confirm(
            f"This will wipe facts created on/after {args.since}.",
            args.yes,
        ):
            print("Aborted.")
            sys.exit(1)
        _report(reset_facts(
            where_clause="WHERE created_at >= ?",
            params=(args.since,),
        ))
        return


if __name__ == "__main__":
    main()