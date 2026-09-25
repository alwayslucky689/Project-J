# memory/store.py - SQLite connection + migrations + low-level helpers
#
# This is the storage layer. It knows about tables and rows.
# It does NOT know about "facts" as a concept — that's fact_memory.py.
#
# Thread safety:
#   The assistant runs multiple threads (audio callback, STT processing,
#   TTS threads). SQLite connections are per-thread by default, but we
#   use check_same_thread=False so a single connection can be shared, and
#   serialize writes with a lock. Reads are lock-free (SQLite handles
#   those internally with WAL mode).

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.paths import MEMORY_DB

_conn: Optional[sqlite3.Connection] = None
_conn_lock = threading.Lock()
_write_lock = threading.Lock()

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


# ===== Connection =====

def get_conn() -> sqlite3.Connection:
    """Return the process-wide SQLite connection. Opens + migrates on first call."""
    global _conn
    if _conn is not None:
        return _conn
    with _conn_lock:
        if _conn is not None:
            return _conn
        _conn = _open_connection(MEMORY_DB)
        _apply_migrations(_conn)
        return _conn


def _open_connection(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        check_same_thread=False,
        # NOTE: PARSE_DECLTYPES is deliberately omitted. It installs a
        # converter that expects "YYYY-MM-DD HH:MM:SS" but we store ISO
        # format "YYYY-MM-DDTHH:MM:SS" — the converter crashes on the "T".
        # We work with raw strings throughout, so no conversion is needed.
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")

    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        print("✅ sqlite-vec loaded.")
    except Exception as e:
        print(f"⚠️ sqlite-vec unavailable: {e}")

    return conn


# ===== Migrations =====

def _migration_001_initial_schema(conn: sqlite3.Connection):
    """Apply the full initial schema and seed the default user."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(sql)
    now = datetime.now().isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO users (id, name, timezone, created_at) "
        "VALUES (?, ?, ?, ?)",
        ("default", "User", "UTC", now),
    )

def _migration_002_embeddings(conn: sqlite3.Connection):
    """Vector search tables. Canonical data stays in `facts`; embeddings
    live here so the index can be rebuilt without touching memory meaning."""
    # vec0 virtual tables require sqlite-vec to be loaded. If it isn't,
    # this migration still succeeds but creates nothing — retrieval will
    # fall back to recency.
    try:
        conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fact_embeddings USING vec0(
                fact_id INTEGER PRIMARY KEY,
                embedding FLOAT[384]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS turn_embeddings USING vec0(
                turn_id INTEGER PRIMARY KEY,
                embedding FLOAT[384]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS topic_embeddings USING vec0(
                topic_id INTEGER PRIMARY KEY,
                embedding FLOAT[384]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS session_embeddings USING vec0(
                session_id INTEGER PRIMARY KEY,
                embedding FLOAT[384]
            );
        """)
    except sqlite3.OperationalError as e:
        print(f"⚠️ Could not create vector tables (sqlite-vec missing?): {e}")
        return

    # Record which embedding model produced these vectors.
    # If we ever switch models, this row tells us what to re-embed.
    now = datetime.now().isoformat()

    conn.execute(
        "INSERT OR IGNORE INTO embedding_versions "
        "(model_id, dimensions, created_at) VALUES (?, ?, ?)",
        ("BAAI/bge-small-en-v1.5", 384, now),

        )
def _migration_003_cosine_metric(conn: sqlite3.Connection):
    """
    Rebuild fact_embeddings with cosine distance.

    L2 on unnormalized MiniLM embeddings doesn't discriminate between short
    factual sentences — all distances cluster in a narrow band (1.2-1.5),
    producing no meaningful ranking. Cosine is the metric MiniLM is trained
    for, and separates 0.2 (near-duplicate) to 1.0 (unrelated) cleanly.

    Dropping the table wipes embeddings only. Facts themselves are untouched,
    and backfill_fact_embeddings() re-creates the vectors.
    """
    try:
        conn.executescript("""
            DROP TABLE IF EXISTS fact_embeddings;
            CREATE VIRTUAL TABLE fact_embeddings USING vec0(
                fact_id INTEGER PRIMARY KEY,
                embedding FLOAT[384] distance_metric=cosine
            );
        """)
    except sqlite3.OperationalError as e:
        print(f"⚠️ Cosine migration failed: {e}")

def _migration_004_bge_small(conn: sqlite3.Connection):
    """Switch embedding model from MiniLM-L6 to BGE-small-en-v1.5.
    Drops and recreates the embeddings table so backfill uses the new model."""
    try:
        conn.executescript("""
            DROP TABLE IF EXISTS fact_embeddings;
            CREATE VIRTUAL TABLE fact_embeddings USING vec0(
                fact_id INTEGER PRIMARY KEY,
                embedding FLOAT[384] distance_metric=cosine
            );
        """)
    except sqlite3.OperationalError as e:
        print(f"⚠️ Model swap migration failed: {e}")


def _migration_005_session_context(conn: sqlite3.Connection):
    """Sessions belong to exactly one context permanently.
    Global-only for now; topics get their own context string later."""
    conn.executescript("""
        ALTER TABLE sessions ADD COLUMN context TEXT NOT NULL DEFAULT 'global';
        CREATE INDEX IF NOT EXISTS idx_sessions_context_open
            ON sessions(context, started_at DESC);
    """)


def _migration_006_pinned_facts(conn: sqlite3.Connection):
    """Facts can be pinned to always appear in the system prompt's
    profile block. Intended for a very small set — identity, language,
    response preferences, hard constraints."""
    conn.executescript("""
        ALTER TABLE facts ADD COLUMN pinned BOOLEAN DEFAULT 0;
        CREATE INDEX IF NOT EXISTS idx_facts_pinned
            ON facts(user_id, pinned, status);
    """)

def _migration_007_topic_fields(conn: sqlite3.Connection):
    """Add summary/description bookkeeping and origin tracking to topics."""
    conn.executescript("""
        ALTER TABLE topics ADD COLUMN summary TEXT;
        ALTER TABLE topics ADD COLUMN summary_updated_at TIMESTAMP;
        ALTER TABLE topics ADD COLUMN created_from TEXT DEFAULT 'explicit';
        CREATE INDEX IF NOT EXISTS idx_topics_user_active
            ON topics(user_id, status, last_active_at DESC);
    """)

def _migration_008_topic_scope_index(conn: sqlite3.Connection):
    """Index for retrieving facts by scope_topic_id."""
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_facts_topic_scope
            ON facts(scope_topic_id, status);
    """)


MIGRATIONS = [
    # ... existing migrations 1-7 ...
    (8, "topic_scope_index", _migration_008_topic_scope_index),
]



def _apply_migrations(conn: sqlite3.Connection):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY, applied_at TIMESTAMP NOT NULL)"
    )
    applied = {row["version"] for row in conn.execute(
        "SELECT version FROM schema_version"
    )}
    for version, name, fn in MIGRATIONS:
        if version in applied:
            continue
        print(f"📦 Applying memory migration {version}: {name}")
        fn(conn)
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (version, datetime.now().isoformat()),
        )
        conn.commit()


# ===== Write helper =====

def write_transaction():
    """
    Context manager that serializes writes and commits on success.

    Usage:
        with write_transaction() as conn:
            conn.execute("INSERT ...")
    """
    class _Ctx:
        def __enter__(self):
            _write_lock.acquire()
            self.conn = get_conn()
            self.conn.execute("BEGIN")
            return self.conn

        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                if exc_type is None:
                    self.conn.commit()
                else:
                    self.conn.rollback()
            finally:
                _write_lock.release()
            return False

    return _Ctx()


# ===== Utility =====

def now_iso() -> str:
    return datetime.now().isoformat()


def json_or_none(value):
    """Encode a Python value as JSON, or return None if empty."""
    if value is None or value == [] or value == {}:
        return None
    return json.dumps(value)


def close():
    """Close the connection. Used by tests and CLI shutdown."""
    global _conn
    with _conn_lock:
        if _conn is not None:
            _conn.close()
            _conn = None
def backfill_fact_embeddings() -> int:
    """
    Embed any active facts that don't yet have vectors. Idempotent and safe
    to call on startup. Returns the number of facts embedded.
    """
    try:
        from memory.embed import embed_batch
    except ImportError:
        return 0

    conn = get_conn()

    # Skip if vec tables don't exist
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='fact_embeddings'"
    ).fetchone()
    if not row:
        return 0

    missing = conn.execute("""
        SELECT id, text FROM facts
        WHERE status='active'
          AND id NOT IN (SELECT fact_id FROM fact_embeddings)
    """).fetchall()

    if not missing:
        return 0

    texts = [r["text"] for r in missing]
    vectors = embed_batch(texts)
    if not vectors:
        return 0

    with write_transaction() as tx:
        for row, vec in zip(missing, vectors):
            tx.execute(
                "INSERT INTO fact_embeddings (fact_id, embedding) VALUES (?, ?)",
                (row["id"], _vec_to_blob(vec)),
            )

    print(f"🧠 Backfilled {len(missing)} fact embeddings.")
    return len(missing)


def _vec_to_blob(vector: list[float]) -> bytes:
    """sqlite-vec expects embeddings as raw float32 bytes."""
    import struct
    return struct.pack(f"{len(vector)}f", *vector)


def _blob_to_vec(blob: bytes) -> list[float]:
    import struct
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))