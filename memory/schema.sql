-- memory/schema.sql
-- Canonical schema for Project-J memory.
--
-- Design notes:
--   * Canonical facts live separately from embeddings (embeddings table is
--     added in migration 2, deferred to Phase 6c). This means we can rebuild
--     the search index without touching what a fact means.
--   * Every fact has a typed structure (subject, property, value) AND a
--     natural-language text field. The structured fields are how we resolve
--     conflicts; the text is what gets injected into prompts.
--   * scope_type distinguishes:
--       - 'global'   : applies everywhere (user prefers dark mode)
--       - 'unscoped' : not yet classified, retrieved at lower rank
--       - 'topic'    : scoped to a specific project/topic
--   * source/epistemic_status encode HOW we know something, which is
--     different from WHAT we know.

-- ===== Schema version tracking =====
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL
);

-- ===== Users (single-user for now, multi-user later) =====
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT,
    timezone TEXT DEFAULT 'UTC',
    created_at TIMESTAMP NOT NULL
);

-- ===== Sessions =====
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'default',
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    summary TEXT,
    active_topic_id INTEGER,
    private_session BOOLEAN DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_time
    ON sessions(user_id, started_at DESC);

-- ===== Raw turns (evidence) =====
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    role TEXT NOT NULL,           -- 'user' | 'assistant'
    text TEXT NOT NULL,
    ts TIMESTAMP NOT NULL,
    excluded_from_extraction BOOLEAN DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_turns_session
    ON turns(session_id, ts);

-- ===== Topics/projects =====
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    aliases TEXT,                 -- JSON array: ["Project-J", "the app"]
    description TEXT,
    status TEXT NOT NULL DEFAULT 'candidate',  -- candidate | active | archived
    created_at TIMESTAMP NOT NULL,
    last_active_at TIMESTAMP NOT NULL,
    UNIQUE(user_id, name),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- ===== Facts (typed claims) =====
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'default',

    -- Typed structure (structured extraction lands in Phase 6e;
    -- until then, subject defaults to 'user' and property/value are null)
    subject TEXT NOT NULL,
    property TEXT,
    value TEXT,
    text TEXT NOT NULL,

    -- Provenance
    source TEXT NOT NULL,           -- user_explicit | user_asserted | inferred | web
    epistemic_status TEXT NOT NULL, -- assertion | plan | observation | inference | hypothetical
    extraction_confidence REAL,     -- NULL for explicit, 0-1 for inferred

    -- Scope
    scope_type TEXT NOT NULL,       -- global | unscoped | topic
    scope_topic_id INTEGER,

    -- Validity
    valid_from TIMESTAMP,
    valid_until TIMESTAMP,

    -- Lifecycle
    status TEXT NOT NULL DEFAULT 'candidate',
        -- candidate | active | superseded | disputed | deleted
    supersedes INTEGER,
    superseded_by INTEGER,

    -- Evidence
    evidence_turn_ids TEXT,         -- JSON array of turn ids

    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,

    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (scope_topic_id) REFERENCES topics(id) ON DELETE SET NULL,
    FOREIGN KEY (supersedes) REFERENCES facts(id) ON DELETE SET NULL,
    FOREIGN KEY (superseded_by) REFERENCES facts(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_active_lookup
    ON facts(user_id, status, scope_type, subject, property);

CREATE INDEX IF NOT EXISTS idx_facts_recent
    ON facts(user_id, created_at DESC);

-- ===== Fact <-> Topic many-to-many =====
CREATE TABLE IF NOT EXISTS fact_topics (
    fact_id INTEGER NOT NULL,
    topic_id INTEGER NOT NULL,
    PRIMARY KEY (fact_id, topic_id),
    FOREIGN KEY (fact_id) REFERENCES facts(id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
);

-- ===== Tombstones (prevent re-extraction after forget) =====
CREATE TABLE IF NOT EXISTS tombstones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'default',
    subject TEXT NOT NULL,
    property TEXT,
    scope_type TEXT NOT NULL,
    scope_topic_id INTEGER,
    deleted_at TIMESTAMP NOT NULL,
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_tombstones_lookup
    ON tombstones(user_id, subject, property, scope_type);

-- ===== Habits (Phase 6h, unused for now) =====
CREATE TABLE IF NOT EXISTS habits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'default',
    name TEXT,
    description TEXT,
    actions TEXT NOT NULL,
    context TEXT,
    status TEXT NOT NULL DEFAULT 'candidate',
    occurrences INTEGER DEFAULT 0,
    opportunities INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0.0,
    first_seen_at TIMESTAMP NOT NULL,
    last_seen_at TIMESTAMP NOT NULL,
    cooldown_until TIMESTAMP,
    last_suggested_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- ===== Extraction checkpoint (Phase 6e, unused for now) =====
CREATE TABLE IF NOT EXISTS extraction_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP NOT NULL
);

-- ===== Embedding model versioning (Phase 6c, unused for now) =====
CREATE TABLE IF NOT EXISTS embedding_versions (
    model_id TEXT PRIMARY KEY,
    dimensions INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL
);

-- ===== Multi-agent coordination (future, unused) =====
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assigned_to_agent TEXT,
    created_by_agent TEXT,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT,
    event_type TEXT NOT NULL,
    payload TEXT,
    ts TIMESTAMP NOT NULL
);