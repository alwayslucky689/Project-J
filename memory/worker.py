# memory/worker.py - Background extraction worker
#
# Runs in a daemon thread. Every N turns OR 60s of idle, it wakes up,
# loads unprocessed turns from the checkpoint, and (in 6e-2) extracts
# candidate facts.
#
# For 6e-1, it just logs what it sees. This validates the trigger logic
# and checkpoint mechanism before adding the extraction pipeline.
from debug import debug_log as dbg
import threading
import time

from memory.sessions import (
    get_unprocessed_turns,
    save_checkpoint,
    _load_checkpoint,
)
from config import settings
def _log(msg: str):
    dbg.event("WORKER", msg)

def _log(msg: str):
    dbg.event("WORKER", msg)

# Trigger thresholds
TURNS_PER_BATCH = 3        # Run after this many new turns
IDLE_SECONDS = 60.0        # Run after this long with no activity
FAILURE_BACKOFF_SECONDS = 30.0
POLL_INTERVAL = 2.0        # How often the loop wakes to check triggers


class _Worker:
    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_activity = time.time()

    # ---- Public API ----

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="memory-worker"
        )
        self._thread.start()
       

    def stop(self, timeout: float = 60.0):
        """Stop the worker. Waits for any in-flight batch to complete so
        extraction events reach disk before the process exits. Extraction
        calls the LLM once per hard turn, so a batch can take 15-30s."""
        if not self._stop.is_set():
            dbg.event("WORKER", "stop requested; finishing current batch")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                dbg.event("WORKER",
                          f"stop timed out after {timeout}s; "
                          f"thread still running")
        

    def notify_activity(self):
        """Called when a new turn is written, to reset the idle timer."""
        self._last_activity = time.time()

    # ---- Loop ----

    def _loop(self):
        dbg.event("WORKER", "loop entered")
        idle_logs = 0
        while not self._stop.is_set():
            try:
                if self._should_run():
                    dbg.event("WORKER", "should_run=True")
                    self._run_batch()
                else:
                    idle_logs += 1
                    # Log once every ~30s that we're alive and polling.
                    if idle_logs % 15 == 0:
                        from memory.sessions import _load_checkpoint
                        ck = _load_checkpoint()
                        last_id = ck.get("last_processed_turn_id", 0)
                        pending = get_unprocessed_turns(
                            limit=TURNS_PER_BATCH + 1
                        )
                        dbg.event(
                            "WORKER",
                            f"idle: checkpoint={last_id} "
                            f"pending={len(pending)}"
                        )

                if self._stop.is_set():
                    break
                self._drain_summaries()
            except Exception as e:
                dbg.event("WORKER", f"loop error: {e}")
                self._sleep(FAILURE_BACKOFF_SECONDS)
                continue
            self._sleep(POLL_INTERVAL)
    def _drain_summaries(self):
        from memory.sessions import pop_pending_summary
        while True:
            sid = pop_pending_summary()
            if sid is None:
                break
            try:
                from memory.summarize import summarize_session
                summary = summarize_session(sid)
                if summary:
                    _log(f"📝 Summarized session {sid}: {summary[:60]}...")
                    self._merge_topic_state(sid, summary)
            except Exception as e:
                _log(f"⚠️ Summary failed for session {sid}: {e}")

    def _merge_topic_state(self, session_id: int, session_summary: str):
        """If the session belonged to a topic, fold its summary into the
        topic's structured state. No-op for global sessions."""
        from memory.sessions import get_session_context
        ctx = get_session_context(session_id)
        if not ctx or not ctx.startswith("topic:"):
            return
        try:
            topic_id = int(ctx.split(":", 1)[1])
        except (ValueError, IndexError):
            return
        try:
            from memory.topic_state import merge_session_into_state
            new_state = merge_session_into_state(topic_id, session_summary)
            if new_state is not None:
                _log(f"🧩 Topic {topic_id} state updated.")
        except Exception as e:
            _log(f"⚠️ Topic state merge failed for topic {topic_id}: {e}")

    def _should_run(self) -> bool:
        checkpoint = _load_checkpoint()
        last_id = checkpoint.get("last_processed_turn_id", 0)

        pending = get_unprocessed_turns(limit=TURNS_PER_BATCH + 1)
        if not pending:
            return False

        # Turn-count trigger: at least N pending turns
        if len(pending) >= TURNS_PER_BATCH:
            return True

        # Idle trigger: no activity for IDLE_SECONDS, but pending turns exist
        if (time.time() - self._last_activity) > IDLE_SECONDS:
            return True

        return False

    def _run_batch(self):
        dbg.event("WORKER", ">>> RUN_BATCH ENTERED <<<")
        pending = get_unprocessed_turns(limit=50)
        dbg.event("WORKER", f">>> got {len(pending)} turns <<<")
        if not pending:
            return

        dbg.event("WORKER", f"batch: {len(pending)} pending turns")

        try:
            from memory.extract import extract_candidates
            raw = extract_candidates(pending)
            dbg.event("WORKER", f"extract: {len(raw)} candidates")
        except Exception as e:
            dbg.event("WORKER", f"extract crashed: {e}")
            raw = []

        kept = []
        for c in raw:
            c["source_text"] = c.get("text", "")
            from memory.validate import validate_candidate
            ok, revised, reason = validate_candidate(c)
            if ok:
                kept.append(revised)
            else:
                dbg.event("WORKER", f"validate rejected: {reason} | "
                                    f"{c.get('text','?')[:60]}")

        dbg.event("WORKER", f"validate: {len(kept)}/{len(raw)} kept")

        actions = {}
        for c in kept:
            try:
                from memory.resolve import resolve_candidate
                action = resolve_candidate(c)
                actions[action] = actions.get(action, 0) + 1
            except Exception as e:
                dbg.event("WORKER", f"resolve failed: {e} | "
                                    f"{c.get('text','?')[:60]}")

        if actions:
            dbg.event("WORKER", f"resolved: {actions}")
        else:
            dbg.event("WORKER", "resolved: no actions")

        highest_id = max(t["id"] for t in pending)
        checkpoint = _load_checkpoint()
        checkpoint["last_processed_turn_id"] = highest_id
        checkpoint["last_run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        checkpoint["consecutive_failures"] = 0
        save_checkpoint(checkpoint)
        self._last_activity = time.time()

    def _sleep(self, seconds: float):
        """
        Interruptible sleep. Returns immediately if stop() is called,
        so shutdown doesn't wait out a full polling interval.
        """
        self._stop.wait(timeout=seconds)
# ===== Module-level singleton =====

_worker: _Worker | None = None
_lock = threading.Lock()


def get_worker() -> _Worker:
    global _worker
    if _worker is None:
        with _lock:
            if _worker is None:
                _worker = _Worker()
    return _worker

def try_inline_extract(user_input: str, session_id: int) -> bool:
    """
    Synchronous single-turn extraction. Runs after the assistant has
    responded and TTS has finished, before the next input() prompt.

    Uses ONLY the spaCy path — no LLM call — to keep latency near zero
    (~50ms on hits, 0ms on prefilter misses). Facts needing the LLM
    fallback are left for the worker to pick up in its next batch.

    The checkpoint is NOT advanced here. The worker re-processes the
    turn later; resolve_candidate dedups via the exact-duplicate check.

    Returns True if extraction ran (whether or not anything was saved),
    False if the prefilter rejected the turn (no work needed).
    """
    from memory.extract import _looks_memory_worthy, extract_candidates
    from memory.validate import validate_candidate
    from memory.resolve import resolve_candidate
    from memory.store import get_conn

    if not user_input or not _looks_memory_worthy(user_input):
        return False

    conn = get_conn()

    # Most recent user turn — the one we just persisted
    turn_row = conn.execute(
        "SELECT id FROM turns WHERE session_id = ? AND role='user' "
        "ORDER BY id DESC LIMIT 1",
        (session_id,)
    ).fetchone()
    if not turn_row:
        return False

    sess_row = conn.execute(
        "SELECT context FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    context = sess_row["context"] if sess_row else "global"

    synthetic = {
        "id": turn_row["id"],
        "role": "user",
        "text": user_input,
        "context": context,
    }

    try:
        candidates = extract_candidates([synthetic], spacy_only=True)
    except Exception as e:
        dbg.event("EXTRACT", f"turn {turn_row['id']} extract error: {e}")
        return False

    if not candidates:
        dbg.event("EXTRACT", f"turn {turn_row['id']}: no spaCy candidates "
                             f"(LLM fallback left to worker)")
        return True

    for c in candidates:
        ok, revised, reason = validate_candidate(c)
        if not ok:
            dbg.event("EXTRACT", f"turn {turn_row['id']} rejected: {reason} | "
                                 f"{c.get('text', '?')[:60]}")
            continue
        try:
            action = resolve_candidate(revised)
        except Exception as e:
            dbg.event("EXTRACT", f"turn {turn_row['id']} resolve error: {e}")
            continue
        dbg.event("EXTRACT", f"turn {turn_row['id']}: {action} | "
                             f"{revised.get('text', '?')[:80]}")

    return True

def start_worker():
    get_worker().start()

def stop_worker():
    if _worker is not None:
        _worker.stop(timeout=60.0)


def notify_activity():
    if _worker is not None:
        _worker.notify_activity()