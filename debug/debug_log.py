# core/debug_log.py - Async structured logger for the assistant
#
# Writes one log file per session. On startup, rotates the previous
# session.log to session.prev.log. All writes go through a queue so
# callers never block on I/O. GPU stats come from a background thread
# (pynvml if available, subprocess fallback otherwise).
#
# Thread-safe. Callers use module-level functions — no need to pass
# the logger around.

import atexit
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

LOG_DIR = Path("logs")
LOG_CURRENT = LOG_DIR / "session.log"
LOG_PREVIOUS = LOG_DIR / "session.prev.log"

_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue()
_writer_thread: Optional[threading.Thread] = None
_stop = threading.Event()
_session_id: Optional[int] = None
_debug_console = False
_gpu_stop = threading.Event()
_gpu_thread: Optional[threading.Thread] = None


def _ts() -> str:
    t = time.time()
    lt = time.localtime(t)
    ms = int((t - int(t)) * 1000)
    return f"{lt.tm_hour:02d}:{lt.tm_min:02d}:{lt.tm_sec:02d}.{ms:03d}"


# ===== Public API =====

def init(session_id: Optional[int] = None, debug: bool = False,
         gpu_poll_seconds: float = 5.0):
    """Start the logger. Idempotent."""
    global _session_id, _debug_console, _writer_thread, _gpu_thread
    with _lock:
        if _writer_thread is not None:
            return
        _session_id = session_id
        _debug_console = debug

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        # Rotate current → previous
        if LOG_CURRENT.exists():
            try:
                shutil.move(str(LOG_CURRENT), str(LOG_PREVIOUS))
            except Exception:
                pass

        _stop.clear()
        _writer_thread = threading.Thread(
            target=_writer_loop, daemon=True, name="debug-log-writer"
        )
        _writer_thread.start()

        _gpu_stop.clear()
        _gpu_thread = threading.Thread(
            target=_gpu_loop, args=(gpu_poll_seconds,),
            daemon=True, name="debug-log-gpu",
        )
        _gpu_thread.start()

        event("SESSION", f"{session_id or '?'} started")


_shutdown_done = threading.Event()

def shutdown():
    if _shutdown_done.is_set():
        return
    _shutdown_done.set()
    global _writer_thread, _gpu_thread
    _gpu_stop.set()
    _stop.set()
    if _writer_thread is not None:
        _writer_thread.join(timeout=2.0)
        _writer_thread = None
    _gpu_thread = None
    try:
        with open(LOG_CURRENT, "a", encoding="utf-8") as f:
            f.write(f"[{_ts()}] SESSION ended\n")
    except Exception:
        pass


def event(category: str, message: str):
    """Log one event line. category is a short tag like 'ROUTE', 'LLM'."""
    line = f"[{_ts()}] {category}: {message}"
    _queue.put(line)
    if _debug_console:
        print(line)


def turn_start(user_text: str):
    event("────", "─" * 60)
    event("USER", user_text)


def route(tool: str, elapsed_ms: float, confidence: Optional[float] = None,
          candidates: Optional[list[str]] = None):
    conf = f", conf={confidence:.2f}" if confidence is not None else ""
    cand = f", candidates={candidates}" if candidates else ""
    event("ROUTE", f"→ {tool} ({elapsed_ms:.0f}ms{conf}{cand})")


def memory(records: list):
    if not records:
        event("MEMORY", "0 records")
        return
    facts = sum(1 for r in records if r.get("kind") == "fact")
    sessions = sum(1 for r in records if r.get("kind") == "session")
    event("MEMORY", f"{len(records)} records ({facts} facts, {sessions} sessions)")


def llm_start(backend: str, model: str):
    event("LLM", f"start: backend={backend} model={model}")


def llm_first_token(elapsed_ms: float):
    event("LLM", f"first token at {elapsed_ms:.0f}ms")


def llm_done(tokens: int, elapsed_ms: float):
    tps = (tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0
    event("LLM", f"{tokens} tokens in {elapsed_ms:.0f}ms ({tps:.2f} tok/s)")


def assistant(text: str, max_len: int = 4000):
    """
    Log the full assistant response. Long responses are wrapped onto
    multiple indented lines so the log stays greppable and readable.
    Anything over max_len is truncated with a marker — raise it if you
    need the full text, but be aware the log grows fast.
    """
    if not text:
        return
    body = text.strip()
    if len(body) > max_len:
        body = body[:max_len] + f"\n[...truncated at {max_len} chars]"

    # First line keeps the ASSISTANT: prefix so grep still works
    lines = body.split("\n")
    event("ASSISTANT", lines[0])
    for line in lines[1:]:
        _queue.put(f"[{_ts()}]   {line}")
    if _debug_console:
        for line in lines[1:]:
            print(f"[{_ts()}]   {line}")


def tts_start():
    event("TTS", "start")


def tts_done(elapsed_ms: float):
    event("TTS", f"{elapsed_ms:.0f}ms")


def turn_end(elapsed_ms: float):
    event("TURN TOTAL", f"{elapsed_ms:.0f}ms")


def gpu(vram_used_mb: int, vram_total_mb: int, util_pct: int):
    event("GPU", f"vram={vram_used_mb}/{vram_total_mb}MB util={util_pct}%")


def error(msg: str):
    event("ERROR", msg)


# ===== Writer thread =====

def _writer_loop():
    with open(LOG_CURRENT, "a", encoding="utf-8", buffering=1) as f:
        while not _stop.is_set() or not _queue.empty():
            try:
                line = _queue.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                f.write(line + "\n")
            except Exception:
                pass


# ===== GPU polling =====

def _gpu_loop(interval: float):
    reader = _make_gpu_reader()
    if reader is None:
        event("GPU monitor", "unavailable")
        return
    event("GPU monitor", reader[0])
    read = reader[1]
    while not _gpu_stop.wait(timeout=interval):
        try:
            stats = read()
            if stats is not None:
                gpu(*stats)
        except Exception:
            pass


def _make_gpu_reader():
    """Return (source_name, reader_fn) or None."""
    # Prefer pynvml — in-process, no subprocess spawn
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        total = pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024 * 1024)

        def _read():
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            u = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return (m.used // (1024 * 1024), total, u.gpu)

        return ("pynvml", _read)
    except Exception:
        pass

    # Fallback: subprocess nvidia-smi
    try:
        import subprocess
        def _read():
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=2.0,
            ).strip().splitlines()[0]
            used, total, util = [int(x.strip()) for x in out.split(",")]
            return (used, total, util)
        return ("nvidia-smi", _read)
    except Exception:
        return None


atexit.register(shutdown)