# core/bonsai_server.py - Manage the Bonsai llama-server subprocess
#
# Started by the assistant on launch (if CHAT_BACKEND == "bonsai" and
# BONSAI_AUTOSTART is True) and killed on shutdown. If a server is already
# running on the configured port, we don't spawn another — we assume the
# user started it manually and leave it alone.
#
# Readiness is detected by polling the TCP port, which works regardless
# of llama-server's log format.

import os
import socket
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path

from config import settings


_process = None
_lock = threading.Lock()


def _host_port() -> tuple[str, int]:
    parsed = urllib.parse.urlparse(settings.BONSAI_URL)
    return parsed.hostname or "127.0.0.1", parsed.port or 8080


def is_running(timeout: float = 0.5) -> bool:
    """True if something is listening on the configured Bonsai port."""
    host, port = _host_port()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def start() -> bool:
    """
    Ensure the Bonsai server is running. Returns True on success.
    If a server is already listening, returns True without spawning.
    """
    global _process
    with _lock:
        if is_running():
            print("🌳 Bonsai server already running.")
            return True

        exe = Path(getattr(settings, "BONSAI_SERVER_EXE", "")).expanduser()
        model = Path(getattr(settings, "BONSAI_MODEL_PATH", "")).expanduser()
        if not exe.exists():
            print(f"❌ Bonsai exe not found: {exe}")
            return False
        if not model.exists():
            print(f"❌ Bonsai model not found: {model}")
            return False

        log_path = Path(getattr(settings, "BONSAI_LOG_PATH",
                                "data/bonsai_server.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")

        host, port = _host_port()
        cmd = [
            str(exe),
            "-m", str(model),
            "-ngl", str(getattr(settings, "BONSAI_NGL", 99)),
            "-c", str(getattr(settings, "BONSAI_CTX", 8192)),
            "--jinja",
            "--host", host,
            "--port", str(port),
        ]

        env = os.environ.copy()
        if getattr(settings, "BONSAI_DISABLE_THINKING", True):
            # Their llama-server version still reads this env var even
            # though it prints a deprecation notice.
            env["LLAMA_ARG_CHAT_TEMPLATE_KWARGS"] = '{"enable_thinking": false}'

        print(f"🌳 Starting Bonsai server ({exe.name})...")
        print(f"   model:  {model.name}")
        print(f"   log:    {log_path}")

        try:
            _process = subprocess.Popen(
                cmd,
                cwd=str(exe.parent),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
        except Exception as e:
            print(f"❌ Failed to spawn Bonsai server: {e}")
            return False

    # Wait outside the lock so is_running() (called from here) can
    # acquire it if needed by a future change.
    deadline = time.time() + getattr(settings, "BONSAI_STARTUP_TIMEOUT", 120.0)
    t0 = time.time()
    while time.time() < deadline:
        if _process.poll() is not None:
            print(f"❌ Bonsai server exited early "
                  f"(code {_process.returncode}). Check {log_path}")
            return False
        if is_running(timeout=0.3):
            elapsed = time.time() - t0
            print(f"✅ Bonsai server ready ({elapsed:.1f}s).")
            return True
        time.sleep(0.5)

    print(f"⚠️ Bonsai server didn't respond within "
          f"{getattr(settings, 'BONSAI_STARTUP_TIMEOUT', 120.0):.0f}s")
    return False


def stop():
    """Kill the server if we started it. No-op if it was already running."""
    global _process
    with _lock:
        if _process is None:
            return
        if _process.poll() is not None:
            _process = None
            return
        print("🌳 Stopping Bonsai server...")
        try:
            _process.terminate()
            _process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _process.kill()
        except Exception as e:
            print(f"⚠️ Error stopping Bonsai: {e}")
        _process = None