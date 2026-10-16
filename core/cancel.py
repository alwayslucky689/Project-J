# core/cancel.py - Cooperative cancellation for LLM/TTS pipelines
#
# A shared token. Any layer can call .cancel() to signal, any layer can
# call .is_cancelled() to check. When cancelled mid-stream, the streaming
# layers close their HTTP responses and drain their queues; the caller
# stores whatever partial content was emitted.

import threading
from typing import Optional
# ===== Esc / typed-word cancellation listener =====
#
# One background thread that owns stdin while a request is active and
# leaves it alone when idle, so input() still works normally.

import msvcrt
import time

_listener_thread = None
_listener_stop = threading.Event()

# Words that cancel an active request when typed and Enter is pressed.
_CANCEL_WORDS = {"stop", "cancel", "abort", "halt"}


def start_listener():
    """Start the background cancel listener. Idempotent.

    Behavior:
      - Idle (no active token): does not read stdin at all, so input()
        works normally.
      - Active token: consumes keystrokes. Esc cancels immediately.
        Typing one of the cancel words followed by Enter cancels.
    """
    global _listener_thread
    if _listener_thread is not None and _listener_thread.is_alive():
        return
    _listener_stop.clear()

    def _listen():
        typed_buffer = ""
        while not _listener_stop.is_set():
            try:
                tok = get_current()
                active = tok is not None and not tok.is_cancelled()

                # Idle: don't eat characters — let input() handle stdin.
                if not active:
                    typed_buffer = ""
                    time.sleep(0.05)
                    continue

                if msvcrt.kbhit():
                    ch = msvcrt.getwch()

                    if ch == "\x1b":               # Esc
                        print("\n⏹️  Cancelled (Esc).")
                        tok.cancel()
                        typed_buffer = ""

                    elif ch in ("\r", "\n"):       # Enter — evaluate word
                        word = typed_buffer.strip().lower()
                        if word in _CANCEL_WORDS:
                            print(f"\n⏹️  Cancelled ({word}).")
                            tok.cancel()
                        typed_buffer = ""

                    elif ch in ("\b", "\x7f"):     # Backspace
                        typed_buffer = typed_buffer[:-1]

                    elif ch.isprintable():
                        typed_buffer += ch
                else:
                    time.sleep(0.05)

            except Exception:
                time.sleep(0.2)

    _listener_thread = threading.Thread(
        target=_listen, daemon=True, name="cancel-listener"
    )
    _listener_thread.start()


def stop_listener():
    """Signal the listener thread to exit. Safe to call even if not running."""
    _listener_stop.set()

class CancelToken:
    def __init__(self):
        self._event = threading.Event()
        self._response = None       # HTTP response to close if cancelled
        self._streamer = None       # StreamingTTS to stop if cancelled
        self._lock = threading.Lock()

    def register_response(self, response):
        """Called by the streaming layer when it opens an HTTP response."""
        with self._lock:
            if self._event.is_set():
                try: response.close()
                except Exception: pass
                return
            self._response = response

    def register_streamer(self, streamer):
        """Called by the chat path when it creates a StreamingTTS."""
        with self._lock:
            if self._event.is_set():
                try: streamer.stop()
                except Exception: pass
                return
            self._streamer = streamer

    def cancel(self):
        """Idempotent. Closes the HTTP response and stops TTS."""
        if self._event.is_set():
            return
        self._event.set()
        with self._lock:
            if self._response is not None:
                try: self._response.close()
                except Exception: pass
            if self._streamer is not None:
                try: self._streamer.stop()
                except Exception: pass

    def is_cancelled(self) -> bool:
        return self._event.is_set()


# Module-level current token. Set by the chat path before streaming,
# checked by the keyboard listener. Only one active request at a time
# (the assistant is single-threaded in its request loop).
_current: Optional[CancelToken] = None
_current_lock = threading.Lock()


def set_current(token: Optional[CancelToken]):
    global _current
    with _current_lock:
        _current = token


def get_current() -> Optional[CancelToken]:
    with _current_lock:
        return _current