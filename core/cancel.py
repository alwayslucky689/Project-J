# core/cancel.py - Cooperative cancellation for LLM/TTS pipelines
#
# A shared token. Any layer can call .cancel() to signal, any layer can
# call .is_cancelled() to check. When cancelled mid-stream, the streaming
# layers close their HTTP responses and drain their queues; the caller
# stores whatever partial content was emitted.

import threading
from typing import Optional


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