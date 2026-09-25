# core/conversation.py - Session-keyed conversation buffers
#
# Each open session has its own buffer. Buffers survive context switches
# (suspend) but are discarded when a session closes. On startup, if a
# session is resumed, its buffer is reconstructed from the turns table.

import threading
from collections import deque
from typing import Callable, Optional


class ConversationManager:
    """
    Manages per-session conversation buffers.

    Public API:
        get_messages(session_id, current_user_input="") -> list[dict]
        add_user(session_id, text)
        add_assistant(session_id, text)
        load_from_db(session_id, limit=20) -> int
        discard(session_id)
        clear_all()
    """

    def __init__(self, system_prompt_fn: Callable[[str], str],
                 max_turns: int = 10):
        self._system_prompt_fn = system_prompt_fn
        self._max_turns = max_turns
        self._buffers: dict[int, deque] = {}
        self._lock = threading.Lock()

    # ===== Internal =====

    def _buffer(self, session_id: int) -> deque:
        buf = self._buffers.get(session_id)
        if buf is None:
            buf = deque(maxlen=self._max_turns * 2)
            self._buffers[session_id] = buf
        return buf

    # ===== Mutation =====

    def add_user(self, session_id: int, text: str):
        with self._lock:
            self._buffer(session_id).append(
                {"role": "user", "content": text}
            )

    def add_assistant(self, session_id: int, text: str):
        with self._lock:
            self._buffer(session_id).append(
                {"role": "assistant", "content": text}
            )

    def discard(self, session_id: int):
        with self._lock:
            self._buffers.pop(session_id, None)

    def clear_all(self):
        with self._lock:
            self._buffers.clear()

    # ===== Read =====

    def get_messages(self, session_id: int,
                     current_user_input: str = "") -> list[dict]:
        """
        Build the message list to send to the LLM.
        Order: system prompt, memory exchange (if any), buffer, current query.
        """
        with self._lock:
            msgs = list(self._buffer(session_id))

        system = {
            "role": "system",
            "content": self._system_prompt_fn(current_user_input),
        }
        result = [system] + msgs

        if current_user_input:
            result.append({"role": "user", "content": current_user_input})

            try:
                from memory.retrieval import assemble_memory
                from memory.inject import inject_into_messages
                records = assemble_memory(current_user_input, session_id)
                if records:
                    result = inject_into_messages(result, records)
            except Exception as e:
                print(f"⚠️ Memory assembly failed: {e}")

        return result

    def load_from_db(self, session_id: int, limit: int = 20) -> int:
        """
        Reconstruct a session's buffer from the turns table.
        Returns the number of turns loaded.
        """
        from memory.sessions import load_recent_turns
        rows = load_recent_turns(session_id, limit=limit)
        with self._lock:
            buf = deque(maxlen=self._max_turns * 2)
            for r in rows:
                buf.append({"role": r["role"], "content": r["text"]})
            self._buffers[session_id] = buf
        return len(rows)


# ===== Module-level singleton =====

_manager: Optional[ConversationManager] = None
_manager_lock = threading.Lock()


def get_manager(system_prompt_fn: Optional[Callable[[str], str]] = None,
                max_turns: int = 10) -> ConversationManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                if system_prompt_fn is None:
                    raise RuntimeError(
                        "get_manager() must be called once with system_prompt_fn"
                    )
                _manager = ConversationManager(system_prompt_fn, max_turns)
    return _manager