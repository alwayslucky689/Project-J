# tts/playback.py - Persistent audio output stream
#
# Why this exists:
#   sd.play() + sd.wait() opens and closes the audio device on every call.
#   On Windows especially, that device reconfiguration produces an audible
#   click/pop at the start of every utterance and adds ~50-100 ms of
#   startup latency. Phase 4 replaces that with one OutputStream kept open
#   for the lifetime of the process. New audio is queued; the stream's
#   callback drains the queue into the device.
#
# Threading model:
#   - play() may be called from any thread.
#   - The sounddevice callback runs on its own thread; it must be fast and
#     non-blocking, so it only touches: the chunk queue, its own "current
#     chunk" state, and a completion queue.
#   - A dedicated worker thread fires on_complete callbacks, so user code
#     never runs on the audio thread.
#
# Completion precision:
#   sounddevice gives us no direct signal that the hardware has finished
#   emitting the last sample. We approximate it: after writing the last
#   frame of an utterance to the output buffer, we output TAIL_SILENCE_MS
#   of silence before firing on_complete. This gives the hardware buffer
#   time to physically drain. Tune TAIL_SILENCE_MS if completion feels
#   early (raise it) or sluggish (lower it).

import atexit
import queue
import threading
from typing import Callable, Optional

import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("⚠️ sounddevice not installed — persistent playback unavailable")


class PersistentPlayer:
    """
    Single OutputStream that stays open for the process lifetime.

    Usage:
        player = get_player()
        player.start()                       # opens the stream (idempotent)
        player.play(audio)                   # queue and return
        player.play_and_wait(audio, 30.0)    # queue and block until played
    """

    # Silence appended after each utterance before firing on_complete.
    # Compensates for hardware buffer latency. See file header.
    TAIL_SILENCE_MS = 150

    def __init__(self, sample_rate: int = 24000, channels: int = 1, device=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device

        self._chunks: queue.Queue = queue.Queue()
        self._completions: queue.Queue = queue.Queue()
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._next_id = 0
        self._callbacks_lock = threading.Lock()

        # Audio-thread-only state (no lock needed — single consumer)
        self._current: Optional[np.ndarray] = None
        self._current_id: Optional[int] = None
        self._offset = 0
        self._tail_frames_remaining = 0
        self._tail_chunk_id: Optional[int] = None

        self._stream = None
        self._running = False
        self._start_lock = threading.Lock()

        self._completion_thread = threading.Thread(
            target=self._completion_loop, daemon=True, name="tts-completion"
        )
        self._completion_thread.start()

    # ---------- Public API ----------

    def start(self):
        """Open the OutputStream. Idempotent."""
        if self._stream is not None or not HAS_SOUNDDEVICE:
            return
        with self._start_lock:
            if self._stream is not None:
                return
            self._running = True
            try:
                self._stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=np.float32,
                    blocksize=0,   # let PortAudio pick
                    device=self.device,
                    callback=self._callback,
                )
                self._stream.start()
                print(f"🔊 Persistent playback stream open "
                      f"({self.sample_rate} Hz, {self.channels} ch).")
            except Exception as e:
                print(f"❌ Failed to open playback stream: {e}")
                self._stream = None
                self._running = False

    def play(self, audio: np.ndarray, on_complete: Optional[Callable[[], None]] = None):
        """Queue audio for playback. Returns immediately."""
        if not HAS_SOUNDDEVICE:
            if on_complete:
                on_complete()
            return

        if audio is None or audio.size == 0:
            if on_complete:
                on_complete()
            return

        if self._stream is None:
            self.start()
            if self._stream is None:
                # Stream failed to open — fire completion so the caller
                # isn't stuck waiting forever.
                if on_complete:
                    on_complete()
                return

        if audio.ndim > 1:
            audio = audio.flatten()
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        with self._callbacks_lock:
            chunk_id = self._next_id
            self._next_id += 1
            if on_complete is not None:
                self._callbacks[chunk_id] = on_complete

        self._chunks.put((chunk_id, audio))

    def play_and_wait(self, audio: np.ndarray, timeout: float = 120.0) -> bool:
        """Queue audio and block until it has finished playing."""
        done = threading.Event()
        self.play(audio, on_complete=done.set)
        return done.wait(timeout=timeout)

    def is_playing(self) -> bool:
        """True if audio is queued or currently being written to the device."""
        return (
            self._current is not None
            or not self._chunks.empty()
            or self._tail_frames_remaining > 0
        )

    def stop(self):
        """Close the stream cleanly."""
        self._running = False
        with self._start_lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    # ---------- Audio callback (runs on sounddevice's thread) ----------

    def _callback(self, outdata, frames, time_info, status):
        # Never raise from here — a crashed callback takes the whole stream down.
        if status:
            # xrun or similar; ignore but don't propagate
            pass

        outdata.fill(0)
        written = 0

        while written < frames:
            # No current chunk — pull one, or emit tail silence, or stop.
            if self._current is None:
                if self._tail_frames_remaining > 0:
                    take = min(frames - written, self._tail_frames_remaining)
                    written += take
                    self._tail_frames_remaining -= take
                    if self._tail_frames_remaining == 0:
                        cid = self._tail_chunk_id
                        self._tail_chunk_id = None
                        if cid is not None:
                            self._completions.put(cid)
                    continue

                try:
                    chunk_id, audio = self._chunks.get_nowait()
                except queue.Empty:
                    break  # nothing more; remainder of outdata stays silence

                self._current = audio
                self._current_id = chunk_id
                self._offset = 0

            # Copy from current chunk into outdata
            remaining = len(self._current) - self._offset
            take = min(frames - written, remaining)
            outdata[written:written + take, 0] = \
                self._current[self._offset:self._offset + take]
            self._offset += take
            written += take

            # Chunk finished?
            if self._offset >= len(self._current):
                cid = self._current_id
                self._current = None
                self._current_id = None
                self._offset = 0

                # If another chunk is already queued, chain seamlessly.
                # Otherwise enter the tail-silence phase so the hardware
                # can drain before we signal on_complete.
                if not self._chunks.empty():
                    self._completions.put(cid)
                else:
                    tail_frames = int(
                        self.sample_rate * (self.TAIL_SILENCE_MS / 1000.0)
                    )
                    if tail_frames > 0:
                        self._tail_frames_remaining = tail_frames
                        self._tail_chunk_id = cid
                    else:
                        self._completions.put(cid)

    # ---------- Completion worker (off the audio thread) ----------

    def _completion_loop(self):
        while True:
            try:
                cid = self._completions.get(timeout=0.2)
            except queue.Empty:
                if not self._running and self._stream is None:
                    return
                continue

            with self._callbacks_lock:
                cb = self._callbacks.pop(cid, None)

            if cb is not None:
                try:
                    cb()
                except Exception as e:
                    print(f"⚠️ TTS completion callback error: {e}")
    def wait_until_idle(self, timeout: float = 120.0):
        import time
        """Block until all queued audio has finished playing."""
        start = time.perf_counter()
        while self.is_playing() and (time.perf_counter() - start) < timeout:
            time.sleep(0.02)

# ===== Module-level singleton =====

_player: Optional[PersistentPlayer] = None
_player_lock = threading.Lock()


def get_player(sample_rate: int = 24000) -> PersistentPlayer:
    global _player
    if _player is None:
        with _player_lock:
            if _player is None:
                _player = PersistentPlayer(sample_rate=sample_rate)
    return _player


def shutdown_player():
    global _player
    if _player is not None:
        _player.stop()
        _player = None


atexit.register(shutdown_player)