# streaming_tts.py - Chunked streaming TTS wrapper for OmniVoice
#
# Splits LLM token stream into sentences, synthesizes each in a background
# thread, plays them back-to-back in another background thread.
#
# Usage:
#     streamer = StreamingTTS()
#     streamer.start()
#     for char in llm_stream:
#         streamer.push_token(char)
#     streamer.finish()  # blocks until all audio has played

import re
import threading
import queue
import time


_SENTINEL = object()
_SENTENCE_END = re.compile(r'[.!?]+(?=\s|$)')


class StreamingTTS:
    def __init__(self, min_chunk_chars=15, max_chunk_chars=200):
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars

        self.text_buffer = ""
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        self._synth_thread = None
        self._player_thread = None
        self._stopped = False

    # ============ Public API ============

    def start(self):
        """Spawn the two background workers."""
        self._synth_thread = threading.Thread(
            target=self._synth_loop, daemon=True, name="tts-synth"
        )
        self._player_thread = threading.Thread(
            target=self._player_loop, daemon=True, name="tts-player"
        )
        self._synth_thread.start()
        self._player_thread.start()

    def push_token(self, token):
        """Feed a single character/token from the LLM stream."""
        if self._stopped:
            return
        self.text_buffer += token
        chunk = self._try_extract_chunk()
        if chunk:
            self.text_queue.put(chunk)

    def finish(self):
        """Flush remaining text, wait for all audio to finish playing."""
        if self._stopped:
            return

        # Flush whatever's left in the buffer
        leftover = self.text_buffer.strip()
        if leftover:
            self.text_queue.put(leftover)
        self.text_buffer = ""

        # Signal synth thread to stop
        self.text_queue.put(_SENTINEL)
        if self._synth_thread:
            self._synth_thread.join()

        # Signal player thread to stop
        self.audio_queue.put(_SENTINEL)
        if self._player_thread:
            self._player_thread.join()

        # Let the hardware buffer drain (pygame reports "not busy" before
        # the speaker has physically finished emitting the last sample)
        time.sleep(0.25)

        self._stopped = True

    # ============ Internal — Chunking ============

    def _try_extract_chunk(self):
        """Return the next ready chunk, or None if we should wait for more."""
        buf = self.text_buffer

        if len(buf) < self.min_chunk_chars:
            return None

        # Look for the last sentence boundary within max_chunk_chars
        search_region = buf[:self.max_chunk_chars]
        matches = list(_SENTENCE_END.finditer(search_region))
        if matches:
            end = matches[-1].end()
            chunk = buf[:end].strip()
            self.text_buffer = buf[end:].lstrip()
            return chunk if chunk else None

        # No sentence boundary but buffer is too long — split at last space
        if len(buf) >= self.max_chunk_chars:
            split_at = buf.rfind(" ", 0, self.max_chunk_chars)
            if split_at == -1:
                # No space at all — force-cut (very rare)
                split_at = self.max_chunk_chars
            chunk = buf[:split_at].strip()
            self.text_buffer = buf[split_at:].lstrip()
            return chunk if chunk else None

        return None

    # ============ Internal — Worker Threads ============

    def _synth_loop(self):
        """Pull sentences, synthesize to audio, push to audio_queue."""
        from tts.tts_manager import generate_chunk_for_streaming

        while True:
            sentence = self.text_queue.get()
            if sentence is _SENTINEL:
                break
            try:
                audio = generate_chunk_for_streaming(sentence)
                if audio is not None and len(audio) > 0:
                    self.audio_queue.put(audio)
            except Exception as e:
                print(f"⚠️ [streaming_tts] synth error: {e}")

    def _player_loop(self):
        """Pull audio arrays, play them sequentially."""
        from tts.tts_manager import play_chunk_blocking

        while True:
            audio = self.audio_queue.get()
            if audio is _SENTINEL:
                break
            try:
                play_chunk_blocking(audio)
            except Exception as e:
                print(f"⚠️ [streaming_tts] playback error: {e}")