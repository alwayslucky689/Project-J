"""Phase 3a verification: TTS starts before the LLM has finished."""
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Simulate an LLM token stream: a paragraph that trickles in over ~3 seconds.
# If TTS is streaming, playback should begin well before the last token.
FAKE_TOKENS = (
    "Certainly, Sir. "               # sentence 1
    "The matter you raise is one of some interest. "  # sentence 2
    "I have examined the parameters. "                 # sentence 3
    "All systems are operating within normal ranges."  # sentence 4
).split(" ")  # crude tokenization, close enough


def main():
    from tts.streaming_tts import StreamingTTS
    from tts.tts_manager import _get

    # Preload the model so the first token isn't stuck on model load.
    _get()._ensure_model()

    print("Streaming fake tokens into TTS...")
    streamer = StreamingTTS()
    streamer.start()

    t0 = time.perf_counter()
    first_audio_played = None

    for tok in FAKE_TOKENS:
        streamer.push_token(tok + " ")
        # Simulate LLM token cadence: ~60ms per token
        time.sleep(0.06)
        # Flag when the first chunk has likely started playing
        if first_audio_played is None and streamer.audio_queue.qsize() > 0:
            first_audio_played = time.perf_counter() - t0
            print(f"  → first synthesized chunk queued at t={first_audio_played:.2f}s")

    print(f"All tokens pushed at t={time.perf_counter() - t0:.2f}s. Finishing...")
    streamer.finish()
    print(f"All audio played at t={time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()