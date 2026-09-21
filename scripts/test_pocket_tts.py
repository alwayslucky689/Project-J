"""
Standalone Pocket-TTS test harness for Project-J.

Uses the real pocket_tts Python API documented at:
    https://github.com/kyutai-labs/pocket-tts#using-it-as-a-python-library

Goals:
  1. Confirm the engine loads on CPU with no GPU fallback.
  2. Clone the JARVIS reference voice from prompts/system/jarvis_sample.wav.
  3. Measure per-phrase synthesis time and real-time factor.
  4. Save each output WAV to data/tts_tests/ for A/B listening vs OmniVoice.
  5. Optionally export the JARVIS voice state to safetensors for fast
     reload on subsequent runs (one-time cost, then instant loads).

Run from project root, inside the pocket-tts venv:
    python scripts/test_pocket_tts.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False
    print("⚠️  soundfile not installed — will fall back to scipy.io.wavfile")

try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False
    print("⚠️  sounddevice not installed — will only write WAVs")

from config.paths import JARVIS_VOICE_SAMPLE

OUT_DIR = PROJECT_ROOT / "data" / "tts_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_AUDIO = str(JARVIS_VOICE_SAMPLE)
# Cached voice embedding — exported once, then loaded instantly on later runs.
REF_SAFETENSORS = OUT_DIR / "jarvis_voice.safetensors"

# Short, medium, long — covers the range of responses the assistant emits.
TEST_PHRASES = [
    "Right away, Sir.",
    "I'm afraid that didn't work, Sir.",
    "Running diagnostics now. This will take a moment, Sir.",
    "Now playing Bohemian Rhapsody by Queen, Sir.",
    "The proposed element should serve as a viable alternative for palladium.",
]


# =====================================================================
# ENGINE — real pocket_tts API per README
# =====================================================================

_MODEL = None
_VOICE_STATE = None
_SAMPLE_RATE = None


def _load_engine():
    """Load TTSModel on CPU and prepare the JARVIS voice state."""
    global _MODEL, _VOICE_STATE, _SAMPLE_RATE

    from pocket_tts import TTSModel, export_model_state

    print("  → TTSModel.load_model() ...")
    # No device arg in the public API — defaults to CPU.
    # If you want int8 dynamic quantization (CPU-only, faster):
    #   TTSModel.load_model(quantize=True)
    # Requires: pip install "pocket-tts[quantize]"
    _MODEL = TTSModel.load_model()
    _SAMPLE_RATE = _MODEL.sample_rate
    print(f"  → sample_rate = {_SAMPLE_RATE}")

    # Voice state: prefer cached safetensors, fall back to raw wav clone.
    if REF_SAFETENSORS.exists():
        print(f"  → Loading cached voice state: {REF_SAFETENSORS.name}")
        _VOICE_STATE = _MODEL.get_state_for_audio_prompt(str(REF_SAFETENSORS))
    else:
        if not Path(REF_AUDIO).exists():
            raise FileNotFoundError(f"Reference audio missing: {REF_AUDIO}")
        print(f"  → Cloning voice from {Path(REF_AUDIO).name} (slow, one-time)")
        _VOICE_STATE = _MODEL.get_state_for_audio_prompt(REF_AUDIO)

        # Export so the next run loads instantly (README: "quite fast …
        # just reading the kvcache from disk").
        try:
            print(f"  → Exporting voice state → {REF_SAFETENSORS.name}")
            export_model_state(_VOICE_STATE, str(REF_SAFETENSORS))
            print("  → Export complete. Future runs will skip cloning.")
        except Exception as e:
            print(f"  ⚠️  Export failed (non-fatal): {e}")


def _synth(text: str) -> np.ndarray:
    """Synthesize text with the JARVIS voice. Returns float32 mono [-1,1]."""
    audio = _MODEL.generate_audio(_VOICE_STATE, text)
    # generate_audio returns a 1D torch tensor on the model's device.
    return audio.detach().cpu().numpy().astype(np.float32).flatten()


# =====================================================================
# HARNESS
# =====================================================================

def _vram_mb() -> int | None:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        ).strip().splitlines()[0]
        return int(out)
    except Exception:
        return None


def _write_wav(path: Path, audio: np.ndarray, rate: int):
    if HAS_SF:
        sf.write(str(path), audio, rate)
    else:
        import scipy.io.wavfile as wav
        wav.write(str(path), rate, (np.clip(audio, -1, 1) * 32767).astype(np.int16))


def main():
    print("=" * 64)
    print("Pocket-TTS Test Harness — Project-J")
    print("=" * 64)

    vram_before = _vram_mb()
    print(f"Reference clip : {REF_AUDIO}")
    print(f"VRAM before    : {vram_before} MB" if vram_before else "VRAM: n/a")

    print("\nLoading engine...")
    t0 = time.perf_counter()
    _load_engine()
    load_s = time.perf_counter() - t0
    print(f"✅ Engine + voice ready in {load_s:.2f}s")

    vram_after = _vram_mb()
    if vram_before and vram_after:
        print(f"VRAM after     : {vram_after} MB  (Δ {vram_after - vram_before:+d})")
    elif vram_after:
        print(f"VRAM after     : {vram_after} MB")

    print("\n" + "-" * 64)
    print(f"{'phrase':<48} {'total':>9} {'RTF':>7}")
    print("-" * 64)

    results = []

    for i, phrase in enumerate(TEST_PHRASES, 1):
        t_start = time.perf_counter()
        audio = _synth(phrase)
        t_end = time.perf_counter()

        total_ms = (t_end - t_start) * 1000
        audio_sec = len(audio) / _SAMPLE_RATE
        rtf = audio_sec / (t_end - t_start)

        out_path = OUT_DIR / f"pocket_{i:02d}.wav"
        _write_wav(out_path, audio, _SAMPLE_RATE)

        marker = "  (warmup)" if i == 1 else ""
        print(f"{phrase[:46]:<48} {total_ms:7.0f}ms {rtf:6.2f}x{marker}")
        print(f"    → audio {audio_sec:.2f}s → {out_path.name}")

        results.append({
            "phrase": phrase,
            "total_ms": total_ms,
            "audio_sec": audio_sec,
            "rtf": rtf,
        })

        if HAS_SD:
            try:
                sd.play(audio, _SAMPLE_RATE)
                sd.wait()
            except Exception as e:
                print(f"    ⚠️  playback error: {e}")

    # Summary — exclude warmup
    real = results[1:] if len(results) > 1 else results
    avg_ms = sum(r["total_ms"] for r in real) / len(real)
    avg_rtf = sum(r["rtf"] for r in real) / len(real)

    print("-" * 64)
    print(f"Average synthesis : {avg_ms:.0f} ms per phrase")
    print(f"Average RTF       : {avg_rtf:.2f}x  "
          f"({'FASTER' if avg_rtf > 1 else 'SLOWER'} than real time)")
    print(f"Outputs           : {OUT_DIR}")
    print()
    print("Next steps:")
    print("  1. Listen to the pocket_*.wav files.")
    print("  2. Compare against your OmniVoice output (same JARVIS clip).")
    print("  3. Note whether the voice sounds like the reference,")
    print("     and whether latency feels acceptable for a live assistant.")
    print()
    print("Integration notes:")
    print(f"  • Voice state cached at: {REF_SAFETENSORS.name}")
    print("    (delete this file to force re-cloning from the raw wav)")
    print("  • If RTF feels tight, try TTSModel.load_model(quantize=True)")
    print("    after: pip install \"pocket-tts[quantize]\"")


if __name__ == "__main__":
    main()