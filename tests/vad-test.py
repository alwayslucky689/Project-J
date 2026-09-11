# vad_test.py - JaVAD + volume gating + valid chunk size

import numpy as np
import sounddevice as sd
import torch
import time as python_time
from collections import deque

from javad.stream import Pipeline

# ===== Configuration =====
SAMPLE_RATE = 16000
CHUNK_MS = 160                              # 160ms — divisible by JaVAD's hop (160 samples)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 2560 samples
assert CHUNK_SIZE % 160 == 0, f"Chunk size {CHUNK_SIZE} not divisible by 160"

# ===== JaVAD parameters =====
MODEL_NAME = 'precise'
PIPELINE_THRESHOLD = 0.55
PIPELINE_MODE = 'gradual'

# ===== Custom post-processing =====
MIN_SPEECH_CHUNKS = 1
MIN_SILENCE_CHUNKS = 3      # 3 × 160ms = 480ms
SMOOTHING_WINDOW = 3        # 3 × 160ms = 480ms

# ===== Volume gating =====
# Start with 0 to DISABLE gating — we'll calibrate based on your output.
# You saw ambient RMS ~0.03, so gate needs to be higher than that, but
# let's first see YOUR speech RMS before deciding.
RMS_GATE = 0.0              # 0.0 = disabled. Set higher once calibrated.

# ===== Load =====
print("\n🔊 Loading JaVAD Pipeline...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")
print(f"   Chunk size: {CHUNK_SIZE} samples ({CHUNK_MS}ms)")

try:
    pipeline = Pipeline(
        model_name=MODEL_NAME,
        mode=PIPELINE_MODE,
        threshold=PIPELINE_THRESHOLD,
        device=device,
    )
    if hasattr(pipeline, 'to'):
        pipeline.to(device)
    print(f"✅ Loaded. Config: {pipeline.config}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback; traceback.print_exc()
    exit(1)

# ===== Sanity check =====
silence = np.zeros(CHUNK_SIZE, dtype=np.float32)
try:
    result = pipeline.detect(silence)
    print(f"✅ Silence check: {result}")
    if hasattr(pipeline, 'reset'):
        pipeline.reset()
except Exception as e:
    print(f"❌ Sanity failed: {e}")
    exit(1)

# ===== State =====
recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
consecutive_speech = 0
consecutive_silence = 0
is_speech_state = False
last_print_time = 0
last_state_printed = None


def audio_callback(indata, frames, pa_time, status):
    global consecutive_speech, consecutive_silence, is_speech_state
    global last_print_time, last_state_printed

    try:
        audio = indata.flatten().astype(np.float32) / 32768.0

        # --- RMS for volume gating ---
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # --- Volume gate ---
        if RMS_GATE > 0 and rms < RMS_GATE:
            val = 0.0
            gated = True
        else:
            raw_speech = pipeline.detect(audio)
            val = 1.0 if raw_speech else 0.0
            gated = False

        recent_predictions.append(val)
        smoothed = float(np.mean(recent_predictions))
        chunk_says_speech = smoothed > 0.5

        # --- Onset/offset ---
        if chunk_says_speech:
            consecutive_speech += 1
            consecutive_silence = 0
        else:
            consecutive_silence += 1
            consecutive_speech = 0

        if not is_speech_state and consecutive_speech >= MIN_SPEECH_CHUNKS:
            is_speech_state = True
        elif is_speech_state and consecutive_silence >= MIN_SILENCE_CHUNKS:
            is_speech_state = False

        # --- Display ---
        now = python_time.time()
        if (is_speech_state != last_state_printed) or (now - last_print_time) > 0.3:
            gate_str = "GATED" if gated else "     "
            if is_speech_state:
                bar = "█" * int(smoothed * 20)
                print(f"\r🎤 SPEECH  [{bar:<20}] {smoothed:.2f} rms={rms:.3f} {gate_str}   ",
                      end="", flush=True)
            else:
                bar = "░" * int((1 - smoothed) * 20)
                print(f"\r🔇 silence [{bar:<20}] {smoothed:.2f} rms={rms:.3f} {gate_str}   ",
                      end="", flush=True)
            last_state_printed = is_speech_state
            last_print_time = now

    except Exception as e:
        if not hasattr(audio_callback, '_error_shown'):
            print(f"\n⚠️ Error: {e}")
            import traceback; traceback.print_exc()
            audio_callback._error_shown = True


print("\n" + "=" * 60)
print("Real-time microphone")
print("=" * 60)
print(f"🎚️  RMS gate = {RMS_GATE} ({'DISABLED' if RMS_GATE == 0 else 'active'})")
print()
print("📊 CALIBRATION — note the rms= values below:")
print("   1. Stay silent for 3s → note rms (this is your ambient noise)")
print("   2. Say 'yes' near mic → note rms (this is your speech)")
print("   3. Have roommate talk → note rms (this is cross-talk)")
print()
print("   Then we'll set RMS_GATE between roommate and you.\n")

try:
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16,
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    )
    stream.start()
    print("🎤 Listening...\n")
    while True:
        python_time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\n🛑 Stopping...")
finally:
    try:
        stream.stop()
        stream.close()
    except:
        pass
    print("✅ Done.")