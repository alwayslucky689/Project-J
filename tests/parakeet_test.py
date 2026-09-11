# tests/parakeet_test.py - Test Parakeet in venv_stt
import nemo.collections.asr as nemo_asr
import time
import os
import sounddevice as sd
import soundfile as sf

print("Recording 4 seconds...")
audio = sd.rec(int(4 * 16000), samplerate=16000, channels=1, dtype='int16')
sd.wait()
sf.write("tests/sample.wav", audio, 16000)
print("✅ Saved tests/sample.wav")
print("🔊 Loading Parakeet TDT 0.6B v2...")
start = time.time()
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
print(f"✅ Loaded in {time.time() - start:.1f}s")
print(f"   Device: {model.device}")

# Check a WAV file exists
audio_path = "tests/sample.wav"
if not os.path.exists(audio_path):
    print(f"\n⚠️ Put a WAV file at: {audio_path}")
    print("   Record a few seconds of speech with Audacity or Windows Voice Recorder.")
    print("   Save as 16kHz mono WAV.")
    exit(0)

print(f"\n📝 Transcribing: {audio_path}")
start = time.time()
output = model.transcribe([audio_path])
elapsed = time.time() - start

print(f"✅ Transcribed in {elapsed:.2f}s")
print(f"\n📄 Result:")
print(f"   {output[0].text}")