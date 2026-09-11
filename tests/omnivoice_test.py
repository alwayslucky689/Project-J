# tests/omnivoice_gibberish_test.py - Track how often OmniVoice garbles

import os
import time
import numpy as np
import soundfile as sf
import torch

os.environ["HF_HOME"] = "C:\\Users\\pstef\\.cache\\huggingface"
from omnivoice import OmniVoice
from omnivoice import VoiceClonePrompt

print("Loading OmniVoice...")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16,
    local_files_only=True,
)

prompt = None
if os.path.exists("prompts/system/jarvis_voice.pt"):
    prompt = VoiceClonePrompt.load("prompts/system/jarvis_voice.pt")

os.makedirs("tests/gibberish_out", exist_ok=True)

# The sentences from your earlier run — these are the kind that garbled
test_sentences = [
    "Good evening, sir.",
    "The weather in London is mild today, with a chance of rain later this evening.",
    "I've completed the diagnostic on all primary systems.",
    "The arc reactor is operating at peak efficiency, repulsor calibration is within tolerance, and the suit's navigation array reports all satellites in range.",
    "Unfortunately, California is a big place, so the weather can vary depending on where you are.",
]

print("\n" + "=" * 60)
print("Generating each sentence 3 times at num_step=16")
print("=" * 60)

for i, text in enumerate(test_sentences):
    print(f"\n[{i+1}] {text[:70]}...")
    for run in range(3):
        # Force a different seed each run
        torch.manual_seed(run * 100 + i)

        start = time.time()
        try:
            audio = model.generate(
                text=text,
                num_step=16,
                speed=1.0,
                voice_clone_prompt=prompt,
            )
        except Exception as e:
            print(f"  Run {run+1}: FAILED - {e}")
            continue
        elapsed = time.time() - start
        a = np.array(audio).squeeze()

        out = f"tests/gibberish_out/s{i+1}_r{run+1}.wav"
        sf.write(out, a, 24000)
        dur = len(a) / 24000

        # Compute simple audio stats to spot garbling
        rms = float(np.sqrt(np.mean(a ** 2)))
        peak = float(np.max(np.abs(a)))

        print(f"  Run {run+1}: {dur:.2f}s, {elapsed:.2f}s gen, rms={rms:.3f}, peak={peak:.3f} → {out}")

print("\n✅ Done.")
print("\nNow:")
print("  1. Listen to each wav in tests/gibberish_out/")
print("  2. Note which ones garble (say 'a word that isn't in the text')")
print("  3. Report the pattern:")
print("     - Does the SAME sentence garble sometimes but not always?")
print("     - Does the SAME SENTENCE garble on every run?")
print("     - Is it random?")
print("  4. That tells us if it's a stochastic sampling problem (fix: seed)")
print("     or a text-specific problem (fix: chunk differently)")