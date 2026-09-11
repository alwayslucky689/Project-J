# tests/voxcpm_diag.py - Convert both sub-models to fp16, then measure

import os
import time
import numpy as np
import torch

print("=" * 60)
print("Diagnostics")
print("=" * 60)
print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"GPU:   {torch.cuda.get_device_name(0)}")
print(f"Cap:   {torch.cuda.get_device_capability(0)}")

# ===== Load with optimize=False (avoid torch.compile warning) =====
print("\n🔊 Loading VoxCPM2...")
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    load_denoiser=False,
    optimize=False,   # skip torch.compile since triton isn't available
)

# ===== Inspect submodels =====
print("\n📋 Before conversion:")
print(f"  model: {type(model).__name__}")
print(f"  model.tts_model: {type(model.tts_model).__name__}")

# Find every nn.Module attached to model
submodules = []
for name in dir(model):
    if name.startswith("_"):
        continue
    try:
        val = getattr(model, name)
    except Exception:
        continue
    if isinstance(val, torch.nn.Module):
        submodules.append((f"model.{name}", val))

for name in dir(model.tts_model):
    if name.startswith("_"):
        continue
    try:
        val = getattr(model.tts_model, name)
    except Exception:
        continue
    if isinstance(val, torch.nn.Module):
        submodules.append((f"model.tts_model.{name}", val))

print(f"\n  Found {len(submodules)} submodules:")
for name, mod in submodules:
    try:
        p = next(mod.parameters())
        print(f"    {name}: dtype={p.dtype}, device={p.device}")
    except StopIteration:
        print(f"    {name}: (no parameters)")
    except Exception as e:
        print(f"    {name}: (error: {e})")

# ===== Convert ALL submodules to fp16 =====
print("\n🔄 Converting all submodules to fp16...")
for name, mod in submodules:
    try:
        mod.half()
        print(f"  ✅ {name}.half()")
    except Exception as e:
        print(f"  ⚠️  {name}.half() failed: {e}")

# ===== Test =====
text = "The weather in London is mild today, with a chance of rain later this evening."
ref = "prompts/system/jarvis_sample.wav"
assert os.path.exists(ref), f"Missing: {ref}"

print("\n" + "=" * 60)
print("Warm-up")
print("=" * 60)
try:
    _ = model.generate(
        text="Hello.",
        reference_wav_path=ref,
        cfg_value=2.0,
        inference_timesteps=10,
    )
    print("✅ Warm-up OK")
except Exception as e:
    print(f"❌ Warm-up failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("Timing tests")
print("=" * 60)

for steps in [10, 5, 3]:
    print(f"\n--- inference_timesteps={steps} ---")
    times = []
    for i in range(2):
        torch.cuda.synchronize()
        start = time.time()
        try:
            wav = model.generate(
                text=text,
                reference_wav_path=ref,
                cfg_value=2.0,
                inference_timesteps=steps,
            )
        except Exception as e:
            print(f"  Run {i+1} FAILED: {e}")
            break
        torch.cuda.synchronize()
        elapsed = time.time() - start
        wav = np.asarray(wav).squeeze()
        duration = len(wav) / 48000
        rtf = elapsed / duration
        print(f"  Run {i+1}: {duration:.2f}s audio in {elapsed:.2f}s → RTF={rtf:.2f}")
        times.append(rtf)
    if times:
        print(f"  Average RTF: {np.mean(times):.2f}")