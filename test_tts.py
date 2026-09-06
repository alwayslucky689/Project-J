# test_tts_gpu.py
import os
os.environ["HF_HOME"] = "C:\\Users\\pstef\\.cache\\huggingface"

from omnivoice import OmniVoice
import soundfile as sf
import torch

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("🎤 Loading model on GPU...")

# Load model on GPU
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16, 
    local_files_only=True # Use float16 for GPU speed
)
print("✅ Model loaded!")

text = "Hello, I am Jarvis. How can I help you today?"
print(f"🔊 Generating: {text}")

# Fast generation with GPU
audio = model.generate(
    text=text,
    num_step=16,
    speed=1.0,
)

sf.write("output_gpu.wav", audio[0], 24000)
print("✅ Saved to output_gpu.wav")