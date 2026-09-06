# jarvis_tts.py - WITHOUT FLASHINFER (Still fast!)
import os
os.environ["HF_HOME"] = "C:\\Users\\pstef\\.cache\\huggingface"

from omnivoice import OmniVoice
import soundfile as sf
import torch
import time

class JarvisTTS:
    def __init__(self):
        print("🎤 Loading model on GPU...")
        self.model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="cuda:0",
            dtype=torch.float16,
            local_files_only=True
        )
        self.cache = {}
        print(f"✅ Model loaded! Generation ready.")
    
    def say(self, text):
        if text in self.cache:
            sf.write("output.wav", self.cache[text][0], 24000)
            print("⚡ Cached (instant)")
            return "output.wav"
        
        start = time.time()
        audio = self.model.generate(
            text=text,
            num_step=16,  # 2x faster than 32
            speed=1.0,
        )
        gen_time = time.time() - start
        print(f"⏱️ Generation: {gen_time:.2f}s")
        
        self.cache[text] = audio
        sf.write("output.wav", audio[0], 24000)
        return "output.wav"

# Initialize once
tts = JarvisTTS()

# Test
tts.say("Hello, I am Jarvis.")