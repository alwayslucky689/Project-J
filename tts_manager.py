# tts_manager.py - Fixed version

import os
import time
import threading
import subprocess
import platform
import torch
import soundfile as sf
import numpy as np
from omnivoice import OmniVoice

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("⚠️ sounddevice not installed. Install with: pip install sounddevice")

os.environ["HF_HOME"] = "C:\\Users\\pstef\\.cache\\huggingface"

class TTSManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.model = None
        self.cache = {}
        self.muted = False
        self.sample_rate = 24000
        self._load_model()

    def _load_model(self):
        print("🎤 Loading TTS model on GPU (this may take a few seconds)...")
        try:
            self.model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map="cuda:0",
                dtype=torch.float16,
                local_files_only=True
            )
            print("✅ TTS model loaded successfully.")
            if HAS_SOUNDDEVICE:
                print("🔊 Using sounddevice for playback.")
            else:
                print("⚠️ sounddevice not available. Using file playback fallback.")
        except Exception as e:
            print(f"❌ Failed to load TTS model: {e}")
            self.model = None

    def set_mute(self, muted: bool):
        self.muted = muted
        state = "🔇 MUTED (text only)" if muted else "🔊 Voice output ENABLED"
        print(f"TTS: {state}")

    def toggle_mute(self):
        self.set_mute(not self.muted)

    def speak(self, text, voice=None, block=False):
        if not text or not self.model:
            return
        if len(text) < 5:
            return
        if self.muted:
            print(f"Assistant: {text[:60]}...")
            return

        # Check cache
        if text in self.cache:
            audio = self.cache[text]
        else:
            try:
                # Generate audio with explicit parameters
                audio = self.model.generate(
                    text=text,
                    num_step=32,  # More steps = better quality (default is 32)
                    speed=1.0,
                )
                self.cache[text] = audio
            except Exception as e:
                print(f"❌ TTS generation error: {e}")
                return

        # Ensure audio is a numpy array
        if isinstance(audio, list):
            audio = np.array(audio)
        if audio.ndim > 1:
            audio = audio.flatten()
        
        # Normalize to float32 range (-1 to 1)
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        # Trim silence from ends (optional)
        # audio = self._trim_silence(audio)

        if block:
            self._play_audio(audio)
        else:
            threading.Thread(target=self._play_audio, args=(audio,), daemon=True).start()

    def _trim_silence(self, audio, threshold=0.02):
        """Trim leading/trailing silence"""
        mask = np.abs(audio) > threshold
        if not np.any(mask):
            return audio
        start = np.argmax(mask)
        end = len(mask) - np.argmax(mask[::-1])
        return audio[start:end]

    def _play_audio(self, audio):
        try:
            # Always use sounddevice if available
            if HAS_SOUNDDEVICE:
                try:
                    # Play with explicit parameters
                    sd.play(audio, self.sample_rate)
                    sd.wait()
                    return
                except Exception as e:
                    print(f"❌ sounddevice playback error: {e}")
                    # Fall through to file fallback
            
            # Fallback: Save to WAV and play
            temp_file = "tts_output.wav"
            sf.write(temp_file, audio, self.sample_rate)
            time.sleep(0.1)
            
            system = platform.system()
            if system == "Windows":
                subprocess.run(["start", temp_file], shell=True, check=False, capture_output=True)
            elif system == "Darwin":
                subprocess.run(["open", temp_file], check=False, capture_output=True)
            else:
                subprocess.run(["xdg-open", temp_file], check=False, capture_output=True)
        except Exception as e:
            print(f"❌ Playback error: {e}")

    def speak_async(self, text, voice=None):
        self.speak(text, voice=voice, block=False)

# Global instance
_tts_instance = TTSManager()

def speak(text, voice=None, block=False):
    _tts_instance.speak(text, voice=voice, block=block)

def speak_async(text, voice=None):
    _tts_instance.speak_async(text, voice=voice)

def mute_tts():
    _tts_instance.set_mute(True)

def unmute_tts():
    _tts_instance.set_mute(False)

def toggle_mute():
    _tts_instance.toggle_mute()

def is_muted():
    return _tts_instance.muted