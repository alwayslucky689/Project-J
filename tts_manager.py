# tts_manager.py
import os
import time
import threading
import subprocess
import platform
import torch
import soundfile as sf
from omnivoice import OmniVoice

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
        self.muted = False  # <- NEW: mute state
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
        except Exception as e:
            print(f"❌ Failed to load TTS model: {e}")
            self.model = None

    def set_mute(self, muted: bool):
        """Set mute state: True = no speech output, False = speech enabled"""
        self.muted = muted
        state = "🔇 " if muted else "🔊 Voice output ENABLED"
        print(f"TTS: {state}")

    def toggle_mute(self):
        """Toggle mute state on/off"""
        self.set_mute(not self.muted)

    def speak(self, text, voice=None, block=False):
        """Generate speech ONLY if not muted"""
        if not text or not self.model:
            return

        # Skip very short messages
        if len(text) < 5:
            return

        # If muted, skip TTS entirely
        if self.muted:
            print(f"Assistant: {text[:60]}...")
            return

        # Rest of your existing speak logic...
        if text in self.cache:
            audio = self.cache[text]
        else:
            start = time.time()
            try:
                audio = self.model.generate(text=text, num_step=16, speed=1.0)
                self.cache[text] = audio
            except Exception as e:
                print(f"❌ TTS generation error: {e}")
                return

        temp_file = "tts_output.wav"
        try:
            sf.write(temp_file, audio[0], 24000)
        except Exception as e:
            print(f"❌ Failed to save audio: {e}")
            return

        if block:
            self._play_audio_blocking(temp_file)
        else:
            threading.Thread(target=self._play_audio_blocking, args=(temp_file,), daemon=True).start()

    def _play_audio_blocking(self, filepath):
        try:
            system = platform.system()
            if system == "Windows":
                subprocess.run(["start", filepath], shell=True, check=False, capture_output=True)
            elif system == "Darwin":
                subprocess.run(["open", filepath], check=False, capture_output=True)
            else:
                subprocess.run(["xdg-open", filepath], check=False, capture_output=True)
        except Exception as e:
            print(f"❌ Could not play audio: {e}")

    def speak_async(self, text, voice=None):
        self.speak(text, voice=voice, block=False)

# Global instance
_tts_instance = TTSManager()

def speak(text, voice=None, block=False):
    _tts_instance.speak(text, voice=voice, block=block)

def speak_async(text, voice=None):
    _tts_instance.speak_async(text, voice=voice)

# NEW: Public functions to control mute
def mute_tts():
    """Mute all TTS output"""
    _tts_instance.set_mute(True)

def unmute_tts():
    """Unmute TTS output"""
    _tts_instance.set_mute(False)

def toggle_mute():
    """Toggle mute state"""
    _tts_instance.toggle_mute()

def is_muted():
    """Check if TTS is currently muted"""
    return _tts_instance.muted