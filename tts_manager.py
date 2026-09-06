# tts_manager.py
import os
import time
import threading
import subprocess
import platform
import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice

os.environ["HF_HOME"] = "C:\\Users\\pstef\\.cache\\huggingface"

# Try to import sounddevice; if not available, fall back to file playback
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("⚠️ sounddevice not installed. Install with: pip install sounddevice")

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
        # Voice profile description
        self.voice_profile = {
            "name": "Jarvis",
            "description": "A calm, British-accented male voice with crisp enunciation and subtle warmth.",
            "speed": 1.0,
            "sample_rate": 24000,
            "traits": ["Professional", "Slightly humorous", "Reassuring"]
        }
        self._load_model()

    def _load_model(self):
        print("🎤 Loading TTS model on GPU...")
        try:
            self.model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map="cuda:0",
                dtype=torch.float16,
                local_files_only=True
            )
            print(f"✅ TTS model loaded. Voice: {self.voice_profile['name']}")
            print(f"   {self.voice_profile['description']}")
        except Exception as e:
            print(f"❌ Failed to load TTS model: {e}")
            self.model = None

    def speak(self, text, voice=None, block=False, speed=None):
        if not text or not self.model or len(text.strip()) < 10:
            return

        # Check cache
        if text in self.cache:
            audio = self.cache[text]
        else:
            speed = speed or self.voice_profile["speed"]
            try:
                audio = self.model.generate(
                    text=text,
                    num_step=16,
                    speed=speed,
                )
                self.cache[text] = audio
            except Exception as e:
                print(f"❌ TTS generation error: {e}")
                return

        # Ensure audio is a numpy array
        audio = np.asarray(audio, dtype=np.float32)
        # If it's 2D with shape (samples, 1), squeeze to 1D
        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio.squeeze(1)
        elif audio.ndim > 1:
            audio = audio.flatten()  # Fallback

        if block:
            self._play_audio_blocking(audio)
        else:
            threading.Thread(target=self._play_audio_blocking, args=(audio,), daemon=True).start()

    def _play_audio_blocking(self, audio):
        """Play audio using sounddevice if available, otherwise fallback to file."""
        if HAS_SOUNDDEVICE:
            try:
                # Play with correct parameters - no extra 'channels' argument
                sd.play(audio, samplerate=self.voice_profile["sample_rate"])
                sd.wait()  # Wait for playback to finish
                return
            except Exception as e:
                print(f"❌ sounddevice playback error: {e}")
                # Fall through to file fallback

        # Fallback: save to WAV and play with system player
        try:
            temp_file = "tts_output.wav"
            sf.write(temp_file, audio, self.voice_profile["sample_rate"])
            time.sleep(0.1)  # Ensure file is written
            system = platform.system()
            if system == "Windows":
                subprocess.run(["start", temp_file], shell=True, check=False)
            elif system == "Darwin":
                subprocess.run(["open", temp_file], check=False)
            else:
                subprocess.run(["xdg-open", temp_file], check=False)
        except Exception as e:
            print(f"❌ Fallback playback error: {e}")

    def speak_async(self, text, voice=None, speed=None):
        self.speak(text, voice=voice, block=False, speed=speed)

    def get_voice_profile(self):
        return self.voice_profile

# Global instance
_tts_instance = TTSManager()

def speak(text, voice=None, block=False, speed=None):
    _tts_instance.speak(text, voice=voice, block=block, speed=speed)

def speak_async(text, voice=None, speed=None):
    _tts_instance.speak_async(text, voice=voice, speed=speed)

def get_voice_info():
    return _tts_instance.get_voice_profile()