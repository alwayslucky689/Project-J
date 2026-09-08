# tts_manager.py - Complete fixed version
import os
import time
import threading
import subprocess
import platform
import torch
import soundfile as sf
import numpy as np
from omnivoice import OmniVoice
from omnivoice import VoiceClonePrompt

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("⚠️ pygame-ce not installed. Install with: pip install pygame-ce")

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
        self.volume = 0.8  # Default volume
        self.voice_prompt = None
        self._load_model()
        self._load_voice_prompt()
        
        # Initialize pygame mixer with STEREO
        if HAS_PYGAME:
            try:
                pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=2)
                print("🔊 Pygame-ce mixer initialized (stereo).")
            except Exception as e:
                print(f"⚠️ Could not initialize pygame mixer: {e}")

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

    def _load_voice_prompt(self):
        prompt_path = "prompts/system/jarvis_voice.pt"
        
        if os.path.exists(prompt_path):
            try:
                self.voice_prompt = VoiceClonePrompt.load(prompt_path)
                return
            except:
                self.voice_prompt = None
                return
        
        ref_audio_path = "prompts/system/jarvis_sample.wav"
        if os.path.exists(ref_audio_path) and self.model is not None:
            try:
                self.voice_prompt = self.model.create_voice_clone_prompt(
                    ref_audio=ref_audio_path,
                    ref_text="Your actual transcription here."
                )
                self.voice_prompt.save(prompt_path)
            except:
                self.voice_prompt = None

    def set_volume(self, volume):
        """Set volume level (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        print(f"🔊 TTS volume set to {int(self.volume * 100)}%")

    def get_volume(self):
        return self.volume

    def set_mute(self, muted: bool):
        self.muted = muted
        state = "🔇 MUTED (text only)" if muted else "🔊 Voice output ENABLED"
        print(f"TTS: {state}")

    def toggle_mute(self):
        self.set_mute(not self.muted)

    def speak(self, text, voice=None, block=False):
        """Generate and play speech"""
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
                if self.voice_prompt:
                    audio = self.model.generate(
                        text=text,
                        num_step=32,
                        speed=1.0,
                        voice_clone_prompt=self.voice_prompt,
                    )
                else:
                    audio = self.model.generate(
                        text=text,
                        num_step=32,
                        speed=1.0,
                        instruct="male, british accent, medium pitch",
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
        
        # Normalize
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        if block:
            self._play_audio(audio)
        else:
            threading.Thread(target=self._play_audio, args=(audio,), daemon=True).start()

    def _play_audio(self, audio):
        """Play audio using pygame-ce"""
        try:
            # Apply volume
            scaled_audio = audio * self.volume
            
            if HAS_PYGAME:
                try:
                    # Convert to int16
                    audio_int16 = (scaled_audio * 32767).astype(np.int16)
                    # Convert mono to stereo if needed
                    if audio_int16.ndim == 1:
                        audio_int16 = np.column_stack((audio_int16, audio_int16))
                    sound = pygame.sndarray.make_sound(audio_int16)
                    sound.play()
                    while pygame.mixer.get_busy():
                        pygame.time.wait(10)
                    return
                except Exception as e:
                    print(f"❌ Pygame playback error: {e}")
                    # Fall through to file fallback
            
            # Fallback: Save to WAV and play
            temp_file = "tts_output.wav"
            sf.write(temp_file, scaled_audio, self.sample_rate)
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
        """Non-blocking speech"""
        self.speak(text, voice=voice, block=False)

# Global instance
_tts_instance = TTSManager()

# Public functions
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

def set_tts_volume(volume):
    _tts_instance.set_volume(volume / 100)

def get_tts_volume():
    return int(_tts_instance.get_volume() * 100)