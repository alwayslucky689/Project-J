# tts_manager.py
import os
import time
import threading
import subprocess
import platform
import torch
import soundfile as sf
from omnivoice import OmniVoice

# Set Hugging Face cache directory (you can move this to settings later)
os.environ["HF_HOME"] = "C:\\Users\\pstef\\.cache\\huggingface"

class TTSManager:
    """
    Singleton TTS manager that loads the model once and keeps it in memory.
    Uses GPU acceleration and caches generated audio for speed.
    """
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
        self._load_model()

    def _load_model(self):
        """Load the Omnivoice model on GPU (called once at startup)."""
        print("🎤 Loading TTS model on GPU (this may take a few seconds)...")
        try:
            self.model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map="cuda:0",
                dtype=torch.float16,
                local_files_only=True  # Assumes you've already downloaded the model
            )
            print("✅ TTS model loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load TTS model: {e}")
            print("   Ensure you have the model cached or remove 'local_files_only=True' to download.")
            self.model = None

    def speak(self, text, voice=None, block=False):
        """
        Generate speech for the given text and play it.
        If block=True, waits for playback to finish; otherwise runs in background.
        """
        if not text or not self.model:
            return

        # Optional: skip very short messages (like "✅ Opened YouTube")
        if len(text) < 10:
            return

        # Check cache
        if text in self.cache:
            audio = self.cache[text]
            # print("⚡ Using cached audio")
        else:
            start = time.time()
            try:
                audio = self.model.generate(
                    text=text,
                    num_step=16,    # Fast generation
                    speed=1.0,
                )
                gen_time = time.time() - start
                # print(f"⏱️ TTS generation: {gen_time:.2f}s")
                self.cache[text] = audio
            except Exception as e:
                print(f"❌ TTS generation error: {e}")
                return

        # Save to a temporary WAV file
        temp_file = "tts_output.wav"
        try:
            sf.write(temp_file, audio[0], 24000)
        except Exception as e:
            print(f"❌ Failed to save audio: {e}")
            return

        # Play the file using system default player
        if block:
            self._play_audio_blocking(temp_file)
        else:
            # Non-blocking: launch in a daemon thread
            threading.Thread(target=self._play_audio_blocking, args=(temp_file,), daemon=True).start()

    def _play_audio_blocking(self, filepath):
        """Play a WAV file using the system's default audio player."""
        try:
            system = platform.system()
            if system == "Windows":
                subprocess.run(["start", filepath], shell=True, check=False, capture_output=True)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", filepath], check=False, capture_output=True)
            else:  # Linux and others
                subprocess.run(["xdg-open", filepath], check=False, capture_output=True)
        except Exception as e:
            print(f"❌ Could not play audio: {e}")

    def speak_async(self, text, voice=None):
        """Convenience method for non-blocking speech."""
        self.speak(text, voice=voice, block=False)

# Optional: Preload the TTS model at import time (so it's ready when assistant starts)
# This will run when the module is imported.
_tts_instance = TTSManager()

# Public functions for easy use
def speak(text, voice=None, block=False):
    """Speak the given text using the global TTS manager."""
    _tts_instance.speak(text, voice=voice, block=block)

def speak_async(text, voice=None):
    """Speak asynchronously (non-blocking)."""
    _tts_instance.speak_async(text, voice=voice)