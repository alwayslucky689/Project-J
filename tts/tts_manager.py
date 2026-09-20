# tts/tts_manager.py - Lazy-loaded singleton with voice cloning

import os
import time
import threading
import subprocess
import platform
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("⚠️ sounddevice not installed. Install with: pip install sounddevice")

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


from config.paths import JARVIS_VOICE_SAMPLE

VOICE_REF_AUDIO = str(JARVIS_VOICE_SAMPLE)

# Exact transcription of the reference audio.
# Set to None to let OmniVoice auto-transcribe via Whisper (slower, less accurate).
VOICE_REF_TEXT = "The proposed element should serve as a viable alternative for palladium. Unfortunately it is impossible to synthesize."  


class TTSManager:
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.model = None
        self.cache = {}
        self.muted = False
        self.sample_rate = 24000
        self.volume = 70
        self._load_lock = threading.Lock()
        self._loading = False

        # Voice cloning reference
        self.ref_audio = VOICE_REF_AUDIO
        self.ref_text = VOICE_REF_TEXT

        # Validate reference audio exists
        if not os.path.exists(self.ref_audio):
            print(f"⚠️ Reference audio not found: {self.ref_audio}")
            print("   TTS will use a random voice until this is fixed.")
            self.ref_audio = None
            self.ref_text = None

    # ---------- Lazy model load ----------
    def _ensure_model(self):
        if self.model is not None:
            return True
        with self._load_lock:
            if self.model is not None:
                return True
            if self._loading:
                while self._loading:
                    time.sleep(0.05)
                return self.model is not None
            self._loading = True
            try:
                print("🎤 Loading TTS model on GPU (first speak only)...")
                import torch
                from omnivoice import OmniVoice
                self.model = OmniVoice.from_pretrained(
                    "k2-fsa/OmniVoice",
                    device_map="cuda:0",
                    dtype=torch.float16,
                    local_files_only=True,
                )
                print("✅ TTS model loaded.")
                if self.ref_audio:
                    print(f"🎙️ Voice cloning reference: {self.ref_audio}")
                else:
                    print("⚠️ No reference audio — using random voice.")
                return True
            except Exception as e:
                print(f"❌ Failed to load TTS model: {e}")
                self.model = None
                return False
            finally:
                self._loading = False

    # ---------- Mute + volume ----------
    def set_mute(self, muted: bool):
        self.muted = muted
        state = "🔇 MUTED (text only)" if muted else "🔊 Voice output ENABLED"
        print(f"TTS: {state}")

    def toggle_mute(self):
        self.set_mute(not self.muted)

    def set_volume(self, volume: int):
        self.volume = max(0, min(100, int(volume)))
        print(f"🔊 TTS volume: {self.volume}%")

    def get_volume(self) -> int:
        return self.volume

    # ---------- Speak ----------
    def speak(self, text, voice=None, block=False, on_complete=None):
        if not text or len(text) < 5:
            if on_complete:
                on_complete()
            return

        if self.muted:
            print(f"Assistant: {text[:60]}...")
            if on_complete:
                on_complete()
            return

        if not self._ensure_model():
            if on_complete:
                on_complete()
            return

        # Cache lookup
        if text in self.cache:
            audio = self.cache[text]
        else:
            try:
                # Build generation kwargs — this is where voice cloning happens
                gen_kwargs = {
                    "text": text,
                    "num_step": 32,
                    "speed": 1.0,
                }

                # Pass the reference audio for voice cloning
                if self.ref_audio and os.path.exists(self.ref_audio):
                    gen_kwargs["ref_audio"] = self.ref_audio
                    if self.ref_text:
                        gen_kwargs["ref_text"] = self.ref_text
                    # If ref_text is None, OmniVoice auto-transcribes via Whisper

                audio = self.model.generate(**gen_kwargs)
                self.cache[text] = audio
            except Exception as e:
                print(f"❌ TTS generation error: {e}")
                if on_complete:
                    on_complete()
                return

        # Post-process
        if isinstance(audio, list):
            audio = np.array(audio)
        if audio.ndim > 1:
            audio = audio.flatten()
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio)) if audio.size else 0.0
        if max_val > 0:
            audio = audio / max_val

        # Apply volume
        if self.volume < 100:
            audio = audio * (self.volume / 100.0)

        if block:
            try:
                self._play_audio(audio)
            finally:
                if on_complete:
                    on_complete()
        else:
            def _run():
                try:
                    self._play_audio(audio)
                finally:
                    if on_complete:
                        on_complete()
            threading.Thread(target=_run, daemon=True).start()

    def _play_audio(self, audio):
        try:
            if HAS_SOUNDDEVICE:
                try:
                    sd.play(audio, self.sample_rate)
                    sd.wait()
                    return
                except Exception as e:
                    print(f"❌ sounddevice playback error: {e}")

            import soundfile as sf
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


# ===== Module-level API =====
def _get():
    return TTSManager()


def speak(text, voice=None, block=False, on_complete=None):
    _get().speak(text, voice=voice, block=block, on_complete=on_complete)


def speak_async(text, voice=None, on_complete=None):
    _get().speak(text, voice=voice, block=False, on_complete=on_complete)


def mute_tts():
    _get().set_mute(True)


def unmute_tts():
    _get().set_mute(False)


def toggle_mute():
    _get().toggle_mute()


def is_muted():
    return _get().muted


def set_tts_volume(volume):
    _get().set_volume(volume)


def get_tts_volume():
    return _get().get_volume()