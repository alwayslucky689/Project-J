# tts/tts_manager.py - Lazy-loaded singleton using Pocket-TTS (CPU)
#
# Phase 5: TTS runs entirely on CPU. Zero VRAM cost, no stutter.
# Public API is unchanged from the OmniVoice version — nothing outside
# this file needs to know the engine changed.

import os
import time
import threading
import platform
import subprocess
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("⚠️ sounddevice not installed. Install with: pip install sounddevice")

from config.paths import JARVIS_VOICE_SAMPLE, SYSTEM_PROMPTS


# Reference clip used for voice cloning on first run.
VOICE_REF_AUDIO = str(JARVIS_VOICE_SAMPLE)

# Cached voice state. First run clones the reference clip (slow) and writes
# this file. Every run after that loads it in well under a second.
VOICE_STATE_CACHE = SYSTEM_PROMPTS / "jarvis_voice.safetensors"

# Built-in Pocket-TTS voice used as a fallback if cloning is unavailable.
FALLBACK_VOICE = "alba"


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
        self.voice_state = None
        self.cache = {}
        self.muted = False
        self.sample_rate = 24000
        self.volume = 70
        self._load_lock = threading.Lock()
        self._loading = False

    # ---------- Lazy model load ----------
    def _ensure_model(self):
        if self.model is not None and self.voice_state is not None:
            return True
        with self._load_lock:
            if self.model is not None and self.voice_state is not None:
                return True
            if self._loading:
                while self._loading:
                    time.sleep(0.05)
                return self.model is not None and self.voice_state is not None
            self._loading = True
            try:
                print("🎤 Loading Pocket-TTS on CPU (first speak only)...")
                from pocket_tts import TTSModel, export_model_state

                self.model = TTSModel.load_model()
                self.sample_rate = self.model.sample_rate
                self.voice_state = self._load_voice_state(export_model_state)

                print(f"✅ Pocket-TTS ready (sample rate {self.sample_rate}).")
                return True
            except Exception as e:
                print(f"❌ Failed to load Pocket-TTS: {e}")
                self.model = None
                self.voice_state = None
                return False
            finally:
                self._loading = False

    def _load_voice_state(self, export_fn):
        """
        Prefer the cached safetensors file. If absent, clone from the
        reference WAV (slow) and cache the result for next time.
        """
        if VOICE_STATE_CACHE.exists():
            print(f"🎙️ Loading cached voice state: {VOICE_STATE_CACHE.name}")
            return self.model.get_state_for_audio_prompt(str(VOICE_STATE_CACHE))

        if not os.path.exists(VOICE_REF_AUDIO):
            print(f"⚠️ Reference audio missing: {VOICE_REF_AUDIO}")
            print(f"   Falling back to built-in voice: {FALLBACK_VOICE}")
            return self.model.get_state_for_audio_prompt(FALLBACK_VOICE)

        print(f"🎙️ Cloning voice from {os.path.basename(VOICE_REF_AUDIO)} "
              f"(slow, one-time — will be cached)")
        state = self.model.get_state_for_audio_prompt(VOICE_REF_AUDIO)

        try:
            VOICE_STATE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            export_fn(state, str(VOICE_STATE_CACHE))
            print(f"✅ Cached voice state → {VOICE_STATE_CACHE.name}")
        except Exception as e:
            print(f"⚠️ Could not cache voice state (will re-clone next run): {e}")

        return state

    # ---------- Mute + volume ----------
    def set_mute(self, muted: bool):
        self.muted = muted
        state = "🔇 MUTED (text only)" if muted else "🔊 Voice output ENABLED"
        print(state)

    def toggle_mute(self):
        self.set_mute(not self.muted)

    def set_volume(self, volume: int):
        self.volume = max(0, min(100, int(volume)))

    def get_volume(self) -> int:
        return self.volume

    # ---------- Speak ----------
    def speak(self, text, voice=None, block=False, on_complete=None):
        if not text or len(text) < 5:
            if on_complete:
                on_complete()
            return

        if self.muted:
            if on_complete:
                on_complete()
            return

        if not self._ensure_model():
            if on_complete:
                on_complete()
            return

        # Cache lookup — same text, same audio.
        if text in self.cache:
            audio = self.cache[text]
        else:
            try:
                # Pocket-TTS: generate_audio(voice_state, text) -> torch.Tensor
                tensor = self.model.generate_audio(self.voice_state, text)
                audio = tensor.detach().cpu().numpy().astype(np.float32).flatten()
                self.cache[text] = audio
            except Exception as e:
                print(f"❌ TTS generation error: {e}")
                if on_complete:
                    on_complete()
                return

        # Normalize
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
        """Unchanged from the OmniVoice version. Phase 4 will replace with a
        persistent OutputStream to eliminate per-utterance device reconfig."""
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
                subprocess.run(["start", temp_file], shell=True,
                               check=False, capture_output=True)
            elif system == "Darwin":
                subprocess.run(["open", temp_file], check=False, capture_output=True)
            else:
                subprocess.run(["xdg-open", temp_file], check=False, capture_output=True)
        except Exception as e:
            print(f"❌ Playback error: {e}")


# ===== Module-level API (unchanged) =====
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