# tts/tts_manager.py - Lazy-loaded singleton using Pocket-TTS (CPU)
#
# Phase 5: TTS runs entirely on CPU. Zero VRAM cost, no stutter.
# Personality-aware: reads voice_sample from the active personality YAML.
# Voice state is cached per reference clip (faye_sample3 → faye_sample3.safetensors).

import torch as _torch
_torch.set_num_threads(3)
_torch.set_num_interop_threads(1)

import os
import time
import threading
import platform
import subprocess
from pathlib import Path
from typing import Callable, Optional
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("⚠️ sounddevice not installed. Install with: pip install sounddevice")

from config.paths import SYSTEM_PROMPTS, PROJECT_ROOT


# Built-in Pocket-TTS voice used as fallback if no reference is available.
FALLBACK_VOICE = "alba"


# Voice provider — set once from assistant.py to personality_mgr.get_voice_sample
_voice_provider: Optional[Callable[[], Optional[str]]] = None


def set_voice_provider(fn: Callable[[], Optional[str]]):
    """Register a callback that returns the voice_sample path for the
    currently active personality. Called once at startup."""
    global _voice_provider
    _voice_provider = fn


def _resolve_ref_path(raw: Optional[str]) -> Optional[str]:
    """Turn a YAML voice_sample value into an absolute path that exists.
    YAML paths are project-relative (e.g. 'prompts/system/faye_sample3.wav')."""
    if not raw:
        return None
    # Try project-relative first, then absolute
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return str(candidate) if candidate.exists() else None


def _voice_cache_path(ref_audio: str) -> Path:
    """Per-clip cache: faye_sample3.wav → faye_sample3_voice.safetensors."""
    stem = Path(ref_audio).stem
    return SYSTEM_PROMPTS / f"{stem}_voice.safetensors"


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
        self._active_ref: Optional[str] = None
        self._voice_states: dict[str, object] = {}   # ref_path -> voice_state
        self.cache = {}
        self.muted = False
        self.sample_rate = 24000
        self.volume = 70
        self._load_lock = threading.Lock()
        self._loading = False

    # ---------- Model + voice loading ----------

    def _ensure_model(self):
        """Load the Pocket-TTS model. Voice is loaded separately."""
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
                print("🎤 Loading Pocket-TTS on CPU (first speak only)...")
                from pocket_tts import TTSModel
                self.model = TTSModel.load_model()
                self.sample_rate = self.model.sample_rate
                print(f"✅ Pocket-TTS ready (sample rate {self.sample_rate}).")
                from tts.playback import get_player
                get_player(sample_rate=self.sample_rate).start()
                return True
            except Exception as e:
                print(f"❌ Failed to load Pocket-TTS: {e}")
                self.model = None
                return False
            finally:
                self._loading = False

    def _current_ref(self) -> Optional[str]:
        """Path to the active personality's voice reference, or None."""
        if _voice_provider is None:
            return None
        raw = _voice_provider()
        return _resolve_ref_path(raw)

    def _ensure_voice(self) -> bool:
        """Ensure voice_state matches the active personality. Reloads when
        the personality changes."""
        if self.model is None:
            return False

        ref = self._current_ref()

        # No ref → fall back to built-in catalog
        if ref is None:
            if self._active_ref != "__fallback__":
                try:
                    self.voice_state = self.model.get_state_for_audio_prompt(
                        FALLBACK_VOICE
                    )
                    self._active_ref = "__fallback__"
                    print(f"⚠️  No voice_sample found; using built-in '{FALLBACK_VOICE}'.")
                except Exception as e:
                    print(f"❌ Fallback voice load failed: {e}")
                    return False
            return True

        # Already loaded for this ref?
        if ref == self._active_ref and ref in self._voice_states:
            self.voice_state = self._voice_states[ref]
            return True

        # Cached in memory?
        if ref in self._voice_states:
            self.voice_state = self._voice_states[ref]
            self._active_ref = ref
            return True

        # Load from disk cache or clone
        from pocket_tts import export_model_state
        cache_path = _voice_cache_path(ref)
        try:
            if cache_path.exists():
                print(f"🎙️ Loading cached voice state: {cache_path.name}")
                state = self.model.get_state_for_audio_prompt(str(cache_path))
            else:
                print(f"🎙️ Cloning voice from {Path(ref).name} (one-time)...")
                state = self.model.get_state_for_audio_prompt(ref)
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    export_model_state(state, str(cache_path))
                    print(f"✅ Cached voice state → {cache_path.name}")
                except Exception as e:
                    print(f"⚠️  Could not cache voice state: {e}")
        except Exception as e:
            print(f"❌ Voice load failed for {ref}: {e}")
            return False

        self._voice_states[ref] = state
        self.voice_state = state
        self._active_ref = ref
        return True

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
            if on_complete: on_complete()
            return
        if self.muted:
            if on_complete: on_complete()
            return
        if not self._ensure_model() or not self._ensure_voice():
            if on_complete: on_complete()
            return

        # Voice-aware cache key: same text different personality = different audio
        cache_key = (self._active_ref, text)

        if cache_key in self.cache:
            audio = self.cache[cache_key]
        else:
            try:
                tensor = self.model.generate_audio(self.voice_state, text)
                audio = tensor.detach().cpu().numpy().astype(np.float32).flatten()
                self.cache[cache_key] = audio
            except Exception as e:
                print(f"❌ TTS generation error: {e}")
                if on_complete: on_complete()
                return

        max_val = np.max(np.abs(audio)) if audio.size else 0.0
        if max_val > 0:
            audio = audio / max_val
        if self.volume < 100:
            audio = audio * (self.volume / 100.0)

        if block:
            try:
                self._play_audio(audio)
            finally:
                if on_complete: on_complete()
        else:
            def _run():
                try:
                    self._play_audio(audio)
                finally:
                    if on_complete: on_complete()
            threading.Thread(target=_run, daemon=True).start()

    def _play_audio(self, audio):
        try:
            from tts.playback import get_player
            player = get_player(sample_rate=self.sample_rate)
            player.play_and_wait(audio, timeout=120.0)
        except Exception as e:
            print(f"❌ Playback error: {e}")


# ===== Module-level API (unchanged) =====

def _get():
    return TTSManager()


def speak(text, voice=None, block=False, on_complete=None):
    _get().speak(text, voice=voice, block=block, on_complete=on_complete)


def speak_async(text, voice=None, on_complete=None):
    _get().speak(text, voice=voice, block=False, on_complete=on_complete)


def mute_tts(): _get().set_mute(True)
def unmute_tts(): _get().set_mute(False)
def toggle_mute(): _get().toggle_mute()
def is_muted(): return _get().muted
def set_tts_volume(volume): _get().set_volume(volume)
def get_tts_volume(): return _get().get_volume()


# ===== Streaming support (Phase 3a) =====

def generate_chunk_for_streaming(text: str):
    """Yield audio chunks from Pocket-TTS native streaming, using the
    active personality's voice."""
    if not text or not text.strip():
        return
    mgr = _get()
    if mgr.muted:
        return
    if not mgr._ensure_model() or not mgr._ensure_voice():
        return
    try:
        for chunk in mgr.model.generate_audio_stream(mgr.voice_state, text):
            audio = chunk.detach().cpu().numpy().astype(np.float32).flatten()
            if audio.size == 0:
                continue
            audio = np.clip(audio, -1.0, 1.0)
            if mgr.volume < 100:
                audio = audio * (mgr.volume / 100.0)
            yield audio
    except Exception as e:
        print(f"⚠️ [tts] chunk synthesis failed for {text!r}: {e}")


def play_chunk(audio):
    if audio is None or len(audio) == 0:
        return
    from tts.playback import get_player
    mgr = _get()
    player = get_player(sample_rate=mgr.sample_rate)
    player.play(audio)


def play_chunk_blocking(audio):
    if audio is None or len(audio) == 0:
        return
    from tts.playback import get_player
    mgr = _get()
    player = get_player(sample_rate=mgr.sample_rate)
    player.play_and_wait(audio, timeout=60.0)