# config/paths.py
"""
Centralized path resolution for Project-J.

All project paths are resolved relative to PROJECT_ROOT so the app works
regardless of current working directory and can be moved without edits.

System executables (Ollama, Discord) are resolved from PATH first, then
standard install locations. No username-specific paths anywhere.
"""

import os
import shutil
import sys
from pathlib import Path

# ===== Project root =====
# This file lives at <root>/config/paths.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ===== Project-relative directories =====
DATA_DIR       = PROJECT_ROOT / "data"
TEMP_DIR       = DATA_DIR / "tmp"
SESSIONS_DIR   = DATA_DIR / "sessions"
VOICES_DIR     = DATA_DIR / "voices"
MODELS_DIR     = PROJECT_ROOT / "models"
WAKEWORD_DIR   = MODELS_DIR / "wakeword"
PROMPTS_DIR    = PROJECT_ROOT / "prompts"
SYSTEM_PROMPTS = PROMPTS_DIR / "system"
LOGS_DIR       = PROJECT_ROOT / "logs"
PERSONALITIES_DIR = PROJECT_ROOT / "personalities"
# Create writable dirs on import
for _d in (DATA_DIR, TEMP_DIR, SESSIONS_DIR, VOICES_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ===== Specific files =====
WAKEWORD_MODEL_PATH = WAKEWORD_DIR / "jarvis_robust_final" / "jarvis_robust.onnx"
JARVIS_VOICE_SAMPLE = SYSTEM_PROMPTS / "jarvis_sample.wav"
SPOTIFY_CACHE       = PROJECT_ROOT / ".spotify_cache"
FACTS_FILE          = DATA_DIR / "facts.json"   # becomes facts.db in Phase 6
STT_UTTERANCE_WAV   = TEMP_DIR / "utterance.wav"

# ===== STT service =====
STT_VENV_PYTHON = PROJECT_ROOT / "venv_stt" / (
    "Scripts" if sys.platform == "win32" else "bin"
) / ("python.exe" if sys.platform == "win32" else "python")
STT_SERVICE_SCRIPT = PROJECT_ROOT / "stt" / "stt_service.py"


# ===== System executables =====

def _find_ollama() -> str | None:
    """PATH → standard install dirs → None."""
    which = shutil.which("ollama")
    if which:
        return which

    candidates: list[Path] = []
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA")
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        if local_appdata:
            candidates.append(Path(local_appdata) / "Programs" / "Ollama" / "ollama.exe")
        candidates.append(Path(program_files) / "Ollama" / "ollama.exe")
    elif sys.platform == "darwin":
        candidates += [Path("/usr/local/bin/ollama"), Path("/opt/homebrew/bin/ollama")]
    else:
        candidates += [Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama")]

    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _find_discord() -> str | None:
    """Windows only. Returns path to Update.exe or None."""
    if sys.platform != "win32":
        return None
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        return None
    p = Path(local_appdata) / "Discord" / "Update.exe"
    return str(p) if p.exists() else None


OLLAMA_EXE  = _find_ollama()
DISCORD_EXE = _find_discord()