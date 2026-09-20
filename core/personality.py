"""
Personality manager — loads personalities from YAML and provides
prompt generation + template rendering.

Replaces config/personality_manager.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from config.paths import PERSONALITIES_DIR


class PersonalityManager:
    def __init__(self, default_name: str = "computah"):
        self.personalities: dict[str, dict] = {}
        self.current: str = default_name
        self._load_all()

        if default_name not in self.personalities:
            available = list(self.personalities.keys())
            if not available:
                raise RuntimeError(f"No personality YAML files found in {PERSONALITIES_DIR}")
            print(f"⚠️ Default personality '{default_name}' not found. Using '{available[0]}'.")
            self.current = available[0]

    # ===== Loading =====

    def _load_all(self):
        if not PERSONALITIES_DIR.exists():
            print(f"⚠️ Personalities directory missing: {PERSONALITIES_DIR}")
            return
        for path in sorted(PERSONALITIES_DIR.glob("*.yaml")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                name = data.get("name") or path.stem
                data["name"] = name
                self.personalities[name] = data
            except Exception as e:
                print(f"⚠️ Failed to load personality {path.name}: {e}")

    # ===== Queries =====

    def _current_data(self) -> dict:
        return self.personalities.get(self.current, {})

    def get_personality_data(self, name: str | None = None) -> dict:
        if name is None:
            name = self.current
        return self.personalities.get(name, {})

    def get_current_name(self) -> str:
        return self.current

    def get_display_name(self, name: str | None = None) -> str:
        data = self.get_personality_data(name)
        return data.get("display_name", self.current if name is None else name or "?")

    def list_personalities(self) -> list[str]:
        return list(self.personalities.keys())

    def get_wake_words(self, name: str | None = None) -> list[str]:
        return self.get_personality_data(name).get("wake_words", [])

    # ===== Switching =====

    def detect_personality(self, user_input: str) -> str | None:
        """Return the first personality whose wake word appears in the input."""
        lower = user_input.lower()
        for name, data in self.personalities.items():
            for word in data.get("wake_words", []):
                if word.lower() in lower:
                    return name
        return None

    def switch_to(self, name: str) -> bool:
        if name in self.personalities:
            self.current = name
            return True
        return False

    # ===== Prompt pieces =====

    def get_persona_prompt(self, name: str | None = None) -> str:
        return self.get_personality_data(name).get("persona_prompt", "").strip()

    def get_chat_prompt(self, name: str | None = None) -> str:
        return self.get_personality_data(name).get("chat_prompt", "").strip()

    # ===== Voice =====

    def get_voice_sample(self, name: str | None = None) -> str | None:
        return self.get_personality_data(name).get("voice_sample")

    def get_voice_ref_text(self, name: str | None = None) -> str | None:
        return self.get_personality_data(name).get("voice_ref_text")

    # ===== Response rendering =====

    def render(self, key: str, **variables) -> str | None:
        """
        Render a personality response template.

        Returns None if the personality has no template for this key,
        so the caller can fall back to the tool's default formatter.
        """
        responses = self._current_data().get("responses", {})
        template = responses.get(key)
        if template is None:
            return None
        try:
            return template.format(**variables)
        except KeyError:
            # Template references a variable we weren't given — return raw
            return template