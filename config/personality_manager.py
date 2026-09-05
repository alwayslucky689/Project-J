import json
import os
import re

CONFIG_PATH = "config/default_config.json"

class PersonalityManager:
    def __init__(self):
        with open(CONFIG_PATH, "r") as f:
            self.config = json.load(f)
        self.personalities = self.config.get("personalities", {})
        self.current = "default"

    def detect_personality(self, user_input):
        """Checks if the input contains a wake word and returns the personality name."""
        user_input_lower = user_input.lower()
        for name, data in self.personalities.items():
            for word in data.get("wake_words", []):
                if word.lower() in user_input_lower:
                    return name
        return None  # No wake word detected

    def switch_to(self, personality_name):
        """Switches the current personality if it exists."""
        if personality_name in self.personalities:
            self.current = personality_name
            return True
        return False

    def get_current_prompt(self, prompt_type="system"):
        """
        Loads the correct prompt file for the current personality.
        prompt_type: 'system', 'command', or 'question'
        """
        personality_data = self.personalities.get(self.current, {})
        prompt_file = personality_data.get(f"{prompt_type}_prompt_file", f"{prompt_type}.txt")
        prompt_path = os.path.join("prompts", prompt_type, prompt_file)
        
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            # Fallback to default prompt
            fallback_path = os.path.join("prompts", prompt_type, f"{prompt_type}.txt")
            with open(fallback_path, "r", encoding="utf-8") as f:
                return f.read().strip()

    def get_current_name(self):
        return self.current

    def list_personalities(self):
        return list(self.personalities.keys())