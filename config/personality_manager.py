# config/personality_manager.py

import os
from config import settings

class PersonalityManager:
    def __init__(self):
        # Load directly from settings.py
        self.personalities = settings.PERSONALITIES
        self.current = settings.DEFAULT_PERSONALITY
        self.prompts_dir = settings.PROMPTS_DIR

    def detect_personality(self, user_input):
        """
        Checks if the input contains a wake word.
        Returns the personality name (str) or None.
        """
        user_input_lower = user_input.lower()
        for name, data in self.personalities.items():
            for word in data.get("wake_words", []):
                if word.lower() in user_input_lower:
                    return name
        return None

    def switch_to(self, personality_name):
        """Switches the current personality if it exists."""
        if personality_name in self.personalities:
            self.current = personality_name
            return True
        return False

    def get_prompt(self, prompt_type="system"):
        """
        Loads the correct prompt file for the current personality.
        prompt_type: 'system', 'command', or 'question'
        """
        personality_data = self.personalities.get(self.current, {})
        # The key in settings is e.g. "system_prompt_file"
        prompt_filename = personality_data.get(f"{prompt_type}_prompt_file")
        
        if not prompt_filename:
            # Fallback: if the personality doesn't specify this prompt type,
            # use the default file (e.g., prompts/system/default.txt)
            prompt_filename = f"{prompt_type}.txt"

        prompt_path = os.path.join(self.prompts_dir, prompt_type, prompt_filename)
        
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            # Ultimate fallback: hardcoded prompt if file missing
            if prompt_type == "system":
                return "You are a helpful AI assistant."
            elif prompt_type == "command":
                return "You are a command router. Output JSON only. User: {user_input}"
            else:  # question
                return "Answer the user's question naturally. User: {user_input}"

    def get_current_name(self):
        return self.current

    def get_wake_words(self, personality_name=None):
        """Returns the list of wake words for a given personality."""
        if personality_name is None:
            personality_name = self.current
        return self.personalities.get(personality_name, {}).get("wake_words", [])

    def list_personalities(self):
        return list(self.personalities.keys())