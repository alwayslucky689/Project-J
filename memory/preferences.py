# memory/preferences.py
import json
import os

PREFERENCES_FILE = "config/user_preferences.json"

def load_preferences():
    """Loads user preferences from a JSON file."""
    try:
        with open(PREFERENCES_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_preference(key, value):
    """Saves a user preference."""
    prefs = load_preferences()
    prefs[key] = value
    with open(PREFERENCES_FILE, "w") as f:
        json.dump(prefs, f, indent=2)

def get_preference(key, default=None):
    """Gets a user preference."""
    return load_preferences().get(key, default)

# Add this to your tools:
def remember_fact(fact):
    """
    Tool for the AI to remember facts about the user.
    Example: "Remember that I like cats" → {"key": "likes", "value": "cats"}
    """
    # The AI extracts key-value pairs from the fact
    # You'll need to prompt it to output structured data
    save_preference(fact_key, fact_value)
    return f"✓ I'll remember that {fact_key} is {fact_value}"

def recall_fact(key):
    """Recalls a fact about the user."""
    return get_preference(key, "I don't have any information about that.")