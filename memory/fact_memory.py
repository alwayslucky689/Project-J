import json
import os

MEMORY_DIR = "data"
FACTS_FILE = os.path.join(MEMORY_DIR, "facts.json")

# Ensure the data folder exists
os.makedirs(MEMORY_DIR, exist_ok=True)

def load_facts():
    """Loads all stored facts from the JSON file."""
    try:
        with open(FACTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_fact(fact: str):
    """Saves a new natural language fact to the file."""
    facts = load_facts()
    facts.append(fact)
    with open(FACTS_FILE, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)
    return f"✓ Remembered: {fact}"

def get_facts_context():
    """Returns a formatted string of all facts for the system prompt."""
    facts = load_facts()
    if not facts:
        return ""
    return "Known facts about the user:\n- " + "\n- ".join(facts)