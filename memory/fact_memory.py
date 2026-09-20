import json
import os
from core.registry import register, Tool, Permission
from config.paths import FACTS_FILE

# FACTS_FILE is a Path object; convert to str for json/open()
FACTS_FILE = str(FACTS_FILE)

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
def _remember_fact(fact):
    return save_fact(fact)


register(Tool(
    name="remember_fact",
    description='Stores a fact about the user. Takes "fact" (string).',
    handler=_remember_fact,
    formatter=lambda s: s,  # returns "✓ Remembered: ..." already
    permission=Permission.SAFE,
    response_key="fact_saved",
))