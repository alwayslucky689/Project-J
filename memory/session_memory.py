import json
import os
from datetime import datetime

SESSION_DIR = "data/sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

def save_conversation(session_name, messages):
    """Saves a list of messages to a session file."""
    filename = os.path.join(SESSION_DIR, f"{session_name}.json")
    data = {
        "session_name": session_name,
        "updated_at": datetime.now().isoformat(),
        "messages": messages
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_conversation(session_name):
    """Loads a session file."""
    filename = os.path.join(SESSION_DIR, f"{session_name}.json")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("messages", [])
    except FileNotFoundError:
        return []

def list_sessions():
    """Returns a list of all saved session names."""
    files = os.listdir(SESSION_DIR)
    return [f.replace(".json", "") for f in files if f.endswith(".json")]