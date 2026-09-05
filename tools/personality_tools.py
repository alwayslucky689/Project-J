from config.personality_manager import PersonalityManager

# We'll pass the manager instance from assistant.py
_personality_manager = None

def init_personality_tools(manager):
    global _personality_manager
    _personality_manager = manager

def change_personality(name):
    """Tool for the AI to switch personalities."""
    if _personality_manager.switch_to(name):
        return f"✓ Switched personality to: {name}"
    available = _personality_manager.list_personalities()
    return f"❌ Personality '{name}' not found. Available: {', '.join(available)}"