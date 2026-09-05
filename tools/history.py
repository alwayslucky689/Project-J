# At the top of your script
conversation_history = []

def add_to_history(role, content):
    """Adds a message to the conversation history."""
    conversation_history.append({"role": role, "content": content})
    # Keep only the last 20 messages (prevents token overflow)
    if len(conversation_history) > 20:
        conversation_history.pop(0)

def get_context():
    """Returns the conversation history as a string for the prompt."""
    context = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in conversation_history
    ])
    return context

