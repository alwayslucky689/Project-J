from memory.fact_memory import save_fact, get_facts_context

def remember_fact(fact: str):
    """Tool wrapper for the AI to call."""
    return save_fact(fact)

def get_memory_context():
    """Used by the main prompt builder to inject facts."""
    return get_facts_context()