"""
Ollama/AI-related tools for the AI Assistant.

The ask_question tool is registered here. In practice, the router in
assistant.py intercepts ask_question as a signal to fall back to the
streaming chat path, but keeping the tool registered means the router
has a name it can return when it decides nothing else fits.
"""

from config import settings
from core.llm import ask_ollama
from core.registry import register, Tool, Permission


def ask_question(question: str) -> str:
    """Answer a general question via the reasoning model."""
    print(f"🤔 Answering: {question}")
    prompt = f"""Answer the user's question naturally, conversationally, and accurately.
Be concise but helpful. Don't mention that you're an AI.

User: {question}
Assistant:"""
    return ask_ollama(prompt, is_json=False, model=settings.REASONING_MODEL)


register(Tool(
    name="ask_question",
    description='Answers a general question. Takes "question" (string). Used as a fallback when no other tool fits.',
    handler=ask_question,
    formatter=lambda s: s,
    permission=Permission.SAFE,
    response_key="asked_question",
))