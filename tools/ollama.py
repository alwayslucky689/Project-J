"""
Ollama/AI-related tools.

The 'chat' tool is the explicit fallback for anything that isn't a command.
The router is instructed to pick it when nothing else fits, which gives
Needle a valid destination instead of forcing a bad tool match.
"""

from config import settings
from core.llm import ask_ollama
from core.registry import register, Tool, Permission
from core.tool_descriptions import DESCRIPTIONS


def chat(message: str) -> str:
    """Answer a general message via the reasoning model."""
    print(f"🤔 Answering: {message}")
    prompt = f"""Answer the user's message naturally, conversationally, and accurately.
Be concise but helpful. Don't mention that you're an AI.

User: {message}
Assistant:"""
    return ask_ollama(prompt, is_json=False, model=settings.REASONING_MODEL)


register(Tool(
    name="chat",
    description=DESCRIPTIONS["chat"],
    handler=chat,
    formatter=lambda s: s,
    permission=Permission.SAFE,
    response_key="asked_question",
))