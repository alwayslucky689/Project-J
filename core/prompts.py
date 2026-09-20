"""
Prompt generation — builds the tool schema from the registry.

The tool list is written down exactly once, in core/registry.py, and
materialised here for injection into the system prompt. Adding a tool
no longer requires editing any prompt file.
"""

from core.registry import list_tools


# The JSON output rules are identical for every personality. Only the
# persona description above these rules changes.
COMMAND_JSON_RULES = """You are a computer control AI. You MUST output ONLY a JSON object (or a JSON array of objects) that EXACTLY matches one of the available tools.

CRITICAL RULES:
1. You MUST use the exact tool names listed above.
2. You MUST output ONLY valid JSON.
3. If the user's request doesn't match any tool, use: {"tool": "ask_question", "question": "the user's original request"}
4. Do NOT include any other text, explanations, or markdown.
5. Do NOT wrap the JSON in backticks or code blocks.

User: {user_input}
Assistant:"""


def build_tool_schema() -> str:
    """Generate the AVAILABLE TOOLS section from the registry."""
    lines = ["AVAILABLE TOOLS (you MUST use these exact names):"]
    for tool in list_tools():
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)