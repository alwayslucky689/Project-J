"""
Tool registry — the single source of truth for AI-callable tools.

A tool is defined once here. Everything else (prompt schema generation,
dispatch, permission checks, per-personality filtering) derives from
the registry.

Phase 1a: minimal shape. Only the fields needed to migrate YouTube.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class Permission(Enum):
    SAFE = "safe"           # read-only, no side effects
    ACTION = "action"       # opens apps, changes state
    DANGEROUS = "dangerous" # purchases, logins, irreversible


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]
    formatter: Optional[Callable[[Any], str]] = None
    permission: Permission = Permission.SAFE
    response_key: str = "tool_success"    # for personality templates (Phase 1d)
    llm_acknowledge: bool = False         # if True, let LLM compose the ack (Phase 1d)
    personalities: Optional[set[str]] = None  # None = available to all

    def format(self, result: Any) -> str:
        """Convert the handler's raw return value into a response string."""
        if self.formatter is None:
            return str(result) if result is not None else ""
        try:
            return self.formatter(result)
        except Exception as e:
            return f"⚠️ Formatter error for {self.name}: {e}"


TOOL_REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    """Register a tool. Raises on duplicate names."""
    if tool.name in TOOL_REGISTRY:
        raise ValueError(f"Tool already registered: {tool.name}")
    TOOL_REGISTRY[tool.name] = tool
    return tool


def get_tool(name: str) -> Optional[Tool]:
    return TOOL_REGISTRY.get(name)


def has_tool(name: str) -> bool:
    return name in TOOL_REGISTRY


def list_tools() -> list[Tool]:
    return list(TOOL_REGISTRY.values())


def list_tool_names() -> list[str]:
    return list(TOOL_REGISTRY.keys())