"""
Ollama HTTP client — the single place all LLM calls go through.

Both streaming and non-streaming. No subprocesses, no hardcoded URLs,
no duplicated error handling.
"""

import json
import re
from typing import Generator, Optional

import requests

from config import settings
# ===== Needle-based routing =====

import re as _re
from core.routing import prefilter_schemas

_NEEDLE_ARG_PATTERN = _re.compile(r'(optional\s+)?"([^"]+)"\s+\((\w+)[^)]*\)')
_NEEDLE_TYPE_MAP = {
    "string": "string",
    "integer": "integer",
    "boolean": "boolean",
    "number": "number",
    "float": "number",
}

_needle_schemas_cache: list[dict] | None = None


def _build_all_schemas() -> list[dict]:
    """Build the full tool schema list from the registry. Cached."""
    global _needle_schemas_cache
    if _needle_schemas_cache is not None:
        return _needle_schemas_cache

    from core.registry import list_tools
    from core.tool_descriptions import DESCRIPTIONS

    schemas = []
    for t in list_tools():
        desc = DESCRIPTIONS.get(t.name, t.description)
        props: dict = {}
        req: list[str] = []
        for m in _NEEDLE_ARG_PATTERN.finditer(desc):
            optional_flag, name, type_name = m.groups()
            props[name] = {"type": _NEEDLE_TYPE_MAP.get(type_name, "string")}
            if not optional_flag:
                req.append(name)
        params: dict = {"type": "object", "properties": props}
        if req:
            params["required"] = req
        schemas.append({"name": t.name, "description": desc, "parameters": params})

    _needle_schemas_cache = schemas
    return schemas


def route_to_tool(query: str) -> dict:
    """
    Route a query to a tool call using Needle + keyword prefilter.

    Returns a dict in the same shape the JSON router used to produce:
      {"tool": "play_spotify_song", "song": "...", "_confidence": 0.98}
      {"tool": "chat", "message": query, "_confidence": 0.42}

    Keys starting with "_" are metadata for logging and are stripped by
    execute_single_action before the handler is called.
    """
    try:
        import needle
    except ImportError as e:
        # Fall back to chat if Needle isn't installed
        print(f"⚠️ Needle not available: {e}")
        return {"tool": "chat", "message": query, "_error": str(e)}

    schemas = _build_all_schemas()
    filtered = prefilter_schemas(query, schemas)

    try:
        agent = needle.Needle(tools=filtered)
        response = agent.complete(query)
    except Exception as e:
        return {"tool": "chat", "message": query, "_error": str(e)}

    if response.get("error"):
        return {"tool": "chat", "message": query, "_error": response["error"]}

    calls = response.get("function_calls") or []
    confidence = response.get("confidence", 0.0)

    if not calls:
        return {"tool": "chat", "message": query, "_confidence": confidence}

    call = calls[0]
    args = call.get("arguments") or {}
    return {
        "tool": call.get("name"),
        **args,
        "_confidence": confidence,
        "_candidates": [s["name"] for s in filtered],
    }
class OllamaError(Exception):
    """Raised for all Ollama communication failures."""


def _endpoint() -> str:
    return settings.OLLAMA_URL


def _post(payload: dict, timeout, stream: bool = False) -> requests.Response:
    try:
        r = requests.post(_endpoint(), json=payload, timeout=timeout, stream=stream)
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(f"Cannot connect to Ollama: {e}") from e
    except requests.exceptions.Timeout as e:
        raise OllamaError(f"Ollama timed out: {e}") from e
    except requests.exceptions.RequestException as e:
        raise OllamaError(f"Ollama request failed: {e}") from e

    if r.status_code != 200:
        raise OllamaError(f"Ollama returned HTTP {r.status_code}")
    return r


def _extract_json(text: str):
    """Pull the first JSON object or array out of an LLM response."""
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


def ask_ollama(prompt_text: str, is_json: bool = False, model: Optional[str] = None):
    """
    Non-streaming generation.

    Returns a dict (if is_json) or a string. On error, returns either
    {"tool": "error", "message": ...} or an "Error: ..." string so callers
    can handle both shapes uniformly.
    """
    model = model or settings.FAST_MODEL
    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": False,
        "temperature": 0.1 if is_json else 0.7,
    }

    try:
        response = _post(payload, timeout=(10, 60))
    except OllamaError as e:
        return {"tool": "error", "message": str(e)} if is_json else f"Error: {e}"

    try:
        data = response.json()
        text = data.get("response", "").strip()
    except json.JSONDecodeError:
        return {"tool": "error", "message": "Invalid JSON response"} if is_json else "Error: invalid response"

    if not text:
        return {"tool": "error", "message": "Empty response"} if is_json else "I didn't get a response."

    if is_json:
        parsed = _extract_json(text)
        if parsed is None:
            return {"tool": "error", "message": f"No JSON found in: {text[:200]}"}
        return parsed

    return text


def stream_tokens(prompt_text: str, model: Optional[str] = None) -> Generator[str, None, None]:
    """
    Yield tokens as they arrive over HTTP.

    Uses Ollama's NDJSON streaming endpoint. On connection error, prints
    a message and yields nothing, so callers get an empty stream rather
    than an exception.
    """
    model = model or settings.FAST_MODEL
    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": True,
        "temperature": 0.7,
    }

    try:
        response = _post(payload, timeout=(10, 180), stream=True)
    except OllamaError as e:
        print(f"\n❌ {e}")
        return

    try:
        for raw in response.iter_lines(decode_unicode=False):
            if not raw:
                continue
            try:
                chunk = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if chunk.get("done"):
                break
            token = chunk.get("response", "")
            if token:
                yield token
    finally:
        response.close()


def ask_ollama_streaming(prompt_text: str, model: Optional[str] = None) -> str:
    """
    Print tokens to stdout as they arrive, return the full text.

    Contract preserved from the old subprocess-based version:
    - Prints "🤖 " prefix once
    - Prints tokens without newlines as they arrive
    - Prints a trailing newline if anything was emitted
    - Returns the stripped full text
    """
    print("🤖 ", end="", flush=True)
    full = ""
    for token in stream_tokens(prompt_text, model=model):
        print(token, end="", flush=True)
        full += token
    if full:
        print()
    return full.strip()