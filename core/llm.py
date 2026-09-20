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