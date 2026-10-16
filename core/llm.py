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

def ask_ollama_streaming(
    prompt_text: str,
    model: Optional[str] = None,
    on_token=None,
) -> str:
    """
    Print tokens to stdout as they arrive, return the full text.

    If `on_token` is provided, it is called once per token with that token
    string. Used by the streaming TTS path to feed a StreamingTTS instance
    without changing this function's primary contract.
    """
    print("🤖 ", end="", flush=True)
    full = ""
    for token in stream_tokens(prompt_text, model=model):
        print(token, end="", flush=True)
        if on_token is not None:
            try:
                on_token(token)
            except Exception as e:
                print(f"\n⚠️ on_token error: {e}")
        full += token
    if full:
        print()
    return full.strip()
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
# ===== Backend-agnostic single-shot text generation =====
#
# The chat path already dispatches on CHAT_BACKEND. This is the same
# pattern for one-shot generation used by the memory pipeline
# (extraction, summarization, topic-state merge). Without it, every
# call silently targets Ollama and fails when the user is on Bonsai.

def generate_text(prompt: str,
                  model: Optional[str] = None,
                  temperature: float = 0.3,
                  timeout=(10, 60)) -> str:
    """
    Single-shot text generation. Returns the response text, or an empty
    string on any failure — callers treat empty as "no result".
    """
    backend = getattr(settings, "CHAT_BACKEND", "ollama")
    if backend == "bonsai":
        return _generate_text_openai(prompt, model, temperature, timeout)
    return _generate_text_ollama(prompt, model, temperature, timeout)


def _generate_text_ollama(prompt, model, temperature, timeout) -> str:
    model = model or settings.FAST_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
    }
    try:
        response = _post(payload, timeout=timeout)
    except OllamaError as e:
        print(f"⚠️ generate_text (ollama) failed: {e}")
        return ""
    try:
        return (response.json().get("response") or "").strip()
    except Exception:
        return ""


def _generate_text_openai(prompt, model, temperature, timeout) -> str:
    model = (model or getattr(settings, "BONSAI_MODEL", None)
             or settings.REASONING_MODEL)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": temperature,
    }
    try:
        response = _post_openai_chat(payload, timeout=timeout)
    except OllamaError as e:
        print(f"⚠️ generate_text (bonsai) failed: {e}")
        return ""
    try:
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message", {}).get("content") or "").strip()
    except Exception:
        return ""
# ===== Chat endpoint (multi-turn conversation) =====

def _post_chat(payload: dict, timeout, stream: bool = False) -> requests.Response:
    """POST to Ollama's /api/chat endpoint."""
    url = settings.OLLAMA_URL.replace("/api/generate", "/api/chat")
    try:
        r = requests.post(url, json=payload, timeout=timeout, stream=stream)
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(f"Cannot connect to Ollama: {e}") from e
    except requests.exceptions.Timeout as e:
        raise OllamaError(f"Ollama timed out: {e}") from e
    except requests.exceptions.RequestException as e:
        raise OllamaError(f"Ollama request failed: {e}") from e

    if r.status_code != 200:
        raise OllamaError(f"Ollama returned HTTP {r.status_code}")
    return r


def chat_stream_tokens_openai(messages, model=None):
    from core.cancel import get_current
    model = model or settings.BONSAI_MODEL
    payload = {"model": model, "messages": messages, "stream": True}

    try:
        response = _post_openai_chat(payload, timeout=(10, 600), stream=True)
    except OllamaError as e:
        print(f"\n❌ {e}")
        return

    token = get_current()
    if token is not None:
        token.register_response(response)

    try:
        for raw in response.iter_lines(decode_unicode=False):
            if token is not None and token.is_cancelled():
                return
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            tok = delta.get("content", "")
            if tok:
                yield tok
    except Exception as e:
        # Response closed mid-iteration by cancel() — treat as normal end
        if token is not None and token.is_cancelled():
            return
        raise
    finally:
        try: response.close()
        except Exception: pass


def chat_stream_tokens(messages, model=None):
    from core.cancel import get_current
    model = model or settings.REASONING_MODEL
    payload = {"model": model, "messages": messages, "stream": True}

    try:
        response = _post_chat(payload, timeout=(10, 180), stream=True)
    except OllamaError as e:
        print(f"\n❌ {e}")
        return

    token = get_current()
    if token is not None:
        token.register_response(response)

    try:
        for raw in response.iter_lines(decode_unicode=False):
            if token is not None and token.is_cancelled():
                return
            if not raw:
                continue
            try:
                chunk = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if chunk.get("done"):
                break
            tok = chunk.get("response", "")
            if tok:
                yield tok
    except Exception:
        if token is not None and token.is_cancelled():
            return
        raise
    finally:
        try: response.close()
        except Exception: pass

# ===== Native tool calling (Phase 7a) =====

# ===== Native tool calling (Phase 7a) =====

def _chat_backend() -> str:
    return getattr(settings, "CHAT_BACKEND", "ollama")


def chat_with_tools(messages: list[dict],
                    tools: list[dict],
                    model: Optional[str] = None,
                    timeout=(10, 120)) -> dict:
    """
    Non-streaming chat with native tool support. Dispatches to Ollama
    (/api/chat) or Bonsai (OpenAI-compatible /v1/chat/completions).

    Returns:
        {"content": str, "tool_calls": list[dict], "raw": dict}

    Each tool_call is normalized to:
        {"id": str | None, "name": str, "arguments": dict, "_raw": dict}

    The `_raw` field is the backend-native dict; append_tool_exchange
    uses it to echo the call back in the shape the backend expects.
    """
    backend = _chat_backend()
    if backend == "bonsai":
        return _chat_with_tools_openai(messages, tools, model, timeout)
    return _chat_with_tools_ollama(messages, tools, model, timeout)


def _chat_with_tools_ollama(messages, tools, model, timeout) -> dict:
    model = model or settings.REASONING_MODEL
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "tools": tools,
    }

    try:
        response = _post_chat(payload, timeout=timeout)
    except OllamaError as e:
        return {"content": "", "tool_calls": [], "raw": {"error": str(e)}}

    try:
        data = response.json()
    except Exception as e:
        return {"content": "", "tool_calls": [],
                "raw": {"error": f"invalid json: {e}"}}

    msg = data.get("message") or {}
    raw_calls = msg.get("tool_calls") or []

    normalized = []
    for tc in raw_calls:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, dict):
            args = _parse_args(args)
        normalized.append({
            "id": None,
            "name": fn.get("name", ""),
            "arguments": args,
            "_raw": tc,
        })

    return {
        "content": (msg.get("content") or "").strip(),
        "tool_calls": normalized,
        "raw": data,
    }


def _chat_with_tools_openai(messages, tools, model, timeout) -> dict:
    """OpenAI-compatible path — used by Bonsai (llama-server)."""
    model = model or getattr(settings, "BONSAI_MODEL", None) \
        or settings.REASONING_MODEL
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
    }

    try:
        response = _post_openai_chat(payload, timeout=timeout)
    except OllamaError as e:
        return {"content": "", "tool_calls": [], "raw": {"error": str(e)}}

    try:
        data = response.json()
    except Exception as e:
        return {"content": "", "tool_calls": [],
                "raw": {"error": f"invalid json: {e}"}}

    choices = data.get("choices") or []
    if not choices:
        return {"content": "", "tool_calls": [], "raw": data}

    msg = choices[0].get("message") or {}
    raw_calls = msg.get("tool_calls") or []

    normalized = []
    for tc in raw_calls:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            args = _parse_args(args)
        elif not isinstance(args, dict):
            args = {}
        normalized.append({
            "id": tc.get("id"),
            "name": fn.get("name", ""),
            "arguments": args,
            "_raw": tc,
        })

    return {
        "content": (msg.get("content") or "").strip(),
        "tool_calls": normalized,
        "raw": data,
    }


def _parse_args(raw) -> dict:
    """Tool arguments can arrive as a dict, a JSON string, or nothing."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def append_tool_exchange(messages: list[dict],
                         tool_call: dict,
                         result_text: str) -> list[dict]:
    """
    Append the assistant's tool call + the tool result to the message
    stream, in whichever shape the active backend expects.
    """
    backend = _chat_backend()

    if backend == "bonsai":
        raw = tool_call.get("_raw") or {
            "id": tool_call.get("id") or "call_0",
            "type": "function",
            "function": {
                "name": tool_call.get("name", ""),
                "arguments": json.dumps(tool_call.get("arguments") or {}),
            },
        }
        call_id = raw.get("id") or "call_0"
        return [
            *messages,
            {"role": "assistant", "content": None, "tool_calls": [raw]},
            {"role": "tool", "tool_call_id": call_id, "content": result_text},
        ]

    # Ollama
    raw = tool_call.get("_raw") or {
        "function": {
            "name": tool_call.get("name", ""),
            "arguments": tool_call.get("arguments") or {},
        },
    }
    name = (raw.get("function") or {}).get("name", "")
    return [
        *messages,
        {"role": "assistant", "content": "", "tool_calls": [raw]},
        {"role": "tool", "tool_name": name, "content": result_text},
    ]
def chat_streaming(messages: list[dict],
                   model: Optional[str] = None,
                   on_token=None) -> str:
    """Stream a chat completion, printing tokens and returning the full text.
    Dispatches to Ollama or Bonsai based on settings.CHAT_BACKEND."""
    print("🤖 ", end="", flush=True)
    full = ""

    backend = getattr(settings, "CHAT_BACKEND", "ollama")
    if backend == "bonsai":
        gen = chat_stream_tokens_openai(messages, model=model)
    else:
        gen = chat_stream_tokens(messages, model=model)

    for token in gen:
        print(token, end="", flush=True)
        if on_token is not None:
            try:
                on_token(token)
            except Exception as e:
                print(f"\n⚠️ on_token error: {e}")
        full += token
    if full:
        print()
    return full.strip()
# ===== OpenAI-compatible chat client (for Bonsai via llama-server) =====

def _post_openai_chat(payload: dict, timeout, stream: bool = False) -> requests.Response:
    url = settings.BONSAI_URL
    try:
        r = requests.post(url, json=payload, timeout=timeout, stream=stream)
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(f"Cannot connect to Bonsai server: {e}") from e
    except requests.exceptions.Timeout as e:
        raise OllamaError(f"Bonsai timed out: {e}") from e
    except requests.exceptions.RequestException as e:
        raise OllamaError(f"Bonsai request failed: {e}") from e
    if r.status_code != 200:
        raise OllamaError(f"Bonsai returned HTTP {r.status_code}")
    return r


def chat_stream_tokens_openai(messages: list[dict],
                              model: Optional[str] = None
                              ) -> Generator[str, None, None]:
    """Stream tokens from an OpenAI-compatible /v1/chat/completions endpoint.
    Parses SSE format: 'data: {json}' lines terminated by 'data: [DONE]'.
    Only yields delta.content — reasoning deltas are skipped."""
    model = model or settings.BONSAI_MODEL
    payload = {"model": model, "messages": messages, "stream": True}

    try:
        response = _post_openai_chat(payload, timeout=(10, 600), stream=True)
    except OllamaError as e:
        print(f"\n❌ {e}")
        return

    try:
        for raw in response.iter_lines(decode_unicode=False):
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            token = delta.get("content", "")
            if token:
                yield token
    finally:
        response.close()