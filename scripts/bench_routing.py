"""
Router benchmark harness.

Runs every query in data/routing_test_set.jsonl through a routing backend
and reports tool-name accuracy, argument accuracy, and latency.

Usage:
    python scripts/bench_routing.py --backend llama
    python scripts/bench_routing.py --backend needle
    python scripts/bench_routing.py --backend llama --verbose
"""# Make project root importable when run from anywhere
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import tools  # noqa: F401 — triggers tool registration
import argparse
import json
import statistics
import re
import time
from datetime import datetime

from core.routing import prefilter_schemas


from config import settings
from core.llm import ask_ollama
from core.prompts import build_tool_schema, COMMAND_JSON_RULES


TEST_SET_PATH = PROJECT_ROOT / "data" / "routing_test_set.jsonl"
RESULTS_DIR = PROJECT_ROOT / "data" / "bench_results"

# Short-circuits for deterministic commands (no LLM routing needed)
import re

_YOUTUBE_REF = re.compile(
    r"\b(?:play|show)\s+(?:the\s+)?(?:video\s+|result\s+)?"
    r"(\d+|first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\b",
    re.I,
)
_WORD_NUM = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
}


def _try_youtube_ref(query: str) -> dict | None:
    m = _YOUTUBE_REF.search(query)
    if not m:
        return None
    tok = m.group(1).lower()
    if tok.isdigit():
        return {"tool": "play_youtube_video", "index": int(tok)}
    if tok in _WORD_NUM:
        return {"tool": "play_youtube_video", "index": _WORD_NUM[tok]}
    return None
# ===== Test set loading =====

def load_test_set() -> list[dict]:
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(f"Test set not found: {TEST_SET_PATH}")
    cases = []
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping malformed line {i}: {e}")
    return cases


# ===== Backends =====

def route_llama(query: str) -> dict:
    """Current router: llama3.2:3b via JSON mode."""
    prompt = (
        build_tool_schema()
        + "\n\n"
        + COMMAND_JSON_RULES.replace("{user_input}", query)
    )
    decision = ask_ollama(prompt, is_json=True, model=settings.FAST_MODEL)
    if not isinstance(decision, dict):
        return {"tool": None, "raw": decision}
    return decision


# ===== Needle backend =====

# Our tool descriptions follow a fixed format:
#   'Plays a song. Takes "song" (string) and optional "artist" (string).'
#   'Takes no arguments.'
# We parse them into JSON schemas so Needle's grammar can constrain output.
_ARG_PATTERN = re.compile(r'(optional\s+)?"([^"]+)"\s+\((\w+)[^)]*\)')

_TYPE_MAP = {
    "string": "string",
    "integer": "integer",
    "boolean": "boolean",
    "number": "number",
    "float": "number",
}


def _parse_args_from_description(description: str) -> dict:
    """Extract a JSON schema object from a 'Takes ...' description."""
    properties: dict = {}
    required: list[str] = []
    for match in _ARG_PATTERN.finditer(description):
        optional_flag, name, type_name = match.groups()
        properties[name] = {"type": _TYPE_MAP.get(type_name, "string")}
        if not optional_flag:
            required.append(name)
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _build_schemas() -> list[dict]:
    """Build the tool schema list Needle expects, from the registry."""
    from core.registry import list_tools
    schemas = []
    for tool in list_tools():
        schemas.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": _parse_args_from_description(tool.description),
        })
    return schemas


# Cache the schemas (they don't change between calls)
_schemas_cache = None


def _get_schemas() -> list[dict]:
    global _schemas_cache
    if _schemas_cache is None:
        _schemas_cache = _build_schemas()
    return _schemas_cache


def route_needle(query: str) -> dict:
    # Deterministic short-circuits first
    ref = _try_youtube_ref(query)
    if ref is not None:
        return ref
    """
    Route via Needle with keyword prefiltering.

    Prefilter narrows the tool list from ~28 to ~5-10 based on the query.
    Needle's base checkpoint requires this; without it, selection collapses.
    """
    import needle

    schemas = _get_schemas()
    filtered = prefilter_schemas(query, schemas)

    try:
        agent = needle.Needle(tools=filtered)
    except Exception as e:
        return {"tool": None, "raw": f"Needle init failed: {e}"}

    try:
        response = agent.complete(query)
    except Exception as e:
        return {"tool": None, "raw": f"Needle call failed: {e}"}

    if response.get("error"):
        return {"tool": None, "_error": response["error"]}

    calls = response.get("function_calls") or []
    if not calls:
        return {"tool": None, "_confidence": response.get("confidence", 0.0)}

    call = calls[0]
    args = call.get("arguments") or {}
    return {
        "tool": call.get("name"),
        **args,
        "_confidence": response.get("confidence", 0.0),
        "_candidates": [s["name"] for s in filtered],  # helps debug failures
    }
# ===== Backend registry =====

BACKENDS = {
    "llama": route_llama,
    "needle": route_needle,
}
# ===== Comparison =====

def tools_match(expected: str | None, actual: str | None) -> bool:
    """Treat None and 'ask_question' as equivalent (both mean chat fallback)."""
    def norm(t):
        if t is None or t in ("ask_question", "chat"):
            return None
        return t
    return norm(expected) == norm(actual)


def args_match(expected: dict, actual: dict) -> bool:
    """Compare argument dicts, ignoring extra keys the router added."""
    # Strip the tool key from actual if present
    actual = {k: v for k, v in actual.items() if k != "tool"}

    # Empty expected means "no arguments matter"
    if not expected:
        return True

    # Every expected key must be present with an equal value
    for k, v in expected.items():
        if k not in actual:
            return False
        # Loose equality: ints vs strings, casing on strings
        if isinstance(v, str) and isinstance(actual[k], str):
            if v.lower().strip() != actual[k].lower().strip():
                return False
        elif v != actual[k]:
            return False
    return True


# ===== Runner =====

def run_benchmark(backend_name: str, verbose: bool = False) -> dict:
    backend = BACKENDS[backend_name]
    cases = load_test_set()
    print(f"Running {len(cases)} cases through backend: {backend_name}")
    print("-" * 70)

    results = []
    latencies = []

    for i, case in enumerate(cases, 1):
        query = case["query"]
        expected_tool = case.get("expected_tool")
        expected_args = case.get("expected_args", {})
        notes = case.get("notes", "")

        start = time.perf_counter()
        try:
            decision = backend(query)
            error = None
        except Exception as e:
            decision = {"tool": None, "raw": str(e)}
            error = str(e)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        actual_tool = decision.get("tool") if isinstance(decision, dict) else None
        actual_args = decision if isinstance(decision, dict) else {}

        tool_ok = tools_match(expected_tool, actual_tool)
        args_ok = tool_ok and args_match(expected_args, actual_args)
        passed = tool_ok and args_ok

        results.append({
            "query": query,
            "expected_tool": expected_tool,
            "actual_tool": actual_tool,
            "expected_args": expected_args,
            "actual_args": {k: v for k, v in actual_args.items() if k != "tool"},
            "tool_ok": tool_ok,
            "args_ok": args_ok,
            "passed": passed,
            "latency_ms": round(elapsed_ms, 1),
            "error": error,
            "notes": notes,
        })

        marker = "✅" if passed else ("🟡" if tool_ok else "❌")
        if verbose or not passed:
            print(f"{marker} [{elapsed_ms:6.1f}ms] {query!r}")
            if not passed:
                print(f"    expected: {expected_tool} {expected_args}")
                print(f"    actual:   {actual_tool} {actual_args}")
                if notes:
                    print(f"    note:     {notes}")
                if error:
                    print(f"    error:    {error}")

    # Summary
    n = len(results)
    tool_correct = sum(1 for r in results if r["tool_ok"])
    full_correct = sum(1 for r in results if r["passed"])
    parse_failures = sum(1 for r in results if r["error"])

    summary = {
        "backend": backend_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total": n,
        "tool_accuracy": round(100 * tool_correct / n, 1),
        "full_accuracy": round(100 * full_correct / n, 1),
        "parse_failures": parse_failures,
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(
            statistics.quantiles(latencies, n=20)[18] if n >= 20 else max(latencies), 1
        ),
        "latency_mean_ms": round(statistics.mean(latencies), 1),
        "results": results,
    }

    print("-" * 70)
    print(f"Tool accuracy:  {summary['tool_accuracy']}%  ({tool_correct}/{n})")
    print(f"Full accuracy:  {summary['full_accuracy']}%  ({full_correct}/{n})")
    print(f"Parse failures: {parse_failures}")
    print(f"Latency p50:    {summary['latency_p50_ms']}ms")
    print(f"Latency p95:    {summary['latency_p95_ms']}ms")
    print(f"Latency mean:   {summary['latency_mean_ms']}ms")

    return summary


def save_results(summary: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{summary['backend']}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the tool router")
    parser.add_argument("--backend", default="llama", choices=list(BACKENDS.keys()))
    parser.add_argument("--verbose", action="store_true", help="Print every case")
    args = parser.parse_args()

    summary = run_benchmark(args.backend, verbose=args.verbose)
    save_results(summary)


if __name__ == "__main__":
    main()