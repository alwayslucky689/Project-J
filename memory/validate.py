# memory/validate.py - Validate extracted candidates before writing
#
# Rejects facts that are hypothetical, unattributed, negated incorrectly,
# or otherwise unsafe to store without confirmation.
#
# Returns (valid: bool, revised: dict | None, reason: str).

import re


# Words indicating a correction or update to a prior fact
_CORRECTION_MARKERS = (
    "no longer", "not anymore", "not any more",
    "used to ", "i stopped", "i quit",
    "i changed", "anymore",
)

# Negation scopes we don't handle well enough to store safely
_SIMPLE_NEGATION = re.compile(
    r"\b(don't|doesn't|didn't|won't|can't|cannot|never)\s+\w+",
    re.IGNORECASE,
)


def validate_candidate(candidate: dict) -> tuple[bool, dict | None, str]:
    """
    Returns:
        (True, revised_candidate, "") if valid
        (False, None, reason) if rejected
    """
    # 1. Epistemic filter
    status = candidate.get("epistemic_status", "assertion")
    if status == "hypothetical":
        return False, None, "hypothetical statement"

    # 2. Subject attribution
    subject = candidate.get("subject", "").strip()
    if not subject:
        return False, None, "no subject"
    if subject not in ("user",) and not subject.startswith("user's "):
        return False, None, f"third-party subject: {subject}"

    # 3. Value presence
    value = candidate.get("value", "").strip()
    if not value or len(value) < 2:
        return False, None, "empty or trivial value"

    # 4. Detection of correction markers on the source text
    src = candidate.get("source_text", candidate.get("text", ""))
    is_correction = any(m in src.lower() for m in _CORRECTION_MARKERS)

    # 5. Simple negation on the verb → reject (too fragile to interpret)
    if _SIMPLE_NEGATION.search(src) and not is_correction:
        return False, None, "simple negation"

    # Attach the correction flag for resolve.py
    revised = dict(candidate)
    revised["is_correction"] = is_correction
    revised["status"] = "active"  # inferred but confirmed enough

    return True, revised, ""