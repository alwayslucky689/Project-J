# memory/summarize.py - Generate session summaries via the fast model
#
# Called by the worker after a session closes. Produces a 1-2 sentence
# factual summary, embeds it, and writes it back to the sessions row and
# session_embeddings vec table. Idempotent: safe to re-run.

from typing import Optional

from memory.store import get_conn, write_transaction, _vec_to_blob


def summarize_session(session_id: int) -> Optional[str]:
    """
    Summarize a closed session. Returns the summary text, or None on
    failure or if the session is too short.
    """
    conn = get_conn()

    row = conn.execute(
        "SELECT summary, started_at, ended_at FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if not row:
        return None
    if row["summary"]:
        return row["summary"]

    turns = conn.execute(
        "SELECT role, text FROM turns WHERE session_id = ? "
        "AND excluded_from_extraction = 0 "
        "ORDER BY ts",
        (session_id,),
    ).fetchall()
    if len(turns) < 2:
        return None

    transcript = _build_transcript(turns)
    summary = _generate_summary(transcript)
    if not summary:
        return None

    _write_summary(session_id, summary)
    return summary


def _build_transcript(turns) -> str:
    """Compact U:/A: transcript, with long turns truncated."""
    lines = []
    for t in turns:
        prefix = "U:" if t["role"] == "user" else "A:"
        text = t["text"].strip().replace("\n", " ")
        if len(text) > 200:
            text = text[:197] + "..."
        lines.append(f"{prefix} {text}")
    return "\n".join(lines)


def _generate_summary(transcript: str) -> Optional[str]:
    try:
        from config import settings
        from core.llm import ask_ollama
    except ImportError:
        return None

    prompt = f"""Summarize this conversation in 1-2 sentences. Focus on:
- what the user was working on or asking about
- any decisions, conclusions, or facts stated
- topics that might be worth remembering later

Skip greetings, small talk, and meta-discussion about the assistant.
Write in third person, factual tone. No first-person pronouns.

Conversation:
{transcript}

Summary:"""

    try:
        result = ask_ollama(prompt, is_json=False, model=settings.FAST_MODEL)
    except Exception as e:
        print(f"⚠️ Summary generation failed for transcript: {e}")
        return None

    summary = result.strip().strip('"').strip("'")
    if not summary or len(summary) < 15 or len(summary) > 600:
        return None
    return summary


def _write_summary(session_id: int, summary: str):
    with write_transaction() as conn:
        conn.execute(
            "UPDATE sessions SET summary = ? WHERE id = ?",
        (summary, session_id),
        )

    # Embed outside the write lock — model call is ~12ms
    try:
        from memory.embed import embed
        vec = embed(summary)
        if vec is None:
            return
        conn = get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO session_embeddings "
            "(session_id, embedding) VALUES (?, ?)",
            (session_id, _vec_to_blob(vec)),
        )
        conn.commit()
    except Exception as e:
        print(f"⚠️ Could not embed session {session_id}: {e}")