# memory/review.py - User-facing memory review CLI (Phase 6o)
#
# Browse, edit, delete, merge, and promote/demote facts. Interactive mode
# walks through candidates one by one.
#
# A memory system the user can't correct is a liability. This is the
# correction surface.
#
# Usage:
#   python -m memory.review                      # browse active facts
#   python -m memory.review --all                # include non-active
#   python -m memory.review --status candidate   # only candidates
#   python -m memory.review --interactive        # walk through candidates
#   python -m memory.review interactive --scope unscoped
#   python -m memory.review edit <id>
#   python -m memory.review delete <id>
#   python -m memory.review merge <target_id> <source_id>
#   python -m memory.review promote <id>
#   python -m memory.review demote <id>

import argparse
import sys

from memory.store import get_conn, write_transaction, now_iso
from memory.fact_memory import forget_fact


# ===== Helpers =====

def _confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def _format_scope(row) -> str:
    s = row["scope_type"]
    if row["scope_topic_id"]:
        s = f"{s}:{row['scope_topic_id']}"
    return s


def _fetch_fact(fact_id: int):
    return get_conn().execute(
        "SELECT * FROM facts WHERE id = ?", (fact_id,)
    ).fetchone()


def _set_status(fact_id: int, status: str):
    with write_transaction() as tx:
        tx.execute(
            "UPDATE facts SET status = ?, updated_at = ? WHERE id = ?",
            (status, now_iso(), fact_id),
        )


def _activate_and_pin(fact_id: int):
    with write_transaction() as tx:
        tx.execute(
            "UPDATE facts SET status='active', pinned=1, updated_at=? "
            "WHERE id = ?",
            (now_iso(), fact_id),
        )


def _edit_text(fact_id: int, new_text: str):
    new_text = new_text.strip()
    if not new_text:
        return
    with write_transaction() as tx:
        tx.execute(
            "UPDATE facts SET text = ?, updated_at = ? WHERE id = ?",
            (new_text, now_iso(), fact_id),
        )
    _reembed_fact(fact_id, new_text)


def _reembed_fact(fact_id: int, new_text: str):
    """Replace the fact's embedding. Non-fatal on failure — backfill on
    next startup will catch it if it doesn't get written now."""
    try:
        from memory.embed import embed
        from memory.store import _vec_to_blob
    except ImportError:
        return
    vec = embed(new_text)
    if vec is None:
        print("  ⚠️  Embedding failed; fact updated but not re-embedded.")
        return
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO fact_embeddings (fact_id, embedding) "
            "VALUES (?, ?)",
            (fact_id, _vec_to_blob(vec)),
        )
        conn.commit()
    except Exception as e:
        print(f"  ⚠️  Could not re-embed fact {fact_id}: {e}")


# ===== Commands =====

def cmd_browse(args):
    conn = get_conn()

    if args.status:
        where = "WHERE status = ?"
        params = [args.status]
    elif args.all:
        where = ""
        params = []
    else:
        where = "WHERE status = 'active'"
        params = []

    rows = conn.execute(
        f"""
        SELECT id, status, scope_type, scope_topic_id, source, text, pinned
        FROM facts
        {where}
        ORDER BY pinned DESC, created_at DESC
        LIMIT ?
        """,
        (*params, args.limit),
    ).fetchall()

    if not rows:
        print("(no facts match)")
        return

    for r in rows:
        marker = "📌" if r["pinned"] else "  "
        scope = _format_scope(r)
        text = r["text"].replace("\n", " ")
        print(
            f"{marker} [{r['id']:>4}] "
            f"{r['status']:<10} {scope:<14} {r['source']:<14}  {text}"
        )


def cmd_edit(args):
    row = _fetch_fact(args.id)
    if not row:
        print(f"No fact with id {args.id}")
        return
    if row["status"] == "deleted":
        print("❌ Fact is deleted. Cannot edit.")
        return

    print(f"Current: {row['text']}")
    try:
        new_text = input("New text (blank cancels): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not new_text:
        print("Canceled.")
        return

    _edit_text(args.id, new_text)
    print(f"✏️  Updated fact {args.id}.")


def cmd_delete(args):
    row = _fetch_fact(args.id)
    if not row:
        print(f"No fact with id {args.id}")
        return
    if row["status"] == "deleted":
        print("Already deleted.")
        return

    print(f"[{args.id}] {row['text']}")
    if not _confirm("Delete this fact?", args.yes):
        print("Aborted.")
        return
    print(forget_fact(fact_id=args.id))


def cmd_merge(args):
    if args.target_id == args.source_id:
        print("❌ Cannot merge a fact with itself.")
        return

    target = _fetch_fact(args.target_id)
    source = _fetch_fact(args.source_id)
    if not target or not source:
        print("❌ One or both fact ids not found.")
        return
    if target["status"] != "active" or source["status"] != "active":
        print("❌ Both facts must be active to merge.")
        return

    print(f"Keep:   [{target['id']}] {target['text']}")
    print(f"Merge:  [{source['id']}] {source['text']}")
    if not _confirm("Proceed?", args.yes):
        print("Aborted.")
        return

    print(forget_fact(
        fact_id=source["id"],
        reason=f"merged into fact {target['id']}",
    ))
    print(f"✅ Merged into [{target['id']}].")


def cmd_promote(args):
    row = _fetch_fact(args.id)
    if not row:
        print(f"No fact with id {args.id}")
        return
    if row["status"] == "active":
        print("Already active.")
        return
    if row["status"] == "deleted":
        print("❌ Fact is deleted. Cannot promote.")
        return

    _set_status(args.id, "active")
    if args.pin:
        _activate_and_pin(args.id)
        print(f"📌 Promoted and pinned: {row['text']}")
    else:
        print(f"✅ Promoted: {row['text']}")


def cmd_demote(args):
    row = _fetch_fact(args.id)
    if not row:
        print(f"No fact with id {args.id}")
        return
    if row["status"] == "candidate":
        print("Already a candidate.")
        return
    if row["status"] == "deleted":
        print("❌ Fact is deleted. Cannot demote.")
        return

    _set_status(args.id, "candidate")
    if args.unpin:
        with write_transaction() as tx:
            tx.execute(
                "UPDATE facts SET pinned=0, updated_at=? WHERE id=?",
                (now_iso(), args.id),
            )
    print(f"⬇️  Demoted to candidate: {row['text']}")


def cmd_interactive(args):
    conn = get_conn()

    status_filter = args.status
    if args.all:
        status_filter = "all"

    clauses = []
    params: list = []
    if status_filter != "all":
        clauses.append("status = ?")
        params.append(status_filter)
    if args.scope:
        clauses.append("scope_type = ?")
        params.append(args.scope)
    if args.source:
        clauses.append("source = ?")
        params.append(args.source)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    rows = conn.execute(
        f"""
        SELECT id, status, scope_type, scope_topic_id, source, text,
               property, value, epistemic_status, pinned
        FROM facts
        {where}
        ORDER BY created_at DESC
        """,
        params,
    ).fetchall()

    if not rows:
        print("Nothing to review.")
        return

    print(f"Interactive review — {len(rows)} fact(s).")
    print("Commands: [k]eep  [a]ctivate  [p]in  [e]dit  [d]elete  [q]uit")
    print()

    for row in rows:
        scope = _format_scope(row)
        marker = "📌" if row["pinned"] else "  "
        print("─" * 66)
        print(f"{marker} [{row['id']}] ({row['status']}, {row['source']}, {scope})")
        print(f"    {row['text']}")
        if row["property"]:
            print(f"    property: {row['property']}   "
                  f"value: {row['value']}")
        print("─" * 66)

        while True:
            try:
                action = input("[k/a/p/e/d/q] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                print("Stopped.")
                return

            if action in ("", "k", "keep"):
                break

            if action in ("a", "activate"):
                _set_status(row["id"], "active")
                print("  ✅ Activated.")
                break

            if action in ("p", "pin"):
                _activate_and_pin(row["id"])
                print("  📌 Pinned and activated.")
                break

            if action in ("e", "edit"):
                try:
                    new_text = input("  New text (blank cancels): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if new_text:
                    _edit_text(row["id"], new_text)
                    print("  ✏️  Updated.")
                else:
                    print("  Canceled.")
                break

            if action in ("d", "delete"):
                if _confirm("  Delete this fact?", args.yes):
                    forget_fact(fact_id=row["id"])
                    print("  🗑️  Deleted.")
                else:
                    print("  Kept.")
                break

            if action in ("q", "quit"):
                print("Stopped.")
                return

            print("  Unknown command. Try k, a, p, e, d, or q.")


# ===== CLI =====

def main():
    p = argparse.ArgumentParser(
        prog="python -m memory.review",
        description="Browse, edit, delete, merge, and promote/demote facts.",
    )
    sub = p.add_subparsers(dest="cmd")

    # --- browse ---
    pb = sub.add_parser("browse", help="List facts")
    pb.add_argument("--all", action="store_true",
                    help="Include non-active facts")
    pb.add_argument("--status",
                    choices=["candidate", "active", "superseded", "deleted"],
                    help="Filter by a single status")
    pb.add_argument("--limit", type=int, default=50)
    pb.set_defaults(func=cmd_browse)

    # --- interactive ---
    pi = sub.add_parser("interactive",
                        help="Walk through facts one by one")
    pi.add_argument("--all", action="store_true",
                    help="Include non-active facts "
                         "(overrides --status default of 'candidate')")
    pi.add_argument("--status", default="candidate",
                    choices=["candidate", "active", "superseded", "deleted"])
    pi.add_argument("--scope",
                    choices=["global", "unscoped", "topic"])
    pi.add_argument("--source",
                    choices=["user_explicit", "user_asserted",
                             "inferred", "web"])
    pi.add_argument("-y", "--yes", action="store_true",
                    help="Skip the delete confirmation prompt.")
    pi.set_defaults(func=cmd_interactive)

    # --- edit ---
    pe = sub.add_parser("edit", help="Edit a fact's text")
    pe.add_argument("id", type=int)
    pe.set_defaults(func=cmd_edit)

    # --- delete ---
    pd = sub.add_parser("delete", help="Delete a fact (tombstone + soft delete)")
    pd.add_argument("id", type=int)
    pd.add_argument("-y", "--yes", action="store_true")
    pd.set_defaults(func=cmd_delete)

    # --- merge ---
    pm = sub.add_parser("merge",
                        help="Merge two facts, keeping the first")
    pm.add_argument("target_id", type=int,
                    help="The fact to keep")
    pm.add_argument("source_id", type=int,
                    help="The fact to merge in and tombstone")
    pm.add_argument("-y", "--yes", action="store_true")
    pm.set_defaults(func=cmd_merge)

    # --- promote ---
    pp = sub.add_parser("promote",
                        help="candidate → active")
    pp.add_argument("id", type=int)
    pp.add_argument("--pin", action="store_true",
                    help="Also pin the fact")
    pp.set_defaults(func=cmd_promote)

    # --- demote ---
    pr = sub.add_parser("demote",
                        help="active → candidate")
    pr.add_argument("id", type=int)
    pr.add_argument("--unpin", action="store_true",
                    help="Also clear the pinned flag")
    pr.set_defaults(func=cmd_demote)

    # --- argv preprocessing ---
    # 1. Support `--interactive` as a top-level shorthand for the
    #    `interactive` subcommand.
    # 2. Default to `browse` when no subcommand is given.
    KNOWN = {"browse", "interactive", "edit", "delete",
             "merge", "promote", "demote"}
    argv = sys.argv[1:]

    if "--interactive" in argv:
        argv = [a for a in argv if a != "--interactive"]
        argv = ["interactive"] + argv

    if not argv or argv[0].startswith("-") or argv[0] not in KNOWN:
        argv = ["browse"] + argv

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()