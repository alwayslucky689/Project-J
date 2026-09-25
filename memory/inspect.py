# memory/inspect.py - CLI for inspecting and editing memory
#
# Usage:
#   python -m memory.inspect list          # recent active facts
#   python -m memory.inspect list --all    # include superseded/deleted
#   python -m memory.inspect show <id>     # full record for one fact
#   python -m memory.inspect delete <id>   # soft delete + tombstone
#   python -m memory.inspect stats         # counts by status/scope/source
#   python -m memory.inspect tombstones    # what's been forgotten
#   python -m memory.inspect migrate       # apply pending migrations

import argparse
import sys

from memory.store import get_conn, write_transaction, now_iso


def cmd_list(args):
    conn = get_conn()
    where = "" if args.all else "WHERE status='active'"
    rows = conn.execute(
        f"""
        SELECT id, status, scope_type, scope_topic_id, source, subject,
        text, created_at, pinned
        {where}
        ORDER BY pinned DESC, created_at DESC
        LIMIT ?
        """,
        (args.limit,),
    ).fetchall()
    if not rows:
        print("(no facts)")
        return
    for r in rows:
        marker = "📌" if r["pinned"] else "  "
        scope_str = r["scope_type"]
        if r["scope_topic_id"]:
            scope_str = f"{scope_str}:{r['scope_topic_id']}"
        meta = f"{r['status']}/{scope_str}/{r['source']}"
        print(f"{marker} {tag:>6} {meta:<32} {r['text']}")

def cmd_scope(args):
    """Move a fact to a new scope: 'global', 'unscoped', or 'topic:<id>'."""
    from memory.store import get_conn, write_transaction, now_iso
    spec = args.scope
    if spec == "global":
        scope_type, scope_topic_id = "global", None
    elif spec == "unscoped":
        scope_type, scope_topic_id = "unscoped", None
    elif spec.startswith("topic:"):
        try:
            scope_topic_id = int(spec.split(":", 1)[1])
        except (ValueError, IndexError):
            print(f"Bad topic spec: {spec}")
            return
        scope_type = "topic"
    else:
        print("Scope must be 'global', 'unscoped', or 'topic:<id>'")
        return

    with write_transaction() as conn:
        n = conn.execute(
            "UPDATE facts SET scope_type=?, scope_topic_id=?, updated_at=? "
            "WHERE id=?",
            (scope_type, scope_topic_id, now_iso(), args.id),
        ).rowcount
    if n:
        print(f"✅ Fact {args.id} → {spec}")
    else:
        print(f"No fact with id {args.id}")


# In main():
    psc = sub.add_parser("scope", help="Change a fact's scope")
    psc.add_argument("id", type=int)
    psc.add_argument("scope", help="global | unscoped | topic:<id>")
    psc.set_defaults(func=cmd_scope)
    
def cmd_show(args):
    conn = get_conn()
    row = conn.execute("SELECT * FROM facts WHERE id=?", (args.id,)).fetchone()
    if not row:
        print(f"No fact with id {args.id}")
        return
    for key in row.keys():
        print(f"  {key:<24} {row[key]}")
def cmd_pin(args):
    from memory.fact_memory import pin_fact
    if args.id is not None:
        print(pin_fact(fact_id=args.id))
    elif args.search:
        print(pin_fact(search=args.search))
    else:
        print("Specify an id or --search")


def cmd_unpin(args):
    from memory.fact_memory import unpin_fact
    if args.id is not None:
        print(unpin_fact(fact_id=args.id))
    elif args.search:
        print(unpin_fact(search=args.search))
    else:
        print("Specify an id or --search")



    

def cmd_delete(args):
    from memory.fact_memory import forget_fact
    print(forget_fact(fact_id=args.id))


def cmd_stats(args):
    conn = get_conn()
    print("By status:")
    for row in conn.execute(
        "SELECT status, COUNT(*) c FROM facts GROUP BY status ORDER BY c DESC"
    ):
        print(f"  {row['status']:<12} {row['c']}")
    print("\nBy scope_type:")
    for row in conn.execute(
        "SELECT scope_type, COUNT(*) c FROM facts GROUP BY scope_type ORDER BY c DESC"
    ):
        print(f"  {row['scope_type']:<12} {row['c']}")
    print("\nBy source:")
    for row in conn.execute(
        "SELECT source, COUNT(*) c FROM facts GROUP BY source ORDER BY c DESC"
    ):
        print(f"  {row['source']:<16} {row['c']}")
    total_t = conn.execute("SELECT COUNT(*) c FROM turns").fetchone()["c"]
    total_s = conn.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"]
    total_tomb = conn.execute("SELECT COUNT(*) c FROM tombstones").fetchone()["c"]
    print(f"\nTurns:      {total_t}")
    print(f"Sessions:   {total_s}")
    print(f"Tombstones: {total_tomb}")


def cmd_tombstones(args):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, subject, property, scope_type, deleted_at, reason "
        "FROM tombstones ORDER BY deleted_at DESC LIMIT ?",
        (args.limit,),
    ).fetchall()
    if not rows:
        print("(no tombstones)")
        return
    for r in rows:
        print(f"[{r['id']}] {r['deleted_at']}  {r['subject']}/{r['property']}  {r['reason']}")


def cmd_migrate(args):
    from memory.store import _apply_migrations, get_conn
    _apply_migrations(get_conn())
    print("Migrations applied.")


def main():
    p = argparse.ArgumentParser(prog="python -m memory.inspect")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list")
    pl.add_argument("--all", action="store_true")
    pl.add_argument("--limit", type=int, default=30)
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("show")
    ps.add_argument("id", type=int)
    ps.set_defaults(func=cmd_show)

    pd = sub.add_parser("delete")
    pd.add_argument("id", type=int)
    pd.set_defaults(func=cmd_delete)

    sub.add_parser("stats").set_defaults(func=cmd_stats)

    pt = sub.add_parser("tombstones")
    pt.add_argument("--limit", type=int, default=20)
    pt.set_defaults(func=cmd_tombstones)

    sub.add_parser("migrate").set_defaults(func=cmd_migrate)

    pp = sub.add_parser("pin", help="Pin a fact to always appear in prompts")
    pp.add_argument("id", nargs="?", type=int)
    pp.add_argument("--search", type=str)
    pp.set_defaults(func=cmd_pin)

    pu = sub.add_parser("unpin", help="Unpin a previously pinned fact")
    pu.add_argument("id", nargs="?", type=int)
    pu.add_argument("--search", type=str)
    pu.set_defaults(func=cmd_unpin)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()