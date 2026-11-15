"""rem command line. Runs the Memory façade in-process (the daemon arrives in M2)."""

import argparse
import json
import sys
from pathlib import Path

from rem.config import load_config
from rem.errors import MemoryNotFound, RemError
from rem.models import HistoryRow, MemoryRecord, Provenance, Scope, SearchResult, Status

SCOPES = [s.value for s in Scope]
STATUSES = [s.value for s in Status]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rem", description="Local-first personal memory.")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="store a memory you confirm as true")
    a.add_argument("text")
    a.add_argument("--scope", choices=SCOPES, default="global")

    s = sub.add_parser("search", help="find relevant memories")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--scope", choices=SCOPES)

    ls = sub.add_parser("list", help="list memories")
    ls.add_argument("--scope", choices=SCOPES)
    ls.add_argument("--status", choices=STATUSES)

    for name, help_ in [("get", "show one memory"), ("delete", "delete a memory"),
                        ("confirm", "mark a memory as confirmed by you"), ("history", "show a memory's history")]:
        sub.add_parser(name, help=help_).add_argument("id")

    u = sub.add_parser("update", help="change a memory's text")
    u.add_argument("id")
    u.add_argument("text")
    return p


def _line(r: MemoryRecord) -> str:
    where = f"{r.scope}:{r.project_name}" if r.project_name else str(r.scope)
    return f"{r.id}  [{where}] {r.status}  {r.text}"


def _emit(args, value) -> None:
    if args.json:
        print(json.dumps(_jsonable(value), indent=2))
        return
    if value is None:
        return
    items = value if isinstance(value, list) else [value]
    for v in items:
        if isinstance(v, SearchResult):
            print(f"{v.score:.2f}  {_line(v.record)}")
        elif isinstance(v, HistoryRow):
            change = f"{v.old_value!r} → {v.new_value!r}"
            print(f"{v.created_at}  {v.event:<11} {v.actor:<9} {change}")
        else:
            print(_line(v))


def _jsonable(value):
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, SearchResult):
        return {**value.record.to_payload(), "score": value.score, "semantic": value.semantic}
    if isinstance(value, MemoryRecord):
        return value.to_payload()
    if isinstance(value, HistoryRow):
        return value.model_dump(mode="json")
    return value


def _run(args) -> object:
    from rem.memory import Memory  # keeps `rem --help` fast

    mem = Memory(load_config())
    try:
        match args.cmd:
            case "add":
                return mem.add(args.text, scope=Scope(args.scope), cwd=Path.cwd(),
                               provenance=Provenance.USER_CONFIRMED, status=Status.VERIFIED,
                               source="cli", actor="cli")
            case "search":
                return mem.search(args.query, cwd=Path.cwd(), limit=args.limit,
                                  scope=Scope(args.scope) if args.scope else None)
            case "list":
                return mem.list(scope=Scope(args.scope) if args.scope else None,
                                status=Status(args.status) if args.status else None)
            case "get":
                record = mem.get(args.id)
                if record is None:
                    raise MemoryNotFound(args.id)
                return record
            case "update":
                return mem.update(args.id, args.text, actor="cli")
            case "delete":
                mem.delete(args.id, actor="cli")
                return None
            case "confirm":
                return mem.confirm(args.id, actor="cli")
            case "history":
                return mem.history(args.id)
    finally:
        mem.close()


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    try:
        _emit(args, _run(args))
    except RemError as e:
        print(f"rem: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"rem: {e}", file=sys.stderr)
        return 2
    return 0
