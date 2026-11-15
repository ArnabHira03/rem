"""rem command line: a thin client of the rem daemon (started automatically when needed)."""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

from rem.config import load_config
from rem.errors import RemError
from rem.models import Scope, Status

SCOPES = [s.value for s in Scope]
STATUSES = [s.value for s in Status]


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="machine-readable output")

    p = argparse.ArgumentParser(prog="rem", description="Local-first personal memory.")
    p.add_argument("--json", action="store_true", default=False, help="machine-readable output")
    sub = p.add_subparsers(dest="cmd", required=True)

    def cmd(name, help_):
        return sub.add_parser(name, help=help_, parents=[common])

    a = cmd("add", "store a memory you confirm as true")
    a.add_argument("text")
    a.add_argument("--scope", choices=SCOPES, default="global")

    s = cmd("search", "find relevant memories")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--scope", choices=SCOPES)

    ls = cmd("list", "list memories")
    ls.add_argument("--scope", choices=SCOPES)
    ls.add_argument("--status", choices=STATUSES)

    for name, help_ in [("get", "show one memory"), ("delete", "delete a memory"),
                        ("confirm", "mark a memory as confirmed by you"), ("history", "show a memory's history")]:
        cmd(name, help_).add_argument("id")

    u = cmd("update", "change a memory's text")
    u.add_argument("id")
    u.add_argument("text")

    cmd("retry-failed", "requeue extraction jobs that failed (e.g. after fixing the LLM config)")

    cmd("daemon", "run the rem daemon in the foreground")
    cmd("start", "start the rem daemon in the background")
    cmd("stop", "stop the rem daemon")
    cmd("status", "show daemon status")
    i = cmd("install-service", "install the systemd user service")
    i.add_argument("--enable", action="store_true", help="also enable and start it now")

    # listed for --help only: main() dispatches `rem hook ...` before argparse
    h = cmd("hook", "handle an agent hook event read from stdin (installed by `rem install`)")
    h.add_argument("agent", choices=["claude-code"])

    ins = cmd("install", "install rem's hooks into an agent's settings")
    ins.add_argument("agent", choices=["claude-code"])
    ins.add_argument("--project", action="store_true", help="use <git root>/.claude/settings.local.json")
    ins.add_argument("--uninstall", action="store_true", help="remove rem's hooks")
    ins.add_argument("--dry-run", action="store_true", help="print the resulting settings, write nothing")

    ev = cmd("eval", "score the configured LLM (runs in-process, no daemon)")
    ev.add_argument("what", choices=["extraction"])
    ev.add_argument("--dataset", type=Path, help="JSONL dataset (default: the one shipped with rem)")
    ev.add_argument("--limit", type=int, help="only run the first N cases")
    return p


def _line(m: dict) -> str:
    where = f"{m['scope']}:{m['project_name']}" if m.get("project_name") else m["scope"]
    return f"{m['id']}  [{where}] {m['status']}  {m['text']}"


def _emit(args, value) -> None:
    if args.json:
        print(json.dumps(value, indent=2))
        return
    if value is None:
        return
    for v in value if isinstance(value, list) else [value]:
        if "event" in v:
            print(f"{v['created_at']}  {v['event']:<11} {v['actor']:<9} {v['old_value']!r} → {v['new_value']!r}")
        elif "score" in v:
            print(f"{v['score']:.2f}  {_line(v)}")
        else:
            print(_line(v))


def _memory_command(args, cfg):
    from rem.client import ensure_daemon

    c = ensure_daemon(cfg)
    cwd = Path.cwd()
    match args.cmd:
        case "add":
            return c.add(args.text, args.scope, cwd)
        case "search":
            return c.search(args.query, cwd, args.limit, args.scope)
        case "list":
            return c.list(args.scope, args.status)
        case "get":
            return c.get(args.id)
        case "update":
            return c.update(args.id, args.text)
        case "delete":
            return c.delete(args.id)
        case "confirm":
            return c.confirm(args.id)
        case "history":
            return c.history(args.id)


def _eval(args, cfg) -> int:
    from rem.embed.factory import create_embedder
    from rem.errors import LLMError, RemoteLLMNotAllowed
    from rem.evals.extraction import DEFAULT_DATASET, load_cases, run_eval
    from rem.llm.factory import create_llm

    try:
        llm = create_llm(cfg.llm)
    except (RemoteLLMNotAllowed, LLMError) as e:  # remote not allowed, or no API key
        print(f"rem: {e}", file=sys.stderr)
        return 3
    if llm is None:
        print("rem: no [llm] configured; set [llm] provider/model", file=sys.stderr)
        return 3
    try:
        cases = load_cases(args.dataset or DEFAULT_DATASET)
    except (OSError, ValueError) as e:  # JSONDecodeError is a ValueError
        print(f"rem: cannot read dataset: {e}", file=sys.stderr)
        return 2
    if args.limit is not None:
        cases = cases[: args.limit]
    report = run_eval(llm, create_embedder(cfg.embedder), cases, cfg)
    if args.json:
        print(json.dumps({"llm": llm.name, **report.to_dict()}, indent=2))
        return 0
    print(f"extraction eval: {llm.name}, {report.cases} cases")
    for label, value in [
        ("precision", report.precision), ("recall", report.recall), ("scope accuracy", report.scope_acc),
        ("origin accuracy", report.origin_acc), ("JSON validity", report.json_valid_rate),
        ("dedup accuracy", report.dedup_acc),
    ]:
        print(f"  {label:<16}" + (f"{value:6.2f}" if value is not None else f"{'n/a':>6}"))
    print(f"  {'latency p50':<16}{report.latency_p50:5.2f}s")
    print(f"  {'latency p95':<16}{report.latency_p95:5.2f}s")
    if report.failures:
        print(f"{len(report.failures)} case(s) with problems:")
        for f in report.failures:
            print(f"  {f['id']}: " + "; ".join(f"{k}={v}" for k, v in f.items() if k != "id"))
    return 0


def _install(args) -> int:
    from rem import install_claude_code as ic

    path = ic.settings_path(args.project, Path.cwd())
    if args.uninstall:
        print(ic.uninstall(path, dry_run=args.dry_run))
    else:
        print(ic.install(path, ic.hook_command(), dry_run=args.dry_run))
    return 0


def _status(args, cfg) -> int:
    from rem.client import DaemonUnavailable, RemClient
    from rem.install_claude_code import installed_where

    from rem.spool import spool_count

    hooks = ",".join(installed_where()) or "none"  # local checks: work with the daemon down
    spooled = spool_count(cfg.data_dir)
    try:
        st = RemClient(cfg.socket_path, timeout=5).status()
    except DaemonUnavailable:
        print(f"claude_code_hooks: {hooks}")
        print(f"spooled: {spooled}")
        if spooled:
            print(f"rem daemon is not running; spooled turns will be ingested when it starts (see {cfg.log_path})")
        else:
            print("rem daemon is not running")
        return 1
    if args.json:
        print(json.dumps({**st, "claude_code_hooks": hooks, "spooled": spooled}, indent=2))
    else:
        print(f"claude_code_hooks: {hooks}")
        print(f"spooled: {spooled}")
        for k, v in st.items():
            print(f"{k}: {v}")
    return 0


def _stop(cfg) -> int:
    from rem.daemon.lock import InstanceLock

    pid = InstanceLock.holder_pid(cfg.lock_path)
    if pid is None:
        print("rem daemon is not running")
        return 0
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while InstanceLock.holder_pid(cfg.lock_path) is not None:
        if time.monotonic() > deadline:
            print(f"rem: daemon (pid {pid}) did not stop within 10s", file=sys.stderr)
            return 1
        time.sleep(0.05)
    print("rem daemon stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    args_in = sys.argv[1:] if argv is None else argv
    if args_in[:1] == ["hook"]:  # before argparse/config/daemon: a hook must never fail, print or block
        from rem.adapters import claude_code

        return claude_code.main() if args_in[1:] == ["claude-code"] else claude_code.bad_args()
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    if args.cmd == "install":  # edits Claude Code settings only: works even with a broken rem config
        try:
            return _install(args)
        except RemError as e:
            print(f"rem: {e}", file=sys.stderr)
            return 1
    try:
        cfg = load_config()
    except ValueError as e:
        print(f"rem: {e}", file=sys.stderr)
        return 3 if args.cmd == "daemon" else 2

    if args.cmd == "daemon":
        from rem.daemon.server import run_daemon

        return run_daemon(cfg)
    try:
        match args.cmd:
            case "eval":
                return _eval(args, cfg)
            case "status":
                return _status(args, cfg)
            case "stop":
                return _stop(cfg)
            case "install-service":
                from rem.systemd import ENABLE_HINT, install_unit

                print(f"wrote {install_unit(enable=args.enable)}")
                print("rem.service enabled and started" if args.enable else f"next: {ENABLE_HINT}")
                return 0
            case "retry-failed":
                from rem.client import ensure_daemon

                n = ensure_daemon(cfg).retry_failed()
                print(json.dumps({"requeued": n}) if args.json else f"requeued {n} job(s)")
                return 0
            case "start":
                from rem.client import ensure_daemon

                pid = ensure_daemon(cfg).status()["pid"]
                print(f"rem daemon running (pid {pid})")
                return 0
        _emit(args, _memory_command(args, cfg))
    except RemError as e:
        print(f"rem: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"rem: {e}", file=sys.stderr)
        return 2
    return 0
