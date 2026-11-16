"""rem's seven MCP memory tools, and `rem mcp` (serve them over stdio).

The tools talk to the daemon through `RemClient` only. Facts an agent adds are stored as
assistant-generated and unverified; every write is recorded with actor "mcp".
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

from rem.client import DaemonUnavailable, RemClient, ensure_daemon
from rem.config import RemConfig
from rem.errors import MemoryNotFound, RemError
from rem.format import history_line, memory_line, search_line
from rem.mcp.protocol import Tool, ToolResult, serve
from rem.projects import project_root

INSTRUCTIONS = (
    "rem is the user's long-term memory shared across AI agents. Search it before asking the user about "
    "their preferences, environment, or project conventions. Save durable facts with add_memory; never "
    "save secrets."
)

SCOPES = ["global", "system", "project"]
STATUSES = ["verified", "unverified"]
NONE_FOUND = "No memories found."

_SCOPE = {"type": "string", "enum": SCOPES}
_ID = {"type": "string", "description": "The memory id."}


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    s = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        s["required"] = required
    return s


class _Args:
    """Checks a tool's arguments; every problem is a ValueError with a message the agent can act on."""

    def __init__(self, args: dict, allowed: set[str]):
        unknown = sorted(set(args) - allowed)
        if unknown:
            raise ValueError(f"unknown argument(s): {', '.join(unknown)}")
        self.args = args

    def str(self, key: str, *, required: bool = False, choices: list[str] | None = None,
            default: str | None = None) -> str | None:
        v = self.args.get(key)
        if v is None:
            if required:
                raise ValueError(f"{key} is required")
            return default
        if not isinstance(v, str):
            raise ValueError(f"{key} must be a string")
        if choices is not None and v not in choices:
            raise ValueError(f"{key} must be one of: {', '.join(choices)}")
        return v

    def int(self, key: str, *, lo: int, hi: int, default: int) -> int:
        v = self.args.get(key)
        if v is None:
            return default
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"{key} must be an integer")
        if not lo <= v <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}")
        return v


def _lines(rows: list[str]) -> str:
    return "\n".join(rows) if rows else NONE_FOUND


def build_tools(get_client: Callable[[], RemClient], cwd: Path) -> list[Tool]:
    def guarded(fn: Callable[[dict], str]) -> Callable[[dict], ToolResult]:
        def handler(args: dict) -> ToolResult:
            try:
                return ToolResult(fn(args))
            except MemoryNotFound as e:
                return ToolResult(f"No memory with id {e.memory_id}", is_error=True)
            except (ValueError, RemError) as e:  # daemon `invalid`, bad arguments, daemon down
                return ToolResult(str(e), is_error=True)
        return handler

    def search_memory(args: dict) -> str:
        a = _Args(args, {"query", "limit", "scope"})
        query = a.str("query", required=True)
        limit = a.int("limit", lo=1, hi=50, default=10)
        scope = a.str("scope", choices=SCOPES)
        return _lines([search_line(m) for m in get_client().search(query, cwd, limit, scope)])

    def list_memories(args: dict) -> str:
        a = _Args(args, {"scope", "status", "limit"})
        scope = a.str("scope", choices=SCOPES)
        status = a.str("status", choices=STATUSES)
        limit = a.int("limit", lo=1, hi=200, default=50)
        return _lines([memory_line(m) for m in get_client().list(scope, status)[:limit]])

    def add_memory(args: dict) -> str:
        a = _Args(args, {"text", "scope"})
        text = a.str("text", required=True)
        scope = a.str("scope", choices=SCOPES, default="global")
        if scope == "project" and not (project_root(cwd) / ".git").exists():
            # the server's cwd may be / or $HOME: never invent a project from it
            raise ValueError(f"project scope needs a git repository; rem mcp is running in {cwd}")
        m = get_client().add(text, scope, cwd, provenance="assistant_generated", status="unverified",
                             source="mcp", actor="mcp")
        return "Saved as unverified (the user can confirm it): " + memory_line(m)

    def confirm_memory(args: dict) -> str:
        id_ = _Args(args, {"id"}).str("id", required=True)
        return "Confirmed: " + memory_line(get_client().confirm(id_, actor="mcp"))

    def update_memory(args: dict) -> str:
        a = _Args(args, {"id", "text"})
        id_ = a.str("id", required=True)
        text = a.str("text", required=True)
        return "Updated: " + memory_line(get_client().update(id_, text, actor="mcp"))

    def delete_memory(args: dict) -> str:
        id_ = _Args(args, {"id"}).str("id", required=True)
        get_client().delete(id_, actor="mcp")
        return f"Deleted {id_}"

    def memory_history(args: dict) -> str:
        id_ = _Args(args, {"id"}).str("id", required=True)
        return "\n".join(history_line(h) for h in get_client().history(id_))

    return [
        Tool("search_memory",
             "Search the user's memories (preferences, environment, project conventions) by meaning. "
             "Memories from the current project rank highest.",
             _schema({
                 "query": {"type": "string", "description": "What to look for."},
                 "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                 "scope": _SCOPE,
             }, ["query"]),
             guarded(search_memory)),
        Tool("list_memories",
             "List stored memories, newest first, optionally filtered by scope and status.",
             _schema({
                 "scope": _SCOPE,
                 "status": {"type": "string", "enum": STATUSES},
                 "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
             }),
             guarded(list_memories)),
        Tool("add_memory",
             "Save a durable fact about the user, their machine, or this project. It is stored as unverified "
             "until the user confirms it.",
             _schema({
                 "text": {"type": "string", "description": "The fact, as one short self-contained sentence."},
                 "scope": {**_SCOPE, "default": "global",
                           "description": "global: about the user; system: about this machine; "
                                          "project: about the current git repository."},
             }, ["text"]),
             guarded(add_memory)),
        Tool("confirm_memory",
             "Mark a memory as confirmed true. Only call this when the user has explicitly confirmed the fact "
             "in this conversation; never to verify your own claims.",
             _schema({"id": _ID}, ["id"]),
             guarded(confirm_memory)),
        Tool("update_memory",
             "Change a memory's text, e.g. when the user corrects it.",
             _schema({"id": _ID, "text": {"type": "string", "description": "The new text."}}, ["id", "text"]),
             guarded(update_memory)),
        Tool("delete_memory",
             "Delete a memory that is wrong or that the user wants forgotten.",
             _schema({"id": _ID}, ["id"]),
             guarded(delete_memory)),
        Tool("memory_history",
             "Show a memory's change history (added, updated, confirmed, deleted, and by whom).",
             _schema({"id": _ID}, ["id"]),
             guarded(memory_history)),
    ]


class _LazyClient:
    """Connects (starting the daemon if needed) on first use and caches the client. A
    DaemonUnavailable, from connecting or from a call, clears the cache so the next call retries."""

    def __init__(self, cfg: RemConfig):
        self._cfg = cfg
        self._client: RemClient | None = None

    def __call__(self) -> _LazyClient:
        if self._client is None:
            self._client = ensure_daemon(self._cfg)
        return self

    def __getattr__(self, name: str):
        method = getattr(self._client, name)

        def call(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except DaemonUnavailable:
                self._client = None
                raise
        return call


def run_mcp(cfg: RemConfig) -> int:
    cwd = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
    try:
        return serve(build_tools(_LazyClient(cfg), cwd), sys.stdin, sys.stdout, instructions=INSTRUCTIONS)
    except BrokenPipeError:  # the client closed the pipe: a normal way to end
        try:  # keep the interpreter's final stdout flush from reporting the broken pipe
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except OSError:
            pass
        return 0
    except KeyboardInterrupt:
        return 0
