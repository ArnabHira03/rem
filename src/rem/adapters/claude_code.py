"""`rem hook claude-code`: capture Claude Code turns and inject verified context at SessionStart.

Runs inside the user's Claude Code session, so it must never break, block or pollute it:
every path exits 0, nothing goes to stderr, and only SessionStart writes to stdout.
Turns come from the hook payloads (`prompt`, `last_assistant_message`), never the transcript.
Keep imports light: no rem.memory, embedder, vector store, qdrant or fastembed here.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from rem.client import DaemonUnavailable, RemClient, start_daemon_background
from rem.config import RemConfig, ensure_private_dir, load_config
from rem.redact import redact
from rem.spool import write_spool

SOURCE = "claude-code"
_EVENTS = {"SessionStart", "UserPromptSubmit", "Stop", "SessionEnd"}


def _excluded(cwd: str | None, cfg: RemConfig) -> bool:
    if not cwd:
        return False
    here = Path(cwd).resolve()
    return any(here.is_relative_to(Path(p).expanduser().resolve()) for p in cfg.claude_code.exclude)


def _timed_out(e: DaemonUnavailable) -> bool:
    """The daemon accepted the request but was slow: don't start a second one."""
    return isinstance(e.__cause__, httpx.TimeoutException)


def _turn(role: str, text: str, prompt_id) -> dict:
    content = redact(text)
    turn = {"role": role, "content": content}
    if prompt_id:
        # One prompt can have several Stops (e.g. a Stop hook makes Claude continue): the same reply
        # re-sent keeps its id (idempotent), a different continuation becomes a new turn.
        suffix = f":{hashlib.sha256(content.encode()).hexdigest()[:12]}" if role == "assistant" else ""
        turn["message_id"] = f"cc:{prompt_id}:{role}{suffix}"
    return turn


def handle_hook(payload: dict, cfg: RemConfig, client_factory=RemClient) -> str:
    """Handle one hook payload; returns the text for stdout (empty except for SessionStart)."""
    event = payload.get("hook_event_name")
    if not isinstance(event, str) or event not in _EVENTS:
        return ""
    cwd = payload.get("cwd") or None
    if _excluded(cwd, cfg):
        return ""
    cc = cfg.claude_code

    if event == "SessionStart":
        client = client_factory(cfg.socket_path, timeout=cc.hook_budget_seconds)
        try:
            return client.context(cwd, cc.context_max_chars) or ""
        except DaemonUnavailable as e:
            if not _timed_out(e):
                start_daemon_background(cfg)
            return ""

    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return ""  # nothing to file the turn under; the daemon would only reject it
    prompt_id = payload.get("prompt_id")
    match event:
        case "UserPromptSubmit":
            prompt = payload.get("prompt") or ""
            if not prompt.strip():
                return ""
            turns, enqueue = [_turn("user", prompt, prompt_id)], False
        case "Stop":
            reply = payload.get("last_assistant_message") or ""
            turns, enqueue = ([_turn("assistant", reply, prompt_id)] if reply.strip() else []), True
        case _:  # SessionEnd: flush any held prompt
            turns, enqueue = [], True

    body = {"source": SOURCE, "session_id": session_id, "cwd": cwd, "turns": turns, "enqueue": enqueue}
    client = client_factory(cfg.socket_path, timeout=cc.hook_budget_seconds)
    try:
        client.ingest(**body)
    except ValueError:
        raise  # the daemon rejected the body: spooling it would only quarantine it later
    except Exception as e:
        # Keep the turn for a later drain. A timeout is ambiguous (the daemon may have stored it);
        # ingest is idempotent per message_id.
        try:
            write_spool(cfg.data_dir, body)
        finally:  # a failed spool must not also keep the daemon down
            if isinstance(e, DaemonUnavailable) and not _timed_out(e):
                start_daemon_background(cfg)
    return ""


HOOK_LOG_MAX_BYTES = 1_000_000


def _log_line(cfg: RemConfig, event, what: str) -> None:
    """Append one line to hook.log: never prompt, reply or argv text, only fixed strings and class names."""
    try:
        name = event if isinstance(event, str) and event in _EVENTS else "unknown"
        ensure_private_dir(cfg.data_dir)
        line = f"{datetime.now(timezone.utc).isoformat()} {name} {what}\n"
        path = cfg.data_dir / "hook.log"
        try:
            if path.stat().st_size >= HOOK_LOG_MAX_BYTES:
                os.replace(path, cfg.data_dir / "hook.log.1")
        except OSError:
            pass
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a") as f:
            f.write(line)
    except Exception:
        pass


def _log_error(cfg: RemConfig, event, exc: BaseException) -> None:
    _log_line(cfg, event, f"{type(exc).__name__}: hook failed")


def _silence_stdout() -> None:
    """After a failed write (e.g. Claude Code closed the pipe), point fd 1 at /dev/null so the
    interpreter's shutdown flush can't print "Exception ignored ... BrokenPipeError" to stderr."""
    try:
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    except Exception:
        pass


def bad_args() -> int:
    """`rem hook` with arguments this version doesn't know: log a fixed line and exit 0, silently."""
    try:
        _log_line(load_config(), None, "unknown hook args")
    except BaseException:
        pass
    return 0


def main(stdin=None, stdout=None) -> int:
    """Entry point for `rem hook claude-code`. Always returns 0."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    try:
        cfg = load_config()
    except BaseException:
        return 0  # no reliable data dir to log into
    event = None
    try:
        payload = json.loads(stdin.read())
        if not isinstance(payload, dict):
            raise ValueError("hook payload is not a JSON object")
        event = payload.get("hook_event_name")
        text = handle_hook(payload, cfg, client_factory=RemClient)
        if text:
            try:
                stdout.write(text)
                stdout.flush()
            except Exception:
                _silence_stdout()
                raise
    except BaseException as e:  # KeyboardInterrupt and SystemExit too: the hook always exits 0
        _log_error(cfg, event, e)
    return 0
