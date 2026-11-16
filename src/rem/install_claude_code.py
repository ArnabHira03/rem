"""Write rem's hook entries into a Claude Code settings file (idempotent, atomic, reversible)."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from rem.errors import RemError
from rem.projects import project_root

MARKER = " -m rem hook claude-code"
EVENTS = {"SessionStart": 10, "UserPromptSubmit": 10, "Stop": 10, "SessionEnd": 5}


def settings_path(project: bool, cwd: Path) -> Path:
    if project:
        return project_root(cwd) / ".claude" / "settings.local.json"
    return Path((os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude")).expanduser() / "settings.json"


def hook_command() -> str:
    return f"{sys.executable}{MARKER}"


def _is_rem(handler) -> bool:
    return isinstance(handler, dict) and MARKER in str(handler.get("command", ""))


def removed(settings: dict) -> dict:
    """A copy of settings without rem's handlers; groups, events and `hooks` left empty are dropped."""
    out = dict(settings)
    hooks = out.get("hooks")
    if not isinstance(hooks, dict):
        return out
    new_hooks = {}
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            new_hooks[event] = groups
            continue
        kept = []
        for group in groups:
            inner = group.get("hooks") if isinstance(group, dict) else None
            if not isinstance(inner, list) or not any(_is_rem(h) for h in inner):
                kept.append(group)
                continue
            rest = [h for h in inner if not _is_rem(h)]
            if rest:
                kept.append({**group, "hooks": rest})
        if kept:
            new_hooks[event] = kept
    if new_hooks:
        out["hooks"] = new_hooks
    else:
        del out["hooks"]
    return out


def merged(settings: dict, command: str) -> dict:
    out = removed(settings)
    hooks = out.get("hooks")
    hooks = dict(hooks) if isinstance(hooks, dict) else {}
    for event, timeout in EVENTS.items():
        group = {"matcher": "*", "hooks": [{"type": "command", "command": command, "timeout": timeout}]}
        hooks[event] = [*hooks.get(event, []), group]
    out["hooks"] = hooks
    return out


def _read(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        raise RemError(f"{path} is not valid JSON; not modified") from None
    if not isinstance(data, dict):
        raise RemError(f"{path} is not a JSON object; not modified")
    return data


def _backup_path(path: Path) -> Path:
    return path.with_name(path.name + ".rem-backup")


def _write(path: Path, data: dict) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    mode = None
    if path.exists():
        mode = path.stat().st_mode & 0o777
        backup = _backup_path(path)
        if not backup.exists():  # the first backup is the user's pre-rem settings: keep it
            # created with the original's mode from the start: no window where it is umask-readable
            fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
            try:
                with os.fdopen(fd, "wb") as f:
                    os.fchmod(f.fileno(), mode)  # os.open's mode is still masked by the umask
                    f.write(path.read_bytes())
            except BaseException:
                backup.unlink(missing_ok=True)
                raise
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _apply(path: Path, compute, dry_run: bool, done: str, noop: str, check=False) -> str:
    path = path.resolve()  # a symlinked settings file: write the real target, keep the link
    original = _read(path)
    if check:
        hooks = original.get("hooks")
        if hooks is not None and not isinstance(hooks, dict):
            raise RemError(f"{path}: hooks is not an object; not modified")
        for event in EVENTS:
            if event in (hooks or {}) and not isinstance(hooks[event], list):
                raise RemError(f"{path}: hooks.{event} is not a list; not modified")
    result = compute(original)
    if dry_run:
        return json.dumps(result, indent=2, ensure_ascii=False)
    if result == original:
        return noop
    _write(path, result)
    return done


def install(path: Path, command: str, *, dry_run: bool = False) -> str:
    return _apply(path, lambda s: merged(s, command), dry_run,
                  f"installed rem hooks in {path}", f"rem hooks already installed in {path}", check=True)


def uninstall(path: Path, *, dry_run: bool = False) -> str:
    msg = _apply(path, removed, dry_run, f"removed rem hooks from {path}", f"nothing to remove in {path}")
    backup = _backup_path(path.resolve())
    if not dry_run and backup.exists():
        msg += f"\nyour settings from before rem was installed are kept in {backup}"
    return msg


def _has_marker(path: Path) -> bool:
    try:
        hooks = json.loads(path.read_text(encoding="utf-8")).get("hooks")
    except (OSError, ValueError, AttributeError):
        return False
    if not isinstance(hooks, dict):
        return False
    return any(
        _is_rem(h)
        for groups in hooks.values() if isinstance(groups, list)
        for g in groups if isinstance(g, dict) and isinstance(g.get("hooks"), list)
        for h in g["hooks"]
    )


def installed_where(cwd: Path | None = None) -> list[str]:
    where = ["user"] if _has_marker(settings_path(False, Path("/"))) else []
    try:
        if _has_marker(settings_path(True, cwd or Path.cwd())):
            where.append("project")
    except OSError:  # e.g. the current directory was deleted: the project is unknown
        pass
    return where


MCP_NAME = "rem"
MCP_TIMEOUT = 30


def mcp_command() -> list[str]:
    return [sys.executable, "-m", "rem", "mcp"]


def _remove_argv(claude: str, scope: str) -> list[str]:
    return [claude, "mcp", "remove", "--scope", scope, MCP_NAME]


def _add_argv(claude: str, scope: str) -> list[str]:
    return [claude, "mcp", "add", "--scope", scope, MCP_NAME, "--", *mcp_command()]


def _line(argv: list[str]) -> str:
    return " ".join(argv)


def register_mcp(scope: Literal["user", "local"], cwd: Path, *, dry_run=False,
                 run=subprocess.run, which=shutil.which) -> str:
    claude = which("claude")
    if claude is None:
        return "claude CLI not found; to add rem's MCP server run: " + _line(_add_argv("claude", scope))
    if dry_run:
        return "would run:\n" + _line(_remove_argv(claude, scope)) + "\n" + _line(_add_argv(claude, scope))
    try:
        run(_remove_argv(claude, scope), cwd=cwd, capture_output=True, text=True, timeout=MCP_TIMEOUT)  # may not exist yet
        r = run(_add_argv(claude, scope), cwd=cwd, capture_output=True, text=True, timeout=MCP_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as e:
        return f"could not add rem's MCP server: {e}"
    if r.returncode != 0:
        return f"could not add rem's MCP server: {_first_line(r.stderr)}"
    return f"registered rem MCP server ({scope} scope)"


def unregister_mcp(scope: Literal["user", "local"], cwd: Path, *, dry_run=False,
                   run=subprocess.run, which=shutil.which) -> str:
    claude = which("claude")
    if claude is None:
        return "claude CLI not found; to remove rem's MCP server run: " + _line(_remove_argv("claude", scope))
    if dry_run:
        return "would run: " + _line(_remove_argv(claude, scope))
    try:
        r = run(_remove_argv(claude, scope), cwd=cwd, capture_output=True, text=True, timeout=MCP_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as e:
        return f"could not remove rem's MCP server: {e}"
    if r.returncode != 0:
        return f"could not remove rem's MCP server: {_first_line(r.stderr)}"
    return f"removed rem's MCP server ({scope} scope)"


def _first_line(text: str) -> str:
    lines = (text or "").strip().splitlines()
    return lines[0] if lines else "claude exited with an error"
