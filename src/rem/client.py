"""Talk to the rem daemon over its Unix socket, starting it on demand."""

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from rem.config import RemConfig, ensure_private_dir
from rem.errors import MemoryNotFound, RemError


class DaemonUnavailable(RemError):
    """The daemon is not running or could not be reached."""


class RemClient:
    def __init__(self, socket_path: Path, timeout: float = 30):
        self.socket_path = socket_path
        self._http = httpx.Client(
            transport=httpx.HTTPTransport(uds=str(socket_path)), base_url="http://rem", timeout=timeout
        )

    def _call(self, method: str, url: str, *, ref: str | None = None, **kw):
        try:
            r = self._http.request(method, url, **kw)
        except httpx.TransportError as e:
            raise DaemonUnavailable(f"rem daemon is not reachable at {self.socket_path}") from e
        if r.status_code == 204:
            return None
        try:
            body = r.json()
        except ValueError:
            raise RemError(f"rem daemon error (HTTP {r.status_code}); see daemon.log") from None
        if r.status_code >= 400:
            code, message = body.get("error"), body.get("message", r.text)
            if code == "not_found":
                raise MemoryNotFound(ref or "?")
            if code == "invalid":
                raise ValueError(message)
            raise RemError(message)
        return body

    def status(self) -> dict:
        return self._call("GET", "/status")

    def add(self, text, scope="global", cwd=None, provenance="user_confirmed", status="verified",
            source="cli", actor="cli") -> dict:
        return self._call("POST", "/memories", json={
            "text": text, "scope": scope, "cwd": str(cwd) if cwd else None, "provenance": provenance,
            "status": status, "source": source, "actor": actor,
        })

    def search(self, query, cwd=None, limit=10, scope=None) -> list[dict]:
        return self._call("POST", "/search", json={
            "query": query, "cwd": str(cwd) if cwd else None, "limit": limit, "scope": scope,
        })

    def context(self, cwd, max_chars=1600) -> str:
        return self._call("POST", "/context", json={"cwd": str(cwd) if cwd else None, "max_chars": max_chars})["text"]

    def list(self, scope=None, status=None) -> list[dict]:
        params = {k: v for k, v in {"scope": scope, "status": status}.items() if v}
        return self._call("GET", "/memories", params=params)

    def get(self, id) -> dict:
        return self._call("GET", f"/memories/{id}", ref=id)

    def update(self, id, text, actor="cli") -> dict:
        return self._call("PATCH", f"/memories/{id}", ref=id, json={"text": text, "actor": actor})

    def delete(self, id, actor="cli") -> None:
        return self._call("DELETE", f"/memories/{id}", ref=id, params={"actor": actor})

    def confirm(self, id, actor="cli") -> dict:
        return self._call("POST", f"/memories/{id}/confirm", ref=id, json={"actor": actor})

    def history(self, id) -> list[dict]:
        return self._call("GET", f"/memories/{id}/history", ref=id)

    def ingest(self, source, session_id, cwd, turns: list[dict], enqueue: bool = True) -> dict:
        return self._call("POST", "/ingest", json={
            "source": source, "session_id": session_id, "cwd": str(cwd) if cwd else None, "turns": turns, "enqueue": enqueue,
        })

    def retry_failed(self) -> int:
        return self._call("POST", "/jobs/retry")["requeued"]


def _log_tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except FileNotFoundError:
        return "(no daemon log)"


def _spawn_daemon(cfg: RemConfig) -> subprocess.Popen:
    ensure_private_dir(cfg.data_dir)
    # daemon.log carries job-failure text: create it 0600 here, before the daemon sets its umask
    fd = os.open(cfg.log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as log, open(os.devnull) as devnull:
        return subprocess.Popen(
            [sys.executable, "-m", "rem", "daemon"],
            stdin=devnull, stdout=log, stderr=log, start_new_session=True, cwd="/",
        )


START_BACKOFF_SECONDS = 60


def start_daemon_background(cfg: RemConfig) -> bool:
    """Spawn a detached `rem daemon` and return immediately (used by hooks).

    At most one spawn per START_BACKOFF_SECONDS (stamp file `daemon-start.stamp`): a daemon that
    can't start must not be respawned by every hook. Returns whether a daemon was spawned.
    """
    stamp = cfg.data_dir / "daemon-start.stamp"
    try:
        if 0 <= time.time() - stamp.stat().st_mtime < START_BACKOFF_SECONDS:  # a future stamp (clock stepped back) doesn't block
            return False
    except FileNotFoundError:
        pass
    ensure_private_dir(cfg.data_dir)
    os.close(os.open(stamp, os.O_WRONLY | os.O_CREAT, 0o600))
    os.utime(stamp)
    _spawn_daemon(cfg)
    return True


def ensure_daemon(cfg: RemConfig) -> RemClient:
    """Return a client for a running daemon, starting one in the background if needed.
    Not subject to the hook's start backoff: an interactive user wants the real error."""
    client = RemClient(cfg.socket_path)
    try:
        client.status()
        return client
    except DaemonUnavailable:
        pass

    proc = _spawn_daemon(cfg)
    deadline = time.monotonic() + cfg.daemon.start_timeout_seconds
    while time.monotonic() < deadline:
        try:
            client.status()
            return client
        except DaemonUnavailable:
            pass
        code = proc.poll()
        if code not in (None, 2):  # 2 = another starter won the race; keep waiting for it
            raise DaemonUnavailable("rem daemon failed to start:\n" + _log_tail(cfg.log_path))
        time.sleep(0.1)
    if proc.poll() is None:
        raise DaemonUnavailable(
            "rem daemon is still starting (the first run downloads the embedding model); "
            f"try again shortly, or watch {cfg.log_path}"
        )
    raise DaemonUnavailable("rem daemon failed to start:\n" + _log_tail(cfg.log_path))
