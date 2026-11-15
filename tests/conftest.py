import pytest

from rem.config import EmbedderConfig, RemConfig, VectorStoreConfig
from rem.models import Provenance, Scope, Status


@pytest.fixture(autouse=True)
def _never_touch_real_rem(tmp_path_factory, monkeypatch):
    """Default every test (and the subprocesses it inherits os.environ into) to a throwaway store,
    so nothing can reach ~/.local/share/rem, ~/.config/rem or the real daemon's socket.
    Fixtures and tests that need other values set them on top of these."""
    import os

    home = tmp_path_factory.mktemp("rem-home")
    monkeypatch.setenv("REM_HOME", str(home))
    monkeypatch.setenv("REM_CONFIG", str(home / "no-config.toml"))
    monkeypatch.setenv("REM_SOCKET", f"/tmp/rem-test-{os.getpid()}-none.sock")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / "claude"))  # never the real ~/.claude


@pytest.fixture
def rem_home(tmp_path, monkeypatch):
    monkeypatch.setenv("REM_HOME", str(tmp_path))
    monkeypatch.delenv("REM_CONFIG", raising=False)
    return tmp_path


@pytest.fixture
def cfg(tmp_path):
    """In-memory vector store + offline hashing embedder, SQLite under tmp_path."""
    return RemConfig(
        data_dir=tmp_path,
        vector_store=VectorStoreConfig(provider="memory"),
        embedder=EmbedderConfig(provider="hashing", model="hash-64"),
    )


@pytest.fixture
def mem(cfg):
    from rem.memory import Memory

    m = Memory(cfg)
    yield m
    m.close()


def add(mem, text="Prefers uv over pip", **kw):
    """Add a user-confirmed memory (global scope unless given)."""
    kw.setdefault("scope", Scope.GLOBAL)
    return mem.add(text, provenance=Provenance.USER_CONFIRMED, status=Status.VERIFIED, source="cli", actor="cli", **kw)


@pytest.fixture
def short_dir():
    """A short /tmp directory: Unix socket paths must stay under ~100 bytes."""
    import shutil
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp(prefix="rem-", dir="/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def daemon_cfg(short_dir):
    from rem.config import DaemonConfig

    return RemConfig(
        data_dir=short_dir,
        vector_store=VectorStoreConfig(provider="qdrant", path=short_dir / "qdrant"),
        embedder=EmbedderConfig(provider="hashing", model="hash-64"),
        daemon=DaemonConfig(socket_path=short_dir / "s.sock", debounce_seconds=0.2, lease_seconds=2),
    )


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def wait_until(predicate, timeout=2.0, interval=0.01):
    """Poll a sync predicate inside the event loop."""
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(interval)


async def wait_until_async(producer, predicate, timeout=2.0, interval=0.01):
    """Poll an async producer until predicate(value) holds."""
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate(await producer()):
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(interval)


class DaemonEnv:
    def __init__(self, root):
        import os

        self.root = root
        self.sock = root / "s.sock"
        self.lock = root / "rem.lock"
        self.log = root / "daemon.log"
        self.config = root / "config.toml"
        self.config.write_text(
            '[embedder]\nprovider = "hashing"\nmodel = "hash-64"\n'
            f'[daemon]\nsocket_path = "{self.sock}"\ndebounce_seconds = 0.2\nlease_seconds = 2\n'
        )
        self.env = {**os.environ, "REM_HOME": str(root), "REM_CONFIG": str(self.config)}
        self.env.pop("REM_SOCKET", None)


@pytest.fixture
def daemon_env(short_dir, monkeypatch):
    env = DaemonEnv(short_dir)
    monkeypatch.setenv("REM_HOME", str(short_dir))
    monkeypatch.setenv("REM_CONFIG", str(env.config))
    monkeypatch.delenv("REM_SOCKET", raising=False)
    yield env
    _stop_daemon(env)


def _stop_daemon(env):
    import os
    import signal
    import time

    from rem.daemon.lock import InstanceLock

    pid = InstanceLock.holder_pid(env.lock)
    if pid:
        os.kill(pid, signal.SIGTERM)
        deadline = time.time() + 10
        while InstanceLock.holder_pid(env.lock) and time.time() < deadline:
            time.sleep(0.05)


def spawn_daemon(env):
    import subprocess
    import sys

    log = open(env.log, "a")
    return subprocess.Popen([sys.executable, "-m", "rem", "daemon"], env=env.env, stdout=log, stderr=log)


def uds_client(sock):
    import httpx

    return httpx.Client(transport=httpx.HTTPTransport(uds=str(sock)), base_url="http://rem", timeout=5)


def wait_for_socket(sock, timeout=10.0, fresh_after=None):
    """Wait until a daemon answers /status on sock (with a pid other than fresh_after)."""
    import time

    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            pid = uds_client(sock).get("/status").json()["pid"]
            if fresh_after is None or pid != fresh_after:
                return pid
        except (httpx.HTTPError, OSError):
            pass
        time.sleep(0.05)
    raise AssertionError(f"daemon did not answer on {sock}")
