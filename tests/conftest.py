import pytest

from rem.config import EmbedderConfig, RemConfig, VectorStoreConfig
from rem.models import Provenance, Scope, Status


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
