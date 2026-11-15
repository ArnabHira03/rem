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
