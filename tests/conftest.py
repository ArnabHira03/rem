import pytest

from rem.config import EmbedderConfig, RemConfig, VectorStoreConfig


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
