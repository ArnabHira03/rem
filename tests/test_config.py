import pytest

from rem.config import load_config


def test_defaults_use_rem_home(tmp_path, monkeypatch):
    monkeypatch.setenv("REM_HOME", str(tmp_path))
    monkeypatch.delenv("REM_CONFIG", raising=False)
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.data_dir == tmp_path
    assert cfg.db_path == tmp_path / "rem.db"
    assert cfg.qdrant_path == tmp_path / "qdrant"
    assert cfg.vector_store.provider == "qdrant"
    assert cfg.embedder.model == "BAAI/bge-small-en-v1.5"


def test_toml_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("REM_HOME", str(tmp_path))
    p = tmp_path / "config.toml"
    p.write_text(
        '[embedder]\nprovider = "hashing"\nmodel = "hash-64"\n'
        '[vector_store]\nprovider = "memory"\n'
    )
    cfg = load_config(p)
    assert cfg.embedder.provider == "hashing" and cfg.embedder.model == "hash-64"
    assert cfg.vector_store.provider == "memory"


def test_invalid_provider_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("REM_HOME", str(tmp_path))
    p = tmp_path / "config.toml"
    p.write_text('[vector_store]\nprovider = "faiss"\n')
    with pytest.raises(ValueError):
        load_config(p)
