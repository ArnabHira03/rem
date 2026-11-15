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


def test_claude_code_config_defaults_and_expanduser(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("REM_HOME", str(tmp_path))
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.claude_code.exclude == []
    assert cfg.claude_code.context_max_chars == 1600
    assert cfg.claude_code.hook_budget_seconds == 2.0
    path = tmp_path / "config.toml"
    path.write_text('[claude_code]\nexclude = ["~/work/client"]\n')
    cfg = load_config(path)
    assert cfg.claude_code.exclude == [Path.home() / "work/client"]


@pytest.mark.parametrize("line,ok", [
    ("context_max_chars = 1", True), ("context_max_chars = 10000", True),
    ("context_max_chars = 0", False), ("context_max_chars = 10001", False),
    ("hook_budget_seconds = 9", True), ("hook_budget_seconds = 0.5", True),
    ("hook_budget_seconds = 0", False), ("hook_budget_seconds = -1", False), ("hook_budget_seconds = 9.5", False),
])
def test_claude_code_limits_are_validated(tmp_path, monkeypatch, line, ok):
    monkeypatch.setenv("REM_HOME", str(tmp_path))
    p = tmp_path / "config.toml"
    p.write_text(f"[claude_code]\n{line}\n")
    if ok:
        load_config(p)
    else:
        with pytest.raises(ValueError, match="claude_code"):
            load_config(p)
