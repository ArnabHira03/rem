import json

import pytest

from rem.cli import main
from rem.storage.vector.qdrant import QdrantVectorStore


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Real Qdrant persistence under tmp_path, offline hashing embedder."""
    cfg = tmp_path / "config.toml"
    cfg.write_text('[embedder]\nprovider = "hashing"\nmodel = "hash-64"\n')
    monkeypatch.setenv("REM_HOME", str(tmp_path))
    monkeypatch.setenv("REM_CONFIG", str(cfg))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_add_then_search_json(cli_env, capsys):
    assert main(["add", "Prefers uv over pip"]) == 0
    capsys.readouterr()
    assert main(["--json", "search", "uv or pip"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["text"] == "Prefers uv over pip" and out[0]["provenance"] == "user_confirmed"
    assert "score" in out[0] and "semantic" in out[0]


def test_human_output_line(cli_env, capsys):
    main(["add", "OS is Fedora 44", "--scope", "system"])
    capsys.readouterr()
    assert main(["list"]) == 0
    assert "[system] verified  OS is Fedora 44" in capsys.readouterr().out


def test_history_and_delete(cli_env, capsys):
    main(["--json", "add", "Uses npm"])
    mid = json.loads(capsys.readouterr().out)["id"]
    assert main(["update", mid, "Uses pnpm"]) == 0 and main(["delete", mid]) == 0
    capsys.readouterr()
    main(["--json", "history", mid])
    assert [h["event"] for h in json.loads(capsys.readouterr().out)] == ["ADD", "UPDATE", "DELETE"]


def test_unknown_id_exit_1(cli_env, capsys):
    assert main(["confirm", "00000000-0000-0000-0000-000000000000"]) == 1
    assert "no memory with id" in capsys.readouterr().err
    assert main(["get", "00000000-0000-0000-0000-000000000000"]) == 1


def test_empty_text_exit_2(cli_env, capsys):
    assert main(["add", "   "]) == 2 and "text must not be empty" in capsys.readouterr().err


def test_store_locked_exit_1(cli_env, capsys):
    holder = QdrantVectorStore(path=cli_env / "qdrant")
    try:
        assert main(["list"]) == 1 and "in use by another rem process" in capsys.readouterr().err
    finally:
        holder.close()
