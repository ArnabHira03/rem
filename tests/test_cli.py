import json

import pytest

from rem.cli import main


@pytest.fixture
def cli_env(daemon_env, monkeypatch):
    """CLI against a real daemon (auto-started) in a short temp dir, offline embedder."""
    monkeypatch.chdir(daemon_env.root)
    return daemon_env


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
