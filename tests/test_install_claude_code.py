import json
import os
import stat
import subprocess
import sys

import pytest

from rem import install_claude_code as ic
from rem.cli import main
from rem.errors import RemError

CMD = f"{sys.executable} -m rem hook claude-code"
EVENTS = {"SessionStart", "UserPromptSubmit", "Stop", "SessionEnd"}


def rem_handlers(settings):
    return [
        (ev, h)
        for ev, groups in settings.get("hooks", {}).items()
        for g in groups for h in g["hooks"] if ic.MARKER in h["command"]
    ]


@pytest.fixture(autouse=True)
def _claude_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))


@pytest.fixture
def path(tmp_path):
    return tmp_path / "claude" / "settings.json"


def test_fresh_install_creates_four_events(path):
    ic.install(path, CMD)
    data = json.loads(path.read_text())
    assert set(data["hooks"]) == EVENTS
    for ev, timeout in ic.EVENTS.items():
        assert data["hooks"][ev] == [
            {"matcher": "*", "hooks": [{"type": "command", "command": CMD, "timeout": timeout}]}
        ]
    assert path.read_text().endswith("}\n")
    assert not path.with_name("settings.json.rem-backup").exists()


def test_install_twice_is_identical_and_backs_up(path):
    ic.install(path, CMD)
    first = path.read_text()
    ic.install(path, CMD)
    assert path.read_text() == first
    assert sorted(ev for ev, _ in rem_handlers(json.loads(first))) == sorted(EVENTS)
    assert path.with_name("settings.json.rem-backup").read_text() == first


def test_user_settings_survive_install_and_uninstall(path):
    path.parent.mkdir(parents=True)
    own = {"matcher": "Bash", "hooks": [{"type": "command", "command": "mine.sh"}]}
    original = {"model": "x", "permissions": {"allow": ["Bash(ls)"]}, "hooks": {"Stop": [own]}}
    path.write_text(json.dumps(original))
    ic.install(path, CMD)
    data = json.loads(path.read_text())
    assert data["model"] == "x" and data["permissions"] == original["permissions"]
    assert data["hooks"]["Stop"][0] == own and len(data["hooks"]["Stop"]) == 2
    ic.uninstall(path)
    assert json.loads(path.read_text()) == original


def test_shared_group_only_rem_handler_dropped():
    mine = {"type": "command", "command": "mine.sh"}
    rem = {"type": "command", "command": CMD, "timeout": 10}
    s = {"hooks": {"Stop": [{"matcher": "*", "hooks": [mine, rem]}]}}
    assert ic.removed(s) == {"hooks": {"Stop": [{"matcher": "*", "hooks": [mine]}]}}


def test_older_command_replaced_not_duplicated(path):
    ic.install(path, "/old/python -m rem hook claude-code")
    ic.install(path, CMD)
    handlers = rem_handlers(json.loads(path.read_text()))
    assert len(handlers) == 4 and {h["command"] for _, h in handlers} == {CMD}


def test_uninstall_removes_hooks_key_when_empty(path):
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"model": "x"}))
    ic.install(path, CMD)
    ic.uninstall(path)
    assert json.loads(path.read_text()) == {"model": "x"}


def test_invalid_json_refused_untouched(path):
    path.parent.mkdir(parents=True)
    path.write_text('{"model": // comment\n}')
    with pytest.raises(RemError, match="not valid JSON; not modified"):
        ic.install(path, CMD)
    with pytest.raises(RemError):
        ic.uninstall(path)
    assert path.read_text() == '{"model": // comment\n}'
    assert not path.with_name("settings.json.rem-backup").exists()


def test_permissions_preserved(path):
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    os.chmod(path, 0o600)
    ic.install(path, CMD)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.with_name("settings.json.rem-backup").stat().st_mode) == 0o600
    assert not [p for p in path.parent.iterdir() if p.suffix == ".tmp"]


def test_dry_run_writes_nothing(path, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["install", "claude-code", "--dry-run"]) == 0
    assert set(json.loads(capsys.readouterr().out)["hooks"]) == EVENTS
    assert not path.exists()
    path.parent.mkdir(parents=True)
    path.write_text('{"model": "x"}')
    assert main(["install", "claude-code", "--dry-run"]) == 0
    assert path.read_text() == '{"model": "x"}'
    assert not path.with_name("settings.json.rem-backup").exists()


def test_settings_path_user_and_project(tmp_path, monkeypatch):
    assert ic.settings_path(False, tmp_path) == tmp_path / "claude" / "settings.json"
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    assert ic.settings_path(False, tmp_path) == ic.Path("~/.claude").expanduser() / "settings.json"
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    assert ic.settings_path(True, repo / "sub") == repo.resolve() / ".claude" / "settings.local.json"


def test_cli_install_uninstall_project_and_status_without_daemon(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.chdir(repo)
    assert main(["status"]) == 1
    assert capsys.readouterr().out == "claude_code_hooks: none\nrem daemon is not running\n"
    assert main(["install", "claude-code"]) == 0
    assert main(["install", "claude-code", "--project"]) == 0
    capsys.readouterr()
    assert main(["status"]) == 1
    assert capsys.readouterr().out.startswith("claude_code_hooks: user,project\n")
    assert main(["install", "claude-code", "--uninstall"]) == 0
    capsys.readouterr()
    main(["status"])
    assert capsys.readouterr().out.startswith("claude_code_hooks: project\n")
    assert main(["install", "claude-code", "--project", "--uninstall"]) == 0
    capsys.readouterr()
    main(["status"])
    assert capsys.readouterr().out.startswith("claude_code_hooks: none\n")


def test_cli_invalid_json_exit_1(path, tmp_path, monkeypatch, capsys):
    path.parent.mkdir(parents=True)
    path.write_text("nope")
    monkeypatch.chdir(tmp_path)
    assert main(["install", "claude-code"]) == 1
    assert "not valid JSON" in capsys.readouterr().err
    assert path.read_text() == "nope"
