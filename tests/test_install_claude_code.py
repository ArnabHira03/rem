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


def test_install_twice_is_noop(path):
    ic.install(path, CMD)
    first = path.read_text()
    mtime = path.stat().st_mtime_ns
    assert "already installed" in ic.install(path, CMD)
    assert path.read_text() == first and path.stat().st_mtime_ns == mtime
    assert sorted(ev for ev, _ in rem_handlers(json.loads(first))) == sorted(EVENTS)
    assert not path.with_name("settings.json.rem-backup").exists()


def test_backup_is_first_pre_rem_content_and_never_overwritten(path):
    path.parent.mkdir(parents=True)
    path.write_text('{"model": "x"}')
    ic.install(path, CMD)
    backup = path.with_name("settings.json.rem-backup")
    assert backup.read_text() == '{"model": "x"}'
    ic.install(path, "/other/python -m rem hook claude-code")  # a real change
    ic.uninstall(path)
    assert backup.read_text() == '{"model": "x"}'


def test_uninstall_noops(path):
    assert "nothing to remove" in ic.uninstall(path)
    assert not path.parent.exists()
    path.parent.mkdir(parents=True)
    path.write_text('{"model":   "x"}')
    assert "nothing to remove" in ic.uninstall(path)
    assert path.read_text() == '{"model":   "x"}'
    assert not path.with_name("settings.json.rem-backup").exists()


def test_symlinked_settings_keeps_link(path, tmp_path):
    target = tmp_path / "dotfiles" / "settings.json"
    target.parent.mkdir()
    target.write_text('{"model": "x"}')
    path.parent.mkdir(parents=True)
    path.symlink_to(target)
    ic.install(path, CMD)
    assert path.is_symlink() and path.resolve() == target
    assert set(json.loads(target.read_text())["hooks"]) == EVENTS
    ic.uninstall(path)
    assert path.is_symlink() and json.loads(target.read_text()) == {"model": "x"}


def test_non_ascii_preserved(path):
    path.parent.mkdir(parents=True)
    path.write_text('{"note": "héllo ✓"}', encoding="utf-8")
    ic.install(path, CMD)
    assert "héllo ✓" in path.read_text(encoding="utf-8")


def test_non_list_event_refused(path):
    path.parent.mkdir(parents=True)
    text = '{"hooks": {"Stop": {"oops": 1}}}'
    path.write_text(text)
    with pytest.raises(RemError, match=r"hooks\.Stop is not a list; not modified"):
        ic.install(path, CMD)
    assert path.read_text() == text
    assert not path.with_name("settings.json.rem-backup").exists()


def test_empty_config_dir_means_default(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "")
    assert ic.settings_path(False, tmp_path) == ic.Path("~/.claude").expanduser() / "settings.json"


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
    assert capsys.readouterr().out == "claude_code_hooks: none\nspooled: 0\nrem daemon is not running\n"
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


# --- final fix wave ---------------------------------------------------------------------------

def test_backup_created_with_original_mode_under_umask_022(path, monkeypatch):
    path.parent.mkdir(parents=True)
    path.write_text('{"model": "x"}')
    os.chmod(path, 0o600)
    modes = []
    real_open = os.open

    def spy(p, flags, mode=0o777, *a, **kw):
        fd = real_open(p, flags, mode, *a, **kw)
        if str(p).endswith(".rem-backup"):
            modes.append(stat.S_IMODE(os.fstat(fd).st_mode))  # the mode the file is born with
        return fd

    monkeypatch.setattr(ic.os, "open", spy)
    old = os.umask(0o022)
    try:
        ic.install(path, CMD)
    finally:
        os.umask(old)
    backup = path.with_name("settings.json.rem-backup")
    assert modes == [0o600]
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert backup.read_text() == '{"model": "x"}'


def test_uninstall_mentions_kept_backup(path):
    path.parent.mkdir(parents=True)
    path.write_text('{"model": "x"}')
    ic.install(path, CMD)
    msg = ic.uninstall(path)
    backup = path.with_name("settings.json.rem-backup")
    assert msg.splitlines()[0] == f"removed rem hooks from {path}"
    assert str(backup) in msg.splitlines()[1]
    assert backup.exists()


def test_uninstall_without_backup_says_nothing_about_it(path):
    path.parent.mkdir(parents=True)
    path.write_text('{"model": "x"}')
    assert ic.uninstall(path) == f"nothing to remove in {path}"


TIP = "tip: run 'rem install-service --enable' so the daemon is up before your first Claude Code session"


def test_cli_install_tips_systemd_unit_only_when_missing(path, tmp_path, monkeypatch, capsys):
    from rem.systemd import unit_path

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)
    assert main(["install", "claude-code"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"installed rem hooks in {path}", TIP]
    for extra in (["--dry-run"], ["--uninstall"], ["--uninstall", "--dry-run"]):
        assert main(["install", "claude-code", *extra]) == 0
        assert TIP not in capsys.readouterr().out
    unit_path().parent.mkdir(parents=True)
    unit_path().write_text("[Unit]\n")
    assert main(["install", "claude-code"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"installed rem hooks in {path}"]


@pytest.mark.parametrize("extra", [[], ["--uninstall"]])
def test_cli_install_ignores_broken_rem_config(path, tmp_path, monkeypatch, capsys, extra):
    bad = tmp_path / "bad.toml"
    bad.write_text("[llm]\nprovider = 'nope'\n")
    monkeypatch.setenv("REM_CONFIG", str(bad))
    monkeypatch.chdir(tmp_path)
    assert main(["install", "claude-code", *extra]) == 0
    assert capsys.readouterr().err == ""


def test_symlink_loop_is_a_clean_error(tmp_path, capsys):
    loop = tmp_path / "claude"
    loop.mkdir()
    (loop / "settings.json").symlink_to(loop / "settings.json")
    assert main(["install", "claude-code"]) == 1
    assert capsys.readouterr().err.startswith("rem: ")
