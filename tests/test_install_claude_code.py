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


@pytest.fixture(autouse=True)
def _no_real_claude(monkeypatch):
    """No test may reach the real `claude` CLI (it would edit ~/.claude.json): `_install` looks it up
    through ic.shutil.which at call time, so make it absent here. MCP tests inject their own fakes."""
    monkeypatch.setattr(ic.shutil, "which", lambda name: None)


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
    out = capsys.readouterr().out
    assert set(json.JSONDecoder().raw_decode(out)[0]["hooks"]) == EVENTS
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


NO_CLAUDE_ADD = ("claude CLI not found; to add rem's MCP server run: "
                 + " ".join(["claude", "mcp", "add", "--scope", "user", "rem", "--", *ic.mcp_command()]))


def test_cli_install_tips_systemd_unit_only_when_missing(path, tmp_path, monkeypatch, capsys):
    from rem.systemd import unit_path

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)
    assert main(["install", "claude-code"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"installed rem hooks in {path}", NO_CLAUDE_ADD, TIP]
    for extra in (["--dry-run"], ["--uninstall"], ["--uninstall", "--dry-run"]):
        assert main(["install", "claude-code", *extra]) == 0
        assert TIP not in capsys.readouterr().out
    unit_path().parent.mkdir(parents=True)
    unit_path().write_text("[Unit]\n")
    assert main(["install", "claude-code"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"installed rem hooks in {path}", NO_CLAUDE_ADD]


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


# --- MCP registration (fake run / which: the real claude CLI is never called) -------------------
CLAUDE = "/fake/bin/claude"


class FakeRun:
    def __init__(self, returncodes=(0, 0), stderr=""):
        self.calls, self.rcs, self.stderr = [], list(returncodes), stderr

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        rc = self.rcs.pop(0) if self.rcs else 0
        return subprocess.CompletedProcess(argv, rc, "", self.stderr if rc else "")


def fake_which(name):
    return CLAUDE if name == "claude" else None


def test_mcp_command():
    assert ic.MCP_NAME == "rem"
    assert ic.mcp_command() == [sys.executable, "-m", "rem", "mcp"]


def test_register_runs_remove_then_add_user_scope(tmp_path):
    run = FakeRun()
    msg = ic.register_mcp("user", tmp_path, run=run, which=fake_which)
    assert msg == "registered rem MCP server (user scope)"
    assert [c[0] for c in run.calls] == [
        [CLAUDE, "mcp", "remove", "--scope", "user", "rem"],
        [CLAUDE, "mcp", "add", "--scope", "user", "rem", "--", sys.executable, "-m", "rem", "mcp"],
    ]
    assert all(c[1]["cwd"] == tmp_path and c[1]["timeout"] == 30 for c in run.calls)


def test_remove_failure_is_ignored(tmp_path):
    run = FakeRun(returncodes=(1, 0))
    assert ic.register_mcp("user", tmp_path, run=run, which=fake_which).startswith("registered")
    assert len(run.calls) == 2


def test_project_uses_local_scope_and_git_root_cwd(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.chdir(repo / "sub")
    run = FakeRun()
    monkeypatch.setattr(ic.subprocess, "run", run)
    monkeypatch.setattr(ic.shutil, "which", fake_which)
    assert main(["install", "claude-code", "--project"]) == 0
    assert "registered rem MCP server (local scope)" in capsys.readouterr().out
    assert [c[0][3:5] for c in run.calls] == [["--scope", "local"]] * 2
    assert all(c[1]["cwd"] == repo.resolve() for c in run.calls)


def test_default_cli_install_uses_user_scope_and_uninstall_removes(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run = FakeRun()
    monkeypatch.setattr(ic.subprocess, "run", run)
    monkeypatch.setattr(ic.shutil, "which", fake_which)
    assert main(["install", "claude-code"]) == 0
    assert [c[0][2] for c in run.calls] == ["remove", "add"]
    run.calls.clear()
    assert main(["install", "claude-code", "--uninstall"]) == 0
    assert [c[0] for c in run.calls] == [[CLAUDE, "mcp", "remove", "--scope", "user", "rem"]]
    assert "removed rem's MCP server (user scope)" in capsys.readouterr().out


def test_no_claude_cli_prints_manual_command(tmp_path):
    run = FakeRun()
    msg = ic.register_mcp("user", tmp_path, run=run, which=lambda n: None)
    assert msg == NO_CLAUDE_ADD
    assert run.calls == []
    assert ic.unregister_mcp("user", tmp_path, run=run, which=lambda n: None) == (
        "claude CLI not found; to remove rem's MCP server run: claude mcp remove --scope user rem")


def test_dry_run_runs_nothing(tmp_path):
    run = FakeRun()
    msg = ic.register_mcp("local", tmp_path, dry_run=True, run=run, which=fake_which)
    assert run.calls == []
    assert f"{CLAUDE} mcp remove --scope local rem" in msg
    assert f"{CLAUDE} mcp add --scope local rem -- {sys.executable} -m rem mcp" in msg
    assert ic.unregister_mcp("user", tmp_path, dry_run=True, run=run, which=fake_which)
    assert run.calls == []


def test_uninstall_removes(tmp_path):
    run = FakeRun()
    assert ic.unregister_mcp("user", tmp_path, run=run, which=fake_which) == (
        "removed rem's MCP server (user scope)")
    assert [c[0] for c in run.calls] == [[CLAUDE, "mcp", "remove", "--scope", "user", "rem"]]


def test_uninstall_not_found_is_not_a_failure(tmp_path):
    for msg in ("No MCP server found with name: rem", "Error: server Not Found", "no mcp server named rem"):
        run = FakeRun(returncodes=(1,), stderr=msg)
        assert ic.unregister_mcp("local", tmp_path, run=run, which=fake_which) == (
            "no rem MCP server registered (local scope)")
    run = FakeRun(returncodes=(1,), stderr="permission denied")
    assert ic.unregister_mcp("user", tmp_path, run=run, which=fake_which) == (
        "could not remove rem's MCP server: permission denied")


def test_add_failure_reported_not_raised(tmp_path, monkeypatch, capsys):
    run = FakeRun(returncodes=(0, 1), stderr="boom\nsecond line")
    assert ic.register_mcp("user", tmp_path, run=run, which=fake_which) == "could not add rem's MCP server: boom"

    def raising(argv, **kw):
        raise OSError("nope")

    assert ic.register_mcp("user", tmp_path, run=raising, which=fake_which).startswith("could not add rem's MCP server:")

    def slow(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 30)

    assert ic.register_mcp("user", tmp_path, run=slow, which=fake_which).startswith("could not add rem's MCP server:")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ic.subprocess, "run", FakeRun(returncodes=(0, 1), stderr="boom"))
    monkeypatch.setattr(ic.shutil, "which", fake_which)
    assert main(["install", "claude-code"]) == 0  # an MCP failure never changes the exit code
    assert "could not add rem's MCP server: boom" in capsys.readouterr().out
