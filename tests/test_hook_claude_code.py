import io
import json
import subprocess
import sys
import time

import httpx
import pytest

from rem.client import DaemonUnavailable
from rem.config import ClaudeCodeConfig, RemConfig

KEY = "sk-ant-" + "a" * 30


@pytest.fixture
def hcfg(tmp_path):
    return RemConfig(data_dir=tmp_path / "data")


class Fake:
    """Client factory recording every construction and call; `fail` makes calls raise."""

    def __init__(self, fail: BaseException | None = None, context_text: str = ""):
        self.fail, self.context_text = fail, context_text
        self.constructed: list[tuple] = []
        self.calls: list[tuple] = []

    def __call__(self, socket_path, timeout=None):
        self.constructed.append((socket_path, timeout))
        return self

    def ingest(self, source, session_id, cwd, turns, enqueue=True):
        self.calls.append(("ingest", dict(source=source, session_id=session_id, cwd=cwd, turns=turns, enqueue=enqueue)))
        if self.fail:
            raise self.fail
        return {}

    def context(self, cwd, max_chars=1600):
        self.calls.append(("context", dict(cwd=cwd, max_chars=max_chars)))
        if self.fail:
            raise self.fail
        return self.context_text


def down():
    try:
        raise httpx.ConnectError("refused")
    except httpx.ConnectError as e:
        try:
            raise DaemonUnavailable("down") from e
        except DaemonUnavailable as d:
            return d


def slow():
    try:
        raise httpx.ReadTimeout("slow")
    except httpx.ReadTimeout as e:
        try:
            raise DaemonUnavailable("slow") from e
        except DaemonUnavailable as d:
            return d


@pytest.fixture
def starts(monkeypatch):
    calls = []
    monkeypatch.setattr("rem.adapters.claude_code.start_daemon_background", lambda cfg: calls.append(cfg))
    return calls


def payload(event, cwd="/tmp", **kw):
    return {"session_id": "s1", "transcript_path": "/x.jsonl", "cwd": cwd, "hook_event_name": event, **kw}


def spooled(cfg):
    d = cfg.data_dir / "spool"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.is_dir() else []


# --- handle_hook: capture -------------------------------------------------------------

def test_user_prompt_submit_ingests_held_user_turn(hcfg, starts):
    from rem.adapters.claude_code import handle_hook

    f = Fake()
    out = handle_hook(payload("UserPromptSubmit", prompt="use uv", prompt_id="p1"), hcfg, client_factory=f)
    assert out == ""
    assert f.constructed == [(hcfg.socket_path, 2.0)]
    assert f.calls == [("ingest", dict(source="claude-code", session_id="s1", cwd="/tmp", enqueue=False,
                                       turns=[{"role": "user", "content": "use uv", "message_id": "cc:p1:user"}]))]
    assert starts == []


def test_blank_prompt_is_not_ingested(hcfg, starts):
    from rem.adapters.claude_code import handle_hook

    f = Fake()
    assert handle_hook(payload("UserPromptSubmit", prompt="  \n", prompt_id="p1"), hcfg, client_factory=f) == ""
    assert f.calls == []


def test_stop_ingests_assistant_turn_and_enqueues(hcfg, starts):
    from rem.adapters.claude_code import handle_hook

    f = Fake()
    p = payload("Stop", last_assistant_message="done", prompt_id="p1", stop_hook_active=True)
    assert handle_hook(p, hcfg, client_factory=f) == ""
    assert f.calls == [("ingest", dict(source="claude-code", session_id="s1", cwd="/tmp", enqueue=True,
                                       turns=[{"role": "assistant", "content": "done", "message_id": "cc:p1:assistant"}]))]


def test_stop_without_reply_still_flushes(hcfg, starts):
    from rem.adapters.claude_code import handle_hook

    f = Fake()
    handle_hook(payload("Stop", last_assistant_message="  ", prompt_id="p1"), hcfg, client_factory=f)
    handle_hook(payload("Stop", prompt_id="p1"), hcfg, client_factory=f)
    assert [c[1]["turns"] for c in f.calls] == [[], []]
    assert all(c[1]["enqueue"] is True for c in f.calls)


def test_session_end_flushes(hcfg, starts):
    from rem.adapters.claude_code import handle_hook

    f = Fake()
    assert handle_hook(payload("SessionEnd", reason="exit"), hcfg, client_factory=f) == ""
    assert f.calls == [("ingest", dict(source="claude-code", session_id="s1", cwd="/tmp", turns=[], enqueue=True))]


def test_missing_prompt_id_sends_no_message_id(hcfg, starts):
    from rem.adapters.claude_code import handle_hook

    f = Fake()
    handle_hook(payload("UserPromptSubmit", prompt="hi"), hcfg, client_factory=f)
    handle_hook(payload("Stop", last_assistant_message="yo"), hcfg, client_factory=f)
    assert [c[1]["turns"] for c in f.calls] == [[{"role": "user", "content": "hi"}],
                                                [{"role": "assistant", "content": "yo"}]]


def test_missing_cwd_is_none(hcfg, starts):
    from rem.adapters.claude_code import handle_hook

    f = Fake()
    p = payload("UserPromptSubmit", prompt="hi")
    del p["cwd"]
    handle_hook(p, hcfg, client_factory=f)
    assert f.calls[0][1]["cwd"] is None


def test_unknown_event_does_nothing(hcfg, starts):
    from rem.adapters.claude_code import handle_hook

    f = Fake()
    assert handle_hook(payload("PreToolUse"), hcfg, client_factory=f) == ""
    assert f.constructed == [] and starts == []


def test_redaction_before_client_and_spool(hcfg, starts):
    from rem.adapters.claude_code import handle_hook

    ok = Fake()
    handle_hook(payload("UserPromptSubmit", prompt=f"my key {KEY}", prompt_id="p1"), hcfg, client_factory=ok)
    assert KEY not in json.dumps(ok.calls) and "[REDACTED:anthropic_key]" in json.dumps(ok.calls)

    handle_hook(payload("Stop", last_assistant_message=f"key {KEY}", prompt_id="p1"), hcfg, client_factory=Fake(down()))
    raw = json.dumps(spooled(hcfg))
    assert KEY not in raw and "[REDACTED:anthropic_key]" in raw


# --- handle_hook: daemon down / slow ----------------------------------------------------

def test_capture_with_daemon_down_spools_and_starts_once(hcfg, starts):
    from rem.adapters.claude_code import handle_hook

    out = handle_hook(payload("UserPromptSubmit", prompt="hi", prompt_id="p1"), hcfg, client_factory=Fake(down()))
    assert out == ""
    assert spooled(hcfg) == [{"source": "claude-code", "session_id": "s1", "cwd": "/tmp", "enqueue": False,
                              "turns": [{"role": "user", "content": "hi", "message_id": "cc:p1:user"}]}]
    assert starts == [hcfg]


def test_capture_timeout_spools_without_starting(hcfg, starts):
    from rem.adapters.claude_code import handle_hook

    handle_hook(payload("SessionEnd"), hcfg, client_factory=Fake(slow()))
    assert spooled(hcfg) == [{"source": "claude-code", "session_id": "s1", "cwd": "/tmp", "turns": [], "enqueue": True}]
    assert starts == []


def test_session_start_returns_context_exactly(hcfg, starts):
    from rem.adapters.claude_code import handle_hook

    text = "Facts rem remembers from earlier sessions (verified):\nAbout the user:\n- Prefers uv"
    hcfg.claude_code = ClaudeCodeConfig(context_max_chars=900)
    f = Fake(context_text=text)
    assert handle_hook(payload("SessionStart", source="startup"), hcfg, client_factory=f) == text
    assert f.calls == [("context", dict(cwd="/tmp", max_chars=900))]


def test_session_start_daemon_down_starts_daemon(hcfg, starts):
    from rem.adapters.claude_code import handle_hook

    assert handle_hook(payload("SessionStart", source="startup"), hcfg, client_factory=Fake(down())) == ""
    assert starts == [hcfg]
    assert spooled(hcfg) == []


def test_session_start_timeout_gives_up_quietly(hcfg, starts):
    from rem.adapters.claude_code import handle_hook

    assert handle_hook(payload("SessionStart", source="resume"), hcfg, client_factory=Fake(slow())) == ""
    assert starts == []


@pytest.mark.parametrize("sub", ["", "deep/er"])
def test_excluded_cwd_does_nothing(hcfg, starts, tmp_path, sub):
    from rem.adapters.claude_code import handle_hook

    secret = tmp_path / "secret"
    (secret / sub).mkdir(parents=True, exist_ok=True)
    hcfg.claude_code = ClaudeCodeConfig(exclude=[secret])
    f = Fake(down())
    for p in [payload("SessionStart", cwd=str(secret / sub)),
              payload("UserPromptSubmit", cwd=str(secret / sub), prompt="hi", prompt_id="p1"),
              payload("Stop", cwd=str(secret / sub), last_assistant_message="yo"),
              payload("SessionEnd", cwd=str(secret / sub))]:
        assert handle_hook(p, hcfg, client_factory=f) == ""
    assert f.constructed == [] and starts == [] and spooled(hcfg) == []


def test_sibling_of_excluded_dir_is_not_excluded(hcfg, starts, tmp_path):
    from rem.adapters.claude_code import handle_hook

    (tmp_path / "secret").mkdir()
    (tmp_path / "secret-not").mkdir()
    hcfg.claude_code = ClaudeCodeConfig(exclude=[tmp_path / "secret"])
    f = Fake()
    handle_hook(payload("SessionEnd", cwd=str(tmp_path / "secret-not")), hcfg, client_factory=f)
    assert len(f.calls) == 1


# --- main() ------------------------------------------------------------------------------

@pytest.fixture
def hook_env(tmp_path, monkeypatch, starts):
    monkeypatch.setenv("REM_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("REM_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setenv("REM_SOCKET", "/tmp/rem-hook-test-nonexistent.sock")
    return tmp_path / "data"


def test_main_invalid_json_returns_0_and_prints_nothing(hook_env):
    from rem.adapters.claude_code import main

    out = io.StringIO()
    assert main(stdin=io.StringIO("{not json"), stdout=out) == 0
    assert out.getvalue() == ""
    lines = (hook_env / "hook.log").read_text().splitlines()
    assert len(lines) == 1 and "JSONDecodeError" in lines[0]


def test_main_logs_unexpected_error_without_prompt_text(hook_env, monkeypatch):
    from rem.adapters import claude_code

    monkeypatch.setattr(claude_code, "RemClient", Fake(RuntimeError("boom: the secret prompt")))
    out = io.StringIO()
    p = payload("UserPromptSubmit", prompt="the secret prompt", prompt_id="p1")
    assert claude_code.main(stdin=io.StringIO(json.dumps(p)), stdout=out) == 0
    assert out.getvalue() == ""
    lines = (hook_env / "hook.log").read_text().splitlines()
    assert len(lines) == 1
    assert "UserPromptSubmit" in lines[0] and "RuntimeError" in lines[0]
    assert "secret prompt" not in lines[0]


def test_main_writes_context_for_session_start(hook_env, monkeypatch):
    from rem.adapters import claude_code

    monkeypatch.setattr(claude_code, "RemClient", Fake(context_text="Facts rem remembers"))
    out = io.StringIO()
    assert claude_code.main(stdin=io.StringIO(json.dumps(payload("SessionStart"))), stdout=out) == 0
    assert out.getvalue() == "Facts rem remembers"


def test_main_bad_config_is_silent(tmp_path, monkeypatch):
    from rem.adapters.claude_code import main

    bad = tmp_path / "bad.toml"
    bad.write_text("[llm]\nprovider = 'nope'\n")
    monkeypatch.setenv("REM_CONFIG", str(bad))
    monkeypatch.setenv("REM_HOME", str(tmp_path / "data"))
    out = io.StringIO()
    assert main(stdin=io.StringIO(json.dumps(payload("SessionStart"))), stdout=out) == 0
    assert out.getvalue() == ""


# --- subprocess: the real `rem hook claude-code` -----------------------------------------

def run_hook(env, body: str):
    return subprocess.run([sys.executable, "-m", "rem", "hook", "claude-code"], input=body.encode(), env=env,
                          capture_output=True, timeout=30)


def test_subprocess_capture_daemon_down_is_silent_spools_and_starts(daemon_env):
    from rem.client import RemClient

    r = run_hook(daemon_env.env, json.dumps(payload("UserPromptSubmit", prompt="hi", prompt_id="p1")))
    assert (r.returncode, r.stdout, r.stderr) == (0, b"", b"")
    spool = daemon_env.root / "spool"
    assert spool.is_dir()  # only the hook creates it
    deadline = time.time() + 15
    while True:  # the hook started the daemon, which drains the spooled turn cleanly
        try:
            RemClient(daemon_env.sock, timeout=2).status()
            if not list(spool.glob("*.json")):
                break
        except DaemonUnavailable:
            pass
        if time.time() > deadline:
            pytest.fail("hook did not start the daemon, or the spool was not drained")
        time.sleep(0.1)
    assert not (spool / "bad").exists()


def test_subprocess_with_running_daemon(daemon_env):
    from tests.conftest import spawn_daemon

    from rem.client import RemClient

    spawn_daemon(daemon_env)
    c = RemClient(daemon_env.sock, timeout=2)
    deadline = time.time() + 15
    while True:
        try:
            c.status()
            break
        except DaemonUnavailable:
            if time.time() > deadline:
                pytest.fail("daemon did not start")
            time.sleep(0.1)
    c.add("Prefers uv over pip", cwd=None)

    r = run_hook(daemon_env.env, json.dumps(payload("UserPromptSubmit", prompt="hello", prompt_id="p1")))
    assert (r.returncode, r.stdout, r.stderr) == (0, b"", b"")
    r = run_hook(daemon_env.env, json.dumps(payload("SessionStart", source="startup")))
    assert r.returncode == 0 and r.stderr == b""
    assert r.stdout.decode() == c.context("/tmp")
    assert "Prefers uv over pip" in r.stdout.decode()


def test_subprocess_invalid_json_is_silent(daemon_env):
    r = run_hook(daemon_env.env, "garbage")
    assert (r.returncode, r.stdout, r.stderr) == (0, b"", b"")


def test_hook_imports_are_light():
    code = ("import rem.adapters.claude_code, sys; print(sorted(m for m in sys.modules if m.startswith("
            "('rem.memory','rem.embed','rem.storage.vector','qdrant_client','fastembed'))))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "[]"


# --- fix round 1 ---------------------------------------------------------------------------

@pytest.mark.parametrize("sid", [None, "", 123])
@pytest.mark.parametrize("p", [payload("UserPromptSubmit", prompt="hi", prompt_id="p1"),
                               payload("Stop", last_assistant_message="yo"), payload("SessionEnd")])
def test_capture_without_session_id_does_nothing(hcfg, starts, sid, p):
    from rem.adapters.claude_code import handle_hook

    p = {**p, "session_id": sid}
    f = Fake(down())
    assert handle_hook(p, hcfg, client_factory=f) == ""
    assert f.calls == [] and starts == [] and spooled(hcfg) == []
    assert not (hcfg.data_dir / "hook.log").exists()


@pytest.mark.parametrize("err,started", [(down, True), (slow, False)])
def test_failed_spool_still_starts_daemon_on_connection_failure(hcfg, starts, monkeypatch, err, started):
    from rem.adapters import claude_code

    def broken(*a):
        raise OSError("disk full")

    monkeypatch.setattr(claude_code, "write_spool", broken)
    with pytest.raises(OSError):
        claude_code.handle_hook(payload("SessionEnd"), hcfg, client_factory=Fake(err()))
    assert starts == ([hcfg] if started else [])


@pytest.mark.parametrize("argv", [["claude-code", "--x"], ["claude-cod"], ["claude-code", "extra"], []])
def test_subprocess_unknown_hook_args_are_silent(daemon_env, argv):
    r = subprocess.run([sys.executable, "-m", "rem", "hook", *argv], input=b"{}", env=daemon_env.env,
                       capture_output=True, timeout=30)
    assert (r.returncode, r.stdout, r.stderr) == (0, b"", b"")
    lines = (daemon_env.root / "hook.log").read_text().splitlines()
    assert len(lines) == 1 and lines[0].endswith(" unknown unknown hook args")
    assert not any(a in lines[0] for a in ["--x", "claude-cod", "extra"])


def test_subprocess_broken_stdout_pipe_is_silent(tmp_path):
    import os

    code = (
        "import sys\n"
        "from rem.adapters import claude_code\n"
        "class F:\n"
        "    def __init__(self, *a, **k): pass\n"
        # small enough to sit in the stdout buffer: without the fix the shutdown flush prints
        # "Exception ignored ... BrokenPipeError" and the process exits 120
        "    def context(self, cwd, max_chars): return 'x' * 100\n"
        "claude_code.RemClient = F\n"
        "sys.exit(claude_code.main())\n"
    )
    env = {**os.environ, "REM_HOME": str(tmp_path / "data"), "REM_CONFIG": str(tmp_path / "none.toml"),
           "REM_SOCKET": "/tmp/rem-hook-test-nonexistent.sock"}
    r_fd, w_fd = os.pipe()
    os.close(r_fd)  # nobody reads: the hook's write hits EPIPE
    try:
        r = subprocess.run([sys.executable, "-c", code], input=json.dumps(payload("SessionStart")).encode(),
                           stdout=w_fd, stderr=subprocess.PIPE, env=env, timeout=30)
    finally:
        os.close(w_fd)
    assert (r.returncode, r.stderr) == (0, b"")
    assert "BrokenPipeError" in (tmp_path / "data" / "hook.log").read_text()


# --- final fix wave ---------------------------------------------------------------------------

def test_first_hook_log_under_umask_022_is_private(hook_env):
    import os
    import stat

    from rem.adapters.claude_code import main

    old = os.umask(0o022)
    try:
        assert main(stdin=io.StringIO("{not json"), stdout=io.StringIO()) == 0
    finally:
        os.umask(old)
    assert stat.S_IMODE(hook_env.stat().st_mode) == 0o700
    assert stat.S_IMODE((hook_env / "hook.log").stat().st_mode) == 0o600
