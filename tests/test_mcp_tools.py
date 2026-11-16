"""The seven MCP memory tools against a real daemon, and `rem mcp` as a subprocess."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rem.client import DaemonUnavailable, RemClient
from rem.mcp.tools import INSTRUCTIONS, build_tools

from .conftest import spawn_daemon, wait_for_socket

NAMES = {"search_memory", "list_memories", "add_memory", "confirm_memory", "update_memory", "delete_memory",
         "memory_history"}


@pytest.fixture
def client(daemon_env):
    spawn_daemon(daemon_env)
    wait_for_socket(daemon_env.sock)
    return RemClient(daemon_env.sock)


def tools_for(client, cwd):
    return {t.name: t for t in build_tools(lambda: client, cwd)}


def call(tools, name, **args):
    return tools[name].handler(args)


def added_id(result) -> str:
    assert not result.is_error, result.text
    prefix = "Saved as unverified (the user can confirm it): "
    assert result.text.startswith(prefix)
    return result.text[len(prefix):].split()[0]


def test_tool_names_and_schemas(tmp_path):
    tools = build_tools(lambda: None, tmp_path)
    assert {t.name for t in tools} == NAMES
    for t in tools:
        assert t.input_schema["type"] == "object"
        assert t.input_schema["additionalProperties"] is False
    by = {t.name: t for t in tools}
    assert by["add_memory"].input_schema["properties"]["scope"]["enum"] == ["global", "system", "project"]
    assert by["list_memories"].input_schema["properties"]["status"]["enum"] == ["verified", "unverified"]
    assert "This does not verify it by itself" in by["confirm_memory"].description
    assert "Never use it for your own claims" in by["confirm_memory"].description
    assert "only the current git repository's" in by["list_memories"].description
    assert "unverified until the user confirms it" in by["add_memory"].description
    assert by["update_memory"].description == (
        "Change a memory's text. The memory becomes unverified until the user confirms it.")
    assert INSTRUCTIONS.startswith("rem is the user's long-term memory")


def test_add_memory_is_unverified_assistant_generated(client, tmp_path):
    tools = tools_for(client, tmp_path)
    mid = added_id(call(tools, "add_memory", text="Prefers uv over pip"))
    m = client.get(mid)
    assert m["provenance"] == "assistant_generated"
    assert m["status"] == "unverified"
    assert m["source"] == "mcp"
    assert m["scope"] == "global"
    assert [(h["event"], h["actor"]) for h in client.history(mid)] == [("ADD", "mcp")]


def test_search_and_list_text(client, tmp_path):
    tools = tools_for(client, tmp_path)
    assert call(tools, "search_memory", query="anything").text == "No memories found."
    assert call(tools, "list_memories").text == "No memories found."
    mid = added_id(call(tools, "add_memory", text="Prefers uv over pip"))
    client.add("Uses fish shell")

    r = call(tools, "search_memory", query="uv pip", limit=1)
    assert not r.is_error
    lines = r.text.splitlines()
    assert len(lines) == 1
    score, rest = lines[0].split("  ", 1)
    float(score)
    assert rest == f"{mid}  [global] unverified  Prefers uv over pip"

    r = call(tools, "list_memories", status="unverified")
    assert r.text == f"{mid}  [global] unverified  Prefers uv over pip"
    assert len(call(tools, "list_memories").text.splitlines()) == 2
    assert len(call(tools, "list_memories", limit=1).text.splitlines()) == 1


def test_confirm_memory_only_requests_confirmation(client, tmp_path):
    tools = tools_for(client, tmp_path)
    mid = added_id(call(tools, "add_memory", text="Prefers uv over pip"))
    r = call(tools, "confirm_memory", id=mid)
    assert not r.is_error
    assert r.text == (f"Confirmation requested (still unverified): {mid}  [global] unverified  Prefers uv over pip\n"
                      "rem verifies it only if the user's own message in this conversation confirms it in words "
                      "that name the fact (a bare \"yes\" is not enough), which it checks after this turn ends; "
                      "otherwise it stays unverified.")
    m = client.get(mid)
    assert m["status"] == "unverified" and m["provenance"] == "assistant_generated"
    assert m["confirm_requested_at"]
    h = call(tools, "memory_history", id=mid)
    assert not h.is_error
    rows = h.text.splitlines()
    assert len(rows) == 2
    assert "CONFIRM_REQUEST" in rows[1] and "mcp" in rows[1]
    assert [(x["event"], x["actor"]) for x in client.history(mid)] == [("ADD", "mcp"), ("CONFIRM_REQUEST", "mcp")]


def test_confirm_memory_of_verified_memory_changes_nothing(client, tmp_path):
    tools = tools_for(client, tmp_path)
    mid = client.add("Uses fish shell")["id"]  # user-confirmed via the CLI
    r = call(tools, "confirm_memory", id=mid)
    assert r.text == f"Already verified: {mid}  [global] verified  Uses fish shell"
    assert [x["event"] for x in client.history(mid)] == ["ADD"]


def _repo(root: Path, name: str) -> Path:
    d = root / name
    (d / ".git").mkdir(parents=True)
    return d


def _texts(tools, **args):
    r = call(tools, "list_memories", **args)
    assert not r.is_error, r.text
    return [] if r.text == "No memories found." else [line.split("  ", 2)[2] for line in r.text.splitlines()]


def test_list_memories_shows_only_the_current_repositorys_project_memories(client, tmp_path):
    a, b = _repo(tmp_path, "repo-a"), _repo(tmp_path, "repo-b")
    (b / "sub").mkdir()
    ta, tb = tools_for(client, a), tools_for(client, b / "sub")
    added_id(call(ta, "add_memory", text="Uses tabs in repo A", scope="project"))
    added_id(call(tb, "add_memory", text="Uses nox in repo B", scope="project"))
    client.add("Prefers uv over pip")  # global

    assert _texts(ta, scope="project") == ["Uses tabs in repo A"]
    assert _texts(tb, scope="project") == ["Uses nox in repo B"]
    assert sorted(_texts(ta)) == ["Prefers uv over pip", "Uses tabs in repo A"]
    assert sorted(_texts(tb)) == ["Prefers uv over pip", "Uses nox in repo B"]
    assert _texts(ta, scope="global") == ["Prefers uv over pip"]
    assert _texts(ta, scope="project", status="verified") == []
    # a repository rem has never stored anything for has no project memories
    assert _texts(tools_for(client, _repo(tmp_path, "repo-c")), scope="project") == []
    assert len(client.list()) == 3  # the CLI view is unchanged


def test_list_memories_outside_a_repository_exposes_no_project_memories(client, tmp_path):
    a = _repo(tmp_path, "repo-a")
    added_id(call(tools_for(client, a), "add_memory", text="Uses tabs in repo A", scope="project"))
    client.add("Prefers uv over pip")
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    tools = tools_for(client, plain)
    r = call(tools, "list_memories", scope="project")
    assert r.is_error
    assert r.text == (f"no project detected: rem mcp is running in {plain}, which is not in a git repository; "
                      "only the current repository's project memories can be listed")
    assert _texts(tools) == ["Prefers uv over pip"]


def test_update_and_delete(client, tmp_path):
    tools = tools_for(client, tmp_path)
    mid = added_id(call(tools, "add_memory", text="Uses npm"))
    r = call(tools, "update_memory", id=mid, text="Uses pnpm")
    assert not r.is_error
    assert r.text == f"Updated (now unverified until the user confirms): {mid}  [global] unverified  Uses pnpm"
    r = call(tools, "delete_memory", id=mid)
    assert not r.is_error and r.text == f"Deleted {mid}"
    assert [(x["event"], x["actor"]) for x in client.history(mid)] == [
        ("ADD", "mcp"), ("UPDATE", "mcp"), ("DELETE", "mcp")]


def test_update_of_confirmed_memory_makes_it_unverified(client, tmp_path):
    tools = tools_for(client, tmp_path)
    mid = client.add("Prefers uv over pip")["id"]  # user-confirmed, verified
    assert client.get(mid)["status"] == "verified"
    r = call(tools, "update_memory", id=mid, text="Prefers pip")
    assert not r.is_error
    m = client.get(mid)
    assert m["text"] == "Prefers pip"
    assert m["status"] == "unverified" and m["provenance"] == "assistant_generated"
    assert not call(tools, "confirm_memory", id=mid).is_error
    assert client.get(mid)["status"] == "unverified"  # only the user's own words can verify it again


def test_unknown_id_is_error_result(client, tmp_path):
    tools = tools_for(client, tmp_path)
    bad = "00000000-0000-0000-0000-000000000000"
    for name, args in [("confirm_memory", {"id": bad}), ("update_memory", {"id": bad, "text": "x"}),
                       ("delete_memory", {"id": bad})]:
        r = tools[name].handler(args)
        assert r.is_error and r.text == f"No memory with id {bad}", name


def test_bad_arguments_are_error_results(client, tmp_path):
    tools = tools_for(client, tmp_path)
    assert call(tools, "add_memory").is_error  # text missing
    assert call(tools, "add_memory", text="x", scope="galaxy").is_error
    assert call(tools, "search_memory", query="x", limit=0).is_error
    assert call(tools, "search_memory", query="x", limit=True).is_error
    assert call(tools, "list_memories", limit=201).is_error
    assert call(tools, "confirm_memory", id=3).is_error
    r = call(tools, "add_memory", text="   ")  # daemon `invalid` -> ValueError message
    assert r.is_error and r.text
    assert client.list() == []


def test_project_scope_outside_project_is_error(client):
    tools = tools_for(client, Path("/"))
    r = call(tools, "add_memory", text="Uses tabs here", scope="project")
    assert r.is_error
    assert r.text == "project scope needs a git repository; rem mcp is running in /"
    assert client.list() == []


def test_project_scope_inside_git_repo(client, tmp_path):
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "src"
    sub.mkdir()
    tools = tools_for(client, sub)
    mid = added_id(call(tools, "add_memory", text="Uses tabs here", scope="project"))
    m = client.get(mid)
    assert m["scope"] == "project" and m["project_name"] == tmp_path.name


def test_daemon_down_is_error_then_recovers(client, tmp_path):
    calls = {"n": 0}

    def get_client():
        calls["n"] += 1
        if calls["n"] == 1:
            raise DaemonUnavailable("rem daemon is not reachable at /nowhere")
        return client

    tools = {t.name: t for t in build_tools(get_client, tmp_path)}
    r = call(tools, "search_memory", query="uv")
    assert r.is_error and r.text == "rem daemon is not reachable at /nowhere"
    r = call(tools, "search_memory", query="uv")
    assert not r.is_error and r.text == "No memories found."


def test_run_mcp_client_cache_retries_after_daemon_unavailable(monkeypatch):
    from rem.mcp import tools as mt

    made = []

    class FakeClient:
        def __init__(self, fail):
            self.fail = fail

        def list(self, scope=None, status=None, cwd=None):
            if self.fail:
                raise DaemonUnavailable("down")
            return []

    results = [FakeClient(fail=False)]

    def fake_ensure(cfg):
        if not made:
            made.append("err")
            raise DaemonUnavailable("cannot start")
        c = results[0]
        made.append(c)
        return c

    captured = {}

    def fake_serve(tools, stdin, stdout, *, instructions):
        captured["tools"] = {t.name: t for t in tools}
        return 0

    monkeypatch.setattr(mt, "ensure_daemon", fake_ensure)
    monkeypatch.setattr(mt, "serve", fake_serve)
    assert mt.run_mcp(object()) == 0
    lm = captured["tools"]["list_memories"]
    assert lm.handler({}).text == "cannot start"
    assert lm.handler({}).text == "No memories found."
    assert lm.handler({}).text == "No memories found."
    assert made == ["err", results[0]]  # cached after the first success

    results[0].fail = True  # the daemon went away: the cache is cleared, the next call reconnects
    assert lm.handler({}).is_error
    results[0].fail = False
    assert lm.handler({}).text == "No memories found."
    assert made == ["err", results[0], results[0]]


def _rpc(proc, msg):
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    if "id" not in msg:
        return None
    return json.loads(proc.stdout.readline())


def test_rem_mcp_subprocess_roundtrip(client, daemon_env, tmp_path):
    client.add("Prefers uv over pip")
    proc = subprocess.Popen([sys.executable, "-m", "rem", "mcp"], env={**daemon_env.env, "CLAUDE_PROJECT_DIR": str(tmp_path)},
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        r = _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                   "clientInfo": {"name": "t", "version": "1"}}})
        assert r["result"]["serverInfo"]["name"] == "rem"
        assert r["result"]["instructions"] == INSTRUCTIONS
        assert _rpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
        r = _rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert {t["name"] for t in r["result"]["tools"]} == NAMES
        r = _rpc(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": "search_memory", "arguments": {"query": "uv pip"}}})
        assert r["id"] == 3 and r["result"]["isError"] is False
        assert "Prefers uv over pip" in r["result"]["content"][0]["text"]
        proc.stdin.close()
        assert proc.wait(timeout=10) == 0
        stderr = proc.stderr.read()
        assert "Traceback" not in stderr
        assert proc.stdout.read() == ""
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _spawn_mcp(daemon_env, **kw):
    return subprocess.Popen([sys.executable, "-m", "rem", "mcp"], env=daemon_env.env, stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE, **kw)


def test_rem_mcp_sigint_exits_quietly(daemon_env):
    import signal

    proc = _spawn_mcp(daemon_env, stdout=subprocess.PIPE)
    proc.stdin.write(b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n')
    proc.stdin.flush()
    assert json.loads(proc.stdout.readline())["result"] == {}  # serving, imports done
    proc.send_signal(signal.SIGINT)
    assert proc.wait(timeout=10) == 0
    assert b"Traceback" not in proc.stderr.read()
    proc.stdin.close()


def test_rem_mcp_closed_stdout_exits_quietly(daemon_env):
    import os

    r, w = os.pipe()
    proc = _spawn_mcp(daemon_env, stdout=w)
    os.close(w)
    os.close(r)  # the client went away: every write is a broken pipe
    try:
        proc.stdin.write(b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n' * 50)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    assert proc.wait(timeout=10) == 0
    err = proc.stderr.read()
    assert b"Traceback" not in err and b"BrokenPipe" not in err
