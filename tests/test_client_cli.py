import json
import os
import subprocess
import sys

import pytest

from rem.cli import main
from rem.client import RemClient
from rem.daemon.lock import InstanceLock
from rem.errors import MemoryNotFound


def test_cli_auto_starts_daemon_and_persists(daemon_env, capsys):
    assert main(["add", "Prefers uv over pip"]) == 0
    pid = InstanceLock.holder_pid(daemon_env.lock)
    assert pid and pid != os.getpid()
    capsys.readouterr()
    assert main(["search", "uv pip", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["text"] == "Prefers uv over pip"


def test_status_start_stop(daemon_env, capsys):
    assert main(["status"]) == 1 and "not running" in capsys.readouterr().out
    assert main(["start"]) == 0 and "running (pid" in capsys.readouterr().out
    assert main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["jobs"]["pending"] == 0
    assert main(["stop"]) == 0 and "stopped" in capsys.readouterr().out
    assert InstanceLock.holder_pid(daemon_env.lock) is None
    assert main(["stop"]) == 0 and "not running" in capsys.readouterr().out


SPOOL_BODY = {"source": "claude-code", "session_id": "s", "cwd": None,
              "turns": [{"role": "user", "content": "hi"}], "enqueue": True}


def test_status_shows_spooled_count_and_hint_with_daemon_down(daemon_env, capsys):
    from rem.spool import write_spool

    assert main(["status"]) == 1
    assert capsys.readouterr().out.splitlines()[1:] == ["spooled: 0", "rem daemon is not running"]
    write_spool(daemon_env.root, SPOOL_BODY)
    write_spool(daemon_env.root, SPOOL_BODY)
    assert main(["status"]) == 1
    assert capsys.readouterr().out.splitlines()[1:] == [
        "spooled: 2",
        f"rem daemon is not running; spooled turns will be ingested when it starts (see {daemon_env.log})",
    ]


def test_status_shows_spooled_count_with_daemon_up(daemon_env, capsys):
    assert main(["start"]) == 0
    capsys.readouterr()
    assert main(["status"]) == 0
    assert capsys.readouterr().out.splitlines()[1] == "spooled: 0"
    assert main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["spooled"] == 0


def test_status_survives_deleted_cwd(daemon_env, tmp_path, monkeypatch, capsys):
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    assert main(["status"]) == 1
    assert capsys.readouterr().out.startswith("claude_code_hooks: none\nspooled: 0\n")


def test_concurrent_cold_start_single_daemon(daemon_env):
    procs = [
        subprocess.Popen([sys.executable, "-m", "rem", "add", f"fact {i}"], env=daemon_env.env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        for i in range(4)
    ]
    results = [(p.wait(30), p.stderr.read()) for p in procs]
    assert [code for code, _ in results] == [0, 0, 0, 0], results
    assert len(RemClient(daemon_env.sock).list()) == 4


def test_failed_auto_start_shows_daemon_error(daemon_env, capsys):
    main(["start"])
    main(["stop"])
    capsys.readouterr()
    daemon_env.config.write_text(daemon_env.config.read_text().replace("hash-64", "hash-32"))
    assert main(["list"]) == 1
    err = capsys.readouterr().err
    assert "failed to start" in err and "rem reembed" in err


def test_errors_keep_m1_exit_codes(daemon_env, capsys):
    assert main(["add", "  "]) == 2
    assert main(["get", "00000000-0000-0000-0000-000000000000"]) == 1
    assert "no memory with id" in capsys.readouterr().err


def test_client_ingest(daemon_env):
    main(["start"])
    r = RemClient(daemon_env.sock).ingest("cli", "s1", None, [{"role": "user", "content": "hi"}])
    assert r["accepted"] == 1


def test_cli_retry_failed(daemon_env, capsys):
    assert main(["retry-failed"]) == 0  # auto-starts the daemon
    assert capsys.readouterr().out.strip().endswith("requeued 0 job(s)")
    assert main(["retry-failed", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"requeued": 0}
    assert RemClient(daemon_env.sock).retry_failed() == 0


def test_parallel_clients_reads_and_writes(daemon_env):
    from concurrent.futures import ThreadPoolExecutor

    main(["start"])

    def work(i):
        c = RemClient(daemon_env.sock)
        c.add(f"fact number {i}")
        return c.search("fact number", None, 5)

    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(work, range(40)))
    assert all(results) and len(RemClient(daemon_env.sock).list()) == 40


def test_non_json_server_error_is_a_daemon_error(daemon_env):
    import httpx
    import pytest

    from rem.errors import RemError

    c = RemClient(daemon_env.sock)
    c._http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500, text="Internal Server Error")),
                           base_url="http://rem")
    with pytest.raises(RemError, match="daemon.log") as e:
        c.list()
    assert not isinstance(e.value, ValueError)


def test_slow_first_start_says_still_starting(daemon_env, monkeypatch):
    import pytest

    from rem.client import DaemonUnavailable, ensure_daemon
    from rem.config import load_config

    cfg = load_config()
    cfg = cfg.model_copy(update={"daemon": cfg.daemon.model_copy(update={"start_timeout_seconds": 0.0})})
    with pytest.raises(DaemonUnavailable, match="still starting"):
        ensure_daemon(cfg)
    # let the daemon finish starting so the fixture's teardown can stop it
    from tests.conftest import wait_for_socket

    wait_for_socket(daemon_env.sock)


def test_cli_reembed_and_finalize(daemon_env, capsys):
    assert main(["reembed", "--model", "hash-32", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["count"] == 0 and out["collection"].endswith("hash-32_32")
    assert main(["reembed", "--status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["model"] == "hash-32"
    assert main(["reembed", "--finalize"]) == 0
    assert "switched" in capsys.readouterr().out


def test_cli_entities_empty(daemon_env, capsys):
    assert main(["entities", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert main(["entities"]) == 0
    assert capsys.readouterr().out == ""


def test_cli_entities_lists_lines_and_json(daemon_env, capsys, monkeypatch):
    ents = [{"id": "e1", "name": "uv", "type": "tool", "memory_ids": ["a", "b"]},
            {"id": "e2", "name": "Rem", "type": "project", "memory_ids": ["c"]}]
    monkeypatch.setattr(RemClient, "list_entities", lambda self: ents)
    assert main(["entities"]) == 0
    assert capsys.readouterr().out.splitlines() == ["e1  uv  tool  2 memories", "e2  Rem  project  1 memories"]
    assert main(["--json", "entities"]) == 0
    assert json.loads(capsys.readouterr().out) == ents


def test_cli_relations_lines_and_json(daemon_env, capsys, monkeypatch):
    base = {"source": "uv", "type": "preferred_over", "target": "pip", "status": "verified", "scope": "global",
            "project_name": None, "evidence": 2, "score": None}
    rels = [{**base, "id": "r1"},
            {**base, "id": "r2", "scope": "project", "project_name": "rem", "score": 0.834}]
    monkeypatch.setattr(RemClient, "relations", lambda self, **kw: rels)
    assert main(["relations"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "r1  uv —preferred_over→ pip  [verified, global, 2 evidence]",
        "r2  uv —preferred_over→ pip  [verified, project rem, 2 evidence]  0.83",
    ]
    assert main(["--json", "relations"]) == 0
    assert json.loads(capsys.readouterr().out) == rels
    assert main(["relations", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == rels


def test_cli_relations_passes_query_entity_scope(daemon_env, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(RemClient, "relations", lambda self, **kw: seen.append(kw) or [])
    monkeypatch.chdir(tmp_path)
    assert main(["relations"]) == 0
    assert main(["relations", "--query", "uv", "--entity", "pip", "--scope", "global", "--limit", "4"]) == 0
    assert seen[0] == {"query": None, "entity": None, "scope": None, "limit": 10, "cwd": None}
    assert seen[1] == {"query": "uv", "entity": "pip", "scope": "global", "limit": 4, "cwd": tmp_path}


def test_client_graph_round_trip(daemon_env):
    assert main(["start"]) == 0
    snap = RemClient(daemon_env.sock).graph(None)
    assert set(snap) == {"generated_at", "current_project_id", "projects", "entities", "relations", "memories"}


def test_cli_graph_without_extra_exits_3(daemon_env, monkeypatch, capsys):
    from rem.graph import window

    def boom():
        raise ImportError("no webview")

    monkeypatch.setattr(window, "import_webview", boom)
    assert main(["graph"]) == 3
    assert "uv sync --extra graph" in capsys.readouterr().err


def test_cli_graph_opens_window_with_cwd(daemon_env, tmp_path, monkeypatch):
    from rem.graph import window

    seen = []
    monkeypatch.setattr(window, "import_webview", lambda: object())
    monkeypatch.setattr(window, "run_window", lambda client, cwd: seen.append((client, cwd)))
    monkeypatch.chdir(tmp_path)
    assert main(["graph"]) == 0
    assert seen[0][1] == tmp_path.resolve()


def test_cli_graph_window_failure_exits_1(daemon_env, monkeypatch, capsys):
    from rem.graph import window

    def fail(client, cwd):
        raise RuntimeError("no display")

    monkeypatch.setattr(window, "import_webview", lambda: object())
    monkeypatch.setattr(window, "run_window", fail)
    assert main(["graph"]) == 1
    err = capsys.readouterr().err
    assert "could not open the graph window: no display" in err
    assert "Traceback" not in err


def test_client_jobs_round_trip(daemon_env):
    main(["start"])
    c = RemClient(daemon_env.sock)
    assert set(c.jobs()) == {"runs", "next_before"}
    assert c.jobs(status="done", flagged=True, limit=5, before=3, project_id=1)["runs"] == []
    with pytest.raises(MemoryNotFound):
        c.job(999)
