import json
import os
import subprocess
import sys

from rem.cli import main
from rem.client import RemClient
from rem.daemon.lock import InstanceLock


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
