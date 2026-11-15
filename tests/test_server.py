import os
import signal

from rem.daemon.lock import InstanceLock
from tests.conftest import spawn_daemon, uds_client, wait_for_socket


def test_start_serves_status_and_sigterm_cleans_up(daemon_env):
    p = spawn_daemon(daemon_env)
    assert wait_for_socket(daemon_env.sock) == p.pid
    assert oct(os.stat(daemon_env.sock).st_mode & 0o777) == "0o600"
    p.send_signal(signal.SIGTERM)
    assert p.wait(10) == 0
    assert not daemon_env.sock.exists() and InstanceLock.holder_pid(daemon_env.lock) is None


def test_second_daemon_exits_2(daemon_env):
    p = spawn_daemon(daemon_env)
    wait_for_socket(daemon_env.sock)
    assert spawn_daemon(daemon_env).wait(10) == 2
    assert uds_client(daemon_env.sock).get("/status").json()["pid"] == p.pid
    p.terminate()
    p.wait(10)


def test_kill_9_then_restart_removes_stale_socket(daemon_env):
    p = spawn_daemon(daemon_env)
    wait_for_socket(daemon_env.sock)
    uds_client(daemon_env.sock).post("/memories", json={"text": "Prefers uv"})
    p.kill()
    p.wait()
    assert daemon_env.sock.exists()
    p2 = spawn_daemon(daemon_env)
    wait_for_socket(daemon_env.sock, fresh_after=p.pid)
    assert [m["text"] for m in uds_client(daemon_env.sock).get("/memories").json()] == ["Prefers uv"]
    p2.terminate()
    p2.wait(10)


def test_ingested_job_stays_pending_without_llm(daemon_env):
    import time

    p = spawn_daemon(daemon_env)
    wait_for_socket(daemon_env.sock)
    c = uds_client(daemon_env.sock)
    c.post("/ingest", json={"source": "cli", "session_id": "s", "turns": [{"role": "user", "content": "hi"}]})
    time.sleep(0.6)  # past the 0.2s debounce: a running worker would have claimed the job by now
    status = c.get("/status").json()
    assert status["jobs"]["pending"] == 1 and status["jobs"]["done"] == 0
    assert status["llm"] == {"provider": "none"}
    p.terminate()
    assert p.wait(10) == 0
    assert "extraction disabled: no [llm] configured; jobs stay pending" in daemon_env.log.read_text()


def test_remote_llm_without_allow_remote_exits_3(daemon_env):
    daemon_env.config.write_text(
        daemon_env.config.read_text() + '[llm]\nprovider = "anthropic"\nmodel = "claude-haiku-5-5"\n'
    )
    assert spawn_daemon(daemon_env).wait(10) == 3
    assert "allow_remote" in daemon_env.log.read_text()
    assert InstanceLock.holder_pid(daemon_env.lock) is None
    assert not daemon_env.sock.exists()


def test_embedder_mismatch_exits_3(daemon_env):
    p = spawn_daemon(daemon_env)
    wait_for_socket(daemon_env.sock)
    p.terminate()
    p.wait(10)
    daemon_env.config.write_text(daemon_env.config.read_text().replace("hash-64", "hash-32"))
    assert spawn_daemon(daemon_env).wait(10) == 3
    assert "rem reembed" in daemon_env.log.read_text()
