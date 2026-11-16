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


def test_missing_api_key_exits_3(daemon_env):
    daemon_env.config.write_text(
        daemon_env.config.read_text()
        + '[llm]\nprovider = "anthropic"\nmodel = "claude-haiku-5-5"\nallow_remote = true\napi_key_env = "REM_TEST_NO_SUCH_KEY"\n'
    )
    daemon_env.env.pop("REM_TEST_NO_SUCH_KEY", None)
    assert spawn_daemon(daemon_env).wait(10) == 3
    assert "set REM_TEST_NO_SUCH_KEY in the daemon's environment" in daemon_env.log.read_text()
    assert InstanceLock.holder_pid(daemon_env.lock) is None


def test_embedder_mismatch_exits_3(daemon_env):
    p = spawn_daemon(daemon_env)
    wait_for_socket(daemon_env.sock)
    p.terminate()
    p.wait(10)
    daemon_env.config.write_text(daemon_env.config.read_text().replace("hash-64", "hash-32"))
    assert spawn_daemon(daemon_env).wait(10) == 3
    assert "rem reembed" in daemon_env.log.read_text()


STUCK_LLM_DAEMON = """
import sys, threading
from rem.config import ExtractionConfig, load_config
from rem.daemon.extraction import ExtractionHandler
from rem.daemon.server import run_daemon
from rem.llm.fake import FakeLLM

class StuckLLM(FakeLLM):
    def complete_json(self, system, user, schema):
        print("llm call entered", file=sys.stderr, flush=True)
        threading.Event().wait()

sys.exit(run_daemon(load_config(), ExtractionHandler(StuckLLM([]), ExtractionConfig())))
"""


def test_sigterm_exits_promptly_with_llm_call_in_flight(daemon_env):
    import subprocess
    import sys
    import time

    script = daemon_env.root / "stuck.py"
    script.write_text(STUCK_LLM_DAEMON)
    log = open(daemon_env.log, "a")
    p = subprocess.Popen([sys.executable, str(script)], env=daemon_env.env, stdout=log, stderr=log)
    try:
        wait_for_socket(daemon_env.sock)
        uds_client(daemon_env.sock).post(
            "/ingest", json={"source": "cli", "session_id": "s", "turns": [{"role": "user", "content": "hi"}]}
        )
        deadline = time.time() + 5
        while "llm call entered" not in daemon_env.log.read_text():
            assert time.time() < deadline
            time.sleep(0.05)
        p.send_signal(signal.SIGTERM)
        assert p.wait(10) == 0  # grace is lease/2 = 1s; the stuck call must not hold process exit
        assert InstanceLock.holder_pid(daemon_env.lock) is None
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()


def test_spool_is_drained_once_at_startup(daemon_env):
    from rem.spool import write_spool

    write_spool(daemon_env.root, {"source": "claude-code", "session_id": "sp", "cwd": None,
                                  "turns": [{"role": "user", "content": "hello from the spool"}],
                                  "enqueue": True})
    write_spool(daemon_env.root, {"source": "claude-code", "turns": "invalid"})
    p = spawn_daemon(daemon_env)
    wait_for_socket(daemon_env.sock)
    assert uds_client(daemon_env.sock).get("/status").json()["jobs"]["pending"] == 1
    spool = daemon_env.root / "spool"
    assert list(spool.glob("*.json")) == []
    assert len(list((spool / "bad").iterdir())) == 1
    p.terminate()
    assert p.wait(10) == 0

    p2 = spawn_daemon(daemon_env)
    wait_for_socket(daemon_env.sock)
    assert uds_client(daemon_env.sock).get("/status").json()["jobs"]["pending"] == 1
    assert daemon_env.log.read_text().count("drained 1 spooled") == 1
    p2.terminate()
    assert p2.wait(10) == 0


SPOOL_PATCH_DAEMON = """
import sys
from rem.config import load_config
import rem.daemon.server as server
from rem.daemon.service import RemService
from rem.spool import write_spool

MODE = sys.argv[1]
real_drain, real_close, calls = server.drain_spool, RemService.close, []

def drain(data_dir, ingest, failures=None):
    calls.append(1)
    if MODE == "raise":
        raise OSError("disk gone")
    n = real_drain(data_dir, ingest, failures)
    if len(calls) == 1:  # a hook spools while we are still starting up
        write_spool(data_dir, {"source": "claude-code", "session_id": "late", "cwd": None,
                               "turns": [{"role": "user", "content": "late turn"}], "enqueue": True})
    return n

def close(self):
    print("service closed", file=sys.stderr, flush=True)
    real_close(self)

server.drain_spool = drain
RemService.close = close
sys.exit(server.run_daemon(load_config()))
"""


def _spawn_patched(daemon_env, mode):
    import subprocess
    import sys

    script = daemon_env.root / "patched.py"
    script.write_text(SPOOL_PATCH_DAEMON)
    log = open(daemon_env.log, "a")
    return subprocess.Popen([sys.executable, str(script), mode], env=daemon_env.env, stdout=log, stderr=log)


def test_spool_file_written_during_startup_is_drained_without_restart(daemon_env):
    import time

    p = _spawn_patched(daemon_env, "late")
    try:
        wait_for_socket(daemon_env.sock)
        c = uds_client(daemon_env.sock)
        deadline = time.time() + 5
        while c.get("/status").json()["jobs"]["pending"] != 1 and time.time() < deadline:
            time.sleep(0.05)
        assert c.get("/status").json()["jobs"]["pending"] == 1
        assert list((daemon_env.root / "spool").glob("*.json")) == []
    finally:
        p.terminate()
        assert p.wait(10) == 0


def test_startup_drain_failure_still_closes_service_and_releases_lock(daemon_env):
    p = _spawn_patched(daemon_env, "raise")
    assert p.wait(20) != 0
    assert "service closed" in daemon_env.log.read_text()
    assert InstanceLock.holder_pid(daemon_env.lock) is None


FAST_POLL_DAEMON = """
import sys
from rem.config import load_config
import rem.daemon.server as server

server.SPOOL_POLL_SECONDS = 0.2
sys.exit(server.run_daemon(load_config()))
"""


def test_spool_file_written_while_serving_is_drained_without_restart(daemon_env):
    import subprocess
    import sys
    import time

    from rem.spool import write_spool

    script = daemon_env.root / "fastpoll.py"
    script.write_text(FAST_POLL_DAEMON)
    p = subprocess.Popen([sys.executable, str(script)], env=daemon_env.env,
                         stdout=open(daemon_env.log, "a"), stderr=subprocess.STDOUT)
    try:
        pid = wait_for_socket(daemon_env.sock)
        c = uds_client(daemon_env.sock)
        time.sleep(0.5)  # well past the first drain: only the periodic one can pick this up
        assert c.get("/status").json()["jobs"]["pending"] == 0
        write_spool(daemon_env.root, {"source": "claude-code", "session_id": "timeout", "cwd": None,
                                      "turns": [{"role": "user", "content": "spooled after a timeout"}],
                                      "enqueue": True})
        deadline = time.time() + 5
        while c.get("/status").json()["jobs"]["pending"] != 1 and time.time() < deadline:
            time.sleep(0.05)
        assert c.get("/status").json()["jobs"]["pending"] == 1
        assert list((daemon_env.root / "spool").glob("*.json")) == []
        assert c.get("/status").json()["pid"] == pid  # same daemon, no restart
    finally:
        p.terminate()
        assert p.wait(10) == 0
