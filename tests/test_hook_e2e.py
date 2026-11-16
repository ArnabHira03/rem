"""The hook flow across real processes: a daemon subprocess and `python -m rem hook claude-code` subprocesses.

No [llm] is configured, so the worker never runs and jobs stay pending; that is what we inspect.
"""

import hashlib
import json
import sqlite3
import subprocess
import sys
import time

from rem.client import RemClient

from .conftest import _stop_daemon, spawn_daemon, wait_for_socket

PROMPT = "I prefer tabs over spaces; reply OK"
REPLY = "OK"


def payload(event, cwd, **kw):
    return {"session_id": "sess-e2e", "transcript_path": "/nonexistent/t.jsonl", "cwd": str(cwd),
            "hook_event_name": event, **kw}


def run_hook(env, p):
    r = subprocess.run([sys.executable, "-m", "rem", "hook", "claude-code"], input=json.dumps(p), text=True,
                       capture_output=True, env=env.env, timeout=60)
    assert r.returncode == 0
    assert r.stderr == ""
    return r.stdout


def query(env, sql, *args):
    con = sqlite3.connect(f"file:{env.root / 'rem.db'}?mode=ro", uri=True)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def wait_for(fn, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = fn()
        if v:
            return v
        time.sleep(0.1)
    raise AssertionError("condition not met in time")


def test_hook_flow_across_processes(daemon_env, tmp_path):
    env = daemon_env
    proc = spawn_daemon(env)
    try:
        wait_for_socket(env.sock)
        work = tmp_path / "work"
        work.mkdir()

        # UserPromptSubmit then Stop: capture hooks print nothing, one pending job holds user then assistant
        assert run_hook(env, payload("UserPromptSubmit", work, prompt_id="p1", prompt=PROMPT)) == ""
        assert query(env, "SELECT COUNT(*) FROM jobs") == [(0,)]  # the prompt turn is held, no job yet
        assert run_hook(env, payload("Stop", work, prompt_id="p1", stop_hook_active=False,
                                     last_assistant_message=REPLY)) == ""
        jobs = query(env, "SELECT id, status FROM jobs")
        assert len(jobs) == 1 and jobs[0][1] == "pending"
        # messages are bound to a job only when the worker claims it; until then the pending job owns the unclaimed ones
        msgs = query(env, "SELECT message_id, role, content FROM messages WHERE job_id IS NULL ORDER BY id")
        reply_id = "cc:p1:assistant:" + hashlib.sha256(REPLY.encode()).hexdigest()[:12]
        assert msgs == [("cc:p1:user", "user", PROMPT), (reply_id, "assistant", REPLY)]
        assert query(env, "SELECT COUNT(*) FROM messages") == [(2,)]

        # SessionEnd with nothing held is harmless
        assert run_hook(env, payload("SessionEnd", work, reason="other")) == ""

        # SessionStart prints the context of verified memories
        RemClient(env.sock).add("Prefers tabs over spaces")
        out = run_hook(env, payload("SessionStart", work, source="startup"))
        assert out == "Facts rem remembers from earlier sessions (verified):\nAbout the user:\n- Prefers tabs over spaces"
    finally:
        _stop_daemon(env)
        proc.wait(timeout=10)


def test_spooled_when_daemon_down_then_drained(daemon_env, tmp_path):
    env = daemon_env
    work = tmp_path / "work"
    work.mkdir()
    spool = env.root / "spool"

    # A fresh start stamp makes the hook back off, so it spools without spawning a daemon.
    stamp = env.root / "daemon-start.stamp"
    stamp.touch()
    before = stamp.stat().st_mtime_ns
    assert run_hook(env, payload("UserPromptSubmit", work, prompt_id="p9", prompt="remember the spool")) == ""
    files = list(spool.glob("*.json"))
    assert len(files) == 1
    body = json.loads(files[0].read_text())
    assert body["source"] == "claude-code" and body["session_id"] == "sess-e2e"
    assert [t["content"] for t in body["turns"]] == ["remember the spool"]
    assert stamp.stat().st_mtime_ns == before  # no daemon was spawned
    assert not env.sock.exists()

    proc = spawn_daemon(env)
    try:
        wait_for_socket(env.sock)
        wait_for(lambda: not list(spool.glob("*.json")))
        rows = wait_for(lambda: query(env, "SELECT m.message_id, m.content FROM messages m "
                                            "JOIN sessions s ON s.id = m.session_pk WHERE s.session_id = 'sess-e2e'"))
        assert rows == [("cc:p9:user", "remember the spool")]
    finally:
        _stop_daemon(env)
        proc.wait(timeout=10)
