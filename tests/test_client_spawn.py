"""Starting the daemon from a client: file modes, working directory and the hook's start backoff."""

import os
import stat
import time

import pytest

from rem import client
from rem.config import RemConfig


class FakePopen:
    calls: list[dict] = []

    def __init__(self, argv, **kw):
        FakePopen.calls.append({"argv": argv, **kw})

    def poll(self):
        return 1  # exited with an error


@pytest.fixture
def popen(monkeypatch):
    FakePopen.calls = []
    monkeypatch.setattr(client.subprocess, "Popen", FakePopen)
    return FakePopen.calls


@pytest.fixture
def scfg(tmp_path):
    return RemConfig(data_dir=tmp_path / "data")


def test_spawn_under_umask_022_makes_private_data_dir_and_log(scfg, popen):
    old = os.umask(0o022)
    try:
        client._spawn_daemon(scfg)
    finally:
        os.umask(old)
    assert stat.S_IMODE(scfg.data_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(scfg.log_path.stat().st_mode) == 0o600


def test_spawn_runs_daemon_from_root_detached(scfg, popen):
    client._spawn_daemon(scfg)
    (call,) = popen
    assert call["argv"][1:] == ["-m", "rem", "daemon"]
    assert call["cwd"] == "/" and call["start_new_session"] is True


def test_background_start_backs_off_for_a_minute(scfg, popen):
    assert client.start_daemon_background(scfg) is True
    stamp = scfg.data_dir / "daemon-start.stamp"
    assert stat.S_IMODE(stamp.stat().st_mode) == 0o600
    assert client.start_daemon_background(scfg) is False  # every hook in the next minute skips the spawn
    assert len(popen) == 1
    old = time.time() - client.START_BACKOFF_SECONDS - 1
    os.utime(stamp, (old, old))
    assert client.start_daemon_background(scfg) is True
    assert len(popen) == 2
    assert stamp.stat().st_mtime > old + 1  # the stamp was refreshed


def test_ensure_daemon_ignores_the_backoff(scfg, popen, monkeypatch):
    from rem.config import DaemonConfig

    monkeypatch.setenv("REM_SOCKET", "/tmp/rem-test-spawn-none.sock")
    scfg.daemon = DaemonConfig(start_timeout_seconds=1)
    client.start_daemon_background(scfg)  # fresh stamp
    with pytest.raises(client.DaemonUnavailable, match="failed to start"):
        client.ensure_daemon(scfg)  # FakePopen exits 1 at once: the CLI shows the real error
    assert len(popen) == 2
