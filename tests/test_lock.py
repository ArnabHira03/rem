import os
import subprocess
import sys

import pytest

from rem.daemon.lock import AlreadyRunning, InstanceLock


def test_second_acquire_fails_with_pid(tmp_path):
    a = InstanceLock(tmp_path / "rem.lock")
    a.acquire()
    with pytest.raises(AlreadyRunning, match=f"pid {os.getpid()}"):
        InstanceLock(tmp_path / "rem.lock").acquire()
    assert InstanceLock.holder_pid(tmp_path / "rem.lock") == os.getpid()
    a.release()
    assert InstanceLock.holder_pid(tmp_path / "rem.lock") is None


def test_holder_pid_when_no_file(tmp_path):
    assert InstanceLock.holder_pid(tmp_path / "missing.lock") is None


def test_lock_released_when_holder_process_dies(tmp_path):
    path = tmp_path / "rem.lock"
    code = (
        "import pathlib, time; from rem.daemon.lock import InstanceLock; "
        f"InstanceLock(pathlib.Path({str(path)!r})).acquire(); print('ok', flush=True); time.sleep(60)"
    )
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ok"
    assert InstanceLock.holder_pid(path) == p.pid
    p.kill()
    p.wait()
    InstanceLock(path).acquire()
