import subprocess
import sys
from pathlib import Path

import pytest

from rem.config import DaemonConfig, RemConfig, load_config


def test_socket_path_resolution_order(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    monkeypatch.delenv("REM_SOCKET", raising=False)
    cfg = RemConfig(data_dir=tmp_path)
    assert cfg.socket_path == Path("/run/user/1000/rem.sock")
    monkeypatch.setenv("REM_SOCKET", "/tmp/x.sock")
    assert cfg.socket_path == Path("/tmp/x.sock")
    cfg2 = cfg.model_copy(update={"daemon": DaemonConfig(socket_path=Path("/tmp/y.sock"))})
    assert cfg2.socket_path == Path("/tmp/y.sock")
    monkeypatch.delenv("REM_SOCKET")
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    assert cfg.socket_path == tmp_path / "rem.sock"


def test_socket_path_too_long_is_clear(monkeypatch):
    monkeypatch.delenv("REM_SOCKET", raising=False)
    cfg = RemConfig(data_dir=Path("/tmp"), daemon=DaemonConfig(socket_path=Path("/tmp/" + "a" * 120)))
    with pytest.raises(ValueError, match="socket path too long"):
        cfg.socket_path


def test_lock_and_log_paths(tmp_path):
    cfg = RemConfig(data_dir=tmp_path)
    assert cfg.lock_path == tmp_path / "rem.lock" and cfg.log_path == tmp_path / "daemon.log"


def test_daemon_section_from_toml(tmp_path, monkeypatch):
    monkeypatch.setenv("REM_HOME", str(tmp_path))
    p = tmp_path / "c.toml"
    p.write_text("[daemon]\ndebounce_seconds = 1\n")
    assert load_config(p).daemon.debounce_seconds == 1


def test_python_dash_m_runs_cli():
    out = subprocess.run([sys.executable, "-m", "rem", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "rem" in out.stdout
