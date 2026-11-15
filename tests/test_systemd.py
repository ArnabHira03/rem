import subprocess

import pytest

from rem.errors import RemError
from rem.systemd import install_unit, render_unit


def test_unit_content():
    u = render_unit("/usr/bin/python3")
    assert "ExecStart=/usr/bin/python3 -m rem daemon" in u
    assert "Restart=on-failure" in u and "RestartPreventExitStatus=2 3" in u
    assert "WantedBy=default.target" in u


def test_install_writes_under_xdg_config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    p = install_unit(enable=False)
    assert p == tmp_path / "systemd/user/rem.service" and p.read_text() == render_unit()


def test_install_enable_reports_systemctl_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(
        "rem.systemd.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "Failed to connect to bus"),
    )
    with pytest.raises(RemError, match="Failed to connect to bus"):
        install_unit(enable=True)


def test_cli_install_service_prints_next_step(tmp_path, monkeypatch, capsys):
    from rem.cli import main

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert main(["install-service"]) == 0
    out = capsys.readouterr().out
    assert str(tmp_path / "systemd/user/rem.service") in out and "systemctl --user enable --now rem.service" in out
