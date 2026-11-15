"""systemd user unit for the rem daemon: the normal way it starts on Linux."""

import os
import subprocess
import sys
from pathlib import Path

from rem.errors import RemError

ENABLE_HINT = "systemctl --user daemon-reload && systemctl --user enable --now rem.service"


def render_unit(python: str = sys.executable) -> str:
    return (
        "[Unit]\n"
        "Description=rem personal memory daemon\n"
        "\n"
        "[Service]\n"
        f"ExecStart={python} -m rem daemon\n"
        "Restart=on-failure\n"
        "RestartSec=2\n"
        "RestartPreventExitStatus=2 3\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def unit_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "systemd" / "user" / "rem.service"


def install_unit(enable: bool) -> Path:
    path = unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_unit())
    if enable:
        for cmd in (["systemctl", "--user", "daemon-reload"], ["systemctl", "--user", "enable", "--now", "rem.service"]):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RemError(f"{' '.join(cmd)} failed: {r.stderr.strip()}")
    return path
