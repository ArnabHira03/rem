import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectInfo:
    root_path: str
    name: str
    git_remote: str | None


def resolve_project(cwd: Path) -> ProjectInfo:
    """The project a directory belongs to: its git root, or the directory itself outside git."""
    start = cwd.resolve()
    root = next((d for d in (start, *start.parents) if (d / ".git").exists()), start)
    remote = None
    if (root / ".git").exists():
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=2,
            )
            remote = out.stdout.strip() or None if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            remote = None
    return ProjectInfo(root_path=str(root), name=root.name, git_remote=remote)
