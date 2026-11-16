import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectInfo:
    root_path: str
    name: str
    git_remote: str | None


def project_root(cwd: Path) -> Path:
    """The git root containing cwd, or cwd itself outside git. Filesystem only, no subprocess."""
    start = cwd.resolve()
    return next((d for d in (start, *start.parents) if (d / ".git").exists()), start)


def resolve_project(cwd: Path) -> ProjectInfo:
    """The project a directory belongs to, with its origin remote (runs git: never call on the store thread)."""
    root = project_root(cwd)
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
