"""One rem daemon per data directory, enforced with flock on <data_dir>/rem.lock.

The kernel releases the lock when the holder exits, even after kill -9, so a
crashed daemon never blocks the next start.
"""

import fcntl
import os
from pathlib import Path

from rem.errors import RemError


class AlreadyRunning(RemError):
    def __init__(self, pid: int | None):
        super().__init__(f"rem daemon already running (pid {pid})")
        self.pid = pid


def _read_pid(fd: int) -> int | None:
    os.lseek(fd, 0, os.SEEK_SET)
    raw = os.read(fd, 32).decode().strip()
    return int(raw) if raw.isdigit() else None


class InstanceLock:
    def __init__(self, path: Path):
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pid = _read_pid(fd)
            os.close(fd)
            raise AlreadyRunning(pid) from None
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        self._fd = fd

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    @staticmethod
    def holder_pid(path: Path) -> int | None:
        """PID of the running daemon, or None when nobody holds the lock."""
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            pid = _read_pid(fd)
            os.close(fd)
            return pid
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        return None
