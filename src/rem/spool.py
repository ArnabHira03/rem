"""Spool for turns captured while the daemon is down.

A hook writes the (already redacted) /ingest body here; the daemon drains it at startup and
then periodically while it runs. The spool is capped at SPOOL_MAX_FILES files.
"""

import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

from rem.config import ensure_private_dir
from rem.errors import RemError

log = logging.getLogger("rem.spool")

SPOOL_MAX_FILES = 1000


class SpoolFull(RemError):
    """The spool already holds SPOOL_MAX_FILES files (the daemon has not drained it for a long time)."""


def spool_count(data_dir: Path) -> int:
    """How many bodies are waiting in the spool (quarantined spool/bad files not counted)."""
    try:
        return sum(1 for _ in (data_dir / "spool").glob("*.json"))
    except OSError:
        return 0


def _spool_dir(data_dir: Path) -> Path:
    ensure_private_dir(data_dir)
    d = data_dir / "spool"
    d.mkdir(mode=0o700, exist_ok=True)
    os.chmod(d, 0o700)
    return d


def write_spool(data_dir: Path, body: dict) -> Path:
    if spool_count(data_dir) >= SPOOL_MAX_FILES:
        raise SpoolFull(f"spool holds {SPOOL_MAX_FILES} files or more")
    d = _spool_dir(data_dir)
    name = f"{time.time_ns()}-{os.getpid()}"
    tmp, final = d / f".{name}.tmp", d / f"{name}.json"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(body, f)
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return final


def _quarantine(path: Path) -> None:
    bad = path.parent / "bad"
    ensure_private_dir(bad)
    os.replace(path, bad / path.name)


def prune_bad(data_dir: Path, keep: int = 100) -> int:
    """Delete the oldest spool/bad files beyond `keep` (names are time-ordered); returns how many."""
    bad = data_dir / "spool" / "bad"
    try:
        files = sorted(bad.glob("*.json"))
    except OSError:
        return 0
    removed = 0
    for path in files[: max(0, len(files) - keep)]:
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def drain_spool(data_dir: Path, ingest: Callable[[dict], None],
                failures: dict[str, int] | None = None, max_failures: int = 3) -> int:
    """Ingest every spooled body in name order; returns how many were ingested.

    Unparseable files and bodies `ingest` rejects with ValueError move to spool/bad/.
    Any other error stops the drain, leaving the rest for the next round, and counts against
    that file in `failures`; at `max_failures` the file is quarantined and the drain continues.
    """
    d = data_dir / "spool"
    if not d.is_dir():
        return 0
    if failures is None:
        failures = {}
    done = 0
    for path in sorted(d.glob("*.json")):
        try:
            body = json.loads(path.read_text())
            if not isinstance(body, dict):
                raise ValueError("spool body must be a JSON object")
            ingest(body)
        except ValueError as e:  # includes json.JSONDecodeError, UnicodeDecodeError, pydantic ValidationError
            log.warning("bad spool file %s: %s", path.name, type(e).__name__)
            failures.pop(path.name, None)
            _quarantine(path)
            continue
        except Exception as e:
            failures[path.name] = failures.get(path.name, 0) + 1
            if failures[path.name] >= max_failures:
                log.warning("bad spool file %s: %s (failed %d times)", path.name, type(e).__name__, failures[path.name])
                failures.pop(path.name, None)
                _quarantine(path)
                continue
            log.error("spool drain stopped at %s: %s", path.name, type(e).__name__)
            break
        failures.pop(path.name, None)
        path.unlink(missing_ok=True)
        done += 1
    prune_bad(data_dir)
    return done
