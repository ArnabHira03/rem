"""Conversation ingestion and the durable job queue (SQLite, inside rem.db).

Ingest is idempotent per (session, message_id) and debounced per session: a burst
of hook calls becomes one pending job. Jobs are claimed under a lease so a crashed
run is picked up again with exactly the same messages.
"""

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rem.config import DaemonConfig
from rem.models import utcnow
from rem.projects import resolve_project
from rem.redact import redact
from rem.storage.sqlite import RemDB

STATUSES = ("pending", "processing", "done", "failed")
MAX_BACKOFF_SECONDS = 300


@dataclass
class Turn:
    role: str
    content: str
    message_id: str | None = None
    ts: str | None = None


@dataclass
class Job:
    id: int
    session_pk: int
    source: str
    session_id: str
    project_id: int | None
    attempts: int


@dataclass
class IngestResult:
    session_pk: int
    accepted: int
    duplicates: int
    job_id: int | None


def backoff_seconds(attempts: int) -> float:
    return min(5 * 2 ** (attempts - 1), MAX_BACKOFF_SECONDS)


class IngestStore:
    def __init__(self, db: RemDB, settings: DaemonConfig, clock: Callable[[], float] = time.time):
        self.db = db
        self.conn = db.conn
        self.settings = settings
        self.clock = clock

    # ---- ingest -------------------------------------------------------------

    def ingest(self, source: str, session_id: str, cwd: Path | None, turns: list[Turn]) -> IngestResult:
        project_id = None
        if cwd is not None:
            info = resolve_project(cwd)
        now, stamp = self.clock(), utcnow()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            if cwd is not None:
                project_id = self.db.upsert_project(info.root_path, info.name, info.git_remote)
            self.conn.execute(
                "INSERT INTO sessions (source, session_id, project_id, cwd, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(source, session_id) DO UPDATE SET "
                "last_seen = excluded.last_seen, cwd = COALESCE(excluded.cwd, cwd), "
                "project_id = COALESCE(excluded.project_id, project_id)",
                (source, session_id, project_id, str(cwd) if cwd else None, stamp, stamp),
            )
            session_pk = self.conn.execute(
                "SELECT id FROM sessions WHERE source = ? AND session_id = ?", (source, session_id)
            ).fetchone()[0]

            accepted = 0
            for turn in turns:
                content = redact(turn.content)
                mid = turn.message_id or hashlib.sha256(f"{turn.role}\n{content}".encode()).hexdigest()[:32]
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO messages (session_pk, message_id, role, content, ts) VALUES (?, ?, ?, ?, ?)",
                    (session_pk, mid, turn.role, content, turn.ts),
                )
                accepted += cur.rowcount

            job_id = None
            if accepted:
                not_before = now + self.settings.debounce_seconds
                # Only a job that has never been claimed may take new turns: a claimed job's
                # inputs are fixed so that a retry or replay sees exactly the same messages.
                row = self.conn.execute(
                    "SELECT j.id FROM jobs j WHERE j.session_pk = ? AND j.status = 'pending' "
                    "AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.job_id = j.id)",
                    (session_pk,),
                ).fetchone()
                if row:
                    job_id = row[0]
                    self.conn.execute(
                        "UPDATE jobs SET not_before = ?, updated_at = ? WHERE id = ?", (not_before, stamp, job_id)
                    )
                else:
                    job_id = self.conn.execute(
                        "INSERT INTO jobs (session_pk, status, not_before, created_at, updated_at) "
                        "VALUES (?, 'pending', ?, ?, ?)",
                        (session_pk, not_before, stamp, stamp),
                    ).lastrowid
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return IngestResult(session_pk, accepted, len(turns) - accepted, job_id)

    def session_project(self, session_pk: int) -> int | None:
        row = self.conn.execute("SELECT project_id FROM sessions WHERE id = ?", (session_pk,)).fetchone()
        return row[0] if row else None

    # ---- queue --------------------------------------------------------------

    def claim(self) -> Job | None:
        now = self.clock()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                """
                SELECT j.id, j.status FROM jobs j
                WHERE ((j.status = 'pending' AND j.not_before <= :now)
                       OR (j.status = 'processing' AND j.lease_until < :now))
                  -- a session's jobs run strictly in order, one at a time
                  AND NOT EXISTS (
                      SELECT 1 FROM jobs o WHERE o.session_pk = j.session_pk AND o.id < j.id
                      AND o.status IN ('pending', 'processing'))
                  AND NOT EXISTS (
                      SELECT 1 FROM jobs p WHERE p.session_pk = j.session_pk AND p.id != j.id
                      AND p.status = 'processing' AND p.lease_until >= :now)
                ORDER BY j.not_before, j.id LIMIT 1
                """,
                {"now": now},
            ).fetchone()
            if row is None:
                self.conn.execute("COMMIT")
                return None
            job_id = row[0]
            self.conn.execute(
                "UPDATE jobs SET status = 'processing', lease_until = ?, updated_at = ? WHERE id = ?",
                (now + self.settings.lease_seconds, utcnow(), job_id),
            )
            first_claim = not self.conn.execute(
                "SELECT 1 FROM messages WHERE job_id = ? LIMIT 1", (job_id,)
            ).fetchone()
            if first_claim:  # later claims (retry, release, crash) keep the messages it was first given
                self.conn.execute(
                    "UPDATE messages SET job_id = ? WHERE job_id IS NULL AND session_pk = "
                    "(SELECT session_pk FROM jobs WHERE id = ?)",
                    (job_id, job_id),
                )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return self._job(job_id)

    def _job(self, job_id: int) -> Job:
        r = self.conn.execute(
            "SELECT j.id, j.session_pk, s.source, s.session_id, s.project_id, j.attempts "
            "FROM jobs j JOIN sessions s ON s.id = j.session_pk WHERE j.id = ?",
            (job_id,),
        ).fetchone()
        return Job(*r)

    def messages_for(self, job_id: int) -> list[Turn]:
        rows = self.conn.execute(
            "SELECT role, content, message_id, ts FROM messages WHERE job_id = ? ORDER BY id", (job_id,)
        ).fetchall()
        return [Turn(*r) for r in rows]

    def context_for(self, job_id: int, limit: int = 10) -> list[Turn]:
        rows = self.conn.execute(
            """
            SELECT role, content, message_id, ts FROM messages
            WHERE session_pk = (SELECT session_pk FROM jobs WHERE id = :job)
              AND id < COALESCE((SELECT MIN(id) FROM messages WHERE job_id = :job), 0)
            ORDER BY id DESC LIMIT :limit
            """,
            {"job": job_id, "limit": limit},
        ).fetchall()
        return [Turn(*r) for r in reversed(rows)]

    def complete(self, job_id: int) -> None:
        self.conn.execute(
            "UPDATE jobs SET status = 'done', lease_until = NULL, updated_at = ? WHERE id = ?", (utcnow(), job_id)
        )

    def fail(self, job_id: int, error: str) -> None:
        attempts = self.conn.execute("SELECT attempts FROM jobs WHERE id = ?", (job_id,)).fetchone()[0] + 1
        if attempts >= self.settings.max_attempts:
            self.conn.execute(
                "UPDATE jobs SET status = 'failed', attempts = ?, last_error = ?, lease_until = NULL, "
                "updated_at = ? WHERE id = ?",
                (attempts, error[:2000], utcnow(), job_id),
            )
        else:
            self.conn.execute(
                "UPDATE jobs SET status = 'pending', attempts = ?, last_error = ?, lease_until = NULL, "
                "not_before = ?, updated_at = ? WHERE id = ?",
                (attempts, error[:2000], self.clock() + backoff_seconds(attempts), utcnow(), job_id),
            )

    def release(self, job_id: int) -> None:
        self.conn.execute(
            "UPDATE jobs SET status = 'pending', lease_until = NULL, updated_at = ? WHERE id = ?", (utcnow(), job_id)
        )

    def recover(self) -> int:
        """Startup only: the instance lock proves no other worker exists, so every
        'processing' job is orphaned by a crash and goes back to pending now."""
        cur = self.conn.execute(
            "UPDATE jobs SET status = 'pending', lease_until = NULL, updated_at = ? WHERE status = 'processing'",
            (utcnow(),),
        )
        return cur.rowcount

    def counts(self) -> dict[str, int]:
        found = dict(self.conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())
        return {s: found.get(s, 0) for s in STATUSES}
