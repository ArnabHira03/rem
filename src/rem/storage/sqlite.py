"""SQLite side of rem's storage: metadata, projects and the history audit log.

Current memory text lives in the vector store; SQLite only keeps history old/new values.
"""

import sqlite3
from pathlib import Path

from rem.errors import RemError
from rem.models import HistoryEvent, HistoryRow, TargetKind, utcnow

SCHEMA_VERSION = 4

_MIGRATIONS = {
    1: """
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY,
        root_path TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        git_remote TEXT
    );
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY,
        target_kind TEXT NOT NULL,
        target_id TEXT NOT NULL,
        event TEXT NOT NULL,
        old_value TEXT,
        new_value TEXT,
        actor TEXT NOT NULL,
        job_id TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (job_id, target_id, event)
    );
    CREATE INDEX IF NOT EXISTS history_target ON history (target_id);
    """,
    2: """
    CREATE TABLE sessions (
        id INTEGER PRIMARY KEY,
        source TEXT NOT NULL,
        session_id TEXT NOT NULL,
        project_id INTEGER,
        cwd TEXT,
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        UNIQUE (source, session_id)
    );
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY,
        session_pk INTEGER NOT NULL REFERENCES sessions(id),
        message_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        ts TEXT,
        job_id INTEGER,
        UNIQUE (session_pk, message_id)
    );
    CREATE TABLE jobs (
        id INTEGER PRIMARY KEY,
        session_pk INTEGER NOT NULL REFERENCES sessions(id),
        status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'done', 'failed')),
        attempts INTEGER NOT NULL DEFAULT 0,
        lease_until REAL,
        not_before REAL NOT NULL,
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX jobs_status ON jobs (status, not_before);
    CREATE INDEX messages_job ON messages (session_pk, job_id);
    """,
    3: """
    CREATE TABLE extractions (
        job_id INTEGER PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
        model TEXT NOT NULL,
        candidates TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    # When a job last took new turns (its exchange's end); updated_at also moves on claim and retry.
    4: """
    ALTER TABLE jobs ADD COLUMN enqueued_at TEXT;
    UPDATE jobs SET enqueued_at = created_at;
    """,
}


class RemDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self) -> None:
        self._conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        current = int(self.get_meta("schema_version") or 0)
        if current > SCHEMA_VERSION:
            raise RemError(f"rem.db was written by a newer rem (schema {current}); upgrade rem")
        for version in range(current + 1, SCHEMA_VERSION + 1):
            # DDL and the version bump commit together, or not at all.
            try:
                self._conn.executescript(
                    f"BEGIN; {_MIGRATIONS[version]}\n"
                    f"INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '{version}'); COMMIT;"
                )
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def upsert_project(self, root_path: str, name: str, git_remote: str | None) -> int:
        self._conn.execute(
            "INSERT INTO projects (root_path, name, git_remote) VALUES (?, ?, ?) "
            "ON CONFLICT(root_path) DO UPDATE SET name = excluded.name, "
            "git_remote = COALESCE(excluded.git_remote, projects.git_remote)",
            (root_path, name, git_remote),
        )
        return self.find_project(root_path)  # type: ignore[return-value]

    def find_project(self, root_path: str) -> int | None:
        row = self._conn.execute("SELECT id FROM projects WHERE root_path = ?", (root_path,)).fetchone()
        return row["id"] if row else None

    def get_project(self, id: int) -> tuple[str, str, str | None] | None:
        row = self._conn.execute("SELECT root_path, name, git_remote FROM projects WHERE id = ?", (id,)).fetchone()
        return (row["root_path"], row["name"], row["git_remote"]) if row else None

    def add_history(
        self,
        target_kind: TargetKind,
        target_id: str,
        event: HistoryEvent,
        old_value: str | None,
        new_value: str | None,
        actor: str,
        job_id: str | None = None,
    ) -> bool:
        """Append a history row. Returns False when (job_id, target_id, event) was already recorded."""
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO history "
            "(target_kind, target_id, event, old_value, new_value, actor, job_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(target_kind), target_id, str(event), old_value, new_value, actor, job_id, utcnow()),
        )
        return cur.rowcount == 1

    def get_history(self, target_id: str) -> list[HistoryRow]:
        rows = self._conn.execute(
            "SELECT * FROM history WHERE target_id = ? ORDER BY id", (target_id,)
        ).fetchall()
        return [HistoryRow(**dict(r)) for r in rows]

    def save_extraction(self, job_id: int, model: str, candidates_json: str) -> None:
        """Cache a job's extraction. The first write wins, so a replay reuses the original candidates."""
        self._conn.execute(
            "INSERT OR IGNORE INTO extractions (job_id, model, candidates, created_at) VALUES (?, ?, ?, ?)",
            (job_id, model, candidates_json, utcnow()),
        )

    def get_extraction(self, job_id: int) -> tuple[str, str] | None:
        """(model, candidates_json) cached for a job, if any."""
        row = self._conn.execute("SELECT model, candidates FROM extractions WHERE job_id = ?", (job_id,)).fetchone()
        return (row["model"], row["candidates"]) if row else None

    def close(self) -> None:
        self._conn.close()
