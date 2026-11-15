"""SQLite side of rem's storage: metadata, projects and the history audit log.

Current memory text lives in the vector store; SQLite only keeps history old/new values.
"""

import sqlite3
from pathlib import Path

from rem.models import HistoryEvent, HistoryRow, TargetKind, utcnow

SCHEMA_VERSION = 1

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
        for version in range(current + 1, SCHEMA_VERSION + 1):
            self._conn.executescript(f"BEGIN; {_MIGRATIONS[version]} COMMIT;")
            self.set_meta("schema_version", str(version))

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

    def close(self) -> None:
        self._conn.close()
