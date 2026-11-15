import sqlite3

import pytest

import rem.storage.sqlite
from rem.errors import RemError
from rem.storage.sqlite import SCHEMA_VERSION, RemDB

V1_SCHEMA_SQL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE projects (id INTEGER PRIMARY KEY, root_path TEXT UNIQUE NOT NULL, name TEXT NOT NULL, git_remote TEXT);
CREATE TABLE history (
    id INTEGER PRIMARY KEY, target_kind TEXT NOT NULL, target_id TEXT NOT NULL, event TEXT NOT NULL,
    old_value TEXT, new_value TEXT, actor TEXT NOT NULL, job_id TEXT, created_at TEXT NOT NULL,
    UNIQUE (job_id, target_id, event)
);
INSERT INTO meta VALUES ('schema_version', '1');
"""


def _tables(path):
    return {r[0] for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_fresh_db_is_v2_with_queue_tables(tmp_path):
    db = RemDB(tmp_path / "rem.db")
    assert db.get_meta("schema_version") == str(SCHEMA_VERSION)
    assert {"sessions", "messages", "jobs", "history", "projects", "meta"} <= _tables(tmp_path / "rem.db")
    assert db.conn is not None


def test_v1_db_upgrades_in_place(tmp_path):
    p = tmp_path / "rem.db"
    c = sqlite3.connect(p)
    c.executescript(V1_SCHEMA_SQL)
    c.execute("INSERT INTO history (target_kind,target_id,event,actor,created_at) VALUES ('memory','m1','ADD','cli','t')")
    c.commit()
    c.close()
    db = RemDB(p)
    assert db.get_meta("schema_version") == str(SCHEMA_VERSION) and len(db.get_history("m1")) == 1


def test_newer_schema_refused(tmp_path):
    p = tmp_path / "rem.db"
    RemDB(p).set_meta("schema_version", "99")
    with pytest.raises(RemError, match="newer rem"):
        RemDB(p)


def test_failed_migration_leaves_version_unchanged(tmp_path, monkeypatch):
    p = tmp_path / "rem.db"
    c = sqlite3.connect(p)
    c.executescript(V1_SCHEMA_SQL)
    c.commit()
    c.close()
    monkeypatch.setitem(rem.storage.sqlite._MIGRATIONS, 2, "CREATE TABLE sessions (x); THIS IS NOT SQL;")
    with pytest.raises(sqlite3.Error):
        RemDB(p)
    assert sqlite3.connect(p).execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "1"
    assert "sessions" not in _tables(p)
