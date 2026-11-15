import sqlite3

from rem.storage.sqlite import _MIGRATIONS, SCHEMA_VERSION, RemDB


def _make_v2(path):
    c = sqlite3.connect(path)
    c.executescript(_MIGRATIONS[1] + _MIGRATIONS[2] + "INSERT INTO meta VALUES ('schema_version', '2');")
    c.execute("INSERT INTO sessions (source, session_id, first_seen, last_seen) VALUES ('cc', 's1', 't', 't')")
    c.execute(
        "INSERT INTO jobs (session_pk, status, not_before, created_at, updated_at) VALUES (1, 'pending', 0, 't', 't')"
    )
    c.execute("INSERT INTO history (target_kind,target_id,event,actor,created_at) VALUES ('memory','m1','ADD','cli','t')")
    c.commit()
    c.close()


def test_schema_version_is_3():
    assert SCHEMA_VERSION == 3


def test_v2_db_migrates_to_v3_and_keeps_rows(tmp_path):
    p = tmp_path / "rem.db"
    _make_v2(p)
    db = RemDB(p)
    assert db.get_meta("schema_version") == "3"
    assert len(db.get_history("m1")) == 1
    assert db.conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
    assert db.conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1
    db.save_extraction(1, "m", "[]")
    assert db.get_extraction(1) == ("m", "[]")


def test_save_extraction_first_write_wins(tmp_path):
    p = tmp_path / "rem.db"
    _make_v2(p)
    db = RemDB(p)
    db.save_extraction(1, "model-a", '[{"text": "a"}]')
    db.save_extraction(1, "model-b", "[]")
    assert db.get_extraction(1) == ("model-a", '[{"text": "a"}]')


def test_get_extraction_unknown_job_is_none(tmp_path):
    assert RemDB(tmp_path / "rem.db").get_extraction(999) is None


def test_extraction_deleted_with_its_job(tmp_path):
    p = tmp_path / "rem.db"
    _make_v2(p)
    db = RemDB(p)
    db.save_extraction(1, "m", "[]")
    db.conn.execute("DELETE FROM jobs WHERE id = 1")
    assert db.get_extraction(1) is None
