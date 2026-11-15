from rem.models import HistoryEvent, TargetKind
from rem.storage.sqlite import SCHEMA_VERSION, RemDB


def test_schema_and_meta(tmp_path):
    db = RemDB(tmp_path / "rem.db")
    assert db.get_meta("schema_version") == str(SCHEMA_VERSION)
    db.set_meta("active_embedder", "hash-64")
    assert db.get_meta("active_embedder") == "hash-64"


def test_history_insert_or_ignore_with_job_id(tmp_path):
    db = RemDB(tmp_path / "rem.db")
    assert db.add_history(TargetKind.MEMORY, "m1", HistoryEvent.ADD, None, "a", "extractor", job_id="j1") is True
    assert db.add_history(TargetKind.MEMORY, "m1", HistoryEvent.ADD, None, "a", "extractor", job_id="j1") is False
    db.add_history(TargetKind.MEMORY, "m1", HistoryEvent.UPDATE, "a", "b", "cli")
    assert [h.event for h in db.get_history("m1")] == ["ADD", "UPDATE"]


def test_history_accepts_entity_and_relation_targets(tmp_path):
    db = RemDB(tmp_path / "rem.db")
    db.add_history(TargetKind.ENTITY, "e1", HistoryEvent.MERGE, "uv", "astral uv", "extractor")
    assert db.get_history("e1")[0].target_kind == "entity"


def test_upsert_project_is_stable(tmp_path):
    db = RemDB(tmp_path / "rem.db")
    a = db.upsert_project("/x/rem", "rem", None)
    b = db.upsert_project("/x/rem", "rem", "git@x:rem.git")
    assert a == b and db.get_project(a) == ("/x/rem", "rem", "git@x:rem.git")
    assert db.find_project("/x/rem") == a and db.find_project("/elsewhere") is None


def test_reopen_keeps_data(tmp_path):
    RemDB(tmp_path / "rem.db").set_meta("k", "v")
    assert RemDB(tmp_path / "rem.db").get_meta("k") == "v"
