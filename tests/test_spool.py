import json
import stat

from rem.spool import drain_spool, write_spool


def body(sid):
    return {"source": "claude-code", "session_id": sid, "cwd": None,
            "turns": [{"role": "user", "content": "hi"}], "enqueue": True}


def test_write_then_drain_in_name_order(tmp_path):
    for sid in ("a", "b", "c"):
        write_spool(tmp_path, body(sid))
    seen = []
    assert drain_spool(tmp_path, lambda b: seen.append(b["session_id"])) == 3
    assert seen == ["a", "b", "c"]
    assert list((tmp_path / "spool").iterdir()) == []


def test_drain_without_spool_dir(tmp_path):
    assert drain_spool(tmp_path, lambda b: None) == 0


def test_permissions(tmp_path):
    p = write_spool(tmp_path, body("a"))
    assert stat.S_IMODE(p.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert json.loads(p.read_text())["session_id"] == "a"


def test_corrupt_and_rejected_files_go_to_bad(tmp_path):
    write_spool(tmp_path, body("a"))
    corrupt = write_spool(tmp_path, body("x"))
    corrupt.write_text("{not json")
    write_spool(tmp_path, body("reject"))
    write_spool(tmp_path, body("b"))
    seen = []

    def ingest(b):
        if b["session_id"] == "reject":
            raise ValueError("nope")
        seen.append(b["session_id"])

    assert drain_spool(tmp_path, ingest) == 2
    assert seen == ["a", "b"]
    spool = tmp_path / "spool"
    assert len(list((spool / "bad").iterdir())) == 2
    assert [p for p in spool.iterdir() if p.is_file()] == []


def test_other_error_stops_drain_and_keeps_rest(tmp_path):
    for sid in ("a", "b", "c"):
        write_spool(tmp_path, body(sid))

    def ingest(b):
        if b["session_id"] == "b":
            raise RuntimeError("boom")

    assert drain_spool(tmp_path, ingest) == 1
    left = sorted((tmp_path / "spool").glob("*.json"))
    assert [json.loads(p.read_text())["session_id"] for p in left] == ["b", "c"]


def test_valid_json_that_is_not_an_object_goes_to_bad(tmp_path):
    p = write_spool(tmp_path, body("a"))
    p.write_text("[1, 2]")
    seen = []
    assert drain_spool(tmp_path, seen.append) == 0
    assert seen == []
    assert [q.name for q in (tmp_path / "spool" / "bad").iterdir()] == [p.name]


def reject(b):
    raise ValueError("bad")


def test_first_spool_under_umask_022_is_private(tmp_path):
    import os

    data = tmp_path / "data"  # does not exist yet: rem creates it
    old = os.umask(0o022)
    try:
        p = write_spool(data, body("a"))
        assert drain_spool(data, reject) == 0  # quarantine creates spool/bad
    finally:
        os.umask(old)
    assert stat.S_IMODE(data.stat().st_mode) == 0o700
    assert stat.S_IMODE(p.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE((p.parent / "bad").stat().st_mode) == 0o700
    assert stat.S_IMODE((p.parent / "bad" / p.name).stat().st_mode) == 0o600


def test_full_spool_refuses_new_files(tmp_path, monkeypatch):
    import pytest

    from rem import spool
    from rem.errors import RemError

    monkeypatch.setattr(spool, "SPOOL_MAX_FILES", 3)
    for sid in ("a", "b", "c"):
        write_spool(tmp_path, body(sid))
    with pytest.raises(spool.SpoolFull) as e:
        write_spool(tmp_path, body("d"))
    assert isinstance(e.value, RemError)
    assert spool.spool_count(tmp_path) == 3
    assert not list((tmp_path / "spool").glob(".*.tmp"))


def test_spool_count_ignores_quarantined_and_missing(tmp_path):
    from rem.spool import spool_count

    assert spool_count(tmp_path) == 0
    write_spool(tmp_path, body("a"))
    drain_spool(tmp_path, reject)  # a is now in spool/bad
    write_spool(tmp_path, body("b"))
    assert spool_count(tmp_path) == 1


def test_existing_data_dir_mode_is_left_alone(tmp_path):
    import os

    data = tmp_path / "data"
    data.mkdir()
    os.chmod(data, 0o750)
    write_spool(data, body("a"))
    assert stat.S_IMODE(data.stat().st_mode) == 0o750


def test_poison_file_is_quarantined_after_three_failures(tmp_path):
    first = write_spool(tmp_path, body("poison"))
    second = write_spool(tmp_path, body("ok"))
    seen = []

    def ingest(b):
        if b["session_id"] == "poison":
            raise RuntimeError("boom")
        seen.append(b["session_id"])

    failures: dict[str, int] = {}
    assert drain_spool(tmp_path, ingest, failures) == 0
    assert drain_spool(tmp_path, ingest, failures) == 0
    assert first.exists() and second.exists()
    assert drain_spool(tmp_path, ingest, failures) == 1
    assert not first.exists() and (tmp_path / "spool" / "bad" / first.name).exists()
    assert not second.exists() and seen == ["ok"]


def test_prune_bad_keeps_newest(tmp_path):
    from rem.spool import prune_bad

    bad = tmp_path / "spool" / "bad"
    bad.mkdir(parents=True)
    names = [f"{i:04d}.json" for i in range(105)]
    for n in names:
        (bad / n).write_text("{}")
    assert prune_bad(tmp_path) == 5
    left = sorted(p.name for p in bad.glob("*.json"))
    assert left == names[5:]
