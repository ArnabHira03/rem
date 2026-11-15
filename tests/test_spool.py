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
