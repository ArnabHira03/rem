import pytest

from rem.daemon import runs
from rem.models import Candidate
from rem.storage.queue import IngestStore, Turn
import json

from tests.test_memory_entities import apply


@pytest.fixture
def ing(cfg, mem):
    cfg.daemon.debounce_seconds = 0
    return IngestStore(mem.db, cfg.daemon)


def repo(tmp_path, name):
    d = tmp_path / name
    (d / ".git").mkdir(parents=True, exist_ok=True)
    return d


def cand(text, *, scope="global", origin="user", names=(), rels=()):
    return Candidate(
        text=text, scope=scope, origin=origin,
        entities=[{"name": n, "type": "tool"} for n in names],
        relations=[{"source": s, "type": t, "target": d} for s, t, d in rels],
    )


def seed(mem, ing, session, turns, cands=None, *, cwd=None, outcome="done"):
    """One job through the real queue: ingest, claim, cache the extraction, apply, then finish it."""
    ing.ingest("cli", session, cwd, [Turn(r, c, f"{session}-{i}") for i, (r, c) in enumerate(turns)])
    job = ing.claim()
    assert job is not None
    if cands is not None:
        mem.db.save_extraction(job.id, "test-model", json.dumps({"candidates": [c.model_dump(mode="json") for c in cands]}))
        apply(mem, cands, job_key=f"job:{job.id}", project_id=job.project_id,
              user_texts=[c for r, c in turns if r == "user"])
    if outcome == "done":
        ing.complete(job.id)
    elif outcome == "failed":
        ing.fail(job.id, "boom")
        ing.conn.execute("UPDATE jobs SET status = 'failed' WHERE id = ?", (job.id,))
    return job.id


def test_get_run_end_to_end(mem, ing):
    c = cand("Prefers uv over pip", names=["uv", "pip"], rels=[("uv", "preferred_over", "pip")])
    jid = seed(mem, ing, "s1", [("user", "I prefer uv over pip"), ("assistant", "ok")], [c])
    rows = runs.gather(mem, ing, jid)
    assert [t["role"] for t in rows.turns] == ["user", "assistant"]
    (m,) = mem.list()
    assert rows.memories_now[m.id]["text"] == "Prefers uv over pip"
    assert len(rows.relations_by_memory[m.id]) == 1
    run = runs.get_run(mem, ing, jid)
    (fact,) = run["facts"]
    assert fact["outcome"]["kind"] == "added" and fact["outcome"]["memory_id"] == m.id
    assert fact["flags"] == []  # the relation was found through the store
    assert ("memory", "ADD", m.id) in [(w["target_kind"], w["event"], w["target_id"]) for w in run["writes"]]
    assert run["run"]["id"] == jid and run["run"]["n_added"] == 1 and run["run"]["model"] == "test-model"
    assert run["context_turns"] == []


def test_get_run_unknown_is_none(mem, ing):
    assert runs.get_run(mem, ing, 999) is None
    assert runs.gather(mem, ing, 999) is None


def test_list_runs_newest_first_and_filters(mem, ing, tmp_path):
    a = seed(mem, ing, "s1", [("user", "I prefer uv over pip")], [cand("Prefers uv over pip")], cwd=repo(tmp_path, "pa"))
    b = seed(mem, ing, "s2", [("user", "hello")], None, outcome="failed", cwd=repo(tmp_path, "pb"))
    c = seed(mem, ing, "s3", [("user", "hey")], [cand("Uses vim")], cwd=repo(tmp_path, "pa"))
    q = dict(project_id=None, status=None, flagged=False, limit=50, before=None)
    assert [r["id"] for r in runs.list_runs(mem, ing, **q)["runs"]] == [c, b, a]
    assert runs.list_runs(mem, ing, **q)["next_before"] is None
    assert [r["id"] for r in runs.list_runs(mem, ing, **{**q, "status": "failed"})["runs"]] == [b]
    assert [r["id"] for r in runs.list_runs(mem, ing, **{**q, "status": "done"})["runs"]] == [c, a]
    pa = ing.job_rows(a)["project_id"]
    assert [r["id"] for r in runs.list_runs(mem, ing, **{**q, "project_id": pa})["runs"]] == [c, a]
    page = runs.list_runs(mem, ing, **{**q, "limit": 2})
    assert [r["id"] for r in page["runs"]] == [c, b] and page["next_before"] == b
    page = runs.list_runs(mem, ing, **{**q, "limit": 2, "before": b})
    assert [r["id"] for r in page["runs"]] == [a] and page["next_before"] is None
    flagged = runs.list_runs(mem, ing, **{**q, "flagged": True})
    # c: the user turn "hey" names no fact, so "Uses vim" is downgraded; b failed; a is clean
    assert [(r["id"], r["flags"]) for r in flagged["runs"]] == [(c, ["downgraded"]), (b, ["failed"])]
    assert flagged["next_before"] is None
    one = runs.list_runs(mem, ing, **{**q, "flagged": True, "limit": 1})
    assert [r["id"] for r in one["runs"]] == [c] and one["next_before"] == c


def test_jobs_flagged_scan_is_bounded(mem, ing, monkeypatch):
    ids = [seed(mem, ing, f"s{i}", [("user", f"t{i}")], None) for i in range(5)]
    q = dict(project_id=None, status=None, flagged=True, limit=50, before=None)
    assert runs.list_runs(mem, ing, **q) == {"runs": [], "next_before": None}
    monkeypatch.setattr(runs, "SCAN_CAP", 3)
    assert runs.list_runs(mem, ing, **q) == {"runs": [], "next_before": ids[2]}
    assert runs.list_runs(mem, ing, **{**q, "before": ids[2]}) == {"runs": [], "next_before": None}


def test_pending_includes_processing(mem, ing):
    ing.ingest("cli", "s1", None, [Turn("user", "hello", "m1")])
    unclaimed = ing.conn.execute("SELECT id FROM jobs").fetchone()[0]
    assert runs.get_run(mem, ing, unclaimed)["turns"] == []  # not run yet: no turns linked
    claimed = ing.claim().id
    assert claimed == unclaimed
    q = dict(project_id=None, flagged=False, limit=50, before=None)
    (r,) = runs.list_runs(mem, ing, status="pending", **q)["runs"]
    assert r["id"] == claimed and r["status"] == "processing" and r["n_turns"] == 1
    assert runs.list_runs(mem, ing, status="done", **q)["runs"] == []


def test_runs_read_only(mem, ing):
    c = cand("Prefers uv over pip", names=["uv", "pip"], rels=[("uv", "preferred_over", "pip")])
    jid = seed(mem, ing, "s1", [("user", "I prefer uv over pip")], [c])
    seed(mem, ing, "s2", [("user", "x")], None, outcome="failed")

    def snapshot():
        cols = [mem.col, mem.entity_col, mem.relation_col]
        return (
            [tuple(r) for r in mem.db.conn.execute("SELECT * FROM history ORDER BY id")],
            [tuple(r) for r in mem.db.conn.execute("SELECT * FROM jobs ORDER BY id")],
            [tuple(r) for r in mem.db.conn.execute("SELECT * FROM extractions ORDER BY job_id")],
            {col: sorted((p.id, repr(sorted(p.payload.items()))) for p in mem.vectors.list_points(col))
             for col in cols},
        )

    before = snapshot()
    assert before[0] and before[3][mem.relation_col]
    runs.list_runs(mem, ing, project_id=None, status=None, flagged=False, limit=50, before=None)
    runs.list_runs(mem, ing, project_id=None, status=None, flagged=True, limit=50, before=None)
    runs.get_run(mem, ing, jid)
    assert snapshot() == before


def test_relation_ids_by_memory_one_scan(mem):
    from tests.test_memory_entities import insert_entity
    from tests.test_memory_relations import insert_relation
    from rem.models import Scope

    uv = insert_entity(mem, name="uv", scope=Scope.GLOBAL, project_id=None)
    pip = insert_entity(mem, name="pip", scope=Scope.GLOBAL, project_id=None)
    rem_ = insert_entity(mem, name="rem", scope=Scope.PROJECT, project_id=1)
    qd = insert_entity(mem, name="Qdrant", scope=Scope.PROJECT, project_id=1)
    a = insert_relation(mem, uv, "preferred_over", pip, ["m1", "m2"])
    b = insert_relation(mem, uv, "related_to", pip, ["m1"])
    insert_relation(mem, rem_, "uses", qd, ["m1"])
    got = mem.relation_ids_by_memory(Scope.GLOBAL, None)
    assert {k: sorted(v) for k, v in got.items()} == {"m1": sorted([a.id, b.id]), "m2": [a.id]}
    assert mem.relation_ids_by_memory(Scope.PROJECT, 2) == {}
    # same filter as relations_for
    assert sorted(r.id for r in mem.relations_for("m1", Scope.GLOBAL, None)) == got["m1"]


def test_messages_for_jobs_matches_messages_for(mem, ing):
    a = seed(mem, ing, "s1", [("user", "one"), ("assistant", "two")], None)
    ing.ingest("cli", "s1", None, [Turn("user", "three", "s1-later")])
    b = ing.claim().id
    c = seed(mem, ing, "s2", [("user", "four")], None)
    unclaimed_session = ing.ingest("cli", "s3", None, [Turn("user", "five", "s3-0")])
    got = ing.messages_for_jobs([c, a, b, 999])
    assert set(got) == {a, b, c, 999}
    for jid in (a, b, c, 999):
        assert got[jid] == ing.messages_for(jid)
    assert [t.content for t in got[a]] == ["one", "two"] and got[999] == []
    assert ing.messages_for_jobs([]) == {}
    del unclaimed_session


def test_messages_for_jobs_uses_the_index(mem, ing):
    plan = ing.conn.execute(
        "EXPLAIN QUERY PLAN " + ing._MESSAGES_FOR_JOBS_SQL.format(marks="?,?"), (1, 2)).fetchall()
    detail = " ".join(r[-1] for r in plan)
    assert "messages_job" in detail and "SCAN m" not in detail and "SCAN messages" not in detail


def test_list_runs_fetches_turns_once_per_batch(mem, ing, monkeypatch):
    ids = [seed(mem, ing, f"s{i}", [("user", f"I use tool{i}")], [cand(f"Uses tool{i}")]) for i in range(7)]
    calls = []
    real = ing.messages_for_jobs
    monkeypatch.setattr(ing, "messages_for_jobs", lambda job_ids: calls.append(list(job_ids)) or real(job_ids))
    monkeypatch.setattr(ing, "messages_for", lambda job_id: pytest.fail("per-job turn query in the list path"))
    monkeypatch.setattr(ing, "context_for", lambda job_id: pytest.fail("context turns in the list path"))
    q = dict(project_id=None, status=None, flagged=False, limit=50, before=None)
    page = runs.list_runs(mem, ing, **q)
    assert len(page["runs"]) == 7 and all(r["n_turns"] == 1 for r in page["runs"])
    assert len(calls) == 1
    calls.clear()
    monkeypatch.setattr(runs, "SCAN_BATCH", 3)
    page = runs.list_runs(mem, ing, **{**q, "flagged": True, "limit": 1})
    assert page == {"runs": [], "next_before": None}  # every fact is the user's own: nothing flagged
    assert [len(c) for c in calls] == [3, 3, 1]
    assert sorted(i for c in calls for i in c) == sorted(ids)
