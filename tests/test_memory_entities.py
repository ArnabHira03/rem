import pytest

from rem.entities import entity_id, normalize_name
from rem.memory import Memory
from rem.models import Candidate, EntityRecord, HistoryEvent, Scope, TargetKind, utcnow
from rem.storage.vector.base import VectorPoint

from tests.conftest import add


def insert_entity(mem, *, name, scope, project_id, aliases=(), type="tool"):
    """Test-only stand-in for the real entity writes (a later task)."""
    now = utcnow()
    normalized = normalize_name(name)
    rec = EntityRecord(
        id=entity_id(scope, project_id, normalized), name=name, normalized=normalized, type=type,
        aliases=list(aliases), scope=scope, project_id=project_id, memory_ids=[], created_at=now, updated_at=now,
    )
    mem.vectors.upsert(mem.entity_col, [VectorPoint(rec.id, [0.0] * mem.embedder.dims, rec.to_payload())])
    return rec


def test_entity_collection_exists_and_is_empty(mem):
    assert mem.vectors.col_exists(mem.entity_col)
    assert not mem.vectors.col_has_sparse(mem.entity_col)
    assert mem.entity_col.startswith("rem_entities__")
    assert mem.list_entities() == []


def test_get_entity(mem):
    e = insert_entity(mem, name="Python", scope=Scope.GLOBAL, project_id=None)
    assert mem.get_entity(e.id) == e
    assert mem.get_entity("missing") is None


def test_list_entities_filters(mem):
    a = insert_entity(mem, name="Python", scope=Scope.GLOBAL, project_id=None)
    b = insert_entity(mem, name="Rem", scope=Scope.PROJECT, project_id=1)
    c = insert_entity(mem, name="Rem", scope=Scope.PROJECT, project_id=2)
    assert {e.id for e in mem.list_entities()} == {a.id, b.id, c.id}
    assert [e.id for e in mem.list_entities(scope=Scope.GLOBAL)] == [a.id]
    assert [e.id for e in mem.list_entities(scope=Scope.PROJECT, project_id=2)] == [c.id]


def test_find_entity_by_name_and_alias_within_bucket(mem):
    e = insert_entity(mem, name="Python", scope=Scope.GLOBAL, project_id=None, aliases=["py"])
    assert mem.find_entity("python", Scope.GLOBAL, None).id == e.id
    assert mem.find_entity("py", Scope.GLOBAL, None).id == e.id
    assert mem.find_entity("py", Scope.PROJECT, 1) is None
    assert mem.find_entity("rust", Scope.GLOBAL, None) is None


def test_find_entity_buckets_global_ignores_project_id_and_none_filter_is_exact(mem):
    g = insert_entity(mem, name="Python", scope=Scope.GLOBAL, project_id=None, aliases=["py"])
    insert_entity(mem, name="Other", scope=Scope.PROJECT, project_id=3, aliases=["py"])
    # a global lookup bucketed with a stray project id still finds the global entity, never the project one
    assert mem.find_entity("py", Scope.GLOBAL, 7).id == g.id
    assert mem.find_entity("py", Scope.SYSTEM, None) is None


def test_existing_memory_collection_gets_entity_collection_on_open(mem):
    rec = add(mem, "Prefers uv over pip")
    mem.vectors.delete_col(mem.entity_col)  # the store as an M6 install left it
    again = Memory(mem.cfg, vector_store=mem.vectors, embedder=mem.embedder)
    try:
        assert again.vectors.col_exists(again.entity_col)
        assert again.list_entities() == []
        assert again.get(rec.id).text == rec.text
        assert again.keyword_enabled
    finally:
        again.db.close()


# ---- entity resolution and linking on the write path (apply_candidates) ----

def apply(mem, cands, *, job_key="job:1", project_id=None, user_texts=None):
    return mem.apply_candidates(
        cands, job_key=job_key, project_id=project_id, session_id=None, source="cli",
        restate_min_similarity=0.9, has_user_turn=True,
        user_texts=user_texts if user_texts is not None else [c.text for c in cands],
    )


def cand(text, *names, scope="global", origin="user"):
    return Candidate(text=text, scope=scope, origin=origin, entities=[{"name": n, "type": "tool"} for n in names])


def entity_history(mem, eid):
    return [(h.target_kind, h.event, h.job_id) for h in mem.db.get_history(eid)]


def test_candidate_entities_create_entities_and_link_both_sides(mem):
    res = apply(mem, [cand("Uses uv", "uv")])
    (ent,) = mem.list_entities()
    (rec,) = mem.list()
    assert ent.name == "uv" and ent.normalized == "uv" and ent.type == "tool"
    assert ent.scope == Scope.GLOBAL and ent.project_id is None and ent.aliases == []
    assert ent.memory_ids == [rec.id] and rec.entity_ids == [ent.id]
    assert ent.id == entity_id(Scope.GLOBAL, None, "uv")
    assert entity_history(mem, ent.id) == [(TargetKind.ENTITY, HistoryEvent.ADD, "job:1")]
    assert res.entities == 1


def test_one_candidate_with_several_entities_links_them_all_sorted(mem):
    apply(mem, [cand("Uses uv and ruff", "uv", "ruff", "UV")])  # "UV" repeats uv: first spelling wins
    ents = {e.name: e for e in mem.list_entities()}
    assert set(ents) == {"uv", "ruff"} and ents["uv"].aliases == []
    (rec,) = mem.list()
    assert rec.entity_ids == sorted(e.id for e in ents.values())


def test_same_name_in_two_projects_is_two_entities(mem, tmp_path):  # Review Focus 1
    p1 = mem.db.upsert_project(str(tmp_path / "a"), "a", None)
    p2 = mem.db.upsert_project(str(tmp_path / "b"), "b", None)
    apply(mem, [cand("Rem uses Django", "Django", scope="project")], job_key="job:1", project_id=p1)
    apply(mem, [cand("Rem uses Django", "Django", scope="project")], job_key="job:2", project_id=p2)
    ents = mem.list_entities()
    assert len(ents) == 2
    by_pid = {e.project_id: e for e in ents}
    assert set(by_pid) == {p1, p2}
    for pid, e in by_pid.items():
        (m,) = mem.list(scope=Scope.PROJECT, project_id=pid)
        assert e.scope == Scope.PROJECT and e.memory_ids == [m.id] and m.entity_ids == [e.id]
        assert e.id == entity_id(Scope.PROJECT, pid, "django")


def test_global_and_project_mentions_are_separate_buckets(mem, tmp_path):
    p1 = mem.db.upsert_project(str(tmp_path / "a"), "a", None)
    apply(mem, [cand("Prefers Python", "Python"), cand("Rem is written in Python", "Python", scope="project")],
          project_id=p1)
    ents = mem.list_entities()
    assert {(e.scope, e.project_id) for e in ents} == {(Scope.GLOBAL, None), (Scope.PROJECT, p1)}
    for e in ents:
        assert len(e.memory_ids) == 1 and mem.get(e.memory_ids[0]).scope == e.scope


def test_replaying_a_job_does_not_duplicate_links(mem):  # Review Focus 2
    cands = [cand("Uses uv", "uv", "uv"), cand("Prefers uv over pip", "uv", "pip")]
    first = apply(mem, cands)
    snapshot = ({e.id: e for e in mem.list_entities()}, {m.id: m for m in mem.list()})
    rows = mem.db.conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    second = apply(mem, cands)
    ents, mems = {e.id: e for e in mem.list_entities()}, {m.id: m for m in mem.list()}
    assert (ents, mems) == snapshot  # nothing changed, not even updated_at
    assert mem.db.conn.execute("SELECT COUNT(*) FROM history").fetchone()[0] == rows
    assert first.entities == 2 and second.entities == 0
    uv = mem.find_entity("uv", Scope.GLOBAL, None)
    assert len(uv.memory_ids) == 2 == len(set(uv.memory_ids))
    for m in mems.values():
        assert m.entity_ids == sorted(set(m.entity_ids)) and uv.id in m.entity_ids


def test_replay_after_a_crash_before_the_memory_link_completes_it(mem, monkeypatch):
    real = mem.vectors.update_payload

    def crash_on_memory_link(col, id, payload):
        if col == mem.col and "entity_ids" in payload:
            raise RuntimeError("crash")
        return real(col, id, payload)

    monkeypatch.setattr(mem.vectors, "update_payload", crash_on_memory_link)
    with pytest.raises(RuntimeError):
        apply(mem, [cand("Uses uv", "uv")])
    monkeypatch.setattr(mem.vectors, "update_payload", real)
    apply(mem, [cand("Uses uv", "uv")])
    (ent,), (rec,) = mem.list_entities(), mem.list()
    assert ent.memory_ids == [rec.id] and rec.entity_ids == [ent.id]
    assert entity_history(mem, ent.id) == [(TargetKind.ENTITY, HistoryEvent.ADD, "job:1")]


def test_replay_after_a_crash_before_entity_history_repairs_the_add_row(mem, monkeypatch):
    real = mem.db.add_history

    def crash_on_entity_history(kind, *a, **kw):
        if kind == TargetKind.ENTITY:
            raise RuntimeError("crash")
        return real(kind, *a, **kw)

    monkeypatch.setattr(mem.db, "add_history", crash_on_entity_history)
    with pytest.raises(RuntimeError):
        apply(mem, [cand("Uses uv", "uv")])
    monkeypatch.setattr(mem.db, "add_history", real)
    apply(mem, [cand("Uses uv", "uv")])
    (ent,), (rec,) = mem.list_entities(), mem.list()
    assert ent.memory_ids == [rec.id] and rec.entity_ids == [ent.id]
    assert entity_history(mem, ent.id) == [(TargetKind.ENTITY, HistoryEvent.ADD, "job:1")]


def test_case_and_space_variants_merge(mem):  # Review Focus 3
    apply(mem, [cand("Prefers Python", "Python")], job_key="job:1")
    res2 = apply(mem, [cand("Writes Python daily", " python ")], job_key="job:2")
    res3 = apply(mem, [cand("Knows Python well", "PYTHON"), cand("Teaches Python", "python", "Python")],
                 job_key="job:3")
    (ent,) = mem.list_entities()
    assert ent.name == "Python" and ent.aliases == ["python", "PYTHON"]
    assert len(ent.memory_ids) == 4 == len(mem.list())
    assert res2.entities == 1 and res3.entities == 1  # "python" is already an alias in job 3
    assert [(h.event, h.job_id, h.new_value) for h in mem.db.get_history(ent.id)] == [
        (HistoryEvent.ADD, "job:1", "Python"), (HistoryEvent.MERGE, "job:2", "python"),
        (HistoryEvent.MERGE, "job:3", "PYTHON"),
    ]


def test_same_spelling_again_is_none_and_only_links(mem):
    apply(mem, [cand("Prefers Python", "Python")], job_key="job:1")
    (before,) = mem.list_entities()
    res = apply(mem, [cand("Writes Python daily", "Python")], job_key="job:2")
    (ent,) = mem.list_entities()
    assert res.entities == 0 and ent.aliases == [] and len(ent.memory_ids) == 2
    assert ent.updated_at >= before.updated_at
    assert [h.event for h in mem.db.get_history(ent.id)] == [HistoryEvent.ADD]


def test_two_merges_of_one_entity_in_one_job_do_not_raise(mem):
    apply(mem, [cand("Prefers Python", "Python")], job_key="job:1")
    apply(mem, [cand("Writes python", "python"), cand("Knows PYTHON", "PYTHON")], job_key="job:2")
    (ent,) = mem.list_entities()
    assert ent.aliases == ["python", "PYTHON"]
    assert [h.event for h in mem.db.get_history(ent.id)] == [HistoryEvent.ADD, HistoryEvent.MERGE]


def test_near_identical_name_merges_by_cosine(mem):
    # the hashing embedder is bag-of-words: "Node.js" and "node js" embed identically (cosine 1.0)
    # but normalize differently, so only the cosine branch can join them
    apply(mem, [cand("Uses Node.js", "Node.js")], job_key="job:1")
    res = apply(mem, [cand("Deploys with node js", "node js")], job_key="job:2")
    (ent,) = mem.list_entities()
    assert ent.name == "Node.js" and ent.aliases == ["node js"] and len(ent.memory_ids) == 2
    assert res.entities == 1 and mem.find_entity("node js", Scope.GLOBAL, None).id == ent.id
    assert [h.event for h in mem.db.get_history(ent.id)] == [HistoryEvent.ADD, HistoryEvent.MERGE]


def test_cosine_merge_stays_within_the_bucket(mem, tmp_path):
    p1 = mem.db.upsert_project(str(tmp_path / "a"), "a", None)
    apply(mem, [cand("Uses Node.js", "Node.js")], job_key="job:1")
    apply(mem, [cand("Rem deploys with node js", "node js", scope="project")], job_key="job:2", project_id=p1)
    assert {(e.name, e.scope) for e in mem.list_entities()} == {("Node.js", Scope.GLOBAL), ("node js", Scope.PROJECT)}


def test_identical_existing_memory_gets_linked(mem):
    add(mem, "Uses uv")  # already stored and verified, no entities
    res = apply(mem, [cand("Uses uv", "uv")], job_key="job:9")
    (rec,) = mem.list()
    (ent,) = mem.list_entities()
    assert res.added == [] and res.skipped == 1
    assert rec.entity_ids == [ent.id] and ent.memory_ids == [rec.id]


def test_corroborated_memory_gets_linked(mem):
    apply(mem, [cand("Uses uv", origin="assistant")], job_key="job:1")
    res = apply(mem, [cand("Uses uv", "uv")], job_key="job:2")
    (rec,) = mem.list()
    (ent,) = mem.list_entities()
    assert res.corroborated == [rec.id] and rec.entity_ids == [ent.id] and ent.memory_ids == [rec.id]


def test_dropped_candidate_creates_no_entities(mem):
    res = apply(mem, [cand("Uses Django", "Django", scope="project")], project_id=None)
    assert res.dropped == 1 and res.entities == 0 and mem.list_entities() == []


def test_candidate_without_entities_writes_the_memory_only(mem):
    res = apply(mem, [cand("Uses uv")])
    (rec,) = mem.list()
    assert rec.entity_ids == [] and mem.list_entities() == [] and res.entities == 0
