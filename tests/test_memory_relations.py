import pytest

from rem.memory import Memory
from rem.models import Candidate, Confirmation, HistoryEvent, Provenance, RelationRecord, Scope, Status, TargetKind, utcnow
from rem.relations import relation_id, relation_sentence
from rem.storage.vector.base import VectorPoint

from tests.test_memory_entities import apply, insert_entity


def insert_relation(mem, src, type, dst, memory_ids=(), status=Status.UNVERIFIED,
                    provenance=Provenance.ASSISTANT_GENERATED):
    """Test-only stand-in for the real relation writes (a later task)."""
    now = utcnow()
    rec = RelationRecord(
        id=relation_id(src.id, type, dst.id), source_entity_id=src.id, target_entity_id=dst.id, type=type,
        sentence=relation_sentence(src.name, type, dst.name), memory_ids=sorted(memory_ids),
        provenance=provenance, status=status, scope=src.scope, project_id=src.project_id,
        created_at=now, updated_at=now,
    )
    mem.vectors.upsert(mem.relation_col, [VectorPoint(rec.id, [0.0] * mem.embedder.dims, rec.to_payload())])
    return rec


def test_relation_collection_exists_dense_only(mem):
    assert mem.relation_col.startswith("rem_relations__")
    assert mem.vectors.col_exists(mem.relation_col)
    assert not mem.vectors.col_has_sparse(mem.relation_col)
    assert mem.list_relations() == []


def test_existing_store_gets_relation_collection_on_open(mem):
    mem.vectors.delete_col(mem.relation_col)
    again = Memory(mem.cfg, vector_store=mem.vectors, embedder=mem.embedder)
    try:
        assert again.vectors.col_exists(again.relation_col)
        assert again.list_relations() == []
    finally:
        again.db.close()


def test_list_relations_resolves_names_and_filters(mem):
    uv = insert_entity(mem, name="uv", scope=Scope.GLOBAL, project_id=None)
    pip = insert_entity(mem, name="pip", scope=Scope.GLOBAL, project_id=None, aliases=["pip3"])
    rem = insert_entity(mem, name="rem", scope=Scope.PROJECT, project_id=1)
    qd = insert_entity(mem, name="Qdrant", scope=Scope.PROJECT, project_id=1)
    a = insert_relation(mem, uv, "preferred_over", pip, ["m1"])
    b = insert_relation(mem, rem, "uses", qd, ["m2", "m3"])
    got = {r.id: r for r in mem.list_relations()}
    assert set(got) == {a.id, b.id}
    assert (got[a.id].source, got[a.id].target, got[a.id].evidence) == ("uv", "pip", 1)
    assert got[b.id].evidence == 2
    assert got[b.id].project_name is None  # project 1 is not registered
    assert [r.id for r in mem.list_relations(scope=Scope.PROJECT, project_id=1)] == [b.id]
    assert [r.id for r in mem.list_relations(entity="PIP")] == [a.id]
    assert [r.id for r in mem.list_relations(entity="pip3")] == [a.id]
    assert [r.id for r in mem.list_relations(entity="uv")] == [a.id]
    assert mem.list_relations(entity="nothing") == []
    assert [r.source for r in mem.list_relations()] == ["rem", "uv"]


def test_relations_for_memory(mem):
    uv = insert_entity(mem, name="uv", scope=Scope.GLOBAL, project_id=None)
    pip = insert_entity(mem, name="pip", scope=Scope.GLOBAL, project_id=None)
    rem = insert_entity(mem, name="rem", scope=Scope.PROJECT, project_id=1)
    qd = insert_entity(mem, name="Qdrant", scope=Scope.PROJECT, project_id=1)
    a = insert_relation(mem, uv, "preferred_over", pip, ["m1"])
    insert_relation(mem, uv, "related_to", pip, ["m9"])
    insert_relation(mem, rem, "uses", qd, ["m1"])
    assert [r.id for r in mem.relations_for("m1", Scope.GLOBAL, None)] == [a.id]
    assert mem.get_relation(a.id) == a
    assert mem.get_relation("missing") is None


# ---- writing relations from a job (apply_candidates) ----

def rcand(text, names, rels, *, scope="global", origin="user", restates=None):
    return Candidate(
        text=text, scope=scope, origin=origin, restates=restates,
        entities=[{"name": n, "type": "tool"} for n in names],
        relations=[{"source": s, "type": t, "target": d} for s, t, d in rels],
    )


def rhist(mem, rid):
    return [(h.target_kind, h.event, h.job_id) for h in mem.db.get_history(rid)]


def test_relation_written_with_evidence_trust_and_add_row(mem):
    res = apply(mem, [rcand("Prefers uv over pip", ["uv", "pip"], [("uv", "preferred_over", "pip")])])
    (r,) = mem.list_relations()
    (m,) = mem.list()
    assert (r.source, r.type, r.target, r.evidence, r.status) == ("uv", "preferred_over", "pip", 1, "verified")
    rec = mem.get_relation(r.id)
    assert rec.memory_ids == [m.id] and rec.sentence == "uv preferred over pip"
    assert rhist(mem, r.id) == [(TargetKind.RELATION, HistoryEvent.ADD, "job:1")]
    assert res.relations == 1


def test_assistant_fact_gives_unverified_relation(mem):
    apply(mem, [rcand("Uses uv", ["uv", "Python"], [("uv", "related_to", "Python")], origin="assistant")])
    (r,) = mem.list_relations()
    assert r.status == "unverified" and r.provenance == "assistant_generated"


def test_replay_writes_nothing_new(mem):
    c = [rcand("Uses uv", ["uv", "Python"], [("uv", "related_to", "Python")])]
    apply(mem, c)
    res = apply(mem, c)
    (r,) = mem.list_relations()
    assert r.evidence == 1 and res.relations == 0
    assert len(rhist(mem, r.id)) == 1


def test_replay_after_crash_before_add_row_repairs_it(mem, monkeypatch):
    c = [rcand("Uses uv", ["uv", "Python"], [("uv", "related_to", "Python")])]
    real = mem.db.add_history
    state = {"boom": True}

    def flaky(kind, *a, **kw):
        if kind == TargetKind.RELATION and state["boom"]:
            state["boom"] = False
            raise RuntimeError("crash")
        return real(kind, *a, **kw)

    monkeypatch.setattr(mem.db, "add_history", flaky)
    with pytest.raises(RuntimeError):
        apply(mem, c)
    (r,) = mem.list_relations()
    assert rhist(mem, r.id) == []
    apply(mem, c)
    assert rhist(mem, r.id) == [(TargetKind.RELATION, HistoryEvent.ADD, "job:1")]
    (again,) = mem.list_relations()
    assert again.evidence == 1


def test_two_facts_same_edge_one_relation_two_evidence(mem):
    rel = [("uv", "uses", "Python")]
    apply(mem, [rcand("uv runs Python scripts", ["uv", "Python"], rel),
                rcand("uv is a Python tool", ["uv", "Python"], rel)])
    (r,) = mem.list_relations()
    assert r.evidence == 2 and len(mem.list()) == 2
    assert [(k, e) for k, e, _ in rhist(mem, r.id)] == [(TargetKind.RELATION, HistoryEvent.ADD)]


def test_second_fact_verifies_unverified_relation_with_corroborate_row(mem):
    rel = [("uv", "uses", "Python")]
    apply(mem, [rcand("uv runs Python scripts", ["uv", "Python"], rel, origin="assistant")], job_key="job:1")
    (r,) = mem.list_relations()
    assert r.status == "unverified"
    apply(mem, [rcand("uv is a Python tool", ["uv", "Python"], rel)], job_key="job:2")
    (r2,) = mem.list_relations()
    assert r2.status == "verified" and r2.provenance == "user_asserted" and r2.evidence == 2
    assert rhist(mem, r.id) == [
        (TargetKind.RELATION, HistoryEvent.ADD, "job:1"),
        (TargetKind.RELATION, HistoryEvent.CORROBORATE, "job:2"),
    ]


def test_restating_candidate_adds_no_relation(mem):
    apply(mem, [rcand("Uses uv for Python", ["uv", "Python"], [])], job_key="job:1")
    (m,) = mem.list()
    mem.apply_candidates(
        [rcand("The user relies on uv for Python work", ["uv", "Python"], [("uv", "uses", "Python")],
               origin="assistant", restates=m.id)],
        job_key="job:2", project_id=None, session_id=None, source="cli",
        restate_min_similarity=0.0, has_user_turn=True, user_texts=[],
    )
    assert len(mem.list()) == 1
    assert mem.list_relations() == []


def test_same_text_existing_memory_gets_the_relation(mem):
    apply(mem, [rcand("Uses uv for Python", ["uv", "Python"], [])], job_key="job:1")
    assert mem.list_relations() == []
    apply(mem, [rcand("Uses uv for Python", ["uv", "Python"], [("uv", "uses", "Python")])], job_key="job:2")
    (r,) = mem.list_relations()
    (m,) = mem.list()
    assert mem.get_relation(r.id).memory_ids == [m.id]


def test_merged_spellings_make_no_self_loop(mem):
    apply(mem, [rcand("Uses Postgres", ["Postgres", "postgres"], [("Postgres", "related_to", "postgres")])])
    assert mem.list_relations() == []


def test_same_names_two_projects_two_relations(mem, tmp_path):
    p1 = mem.db.upsert_project(str(tmp_path / "a"), "a", None)
    p2 = mem.db.upsert_project(str(tmp_path / "b"), "b", None)
    c = [rcand("Rem uses Django", ["Rem", "Django"], [("Rem", "uses", "Django")], scope="project")]
    apply(mem, c, job_key="job:1", project_id=p1)
    apply(mem, c, job_key="job:2", project_id=p2)
    rels = mem.list_relations()
    assert len(rels) == 2 and {r.project_id for r in rels} == {p1, p2}


def test_dropped_project_candidate_writes_no_relation(mem):
    apply(mem, [rcand("Rem uses Django", ["Rem", "Django"], [("Rem", "uses", "Django")], scope="project")])
    assert mem.list_relations() == []


def test_apply_trust_deletes_relation_without_evidence(mem):
    apply(mem, [rcand("Uses uv", ["uv", "Python"], [("uv", "related_to", "Python")])])
    (r,) = mem.list_relations()
    (m,) = mem.list()
    mem.vectors.delete(mem.col, [m.id])
    assert mem._apply_trust(mem.get_relation(r.id), "t", None) is None
    assert mem.list_relations() == []
    assert rhist(mem, r.id)[-1][1] == HistoryEvent.DELETE


def test_user_restating_unverified_memory_adds_no_relation(mem):
    apply(mem, [rcand("Uses uv for Python", ["uv", "Python"], [], origin="assistant")], job_key="job:1")
    (m,) = mem.list()
    assert m.status == "unverified"
    mem.apply_candidates(
        [rcand("The user relies on uv for Python work", ["uv", "Python"], [("uv", "uses", "Python")],
               restates=m.id)],
        job_key="job:2", project_id=None, session_id=None, source="cli",
        restate_min_similarity=0.0, has_user_turn=True, user_texts=["I rely on uv for Python work"],
    )
    (m2,) = mem.list()
    assert m2.status == "verified"  # corroborated
    assert mem.list_relations() == []


# ---- trust refresh and lifecycle ----

REL = [("uv", "uses", "Python")]


def one_relation(mem, text="uv runs Python scripts", origin="assistant", job_key="job:1"):
    apply(mem, [rcand(text, ["uv", "Python"], REL, origin=origin)], job_key=job_key)
    (r,) = mem.list_relations()
    return r


def events(mem, rid):
    return [e for _, e, _ in rhist(mem, rid)]


def test_confirming_evidence_verifies_relation(mem):
    r = one_relation(mem)
    assert r.status == "unverified"
    (m,) = mem.list()
    mem.confirm(m.id, actor="cli")
    (r2,) = mem.list_relations()
    assert (r2.status, r2.provenance) == ("verified", "user_confirmed")
    assert events(mem, r.id) == [HistoryEvent.ADD, HistoryEvent.CORROBORATE]


def test_corroborating_evidence_verifies_relation_with_job(mem):
    r = one_relation(mem)
    (m,) = mem.list()
    mem.corroborate(m.id, provenance=Provenance.USER_ASSERTED, job_id="job:2", actor="extractor")
    (r2,) = mem.list_relations()
    assert r2.status == "verified"
    assert rhist(mem, r.id)[-1] == (TargetKind.RELATION, HistoryEvent.CORROBORATE, "job:2")


def test_mcp_writes_never_verify_a_relation(mem):
    r = one_relation(mem, origin="user")
    assert r.status == "verified"
    (m,) = mem.list()
    mem.update(m.id, m.text, actor="mcp", unverify=True)
    (r2,) = mem.list_relations()
    assert r2.status == "unverified"
    assert events(mem, r.id) == [HistoryEvent.ADD, HistoryEvent.UPDATE]


def test_deleting_verified_evidence_leaves_unverified_relation(mem):
    rel = [("uv", "uses", "Python")]
    apply(mem, [rcand("uv runs Python scripts", ["uv", "Python"], rel, origin="assistant")], job_key="job:1")
    apply(mem, [rcand("uv is a Python tool", ["uv", "Python"], rel)], job_key="job:2")
    (r,) = mem.list_relations()
    assert r.status == "verified" and r.evidence == 2
    verified = next(m for m in mem.list() if m.status == "verified")
    mem.delete(verified.id, actor="cli")
    (r2,) = mem.list_relations()
    assert (r2.status, r2.evidence) == ("unverified", 1)
    assert events(mem, r.id)[-1] == HistoryEvent.UPDATE


def test_deleting_last_evidence_deletes_relation_then_entities(mem):
    r = one_relation(mem)
    (m,) = mem.list()
    mem.delete(m.id, actor="cli")
    assert mem.list_relations() == [] and mem.list_entities() == []
    row = mem.db.get_history(r.id)[-1]
    assert (row.target_kind, row.event, row.old_value) == (TargetKind.RELATION, HistoryEvent.DELETE, "uv uses Python")


def test_delete_converges_after_crash_between_relation_and_entity_unlink(mem, monkeypatch):
    one_relation(mem)
    (m,) = mem.list()
    real = Memory._unlink
    state = {"boom": True}

    def flaky(self, *a, **kw):
        if state["boom"]:
            state["boom"] = False
            raise RuntimeError("crash")
        return real(self, *a, **kw)

    monkeypatch.setattr(Memory, "_unlink", flaky)
    with pytest.raises(RuntimeError):
        mem.delete(m.id, actor="cli")
    assert mem.list_relations() == [] and mem.get(m.id) is not None
    mem.delete(m.id, actor="cli")
    assert mem.list_relations() == [] and mem.list_entities() == [] and mem.list() == []


def test_text_update_unlinks_relations(mem):
    r = one_relation(mem)
    (m,) = mem.list()
    mem.update(m.id, "Something else entirely", actor="cli")
    assert mem.list_relations() == [] and mem.list_entities() == []
    assert events(mem, r.id)[-1] == HistoryEvent.DELETE


def test_same_text_update_keeps_relations(mem):
    r = one_relation(mem)
    (m,) = mem.list()
    mem.update(m.id, m.text, actor="cli")
    (r2,) = mem.list_relations()
    assert r2.id == r.id and r2.evidence == 1 and len(mem.list_entities()) == 2


def test_apply_confirmations_verifies_relation(mem):
    apply(mem, [rcand("Prefers uv over pip", ["uv", "pip"], [("uv", "preferred_over", "pip")],
                      origin="assistant")], job_key="job:1")
    (r,) = mem.list_relations()
    assert r.status == "unverified"
    (m,) = mem.list()
    m = mem.request_confirmation(m.id, actor="mcp")
    c = Confirmation(memory_id=m.id, hash=m.hash, requested_at=m.confirm_requested_at, quote="yes, uv over pip")
    assert mem.apply_confirmations([c], user_texts=["yes, uv over pip"], job_key="job:9") == [m.id]
    (r2,) = mem.list_relations()
    assert (r2.status, r2.provenance) == ("verified", "user_confirmed")
    assert rhist(mem, r.id)[-1] == (TargetKind.RELATION, HistoryEvent.CORROBORATE, "job:9")


@pytest.mark.parametrize("how", ["delete", "update"])
def test_unlink_crash_never_leaves_relation_over_verified(mem, monkeypatch, how):
    rel = [("uv", "uses", "Python")]
    apply(mem, [rcand("uv runs Python scripts", ["uv", "Python"], rel, origin="assistant")], job_key="job:1")
    apply(mem, [rcand("uv is a Python tool", ["uv", "Python"], rel)], job_key="job:2")
    (r,) = mem.list_relations()
    assert r.status == "verified" and r.evidence == 2
    verified = next(m for m in mem.list() if m.status == "verified")
    other = next(m for m in mem.list() if m.status == "unverified")
    real = mem.vectors.update_payload
    state = {"boom": True}

    def flaky(col, id, payload):
        if col == mem.relation_col and "status" in payload and state["boom"]:
            state["boom"] = False
            raise RuntimeError("crash")
        return real(col, id, payload)

    monkeypatch.setattr(mem.vectors, "update_payload", flaky)

    def act():
        if how == "delete":
            mem.delete(verified.id, actor="cli")
        else:
            mem.update(verified.id, "Something else entirely", actor="cli")

    with pytest.raises(RuntimeError):
        act()
    assert verified.id in mem.get_relation(r.id).memory_ids  # nothing was written: a retry finds it
    act()
    rec = mem.get_relation(r.id)
    assert rec.memory_ids == [other.id] and rec.status == Status.UNVERIFIED


# ---- searching relations by meaning ----

def embedded_relation(mem, src, type, dst, **kw):
    """insert_relation, but with the real vector of its sentence."""
    rec = insert_relation(mem, src, type, dst, **kw)
    mem.vectors.upsert(mem.relation_col, [VectorPoint(rec.id, mem.embedder.embed([rec.sentence])[0], rec.to_payload())])
    return rec


def test_search_relations_finds_by_sentence(mem):
    uv = insert_entity(mem, name="uv", scope=Scope.GLOBAL, project_id=None)
    pip = insert_entity(mem, name="pip", scope=Scope.GLOBAL, project_id=None)
    qd = insert_entity(mem, name="Qdrant", scope=Scope.GLOBAL, project_id=None)
    rem = insert_entity(mem, name="rem", scope=Scope.GLOBAL, project_id=None)
    a = embedded_relation(mem, uv, "preferred_over", pip)
    embedded_relation(mem, rem, "uses", qd)
    got = mem.search_relations("uv preferred over pip")
    assert got[0].id == a.id
    assert got[0].score == pytest.approx(1.0, abs=1e-4)
    assert (got[0].source, got[0].target) == ("uv", "pip")
    with pytest.raises(Exception):
        mem.search_relations("   ")


def test_search_relations_other_project_needs_t_other(mem, monkeypatch, tmp_path):
    from rem.storage.vector.base import ScoredPoint

    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    pid = mem.db.upsert_project(str(root), "repo", None)
    a = insert_entity(mem, name="rem", scope=Scope.PROJECT, project_id=pid)
    b = insert_entity(mem, name="qdrant", scope=Scope.PROJECT, project_id=pid)
    c = insert_entity(mem, name="foo", scope=Scope.PROJECT, project_id=pid + 1)
    d = insert_entity(mem, name="bar", scope=Scope.PROJECT, project_id=pid + 1)
    mine = insert_relation(mem, a, "uses", b)
    theirs = insert_relation(mem, c, "uses", d)
    pts = {r.id: ScoredPoint(r.id, 0.4, r.to_payload()) for r in (mine, theirs)}
    monkeypatch.setattr(mem.vectors, "search", lambda col, vec, limit, filters=None: list(pts.values()))
    assert [r.id for r in mem.search_relations("anything", cwd=root)] == [mine.id]
    assert [r.id for r in mem.search_relations("anything")] == []  # no cwd: every project is "other"
    pts[theirs.id] = ScoredPoint(theirs.id, 0.6, theirs.to_payload())
    assert {r.id for r in mem.search_relations("anything", cwd=root)} == {mine.id, theirs.id}


def test_search_relations_entity_filter_and_limit(mem):
    uv = insert_entity(mem, name="uv", scope=Scope.GLOBAL, project_id=None)
    pip = insert_entity(mem, name="pip", scope=Scope.GLOBAL, project_id=None)
    poe = insert_entity(mem, name="poetry", scope=Scope.GLOBAL, project_id=None)
    a = embedded_relation(mem, uv, "preferred_over", pip)
    b = embedded_relation(mem, uv, "preferred_over", poe)
    embedded_relation(mem, pip, "related_to", poe)
    q = "uv preferred over pip"
    assert len(mem.search_relations(q, limit=1)) == 1
    got = mem.search_relations(q, entity="UV")
    assert {r.id for r in got} == {a.id, b.id}
    assert got[0].id == a.id
    assert [r.id for r in mem.search_relations(q, entity="poetry", scope=Scope.GLOBAL)][0:1] in ([a.id], [b.id])
    assert mem.search_relations(q, entity="nothing") == []


def test_search_relations_does_not_change_memory_search(mem):
    from tests.conftest import add

    add(mem, "Prefers uv over pip")
    add(mem, "Uses Qdrant for storage")
    before = [(r.record.id, r.score) for r in mem.search("uv pip")]
    uv = insert_entity(mem, name="uv", scope=Scope.GLOBAL, project_id=None)
    pip = insert_entity(mem, name="pip", scope=Scope.GLOBAL, project_id=None)
    embedded_relation(mem, uv, "preferred_over", pip)
    mem.search_relations("uv pip")
    after = [(r.record.id, r.score) for r in mem.search("uv pip")]  # scores drift by recency only
    assert [i for i, _ in after] == [i for i, _ in before]
    assert [s for _, s in after] == pytest.approx([s for _, s in before], abs=1e-6)


# ---- final-review fixes ----

def test_unverifying_update_crash_never_leaves_relation_over_verified(mem, monkeypatch):
    r = one_relation(mem, origin="user")
    assert r.status == "verified"
    (m,) = mem.list()
    real = mem.vectors.upsert
    state = {"boom": True}

    def flaky(col, points):
        if col == mem.col and state["boom"]:
            state["boom"] = False
            raise RuntimeError("crash")
        return real(col, points)

    monkeypatch.setattr(mem.vectors, "upsert", flaky)
    with pytest.raises(RuntimeError):
        mem.update(m.id, m.text, actor="mcp", unverify=True)
    # the memory write never landed; the relation must not be verified on an unverified-only claim
    assert mem.get_relation(r.id).status != Status.VERIFIED or mem.get(m.id).status == Status.VERIFIED
    mem.update(m.id, m.text, actor="mcp", unverify=True)
    assert mem.get(m.id).status == Status.UNVERIFIED
    assert mem.get_relation(r.id).status == Status.UNVERIFIED


def test_unverifying_update_crash_after_memory_write_is_safe(mem, monkeypatch):
    r = one_relation(mem, origin="user")
    (m,) = mem.list()

    real = mem.db.add_history

    def boom(kind, *a, **kw):
        if kind == TargetKind.MEMORY:
            raise RuntimeError("crash")
        return real(kind, *a, **kw)

    monkeypatch.setattr(mem.db, "add_history", boom)
    with pytest.raises(RuntimeError):
        mem.update(m.id, m.text, actor="mcp", unverify=True)
    assert mem.get(m.id).status == Status.UNVERIFIED
    assert mem.get_relation(r.id).status == Status.UNVERIFIED


def test_confirm_on_already_confirmed_memory_repairs_relation(mem):
    r = one_relation(mem)
    assert r.status == "unverified"
    (m,) = mem.list()
    # simulate a crash after the confirm payload write, before the relation refresh
    mem.vectors.update_payload(mem.col, m.id, {"provenance": "user_confirmed", "status": "verified"})
    assert mem.get_relation(r.id).status == Status.UNVERIFIED
    mem.confirm(m.id, actor="cli")
    assert mem.get_relation(r.id).status == Status.VERIFIED


def test_search_relations_entity_filter_not_crowded_out(mem, monkeypatch):
    from rem.retrieve import overfetch
    from rem.storage.vector.base import ScoredPoint

    uv = insert_entity(mem, name="uv", scope=Scope.GLOBAL, project_id=None)
    pip = insert_entity(mem, name="pip", scope=Scope.GLOBAL, project_id=None)
    mine = insert_relation(mem, uv, "uses", pip)
    others = []
    for i in range(overfetch(5) + 10):
        a = insert_entity(mem, name=f"a{i}", scope=Scope.GLOBAL, project_id=None)
        b = insert_entity(mem, name=f"b{i}", scope=Scope.GLOBAL, project_id=None)
        others.append(insert_relation(mem, a, "uses", b))
    ranked = [ScoredPoint(r.id, 0.9, r.to_payload()) for r in others] + [ScoredPoint(mine.id, 0.5, mine.to_payload())]
    monkeypatch.setattr(mem.vectors, "search", lambda col, vec, limit, filters=None: ranked[:limit])
    assert [r.id for r in mem.search_relations("anything", entity="uv", limit=5)] == [mine.id]


def test_add_row_repaired_after_delete_and_recreate_in_later_job(mem, monkeypatch):
    r = one_relation(mem, job_key="job:1")
    (m,) = mem.list()
    mem.delete(m.id, actor="cli")
    assert mem.get_relation(r.id) is None
    c = [rcand("uv runs Python scripts", ["uv", "Python"], REL, origin="assistant")]
    real = mem.db.add_history
    state = {"boom": True}

    def flaky(kind, *a, **kw):
        if kind == TargetKind.RELATION and state["boom"]:
            state["boom"] = False
            raise RuntimeError("crash")
        return real(kind, *a, **kw)

    monkeypatch.setattr(mem.db, "add_history", flaky)
    with pytest.raises(RuntimeError):
        apply(mem, c, job_key="job:3")
    apply(mem, c, job_key="job:3")
    assert (TargetKind.RELATION, HistoryEvent.ADD, "job:3") in rhist(mem, r.id)


def test_context_is_unchanged_by_relations(mem):
    base = [rcand("Prefers uv over pip", ["uv", "pip"], []), rcand("Uses Python", ["Python"], [], scope="system")]
    apply(mem, base, job_key="job:1")
    before = mem.context(None, 1600)
    withrel = [rcand("Prefers uv over pip", ["uv", "pip"], [("uv", "preferred_over", "pip")]),
               rcand("Uses Python", ["Python"], [], scope="system")]
    apply(mem, withrel, job_key="job:2")
    assert mem.list_relations() == [] or len(mem.list_relations()) == 1
    assert len(mem.list_relations()) == 1
    assert mem.context(None, 1600) == before
    assert before.count == 2
