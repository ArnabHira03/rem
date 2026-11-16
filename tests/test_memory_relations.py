import pytest

from rem.memory import Memory
from rem.models import Provenance, RelationRecord, Scope, Status, utcnow
from rem.relations import relation_id, relation_sentence
from rem.storage.vector.base import VectorPoint

from tests.test_memory_entities import insert_entity


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

from rem.models import Candidate, HistoryEvent, TargetKind  # noqa: E402
from tests.test_memory_entities import apply  # noqa: E402


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
    assert r.evidence == 1


def test_two_facts_same_edge_one_relation_two_evidence(mem):
    rel = [("uv", "uses", "Python")]
    apply(mem, [rcand("uv runs Python scripts", ["uv", "Python"], rel),
                rcand("uv is a Python tool", ["uv", "Python"], rel)])
    (r,) = mem.list_relations()
    assert r.evidence == 2 and len(mem.list()) == 2


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
