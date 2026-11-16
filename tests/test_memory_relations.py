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
