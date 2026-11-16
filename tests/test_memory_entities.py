from rem.entities import entity_id, normalize_name
from rem.memory import Memory
from rem.models import EntityRecord, Scope, utcnow
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
