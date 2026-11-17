from rem.memory import MEMORY_GRAPH_FIELDS

from tests.test_memory_entities import apply
from tests.test_memory_relations import rcand


def _scroll(mem, col):
    return sorted((p.payload for p in mem.vectors.list_points(col)), key=lambda p: p["id"])


def _repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    return repo


def test_graph_snapshot_returns_stored_records_verbatim(mem):
    apply(mem, [rcand("Prefers uv over pip", ["uv", "pip"], [("uv", "preferred_over", "pip")])])
    snap = mem.graph_snapshot(None)
    assert set(snap) == {"generated_at", "current_project_id", "projects", "entities", "relations", "memories"}
    assert snap["entities"] == sorted((e.to_payload() for e in mem.list_entities()), key=lambda p: p["id"])
    assert len(snap["entities"]) == 2
    assert snap["relations"] == _scroll(mem, mem.relation_col) and len(snap["relations"]) == 1
    stored = {p["id"]: p for p in _scroll(mem, mem.col)}
    assert len(snap["memories"]) == 1
    for m in snap["memories"]:
        assert set(m) == set(MEMORY_GRAPH_FIELDS)
        assert m == {k: stored[m["id"]][k] for k in MEMORY_GRAPH_FIELDS}


def test_graph_snapshot_lists_projects_and_current_project(mem, tmp_path):
    repo = _repo(tmp_path)
    pid = mem.db.upsert_project(str(repo), "repo", None)
    (repo / "sub").mkdir()
    snap = mem.graph_snapshot(repo / "sub")
    assert snap["current_project_id"] == pid
    assert snap["projects"] == [{"id": pid, "name": "repo", "root_path": str(repo)}]


def test_graph_snapshot_unknown_repo_has_no_current_project(mem, tmp_path):
    repo = _repo(tmp_path)
    assert mem.graph_snapshot(repo)["current_project_id"] is None
    assert mem.graph_snapshot(None)["current_project_id"] is None


def test_graph_snapshot_writes_nothing(mem):
    apply(mem, [rcand("Prefers uv over pip", ["uv", "pip"], [("uv", "preferred_over", "pip")])])

    def state():
        n = mem.db._conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        return n, [_scroll(mem, c) for c in (mem.col, mem.entity_col, mem.relation_col)]

    before = state()
    mem.graph_snapshot(None)
    mem.graph_snapshot(None)
    assert state() == before


def test_graph_snapshot_tolerates_payloads_missing_model_defaults(mem):
    """A memory written before M5 has no `agent_claim`; an entity payload may lack `aliases`."""
    from rem.models import EntityRecord, MemoryRecord, utcnow
    from rem.storage.vector.base import VectorPoint

    now = utcnow()
    old_mem = MemoryRecord(
        id="00000000-0000-0000-0000-000000000001", text="Prefers uv", hash="h", scope="global", project_id=None,
        project_name=None, provenance="user_asserted", status="verified", evidence_source=None, observed_at=None,
        source="cli", session_id=None, job_id=None, created_at=now, updated_at=now,
    ).to_payload()
    del old_mem["agent_claim"]
    old_ent = EntityRecord(
        id="00000000-0000-0000-0000-000000000002", name="uv", normalized="uv", type="tool", scope="global",
        project_id=None, memory_ids=[old_mem["id"]], created_at=now, updated_at=now,
    ).to_payload()
    del old_ent["aliases"]
    zero = [0.0] * mem.embedder.dims
    mem.vectors.upsert(mem.col, [VectorPoint(old_mem["id"], zero, old_mem)])
    mem.vectors.upsert(mem.entity_col, [VectorPoint(old_ent["id"], zero, old_ent)])

    snap = mem.graph_snapshot(None)
    [m] = snap["memories"]
    assert m["agent_claim"] is False and set(m) == set(MEMORY_GRAPH_FIELDS)
    [e] = snap["entities"]
    assert e["aliases"] == []
    # read-only: the stored payloads are not rewritten
    assert "agent_claim" not in _scroll(mem, mem.col)[0]
    assert "aliases" not in _scroll(mem, mem.entity_col)[0]
