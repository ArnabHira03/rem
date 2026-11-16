import pytest

from rem.config import EmbedderConfig
from rem.errors import EmbedderMismatchError, MemoryNotFound
from rem.memory import Memory
from rem.models import Provenance, Scope, Status
from rem.storage.vector.memory import InMemoryVectorStore
from tests.conftest import add


def test_add_get_and_history(mem):
    r = add(mem)
    assert mem.get(r.id).text == "Prefers uv over pip" and r.entity_ids == []
    assert [h.event for h in mem.history(r.id)] == ["ADD"]


def test_duplicate_add_returns_existing_without_new_history(mem):
    a = add(mem)
    b = add(mem, "  prefers UV over pip ")
    assert a.id == b.id and len(mem.list()) == 1 and len(mem.history(a.id)) == 1


def test_project_scope_resolves_project(mem, tmp_path):
    (tmp_path / "p" / ".git").mkdir(parents=True)
    r = add(mem, "rem uses Qdrant", scope=Scope.PROJECT, cwd=tmp_path / "p")
    assert r.project_name == "p" and r.project_id is not None
    with pytest.raises(ValueError, match="needs a working directory"):
        add(mem, "x", scope=Scope.PROJECT)


def test_list_filters(mem, tmp_path):
    add(mem, "Prefers uv")
    add(mem, "OS is Fedora", scope=Scope.SYSTEM)
    assert [r.text for r in mem.list(scope=Scope.SYSTEM)] == ["OS is Fedora"]
    assert len(mem.list(status=Status.VERIFIED)) == 2


def test_update_confirm_delete_and_history(mem):
    r = mem.add("Uses npm", scope=Scope.GLOBAL, provenance=Provenance.ASSISTANT_GENERATED,
                status=Status.UNVERIFIED, source="mcp", actor="mcp")
    mem.update(r.id, "Uses pnpm", actor="cli")
    c = mem.confirm(r.id, actor="cli")
    assert c.status == "verified" and c.provenance == "user_confirmed" and c.text == "Uses pnpm"
    mem.delete(r.id, actor="cli")
    assert mem.get(r.id) is None
    h = mem.history(r.id)
    assert [x.event for x in h] == ["ADD", "UPDATE", "CONFIRM", "DELETE"]
    assert (h[1].old_value, h[1].new_value) == ("Uses npm", "Uses pnpm") and h[3].old_value == "Uses pnpm"


def test_unknown_id(mem):
    assert mem.get("00000000-0000-0000-0000-000000000000") is None
    for call in (
        lambda: mem.update("nope", "x", actor="cli"),
        lambda: mem.delete("nope", actor="cli"),
        lambda: mem.confirm("nope", actor="cli"),
        lambda: mem.history("nope"),
    ):
        with pytest.raises(MemoryNotFound):
            call()


def test_empty_text_rejected(mem):
    with pytest.raises(ValueError, match="text must not be empty"):
        add(mem, "   ")
    r = add(mem)
    with pytest.raises(ValueError, match="text must not be empty"):
        mem.update(r.id, " ", actor="cli")


def test_crash_between_vector_write_and_history_replays_cleanly(mem, monkeypatch):
    real = mem.db.add_history
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("crash")
        return real(*a, **k)

    monkeypatch.setattr(mem.db, "add_history", flaky)
    with pytest.raises(RuntimeError):
        add(mem, job_id="job-1")
    r = add(mem, job_id="job-1")
    assert len(mem.list()) == 1 and [h.event for h in mem.history(r.id)] == ["ADD"]


def test_embedder_mismatch_refuses(cfg):
    Memory(cfg).close()
    other = cfg.model_copy(update={"embedder": EmbedderConfig(provider="hashing", model="hash-32")})
    with pytest.raises(EmbedderMismatchError, match="rem reembed"):
        Memory(other, vector_store=InMemoryVectorStore())


def test_add_after_update_stores_the_new_fact(mem):
    a = add(mem, "Prefers uv")
    mem.update(a.id, "Prefers pip", actor="cli")
    b = add(mem, "Prefers uv")
    assert b.id != a.id and b.text == "Prefers uv"
    assert sorted(r.text for r in mem.list()) == ["Prefers pip", "Prefers uv"]


def test_add_matching_updated_text_returns_existing(mem):
    a = add(mem, "Prefers uv")
    mem.update(a.id, "Prefers pip", actor="cli")
    assert add(mem, "prefers  PIP").id == a.id and len(mem.list()) == 1


def test_update_to_text_of_another_memory_is_refused(mem):
    add(mem, "Prefers pip")
    b = add(mem, "Prefers uv")
    with pytest.raises(ValueError, match="already has this text"):
        mem.update(b.id, "prefers PIP", actor="cli")
    assert mem.get(b.id).text == "Prefers uv"


def test_crash_during_direct_add_recovers_history_on_retry(mem, monkeypatch):
    real = mem.db.add_history
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("crash")
        return real(*a, **k)

    monkeypatch.setattr(mem.db, "add_history", flaky)
    with pytest.raises(RuntimeError):
        add(mem)
    r = add(mem)
    assert [h.event for h in mem.history(r.id)] == ["ADD"]
    add(mem)
    assert [h.event for h in mem.history(r.id)] == ["ADD"]


def test_unverified_update_not_reverified_by_replay(mem):
    m = add(mem, "Prefers uv over pip")
    mem.corroborate(m.id, provenance=Provenance.USER_ASSERTED, job_id="J", actor="worker")
    mem.update(m.id, "Prefers pip", actor="mcp", unverify=True)
    r = mem.corroborate(m.id, provenance=Provenance.USER_ASSERTED, job_id="J", actor="worker")
    assert r.status == Status.UNVERIFIED
    assert mem.get(m.id).status == Status.UNVERIFIED
    assert mem.get(m.id).provenance == Provenance.ASSISTANT_GENERATED


def test_update_unverify_downgrades_a_verified_memory(mem):
    m = add(mem, "Prefers uv over pip")
    assert m.status == Status.VERIFIED and m.provenance == Provenance.USER_CONFIRMED
    kept = mem.update(m.id, "Prefers uv", actor="cli")
    assert kept.status == Status.VERIFIED and kept.provenance == Provenance.USER_CONFIRMED
    r = mem.update(m.id, "Prefers pip", actor="mcp", unverify=True)
    assert r.text == "Prefers pip" and r.corroborated_by is None  # unchanged (was None)
    assert r.status == Status.UNVERIFIED and r.provenance == Provenance.ASSISTANT_GENERATED
    stored = mem.get(m.id)
    assert stored.status == Status.UNVERIFIED and stored.provenance == Provenance.ASSISTANT_GENERATED
    assert stored.text == "Prefers pip"
    assert [(h.event, h.old_value, h.new_value) for h in mem.history(m.id)][1:] == [
        ("UPDATE", "Prefers uv over pip", "Prefers uv"), ("UPDATE", "Prefers uv", "Prefers pip")]


def test_add_stores_bm25_slot(mem):
    r = add(mem, "uses Qdrant for storage")
    (p,) = mem.vectors.get(mem.col, [r.id], with_vectors=True)
    assert p.sparse is not None and len(p.sparse.indices) > 0


def test_update_refreshes_bm25_slot(mem):
    r = add(mem, "uses Qdrant for storage")
    mem.update(r.id, "uses Postgres for storage", actor="cli")
    (p,) = mem.vectors.get(mem.col, [r.id], with_vectors=True)
    assert p.sparse == mem.sparse_embedder.embed_docs(["uses Postgres for storage"])[0]


def test_dense_only_collection_is_refused_before_writing(cfg):
    from rem.embed.base import collection_slug
    from rem.embed.hashing import HashingEmbedder
    from rem.storage.sqlite import RemDB

    e = HashingEmbedder(64)
    vs = InMemoryVectorStore()
    vs.create_col(f"rem_memories__{collection_slug(e.model_id, e.dims)}", e.dims)  # M5 layout: no BM25 slot
    with pytest.raises(EmbedderMismatchError, match="rem reembed"):
        Memory(cfg, vector_store=vs, embedder=e)
    assert RemDB(cfg.db_path).get_meta("active_embedder") is None
