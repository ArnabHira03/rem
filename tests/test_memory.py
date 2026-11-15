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
