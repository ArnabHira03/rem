import json

import pytest

from rem import reembed
from rem.embed.base import collection_slug
from rem.errors import ReembedError
from tests.conftest import add


def points(mem, col):
    return {p.id: p.payload for p in mem.vectors.list_points(col)}


def test_reembed_keeps_ids_and_payloads(mem):
    ids = [add(mem, t).id for t in ["uses tabs", "uses spaces", "uses qdrant"]]
    before = points(mem, mem.col)
    res = reembed.run(mem, "hash-32", batch=2)
    assert res.count == 3 and set(points(mem, res.collection)) == set(ids)
    assert points(mem, res.collection) == before
    assert json.loads(mem.db.get_meta(reembed.TARGET_KEY))["collection"] == res.collection


def test_reembed_twice_is_idempotent(mem):
    add(mem, "uses tabs")
    first = reembed.run(mem, "hash-32")
    second = reembed.run(mem, "hash-32")
    assert first.collection == second.collection
    assert len(points(mem, second.collection)) == 1


def test_reembed_to_the_same_model_uses_a_new_generation(mem):
    add(mem, "uses tabs")
    res = reembed.run(mem, "hash-64")  # M5 layout had no BM25 slot: a same-model rebuild must be possible
    assert res.collection != mem.col and res.collection.endswith("__g1")


def test_reembed_to_another_model_while_one_is_pending_is_refused(mem):
    add(mem, "uses tabs")
    reembed.run(mem, "hash-32")
    with pytest.raises(ReembedError, match="pending"):
        reembed.run(mem, "hash-16")


def test_finalize_switches_to_target_and_drops_old(mem):
    add(mem, "uses tabs")
    old = mem.col
    res = reembed.run(mem, "hash-32")
    assert reembed.finalize(mem) == res.collection
    assert mem.col == res.collection and mem.embedder.model_id == "hash-32"
    assert not mem.vectors.col_exists(old)
    assert mem.db.get_meta(reembed.TARGET_KEY) is None
    assert [r.record.text for r in mem.search("tabs")] == ["uses tabs"]


def test_finalize_catches_up_writes_made_during_reembed(mem):
    a = add(mem, "uses tabs")
    reembed.run(mem, "hash-32")
    b = add(mem, "uses spaces")                 # written to the old collection after the copy
    mem.update(a.id, "uses tabs everywhere", actor="cli")
    mem.delete(b.id, actor="cli")
    reembed.finalize(mem)
    assert {p.id: p.payload["text"] for p in mem.vectors.list_points(mem.col)} == {a.id: "uses tabs everywhere"}


def test_finalize_without_a_pending_reembed_is_refused(mem):
    with pytest.raises(ReembedError, match="no reembed"):
        reembed.finalize(mem)


def test_same_model_target_never_is_the_active_collection(mem):
    """F2: an orphan at the base name pushes the copy to __g1; a second same-model copy must not pick the
    active __g1 (which finalize would then delete)."""
    a = add(mem, "uses tabs")
    base = f"rem_memories__{collection_slug('hash-32', 32)}"
    mem.vectors.create_col(base, 32)                # orphan left by a crashed copy
    reembed.run(mem, "hash-32")
    reembed.finalize(mem)                           # active is now __g1
    active = mem.col
    assert active.endswith("__g1")
    res = reembed.run(mem, "hash-32")
    assert res.collection not in (active,)
    reembed.finalize(mem)
    assert mem.vectors.col_exists(mem.col)
    assert [h.record.id for h in mem.search("tabs")] == [a.id]


def test_finalize_refuses_a_target_that_is_the_active_collection(mem):
    """F2 guard: a corrupt pending record that points at the active collection must not be switched or deleted."""
    add(mem, "uses tabs")
    mem.db.set_meta(reembed.TARGET_KEY, json.dumps({"collection": mem.col, "slug": "x", "model": "hash-64"}))
    with pytest.raises(ReembedError, match="active"):
        reembed.finalize(mem)
    assert mem.vectors.col_exists(mem.col)


def test_pending_target_is_recorded_before_the_copy_and_a_crash_resumes(mem, monkeypatch):
    """F3: a copy killed after its first batch must be resumed by the rerun, not orphaned, and finalize completes it."""
    ids = [add(mem, t).id for t in ["uses tabs", "uses spaces", "uses qdrant"]]
    real_target = reembed._target_embedder(mem, "hash-32")

    class Dies:
        model_id = real_target.model_id
        dims = real_target.dims
        calls = 0

        def embed(self, texts):
            Dies.calls += 1
            if Dies.calls == 2:
                raise RuntimeError("killed")
            return real_target.embed(texts)

    monkeypatch.setattr(reembed, "_target_embedder", lambda m, model: Dies())
    with pytest.raises(RuntimeError):
        reembed.run(mem, "hash-32", batch=1)
    pending = json.loads(mem.db.get_meta(reembed.TARGET_KEY))   # recorded before the copy finished
    monkeypatch.undo()
    res = reembed.run(mem, "hash-32", batch=1)
    assert res.collection == pending["collection"]               # resumed, no orphan
    assert reembed.finalize(mem) == res.collection
    assert set(points(mem, mem.col)) == set(ids)
