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


# ---- entities (M7) ----------------------------------------------------------------


def with_entities(mem, text, *names, job_key="job:1"):
    """Write a memory through the extraction path, so its entities are created and linked."""
    from rem.models import Candidate

    cand = Candidate(text=text, scope="global", origin="user", entities=[{"name": n, "type": "tool"} for n in names])
    mem.apply_candidates([cand], job_key=job_key, project_id=None, session_id=None, source="cli",
                         restate_min_similarity=0.9, has_user_turn=True, user_texts=[text])
    return next(r for r in mem.list() if r.text == text)


def vectors(mem, col):
    return {p.id: p.vector for p in mem.vectors.get(col, list(points(mem, col)), with_vectors=True)}


def test_reembed_copies_entities_with_same_ids_and_payloads(mem):
    with_entities(mem, "Uses uv", "uv")
    with_entities(mem, "Uses Qdrant", "Qdrant", job_key="job:2")
    before = points(mem, mem.entity_col)
    assert len(before) == 2
    reembed.run(mem, "hash-32")
    target = json.loads(mem.db.get_meta(reembed.TARGET_KEY))["entity_collection"]
    assert target == f"rem_entities__{collection_slug('hash-32', 32)}" and target != mem.entity_col
    assert points(mem, target) == before
    vecs = vectors(mem, target)
    assert all(len(v) == 32 for v in vecs.values())
    expected = reembed._target_embedder(mem, "hash-32")
    assert all(vecs[i] == expected.embed([p["normalized"]])[0] for i, p in before.items())
    assert not mem.vectors.col_has_sparse(target)


def test_same_model_reembed_puts_entities_in_a_new_generation(mem):
    with_entities(mem, "Uses uv", "uv")
    reembed.run(mem, "hash-64")
    target = json.loads(mem.db.get_meta(reembed.TARGET_KEY))["entity_collection"]
    assert target != mem.entity_col and target.endswith("__g1")


def test_finalize_switches_the_entity_collection_and_drops_the_old_one(mem, cfg):
    from rem.config import EmbedderConfig
    from rem.memory import ACTIVE_ENTITY_COLLECTION_KEY, Memory

    with_entities(mem, "Uses uv", "uv")
    before = points(mem, mem.entity_col)
    old = mem.entity_col
    reembed.run(mem, "hash-32")
    target = json.loads(mem.db.get_meta(reembed.TARGET_KEY))["entity_collection"]
    reembed.finalize(mem)
    assert mem.entity_col == target
    assert mem.db.get_meta(ACTIVE_ENTITY_COLLECTION_KEY) == target
    assert not mem.vectors.col_exists(old)
    assert points(mem, target) == before
    fresh = Memory(cfg.model_copy(update={"embedder": EmbedderConfig(provider="hashing", model="hash-32")}),
                   vector_store=mem.vectors)
    try:
        assert fresh.entity_col == target
        assert [e.id for e in fresh.list_entities()] == list(before)
    finally:
        fresh.db.close()


def test_finalize_syncs_entities_changed_after_the_copy(mem):
    a = with_entities(mem, "Uses uv", "uv")
    b = with_entities(mem, "Uses Qdrant", "Qdrant", job_key="job:2")
    reembed.run(mem, "hash-32")
    with_entities(mem, "Uses ruff", "ruff", job_key="job:3")      # new entity
    with_entities(mem, "Uses UV everywhere", "UV", job_key="job:4")  # memory_ids change on uv (no alias: F6)
    mem.delete(b.id, actor="cli")                                  # drops the Qdrant entity
    expected = points(mem, mem.entity_col)
    assert len(expected) == 2
    reembed.finalize(mem)
    assert points(mem, mem.entity_col) == expected
    assert all(len(v) == 32 for v in vectors(mem, mem.entity_col).values())
    (uv,) = [e for e in mem.list_entities() if e.normalized == "uv"]
    assert uv.aliases == [] and a.id in uv.memory_ids and len(uv.memory_ids) == 2


def test_finalize_of_a_reembed_pending_from_before_m7_copies_entities(mem):
    """A pending target without `entity_collection` (recorded by a pre-M7 rem): finalize picks one and copies all."""
    with_entities(mem, "Uses uv", "uv")
    before = points(mem, mem.entity_col)
    old = mem.entity_col
    reembed.run(mem, "hash-32")
    pending = json.loads(mem.db.get_meta(reembed.TARGET_KEY))
    entity_target = pending.pop("entity_collection")
    mem.vectors.delete_col(entity_target)
    mem.db.set_meta(reembed.TARGET_KEY, json.dumps(pending))
    reembed.finalize(mem)
    assert mem.entity_col == entity_target and mem.entity_col != old
    assert not mem.vectors.col_exists(old)
    assert points(mem, mem.entity_col) == before
    assert all(len(v) == 32 for v in vectors(mem, mem.entity_col).values())


def test_entity_boost_still_works_after_finalize(mem):
    from rem.entities import ENTITY_BOOST

    a = with_entities(mem, "the build uses tabs for indentation", "tabs")
    b = add(mem, "the build uses spaces for indentation")
    reembed.run(mem, "hash-32")
    reembed.finalize(mem)
    scores = {r.record.id: r.score for r in mem.search("tabs")}
    assert a.id in scores
    # same query against the memories alone: the gap is exactly the entity boost
    mem.vectors.delete_col(mem.entity_col)
    plain = {r.record.id: r.score for r in mem.search("tabs")}
    assert scores[a.id] == pytest.approx(plain[a.id] + ENTITY_BOOST)
    assert scores.get(b.id) == plain.get(b.id)


def test_finalize_meta_writes_are_one_transaction(mem, monkeypatch):
    add(mem, "uses tabs")
    reembed.run(mem, "hash-32")
    keys = [reembed.ACTIVE_COLLECTION_KEY, reembed.ACTIVE_ENTITY_COLLECTION_KEY, reembed.ACTIVE_EMBEDDER_KEY,
            reembed.ACTIVE_MODEL_KEY, reembed.TARGET_KEY]
    before = {k: mem.db.get_meta(k) for k in keys}
    old_col, old_entity = mem.col, mem.entity_col
    real = mem.db.delete_meta

    def crash(key):
        raise RuntimeError("crash")

    monkeypatch.setattr(mem.db, "delete_meta", crash)  # the last meta write of finalize
    with pytest.raises(RuntimeError):
        reembed.finalize(mem)
    assert {k: mem.db.get_meta(k) for k in keys} == before  # nothing was switched in the store
    assert (mem.col, mem.entity_col) == (old_col, old_entity)  # nor in this process
    assert mem.vectors.col_exists(old_col) and mem.vectors.col_exists(old_entity)
    monkeypatch.setattr(mem.db, "delete_meta", real)
    col = reembed.finalize(mem)  # the pending target is still there: finalize simply runs again
    assert mem.col == col and mem.db.get_meta(reembed.TARGET_KEY) is None
    assert [r.record.text for r in mem.search("uses tabs")] == ["uses tabs"]


# ---- relations (M8) ---------------------------------------------------------------


def with_relation(mem, text, src, dst, job_key="job:1"):
    from rem.models import Candidate, RelationMention

    cand = Candidate(text=text, scope="global", origin="user",
                     entities=[{"name": src, "type": "tool"}, {"name": dst, "type": "tool"}],
                     relations=[RelationMention(source=src, type="preferred_over", target=dst)])
    mem.apply_candidates([cand], job_key=job_key, project_id=None, session_id=None, source="cli",
                         restate_min_similarity=0.9, has_user_turn=True, user_texts=[text])
    return next(r for r in mem.list() if r.text == text)


def test_reembed_copies_relations_same_ids_and_payloads(mem):
    with_relation(mem, "uv is preferred over pip", "uv", "pip")
    with_relation(mem, "ruff is preferred over flake8", "ruff", "flake8", job_key="job:2")
    before = points(mem, mem.relation_col)
    assert len(before) == 2
    reembed.run(mem, "hash-32")
    target = json.loads(mem.db.get_meta(reembed.TARGET_KEY))["relation_collection"]
    assert target == f"rem_relations__{collection_slug('hash-32', 32)}" and target != mem.relation_col
    assert points(mem, target) == before
    vecs = vectors(mem, target)
    expected = reembed._target_embedder(mem, "hash-32")
    assert all(vecs[i] == expected.embed([p["sentence"]])[0] for i, p in before.items())
    assert not mem.vectors.col_has_sparse(target)


def test_finalize_switches_relation_collection_and_meta(mem):
    from rem.memory import ACTIVE_RELATION_COLLECTION_KEY

    with_relation(mem, "uv is preferred over pip", "uv", "pip")
    before = points(mem, mem.relation_col)
    old = mem.relation_col
    reembed.run(mem, "hash-32")
    target = json.loads(mem.db.get_meta(reembed.TARGET_KEY))["relation_collection"]
    reembed.finalize(mem)
    assert mem.relation_col == target
    assert mem.db.get_meta(ACTIVE_RELATION_COLLECTION_KEY) == target
    assert not mem.vectors.col_exists(old)
    assert points(mem, target) == before
    assert [r.id for r in mem.list_relations()] == list(before)


def test_finalize_with_pre_m8_pending_target_copies_all_relations(mem):
    with_relation(mem, "uv is preferred over pip", "uv", "pip")
    before = points(mem, mem.relation_col)
    old = mem.relation_col
    reembed.run(mem, "hash-32")
    pending = json.loads(mem.db.get_meta(reembed.TARGET_KEY))
    rel_target = pending.pop("relation_collection")
    mem.vectors.delete_col(rel_target)
    mem.db.set_meta(reembed.TARGET_KEY, json.dumps(pending))
    reembed.finalize(mem)
    assert mem.relation_col == rel_target and mem.relation_col != old
    assert not mem.vectors.col_exists(old)
    assert points(mem, mem.relation_col) == before
    assert all(len(v) == 32 for v in vectors(mem, mem.relation_col).values())


def test_finalize_syncs_relations_written_after_the_copy(mem):
    a = with_relation(mem, "uv is preferred over pip", "uv", "pip")
    b = with_relation(mem, "ruff is preferred over flake8", "ruff", "flake8", job_key="job:2")
    reembed.run(mem, "hash-32")
    with_relation(mem, "poetry is preferred over pipenv", "poetry", "pipenv", job_key="job:3")  # new
    mem.confirm(a.id, actor="cli")  # status change on an existing relation
    mem.delete(b.id, actor="cli")   # drops one
    expected = points(mem, mem.relation_col)
    assert len(expected) == 2
    reembed.finalize(mem)
    assert points(mem, mem.relation_col) == expected
    assert all(len(v) == 32 for v in vectors(mem, mem.relation_col).values())
