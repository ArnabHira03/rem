import json

import pytest

from rem import reembed
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
