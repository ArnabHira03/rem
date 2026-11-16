"""Behaviour every VectorStore backend must share (Mem0-style VectorStoreBase contract)."""

import pytest

from rem.storage.vector.base import VectorPoint
from rem.storage.vector.memory import InMemoryVectorStore
from rem.storage.vector.qdrant import QdrantVectorStore

ID1 = "00000000-0000-0000-0000-000000000001"
ID2 = "00000000-0000-0000-0000-000000000002"
ID3 = "00000000-0000-0000-0000-000000000003"


def make_store(kind, tmp_path):
    if kind == "memory":
        return InMemoryVectorStore()
    if kind == "qdrant":
        return QdrantVectorStore(location=":memory:")
    if kind == "qdrant-path":
        return QdrantVectorStore(path=tmp_path / "q")
    raise ValueError(kind)


@pytest.fixture(params=["memory", "qdrant", "qdrant-path"])
def store(request, tmp_path):
    s = make_store(request.param, tmp_path)
    s.create_col("c", 3)
    yield s
    s.close()


def test_upsert_get_and_idempotent_create(store):
    store.create_col("c", 3)
    store.upsert("c", [VectorPoint(ID1, [1, 0, 0], {"scope": "global", "text": "a"})])
    [p] = store.get("c", [ID1])
    assert p.payload["text"] == "a"
    assert store.col_exists("c") and not store.col_exists("nope")


def test_search_orders_by_cosine_and_filters(store):
    store.upsert("c", [
        VectorPoint(ID1, [1, 0, 0], {"scope": "global"}),
        VectorPoint(ID2, [0.9, 0.1, 0], {"scope": "project", "project_id": 7}),
        VectorPoint(ID3, [0, 1, 0], {"scope": "system"}),
    ])
    assert [r.id for r in store.search("c", [1, 0, 0], 3)] == [ID1, ID2, ID3]
    assert [r.id for r in store.search("c", [1, 0, 0], 3, {"scope": ["global", "system"]})] == [ID1, ID3]
    assert [r.id for r in store.search("c", [1, 0, 0], 3, {"scope": "project", "project_id": 7})] == [ID2]
    assert store.search("c", [1, 0, 0], 1)[0].score == pytest.approx(1.0, abs=1e-6)


def test_update_payload_merges_and_delete_removes(store):
    store.upsert("c", [VectorPoint(ID1, [1, 0, 0], {"status": "unverified", "text": "a"})])
    store.update_payload("c", ID1, {"status": "verified"})
    assert store.get("c", [ID1])[0].payload == {"status": "verified", "text": "a"}
    store.delete("c", [ID1])
    assert store.get("c", [ID1]) == [] and store.list_points("c") == []


def test_upsert_same_id_overwrites(store):
    store.upsert("c", [VectorPoint(ID1, [1, 0, 0], {"text": "a"})])
    store.upsert("c", [VectorPoint(ID1, [1, 0, 0], {"text": "a"})])
    assert len(store.list_points("c")) == 1


def test_list_points_filters_including_null(store):
    store.upsert("c", [
        VectorPoint(ID1, [1, 0, 0], {"scope": "global", "project_id": None}),
        VectorPoint(ID2, [0, 1, 0], {"scope": "project", "project_id": 7}),
    ])
    assert [p.id for p in store.list_points("c", {"project_id": None})] == [ID1]
    assert [p.id for p in store.list_points("c", {"project_id": 7})] == [ID2]


def test_keyword_search_unsupported_returns_none(store):
    assert store.keyword_search("c", "anything", 5) is None


def test_search_filters_on_payload_id(store):
    # Memory.similarity scores one memory by filtering on its payload "id" field.
    store.upsert("c", [
        VectorPoint(ID1, [1, 0, 0], {"id": ID1, "scope": "global"}),
        VectorPoint(ID2, [0.6, 0.8, 0], {"id": ID2, "scope": "global"}),
    ])
    [hit] = store.search("c", [1, 0, 0], 1, {"id": ID2, "scope": "global"})
    assert hit.id == ID2 and hit.score == pytest.approx(0.6, abs=1e-6)
    assert store.search("c", [1, 0, 0], 1, {"id": ID2, "scope": "system"}) == []


def test_list_points_returns_all_beyond_a_page(store, monkeypatch):
    monkeypatch.setattr("rem.storage.vector.qdrant.SCROLL_PAGE", 7)
    store.upsert("c", [
        VectorPoint(f"00000000-0000-0000-0000-{i:012d}", [1, 0, 0], {"scope": "global"}) for i in range(1, 26)
    ])
    assert len(store.list_points("c")) == 25
    assert len(store.list_points("c", limit=3)) == 3
