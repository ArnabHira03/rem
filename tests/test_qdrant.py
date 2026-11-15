import pytest

from rem.config import VectorStoreConfig
from rem.errors import StoreLockedError
from rem.storage.vector.base import VectorPoint
from rem.storage.vector.factory import create_vector_store
from rem.storage.vector.memory import InMemoryVectorStore
from rem.storage.vector.qdrant import QdrantVectorStore

ID1 = "00000000-0000-0000-0000-000000000001"


def test_persists_across_reopen(tmp_path):
    s = QdrantVectorStore(path=tmp_path / "q")
    s.create_col("c", 3)
    s.upsert("c", [VectorPoint(ID1, [1, 0, 0], {"text": "a"})])
    s.close()
    s2 = QdrantVectorStore(path=tmp_path / "q")
    assert s2.get("c", [ID1])[0].payload["text"] == "a"
    s2.close()


def test_second_open_raises_store_locked(tmp_path):
    s = QdrantVectorStore(path=tmp_path / "q")
    with pytest.raises(StoreLockedError, match="in use by another rem process"):
        QdrantVectorStore(path=tmp_path / "q")
    s.close()


def test_factory_picks_provider(cfg, tmp_path):
    assert isinstance(create_vector_store(cfg), InMemoryVectorStore)
    q = create_vector_store(
        cfg.model_copy(update={"vector_store": VectorStoreConfig(provider="qdrant", path=tmp_path / "q")})
    )
    assert isinstance(q, QdrantVectorStore)
    q.close()
