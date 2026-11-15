from rem.config import RemConfig
from rem.storage.vector.base import VectorStore
from rem.storage.vector.memory import InMemoryVectorStore
from rem.storage.vector.qdrant import QdrantVectorStore


def create_vector_store(cfg: RemConfig) -> VectorStore:
    if cfg.vector_store.provider == "memory":
        return InMemoryVectorStore()
    if cfg.vector_store.url:
        return QdrantVectorStore(url=cfg.vector_store.url)
    cfg.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantVectorStore(path=cfg.qdrant_path)
