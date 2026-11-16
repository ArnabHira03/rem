"""BM25 sparse embeddings for the keyword slot. FastEmbed's Qdrant/bm25 model, or a hashing stand-in for tests."""

import re
import zlib
from abc import ABC, abstractmethod
from pathlib import Path

from rem.errors import EmbedderUnavailableError
from rem.storage.vector.base import SparseVector

BM25_MODEL = "Qdrant/bm25"
HASH_SPACE = 2**20  # term index space for HashingBM25


class SparseEmbedder(ABC):
    @abstractmethod
    def embed_docs(self, texts: list[str]) -> list[SparseVector]: ...

    @abstractmethod
    def embed_query(self, text: str) -> SparseVector: ...


class FastEmbedBM25(SparseEmbedder):
    """BM25 term weights from fastembed's Qdrant/bm25 model (downloaded once into cache_dir)."""

    def __init__(self, cache_dir: Path):
        from fastembed import SparseTextEmbedding  # heavy import, only when used

        try:
            self._model = SparseTextEmbedding(model_name=BM25_MODEL, cache_dir=str(cache_dir))
        except Exception as e:
            raise EmbedderUnavailableError(
                f"could not load keyword model {BM25_MODEL} (cache: {cache_dir}); "
                "the first run needs network access to download it"
            ) from e

    def embed_docs(self, texts):
        return [SparseVector(list(v.indices), list(v.values)) for v in self._model.embed(texts)]

    def embed_query(self, text):
        v = next(iter(self._model.query_embed(text)))
        return SparseVector(list(v.indices), list(v.values))


class HashingBM25(SparseEmbedder):
    """Token counts keyed by a stable hash: no model, no network. Same tokenization for docs and queries."""

    def _vector(self, text: str) -> SparseVector:
        counts: dict[int, float] = {}
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            idx = zlib.crc32(tok.encode()) % HASH_SPACE
            counts[idx] = counts.get(idx, 0.0) + 1.0
        return SparseVector(list(counts), list(counts.values()))

    def embed_docs(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)
