import hashlib
import math
import re

from rem.embed.base import Embedder


class HashingEmbedder(Embedder):
    """Deterministic bag-of-words embedder: no model, no network. For tests and offline use."""

    def __init__(self, dims: int = 64):
        self._dims = dims

    @property
    def model_id(self) -> str:
        return f"hash-{self._dims}"

    @property
    def dims(self) -> int:
        return self._dims

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self._dims
        for token in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.sha1(token.encode()).hexdigest(), 16) % self._dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)
