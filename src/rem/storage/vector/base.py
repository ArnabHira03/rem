"""Vector store interface, modelled on Mem0's VectorStoreBase.

The payload of each point is the canonical memory record; the vector is its embedding.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Scalar = equality, list = match any; keys are ANDed. None matches a null/missing field.
Filters = dict[str, str | int | None | list[str | int]]


@dataclass
class SparseVector:
    """Sparse term weights for BM25: parallel lists of term index and weight."""

    indices: list[int]
    values: list[float]


@dataclass
class VectorPoint:
    id: str
    vector: list[float] | None
    payload: dict
    sparse: SparseVector | None = None


@dataclass
class ScoredPoint:
    id: str
    score: float
    payload: dict


class VectorStore(ABC):
    @abstractmethod
    def create_col(self, name: str, vector_size: int, *, sparse: bool = False) -> None:
        """Create a collection if it does not exist. sparse=True adds the BM25 slot ``bm25``."""

    @abstractmethod
    def col_exists(self, name: str) -> bool: ...

    @abstractmethod
    def col_has_sparse(self, name: str) -> bool:
        """True when the collection has the BM25 sparse slot."""

    @abstractmethod
    def delete_col(self, name: str) -> None: ...

    @abstractmethod
    def upsert(self, col: str, points: list[VectorPoint]) -> None: ...

    @abstractmethod
    def search(
        self, col: str, vector: list[float], limit: int, filters: Filters | None = None
    ) -> list[ScoredPoint]:
        """Cosine-similarity search, highest score first."""

    @abstractmethod
    def get(self, col: str, ids: list[str], with_vectors: bool = False) -> list[VectorPoint]:
        """Points for the given ids; missing ids are omitted. with_vectors also returns vector and sparse."""

    @abstractmethod
    def update_payload(self, col: str, id: str, payload: dict) -> None:
        """Merge keys into an existing point's payload."""

    @abstractmethod
    def delete(self, col: str, ids: list[str]) -> None: ...

    @abstractmethod
    def list_points(
        self, col: str, filters: Filters | None = None, limit: int | None = None
    ) -> list[VectorPoint]:
        """Every point matching filters; limit=None means all of them."""

    def keyword_search(
        self, col: str, query_sparse: SparseVector, limit: int, filters: Filters | None = None
    ) -> list[ScoredPoint] | None:
        """BM25 keyword search over the sparse slot; None when the collection has no such slot."""
        return None

    def close(self) -> None:
        return None
