"""Vector store interface, modelled on Mem0's VectorStoreBase.

The payload of each point is the canonical memory record; the vector is its embedding.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Scalar = equality, list = match any; keys are ANDed. None matches a null/missing field.
Filters = dict[str, str | int | None | list[str | int]]


@dataclass
class VectorPoint:
    id: str
    vector: list[float] | None
    payload: dict


@dataclass
class ScoredPoint:
    id: str
    score: float
    payload: dict


class VectorStore(ABC):
    @abstractmethod
    def create_col(self, name: str, vector_size: int) -> None:
        """Create a collection if it does not exist."""

    @abstractmethod
    def col_exists(self, name: str) -> bool: ...

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
    def get(self, col: str, ids: list[str]) -> list[VectorPoint]:
        """Points for the given ids; missing ids are omitted."""

    @abstractmethod
    def update_payload(self, col: str, id: str, payload: dict) -> None:
        """Merge keys into an existing point's payload."""

    @abstractmethod
    def delete(self, col: str, ids: list[str]) -> None: ...

    @abstractmethod
    def list_points(
        self, col: str, filters: Filters | None = None, limit: int = 10_000
    ) -> list[VectorPoint]: ...

    def keyword_search(
        self, col: str, query: str, limit: int, filters: Filters | None = None
    ) -> list[ScoredPoint] | None:
        """BM25 keyword search; None when the backend does not support it (M6)."""
        return None

    def close(self) -> None:
        return None
