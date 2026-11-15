import re
from abc import ABC, abstractmethod


class Embedder(ABC):
    """Turns text into vectors. Configured independently of the extraction LLM."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def dims(self) -> int: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


def collection_slug(model_id: str, dims: int) -> str:
    """Name fragment tying a collection to one embedding model, e.g. 'baai-bge-small-en-v1-5_384'."""
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")
    return f"{slug}_{dims}"
