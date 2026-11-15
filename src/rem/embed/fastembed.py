from pathlib import Path

from fastembed import TextEmbedding

from rem.embed.base import Embedder
from rem.errors import EmbedderUnavailableError


class FastEmbedEmbedder(Embedder):
    """Local ONNX embeddings on CPU via fastembed (the model downloads once into cache_dir)."""

    def __init__(self, model: str, cache_dir: Path):
        self._model_name = model
        try:
            self._model = TextEmbedding(model_name=model, cache_dir=str(cache_dir))
            self._dims = len(next(iter(self._model.embed(["dimension probe"]))))
        except Exception as e:
            raise EmbedderUnavailableError(
                f"could not load embedding model {model} (cache: {cache_dir}); "
                "the first run needs network access to download it"
            ) from e

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def dims(self) -> int:
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()
