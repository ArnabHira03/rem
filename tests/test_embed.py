import pytest

from rem.config import EmbedderConfig
from rem.embed.base import collection_slug
from rem.embed.factory import create_embedder
from rem.embed.fastembed import FastEmbedEmbedder
from rem.embed.hashing import HashingEmbedder
from rem.errors import EmbedderUnavailableError


def test_slug():
    assert collection_slug("BAAI/bge-small-en-v1.5", 384) == "baai-bge-small-en-v1-5_384"


def test_hashing_embedder_similarity_tracks_word_overlap():
    e = HashingEmbedder(64)
    a, b, c = e.embed(["prefers uv over pip", "uv over pip preferred", "laptop has 15 GB RAM"])

    def cos(x, y):
        return sum(i * j for i, j in zip(x, y))

    assert cos(a, b) > cos(a, c)
    assert len(a) == 64 and e.model_id == "hash-64"


def test_factory_builds_hashing_with_dims_from_model():
    e = create_embedder(EmbedderConfig(provider="hashing", model="hash-32"))
    assert isinstance(e, HashingEmbedder) and e.dims == 32


def test_fastembed_unavailable_is_clear(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr("rem.embed.fastembed.TextEmbedding", boom)
    with pytest.raises(EmbedderUnavailableError, match="BAAI/bge-small-en-v1.5"):
        FastEmbedEmbedder("BAAI/bge-small-en-v1.5", tmp_path)


@pytest.mark.slow
def test_fastembed_real_model(tmp_path):
    e = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", tmp_path)
    assert e.dims == 384 and len(e.embed_query("which package manager?")) == 384
