from rem.config import DEFAULT_CACHE_DIR, EmbedderConfig
from rem.embed.base import Embedder
from rem.embed.hashing import HashingEmbedder
from rem.embed.sparse import HashingBM25, SparseEmbedder


def create_embedder(cfg: EmbedderConfig) -> Embedder:
    if cfg.provider == "hashing":
        _, _, dims = cfg.model.partition("-")
        return HashingEmbedder(int(dims) if dims.isdigit() else 64)
    from rem.embed.fastembed import FastEmbedEmbedder  # heavy import, only when used

    return FastEmbedEmbedder(cfg.model, cfg.cache_dir or DEFAULT_CACHE_DIR)


def create_sparse_embedder(cfg: EmbedderConfig) -> SparseEmbedder:
    if cfg.provider == "hashing":
        return HashingBM25()
    from rem.embed.sparse import FastEmbedBM25  # heavy import, only when used

    return FastEmbedBM25(cfg.cache_dir or DEFAULT_CACHE_DIR)
