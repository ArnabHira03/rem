from rem.config import DEFAULT_CACHE_DIR, EmbedderConfig
from rem.embed.base import Embedder
from rem.embed.hashing import HashingEmbedder


def create_embedder(cfg: EmbedderConfig) -> Embedder:
    if cfg.provider == "hashing":
        _, _, dims = cfg.model.partition("-")
        return HashingEmbedder(int(dims) if dims.isdigit() else 64)
    from rem.embed.fastembed import FastEmbedEmbedder  # heavy import, only when used

    return FastEmbedEmbedder(cfg.model, cfg.cache_dir or DEFAULT_CACHE_DIR)
