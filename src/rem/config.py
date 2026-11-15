import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

DEFAULT_DATA_DIR = Path("~/.local/share/rem").expanduser()
DEFAULT_CONFIG_PATH = Path("~/.config/rem/config.toml").expanduser()
DEFAULT_CACHE_DIR = Path("~/.cache/rem/fastembed").expanduser()


class VectorStoreConfig(BaseModel):
    provider: Literal["qdrant", "memory"] = "qdrant"
    path: Path | None = None
    url: str | None = None


class EmbedderConfig(BaseModel):
    provider: Literal["fastembed", "hashing"] = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"
    cache_dir: Path | None = None


class RemConfig(BaseModel):
    data_dir: Path
    vector_store: VectorStoreConfig = VectorStoreConfig()
    embedder: EmbedderConfig = EmbedderConfig()

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rem.db"

    @property
    def qdrant_path(self) -> Path:
        return self.vector_store.path or self.data_dir / "qdrant"


def load_config(path: Path | None = None) -> RemConfig:
    """Load config from TOML (if present), with REM_HOME overriding the data dir."""
    if path is None:
        env = os.environ.get("REM_CONFIG")
        path = Path(env) if env else DEFAULT_CONFIG_PATH
    raw: dict = {}
    if path.is_file():
        raw = tomllib.loads(path.read_text())

    home = os.environ.get("REM_HOME")
    raw["data_dir"] = Path(home) if home else Path(raw.get("data_dir", DEFAULT_DATA_DIR)).expanduser()
    raw.setdefault("embedder", {}).setdefault("cache_dir", DEFAULT_CACHE_DIR)
    try:
        return RemConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"invalid config {path}: {e}") from e
