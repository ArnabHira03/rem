import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError, model_validator

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


MAX_SOCKET_PATH_BYTES = 100  # AF_UNIX paths are limited to ~107 bytes on Linux


class DaemonConfig(BaseModel):
    socket_path: Path | None = None
    debounce_seconds: float = 5
    lease_seconds: float = 120
    max_attempts: int = 5
    start_timeout_seconds: float = 15


class LLMConfig(BaseModel):
    provider: Literal["none", "anthropic", "openai", "ollama"] = "none"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 60
    effort: Literal["low", "medium", "high"] = "low"
    think: bool | None = None  # openai/ollama: false sends reasoning_effort "none"; unset = model default
    allow_remote: bool = False

    @model_validator(mode="after")
    def _model_required(self):
        if self.provider != "none" and not self.model:
            raise ValueError(f"[llm] model is required when provider = '{self.provider}'")
        return self


class ExtractionConfig(BaseModel):
    restate_min_similarity: float = 0.75  # floor for an LLM "restates" link to count
    related_limit: int = 10
    max_candidates: int = 20


class RemConfig(BaseModel):
    data_dir: Path
    vector_store: VectorStoreConfig = VectorStoreConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    daemon: DaemonConfig = DaemonConfig()
    llm: LLMConfig = LLMConfig()
    extraction: ExtractionConfig = ExtractionConfig()

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rem.db"

    @property
    def qdrant_path(self) -> Path:
        return self.vector_store.path or self.data_dir / "qdrant"

    @property
    def socket_path(self) -> Path:
        """Config, then REM_SOCKET, then $XDG_RUNTIME_DIR/rem.sock, then <data_dir>/rem.sock."""
        if self.daemon.socket_path:
            path = self.daemon.socket_path
        elif env := os.environ.get("REM_SOCKET"):
            path = Path(env)
        elif runtime := os.environ.get("XDG_RUNTIME_DIR"):
            path = Path(runtime) / "rem.sock"
        else:
            path = self.data_dir / "rem.sock"
        size = len(os.fsencode(path))
        if size > MAX_SOCKET_PATH_BYTES:
            raise ValueError(f"socket path too long ({size} bytes, max {MAX_SOCKET_PATH_BYTES}): {path}")
        return path

    @property
    def lock_path(self) -> Path:
        return self.data_dir / "rem.lock"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "daemon.log"


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
