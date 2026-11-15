import os
from urllib.parse import urlparse

from rem.config import LLMConfig
from rem.errors import LLMError, RemoteLLMNotAllowed
from rem.llm.base import LLM, is_loopback

OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"
ANTHROPIC_HOST = "api.anthropic.com"
DEFAULT_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def resolved_base_url(cfg: LLMConfig) -> str | None:
    if cfg.base_url:
        return cfg.base_url
    if cfg.provider == "ollama":
        return OLLAMA_DEFAULT_URL
    return None


def is_remote(cfg: LLMConfig) -> bool:
    if cfg.provider == "none":
        return False
    url = resolved_base_url(cfg)
    return url is None or not is_loopback(url)


def check_remote_allowed(cfg: LLMConfig) -> None:
    if is_remote(cfg) and not cfg.allow_remote:
        url = resolved_base_url(cfg)
        if url:
            where = urlparse(url).hostname or url
        elif cfg.provider == "anthropic":
            where = ANTHROPIC_HOST
        else:
            where = "a remote endpoint (base_url is unset)"
        raise RemoteLLMNotAllowed(
            f"llm provider '{cfg.provider}' sends conversation data to {where}; "
            "set [llm] allow_remote = true to allow it"
        )


def api_key_env(cfg: LLMConfig) -> str | None:
    """The env var holding the API key: [llm] api_key_env, else the provider's default."""
    return cfg.api_key_env or DEFAULT_KEY_ENV.get(cfg.provider)


def api_key(cfg: LLMConfig) -> str | None:
    var = api_key_env(cfg)
    return (os.environ.get(var) or None) if var else None


def check_api_key(cfg: LLMConfig) -> None:
    """Anthropic, and OpenAI at a non-loopback endpoint, cannot work without a key: fail at startup."""
    if cfg.provider == "anthropic" or (cfg.provider == "openai" and is_remote(cfg)):
        if api_key(cfg) is None:
            raise LLMError(
                f"llm provider '{cfg.provider}' needs an API key: "
                f"set {api_key_env(cfg)} in the daemon's environment"
            )


def create_llm(cfg: LLMConfig) -> LLM | None:
    if cfg.provider == "none":
        return None
    check_remote_allowed(cfg)
    check_api_key(cfg)
    if cfg.provider == "anthropic":
        from rem.llm.anthropic import AnthropicLLM  # lazy: SDK only imported when used

        return AnthropicLLM(cfg)
    if cfg.provider in ("openai", "ollama"):
        from rem.llm.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(cfg)
    raise LLMError(f"unknown llm provider '{cfg.provider}'")


def describe_llm(cfg: LLMConfig) -> dict:
    if cfg.provider == "none":
        return {"provider": "none"}
    return {"provider": cfg.provider, "model": cfg.model, "remote": is_remote(cfg)}
