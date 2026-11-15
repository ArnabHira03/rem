from urllib.parse import urlparse

from rem.config import LLMConfig
from rem.errors import RemoteLLMNotAllowed
from rem.llm.base import LLM, is_loopback

OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"
ANTHROPIC_HOST = "api.anthropic.com"


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
        host = (urlparse(url).hostname if url else None) or ANTHROPIC_HOST
        raise RemoteLLMNotAllowed(
            f"llm provider '{cfg.provider}' sends conversation data to {host}; "
            "set [llm] allow_remote = true to allow it"
        )


def create_llm(cfg: LLMConfig) -> LLM | None:
    if cfg.provider == "none":
        return None
    check_remote_allowed(cfg)
    if cfg.provider == "anthropic":
        from rem.llm.anthropic import AnthropicLLM  # lazy: SDK only imported when used

        return AnthropicLLM(cfg)
    from rem.llm.openai_compat import OpenAICompatLLM

    return OpenAICompatLLM(cfg)


def describe_llm(cfg: LLMConfig) -> dict:
    if cfg.provider == "none":
        return {"provider": "none"}
    return {"provider": cfg.provider, "model": cfg.model, "remote": is_remote(cfg)}
