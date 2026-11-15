import pytest

from rem.config import LLMConfig, RemConfig
from rem.errors import LLMError, RemoteLLMNotAllowed
from rem.llm.base import is_loopback
from rem.llm.factory import check_remote_allowed, create_llm, describe_llm, is_remote, resolved_base_url
from rem.llm.fake import FakeLLM


def test_llm_defaults_off(cfg):
    assert cfg.llm.provider == "none" and cfg.llm.allow_remote is False
    assert cfg.extraction.dedup_threshold == 0.90 and cfg.extraction.related_limit == 10
    assert create_llm(cfg.llm) is None and describe_llm(cfg.llm) == {"provider": "none"}


@pytest.mark.parametrize(
    "url,ok",
    [
        ("http://localhost:11434/v1", True),
        ("http://127.0.0.1:8080", True),
        ("http://[::1]:8000/v1", True),
        ("https://api.openai.com/v1", False),
    ],
)
def test_is_loopback(url, ok):
    assert is_loopback(url) is ok


def test_anthropic_requires_allow_remote():
    with pytest.raises(RemoteLLMNotAllowed, match="allow_remote"):
        check_remote_allowed(LLMConfig(provider="anthropic", model="claude-haiku-5-5"))
    check_remote_allowed(LLMConfig(provider="anthropic", model="claude-haiku-5-5", allow_remote=True))


def test_ollama_is_local_by_default():
    c = LLMConfig(provider="ollama", model="qwen3:4b")
    assert is_remote(c) is False and resolved_base_url(c) == "http://localhost:11434/v1"
    check_remote_allowed(c)


def test_openai_remote_url_refused():
    with pytest.raises(RemoteLLMNotAllowed):
        check_remote_allowed(LLMConfig(provider="openai", model="m", base_url="https://api.example.com/v1"))


def test_provider_without_model_is_invalid():
    with pytest.raises(ValueError, match="model"):
        RemConfig(data_dir="/tmp/x", llm={"provider": "anthropic", "allow_remote": True})


def test_fake_llm_scripts_and_records():
    f = FakeLLM([{"facts": []}, LLMError("boom")])
    assert f.name == "fake"
    assert f.complete_json("s", "u", {}) == {"facts": []} and f.calls == [("s", "u")]
    with pytest.raises(LLMError, match="boom"):
        f.complete_json("s", "u", {})
    with pytest.raises(LLMError, match="exhausted"):
        f.complete_json("s", "u", {})
