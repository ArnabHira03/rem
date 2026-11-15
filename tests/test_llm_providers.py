import json
import os
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from rem.config import LLMConfig
from rem.errors import LLMError, RemoteLLMNotAllowed
from rem.llm.anthropic import AnthropicLLM
from rem.llm.factory import create_llm, describe_llm
from rem.llm.openai_compat import OpenAICompatLLM

SCHEMA = {"type": "object", "properties": {"facts": {"type": "array"}}, "required": ["facts"]}
ANTH_CFG = LLMConfig(provider="anthropic", model="claude-haiku-5-5", allow_remote=True)


class StubAnthropic:
    def __init__(self, stop_reason="end_turn", text="", raises=None, content=None):
        self.stop_reason, self.text, self.raises, self.last = stop_reason, text, raises, None
        self.content = content
        self.messages = self

    def create(self, **kw):
        self.last = kw
        if self.raises:
            raise self.raises
        content = self.content or [SimpleNamespace(type="text", text=self.text)]
        return SimpleNamespace(stop_reason=self.stop_reason, content=content)


def test_anthropic_request_shape():
    stub = StubAnthropic(text='{"facts": []}')
    llm = AnthropicLLM(ANTH_CFG, client=stub)
    assert llm.complete_json("SYS", "USER", SCHEMA) == {"facts": []}
    kw = stub.last
    assert kw["model"] == "claude-haiku-5-5" and kw["system"] == "SYS"
    assert kw["messages"] == [{"role": "user", "content": "USER"}]
    assert kw["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert "temperature" not in kw and kw["max_tokens"] == 4096
    assert llm.name == "anthropic:claude-haiku-5-5"


def test_anthropic_skips_thinking_blocks():
    content = [SimpleNamespace(type="thinking", thinking="hm"), SimpleNamespace(type="text", text='{"facts": [1]}')]
    llm = AnthropicLLM(ANTH_CFG, client=StubAnthropic(content=content))
    assert llm.complete_json("s", "u", SCHEMA) == {"facts": [1]}


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_anthropic_bad_stop_reason_raises(stop):
    llm = AnthropicLLM(ANTH_CFG, client=StubAnthropic(stop_reason=stop, text=""))
    with pytest.raises(LLMError, match=stop):
        llm.complete_json("s", "u", SCHEMA)


def test_anthropic_sdk_error_becomes_llm_error():
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    llm = AnthropicLLM(ANTH_CFG, client=StubAnthropic(raises=err))
    with pytest.raises(LLMError, match="anthropic request failed"):
        llm.complete_json("s", "u", SCHEMA)


def test_anthropic_invalid_json_raises():
    llm = AnthropicLLM(ANTH_CFG, client=StubAnthropic(text="not json"))
    with pytest.raises(LLMError, match="invalid JSON"):
        llm.complete_json("s", "u", SCHEMA)


def _ollama(handler):
    cfg = LLMConfig(provider="ollama", model="qwen3:4b")
    return OpenAICompatLLM(cfg, transport=httpx.MockTransport(handler))


def _ok(content, finish="stop"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]})


def test_openai_compat_request_and_parse():
    seen = {}

    def handler(req):
        seen.update(url=str(req.url), body=json.loads(req.content), auth=req.headers.get("authorization"))
        return _ok('{"facts": []}')

    llm = _ollama(handler)
    assert llm.complete_json("S", "U", SCHEMA) == {"facts": []}
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["body"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "extraction", "schema": SCHEMA, "strict": True},
    }
    assert seen["body"]["messages"][0] == {"role": "system", "content": "S"}
    assert seen["body"]["messages"][1] == {"role": "user", "content": "U"}
    assert seen["auth"] is None
    assert llm.name == "ollama:qwen3:4b"


def test_openai_compat_sends_bearer_when_key_set(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sekrit")
    seen = {}

    def handler(req):
        seen["auth"] = req.headers.get("authorization")
        return _ok("{}")

    cfg = LLMConfig(provider="openai", model="m", base_url="http://localhost:1/v1", api_key_env="MY_KEY")
    OpenAICompatLLM(cfg, transport=httpx.MockTransport(handler)).complete_json("s", "u", SCHEMA)
    assert seen["auth"] == "Bearer sekrit"


def test_openai_compat_http_error_and_bad_json_raise_llm_error():
    with pytest.raises(LLMError, match="HTTP 500"):
        _ollama(lambda req: httpx.Response(500)).complete_json("s", "u", SCHEMA)
    with pytest.raises(LLMError, match="invalid JSON"):
        _ollama(lambda req: _ok("not json")).complete_json("s", "u", SCHEMA)
    with pytest.raises(LLMError, match="finish_reason=length"):
        _ollama(lambda req: _ok("{}", finish="length")).complete_json("s", "u", SCHEMA)
    with pytest.raises(LLMError, match="malformed"):
        _ollama(lambda req: httpx.Response(200, json={"nope": 1})).complete_json("s", "u", SCHEMA)


def test_create_llm_refuses_remote_without_allow_remote():
    with pytest.raises(RemoteLLMNotAllowed, match="api.anthropic.com"):
        create_llm(LLMConfig(provider="anthropic", model="m"))
    with pytest.raises(RemoteLLMNotAllowed) as e:
        create_llm(LLMConfig(provider="openai", model="m"))
    assert "anthropic" not in str(e.value) and "base_url is unset" in str(e.value)
    with pytest.raises(RemoteLLMNotAllowed, match="example.com"):
        create_llm(LLMConfig(provider="openai", model="m", base_url="https://example.com/v1"))


def test_create_llm_maps_providers(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert isinstance(create_llm(LLMConfig(provider="anthropic", model="m", allow_remote=True)), AnthropicLLM)
    assert isinstance(create_llm(LLMConfig(provider="openai", model="m", allow_remote=True)), OpenAICompatLLM)
    assert isinstance(create_llm(LLMConfig(provider="ollama", model="m")), OpenAICompatLLM)


def test_describe_llm_remote_flag():
    assert describe_llm(LLMConfig(provider="anthropic", model="m"))["remote"] is True
    assert describe_llm(LLMConfig(provider="ollama", model="m"))["remote"] is False


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY unset")
def test_anthropic_live():
    llm = AnthropicLLM(ANTH_CFG)
    out = llm.complete_json(
        "Extract durable facts about the user as a JSON object with a `facts` list of strings.",
        "I always use tabs, never spaces.",
        {
            "type": "object",
            "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
            "required": ["facts"],
            "additionalProperties": False,
        },
    )
    assert isinstance(out, dict) and isinstance(out["facts"], list)
