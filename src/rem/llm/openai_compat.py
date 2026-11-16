import json

import httpx

from rem.config import LLMConfig
from rem.errors import LLMError
from rem.llm.base import LLM
from rem.llm.factory import api_key, resolved_base_url

OPENAI_DEFAULT_URL = "https://api.openai.com/v1"
MAX_ERROR_DETAIL = 200  # chars of the provider's error body kept in the message


class OpenAICompatLLM(LLM):
    def __init__(self, cfg: LLMConfig, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        headers = {}
        if key := api_key(cfg):
            headers["Authorization"] = f"Bearer {key}"
        self.client = httpx.Client(
            base_url=resolved_base_url(cfg) or OPENAI_DEFAULT_URL,
            timeout=cfg.timeout_seconds,
            transport=transport,
            headers=headers,
        )

    def close(self) -> None:
        self.client.close()

    @property
    def name(self) -> str:
        return f"{self.cfg.provider}:{self.cfg.model}"

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": schema, "strict": True},
            },
        }
        if self.cfg.think is False:
            body["reasoning_effort"] = "none"  # Ollama's OpenAI endpoint ignores `think`; this turns thinking off
        try:
            resp = self.client.post("chat/completions", json=body)
        except Exception as e:  # httpx errors and anything else on the request path
            raise LLMError(f"{self.cfg.provider} request failed: {e}") from e
        if resp.status_code != 200:
            detail = " ".join(resp.text.split())[:MAX_ERROR_DETAIL]
            raise LLMError(f"{self.cfg.provider} request failed: HTTP {resp.status_code}" + (f": {detail}" if detail else ""))
        try:
            choice = resp.json()["choices"][0]
            finish, content = choice.get("finish_reason"), choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMError(f"{self.cfg.provider} returned a malformed response") from e
        if finish != "stop":
            raise LLMError(f"{self.cfg.provider} finish_reason={finish}")
        try:
            data = json.loads(content)
        except (TypeError, json.JSONDecodeError) as e:
            raise LLMError(f"{self.cfg.provider} returned invalid JSON") from e
        if not isinstance(data, dict):
            raise LLMError(f"{self.cfg.provider} returned JSON that is not an object")
        return data
