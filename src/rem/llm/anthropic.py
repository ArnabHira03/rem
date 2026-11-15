import json
import os

import anthropic

from rem.config import LLMConfig
from rem.errors import LLMError
from rem.llm.base import LLM

MAX_TOKENS = 4096


class AnthropicLLM(LLM):
    def __init__(self, cfg: LLMConfig, client=None):
        self.cfg = cfg
        if client is None:
            # Unset env var -> None: the SDK resolves credentials itself.
            client = anthropic.Anthropic(
                api_key=os.environ.get(cfg.api_key_env or "ANTHROPIC_API_KEY"),
                base_url=cfg.base_url,
                timeout=cfg.timeout_seconds,
                max_retries=2,
            )
        self.client = client

    @property
    def name(self) -> str:
        return f"anthropic:{self.cfg.model}"

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        try:
            resp = self.client.messages.create(
                model=self.cfg.model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.cfg.effort, "format": {"type": "json_schema", "schema": schema}},
            )
        except anthropic.APIError as e:
            raise LLMError(f"anthropic request failed: {e}") from e
        if resp.stop_reason != "end_turn":
            raise LLMError(f"anthropic stop_reason={resp.stop_reason}")
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            raise LLMError("anthropic response has no text block")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"anthropic returned invalid JSON: {e}") from e
        if not isinstance(data, dict):
            raise LLMError("anthropic returned JSON that is not an object")
        return data
