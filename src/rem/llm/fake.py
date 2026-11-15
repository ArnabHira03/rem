from rem.errors import LLMError
from rem.llm.base import LLM


class FakeLLM(LLM):
    """Scripted LLM for tests: pops one response per call, raising it if it is an exception."""

    def __init__(self, responses: list[dict | Exception]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "fake"

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append((system, user))
        if not self.responses:
            raise LLMError("FakeLLM exhausted")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
