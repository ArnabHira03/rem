import logging
from typing import Literal

from pydantic import BaseModel, ValidationError, field_validator

from rem.errors import LLMError
from rem.llm.base import LLM
from rem.models import MemoryRecord, Scope, normalize_text
from rem.redact import redact
from rem.storage.queue import Turn

log = logging.getLogger(__name__)

Origin = Literal["user", "assistant", "tool"]


class Candidate(BaseModel):
    text: str
    scope: Scope
    origin: Origin
    evidence_source: str | None = None

    @field_validator("text")
    @classmethod
    def _text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        return v


class Extraction(BaseModel):
    candidates: list[Candidate]
    model: str


EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "scope": {"type": "string", "enum": ["global", "system", "project"]},
                    "origin": {"type": "string", "enum": ["user", "assistant", "tool"]},
                },
                "required": ["text", "scope", "origin"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You extract durable memories from a conversation between a user and an AI coding assistant.

Rules:
1. Extract durable facts about the user, their preferences, their system or environment, and \
their projects. Skip chit-chat, transient task state, and anything a secret placeholder like \
[REDACTED:...] stands for.
2. Write each fact as one standalone sentence in the third person, for example \
"Prefers uv over pip".
3. Set "scope" to one of: "global" (about the person), "system" (about this machine or OS), \
"project" (about the current repository).
4. Set "origin" to one of: "user" (the user stated it), "assistant" (the assistant asserted or \
inferred it and the user did not confirm it). Never use "tool"; tool output is not provided.
5. Existing memories are listed with integer ids and a status. Do not restate memories marked \
"verified". But if the new messages show the USER stating or confirming something listed as \
"unverified", output it again (origin "user") so it can be verified. Otherwise only output new \
or changed information.
6. Only the "New messages" section is a source of facts; "Earlier conversation" is context only. \
If there is nothing durable, return {"facts": []}.

Respond with JSON: {"facts": [{"text": ..., "scope": ..., "origin": ...}]}"""


def build_prompt(
    turns: list[Turn], context: list[Turn], related: list[MemoryRecord]
) -> tuple[str, str]:
    existing = "\n".join(
        f"[{i}] ({m.scope.value}, {m.status.value}) {redact(m.text)}" for i, m in enumerate(related)
    )
    earlier = "\n".join(f"{t.role}: {redact(t.content)}" for t in context)
    new = "\n".join(f"{t.role}: {redact(t.content)}" for t in turns)
    user = (
        f"## Existing memories\n{existing}\n\n"
        f"## Earlier conversation\n{earlier}\n\n"
        f"## New messages\n{new}\n"
    )
    return redact(SYSTEM_PROMPT), user


def extract(
    llm: LLM,
    turns: list[Turn],
    context: list[Turn],
    related: list[MemoryRecord],
    max_candidates: int,
) -> Extraction:
    system, user = build_prompt(turns, context, related)
    out = llm.complete_json(system, user, EXTRACTION_SCHEMA)
    facts = out.get("facts") if isinstance(out, dict) else None
    if not isinstance(facts, list):
        raise LLMError("extraction response has no 'facts' list")
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for item in facts:
        try:
            cand = Candidate.model_validate(item)
        except ValidationError as e:
            log.warning("dropping invalid extracted fact %r: %s", item, e.errors()[0]["msg"])
            continue
        key = normalize_text(cand.text)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cand)
    if len(candidates) > max_candidates:
        log.warning(
            "dropping %d candidates over the cap of %d", len(candidates) - max_candidates, max_candidates
        )
        candidates = candidates[:max_candidates]
    return Extraction(candidates=candidates, model=llm.name)
