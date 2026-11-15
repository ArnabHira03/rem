import hashlib
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, field_validator

REM_NAMESPACE = uuid.UUID("6f1c8e5a-3b7d-4c2e-9a10-5d4e3f2a1b00")


class Scope(StrEnum):
    GLOBAL = "global"
    SYSTEM = "system"
    PROJECT = "project"


class Provenance(StrEnum):
    USER_ASSERTED = "user_asserted"
    ASSISTANT_GENERATED = "assistant_generated"
    TOOL_OBSERVED = "tool_observed"
    USER_CONFIRMED = "user_confirmed"


class Status(StrEnum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


class HistoryEvent(StrEnum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CONFIRM = "CONFIRM"
    CORROBORATE = "CORROBORATE"
    MERGE = "MERGE"
    REEMBED = "REEMBED"


class TargetKind(StrEnum):
    MEMORY = "memory"
    ENTITY = "entity"
    RELATION = "relation"


def normalize_text(text: str) -> str:
    norm = " ".join(text.split()).lower()
    if not norm:
        raise ValueError("text must not be empty")
    return norm


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def memory_id(text: str, scope: Scope, project_id: int | None, job_id: str | None = None) -> str:
    key = f"{job_id or 'direct'}:{scope}:{project_id if project_id is not None else ''}:{text_hash(text)}"
    return str(uuid.uuid5(REM_NAMESPACE, key))


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


class MemoryRecord(BaseModel):
    """A memory as stored in the vector store payload (the canonical record)."""

    id: str
    text: str
    hash: str
    scope: Scope
    project_id: int | None
    project_name: str | None
    provenance: Provenance
    status: Status
    evidence_source: str | None
    observed_at: str | None
    source: str
    session_id: str | None
    job_id: str | None
    entity_ids: list[str] = []
    corroborated_by: str | None = None
    created_at: str
    updated_at: str

    def to_payload(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: dict) -> "MemoryRecord":
        return cls.model_validate(payload)


class SearchResult(BaseModel):
    record: MemoryRecord
    score: float
    semantic: float


class HistoryRow(BaseModel):
    id: int
    target_kind: TargetKind
    target_id: str
    event: HistoryEvent
    old_value: str | None
    new_value: str | None
    actor: str
    job_id: str | None
    created_at: str


Origin = Literal["user", "assistant", "tool"]


class Candidate(BaseModel):
    text: str
    scope: Scope
    origin: Origin
    evidence_source: str | None = None
    # Id of an existing memory the LLM says this fact restates (resolved from its integer index at
    # extraction time, so a cached extraction replays against the same memory).
    restates: str | None = None

    @field_validator("text")
    @classmethod
    def _text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        return v
