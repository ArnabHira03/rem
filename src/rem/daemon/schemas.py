"""Request bodies shared by the HTTP route and the startup spool drain."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, StrictBool

from rem.storage.queue import Turn


class TurnBody(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    message_id: str | None = None
    ts: str | None = None


class IngestBody(BaseModel):
    source: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    cwd: str | None = None
    turns: list[TurnBody] = Field(max_length=500)
    enqueue: StrictBool = True

    def turns_for_ingest(self) -> list[Turn]:
        return [Turn(t.role, t.content, t.message_id, t.ts) for t in self.turns]

    def cwd_path(self) -> Path | None:
        return Path(self.cwd) if self.cwd else None
