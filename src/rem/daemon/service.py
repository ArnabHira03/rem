"""The daemon's view of rem: the Memory façade and the ingest queue behind one store thread."""

import asyncio
import os
import time
from collections.abc import Callable
from pathlib import Path

from rem.config import RemConfig
from rem.embed.base import collection_slug
from rem.llm.factory import describe_llm
from rem.memory import Memory
from rem.models import Provenance, Scope, Status
from rem.projects import resolve_project
from rem.storage.queue import IngestStore, Turn
from rem.daemon.store_thread import StoreThread


class RemService:
    def __init__(self, cfg: RemConfig, memory: Memory | None = None, clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.store = StoreThread()
        try:
            self.memory = memory or self.store.call(Memory, cfg)
        except BaseException:
            self.store.shutdown()
            raise
        self.ingest_store = IngestStore(self.memory.db, cfg.daemon, clock)
        self.started_at = time.time()

    # memories
    async def add(self, text: str, *, scope: Scope, cwd: Path | None, provenance: Provenance,
                  status: Status, source: str, actor: str):
        project = await asyncio.to_thread(resolve_project, cwd) if scope == Scope.PROJECT and cwd is not None else None
        # anything an agent writes over MCP is its claim until the user confirms it
        return await self.store.run(self.memory.add, text, scope=scope, cwd=cwd, project=project, provenance=provenance,
                                    status=status, source=source, actor=actor, agent_claim=actor == "mcp")

    async def search(self, query: str, *, cwd: Path | None, limit: int, scope: Scope | None):
        return await self.store.run(self.memory.search, query, cwd=cwd, limit=limit, scope=scope)

    async def context(self, cwd: Path | None, max_chars: int):
        return await self.store.run(self.memory.context, cwd, max_chars)

    async def list(self, *, scope: Scope | None, status: Status | None, project_id: int | None,
                   cwd: Path | None = None):
        if cwd is not None:  # only cwd's project's memories (project_root is filesystem-only)
            return await self.store.run(self.memory.list_for, cwd, scope=scope, status=status)
        return await self.store.run(self.memory.list, scope=scope, status=status, project_id=project_id)

    async def get(self, id: str):
        return await self.store.run(self.memory.get, id)

    async def update(self, id: str, text: str, *, actor: str, unverify: bool = False):
        return await self.store.run(self.memory.update, id, text, actor=actor, unverify=unverify)

    async def delete(self, id: str, *, actor: str):
        return await self.store.run(self.memory.delete, id, actor=actor)

    async def confirm(self, id: str, *, actor: str):
        return await self.store.run(self.memory.confirm, id, actor=actor)

    async def request_confirmation(self, id: str, *, actor: str, cwd: Path | None = None):
        return await self.store.run(self.memory.request_confirmation, id, actor=actor, cwd=cwd)

    async def history(self, id: str):
        return await self.store.run(self.memory.history, id)

    # ingestion
    async def ingest(self, source: str, session_id: str, cwd: Path | None, turns: list[Turn],
                     enqueue: bool = True):
        project = await asyncio.to_thread(resolve_project, cwd) if cwd is not None else None
        return await self.store.run(self.ingest_store.ingest, source, session_id, cwd, turns, enqueue, project)

    async def retry_failed(self) -> int:
        return await self.store.run(self.ingest_store.retry_failed)

    async def status(self) -> dict:
        def snapshot():
            return len(self.memory.vectors.list_points(self.memory.col)), self.ingest_store.counts()

        memories, jobs = await self.store.run(snapshot)
        emb = self.memory.embedder
        return {
            "pid": os.getpid(),
            "started_at": self.started_at,
            "socket": str(self.cfg.socket_path),
            "data_dir": str(self.cfg.data_dir),
            "embedder": collection_slug(emb.model_id, emb.dims),
            "memories": memories,
            "jobs": jobs,
            "llm": describe_llm(self.cfg.llm),
        }

    def close(self) -> None:
        try:
            self.store.call(self.memory.close)
        finally:
            self.store.shutdown()
