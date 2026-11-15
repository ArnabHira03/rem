"""Drives queued jobs through a JobHandler. M2 ships a no-op handler; M3 plugs in extraction."""

import asyncio
import logging
from typing import TYPE_CHECKING, Protocol

from rem.storage.queue import Job, Turn

if TYPE_CHECKING:
    from rem.daemon.service import RemService

log = logging.getLogger("rem.worker")


class JobHandler(Protocol):
    async def handle(self, job: Job, turns: list[Turn], context: list[Turn], service: "RemService") -> None:
        """Process one job. Raising marks the attempt as failed (it is retried with backoff)."""


class NoopHandler:
    async def handle(self, job: Job, turns: list[Turn], context: list[Turn], service: "RemService") -> None:
        return None


class Worker:
    def __init__(self, service: "RemService", handler: JobHandler, poll_seconds: float = 0.5):
        self.service = service
        self.handler = handler
        self.poll_seconds = poll_seconds

    async def run(self, stop: asyncio.Event) -> None:
        q, store = self.service.ingest_store, self.service.store
        while not stop.is_set():
            job = await store.run(q.claim)
            if job is None:
                try:
                    await asyncio.wait_for(stop.wait(), self.poll_seconds)
                except TimeoutError:
                    pass
                continue
            turns = await store.run(q.messages_for, job.id)
            context = await store.run(q.context_for, job.id)
            await self._process(job, turns, context, stop)

    async def _process(self, job: Job, turns: list[Turn], context: list[Turn], stop: asyncio.Event) -> None:
        q, store = self.service.ingest_store, self.service.store
        handling = asyncio.create_task(self.handler.handle(job, turns, context, self.service))
        stopping = asyncio.create_task(stop.wait())
        await asyncio.wait({handling, stopping}, return_when=asyncio.FIRST_COMPLETED)
        if not handling.done():
            # Shutting down: give the handler half a lease to finish, then hand the job back.
            grace = self.service.cfg.daemon.lease_seconds / 2
            await asyncio.wait({handling}, timeout=grace)
            if not handling.done():
                handling.cancel()
                await asyncio.gather(handling, return_exceptions=True)
                await store.run(q.release, job.id)
                log.info("released job %s on shutdown", job.id)
                return
        stopping.cancel()
        exc = handling.exception()
        if exc is None:
            await store.run(q.complete, job.id)
        else:
            log.warning("job %s failed: %s", job.id, exc)
            await store.run(q.fail, job.id, str(exc))
