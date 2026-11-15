"""The M3 job handler: extract candidate facts from a job's turns and write them through Memory."""

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from rem.config import ExtractionConfig
from rem.extract import Candidate, extract
from rem.llm.base import LLM
from rem.storage.queue import Job, Turn

if TYPE_CHECKING:
    from rem.daemon.service import RemService

log = logging.getLogger("rem.extraction")


async def _in_daemon_thread[T](fn: Callable[..., T], *args) -> T:
    """Run fn in a daemon thread and await its result.

    Not asyncio.to_thread: a slow LLM call (timeout x retries can be minutes) would hold
    asyncio.run's default-executor shutdown, and a ThreadPoolExecutor's threads are joined at
    interpreter exit even after shutdown(wait=False). A daemon thread is simply abandoned on
    exit. If the awaiting task is cancelled (shutdown grace expired), the late result is dropped,
    so it is never saved or applied.
    """
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[T] = loop.create_future()

    def deliver(ok: bool, value) -> None:
        if not fut.done():
            fut.set_result(value) if ok else fut.set_exception(value)

    def target() -> None:
        try:
            ok, value = True, fn(*args)
        except BaseException as e:  # anything, so the awaiting future always resolves
            ok, value = False, e
            if not isinstance(e, Exception):  # e.g. SystemExit: fail the job, don't raise it in the loop
                value = RuntimeError(f"LLM call aborted: {type(e).__name__}")
                value.__cause__ = e
        try:
            loop.call_soon_threadsafe(deliver, ok, value)
        except RuntimeError:
            pass  # the loop is closed: the daemon stopped and nobody wants this result

    threading.Thread(target=target, name="rem-llm", daemon=True).start()
    return await fut


class ExtractionHandler:
    def __init__(self, llm: LLM, settings: ExtractionConfig):
        self.llm = llm
        self.settings = settings

    async def handle(self, job: Job, turns: list[Turn], context: list[Turn], service: "RemService") -> None:
        store, memory, db = service.store, service.memory, service.memory.db
        cached = await store.run(db.get_extraction, job.id)
        if cached is None:
            text = "\n".join(t.content for t in turns)
            related = await store.run(memory.related, text, job.project_id, self.settings.related_limit)
            # The LLM call runs off the store thread so the API keeps being served meanwhile.
            extraction = await _in_daemon_thread(
                extract, self.llm, turns, context, related, self.settings.max_candidates
            )
            payload = json.dumps([c.model_dump(mode="json") for c in extraction.candidates])
            await store.run(db.save_extraction, job.id, extraction.model, payload)
            # First write wins: on a race the stored row is authoritative.
            cached = await store.run(db.get_extraction, job.id)
        model, candidates_json = cached
        candidates = [Candidate.model_validate(c) for c in json.loads(candidates_json)]
        result = await store.run(
            memory.apply_candidates,
            candidates,
            job_key=f"job:{job.id}",
            project_id=job.project_id,
            session_id=job.session_id,
            source=job.source,
            restate_min_similarity=self.settings.restate_min_similarity,
            has_user_turn=any(t.role == "user" for t in turns),
        )
        log.info(
            "job %s (%s): %d candidate(s), added %d, corroborated %d, skipped %d, dropped %d",
            job.id, model, len(candidates), len(result.added), len(result.corroborated),
            result.skipped, result.dropped,
        )
