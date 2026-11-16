"""The M3 job handler: extract candidate facts from a job's turns and write them through Memory."""

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from rem.config import ExtractionConfig
from rem.extract import Candidate, Confirmation, confirm_pending, extract
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
        user_texts = [t.content for t in turns if t.role == "user"]
        cached = await store.run(db.get_extraction, job.id)
        if cached is None:
            text = "\n".join(t.content for t in turns)
            related = await store.run(memory.related, text, job.project_id, self.settings.related_limit)
            # The LLM call runs off the store thread so the API keeps being served meanwhile.
            extraction = await _in_daemon_thread(
                extract, self.llm, turns, context, related, self.settings.max_candidates
            )
            confirmations: list[Confirmation] = []
            if user_texts:  # only the user's own turn can confirm a memory an agent asked about
                since, until = await store.run(service.ingest_store.job_window, job.id)
                pending = await store.run(memory.pending_confirmations, since, until, job.project_id)
                if pending:  # a second, separate call: its output can confirm but never add a memory
                    confirmations = await _in_daemon_thread(confirm_pending, self.llm, turns, context, pending)
            payload = json.dumps({
                "candidates": [c.model_dump(mode="json") for c in extraction.candidates],
                "confirmations": [c.model_dump(mode="json") for c in confirmations],
            })
            await store.run(db.save_extraction, job.id, extraction.model, payload)
            # First write wins: on a race the stored row is authoritative.
            cached = await store.run(db.get_extraction, job.id)
        model, cached_json = cached
        candidates, confirmations = _load_cached(cached_json)
        confirmed = await store.run(
            memory.apply_confirmations, confirmations, user_texts=user_texts, job_key=f"job:{job.id}"
        )
        result = await store.run(
            memory.apply_candidates,
            candidates,
            job_key=f"job:{job.id}",
            project_id=job.project_id,
            session_id=job.session_id,
            source=job.source,
            restate_min_similarity=self.settings.restate_min_similarity,
            has_user_turn=bool(user_texts),
        )
        log.info(
            "job %s (%s): %d candidate(s), added %d, corroborated %d, skipped %d, dropped %d; confirmed %d of %d",
            job.id, model, len(candidates), len(result.added), len(result.corroborated),
            result.skipped, result.dropped, len(confirmed), len(confirmations),
        )


def _load_cached(cached_json: str) -> tuple[list[Candidate], list[Confirmation]]:
    """Candidates and confirmations from a cached extraction (before M5 it was a bare candidate list)."""
    data = json.loads(cached_json)
    if isinstance(data, list):
        data = {"candidates": data}
    return ([Candidate.model_validate(c) for c in data.get("candidates", [])],
            [Confirmation.model_validate(c) for c in data.get("confirmations", [])])
