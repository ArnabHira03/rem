"""The M3 job handler: extract candidate facts from a job's turns and write them through Memory."""

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from rem.config import ExtractionConfig
from rem.extract import Candidate, extract
from rem.llm.base import LLM
from rem.storage.queue import Job, Turn

if TYPE_CHECKING:
    from rem.daemon.service import RemService

log = logging.getLogger("rem.extraction")


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
            extraction = await asyncio.to_thread(
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
            threshold=self.settings.dedup_threshold,
        )
        log.info(
            "job %s (%s): %d candidate(s), added %d, corroborated %d, skipped %d, dropped %d",
            job.id, model, len(candidates), len(result.added), len(result.corroborated),
            result.skipped, result.dropped,
        )
