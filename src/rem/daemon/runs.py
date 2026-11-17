"""Read-only views of extraction runs: gathers a job's rows and hands them to rem.review.

Everything here runs in one store-thread call and only reads (SELECTs and vector gets/scrolls)."""

from rem.memory import Memory
from rem.models import Scope
from rem.review import RunRows, parse_extraction, review_run, run_summary
from rem.storage.queue import IngestStore, Turn

SCAN_CAP = 2000  # most jobs one flagged-only listing looks at
SCAN_BATCH = 100  # jobs a flagged-only listing reviews per batch (one turn query each)
STATUS_GROUPS = {"done": ("done",), "failed": ("failed",), "pending": ("pending", "processing")}

# {(scope, project_id): {memory_id: [relation ids]}}, built once per list_runs / get_run call
_RelationCache = dict[tuple[str, int | None], dict[str, list[str]]]


def _turns(turns) -> list[dict]:
    return [{"role": t.role, "content": t.content, "ts": t.ts} for t in turns]


def gather(memory: Memory, ingest: IngestStore, job_id: int, *, history: list[dict] | None = None,
           with_context: bool = True, relations: _RelationCache | None = None,
           turns: list[Turn] | None = None) -> RunRows | None:
    """A job's rows for rem.review. `turns` and `history`, when given, were fetched in a batch."""
    job = ingest.job_rows(job_id)
    if job is None:
        return None
    relations = {} if relations is None else relations
    if history is None:
        history = [h.model_dump(mode="json") for h in memory.db.history_for_job(f"job:{job_id}")]
    if turns is None:
        turns = ingest.messages_for_jobs([job_id])[job_id]
    extraction = memory.db.get_extraction(job_id)
    candidates = parse_extraction(extraction[1])[0] if extraction else []

    ids = {h["target_id"] for h in history if h["target_kind"] == "memory"}
    ids |= {c["restates"] for c in candidates if c.get("restates")}
    memories_now = {}
    for mid in sorted(ids):
        rec = memory.get(mid)
        memories_now[mid] = None if rec is None else {
            "id": rec.id, "text": rec.text, "status": str(rec.status), "provenance": str(rec.provenance),
            "scope": str(rec.scope), "project_id": rec.project_id,
        }

    # relations of every memory this run added and that still exists (review credits ADD rows to facts)
    by_memory: dict[str, list[str]] = {}
    for h in history:
        now = memories_now.get(h["target_id"])
        if h["target_kind"] != "memory" or h["event"] != "ADD" or now is None:
            continue
        key = (now["scope"], now["project_id"])
        if key not in relations:
            relations[key] = memory.relation_ids_by_memory(Scope(now["scope"]), now["project_id"])
        by_memory[h["target_id"]] = relations[key].get(h["target_id"], [])

    return RunRows(
        job=job, turns=_turns(turns),
        context_turns=_turns(ingest.context_for(job_id)) if with_context else [],
        extraction=extraction, history=history, memories_now=memories_now, relations_by_memory=by_memory,
    )


def get_run(memory: Memory, ingest: IngestStore, job_id: int) -> dict | None:
    rows = gather(memory, ingest, job_id)
    return None if rows is None else review_run(rows)


def list_runs(memory: Memory, ingest: IngestStore, *, project_id: int | None, status: str | None,
              flagged: bool, limit: int, before: int | None) -> dict:
    statuses = STATUS_GROUPS[status] if status else None
    relations: _RelationCache = {}

    def summaries(ids: list[int]) -> list[dict]:
        # one history query and one turn query per batch; a summary needs no context turns
        history = memory.db.history_for_jobs([f"job:{i}" for i in ids])
        turns = ingest.messages_for_jobs(ids)
        return [run_summary(gather(
            memory, ingest, i, history=[h.model_dump(mode="json") for h in history[f"job:{i}"]],
            with_context=False, relations=relations, turns=turns[i])) for i in ids]

    if not flagged:
        ids = ingest.job_ids(project_id=project_id, statuses=statuses, before=before, limit=limit + 1)
        return {"runs": summaries(ids[:limit]), "next_before": ids[limit - 1] if len(ids) > limit else None}

    found: list[dict] = []
    scanned, cursor = 0, before
    while len(found) < limit and scanned < SCAN_CAP:
        ids = ingest.job_ids(project_id=project_id, statuses=statuses, before=cursor,
                             limit=min(max(limit, SCAN_BATCH), SCAN_CAP - scanned))
        if not ids:
            return {"runs": found, "next_before": None}
        for run in summaries(ids):
            scanned += 1
            cursor = run["id"]
            if run["flags"]:
                found.append(run)
                if len(found) == limit:
                    break
    return {"runs": found, "next_before": cursor}
