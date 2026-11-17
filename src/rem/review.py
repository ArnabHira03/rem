"""Pure review of one extraction run: what the model proposed and what rem did with it.

No I/O. The daemon reads the rows in one store-thread call; this module only computes, and mirrors
the rules of Memory.apply_candidates without calling it. Outcomes are derived from the run's
history rows, never guessed."""

import json
import re
from dataclasses import dataclass, field

from rem.models import ENTITY_TYPES, Candidate, Scope, Status, memory_id, normalize_text
from rem.trust import assign_trust, user_states

RULES = ("downgraded", "invalid-type", "noise", "narration", "scope-hint", "relation-not-written", "failed")
NARRATION_VERBS = frozenset({
    "fixed", "added", "modified", "changed", "updated", "removed", "created", "commits", "opens",
    "pushes", "merged", "renamed", "implemented", "refactored", "deleted",
})
_NOISE = re.compile(r"https?://|\b(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b")
_PROJECT_WORDS = re.compile(r"\b(?:this repo|this project|the project)\b", re.IGNORECASE)

_REASONS = {
    "downgraded": "the model proposed a user origin, but no user sentence names this fact",
    "invalid-type": "entity type is not one of the known types",
    "noise": "the text holds a URL or a commit-like id",
    "narration": "an assistant fact that starts with an action verb, like a change log",
    "scope-hint": "scoped system or global, but the text names this project",
    "relation-not-written": "the model proposed relations, but no stored relation cites this memory",
}


@dataclass
class RunRows:
    job: dict
    turns: list[dict]
    context_turns: list[dict]
    extraction: tuple[str, str] | None
    history: list[dict]
    memories_now: dict[str, dict | None] = field(default_factory=dict)
    relations_by_memory: dict[str, list[str]] = field(default_factory=dict)


def parse_extraction(candidates_json: str) -> tuple[list[dict], list[dict]]:
    """(candidates, confirmations) from a cached extraction; a bare list is the pre-M5 form."""
    data = json.loads(candidates_json)
    if isinstance(data, list):
        return data, []
    return list(data.get("candidates") or []), list(data.get("confirmations") or [])


def _flag(rule: str, reason: str | None = None) -> dict:
    return {"rule": rule, "reason": reason or _REASONS[rule]}


def candidate_memory_id(candidate: dict, project_id: int | None, job_id: int) -> str:
    """The id Memory.add derives first for a candidate of job `job_id` (models.memory_id)."""
    scope = Scope(candidate.get("scope", "global"))
    pid = project_id if scope == Scope.PROJECT else None
    return memory_id(candidate.get("text", ""), scope, pid, f"job:{job_id}")


def _verified(c: dict, origin: str) -> bool:
    return assign_trust(Candidate(text=c.get("text", ""), scope=c.get("scope", "global"), origin=origin,
                                  evidence_source=c.get("evidence_source"))).status == Status.VERIFIED


def _facts(rows: RunRows, candidates: list[dict]) -> list[dict]:
    """One reviewed fact per candidate, in the model's order.

    Duplicates: like Memory._collapse_duplicates, each (normalised text, scope) group has one
    representative, the first copy unless a later one is verified and it is not (then the first
    verified copy). Only the representative is credited with the group's write (added or
    corroborated) and is checked by relation-not-written with the group's pooled relations; the
    other copies show the outcome "duplicate".

    Added: the ADD row of the id Memory.add derives first (candidate_memory_id). When that id was
    taken, add() re-derives another one, so a fact with no ADD row under its id falls back to an
    uncredited memory ADD row of this run with the same normalised text. Each ADD or CORROBORATE
    row is credited to at most one fact."""
    job = rows.job
    project_id = job.get("project_id")
    user_texts = [t["content"] for t in rows.turns if t["role"] == "user"]
    mem = [h for h in rows.history if h["target_kind"] == "memory"]
    add_rows = [h for h in mem if h["event"] == "ADD"]
    corroborated = [h["target_id"] for h in mem if h["event"] == "CORROBORATE"]
    exact_ids = {candidate_memory_id(c, project_id, job["id"]) for c in candidates}
    credited_adds: set[int] = set()  # indexes into add_rows
    credited: set[int] = set()  # indexes into corroborated
    name = job.get("project_name")
    name_re = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE) if name else None

    effective_of, groups = [], {}
    for i, c in enumerate(candidates):
        text, proposed = c.get("text", ""), c.get("origin", "assistant")
        effective = proposed
        if proposed == "user" and (not user_texts or not user_states(text, user_texts)):
            effective = "assistant"
        effective_of.append(effective)
        groups.setdefault((normalize_text(text), c.get("scope", "global")), []).append(i)
    representative: dict[tuple[str, str], int] = {}
    for key, members in groups.items():
        best = members[0]
        for i in members[1:]:
            if _verified(candidates[i], effective_of[i]) and not _verified(candidates[best], effective_of[best]):
                best = i
        representative[key] = best

    facts = []
    for i, c in enumerate(candidates):
        text, scope = c.get("text", ""), c.get("scope", "global")
        proposed, effective = c.get("origin", "assistant"), effective_of[i]
        norm = normalize_text(text)
        restates = c.get("restates")
        key = (norm, scope)
        outcome: dict
        if representative[key] != i:
            outcome = {"kind": "none", "reason": "duplicate"}
        elif scope == Scope.PROJECT and project_id is None:
            outcome = {"kind": "none", "reason": "no_project"}
        else:
            outcome = {"kind": "none", "reason": "existing_or_skipped"}
            mid = candidate_memory_id(c, project_id, job["id"])
            hit = next((j for j, h in enumerate(add_rows) if j not in credited_adds and h["target_id"] == mid), None)
            if hit is None:
                hit = next((j for j, h in enumerate(add_rows) if j not in credited_adds
                            and h["target_id"] not in exact_ids and h.get("new_value")
                            and normalize_text(h["new_value"]) == norm), None)
            if hit is not None:
                credited_adds.add(hit)
                added = add_rows[hit]["target_id"]
                outcome = {"kind": "added", "memory_id": added, "memory_now": rows.memories_now.get(added)}
            elif _verified(c, effective):
                # only a verified fact can corroborate (trust.decide)
                for j, target in enumerate(corroborated):
                    now = rows.memories_now.get(target)
                    if j in credited or not (target == restates or (
                            now and normalize_text(now["text"]) == norm)):
                        continue
                    credited.add(j)
                    outcome = {"kind": "corroborated", "memory_id": target, "memory_now": now}
                    break
        entities = c.get("entities") or []
        relations = c.get("relations") or []
        pooled = [r for j in groups[key] for r in (candidates[j].get("relations") or [])]
        flags = []
        if proposed == "user" and effective == "assistant":
            flags.append(_flag("downgraded"))
        for m in entities:
            if m.get("type") not in ENTITY_TYPES:
                flags.append(_flag("invalid-type", f"entity {m.get('name')!r} has type {m.get('type')!r}"))
        if _NOISE.search(text):
            flags.append(_flag("noise"))
        if effective == "assistant" and text.split() and text.split()[0].lower() in NARRATION_VERBS:
            flags.append(_flag("narration"))
        if scope in ("system", "global") and ((name_re and name_re.search(text)) or _PROJECT_WORDS.search(text)):
            flags.append(_flag("scope-hint"))
        now = outcome.get("memory_now")
        # rem delete and a text-changing update unlink the memory's relations, so only a memory that
        # still holds the fact's text can show a relation that was never written
        if (pooled and outcome["kind"] == "added" and now is not None and normalize_text(now["text"]) == norm
                and not rows.relations_by_memory.get(outcome["memory_id"])):
            flags.append(_flag("relation-not-written"))
        facts.append({
            "index": i, "text": text, "scope": scope, "origin_proposed": proposed,
            "origin_effective": effective, "restates": restates, "entities": entities,
            "relations": relations, "outcome": outcome, "flags": flags,
        })
    return facts


def _other_memory_writes(rows: RunRows, facts: list[dict]) -> list[dict]:
    """This run's memory history rows no fact accounts for: every CONFIRM (from the cached
    confirmations) and any ADD or CORROBORATE not credited to a fact, oldest first."""
    credited = {(f["outcome"]["kind"], f["outcome"].get("memory_id")) for f in facts}
    kinds = {"ADD": "added", "CORROBORATE": "corroborated"}
    out = []
    for h in rows.history:
        if h["target_kind"] != "memory" or h["event"] not in ("CONFIRM", *kinds):
            continue
        if h["event"] in kinds and (kinds[h["event"]], h["target_id"]) in credited:
            continue
        out.append({"event": h["event"], "target_id": h["target_id"], "new_value": h.get("new_value"),
                    "memory_now": rows.memories_now.get(h["target_id"])})
    return out


def _parsed(rows: RunRows) -> tuple[list[dict], list[dict]]:
    return parse_extraction(rows.extraction[1]) if rows.extraction else ([], [])


def _run(rows: RunRows, facts: list[dict]) -> dict:
    job = rows.job
    run_flags = []
    if job["status"] == "failed":
        run_flags.append(_flag("failed", job.get("last_error") or "the run failed"))
    names = {f["rule"] for f in run_flags} | {f["rule"] for fact in facts for f in fact["flags"]}
    return {
        "id": job["id"], "status": job["status"], "created_at": job["created_at"],
        "updated_at": job["updated_at"], "project_id": job.get("project_id"),
        "project_name": job.get("project_name"), "source": job.get("source"),
        "model": rows.extraction[0] if rows.extraction else None,
        "n_turns": len(rows.turns), "n_proposed": len(facts),
        "n_added": sum(f["outcome"]["kind"] == "added" for f in facts),
        "n_corroborated": sum(f["outcome"]["kind"] == "corroborated" for f in facts),
        "flags": [r for r in RULES if r in names], "error": job.get("last_error"),
        "run_flags": run_flags,
    }


def run_summary(rows: RunRows) -> dict:
    """The list item for one run."""
    run = _run(rows, _facts(rows, _parsed(rows)[0]))
    del run["run_flags"]
    return run


def review_run(rows: RunRows) -> dict:
    """The body of GET /jobs/{id}."""
    candidates, confirmations = _parsed(rows)
    facts = _facts(rows, candidates)
    return {
        "run": _run(rows, facts), "turns": rows.turns, "context_turns": rows.context_turns,
        "facts": facts, "confirmations": confirmations,
        "other_memory_writes": _other_memory_writes(rows, facts),
        "writes": [{k: h[k] for k in ("target_kind", "event", "target_id", "new_value")} for h in rows.history],
    }
