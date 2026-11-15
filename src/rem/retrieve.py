"""Ranking for search results. Reads memory status; never changes it."""

from datetime import datetime

from rem.models import MemoryRecord, Scope, SearchResult, Status

T_BASE = 0.30  # global, system and current-project memories
T_OTHER = 0.55  # memories from other projects must match more strongly
BOOST_CURRENT = 0.15
PENALTY_OTHER = 0.05
PENALTY_UNVERIFIED = 0.05
RECENCY_MAX = 0.02


def overfetch(limit: int) -> int:
    return max(limit * 4, 40)


def rank(
    candidates: list[tuple[MemoryRecord, float]],
    current_project_id: int | None,
    limit: int,
    now: datetime,
) -> list[SearchResult]:
    results = []
    for record, semantic in candidates:
        is_project = record.scope == Scope.PROJECT
        current = is_project and current_project_id is not None and record.project_id == current_project_id
        other = is_project and not current
        if semantic < (T_OTHER if other else T_BASE):
            continue
        score = semantic
        if current:
            score += BOOST_CURRENT
        elif other:
            score -= PENALTY_OTHER
        if record.status == Status.UNVERIFIED:
            score -= PENALTY_UNVERIFIED
        age_days = (now - datetime.fromisoformat(record.created_at)).total_seconds() / 86400
        score += RECENCY_MAX * max(0.0, 1 - age_days / 365)
        results.append(SearchResult(record=record, score=score, semantic=semantic))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]
