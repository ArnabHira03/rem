"""Ranking for search results. Reads memory status; never changes it."""

import math
from datetime import datetime

from rem.models import MemoryRecord, Scope, SearchResult, Status

T_BASE = 0.30  # global, system and current-project memories
T_OTHER = 0.55  # memories from other projects must match more strongly
BOOST_CURRENT = 0.15
PENALTY_OTHER = 0.05
PENALTY_UNVERIFIED = 0.05
RECENCY_MAX = 0.02
KEYWORD_BOOST = 0.10  # added to the dense score, times the memory's BM25 score normalized to the best hit


def overfetch(limit: int) -> int:
    return max(limit * 4, 40)


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def fuse(
    dense: list[tuple[MemoryRecord, float]],
    keyword: dict[str, float],
) -> list[tuple[MemoryRecord, float, float]]:
    """Hybrid score per candidate: dense cosine plus KEYWORD_BOOST times its BM25 score / the best BM25 score.

    Returns (record, hybrid, dense). Dense scores are unchanged when no candidate has a keyword hit.
    The threshold in score_and_rank still gates on dense only, so a keyword-only match with an
    unrelated meaning is dropped.
    """
    top = max(keyword.values(), default=0.0)
    out = []
    for record, semantic in dense:
        kw = keyword.get(record.id, 0.0) / top if top > 0 else 0.0
        out.append((record, semantic + KEYWORD_BOOST * kw, semantic))
    return out


def score_and_rank(
    candidates: list[tuple[MemoryRecord, float, float]],
    current_project_id: int | None,
    limit: int,
    now: datetime,
) -> list[SearchResult]:
    """Threshold on dense, rank on hybrid, then the project, status and recency adjustments."""
    results = []
    for record, hybrid, semantic in candidates:
        is_project = record.scope == Scope.PROJECT
        current = is_project and current_project_id is not None and record.project_id == current_project_id
        other = is_project and not current
        if semantic < (T_OTHER if other else T_BASE):
            continue
        score = hybrid
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
