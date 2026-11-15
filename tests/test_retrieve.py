from datetime import UTC, datetime, timedelta

import pytest

from rem.models import MemoryRecord
from rem.retrieve import overfetch, rank

NOW = datetime(2026, 11, 15, tzinfo=UTC)
_n = 0


def rec(scope, project_id=None, status="verified", age_days=0):
    global _n
    _n += 1
    ts = (NOW - timedelta(days=age_days)).isoformat()
    return MemoryRecord(
        id=f"m{_n}", text="t", hash="h", scope=scope, project_id=project_id,
        project_name=f"p{project_id}" if project_id else None, provenance="user_confirmed",
        status=status, evidence_source=None, observed_at=None, source="cli",
        session_id=None, job_id=None, created_at=ts, updated_at=ts,
    )


def test_current_project_outranks_equal_global_and_other_project():
    out = rank([(rec("global"), 0.60), (rec("project", 1), 0.60), (rec("project", 2), 0.60)], 1, 10, NOW)
    assert [(r.record.scope, r.record.project_id) for r in out] == [("project", 1), ("global", None), ("project", 2)]
    assert out[0].score == pytest.approx(0.60 + 0.15 + 0.02)
    assert out[0].semantic == pytest.approx(0.60)


def test_other_project_needs_stricter_threshold():
    out = rank([(rec("project", 2), 0.50), (rec("system"), 0.31)], 1, 10, NOW)
    assert [r.record.scope for r in out] == ["system"]


def test_no_current_project_treats_all_projects_as_other():
    assert rank([(rec("project", 1), 0.50)], None, 10, NOW) == []


def test_unverified_penalized_and_recency_decays():
    out = rank([(rec("global", status="unverified"), 0.60), (rec("global", age_days=365), 0.60)], None, 10, NOW)
    assert out[0].record.status == "verified"
    assert out[0].score == pytest.approx(0.60) and out[1].score == pytest.approx(0.60 - 0.05 + 0.02)


def test_limit_and_overfetch():
    assert overfetch(5) == 40 and overfetch(20) == 80
    assert len(rank([(rec("global"), 0.9)] * 5, None, 2, NOW)) == 2
