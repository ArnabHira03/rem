import pytest

from rem.extract import Candidate
from rem.models import MemoryRecord, Provenance, Scope, Status, utcnow
from rem.trust import Action, TrustInfo, assign_trust, decide


def _rec(status: Status, corroborated_by: str | None = None) -> MemoryRecord:
    return MemoryRecord(
        id="x", text="t", hash="h", scope=Scope.GLOBAL, project_id=None, project_name=None,
        provenance=Provenance.USER_ASSERTED if status == Status.VERIFIED
        else Provenance.ASSISTANT_GENERATED,
        status=status, evidence_source=None, observed_at=None, source="t",
        session_id=None, job_id=None, corroborated_by=corroborated_by,
        created_at=utcnow(), updated_at=utcnow(),
    )


unverified_rec = _rec(Status.UNVERIFIED)
verified_rec = _rec(Status.VERIFIED)


def verified_rec_corroborated_by(job: str) -> MemoryRecord:
    return _rec(Status.VERIFIED, corroborated_by=job)


USER = TrustInfo(Provenance.USER_ASSERTED, Status.VERIFIED, None)
ASSISTANT = TrustInfo(Provenance.ASSISTANT_GENERATED, Status.UNVERIFIED, None)


@pytest.mark.parametrize("origin,ev,prov,status", [
    ("user", None, Provenance.USER_ASSERTED, Status.VERIFIED),
    ("assistant", None, Provenance.ASSISTANT_GENERATED, Status.UNVERIFIED),
    ("tool", "pytest output", Provenance.TOOL_OBSERVED, Status.VERIFIED),
    ("tool", None, Provenance.ASSISTANT_GENERATED, Status.UNVERIFIED),
])
def test_assign_trust(origin, ev, prov, status):
    t = assign_trust(Candidate(text="x", scope=Scope.GLOBAL, origin=origin, evidence_source=ev))
    assert (t.provenance, t.status, t.evidence_source) == (prov, status, ev)


def test_decide_add_when_nothing_close():
    assert decide(USER, None, 0.0, 0.9, "job:1") == Action.ADD


def test_decide_add_below_threshold():
    assert decide(USER, unverified_rec, 0.89, 0.9, "job:1") == Action.ADD


def test_decide_corroborates_unverified():
    assert decide(USER, unverified_rec, 0.95, 0.9, "job:1") == Action.CORROBORATE


def test_decide_skips_verified_duplicate():
    assert decide(USER, verified_rec, 0.99, 0.9, "job:1") == Action.SKIP


def test_unverified_never_corroborates():
    assert decide(ASSISTANT, unverified_rec, 0.99, 0.9, "job:1") == Action.SKIP


def test_replay_of_own_corroboration():
    rec = verified_rec_corroborated_by("job:1")
    assert decide(USER, rec, 0.99, 0.9, "job:1") == Action.CORROBORATE


def test_other_jobs_corroboration_skips():
    rec = verified_rec_corroborated_by("job:7")
    assert decide(USER, rec, 0.99, 0.9, "job:1") == Action.SKIP
