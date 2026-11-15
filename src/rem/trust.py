from dataclasses import dataclass
from enum import StrEnum

from rem.models import Candidate, MemoryRecord, Provenance, Status


@dataclass(frozen=True)
class TrustInfo:
    provenance: Provenance
    status: Status
    evidence_source: str | None


class Action(StrEnum):
    ADD = "ADD"
    SKIP = "SKIP"
    CORROBORATE = "CORROBORATE"


def assign_trust(c: Candidate) -> TrustInfo:
    if c.origin == "user":
        return TrustInfo(Provenance.USER_ASSERTED, Status.VERIFIED, c.evidence_source)
    if c.origin == "tool" and c.evidence_source:
        return TrustInfo(Provenance.TOOL_OBSERVED, Status.VERIFIED, c.evidence_source)
    return TrustInfo(Provenance.ASSISTANT_GENERATED, Status.UNVERIFIED, c.evidence_source)


def decide(
    trust: TrustInfo,
    target: MemoryRecord | None,
    similarity: float,
    threshold: float,
    job_key: str,
) -> Action:
    if target is None or similarity < threshold:
        return Action.ADD
    if target.status == Status.UNVERIFIED and trust.status == Status.VERIFIED:
        return Action.CORROBORATE
    if target.corroborated_by == job_key:
        return Action.CORROBORATE  # replay of this job's own corroboration
    return Action.SKIP
