import re
from dataclasses import dataclass
from enum import StrEnum

from rem.models import Candidate, MemoryRecord, Provenance, Status
from rem.redact import redact

_QUOTE_MARKS = " \t\n\"'`“”‘’«»"
MIN_QUOTE_WORD_CHARS = 2


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


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def quote_in_user_turns(quote: str, user_texts: list[str]) -> bool:
    """Whether quote is words the user actually wrote: whole words of one of user_texts.

    The check is literal (case and whitespace aside), so an LLM's paraphrase, an assistant turn, or
    an agent's summary of what the user said never passes. A quote needs at least
    MIN_QUOTE_WORD_CHARS letters or digits, so a stray character can't match every message.
    """
    q = _norm(quote.strip(_QUOTE_MARKS))
    if sum(ch.isalnum() for ch in q) < MIN_QUOTE_WORD_CHARS:
        return False
    pattern = re.compile(r"(?<!\w)" + re.escape(q) + r"(?!\w)")
    # turns are stored redacted already; the prompt redacts again, so accept a quote of either form
    return any(pattern.search(_norm(t)) or pattern.search(_norm(redact(t))) for t in user_texts)
