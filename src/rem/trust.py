import re
from dataclasses import dataclass
from enum import StrEnum

from rem.models import Candidate, MemoryRecord, Provenance, Status
from rem.redact import redact

_QUOTE_MARKS = " \t\n\"'`“”‘’«»"
MIN_QUOTE_WORD_CHARS = 2
# words too common to tie a quote to a fact
_STOPWORDS = frozenset("""
about after also always been before being could does doing from have just like make more most much never
only other over same should some such than that their them then there these they this very want what
when where which while will with would your yours
""".split())


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


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _key_words(text: str) -> set[str]:
    """Words that tie a sentence to a fact, cut to 5 letters so "deploy" matches "deploys"."""
    return {w[:5] for w in _words(text) if len(w) >= 4 and w not in _STOPWORDS}


def quote_in_user_turns(quote: str, user_texts: list[str], fact: str) -> bool:
    """Whether quote is the user's own words confirming fact, as far as can be checked without an LLM.

    The quote must be whole words of one of user_texts (case and whitespace aside), so an LLM's
    paraphrase, an assistant turn, or an agent's summary of what the user said never passes. And
    the user's sentence containing it must itself name the fact (share a key word with it): "yep,
    tabs" confirms "Prefers tabs over spaces", a bare "yes" does not, because what a bare yes
    agrees to is defined by the assistant's question, which the agent wrote.
    """
    q = _norm(quote.strip(_QUOTE_MARKS))
    if sum(ch.isalnum() for ch in q) < MIN_QUOTE_WORD_CHARS:
        return False
    pattern = re.compile(r"(?<!\w)" + re.escape(q) + r"(?!\w)")
    fact_words = _key_words(fact)
    # turns are stored redacted already; the prompt redacts again, so accept a quote of either form
    for text in user_texts:
        for form in (_norm(text), _norm(redact(text))):
            for m in pattern.finditer(form):
                if _key_words(_sentence_around(form, m.start(), m.end())) & fact_words:
                    return True
    return False


def _sentence_around(text: str, start: int, end: int) -> str:
    """The sentence(s) of text that the span [start, end) touches."""
    stops = ".!?;\n"
    left = max(text.rfind(c, 0, start) for c in stops) + 1
    rights = [i for c in stops if (i := text.find(c, end)) != -1]
    return text[left:min(rights) if rights else len(text)]
