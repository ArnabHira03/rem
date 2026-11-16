import re
from dataclasses import dataclass
from enum import StrEnum

from rem.models import Candidate, MemoryRecord, Provenance, Status
from rem.redact import redact

_QUOTE_MARKS = " \t\n\"'`“”‘’«»"
MIN_QUOTE_WORD_CHARS = 2
# Words that don't tie a sentence to a fact: function words, answers, and the verbs facts are
# phrased with ("Prefers X", "Uses Y"), so overlap means the fact's actual subject.
_STOPWORDS = frozenset("""
a an and are as at be been but by can do does for from got had has have he her here him his how i if im in
into is it its just me my no not of off ok okay on or our out she so than that the their them then there
these they this to too up us was we were what when where which who why will with yes yeah yep sure you your
about after again all also always any around because before being both could each even ever every few just
like make many more most much must never only other over own same should some such very want wants would
use uses used using prefer prefers preferred like likes work works run runs need needs usually tends
without within project projects repo repository codebase user users person machine system
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


def _stem(w: str) -> str:
    """Crude stem so "tab"/"tabs" and "deploy"/"deploys"/"deploying" compare equal."""
    if len(w) > 3 and w.endswith("s"):
        w = w[:-1]
    return w[:5]


def key_words(text: str) -> set[str]:
    """What a fact is about: its non-stopword words (2+ letters, so "uv" and "go" count), stemmed."""
    return {_stem(w) for w in re.findall(r"\w+", text.casefold()) if len(w) >= 2 and w not in _STOPWORDS}


def _sentences(text: str) -> list[str]:
    """Split raw text into sentences: at line breaks, and at . ! ? ; followed by space or the end
    (so "config.toml" stays one word)."""
    return [p for p in re.split(r"\n+|[.!?;]+(?=\s|$)", text) if p.strip()]


def sentence_supports(sentence: str, fact: str, *, strict: bool = True) -> bool:
    """Whether one sentence of the user's names the fact: shares at least one of its key words, and
    with strict most of them (one of two, two of three, three of four or five). Strict is for facts
    an agent wrote (confirmations): it can't word a claim around a word or two the user typed."""
    want = key_words(fact)
    shared = len(key_words(sentence) & want)
    need = 1 if not strict or len(want) <= 2 else len(want) // 2 + 1
    return bool(want) and shared >= need


def user_states(fact: str, user_texts: list[str]) -> bool:
    """Whether some sentence the user actually wrote names fact (one shared key word; for facts the
    extractor proposed, where the risk is the LLM taking assistant text for the user's)."""
    return any(sentence_supports(sent, fact, strict=False) for t in user_texts for form in (t, redact(t))
               for sent in _sentences(form))


def quote_in_user_turns(quote: str, user_texts: list[str], fact: str) -> bool:
    """Whether quote is the user's own words confirming fact, as far as can be checked without an LLM.

    The quote must be whole words of a sentence of one of user_texts (case and whitespace aside),
    so an LLM's paraphrase, an assistant turn, or an agent's summary never passes; and that sentence
    must name the fact (sentence_supports). A bare "yes" never does: what it agrees to is set by the
    assistant's question, which the agent wrote.
    """
    # sentences are split at their end punctuation, so a quote ending "…that." is matched without it
    q = _norm(quote.strip(_QUOTE_MARKS + ".!?;,"))
    if sum(ch.isalnum() for ch in q) < MIN_QUOTE_WORD_CHARS:
        return False
    pattern = re.compile(r"(?<!\w)" + re.escape(q) + r"(?!\w)")
    # turns are stored redacted already; the prompt redacts again, so accept a quote of either form
    return any(pattern.search(_norm(sent)) and sentence_supports(sent, fact)
               for t in user_texts for form in (t, redact(t)) for sent in _sentences(form))
