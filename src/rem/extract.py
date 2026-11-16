import hashlib
import logging

from pydantic import BaseModel, ValidationError

from rem.errors import LLMError
from rem.llm.base import LLM
from rem.models import Candidate, Confirmation, MemoryRecord, Origin, Scope  # noqa: F401  (re-exported)
from rem.redact import redact
from rem.storage.queue import Turn

log = logging.getLogger(__name__)

class Extraction(BaseModel):
    candidates: list[Candidate]
    model: str


EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "scope": {"type": "string", "enum": ["global", "system", "project"]},
                    "origin": {"type": "string", "enum": ["user", "assistant", "tool"]},
                    "restates": {"type": ["integer", "null"]},
                },
                "required": ["text", "scope", "origin", "restates"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You extract durable memories from a conversation between a user and an AI coding assistant.

Rules:
1. Extract durable facts about the user, their preferences, their system or environment, and \
their projects. Skip chit-chat, transient task state, and anything a secret placeholder like \
[REDACTED:...] stands for.
2. Write each fact as one standalone sentence in the third person, for example \
"Prefers uv over pip".
3. Set "scope" by what the fact is ABOUT, not by what it mentions:
   - "global": the person. Their preferences, habits and the tools they choose to use anywhere, \
even when the tool is software (e.g. "Uses fish shell", "Prefers uv over pip"). A preference stays global even \
when it is voiced while working in a repository: only a fact the user ties to this repository \
is "project". If unsure between \
global and system for a preference, use global.
   - "system": facts true of this machine regardless of who asks: OS, hardware, installed \
versions, paths, running services.
   - "project": only the current repository: its stack, layout, commands and conventions.
4. Set "origin" to one of: "user" (the user stated it), "assistant" (the assistant asserted or \
inferred it and the user did not confirm it). Never use "tool"; tool output is not provided.
5. Existing memories are listed with integer ids and a status. Do not restate memories marked \
"verified". But if the new messages show the USER stating or confirming something listed as \
"unverified", output it again (origin "user") so it can be verified. Otherwise only output new \
or changed information.
6. Set "restates" to the integer id of an existing memory ONLY when the fact states the same \
thing as that memory (same meaning). A changed value or the opposite is NOT a restatement: \
output it as a new fact with "restates": null. Every other fact has "restates": null.
7. Only the "New messages" section is a source of facts; "Earlier conversation" is context only. \
If there is nothing durable, return {"facts": []}.

Respond with JSON: {"facts": [{"text": ..., "scope": ..., "origin": ..., "restates": ...}]}"""


MAX_TURN_CHARS = 4000  # per turn sent to the LLM: a pasted log or file must not blow up the prompt


def clip_turn(text: str, limit: int = MAX_TURN_CHARS) -> str:
    """Keep the first and last limit/2 characters of an over-long turn, marking what was left out."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[… {len(text) - 2 * half} chars omitted …]\n{text[-half:]}"


def build_prompt(
    turns: list[Turn], context: list[Turn], related: list[MemoryRecord]
) -> tuple[str, str]:
    existing = "\n".join(
        f"[{i}] ({m.scope.value}, {m.status.value}) {redact(m.text)}" for i, m in enumerate(related)
    )
    # redact before clipping, so a cut can't leave half a secret the patterns no longer match
    earlier = "\n".join(f"{t.role}: {clip_turn(redact(t.content))}" for t in context)
    new = "\n".join(f"{t.role}: {clip_turn(redact(t.content))}" for t in turns)
    user = (
        f"## Existing memories\n{existing}\n\n"
        f"## Earlier conversation\n{earlier}\n\n"
        f"## New messages\n{new}\n"
    )
    return redact(SYSTEM_PROMPT), user


def extract(
    llm: LLM,
    turns: list[Turn],
    context: list[Turn],
    related: list[MemoryRecord],
    max_candidates: int,
) -> Extraction:
    system, user = build_prompt(turns, context, related)
    out = llm.complete_json(system, user, EXTRACTION_SCHEMA)
    facts = out.get("facts") if isinstance(out, dict) else None
    if not isinstance(facts, list):
        raise LLMError("extraction response has no 'facts' list")
    candidates: list[Candidate] = []
    for n, item in enumerate(facts):
        if not isinstance(item, dict):
            log.warning("dropping extracted fact #%d: not an object", n)
            continue
        try:
            cand = Candidate.model_validate({k: v for k, v in item.items() if k != "restates"})
        except ValidationError as e:
            err = e.errors()[0]
            log.warning("dropping invalid extracted fact #%d (%s): %s: %s", n, _fingerprint(item.get("text")),
                        ".".join(str(p) for p in err["loc"]), err["msg"])
            continue
        # The schema requires "restates", but a lenient provider may omit it: missing means null.
        candidates.append(cand.model_copy(update={"restates": _resolve(item.get("restates"), related, n)}))
    if len(candidates) > max_candidates:
        log.warning(
            "dropping %d candidates over the cap of %d", len(candidates) - max_candidates, max_candidates
        )
        candidates = candidates[:max_candidates]
    return Extraction(candidates=candidates, model=llm.name)


def _resolve(index, related: list[MemoryRecord], n: int) -> str | None:
    """Map the LLM's integer memory index to that memory's id; anything invalid becomes None."""
    if index is None:
        return None
    if type(index) is int and 0 <= index < len(related):
        return related[index].id
    shown = index if type(index) is int else type(index).__name__
    log.warning("fact #%d: ignoring invalid restates index %s (%d related memories)", n, shown, len(related))
    return None


def _fingerprint(text) -> str:
    """Identify a fact in logs without its text: length and a short hash."""
    if not isinstance(text, str):
        return f"text: {type(text).__name__}"
    return f"{len(text)} chars, sha256 {hashlib.sha256(text.encode()).hexdigest()[:12]}"


# ---- confirmation of memories an agent asked about (MCP confirm_memory) -------------------------
#
# A separate, narrow call: the extraction prompt never sees the agent's claims, so they can't leak
# into "facts", and this call's output can only ever confirm, never create, a memory. Its quotes are
# then checked literally against the user's turns (trust.quote_in_user_turns).

CONFIRM_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "confirmations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "quote": {"type": "string"}},
                "required": ["id", "quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["confirmations"],
    "additionalProperties": False,
}

CONFIRM_SYSTEM_PROMPT = """\
An AI assistant wants to remember some facts about a user, and says the user confirmed them. You \
check that against what the user actually wrote.

A fact is confirmed ONLY if a "user:" line in the conversation clearly agrees that THIS fact is true: \
the user states it, or answers yes to the assistant asking about this fact. It is NOT confirmed by \
a yes to a different question, a vague or partial agreement, a correction, a refusal, or anything \
only the assistant writes (including the assistant saying the user agreed).

For each confirmed fact, return {"id": <its number>, "quote": "<the words from that user line that \
agree, copied character for character>"}. When no fact is confirmed, or you are unsure, return \
{"confirmations": []}.

Respond with JSON: {"confirmations": [{"id": ..., "quote": ...}]}"""


def build_confirm_prompt(turns: list[Turn], context: list[Turn], pending: list[MemoryRecord]) -> tuple[str, str]:
    # the assistant's last message before this job is what a bare "yes" answers
    before = next((t for t in reversed(context) if t.role == "assistant"), None)
    lines = [before] if before is not None else []
    convo = "\n".join(f"{t.role}: {clip_turn(redact(t.content))}" for t in [*lines, *turns])
    facts = "\n".join(f"[{i}] {redact(m.text)}" for i, m in enumerate(pending))
    return CONFIRM_SYSTEM_PROMPT, f"## Conversation\n{convo}\n\n## Facts awaiting the user's confirmation\n{facts}\n"


def confirm_pending(
    llm: LLM, turns: list[Turn], context: list[Turn], pending: list[MemoryRecord]
) -> list[Confirmation]:
    """Ask the LLM which pending memories the user's own words confirm; each comes back with the
    memory's hash (as shown) and the user's quote, for Memory.apply_confirmations to check."""
    if not pending or not any(t.role == "user" for t in turns):
        return []
    system, user = build_confirm_prompt(turns, context, pending)
    out = llm.complete_json(system, user, CONFIRM_SCHEMA)
    items = out.get("confirmations") if isinstance(out, dict) else None
    if not isinstance(items, list):
        raise LLMError("confirmation response has no 'confirmations' list")
    found: dict[str, Confirmation] = {}
    for n, item in enumerate(items):
        index = item.get("id") if isinstance(item, dict) else None
        quote = item.get("quote") if isinstance(item, dict) else None
        if type(index) is not int or not 0 <= index < len(pending) or not isinstance(quote, str) or not quote.strip():
            log.warning("dropping invalid confirmation #%d", n)
            continue
        m = pending[index]
        found.setdefault(m.id, Confirmation(memory_id=m.id, hash=m.hash, quote=quote.strip()))
    return list(found.values())
