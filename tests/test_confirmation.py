"""An agent's confirm_memory only requests confirmation; the user's own words verify (M5 trust fix)."""

import pytest

from rem.errors import LLMError
from rem.extract import CONFIRM_SCHEMA, build_confirm_prompt, build_prompt, confirm_pending
from rem.extract import Candidate
from rem.llm.fake import FakeLLM
from rem.memory import MAX_PENDING_CONFIRMATIONS
from rem.models import Confirmation, HistoryEvent as E, Provenance as P, Scope, Status as S, utcnow
from rem.storage.queue import Turn
from rem.trust import quote_in_user_turns

EVER = ("", "9999")  # a window that holds every request

JOB = "job:7"


def agent_fact(mem, text="Prefers tabs over spaces"):
    return mem.add(text, scope=Scope.GLOBAL, provenance=P.ASSISTANT_GENERATED, status=S.UNVERIFIED,
                   source="mcp", actor="mcp")


def requested(mem, text="Prefers tabs over spaces"):
    return mem.request_confirmation(agent_fact(mem, text).id, actor="mcp")


def conf(rec, quote):
    return Confirmation(memory_id=rec.id, hash=rec.hash, quote=quote)


# ---- the literal quote check -----------------------------------------------------------------


@pytest.mark.parametrize("quote", ["yes", "Yes, that's right", "  YES,   that's   right ", "\"that's right\"",
                                   "“yes”"])
def test_quote_in_user_turns_accepts_the_users_own_words(quote):
    assert quote_in_user_turns(quote, ["Yes, that's right. Now fix the test."])


@pytest.mark.parametrize("quote", [
    "The user confirmed they prefer tabs",  # the agent's / LLM's summary
    "yes I prefer tabs",                    # a paraphrase stitched together
    "ye",                                    # part of a word
    "y",                                     # too short to mean anything
    "",
    "…",
])
def test_quote_in_user_turns_rejects_anything_else(quote):
    assert not quote_in_user_turns(quote, ["Yes, that's right. Now fix the test."])


def test_quote_must_come_from_a_user_turn():
    assert not quote_in_user_turns("yes", [])


# ---- Memory.request_confirmation / pending_confirmations / apply_confirmations -----------------


def test_request_keeps_the_memory_unverified(mem):
    r = requested(mem)
    stored = mem.get(r.id)
    assert stored.status == S.UNVERIFIED and stored.provenance == P.ASSISTANT_GENERATED
    assert stored.confirm_requested_at is not None
    assert [(h.event, h.actor) for h in mem.history(r.id)] == [(E.ADD, "mcp"), (E.CONFIRM_REQUEST, "mcp")]
    assert [m.id for m in mem.pending_confirmations(*EVER)] == [r.id]


def test_request_on_a_verified_memory_is_a_no_op(mem):
    r = mem.add("Uses fish shell", scope=Scope.GLOBAL, provenance=P.USER_CONFIRMED, status=S.VERIFIED,
                source="cli", actor="cli")
    assert mem.request_confirmation(r.id, actor="mcp").confirm_requested_at is None
    assert [h.event for h in mem.history(r.id)] == [E.ADD]
    assert mem.pending_confirmations(*EVER) == []


def test_pending_is_limited_to_the_window_and_capped(mem):
    before = utcnow()
    r = requested(mem)
    after = utcnow()
    assert [m.id for m in mem.pending_confirmations(before, after)] == [r.id]
    assert mem.pending_confirmations(after, "9999") == []  # requested before this exchange began
    assert mem.pending_confirmations("", before) == []  # requested after this exchange's job was enqueued
    for i in range(MAX_PENDING_CONFIRMATIONS + 2):
        requested(mem, f"Fact number {i}")
    pending = mem.pending_confirmations(*EVER)
    assert len(pending) == MAX_PENDING_CONFIRMATIONS and r.id not in [m.id for m in pending]  # newest first


def test_a_requested_memory_is_not_verified_by_a_restating_candidate(mem):
    r = requested(mem)
    out = mem.apply_candidates([Candidate(text=r.text, scope=Scope.GLOBAL, origin="user")], job_key=JOB,
                               project_id=None, session_id="s", source="claude-code", restate_min_similarity=0.75,
                               has_user_turn=True)
    assert out.corroborated == [] and out.added == [] and out.skipped == 1
    assert mem.get(r.id).status == S.UNVERIFIED


def test_users_words_verify_a_requested_memory(mem):
    r = requested(mem)
    assert mem.apply_confirmations([conf(r, "yes, that's right")], user_texts=["Yes, that's right"],
                                   job_key=JOB) == [r.id]
    m = mem.get(r.id)
    assert m.status == S.VERIFIED and m.provenance == P.USER_CONFIRMED
    assert m.corroborated_by == JOB and m.confirm_requested_at is None
    assert [(h.event, h.actor, h.job_id) for h in mem.history(r.id)][-1] == (E.CONFIRM, "extractor", JOB)
    # replaying the job only fills in history; nothing is duplicated
    assert mem.apply_confirmations([conf(r, "yes, that's right")], user_texts=["Yes, that's right"],
                                   job_key=JOB) == [r.id]
    assert [h.event for h in mem.history(r.id)].count(E.CONFIRM) == 1


@pytest.mark.parametrize("quote,user_texts", [
    ("The user confirmed it", ["Yes, that's right"]),  # summary, not the user's words
    ("yes", []),                                       # no user turn at all
    ("yes", ["no, I prefer spaces"]),                  # words that aren't there
])
def test_without_the_users_words_the_memory_stays_unverified(mem, quote, user_texts):
    r = requested(mem)
    assert mem.apply_confirmations([conf(r, quote)], user_texts=user_texts, job_key=JOB) == []
    m = mem.get(r.id)
    assert m.status == S.UNVERIFIED and m.confirm_requested_at is not None
    assert E.CONFIRM not in [h.event for h in mem.history(r.id)]


def test_no_request_no_confirmation(mem):
    r = agent_fact(mem)  # added but never requested
    assert mem.apply_confirmations([conf(r, "yes")], user_texts=["yes"], job_key=JOB) == []
    assert mem.get(r.id).status == S.UNVERIFIED


def test_edit_after_the_request_withdraws_it(mem):
    r = requested(mem)
    shown = conf(r, "yes")  # what the LLM saw: the old text
    mem.update(r.id, "Prefers spaces over tabs", actor="mcp", unverify=True)
    assert mem.get(r.id).confirm_requested_at is None
    assert mem.apply_confirmations([shown], user_texts=["yes"], job_key=JOB) == []
    assert mem.get(r.id).status == S.UNVERIFIED


def test_text_changed_since_extraction_is_not_confirmed(mem):
    r = requested(mem)
    stale = Confirmation(memory_id=r.id, hash="0" * 64, quote="yes")
    assert mem.apply_confirmations([stale], user_texts=["yes"], job_key=JOB) == []


# ---- the separate confirmation call --------------------------------------------------------------


def test_extraction_prompt_never_shows_pending_memories(mem):
    r = requested(mem)
    _, u = build_prompt([Turn("user", "yes")], [], [])
    assert r.text not in u


def test_confirm_prompt_shows_the_question_answered_and_the_facts(mem):
    r = requested(mem)
    ctx = [Turn("user", "old"), Turn("assistant", "Should I remember that you prefer tabs?"), Turn("user", "hm")]
    s, u = build_confirm_prompt([Turn("user", "yes"), Turn("assistant", "Saved.")], ctx, [r])
    assert "only the assistant writes" in s
    assert u == ("## Conversation\nassistant: Should I remember that you prefer tabs?\nuser: yes\nassistant: Saved.\n\n"
                 "## Facts awaiting the user's confirmation\n[0] Prefers tabs over spaces\n")
    assert CONFIRM_SCHEMA["required"] == ["confirmations"]


def test_confirm_pending_resolves_and_drops_bad_items(mem):
    a, b = requested(mem, "Prefers tabs"), requested(mem, "Uses fish shell")
    out = {"confirmations": [
        {"id": 1, "quote": " yes "}, {"id": 1, "quote": "again"}, {"id": 5, "quote": "x"},
        {"id": "0", "quote": "yes"}, {"id": 0, "quote": ""}, "junk", {"id": True, "quote": "yes"},
    ]}
    assert confirm_pending(FakeLLM([out]), [Turn("user", "yes")], [], [a, b]) == [
        Confirmation(memory_id=b.id, hash=b.hash, quote="yes")]


def test_confirm_pending_needs_a_user_turn_and_pending_memories(mem):
    r = requested(mem)
    llm = FakeLLM([])
    assert confirm_pending(llm, [Turn("assistant", "the user said yes")], [], [r]) == []
    assert confirm_pending(llm, [Turn("user", "yes")], [], []) == []
    assert llm.calls == []


def test_confirm_pending_rejects_a_malformed_response(mem):
    with pytest.raises(LLMError):
        confirm_pending(FakeLLM([{"nope": 1}]), [Turn("user", "yes")], [], [requested(mem)])
