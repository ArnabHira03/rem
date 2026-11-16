import asyncio
import json
import threading

import pytest

from rem.config import DaemonConfig, ExtractionConfig
from rem.daemon.extraction import ExtractionHandler, _in_daemon_thread
from rem.daemon.service import RemService
from rem.daemon.worker import Worker
from rem.errors import LLMError
from rem.llm.fake import FakeLLM
from rem.models import HistoryEvent, Provenance as P, Scope, Status as S
from rem.storage.queue import Turn
from tests.conftest import wait_until, wait_until_async
from tests.test_queue import FakeClock

pytestmark = pytest.mark.anyio


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def svc(cfg, clock):
    s = RemService(
        cfg.model_copy(update={"daemon": DaemonConfig(debounce_seconds=0, lease_seconds=0.2, max_attempts=2)}),
        clock=clock,
    )
    yield s
    s.close()


@pytest.fixture
def git_dir(tmp_path):
    d = tmp_path / "myproj"
    (d / ".git").mkdir(parents=True)
    return d


async def run_worker_until(svc, handler, predicate):
    stop = asyncio.Event()
    task = asyncio.create_task(Worker(svc, handler, 0.01).run(stop))
    try:
        await wait_until_async(svc.status, predicate)
    finally:
        stop.set()
        await task


async def run_worker_until_done(svc, handler):
    await run_worker_until(svc, handler, lambda s: s["jobs"]["done"] == 1)


def job_row(svc):
    return svc.memory.db.conn.execute("SELECT status, attempts, last_error FROM jobs").fetchone()


async def test_job_becomes_memories(svc, git_dir):
    llm = FakeLLM([{"facts": [{"text": "Prefers uv over pip", "scope": "global", "origin": "user", "restates": None},
                              {"text": "Uses Django 5", "scope": "project", "origin": "assistant", "restates": None}]}])
    await svc.ingest("claude-code", "s1", git_dir, [Turn("user", "I prefer uv over pip")])
    await run_worker_until_done(svc, ExtractionHandler(llm, ExtractionConfig()))
    mems = {m.text: m for m in await svc.list(scope=None, status=None, project_id=None)}
    assert mems["Prefers uv over pip"].status == S.VERIFIED
    assert mems["Uses Django 5"].status == S.UNVERIFIED and mems["Uses Django 5"].project_name == git_dir.name
    assert all(m.job_id == "job:1" and m.session_id == "s1" and m.source == "claude-code" for m in mems.values())
    assert "I prefer uv over pip" in llm.calls[0][1]


async def test_prompt_includes_related_memories(svc):
    await svc.add("Prefers uv over pip", scope=Scope.GLOBAL, cwd=None, provenance=P.USER_CONFIRMED,
                  status=S.VERIFIED, source="cli", actor="cli")
    llm = FakeLLM([{"facts": []}])
    await svc.ingest("claude-code", "s1", None, [Turn("user", "I prefer uv over pip")])
    await run_worker_until_done(svc, ExtractionHandler(llm, ExtractionConfig()))
    assert "[0] (global, verified) Prefers uv over pip" in llm.calls[0][1]


async def test_replay_after_crash_reuses_cached_extraction(svc, clock, monkeypatch):
    llm = FakeLLM([{"facts": [{"text": "Prefers uv over pip", "scope": "global", "origin": "user", "restates": None},
                              {"text": "Runs Fedora 44", "scope": "system", "origin": "user", "restates": None}]}])
    real_add = svc.memory.add
    adds = {"n": 0}

    def crashing_add(*a, **kw):
        adds["n"] += 1
        if adds["n"] == 2:
            raise RuntimeError("crash mid-apply")
        return real_add(*a, **kw)

    monkeypatch.setattr(svc.memory, "add", crashing_add)
    await svc.ingest("claude-code", "s1", None, [Turn("user", "I prefer uv; I run Fedora 44")])
    handler = ExtractionHandler(llm, ExtractionConfig())
    stop = asyncio.Event()
    task = asyncio.create_task(Worker(svc, handler, 0.01).run(stop))
    try:
        await wait_until(lambda: job_row(svc)["attempts"] == 1)
        assert job_row(svc)["status"] == "pending" and "crash mid-apply" in job_row(svc)["last_error"]
        assert [m.text for m in await svc.list(scope=None, status=None, project_id=None)] == ["Prefers uv over pip"]
        clock.t += 60  # past the retry backoff
        await wait_until_async(svc.status, lambda s: s["jobs"]["done"] == 1)
    finally:
        stop.set()
        await task
    assert len(llm.calls) == 1
    mems = await svc.list(scope=None, status=None, project_id=None)
    assert sorted(m.text for m in mems) == ["Prefers uv over pip", "Runs Fedora 44"]
    for m in mems:
        assert [h.event for h in await svc.history(m.id)] == [HistoryEvent.ADD]


async def test_llm_failure_retries_and_never_marks_done(svc, clock):
    llm = FakeLLM([LLMError("down")])
    await svc.ingest("claude-code", "s1", None, [Turn("user", "I prefer uv over pip")])
    await run_worker_until(svc, ExtractionHandler(llm, ExtractionConfig()),
                           lambda s: s["jobs"]["pending"] == 1 and job_row(svc)["attempts"] == 1)
    row = job_row(svc)
    assert row["status"] == "pending" and row["attempts"] == 1 and "down" in row["last_error"]
    assert await svc.list(scope=None, status=None, project_id=None) == []
    assert svc.memory.db.get_extraction(1) is None
    assert (await svc.status())["jobs"]["done"] == 0


async def test_project_fact_without_cwd_is_dropped(svc):
    llm = FakeLLM([{"facts": [{"text": "Uses Django 5", "scope": "project", "origin": "user", "restates": None}]}])
    await svc.ingest("claude-code", "s1", None, [Turn("user", "this project uses Django 5")])
    await run_worker_until_done(svc, ExtractionHandler(llm, ExtractionConfig()))
    assert await svc.list(scope=None, status=None, project_id=None) == []


async def test_store_thread_not_blocked_during_llm_call(svc):
    entered, release = threading.Event(), threading.Event()

    class BlockingLLM(FakeLLM):
        def complete_json(self, system, user, schema):
            entered.set()
            release.wait(5)
            return super().complete_json(system, user, schema)

    llm = BlockingLLM([{"facts": [{"text": "Prefers uv over pip", "scope": "global", "origin": "user", "restates": None}]}])
    await svc.ingest("claude-code", "s1", None, [Turn("user", "I prefer uv over pip")])
    stop = asyncio.Event()
    task = asyncio.create_task(Worker(svc, ExtractionHandler(llm, ExtractionConfig()), 0.01).run(stop))
    try:
        await wait_until(entered.is_set)
        assert await asyncio.wait_for(svc.list(scope=None, status=None, project_id=None), 1) == []
        release.set()
        await wait_until_async(svc.status, lambda s: s["jobs"]["done"] == 1)
    finally:
        release.set()
        stop.set()
        await task
    assert [m.text for m in await svc.list(scope=None, status=None, project_id=None)] == ["Prefers uv over pip"]


async def test_status_reports_llm(svc):
    assert (await svc.status())["llm"] == {"provider": "none"}


async def test_stuck_llm_call_does_not_block_worker_stop(svc):
    entered, never = threading.Event(), threading.Event()

    class StuckLLM(FakeLLM):
        def complete_json(self, system, user, schema):
            entered.set()
            never.wait(10)
            return {"facts": [{"text": "Prefers uv over pip", "scope": "global", "origin": "user", "restates": None}]}

    await svc.ingest("claude-code", "s1", None, [Turn("user", "I prefer uv over pip")])
    stop = asyncio.Event()
    task = asyncio.create_task(Worker(svc, ExtractionHandler(StuckLLM([]), ExtractionConfig()), 0.01).run(stop))
    await wait_until(entered.is_set)
    stop.set()
    await asyncio.wait_for(task, 2)  # grace is lease/2 = 0.1s, then the job is released
    assert job_row(svc)["status"] == "pending"
    never.set()  # the abandoned call now returns; its result must be dropped
    await asyncio.sleep(0.1)
    assert svc.memory.db.get_extraction(1) is None
    assert await svc.list(scope=None, status=None, project_id=None) == []


async def test_user_restatement_corroborates_across_jobs(svc):
    fact = "Project uses Postgres 16"
    llm = FakeLLM([
        {"facts": [{"text": fact, "scope": "global", "origin": "assistant", "restates": None}]},
        # The only related memory is the assistant fact, so it is index 0 in job 2's prompt.
        {"facts": [{"text": "The project uses Postgres 16", "scope": "global", "origin": "user", "restates": 0}]},
    ])
    handler = ExtractionHandler(llm, ExtractionConfig())
    await svc.ingest("claude-code", "s1", None, [Turn("assistant", "This project looks like it uses Postgres 16.")])
    await run_worker_until_done(svc, handler)
    await svc.ingest("claude-code", "s1", None, [Turn("user", "Yes, the project uses Postgres 16.")])
    await run_worker_until(svc, handler, lambda s: s["jobs"]["done"] == 2)

    assert f"[0] (global, unverified) {fact}" in llm.calls[1][1]
    [m] = await svc.list(scope=None, status=None, project_id=None)
    assert m.text == fact and m.provenance == P.USER_ASSERTED and m.status == S.VERIFIED
    assert m.job_id == "job:1" and m.corroborated_by == "job:2"
    assert [(h.event, h.job_id) for h in await svc.history(m.id)] == [
        (HistoryEvent.ADD, "job:1"), (HistoryEvent.CORROBORATE, "job:2")]
    _, cached = svc.memory.db.get_extraction(2)
    assert json.loads(cached)["candidates"][0]["restates"] == m.id  # the resolved id is what a replay uses


async def test_job_without_user_turn_cannot_assert_user_facts(svc):
    llm = FakeLLM([{"facts": [{"text": "Prefers uv over pip", "scope": "global", "origin": "user", "restates": None}]}])
    await svc.ingest("claude-code", "s1", None, [Turn("assistant", "You prefer uv over pip.")])
    await run_worker_until_done(svc, ExtractionHandler(llm, ExtractionConfig()))
    [m] = await svc.list(scope=None, status=None, project_id=None)
    assert m.provenance == P.ASSISTANT_GENERATED and m.status == S.UNVERIFIED


async def test_in_daemon_thread_resolves_on_base_exception():
    def boom():
        raise SystemExit(4)

    with pytest.raises(RuntimeError, match="aborted: SystemExit") as e:
        await asyncio.wait_for(_in_daemon_thread(boom), 2)
    assert isinstance(e.value.__cause__, SystemExit)


# ---- an agent's confirmation request is settled by the user's own words (M5) ---------------------


async def _agent_fact(svc, text="Prefers tabs over spaces"):
    return await svc.add(text, scope=Scope.GLOBAL, cwd=None, provenance=P.ASSISTANT_GENERATED,
                         status=S.UNVERIFIED, source="mcp", actor="mcp")


async def _exchange(svc, user, reply, request_ids=(), session="s1"):
    """One Claude Code exchange: the prompt is held, the agent calls confirm_memory, Stop enqueues the job."""
    await svc.ingest("claude-code", session, None, [Turn("user", user)], enqueue=False)
    for mid in request_ids:
        await svc.request_confirmation(mid, actor="mcp")
    await svc.ingest("claude-code", session, None, [Turn("assistant", reply)])


NO_FACTS = {"facts": []}


async def test_users_reply_confirms_a_requested_memory(svc):
    m = await _agent_fact(svc)
    llm = FakeLLM([NO_FACTS, {"confirmations": [{"id": 0, "quote": "yes, remember that"}]}])
    await _exchange(svc, "Yes, remember that.", "Saved.", [m.id])
    await run_worker_until_done(svc, ExtractionHandler(llm, ExtractionConfig()))
    assert m.text not in llm.calls[0][1]  # the extraction prompt never sees the agent's claim
    assert "[0] Prefers tabs over spaces" in llm.calls[1][1] and "user: Yes, remember that." in llm.calls[1][1]
    got = await svc.get(m.id)
    assert got.status == S.VERIFIED and got.provenance == P.USER_CONFIRMED and got.corroborated_by == "job:1"
    assert [(h.event, h.actor) for h in await svc.history(m.id)] == [
        (HistoryEvent.ADD, "mcp"), (HistoryEvent.CONFIRM_REQUEST, "mcp"), (HistoryEvent.CONFIRM, "extractor")]
    _, cached = svc.memory.db.get_extraction(1)
    assert json.loads(cached)["confirmations"] == [{"memory_id": m.id, "hash": m.hash, "quote": "yes, remember that"}]


async def test_llm_claim_without_the_users_words_leaves_it_unverified(svc):
    m = await _agent_fact(svc)
    # the LLM "confirms" with the assistant's words: the literal check rejects it
    llm = FakeLLM([NO_FACTS, {"confirmations": [{"id": 0, "quote": "The user prefers tabs over spaces"}]}])
    await _exchange(svc, "ok, now run the tests", "Done. The user prefers tabs over spaces.", [m.id])
    await run_worker_until_done(svc, ExtractionHandler(llm, ExtractionConfig()))
    got = await svc.get(m.id)
    assert got.status == S.UNVERIFIED and got.confirm_requested_at is not None
    assert HistoryEvent.CONFIRM not in [h.event for h in await svc.history(m.id)]


async def test_extractor_restating_a_requested_memory_does_not_verify_it(svc):
    m = await _agent_fact(svc)
    llm = FakeLLM([{"facts": [{"text": m.text, "scope": "global", "origin": "user", "restates": None}]},
                   {"confirmations": []}])
    await _exchange(svc, "sure", "Saved.", [m.id])
    await run_worker_until_done(svc, ExtractionHandler(llm, ExtractionConfig()))
    assert (await svc.get(m.id)).status == S.UNVERIFIED
    assert [x.id for x in await svc.list(scope=None, status=None, project_id=None)] == [m.id]


async def test_request_from_an_earlier_exchange_is_not_offered(svc):
    m = await _agent_fact(svc)
    handler = ExtractionHandler(FakeLLM([NO_FACTS, {"confirmations": []}, NO_FACTS]), ExtractionConfig())
    await _exchange(svc, "should we remember my indentation?", "Requested.", [m.id])
    await run_worker_until_done(svc, handler)
    assert len(handler.llm.calls) == 2  # job 1 offered the request (the LLM didn't confirm it)
    await _exchange(svc, "yep, run the tests", "Running tests.")  # a yes to something else, later
    await run_worker_until(svc, handler, lambda st: st["jobs"]["done"] == 2)
    assert len(handler.llm.calls) == 2 + 1  # job 2: extraction only, no confirmation call
    assert (await svc.get(m.id)).status == S.UNVERIFIED


async def test_job_without_a_user_turn_is_not_asked_to_confirm(svc):
    m = await _agent_fact(svc)
    llm = FakeLLM([NO_FACTS])
    await svc.request_confirmation(m.id, actor="mcp")
    await svc.ingest("claude-code", "s1", None, [Turn("assistant", "The user said yes.")])
    await run_worker_until_done(svc, ExtractionHandler(llm, ExtractionConfig()))
    assert len(llm.calls) == 1
    assert (await svc.get(m.id)).status == S.UNVERIFIED


async def test_cached_extraction_from_before_m5_still_replays(svc):
    await svc.ingest("claude-code", "s1", None, [Turn("user", "I prefer uv over pip")])
    await svc.store.run(svc.memory.db.save_extraction, 1, "fake", json.dumps(
        [{"text": "Prefers uv over pip", "scope": "global", "origin": "user", "restates": None}]))
    llm = FakeLLM([])  # the cache is used: no LLM call
    await run_worker_until_done(svc, ExtractionHandler(llm, ExtractionConfig()))
    [m] = await svc.list(scope=None, status=None, project_id=None)
    assert m.text == "Prefers uv over pip" and m.status == S.VERIFIED and llm.calls == []
