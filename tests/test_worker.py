import asyncio

import pytest

from rem.config import DaemonConfig
from rem.daemon.service import RemService
from rem.daemon.worker import NoopHandler, Worker
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


async def test_worker_completes_jobs_and_passes_turns(svc):
    seen = []

    class H:
        async def handle(self, job, turns, context, service):
            seen.append([t.content for t in turns])

    await svc.ingest("cli", "s", None, [Turn("user", "a", "1")])
    stop = asyncio.Event()
    task = asyncio.create_task(Worker(svc, H(), 0.01).run(stop))
    await wait_until(lambda: seen == [["a"]])
    await wait_until_async(svc.status, lambda s: s["jobs"]["done"] == 1)
    stop.set()
    await task


async def test_noop_handler_drives_jobs_to_done(svc):
    await svc.ingest("cli", "s", None, [Turn("user", "a", "1")])
    stop = asyncio.Event()
    task = asyncio.create_task(Worker(svc, NoopHandler(), 0.01).run(stop))
    await wait_until_async(svc.status, lambda s: s["jobs"]["done"] == 1)
    stop.set()
    await task


async def test_failing_handler_retries_then_fails(svc, clock):
    calls = []

    class Boom:
        async def handle(self, job, *a):
            calls.append(job.attempts)
            raise RuntimeError("llm down")

    await svc.ingest("cli", "s", None, [Turn("user", "a", "1")])
    stop = asyncio.Event()
    task = asyncio.create_task(Worker(svc, Boom(), 0.01).run(stop))
    await wait_until(lambda: calls == [0])
    await wait_until_async(svc.status, lambda s: s["jobs"]["pending"] == 1)
    clock.t += 5
    await wait_until_async(svc.status, lambda s: s["jobs"]["failed"] == 1)
    stop.set()
    await task
    assert calls == [0, 1]
    assert svc.memory.db.conn.execute("SELECT last_error FROM jobs").fetchone()[0] == "llm down"


async def test_stop_releases_unfinished_job(svc):
    gate = asyncio.Event()

    class Slow:
        async def handle(self, *a):
            await gate.wait()

    await svc.ingest("cli", "s", None, [Turn("user", "a", "1")])
    stop = asyncio.Event()
    task = asyncio.create_task(Worker(svc, Slow(), 0.01).run(stop))
    await wait_until_async(svc.status, lambda s: s["jobs"]["processing"] == 1)
    stop.set()
    await asyncio.wait_for(task, 2)
    assert (await svc.status())["jobs"] == {"pending": 1, "processing": 0, "done": 0, "failed": 0}


async def test_stop_lets_a_quick_job_finish(svc):
    started = asyncio.Event()

    class Quick:
        async def handle(self, *a):
            started.set()
            await asyncio.sleep(0.05)

    await svc.ingest("cli", "s", None, [Turn("user", "a", "1")])
    stop = asyncio.Event()
    task = asyncio.create_task(Worker(svc, Quick(), 0.01).run(stop))
    await started.wait()
    stop.set()
    await task
    assert (await svc.status())["jobs"]["done"] == 1


async def test_worker_survives_an_internal_error(svc, monkeypatch):
    real = svc.ingest_store.claim
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk hiccup")
        return real()

    monkeypatch.setattr(svc.ingest_store, "claim", flaky)
    monkeypatch.setattr("rem.daemon.worker.ERROR_BACKOFF_SECONDS", 0.01)
    await svc.ingest("cli", "s", None, [Turn("user", "a", "1")])
    stop = asyncio.Event()
    task = asyncio.create_task(Worker(svc, NoopHandler(), 0.01).run(stop))
    await wait_until_async(svc.status, lambda s: s["jobs"]["done"] == 1)
    stop.set()
    await task
