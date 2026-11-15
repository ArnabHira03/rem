import asyncio
import os
import threading

import httpx
import pytest

from rem.daemon.app import build_app
from rem.daemon.service import RemService

pytestmark = pytest.mark.anyio


@pytest.fixture
async def api(cfg):
    svc = RemService(cfg)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(svc)), base_url="http://rem") as c:
        yield c
    svc.close()


async def test_memory_crud_round_trip(api):
    r = await api.post("/memories", json={"text": "Prefers uv over pip"})
    assert r.status_code == 201
    mid = r.json()["id"]
    assert (await api.get(f"/memories/{mid}")).json()["text"] == "Prefers uv over pip"
    assert (await api.patch(f"/memories/{mid}", json={"text": "Prefers uv"})).json()["text"] == "Prefers uv"
    assert (await api.post(f"/memories/{mid}/confirm", json={})).json()["status"] == "verified"
    hits = (await api.post("/search", json={"query": "uv"})).json()
    assert hits[0]["id"] == mid and "score" in hits[0]
    assert [m["id"] for m in (await api.get("/memories", params={"scope": "global"})).json()] == [mid]
    assert (await api.delete(f"/memories/{mid}")).status_code == 204
    assert [h["event"] for h in (await api.get(f"/memories/{mid}/history")).json()] == ["ADD", "UPDATE", "DELETE"]


async def test_error_mapping(api):
    r = await api.get("/memories/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    r = await api.post("/memories", json={"text": "  "})
    assert r.status_code == 400 and r.json() == {"error": "invalid", "message": "text must not be empty"}
    assert (await api.post("/search", json={"query": "x", "limit": 0})).status_code == 400
    assert (await api.post("/memories", json={"text": "x", "scope": "galaxy"})).status_code == 400
    assert (await api.post("/memories", json={})).status_code == 400
    assert (await api.post("/memories", content=b"not json")).status_code == 400


async def test_ingest_accepts_and_dedupes(api):
    body = {"source": "claude-code", "session_id": "s1", "turns": [{"role": "user", "content": "hi", "message_id": "m1"}]}
    r1 = await api.post("/ingest", json=body)
    r2 = await api.post("/ingest", json=body)
    assert r1.status_code == 202 and r1.json()["accepted"] == 1 and r2.json()["duplicates"] == 1
    assert (await api.post("/ingest", json={**body, "turns": [{"role": "robot", "content": "x"}]})).status_code == 400
    assert (await api.post("/ingest", json={**body, "session_id": ""})).status_code == 400
    assert (await api.post("/ingest", json={**body, "turns": body["turns"] * 501})).status_code == 400


async def test_ingest_empty_turns_and_enqueue_flag(api):
    body = {"source": "claude-code", "session_id": "s1", "turns": []}
    assert (await api.post("/ingest", json={**body, "enqueue": True})).status_code == 202
    assert (await api.post("/ingest", json={**body, "enqueue": "x"})).status_code == 400


async def test_status_reports_counts(api):
    s = (await api.get("/status")).json()
    assert s["memories"] == 0 and s["jobs"] == {"pending": 0, "processing": 0, "done": 0, "failed": 0}
    assert s["embedder"] == "hash-64_64" and s["pid"] == os.getpid()


async def test_retry_failed_endpoint(cfg):
    svc = RemService(cfg)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(svc)), base_url="http://rem") as api:
            assert (await api.post("/jobs/retry")).json() == {"requeued": 0}
            await api.post("/ingest", json={"source": "cli", "session_id": "s", "turns": [{"role": "user", "content": "hi"}]})
            svc.memory.db.conn.execute("UPDATE jobs SET status = 'failed', attempts = 5")
            r = await api.post("/jobs/retry")
            assert r.status_code == 200 and r.json() == {"requeued": 1}
            assert (await api.get("/status")).json()["jobs"]["pending"] == 1
    finally:
        svc.close()


def test_store_calls_run_on_one_thread(cfg):
    svc = RemService(cfg)
    names = set()

    async def go():
        await asyncio.gather(*[svc.store.run(lambda: names.add(threading.current_thread().name)) for _ in range(20)])

    asyncio.run(go())
    svc.close()
    assert len(names) == 1 and next(iter(names)).startswith("rem-store")
