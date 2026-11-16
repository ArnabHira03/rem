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


async def test_context_route(api):
    await api.post("/memories", json={"text": "Prefers uv over pip"})
    r = await api.post("/context", json={"cwd": None})
    assert r.status_code == 200
    assert r.json() == {
        "text": "Facts rem remembers from earlier sessions (verified):\nAbout the user:\n- Prefers uv over pip",
        "count": 1,
    }
    assert (await api.post("/context", json={"max_chars": 0})).status_code == 400


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


async def test_store_thread_never_runs_git(cfg, tmp_path, monkeypatch):
    import subprocess as sp

    import rem.projects as projects

    repo = tmp_path / "repo"
    repo.mkdir()
    sp.run(["git", "init", "-q", str(repo)], check=True)
    sp.run(["git", "-C", str(repo), "remote", "add", "origin", "https://example.com/r.git"], check=True)
    real_run, names = sp.run, []

    def recorder(*a, **k):
        names.append(threading.current_thread().name)
        return real_run(*a, **k)

    monkeypatch.setattr(projects.subprocess, "run", recorder)
    svc = RemService(cfg)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(svc)), base_url="http://rem") as c:
            r = await c.post("/memories", json={"text": "Uses ruff here", "scope": "project", "cwd": str(repo)})
            assert r.status_code == 201
            assert (await c.post("/search", json={"query": "ruff", "cwd": str(repo)})).status_code == 200
            assert (await c.post("/context", json={"cwd": str(repo)})).status_code == 200
            body = {"source": "claude-code", "session_id": "s1", "cwd": str(repo),
                    "turns": [{"role": "user", "content": "hi", "message_id": "m1"}]}
            assert (await c.post("/ingest", json=body)).status_code == 202
    finally:
        svc.close()
    assert names and not any(n.startswith("rem-store") for n in names)


async def test_patch_passes_unverify(api):
    mid = (await api.post("/memories", json={"text": "Prefers uv over pip"})).json()["id"]
    r = (await api.patch(f"/memories/{mid}", json={"text": "Prefers uv"})).json()
    assert r["status"] == "verified" and r["provenance"] == "user_confirmed"
    r = (await api.patch(f"/memories/{mid}", json={"text": "Prefers pip", "unverify": True, "actor": "mcp"})).json()
    assert r["status"] == "unverified" and r["provenance"] == "assistant_generated"
    assert (await api.patch(f"/memories/{mid}", json={"text": "x", "unverify": "maybe"})).status_code == 400


def test_client_update_sends_unverify():
    from rem.client import RemClient

    sent = []

    def handler(request):
        import json as _json

        sent.append(_json.loads(request.content))
        return httpx.Response(200, json={"id": "m"})

    c = RemClient.__new__(RemClient)
    c.socket_path = "x"
    c._http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://rem")
    c.update("m", "t")
    c.update("m", "t", actor="mcp", unverify=True)
    assert sent == [{"text": "t", "actor": "cli", "unverify": False}, {"text": "t", "actor": "mcp", "unverify": True}]


async def test_request_confirmation_route_keeps_memory_unverified(api):
    mid = (await api.post("/memories", json={"text": "Uses tabs", "provenance": "assistant_generated",
                                             "status": "unverified", "actor": "mcp"})).json()["id"]
    r = await api.post(f"/memories/{mid}/request-confirmation", json={"actor": "mcp"})
    assert r.status_code == 200
    assert r.json()["status"] == "unverified" and r.json()["confirm_requested_at"]
    missing = await api.post("/memories/00000000-0000-0000-0000-000000000000/request-confirmation", json={})
    assert missing.status_code == 404


async def test_list_by_cwd_keeps_only_that_projects_memories(api, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        (d / ".git").mkdir(parents=True)
        await api.post("/memories", json={"text": f"Convention of {d.name}", "scope": "project", "cwd": str(d)})
    await api.post("/memories", json={"text": "Prefers uv"})

    def texts(r):
        return sorted(m["text"] for m in r.json())

    assert texts(await api.get("/memories", params={"cwd": str(a)})) == ["Convention of a", "Prefers uv"]
    assert texts(await api.get("/memories", params={"cwd": str(b), "scope": "project"})) == ["Convention of b"]
    assert texts(await api.get("/memories", params={"cwd": str(tmp_path), "scope": "project"})) == []
    assert len((await api.get("/memories")).json()) == 3
    r = await api.get("/memories", params={"cwd": str(a), "project_id": "1"})
    assert r.status_code == 400


async def test_reembed_endpoints(api):
    r = await api.post("/reembed", json={"model": "hash-32"})
    assert r.status_code == 200 and r.json()["count"] == 0 and r.json()["collection"].endswith("hash-32_32")
    assert (await api.get("/reembed")).json()["model"] == "hash-32"
    r = await api.post("/reembed/finalize")
    assert r.status_code == 200 and r.json()["collection"].endswith("hash-32_32")
    assert (await api.get("/reembed")).json() is None
    assert (await api.post("/reembed/finalize")).status_code == 409
