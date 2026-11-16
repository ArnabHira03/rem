"""HTTP API served on the daemon's Unix socket."""

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from rem.daemon.schemas import IngestBody
from rem.daemon.service import RemService
from rem.errors import MemoryNotFound, RemError
from rem.models import HistoryRow, MemoryRecord, Provenance, Scope, SearchResult, Status


class AddBody(BaseModel):
    text: str
    scope: Scope = Scope.GLOBAL
    cwd: str | None = None
    provenance: Provenance = Provenance.USER_CONFIRMED
    status: Status = Status.VERIFIED
    source: str = "cli"
    actor: str = "cli"


class UpdateBody(BaseModel):
    text: str
    actor: str = "cli"
    unverify: bool = False


class ActorBody(BaseModel):
    actor: str = "cli"


class RequestConfirmationBody(BaseModel):
    actor: str = "mcp"
    cwd: str | None = None


class SearchBody(BaseModel):
    query: str
    cwd: str | None = None
    limit: int = Field(10, ge=1, le=100)
    scope: Scope | None = None


class ContextBody(BaseModel):
    cwd: str | None = None
    max_chars: int = Field(1600, ge=1, le=10000)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": code, "message": message}, status_code=status)


def _json(value, status: int = 200) -> JSONResponse:
    def conv(v):
        if isinstance(v, list):
            return [conv(x) for x in v]
        if isinstance(v, SearchResult):
            return {**v.record.to_payload(), "score": v.score, "semantic": v.semantic}
        if isinstance(v, MemoryRecord):
            return v.to_payload()
        if isinstance(v, HistoryRow):
            return v.model_dump(mode="json")
        return v

    return JSONResponse(conv(value), status_code=status)


async def _body[T: BaseModel](request: Request, model: type[T]) -> T:
    raw = await request.body()
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        raise ValueError("request body must be JSON") from None
    return model.model_validate(data)


def _path(cwd: str | None) -> Path | None:
    return Path(cwd) if cwd else None


def build_app(service: RemService) -> Starlette:
    async def status(request: Request):
        return _json(await service.status())

    async def add(request: Request):
        b = await _body(request, AddBody)
        rec = await service.add(b.text, scope=b.scope, cwd=_path(b.cwd), provenance=b.provenance,
                                status=b.status, source=b.source, actor=b.actor)
        return _json(rec, 201)

    async def list_memories(request: Request):
        q = request.query_params
        if q.get("cwd") and q.get("project_id"):
            raise ValueError("give cwd or project_id, not both")
        rec = await service.list(
            scope=Scope(q["scope"]) if q.get("scope") else None,
            status=Status(q["status"]) if q.get("status") else None,
            project_id=int(q["project_id"]) if q.get("project_id") else None,
            cwd=_path(q.get("cwd")),
        )
        return _json(rec)

    async def get(request: Request):
        mid = request.path_params["id"]
        rec = await service.get(mid)
        if rec is None:
            raise MemoryNotFound(mid)
        return _json(rec)

    async def update(request: Request):
        b = await _body(request, UpdateBody)
        return _json(await service.update(request.path_params["id"], b.text, actor=b.actor, unverify=b.unverify))

    async def delete(request: Request):
        await service.delete(request.path_params["id"], actor=request.query_params.get("actor", "cli"))
        return Response(status_code=204)

    async def confirm(request: Request):
        b = await _body(request, ActorBody)
        return _json(await service.confirm(request.path_params["id"], actor=b.actor))

    async def request_confirmation(request: Request):
        b = await _body(request, RequestConfirmationBody)
        return _json(await service.request_confirmation(request.path_params["id"], actor=b.actor, cwd=_path(b.cwd)))

    async def history(request: Request):
        return _json(await service.history(request.path_params["id"]))

    async def search(request: Request):
        b = await _body(request, SearchBody)
        return _json(await service.search(b.query, cwd=_path(b.cwd), limit=b.limit, scope=b.scope))

    async def context(request: Request):
        b = await _body(request, ContextBody)
        r = await service.context(_path(b.cwd), b.max_chars)
        return _json({"text": r.text, "count": r.count})

    async def ingest(request: Request):
        b = await _body(request, IngestBody)
        r = await service.ingest(b.source, b.session_id, b.cwd_path(), b.turns_for_ingest(), b.enqueue)
        return _json({"session_pk": r.session_pk, "accepted": r.accepted,
                      "duplicates": r.duplicates, "job_id": r.job_id}, 202)

    async def retry_failed(request: Request):
        return _json({"requeued": await service.retry_failed()})

    async def on_not_found(request, exc: MemoryNotFound):
        return _error(404, "not_found", str(exc))

    async def on_invalid(request, exc: Exception):
        if isinstance(exc, ValidationError):
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"])
            return _error(400, "invalid", f"{where}: {first['msg']}" if where else first["msg"])
        return _error(400, "invalid", str(exc))

    async def on_rem_error(request, exc: RemError):
        return _error(409, "conflict", str(exc))

    return Starlette(
        routes=[
            Route("/status", status, methods=["GET"]),
            Route("/memories", add, methods=["POST"]),
            Route("/memories", list_memories, methods=["GET"]),
            Route("/memories/{id}", get, methods=["GET"]),
            Route("/memories/{id}", update, methods=["PATCH"]),
            Route("/memories/{id}", delete, methods=["DELETE"]),
            Route("/memories/{id}/confirm", confirm, methods=["POST"]),
            Route("/memories/{id}/request-confirmation", request_confirmation, methods=["POST"]),
            Route("/memories/{id}/history", history, methods=["GET"]),
            Route("/search", search, methods=["POST"]),
            Route("/context", context, methods=["POST"]),
            Route("/ingest", ingest, methods=["POST"]),
            Route("/jobs/retry", retry_failed, methods=["POST"]),
        ],
        exception_handlers={
            MemoryNotFound: on_not_found,
            ValidationError: on_invalid,
            ValueError: on_invalid,
            RemError: on_rem_error,
        },
    )
