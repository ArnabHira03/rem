"""The Memory façade: the only code that writes to rem's stores.

Like Mem0's Memory class, but with a fixed write order so a crash never leaves
duplicates: the vector store (canonical record) is written first, history second.
"""

from datetime import UTC, datetime
import uuid
from pathlib import Path

from rem.config import RemConfig
from rem.embed.base import Embedder, collection_slug
from rem.embed.factory import create_embedder
from rem.errors import EmbedderMismatchError, MemoryNotFound
from rem.models import (
    HistoryEvent,
    REM_NAMESPACE,
    HistoryRow,
    MemoryRecord,
    Provenance,
    SearchResult,
    Scope,
    Status,
    TargetKind,
    memory_id,
    normalize_text,
    text_hash,
    utcnow,
)
from rem.projects import resolve_project
from rem.retrieve import overfetch, rank
from rem.storage.sqlite import RemDB
from rem.storage.vector.base import VectorPoint, VectorStore
from rem.storage.vector.factory import create_vector_store

ACTIVE_EMBEDDER_KEY = "active_embedder"


class Memory:
    def __init__(
        self,
        cfg: RemConfig,
        *,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
    ):
        self.db = RemDB(cfg.db_path)
        try:
            self.embedder = embedder or create_embedder(cfg.embedder)
            slug = collection_slug(self.embedder.model_id, self.embedder.dims)
            stored = self.db.get_meta(ACTIVE_EMBEDDER_KEY)
            if stored and stored != slug:
                raise EmbedderMismatchError(
                    f"stored memories use {stored}; configured embedder is {slug}. "
                    "Run `rem reembed` (coming in M6) or change the config back."
                )
            self.vectors = vector_store or create_vector_store(cfg)
        except BaseException:
            self.db.close()
            raise
        self.db.set_meta(ACTIVE_EMBEDDER_KEY, slug)
        self.col = f"rem_memories__{slug}"
        self.vectors.create_col(self.col, self.embedder.dims)

    # ---- reads -------------------------------------------------------------

    def get(self, id: str) -> MemoryRecord | None:
        found = self.vectors.get(self.col, [id])
        return MemoryRecord.from_payload(found[0].payload) if found else None

    def list(
        self,
        *,
        scope: Scope | None = None,
        project_id: int | None = None,
        status: Status | None = None,
    ) -> list[MemoryRecord]:
        filters = {k: str(v) if k != "project_id" else v for k, v in
                   {"scope": scope, "project_id": project_id, "status": status}.items() if v is not None}
        records = [MemoryRecord.from_payload(p.payload) for p in self.vectors.list_points(self.col, filters)]
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def search(
        self,
        query: str,
        *,
        cwd: Path | None = None,
        limit: int = 10,
        scope: Scope | None = None,
    ) -> list[SearchResult]:
        normalize_text(query)  # rejects empty queries
        current = None
        if cwd is not None:  # search never registers new projects
            current = self.db.find_project(resolve_project(cwd).root_path)
        hits = self.vectors.search(
            self.col,
            self.embedder.embed_query(query),
            overfetch(limit),
            {"scope": str(scope)} if scope else None,
        )
        candidates = [(MemoryRecord.from_payload(h.payload), h.score) for h in hits]
        return rank(candidates, current, limit, datetime.now(UTC))

    def history(self, id: str) -> list[HistoryRow]:
        rows = self.db.get_history(id)
        if not rows and self.get(id) is None:
            raise MemoryNotFound(id)
        return rows

    # ---- writes (vector store first, history second) -------------------------

    def add(
        self,
        text: str,
        *,
        scope: Scope,
        cwd: Path | None = None,
        provenance: Provenance,
        status: Status,
        source: str,
        actor: str,
        job_id: str | None = None,
        session_id: str | None = None,
    ) -> MemoryRecord:
        normalize_text(text)  # rejects empty text
        text = text.strip()
        project_id = project_name = None
        if scope == Scope.PROJECT:
            if cwd is None:
                raise ValueError("project scope needs a working directory")
            info = resolve_project(cwd)
            project_id = self.db.upsert_project(info.root_path, info.name, info.git_remote)
            project_name = info.name

        duplicate = self._find_by_text(text, scope, project_id)
        if duplicate is not None:
            # Same fact already stored (or a crashed add being retried): make sure its ADD row exists.
            if not any(h.event == HistoryEvent.ADD for h in self.db.get_history(duplicate.id)):
                self.db.add_history(TargetKind.MEMORY, duplicate.id, HistoryEvent.ADD, None, duplicate.text, actor, job_id)
            return duplicate

        mid = memory_id(text, scope, project_id, job_id)
        while self.get(mid) is not None:
            # An updated memory keeps the id derived from its old text; derive the next id deterministically.
            mid = str(uuid.uuid5(REM_NAMESPACE, mid))

        now = utcnow()
        record = MemoryRecord(
            id=mid, text=text, hash=text_hash(text), scope=scope, project_id=project_id,
            project_name=project_name, provenance=provenance, status=status, evidence_source=None,
            observed_at=None, source=source, session_id=session_id, job_id=job_id,
            created_at=now, updated_at=now,
        )
        self.vectors.upsert(self.col, [VectorPoint(mid, self.embedder.embed([text])[0], record.to_payload())])
        self.db.add_history(TargetKind.MEMORY, mid, HistoryEvent.ADD, None, text, actor, job_id)
        return record

    def update(self, id: str, text: str, *, actor: str) -> MemoryRecord:
        normalize_text(text)
        text = text.strip()
        old = self._require(id)
        other = self._find_by_text(text, old.scope, old.project_id)
        if other is not None and other.id != id:
            raise ValueError(f"another memory already has this text: {other.id}")
        record = old.model_copy(update={"text": text, "hash": text_hash(text), "updated_at": utcnow()})
        self.vectors.upsert(self.col, [VectorPoint(id, self.embedder.embed([text])[0], record.to_payload())])
        self.db.add_history(TargetKind.MEMORY, id, HistoryEvent.UPDATE, old.text, text, actor)
        return record

    def confirm(self, id: str, *, actor: str) -> MemoryRecord:
        old = self._require(id)
        if old.provenance == Provenance.USER_CONFIRMED and old.status == Status.VERIFIED:
            return old
        confirmed = old.model_copy(
            update={"provenance": Provenance.USER_CONFIRMED, "status": Status.VERIFIED, "updated_at": utcnow()}
        )
        payload = confirmed.to_payload()
        self.vectors.update_payload(self.col, id, {k: payload[k] for k in ("provenance", "status", "updated_at")})
        self.db.add_history(TargetKind.MEMORY, id, HistoryEvent.CONFIRM, str(old.status), str(Status.VERIFIED), actor)
        return confirmed

    def delete(self, id: str, *, actor: str) -> None:
        old = self._require(id)
        self.vectors.delete(self.col, [id])
        self.db.add_history(TargetKind.MEMORY, id, HistoryEvent.DELETE, old.text, None, actor)

    def _find_by_text(self, text: str, scope: Scope, project_id: int | None) -> MemoryRecord | None:
        """A live memory with the same normalized text in the same scope and project, if any."""
        found = self.vectors.list_points(
            self.col, {"hash": text_hash(text), "scope": str(scope), "project_id": project_id}, limit=1
        )
        return MemoryRecord.from_payload(found[0].payload) if found else None

    def _require(self, id: str) -> MemoryRecord:
        record = self.get(id)
        if record is None:
            raise MemoryNotFound(id)
        return record

    def close(self) -> None:
        self.vectors.close()
        self.db.close()
