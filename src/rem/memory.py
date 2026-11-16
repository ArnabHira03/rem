"""The Memory façade: the only code that writes to rem's stores.

Like Mem0's Memory class, but with a fixed write order so a crash never leaves
duplicates: the vector store (canonical record) is written first, history second.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rem.config import RemConfig
from rem.embed.base import Embedder, collection_slug
from rem.embed.factory import create_embedder, create_sparse_embedder
from rem.embed.sparse import SparseEmbedder
from rem.errors import EmbedderMismatchError, MemoryNotFound
from rem.models import (
    Candidate,
    Confirmation,
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
from rem.projects import ProjectInfo, project_root, resolve_project
from rem.retrieve import cosine, fuse, overfetch, score_and_rank
from rem.storage.sqlite import RemDB
from rem.storage.vector.base import VectorPoint, VectorStore
from rem.storage.vector.factory import create_vector_store
from rem.trust import Action, assign_trust, decide, quote_in_user_turns, user_states

log = logging.getLogger(__name__)

ACTIVE_EMBEDDER_KEY = "active_embedder"
MAX_PENDING_CONFIRMATIONS = 10  # requests one job checks, newest first


@dataclass
class ApplyResult:
    added: list[str] = field(default_factory=list)
    corroborated: list[str] = field(default_factory=list)
    confirmed: list[str] = field(default_factory=list)
    skipped: int = 0
    dropped: int = 0


@dataclass
class ContextResult:
    text: str
    count: int


CONTEXT_HEADER = "Facts rem remembers from earlier sessions (verified):"


class Memory:
    def __init__(
        self,
        cfg: RemConfig,
        *,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
        sparse_embedder: SparseEmbedder | None = None,
    ):
        self.db = RemDB(cfg.db_path)
        try:
            self.embedder = embedder or create_embedder(cfg.embedder)
            self.sparse_embedder = sparse_embedder or create_sparse_embedder(cfg.embedder)
            slug = collection_slug(self.embedder.model_id, self.embedder.dims)
            stored = self.db.get_meta(ACTIVE_EMBEDDER_KEY)
            if stored and stored != slug:
                raise EmbedderMismatchError(
                    f"stored memories use {stored}; configured embedder is {slug}. "
                    "Run `rem reembed` to move them, or change the config back."
                )
            self.vectors = vector_store or create_vector_store(cfg)
            self.col = f"rem_memories__{slug}"
            if self.vectors.col_exists(self.col) and not self.vectors.col_has_sparse(self.col):
                # created before M6: keyword search would silently be off, so refuse before writing anything
                raise EmbedderMismatchError(
                    f"{self.col} was created before BM25 search existed; run `rem reembed` to rebuild it."
                )
        except BaseException:
            self.db.close()
            raise
        self.db.set_meta(ACTIVE_EMBEDDER_KEY, slug)
        self.vectors.create_col(self.col, self.embedder.dims, sparse=True)

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

    def list_for(self, cwd: Path, *, scope: Scope | None = None, status: Status | None = None) -> list[MemoryRecord]:
        """Like list(), but project memories only from cwd's project (its git root), never another
        repository's. A directory rem knows no project for has no project memories. Read-only."""
        current = self.db.find_project(str(project_root(cwd)))
        if scope == Scope.PROJECT:
            return [] if current is None else self.list(scope=scope, project_id=current, status=status)
        return [r for r in self.list(scope=scope, status=status)
                if r.scope != Scope.PROJECT or (current is not None and r.project_id == current)]

    def pending_confirmations(self, since: str, until: str, project_id: int | None) -> list[MemoryRecord]:
        """Unverified memories an agent in project_id asked to confirm in (since, until], newest first.

        The window is one job's exchange (IngestStore.job_window): an agent asks for confirmation
        after the user's reply, before that exchange's job is enqueued, so an older request (or one
        from a later exchange) is never matched against this user's words. A request from another
        project's session is never offered either.
        """
        recs = [r for r in self.list(status=Status.UNVERIFIED)
                if r.confirm_requested_at and since < r.confirm_requested_at <= until
                and r.confirm_request_project_id == project_id]
        recs.sort(key=lambda r: r.confirm_requested_at or "", reverse=True)
        return recs[:MAX_PENDING_CONFIRMATIONS]

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
            current = self.db.find_project(str(project_root(cwd)))
        filters = {"scope": str(scope)} if scope else None
        dense_vec = self.embedder.embed_query(query)
        sparse_vec = self.sparse_embedder.embed_query(query)
        hits = self.vectors.search(self.col, dense_vec, overfetch(limit), filters)
        keyword_hits = (
            self.vectors.keyword_search(self.col, sparse_vec, overfetch(limit), filters) or []
            if sparse_vec.indices else []
        )
        keyword = {h.id: h.score for h in keyword_hits}
        payloads = {h.id: (h.payload, h.score) for h in hits}
        # keyword-only matches: fetch them and score their dense cosine, so the threshold applies to a real number
        missing = [i for i in keyword if i not in payloads]
        for p in self.vectors.get(self.col, missing, with_vectors=True) if missing else []:
            if filters and any(p.payload.get(k) != v for k, v in filters.items()):
                continue
            payloads[p.id] = (p.payload, cosine(dense_vec, p.vector or []))
        candidates = [(MemoryRecord.from_payload(payload), score) for payload, score in payloads.values()]
        return score_and_rank(fuse(candidates, keyword), current, limit, datetime.now(UTC))

    def context(self, cwd: Path | None, max_chars: int) -> ContextResult:
        """Verified memories for SessionStart: current project, then system, then global; read-only."""
        current = self.db.find_project(str(project_root(cwd))) if cwd is not None else None
        verified = self.list(status=Status.VERIFIED)
        sections: list[tuple[str, list[MemoryRecord]]] = []
        if current is not None:
            proj = [r for r in verified if r.scope == Scope.PROJECT and r.project_id == current]
            name = next((r.project_name for r in proj if r.project_name), None)
            sections.append((f"Project {name or 'this project'}:", proj))
        sections.append(("This machine:", [r for r in verified if r.scope == Scope.SYSTEM]))
        sections.append(("About the user:", [r for r in verified if r.scope == Scope.GLOBAL]))

        lines, used, count, full = [CONTEXT_HEADER], len(CONTEXT_HEADER), 0, False
        for title, recs in sections:
            pending = len(title) + 1  # a section title is only emitted together with a fact
            for r in sorted(recs, key=lambda r: r.updated_at, reverse=True):
                if full:
                    break
                line = "- " + " ".join(r.text.split())
                cost = len(line) + 1 + pending
                if used + cost > max_chars:
                    full = True
                    break
                if pending:
                    lines.append(title)
                    pending = 0
                lines.append(line)
                used += cost
                count += 1
        if count == 0:
            return ContextResult("", 0)
        return ContextResult("\n".join(lines), count)

    def related(self, text: str, project_id: int | None, limit: int) -> list[MemoryRecord]:
        """Memories across all scopes related to text, ranked with project_id as the current project."""
        hits = self.vectors.search(self.col, self.embedder.embed_query(text), overfetch(limit))
        candidates = [(r, h.score) for h in hits
                      # a memory awaiting the user's confirmation is only ever shown to confirm_pending
                      if not (r := MemoryRecord.from_payload(h.payload)).confirm_requested_at]
        return [r.record for r in score_and_rank(fuse(candidates, {}), project_id, limit, datetime.now(UTC))]

    def similarity(self, text: str, id: str, scope: Scope, project_id: int | None) -> tuple[MemoryRecord, float] | None:
        """Memory `id` with its cosine similarity to text, if it is live in the given scope and project."""
        hits = self.vectors.search(
            self.col, self.embedder.embed_query(text), 1,
            {"id": id, "scope": str(scope), "project_id": project_id},
        )
        return (MemoryRecord.from_payload(hits[0].payload), hits[0].score) if hits else None

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
        project: ProjectInfo | None = None,
        provenance: Provenance,
        status: Status,
        source: str,
        actor: str,
        job_id: str | None = None,
        session_id: str | None = None,
        project_id: int | None = None,
        evidence_source: str | None = None,
        agent_claim: bool = False,
    ) -> MemoryRecord:
        normalize_text(text)  # rejects empty text
        text = text.strip()
        project_name = None
        if scope != Scope.PROJECT:
            project_id = None
        elif project_id is not None:
            project = self.db.get_project(project_id)
            if project is None:
                raise ValueError(f"unknown project id {project_id}")
            project_name = project[1]
        elif project is not None or cwd is not None:
            info = project if project is not None else resolve_project(cwd)
            project_id = self.db.upsert_project(info.root_path, info.name, info.git_remote)
            project_name = info.name
        else:
            raise ValueError("project scope needs a working directory or a project id")

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
            project_name=project_name, provenance=provenance, status=status, evidence_source=evidence_source,
            observed_at=None, source=source, session_id=session_id, job_id=job_id, agent_claim=agent_claim,
            created_at=now, updated_at=now,
        )
        self.vectors.upsert(self.col, [self._point(mid, text, record)])
        self.db.add_history(TargetKind.MEMORY, mid, HistoryEvent.ADD, None, text, actor, job_id)
        return record

    def update(self, id: str, text: str, *, actor: str, unverify: bool = False) -> MemoryRecord:
        """Change a memory's text. With ``unverify`` (an agent's edit) the record also becomes
        assistant-generated and unverified, in the same upsert, until the user confirms it."""
        normalize_text(text)
        text = text.strip()
        old = self._require(id)
        other = self._find_by_text(text, old.scope, old.project_id)
        if other is not None and other.id != id:
            raise ValueError(f"another memory already has this text: {other.id}")
        # an edit withdraws any confirmation request: the user never saw the new text
        changes = {"text": text, "hash": text_hash(text), "updated_at": utcnow(), "confirm_requested_at": None}
        if unverify:
            changes |= {"provenance": Provenance.ASSISTANT_GENERATED, "status": Status.UNVERIFIED, "agent_claim": True}
        record = old.model_copy(update=changes)
        self.vectors.upsert(self.col, [self._point(id, text, record)])
        self.db.add_history(TargetKind.MEMORY, id, HistoryEvent.UPDATE, old.text, text, actor)
        return record

    def confirm(self, id: str, *, actor: str) -> MemoryRecord:
        old = self._require(id)
        if old.provenance == Provenance.USER_CONFIRMED and old.status == Status.VERIFIED:
            return old
        confirmed = old.model_copy(
            update={"provenance": Provenance.USER_CONFIRMED, "status": Status.VERIFIED, "updated_at": utcnow(),
                    "confirm_requested_at": None}
        )
        payload = confirmed.to_payload()
        self.vectors.update_payload(
            self.col, id, {k: payload[k] for k in ("provenance", "status", "updated_at", "confirm_requested_at")}
        )
        self.db.add_history(TargetKind.MEMORY, id, HistoryEvent.CONFIRM, str(old.status), str(Status.VERIFIED), actor)
        return confirmed

    def request_confirmation(self, id: str, *, actor: str, cwd: Path | None = None) -> MemoryRecord:
        """Record that an agent working in cwd says the user confirmed this memory. It does NOT verify
        it: the memory stays unverified until apply_confirmations finds the user's own words, in a job
        of the same project, confirming it."""
        old = self._require(id)
        if old.status == Status.VERIFIED:
            return old
        project_id = self.db.find_project(str(project_root(cwd))) if cwd is not None else None
        record = old.model_copy(update={"confirm_requested_at": utcnow(), "confirm_request_project_id": project_id})
        self.vectors.update_payload(self.col, id, {"confirm_requested_at": record.confirm_requested_at,
                                                   "confirm_request_project_id": project_id})
        self.db.add_history(TargetKind.MEMORY, id, HistoryEvent.CONFIRM_REQUEST, None, None, actor)
        return record

    def apply_confirmations(
        self, confirmations: list[Confirmation], *, user_texts: list[str], job_key: str
    ) -> list[str]:
        """Verify the memories a job's extraction says the user confirmed, where that holds up.

        Each needs the same pending request and text the LLM was shown (requested_at, hash), and a
        quote that is the user's own words in this job and about this fact (quote_in_user_turns);
        otherwise the memory stays unverified.
        Replaying the job only fills in a missing CONFIRM history row.
        """
        done: list[str] = []
        for c in confirmations:
            rec = self.get(c.memory_id)
            if rec is None:
                continue
            replay = (rec.status == Status.VERIFIED and rec.corroborated_by == job_key
                      and rec.provenance == Provenance.USER_CONFIRMED)
            if not replay:
                if (rec.status != Status.UNVERIFIED or rec.confirm_requested_at != c.requested_at
                        or rec.hash != c.hash):
                    continue
                if not quote_in_user_turns(c.quote, user_texts, rec.text):
                    log.info("not confirming %s (%s): the quote is not in the user's messages", rec.id, job_key)
                    continue
                rec = rec.model_copy(update={
                    "provenance": Provenance.USER_CONFIRMED, "status": Status.VERIFIED, "corroborated_by": job_key,
                    "confirm_requested_at": None, "updated_at": utcnow(),
                })
                payload = rec.to_payload()
                self.vectors.update_payload(self.col, rec.id, {k: payload[k] for k in (
                    "provenance", "status", "corroborated_by", "confirm_requested_at", "updated_at")})
            self.db.add_history(TargetKind.MEMORY, rec.id, HistoryEvent.CONFIRM, str(Status.UNVERIFIED),
                                str(Status.VERIFIED), "extractor", job_key)
            done.append(rec.id)
        return done

    def corroborate(self, id: str, *, provenance: Provenance, job_id: str, actor: str) -> MemoryRecord:
        """Mark an unverified memory verified because a trusted origin restated it.

        Payload first (one update), then an INSERT OR IGNORE history row, so a replay of the
        same job only fills in a missing history row. A user_confirmed memory keeps its provenance.
        """
        record = self._require(id)
        if record.corroborated_by != job_id:
            if record.provenance == Provenance.USER_CONFIRMED:
                provenance = Provenance.USER_CONFIRMED
            record = record.model_copy(
                update={"provenance": provenance, "status": Status.VERIFIED, "corroborated_by": job_id,
                        "updated_at": utcnow()}
            )
            payload = record.to_payload()
            self.vectors.update_payload(
                self.col, id, {k: payload[k] for k in ("provenance", "status", "corroborated_by", "updated_at")}
            )
        self.db.add_history(
            TargetKind.MEMORY, id, HistoryEvent.CORROBORATE, str(Status.UNVERIFIED), str(Status.VERIFIED), actor, job_id
        )
        return record

    def apply_candidates(
        self,
        candidates: list[Candidate],
        *,
        job_key: str,
        project_id: int | None,
        session_id: str | None,
        source: str,
        restate_min_similarity: float,
        has_user_turn: bool,
        user_texts: list[str] | None = None,
    ) -> ApplyResult:
        """Write a job's extracted candidates, in order, so later ones see earlier writes.

        A candidate is compared with one existing memory only, never by cosine alone: the memory
        with the same normalized text, else the memory the LLM said it restates (same scope and
        project, and at least restate_min_similarity as a guard against a wrong link). A changed
        value or a contradiction has no such memory, so it is added rather than corroborating.

        Without a user turn in the job, nothing can be user-asserted: "user" origins are
        downgraded to "assistant" (unverified). Given the job's user_texts (the daemon always passes
        them), a "user" fact must also be named by a sentence the user actually wrote
        (trust.user_states); otherwise it is downgraded too, so the LLM's say-so that the user said
        it (perhaps misled by an assistant reply) never makes a fact verified on its own.

        Replaying the same job_key is idempotent: memory ids are deterministic, its own ADDs are
        re-taken (repairing a missing ADD row) and its own corroborations only complete history.
        """
        result = ApplyResult()
        if not has_user_turn:
            candidates = [c.model_copy(update={"origin": "assistant"}) if c.origin == "user" else c
                          for c in candidates]
        elif user_texts is not None:
            candidates = [c.model_copy(update={"origin": "assistant"})
                          if c.origin == "user" and not user_states(c.text, user_texts) else c
                          for c in candidates]
        kept, collapsed = _collapse_duplicates(candidates)
        result.skipped += collapsed
        for c in kept:
            if c.scope == Scope.PROJECT and project_id is None:
                log.info("dropping project-scope candidate without a project (job %s)", job_key)
                result.dropped += 1
                continue
            pid = project_id if c.scope == Scope.PROJECT else None
            trust = assign_trust(c)
            record: MemoryRecord | None = None
            own = self._find_by_text(c.text, c.scope, pid)
            if own is not None and own.job_id == job_key:
                action = Action.ADD  # replay of this job's own add: add() returns it and repairs history
            else:
                record, similarity = self._comparison_target(c, pid, own, restate_min_similarity)
                action = decide(trust, record, similarity, restate_min_similarity, job_key)
                if action == Action.CORROBORATE and record is not None and (
                        record.agent_claim or record.confirm_requested_at):
                    # an agent wrote it (or asked to confirm it): the extractor's word that the user
                    # said it is not enough; only the user's quoted words or `rem confirm` verify it
                    action = Action.SKIP
            if action == Action.ADD:
                added = self.add(
                    c.text, scope=c.scope, project_id=pid, provenance=trust.provenance, status=trust.status,
                    evidence_source=trust.evidence_source, source=source, actor="extractor",
                    job_id=job_key, session_id=session_id,
                )
                result.added.append(added.id)
            elif action == Action.CORROBORATE and record is not None and record.id not in result.corroborated:
                self.corroborate(record.id, provenance=trust.provenance, job_id=job_key, actor="extractor")
                result.corroborated.append(record.id)
            else:
                result.skipped += 1
        return result

    def _comparison_target(
        self, c: Candidate, pid: int | None, same_text: MemoryRecord | None, min_similarity: float
    ) -> tuple[MemoryRecord | None, float]:
        if same_text is not None:
            return same_text, 1.0
        if c.restates:
            found = self.similarity(c.text, c.restates, c.scope, pid)
            if found is not None and found[1] >= min_similarity:
                return found
        return None, 0.0

    def delete(self, id: str, *, actor: str) -> None:
        old = self._require(id)
        self.vectors.delete(self.col, [id])
        self.db.add_history(TargetKind.MEMORY, id, HistoryEvent.DELETE, old.text, None, actor)

    def _point(self, id: str, text: str, record: MemoryRecord) -> VectorPoint:
        """The point for a memory: dense vector, BM25 sparse vector and the payload that is the record."""
        return VectorPoint(
            id,
            self.embedder.embed([text])[0],
            record.to_payload(),
            self.sparse_embedder.embed_docs([text])[0],
        )

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


def _collapse_duplicates(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    """Keep one candidate per (normalized text, scope): the most trusted, at the group's first position."""
    kept: dict[tuple[str, Scope], Candidate] = {}
    for c in candidates:
        key = (normalize_text(c.text), c.scope)
        best = kept.get(key)
        if best is None or (
            assign_trust(c).status == Status.VERIFIED and assign_trust(best).status != Status.VERIFIED
        ):
            kept[key] = c  # replacing a value keeps the key's original insertion position
    return list(kept.values()), len(candidates) - len(kept)
