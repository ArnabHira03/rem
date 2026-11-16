"""rem reembed: copy every memory into a collection built with another embedder, then switch to it.

The copy keeps each memory's id and payload and writes the dense vector from the target embedder and the
BM25 vector from the same sparse embedder. The active collection stays in use until finalize, which first
brings the target up to date with any writes made since the copy, then switches this process and the store.
"""

import json
from dataclasses import dataclass

from rem.config import EmbedderConfig
from rem.embed.base import Embedder, collection_slug
from rem.embed.factory import create_embedder
from rem.errors import ReembedError
from rem.memory import ACTIVE_COLLECTION_KEY, ACTIVE_EMBEDDER_KEY, Memory
from rem.models import HistoryEvent, TargetKind
from rem.storage.vector.base import VectorPoint

TARGET_KEY = "reembed_target"  # JSON {"collection", "slug", "model"} while a reembed is pending
BATCH = 64


@dataclass
class ReembedResult:
    collection: str
    count: int


def _target_embedder(mem: Memory, model: str) -> Embedder:
    return create_embedder(EmbedderConfig(provider=mem.cfg.embedder.provider, model=model,
                                          cache_dir=mem.cfg.embedder.cache_dir))


def _pending(mem: Memory) -> dict | None:
    raw = mem.db.get_meta(TARGET_KEY)
    return json.loads(raw) if raw else None


def _choose_collection(mem: Memory, base: str) -> str:
    """The default name when it is free; otherwise the first free __gN generation."""
    if base != mem.col and not mem.vectors.col_exists(base):
        return base
    n = 1
    while mem.vectors.col_exists(f"{base}__g{n}") and f"{base}__g{n}" != mem.col:
        n += 1
    return f"{base}__g{n}"


def _embed_into(mem: Memory, target: Embedder, col: str, points, batch: int = BATCH) -> None:
    """Write points (payload kept as is) into col with fresh dense and BM25 vectors, batch by batch."""
    for i in range(0, len(points), batch):
        chunk = points[i:i + batch]
        texts = [p.payload["text"] for p in chunk]
        dense = target.embed(texts)
        sparse = mem.sparse_embedder.embed_docs(texts)
        mem.vectors.upsert(col, [VectorPoint(p.id, d, p.payload, s) for p, d, s in zip(chunk, dense, sparse)])


def run(mem: Memory, model: str, batch: int = BATCH) -> ReembedResult:
    """Copy every memory of the active collection into the target collection. Safe to repeat."""
    target = _target_embedder(mem, model)
    slug = collection_slug(target.model_id, target.dims)
    pending = _pending(mem)
    if pending and pending["slug"] != slug:
        raise ReembedError(
            f"a reembed to {pending['model']} is pending; run `rem reembed --finalize` first"
        )
    col = pending["collection"] if pending else _choose_collection(mem, f"rem_memories__{slug}")
    mem.vectors.create_col(col, target.dims, sparse=True)
    points = mem.vectors.list_points(mem.col)
    _embed_into(mem, target, col, points, batch)
    mem.db.set_meta(TARGET_KEY, json.dumps({"collection": col, "slug": slug, "model": model}))
    mem.db.add_history(TargetKind.MEMORY, col, HistoryEvent.REEMBED, mem.col, col, "cli", f"reembed:{col}")
    return ReembedResult(col, len(points))


def finalize(mem: Memory) -> str:
    """Bring the pending target up to date with the active collection, then switch to it and drop the old one."""
    pending = _pending(mem)
    if not pending:
        raise ReembedError("no reembed is pending; run `rem reembed --model <model>` first")
    target = _target_embedder(mem, pending["model"])
    col, old = pending["collection"], mem.col
    active = mem.vectors.list_points(old)
    src = {p.id: p.payload for p in active}
    dst = {p.id: p.payload for p in mem.vectors.list_points(col)}
    changed = [p for p in active if dst.get(p.id) != p.payload]
    stale = [i for i in dst if i not in src]
    _embed_into(mem, target, col, changed)
    if stale:
        mem.vectors.delete(col, stale)
    if {p.id for p in mem.vectors.list_points(col)} != set(src):
        raise ReembedError("target collection does not match the active one; nothing was switched")
    mem.db.set_meta(ACTIVE_COLLECTION_KEY, col)
    mem.db.set_meta(ACTIVE_EMBEDDER_KEY, pending["slug"])
    mem.db.delete_meta(TARGET_KEY)
    mem.set_active(col, target)
    mem.vectors.delete_col(old)
    return col


def status(mem: Memory) -> dict | None:
    """The pending reembed, or None."""
    return _pending(mem)
