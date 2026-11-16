"""rem reembed: copy every memory into a collection built with another embedder, then switch to it.

The copy keeps each memory's id and payload and writes the dense vector from the target embedder and the
BM25 vector from the same sparse embedder. Entities are copied the same way into their own target collection
(dense only: the vector is the target embedder applied to the entity's normalized name). The active collections
stay in use until finalize, which first brings the targets up to date with any writes made since the copy, then
switches this process and the store.
"""

import json
from dataclasses import dataclass

from rem.config import EmbedderConfig
from rem.embed.base import Embedder, collection_slug
from rem.embed.factory import create_embedder
from rem.errors import ReembedError
from rem.memory import (
    ACTIVE_COLLECTION_KEY,
    ACTIVE_EMBEDDER_KEY,
    ACTIVE_ENTITY_COLLECTION_KEY,
    ACTIVE_MODEL_KEY,
    TARGET_KEY,
    Memory,
)
from rem.models import HistoryEvent, TargetKind
from rem.storage.vector.base import VectorPoint

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


def _choose_collection(mem: Memory, base: str, active: str | None = None) -> str:
    """The default name when it is free; otherwise the first __gN name that is neither taken nor the active one
    (`active`, the active memory collection by default)."""
    active = mem.col if active is None else active
    if base != active and not mem.vectors.col_exists(base):
        return base
    n = 1
    while True:
        name = f"{base}__g{n}"
        if name != active and not mem.vectors.col_exists(name):
            return name
        n += 1


def _embed_into(mem: Memory, target: Embedder, col: str, points, batch: int = BATCH) -> None:
    """Write points (payload kept as is) into col with fresh dense and BM25 vectors, batch by batch."""
    for i in range(0, len(points), batch):
        chunk = points[i:i + batch]
        texts = [p.payload["text"] for p in chunk]
        dense = target.embed(texts)
        sparse = mem.sparse_embedder.embed_docs(texts)
        mem.vectors.upsert(col, [VectorPoint(p.id, d, p.payload, s) for p, d, s in zip(chunk, dense, sparse)])


def _embed_entities_into(mem: Memory, target: Embedder, col: str, points, batch: int = BATCH) -> None:
    """Write entity points (payload kept as is) into col with a fresh dense vector of the normalized name,
    the same text the entity write path embeds. Entities have no BM25 vector."""
    for i in range(0, len(points), batch):
        chunk = points[i:i + batch]
        dense = target.embed([p.payload["normalized"] for p in chunk])
        mem.vectors.upsert(col, [VectorPoint(p.id, d, p.payload) for p, d in zip(chunk, dense)])


def _entity_target(mem: Memory, pending: dict | None, slug: str) -> str:
    """The pending entity target, or a new one (also for a target recorded before M7, which has none)."""
    if pending and pending.get("entity_collection"):
        return pending["entity_collection"]
    return _choose_collection(mem, f"rem_entities__{slug}", mem.entity_col)


def _sync(mem: Memory, src_col: str, dst_col: str, write) -> None:
    """Bring dst_col up to date with src_col: (re)write points whose payload differs, delete removed ids."""
    active = mem.vectors.list_points(src_col)
    src = {p.id for p in active}
    dst = {p.id: p.payload for p in mem.vectors.list_points(dst_col)}
    write([p for p in active if dst.get(p.id) != p.payload])
    stale = [i for i in dst if i not in src]
    if stale:
        mem.vectors.delete(dst_col, stale)
    if {p.id for p in mem.vectors.list_points(dst_col)} != src:
        raise ReembedError(f"{dst_col} does not match {src_col}; nothing was switched")


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
    entity_col = _entity_target(mem, pending, slug)
    # recorded before the copy, so a copy killed midway is resumed by the rerun (not orphaned), and
    # finalize completes whatever is missing
    mem.db.set_meta(TARGET_KEY, json.dumps(
        {"collection": col, "slug": slug, "model": model, "entity_collection": entity_col}
    ))
    mem.vectors.create_col(col, target.dims, sparse=True)
    mem.vectors.create_col(entity_col, target.dims, sparse=False)
    points = mem.vectors.list_points(mem.col)
    _embed_into(mem, target, col, points, batch)
    _embed_entities_into(mem, target, entity_col, mem.vectors.list_points(mem.entity_col), batch)
    mem.db.add_history(TargetKind.MEMORY, col, HistoryEvent.REEMBED, mem.col, col, "cli", f"reembed:{col}")
    return ReembedResult(col, len(points))


def finalize(mem: Memory) -> str:
    """Bring the pending target up to date with the active collection, then switch to it and drop the old one."""
    pending = _pending(mem)
    if not pending:
        raise ReembedError("no reembed is pending; run `rem reembed --model <model>` first")
    target = _target_embedder(mem, pending["model"])
    col, old = pending["collection"], mem.col
    entity_col, old_entity = _entity_target(mem, pending, pending["slug"]), mem.entity_col
    if col == old:
        raise ReembedError(f"the pending target {col} is the active collection; nothing was switched")
    if entity_col == old_entity:
        raise ReembedError(f"the pending target {entity_col} is the active entity collection; nothing was switched")
    _sync(mem, old, col, lambda pts: _embed_into(mem, target, col, pts))
    if pending.get("entity_collection") != entity_col:
        # a target recorded before M7 has no entity collection: record the one picked here (so a rerun after a
        # failure resumes it, not orphans it), then the sync below creates it and copies every entity
        mem.db.set_meta(TARGET_KEY, json.dumps({**pending, "entity_collection": entity_col}))
    mem.vectors.create_col(entity_col, target.dims, sparse=False)
    _sync(mem, old_entity, entity_col, lambda pts: _embed_entities_into(mem, target, entity_col, pts))
    # one transaction: a crash midway must not leave the store half switched (e.g. a new collection with
    # the old embedder) or the target dropped before the switch; until it commits, finalize can rerun
    with mem.db.transaction():
        mem.db.set_meta(ACTIVE_COLLECTION_KEY, col)
        mem.db.set_meta(ACTIVE_ENTITY_COLLECTION_KEY, entity_col)
        mem.db.set_meta(ACTIVE_EMBEDDER_KEY, pending["slug"])
        mem.db.set_meta(ACTIVE_MODEL_KEY, pending["model"])
        mem.db.delete_meta(TARGET_KEY)
    mem.set_active(col, target)
    mem.set_entity_col(entity_col)
    mem.vectors.delete_col(old)
    mem.vectors.delete_col(old_entity)
    return col


def status(mem: Memory) -> dict | None:
    """The pending reembed, or None."""
    return _pending(mem)
