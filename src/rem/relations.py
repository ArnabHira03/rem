"""Pure relation rules: type normalization, deterministic ids, the embedded sentence and derived trust.

No storage, embedder or vector-store imports. A relation's trust comes from its evidence memories.
"""

import uuid

from rem.models import RELATION_TYPES, MemoryRecord, Provenance, Status, normalize_type

__all__ = [
    "RELATION_NS",
    "RELATION_TYPES",
    "derive_trust",
    "normalize_type",
    "relation_id",
    "relation_sentence",
]

RELATION_NS = uuid.UUID("7a3cd350-88a3-41ae-b0a6-3c15c92209c4")

# Strongest first.
_PROVENANCE_STRENGTH = [
    Provenance.USER_CONFIRMED,
    Provenance.USER_ASSERTED,
    Provenance.TOOL_OBSERVED,
    Provenance.ASSISTANT_GENERATED,
]


def relation_id(source_entity_id: str, type: str, target_entity_id: str) -> str:
    return str(uuid.uuid5(RELATION_NS, f"{source_entity_id}|{type}|{target_entity_id}"))


def relation_sentence(source_name: str, type: str, target_name: str) -> str:
    return f"{source_name} {type.replace('_', ' ')} {target_name}"


def derive_trust(evidence: list[MemoryRecord]) -> tuple[Provenance, Status]:
    """Verified if any evidence memory is; provenance is the strongest among the evidence.
    An empty list raises ValueError: the caller deletes a relation with no evidence."""
    if not evidence:
        raise ValueError("a relation needs at least one evidence memory")
    provenance = min((r.provenance for r in evidence), key=_PROVENANCE_STRENGTH.index)
    verified = any(r.status == Status.VERIFIED for r in evidence)
    return provenance, Status.VERIFIED if verified else Status.UNVERIFIED
