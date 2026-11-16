"""Pure entity rules: name normalization, deterministic ids and scope buckets.

No storage, embedder or vector-store imports; callers pass bucketed values.
"""

import uuid

from rem.models import Scope

ENTITY_NS = uuid.UUID("5bc25946-d1bb-4aed-bf8e-bf4d0e4e8401")
ENTITY_MERGE_MIN = 0.95
ENTITY_QUERY_MIN = 0.5
ENTITY_BOOST = 0.05


def normalize_name(name: str) -> str:
    return " ".join(name.split()).casefold()


def entity_id(scope: Scope, project_id: int | None, normalized: str) -> str:
    return str(uuid.uuid5(ENTITY_NS, f"{scope}|{project_id}|{normalized}"))


def bucket(scope: Scope, project_id: int | None) -> tuple[Scope, int | None]:
    return (scope, project_id if scope == Scope.PROJECT else None)
