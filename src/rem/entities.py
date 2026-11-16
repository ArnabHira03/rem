"""Pure entity rules: name normalization, deterministic ids and scope buckets.

No storage, embedder or vector-store imports; callers pass bucketed values.
"""

import re
import uuid
from collections.abc import Iterable

from rem.models import Scope

ENTITY_NS = uuid.UUID("5bc25946-d1bb-4aed-bf8e-bf4d0e4e8401")
ENTITY_MERGE_MIN = 0.95
ENTITY_BOOST = 0.05


def normalize_name(name: str) -> str:
    return " ".join(name.split()).casefold()


def entity_id(scope: Scope, project_id: int | None, normalized: str) -> str:
    return str(uuid.uuid5(ENTITY_NS, f"{scope}|{project_id}|{normalized}"))


def bucket(scope: Scope, project_id: int | None) -> tuple[Scope, int | None]:
    return (scope, project_id if scope == Scope.PROJECT else None)


def mentions_entity(normalized_query: str, names: Iterable[str]) -> bool:
    """Whether any of names (an entity's name and aliases) occurs in normalized_query as a whole token
    sequence: the characters around it are the string's ends or not alphanumeric (unicode letters and
    digits count; "_" and punctuation are boundaries). So "uv" matches "which uv" but not "uvicorn",
    and "node.js" or "c++" match as written. Names are normalized here; the query must already be."""
    for name in names:
        norm = normalize_name(name)
        if norm and re.search(rf"(?<![^\W_]){re.escape(norm)}(?![^\W_])", normalized_query):
            return True
    return False
