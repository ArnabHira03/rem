import math

from rem.storage.vector.base import Filters, ScoredPoint, VectorPoint, VectorStore


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _matches(payload: dict, filters: Filters | None) -> bool:
    for key, want in (filters or {}).items():
        have = payload.get(key)
        if isinstance(want, list):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


class InMemoryVectorStore(VectorStore):
    """Dict-backed store for tests and experiments; nothing is persisted."""

    def __init__(self) -> None:
        self._cols: dict[str, dict[str, VectorPoint]] = {}

    def create_col(self, name: str, vector_size: int) -> None:
        self._cols.setdefault(name, {})

    def col_exists(self, name: str) -> bool:
        return name in self._cols

    def delete_col(self, name: str) -> None:
        self._cols.pop(name, None)

    def upsert(self, col: str, points: list[VectorPoint]) -> None:
        for p in points:
            self._cols[col][p.id] = VectorPoint(p.id, list(p.vector or []), dict(p.payload))

    def search(self, col, vector, limit, filters=None):
        hits = [
            ScoredPoint(p.id, _cosine(vector, p.vector or []), dict(p.payload))
            for p in self._cols[col].values()
            if _matches(p.payload, filters)
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def get(self, col, ids):
        pts = self._cols[col]
        return [VectorPoint(i, list(pts[i].vector or []), dict(pts[i].payload)) for i in ids if i in pts]

    def update_payload(self, col, id, payload):
        self._cols[col][id].payload.update(payload)

    def delete(self, col, ids):
        for i in ids:
            self._cols[col].pop(i, None)

    def list_points(self, col, filters=None, limit=None):
        return [
            VectorPoint(p.id, list(p.vector or []), dict(p.payload))
            for p in self._cols[col].values()
            if _matches(p.payload, filters)
        ][:limit]
