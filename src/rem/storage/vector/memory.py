import math

from rem.storage.vector.base import Filters, ScoredPoint, SparseVector, VectorPoint, VectorStore


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
        self._sparse: set[str] = set()

    def create_col(self, name: str, vector_size: int, *, sparse: bool = False) -> None:
        self._cols.setdefault(name, {})
        if sparse:
            self._sparse.add(name)

    def col_has_sparse(self, name: str) -> bool:
        return name in self._sparse

    def col_exists(self, name: str) -> bool:
        return name in self._cols

    def delete_col(self, name: str) -> None:
        self._cols.pop(name, None)

    def upsert(self, col: str, points: list[VectorPoint]) -> None:
        for p in points:
            self._cols[col][p.id] = VectorPoint(p.id, list(p.vector or []), dict(p.payload), p.sparse)

    def search(self, col, vector, limit, filters=None):
        hits = [
            ScoredPoint(p.id, _cosine(vector, p.vector or []), dict(p.payload))
            for p in self._cols[col].values()
            if _matches(p.payload, filters)
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def keyword_search(self, col, query_sparse, limit, filters=None):
        if col not in self._sparse:
            return None
        pts = [p for p in self._cols[col].values() if p.sparse and _matches(p.payload, filters)]
        n = len(pts)
        df: dict[int, int] = {}
        for p in pts:
            for t in set(p.sparse.indices):
                df[t] = df.get(t, 0) + 1
        query = dict(zip(query_sparse.indices, query_sparse.values))
        hits = []
        for p in pts:
            weights = dict(zip(p.sparse.indices, p.sparse.values))
            score = 0.0
            for t, qv in query.items():
                if t in weights:
                    d = df[t]
                    idf = math.log(1 + (n - d + 0.5) / (d + 0.5))
                    score += qv * weights[t] * idf
            if score > 0:
                hits.append(ScoredPoint(p.id, score, dict(p.payload)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def get(self, col, ids, with_vectors=False):
        pts = self._cols[col]
        out = []
        for i in ids:
            if i not in pts:
                continue
            p = pts[i]
            if with_vectors:
                out.append(VectorPoint(i, list(p.vector or []), dict(p.payload), p.sparse))
            else:
                out.append(VectorPoint(i, None, dict(p.payload)))
        return out

    def update_payload(self, col, id, payload):
        self._cols[col][id].payload.update(payload)

    def delete(self, col, ids):
        for i in ids:
            self._cols[col].pop(i, None)

    def list_points(self, col, filters=None, limit=None):
        return [
            VectorPoint(p.id, None, dict(p.payload))
            for p in self._cols[col].values()
            if _matches(p.payload, filters)
        ][:limit]
