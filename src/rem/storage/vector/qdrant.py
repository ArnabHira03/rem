from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from rem.errors import StoreLockedError
from rem.storage.vector.base import Filters, ScoredPoint, VectorPoint, VectorStore

SCROLL_PAGE = 1000  # points per scroll request; list_points pages until exhausted


def _to_filter(filters: Filters | None) -> qm.Filter | None:
    if not filters:
        return None
    must: list = []
    for key, want in filters.items():
        if want is None:
            must.append(qm.IsNullCondition(is_null=qm.PayloadField(key=key)))
        elif isinstance(want, list):
            must.append(qm.FieldCondition(key=key, match=qm.MatchAny(any=want)))
        else:
            must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=want)))
    return qm.Filter(must=must)


class QdrantVectorStore(VectorStore):
    """Qdrant behind the VectorStore interface.

    Local mode (``path``) needs no server but allows only one process per path.
    Payload indexes are not created: they have no effect in local mode.
    """

    def __init__(self, path: Path | None = None, url: str | None = None, location: str | None = None):
        try:
            if url:
                self._client = QdrantClient(url=url)
            elif location:
                self._client = QdrantClient(location=location)
            else:
                self._client = QdrantClient(path=str(path))
        except RuntimeError as e:
            if "already accessed" in str(e):
                raise StoreLockedError(
                    f"{path} is in use by another rem process (is the daemon running?)"
                ) from e
            raise

    def create_col(self, name: str, vector_size: int) -> None:
        if not self._client.collection_exists(name):
            self._client.create_collection(
                name, vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE)
            )

    def col_exists(self, name: str) -> bool:
        return self._client.collection_exists(name)

    def delete_col(self, name: str) -> None:
        if self._client.collection_exists(name):
            self._client.delete_collection(name)

    def upsert(self, col: str, points: list[VectorPoint]) -> None:
        self._client.upsert(
            col,
            points=[qm.PointStruct(id=p.id, vector=list(p.vector or []), payload=p.payload) for p in points],
        )

    def search(self, col, vector, limit, filters=None):
        res = self._client.query_points(
            col, query=list(vector), limit=limit, query_filter=_to_filter(filters), with_payload=True
        )
        return [ScoredPoint(str(p.id), p.score, dict(p.payload or {})) for p in res.points]

    def get(self, col, ids):
        found = self._client.retrieve(col, ids=ids, with_payload=True)
        by_id = {str(p.id): p for p in found}
        return [VectorPoint(i, None, dict(by_id[i].payload or {})) for i in ids if i in by_id]

    def update_payload(self, col, id, payload):
        self._client.set_payload(col, payload=payload, points=[id])

    def delete(self, col, ids):
        self._client.delete(col, points_selector=qm.PointIdsList(points=ids))

    def list_points(self, col, filters=None, limit=None):
        flt = _to_filter(filters)
        out: list[VectorPoint] = []
        offset = None
        while limit is None or len(out) < limit:
            page = SCROLL_PAGE if limit is None else min(SCROLL_PAGE, limit - len(out))
            points, offset = self._client.scroll(
                col, scroll_filter=flt, limit=page, offset=offset, with_payload=True
            )
            out.extend(VectorPoint(str(p.id), None, dict(p.payload or {})) for p in points)
            if offset is None:
                break
        return out

    def close(self) -> None:
        self._client.close()
