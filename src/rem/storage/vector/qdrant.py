from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from rem.errors import StoreLockedError
from rem.storage.vector.base import Filters, ScoredPoint, SparseVector, VectorPoint, VectorStore

DENSE = "dense"  # name of the dense vector in sparse-enabled collections
BM25 = "bm25"  # name of the sparse slot

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

    def create_col(self, name: str, vector_size: int, *, sparse: bool = False) -> None:
        if self._client.collection_exists(name):
            return
        if sparse:
            self._client.create_collection(
                name,
                vectors_config={DENSE: qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE)},
                sparse_vectors_config={BM25: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            )
        else:
            self._client.create_collection(
                name, vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE)
            )

    def col_has_sparse(self, name: str) -> bool:
        if not self._client.collection_exists(name):
            return False
        info = self._client.get_collection(name)
        return BM25 in (info.config.params.sparse_vectors or {})

    def col_exists(self, name: str) -> bool:
        return self._client.collection_exists(name)

    def delete_col(self, name: str) -> None:
        if self._client.collection_exists(name):
            self._client.delete_collection(name)

    def upsert(self, col: str, points: list[VectorPoint]) -> None:
        sparse_col = self.col_has_sparse(col)
        structs = []
        for p in points:
            if sparse_col:
                vec: dict = {DENSE: list(p.vector or [])}
                if p.sparse is not None:
                    vec[BM25] = qm.SparseVector(indices=p.sparse.indices, values=p.sparse.values)
            else:
                vec = list(p.vector or [])
            structs.append(qm.PointStruct(id=p.id, vector=vec, payload=p.payload))
        self._client.upsert(col, points=structs)

    def search(self, col, vector, limit, filters=None):
        using = DENSE if self.col_has_sparse(col) else None
        res = self._client.query_points(
            col, query=list(vector), using=using, limit=limit, query_filter=_to_filter(filters), with_payload=True
        )
        return [ScoredPoint(str(p.id), p.score, dict(p.payload or {})) for p in res.points]

    def keyword_search(self, col, query_sparse: SparseVector, limit, filters=None):
        if not self.col_has_sparse(col):
            return None
        res = self._client.query_points(
            col,
            query=qm.SparseVector(indices=query_sparse.indices, values=query_sparse.values),
            using=BM25,
            limit=limit,
            query_filter=_to_filter(filters),
            with_payload=True,
        )
        return [ScoredPoint(str(p.id), p.score, dict(p.payload or {})) for p in res.points]

    def get(self, col, ids, with_vectors=False):
        found = self._client.retrieve(col, ids=ids, with_payload=True, with_vectors=with_vectors)
        by_id = {str(p.id): p for p in found}
        out = []
        for i in ids:
            if i not in by_id:
                continue
            p = by_id[i]
            if not with_vectors:
                out.append(VectorPoint(i, None, dict(p.payload or {})))
                continue
            dense, sparse = p.vector, None
            if isinstance(dense, dict):  # sparse-enabled collection: named vectors
                sv = dense.get(BM25)
                sparse = SparseVector(list(sv.indices), list(sv.values)) if sv is not None else None
                dense = dense.get(DENSE)
            out.append(VectorPoint(i, list(dense or []), dict(p.payload or {}), sparse))
        return out

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
