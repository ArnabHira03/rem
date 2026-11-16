# M6: Hybrid Retrieval + Re-embed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `search` combines dense (cosine) and BM25 keyword scores in Qdrant and ranks them with one `score_and_rank` function; `rem reembed` moves every memory to a new embedder with the same IDs and payloads, keeping the old collection until `--finalize`.

**Architecture:** Each memory point gets a second, named sparse vector (`bm25`) computed by fastembed's `Qdrant/bm25` model at write time, so it is stored in the same point as the dense vector (Mem0's layout). `Memory.search` runs a dense query and a sparse query over the same collection, fuses them into one hybrid score per memory, and passes the candidates to `retrieve.score_and_rank` (the renamed `rank`). `rem reembed` is a store-side job that scrolls the old collection's payloads and writes the same points into a new collection with the new dense vector. Clients still never touch storage; the CLI goes through the daemon.

**Tech Stack:** Python 3.14, `qdrant-client` 1.20 (named sparse vectors, `query_points(using=...)`), `fastembed` 0.9.1 (`SparseTextEmbedding("Qdrant/bm25")`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-rem-design.md` (Storage: vector retrieval; Embeddings and re-embed; ranking, decision 5; M6 row; Verification: re-embed). `CLAUDE.md` (invariants). `plans_completed/m5.md` for the backlog state.

## Global Constraints

- Python ≥ 3.14; **no new runtime dependency** (fastembed already provides the BM25 model).
- Only `Memory` writes the stores; `retrieve` never writes; write order stays Qdrant upsert → history `INSERT OR IGNORE` → job done.
- Memory ids stay deterministic (`uuid5(job_id, normalized_text_hash)`); `reembed` never changes ids or payloads.
- Clients never touch storage: the CLI and MCP shim use `RemClient` only.
- Unix socket paths stay ≤ 100 bytes: tests use the `short_dir` / `daemon_env` fixtures.
- Dense search keeps its existing semantics: `related()` (used by extraction dedup) stays dense-only and unchanged.
- `REEMBED` is a valid `HistoryEvent`; one history row per reembed run, not per memory (spec).
- Ranking order and thresholds from spec decision 5 are preserved: current project +0.15, other projects −0.05 with the stricter threshold, unverified −0.05, recency tiebreak ≤ 0.02. Thresholds apply to the **dense** cosine, as today.
- Paid or local evals are pre-approved; no M6 task needs an LLM.

## Decisions taken in this plan (review these)

1. **Fusion:** `hybrid = α·dense + (1−α)·bm25_norm`, α = 0.7, where `bm25_norm` is the point's BM25 score divided by the best BM25 score in the candidate set (0 when it has none). Reason: BM25 scores have no fixed scale, so they cannot be compared to cosine directly.
2. **Thresholds still gate on dense cosine.** A keyword-only hit with dense cosine below `T_BASE` is dropped. This keeps the spec's "no unrelated results" behaviour; keyword search only reorders and rescues near-misses above the floor. Tunable in one constant (`KEYWORD_FLOOR`) if the evals show it is too strict.
3. **Collection layout:** the existing collection `rem_memories__<embed-slug>` is created with a sparse slot `bm25` from M6 on. A collection without the slot (created by M1–M5) is detected at startup; the daemon refuses with "run `rem reembed`" (same path as the embedder mismatch). No silent migration.
4. **Reembed runs inside the daemon** (it needs the Qdrant lock), as one job in the store thread's batches of 64 points. The daemon keeps serving on the active embedder while it runs. `rem reembed --model X` records a target in `meta` (`reembed_target`); on completion it sets `active_embedder` to the new slug, clears the target, and asks the user to set `[embedder]` in config and restart. Finalize deletes the old collection.

## Review Focus

1. **Crash in the middle of a reembed** (daemon killed after half the points are written): rerunning `rem reembed` produces exactly one point per id, no duplicates, and the same payloads. Test in Task 6.
2. **Empty or punctuation-only query** (`"?"`, `"   "`): BM25 yields no sparse hits and dense search still runs; no error, no crash on an empty sparse vector. Test in Task 4.
3. **Keyword-only match with unrelated meaning** (shares one common word, dense cosine far below the floor): it must not appear in results. Test in Task 4.
4. **Collection created by M5** (no `bm25` slot) opened by M6: the daemon refuses to start with a message naming `rem reembed`, and does not write anything. Test in Task 5.
5. **Reembed to the same model** (`--model` equals active): refused with a clear message, no collection created. Test in Task 6.
6. **Mixed-case and punctuation text** in BM25 (`"Tabs"` vs `"tabs."`): the same memory matches. Test in Task 2.

---

### Task 1: Sparse vector support in the VectorStore interface

**Files:**
- Modify: `src/rem/storage/vector/base.py` (add `SparseVector`, `VectorStore.create_col(..., sparse=...)`, `keyword_search` signature)
- Modify: `src/rem/storage/vector/qdrant.py`
- Modify: `src/rem/storage/vector/memory.py`
- Test: `tests/test_vector_contract.py` (extend the shared contract)

**Interfaces:**
- Consumes: existing `VectorStore` ABC and contract fixture `store` (params `memory`, `qdrant`, `qdrant-path`).
- Produces:
  - `@dataclass SparseVector: indices: list[int]; values: list[float]`
  - `VectorPoint.sparse: SparseVector | None = None` (new optional field, default keeps all existing callers valid)
  - `VectorStore.create_col(name, vector_size, *, sparse: bool = False)`; a sparse-enabled collection has a named sparse slot `"bm25"`.
  - `VectorStore.col_has_sparse(name: str) -> bool`
  - `VectorStore.keyword_search(col, query_sparse: SparseVector, limit, filters=None) -> list[ScoredPoint] | None` (replaces the M1 stub signature; takes a sparse vector, not text, so the store does not embed)
  - `VectorStore.get(...)` gains `with_vectors: bool = False`; returns `VectorPoint.vector` and `sparse` when true (needed by reembed).

- [ ] **Step 1: Write the failing contract tests**

Add to `tests/test_vector_contract.py`, parametrized through the existing `store` fixture:

```python
def test_sparse_collection_reports_slot_and_keyword_search_ranks_exact_terms(store):
    store.create_col("c", 2, sparse=True)
    assert store.col_has_sparse("c")
    store.upsert("c", [
        VectorPoint(ID1, [1.0, 0.0], {"scope": "global"}, sparse=SparseVector([7], [1.0])),
        VectorPoint(ID2, [0.0, 1.0], {"scope": "global"}, sparse=SparseVector([9], [1.0])),
    ])
    hits = store.keyword_search("c", SparseVector([9], [1.0]), 5)
    assert [h.id for h in hits] == [ID2]          # ID1 shares no term: not returned

def test_dense_only_collection_has_no_keyword_slot(store):
    store.create_col("d", 2)
    assert not store.col_has_sparse("d")
    assert store.keyword_search("d", SparseVector([1], [1.0]), 5) is None

def test_get_with_vectors_returns_both_vectors(store):
    store.create_col("c", 2, sparse=True)
    store.upsert("c", [VectorPoint(ID1, [1.0, 0.0], {"k": 1}, sparse=SparseVector([3], [0.5]))])
    (p,) = store.get("c", [ID1], with_vectors=True)
    assert p.vector == [1.0, 0.0] and p.sparse == SparseVector([3], [0.5])
```

(`store` is created with `create_col` on the sparse collection; `InMemoryVectorStore` must match.)

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_vector_contract.py -k "sparse or keyword or with_vectors" -v`
Expected: FAIL (`SparseVector` not importable / unexpected keyword `sparse`).

- [ ] **Step 3: Implement in `base.py`, `qdrant.py`, `memory.py`**

- Qdrant: `create_collection(name, vectors_config=VectorParams(...), sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)})` when `sparse=True`. `col_has_sparse` checks `get_collection(name).config.params.sparse_vectors` contains `"bm25"`.
- Qdrant upsert: when `p.sparse` is set, the point's vector is `{"": dense, "bm25": SparseVector(indices, values)}`, using the unnamed dense vector as default (`{"": ...}` is qdrant's default name for a single unnamed vector only when the collection has no names; use the explicit dict form `{DENSE_NAME: dense, "bm25": sparse}` with `DENSE_NAME = "dense"` for sparse collections, and the plain list for dense-only collections). Apply the same rule in `search` (`using=DENSE_NAME` for sparse collections).
- Qdrant `keyword_search`: `query_points(col, query=qm.SparseVector(...), using="bm25", limit, query_filter=..., with_payload=True)`. Return `ScoredPoint`s; the score is the raw BM25 score.
- In-memory: store `sparse` on the point; `keyword_search` scores with the same formula Qdrant uses (sum over shared indices of query value × point value, with the IDF modifier computed over the collection's sparse points: `idf(t) = ln(1 + (N − df + 0.5)/(df + 0.5))`). The test only checks ordering, so the exact formula need only be consistent between the two backends' results on the test data.
- `get(..., with_vectors)`: the Qdrant call uses `with_vectors=True` only when requested.

- [ ] **Step 4: Run the contract tests to verify they pass**

Run: `uv run pytest tests/test_vector_contract.py -v`
Expected: all pass for `memory`, `qdrant`, `qdrant-path`.

- [ ] **Step 5: Commit**

```bash
git add src/rem/storage/vector tests/test_vector_contract.py
git commit -m "feat: sparse BM25 slot and keyword_search in the vector store contract"
```

---

### Task 2: BM25 sparse embedder

**Files:**
- Create: `src/rem/embed/sparse.py`
- Modify: `src/rem/embed/factory.py` (add `create_sparse_embedder()`)
- Test: `tests/test_embed.py` (extend)

**Interfaces:**
- Consumes: `SparseVector` (Task 1).
- Produces:
  - `class SparseEmbedder(ABC)`: `embed_docs(texts: list[str]) -> list[SparseVector]`, `embed_query(text: str) -> SparseVector`.
  - `class FastEmbedBM25(SparseEmbedder)`: wraps `fastembed.SparseTextEmbedding("Qdrant/bm25")`, cache under `cfg.embedder.cache_dir or DEFAULT_CACHE_DIR`.
  - `class HashingBM25(SparseEmbedder)`: tokens hashed to indices (used with `[embedder] provider = "hashing"` so tests stay offline).
  - `def create_sparse_embedder(cfg: EmbedderConfig) -> SparseEmbedder`.

- [ ] **Step 1: Write the failing tests** (`tests/test_embed.py`)

```python
def test_hashing_bm25_is_case_and_punctuation_insensitive():
    e = HashingBM25()
    assert e.embed_query("Tabs!") == e.embed_query("tabs")

def test_hashing_bm25_empty_query_is_empty_vector():
    v = HashingBM25().embed_query("?  ")
    assert v.indices == [] and v.values == []
```

A slow test (`@pytest.mark.slow`) checks `FastEmbedBM25` returns non-empty indices for `"python tabs"` and that `"Tabs"` and `"tabs."` produce equal vectors.

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest tests/test_embed.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement `src/rem/embed/sparse.py`**

`FastEmbedBM25` uses `SparseTextEmbedding(model_name="Qdrant/bm25", cache_dir=...)`; `embed_docs` returns `SparseVector(list(v.indices), list(v.values))` for each doc. Wrap load failures in `EmbedderUnavailableError`, as `FastEmbedEmbedder` does. `HashingBM25` tokenizes with `re.findall(r"[a-z0-9]+", text.lower())`, maps each token to `zlib.crc32(token) % 2**20`, counts occurrences, and uses the count as the value; the query uses the same mapping.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_embed.py -v` (and `uv run pytest -m slow tests/test_embed.py -k bm25` once, to confirm the real model; it downloads the Qdrant/bm25 model, which is small).
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rem/embed tests/test_embed.py
git commit -m "feat: BM25 sparse embedder (fastembed Qdrant/bm25, hashing for tests)"
```

---

### Task 3: Write the sparse vector with every memory

**Files:**
- Modify: `src/rem/memory.py` (constructor, `add`, `update`, and anywhere `VectorPoint(...)` is built for memories)
- Test: `tests/test_memory.py` (extend)

**Interfaces:**
- Consumes: `create_sparse_embedder` (Task 2); `VectorStore.create_col(..., sparse=True)` (Task 1).
- Produces: `Memory.sparse_embedder: SparseEmbedder`; `Memory.col` collections created with `sparse=True`; `Memory.add`/`update` store `sparse` on the point. `Memory.__init__` takes `sparse_embedder: SparseEmbedder | None = None` (tests inject `HashingBM25()`).

- [ ] **Step 1: Write the failing test**

```python
def test_add_stores_bm25_slot(memory_factory):
    mem = memory_factory()
    rec = mem.add("uses Qdrant for storage", infer=False, actor="cli")
    (p,) = mem.vectors.get(mem.col, [rec.id], with_vectors=True)
    assert p.sparse is not None and len(p.sparse.indices) > 0
```

Use the existing memory fixture in `tests/test_memory.py` (check its name when implementing; it builds `Memory` with the hashing embedder and in-memory vectors).

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_memory.py::test_add_stores_bm25_slot -v`
Expected: FAIL (`sparse` is None).

- [ ] **Step 3: Implement**

In `Memory.__init__`, create the collection with `sparse=True`. In `add` and `update`, compute `self.sparse_embedder.embed_docs([text])[0]` next to the dense vector and pass `sparse=` into `VectorPoint`. The `update` path re-embeds both vectors.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_memory.py -v`
Expected: PASS (existing tests still pass).

- [ ] **Step 5: Commit**

```bash
git add src/rem/memory.py tests/test_memory.py
git commit -m "feat: store a BM25 sparse vector with each memory"
```

---

### Task 4: Hybrid search and `score_and_rank`

**Files:**
- Modify: `src/rem/retrieve.py` (rename `rank` → `score_and_rank`, add `fuse` and `KEYWORD_FLOOR`, `ALPHA_DENSE = 0.7`)
- Modify: `src/rem/memory.py` (`search` only; `related` untouched; `rank` callers updated)
- Test: `tests/test_retrieve.py`, `tests/test_search.py`

**Interfaces:**
- Consumes: `VectorStore.keyword_search` (Task 1), `SparseEmbedder.embed_query` (Task 2).
- Produces:
  - `retrieve.fuse(dense: list[tuple[MemoryRecord, float]], keyword: dict[str, float]) -> list[tuple[MemoryRecord, float, float]]` returns `(record, hybrid, dense)`; a memory missing from `dense` is dropped if its dense score is below `KEYWORD_FLOOR` (the rule from decision 2). Note: a keyword hit that is not in the dense candidate list has no dense score; it is fetched through `vectors.get(..., with_vectors=True)` and scored by cosine against the query dense vector, so the floor applies to a real number.
  - `retrieve.score_and_rank(candidates: list[tuple[MemoryRecord, float, float]], current_project_id, limit, now) -> list[SearchResult]`: the old `rank` logic, where the threshold check uses the third element (dense) and the ranking score starts from the hybrid value. `SearchResult.semantic` keeps the dense value.
  - `KEYWORD_FLOOR = T_BASE` (0.30), `ALPHA_DENSE = 0.7`.

- [ ] **Step 1: Write the failing tests** (`tests/test_retrieve.py`)

```python
def test_keyword_only_hit_below_floor_is_dropped():
    rec = make_record(text="the build uses tabs")  # helper in tests
    out = score_and_rank([(rec, 0.05, 0.05)], None, 10, NOW)   # dense below floor
    assert out == []

def test_keyword_match_lifts_a_near_miss_above_others():
    a, b = make_record(text="a"), make_record(text="b")
    fused = fuse([(a, 0.40), (b, 0.42)], {a.id: 1.0, b.id: 0.0})
    ranked = score_and_rank(fused, None, 10, NOW)
    assert ranked[0].record.id == a.id
```

And in `tests/test_search.py`, an end-to-end check with `HashingBM25` and `HashingEmbedder`: add `"the build uses tabs"` and `"the build uses spaces"`; search `"tabs"` returns the tabs memory first.

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest tests/test_retrieve.py tests/test_search.py -v`
Expected: FAIL (`score_and_rank`/`fuse` not defined).

- [ ] **Step 3: Implement**

`fuse`: normalize keyword scores by their max (0 if none); `hybrid = ALPHA_DENSE * dense + (1 − ALPHA_DENSE) * kw_norm`; drop entries whose dense is below `KEYWORD_FLOOR` only when they came from keyword-only (memories that were in dense results are already gated by `T_BASE`/`T_OTHER` in `score_and_rank`).
`Memory.search`: run `vectors.search` (dense) and `vectors.keyword_search(col, sparse_query, overfetch(limit), filters)` (skip the sparse call when the query vector is empty); fetch keyword-only ids with `vectors.get(col, ids, with_vectors=True)` and score their dense value as `dot(normalized query, normalized point)`; then `fuse` and `score_and_rank`. `search` still never writes.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_retrieve.py tests/test_search.py tests/test_memory.py -v`
Expected: PASS. Also run the eval-free check `uv run pytest -q` to confirm no other caller of `rank` broke (`grep -rn "rank(" src/rem tests`).

- [ ] **Step 5: Commit**

```bash
git add src/rem/retrieve.py src/rem/memory.py tests/test_retrieve.py tests/test_search.py
git commit -m "feat: hybrid dense+BM25 search with score_and_rank"
```

---

### Task 5: Refuse a collection without the BM25 slot

**Files:**
- Modify: `src/rem/memory.py` (`__init__`, after `create_col`)
- Modify: `src/rem/errors.py` (add `CollectionMigrationError(RemError)` if no suitable base exists; reuse the existing mismatch error type if one fits, and check `errors.py` first)
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: `VectorStore.col_has_sparse` (Task 1).
- Produces: `Memory.__init__` raises an error whose message is `"<col> was created before BM25 search; run `rem reembed --model <active>`"` when the collection exists and lacks the slot. Nothing is written in that path (the `set_meta` call happens after the check).

- [ ] **Step 1: Write the failing test**

```python
def test_dense_only_collection_is_refused_before_writing(tmp_path):
    vs = InMemoryVectorStore()
    vs.create_col(f"rem_memories__{slug}", DIMS)     # what M5 created
    with pytest.raises(EmbedderMismatchError, match="rem reembed"):
        Memory(cfg, vector_store=vs, embedder=HashingEmbedder(DIMS))
    assert cfg_db_meta_active_embedder_unset()        # nothing written
```

(`slug` is `collection_slug(...)` of the test embedder; use the existing test helpers in `tests/test_memory.py` for the config.)

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_memory.py -k dense_only -v`
Expected: FAIL (no error raised).

- [ ] **Step 3: Implement**

In `Memory.__init__`, before `self.vectors.create_col(self.col, ...)`, check `self.vectors.col_exists(self.col) and not self.vectors.col_has_sparse(self.col)` and raise `EmbedderMismatchError` (or the dedicated error) with the message above. Move the `db.set_meta` call after this check, so a refused start writes nothing.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_memory.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rem/memory.py src/rem/errors.py tests/test_memory.py
git commit -m "fix: refuse to open a dense-only collection, pointing at rem reembed"
```

---

### Task 6: `rem reembed` (daemon job, same IDs, crash-safe)

**Files:**
- Create: `src/rem/reembed.py` (the job: `reembed_collection(...)`, batching, progress, finalize)
- Modify: `src/rem/memory.py` (`Memory.reembed_status()`, `Memory.finalize_reembed()`; the `reembed_target` meta key)
- Modify: `src/rem/daemon/app.py` and `src/rem/daemon/service.py` (`POST /reembed` with `{"model": str}`, `GET /reembed`, `POST /reembed/finalize`)
- Modify: `src/rem/client.py` (`reembed(model)`, `reembed_status()`, `reembed_finalize()`)
- Modify: `src/rem/cli.py` (`rem reembed --model X`, `rem reembed --status`, `rem reembed --finalize`)
- Modify: `src/rem/daemon/schemas.py` (request model with `model: str`)
- Test: `tests/test_reembed.py` (new)

**Interfaces:**
- Consumes: `Memory.vectors`, `Memory.embedder`, `VectorStore.list_points(col, limit=None)` (already returns every point), `VectorStore.get(..., with_vectors=True)`, `VectorStore.upsert`, `RemDB.add_history` (`HistoryEvent.REEMBED`, `TargetKind.MEMORY`).
- Produces:
  - `reembed.build_embedders(cfg: RemConfig, model: str) -> tuple[Embedder, SparseEmbedder]` (the new dense embedder built from `cfg.embedder` with `model` substituted).
  - `reembed.run(mem: Memory, target: Embedder, batch: int = 64) -> ReembedResult` where `ReembedResult(collection: str, count: int)`. Steps: create `rem_memories__<target-slug>` with `sparse=True`; `list_points(old col)`; for each batch, embed `payload["text"]` with the target dense and sparse embedders; upsert `VectorPoint(id, dense, payload, sparse)` (same id and payload); after the last batch, `db.add_history(TargetKind.MEMORY, target_col, HistoryEvent.REEMBED, old_slug, new_slug, actor="cli", job_id=None)`, and write `meta.reembed_target = new_slug`.
  - `class ReembedError(RemError)` in `rem/errors.py`: raised for a same-model target, a partial target at finalize, or no target recorded.
  - Finalize: `reembed.finalize(mem) -> str` deletes the old collection, sets `active_embedder` to the target slug, deletes `reembed_target`; it refuses if the target is not fully written (`count(new) != count(old)`).
  - Dedupe: the history row's `UNIQUE(job_id, target_id, event)` plus `job_id=None` would collide on a second run, so the REEMBED row uses `job_id = f"reembed:{target_slug}"`.

- [ ] **Step 1: Write the failing tests** (`tests/test_reembed.py`)

```python
def test_reembed_keeps_ids_payloads_and_searchability(mem_factory):
    mem = mem_factory(embedder=HashingEmbedder(64))
    ids = [mem.add(t, infer=False, actor="cli").id for t in ["tabs", "spaces", "qdrant"]]
    before = {p.id: p.payload for p in mem.vectors.list_points(mem.col)}
    res = reembed.run(mem, HashingEmbedder(32), batch=2)
    after = {p.id: p.payload for p in mem.vectors.list_points(res.collection)}
    assert after == before and res.count == 3
    assert mem.db.get_meta("reembed_target") == res.collection.removeprefix("rem_memories__")  # slug recorded

def test_reembed_twice_is_idempotent(mem_factory):
    ...  # run twice with the same target: list_points count stays 3, no duplicate ids

def test_reembed_same_model_is_refused(mem_factory):
    with pytest.raises(ReembedError, match="already"):
        reembed.run(mem, mem.embedder)

def test_finalize_refuses_partial_target_then_deletes_old(mem_factory):
    ...  # delete one point in target, finalize raises; restore, finalize removes old collection
```

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest tests/test_reembed.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** `reembed.py`, the `Memory` helpers, the daemon routes, client methods and CLI.

- Run the job on the store thread in batches (each batch is one `StoreThread` call), so the API keeps serving between batches.
- The CLI `rem reembed --model X` prints: the new collection, the count, and "set `[embedder] model` to X in config and restart the daemon to use it".
- Reembed ignores points whose dense vectors are already in the target (idempotent on rerun because upserts overwrite the same ids).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_reembed.py tests/test_server.py tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rem/reembed.py src/rem/memory.py src/rem/daemon src/rem/client.py src/rem/cli.py tests/test_reembed.py
git commit -m "feat: rem reembed (same ids and payloads, finalize removes the old collection)"
```

---

### Task 7: Verification and documentation

**Files:**
- Modify: `CLAUDE.md` (M6 status, `rem reembed` command, BM25/fusion invariants)
- Modify: `docs/superpowers/specs/2026-11-15-rem-design.md` only if a decision above changes the spec (decision 1–4 are proposed as deviations; record them in the M6 row like M5 did)
- Move: `plans_completed/` gets `m6.md` when done (follow the M5 convention)

- [ ] **Step 1: Full suite**

Run: `uv run pytest`
Expected: all pass (slow and llm excluded).

- [ ] **Step 2: Slow tests**

Run: `uv run pytest -m slow`
Expected: pass; the real `Qdrant/bm25` model downloads once into `~/.cache/rem/fastembed`.

- [ ] **Step 3: End-to-end on a throwaway store**

```bash
# REM_SOCKET must stay ≤ 100 bytes, so use a short directory (the test fixtures use /tmp/rem-*),
# and a config file with [embedder] provider = "hashing" and [vector_store] provider = "qdrant" path set to REM_HOME.
export REM_HOME=<short-dir>/home REM_CONFIG=<short-dir>/rem.toml REM_SOCKET=<short-dir>/s.sock
uv run rem add "the build uses tabs" && uv run rem search tabs
uv run rem reembed --model <other-hashing-model> && uv run rem reembed --finalize
```
Expected: search returns the tabs memory; reembed reports the same count; finalize removes the old collection.

- [ ] **Step 4: Update CLAUDE.md and commit**

```bash
git add CLAUDE.md
git commit -m "docs: M6 status, rem reembed, hybrid search invariants"
```

---

## Self-review notes

- Spec coverage: BM25 in Qdrant (Task 1–3), `score_and_rank` (Task 4), `rem reembed` with same ids/payloads and old collection kept until finalize (Task 6), daemon refusing a mismatched embedder (existing check + Task 5), REEMBED history per run (Task 6).
- Type names used consistently: `SparseVector`, `VectorPoint.sparse`, `keyword_search(col, SparseVector, ...)`, `score_and_rank`, `fuse`, `reembed.run`.
- Open choices the user should confirm: α = 0.7 and the keyword floor (decisions 1–2); reembed running inside the daemon with the active embedder still serving (decision 4); deviations to record in the spec.
