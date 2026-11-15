# M1 — Core Storage & Memory Façade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working in-process memory engine: `rem add/search/list/get/update/delete/confirm/history` from the CLI, with memories stored the Mem0 way (Qdrant local = canonical records, SQLite = history + metadata) and ranked by scope/project/trust.

**Architecture:** A `Memory` façade (like Mem0's `Memory` class) is the only code that writes to either store. Vector storage sits behind a `VectorStore` ABC modelled on Mem0's `VectorStoreBase`, with a Qdrant implementation and an in-memory one for tests. Embedders sit behind an `Embedder` ABC: fastembed by default, plus a dependency-free hashing embedder for tests and offline use. Ranking is a pure function in `retrieve.py`. There is no daemon, LLM or extraction yet (M2/M3); `add` is Mem0's `infer=False` path.

**Tech Stack:** Python 3.14, uv, `qdrant-client` (local mode), `fastembed`, `pydantic` v2, stdlib `sqlite3`, `tomllib`, `argparse`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-11-15-rem-design.md` (milestone M1; sections "Revised storage architecture", "Crash-safe write ordering", "Embeddings", "Graph-ready memory model", "Data model", "Search & ranking" values carried in "Decisions carried over" item 5).

## Global Constraints

- Python `>=3.14`; build backend `uv_build`; package lives in `src/rem/`; console script stays `rem = "rem:main"`.
- Data dir default `~/.local/share/rem/` (Qdrant at `<data>/qdrant/`, SQLite at `<data>/rem.db`); config `~/.config/rem/config.toml`; model cache `~/.cache/rem/fastembed/`. Env `REM_HOME` overrides the data dir, and `REM_CONFIG` the config path (needed for tests).
- **Canonical memory = the Qdrant payload.** SQLite never stores current memory text except as history old/new values.
- Write order for every mutation: **vector store first → history `INSERT OR IGNORE` second.**
- Memory id = `uuid5(REM_NAMESPACE, f"{job_id or 'direct'}:{scope}:{project_id or ''}:{hash}")`; hash = sha256 hex of the normalized text (lowercase, whitespace collapsed, stripped).
- Payload fields (all present on every memory, `None`/`[]` when empty): `id, text, hash, scope, project_id, project_name, provenance, status, evidence_source, observed_at, source, session_id, job_id, entity_ids, created_at, updated_at`. Timestamps are ISO-8601 UTC strings.
- Enums: scope ∈ `global|system|project`; provenance ∈ `user_asserted|assistant_generated|tool_observed|user_confirmed`; status ∈ `verified|unverified`; history event ∈ `ADD|UPDATE|DELETE|CONFIRM|CORROBORATE|MERGE|REEMBED`; history target_kind ∈ `memory|entity|relation`.
- Collection name: `rem_memories__<slug>_<dims>`, where slug = the embedder model id lowercased with each run of non-alphanumerics replaced by `-` and `-` trimmed from both ends (for example `BAAI/bge-small-en-v1.5` → `baai-bge-small-en-v1-5`).
- Ranking constants: thresholds `0.30` (global/system/current project) and `0.55` (other projects); boosts `+0.15` current project, `−0.05` other project, `−0.05` unverified; recency `0.02 * max(0, 1 - age_days/365)`; over-fetch `max(limit*4, 40)`.
- Qdrant local mode: **do not create payload indexes** (no effect locally, emits a warning). A second process opening the same path raises `RuntimeError`, which must surface as `StoreLockedError`.
- `rem add` from the CLI stores `provenance=user_confirmed, status=verified` (spec: "CLI `rem add` → user_confirmed").

## Review Focus

1. **Adding the same text twice** (same scope/project) must return the existing memory, with no second point and no second ADD history row. Test in Task 8.
2. **Empty or whitespace-only text** for add/update/search must raise `ValueError("text must not be empty")`, and the CLI must print it and exit with code 2. Test in Tasks 8 and 10.
3. **Unknown memory id** for get/update/delete/confirm/history: `get` returns `None`; the others raise `MemoryNotFound`; the CLI prints `no memory with id <id>` and exits 1. Test in Tasks 8 and 10.
4. **Store already open by another process** must give a `StoreLockedError` whose message names the path and says another rem process is using it, not a traceback. Test in Tasks 4 and 10.
5. **First run with no network** (fastembed can't download its model) must raise `EmbedderUnavailableError` naming the model and cache dir. Test in Task 5 (simulated).

---

## File Structure

```
src/rem/
├── __init__.py              exports main (from cli)
├── errors.py                RemError + MemoryNotFound, StoreLockedError, EmbedderMismatchError, EmbedderUnavailableError
├── config.py                RemConfig (pydantic) + load_config()
├── models.py                enums, MemoryRecord, SearchResult, HistoryRow, normalize_text, text_hash, memory_id, utcnow
├── projects.py              resolve_project(cwd) -> ProjectInfo (git root / cwd fallback, remote url)
├── retrieve.py              rank(...) pure ranking function + constants
├── memory.py                Memory façade (only writer)
├── cli.py                   argparse CLI, main()
├── embed/
│   ├── base.py              Embedder ABC, collection_slug()
│   ├── hashing.py           HashingEmbedder (deterministic, offline)
│   ├── fastembed.py         FastEmbedEmbedder
│   └── factory.py           create_embedder(cfg)
└── storage/
    ├── sqlite.py            RemDB: meta, projects, history; migrations
    └── vector/
        ├── base.py          VectorStore ABC, VectorPoint, ScoredPoint, Filters
        ├── memory.py        InMemoryVectorStore
        ├── qdrant.py        QdrantVectorStore
        └── factory.py       create_vector_store(cfg)
tests/
├── conftest.py              tmp REM_HOME, hashing-embedder config, memory fixture
├── test_config.py · test_models.py · test_vector_contract.py · test_qdrant.py
├── test_embed.py · test_sqlite.py · test_projects.py · test_retrieve.py
├── test_memory.py · test_search.py · test_cli.py
```

---

### Task 1: Package skeleton, config, test tooling

**Files:**
- Modify: `pyproject.toml`
- Create: `src/rem/__init__.py`, `src/rem/errors.py`, `src/rem/config.py`, `tests/conftest.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `rem.errors`: `RemError(Exception)`, `MemoryNotFound(RemError)`, `StoreLockedError(RemError)`, `EmbedderMismatchError(RemError)`, `EmbedderUnavailableError(RemError)`.
  - `rem.config`:
    - `VectorStoreConfig(provider: Literal["qdrant","memory"]="qdrant", path: Path | None = None, url: str | None = None)`
    - `EmbedderConfig(provider: Literal["fastembed","hashing"]="fastembed", model: str = "BAAI/bge-small-en-v1.5", cache_dir: Path | None = None)`
    - `RemConfig(data_dir: Path, vector_store: VectorStoreConfig, embedder: EmbedderConfig)` with properties `db_path -> data_dir/"rem.db"` and `qdrant_path -> vector_store.path or data_dir/"qdrant"`
    - `load_config(path: Path | None = None) -> RemConfig`. It reads TOML if the file exists; `data_dir` comes from `REM_HOME`, else the TOML `data_dir`, else `~/.local/share/rem`; the default `cache_dir` is `~/.cache/rem/fastembed`; the config path is `REM_CONFIG` if no argument is given.
  - `rem.main`: the placeholder `main()` in `__init__.py` prints `rem: no command yet` until Task 10.

- [ ] **Step 1: Add dependencies and pytest config**

```bash
uv add qdrant-client fastembed pydantic
uv add --dev pytest
```
Then add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: downloads the real embedding model"]
addopts = "-m 'not slow'"
```

- [ ] **Step 2: Write failing tests** in `tests/test_config.py`

```python
def test_defaults_use_rem_home(tmp_path, monkeypatch):
    monkeypatch.setenv("REM_HOME", str(tmp_path))
    monkeypatch.delenv("REM_CONFIG", raising=False)
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.data_dir == tmp_path
    assert cfg.db_path == tmp_path / "rem.db"
    assert cfg.qdrant_path == tmp_path / "qdrant"
    assert cfg.vector_store.provider == "qdrant"
    assert cfg.embedder.model == "BAAI/bge-small-en-v1.5"

def test_toml_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("REM_HOME", str(tmp_path))
    p = tmp_path / "config.toml"
    p.write_text('[embedder]\nprovider = "hashing"\nmodel = "hash-64"\n[vector_store]\nprovider = "memory"\n')
    cfg = load_config(p)
    assert cfg.embedder.provider == "hashing" and cfg.embedder.model == "hash-64"
    assert cfg.vector_store.provider == "memory"

def test_invalid_provider_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("REM_HOME", str(tmp_path))
    p = tmp_path / "config.toml"; p.write_text('[vector_store]\nprovider = "faiss"\n')
    with pytest.raises(ValueError):
        load_config(p)
```

- [ ] **Step 3: Run** `uv run pytest tests/test_config.py -v`. Expected: FAIL (`ModuleNotFoundError: rem.config`).
- [ ] **Step 4: Implement** `errors.py`, `config.py`, the `__init__.py` placeholder `main`, and `tests/conftest.py` with:
  - a `rem_home` fixture (sets `REM_HOME` to `tmp_path`, unsets `REM_CONFIG`)
  - a `cfg` fixture returning `RemConfig` with `provider="memory"` and `EmbedderConfig(provider="hashing", model="hash-64")`
- [ ] **Step 5: Run** `uv run pytest tests/test_config.py -v` and `uv run rem`. Expected: 3 passed; the CLI prints `rem: no command yet`.
- [ ] **Step 6: Commit** `git add pyproject.toml uv.lock src tests && git commit -m "feat: package skeleton, config and test tooling"`

---

### Task 2: Memory model, ids and hashing

**Files:**
- Create: `src/rem/models.py`, `tests/test_models.py`

**Interfaces:**
- Produces (`rem.models`):
  - `Scope`, `Provenance`, `Status`, `HistoryEvent`, `TargetKind`: `StrEnum`s with the Global Constraints values.
  - `REM_NAMESPACE = uuid.UUID("6f1c8e5a-3b7d-4c2e-9a10-5d4e3f2a1b00")`.
  - `normalize_text(text: str) -> str`; raises `ValueError("text must not be empty")` on empty or whitespace-only text.
  - `text_hash(text: str) -> str`: sha256 hex of `normalize_text(text)`.
  - `memory_id(text: str, scope: Scope, project_id: int | None, job_id: str | None = None) -> str`.
  - `utcnow() -> str`: ISO-8601 UTC with `+00:00`.
  - `MemoryRecord` (pydantic) with exactly the payload fields from Global Constraints (`entity_ids: list[str] = []`), plus `to_payload() -> dict` and `@classmethod from_payload(payload: dict) -> MemoryRecord`.
  - `SearchResult(record: MemoryRecord, score: float, semantic: float)`.
  - `HistoryRow(id: int, target_kind: TargetKind, target_id: str, event: HistoryEvent, old_value: str | None, new_value: str | None, actor: str, job_id: str | None, created_at: str)`.

- [ ] **Step 1: Write failing tests**

```python
def test_normalize_and_hash_ignore_case_and_spacing():
    assert normalize_text("  Prefers   UV\n") == "prefers uv"
    assert text_hash("Prefers uv") == text_hash(" prefers  UV ")

def test_empty_text_rejected():
    with pytest.raises(ValueError, match="text must not be empty"):
        normalize_text("   ")

def test_memory_id_deterministic_and_scoped():
    a = memory_id("Prefers uv", Scope.GLOBAL, None)
    assert a == memory_id("prefers  UV", Scope.GLOBAL, None)
    assert a != memory_id("Prefers uv", Scope.PROJECT, 1)
    assert a != memory_id("Prefers uv", Scope.GLOBAL, None, job_id="job-1")
    uuid.UUID(a)  # valid uuid string

def test_payload_roundtrip_keeps_all_fields():
    rec = MemoryRecord(id="x", text="t", hash="h", scope=Scope.GLOBAL, project_id=None, project_name=None,
        provenance=Provenance.USER_CONFIRMED, status=Status.VERIFIED, evidence_source=None, observed_at=None,
        source="cli", session_id=None, job_id=None, created_at=utcnow(), updated_at=utcnow())
    p = rec.to_payload()
    assert set(p) == {"id","text","hash","scope","project_id","project_name","provenance","status",
        "evidence_source","observed_at","source","session_id","job_id","entity_ids","created_at","updated_at"}
    assert p["entity_ids"] == [] and p["scope"] == "global"
    assert MemoryRecord.from_payload(p) == rec
```

- [ ] **Step 2: Run** `uv run pytest tests/test_models.py -v`. Expected: FAIL (import error).
- [ ] **Step 3: Implement `rem/models.py`.** Enums serialize to their string values in `to_payload()` (`model_dump(mode="json")`).
- [ ] **Step 4: Run** `uv run pytest tests/test_models.py -v`. Expected: 4 passed.
- [ ] **Step 5: Commit** `git add src/rem/models.py tests/test_models.py && git commit -m "feat: memory record model, deterministic ids"`

---

### Task 3: VectorStore interface + in-memory store + contract tests

**Files:**
- Create: `src/rem/storage/__init__.py`, `src/rem/storage/vector/__init__.py`, `src/rem/storage/vector/base.py`, `src/rem/storage/vector/memory.py`, `tests/test_vector_contract.py`

**Interfaces:**
- Produces (`rem.storage.vector.base`):
  - `Filters = dict[str, str | int | None | list[str | int]]`. A scalar means equality and a list means match-any; all keys are ANDed.
  - `@dataclass VectorPoint(id: str, vector: list[float] | None, payload: dict)`
  - `@dataclass ScoredPoint(id: str, score: float, payload: dict)` (score = cosine similarity)
  - `class VectorStore(ABC)`:
    - `create_col(name: str, vector_size: int) -> None` (idempotent)
    - `col_exists(name) -> bool`
    - `delete_col(name) -> None`
    - `upsert(col: str, points: list[VectorPoint]) -> None`
    - `search(col, vector: list[float], limit: int, filters: Filters | None = None) -> list[ScoredPoint]`, sorted by descending score
    - `get(col, ids: list[str]) -> list[VectorPoint]` (missing ids are omitted)
    - `update_payload(col, id: str, payload: dict) -> None` (merges keys)
    - `delete(col, ids: list[str]) -> None`
    - `list_points(col, filters: Filters | None = None, limit: int = 10_000) -> list[VectorPoint]`
    - `keyword_search(col, query: str, limit: int, filters=None) -> list[ScoredPoint] | None` (the base returns `None`; implemented in M6)
    - `close() -> None`
- `InMemoryVectorStore()` in `memory.py`.

- [ ] **Step 1: Write the contract tests.** Parametrize over a `store` fixture so Task 4 can add Qdrant to `params` without rewriting them.

```python
@pytest.fixture(params=["memory"])
def store(request, tmp_path):
    s = make_store(request.param, tmp_path)   # helper in this test module; Task 4 adds "qdrant", "qdrant-path"
    s.create_col("c", 3)
    yield s
    s.close()

def test_upsert_get_and_idempotent_create(store):
    store.create_col("c", 3)
    store.upsert("c", [VectorPoint("00000000-0000-0000-0000-000000000001", [1,0,0], {"scope": "global", "text": "a"})])
    [p] = store.get("c", ["00000000-0000-0000-0000-000000000001"])
    assert p.payload["text"] == "a"

def test_search_orders_by_cosine_and_filters(store):
    store.upsert("c", [VectorPoint(ID1, [1,0,0], {"scope": "global"}),
                       VectorPoint(ID2, [0.9,0.1,0], {"scope": "project", "project_id": 7}),
                       VectorPoint(ID3, [0,1,0], {"scope": "system"})])
    assert [r.id for r in store.search("c", [1,0,0], 3)] == [ID1, ID2, ID3]
    assert [r.id for r in store.search("c", [1,0,0], 3, {"scope": ["global", "system"]})] == [ID1, ID3]
    assert [r.id for r in store.search("c", [1,0,0], 3, {"scope": "project", "project_id": 7})] == [ID2]
    assert store.search("c", [1,0,0], 1)[0].score == pytest.approx(1.0, abs=1e-6)

def test_update_payload_merges_and_delete_removes(store):
    store.upsert("c", [VectorPoint(ID1, [1,0,0], {"status": "unverified", "text": "a"})])
    store.update_payload("c", ID1, {"status": "verified"})
    assert store.get("c", [ID1])[0].payload == {"status": "verified", "text": "a"}
    store.delete("c", [ID1])
    assert store.get("c", [ID1]) == [] and store.list_points("c") == []

def test_upsert_same_id_overwrites(store):
    store.upsert("c", [VectorPoint(ID1, [1,0,0], {"text": "a"})])
    store.upsert("c", [VectorPoint(ID1, [1,0,0], {"text": "a"})])
    assert len(store.list_points("c")) == 1

def test_keyword_search_unsupported_returns_none(store):
    assert store.keyword_search("c", "anything", 5) is None
```
(`ID1..ID3` are module-level UUID strings, because Qdrant requires UUID or int ids.)

- [ ] **Step 2: Run** `uv run pytest tests/test_vector_contract.py -v`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** the `VectorStore` ABC and `InMemoryVectorStore` (dict per collection, cosine computed in pure Python; zero vectors score 0).
- [ ] **Step 4: Run** `uv run pytest tests/test_vector_contract.py -v`. Expected: 5 passed.
- [ ] **Step 5: Commit** `git commit -m "feat: VectorStore interface and in-memory store"`

---

### Task 4: Qdrant vector store

**Files:**
- Create: `src/rem/storage/vector/qdrant.py`, `src/rem/storage/vector/factory.py`, `tests/test_qdrant.py`
- Modify: `tests/test_vector_contract.py` (fixture params)

**Interfaces:**
- Consumes: the `VectorStore` ABC and `Filters` (Task 3); `RemConfig` (Task 1); `StoreLockedError` (Task 1).
- Produces:
  - `QdrantVectorStore(path: Path | None = None, url: str | None = None, location: str | None = None)`. Use `location=":memory:"` for tests. Constructor `RuntimeError` mentioning "already accessed" → `StoreLockedError(f"{path} is in use by another rem process (is the daemon running?)")`.
  - `create_vector_store(cfg: RemConfig) -> VectorStore`: `"qdrant"` → `QdrantVectorStore(path=cfg.qdrant_path, url=cfg.vector_store.url)` (url wins when set); `"memory"` → `InMemoryVectorStore()`.

- [ ] **Step 1: Extend the contract fixture** to `params=["memory", "qdrant", "qdrant-path"]`, where `qdrant` = `QdrantVectorStore(location=":memory:")` and `qdrant-path` = `QdrantVectorStore(path=tmp_path/"q")`.
- [ ] **Step 2: Write failing tests** in `tests/test_qdrant.py`:

```python
def test_persists_across_reopen(tmp_path):
    s = QdrantVectorStore(path=tmp_path / "q"); s.create_col("c", 3)
    s.upsert("c", [VectorPoint(ID1, [1,0,0], {"text": "a"})]); s.close()
    s2 = QdrantVectorStore(path=tmp_path / "q")
    assert s2.get("c", [ID1])[0].payload["text"] == "a"; s2.close()

def test_second_open_raises_store_locked(tmp_path):
    s = QdrantVectorStore(path=tmp_path / "q")
    with pytest.raises(StoreLockedError, match="in use by another rem process"):
        QdrantVectorStore(path=tmp_path / "q")
    s.close()

def test_factory_picks_provider(cfg, tmp_path):
    assert isinstance(create_vector_store(cfg), InMemoryVectorStore)
    q = create_vector_store(cfg.model_copy(update={"vector_store": VectorStoreConfig(provider="qdrant", path=tmp_path / "q")}))
    assert isinstance(q, QdrantVectorStore); q.close()
```

- [ ] **Step 3: Run** `uv run pytest tests/test_qdrant.py tests/test_vector_contract.py -v`. Expected: FAIL (import error).
- [ ] **Step 4: Implement `QdrantVectorStore`** with `qdrant_client.QdrantClient`:
  - cosine distance; `query_points` for search; `scroll` for `list_points`; `set_payload` for `update_payload`; `retrieve` for `get`
  - `Filters` → `models.Filter(must=[FieldCondition(MatchValue | MatchAny)])`; a `None` value means `IsNullCondition`
  - no payload indexes
- [ ] **Step 5: Run** `uv run pytest tests/test_qdrant.py tests/test_vector_contract.py -v`. Expected: all pass (15 contract cases + 3).
- [ ] **Step 6: Commit** `git commit -m "feat: Qdrant local vector store behind VectorStore"`

---

### Task 5: Embedders

**Files:**
- Create: `src/rem/embed/__init__.py`, `src/rem/embed/base.py`, `src/rem/embed/hashing.py`, `src/rem/embed/fastembed.py`, `src/rem/embed/factory.py`, `tests/test_embed.py`

**Interfaces:**
- Consumes: `EmbedderConfig`, `EmbedderUnavailableError`.
- Produces:
  - `class Embedder(ABC)`: properties `model_id: str` and `dims: int`; `embed(texts: list[str]) -> list[list[float]]`; `embed_query(text: str) -> list[float]`.
  - `collection_slug(model_id: str, dims: int) -> str`, e.g. `collection_slug("BAAI/bge-small-en-v1.5", 384) == "baai-bge-small-en-v1-5_384"`.
  - `HashingEmbedder(dims: int = 64)`, with `model_id = f"hash-{dims}"`. Each lowercase word token (`\w+`) adds 1.0 at `int(sha1(token).hexdigest(), 16) % dims`; the vector is L2-normalized and the zero vector is returned for no tokens. `embed_query == embed`.
  - `FastEmbedEmbedder(model: str, cache_dir: Path)`. It uses `fastembed.TextEmbedding(model_name=model, cache_dir=str(cache_dir))`, with `query_embed` for queries and `dims` from the first embedding. Any exception while loading the model → `EmbedderUnavailableError(f"could not load embedding model {model} (cache: {cache_dir}); the first run needs network access to download it")`.
  - `create_embedder(cfg: EmbedderConfig) -> Embedder`. `"hashing"` parses dims from a model like `hash-64` (default 64).

- [ ] **Step 1: Write failing tests**

```python
def test_slug():
    assert collection_slug("BAAI/bge-small-en-v1.5", 384) == "baai-bge-small-en-v1-5_384"

def test_hashing_embedder_similarity_tracks_word_overlap():
    e = HashingEmbedder(64)
    a, b, c = e.embed(["prefers uv over pip", "uv over pip preferred", "laptop has 15 GB RAM"])
    cos = lambda x, y: sum(i*j for i, j in zip(x, y))
    assert cos(a, b) > cos(a, c)
    assert len(a) == 64 and e.model_id == "hash-64"

def test_fastembed_unavailable_is_clear(monkeypatch, tmp_path):
    def boom(*a, **k): raise OSError("network down")
    monkeypatch.setattr("rem.embed.fastembed.TextEmbedding", boom)
    with pytest.raises(EmbedderUnavailableError, match="BAAI/bge-small-en-v1.5"):
        FastEmbedEmbedder("BAAI/bge-small-en-v1.5", tmp_path)

@pytest.mark.slow
def test_fastembed_real_model(tmp_path):
    e = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", tmp_path)
    assert e.dims == 384 and len(e.embed_query("which package manager?")) == 384
```

- [ ] **Step 2: Run** `uv run pytest tests/test_embed.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the four modules per the Interfaces.
- [ ] **Step 4: Run** `uv run pytest tests/test_embed.py -v` (3 passed, 1 deselected) and `uv run pytest tests/test_embed.py -m slow -v` (1 passed; downloads about 70 MB once).
- [ ] **Step 5: Commit** `git commit -m "feat: embedder interface, hashing and fastembed embedders"`

---

### Task 6: SQLite metadata, projects and history

**Files:**
- Create: `src/rem/storage/sqlite.py`, `src/rem/projects.py`, `tests/test_sqlite.py`, `tests/test_projects.py`

**Interfaces:**
- Consumes: `HistoryRow`, `HistoryEvent`, `TargetKind`, `utcnow` (Task 2).
- Produces:
  - `RemDB(path: Path)`:
    - WAL mode, `foreign_keys=ON`; creates the parent dir; runs migrations up to `SCHEMA_VERSION = 1`, stored in `meta["schema_version"]`.
    - Tables:
      - `meta(key TEXT PRIMARY KEY, value TEXT)`
      - `projects(id INTEGER PRIMARY KEY, root_path TEXT UNIQUE NOT NULL, name TEXT NOT NULL, git_remote TEXT)`
      - `history(id INTEGER PRIMARY KEY, target_kind TEXT NOT NULL, target_id TEXT NOT NULL, event TEXT NOT NULL, old_value TEXT, new_value TEXT, actor TEXT NOT NULL, job_id TEXT, created_at TEXT NOT NULL, UNIQUE(job_id, target_id, event))`
    - Methods:
      - `get_meta(key) -> str | None`, `set_meta(key, value) -> None`
      - `upsert_project(root_path: str, name: str, git_remote: str | None) -> int`
      - `get_project(id: int) -> tuple[str, str, str | None] | None` (root_path, name, git_remote)
      - `add_history(target_kind, target_id, event, old_value, new_value, actor, job_id=None) -> bool` (False if ignored as a duplicate)
      - `get_history(target_id) -> list[HistoryRow]` (oldest first)
      - `close()`
  - `ProjectInfo(root_path: str, name: str, git_remote: str | None)` and `resolve_project(cwd: Path) -> ProjectInfo`:
    - walks up from `cwd` to the first directory containing `.git`, falling back to `cwd.resolve()`
    - `name` is the directory's basename
    - `git_remote` comes from `git -C <root> remote get-url origin` (any failure → `None`)

- [ ] **Step 1: Write failing tests**

```python
def test_schema_and_meta(tmp_path):
    db = RemDB(tmp_path / "rem.db")
    assert db.get_meta("schema_version") == "1"
    db.set_meta("active_embedder", "hash-64"); assert db.get_meta("active_embedder") == "hash-64"

def test_history_insert_or_ignore_with_job_id(tmp_path):
    db = RemDB(tmp_path / "rem.db")
    assert db.add_history(TargetKind.MEMORY, "m1", HistoryEvent.ADD, None, "a", "extractor", job_id="j1") is True
    assert db.add_history(TargetKind.MEMORY, "m1", HistoryEvent.ADD, None, "a", "extractor", job_id="j1") is False
    db.add_history(TargetKind.MEMORY, "m1", HistoryEvent.UPDATE, "a", "b", "cli")
    assert [h.event for h in db.get_history("m1")] == ["ADD", "UPDATE"]

def test_history_accepts_entity_and_relation_targets(tmp_path):
    db = RemDB(tmp_path / "rem.db")
    db.add_history(TargetKind.ENTITY, "e1", HistoryEvent.MERGE, "uv", "astral uv", "extractor")
    assert db.get_history("e1")[0].target_kind == "entity"

def test_upsert_project_is_stable(tmp_path):
    db = RemDB(tmp_path / "rem.db")
    a = db.upsert_project("/x/rem", "rem", None); b = db.upsert_project("/x/rem", "rem", "git@x:rem.git")
    assert a == b and db.get_project(a) == ("/x/rem", "rem", "git@x:rem.git")
```
`tests/test_projects.py`:
```python
def test_resolves_git_root_from_subdir(tmp_path):
    (tmp_path / "proj" / ".git").mkdir(parents=True); sub = tmp_path / "proj" / "src" / "a"; sub.mkdir(parents=True)
    info = resolve_project(sub)
    assert info.root_path == str((tmp_path / "proj").resolve()) and info.name == "proj"

def test_falls_back_to_cwd_without_git(tmp_path):
    assert resolve_project(tmp_path).root_path == str(tmp_path.resolve())
```

- [ ] **Step 2: Run** `uv run pytest tests/test_sqlite.py tests/test_projects.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `RemDB` and `resolve_project`. Use `check_same_thread=False` (the M2 daemon will own the connection), and run `git` with `timeout=2`.
- [ ] **Step 4: Run** the same command. Expected: 6 passed.
- [ ] **Step 5: Commit** `git commit -m "feat: SQLite meta/projects/history and project resolution"`

---

### Task 7: Ranking (pure function)

**Files:**
- Create: `src/rem/retrieve.py`, `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `MemoryRecord`, `SearchResult`, `Scope`, `Status` (Task 2).
- Produces:
  - Constants `T_BASE = 0.30`, `T_OTHER = 0.55`, `BOOST_CURRENT = 0.15`, `PENALTY_OTHER = 0.05`, `PENALTY_UNVERIFIED = 0.05`, `RECENCY_MAX = 0.02`, and `overfetch(limit: int) -> int` (= `max(limit*4, 40)`).
  - `rank(candidates: list[tuple[MemoryRecord, float]], current_project_id: int | None, limit: int, now: datetime) -> list[SearchResult]`.
    - A project-scope memory counts as "other project" when `current_project_id is None` or its `project_id` differs.
    - The rest follows the Global Constraints formula, sorted by descending `score`. `semantic` is the raw cosine.

- [ ] **Step 1: Write failing tests** (build records with a small helper `rec(scope, project_id=None, status="verified", age_days=0)`)

```python
NOW = datetime(2026, 11, 15, tzinfo=UTC)

def test_current_project_outranks_equal_global_and_other_project():
    out = rank([(rec("global"), 0.60), (rec("project", 1), 0.60), (rec("project", 2), 0.60)], 1, 10, NOW)
    assert [(r.record.scope, r.record.project_id) for r in out] == [("project", 1), ("global", None), ("project", 2)]
    assert out[0].score == pytest.approx(0.60 + 0.15 + 0.02)

def test_other_project_needs_stricter_threshold():
    out = rank([(rec("project", 2), 0.50), (rec("system"), 0.31)], 1, 10, NOW)
    assert [r.record.scope for r in out] == ["system"]

def test_unverified_penalized_and_recency_decays():
    out = rank([(rec("global", status="unverified"), 0.60), (rec("global", age_days=365), 0.60)], None, 10, NOW)
    assert out[0].record.status == "verified"
    assert out[0].score == pytest.approx(0.60) and out[1].score == pytest.approx(0.60 - 0.05 + 0.02)

def test_limit_and_overfetch():
    assert overfetch(5) == 40 and overfetch(20) == 80
    assert len(rank([(rec("global"), 0.9)] * 5, None, 2, NOW)) == 2
```

- [ ] **Step 2: Run** `uv run pytest tests/test_retrieve.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `rank`** exactly per the Global Constraints formula (`age_days` is computed from `created_at`).
- [ ] **Step 4: Run** `uv run pytest tests/test_retrieve.py -v`. Expected: 4 passed.
- [ ] **Step 5: Commit** `git commit -m "feat: scope/project/trust ranking"`

---

### Task 8: Memory façade — writes, reads, history, crash-safe ordering

**Files:**
- Create: `src/rem/memory.py`, `tests/test_memory.py`

**Interfaces:**
- Consumes: `RemConfig`, `create_vector_store`, `create_embedder`, `collection_slug`, `RemDB`, `resolve_project`, the models, the errors.
- Produces:
  - `Memory(cfg: RemConfig, *, vector_store: VectorStore | None = None, embedder: Embedder | None = None)`. Opens `RemDB(cfg.db_path)`.
    - Collection = `f"rem_memories__{collection_slug(e.model_id, e.dims)}"`.
    - If `meta["active_embedder"]` is set and differs from the slug → `EmbedderMismatchError(f"stored memories use {stored}; configured embedder is {slug}. Run `rem reembed` (coming in M6) or change the config back.")`.
    - Otherwise it sets the meta key and `create_col`.
    - Methods:
      - `add(text, *, scope: Scope, cwd: Path | None = None, provenance: Provenance, status: Status, source: str, actor: str, job_id: str | None = None, session_id: str | None = None) -> MemoryRecord`
      - `get(id) -> MemoryRecord | None`
      - `list(*, scope=None, project_id=None, status=None) -> list[MemoryRecord]` (newest first)
      - `update(id, text, *, actor) -> MemoryRecord`
      - `delete(id, *, actor) -> None`
      - `confirm(id, *, actor) -> MemoryRecord`
      - `history(id) -> list[HistoryRow]`
      - `close()`
      - `search(...)` comes in Task 9
- Behaviour:
  - **add:**
    - `project` scope requires `cwd` (`ValueError("project scope needs a working directory")`); `project_id` comes from `resolve_project` + `upsert_project`.
    - The id comes from `memory_id(...)`. **If that id already exists, return the existing record unchanged** (no upsert, no history).
    - Otherwise: upsert the point (vector = `embed([text])[0]`), then `add_history(MEMORY, id, ADD, None, text, actor, job_id)`.
  - **update:**
    - re-embed, then upsert with the same id, new text/hash/`updated_at`
    - then history `UPDATE` with old → new text
    - unknown id → `MemoryNotFound(id)`
  - **delete:** vector `delete` first, then history `DELETE` with old = the text.
  - **confirm:**
    - `update_payload(provenance=user_confirmed, status=verified, updated_at)`, then history `CONFIRM`
    - no-op without history if the memory is already `user_confirmed`
  - All mutations: vector store first, history second.

- [ ] **Step 1: Write failing tests** (fixture `mem` = `Memory(cfg)` using the memory store + hashing embedder from conftest)

```python
def add(mem, text="Prefers uv over pip", **kw):
    kw.setdefault("scope", Scope.GLOBAL)
    return mem.add(text, provenance=Provenance.USER_CONFIRMED, status=Status.VERIFIED, source="cli", actor="cli", **kw)

def test_add_get_and_history(mem):
    r = add(mem)
    assert mem.get(r.id).text == "Prefers uv over pip" and r.entity_ids == []
    assert [h.event for h in mem.history(r.id)] == ["ADD"]

def test_duplicate_add_returns_existing_without_new_history(mem):
    a = add(mem); b = add(mem, "  prefers UV over pip ")
    assert a.id == b.id and len(mem.list()) == 1 and len(mem.history(a.id)) == 1

def test_project_scope_resolves_project(mem, tmp_path):
    (tmp_path / "p" / ".git").mkdir(parents=True)
    r = add(mem, "rem uses Qdrant", scope=Scope.PROJECT, cwd=tmp_path / "p")
    assert r.project_name == "p" and r.project_id is not None
    with pytest.raises(ValueError, match="needs a working directory"):
        add(mem, "x", scope=Scope.PROJECT)

def test_update_confirm_delete_and_history(mem):
    r = mem.add("Uses npm", scope=Scope.GLOBAL, provenance=Provenance.ASSISTANT_GENERATED,
                status=Status.UNVERIFIED, source="mcp", actor="mcp")
    mem.update(r.id, "Uses pnpm", actor="cli")
    c = mem.confirm(r.id, actor="cli")
    assert c.status == "verified" and c.provenance == "user_confirmed" and c.text == "Uses pnpm"
    mem.delete(r.id, actor="cli")
    assert mem.get(r.id) is None
    h = mem.history(r.id)
    assert [x.event for x in h] == ["ADD", "UPDATE", "CONFIRM", "DELETE"]
    assert (h[1].old_value, h[1].new_value) == ("Uses npm", "Uses pnpm") and h[3].old_value == "Uses pnpm"

def test_unknown_id(mem):
    assert mem.get("00000000-0000-0000-0000-000000000000") is None
    for call in (lambda: mem.update("nope", "x", actor="cli"), lambda: mem.delete("nope", actor="cli"),
                 lambda: mem.confirm("nope", actor="cli")):
        with pytest.raises(MemoryNotFound): call()

def test_empty_text_rejected(mem):
    with pytest.raises(ValueError, match="text must not be empty"): add(mem, "   ")

def test_crash_between_vector_write_and_history_replays_cleanly(mem, monkeypatch):
    real = mem.db.add_history; calls = {"n": 0}
    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1: raise RuntimeError("crash")
        return real(*a, **k)
    monkeypatch.setattr(mem.db, "add_history", flaky)
    with pytest.raises(RuntimeError): add(mem, job_id="job-1")
    # replay of the same job: point exists, so add must still complete its history row
    r = add(mem, job_id="job-1")
    assert len(mem.list()) == 1 and [h.event for h in mem.history(r.id)] == ["ADD"]

def test_embedder_mismatch_refuses(cfg):
    m = Memory(cfg); m.close()
    other = cfg.model_copy(update={"embedder": EmbedderConfig(provider="hashing", model="hash-32")})
    with pytest.raises(EmbedderMismatchError, match="rem reembed"):
        Memory(other, vector_store=InMemoryVectorStore())
```
The crash test pins one refinement to the "return existing" rule. **When the id already exists and `job_id` is set**, `add` still calls `add_history` (INSERT OR IGNORE makes this a no-op if the row exists) before returning. That is what completes a replayed job. Direct adds (`job_id=None`) skip history on duplicates.

- [ ] **Step 2: Run** `uv run pytest tests/test_memory.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `Memory`** per the Interfaces and Behaviour.
- [ ] **Step 4: Run** `uv run pytest tests/test_memory.py -v`. Expected: 8 passed.
- [ ] **Step 5: Commit** `git commit -m "feat: Memory façade with crash-safe write ordering"`

---

### Task 9: Memory.search

**Files:**
- Modify: `src/rem/memory.py`
- Create: `tests/test_search.py`

**Interfaces:**
- Consumes: `rank`, `overfetch` (Task 7); `Memory` (Task 8).
- Produces: `Memory.search(query: str, *, cwd: Path | None = None, limit: int = 10, scope: Scope | None = None) -> list[SearchResult]`.
  - Empty query → `ValueError("text must not be empty")`.
  - The current project comes from `resolve_project(cwd)` only if that root is already in `projects` (search never creates projects).
  - It calls `vector_store.search(col, embedder.embed_query(query), overfetch(limit), {"scope": scope} if scope else None)` and passes `(record, score)` pairs to `rank(..., now=datetime.now(UTC))`.

- [ ] **Step 1: Write failing tests**

```python
def test_current_project_ranks_first(mem, tmp_path):
    for p in ("a", "b"): (tmp_path / p / ".git").mkdir(parents=True)
    add(mem, "database is SQLite", scope=Scope.PROJECT, cwd=tmp_path / "a")
    add(mem, "database is Postgres", scope=Scope.PROJECT, cwd=tmp_path / "b")
    out = mem.search("which database", cwd=tmp_path / "a")
    assert out and out[0].record.project_name == "a"

def test_scope_filter_and_labels(mem):
    add(mem, "Prefers uv over pip"); add(mem, "OS is Fedora 44", scope=Scope.SYSTEM)
    out = mem.search("Fedora OS", scope=Scope.SYSTEM)
    assert [r.record.scope for r in out] == ["system"] and out[0].record.provenance == "user_confirmed"

def test_empty_query_rejected(mem):
    with pytest.raises(ValueError, match="text must not be empty"): mem.search("  ")
```
(`add` helper shared via `tests/conftest.py`: move it there in this task.)

- [ ] **Step 2: Run** `uv run pytest tests/test_search.py -v`. Expected: FAIL (`AttributeError: search`).
- [ ] **Step 3: Implement `Memory.search`.**
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: the whole suite passes.
- [ ] **Step 5: Commit** `git commit -m "feat: memory search with scope ranking"`

---

### Task 10: CLI

**Files:**
- Create: `src/rem/cli.py`, `tests/test_cli.py`
- Modify: `src/rem/__init__.py` (`from rem.cli import main`)

**Interfaces:**
- Consumes: `load_config`, `Memory`, the models, the errors.
- Produces: `main(argv: list[str] | None = None) -> int` (the console script calls `sys.exit(main())`).
  - Subcommands:
    - `add TEXT --scope {global,system,project}` (default `global`): stores `user_confirmed`/`verified`, `source="cli"`, `actor="cli"`, `cwd=Path.cwd()`.
    - `search QUERY [--limit N] [--scope S]`
    - `list [--scope S] [--status S]`
    - `get ID`, `update ID TEXT`, `delete ID`, `confirm ID`, `history ID`
  - Global flag `--json` prints machine-readable output (records via `to_payload()`, search results add `score` and `semantic`).
  - Human output, one memory per line: `<id[:8]>  [<scope><:project_name if any>] <status>  <text>`. Search prefixes `score:.2f`.
  - Exit codes:
    - `0` on success
    - `1` for `MemoryNotFound` (`no memory with id <id>`) and other `RemError`s (their message), printed to stderr
    - `2` for `ValueError` (its message) and argparse errors
  - `get` on an unknown id: same message, exit 1.

- [ ] **Step 1: Write failing tests** (they run `main([...])` in-process with `REM_HOME=tmp_path` and a `REM_CONFIG` file selecting `hashing` + `qdrant`, so real persistence is exercised; `capsys` captures output)

```python
def test_add_then_search_json(cli_env, capsys):
    assert main(["add", "Prefers uv over pip"]) == 0
    capsys.readouterr()
    assert main(["--json", "search", "uv or pip"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["text"] == "Prefers uv over pip" and out[0]["provenance"] == "user_confirmed" and "score" in out[0]

def test_history_and_delete(cli_env, capsys):
    main(["--json", "add", "Uses npm"]); mid = json.loads(capsys.readouterr().out)["id"]
    assert main(["update", mid, "Uses pnpm"]) == 0 and main(["delete", mid]) == 0
    capsys.readouterr(); main(["--json", "history", mid])
    assert [h["event"] for h in json.loads(capsys.readouterr().out)] == ["ADD", "UPDATE", "DELETE"]

def test_unknown_id_exit_1(cli_env, capsys):
    assert main(["confirm", "00000000-0000-0000-0000-000000000000"]) == 1
    assert "no memory with id" in capsys.readouterr().err

def test_empty_text_exit_2(cli_env, capsys):
    assert main(["add", "   "]) == 2 and "text must not be empty" in capsys.readouterr().err

def test_store_locked_exit_1(cli_env, capsys):
    holder = QdrantVectorStore(path=cli_env / "qdrant")
    try:
        assert main(["list"]) == 1 and "in use by another rem process" in capsys.readouterr().err
    finally:
        holder.close()
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `cli.py`** with argparse. Open `Memory` once per invocation and always `close()` it in `finally`.
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: all pass. Then a manual smoke test with the real embedder: `REM_HOME=$(mktemp -d) uv run rem add "Prefers uv over pip" && REM_HOME=<same dir> uv run rem search "package manager"`. Expected: one result line containing `[global] verified  Prefers uv over pip`.
- [ ] **Step 5: Update `CLAUDE.md` Commands:**
  - `uv run pytest`
  - `uv run pytest tests/test_x.py::test_name -v`
  - `uv run pytest -m slow` (real model download)
  - `uv run rem <subcommand>`
  - and remove the "no tooling yet" line
- [ ] **Step 6: Commit** `git commit -m "feat: rem CLI over the Memory façade"`

---

## Done when

- `uv run pytest` passes, and `uv run pytest -m slow` passes with network access.
- `uv run rem add/search/list/get/update/delete/confirm/history` work against a real `~/.local/share/rem` (or `REM_HOME`), and data survives between invocations.
- Payloads carry `entity_ids: []`; history rows carry `target_kind`. Nothing from M2+ (daemon, LLM, hooks, MCP, BM25) is built.
