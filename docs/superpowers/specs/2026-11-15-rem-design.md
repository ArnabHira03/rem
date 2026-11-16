# rem — design (rev 2)

_Status: approved 2026-11-15. Architecture diagrams: `2026-11-15-rem-architecture.html` in this folder._

## Context
rem is a **persistent, local-first personal memory layer**. It learns and stores what it finds out about the user, their system, preferences, development environment and projects, and every AI agent on the machine shares it as **one unified memory**. The main goal is to **learn by building**, with Mem0 as the reference.

**Rev 2:** the user asked to revisit storage so it follows **Mem0's separation of memory records / vector retrieval / history**, instead of everything in one SQLite + sqlite-vec file. They also want an **Obsidian-inspired graph interface later**, which means the memory model must be ready now for entities, aliases and typed relationships. No graph DB and no UI in the initial phase.

Background: `plans/initial_plan.md` (Mem0 pipeline analysis) and `plans/Mem0 Memory Pipeline - Claude.pdf`. Repo: empty uv scaffold, Python ≥3.14, `uv_build`; the `rem = "rem:main"` entry point is broken because `src/rem/__init__.py` was deleted. Machine: TigerLake CPU (8 threads), Iris Xe, 15 GB RAM, no Ollama.

## How Mem0 actually stores things (from source at `f1791da`)
| Concern | Where it lives in Mem0 | Code |
|---|---|---|
| **Memory records** (canonical) | **Vector store payload.** Each point = embedding + payload `{data, hash, text_lemmatized, user_id/agent_id/run_id, actor_id, role, created_at, updated_at, …}`. The payload *is* the memory; there's no other fact table. | `Memory._add_to_vector_store`, `main.py` |
| **Vector retrieval** | The same collection: `search()` (dense) and, optionally, `keyword_search()` (BM25). Qdrant does BM25 with fastembed `Qdrant/bm25` sparse vectors in a `bm25` slot. | `vector_stores/base.py::VectorStoreBase`, `vector_stores/qdrant.py:111-180, 471` |
| **Entities** | A **second collection in the same vector store** (`<collection>_entities`); for embedded Qdrant it **shares the client** to avoid the path lock. | `Memory.entity_store`, `main.py:559-580` |
| **History** | **SQLite** `history` table: memory_id, old → new, event, actor_id, role, is_deleted. It's an audit log, not the source of truth. | `memory/storage.py::SQLiteManager` |
| **Conversation context** | The same SQLite file, `messages` table, keyed by session scope. | `SQLiteManager.save_messages/get_last_messages` |
| **Pluggability** | `VectorStoreFactory` + a pydantic config per provider; 25+ backends. **Default `qdrant`, `path=/tmp/qdrant`** (not persistent across reboots). | `vector_stores/configs.py`, `configs/vector_stores/qdrant.py` |
| **Cross-store consistency** | None. A vector insert and a history write are separate; failures are logged and the operation falls back to inserting one at a time. | `main.py` phases 6–7 |

So Mem0 = **one vector store (memories + entities) + one SQLite file (history + messages)**. rem follows that exactly and adds what Mem0 lacks: crash-safe write ordering and persistent default paths.

## Local storage options compared
All of these have Python 3.14 wheels on PyPI today.

| Option | Payload filters | Keyword/BM25 | Persistent embedded | Replaceable later | Notes |
|---|---|---|---|---|---|
| **Qdrant local mode** (`qdrant-client`, `path=`) | yes | yes (sparse + fastembed, the same as Mem0) | yes, no server | **Same API to a Qdrant server** by changing only the config | Mem0's default. Pure Python, brute-force search: fine for thousands to tens of thousands of memories. Single-process lock on the path, which fits the one-daemon design. |
| Chroma persistent | yes | not in Mem0's adapter | yes | own API | Heavier dependency tree; no hybrid search in Mem0's version. |
| LanceDB | yes | yes (FTS) | yes | own API | Strong engine, but not in Mem0, so less reference value. |
| FAISS | **no** | no | index file + pickled payloads | — | Would need a custom metadata store. Rejected. |
| pgvector | yes | yes | needs a Postgres server | — | Extra infrastructure. Rejected. |
| sqlite-vec | via SQL | FTS5 | yes | — | Not a Mem0 backend; would make SQLite the fact ledger. Rejected by the requirements. |

**Recommendation: Qdrant local mode** behind a Mem0-style `VectorStore` interface. It's Mem0's own default, needs no server, has hybrid search built in, and the upgrade path is a config change (`path` → `url`).

## Revised storage architecture
```
~/.local/share/rem/
├── qdrant/            ← VECTOR STORE (canonical knowledge), opened only by the daemon
│   ├── rem_memories__<embed-slug>    memory records: vector + full payload
│   ├── rem_entities__<embed-slug>    (M7) entities + aliases
│   └── rem_relations__<embed-slug>   (M8) typed relationships with provenance
└── rem.db             ← SQLITE (operational + audit, NOT the fact ledger)
    history · messages · sessions · projects · jobs · meta
```
- **Canonical = the vector store payloads.** Everything needed to rebuild a memory (text, scope, provenance…) is in the payload, so re-embedding is lossless.
- **SQLite holds only:** the history audit log, raw conversation turns (redacted), the session/project registry, the job queue, and `meta` (schema version, active embedding model/collections).
- **Two stores total.** No graph DB and no custom storage engine.

### Crash-safe write ordering (what Mem0 doesn't do)
There's no transaction across Qdrant and SQLite, so instead:
1. **Deterministic IDs:** `memory_id = uuid5(job_id, normalized_text_hash)`. Replaying a job produces the same IDs.
2. **Order:** upsert into Qdrant → `INSERT OR IGNORE` history rows (`UNIQUE(job_id, target_id, event)`) → mark the job done.
3. **Crash at any point** → the lease expires → the job replays → upserts overwrite identical points and history inserts are ignored. **No duplicates, no lost writes.**
4. Payload updates (confirm, corroborate) are idempotent "set" operations.

### Embeddings (independent of the extraction LLM)
- Separate config `[embedder]` from `[llm]`. Providers: **fastembed (default, local ONNX on CPU, no torch, no Ollama)**, Ollama, OpenAI-compatible. Default model `BAAI/bge-small-en-v1.5` (384-d). fastembed is also what Mem0 uses for BM25, so it's one dependency covering both.
- **Collection names include the embedding slug + dimension.** `meta.active_embedder` points to the live collections.
- **`rem reembed --model X`:** create the new collections → scroll the old payloads → re-embed text (and entity names and relation sentences) → upsert with **the same IDs and payloads** → switch `meta` → keep the old collections until `rem reembed --finalize`. History records a `REEMBED` event per run (not per memory).
- The daemon refuses to start if the configured embedder doesn't match `meta.active_embedder`, and tells you to run `reembed`.

## Graph-ready memory model (no graph DB)
Decided **now** so the payload and history schema never need a breaking change. Built in M7–M9.

- **Memory payload (from M1):** `entity_ids: []` (empty until M7).
- **Entity (M7)**, in collection `rem_entities`. The vector is the embedding of the name (Mem0 pattern). Payload:
  `{name, normalized, type (person|tool|project|language|org|place|concept…), aliases[], scope, project_id?, memory_ids[], created_at, updated_at}`
  - Merging uses exact normalized/alias match first, then cosine ≥ ~0.95 (Mem0's rule), then appends an alias.
- **Relation (M8)**, in collection `rem_relations`. The vector is the embedding of the sentence "`<source> <type> <target>`", so relations are searchable too. Payload:
  `{source_entity_id, target_entity_id, type (uses|prefers|works_on|depends_on|part_of|runs_on…), memory_ids[] (evidence), provenance, status, scope, project_id?, created_at, updated_at}`
  - It reuses the **same trust model** as memories: a relation proposed by the assistant is `unverified` until corroborated.
- **Memory ↔ entity links** are stored on both sides (`memory.entity_ids`, `entity.memory_ids`), as Mem0 does. Relations cite their evidence memories.
- **The graph interface (M9)** is a **read-only visualization**:
  - It's served by the daemon as a local web page at `/graph`, built from scrolling the three collections. Nodes are entities (and optionally memories, like Obsidian notes); edges are relations plus mention links.
  - It is not a separate store.
  - Editing from the UI, if added later, goes through the same Memory API, so history and trust apply.

## Module boundaries
```
rem/
├── config.py           pydantic config: [vector_store] [embedder] [llm] [daemon] [context]
├── memory.py           Memory façade (like Mem0's Memory): add / search / get / list / update / delete / confirm / history
│                       — the only module that coordinates the stores and enforces write ordering
├── storage/
│   ├── vector/base.py      VectorStore ABC, modelled on Mem0's VectorStoreBase:
│   │                       create_col, upsert, search, search_batch, keyword_search (→None if unsupported),
│   │                       get, update_payload, delete, scroll/list, col_info, delete_col
│   ├── vector/qdrant.py    Qdrant local/server implementation (shares one client across collections)
│   ├── vector/memory.py    in-memory fake for tests (proves the interface is backend-neutral)
│   ├── vector/factory.py   provider → class (like Mem0's VectorStoreFactory)
│   └── sqlite.py           history, messages, sessions, projects, jobs, meta
├── embed/              Embedder ABC + fastembed / ollama / openai-compatible
├── llm/                LLM ABC + anthropic / openai-compatible / ollama
├── extract.py          candidates {text, scope, origin}; no trust decisions
├── trust.py            deterministic provenance/status rules + corroboration decision
├── retrieve.py         search ranking + session context; reads only
├── entities.py         (M7) entity extraction + resolution;  relations.py (M8)
├── daemon/             socket API, job queue worker, lifecycle
├── adapters/claude_code.py
├── mcp_server.py
└── cli.py
```
Rules: `extract` → `trust` → `memory` (writes); `retrieve` never writes; only `memory.py` touches both stores.

## Data model
**Vector store payload — memory** (`rem_memories__<slug>`):
`id, text, hash, scope (global|system|project), project_id?, project_name?, provenance (user_asserted|assistant_generated|tool_observed|user_confirmed), status (verified|unverified), evidence_source?, observed_at?, source (claude-code|cli|mcp…), session_id?, job_id?, entity_ids[], created_at, updated_at`
Indexed payload fields: `scope, project_id, status, provenance`.
Deletes are **hard deletes in Qdrant + a DELETE history row holding the old text** (Mem0's approach; recoverable from history).

**SQLite `rem.db`:**
- `meta(key, value)`: schema_version, active_embedder, collection names
- `projects(id, root_path UNIQUE, name, git_remote)`
- `sessions(id, source, session_id, project_id, cwd, first_seen, last_seen, UNIQUE(source, session_id))`
- `messages(id, session_pk, message_id, role, content, ts, UNIQUE(session_pk, message_id))`
- `jobs(id, session_pk, status, attempts, lease_until, not_before, last_error)`
- `history(id, target_kind memory|entity|relation, target_id, event ADD|UPDATE|DELETE|CONFIRM|CORROBORATE|MERGE|REEMBED, old_value, new_value, actor, job_id?, created_at, UNIQUE(job_id, target_id, event))`
  - `target_kind` exists from day one, so entity and relation history needs no migration.

## Decisions carried over from rev 1 (unchanged)
1. Claude Code is the first client: hooks (capture + SessionStart context) and an MCP server.
2. rem owns extraction with its own LLM. Develop and evaluate against an API model with non-sensitive data; local is the long-term preference. Remote use requires `allow_remote = true` and is shown in `rem status`.
3. An agent-independent ingestion API plus per-agent capture adapters (cursor, session id, incremental sync, dedup). *Deviation:* M4 captures from hook payloads only — no transcript cursor or backfill (decided 2026-11-15; the transcript format is internal and unstable).
4. One daemon owns both stores, the LLM and the durable job queue. systemd user service plus auto-start; `flock`; Unix socket at `$XDG_RUNTIME_DIR/rem.sock`.
5. Scopes `global` / `system` / `project`. Ranking: current project +0.15, other projects −0.05 with a stricter threshold (~0.55 vs ~0.3), unverified −0.05, a small recency tiebreak. Every result shows scope, project, provenance, status and created_at.
6. Session-start injection: verified memories only, current project → system → global, ~400-token cap, 2s hook timeout. No per-prompt injection.
7. Provenance + trust rules (user → user_asserted/verified; assistant → assistant_generated/unverified; tool → tool_observed/verified with evidence; confirm → user_confirmed). Corroboration upgrades instead of duplicating.
8. Extraction pipeline as in Mem0's additive design: context turns, top-10 related memories with integer IDs, one JSON LLM call `{text, scope, origin}`, batch embed, hash and near-dup dedup.
9. Redaction before storage or LLM calls; no backfill in v1 (last ~20 turns of an unseen session); raw messages kept. *Deviation:* M4 captures from hook payloads only — no transcript cursor or backfill (decided 2026-11-15; the transcript format is internal and unstable).
10. MCP tools: `search_memory, list_memories, add_memory (assistant_generated), confirm_memory, update_memory, delete_memory, memory_history`.
11. Extraction eval harness (`rem eval extraction`), measuring recall/precision, scope, origin, JSON validity, dedup and latency.

## Milestones
| # | Builds | Learn |
|---|---|---|
| M1 | Skeleton + fix the entry point; config; `VectorStore` ABC + **Qdrant local** + in-memory fake; fastembed embedder; SQLite `rem.db` (meta, history); `Memory` façade with `add(infer=False)/search/get/list/update/delete/history`; deterministic IDs + write ordering; scope/provenance payload incl. empty `entity_ids`; CLI in-process | Mem0's storage split |
| M2 | Daemon: socket API, durable job queue + leases, single owner of both stores, flock, start/stop/status, systemd, auto-start. CLI switches to talking to the daemon | concurrency, crash safety |
| M3 | LLM providers; `extract`, `trust`, dedup/corroboration; redaction; `allow_remote`; eval harness + dataset | Mem0's write path |
| M4 | Claude Code adapter: ingest hooks, cursor, project detection, SessionStart context, `rem install claude-code`. *Deviation:* M4 captures from hook payloads only — no transcript cursor or backfill (decided 2026-11-15; the transcript format is internal and unstable). | incremental sync |
| M5 | MCP shim + tools; end-to-end in Claude Code. *Deviation:* hand-rolled stdio JSON-RPC (no MCP SDK; its dependency tree is far heavier than rem's), registered via `claude mcp add`; an MCP update makes the memory unverified (decided 2026-11-16); MCP `confirm_memory` only *requests* confirmation, verified later only when the user's own words in that exchange confirm it (a separate LLM check plus a literal quote match), and `list_memories` shows only the current repository's project memories (decided 2026-11-16). | MCP |
| M6 | Hybrid retrieval: BM25 sparse vectors in Qdrant (as in Mem0) + `score_and_rank`; `rem reembed` | Mem0's read path. *Deviations (decided 2026-11-16):* keyword score is an additive boost (`KEYWORD_BOOST` 0.10 × normalized BM25) on the dense cosine, not a 0.7/0.3 blend, so dense-only scores and the 0.30/0.55 thresholds keep their meaning; the threshold still gates on dense only. Reembed runs in one store-thread call, not batches interleaved with the API, so no write lands mid-copy; finalize re-syncs the copy before switching. A same-model reembed is allowed (needed to add the BM25 slot to an M5 collection) and uses a `__gN` generation name. |
| M7 | Entities: extraction (spaCy or LLM), alias resolution, `rem_entities` collection, memory ↔ entity links, entity boost in search | Mem0 entity store. *Deviations (decided 2026-11-16):* the extractor is the already-configured LLM (the extraction call also returns `entities` per fact), not spaCy. An entity exists once per (scope, project) bucket, so there is no cross-project merge. Search uses entities only as a boost (+0.05 when the entity's name or an alias occurs in the query as whole words; a cosine match was dropped because bge-small scores unrelated names >= 0.5); they never add candidates. A spelling that differs only in case or spacing is the same entity, never an alias. Deleting a memory's last link, or editing its text, unlinks it and deletes an orphaned entity. A candidate that restates a memory links its entities to it. Reembed finalize switches its meta keys in one SQLite transaction. Known gaps: some entity history rows can be lost on a crash (MERGE after an alias write, DELETE after an orphan's point delete), a second new alias in one job gets no MERGE row, and the extraction eval has no entity labels. The 0.95 cosine alias merge rarely fires with bge-small (postgres~postgresql 0.934, neovim~nvim 0.912). A crash between finalize's two collection deletes can leave the old entity collection orphaned on disk. |
| M8 | Typed relationships: `rem_relations`, provenance/trust on relations, relation search | graph-shaped memory, no graph DB |
| M9 | **Graph interface:** read-only Obsidian-style view served by the daemon (`/graph`) over entities, relations and memories | visualization |
| Later | Tool-output ingestion + system-facts collector (`tool_observed`); local-model default after evals; backfill; pruning; optional Qdrant server via config | |

## Key trade-offs
- **Two stores instead of one file.** This matches Mem0 and makes the vector backend swappable. The cost: no cross-store transaction. That's handled with deterministic IDs, idempotent upserts and history inserts, and job replay. Backups mean copying one directory (`~/.local/share/rem/`) with the daemon stopped.
- **Qdrant local mode is brute-force and single-process.** That's fine for personal-scale memory and matches the one-daemon design. If it ever gets slow, switching to a Qdrant server binary is a config change, not a rewrite. The CLI can't open the store while the daemon runs, so after M2 the CLI always goes through the daemon.
- **Payload as the canonical record** means `list` and filtering go through Qdrant scroll and filters rather than SQL. Richer ad-hoc queries are harder, which is acceptable at this scale.
- **Hard delete + history (Mem0) vs soft delete.** The vector store stays clean; recovery comes from history's old text.
- **Entities and relations as vector collections rather than a graph DB.** There are no multi-hop graph queries (e.g. "everything 3 hops from X") in the engine. The graph view computes neighbourhoods in Python from scrolled data, which is fine at personal scale, and a graph store can be added behind the same model later.

## Left to the builder
Exact library versions, MCP SDK, HTTP framework, LLM clients, the entity extractor choice in M7, threshold tuning, debounce/retry values, redaction patterns, the graph UI library in M9, test layout.

## Risks to check early (M1)
- Qdrant local-mode support for payload indexes, sparse vectors and `query_points` in the installed `qdrant-client`; persistence across restarts at a real path.
- fastembed model download is a one-time network fetch; cache it under `~/.cache/rem/` and note it in `rem status`.
- Claude Code hook payloads and transcript JSONL format: confirm before M4.

## Verification
- **Interface contract tests** run against both the Qdrant-local and in-memory backends: upsert/search/filter/scroll/update/delete behave identically.
- **Crash replay:** kill the process between the Qdrant upsert and the history insert → replay → exactly one memory and one ADD row.
- **Persistence:** add memories → restart the daemon → search still returns them.
- **Re-embed (M6):** switch the embedder → `rem reembed` → same IDs and payloads, search works, old collections kept until finalize; the daemon refuses a mismatched embedder.
- **Graph readiness:** M1 payloads include `entity_ids`; history accepts `target_kind=entity|relation` without migration.
- All rev-1 checks still apply: idempotent ingest, trust rules, ranking order, context cap, redaction, concurrency, eval run, end-to-end in Claude Code.

## After approval
1. Save this design to `docs/superpowers/specs/2026-11-15-rem-design.md` and commit it; update `plans/initial_plan.md` (storage section + decisions).
2. Invoke writing-plans for M1.
