# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

rem is a local-first personal memory layer for AI agents, modelled on Mem0. Python 3.14, uv, `uv_build` backend, code in `src/rem/`. **M1 (core storage + Memory façade), M2 (daemon) and M3 (LLM extraction) are implemented.** The CLI is a thin client of the daemon and starts it automatically. Later milestones (hooks, MCP, …) are not built yet.

The **approved design (rev 2)** is `docs/superpowers/specs/2026-11-15-rem-design.md`, with diagrams in `docs/superpowers/specs/2026-11-15-rem-architecture.html`. It is implemented milestone by milestone (M1 → M9). Background notes are in `plans/initial_plan.md` and `plans/Mem0 Memory Pipeline - Claude.pdf`; `plans/` is **gitignored** and local only.

Each milestone gets its own implementation plan before any code is written.

## Commands

- `uv sync`: install dependencies into `.venv`
- `uv run pytest`: run the test suite (excludes `slow` and `llm`)
- `uv run pytest tests/test_memory.py::test_add_get_and_history -v`: run one test
- `uv run pytest -m slow`: tests that download the real fastembed model (about 70 MB, once, into `~/.cache/rem/fastembed`)
- `uv run rem <add|search|list|get|update|delete|confirm|history> ...`: memory commands; they auto-start the daemon. `--json` works before or after the subcommand.
- `uv run pytest -m llm`: tests that call a real LLM provider (cost money; need `ANTHROPIC_API_KEY`)
- `uv run rem eval extraction [--dataset PATH] [--limit N] [--json]`: score the configured LLM on the extraction dataset (`src/rem/evals/data/extraction.jsonl`). It runs in-process against throwaway stores, never contacts the daemon, and calls the LLM for every case, so it costs money with a remote provider. Exits 3 with no `[llm]` or without `allow_remote`.
- `uv run rem retry-failed`: requeue extraction jobs that used up their attempts (e.g. after fixing the LLM config or key). They go back to `pending` with attempts reset; they kept their turns and any cached extraction, so a retry only re-runs what failed.
- `uv run rem start|stop|status|daemon|install-service [--enable]`: daemon lifecycle (`daemon` runs it in the foreground; `install-service` writes the systemd user unit)
- Config for extraction: `[llm]` (`provider` = `none`|`anthropic`|`openai`|`ollama`, `model`, `base_url`, `api_key_env`, `timeout_seconds`, `effort`, `allow_remote`) and `[extraction]` (`restate_min_similarity`, `related_limit`, `max_candidates`). A non-loopback endpoint (including the Anthropic API) needs `allow_remote = true`. API keys come only from environment variables, never from config: the one named by `api_key_env`, else `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, with no fallback from a custom name to the default. Anthropic, and OpenAI at a non-loopback endpoint, without a key exit 3 at startup.
- Set `REM_HOME=<dir>` (data), `REM_CONFIG=<file>` (config) **and `REM_SOCKET=<short path>`** to run against a throwaway store. Without `REM_SOCKET` you talk to the real daemon at `$XDG_RUNTIME_DIR/rem.sock`. The config can set `[embedder] provider = "hashing"` for an offline embedder, and `[vector_store] provider = "memory"`.

There is no lint or format tooling configured.

`.env` exists and Claude is denied read access to it (`.claude/settings.json`). Leave it that way.

## Architecture (as designed)

Read the design spec for the details. These are the invariants that span modules:

- **What exists in M2:**
  - `rem.daemon.server.run_daemon` runs uvicorn on a socket rem binds itself, never uvicorn's `uds=`, because uvicorn chmods that to 0666.
  - Every `Memory` and `RemDB` call goes through `StoreThread`, a single thread that is the single writer.
  - `/ingest` stores redacted turns and a debounced per-session job (`storage/queue.py`).
  - `Worker` runs jobs through a `JobHandler`. With an `[llm]` configured it is `ExtractionHandler`; with none the worker doesn't start and jobs stay `pending`.
  - Extraction is cached per job (`extractions` table, schema v3) before anything is written, so replaying a job never makes a second LLM call. The LLM call runs in a daemon thread, off the store thread, so the API keeps being served. On shutdown the worker waits up to `lease_seconds / 2` for an in-flight job, then releases it back to `pending` and abandons the LLM thread; only process exit never waits on that thread.
  - The systemd unit needs `ANTHROPIC_API_KEY` through `systemctl --user edit rem` (`Environment=` or `EnvironmentFile=`), never a key in config.
  - Exit codes: 2 means already running, 3 means a config error. The systemd unit doesn't restart on either.
  - Socket paths must stay ≤ 100 bytes, so tests use `/tmp/rem-*` directories (the `short_dir` and `daemon_env` fixtures).

- **One daemon owns everything.** Clients never touch storage. Claude Code hooks (`rem hook claude-code`), the stdio MCP shim (`rem mcp`) and the CLI all talk to one `rem daemon` over a Unix socket (`$XDG_RUNTIME_DIR/rem.sock`). It runs as a systemd user service with auto-start as a fallback, and `flock` guarantees a single instance.
- **Storage follows Mem0's split.**
  - **Qdrant local mode** (`~/.local/share/rem/qdrant/`) holds the **canonical memory records**: the vector plus a payload that *is* the memory. It sits behind a `VectorStore` interface modelled on Mem0's `VectorStoreBase`, so the backend can be swapped.
  - **SQLite** (`~/.local/share/rem/rem.db`) holds only history (the audit log), redacted messages, sessions, projects, the job queue and `meta`. **SQLite is not the fact ledger.**
- **No cross-store transaction.** Writes are made crash-safe by:
  - deterministic memory ids: `uuid5(job_id, normalized_text_hash)`
  - a fixed write order: Qdrant upsert → history `INSERT OR IGNORE` (`UNIQUE(job_id, target_id, event)`) → job done
  - replaying any job after a crash must never create duplicates
  - only the `Memory` façade writes to the stores
- **Separate responsibilities.**
  - `extract` (LLM) only proposes candidates `{text, scope, origin, restates}`. `restates` is the LLM's link to an existing memory saying the same thing; `extract` resolves its integer index to a memory id, and that resolved id is what the cached extraction replays.
  - `trust` applies deterministic rules (no LLM) to set `provenance` (`user_asserted`, `assistant_generated`, `tool_observed`, `user_confirmed`) and `status` (verified/unverified). Assistant-generated facts start unverified.
  - `retrieve` only reads. It never changes trust.
- **The write path is ADD-only**, like current Mem0. A candidate is compared with one existing memory, never chosen by cosine alone (real embeddings score contradictions like "tabs over spaces"/"spaces over tabs" at 0.99): the memory with the same normalized text, else the one its `restates` link names in the same scope and project, if cosine ≥ `restate_min_similarity` (a guard against a wrong link). A restatement of an unverified memory from a verified origin upgrades that memory (`CORROBORATE` in history) rather than creating a duplicate; a changed value or a contradiction is a new memory. A job with no `user` turn cannot produce user-asserted facts: `user` origins are downgraded to `assistant`.
- **Scopes:** `global`, `system`, `project`. Search ranks the current project (its git root) highest and holds other projects' memories to a stricter similarity threshold. SessionStart injection includes only verified memories, capped at about 400 tokens.
- **The embedder is configured separately from the LLM.** The default is local fastembed. Collection names include the embedding model and dimension; switching models means running `rem reembed`, and the daemon refuses to start on a mismatch.
- **Extraction LLM:** use an API model during development (sending data to it requires `allow_remote = true` in config). Local models are the long-term default, chosen through `rem eval extraction`.
- **Graph-ready from M1.** Memory payloads carry `entity_ids[]` and history rows carry `target_kind` (memory, entity or relation). Entities and typed relations are planned as extra Qdrant collections (M7, M8), and the graph UI (M9) is a read-only view over them. Don't add a graph database.

## Reference

The Mem0 source (`mem0ai/mem0`) is the reference implementation, especially:
- `mem0/memory/main.py` (`_add_to_vector_store`, `_search_vector_store`)
- `mem0/vector_stores/base.py`, `mem0/vector_stores/qdrant.py`
- `mem0/memory/storage.py`
- `mem0/utils/scoring.py`
- `mem0/configs/prompts.py` (`ADDITIVE_EXTRACTION_PROMPT`)

Mem0 no longer calls its old ADD/UPDATE/DELETE prompt from `add()`, and it no longer has a graph store.
