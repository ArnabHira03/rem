# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

rem is a local-first personal memory layer for AI agents, modelled on Mem0. Python 3.14, uv, `uv_build` backend, code in `src/rem/`. **M1 (core storage + Memory façade + CLI) is implemented**; the CLI runs in-process until the M2 daemon exists. Later milestones (daemon, extraction, hooks, MCP, …) are not built yet.

The **approved design (rev 2)** is `docs/superpowers/specs/2026-11-15-rem-design.md`, with diagrams in `docs/superpowers/specs/2026-11-15-rem-architecture.html`. It is implemented milestone by milestone (M1 → M9). Background notes are in `plans/initial_plan.md` and `plans/Mem0 Memory Pipeline - Claude.pdf`; `plans/` is **gitignored** and local only.

Each milestone gets its own implementation plan before any code is written.

## Commands

- `uv sync`: install dependencies into `.venv`
- `uv run pytest`: run the test suite (excludes `slow`)
- `uv run pytest tests/test_memory.py::test_add_get_and_history -v`: run one test
- `uv run pytest -m slow`: tests that download the real fastembed model (about 70 MB, once, into `~/.cache/rem/fastembed`)
- `uv run rem <add|search|list|get|update|delete|confirm|history> ...`: the CLI; `--json` gives machine-readable output
- Set `REM_HOME=<dir>` (data) and `REM_CONFIG=<file>` (config) to run against a throwaway store. The config can set `[embedder] provider = "hashing"` for an offline embedder, and `[vector_store] provider = "memory"`.

There is no lint or format tooling configured.

`.env` exists and Claude is denied read access to it (`.claude/settings.json`). Leave it that way.

## Architecture (as designed)

Read the design spec for the details. These are the invariants that span modules:

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
  - `extract` (LLM) only proposes candidates `{text, scope, origin}`.
  - `trust` applies deterministic rules (no LLM) to set `provenance` (`user_asserted`, `assistant_generated`, `tool_observed`, `user_confirmed`) and `status` (verified/unverified). Assistant-generated facts start unverified.
  - `retrieve` only reads. It never changes trust.
- **The write path is ADD-only**, like current Mem0. A near-duplicate of an unverified memory from a verified origin upgrades that memory (`CORROBORATE` in history) rather than creating a duplicate.
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
