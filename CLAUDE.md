# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

rem is a local-first personal memory layer for AI agents, modelled on Mem0. Python 3.14, uv, `uv_build` backend, code in `src/rem/`. **M1 (core storage + Memory façade), M2 (daemon), M3 (LLM extraction), M4 (Claude Code adapter) and M5 (MCP shim) are implemented.** The CLI is a thin client of the daemon and starts it automatically. Later milestones (BM25, entities, …) are not built yet.

The **approved design (rev 2)** is `docs/superpowers/specs/2026-11-15-rem-design.md`, with diagrams in `docs/superpowers/specs/2026-11-15-rem-architecture.html`. It is implemented milestone by milestone (M1 → M9). Background notes are in `plans/initial_plan.md` and `plans/Mem0 Memory Pipeline - Claude.pdf`; `plans/` is **gitignored** and local only.

Each milestone gets its own implementation plan before any code is written.

## Commands

- `uv sync`: install dependencies into `.venv`
- `uv run pytest`: run the test suite (excludes `slow` and `llm`)
- `uv run pytest tests/test_memory.py::test_add_get_and_history -v`: run one test
- `uv run pytest -m slow`: tests that download the real fastembed model (about 70 MB, once, into `~/.cache/rem/fastembed`)
- `uv run rem <add|search|list|get|update|delete|confirm|history> ...`: memory commands; they auto-start the daemon. `--json` works before or after the subcommand.
- `uv run pytest -m llm`: tests that call a real LLM provider. `test_anthropic_live` needs `ANTHROPIC_API_KEY` (costs money); `test_ollama_live` needs a running Ollama and `REM_TEST_OLLAMA_MODEL=<model>` (e.g. `qwen3.5:4b`). Each skips when its variable is unset.
- `uv run rem eval extraction [--dataset PATH] [--limit N] [--json]`: score the configured LLM on the extraction dataset (`src/rem/evals/data/extraction.jsonl`). It runs in-process against throwaway stores, never contacts the daemon, and calls the LLM for every case, so it costs money with a remote provider. Exits 3 with no `[llm]` or without `allow_remote`.
- `uv run rem retry-failed`: requeue extraction jobs that used up their attempts (e.g. after fixing the LLM config or key). They go back to `pending` with attempts reset; they kept their turns and any cached extraction, so a retry only re-runs what failed.
- `uv run rem start|stop|status|daemon|install-service [--enable]`: daemon lifecycle (`daemon` runs it in the foreground; `install-service` writes the systemd user unit). `rem status` also prints `claude_code_hooks:` (where the hooks are installed, or `none`) and `spooled: N` (hook turns waiting in the spool), even with the daemon down.
- `uv run rem mcp`: the stdio MCP server. Claude Code runs it; don't call it by hand. It is a thin client of the daemon (auto-starts it) and writes only JSON-RPC lines to stdout.
- `uv run rem install claude-code [--project] [--uninstall] [--dry-run]`: add (or remove) rem's hooks in Claude Code settings (user settings by default, `--project` for `<git root>/.claude/settings.local.json`). It also registers the `rem` MCP server through `claude mcp add` (user scope, or local scope with `--project`), or prints the command to run when `claude` isn't on PATH; `--uninstall` removes it. `rem hook claude-code` is the internal command those hooks run; don't call it by hand. It doesn't need a valid rem config. When the systemd unit isn't installed it prints a tip to run `rem install-service --enable`: without the unit the first session after boot gets no SessionStart context (the hook only starts the daemon, it never waits for it).
- Config for extraction: `[llm]` (`provider` = `none`|`anthropic`|`openai`|`ollama`, `model`, `base_url`, `api_key_env`, `timeout_seconds`, `effort`, `think`, `allow_remote`; `think = false` turns off model thinking on openai/ollama by sending `reasoning_effort: "none"`, e.g. for `qwen3.5:4b`) and `[extraction]` (`restate_min_similarity`, `related_limit`, `max_candidates`). A non-loopback endpoint (including the Anthropic API) needs `allow_remote = true`. API keys come only from environment variables, never from config: the one named by `api_key_env`, else `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, with no fallback from a custom name to the default. Anthropic, and OpenAI at a non-loopback endpoint, without a key exit 3 at startup.
- Set `REM_HOME=<dir>` (data), `REM_CONFIG=<file>` (config) **and `REM_SOCKET=<short path>`** to run against a throwaway store. Without `REM_SOCKET` you talk to the real daemon at `$XDG_RUNTIME_DIR/rem.sock`. The config can set `[embedder] provider = "hashing"` for an offline embedder, and `[vector_store] provider = "memory"`.

There is no lint or format tooling configured.

`.env` exists and Claude is denied read access to it (`.claude/settings.json`). Leave it that way.

## Architecture (as designed)

Read the design spec for the details. These are the invariants that span modules:

- **What exists (M1–M5):**
  - `rem.daemon.server.run_daemon` runs uvicorn on a socket rem binds itself, never uvicorn's `uds=`, because uvicorn chmods that to 0666.
  - Every `Memory` and `RemDB` call goes through `StoreThread`, a single thread that is the single writer.
  - `/ingest` stores redacted turns and a debounced per-session job (`storage/queue.py`).
  - `Worker` runs jobs through a `JobHandler`. With an `[llm]` configured it is `ExtractionHandler`; with none the worker doesn't start and jobs stay `pending`.
  - The extraction prompt clips each turn to 4000 chars (first and last 2000, `extract.MAX_TURN_CHARS`); stored messages are not clipped.
  - Extraction is cached per job (`extractions` table, schema v3) before anything is written, so replaying a job never makes a second LLM call. The LLM call runs in a daemon thread, off the store thread, so the API keeps being served. On shutdown the worker waits up to `lease_seconds / 2` for an in-flight job, then releases it back to `pending` and abandons the LLM thread; only process exit never waits on that thread.
  - The systemd unit needs `ANTHROPIC_API_KEY` through `systemctl --user edit rem` (`Environment=` or `EnvironmentFile=`), never a key in config.
  - **Claude Code adapter (M4):** `rem hook claude-code` (`adapters/claude_code.py`) reads one hook payload from stdin and always exits 0, never writes to stderr, and imports no storage, embedder or qdrant code. Config is `[claude_code]`: `exclude` (folders where rem does nothing at all), `context_max_chars` (default 1600, 1..10000) and `hook_budget_seconds` (default 2.0, > 0 and ≤ 9, the HTTP timeout to the daemon).
    - Capture comes from hook payloads only (`prompt` on `UserPromptSubmit`, `last_assistant_message` on `Stop`), never from the transcript file, so there is no transcript cursor or backfill. Turns are redacted in the hook, with ids `cc:<prompt_id>:user` and `cc:<prompt_id>:assistant:<sha256(redacted reply)[:12]>` (a second `Stop` for the same prompt with a different reply is a new turn; the same reply re-sent is a duplicate).
    - A prompt turn is held (`enqueue=false`, no job) until the `Stop` (or `SessionEnd`) ingest enqueues the debounced job, so a user turn and its reply are usually extracted together; a quick follow-up prompt (within the debounce window) can land in the previous job, and its reply then starts the next one.
    - If the daemon is down (or the ingest fails other than as invalid), the hook writes the `/ingest` body to `<data_dir>/spool/` (dir 0700, files 0600, at most 1000 files, then `SpoolFull`). On a connection failure (not a timeout) it also starts the daemon in the background, at most once a minute (`<data_dir>/daemon-start.stamp`; the CLI's auto-start ignores this). The daemon drains the spool at startup (right after interrupted jobs are recovered), when it begins serving and then every 30 s (`SPOOL_POLL_SECONDS`) while files are waiting; unparseable or invalid files go to `spool/bad/`.
    - rem creates its data dir 0700 and `daemon.log`/`hook.log` 0600 whatever the umask (`rem.config.ensure_private_dir`).
    - `SessionStart` prints `POST /context` output: verified memories only, no other projects', ordered project, then system, then global (newest first), whole lines only, within `context_max_chars`. No memories means no output. The other events print nothing.
  - **MCP shim (M5):** `rem mcp` is a hand-rolled stdio JSON-RPC server with no MCP SDK (`rem/mcp/protocol.py` is the protocol layer, `rem/mcp/tools.py` the tools). The 7 tools (`search_memory`, `list_memories`, `add_memory`, `confirm_memory`, `update_memory`, `delete_memory`, `memory_history`) are thin `RemClient` calls.
    - `add_memory` is always stored `assistant_generated`/`unverified` with source `mcp`. An MCP `update_memory` makes the memory unverified too (`Memory.update(unverify=True)`); the CLI's update stays trusted. Every MCP write uses actor `mcp`.
    - `add_memory` with scope `project` needs a git repo (`project_root(cwd)`, no subprocess); otherwise it returns an error and stores nothing.
    - **An agent can never verify a memory.** Memories written over MCP (`add_memory`, `update_memory`) are `agent_claim`s: an extracted restatement never corroborates them; only the user does (`rem confirm`, or the flow below). MCP `confirm_memory` calls `POST /memories/{id}/request-confirmation` (never `/confirm`). That sets `confirm_requested_at` and `confirm_request_project_id` (the MCP cwd's project) and writes a `CONFIRM_REQUEST` history row; the memory stays unverified. The extraction job verifies it (`user_confirmed`, a `CONFIRM` row by `extractor`) only if all of these hold, and otherwise it stays unverified:
      - The request falls inside the job's exchange window (`IngestStore.job_window`, schema v4 `jobs.enqueued_at`): after the session's previous enqueue, up to this job's last enqueue. The window is fixed by enqueueing, never by claim or retry.
      - The request's project is the job's project.
      - A **separate** LLM call (`extract.confirm_pending`) names the memory and quotes the user. It sees only the assistant message the user answered and the job's turns up to the last user turn. The extraction prompt and `Memory.related` never see pending memories.
      - The quote is whole words of one sentence of a `user` turn of the job, and that sentence names the fact (`trust.quote_in_user_turns`, deterministic). The sentence must contain most of the fact's key words: 1 of 2, 2 of 3, 3 of 4 or 5. Key words are stemmed and exclude stopwords and verbs like "uses"/"prefers". Sentences split on raw text at line breaks and end punctuation. A bare "yes" is never enough, because what it agrees to is set by the agent's question.
      - The memory's text (hash) and request time are unchanged since the LLM saw them. Any update clears the request.
    - Prompts indent a turn's own line breaks (`extract.render_turn`), so a turn's content can't fake a `user:` line. Facts are shown on one line (as JSON strings in the confirm prompt).
    - Main extraction path (M3, tightened in M5): the daemon passes the job's `user_texts` to `apply_candidates`. A `user`-origin candidate that no user sentence names (`trust.user_states`, one shared key word) is downgraded to `assistant`/unverified, so the LLM's say-so never verifies a fact on its own. All 33 user facts and restatements in the eval dataset pass this check.
    - Out of scope: an agent with shell access acts as the user (`rem confirm`, the socket API).
    - Without an `[llm]` nothing is confirmed this way. Confirmations are cached with the job's extraction (`{"candidates", "confirmations"}`; a pre-M5 bare list still loads).
    - `list_memories` passes the server's cwd (`GET /memories?cwd=`, `Memory.list_for`). It shows project memories only for that git root's project, never another repository's. With scope `project` outside a git repo it returns an error; unscoped outside a repo it lists no project memories. The CLI `rem list` still lists everything.
    - Claude Code starts stdio servers in the session dir with `CLAUDE_PROJECT_DIR` set to it.
  - **M5 backlog invariants:**
    - Projects are resolved off the store thread (`project_root` has no subprocess).
    - `list_points(limit=None)` returns every match.
    - A spool file that fails to drain 3 times within one daemon run is quarantined in `spool/bad/`, which keeps 100 files; to recover, move the files from `spool/bad/` back to `spool/`, then restart the daemon or wait for the next drain.
    - `hook.log` rotates at 1 MB to `hook.log.1`.
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
