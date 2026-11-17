# Extraction Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a read-only **Extraction** tab in the `rem graph` window. It lists extraction runs and shows each one as conversation → model proposal → what rem stored, with fixed, named flag rules marking likely problems.

**Architecture:** A pure module `src/rem/review.py` turns one run's stored rows into a reviewed run: effective origin, outcome per fact, and flags. `src/rem/daemon/runs.py` reads those rows in one `StoreThread` call and serves `GET /jobs` and `GET /jobs/{id}`. The window's `Api` gains `jobs` and `job` methods, and `graph.js` adds the tab. Nothing is written anywhere.

**Tech Stack:** Python 3.14, Starlette, httpx, pywebview/Qt (optional `graph` extra), cytoscape (unchanged), pytest.

**Spec:** `docs/superpowers/specs/2026-11-17-extraction-panel-design.md`. Read it in full, especially the flag rules table, "Builder's calls after the builder check" and "Window". Also read the M9 invariants in CLAUDE.md, "Graph window (M9)".

## Global Constraints
- **Read-only.** Only `GET` routes are added, and no `Memory`, `RemDB` or `IngestStore` write method is called. No schema change, no new table or collection, no LLM call.
- `review.py` is pure: no I/O and no imports of storage, daemon, Qt or httpx.
- The rule names are exactly `downgraded`, `invalid-type`, `noise`, `narration`, `scope-hint`, `relation-not-written` and `failed`. Every flag is `{"rule": name, "reason": str}`.
- `noise` pattern: the text matches `https?://`, or a word matching `\b(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b` (at least one digit and one a-f letter).
- The effective origin uses `rem.trust.user_states` over the run's user turns. With no user turn, any `user` origin becomes `assistant`. This is exactly `Memory.apply_candidates`'s rule.
- An *added* fact's memory id is `rem.models.memory_id(text, scope, pid, f"job:{id}")`, where `pid` is the run's project id if `scope == "project"`, else `None`.
- Relations for a fact come from the stores (`Memory.relations_for`), never from rebuilding ids from names.
- `GET /jobs`: newest first; `limit` 1..200 (default 50, else 400); `before` is a job id. With `flagged=true`, `limit` counts flagged runs and the scan stops after 2,000 jobs. The response is `{"runs": [...], "next_before": int | null}`.
- `processing` folds into the `pending` filter. A run with no turns linked yet shows "not run yet".
- Window: black and white, `textContent` only (never `innerHTML`), and the bridge exposes only named read methods. No test in the default suite imports `webview`, `qtpy` or `PySide6`.

## Review Focus
1. **A run from before M3, or one that failed before extraction** (no `extractions` row): the detail shows its turns and `facts: []` with `extraction: null`, and the list shows `model: null`. No crash. Test: Task 1 `test_run_without_extraction`.
2. **The pre-M5 cache format** (a bare JSON list instead of `{"candidates", "confirmations"}`): it parses as candidates with no confirmations. Test: Task 1 `test_parse_extraction_bare_list`.
3. **A memory added by the run and deleted since:** the outcome stays `added`, with `memory_now: null` shown as "deleted since". Test: Task 1 `test_added_then_deleted_memory`.
4. **A conversation containing HTML or script text:** it shows literally. Test: Task 3 `test_page_never_uses_innerhtml` (existing) still passes, and the new page code is covered by it.
5. **`flagged=true` on a store with no flagged runs:** `{"runs": [], "next_before": null}` after scanning to the end (or after 2,000 jobs, with `next_before` set), never an endless scan. Test: Task 2 `test_jobs_flagged_scan_is_bounded`.

---

### Task 1: `review.py`, the pure run review

**Files:**
- Create: `src/rem/review.py`
- Test: `tests/test_review.py`

**Interfaces:**
- Produces:
  - `@dataclass RunRows` with fields:
    - `job: dict` (`id, status, created_at, updated_at, last_error, project_id, project_name, source`)
    - `turns: list[dict]` and `context_turns: list[dict]` (`role, content, ts`)
    - `extraction: tuple[str, str] | None` (model, candidates JSON, as `RemDB.get_extraction` returns)
    - `history: list[dict]` (`HistoryRow` dumps with `job_id == f"job:{id}"`)
    - `memories_now: dict[str, dict | None]`: memory id → current `{id, text, status, provenance}`, or `None` when deleted
    - `relations_by_memory: dict[str, list[str]]`: memory id → ids of stored relations citing it
  - `parse_extraction(candidates_json: str) -> tuple[list[dict], list[dict]]`, returning (candidates, confirmations).
  - `review_run(rows: RunRows) -> dict`, the `GET /jobs/{id}` body from the spec: `run`, `turns`, `context_turns`, `facts`, `writes`, `confirmations`.
  - `run_summary(rows: RunRows) -> dict`, the list item. It includes `flags: list[str]` (distinct rule names over the run and its facts) and `model`.
  - Constants `RULES` (the seven names) and `NARRATION_VERBS` (a frozenset: fixed, added, modified, changed, updated, removed, created, commits, opens, pushes, merged, renamed, implemented, refactored, deleted).

**Per-fact algorithm** (fixed by the spec). Go through the candidates in order:
1. `origin_effective`: see Global Constraints. `downgraded` fires when proposed `user` → effective `assistant`.
2. A normalised text seen earlier in this run → outcome `{"kind": "none", "reason": "duplicate"}`.
3. `scope == "project"` and the run's `project_id` is None → `{"kind": "none", "reason": "no_project"}`.
4. The memory id is in this run's memory `ADD` rows → `{"kind": "added", "memory_id", "memory_now": memories_now.get(id)}`.
5. A memory `CORROBORATE` row in this run targets a memory whose current text normalises to this fact's text, or whose id equals the fact's `restates` → `{"kind": "corroborated", "memory_id", "memory_now"}`.
6. Otherwise → `{"kind": "none", "reason": "existing_or_skipped"}`.

Flags per fact:
- `invalid-type`: one flag per entity mention whose type isn't in `ENTITY_TYPES`.
- `noise`.
- `narration`: an assistant effective origin, and the first word lowercased is in `NARRATION_VERBS`.
- `scope-hint`: scope is `system` or `global`, and the text contains the run's `project_name` as a whole word, or `this repo`, `this project` or `the project`, all case-insensitive.
- `relation-not-written`: the fact has relations, the outcome is `added`, and `relations_by_memory.get(memory_id)` is empty.

Run flag: `failed` when `job["status"] == "failed"`, with `last_error` as the reason.

- [ ] **Step 1: Write the failing tests** in `tests/test_review.py`. Use a `rows(**overrides)` builder giving a done job in project 3 ("rem"), one user turn and one assistant turn, and an extraction JSON built from candidate dicts (`text, scope, origin, restates, entities, relations`).
  - `test_added_fact_with_memory_now`: the history has an ADD for `memory_id("Uses tmux", Scope.GLOBAL, None, "job:7")`, and the user turn says "I use tmux". The outcome is `added`, `memory_now` is the given dict, and there are no flags.
  - `test_downgraded_when_user_never_said_it`: proposed `user`, but no user sentence names it. `origin_effective == "assistant"` and the flags include `downgraded`. A run with no user turn at all also downgrades.
  - `test_duplicate_and_no_project_reasons`: two candidates with the same normalised text give `none/duplicate` for the second. A project-scope fact in a run with `project_id=None` gives `none/no_project`.
  - `test_corroborated_by_text_and_by_restates`: a CORROBORATE row on memory `m1`, whose current text equals the fact, gives `corroborated`. Another fact whose `restates == "m2"` with a CORROBORATE on `m2` gives `corroborated`.
  - `test_existing_or_skipped`: no matching rows gives `none/existing_or_skipped`.
  - `test_added_then_deleted_memory`: an ADD row, with `memories_now[id] = None`, gives `added` with `memory_now is None`.
  - `test_flag_rules`: one fact per rule:
    - type `entity_type_commit` → `invalid-type`;
    - "PR URL is https://x/pull/1" and "Commit ID is 82fd0fc" → `noise`;
    - "Fixed the layout" with assistant origin → `narration`, while a user-origin "Fixed …" doesn't fire;
    - "rem uses Qdrant" with scope system in project "rem" → `scope-hint`, and so does "This repo uses uv" with scope global;
    - an added fact with relations and `relations_by_memory` empty → `relation-not-written`; with one relation id → no flag;
    - a failed job → the run flags include `failed` with `last_error` as the reason.

    The rule names are exactly those in `RULES`.
  - `test_run_summary_counts_and_flags`: `n_turns`, `n_proposed`, `n_added`, `n_corroborated`, `model` and the distinct `flags` names.
  - `test_run_without_extraction`: `extraction=None` gives `facts == []`, `run.model is None` and no exception.
  - `test_parse_extraction_bare_list`: `'[{"text": …}]'` gives (one candidate, `[]`). The `{"candidates", "confirmations"}` form gives both.
  - `test_review_is_pure`: `import rem.review` doesn't import `rem.memory`, `rem.storage` or `rem.daemon` (check `sys.modules` after deleting and re-importing).

- [ ] **Step 2: Run them to verify they fail**
  Run: `uv run pytest tests/test_review.py -v`
  Expected: FAIL (`ModuleNotFoundError: rem.review`).

- [ ] **Step 3: Implement** `src/rem/review.py` per the interfaces. Use `rem.models` (`memory_id`, `normalize_text`, `ENTITY_TYPES`, `Scope`) and `rem.trust.user_states`. Both are pure.

- [ ] **Step 4: Run the tests**
  Run: `uv run pytest tests/test_review.py -v`, then `uv run pytest`
  Expected: PASS.

- [ ] **Step 5: Commit**
  `git commit -m "feat: pure extraction run review with named flag rules"`

---

### Task 2: Daemon routes `GET /jobs` and `GET /jobs/{id}`, and the client

**Files:**
- Create: `src/rem/daemon/runs.py` (gathers `RunRows`; read-only)
- Modify:
  - `src/rem/storage/queue.py`: `IngestStore.job_rows(job_id) -> dict | None` and `IngestStore.job_ids(*, project_id, statuses, before, limit) -> list[int]` (newest first), both read-only `SELECT`s joining `sessions` and `projects`;
  - `src/rem/storage/sqlite.py`: `RemDB.history_for_job(job_key: str) -> list[HistoryRow]`;
  - `src/rem/daemon/service.py`, `src/rem/daemon/app.py`, `src/rem/client.py`.
- Test: `tests/test_runs.py` (new), `tests/test_app.py`, `tests/test_client_cli.py`

**Interfaces:**
- Consumes: `RunRows`, `review_run` and `run_summary` from Task 1.
- Produces:
  - `runs.gather(memory, ingest, job_id: int) -> RunRows | None`. It reads the job, turns (`messages_for`), context (`context_for`), extraction (`memory.db.get_extraction`) and history (`history_for_job(f"job:{id}")`). `memories_now` covers every memory id named by the history rows and by the candidates' would-be added ids, plus the `restates` ids, via `memory.get`. `relations_by_memory` covers the added memory ids, via `memory.relations_for(id, scope, pid)`.
  - `runs.list_runs(memory, ingest, *, project_id: int | None, status: str | None, flagged: bool, limit: int, before: int | None) -> dict`, returning `{"runs": [run_summary...], "next_before": int | None}`.
    - The `status` filter maps `pending` to `("pending", "processing")`.
    - With `flagged`, it scans in batches of `limit` ids until `limit` flagged runs are found, or 2,000 jobs have been scanned, or there are no more jobs. `next_before` is the last scanned id when more may remain, else `None`.
  - `runs.get_run(memory, ingest, job_id) -> dict | None`, which is `review_run(gather(...))`.
  - `RemService.jobs(**filters)` and `RemService.job(id)`: one `store.run` each.
  - Routes, registered beside `/graph`: `GET /jobs` (query params `project_id, status, flagged, limit, before`; bad values give 400, as `/relations` does) and `GET /jobs/{id}` (404 `not_found` if unknown).
  - `RemClient.jobs(project_id=None, status=None, flagged=False, limit=50, before=None) -> dict` and `RemClient.job(id: int) -> dict`.

- [ ] **Step 1: Write the failing tests.** Seed a store with `Memory` plus `IngestStore`, using the `cfg`/`mem` fixtures and the `apply`/`rcand` helpers from `tests/test_memory_entities.py`/`tests/test_memory_relations.py`. Ingest turns to create a session and job, `claim()` it, `save_extraction`, then `apply_candidates` with `job_key=f"job:{id}"`, then `complete()`.
  - `test_get_run_end_to_end`: the run has turns, a fact with outcome `added`, the `writes` include the memory ADD row, and the relations are found through the store.
  - `test_list_runs_newest_first_and_filters`: three jobs come back newest first. `status="failed"` returns only the failed one, and `project_id` filters.
  - `test_jobs_flagged_scan_is_bounded`: with no flagged runs, `flagged=True` returns `{"runs": [], "next_before": None}`. Patch the scan cap to 3 with 5 jobs: the result has `next_before` set and `runs == []`.
  - `test_pending_includes_processing`: a claimed (processing) job appears under `status="pending"` and shows no turns, as "not run yet".
  - `test_runs_read_only`: the history count, all collection payloads and the `jobs` rows are identical before and after `list_runs` and `get_run`.
  - In `tests/test_app.py`: `test_jobs_routes` checks 200 with the `{"runs", "next_before"}` keys, `limit=0` → 400, `status=bogus` → 400, and `/jobs/999999` → 404.
  - In `tests/test_client_cli.py`: `test_client_jobs_round_trip(daemon_env)` checks that `RemClient.jobs()` returns the two keys.

- [ ] **Step 2: Run them to verify they fail**
  Run: `uv run pytest tests/test_runs.py tests/test_app.py -k "jobs or runs" -v`
  Expected: FAIL.

- [ ] **Step 3: Implement** the interfaces above.

- [ ] **Step 4: Run the tests**
  Run: `uv run pytest`
  Expected: PASS.

- [ ] **Step 5: Commit**
  `git commit -m "feat: GET /jobs and /jobs/{id} read-only extraction runs"`

---

### Task 3: Window Extraction tab, graph link, docs

**Files:**
- Modify:
  - `src/rem/memory.py`: add `"job_id"` to `MEMORY_GRAPH_FIELDS`;
  - `src/rem/graph/view.py`: memory nodes and memory entries carry `job_id`;
  - `src/rem/graph/window.py`: `Api.jobs(filters: dict) -> dict` and `Api.job(job_id) -> dict`, which validate input, call `RemClient.jobs`/`job`, and keep the client private;
  - `src/rem/graph/static/graph.html`, `graph.css`, `graph.js`;
  - `CLAUDE.md`.
- Test: `tests/test_graph_window.py`, `tests/test_graph_view.py`, `tests/test_graph_snapshot.py`

**Interfaces:**
- Consumes: `RemClient.jobs`/`job` (Task 2), and the response shapes from Task 1.
- Produces:
  - `Api.jobs(filters)`. `filters` has exactly the keys `project` ("all" or an id, coerced like `view`), `status` ("all", "done", "failed" or "pending"), `flagged` (bool) and `before` (int or None). It returns the `/jobs` body.
  - `Api.job(job_id)`: `job_id` is coerced to int (a string from JS is accepted); anything else raises `ValueError`.
  - The public `Api` surface becomes exactly `initial`, `view`, `refresh`, `jobs`, `job`.

**Page behaviour** (spec "Window"):
- Tabs **Graph | Extraction** sit in the top bar. The project dropdown and ⟳ are shared. The Extraction tab adds a status `<select>` and a "flagged only" checkbox.
- **Left list:** time, project, status, `n_proposed → n_added`, and a flag count badge. "Load more" calls `jobs` with `before = next_before`, and is hidden when that is null.
- **Right, three columns:**
  - conversation turns, with context folded under "earlier context";
  - proposed facts (scope, `origin_proposed` → `origin_effective` when they differ, entities `name · type`, relations);
  - the stored outcome, lined up per fact: added ✓/◌, deleted since, corroborated, or none plus a plain-words reason. Then the run's entity and relation writes.
- Flags are bordered badges showing the rule name, with the reason in `title`. A failed run shows its error.
- **Graph link:** a memory panel whose `job_id` matches `^job:(\d+)$` shows a button, "show the run that extracted this". It switches to Extraction and selects that run, by calling `job(n)` directly, so it works even if the run isn't in the loaded list.
- Black and white only. `textContent` and `createElement` only.

- [ ] **Step 1: Write the failing tests:**
  - `test_graph_snapshot` / `test_graph_view`: the memory dict and memory nodes include `job_id`.
  - `test_api_exposes_only_read_methods`: replaces the old three-method test; the public names are exactly `{"initial", "view", "refresh", "jobs", "job"}`.
  - `test_api_jobs_and_job`: with a fake client recording calls, `jobs({"project": "3", "status": "pending", "flagged": True, "before": None})` calls the client with `project_id=3, status="pending", flagged=True`. `"all"` maps to `None`. `job("12")` calls `job(12)`. Bad filter keys, a bad status, or `job("x")` → `ValueError`.
  - The existing page tests (`innerHTML`, greyscale, self-contained) still pass on the new page code.

- [ ] **Step 2: Run them to verify they fail**
  Run: `uv run pytest tests/test_graph_window.py tests/test_graph_view.py tests/test_graph_snapshot.py -v`
  Expected: the new tests FAIL.

- [ ] **Step 3: Implement** the Python changes and the page per "Page behaviour". Then run `node --check src/rem/graph/static/graph.js`.

- [ ] **Step 4: Offscreen smoke test.**
  - Run `uv sync --extra graph`.
  - Write a scratch script under the session scratchpad (not in the repo) that opens the real page with `QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS="--disable-gpu --no-sandbox"` and a fake `Api` client returning one done run with an added fact, one flagged run (noise + downgraded) and one failed run.
  - Through `evaluate_js`, check:
    - the tab switches;
    - the list shows 3 runs;
    - "flagged only" requests `flagged=true`;
    - selecting the flagged run renders the badges `noise` and `downgraded`;
    - the graph-link button switches tabs and selects the run;
    - no JS errors. Never return a cytoscape object from `evaluate_js`.
  - Afterwards run plain `uv sync`.
  - Report what you observed.

- [ ] **Step 5: CLAUDE.md**
  - Add an **"Extraction panel"** block under the M9 block: read-only `GET /jobs` and `/jobs/{id}`; pure `rem/review.py`; the seven rule names and that they are hints; outcome kinds; relations looked up through the store; the flagged paging cap of 2,000; the `Api` surface.
  - In Commands, mention the Extraction tab of `rem graph`.

- [ ] **Step 6: Run the full suite**
  Run: `uv run pytest`
  Expected: PASS.

- [ ] **Step 7: Commit**
  `git commit -m "feat: Extraction tab in the graph window; docs"`
