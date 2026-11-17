# Extraction panel: design

_Status: draft 2026-11-17. Builds on M9 (graph window), M3 (extraction jobs), M7/M8 (entities, relations) and the eval labels work (the same failure kinds)._

## Intent
The graph window shows what rem **knows**. The user wants a second view showing **how rem got there**, so they can check whether extraction picks the right things in their **real** conversations.

On 2026-11-17 the real store showed several kinds of bad extraction:
- commit ids and PR URLs stored as facts;
- an assistant's narration stored as facts;
- a subagent report quoted as a memory;
- project facts filed as *system*;
- made-up entity types;
- no relations.

The panel makes these visible run by run, and points out the likely ones.

## Goals
- A second view in the graph window, reached by a tab: **Graph | Extraction**.
- A list of extraction runs (jobs), newest first, that can be filtered by project, by status, and by "flagged only".
- One run shown as **conversation → model's proposal → what rem did**.
- Fixed, named flag rules that mark likely problems on each proposed fact and on the run.
- From the graph's memory panel, a link to the run that extracted that memory.

## Non-goals
- **Writing anything.** The user's decision is "just see it". No delete, confirm, edit, or "save as eval case"; fixes stay with `rem delete` / `rem confirm`.
- Changing extraction, its prompt, trust rules or storage.
- Re-running a job or calling the LLM from the panel.
- Choosing a model.

## Anti-goals
- A flag that looks like a verdict. Every flag shows its rule name and a one-line reason; it is a hint.
- Flags that change data or rank memories. They exist only in the panel's view.
- The window touching storage. All data comes through the daemon over the socket, like `/graph`.
- Loading every conversation at once. The list is light, and one run's detail is fetched on demand.

## Decided by the user
- **Read-only ("just see it").**
- **Show and flag:** simple fixed rules mark suspicious items, and the run list can be filtered to flagged runs.
- **Approach A:** two read-only daemon routes plus a pure Python review module.
  - Rejected: B (put all runs in the `/graph` snapshot, too heavy) and C (the window reads `rem.db` directly, which breaks "only the daemon touches storage").
- **Layout (accepted from the builder's proposal):** a tab at the top; the run list on the left; three columns on the right (conversation | model proposed | stored); a "show the run that extracted this" link from a memory in the graph panel.

## What rem already stores per run (no new storage)
- **jobs:** `id`, `session_pk` → session (source, session id, project), `status`, `attempts`, `last_error`, `created_at`/`updated_at`/`enqueued_at`.
- **messages** with `job_id`: the redacted turns the run read (`IngestStore.messages_for`). The context turns (`context_for`) were also shown to the model.
- **extractions:** `model`, plus `candidates` JSON `{"candidates": [...], "confirmations": [...]}`. A bare list is the pre-M5 form. These are the model's raw proposals, before trust rules and downgrades. A run before M3, or one that failed before extraction, has none.
- **history** rows with `job_id = "job:<id>"`:
  - memory: `ADD`, `CORROBORATE`, `CONFIRM`;
  - entity: `ADD`, `MERGE`;
  - relation: `ADD`, `CORROBORATE`.

  These show what the run wrote.
- **the stores now:** whether a memory or entity still exists, and its current status.

## The review module (`rem/review.py`, pure, no Qt, no I/O)
It takes the raw rows above and builds the views. The daemon does the reads in one `StoreThread` call; the module only computes.

**Per proposed fact, "what rem did"** is derived, never guessed. Facts are processed in order:
1. **Effective origin.** It is recomputed with the same rules `apply_candidates` uses: no user turn in the run → `assistant`; `user` origin with no user sentence naming it (`trust.user_states` over the run's user turns) → `assistant`. When this differs from the proposal, the fact is marked *downgraded*.
2. **Outcome**, one of:
   - **added:** the run has a memory `ADD` row for `memory_id(text, scope, pid, "job:<id>")`. The fact also shows the memory's current state: still there (✓/◌), confirmed since, or deleted since.
   - **corroborated:** a memory `CORROBORATE` row in this run targets a memory whose text normalises to this fact's text, or the memory the proposal's `restates` link names.
   - **no new memory:** neither of the above. The detail says which reason applies, when one is provable:
     - a duplicate of an earlier proposal in the same run (same normalised text);
     - a project-scope fact in a session with no project;
     - "matched an existing memory or skipped".

     It never claims more than the rows show.
3. **Entities and relations:** each proposed mention is shown with its type, next to the entity `ADD`/`MERGE` and relation `ADD`/`CORROBORATE` rows the run wrote (by entity and relation id).

**Flag rules** (fixed, each with a name and a reason line):

| rule | fires when |
|---|---|
| `downgraded` | the proposed origin was `user` but the effective origin is `assistant` |
| `invalid-type` | an entity mention's type is not in `ENTITY_TYPES` |
| `noise` | the fact text contains a URL (`https?://`) or a commit-like token (`\b(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b`: 7 to 40 hex characters with at least one digit and one a-f letter, so `httpx` and `1000000` don't fire), defined here in `review.py` |
| `narration` | an `assistant`-origin fact starts with a past-tense or third-person action verb from a fixed list (fixed, added, modified, changed, updated, removed, created, commits, opens, pushes, merged, renamed…) |
| `scope-hint` | the fact is `system` or `global` but its text names the run's project, or says "this repo", "this project" or "the project" |
| `relation-not-written` | the proposal had relations, but no stored relation cites that fact's memory as evidence. This is looked up through the relation store (`Memory.relations_for`), not by recomputing ids from names, which would miss alias-merged entities. A fact with no memory of its own in this run (outcome not *added*) is never flagged by this rule. |
| `failed` | the job status is `failed` (run level, with `last_error`) |

A run is **flagged** if any of its facts or the run itself carries a flag. Rules apply only to text and stored rows; they never change data.

## Daemon routes (read-only)
- `GET /jobs?project_id=&status=&flagged=&limit=&before=`
  - Newest first. `limit` is 1..200, default 50. `before` is a job id for paging.
  - Each item: `{id, status, created_at, updated_at, project_id, project_name, source, model, n_turns, n_proposed, n_added, n_corroborated, flags: [rule names], error}`.
  - `flagged=true` keeps runs with any flag.
- `GET /jobs/{id}` returns:
  - `run`: the list item;
  - `turns`: `[{role, content, ts}]`;
  - `context_turns`;
  - `facts`: per proposed fact, `{index, text, scope, origin_proposed, origin_effective, restates, entities, relations, outcome: {kind, memory_id?, memory_now?, reason?}, flags: [{rule, reason}]}`;
  - `writes`: the run's history rows (`target_kind`, `event`, `target_id`, `new_value`);
  - `confirmations`: the cached ones, if any.

  An unknown id gives 404.
- `RemClient.jobs(...)` and `RemClient.job(id)`.

## Window
- **Top bar:** **Graph | Extraction** tabs. The project dropdown and ⟳ are shared. The Extraction view adds a status filter (all, done, failed, pending) and a "flagged only" checkbox.
- **Left:** the run list (time, project, status, number of proposed and added facts, a flag count). "Load more" pages with `before`.
- **Right:** three columns for the selected run:
  - **conversation:** turns in order. Context turns are folded under "earlier context", collapsed.
  - **model proposed:** each fact with its scope, origin, entities (name · type) and relations. A downgraded origin shows `user → assistant`.
  - **stored:** the outcome for each fact, lined up with the proposal, then the run's entity and relation writes.

  Flags show as small labelled badges (rule name), with the reason on hover.
- **Graph link:** a memory node's panel shows "show the run that extracted this" when its `job_id` is `job:<id>`. Clicking it switches to the Extraction tab with that run selected.
- **Style:** black and white like the graph view. Flags are a bordered badge. No hue.
- **Text:** all text goes in through `textContent`, never `innerHTML`, the same as the graph view. Conversations hold arbitrary text.

## Constraints
- All the M9 invariants hold:
  - pywebview on Qt;
  - only `GET` routes added;
  - `js_api` exposes only named read methods (the `Api` gains `jobs(filters)` and `job(id)`);
  - no network;
  - the default test suite imports no Qt.
- Turns are already redacted at ingest; the panel shows them as stored.
- No new dependencies.

## Approaches considered
- **A (chosen):** `GET /jobs` and `GET /jobs/{id}`, with a pure `rem/review.py`.
- **B:** all runs inside the `/graph` snapshot. Rejected as too heavy, and it mixes two views.
- **C:** the window reads `rem.db` directly. Rejected: it breaks "only the daemon touches storage".

## Builder's calls after the builder check
- **Relations per fact are looked up through the stores,** never recomputed from names. The daemon reads, for each added memory, the relations citing it (`relations_for`) and the entities it links (`entity_ids`). It passes them to `review.py`. Ids from merged aliases are therefore always right.
- **Paging when "flagged only" is on:** `limit` applies **after** the flag filter. The daemon walks jobs newest-first from `before`, builds each run's flags, and stops when `limit` runs are collected or 2,000 jobs have been scanned. The response carries `next_before` (the last scanned id, or null at the end), so "load more" continues the scan and a short page never hides older flagged runs. History is read in one query per page (`job_id IN (...)`). Without "flagged only", the walk is just `limit` jobs.
- **The graph link needs each memory's job:** `job_id` is added to the graph snapshot's memory fields (`MEMORY_GRAPH_FIELDS`) and passed through `view.py` to memory entries. This adds a field to the payload; storage doesn't change. The page parses `job:<n>`.
- **Pending and processing runs:** a run that hasn't been claimed yet has no turns linked. It shows "not run yet" and folds `processing` into the `pending` filter.

## Left to the builder
- Exact payload field order, CSS, and the exact narration verb list (keep it short and documented in the module).
- Paging details and the default `limit`.
- Test layout: `review.py` unit tests with constructed rows, route tests with the app test client, and window `Api` tests with a fake client.
