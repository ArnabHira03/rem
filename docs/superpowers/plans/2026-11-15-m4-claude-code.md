# M4: Claude Code Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect Claude Code to rem. Hooks capture each prompt you type and Claude's final reply into the daemon's queue, where M3 extracts memories from them. At session start, a short block of verified memories is injected, ordered current project, then this machine, then global, and capped at about 400 tokens.

**Architecture:**
- One command, `rem hook claude-code`, reads the hook's JSON from stdin and acts on `hook_event_name`. It imports only light modules (`config`, `client`, `redact`) and always exits 0.
- **Capture.** `UserPromptSubmit` stores the prompt as a *held* turn: it is stored, but no job is created. `Stop` stores the reply and creates the job. Because `claim()` already attaches every unjobbed message of the session to the job it claims, a prompt and its reply are extracted together.
- **Daemon down.** If the daemon can't be reached, the hook writes the (already redacted) turn to a spool directory and starts the daemon in the background without waiting. The daemon drains the spool at startup.
- **Context.** `SessionStart` calls `POST /context` and prints the result. `rem install claude-code` writes the hooks into Claude Code's settings, and is safe to run more than once.

**Tech Stack:** stdlib (`json`, `fcntl`, `tempfile`, `os.replace`), plus httpx, Starlette and everything from M1–M3.

**Spec:** `docs/superpowers/specs/2026-11-15-rem-design.md`: the M4 row, rev-1 decisions 1, 3, 6 and 9, and the risk note "Claude Code hook payloads and transcript JSONL format: confirm before M4". Completed context is in `plans_completed/m1.md`, `m2.md` and `m3.md`.

## Decisions taken with the user (2026-11-15)

- **Turns come from the hook payloads, never from the transcript file.** The docs call the transcript "internal … can break on any release", and in practice it mixes injected text into `user` lines (only 15 of 218 `user` lines in a real session were typed by a person). The payload fields `prompt` (on `UserPromptSubmit`) and `last_assistant_message` (on `Stop`) are documented. Consequence: no backfill and no transcript cursor. The spec's "cursor / backfill" items are dropped for M4.
- **Install target:** user settings `~/.claude/settings.json` by default, honouring `$CLAUDE_CONFIG_DIR`. `--project` installs into `<git root of cwd>/.claude/settings.local.json` instead. A `[claude_code] exclude` list of folders turns rem off completely inside them.
- **Scope-accuracy prompt tuning is NOT in M4.** It is a separate small pass afterwards.

## Verified Claude Code facts this plan relies on (docs fetched 2026-11-15)

- **Every hook's stdin** has `session_id`, `transcript_path`, `cwd`, `hook_event_name` and `prompt_id` (absent before the first input).
  - `UserPromptSubmit` adds `prompt`.
  - `Stop` adds `last_assistant_message` and `stop_hook_active`.
  - `SessionStart` adds `source` (`startup`, `resume`, `clear`, `compact` or `fork`).
  - `SessionEnd` adds `reason`.
- **SessionStart context:** plain stdout with exit 0 becomes context. Text over 10,000 characters is moved to a file. The docs recommend factual, non-imperative wording.
- **Plain stdout on `UserPromptSubmit` also becomes context**, so capture hooks must print nothing.
- **Config shape:** `{"hooks": {"<Event>": [{"matcher": "*", "hooks": [{"type": "command", "command": "...", "timeout": <seconds>}]}]}}`.
  - Hooks are synchronous by default.
  - Exit 0 means OK. Exit 2 blocks. Other codes are non-blocking errors.
  - `SessionEnd` has a 1.5 s budget unless the hook sets `timeout`.
  - Hooks from user, project and local settings merge, and identical handlers run once.

## Global Constraints

- **The hook never breaks Claude Code.** Every path exits 0. Exceptions are caught and one line goes to `<data_dir>/hook.log` (event, error class and message, never prompt or reply text). Nothing is ever written to stderr.
- **Capture hooks (`UserPromptSubmit`, `Stop`, `SessionEnd`) print nothing to stdout.** Only `SessionStart` prints, and only the context text.
- **The hook's own time budget is `[claude_code] hook_budget_seconds = 2.0`** (HTTP timeout to the daemon). The installed `timeout` values are 10 s, which is Claude Code's kill limit, not our target.
- **The hook process must not import** `rem.memory`, `rem.storage.vector.*`, `rem.embed.*`, `qdrant_client` or `fastembed`. A test enforces this.
- **Source and ids:**
  - source is `"claude-code"`; `session_id` comes from the payload; `cwd` from the payload's `cwd`
  - user message_id is `f"cc:{prompt_id}:user"` and assistant message_id is `f"cc:{prompt_id}:assistant"`
  - when `prompt_id` is missing, send no id (the daemon hashes role + content)
- **Turns are redacted in the hook before any spooling** with `rem.redact.redact`. The daemon redacts again at ingest, which is idempotent.
- **Spool files** live at `<data_dir>/spool/<time_ns>-<pid>.json`, with the directory mode 0700 and files 0600. Each file's content is exactly an `/ingest` request body.
- **Exclude:** if `Path(cwd).resolve()` is equal to, or inside, any `Path(p).expanduser().resolve()` in `[claude_code] exclude`, the hook does nothing at all: no capture, no context, no daemon start.
- **SessionStart context:**
  - only **verified** memories, never memories from other projects
  - order: current project, then system, then global; newest `updated_at` first within each section
  - capped at `[claude_code] context_max_chars = 1600` (about 400 tokens), with whole lines only
  - exact format:
    ```
    Facts rem remembers from earlier sessions (verified):
    Project <project_name>:
    - <text>
    This machine:
    - <text>
    About the user:
    - <text>
    ```
    Empty sections are omitted. With no memories, the text is `""` and the hook prints nothing.
- **Installed command:** `f"{sys.executable} -m rem hook claude-code"`, using the absolute path of the interpreter running `rem install`. The marker that identifies rem's entries is the substring `" -m rem hook claude-code"`.
- **Installed events and timeouts:** `SessionStart` 10, `UserPromptSubmit` 10, `Stop` 10, `SessionEnd` 5. All use matcher `"*"`, and all run synchronously.
- **Install writes atomically** (temp file in the same directory, then `os.replace`) after copying the previous file to `<file>.rem-backup`. It preserves every other key and every non-rem hook entry. A settings file that isn't valid JSON is refused with exit 1, and nothing is written.
- Existing tests keep passing. Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **The daemon is down (or starting) when a capture hook fires.** The turn is spooled rather than lost, the hook returns within the budget, the daemon is started in the background, and the next daemon start ingests the spool exactly once. Tests in Tasks 3 and 5.
2. **A prompt whose reply never comes** (the user interrupts, so there's no `Stop`). The held user turn is not lost: the next `Stop` in the session, or `SessionEnd`, puts it into a job. Test in Task 2.
3. **Re-running `rem install`, or a settings.json with the user's own hooks and other settings.** There must be exactly one rem entry per event, the user's entries and keys untouched, and a backup written. A malformed settings.json is refused without writing. Tests in Task 6.
4. **Working inside an excluded folder, or a subfolder of one.** Nothing is captured or injected and no daemon is started. Test in Task 5.
5. **Capture hooks write nothing to stdout,** even on error. On `UserPromptSubmit`, plain stdout would be injected into Claude's context. Test in Task 5.

---

## File Structure

```
src/rem/
├── models.py              + Candidate (moved from extract.py; extract re-exports it)
├── config.py              + ClaudeCodeConfig on RemConfig.claude_code
├── memory.py              + context(cwd, max_chars) -> ContextResult
├── storage/queue.py       ingest(..., enqueue: bool = True): held turns + flush
├── spool.py               write_spool(data_dir, body), drain_spool(data_dir, ingest_fn) -> int
├── client.py              + RemClient.context(); start_daemon_background(cfg); ingest(..., enqueue=)
├── daemon/
│   ├── app.py             POST /context; /ingest accepts "enqueue"
│   ├── service.py         + context(); ingest(..., enqueue)
│   └── server.py          drain spool at startup (before serving)
├── adapters/
│   ├── __init__.py
│   └── claude_code.py     handle_hook(payload, cfg) -> str ; main entry for `rem hook claude-code`
├── install_claude_code.py install(path, command) / uninstall(path) / settings_path(project: bool, cwd)
└── cli.py                 + `rem hook claude-code`, `rem install claude-code [--project] [--uninstall] [--dry-run]`, status line
```

---

### Task 1: Move `Candidate` into `models.py`; add the `[claude_code]` config

**Files:** Modify `src/rem/models.py`, `src/rem/extract.py`, `src/rem/memory.py`, `src/rem/trust.py`, `src/rem/config.py`. Test: `tests/test_models.py`, `tests/test_config.py`.

**Interfaces:**
- Produces:
  - `rem.models.Candidate`, unchanged in shape. `rem.extract.Candidate` still resolves to the same class through a re-export, so cached extractions and old tests keep working.
  - `memory.py` and `trust.py` import it from `rem.models`, so neither imports `rem.extract` any more.
  - `ClaudeCodeConfig(exclude: list[Path] = [], context_max_chars: int = 1600, hook_budget_seconds: float = 2.0)`, available as `RemConfig.claude_code`. `exclude` entries are expanded with `expanduser` when loaded.

- [ ] **Step 1: Write the failing tests.**
  - `test_candidate_lives_in_models`: `from rem.models import Candidate; from rem.extract import Candidate as C2; assert Candidate is C2`.
  - `test_memory_does_not_import_extract`: `import rem.memory`, `import rem.trust`; then check (in a subprocess, `python -c "import rem.memory, sys; print('rem.extract' in sys.modules)"`) that the output is `False`.
  - `test_claude_code_config_defaults_and_expanduser`: `load_config` with TOML `[claude_code] exclude = ["~/work/client"]` gives `cfg.claude_code.exclude == [Path.home()/"work/client"]`, and the defaults are 1600 and 2.0.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Move the class and add the config** (a pydantic `field_validator` handles `expanduser`).
- [ ] **Step 4: Run** `uv run pytest`. All tests pass.
- [ ] **Step 5: Commit.** `refactor: Candidate in models; [claude_code] config`

---

### Task 2: Held turns. `ingest(enqueue=False)` and flushing

**Files:** Modify `src/rem/storage/queue.py`, `src/rem/daemon/service.py`, `src/rem/daemon/app.py`, `src/rem/client.py`. Test: `tests/test_queue.py`, `tests/test_app.py`.

**Interfaces:**
- Produces:
  - `IngestStore.ingest(source, session_id, cwd, turns, enqueue: bool = True) -> IngestResult`
  - `RemService.ingest(..., enqueue=True)`
  - `/ingest` body gains `"enqueue": bool = true`; `turns` may be empty (`[]` is valid, at most 500 as before)
  - `RemClient.ingest(source, session_id, cwd, turns, enqueue=True)`
- Semantics:
  - `enqueue=False`: messages are inserted (with `INSERT OR IGNORE`, redacted), the session is upserted, and **no job is created or pushed back**.
  - `enqueue=True`: a job is created or pushed back when `accepted > 0` **or** the session has unjobbed messages (`job_id IS NULL`). The second case is the flush: an empty `turns` with `enqueue=True` creates a job for held turns.
  - Nothing changes in `claim()`. It already attaches every `job_id IS NULL` message of the session on first claim.

- [ ] **Step 1: Write the failing tests** (in `test_queue.py`, using the existing clock fixture pattern):

```python
def test_held_turn_creates_no_job(store):
    r = store.ingest("claude-code", "s", None, [Turn("user", "I use fish", "cc:p1:user")], enqueue=False)
    assert r.accepted == 1 and r.job_id is None and store.counts()["pending"] == 0

def test_reply_job_takes_held_prompt(store, clock):
    store.ingest("claude-code", "s", None, [Turn("user", "I use fish", "cc:p1:user")], enqueue=False)
    store.ingest("claude-code", "s", None, [Turn("assistant", "Noted.", "cc:p1:assistant")])
    clock.advance(10); job = store.claim()
    assert [t.role for t in store.messages_for(job.id)] == ["user", "assistant"]

def test_empty_enqueue_flushes_held_turns(store, clock):
    store.ingest("claude-code", "s", None, [Turn("user", "I use fish", "cc:p1:user")], enqueue=False)
    r = store.ingest("claude-code", "s", None, [], enqueue=True)
    assert r.job_id is not None and r.accepted == 0
    clock.advance(10); assert len(store.messages_for(store.claim().id)) == 1

def test_empty_enqueue_without_held_turns_is_noop(store):
    assert store.ingest("claude-code", "s", None, [], enqueue=True).job_id is None
```

  In `test_app.py`: `POST /ingest` with `"turns": [], "enqueue": true` returns 202; `"enqueue": "x"` returns 400.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement** the condition: create or extend the job when `enqueue and (accepted or EXISTS(unjobbed messages for session_pk))`, and keep the "only a never-claimed job takes new turns" rule.
- [ ] **Step 4: Run** `uv run pytest`. All tests pass.
- [ ] **Step 5: Commit.** `feat: held turns (enqueue=false) and flush for prompt+reply jobs`

---

### Task 3: Spool for turns captured while the daemon is down

**Files:** Create `src/rem/spool.py`. Modify `src/rem/daemon/server.py`, `src/rem/client.py`. Test: `tests/test_spool.py`, `tests/test_server.py`.

**Interfaces:**
- Produces:
  - `write_spool(data_dir: Path, body: dict) -> Path`: an atomic write (temp file then `os.replace`) to `spool/<time_ns>-<pid>.json`, with dir 0700 and file 0600.
  - `drain_spool(data_dir: Path, ingest: Callable[[dict], None]) -> int`:
    - processes files sorted by name and deletes each after `ingest` returns
    - moves a file that isn't valid JSON, or that `ingest` rejects with `ValueError`, into `spool/bad/`
    - returns the number ingested
    - any other exception stops the drain and leaves the remaining files for the next start
  - `start_daemon_background(cfg) -> None`: spawns `python -m rem daemon` detached exactly like `ensure_daemon` does, but returns immediately. `ensure_daemon` is refactored to use it.
  - `run_daemon` drains the spool once, after `recover_jobs` and before the listener starts. Each body goes through `service.ingest(**body)`, and the count is logged.

- [ ] **Step 1: Write the failing tests:**
  - write then drain ingests in name order, and the spool directory ends up empty
  - permissions are 0700 and 0600
  - a corrupt file goes to `bad/` and the others are still ingested
  - an ingest raising `RuntimeError` leaves that file and later ones in place, and the return value counts only those done
  - in `test_server.py`: write a spool body, start the daemon subprocess (`daemon_env`), and the session's message appears once (check through `/status` job counts or the DB). Restart: no second ingest, because the file is gone.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest`. All tests pass.
- [ ] **Step 5: Commit.** `feat: spool turns while the daemon is down; drain at startup`

---

### Task 4: SessionStart context: `Memory.context` and `POST /context`

**Files:** Modify `src/rem/memory.py`, `src/rem/daemon/service.py`, `src/rem/daemon/app.py`, `src/rem/client.py`. Test: `tests/test_context.py`, `tests/test_app.py`.

**Interfaces:**
- Produces:
  - `@dataclass ContextResult(text: str, count: int)`
  - `Memory.context(cwd: Path | None, max_chars: int) -> ContextResult`: a read-only method that never creates projects. It uses `db.find_project(resolve_project(cwd).root_path)`.
  - `RemService.context(cwd, max_chars)`
  - `POST /context {"cwd": str|null, "max_chars": int (1..10000, default 1600)}` returns `{"text", "count"}`
  - `RemClient.context(cwd, max_chars=1600) -> str`
- Rules: see Global Constraints. When the cwd has no known project, the project section is omitted. The header line counts toward `max_chars`. Lines are added in section order until the next one would exceed `max_chars`, then everything after it is dropped; there is no partial line.

- [ ] **Step 1: Write the failing tests** (using `mem` with the hashing embedder; add memories via `add` with explicit provenance and status):
  - order is project, then system, then global, newest first within a section
  - unverified memories are excluded
  - another project's memories are excluded
  - unknown cwd: no project section, and no project row is created
  - the cap keeps whole lines only, `len(text) <= max_chars`, and `count` equals the number of fact lines
  - no memories gives `ContextResult("", 0)`
  - exact text for a fixture of one memory per section (assert the full string)
  - app: 200 with text; `max_chars=0` returns 400
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest`. All tests pass.
- [ ] **Step 5: Commit.** `feat: verified-memory context for SessionStart (POST /context)`

---

### Task 5: `rem hook claude-code`

**Files:** Create `src/rem/adapters/__init__.py`, `src/rem/adapters/claude_code.py`. Modify `src/rem/cli.py`. Test: `tests/test_hook_claude_code.py`.

**Interfaces:**
- Consumes: `RemClient` (Task 2 `enqueue`, Task 4 `context`), `write_spool`, `start_daemon_background` (Task 3), `redact`, `ClaudeCodeConfig` (Task 1).
- Produces:
  - `handle_hook(payload: dict, cfg: RemConfig, client_factory=RemClient) -> str` returns the stdout text (empty except for SessionStart).
  - `main(stdin=sys.stdin, stdout=sys.stdout) -> int` always returns 0. It parses JSON (bad JSON is logged and returns 0), calls `load_config()` and `handle_hook`, and writes stdout only when the text is non-empty.
  - The CLI routes `rem hook claude-code` to `main` **before** any daemon or `ensure_daemon` code, so the subcommand never auto-starts the daemon through the CLI path.
- Behaviour by event, after the exclude check (excluded means return `""` immediately):
  - `UserPromptSubmit`: if `prompt.strip()` is non-empty, ingest `[{"role": "user", "content": redact(prompt), "message_id": f"cc:{prompt_id}:user"}]` with `enqueue=False`.
  - `Stop`: if `last_assistant_message.strip()` is non-empty, ingest the assistant turn with `enqueue=True`. If it is empty, still ingest `[]` with `enqueue=True`, to flush a held prompt.
  - `SessionEnd`: ingest `[]` with `enqueue=True` (flush).
  - `SessionStart`: call `client.context(cwd, max_chars)` and return its text. On `DaemonUnavailable`, call `start_daemon_background(cfg)` and return `""`.
  - Any other event: return `""`.
  - Capture on `DaemonUnavailable`: `write_spool(cfg.data_dir, <ingest body>)`, then `start_daemon_background(cfg)`. Other exceptions go to `hook.log`.
  - The client timeout is `cfg.claude_code.hook_budget_seconds`.

- [ ] **Step 1: Write the failing tests.** Use a fake client class that records calls or raises `DaemonUnavailable`, plus `monkeypatch` on `start_daemon_background`:
  - each event's ingest call: role, content, message_id, `enqueue`
  - a missing `prompt_id` means no message_id
  - redaction happens before the client or the spool sees the text (an `sk-ant-…` prompt)
  - daemon down: the body is spooled, the daemon start is called once, and the return is `""`
  - SessionStart with the daemon down returns `""` and starts the daemon
  - SessionStart with context returns the exact text
  - excluded cwd and an excluded subfolder: the client is never constructed and the daemon is never started
  - `main()` with invalid JSON on stdin returns 0 and prints nothing
  - `main()` with a client raising `RuntimeError` on `UserPromptSubmit` returns 0, **stdout is empty**, and `hook.log` has one line that doesn't contain the prompt text
  - `test_hook_imports_are_light`: in a subprocess, `python -c "import rem.adapters.claude_code, sys; print(sorted(m for m in sys.modules if m.startswith(('rem.memory','rem.embed','rem.storage.vector','qdrant_client','fastembed'))))"` prints `[]`
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest`. All tests pass.
- [ ] **Step 5: Commit.** `feat: rem hook claude-code (capture, flush, SessionStart context, spool fallback)`

---

### Task 6: `rem install claude-code`

**Files:** Create `src/rem/install_claude_code.py`. Modify `src/rem/cli.py`. Test: `tests/test_install_claude_code.py`.

**Interfaces:**
- Produces:
  - `settings_path(project: bool, cwd: Path) -> Path`:
    - user: `Path(os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude")).expanduser() / "settings.json"`
    - project: `resolve_project(cwd).root_path / ".claude/settings.local.json"`
  - `hook_command() -> str` returns `f"{sys.executable} -m rem hook claude-code"`
  - `merged(settings: dict, command: str) -> dict`: a pure function. It removes every handler whose `command` contains the marker (dropping matcher groups and events left empty), then appends one group `{"matcher": "*", "hooks": [{"type": "command", "command": command, "timeout": T}]}` per event in `{"SessionStart": 10, "UserPromptSubmit": 10, "Stop": 10, "SessionEnd": 5}`.
  - `removed(settings: dict) -> dict`: a pure function, the same removal without appending.
  - `install(path, command, *, dry_run=False) -> str` and `uninstall(path, *, dry_run=False) -> str`:
    - they read the file (missing means `{}`; invalid JSON raises `RemError("…/settings.json is not valid JSON; not modified")`)
    - they compute the result
    - if not `dry_run`: write `<file>.rem-backup` (only if the file existed), then write atomically with `indent=2`, creating the parent directory if needed
    - they return a one-line summary
  - `installed_where() -> list[str]` returns which of `user` or `project` (for cwd) contain the marker. It is used by the `rem status` CLI output as a line `claude_code_hooks: user` (or `none`).
  - CLI: `rem install claude-code [--project] [--uninstall] [--dry-run]`. `--dry-run` prints the resulting JSON. Exit 1 on `RemError`.

- [ ] **Step 1: Write the failing tests** (`tmp_path` settings files, with `CLAUDE_CONFIG_DIR` monkeypatched):
  - a fresh install creates the file with exactly the four events
  - installing twice gives identical content, and still exactly one rem handler per event
  - a user's own `Stop` hook and `"model": "x"` key survive install and uninstall
  - an older rem command (different interpreter path, same marker) is replaced, not duplicated
  - uninstall removes only rem's entries, and removes `hooks` entirely if nothing else is left there
  - invalid JSON raises with the file unchanged and no backup written
  - a backup exists after the second install
  - `--dry-run` leaves the file unchanged and prints JSON
  - `settings_path(project=True)` puts it under the git root
  - `rem status` (CLI-level, with the daemon down) still prints the hooks line
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest`. All tests pass.
- [ ] **Step 5: Commit.** `feat: rem install claude-code (idempotent, atomic, --project, --uninstall)`

---

### Task 7: End-to-end check with real Claude Code, and docs

**Files:** Modify `CLAUDE.md`. Test: `tests/test_hook_e2e.py` (a daemon subprocess plus hook subprocesses fed with recorded payload JSON).

- [ ] **Step 1: Automated end to end, offline (FakeLLM is impossible across processes, so use `[llm] provider = "none"` and check the queue).**
  - Start a daemon with `daemon_env`.
  - Pipe `UserPromptSubmit`, then `Stop`, payloads into `python -m rem hook claude-code`, with `REM_HOME`, `REM_CONFIG` and the socket env set.
  - Assert: one pending job holding exactly two messages (user, then assistant); hook stdout is empty; exit code 0.
  - Then `SessionStart` after seeding a verified memory through `RemClient.add` prints the context text.
  - Then stop the daemon, run a `UserPromptSubmit` hook, and assert that a spool file exists. Start the daemon and the spool is drained into the session.
- [ ] **Step 2: Run** `uv run pytest`. All tests pass.
- [ ] **Step 3: Real Claude Code plus local model** (manual; the user pre-approved real-model runs). Use a throwaway `REM_HOME`, `REM_CONFIG` (`[llm] provider = "ollama"`, `model = "qwen3.5:4b"`, `think = false`), and `REM_SOCKET` under `/tmp/rem-*`.
  - Run `rem install claude-code --dry-run`, then write the hooks into a **temporary settings file** passed with `claude -p --settings <tmp.json>`. The user's real settings are never touched in this check.
  - Run `claude -p` with the haiku model and a short prompt that states a preference ("I prefer tabs over spaces; reply OK") in a temp git repo.
  - Wait for the job to finish. `rem list` shows a verified `user_asserted` memory.
  - A second `claude -p` session in the same repo receives the SessionStart context. Check this through the `--settings` debug log (`--debug-file`) or by asking "what do you know about my indentation preference?" and checking the answer.
  - Record the results in the commit message.
- [ ] **Step 4: Update `CLAUDE.md`:**
  - Status: M4 is implemented.
  - Commands: `rem install claude-code [--project] [--uninstall] [--dry-run]` and `rem hook claude-code` (internal).
  - `[claude_code]` config: `exclude`, `context_max_chars`, `hook_budget_seconds`.
  - Under "What exists": capture comes from hook payloads only (never the transcript), held prompt turns, the spool, and SessionStart context rules.
  - Remove the transcript/backfill wording that no longer applies.
- [ ] **Step 5: Commit.** `test: end-to-end hook flow; docs for M4`

---

## Done criteria

- `uv run pytest` passes offline.
- `rem install claude-code` produces exactly one handler per event, is idempotent, keeps the user's own settings, and can be undone with `--uninstall`.
- With the hooks installed, a real `claude -p` session produces memories through the daemon and the local LLM. A later session receives a context block of verified memories, under 1600 characters, ordered project, then system, then global.
- With the daemon down, turns are spooled and ingested later, and Claude Code is never blocked or shown an error.
- In an excluded folder nothing is captured or injected.
- Nothing from M5 onward (MCP, BM25, entities) and no transcript parsing.
