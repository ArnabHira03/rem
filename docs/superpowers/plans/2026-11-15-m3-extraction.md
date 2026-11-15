# M3: LLM Extraction Write Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn queued conversation jobs into memories using Mem0's additive write path. An LLM proposes candidate facts, deterministic trust rules set provenance and status, and the `Memory` façade adds each fact, skips it as a duplicate, or corroborates an existing unverified memory. An eval harness measures extraction quality.

**Architecture:**
- `ExtractionHandler` replaces `NoopHandler` and runs three stages:
  1. On the store thread, it loads related memories and any cached extraction for the job.
  2. **Off** the store thread, it redacts the prompt and makes one JSON-schema LLM call.
  3. On the store thread, it saves the extraction, then applies trust rules and writes through `Memory`.
- The LLM output is cached per job in SQLite before any write. A replayed job therefore reuses exactly the same candidates and the M1 deterministic ids hold, because LLMs are not deterministic.
- When no LLM is configured, the worker doesn't start and jobs stay `pending` until one is configured.

**Tech Stack:** `anthropic` Python SDK (structured outputs via `output_config.format`); httpx for OpenAI-compatible and Ollama endpoints; everything from M1 and M2.

**Spec:** `docs/superpowers/specs/2026-11-15-rem-design.md`: the M3 row, "Separate responsibilities", rev-1 decisions 2, 7, 8, 9 and 11, and "Crash-safe write ordering". Completed context: `plans_completed/m1.md` and `plans_completed/m2.md`.

## Global Constraints

- Only `memory.py` writes to the stores. `extract` proposes candidates and never decides trust. `trust` is pure and deterministic, with no LLM and no I/O. `retrieve` never writes.
- Write order is unchanged: vector store upsert or payload set → history `INSERT OR IGNORE` → job done. In M3, the cached extraction row is written before any of these.
- The job key used as `job_id` on memories and history is `f"job:{job.id}"` (a string, matching `Memory.add(job_id: str | None)`).
- Trust table:
  - `user` → `user_asserted`, `verified`
  - `assistant` → `assistant_generated`, `unverified`
  - `tool` → `tool_observed`, `verified`, **only if** `evidence_source` is set; otherwise it is treated as `assistant`
  - The CLI's `confirm` stays `user_confirmed`, `verified` (M1).
- The near-duplicate threshold is cosine **≥ 0.90**, within the same scope and project (config `[extraction] dedup_threshold`). The prompt includes the top **10** related memories, shown with integer ids 0..n-1 (`related_limit`). At most **20** candidates per job are used (`max_candidates`); extras are dropped and logged.
- `[llm] allow_remote = false` by default. A provider whose endpoint is not loopback (`localhost`, `127.0.0.1`, `::1`) needs `allow_remote = true`, or the daemon exits **3** with `RemoteLLMNotAllowed`.
- `[llm] provider = "none"` is the default. The daemon then serves normally, the worker isn't started, jobs stay `pending`, and `status` reports `"llm": {"provider": "none"}`.
- Default development model: `claude-haiku-5-5` (`provider = "anthropic"`), effort `low`. **Never send `temperature`** to Anthropic (it is a 400 on current models), and never prefill.
- API keys come only from environment variables named in config (`api_key_env`; defaults `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`). Never put a key in config files or logs, and never read `.env`.
- Every text that reaches the LLM goes through `rem.redact.redact` first, including memory texts and context turns.
- Deliberate deviation from the spec's "batch embed": candidates are applied one at a time so each sees the previous one's write (in-job dedup). With at most 20 candidates and a local embedder, that's cheap. Batching can come back in M6 if profiling shows a need.
- Tests run offline. Real-provider tests are marked `llm` and excluded by default (`addopts = "-m 'not slow and not llm'"`).
- Existing tests keep passing, with one ruling: `test_server.py::test_ingested_job_reaches_done` becomes `test_ingested_job_stays_pending_without_llm` (Task 6).

## Review Focus

1. **A crash after the LLM call but before or during the writes.** The replay must not call the LLM again; it reuses the cached extraction and ends with exactly one memory and one history row per fact. Test in Task 6.
2. **The LLM proposes a `project`-scope fact for a session with no project** (no cwd). The fact is skipped and logged, never silently stored as `global`. Test in Task 6.
3. **The LLM returns invalid JSON, refuses, or hits `max_tokens`.** The attempt fails with an `LLMError` that names the reason, the job is retried by the queue, and it is never marked `done` with zero memories. Tests in Tasks 2 and 3.
4. **A secret reaches the prompt** through an existing memory's text or an older context turn rather than a new turn. The prompt sent to the LLM is redacted. Test in Task 3.
5. **The same fact twice in one job, or a user restating an assistant fact that's already stored as unverified.** The first case gives one memory. The second gives one CORROBORATE: the memory becomes `user_asserted`/`verified` with the same id, and there's no second memory. Tests in Tasks 4 and 6.

---

## File Structure

```
src/rem/
├── config.py            + LLMConfig, ExtractionConfig on RemConfig (.llm, .extraction)
├── errors.py            + LLMError, RemoteLLMNotAllowed
├── llm/
│   ├── __init__.py
│   ├── base.py          LLM ABC: complete_json(); is_loopback(url)
│   ├── fake.py          FakeLLM (scripted responses, records prompts) — tests and evals
│   ├── anthropic.py     AnthropicLLM (anthropic SDK, structured outputs)
│   ├── openai_compat.py OpenAICompatLLM (httpx, /chat/completions, json_schema); also Ollama
│   └── factory.py       create_llm(cfg) -> LLM | None; check_remote_allowed(cfg); describe_llm(cfg)
├── extract.py           Candidate, Extraction, build_prompt(), extract()
├── trust.py             TrustInfo, Action, assign_trust(), decide()
├── memory.py            + add(project_id=), related(), nearest(), corroborate(), apply_candidates()
├── models.py            + MemoryRecord.corroborated_by
├── storage/sqlite.py    schema v3: extractions table; save_extraction(), get_extraction()
├── daemon/
│   ├── extraction.py    ExtractionHandler
│   ├── service.py       + status()["llm"]
│   └── server.py        build the handler from config; worker only when an LLM is configured
├── evals/
│   ├── __init__.py
│   ├── extraction.py    run_eval(), EvalReport, match_facts()
│   └── data/extraction.jsonl   ~30 synthetic cases
└── cli.py               + `rem eval extraction [--dataset] [--limit] [--json]`
```

---

### Task 1: LLM config, ABC, fake, factory, remote gate

**Files:**
- Modify: `src/rem/config.py`, `src/rem/errors.py`, `pyproject.toml` (`llm` marker; addopts)
- Create: `src/rem/llm/__init__.py`, `src/rem/llm/base.py`, `src/rem/llm/fake.py`, `src/rem/llm/factory.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces:
  - `LLMConfig(provider: Literal["none","anthropic","openai","ollama"]="none", model: str | None=None, base_url: str | None=None, api_key_env: str | None=None, timeout_seconds: float=60, effort: Literal["low","medium","high"]="low", allow_remote: bool=False)`
  - `ExtractionConfig(dedup_threshold: float=0.90, related_limit: int=10, max_candidates: int=20)`
  - `RemConfig.llm: LLMConfig`, `RemConfig.extraction: ExtractionConfig`
  - `class LLMError(RemError)`, `class RemoteLLMNotAllowed(RemError)`
  - `class LLM(ABC)`: `name: str` property (e.g. `"anthropic:claude-haiku-5-5"`) and `complete_json(system: str, user: str, schema: dict) -> dict`, which raises `LLMError`.
  - `is_loopback(url: str) -> bool`
  - `FakeLLM(responses: list[dict | Exception])`: `name == "fake"`; `.calls: list[tuple[str, str]]`. Each call pops the next response and raises it if it's an exception; when the list is exhausted, it raises `LLMError("FakeLLM exhausted")`.
  - `resolved_base_url(cfg: LLMConfig) -> str | None`: ollama defaults to `http://localhost:11434/v1`; anthropic defaults to `None` (the SDK default).
  - `is_remote(cfg: LLMConfig) -> bool`: anthropic is remote unless `base_url` is loopback; openai/ollama are remote unless the resolved base URL is loopback.
  - `check_remote_allowed(cfg: LLMConfig) -> None`
  - `create_llm(cfg: LLMConfig) -> LLM | None`: `None` for `"none"`; calls `check_remote_allowed` first.
  - `describe_llm(cfg: LLMConfig) -> dict`: `{"provider", "model", "remote"}`, or `{"provider": "none"}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_llm_defaults_off(cfg):
    assert cfg.llm.provider == "none" and cfg.llm.allow_remote is False
    assert cfg.extraction.dedup_threshold == 0.90 and cfg.extraction.related_limit == 10
    assert create_llm(cfg.llm) is None and describe_llm(cfg.llm) == {"provider": "none"}

@pytest.mark.parametrize("url,ok", [("http://localhost:11434/v1", True), ("http://127.0.0.1:8080", True),
                                    ("http://[::1]:8000/v1", True), ("https://api.openai.com/v1", False)])
def test_is_loopback(url, ok): assert is_loopback(url) is ok

def test_anthropic_requires_allow_remote():
    with pytest.raises(RemoteLLMNotAllowed, match="allow_remote"):
        check_remote_allowed(LLMConfig(provider="anthropic", model="claude-haiku-5-5"))
    check_remote_allowed(LLMConfig(provider="anthropic", model="claude-haiku-5-5", allow_remote=True))

def test_ollama_is_local_by_default():
    c = LLMConfig(provider="ollama", model="qwen3:4b")
    assert is_remote(c) is False and resolved_base_url(c) == "http://localhost:11434/v1"
    check_remote_allowed(c)

def test_openai_remote_url_refused():
    with pytest.raises(RemoteLLMNotAllowed):
        check_remote_allowed(LLMConfig(provider="openai", model="m", base_url="https://api.example.com/v1"))

def test_provider_without_model_is_invalid():  # ValueError from load_config / model validator
    with pytest.raises(ValueError, match="model"):
        RemConfig(data_dir="/tmp/x", llm={"provider": "anthropic", "allow_remote": True})

def test_fake_llm_scripts_and_records():
    f = FakeLLM([{"facts": []}, LLMError("boom")])
    assert f.complete_json("s", "u", {}) == {"facts": []} and f.calls == [("s", "u")]
    with pytest.raises(LLMError, match="boom"): f.complete_json("s", "u", {})
    with pytest.raises(LLMError, match="exhausted"): f.complete_json("s", "u", {})
```

- [ ] **Step 2: Run them and confirm they fail.** `uv run pytest tests/test_llm.py -v` should fail on imports.
- [ ] **Step 3: Implement the interfaces above.**
  - A pydantic `model_validator` on `LLMConfig` requires `model` when `provider != "none"`.
  - `RemoteLLMNotAllowed` message: `"llm provider '<p>' sends conversation data to <host or 'api.anthropic.com'>; set [llm] allow_remote = true to allow it"`.
  - The factory imports the provider modules lazily, so `anthropic` is only imported when it's used.
- [ ] **Step 4: Update `pyproject.toml`.** Add `"llm: calls a real LLM provider (costs money; opt-in)"` to markers and set `addopts = "-m 'not slow and not llm'"`. Run `uv run pytest`; everything passes.
- [ ] **Step 5: Commit.** `feat: llm config, provider ABC, fake llm, allow_remote gate`

---

### Task 2: Anthropic and OpenAI-compatible providers

**Files:**
- Modify: `pyproject.toml` (run `uv add anthropic`)
- Create: `src/rem/llm/anthropic.py`, `src/rem/llm/openai_compat.py`
- Test: `tests/test_llm_providers.py`

**Interfaces:**
- Consumes: `LLM`, `LLMError`, `LLMConfig`, `resolved_base_url` (Task 1)
- Produces:
  - `AnthropicLLM(cfg: LLMConfig, client=None)`
  - `OpenAICompatLLM(cfg: LLMConfig, transport: httpx.BaseTransport | None = None)`
  - Both implement `complete_json`. `create_llm` maps `anthropic` → `AnthropicLLM` and `openai`/`ollama` → `OpenAICompatLLM`.

- [ ] **Step 1: Write the failing tests.** Inject a stub `client` whose `messages.create(**kw)` records `kw` and returns an object with `.stop_reason` and `.content=[SimpleNamespace(type="text", text=...)]`. For OpenAI, use `httpx.MockTransport`.

```python
def test_anthropic_request_shape():
    stub = StubAnthropic(stop_reason="end_turn", text='{"facts": []}')
    llm = AnthropicLLM(LLMConfig(provider="anthropic", model="claude-haiku-5-5", allow_remote=True), client=stub)
    assert llm.complete_json("SYS", "USER", SCHEMA) == {"facts": []}
    kw = stub.last
    assert kw["model"] == "claude-haiku-5-5" and kw["system"] == "SYS"
    assert kw["messages"] == [{"role": "user", "content": "USER"}]
    assert kw["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert "temperature" not in kw and kw["max_tokens"] == 4096

@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_anthropic_bad_stop_reason_raises(stop):
    llm = AnthropicLLM(ANTH_CFG, client=StubAnthropic(stop_reason=stop, text=""))
    with pytest.raises(LLMError, match=stop): llm.complete_json("s", "u", SCHEMA)

def test_anthropic_sdk_error_becomes_llm_error():  # stub raises anthropic.APIConnectionError(request=...)
    ...  # assert LLMError("anthropic request failed: ...")

def test_openai_compat_request_and_parse():
    seen = {}
    def handler(req):
        seen.update(url=str(req.url), body=json.loads(req.content), auth=req.headers.get("authorization"))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"facts": []}'}, "finish_reason": "stop"}]})
    llm = OpenAICompatLLM(LLMConfig(provider="ollama", model="qwen3:4b"), transport=httpx.MockTransport(handler))
    assert llm.complete_json("S", "U", SCHEMA) == {"facts": []}
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["body"]["response_format"] == {"type": "json_schema", "json_schema": {"name": "extraction", "schema": SCHEMA, "strict": True}}
    assert seen["body"]["messages"][0] == {"role": "system", "content": "S"}

def test_openai_compat_http_error_and_bad_json_raise_llm_error(): ...  # 500 → LLMError("HTTP 500"); content "not json" → LLMError("invalid JSON")
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - `AnthropicLLM`:
    - Builds `anthropic.Anthropic(api_key=os.environ.get(api_key_env or "ANTHROPIC_API_KEY"), base_url=cfg.base_url, timeout=cfg.timeout_seconds, max_retries=2)` unless a `client` is passed in.
    - Calls `messages.create(model=, max_tokens=4096, system=, messages=[user], output_config={"effort": cfg.effort, "format": {"type": "json_schema", "schema": schema}})`.
    - Checks `stop_reason` before reading the content. Any value other than `end_turn` raises `LLMError(f"anthropic stop_reason={...}")`.
    - Parses the first `text` block with `json.loads`.
    - Wraps `anthropic.APIError` in `LLMError("anthropic request failed: …")`.
  - `OpenAICompatLLM`:
    - Uses `httpx.Client(base_url=resolved_base_url(cfg), timeout=cfg.timeout_seconds, transport=transport)`.
    - Sends a `Bearer` header only if the key env var is set.
    - A `finish_reason` other than `stop` raises `LLMError`.
- [ ] **Step 4: Add the opt-in smoke test** `test_anthropic_live` (`@pytest.mark.llm`, skipped if `ANTHROPIC_API_KEY` is unset). It sends one tiny extraction and asserts it returns a dict with a `facts` list. Run `uv run pytest tests/test_llm_providers.py -v`; the offline tests pass.
- [ ] **Step 5: Commit.** `feat: anthropic and openai-compatible llm providers`

---

### Task 3: `extract.py`: prompt, LLM call, validation

**Files:**
- Create: `src/rem/extract.py`
- Test: `tests/test_extract.py`

**Interfaces:**
- Consumes: `LLM`, `FakeLLM`, `LLMError` (Task 1); `Turn` (`rem.storage.queue`); `MemoryRecord`; `redact`
- Produces:
  - `Origin = Literal["user", "assistant", "tool"]`
  - `class Candidate(BaseModel)`: `text: str` (stripped and non-empty), `scope: Scope`, `origin: Origin`, `evidence_source: str | None = None`
  - `class Extraction(BaseModel)`: `candidates: list[Candidate]`, `model: str`
  - `EXTRACTION_SCHEMA: dict`: `{"facts": [{"text", "scope", "origin"}]}`, with every object `additionalProperties: false` and every field required
  - `build_prompt(turns: list[Turn], context: list[Turn], related: list[MemoryRecord]) -> tuple[str, str]` returns `(system, user)`, both already redacted
  - `extract(llm: LLM, turns, context, related, max_candidates: int) -> Extraction`

**What the system prompt must say** (adapted from Mem0's `ADDITIVE_EXTRACTION_PROMPT`; the wording is the implementer's, and the eval tunes it):
1. Extract durable facts about the user, their preferences, their system or environment, and their projects. Skip chit-chat, transient task state, and anything a secret placeholder like `[REDACTED:…]` stands for.
2. Write each fact as one standalone sentence in the third person ("Prefers uv over pip").
3. `scope`:
   - `global`: about the person
   - `system`: about this machine or OS
   - `project`: about the current repository
4. `origin`:
   - `user`: the user stated it
   - `assistant`: the assistant asserted or inferred it and the user didn't confirm it
   - `tool`: never, in M3, because tool output isn't ingested yet
5. Don't restate the existing memories listed with integer ids. Only output new or changed information.
6. Return `{"facts": []}` when there's nothing durable.

The user message has three sections: `## Existing memories` (`[0] (global) text`…), `## Earlier conversation` (the context turns as `role: content`), and `## New messages`.

- [ ] **Step 1: Write the failing tests**

```python
def test_prompt_numbers_related_and_orders_sections():
    s, u = build_prompt([Turn("user", "I use fish shell")], [Turn("assistant", "hi")], [rec("Prefers uv over pip")])
    assert "[0] (global) Prefers uv over pip" in u
    assert u.index("## Existing memories") < u.index("## Earlier conversation") < u.index("## New messages")
    assert "user: I use fish shell" in u

def test_prompt_is_redacted_everywhere():
    secret = "sk-ant-" + "a" * 30
    _, u = build_prompt([Turn("user", "ok")], [Turn("user", f"key {secret}")], [rec(f"token {secret}")])
    assert secret not in u and "[REDACTED:anthropic_key]" in u

def test_extract_parses_and_caps():
    facts = [{"text": f"fact {i}", "scope": "global", "origin": "user"} for i in range(25)]
    ex = extract(FakeLLM([{"facts": facts}]), [Turn("user", "x")], [], [], max_candidates=20)
    assert len(ex.candidates) == 20 and ex.model == "fake"

def test_extract_drops_invalid_items_but_keeps_valid():
    out = {"facts": [{"text": "  ", "scope": "global", "origin": "user"},
                     {"text": "Uses Fedora", "scope": "planet", "origin": "user"},
                     {"text": "Uses Fedora 44", "scope": "system", "origin": "user"}]}
    assert [c.text for c in extract(FakeLLM([out]), T, [], [], 20).candidates] == ["Uses Fedora 44"]

def test_extract_missing_facts_key_raises():
    with pytest.raises(LLMError, match="facts"): extract(FakeLLM([{"nope": 1}]), T, [], [], 20)

def test_extract_propagates_llm_error():
    with pytest.raises(LLMError): extract(FakeLLM([LLMError("down")]), T, [], [], 20)
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - `FakeLLM.name` is `"fake"`.
  - Validate each item separately with `Candidate.model_validate`, dropping invalid items with a `log.warning`.
  - A response missing the `facts` list raises `LLMError("extraction response has no 'facts' list")`.
  - Duplicate candidate texts within one response (same `normalize_text`) are collapsed to the first.
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit.** `feat: extraction prompt and candidate validation`

---

### Task 4: `trust.py`: deterministic provenance and write decision

**Files:**
- Create: `src/rem/trust.py`
- Test: `tests/test_trust.py`

**Interfaces:**
- Consumes: `Candidate` (Task 3); `Provenance`, `Status`, `MemoryRecord`
- Produces:
  - `@dataclass(frozen=True) class TrustInfo`: `provenance: Provenance`, `status: Status`, `evidence_source: str | None`
  - `class Action(StrEnum)`: `ADD`, `SKIP`, `CORROBORATE`
  - `assign_trust(c: Candidate) -> TrustInfo`
  - `decide(trust: TrustInfo, nearest: MemoryRecord | None, similarity: float, threshold: float, job_key: str) -> Action`

The decision rules, in order:
1. No `nearest`, or `similarity < threshold` → `ADD`.
2. `nearest.status == unverified` and `trust.status == verified` → `CORROBORATE`.
3. `nearest.corroborated_by == job_key` → `CORROBORATE`. This is a replay of this job's own corroboration; `Memory.corroborate` completes the missing history row.
4. Otherwise → `SKIP`. The memory is already at least as trusted, or the candidate is unverified.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("origin,ev,prov,status", [
    ("user", None, Provenance.USER_ASSERTED, Status.VERIFIED),
    ("assistant", None, Provenance.ASSISTANT_GENERATED, Status.UNVERIFIED),
    ("tool", "pytest output", Provenance.TOOL_OBSERVED, Status.VERIFIED),
    ("tool", None, Provenance.ASSISTANT_GENERATED, Status.UNVERIFIED),  # tool claim without evidence is not trusted
])
def test_assign_trust(origin, ev, prov, status): ...

def test_decide_add_when_nothing_close():      assert decide(USER, None, 0.0, 0.9, "job:1") == Action.ADD
def test_decide_add_below_threshold():         assert decide(USER, unverified_rec, 0.89, 0.9, "job:1") == Action.ADD
def test_decide_corroborates_unverified():     assert decide(USER, unverified_rec, 0.95, 0.9, "job:1") == Action.CORROBORATE
def test_decide_skips_verified_duplicate():    assert decide(USER, verified_rec, 0.99, 0.9, "job:1") == Action.SKIP
def test_unverified_never_corroborates():      assert decide(ASSISTANT, unverified_rec, 0.99, 0.9, "job:1") == Action.SKIP
def test_replay_of_own_corroboration():        assert decide(USER, verified_rec_corroborated_by("job:1"), 0.99, 0.9, "job:1") == Action.CORROBORATE
def test_other_jobs_corroboration_skips():     assert decide(USER, verified_rec_corroborated_by("job:7"), 0.99, 0.9, "job:1") == Action.SKIP
```

- [ ] **Step 2: Run them and confirm they fail.** (`corroborated_by` doesn't exist yet. Add it in this task: `MemoryRecord.corroborated_by: str | None = None` in `models.py`, so old payloads still load.)
- [ ] **Step 3: Implement** `assign_trust`, `decide` and the model field. Add `test_models.py::test_payload_without_corroborated_by_loads`.
- [ ] **Step 4: Run** `uv run pytest tests/test_trust.py tests/test_models.py -v`. Everything passes.
- [ ] **Step 5: Commit.** `feat: deterministic trust rules and corroboration decision`

---

### Task 5: Memory façade additions + cached extractions (schema v3)

**Files:**
- Modify: `src/rem/memory.py`, `src/rem/storage/sqlite.py`
- Test: `tests/test_memory_extraction.py`, `tests/test_schema_v3.py`

**Interfaces:**
- Consumes: `Candidate`, `Extraction` (Task 3); `TrustInfo`, `Action`, `assign_trust`, `decide` (Task 4)
- Produces:
  - `Memory.add(..., project_id: int | None = None, evidence_source: str | None = None)`. For `project` scope, either `cwd` or `project_id` must be given. With `project_id`, the name comes from `db.get_project(project_id)`. An unknown id raises `ValueError("unknown project id N")`.
  - `Memory.related(text: str, project_id: int | None, limit: int) -> list[MemoryRecord]`: vector search over all scopes, ranked with `retrieve.rank` using `project_id` as the current project, returning records only.
  - `Memory.nearest(text: str, scope: Scope, project_id: int | None) -> tuple[MemoryRecord, float] | None`: the top-1 raw cosine match filtered by `scope` and `project_id`.
  - `Memory.corroborate(id: str, *, provenance: Provenance, job_id: str, actor: str) -> MemoryRecord`:
    - It sets `provenance`, `status=verified`, `corroborated_by=job_id` and `updated_at` in one `update_payload`, then writes `INSERT OR IGNORE` history `CORROBORATE` (old `unverified` → new `verified`).
    - If the record already has `corroborated_by == job_id`, it skips the payload write and only makes sure the history row exists.
    - It never downgrades: a `user_confirmed` memory keeps its provenance.
  - `@dataclass ApplyResult`: `added: list[str]`, `corroborated: list[str]`, `skipped: int`, `dropped: int`
  - `Memory.apply_candidates(candidates: list[Candidate], *, job_key: str, project_id: int | None, session_id: str | None, source: str, threshold: float) -> ApplyResult`. It processes candidates **in order** so later ones see earlier writes. For each candidate:
    - A `project`-scope candidate with `project_id is None` is dropped and logged.
    - Otherwise it calls `assign_trust`, then `nearest`, then `decide`.
    - `ADD` calls `add(..., job_id=job_key, actor="extractor", provenance/status from trust)`.
    - `CORROBORATE` calls `corroborate(nearest.id, provenance=trust.provenance, job_id=job_key, actor="extractor")`.
  - Schema v3 adds `extractions(job_id INTEGER PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE, model TEXT NOT NULL, candidates TEXT NOT NULL, created_at TEXT NOT NULL)`.
    - `RemDB.save_extraction(job_id: int, model: str, candidates_json: str) -> None` uses `INSERT OR IGNORE`, so the first write wins.
    - `RemDB.get_extraction(job_id: int) -> tuple[str, str] | None` returns `(model, candidates_json)`.

- [ ] **Step 1: Write the failing tests** (they use the `mem` fixture with the hashing embedder, so identical text gives cosine 1.0):

```python
def test_add_with_project_id(mem, tmp_path):
    pid = mem.db.upsert_project(str(tmp_path), "proj", None)
    r = mem.add("Uses Postgres 16", scope=Scope.PROJECT, project_id=pid, provenance=P.USER_ASSERTED,
                status=S.VERIFIED, source="claude-code", actor="extractor", job_id="job:1")
    assert r.project_id == pid and r.project_name == "proj"

def test_apply_adds_user_fact_verified(mem):
    res = mem.apply_candidates([cand("Prefers uv over pip", "user")], job_key="job:1", project_id=None,
                               session_id="s", source="claude-code", threshold=0.9)
    rec = mem.get(res.added[0]); assert rec.status == S.VERIFIED and rec.provenance == P.USER_ASSERTED and rec.job_id == "job:1"

def test_user_restatement_corroborates_assistant_fact(mem):
    a = mem.apply_candidates([cand("Project uses Postgres 16", "assistant", scope="global")], job_key="job:1", **KW).added[0]
    res = mem.apply_candidates([cand("Project uses Postgres 16", "user", scope="global")], job_key="job:2", **KW)
    assert res.corroborated == [a] and res.added == []
    rec = mem.get(a); assert rec.status == S.VERIFIED and rec.provenance == P.USER_ASSERTED and rec.corroborated_by == "job:2"
    assert [h.event for h in mem.history(a)] == [HistoryEvent.ADD, HistoryEvent.CORROBORATE]
    assert len(mem.list()) == 1

def test_duplicate_candidates_in_one_job_make_one_memory(mem):
    res = mem.apply_candidates([cand("Uses fish shell", "user"), cand("uses  FISH shell", "user")], job_key="job:1", **KW)
    assert len(res.added) == 1 and res.skipped == 1

def test_project_fact_without_project_is_dropped(mem):
    res = mem.apply_candidates([cand("Uses Django", "user", scope="project")], job_key="job:1", **KW)
    assert res.dropped == 1 and mem.list() == []

def test_apply_replay_is_idempotent(mem):
    cs = [cand("Uses fish shell", "user"), cand("Runs Fedora 44", "assistant", scope="system")]
    first = mem.apply_candidates(cs, job_key="job:3", **KW); again = mem.apply_candidates(cs, job_key="job:3", **KW)
    assert again.added == first.added and len(mem.list()) == 2
    assert all(len(mem.history(i)) == 1 for i in first.added)

def test_corroborate_replay_after_crash_completes_history(mem, monkeypatch):
    # assistant fact; then corroborate with add_history patched to raise once (crash after payload set)
    ...  # replay apply_candidates with same job_key → exactly one CORROBORATE row, still one memory

def test_corroborate_never_downgrades_user_confirmed(mem): ...

def test_related_ranks_current_project_first(mem, tmp_path): ...
def test_nearest_respects_scope_and_project(mem, tmp_path): ...
```

In `tests/test_schema_v3.py`: a v2 database migrates to v3 and existing rows survive; `save_extraction` twice keeps the first value; `get_extraction` on an unknown job returns `None`.

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - Set `SCHEMA_VERSION = 3` and add migration 3.
  - Implement the Memory methods. `apply_candidates` lives in `memory.py`, because it is the only module that coordinates writes; it calls the pure `trust` functions.
- [ ] **Step 4: Run** `uv run pytest`. Everything passes, including the M1 and M2 suites.
- [ ] **Step 5: Commit.** `feat: memory corroborate/nearest/related/apply_candidates; cache extractions (schema v3)`

---

### Task 6: `ExtractionHandler` and daemon wiring

**Files:**
- Create: `src/rem/daemon/extraction.py`
- Modify: `src/rem/daemon/server.py`, `src/rem/daemon/service.py`, `tests/test_server.py`
- Test: `tests/test_extraction_handler.py`

**Interfaces:**
- Consumes: `Worker`/`JobHandler`, `Job`, `Turn` (M2); `extract`, `Extraction` (Task 3); `Memory.related`, `Memory.apply_candidates`, `RemDB.save_extraction`/`get_extraction` (Task 5); `create_llm`, `describe_llm`, `check_remote_allowed` (Task 1)
- Produces:
  - `ExtractionHandler(llm: LLM, settings: ExtractionConfig)`, which implements `async handle(job, turns, context, service)`.
  - `RemService.status()` gains `"llm": describe_llm(cfg.llm)`.
  - `run_daemon(cfg, handler=None)`:
    - When `handler` is `None`, it builds one with `create_llm(cfg.llm)`. `RemoteLLMNotAllowed`, an invalid LLM config and other `RemError`s exit **3**, after the lock is released, as in the existing path.
    - When `create_llm` returns `None`, it serves without starting the worker and logs `"extraction disabled: no [llm] configured; jobs stay pending"`.

`handle` runs these steps:
1. On the store thread: `cached = db.get_extraction(job.id)`.
2. If `cached` is `None`:
   - On the store thread, compute `related = memory.related(<text of the new turns>, job.project_id, settings.related_limit)`.
   - Then `await asyncio.to_thread(extract, llm, turns, context, related, settings.max_candidates)`. **This never runs on the store thread**, so API calls keep being served during an LLM call.
   - Then save the result on the store thread.
3. On the store thread: `memory.apply_candidates(candidates, job_key=f"job:{job.id}", project_id=job.project_id, session_id=job.session_id, source=job.source, threshold=settings.dedup_threshold)`.
4. Log the `ApplyResult` counts at INFO. Raising from any step fails the attempt, and M2's backoff and retry take over.

- [ ] **Step 1: Write the failing tests.** These are async (anyio), using the `cfg` fixture, `RemService(cfg)` and the M2 `wait_until_async` helpers.

```python
async def test_job_becomes_memories(svc, git_dir):
    llm = FakeLLM([{"facts": [{"text": "Prefers uv over pip", "scope": "global", "origin": "user"},
                              {"text": "Uses Django 5", "scope": "project", "origin": "assistant"}]}])
    await svc.ingest("claude-code", "s1", git_dir, [Turn("user", "I prefer uv over pip")])
    await run_worker_until_done(svc, ExtractionHandler(llm, ExtractionConfig()))
    mems = {m.text: m for m in await svc.list(scope=None, status=None, project_id=None)}
    assert mems["Prefers uv over pip"].status == S.VERIFIED
    assert mems["Uses Django 5"].status == S.UNVERIFIED and mems["Uses Django 5"].project_name == git_dir.name

async def test_replay_after_crash_reuses_cached_extraction(svc):
    # First attempt: FakeLLM returns facts, but apply_candidates is patched to raise after the first add.
    # Second attempt: FakeLLM is empty (would raise if called). Job reaches done;
    # exactly the expected memories exist; each has one ADD row; len(llm.calls) == 1.

async def test_llm_failure_retries_and_never_marks_done(svc, clock):
    llm = FakeLLM([LLMError("down")])
    ...  # job is pending with attempts == 1 and last_error contains "down"; no memories; no extraction cached

async def test_project_fact_without_cwd_is_dropped(svc):
    ...  # ingest with cwd=None; candidate scope=project → no memories; job done

async def test_store_thread_not_blocked_during_llm_call(svc):
    # FakeLLM subclass whose complete_json blocks on a threading.Event; while blocked, await svc.list(...) returns.
```

Change in `tests/test_server.py` (the ruling): `test_ingested_job_reaches_done` becomes `test_ingested_job_stays_pending_without_llm`. It asserts that after the debounce, `status["jobs"]["pending"] == 1` and `status["llm"] == {"provider": "none"}`. Add `test_remote_llm_without_allow_remote_exits_3`: spawn the daemon with `[llm] provider = "anthropic"`, `model = "claude-haiku-5-5"`. It must exit with code 3, `daemon.log` must contain `allow_remote`, and the lock must be released.

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement** the handler, the `status` field and the `run_daemon` wiring.
- [ ] **Step 4: Run** `uv run pytest`. Everything passes.
- [ ] **Step 5: Commit.** `feat: extraction handler drives queued jobs into memories`

---

### Task 7: Eval harness, dataset, `rem eval extraction`, docs

**Files:**
- Create: `src/rem/evals/__init__.py`, `src/rem/evals/extraction.py`, `src/rem/evals/data/extraction.jsonl`
- Modify: `src/rem/cli.py`, `CLAUDE.md`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `extract`, `build_prompt` (Task 3); `Memory.apply_candidates` (Task 5); `create_llm` (Task 1); `Embedder` (M1)
- Produces:
  - Dataset line: `{"id": str, "turns": [{"role", "content"}], "context": [...], "existing": [{"text", "scope", "status"}], "expected": [{"text", "scope", "origin"}], "expect_corroborate": [str]}`. `expected` may be empty: that's a "nothing durable" case.
  - `match_facts(predicted: list[str], expected: list[str], embedder: Embedder, threshold: float = 0.80) -> list[tuple[int, int]]`: a greedy one-to-one match by cosine, highest first.
  - `@dataclass EvalReport`: `cases: int`, `precision: float`, `recall: float`, `scope_acc: float`, `origin_acc: float` (over matched pairs), `json_valid_rate: float`, `dedup_acc: float`, `latency_p50: float`, `latency_p95: float`, `failures: list[dict]`
  - `run_eval(llm: LLM, embedder: Embedder, cases: list[dict], cfg: RemConfig) -> EvalReport`. Each case runs against a **fresh in-memory** `Memory` (memory vector store, the given embedder, a temporary data dir) seeded with `existing`, so evals never touch the user's real store or need the daemon.
  - CLI: `rem eval extraction [--dataset PATH] [--limit N] [--json]`.
    - It uses the configured `[llm]` and the remote gate, and runs in-process.
    - It prints a table, or JSON with `--json`, plus the LLM `name`.
    - Exit codes: 0 when the run completes; 3 when no LLM is configured, with the message `"no [llm] configured; set [llm] provider/model"`, or on `RemoteLLMNotAllowed`.

- [ ] **Step 1: Write the failing tests** (FakeLLM and hashing embedder):

```python
def test_match_facts_one_to_one(hashing): assert match_facts(["Uses fish shell"], ["Uses fish shell", "Uses zsh"], hashing) == [(0, 0)]

def test_perfect_fake_scores_one(cfg, hashing):
    cases = [CASE_FISH, CASE_EMPTY]
    llm = FakeLLM([{"facts": CASE_FISH["expected"]}, {"facts": []}])
    r = run_eval(llm, hashing, cases, cfg)
    assert (r.precision, r.recall, r.scope_acc, r.origin_acc, r.json_valid_rate) == (1, 1, 1, 1, 1)

def test_invalid_json_counts_against_validity(cfg, hashing):
    r = run_eval(FakeLLM([LLMError("invalid JSON")]), hashing, [CASE_FISH], cfg)
    assert r.json_valid_rate == 0 and r.recall == 0 and r.failures[0]["id"] == CASE_FISH["id"]

def test_dedup_case_scores_corroboration(cfg, hashing): ...  # existing unverified + user restatement → dedup_acc == 1

def test_shipped_dataset_is_well_formed():
    cases = load_cases(DEFAULT_DATASET)
    assert len(cases) >= 30 and len({c["id"] for c in cases}) == len(cases)
    assert sum(1 for c in cases if not c["expected"]) >= 4 and sum(1 for c in cases if c["expect_corroborate"]) >= 4

def test_cli_eval_without_llm_exits_3(rem_home, capsys): assert main(["eval", "extraction"]) == 3
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Write the dataset.** About 30 **synthetic, non-sensitive** cases covering:
  - user preferences (global)
  - machine facts (system)
  - repository facts (project)
  - assistant-only claims, which must come out with origin `assistant`
  - nothing-durable chit-chat, at least 4 cases
  - restatements of existing memories, which must not be re-extracted
  - corroboration of existing unverified memories, at least 4 cases
  - a turn containing a `[REDACTED:…]` placeholder, which must not be extracted
- [ ] **Step 4: Implement** `run_eval`, `match_facts`, `load_cases` and the CLI subcommand. A dedup case passes when every `expect_corroborate` text ended up as a CORROBORATE on the seeded memory and no new memory duplicates it.
- [ ] **Step 5: Run** `uv run pytest`. Everything passes. Then do a manual opt-in run. **It costs money, so ask the user first:** with a throwaway `REM_CONFIG` setting `[llm] provider="anthropic"`, `model="claude-haiku-5-5"`, `allow_remote=true`, run `uv run rem eval extraction`. Record the baseline numbers in the commit message.
- [ ] **Step 6: Update `CLAUDE.md`.**
  - Project status: M3 is implemented.
  - Commands: `rem eval extraction`, `uv run pytest -m llm`, and the `[llm]`/`[extraction]` config keys, including `allow_remote`.
  - Under "What exists": extraction is cached per job (schema v3), the LLM runs off the store thread, and with no LLM jobs stay pending.
  - Note that the systemd unit needs `ANTHROPIC_API_KEY` through `systemctl --user edit rem` (`Environment=` or `EnvironmentFile=`), never a key in config.
- [ ] **Step 7: Commit.** `feat: extraction eval harness, dataset, rem eval extraction; docs`

---

## Done criteria

- `uv run pytest` passes offline. `uv run pytest -m llm` passes when `ANTHROPIC_API_KEY` is set.
- With `[llm]` configured, `POST /ingest` → the worker → memories appear with the correct scope, provenance and status. A user restatement corroborates an unverified memory instead of duplicating it.
- `kill -9` during a job, followed by a restart, gives no duplicate memories or history rows and makes no second LLM call for that job.
- With no `[llm]`, the daemon runs and jobs wait in `pending`. A remote provider without `allow_remote` exits 3 with a clear message.
- `rem eval extraction` prints precision, recall, scope and origin accuracy, JSON validity, dedup accuracy and latency.
- Nothing from M4 or later (hooks, SessionStart, MCP, BM25, reembed, entities) is built.
