# M8: Typed Relationships Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Facts can name typed edges between their own entities (`uv —preferred_over→ pip`). Edges are stored in `rem_relations__<slug>`, take their trust from the memories that evidence them, survive replay and reembed, and can be listed and searched by meaning.

**Architecture:** The M7 extraction call also returns `relations` per fact. No second LLM call. `relations.py` holds the pure rules: types, deterministic ids, the sentence that gets embedded, and trust derived from evidence. `Memory` owns every write. `apply_candidates` writes relations after a candidate's entities are linked. `confirm`, `corroborate`, `apply_confirmations`, `update` and `delete` refresh or unlink the relations their memory evidences, always before the entity unlink. Reads are `list_relations` and `search_relations`, exposed through `GET /relations` and `rem relations`. Memory search and `/context` are untouched.

**Tech Stack:** Python 3.14, pydantic, the existing `VectorStore` and embedder, Starlette daemon, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-16-m8-relations-design.md` (approved). The parent spec is `docs/superpowers/specs/2026-11-15-rem-design.md` (M8 row). Read both.

## Global Constraints

- No new runtime dependency. No second LLM call. No graph DB.
- Only `Memory` writes the stores. Every call goes through `StoreThread`. The read paths never write.
- Memory search ranking (`Memory.search`, `retrieve.py`) and `Memory.context` / `POST /context` do not change.
- `RELATION_TYPES = ("uses", "depends_on", "part_of", "runs_on", "works_on", "preferred_over", "related_to")`. An unknown type becomes `related_to`.
- Relation id: `uuid5(RELATION_NS, f"{source_entity_id}|{type}|{target_entity_id}")`. It is directed.
- Relation payload: `{id, source_entity_id, target_entity_id, type, sentence, memory_ids[], provenance, status, scope, project_id, created_at, updated_at}`. `memory_ids` is sorted and de-duplicated.
- Sentence (the embedded text): `f"{source.name} {type.replace('_', ' ')} {target.name}"`, using each entity's canonical `name`.
- Trust: `status = verified` iff any evidence memory is verified. `provenance` is the strongest evidence provenance, ranked `user_confirmed > user_asserted > tool_observed > assistant_generated`.
- History rows for relations use `target_kind = relation`:
  - `ADD`: the relation was created, with the job's `job_id` when written in a job.
  - `CORROBORATE`: unverified → verified.
  - `UPDATE`: verified → unverified (old and new are the status strings).
  - `DELETE`: the old sentence, when the last evidence is gone.
  - Adding evidence with no status change writes no row.
- A relation attaches only to a memory whose normalized text **is the candidate's text** (the candidate was ADDed, or the `own` same-text match). A candidate that only restates another memory adds no relation.
- Relations are unlinked **before** entities in `delete` and in a text-changing `update`.
- Search thresholds reuse `retrieve.T_BASE` (0.30) and `retrieve.T_OTHER` (0.55, for other projects' relations).
- Meta key `active_relation_collection`, default `rem_relations__<slug>`. The collection is dense-only and is created on open if missing.
- No backfill command. A pre-M8 cached extraction replays with no relations.

## Review Focus

1. **A restatement must never produce a verified edge.** An assistant-origin candidate that `restates` a verified user memory carries a relation; no relation is written. Test in Task 4.
2. **Two spellings of one entity in one relation.** The ends merge to the same entity, for example "Postgres" and "postgres", or an alias. No self-loop relation is written. Test in Task 4.
3. **Deleting the verified evidence while unverified evidence remains.** The relation stays but turns unverified, with an `UPDATE` row. Test in Task 5.
4. **A crash between the relation unlink and the entity unlink in `delete`.** The repeated delete converges: no relation points at a missing entity, and no error is raised. Test in Task 5.
5. **An LLM relation naming an entity the fact doesn't list, or naming nothing.** The relation is dropped and the fact and its entities are kept. Test in Task 2.

---

### Task 1: Relation rules and models

**Files:**
- Create: `src/rem/relations.py`
- Modify: `src/rem/models.py` (add `RelationMention`, `RelationRecord`, `RelationResult`; add `Candidate.relations`)
- Test: `tests/test_relations.py`, `tests/test_models.py`

**Interfaces:**
- Produces, in `rem/relations.py`. It is pure: it imports only `rem.models` and the stdlib.
  - `RELATION_NS = uuid.UUID("<fresh uuid4 literal>")` and `RELATION_TYPES` as above.
  - `normalize_type(t: str) -> str`: strip, lowercase, turn spaces and `-` into `_`, unknown becomes `"related_to"`.
  - `relation_id(source_entity_id: str, type: str, target_entity_id: str) -> str`
  - `relation_sentence(source_name: str, type: str, target_name: str) -> str`
  - `derive_trust(evidence: list[MemoryRecord]) -> tuple[Provenance, Status]`: given an empty list it raises `ValueError`, because the caller deletes the relation instead.
- Produces, in `rem/models.py`:
  - `class RelationMention(BaseModel)`:
    - `source: str` and `target: str`, both stripped, 1..80 characters, validated like `EntityMention.name`.
    - `type: str`, run through `normalize_type` in a validator.
  - `class RelationRecord(BaseModel)`: the payload fields above, plus `to_payload()` and `from_payload()` as on `EntityRecord`.
  - `class RelationResult(BaseModel)`: what the API returns.
    - Fields: `id, source, target, type, status, provenance, scope, project_id, project_name: str | None, evidence: int, created_at, score: float | None = None`.
    - `source` and `target` are entity names.
  - `Candidate.relations: list[RelationMention] = []`

- [ ] **Step 1: Write the failing tests**

```python
def test_normalize_type():
    assert normalize_type(" Depends-On ") == "depends_on"
    assert normalize_type("runs on") == "runs_on"
    assert normalize_type("loves") == "related_to"

def test_relation_id_is_deterministic_and_directed():
    assert relation_id("a", "uses", "b") == relation_id("a", "uses", "b")
    assert relation_id("a", "uses", "b") != relation_id("b", "uses", "a")
    assert relation_id("a", "uses", "b") != relation_id("a", "depends_on", "b")

def test_relation_sentence():
    assert relation_sentence("uv", "preferred_over", "pip") == "uv preferred over pip"

def test_derive_trust_any_verified_wins_and_strongest_provenance(...):
    # unverified assistant only -> (ASSISTANT_GENERATED, UNVERIFIED)
    # assistant + user_asserted/verified -> (USER_ASSERTED, VERIFIED)
    # user_asserted + user_confirmed -> (USER_CONFIRMED, VERIFIED)
    # [] -> ValueError

def test_relation_mention_validates():  # in test_models.py
    m = RelationMention(source=" uv ", type="Preferred Over", target="pip")
    assert (m.source, m.type, m.target) == ("uv", "preferred_over", "pip")
    with pytest.raises(ValidationError): RelationMention(source="", type="uses", target="x")
    assert Candidate(text="Uses uv", scope="global", origin="user").relations == []
```

Build the `derive_trust` test records with `MemoryRecord` from `tests.conftest.add`, or construct them directly.

- [ ] **Step 2: Run `uv run pytest tests/test_relations.py tests/test_models.py -v`.** Expected: FAIL (import errors).
- [ ] **Step 3: Implement `relations.py` and the models.** Use the fixed list `[USER_CONFIRMED, USER_ASSERTED, TOOL_OBSERVED, ASSISTANT_GENERATED]` for provenance strength.
- [ ] **Step 4: Run the same tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: relation rules (types, ids, sentence, derived trust) and models`

---

### Task 2: The extraction contract (and the quality gate's baseline)

**Files:**
- Modify: `src/rem/extract.py` (`EXTRACTION_SCHEMA`, `SYSTEM_PROMPT` rule 9 plus the response line, new `_relations`, `extract`)
- Modify: `src/rem/memory.py` (`_collapse_duplicates` pools relations)
- Test: `tests/test_extract.py`, `tests/test_memory_entities.py` (collapse)

**Interfaces:**
- Consumes: `RelationMention`, `normalize_name` (Task 1, M7)
- Produces:
  - `_relations(raw, mentions: list[EntityMention], n: int) -> list[RelationMention]`. A relation is kept only when both ends normalize to the name of one of `mentions`, the ends differ after normalization, and it isn't a duplicate (same normalized source, type and target) of one already kept. It returns the relation with the ends' names as given in the mention.
  - Each `Candidate` from `extract` carries `relations`.

- [ ] **Step 0: Record the baseline before changing anything.**
  - Run: `uv run rem eval extraction --json > <scratchpad>/m8-eval-before.json`. Use the configured local Ollama `qwen3.5:4b` with `think = false`; paid and local eval runs are pre-approved.
  - If it exits 3 (no `[llm]`), stop and report back. Don't guess a config.
  - Note `precision`, `recall` and `json_valid_rate`.
- [ ] **Step 1: Write the failing tests**

```python
def test_extract_keeps_valid_relations(fake_llm_factory):
    # fact with entities uv, pip and relations [{"source":"UV","type":"preferred over","target":"pip"}]
    # -> candidate.relations == [RelationMention(source="uv", type="preferred_over", target="pip")]

def test_extract_drops_bad_relations_keeps_fact_and_entities(fake_llm_factory):  # Review Focus 5
    # relations: [{"source":"uv","type":"uses","target":"cargo"},   # cargo not an entity of the fact
    #             {"source":"uv","type":"uses","target":"uv"},      # self-loop
    #             "nope",                                           # not an object
    #             {"source":"uv","type":"uses","target":"pip"},
    #             {"source":"uv","type":"uses","target":"pip"}]     # duplicate
    # -> one relation (uv uses pip); fact text and both entities kept

def test_extract_missing_relations_key_means_none(fake_llm_factory): ...
def test_schema_requires_relations():
    item = EXTRACTION_SCHEMA["properties"]["facts"]["items"]
    assert "relations" in item["required"]
    assert item["properties"]["relations"]["items"]["required"] == ["source", "type", "target"]
def test_prompt_lists_relation_types_and_no_user_node():
    assert all(t in SYSTEM_PROMPT for t in RELATION_TYPES)
    assert "user is not an entity" in SYSTEM_PROMPT

def test_collapsed_duplicates_pool_relations():  # test_memory_entities.py, next to the mentions test
    # two copies of "Prefers uv over pip" with different relations -> kept candidate has both, de-duped
```

Take the turns and fixtures from the existing `tests/test_extract.py` tests.

- [ ] **Step 2: Run `uv run pytest tests/test_extract.py tests/test_memory_entities.py -v`.** Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
  - Schema: `relations` is an array of `{source, type, target}` strings, strict (`additionalProperties: False`), and required next to `entities`.
  - Rule 9 says three things:
    - A relation joins two entities of the same fact, using their names exactly as listed in that fact's `entities`.
    - Its `type` comes from `RELATION_TYPES`.
    - The user is not an entity, so a fact only about the user and one thing has `"relations": []`. Most facts have none.
  - Add `"relations": [{"source": ..., "type": ..., "target": ...}]` to the response line.
  - Log drops without text, the way `_mentions` does.
  - `_collapse_duplicates` pools relations like mentions: the kept candidate's own first, de-duped by (normalized source, type, normalized target).
- [ ] **Step 4: Run the test files above plus `uv run pytest`.** Expected: PASS, no regressions.
- [ ] **Step 5: Quality gate.**
  - Run: `uv run rem eval extraction --json > <scratchpad>/m8-eval-after.json`.
  - If `precision`, `recall` or `json_valid_rate` dropped by **more than 0.05** against Step 0:
    - Add `ExtractionConfig.relations: bool = True`.
    - Add `extract(..., relations: bool = True)`. When false, the schema has no `relations` key and the prompt has no rule 9. Build both from a helper rather than mutating the module constant.
    - The daemon's extraction handler and the eval pass `cfg.extraction.relations`.
    - Add a test that `relations=False` removes both.
    - Write `relations = false` into the dev config note in CLAUDE.md (Task 8).
  - If nothing dropped, add no switch.
  - Either way, paste both runs' three numbers into the commit message.
- [ ] **Step 6: Commit** `feat: relations in the extraction contract`

---

### Task 3: The relation collection and reads

**Files:**
- Modify: `src/rem/memory.py` (`ACTIVE_RELATION_COLLECTION_KEY`, `self.relation_col` in `__init__`, `get_relation`, `list_relations`, `set_relation_col`, `_relation_result`)
- Test: `tests/test_memory_relations.py` (new)

**Interfaces:**
- Consumes: `RelationRecord`, `RelationResult` (Task 1)
- Produces:
  - `ACTIVE_RELATION_COLLECTION_KEY = "active_relation_collection"`
  - `Memory.relation_col: str`. It is created dense-only on open if missing, right after `entity_col`.
  - `Memory.get_relation(id: str) -> RelationRecord | None`
  - `Memory.relations_for(memory_id: str, scope: Scope, project_id: int | None) -> list[RelationRecord]`: the relations in that bucket whose `memory_ids` contain `memory_id`. It uses `list_points` with a `{"scope", "project_id"}` bucket filter and filters in Python.
  - `Memory.list_relations(*, scope: Scope | None = None, project_id: int | None = None, entity: str | None = None) -> list[RelationResult]`.
    - `entity` matches when the normalized name, or an alias, of either end equals `normalize_name(entity)`.
    - Results are sorted by `(source.casefold(), type, target.casefold(), id)`.
  - `Memory._relation_result(rec: RelationRecord, score: float | None = None) -> RelationResult`.
    - Names come from `get_entity`. A missing entity falls back to the id, which never happens under the invariants.
    - `project_name` comes from `db.get_project`.
  - `Memory.set_relation_col(col: str) -> None`

- [ ] **Step 1: Write the failing tests.** Add an `insert_relation(mem, src, type, dst, memory_ids, status=...)` helper modelled on `insert_entity` in `tests/test_memory_entities.py`. Reuse `insert_entity` for the ends.

```python
def test_relation_collection_exists_dense_only(mem): ...  # startswith("rem_relations__"), not sparse, list_relations() == []
def test_existing_store_gets_relation_collection_on_open(mem): ...  # delete_col then reopen, as the M7 test does
def test_list_relations_resolves_names_and_filters(mem):
    # global uv->pip and project(1) rem->qdrant; list_relations() has both with source/target names;
    # scope=PROJECT, project_id=1 -> only rem->qdrant; entity="PIP" -> only uv->pip; entity matches an alias too
def test_relations_for_memory(mem): ...  # only relations citing that memory, only in its bucket
```

- [ ] **Step 2: Run `uv run pytest tests/test_memory_relations.py -v`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run it, then `uv run pytest`.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: rem_relations collection and relation reads`

---

### Task 4: Writing relations from a job

**Files:**
- Modify: `src/rem/memory.py` (`ApplyResult.relations`, `_link_mentions` return type, `apply_candidates`, new `_write_relations`, new `_evidence`)
- Test: `tests/test_memory_relations.py`

**Interfaces:**
- Consumes: Task 1 rules, Task 3 reads, M7 `_link_mentions`
- Produces:
  - `ApplyResult.relations: int = 0`: the number of relation `ADD` events.
  - `_link_mentions(...) -> tuple[int, dict[str, str]]`: the event count, as before, plus a map from normalized mention name to resolved entity id. Update its one caller.
  - `_write_relations(memory_id: str, relations: list[RelationMention], names_to_ids: dict[str, str], scope: Scope, project_id: int | None, job_key: str, actor: str) -> int`
  - `_evidence(rec: RelationRecord) -> list[MemoryRecord]`: the evidence memories that still exist.
  - `_apply_trust(rec: RelationRecord, actor: str, job_id: str | None) -> RelationRecord | None`: recompute status and provenance from `_evidence`.
    - If there is no evidence, delete the point, write a `DELETE` row and return `None`.
    - On a status change, write `update_payload` first, then the `CORROBORATE` or `UPDATE` row.
    - Otherwise write the payload only if it differs.
    - Task 5 uses this too.

In `apply_candidates`, after `_link_mentions`:
- Call `_write_relations` only when `linked` is the candidate's own text: `action == ADD`, or `own is not None and linked == own.id`.
- For each relation, look up `names_to_ids[normalize_name(end)]` for both ends.
- Skip the relation if an end is missing or both ends are the same id.
- If the relation is new: build the record (evidence `[memory_id]`, `derive_trust`), upsert it with `embed([sentence])[0]`, then write `ADD`.
- If it exists: union the evidence with `update_payload`, then `_apply_trust`.
- On replay, if there's no `ADD` in its history, write the `ADD` row, as `_link_mentions` does.

- [ ] **Step 1: Write the failing tests**

```python
def rcand(text, names, rels, **kw):  # Candidate with entities and relations [{"source","type","target"}]

def test_relation_written_with_evidence_trust_and_add_row(mem):
    res = apply(mem, [rcand("Prefers uv over pip", ["uv", "pip"], [("uv", "preferred_over", "pip")])])
    (r,) = mem.list_relations(); (m,) = mem.list()
    assert (r.source, r.type, r.target, r.evidence, r.status) == ("uv", "preferred_over", "pip", 1, "verified")
    rec = mem.get_relation(r.id)
    assert rec.memory_ids == [m.id] and rec.sentence == "uv preferred over pip"
    assert [(h.target_kind, h.event, h.job_id) for h in mem.db.get_history(r.id)] == [("relation", "ADD", "job:1")]
    assert res.relations == 1

def test_assistant_fact_gives_unverified_relation(mem): ...
def test_replay_writes_nothing_new(mem): ...  # same job twice: one relation, one evidence id, one ADD row
def test_replay_after_crash_before_add_row_repairs_it(mem, monkeypatch): ...  # make add_history raise once for relation ADD
def test_two_facts_same_edge_one_relation_two_evidence(mem): ...  # different texts, both state uv uses python
def test_second_fact_verifies_unverified_relation_with_corroborate_row(mem): ...  # assistant fact then user fact, same edge
def test_restating_candidate_adds_no_relation(mem):  # Review Focus 1
    # add a verified user memory "Uses uv for Python" with entities; then an assistant candidate restating it
    # (restates=its id, different text) carrying relation uv uses python -> list_relations() == []
def test_same_text_existing_memory_gets_the_relation(mem): ...  # own match carries relations
def test_merged_spellings_make_no_self_loop(mem):  # Review Focus 2
    # entities ["Postgres", "postgres"] and relation Postgres->postgres... -> no relation written
def test_same_names_two_projects_two_relations(mem, tmp_path): ...
def test_dropped_project_candidate_writes_no_relation(mem): ...
```

- [ ] **Step 2: Run `uv run pytest tests/test_memory_relations.py -v`.** Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run it, then `uv run pytest`.** Expected: PASS, and the M7 entity tests are unchanged.
- [ ] **Step 5: Commit** `feat: write and link relations when a job's facts are written`

---

### Task 5: Trust refresh and lifecycle

**Files:**
- Modify: `src/rem/memory.py` (`_refresh_relations`, `_unlink_relations`, calls in `confirm`, `corroborate`, `apply_confirmations`, `update`, `delete`)
- Test: `tests/test_memory_relations.py`

**Interfaces:**
- Consumes: `relations_for`, `_apply_trust` (Tasks 3–4)
- Produces:
  - `_refresh_relations(memory_id: str, scope: Scope, project_id: int | None, actor: str, job_id: str | None = None) -> None`: calls `_apply_trust` on each relation the memory evidences.
  - `_unlink_relations(memory_id: str, scope: Scope, project_id: int | None, actor: str) -> None`: removes the id from each such relation's `memory_ids`, then calls `_apply_trust`, which deletes the relation if no evidence is left. It is repeat-safe.

Where each change goes:
- `confirm`, `corroborate` and `apply_confirmations` (only for a memory it actually verified or replays) call `_refresh_relations` **after** their own history row. Pass the job id where there is one.
- `update`:
  - With a text change, call `_unlink_relations` **before** `_unlink` (entities).
  - Without a text change but with `unverify`, call `_refresh_relations` after the upsert.
- `delete` order: `_unlink_relations` → `_unlink` (entities) → memory point delete → history.

The evidence check runs after the memory's payload changes, so the memory's own new status is what counts. For `delete` and a text `update`, `_unlink_relations` removes the memory from the evidence first, so a memory that is about to go never counts.

- [ ] **Step 1: Write the failing tests**

```python
def test_confirming_evidence_verifies_relation(mem): ...  # assistant fact -> unverified relation; mem.confirm(memory) -> verified, CORROBORATE row
def test_mcp_writes_never_verify_a_relation(mem): ...  # update(unverify=True) on the only verified evidence (same text) -> relation unverified, UPDATE row
def test_deleting_verified_evidence_leaves_unverified_relation(mem):  # Review Focus 3
def test_deleting_last_evidence_deletes_relation_then_entities(mem):
    # DELETE row with target_kind relation and old sentence; entities gone too; no relation points at a missing entity
def test_delete_converges_after_crash_between_relation_and_entity_unlink(mem, monkeypatch):  # Review Focus 4
    # make Memory._unlink raise once; delete raises; delete again succeeds; list_relations()==[], list_entities()==[]
def test_text_update_unlinks_relations(mem): ...
def test_same_text_update_keeps_relations(mem): ...
def test_apply_confirmations_verifies_relation(mem): ...  # reuse the setup style of tests/test_confirmation.py
```

- [ ] **Step 2: Run `uv run pytest tests/test_memory_relations.py -v`.** Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run it, then `uv run pytest`.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: relation trust follows its evidence; delete and update unlink relations first`

---

### Task 6: Searching relations by meaning

**Files:**
- Modify: `src/rem/memory.py` (`search_relations`)
- Test: `tests/test_memory_relations.py`

**Interfaces:**
- Consumes: `_relation_result`, `retrieve.T_BASE`, `retrieve.T_OTHER`, `retrieve.overfetch`
- Produces: `Memory.search_relations(query: str, *, cwd: Path | None = None, limit: int = 10, scope: Scope | None = None, entity: str | None = None) -> list[RelationResult]`.
  - It rejects an empty query (`normalize_text`), embeds the query, and searches `relation_col` with `overfetch(limit)` and an optional scope filter.
  - The current project comes from `db.find_project(str(project_root(cwd)))`.
  - The score cutoff is `T_OTHER` for a project relation outside the current project, else `T_BASE`.
  - It applies the `entity` filter as in `list_relations`, sorts by score descending, and cuts to `limit`.
  - `score` is the cosine. There is no boost and no keyword search.

- [ ] **Step 1: Write the failing tests.** Use the `hashing` embedder from the test `cfg`. Assert ordering and thresholds with inserted vectors where the hashing embedder can't separate meanings.

```python
def test_search_relations_finds_by_sentence(mem): ...  # the relation whose sentence matches the query comes first, score set
def test_search_relations_other_project_needs_t_other(mem, monkeypatch): ...  # stub cosine scores 0.4: current project kept, other dropped
def test_search_relations_entity_filter_and_limit(mem): ...
def test_search_relations_does_not_change_memory_search(mem):
    # same store: mem.search(q) results identical before and after relations are written
```

- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them, then `uv run pytest`.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: semantic search over relations`

---

### Task 7: Daemon API, client and CLI

**Files:**
- Modify: `src/rem/daemon/service.py`, `src/rem/daemon/app.py`, `src/rem/client.py`, `src/rem/cli.py`, `src/rem/format.py`
- Test: `tests/test_app.py`, `tests/test_client_cli.py`

**Interfaces:**
- Consumes: `list_relations`, `search_relations`
- Produces:
  - `Service.relations(*, query: str | None, entity: str | None, scope: Scope | None, limit: int, cwd: Path | None) -> list[RelationResult]`: with `query` it calls `search_relations`, otherwise `list_relations(scope=, entity=)`. It runs on the store thread.
  - `GET /relations?query=&entity=&scope=&limit=&cwd=`
    - `limit` defaults to 10 for search and is ignored for listing.
    - A bad `scope` returns 400 through the existing `ValidationError` handler.
  - `RemClient.relations(query=None, entity=None, scope=None, limit=10, cwd=None) -> list[dict]`
  - CLI `rem relations [--query Q] [--entity NAME] [--scope S] [--limit N]`, with `--json` in either position like the other commands. It passes `cwd = Path.cwd()`.
  - `format.relation_line(v: dict) -> str`:
    - Listing: `f"{v['id']}  {v['source']} —{v['type']}→ {v['target']}  [{v['status']}, {where}, {v['evidence']} evidence]"`.
    - `where` is the scope, or `project <project_name>`.
    - A search result also ends with `f"  {v['score']:.2f}"`.
  - `_emit` gets a `"source" in v and "target" in v` branch **before** the `"score"` branch.

- [ ] **Step 1: Write the failing tests.** Copy the patterns of `test_entities_endpoint`, `test_entities_endpoint_lists_entities_sorted` and `test_cli_entities_*`.

```python
async def test_relations_endpoint_empty(api): assert (await api.get("/relations")).json() == []
async def test_relations_endpoint_lists_and_searches(cfg): ...  # write via mem.apply_candidates; GET lists; ?query= returns score
async def test_relations_endpoint_bad_scope_is_400(api): ...
def test_cli_relations_lines_and_json(daemon_env, capsys, monkeypatch):
    rels = [{"id": "r1", "source": "uv", "type": "preferred_over", "target": "pip", "status": "verified",
             "scope": "global", "project_name": None, "evidence": 2, ...}]
    # monkeypatch RemClient.relations; main(["relations"]) prints
    # "r1  uv —preferred_over→ pip  [verified, global, 2 evidence]"; --json round-trips
def test_cli_relations_passes_query_entity_scope(daemon_env, monkeypatch): ...
```

- [ ] **Step 2: Run `uv run pytest tests/test_app.py tests/test_client_cli.py -v`.** Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them, then `uv run pytest`.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: GET /relations and rem relations`

---

### Task 8: Reembed carries relations, and docs

**Files:**
- Modify: `src/rem/reembed.py`, `src/rem/cli.py` (the `--status` text mentions the relation collection)
- Modify: `CLAUDE.md`, `docs/superpowers/specs/2026-11-15-rem-design.md` (M8 row deviations)
- Test: `tests/test_reembed.py`

**Interfaces:**
- Consumes: `relation_col`, `set_relation_col`, `ACTIVE_RELATION_COLLECTION_KEY`
- Produces:
  - `_embed_relations_into(mem, target, col, points, batch=BATCH)`: the vector is `target.embed([payload["sentence"]])` and the payload is kept as is.
  - `_relation_target(mem, pending, slug) -> str`: mirrors `_entity_target`, with key `relation_collection` and base `rem_relations__{slug}`.
  - `run`: records `relation_collection` in `TARGET_KEY` before the copy, creates the collection and copies it.
  - `finalize`:
    - Refuses if the target is the active relation collection.
    - Records a missing `relation_collection` (a pre-M8 pending target), then creates it and `_sync`s it.
    - Sets `ACTIVE_RELATION_COLLECTION_KEY` in the same `transaction()`, calls `mem.set_relation_col`, then `delete_col(old_relation)` last.

- [ ] **Step 1: Write the failing tests.** Mirror the existing entity tests in `tests/test_reembed.py`.

```python
def test_reembed_copies_relations_same_ids_and_payloads(mem): ...
def test_finalize_switches_relation_collection_and_meta(mem): ...  # meta key set, old relation col deleted, list_relations works
def test_finalize_with_pre_m8_pending_target_copies_all_relations(mem): ...  # pending JSON without relation_collection
def test_finalize_syncs_relations_written_after_the_copy(mem): ...
```

- [ ] **Step 2: Run `uv run pytest tests/test_reembed.py -v`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `uv run pytest`, then `uv run pytest -m slow`.** Expected: PASS.
- [ ] **Step 5: Docs.**
  - `CLAUDE.md`:
    - Status line: M8 is implemented.
    - Commands: add `rem relations`.
    - Add an **"Relations (M8)"** invariants block after M7. It covers the contract, the own-text-only rule, derived trust and its history events, the order relations → entities → memory, reembed, the no-backfill gap, and the eval gate's outcome (and the `[extraction] relations` switch if Task 2 added it).
  - Design spec M8 row deviations:
    - `prefers` → `preferred_over`, plus `related_to`.
    - An unknown type becomes `related_to`.
    - Restatements carry no relation.
    - There is no MCP tool and no ranking boost.
    - Point to `2026-11-16-m8-relations-design.md`.
- [ ] **Step 6: Commit** `feat: reembed carries relations; docs: M8 status and invariants`

---

## Ledger (fill in during execution)

- Eval before / after (Task 2): precision, recall and json_valid_rate, plus whether the switch was added.
- Known gaps carried from the design:
  - No backfill.
  - No relation labels in the eval.
  - A crash between a relation write and its status-change history row loses that row (state converges).
