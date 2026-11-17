# Eval Entity/Relation Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `rem eval extraction` scores entities and relations against hand-written labels, and its default dataset gains cases built from real extraction failures.

**Architecture:** Expected facts in `extraction.jsonl` gain optional `entities`/`relations` labels. A pure scoring function in `src/rem/evals/extraction.py` compares each case's predicted entity and relation mentions with the labels, per case, and `run_eval` sums the counts into new `EvalReport` fields. The CLI prints them. Fact scoring is untouched.

**Tech Stack:** Python 3.14, stdlib, pytest; the existing `FakeLLM` and `HashingEmbedder` for tests.

**Spec:** `docs/superpowers/specs/2026-11-17-eval-labels-design.md`. Read it in full: "Dataset format", "Scoring", "New real-failure cases", "Builder's calls".

## Global Constraints
- Fact precision, recall, scope accuracy, origin accuracy, JSON validity and dedup accuracy are computed exactly as today. Don't touch that code path except to add the new counting beside it.
- Names are compared with `rem.entities.normalize_name`. Allowed entity types are `rem.models.ENTITY_TYPES`, and allowed relation types are `rem.models.RELATION_TYPES`.
- **A case is labelled** iff every item in `expected` has an `entities` key. `expected: []` counts as labelled. If `relations` is missing on a labelled fact, it means `[]`.
- Entity-type accuracy and invalid-type rate are counted **per mention**. The invalid-type rate covers all valid-JSON cases, labelled or not.
- A relation matches when the source and target match in the same direction (after alias resolution), one-to-one. Its type is scored separately.
- A labelled case whose LLM call failed adds its expected entities and relations to the recall denominators.
- A zero denominator gives `None`, printed as `n/a`.
- The eval stays in-process and never touches the real store or the daemon. No new dependencies. The default test suite calls no real LLM.
- Out of scope: changing `extract()`, the prompt, entity-type validation in `models.py`, worktree/project resolution, or the default model.
- New case texts are paraphrased, with no real commit ids, URLs or personal data. Their ids start with `real-`.

## Review Focus
1. **A custom `--dataset` file from before labels existed** (no `entities` keys): every entity and relation rate is `None`/`n/a`, `labelled_cases == 0`, and fact scores are unchanged. Test: Task 1 `test_unlabelled_cases_skip_label_scores`.
2. **Same name with different cases or spacing** ("PostgreSQL" vs "postgresql", "Node.js" vs "node.js"): it matches through `normalize_name`, and an alias label matches an alias spelling. Test: Task 1 `test_entity_matching_uses_normalized_names_and_aliases`.
3. **A predicted relation in reverse direction** (`pip → uv` when `uv → pip` is labelled): it doesn't match. Test: Task 1 `test_relation_direction_matters`.
4. **The model types one name differently in two facts:** each mention is scored, so type accuracy is 0.5, not 1. Test: Task 1 `test_type_accuracy_and_invalid_rate_are_per_mention`.
5. **A label typo in the shipped dataset** (an invalid type, or a relation end that isn't one of the fact's entities): the dataset test fails, rather than the eval silently scoring against a bad label. Test: Task 2 `test_shipped_dataset_labels_are_valid`.

---

### Task 1: Score entities and relations

**Files:**
- Modify: `src/rem/evals/extraction.py` (new dataclass, helpers, `EvalReport` fields, counting in `run_eval`)
- Modify: `src/rem/cli.py` `_eval` (print the new lines)
- Test: `tests/test_eval.py`

**Interfaces:**
- Produces:
  - `is_labelled(case: dict) -> bool`
  - `@dataclass LabelScore` with int fields `ent_pred, ent_exp, ent_match, type_mentions, type_ok, rel_pred, rel_exp, rel_match, rel_type_ok`, plus `problems: dict`.
  - `score_labels(candidates: list[Candidate], expected: list[dict]) -> LabelScore`, a pure function using `Candidate.entities` (`EntityMention` name/type) and `Candidate.relations` (`RelationMention` source/type/target).
  - `invalid_type_mentions(candidates: list[Candidate]) -> tuple[int, int]`, returning (invalid, total) over all entity mentions.
  - New `EvalReport` fields, appended after `latency_p95` and before `failures`: `entity_precision`, `entity_recall`, `entity_type_acc`, `invalid_type_rate`, `relation_precision`, `relation_recall`, `relation_type_acc` (all `float | None`) and `labelled_cases: int`.
  - Problem keys: `missing_entities`, `extra_entities`, `wrong_entity_type` (`[{name, got, want}]`), `invalid_entity_type` (`[{name, type}]`), `missing_relations`, `extra_relations` (as `"src -type-> tgt"` strings), `wrong_relation_type` (`[{relation, got, want}]`). Each appears only when non-empty and merges into the case's existing `failures[]` entry.

**Algorithm for `score_labels`** (fixed by the spec):
1. Expected entities: one per label across all expected facts, de-duplicated by `normalize_name(name)`. Each has `keys = {normalize(name)} ∪ {normalize(a) for a in aliases}` and its label type.
2. Predicted names: the set of `normalize(m.name)` over all mentions in all candidates. `ent_pred = len(that set)`.
3. Match one-to-one, going through expected entities in label order: take the first unused predicted name that is in the entity's `keys`.
4. `type_mentions` / `type_ok`: for every predicted mention whose normalised name matched, count it once, and count it as ok if `m.type ==` the label type.
5. Relations: resolve each end to a matched expected entity's canonical name when the normalised end is in its keys, otherwise keep the normalised name. Expected relations are `(norm source, norm target, type)`. Match one-to-one on `(source, target)`, and `rel_type_ok` counts matched pairs with equal type.

- [ ] **Step 1: Write the failing tests** in `tests/test_eval.py`. Helper: `lab(text, ents, rels=())` builds an expected fact with `entities=[{"name": n, "type": t}…]` and `relations=[{"source": s, "type": ty, "target": d}…]`. The `said()` helper already forwards any extra keys, so a "perfect" fake just echoes the labelled facts.
  - `test_perfect_labels_score_one`: a labelled `Prefers uv over pip` fact (uv/tool, pip/tool, uv preferred_over pip), plus `CASE_EMPTY`, with the fake echoing it. All seven new rates are `1` (`invalid_type_rate == 0`), `labelled_cases == 2`, and `failures == []`.
  - `test_unlabelled_cases_skip_label_scores`: `CASE_FISH` (no labels) with the fake returning FISH with an entity. All entity/relation precision, recall and type accuracy are `None`, `labelled_cases == 0`, and fact precision and recall are both `1`. `invalid_type_rate` is still computed (`0`).
  - `test_entity_matching_uses_normalized_names_and_aliases`: the label `{"name": "PostgreSQL", "type": "tool", "aliases": ["Postgres"]}`, predicted `postgres`. Then `entity_recall == 1`.
  - `test_missing_and_extra_entities_reported`: the label has uv and pip, and the prediction has uv and npm. Then `entity_precision == entity_recall == 0.5`, with `missing_entities == ["pip"]` and `extra_entities == ["npm"]` on the case's failure.
  - `test_relation_direction_matters`: label `uv preferred_over pip`, predicted `pip preferred_over uv`. Then `relation_precision == relation_recall == 0`.
  - `test_relation_type_scored_separately`: label `alpha depends_on Qdrant`, predicted `alpha uses Qdrant`. Then `relation_recall == 1`, `relation_type_acc == 0`, and the failure has `wrong_relation_type`.
  - `test_type_accuracy_and_invalid_rate_are_per_mention`: two predicted facts, both mentioning uv, typed `tool` and `entity_type_x`, against the label uv/tool. Then `entity_type_acc == 0.5`, `invalid_type_rate == 0.5`, and the failure lists `invalid_entity_type == [{"name": "uv", "type": "entity_type_x"}]`.
  - `test_failed_llm_case_counts_against_label_recall`: a labelled case whose `FakeLLM` raises `LLMError`. Then `entity_recall == 0` and `relation_recall == 0`, while `entity_precision is None`.
  - `test_cli_eval_prints_label_scores` (copy the setup of `test_cli_eval_prints_na_for_empty_rates`): stdout contains lines for `entity precision`, `relation recall` and `invalid types`. Unlabelled runs show `n/a` for the entity and relation lines.

- [ ] **Step 2: Run them to verify they fail**
  Run: `uv run pytest tests/test_eval.py -v`
  Expected: the new tests FAIL (`AttributeError`/`TypeError` on the new fields), and the old tests still PASS.

- [ ] **Step 3: Implement** the interfaces above.
  - In `run_eval`: for each case, compute `labelled = is_labelled(case)`.
    - On `LLMError` for a labelled case: add the expected entity count (de-duplicated by normalised name) to `ent_exp`, and the expected relation count to `rel_exp`.
    - On success: add `invalid_type_mentions(candidates)` to the global counters. If labelled, add `score_labels(...)` counts and merge its `problems` into the case's `problems` dict before the existing `if problems:` append.
  - In `cli._eval`, print the new rates after `dedup accuracy`, in this order and with these labels:
    - `entity precision`
    - `entity recall`
    - `entity types`
    - `invalid types`
    - `relation precision`
    - `relation recall`
    - `relation types`
  - Then print the latency lines and `labelled cases: N`. Widen the label column from 16 to 20 characters for all rate lines. If an existing CLI test pins the old spacing, update it.

- [ ] **Step 4: Run the tests**
  Run: `uv run pytest tests/test_eval.py -v`, then `uv run pytest`
  Expected: all PASS.

- [ ] **Step 5: Commit**
  `git commit -m "feat: eval scores entities and relations against labels"`

---

### Task 2: Label the existing 40 cases

**Files:**
- Modify: `src/rem/evals/data/extraction.jsonl` (add `entities`/`relations` to all 33 expected facts; no other field changes)
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: the label format from the spec and `is_labelled` from Task 1.

Labelling rules (from the spec, applied by hand):
- An entity is a named person, tool, project, language, org, place or concept the fact is about, written as it appears in the fact. The user is never an entity.
- Use only the seven types. Add `aliases` only for a common alternative spelling.
- Relations go only between two entities of the same fact, with one of `RELATION_TYPES`. A fact about the user plus one thing has none. Preferences between two things use `preferred_over`, preferred first.
- A redacted value (`[REDACTED:…]`) is never an entity.

- [ ] **Step 1: Write the failing test** `test_shipped_dataset_labels_are_valid`:
  - every case `is_labelled`;
  - every entity type is in `ENTITY_TYPES`, and every name is 1–80 characters after stripping;
  - every relation type is in `RELATION_TYPES`;
  - each relation's source and target normalise to a name or alias of an entity in **the same fact**;
  - no relation is a self-loop;
  - at least 5 facts carry a relation.

- [ ] **Step 2: Run it to verify it fails**
  Run: `uv run pytest tests/test_eval.py::test_shipped_dataset_labels_are_valid -v`
  Expected: FAIL (cases are not labelled).

- [ ] **Step 3: Add the labels** to all 33 expected facts. Keep each line one JSON object, keep the key order `text, scope, origin, entities, relations`, and change nothing else.

- [ ] **Step 4: Run the tests**
  Run: `uv run pytest tests/test_eval.py -v`
  Expected: PASS, including the existing `test_shipped_dataset_is_well_formed`.

- [ ] **Step 5: Commit**
  `git commit -m "feat: entity and relation labels for the extraction eval dataset"`

---

### Task 3: Real-failure cases, baseline, docs

**Files:**
- Modify: `src/rem/evals/data/extraction.jsonl` (append the `real-` cases)
- Modify: `tests/test_eval.py` (`test_shipped_dataset_is_well_formed` counts)
- Modify: `CLAUDE.md`. Replace the two "The extraction eval has no entity/relation labels, so … quality is not scored" sentences (M7 and M8 known limitations) with a line saying the eval scores entities and relations against labels. In the `rem eval extraction` command line, list the new report fields briefly.

**Interfaces:**
- Consumes: the label format and scoring from Tasks 1–2.

- [ ] **Step 1: Write the failing test.** Extend `test_shipped_dataset_is_well_formed`:
  - `len(cases) >= 50`;
  - at least 10 cases with ids starting `real-`;
  - at least 6 cases expect nothing;
  - the `real-` cases include at least one id containing each of `noise`, `activity`, `quoted`, `scope`, `assistant`, `types` and `relations`;
  - no expected fact (in any case) contains `http`, or a commit-like token: a word matching `[0-9a-f]{7,40}` that contains at least one digit.

- [ ] **Step 2: Run it to verify it fails**
  Run: `uv run pytest tests/test_eval.py::test_shipped_dataset_is_well_formed -v`
  Expected: FAIL (`len(cases) >= 50`).

- [ ] **Step 3: Write 10–12 `real-` cases,** fully labelled, following the spec's list. The kinds are:
  - `noise`: one case keeping only a decision, and one expecting nothing;
  - `activity`: expects nothing;
  - `quoted`: agent or tool text in an assistant turn, plus one real user statement that is extracted;
  - `scope`: a project fact, `project` scope;
  - `assistant`: one case keeping a lasting fact with origin `assistant`, and one narration expecting nothing;
  - `types`: branch, commit, file path, PR; the expected types come from the seven;
  - `relations`: at least three facts with `runs_on` / `depends_on` / `part_of`.

  Rules for the turns:
  - Turns may mention a fake commit id or PR URL, because that's the noise being tested, but expected facts never contain them.
  - Turns are written as a real Claude Code session would look (user prompt, assistant reply), paraphrased, with no real data.

- [ ] **Step 4: Run the full suite**
  Run: `uv run pytest`
  Expected: PASS.

- [ ] **Step 5: Record a baseline**
  Run: `uv run rem eval extraction --json > /tmp/eval-baseline.json`. This uses the configured local `qwen3.5:4b`, and the user has pre-approved it. It can take 30+ minutes, so run it in the background.
  Expected: exit 0, with every new field present and numeric where labelled. Put the summary numbers (all rates, `labelled_cases`, latency) in the task report so they can go in the PR. Don't commit the JSON file.

- [ ] **Step 6: Update CLAUDE.md** as listed under Files.

- [ ] **Step 7: Commit**
  `git commit -m "feat: real-failure cases in the extraction eval; docs"`
