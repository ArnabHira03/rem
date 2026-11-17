# Eval: entity and relation labels, plus real-failure cases: design

_Status: implemented 2026-11-17. Builds on the extraction eval (`src/rem/evals/extraction.py`, `src/rem/evals/data/extraction.jsonl`), M7 (entities) and M8 (relations)._

## Intent
`rem eval extraction` scores the facts a model extracts: precision, recall, scope, origin, JSON validity and dedup. It doesn't score the **entities** (M7) or **relations** (M8) the same LLM call returns. That gap is listed as a known limitation in CLAUDE.md for both milestones.

Running `qwen3.5:4b` on the user's real store (2026-11-17) showed failures the current 40 cases can't catch:
- made-up entity types (`entity_type_commit`, `branch`, `place` for a file path);
- **no relations at all**;
- one-off session details stored as facts ("Commit ID is `82fd0fc`", "PR URL is …/pull/10");
- subagent report text stored as a memory;
- project facts filed as *system*;
- an assistant's own actions stored as verified facts.

This work makes the eval measure all of that, so models (chosen later, not here) are compared on what actually goes wrong.

## Goals
- Every expected fact in the shipped dataset carries hand-written `entities` and `relations` labels.
- The report adds entity and relation scores. All existing scores are still **computed** exactly as today. The new cases join the default dataset, so the headline numbers on the default dataset move once (builder's call, below). Runs on the same dataset stay comparable.
- New cases built from the real failures above, labelled the same way.
- Older or custom `--dataset` files without labels still load and score. Unlabelled cases are left out of the new numbers, not counted as zero.

## Non-goals
- Choosing or changing the default model (deferred by the user).
- Fixing the code bugs the real failures exposed. They stay on `docs/superpowers/plans/list.md`:
  - entity types not limited to the allowed seven (`EntityMention._type_lower`);
  - a git worktree resolving to its own project. This happens before extraction, so this eval can't see it anyway.
- Changing the extraction prompt or `extract()`.
- A pass/fail gate or threshold. The eval stays a report.

## Anti-goals
- Changing how existing fact scores are computed. Precision, recall, scope, origin, JSON validity and dedup must give the same numbers on the same model output.
- The eval touching the real store or the daemon (it already never does; keep it that way).
- Labels that reward a particular wording. Names are compared normalised (`rem.entities.normalize_name`), with optional aliases.

## Decided by the user
- **Approach A:** labels live inside each expected fact, and scoring is per case.
  - Rejected: B (a separate labels file, which drifts) and C (scoring only inside matched fact pairs, which is stricter but punishes a model twice when its wording differs).
- **Add new cases from the real failures**, not only labels.
- **Label rule for session content:** remember **decisions and lasting project facts**; skip **commit ids, PR URLs and other one-off details**. Builder's reading, confirmed in conversation: "how the project is built or what was decided" is a lasting fact, while "currently working on X" is one-off activity and is noise.

## Dataset format (approach A)
Each item in a case's `expected` list gains two optional keys:

```jsonc
{"text": "Prefers uv over pip", "scope": "global", "origin": "user",
 "entities": [{"name": "uv", "type": "tool"}, {"name": "pip", "type": "tool"}],
 "relations": [{"source": "uv", "type": "preferred_over", "target": "pip"}]}
```

- `entities[]`: `{name, type, aliases?}`. `type` is one of the seven entity types (`rem.models.ENTITY_TYPES`). `aliases` is an optional list of other accepted names, e.g. `["Postgres"]` for `PostgreSQL`.
- `relations[]`: `{source, type, target}`. `source` and `target` are names of that fact's own labelled entities (as M8 requires), and `type` is one of `RELATION_TYPES`. If `relations` is missing but `entities` is present, it means `[]`.
- **A case is labelled** iff every item in `expected` has an `entities` key. A case with `expected: []` counts as labelled, with no entities or relations expected: anything predicted there is a false positive.
- A shipped-dataset test checks that every case is labelled, every type is valid, and every relation end names an entity of the same fact.

## Scoring (per case, over labelled cases only)
For each labelled case whose LLM call produced valid JSON:
- **Predicted entities** = the union, over all predicted candidates, of `normalize_name(name)`. Each name also keeps the list of types it was given, one per mention.
- **Expected entities** = the union over expected facts. Each expected entity matches a predicted name equal to its normalised name or to any normalised alias. Matching is one-to-one.
- **Entity precision** = matched / predicted. **Entity recall** = matched / expected. Both are summed over cases (micro-averaged), like the fact scores.
- **Entity-type accuracy** is counted **per mention**: over every mention of a matched entity, the share whose type equals the label's type. A model that types the same name differently across facts is penalised for each wrong mention.
- **Invalid-type rate** is also counted **per mention**: over *all* predicted entity mentions (labelled cases or not), the share whose type isn't in `ENTITY_TYPES`. This measures the made-up-type bug directly.
- Relation types can't be invalid: `RelationMention` already maps an unknown type to `related_to` inside `extract()`. A made-up relation type therefore shows up only as a wrong relation type against a label. This is a known limit, and measuring it would mean changing `extract()`, which is a non-goal.
- **Relations:** each predicted relation becomes `(normalize(source), normalize(target))` with its type, resolved to an expected entity when the name or an alias matches. A predicted relation matches an expected one when the **source and target match in the same direction**, one-to-one.
  - **Relation precision** = matched / predicted, and **relation recall** = matched / expected.
  - **Relation-type accuracy** = among matched relations, the share with the same type.
- A case whose LLM call failed adds its expected entities and relations to the recall denominators, the same way failed cases count against fact recall today.
- Any rate with a zero denominator is `None`, which the CLI prints as `n/a` (existing `_ratio` behaviour).
- **Per-case problems:** `failures[]` entries gain `missing_entities`, `extra_entities`, `wrong_entity_type` (`{name, got, want}`), `invalid_entity_type` (`{name, type}`), `missing_relations`, `extra_relations` and `wrong_relation_type`, with the same "only when non-empty" rule as today.

`EvalReport` gains `entity_precision`, `entity_recall`, `entity_type_acc`, `invalid_type_rate`, `relation_precision`, `relation_recall`, `relation_type_acc` and `labelled_cases` (int). The CLI's text output prints them after the existing lines in the same format, and `--json` includes them.

## New real-failure cases
The new cases are added to the **default** `extraction.jsonl` (builder's call), so `rem eval extraction` measures the real problems with no extra flag. The PR records a fresh baseline on `qwen3.5:4b`. The shipped-dataset test's minimum counts are updated to match.

About 10–12 new cases, each with ids prefixed `real-`, labelled under the rule above. Each failure kind gets at least one case:
1. **Session noise:** a commit/PR exchange with ids and URLs, where the turns also contain one lasting decision. Expected: only the decision, with no commit id and no PR URL. Plus one case where the exchange holds nothing lasting, so expected is `[]`.
2. **Current activity:** "I'm working on the graph view today". Expected: `[]`.
3. **Agent text in a reply:** an assistant turn that quotes a subagent report or tool output containing fact-like sentences. Expected: no facts from the quoted text; a real user statement in the same exchange is still extracted.
4. **Wrong scope:** a project-specific fact stated in a project session. Expected scope `project`, not `system`.
5. **Assistant actions:** "I modified the relation panel to show ids". Expected: the lasting project fact with origin `assistant`, or `[]` if it's only a narration of work. Each case states which.
6. **Made-up types:** facts mentioning a git branch, a commit, a file path and a pull request. Expected types come from the allowed seven (branch → concept, file path → concept, a repo → project, a person or org → person/org).
7. **Relations present:** at least three cases whose facts clearly carry a relation (`X runs_on Y`, `X depends_on Y`, `X part_of Y`), so relation recall isn't measured only on preference facts.

The case texts are paraphrased, not copied from the user's real conversations, and contain no real ids, URLs or personal data beyond what the dataset already uses.

## Approaches considered
- **A (chosen):** labels inside expected facts, scored per case.
- **B:** a separate labels file keyed by case id. It leaves the dataset untouched, but two files have to be kept in step.
- **C:** score entities and relations only inside matched fact pairs. Stricter, but a fact that misses the text-match threshold also loses its entity credit.

## Constraints
- In-process, throwaway stores, never contacts the daemon (existing eval invariants).
- The default test suite runs no real LLM. The scoring is unit-tested with the existing fake LLM in `tests/test_eval.py`.
- Python 3.14 and the stdlib only; no new dependencies.

## Builder's calls after the builder check
- New cases go into the default dataset rather than a second file, and a fresh baseline is recorded.
- Entity-type accuracy and invalid-type rate are counted per mention, not per unique name.
- Made-up relation types can't be measured without changing `extract()` (noted as a known limit).

## Left to the builder
- The exact wording of the new cases and their labels, within the rules above.
- Field ordering and helper names inside `extraction.py`.
- Test layout, extending `tests/test_eval.py`.
- Whether to also run `rem eval extraction` on the configured `qwen3.5:4b` and record the new baseline in the PR. Recommended: running it is pre-approved, it's free and local.
