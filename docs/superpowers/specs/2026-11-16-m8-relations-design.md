# M8 — typed relationships: design

_Status: approved 2026-11-16. Parent spec: `2026-11-15-rem-design.md` (rev 2), "Graph-ready memory model" and the M8 row. Builds on M7 (entities)._

## Intent
M7 gave rem the named things facts are about (entities). M8 adds the **typed edges between them** — `rem —depends_on→ Qdrant`, `rem —runs_on→ Fedora` — so M9 can draw an Obsidian-style graph whose edges mean something, without a graph database.

The guiding principle (the user's words): **M8 adds relationships to the existing memory architecture; it does not introduce a second independent intelligence or extraction pipeline.** Relations are a by-product of the extraction rem already does, carry the trust of the memories that evidence them, and are written by the same crash-safe path.

## Goals
- Extend M7's extraction output so each fact can also name typed relations between its own entities.
- Store relations in a third Qdrant collection, `rem_relations__<slug>`, with deterministic ids.
- Relation trust is **derived from evidence memories** (the existing trust model), never decided separately.
- List relations and search them by meaning (CLI + daemon API).
- Replaying a job never duplicates a relation or its history; `rem reembed` carries relations along.

## Non-goals
- No change to memory search ranking (no relation boost) and none to SessionStart context.
- No MCP tool for relations (possible later).
- No second LLM call, no rule-based extractor, no spaCy.
- No multi-hop graph queries; no graph DB; no graph UI (M9).
- No relation labels in the extraction eval dataset (same gap as entities in M7).
- **No backfill.** Memories that exist before M8 get relations only if a later job extracts them again; cached pre-M8 extractions replay with none, and there is no re-extraction command (consistent with rev 1's "no backfill in v1"). M9 draws whatever edges exist by then.

## Anti-goals (would be a failure even if it works)
- A relation becoming `verified` by any route other than a verified evidence memory (an agent must never verify an edge).
- A relation outliving all of its evidence, or pointing at a deleted entity.
- Relations linking two projects' entities.
- A replay producing a second relation, a duplicate evidence id or a second ADD row.

## Approaches considered
1. **Extend the existing extraction call (chosen).** Each fact returns `relations: [{source, type, target}]` whose ends are names from that same fact's `entities`. No extra LLM cost; replay uses the cached extraction unchanged. Cost: a larger JSON output for small local models (qwen3.5:4b) to get right.
2. A second LLM call per job over the written facts and resolved entities. Cleaner prompt and cross-fact edges, but doubles LLM calls, needs a second cache entry, and is exactly the "second pipeline" the principle rules out.
3. Deterministic phrase rules ("Uses X", "X runs on Y"). Free and testable, but only catches phrasings we write, and most facts are about the user, who isn't an entity. Kept as a possible later fallback.

The user picked **1**, and **A** for search scope (listing + semantic search only; over B = a ranking boost and C = an MCP tool).

## Decisions

### Extraction contract
- `EXTRACTION_SCHEMA` fact items gain a required `relations` array of `{source: string, type: string, target: string}` (strict, `additionalProperties: false`), alongside `entities`.
- New prompt rule 9: relations connect two entities **of the same fact**, using their names exactly as given in that fact's `entities`; types from the fixed list; a fact about the user and one thing has no relation (the user is not an entity — **no implicit `user` node**); most facts have `"relations": []`.
- Parsing (`extract._relations`) keeps the fact whatever happens to its relations. A relation is **dropped** (logged, no text) when: it isn't an object; `source` or `target` doesn't normalize (`normalize_name`) to the name of one of that fact's valid mentions; source and target are the same entity (self-loop); it repeats an earlier relation of the fact (same source, type, target).
- **Types:** `RELATION_TYPES = (uses, depends_on, part_of, runs_on, works_on, preferred_over, related_to)`. Type is lowercased and spaces/hyphens become `_`. An unknown type becomes `related_to` (the edge is kept, the label is generic). _Builder's call, flagged:_ the parent spec's `prefers` is replaced by `preferred_over` because, without a user node, a preference can only be expressed between two entities ("Prefers uv over pip" → `uv —preferred_over→ pip`).
- `Candidate.relations: list[RelationMention]` (default `[]`). A cached extraction from before M8 has no `relations` and loads with none. `_collapse_duplicates` pools relations from collapsed copies as it pools entity mentions.

### Storage: `rem_relations__<slug>`
- Dense-only collection; meta key `active_relation_collection` (default `rem_relations__<slug>`); created on open if missing (an M7 store just gets it), as M7 does for entities.
- Vector = embedding of the sentence `"<source name> <type with _ → space> <target name>"` using the entity's canonical `name`. The sentence is stored in the payload so reembed is lossless.
- **Payload:** `{id, source_entity_id, target_entity_id, type, sentence, memory_ids[] (evidence, sorted, de-duplicated), provenance, status, scope, project_id, created_at, updated_at}`. Indexed fields as for entities (`scope`, `project_id`) plus `status`.
- **Id:** `uuid5(RELATION_NS, f"{source_entity_id}|{type}|{target_entity_id}")`. Entity ids already encode the (scope, project) bucket, so the relation is bucketed too. Directed: `A uses B` ≠ `B uses A`.
- Both ends are always in the fact's bucket (a fact's entities are resolved in one bucket), so a relation never crosses projects.
- No `relation_ids` field on memories or entities: relations point at their evidence memories and endpoint entities; finding a memory's relations is a scan of the bucket (personal scale). _Builder's call:_ add a reverse index only if a test shows it's needed.

### Trust (derived, never decided)
- `status = verified` iff at least one evidence memory is verified; else `unverified`.
- `provenance` = the strongest provenance among evidence memories, order `user_confirmed > user_asserted > tool_observed > assistant_generated`.
- Recomputed (`Memory._refresh_relations(memory_id)`) whenever an evidence memory's trust or existence changes: `confirm`, `corroborate`, `apply_confirmations`, `update` (text change or `unverify`), `delete`, and when evidence is added.
- History (`target_kind = relation`): `ADD` when created (with job_id in a job), `CORROBORATE` when status goes unverified → verified, `UPDATE` (old/new status) when it goes verified → unverified, `DELETE` (old sentence) when its last evidence is removed. Adding evidence without a status change writes no row (like entity links).

### Write path (inside `Memory.apply_candidates`, after `_link_mentions`)
1. `_link_mentions` also returns the map normalized mention name → resolved entity id for that candidate (an alias merge maps to the existing entity's id).
2. For each relation of the candidate: **attach it only when the evidence memory's text is this candidate's text**, i.e. the candidate was ADDed, or a memory with the same normalized text already exists (the `own` match, in any action). A candidate that only *restates* another memory (the `restates` link, corroborate or skip) contributes **no** relation, even though its entities still link there as in M7: that memory's text may not state the edge, and it may already be verified, so attaching would let an assistant-origin candidate produce a verified edge. _Builder's call, derived from the anti-goal._ Look both ends up in the map; skip if either is missing or they resolve to the same entity (two spellings merged).
3. Upsert relation (new: embed sentence, evidence `[linked]`; existing: union evidence) → recompute trust → `INSERT OR IGNORE` history → job done.
4. Replay: deterministic id, set-union evidence and an idempotent recompute make a replay a no-op; a missing ADD row is repaired as entities do (`no ADD in history → add it`).
- `ApplyResult.relations: int` counts relation ADD events.

### Lifecycle invariants
- **Delete/update order:** relations are unlinked **before** entities. A relation's evidence memory is linked to both its endpoint entities, so an entity cannot be orphaned while a relation still cites a memory linked to it; unlinking relations first keeps "a relation never points at a deleted entity" true at every crash point.
- `Memory.delete(id)`: remove `id` from every relation's evidence; a relation left with none is deleted (`DELETE` row); others are recomputed. Then entity unlink (M7), then the memory point.
- `Memory.update` that changes the normalized text: same unlink as delete (the text no longer evidences the old edges), as M7 does for entities. `unverify=True` without a text change: recompute only.
- Known gaps accepted the same way as M7: a crash between a relation write and its history row can lose that row (replay sees the state already applied); links and relations still converge.

### Reembed
- The relation collection is copied, synced at finalize and switched with memories and entities (vector = target embedder on `sentence`, payload kept). The pending target records `relation_collection`; a target recorded before M8 gets a full relation copy at finalize. Finalize writes `active_relation_collection` in the same SQLite transaction as the other meta keys, then `Memory.set_relation_col`, then drops the old collection.

### Read API (search scope A)
- `Memory.list_relations(*, scope=None, project_id=None, entity=None)` — `entity` filters to relations whose source or target name/alias matches (normalized). Listing (no `query`) shows **every project's** relations, like CLI `rem list`; each line names its project. _Builder's call._ With both `--query` and `--entity`, the entity filter applies to the search results.
- `Memory.search_relations(query, *, cwd=None, limit=10, scope=None)` — dense search over the relation collection; results below the dense threshold (`0.30`, the same constant memory search uses) are dropped; other projects' relations need `0.55` (same rule as memories). No boost, no keyword search.
- Each result carries source and target **names** (resolved from the entity collection), type, status, provenance, scope, project name, evidence count, created_at, and score for searches.
- Daemon: `GET /relations?query=&entity=&scope=&limit=&cwd=`. Client: `RemClient.list_relations / search_relations`.
- CLI: `rem relations [--query Q] [--entity NAME] [--scope S] [--limit N] [--json]`. Text line: `uv —preferred_over→ pip  [verified, global, 2 evidence]`.
- `rem entities` is unchanged.

## Constraints
- No new dependencies. Python 3.14, existing `VectorStore` ABC (runs on Qdrant local and the in-memory fake).
- Only `Memory` writes the stores; all calls go through `StoreThread`.
- `retrieve` and the search path never write.
- Graph data in Qdrant only; SQLite history gets rows with `target_kind = relation` (exists since M1, no migration).

## Verification
- Extract: valid relations kept; bad end name, self-loop, duplicate, non-object dropped with the fact kept; unknown type → `related_to`; missing `relations` key → none; pre-M8 cached extraction replays.
- Write: one fact with two entities and a relation → one relation, evidence = that memory, ADD row; replay → nothing new; two facts evidencing the same edge → one relation, two evidence ids.
- Trust: unverified evidence → unverified; `rem confirm` the memory → relation verified + CORROBORATE row; MCP `add_memory`/`update_memory` can never yield a verified relation; deleting the verified evidence while unverified evidence remains → back to unverified + UPDATE row.
- Lifecycle: deleting the last evidence deletes the relation (DELETE row) before entities are orphaned; text update unlinks; two projects, same names → two relations.
- Reembed: same ids and payloads in the new relation collection; pre-M8 pending target gets a full copy at finalize.
- Read: list filters by scope/entity; search by meaning returns the edge; other-project threshold; daemon + CLI `--json`.
- **Extraction quality gate (decided by the user):** run `rem eval extraction` on qwen3.5:4b (`think = false`) on `master` and on the M8 branch. If fact recall, precision or JSON validity drops by more than ~5 points, add `[extraction] relations = false` (drops `relations` from the schema and prompt) and default it off for that model; if nothing drops, no switch is added. Record both runs in the plan's ledger.
- Contract tests pass on both vector backends; memory search results and `/context` output are byte-identical before and after M8 for the same store.

## Left to the builder
Exact prompt wording and examples for rule 9, logging text, the list/search output formatting, test layout, whether `list_points` filtering for evidence scans happens in Python or via a payload filter, and whether `search_relations` overfetches.
