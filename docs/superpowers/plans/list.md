# rem: what's left to do

_Written 2026-11-17, after M9 (PR #9). Sources: the design spec's "Later" row, the M9 review notes, and the state of the local install._

## Right now (finishing M9)
1. **Review and merge PR #9.** Then `git checkout master && git pull`.
2. **Restart the daemon after merging** (`systemctl --user restart rem`). The running daemon predates `GET /graph` and answers `rem graph` with a 404.
3. **Configure the LLM.** The real store shows `llm: provider none`, `memories: 0` and 9 extraction jobs stuck `pending`, so the graph stays empty until this is done.
   - Set `[llm]` in the rem config (dev setup: Ollama, `qwen3.5:4b`, `think = false`).
   - Restart the daemon, then check with `rem status` that the jobs drain.
4. **Decide about the CLAUDE.md `## Rules` line** that ended up in commit `0fd4b9b`, and fix its stray line break ("before any \ncode is written").
5. **Housekeeping:** `.claude/` and `plans_completed/` are untracked. Either commit them or add them to `.gitignore`. Add `m9.md` to `plans_completed/` if that habit continues.

## Small M9 follow-ups (optional polish from the reviews), done 2026-11-17 on branch `m9-polish`
Also fixed there: the `rem_home` test fixture fell back to the real `~/.config/rem/config.toml`, so `test_cli_eval_without_llm_exits_3` hung running a real eval once an LLM was configured.

6. ~~**Layout jumps on every toggle.**~~ Done: nodes keep their positions, and new nodes start between their neighbours.
7. ~~**Refresh and filter can race.**~~ Done: the latest refresh always updates the project list and time, then re-renders with the current filters.
8. ~~**"Click to copy" may not copy**~~ Done: a click selects the whole id (Ctrl+C copies), and the clipboard API is still tried.
9. ~~**Unshown field.**~~ Done: each relation row shows its relation id and the other end's entity id.
10. ~~**A dead display still crashes.**~~ Done: the X or Wayland socket is probed before Qt starts.

## Found while doing step 3
- **Entity types aren't limited to the allowed list.** `EntityMention._type_lower` (`src/rem/models.py`) only lowercases the type. qwen3.5:4b produced `entity_type_commit` and `entity_type_pull_request`, which became stored types. Relations already map unknown types to `related_to`; entities should do the same and map unknown types to `concept`. Existing odd types stay until those entities are re-extracted. In the graph window they show as an ellipse, since mixed/other falls back to that shape.

## Next milestone candidates
11. **Extraction-quality panel** in the graph window: a second panel for checking that rem picks the right entities, relations and trust.
12. **Entity and relation labels in the eval dataset.** `rem eval extraction` scores neither today (an M7/M8 known limitation).
13. **Choose the default local model** with `rem eval extraction` (the design's goal is a local model by default).

## Later (from the design spec)
14. **Tool-output ingestion and a system-facts collector,** so rem can learn `tool_observed` facts, not only facts from conversations.
15. **Backfill:** re-extract older memories so they get entities and relations. Pre-M7/M8 extractions replay with none.
16. **Pruning** of stale or unused memories.
17. **Optional Qdrant server mode** through config, if local mode ever gets slow.
18. **Editing from the graph window** (confirm, delete), only through the `Memory` API so history and trust apply.

## Known limitations (no action unless they bite)
- Audit-history gaps after a crash (M7/M8 known limitations in CLAUDE.md).
- bge-small rarely triggers the 0.95 entity merge (postgres/postgresql 0.934, neovim/nvim 0.912).
- Graph performance at thousands of entities is untested.

**Suggested order:** 1–3 first. Without an LLM configured, rem isn't learning anything yet, so the graph has nothing to show.
