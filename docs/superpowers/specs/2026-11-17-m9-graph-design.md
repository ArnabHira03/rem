# M9 — graph window: design

_Status: approved 2026-11-17. Parent spec: `2026-11-15-rem-design.md` (rev 2), "Graph-ready memory model" and the M9 row. Builds on M7 (entities) and M8 (relations)._

## Intent
rem extracts entities (M7) and typed relations (M8), but they can only be read as text (`rem entities`, `rem relations`). M9 lets the user **explore what rem knows about them and their projects** as an Obsidian-style graph.

A second panel for **checking extraction quality** (is rem picking the right things?) is planned for later. It is not part of M9, but the layout leaves room for it on the right.

**Change from rev 2:** the parent spec said the daemon serves a web page at `/graph`. The user decided it is a **local desktop window** instead. A browser can't reach the daemon's Unix socket anyway, and a TCP port would weaken the "only this user" guarantee.

## Goals
- `rem graph` opens a native desktop window: entities are nodes, stored relations are edges.
- Read-only: nothing in the window writes to rem.
- Explore by project: one repo (plus global and system), or everything.
- Fully offline, with no network port and no HTTP server.

## Non-goals (M9)
- Editing, confirming or deleting from the window. If added later, it goes only through the `Memory` API.
- The extraction-quality panel.
- Live updates, multi-hop queries, a graph database, a browser page.
- Semantic search from the window.

## Anti-goals (a failure even if it works)
- **Inferring anything.** Every edge drawn is a stored relation or a stored memory↔entity link. The window never connects things by name match, text match or similarity.
- Changing storage: no new collection, table, payload field or history row. No writes at all.
- Hiding stored identity, scope, project or trust. Every entity id, every link, and each record's own scope, project and status stay visible in the panel, even when a node groups several records.
- Opening a TCP port or HTTP server.
- Making the core install heavy: the GUI dependencies are an optional extra.
- The window touching storage: it is a daemon client like the CLI.

## Approaches considered
1. **pywebview on Qt (PySide6 + QtWebEngine), drawing with a vendored JS graph library. Chosen by the user.** A mature force layout, pan/zoom and hover for little code, and a reliable pip install. It costs a large download (~200 MB), so it's an optional extra.
2. pywebview on GTK (system WebKitGTK). The dependency is tiny, but getting PyGObject into the uv venv needs a source build or `--system-site-packages`, which is distro-dependent and fragile.
3. Pure native Qt (QGraphicsView, layout computed in Python). The same weight as 1, with far more hand-written code and a worse graph.

## Decided by the user
- A desktop window, launched by `rem graph`.
- **Layout (the "middle path"):**
  - A slim top bar holds the filters.
  - The graph fills the rest of the window.
  - Hovering a node shows a small tooltip and highlights its neighbours.
  - Clicking a node opens a detail panel on the right. Esc or ✕ closes it.
  - No floating or pinnable cards.
- **Memories as nodes are off by default.** A toggle adds them as small dots, linked to the entities they are stored as mentioning.
- **Same-name entities are grouped into one node (display only).** Entities are stored once per (scope, project) bucket. The view groups the visible records whose `normalized` name is equal into one node, so `uv` in two repos plus a global `uv` is one node. Storage is unchanged. The panel lists every grouped record with its own entity id, scope, project and links. Grouping is by exact `normalized` name only: aliases don't group (an alias already lives on its own record), and there's no cosine or fuzzy grouping.
- **Default view is where `rem graph` was run.** Inside a git repo rem knows, the window opens on that project plus global and system. Outside one, it opens on "all".
- **Memory↔entity edges** (exact rule):
  - Draw an edge from memory M to node N iff some entity record E grouped into N has `E.id ∈ M.entity_ids` **or** `M.id ∈ E.memory_ids`, and both M and E are in the snapshot. Dangling ids are skipped.
  - There is one edge per (M, N).
  - The "or" covers `Memory._link`'s write order: the entity side is written first, so a crash leaves only that side until the job replays.
  - Memories from `rem add` and MCP `add_memory`, and memories whose text was changed, have no links. They show as lone dots.
- **Entity↔entity edges** are stored relations only (`source_entity_id → target_entity_id`, mapped to their nodes). A relation whose two ends are grouped into the same node is drawn as no edge, but it is still listed in the panel.
- **Theme: black and white.**
- **Data:** a snapshot when the window opens, plus a ⟳ refresh button.

## Grouping requirements (user, 2026-11-17)
Visual grouping is kept strictly separate from rem's entity identity and storage. The window must never become an entity-resolution or relationship-inference system.
1. Records are grouped only when their `normalized` names are identical. Aliases, similar names and semantic similarity never merge records.
2. Every original entity id, scope, project, relation and memory association is preserved and shown.
3. **Project and scope filters are applied before grouping.** A filtered view never includes records from unrelated projects.
4. Grouping never creates, modifies or infers a relationship. Every edge drawn is an existing stored relation or memory↔entity association.
5. A relation whose two ends are represented by the same node is drawn as no edge (no self-loop) but stays in the detail panel.
6. The detail panel clearly separates the individual records: one section per member record, each with its own entity id, project, scope, relations and memories.
7. Tests verify that grouping never introduces a relationship and never changes the underlying entity identities.

## Builder's decisions (documented before implementation, as requested)

### Graph library
**cytoscape.js 3.34.4** (MIT), vendored as `src/rem/graph/static/vendor/cytoscape.min.js` with its LICENSE.
- A single file with no dependencies.
- The built-in `cose` force layout is enough, so no layout extensions.
- Supports node shapes, dashed borders, opacity and per-element classes, which is everything a black-and-white theme needs to encode type and trust.
- 3.34.4 is chosen over the newest 3.35.0, which came out a week ago.

### Python dependencies
An optional extra:

```toml
[project.optional-dependencies]
graph = ["pywebview[pyside6]>=6.2.1"]
```

- pywebview's `pyside6` extra pulls in `QtPy` and `PySide6`.
- PySide6 6.12.0 ships `cp310-abi3` manylinux_2_34 x86_64 wheels with `requires_python <3.16,>=3.10`, which covers Python 3.14 (checked 2026-11-17). Fedora 44 has glibc 2.43.
- The core install doesn't change.
- Without the extra, `rem graph` prints `install the graph extra: uv sync --extra graph` and exits 3. It catches the `ImportError` lazily, so no other command imports Qt.
- `pywebview` also depends on `bottle` (its optional local file server). The window loads the page with `html=` (an inline string), so pywebview starts no server.

### Data path
- **Daemon:** `GET /graph?cwd=<dir>` is read-only and built in one `StoreThread` call, `Memory.graph_snapshot(cwd)`. That call scrolls all three collections (`list_points(limit=None)`) and lists projects. It returns raw stored records with no grouping, filtering or inference.
- **Client:** `RemClient.graph(cwd)`.
- **Window process (`rem graph`):**
  - It fetches the snapshot once and keeps it in memory.
  - A pure-Python view builder, `rem.graph.view.build_view(snapshot, project, scopes, show_memories)`, turns it into cytoscape elements plus panel data. It is unit-tested and has no Qt imports.
  - JS calls `api.view(filters)` through pywebview's `js_api` whenever a filter changes, and `api.refresh()` for ⟳, which re-fetches from the daemon.
  - All grouping and filtering logic lives in Python. JS only renders, searches by name and handles interaction.
- **New read-only helper:** `RemDB.list_projects()` (`SELECT id, name, root_path FROM projects`). No schema change.

### `GET /graph` response schema
`cwd` is optional. `current_project_id` is the project rem knows for `cwd`'s git root (`db.find_project(project_root(cwd))`), or `null`. The handler only ever reads. Records are their stored payloads, unchanged except for the memory field subset shown.

```jsonc
{
  "generated_at": "2026-11-17T10:00:00+00:00",
  "current_project_id": 3,                 // or null
  "projects": [ {"id": 3, "name": "rem", "root_path": "/home/…/rem"} ],
  "entities": [                            // EntityRecord payloads, verbatim
    {"id": "…uuid5…", "name": "uv", "normalized": "uv", "type": "tool", "aliases": [],
     "scope": "project", "project_id": 3, "memory_ids": ["…"], "created_at": "…", "updated_at": "…"}
  ],
  "relations": [                           // RelationRecord payloads, verbatim
    {"id": "…", "source_entity_id": "…", "target_entity_id": "…", "type": "uses",
     "sentence": "rem uses uv", "memory_ids": ["…"], "provenance": "user_asserted",
     "status": "verified", "scope": "project", "project_id": 3, "created_at": "…", "updated_at": "…"}
  ],
  "memories": [                            // MemoryRecord subset (no hash/job/session/confirm fields)
    {"id": "…", "text": "rem uses uv for packaging", "scope": "project", "project_id": 3,
     "project_name": "rem", "provenance": "user_asserted", "status": "verified",
     "agent_claim": false, "entity_ids": ["…"], "source": "extraction",
     "created_at": "…", "updated_at": "…"}
  ]
}
```

### View builder output (what `api.view` returns to JS)
`filters = {project: <id> | "all", scopes: ["global","system","project"], show_memories: bool}`. The steps:
1. Keep records in scope: a record passes if its scope is ticked and, when scope is `project`, the project is the selected one (or "all").
2. Group the kept entities by `normalized` into nodes `ent:<normalized>`.
3. Add relation edges `rel:<relation id>` between their ends' nodes, but only when both ends were kept and they aren't the same node.
4. If `show_memories` is on, add memory nodes `mem:<id>` and mention edges `men:<memory id>:<normalized>` using the exact link rule above, among kept records only.

Node data:
- `members`: every grouped record, with its entity id, name, type, aliases, scope, project, and that record's own relations and memories (the panel renders one section per member).
- `type`: the members' type if they all agree, else `mixed`.
- `verified`: true iff any memory with an edge to the node is verified.
- `relations`: each with its id, direction, type, other end, status, provenance, scope and project.
- `memories`: each with its id, text, status, scope, project.
- `n_relations` and `n_memories` for the tooltip.

There are no other node or edge kinds.

### Visual style (black and white)
- White background, black ink and greys only. No hue anywhere.
- **Entity type is shown by shape:** person = ellipse, tool = round-rectangle, project = hexagon, language = diamond, org = rectangle, place = triangle, concept = octagon, mixed or other = ellipse. A small legend sits in the corner.
- **Node trust:** verified is filled black with the label beside it. Unverified is white fill with a dashed black border.
- **Size** scales gently with the number of edges.
- **Relation edges:** verified solid black, unverified dashed grey. Directed arrows. The type label shows only on hover or select.
- **Memory nodes:** small grey dots, filled if verified and hollow if unverified. Mention edges are thin dotted light grey.
- **Hover** dims everything outside the node's neighbourhood to 15% opacity. **Selected** gets a thick black ring.

### Interaction
- **Top bar:**
  - Project dropdown: "all" plus each project, defaulting as decided above.
  - Scope checkboxes for global, system and project.
  - "show memories" toggle.
  - Search box: filters by entity name or alias, case-insensitive substring, in JS. Matches are highlighted and the rest is dimmed. Enter centres on the first match.
  - ⟳ refresh button.
  - Snapshot time.
- **Hover** a node: tooltip (`name · type`, `N relations · N memories`; for a memory node, its first 80 chars) and neighbourhood highlight.
- **Click** an entity node: the right panel opens with:
  - name, type and the list of members (each with project, scope and entity id, small and copyable);
  - aliases;
  - relations (`→ uses → Qdrant ✓`, with the project);
  - memories (full text, ✓/◌, project).

  Clicking a memory node shows its full text, status, provenance, scope, project, id and linked entities. Clicking a relation edge shows its sentence, status, provenance, evidence count and project.
- Clicking a name inside the panel selects that node. Esc or ✕ closes the panel, and clicking empty canvas clears the selection.
- Changing a filter re-runs `api.view` and lays out again, keeping the selection if the node still exists.
- An empty graph shows "rem hasn't extracted any entities here yet".
- `rem graph` blocks the terminal until the window closes. If the daemon can't be reached, it prints the usual client error and exits non-zero before any window opens.

### Code layout
- `src/rem/memory.py`: `graph_snapshot(cwd)`, read-only.
- `src/rem/storage/sqlite.py`: `list_projects()`.
- `src/rem/daemon/service.py`, `src/rem/daemon/app.py`: `GET /graph`.
- `src/rem/client.py`: `graph(cwd)`.
- `src/rem/graph/view.py`: pure view builder.
- `src/rem/graph/window.py`: pywebview window and `Api`. Lazy imports.
- `src/rem/graph/static/`: `graph.html`, `graph.css`, `graph.js`, `vendor/cytoscape.min.js`, `vendor/LICENSE`. These are inlined into one HTML string at launch.
- `src/rem/cli.py`: `rem graph`.

### Tests
- `tests/test_graph_snapshot.py`: `Memory.graph_snapshot` on the in-memory backend. It returns raw records unchanged, all three collections, projects and `current_project_id`.
- `tests/test_graph_view.py`, the most important file, a pure function:
  - grouping by `normalized` keeps every member id;
  - aliases don't group;
  - project and scope filtering;
  - the either-side link rule and dangling ids;
  - one edge per (M, N);
  - a self-loop after grouping gets no edge but stays in the panel;
  - node `verified` derivation;
  - lone memories;
  - no edge exists without a stored relation or link (an anti-inference check over generated cases);
  - grouping leaves identities unchanged: the multiset of member entity ids equals the filtered input ids, each member's scope, project, relation ids and memory ids equal its stored record's, and the input snapshot is not mutated;
  - filtering happens before grouping: a project-X view contains no member, relation or memory from another project.
- `tests/test_app.py` / `tests/test_client_cli.py`: `GET /graph` and `RemClient.graph`. `rem graph` without the extra exits 3 with the hint (simulated with a monkeypatched import).
- The window and JS are smoke-tested by hand against a throwaway store. Nothing in the default suite imports Qt.

## Left open
- The extraction-quality panel (later milestone).
- Performance beyond personal scale (thousands of entities). `cose` and the filters are enough for now.
