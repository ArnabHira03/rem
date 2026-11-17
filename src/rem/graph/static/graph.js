/* rem graph window. Read-only: it only calls api.initial / api.view / api.refresh / api.jobs / api.job.
 * Every piece of text from data enters the page through textContent or createElement, never as markup. */
(function () {
  "use strict";

  var SHAPES = {
    person: "ellipse", tool: "round-rectangle", project: "hexagon", language: "diamond",
    org: "rectangle", place: "triangle", concept: "octagon"
  };
  var INK = "#000000", PAPER = "#ffffff", GREY = "#888888", MEM = "#666666", FAINT = "#bbbbbb";

  var api = null;
  var cy = null;
  var info = new Map();       // element id -> node or edge object from the view
  var selectedId = null;
  var started = false;

  var $ = function (id) { return document.getElementById(id); };

  // ---------- small DOM helpers (text only) ----------
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function mark(status) { return status === "verified" ? "✓" : "◌"; }
  function projectText(o) {
    if (o.project_name) return o.project_name;
    if (o.project_id !== null && o.project_id !== undefined) return "project #" + o.project_id;
    return "—";
  }
  function setStatus(text) { $("status").textContent = text || ""; }
  function fail(err) {
    var msg = err && err.message ? err.message : String(err);
    setStatus("error: " + msg);
  }

  // ---------- filters ----------
  function readFilters() {
    var scopes = [];
    document.querySelectorAll("input.scope").forEach(function (c) { if (c.checked) scopes.push(c.value); });
    return { project: $("project").value, scopes: scopes, show_memories: $("show-memories").checked };
  }

  function setProjects(projects, current) {
    var sel = $("project");
    clear(sel);
    var all = el("option", null, "all");
    all.value = "all";
    sel.appendChild(all);
    projects.forEach(function (p) {
      var o = el("option", null, p.name);
      o.value = String(p.id);
      o.title = p.root_path || "";
      sel.appendChild(o);
    });
    sel.value = String(current);
    if (sel.value !== String(current)) sel.value = "all";
  }

  function applyFilters(f) {
    document.querySelectorAll("input.scope").forEach(function (c) { c.checked = f.scopes.indexOf(c.value) >= 0; });
    $("show-memories").checked = !!f.show_memories;
  }

  function setSnapshotTime(iso) {
    var d = iso ? new Date(iso) : null;
    $("snapshot").textContent = d && !isNaN(d) ? "snapshot " + d.toLocaleString() : "";
  }

  // ---------- cytoscape ----------
  function style() {
    return [
      { selector: "node", style: {
        "label": "", "width": "data(size)", "height": "data(size)",
        "font-size": 11, "color": INK, "text-valign": "center", "text-halign": "right", "text-margin-x": 5,
        "text-outline-color": PAPER, "text-outline-width": 2, "min-zoomed-font-size": 6,
        "overlay-opacity": 0
      } },
      { selector: "node.entity", style: { "label": "data(label)", "shape": "data(shape)" } },
      { selector: "node.entity.verified", style: {
        "background-color": INK, "border-width": 0 } },
      { selector: "node.entity.unverified", style: {
        "background-color": PAPER, "border-width": 2, "border-style": "dashed", "border-color": INK } },
      { selector: "node.memory", style: {
        "shape": "ellipse", "width": 8, "height": 8, "border-width": 1, "border-color": MEM } },
      { selector: "node.memory.verified", style: { "background-color": MEM } },
      { selector: "node.memory.unverified", style: { "background-color": PAPER } },
      { selector: "edge", style: { "curve-style": "bezier", "overlay-opacity": 0, "font-size": 10, "color": INK,
        "text-background-color": PAPER, "text-background-opacity": 1, "text-background-padding": 2 } },
      { selector: "edge.relation", style: {
        "width": 1.5, "target-arrow-shape": "triangle", "arrow-scale": 0.9 } },
      { selector: "edge.relation.verified", style: {
        "line-color": INK, "target-arrow-color": INK, "line-style": "solid" } },
      { selector: "edge.relation.unverified", style: {
        "line-color": GREY, "target-arrow-color": GREY, "line-style": "dashed" } },
      { selector: "edge.mention", style: { "width": 1, "line-color": FAINT, "line-style": "dotted" } },
      { selector: "edge.relation:selected, edge.relation.hover", style: { "label": "data(rel_type)" } },
      { selector: "node:selected", style: {
        "outline-width": 4, "outline-color": INK, "outline-offset": 2, "outline-opacity": 1 } },
      { selector: "edge:selected", style: { "width": 4 } },
      { selector: "node.match", style: { "font-weight": "bold", "font-size": 13 } },
      { selector: ".dim, .faded", style: { "opacity": 0.15 } }
    ];
  }

  function makeCy() {
    cy = cytoscape({
      container: $("cy"), elements: [], style: style(),
      minZoom: 0.1, maxZoom: 4, boxSelectionEnabled: false, selectionType: "single"
    });
    cy.on("select", "node, edge", function (e) { selectedId = e.target.id(); showPanel(e.target.id()); });
    cy.on("tap", function (e) { if (e.target === cy) { cy.elements().unselect(); closePanel(); } });
    cy.on("mouseover", "node", function (e) { hoverNode(e.target, e); });
    cy.on("mouseout", "node", function () { unhover(); });
    cy.on("mousemove", "node", function (e) { placeTooltip(e); });
    cy.on("mouseover", "edge.relation", function (e) { e.target.addClass("hover"); });
    cy.on("mouseout", "edge.relation", function (e) { e.target.removeClass("hover"); });
    cy.on("pan zoom", function () { $("tooltip").hidden = true; });
  }

  function toElements(view) {
    var degree = {};
    view.edges.forEach(function (e) {
      degree[e.source] = (degree[e.source] || 0) + 1;
      degree[e.target] = (degree[e.target] || 0) + 1;
    });
    var out = [];
    view.nodes.forEach(function (n) {
      var trust = n.verified ? "verified" : "unverified";
      var size = n.kind === "memory" ? 8 : Math.min(56, 18 + 6 * Math.sqrt(degree[n.id] || 0));
      out.push({ group: "nodes", classes: n.kind + " " + trust, data: {
        id: n.id, label: n.label, shape: SHAPES[n.type] || "ellipse", size: size } });
    });
    view.edges.forEach(function (e) {
      var classes = e.kind === "relation" ? "relation " + (e.status === "verified" ? "verified" : "unverified") : "mention";
      out.push({ group: "edges", classes: classes, data: {
        id: e.id, source: e.source, target: e.target, rel_type: e.rel_type || "" } });
    });
    return out;
  }

  // Nodes that stay keep their place; a new node starts at the middle of its already-placed neighbours,
  // so a filter change nudges the picture instead of reshuffling it. Only a first layout is random and fitted.
  function placeNew(view, prev) {
    var placed = 0;
    view.nodes.forEach(function (n) { if (prev[n.id]) placed++; });
    if (!placed) return false;
    var neighbours = {};
    view.edges.forEach(function (e) {
      (neighbours[e.source] = neighbours[e.source] || []).push(e.target);
      (neighbours[e.target] = neighbours[e.target] || []).push(e.source);
    });
    cy.nodes().forEach(function (node) {
      var id = node.id();
      if (prev[id]) { node.position(prev[id]); return; }
      var near = (neighbours[id] || []).filter(function (o) { return prev[o]; });
      if (!near.length) return;
      var x = 0, y = 0;
      near.forEach(function (o) { x += prev[o].x; y += prev[o].y; });
      node.position({ x: x / near.length + (Math.random() - 0.5) * 30, y: y / near.length + (Math.random() - 0.5) * 30 });
    });
    return true;
  }

  function render(view) {
    var prev = {};
    cy.nodes().forEach(function (node) { var p = node.position(); prev[node.id()] = { x: p.x, y: p.y }; });
    info = new Map();
    view.nodes.forEach(function (n) { info.set(n.id, n); });
    view.edges.forEach(function (e) { info.set(e.id, e); });
    unhover();
    var kept = false;
    cy.batch(function () {
      cy.elements().remove();
      cy.add(toElements(view));
      kept = placeNew(view, prev);
    });
    $("empty").hidden = view.nodes.some(function (n) { return n.kind === "entity"; });
    if (view.nodes.length) {
      cy.layout({ name: "cose", animate: false, randomize: !kept, fit: !kept,
                  nodeDimensionsIncludeLabels: true, padding: 40 }).run();
    }
    applySearch();
    var keep = selectedId && cy.getElementById(selectedId);
    if (keep && keep.nonempty()) {
      keep.select();
      showPanel(selectedId);
    } else {
      closePanel();
    }
  }

  // ---------- hover ----------
  function hoverNode(node, e) {
    var nh = node.closedNeighborhood();
    cy.elements().not(nh).addClass("faded");
    nh.edges(".relation").addClass("hover");
    var n = info.get(node.id());
    var tip = $("tooltip");
    clear(tip);
    if (n && n.kind === "entity") {
      tip.appendChild(el("div", null, n.label + " · " + n.type));
      tip.appendChild(el("div", "sub", n.n_relations + " relations · " + n.n_memories + " memories"));
    } else if (n) {
      tip.appendChild(el("div", null, n.label));
    }
    tip.hidden = false;
    placeTooltip(e);
  }
  function placeTooltip(e) {
    var tip = $("tooltip");
    if (tip.hidden || !e.renderedPosition) return;
    var stage = $("stage");
    var x = e.renderedPosition.x + 14, y = e.renderedPosition.y + 14;
    x = Math.min(x, stage.clientWidth - tip.offsetWidth - 6);
    y = Math.min(y, stage.clientHeight - tip.offsetHeight - 6);
    tip.style.left = Math.max(4, x) + "px";
    tip.style.top = Math.max(4, y) + "px";
  }
  function unhover() {
    if (!cy) return;
    cy.elements().removeClass("faded");
    cy.edges(".hover").removeClass("hover");
    $("tooltip").hidden = true;
  }

  // ---------- search (JS only, over the current view): entity name or alias only ----------
  function matches(n, q) {
    if (n.kind !== "entity") return false;
    if ((n.label || "").toLowerCase().indexOf(q) >= 0) return true;
    return (n.members || []).some(function (m) {
      if ((m.name || "").toLowerCase().indexOf(q) >= 0) return true;
      return (m.aliases || []).some(function (a) { return String(a).toLowerCase().indexOf(q) >= 0; });
    });
  }
  function applySearch() {
    var q = $("search").value.trim().toLowerCase();
    cy.elements().removeClass("match dim");
    if (!q) return null;
    var hits = cy.nodes().filter(function (node) {
      var n = info.get(node.id());
      return !!n && matches(n, q);
    });
    hits.addClass("match");
    cy.elements().not(hits).addClass("dim");
    return hits;
  }

  // ---------- selection & panel ----------
  function selectId(id) {
    var ele = cy.getElementById(id);
    if (ele.empty()) return;
    cy.elements().unselect();
    ele.select();
    cy.animate({ center: { eles: ele }, duration: 200 });
  }
  function nodeLink(nodeId, text) {
    var exists = cy && cy.getElementById(nodeId).nonempty();
    if (!exists) return el("span", null, text);
    var b = el("button", "link", text);
    b.type = "button";
    b.title = "Select this node";
    b.addEventListener("click", function () { selectId(nodeId); });
    return b;
  }
  // Clicking selects the whole id, so Ctrl+C always copies it. The clipboard API is tried too, but an
  // inline page in QtWebEngine may not be allowed to use it; "copied" shows only when it really worked.
  function copyId(id) {
    var c = el("span", "id", id);
    c.title = "Click to select, then Ctrl+C to copy";
    c.addEventListener("click", function () {
      var range = document.createRange();
      range.selectNodeContents(c);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      var done = function () { c.classList.add("copied"); setTimeout(function () { c.classList.remove("copied"); }, 600); };
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(id).then(done, function () {});
      } catch (err) { /* the selection is there for Ctrl+C */ }
    });
    return c;
  }
  function kv(pairs) {
    var dl = el("dl", "kv");
    pairs.forEach(function (p) {
      dl.appendChild(el("dt", null, p[0]));
      var dd = el("dd");
      if (p[1] instanceof Node) dd.appendChild(p[1]); else dd.textContent = p[1] === null || p[1] === undefined ? "—" : String(p[1]);
      dl.appendChild(dd);
    });
    return dl;
  }
  function relationItem(r) {
    var li = el("li");
    var arrow = r.direction === "out" ? "→ " + r.type + " → " : "← " + r.type + " ← ";
    li.appendChild(el("span", null, arrow));
    li.appendChild(r.drawn ? nodeLink(r.other_node, r.other) : el("span", null, r.other));
    li.appendChild(el("span", null, " " + mark(r.status)));
    li.appendChild(el("span", "meta", " · " + r.scope + " · " + projectText(r)));
    if (!r.drawn) li.appendChild(el("span", "same", " (same node)"));
    li.title = r.provenance + ", " + r.scope + ", relation " + r.id;
    var ids = el("div", "ids");
    ids.appendChild(el("span", null, "relation "));
    ids.appendChild(copyId(r.id));
    ids.appendChild(el("span", null, " · other end "));
    ids.appendChild(copyId(r.other_entity_id));
    li.appendChild(ids);
    return li;
  }
  function memoryItem(m) {
    var li = el("li");
    li.appendChild(el("span", null, mark(m.status) + " "));
    if (cy.getElementById("mem:" + m.id).nonempty()) li.appendChild(nodeLink("mem:" + m.id, m.text));
    else li.appendChild(el("span", null, m.text));
    li.appendChild(el("span", "meta", " · " + projectText(m)));
    return li;
  }
  function list(items, make, emptyText) {
    if (!items.length) return el("div", "meta", emptyText);
    var ul = el("ul");
    items.forEach(function (x) { ul.appendChild(make(x)); });
    return ul;
  }

  function entityPanel(body, n) {
    body.appendChild(el("h2", null, n.label));
    var counts = n.members.length + (n.members.length === 1 ? " record" : " records") +
      " · " + n.n_relations + " relations · " + n.n_memories + " memories";
    body.appendChild(el("div", "meta", n.type + " · " + counts));
    var aliases = [];
    n.members.forEach(function (m) {
      (m.aliases || []).forEach(function (a) { if (aliases.indexOf(a) < 0) aliases.push(a); });
    });
    if (aliases.length) body.appendChild(el("div", "meta", "aliases: " + aliases.join(", ")));
    n.members.forEach(function (m) {
      var s = el("section");
      s.appendChild(el("h3", null, m.name));
      s.appendChild(kv([["type", m.type], ["scope", m.scope], ["project", projectText(m)],
                        ["entity id", copyId(m.entity_id)]]));
      if (n.members.length > 1 && m.aliases && m.aliases.length) s.appendChild(el("div", "meta", "aliases: " + m.aliases.join(", ")));
      s.appendChild(el("h4", null, "relations"));
      s.appendChild(list(m.relations, relationItem, "none"));
      s.appendChild(el("h4", null, "memories"));
      s.appendChild(list(m.memories, memoryItem, "none"));
      body.appendChild(s);
    });
  }

  function memoryPanel(body, n) {
    body.appendChild(el("h2", null, "memory"));
    body.appendChild(el("div", "fulltext", n.text));
    body.appendChild(kv([["status", mark(n.status) + " " + n.status], ["provenance", n.provenance],
                         ["scope", n.scope], ["project", projectText(n)], ["id", copyId(n.id.replace(/^mem:/, ""))]]));
    var job = /^job:(\d+)$/.exec(n.job_id || "");
    if (job) {
      var go = el("button", "link", "show the run that extracted this");
      go.type = "button";
      go.addEventListener("click", function () { showRun(Number(job[1])); });
      body.appendChild(go);
    }
    body.appendChild(el("h4", null, "linked entities"));
    body.appendChild(list(n.entity_nodes, function (nid) {
      var target = info.get(nid);
      var li = el("li");
      li.appendChild(nodeLink(nid, target ? target.label : nid.replace(/^ent:/, "")));
      return li;
    }, "none"));
  }

  function relationPanel(body, e) {
    var src = info.get(e.source), dst = info.get(e.target);
    body.appendChild(el("h2", null, e.rel_type));
    var line = el("div");
    line.appendChild(nodeLink(e.source, src ? src.label : e.source));
    line.appendChild(el("span", null, " → " + e.rel_type + " → "));
    line.appendChild(nodeLink(e.target, dst ? dst.label : e.target));
    body.appendChild(line);
    if (e.sentence) body.appendChild(el("div", "fulltext", e.sentence));
    body.appendChild(kv([["status", mark(e.status) + " " + e.status], ["provenance", e.provenance],
                         ["evidence", e.evidence + (e.evidence === 1 ? " memory" : " memories")],
                         ["scope", e.scope], ["project", projectText(e)],
                         ["relation id", copyId(e.id.replace(/^rel:/, ""))]]));
  }

  function mentionPanel(body, e) {
    var src = info.get(e.source), dst = info.get(e.target);
    body.appendChild(el("h2", null, "mention"));
    var line = el("div");
    line.appendChild(nodeLink(e.source, src ? src.label : e.source));
    line.appendChild(el("span", null, " mentions "));
    line.appendChild(nodeLink(e.target, dst ? dst.label : e.target));
    body.appendChild(line);
  }

  function showPanel(id) {
    var x = info.get(id);
    if (!x) { closePanel(); return; }
    var body = $("panel-body");
    clear(body);
    if (x.kind === "entity") entityPanel(body, x);
    else if (x.kind === "memory") memoryPanel(body, x);
    else if (x.kind === "relation") relationPanel(body, x);
    else mentionPanel(body, x);
    $("panel").hidden = false;
    $("panel").scrollTop = 0;
  }
  function closePanel() {
    selectedId = null;
    $("panel").hidden = true;
    clear($("panel-body"));
  }

  // ---------- api calls ----------
  var seq = 0;
  function update() {
    var mine = ++seq;
    setStatus("");
    api.view(readFilters()).then(function (view) {
      if (mine === seq) render(view);
    }, fail);
  }
  function load(payload) {
    setProjects(payload.projects, payload.filters.project);
    applyFilters(payload.filters);
    setSnapshotTime(payload.generated_at);
    render(payload.view);
  }
  // A refresh replaces the snapshot in Python. If a filter changed while it ran, the filter's view may come
  // from the old snapshot, so the latest refresh always updates the project list and time, then re-renders
  // with whatever filters are on screen now.
  var refreshSeq = 0;
  function refresh() {
    var mine = ++seq, myRefresh = ++refreshSeq;
    setStatus("refreshing…");
    api.refresh(readFilters()).then(function (payload) {
      if (myRefresh !== refreshSeq) return;
      setStatus("");
      if (mine === seq) { load(payload); return; }
      setProjects(payload.projects, $("project").value);  // a project that vanished falls back to "all"
      setSnapshotTime(payload.generated_at);
      update();
    }, fail);
  }


  // ---------- tabs ----------
  var tab = "graph";
  var graphStale = false;   // the project changed while the graph was hidden
  var extStale = true;      // the run list needs (re)loading when the Extraction tab shows
  function switchTab(name) {
    if (name !== "graph" && name !== "extraction") return;
    tab = name;
    document.body.dataset.tab = name;
    document.querySelectorAll("button.tab").forEach(function (b) {
      b.setAttribute("aria-selected", b.dataset.tab === name ? "true" : "false");
    });
    setStatus("");
    if (name === "graph") {
      cy.resize();
      if (graphStale) { graphStale = false; update(); }
    } else if (extStale) {
      loadRuns();
    }
  }

  // ---------- extraction runs ----------
  var REASONS = {
    duplicate: "duplicate of an earlier fact in this run",
    no_project: "project fact in a session with no project",
    existing_or_skipped: "matched an existing memory or skipped"
  };
  var nextBefore = null;
  var selectedRun = null;
  var runsSeq = 0, runSeq = 0;

  function when(iso) {
    var d = iso ? new Date(iso) : null;
    return d && !isNaN(d) ? d.toLocaleString() : String(iso || "—");
  }
  function badge(rule, reason) {
    var b = el("span", "badge", rule);
    if (reason) b.title = reason;
    return b;
  }
  function badges(flags) {
    var wrap = el("div", "flags");
    flags.forEach(function (f) { wrap.appendChild(badge(f.rule, f.reason)); });
    return wrap;
  }
  function readJobFilters(before) {
    return { project: $("project").value, status: $("run-status").value,
             flagged: $("flagged-only").checked, before: before === undefined ? null : before };
  }

  function runItem(r) {
    var li = el("li");
    li.dataset.id = String(r.id);
    if (r.id === selectedRun) li.className = "selected";
    li.appendChild(el("span", "when", when(r.created_at)));
    var counts = el("span", "counts", r.n_proposed + " → " + r.n_added);
    counts.title = r.n_proposed + " proposed, " + r.n_added + " added";
    if (r.flags && r.flags.length) {
      var b = badge(String(r.flags.length), r.flags.join(", "));
      counts.insertBefore(b, counts.firstChild);
    }
    li.appendChild(counts);
    li.appendChild(el("span", "sub muted", projectText(r) + " · " + r.status + " · run #" + r.id));
    li.addEventListener("click", function () { selectRun(r.id); });
    return li;
  }
  function markSelected() {
    document.querySelectorAll("#run-list li").forEach(function (li) {
      li.classList.toggle("selected", li.dataset.id === String(selectedRun));
    });
  }

  // Without `before`, replaces the list; with it, appends the next page.
  function loadRuns(before) {
    var append = before !== undefined && before !== null;
    var mine = ++runsSeq;
    extStale = false;
    if (!append) {
      // a new filter: drop the old cursor now, so "load more" can't append a page of the old filters
      nextBefore = null;
      $("load-more").hidden = true;
    }
    var filters = readJobFilters(append ? before : null);
    setStatus("loading runs…");
    api.jobs(filters).then(function (page) {
      if (mine !== runsSeq) return;
      setStatus("");
      var ul = $("run-list");
      if (!append) clear(ul);
      page.runs.forEach(function (r) { ul.appendChild(runItem(r)); });
      nextBefore = page.next_before === undefined ? null : page.next_before;
      $("load-more").hidden = nextBefore === null;
      var empty = $("runs-empty");
      empty.hidden = ul.childNodes.length > 0;
      empty.textContent = !filters.flagged ? "no extraction runs"
        : nextBefore !== null ? "no flagged runs in the jobs scanned so far" : "no flagged runs";
    }, fail);
  }

  function selectRun(id) {
    var mine = ++runSeq;
    selectedRun = id;
    markSelected();
    api.job(id).then(function (body) {
      if (mine === runSeq) renderRun(body);
    }, fail);
  }

  // From the graph: works even when the run isn't in the loaded list.
  function showRun(id) {
    switchTab("extraction");
    selectRun(id);
  }

  function turnEl(t) {
    var d = el("div", "turn");
    var head = el("div");
    head.appendChild(el("span", "role", t.role));
    if (t.ts) head.appendChild(el("span", "ts", when(t.ts)));
    d.appendChild(head);
    d.appendChild(el("div", "fulltext", t.content));
    return d;
  }

  function proposedCell(f) {
    var c = el("div", "cell fact");
    c.appendChild(el("div", "text", (f.index + 1) + ". " + f.text));
    var origin = f.origin_proposed === f.origin_effective ? f.origin_effective
      : f.origin_proposed + " → " + f.origin_effective;
    c.appendChild(el("div", "meta", f.scope + " · " + origin + (f.restates ? " · restates " + f.restates : "")));
    if (f.entities && f.entities.length) {
      c.appendChild(el("div", "meta", "entities: " + f.entities.map(function (e) {
        return e.name + " · " + e.type;
      }).join(", ")));
    }
    if (f.relations && f.relations.length) {
      var ul = el("ul", "meta");
      f.relations.forEach(function (r) { ul.appendChild(el("li", null, r.source + " → " + r.type + " → " + r.target)); });
      c.appendChild(ul);
    }
    if (f.flags && f.flags.length) c.appendChild(badges(f.flags));
    return c;
  }

  function storedCell(f) {
    var c = el("div", "cell");
    var o = f.outcome || { kind: "none" };
    var now = o.memory_now;
    var line;
    if (o.kind === "none") {
      line = el("div", "outcome", "none");
      c.appendChild(line);
      c.appendChild(el("div", "meta", REASONS[o.reason] || o.reason || ""));
      return c;
    }
    var label = o.kind === "added" ? "added" : "corroborated";
    line = el("div", "outcome", now ? label + " " + mark(now.status) + " " + now.status : label + ", deleted since");
    if (o.memory_id) line.title = "memory " + o.memory_id;
    c.appendChild(line);
    // no run's ADD or CORROBORATE makes a memory user_confirmed: only the user's confirmation does
    if (now && now.provenance === "user_confirmed") c.appendChild(el("div", "meta", "confirmed since"));
    if (now && String(now.text).trim() !== String(f.text).trim()) c.appendChild(el("div", "meta", "now: " + now.text));
    return c;
  }

  function writesCell(body) {
    var c = el("div", "cell");
    var other = body.other_memory_writes || [];
    if (other.length) {
      // memory writes no fact accounts for: confirmations, and adds or corroborations not credited
      c.appendChild(el("h3", null, "other memory writes"));
      var mul = el("ul");
      other.forEach(function (w) {
        var now = w.memory_now;
        var text = now ? now.text : (w.new_value || w.target_id);
        var li = el("li", null, "memory " + w.event.toLowerCase() + " " + text + (now ? " " + mark(now.status) : ", deleted since"));
        li.title = "memory " + w.target_id;
        mul.appendChild(li);
      });
      c.appendChild(mul);
    }
    var confirmations = body.confirmations || [];
    if (confirmations.length) {
      var confirmed = {};
      other.forEach(function (w) { if (w.event === "CONFIRM") confirmed[w.target_id] = true; });
      c.appendChild(el("h3", null, "confirmations"));
      var cul = el("ul");
      confirmations.forEach(function (x) {
        var id = x.memory_id || x.id || "?";
        var li = el("li", null, "memory " + id + (confirmed[id] ? " · confirmed" : " · not applied"));
        if (x.quote) li.appendChild(el("div", "meta fulltext", "user: \u201c" + x.quote + "\u201d"));
        cul.appendChild(li);
      });
      c.appendChild(cul);
    }
    c.appendChild(el("h3", null, "entity and relation writes"));
    var rows = (body.writes || []).filter(function (w) { return w.target_kind === "entity" || w.target_kind === "relation"; });
    if (!rows.length) { c.appendChild(el("div", "meta", "none")); return c; }
    var ul = el("ul");
    rows.forEach(function (w) {
      var li = el("li", null, w.target_kind + " " + w.event.toLowerCase() + " " + (w.new_value || w.target_id));
      li.title = w.target_kind + " " + w.target_id;
      ul.appendChild(li);
    });
    c.appendChild(ul);
    return c;
  }

  function place(node, row, col, span) {
    node.style.gridColumn = String(col);
    node.style.gridRow = span ? row + " / span " + span : String(row);
    return node;
  }

  function renderRun(body) {
    var run = body.run;
    var out = $("run-body");
    clear(out);
    $("run-empty").hidden = true;
    out.appendChild(el("h2", null, "run #" + run.id));
    var meta = [run.status, projectText(run), when(run.created_at)];
    if (run.source) meta.push(run.source);
    if (run.model) meta.push(run.model);
    meta.push(run.n_proposed + " proposed · " + run.n_added + " added · " + run.n_corroborated + " corroborated");
    out.appendChild(el("div", "meta", meta.join(" · ")));
    if (run.run_flags && run.run_flags.length) out.appendChild(badges(run.run_flags));
    if (run.error) out.appendChild(el("div", "error fulltext", "error: " + run.error));
    if (run.status === "pending" && !body.turns.length) {
      out.appendChild(el("div", "meta", "not run yet"));
      $("run").scrollTop = 0;
      return;
    }

    var facts = body.facts || [];
    var grid = el("div", "cols");
    var conv = el("div", "conversation");
    conv.appendChild(el("h3", null, "conversation"));
    if (body.context_turns && body.context_turns.length) {
      var ctx = el("details", "context");
      ctx.appendChild(el("summary", null, "earlier context (" + body.context_turns.length + ")"));
      body.context_turns.forEach(function (t) { ctx.appendChild(turnEl(t)); });
      conv.appendChild(ctx);
    }
    if (!body.turns.length) conv.appendChild(el("div", "meta", "no turns"));
    body.turns.forEach(function (t) { conv.appendChild(turnEl(t)); });
    var writesRow = Math.max(facts.length, 1) + 2;
    grid.appendChild(place(conv, 1, 1, writesRow));

    grid.appendChild(place(el("h3", null, "model proposed"), 1, 2));
    grid.appendChild(place(el("h3", null, "stored"), 1, 3));
    if (!facts.length) grid.appendChild(place(el("div", "meta", "no facts proposed"), 2, 2));
    facts.forEach(function (f, i) {
      grid.appendChild(place(proposedCell(f), i + 2, 2));
      grid.appendChild(place(storedCell(f), i + 2, 3));
    });
    grid.appendChild(place(writesCell(body), writesRow, 3));
    out.appendChild(grid);
    $("run").scrollTop = 0;
  }

  function wire() {
    $("project").addEventListener("change", function () {
      if (tab === "graph") { update(); extStale = true; } else { loadRuns(); graphStale = true; }
    });
    document.querySelectorAll("input.scope").forEach(function (c) { c.addEventListener("change", update); });
    $("show-memories").addEventListener("change", update);
    $("search").addEventListener("input", function () { applySearch(); });
    $("search").addEventListener("keydown", function (e) {
      if (e.key !== "Enter") return;
      var hits = applySearch();
      if (hits && hits.nonempty()) cy.animate({ center: { eles: hits[0] }, duration: 200 });
    });
    $("refresh").addEventListener("click", function () {
      if (tab === "graph") { refresh(); extStale = true; } else { loadRuns(); if (selectedRun !== null) selectRun(selectedRun); }
    });
    document.querySelectorAll("button.tab").forEach(function (b) {
      b.addEventListener("click", function () { switchTab(b.dataset.tab); });
    });
    $("run-status").addEventListener("change", function () { loadRuns(); });
    $("flagged-only").addEventListener("change", function () { loadRuns(); });
    $("load-more").addEventListener("click", function () { if (nextBefore !== null) loadRuns(nextBefore); });
    $("panel-close").addEventListener("click", function () { cy.elements().unselect(); closePanel(); });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") { cy.elements().unselect(); closePanel(); }
    });
  }

  function start() {
    if (started || !window.pywebview || !window.pywebview.api) return;
    started = true;
    api = window.pywebview.api;
    document.body.dataset.tab = "graph";
    makeCy();
    wire();
    setStatus("loading…");
    api.initial().then(function (payload) { setStatus(""); load(payload); }, fail);
  }

  window.addEventListener("pywebviewready", start);
  if (window.pywebview && window.pywebview.api) start();
})();
