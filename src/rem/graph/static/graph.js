/* rem graph window. Read-only: it only calls api.initial / api.view / api.refresh / api.jobs / api.job (and sphere.js, api.sphere).
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
  // The panel builders reach the view they serve only through `nav`: {exists(id), select(id), info(id)}.
  var cyNav = {
    exists: function (id) { return !!cy && cy.getElementById(id).nonempty(); },
    select: function (id) { selectId(id); },
    info: function (id) { return info.get(id); }
  };
  function nodeLink(nav, nodeId, text) {
    if (!nav.exists(nodeId)) return el("span", null, text);
    var b = el("button", "link", text);
    b.type = "button";
    b.title = "Select this node";
    b.addEventListener("click", function () { nav.select(nodeId); });
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
  function relationItem(nav, r) {
    var li = el("li");
    var arrow = r.direction === "out" ? "→ " + r.type + " → " : "← " + r.type + " ← ";
    li.appendChild(el("span", null, arrow));
    li.appendChild(r.drawn ? nodeLink(nav, r.other_node, r.other) : el("span", null, r.other));
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
  function memoryItem(nav, m) {
    var li = el("li");
    li.appendChild(el("span", null, mark(m.status) + " "));
    li.appendChild(nodeLink(nav, "mem:" + m.id, m.text));
    li.appendChild(el("span", "meta", " · " + projectText(m)));
    return li;
  }
  function list(items, make, emptyText) {
    if (!items.length) return el("div", "meta", emptyText);
    var ul = el("ul");
    items.forEach(function (x) { ul.appendChild(make(x)); });
    return ul;
  }

  function entityPanel(body, n, nav) {
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
      s.appendChild(list(m.relations, function (r) { return relationItem(nav, r); }, "none"));
      s.appendChild(el("h4", null, "memories"));
      s.appendChild(list(m.memories, function (x) { return memoryItem(nav, x); }, "none"));
      body.appendChild(s);
    });
  }

  function memoryPanel(body, n, nav) {
    body.appendChild(el("h2", null, "memory"));
    body.appendChild(el("div", "fulltext", n.text));
    body.appendChild(kv([["status", mark(n.status) + " " + n.status], ["provenance", n.provenance],
                         ["scope", n.scope], ["project", projectText(n)], ["id", copyId(n.id.replace(/^mem:/, ""))]]));
    var job = /^job:(\d+)$/.exec(n.job_id || "");
    if (job) {
      var go = el("button", "link", "See the conversation this came from");
      go.type = "button";
      go.addEventListener("click", function () { showRun(Number(job[1]), n.id.replace(/^mem:/, "")); });
      body.appendChild(go);
    }
    body.appendChild(el("h4", null, "linked entities"));
    body.appendChild(list(n.entity_nodes, function (nid) {
      var target = nav.info(nid);
      var li = el("li");
      li.appendChild(nodeLink(nav, nid, target ? target.label : nid.replace(/^ent:/, "")));
      return li;
    }, "none"));
  }

  function relationPanel(body, e, nav) {
    var src = nav.info(e.source), dst = nav.info(e.target);
    body.appendChild(el("h2", null, e.rel_type));
    var line = el("div");
    line.appendChild(nodeLink(nav, e.source, src ? src.label : e.source));
    line.appendChild(el("span", null, " → " + e.rel_type + " → "));
    line.appendChild(nodeLink(nav, e.target, dst ? dst.label : e.target));
    body.appendChild(line);
    if (e.sentence) body.appendChild(el("div", "fulltext", e.sentence));
    body.appendChild(kv([["status", mark(e.status) + " " + e.status], ["provenance", e.provenance],
                         ["evidence", e.evidence + (e.evidence === 1 ? " memory" : " memories")],
                         ["scope", e.scope], ["project", projectText(e)],
                         ["relation id", copyId(e.id.replace(/^rel:/, ""))]]));
  }

  function mentionPanel(body, e, nav) {
    var src = nav.info(e.source), dst = nav.info(e.target);
    body.appendChild(el("h2", null, "mention"));
    var line = el("div");
    line.appendChild(nodeLink(nav, e.source, src ? src.label : e.source));
    line.appendChild(el("span", null, " mentions "));
    line.appendChild(nodeLink(nav, e.target, dst ? dst.label : e.target));
    body.appendChild(line);
  }

  // Fills `body` with the panel for one node or edge of a view (the Graph's or the Sphere's).
  function buildPanel(body, x, nav) {
    if (x.kind === "entity") entityPanel(body, x, nav);
    else if (x.kind === "memory") memoryPanel(body, x, nav);
    else if (x.kind === "relation") relationPanel(body, x, nav);
    else mentionPanel(body, x, nav);
  }
  function showPanel(id) {
    var x = info.get(id);
    if (!x) { closePanel(); return; }
    var body = $("panel-body");
    clear(body);
    buildPanel(body, x, cyNav);
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
    if (tab === "graph") render(payload.view); else graphStale = true;   // a hidden graph renders when it shows
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
      if (tab !== "graph") {
        // the Sphere asks for its own data from the new snapshot; the graph catches up when it shows
        setProjects(payload.projects, $("project").value);
        setSnapshotTime(payload.generated_at);
        graphStale = true;
        if (tab === "sphere" && views.sphere) views.sphere.refreshed(); else sphereStale = true;
        return;
      }
      sphereStale = true;
      if (mine === seq) { load(payload); return; }
      setProjects(payload.projects, $("project").value);  // a project that vanished falls back to "all"
      setSnapshotTime(payload.generated_at);
      update();
    }, fail);
  }


  // ---------- tabs ----------
  // The window opens on the Sphere. Graph and Sphere share the project, scope and search controls; a change
  // made on one marks the other stale, and it catches up when it next shows.
  var tab = "sphere";
  var ready = false;        // the initial payload (projects, default filters) has been applied
  var views = {};           // other views, registered by their own scripts: { sphere: {show, hide, ...} }
  var graphStale = false;   // the filters or snapshot changed while the graph was hidden
  var sphereStale = false;  // the same, for the Sphere
  var extStale = true;      // the run list needs (re)loading when the Extraction tab shows
  function switchTab(name) {
    if (name !== "graph" && name !== "sphere" && name !== "extraction") return;
    var prev = tab;
    tab = name;
    document.body.dataset.tab = name;
    document.querySelectorAll("button.tab").forEach(function (b) {
      b.setAttribute("aria-selected", b.dataset.tab === name ? "true" : "false");
    });
    setStatus("");
    if (prev === "sphere" && name !== "sphere" && views.sphere) views.sphere.hide();
    if (name === "graph") {
      cy.resize();
      if (graphStale) { graphStale = false; update(); } else applySearch();
    } else if (name === "sphere") {
      showSphere();
    } else if (extStale) {
      loadRuns();
    }
  }
  function showSphere() {
    var v = views.sphere;
    if (!v || !ready) return;
    v.search($("search").value);
    var stale = sphereStale;
    sphereStale = false;
    v.show();
    if (stale && tab === "sphere") v.filtersChanged();
  }
  function registerView(name, v) {
    views[name] = v;
    if (name === "sphere" && tab === "sphere") showSphere();
  }
  // The project or a scope changed: the visible view updates now, the other one when it shows.
  function filtersChanged() {
    if (tab === "graph") { update(); sphereStale = true; }
    else if (tab === "sphere") { if (views.sphere) views.sphere.filtersChanged(); graphStale = true; }
  }

  // ---------- extraction ----------
  // Each Claude Code stop makes one job that reads only the newest messages, so on screen a job is never called a
  // "run" or a "conversation": it is described by when it happened and what rem did. All colour is in
  // extraction.css; this code only sets class names.
  var KNOWN_TYPES = ["person", "tool", "project", "language", "org", "place", "concept"];   // models.ENTITY_TYPES
  var TRUST_WHY = {
    user_asserted: "you said it", user_confirmed: "you confirmed it",
    tool_observed: "seen in tool output", assistant_generated: "the assistant's words, not yours"
  };
  var FLAG_TEXT = {
    downgraded: "The model thought you said this, but none of your messages do.",
    narration: "Reads like a note of what the assistant did, not a lasting fact.",
    noise: "Contains a link or a commit id, so it's probably not a lasting fact.",
    "scope-hint": "Filed as true everywhere (or for this computer), but it names this project.",
    "relation-not-written": "The model linked things together here, but the links weren't saved."
  };
  var nextBefore = null;
  var selectedRun = null;
  var runsSeq = 0, runSeq = 0;
  var lastDay = null;          // the day heading the run list ends on, so "Load older" appends under it
  var messagesOpen = false;    // "Show the messages" stays open or closed as you move between entries

  function plural(n, one, many) { return n + " " + (n === 1 ? one : many); }
  function date(iso) { var d = iso ? new Date(iso) : null; return d && !isNaN(d) ? d : null; }
  function when(iso) { var d = date(iso); return d ? d.toLocaleString() : String(iso || "—"); }
  function timeText(iso) {
    var d = date(iso);
    return d ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "—";
  }
  function dayKey(d) { return d ? d.getFullYear() + "-" + d.getMonth() + "-" + d.getDate() : "?"; }
  function dayText(iso) {
    var d = date(iso);
    if (!d) return "Unknown day";
    var today = new Date(), yesterday = new Date();
    yesterday.setDate(today.getDate() - 1);
    if (dayKey(d) === dayKey(today)) return "Today";
    if (dayKey(d) === dayKey(yesterday)) return "Yesterday";
    var opts = { weekday: "short", day: "numeric", month: "short" };
    if (d.getFullYear() !== today.getFullYear()) opts.year = "numeric";
    return d.toLocaleDateString(undefined, opts);
  }
  function projectName(o) {
    if (o.project_name) return o.project_name;
    if (o.project_id !== null && o.project_id !== undefined) return "project #" + o.project_id;
    return null;
  }
  function sourceText(s) { return /^claude[-_]code$/.test(s || "") ? "Claude Code" : s; }
  function readJobFilters(before) {
    return { project: $("project").value, status: $("run-status").value,
             flagged: $("flagged-only").checked, before: before === undefined ? null : before };
  }

  // ----- the list on the left -----
  function listSummary(r) {
    if (r.status === "failed") return "couldn't read";
    if (r.status === "pending") return "waiting";
    if (r.status === "processing") return "being read";
    var parts = [];
    if (r.n_added) parts.push(r.n_added + " saved");
    if (r.n_corroborated) parts.push(r.n_corroborated + " confirmed");
    return parts.length ? parts.join(" · ") : "nothing new";
  }
  function icon(kind, glyph, label) {
    var s = el("span", "icon " + kind, glyph);
    s.title = label;
    s.setAttribute("aria-label", label);
    return s;
  }
  function listIcon(r) {
    if (r.status === "failed") return icon("bad", "✕", "couldn't read");
    if (r.status === "pending" || r.status === "processing") return icon("wait", "⏳", "waiting");
    var flags = (r.flags || []).filter(function (f) { return f !== "failed"; });
    return flags.length ? icon("warn", "⚠", "worth a look") : null;
  }
  function runItem(r) {
    var li = el("li", "run");
    li.dataset.id = String(r.id);
    if (r.id === selectedRun) li.classList.add("selected");
    var top = el("div", "top");
    top.appendChild(el("span", "time", timeText(r.created_at)));
    var name = projectName(r);
    top.appendChild(el("span", name ? "proj" : "proj none", name || "no project"));
    li.appendChild(top);
    li.appendChild(el("div", "summary", listSummary(r)));
    var ic = listIcon(r);
    if (ic) li.appendChild(ic);
    li.title = when(r.created_at);
    li.addEventListener("click", function () { selectRun(r.id); });
    return li;
  }
  function markSelected() {
    document.querySelectorAll("#run-list li.run").forEach(function (li) {
      li.classList.toggle("selected", li.dataset.id === String(selectedRun));
    });
  }

  // Without `before`, replaces the list; with it, appends the next (older) page.
  function loadRuns(before) {
    var append = before !== undefined && before !== null;
    var mine = ++runsSeq;
    extStale = false;
    if (!append) {
      // a new filter: drop the old cursor now, so "Load older" can't append a page of the old filters
      nextBefore = null;
      $("load-more").hidden = true;
    }
    var filters = readJobFilters(append ? before : null);
    setStatus("loading…");
    api.jobs(filters).then(function (page) {
      if (mine !== runsSeq) return;
      setStatus("");
      var ul = $("run-list");
      if (!append) { clear(ul); lastDay = null; }
      page.runs.forEach(function (r) {
        var key = dayKey(date(r.created_at));
        if (key !== lastDay) { ul.appendChild(el("li", "day", dayText(r.created_at))); lastDay = key; }
        ul.appendChild(runItem(r));
      });
      nextBefore = page.next_before === undefined ? null : page.next_before;
      $("load-more").hidden = nextBefore === null;
      var first = ul.querySelector("li.run");
      var empty = $("runs-empty");
      empty.hidden = first !== null;
      var filtered = filters.project !== "all" || filters.status !== "all" || filters.flagged;
      empty.textContent = filters.flagged && nextBefore !== null
        ? "Nothing worth a look so far. Load older to look further back."
        : filtered ? "Nothing matches these filters."
        : "rem hasn't read any conversations yet. They'll show up here after your next message in Claude Code.";
      // never a blank right side: open the newest entry when nothing is picked yet; keep a picked one
      if (!append && selectedRun === null && first) selectRun(Number(first.dataset.id));
      $("run-empty").hidden = selectedRun !== null || first === null;
    }, fail);
  }

  function selectRun(id, memoryId) {
    var mine = ++runSeq;
    selectedRun = id;
    markSelected();
    api.job(id).then(function (body) {
      if (mine === runSeq) renderRun(body, memoryId);
    }, fail);
  }

  // From the Graph or Sphere: works even when the entry isn't in the loaded list. `memoryId` (optional) is the
  // memory whose fact to scroll to and highlight.
  function showRun(id, memoryId) {
    switchTab("extraction");
    selectRun(id, memoryId);
  }

  // ----- the right side -----
  function runHead(run) {
    var h = el("header", "run-head");
    h.appendChild(el("h2", null, dayText(run.created_at) + ", " + timeText(run.created_at) + " · " +
                                 (projectName(run) || "no project")));
    if (run.source) h.appendChild(el("span", "via", "via " + sourceText(run.source)));
    return h;
  }
  function errorDetails(error) {
    var d = el("details");
    d.appendChild(el("summary", null, "Show error"));
    d.appendChild(el("pre", null, error));
    return d;
  }
  function banner(kind, glyph, parts, error) {
    var b = el("div", "banner " + kind);
    b.appendChild(el("span", "icon " + kind, glyph));
    parts.forEach(function (p) { b.appendChild(typeof p === "string" ? document.createTextNode(p) : p); });
    if (error) b.appendChild(errorDetails(error));
    return b;
  }
  function appliedConfirmations(body) {
    var applied = {};
    (body.other_memory_writes || []).forEach(function (w) { if (w.event === "CONFIRM") applied[w.target_id] = true; });
    return applied;
  }
  function runSummary(run, body, facts) {
    if (run.status === "failed") {
      return banner("bad", "✕", ["rem couldn't read these messages. It won't try again by itself. " +
                                 "After fixing the model settings, run ", el("code", null, "rem retry-failed"), "."],
                    run.error);
    }
    if (run.status === "pending" && run.error) {
      return banner("warn", "⚠", ["rem hit a problem and will try again shortly."], run.error);
    }
    if (run.status === "pending") return banner("wait", "⏳", ["Waiting to be read."]);
    if (run.status === "processing") return banner("wait", "⏳", ["Being read now."]);

    var box = el("div", "summary-text");
    var msgs = plural(run.n_turns, "new message", "new messages");
    var added = run.n_added, corr = run.n_corroborated, proposed = run.n_proposed;
    var line;
    if (!proposed) {
      line = "rem read " + msgs + " and found nothing new to remember.";
    } else if (!added && !corr) {
      line = "From " + msgs + ", rem found " + plural(proposed, "fact", "facts") + " but didn't save any.";
    } else {
      var did = [];
      if (added) did.push("saved " + plural(added, "new memory", "new memories"));
      if (corr) did.push("confirmed " + corr + " it already had");
      line = "From " + msgs + ", rem " + did.join(" and ") + ".";
      var rest = proposed - added - corr;
      if (rest > 0) line += " " + (rest === 1 ? "1 other fact wasn't saved." : rest + " other facts weren't saved.");
    }
    box.appendChild(el("p", null, line));
    var flagged = facts.filter(function (f) { return f.flags && f.flags.length; }).length;
    if (flagged) box.appendChild(el("p", "warn-text", "⚠ " + flagged + (flagged === 1 ? " is" : " are") + " worth a look."));
    var nConfirmed = Object.keys(appliedConfirmations(body)).length;
    if (nConfirmed) box.appendChild(el("p", null, "You confirmed " + plural(nConfirmed, "memory", "memories") + "."));
    return box;
  }

  // The result of one fact: its group, card class, label and (only when it isn't obvious) an explanation.
  function resultOf(f) {
    var o = f.outcome || { kind: "none" };
    var now = o.memory_now;
    if (o.kind === "added" || o.kind === "corroborated") {
      var saved = o.kind === "added";
      var r = { group: saved ? "saved" : "confirmed", cls: saved ? "saved" : "confirmed",
                label: saved ? "✚ Saved" : "✓ Confirmed",
                note: saved ? null : "rem had this as a guess; you've now said it, so it's trusted." };
      if (!now) { r.label += " · deleted since"; r.cls += " gone"; }
      else if (String(now.text).trim() !== String(f.text).trim()) { r.label += " · edited since"; r.nowReads = now.text; }
      return r;
    }
    if (o.reason === "existing_or_skipped") return { group: "none", cls: "none", label: "= Already known",
                                                     note: "rem already had this, so nothing changed." };
    if (o.reason === "duplicate") return { group: "none", cls: "none", label: "= Repeat",
                                           note: "Said twice in these messages; kept once." };
    if (o.reason === "no_project") return { group: "none", cls: "none", label: "– Not saved",
                                            note: "About a project, but these messages weren't in one." };
    return { group: "none", cls: "none", label: "– Not saved", note: null };
  }
  function trustChip(now) {
    var verified = now.status === "verified";
    var why = TRUST_WHY[now.provenance] || now.provenance;
    return el("span", verified ? "trust verified" : "trust",
              (verified ? "✓ verified" : "◌ unverified") + " · " + why);
  }
  function flagLines(f) {
    var lines = [], typed = false;
    (f.flags || []).forEach(function (flag) {
      if (flag.rule !== "invalid-type") { lines.push(FLAG_TEXT[flag.rule] || flag.reason); return; }
      if (typed) return;   // one flag per odd entity; they're all listed from the entities at once
      typed = true;
      (f.entities || []).forEach(function (e) {
        if (KNOWN_TYPES.indexOf(e.type) < 0) {
          lines.push("Mentions “" + e.name + "” as an unknown kind of thing (“" + e.type + "”).");
        }
      });
    });
    return lines;
  }
  function originText(o) {
    return o === "user" ? "You" : o === "assistant" ? "The assistant" : o === "tool" ? "A tool" : String(o);
  }
  function scopeText(scope, run) {
    if (scope === "global") return "You, in any project";
    if (scope === "system") return "This computer";
    if (scope === "project") return "This project" + (projectName(run) ? " (" + projectName(run) + ")" : "");
    return String(scope);
  }
  function factDetails(f, run) {
    var o = f.outcome || {};
    var said = originText(f.origin_effective);
    if (f.origin_proposed !== f.origin_effective) {
      said += f.origin_proposed === "user" ? " (the model thought you said it)"
        : " (the model said: " + originText(f.origin_proposed).toLowerCase() + ")";
    }
    var rows = [["Said by", said], ["Applies to", scopeText(f.scope, run)]];
    if (f.entities && f.entities.length) {
      var chips = el("span");
      f.entities.forEach(function (e) {
        var known = KNOWN_TYPES.indexOf(e.type) >= 0;
        var c = el("span", "chip" + (known ? " type-" + e.type : ""), e.name);
        c.title = String(e.type);
        chips.appendChild(c);
      });
      rows.push(["Mentions", chips]);
    }
    if (f.relations && f.relations.length) {
      var links = el("span");
      f.relations.forEach(function (r) { links.appendChild(el("span", "rel", r.source + " → " + r.type + " → " + r.target)); });
      rows.push(["Links", links]);
    }
    if (f.restates) rows.push(["Restates", copyId(f.restates)]);
    if (o.memory_id) rows.push(["Memory id", copyId(o.memory_id)]);
    if (f.flags && f.flags.length) rows.push(["Rules", f.flags.map(function (x) { return x.rule; }).join(", ")]);
    var d = el("details", "more");
    d.appendChild(el("summary", null, "Details"));
    d.appendChild(kv(rows));
    return d;
  }
  function factCard(f, run) {
    var o = f.outcome || { kind: "none" };
    var now = o.memory_now;
    var res = resultOf(f);
    var card = el("article", "fact " + res.cls);
    if (o.memory_id) card.dataset.memoryId = o.memory_id;
    var top = el("div", "top");
    top.appendChild(el("span", "result", res.label));
    if (now) top.appendChild(trustChip(now));
    card.appendChild(top);
    card.appendChild(el("div", "text", f.text));
    if (res.note) card.appendChild(el("div", "note", res.note));
    if (res.nowReads) card.appendChild(el("div", "note", "Now reads: " + res.nowReads));
    // no run's ADD or CORROBORATE makes a memory user_confirmed: only the user's later confirmation does
    if (o.kind === "added" && now && now.provenance === "user_confirmed") card.appendChild(el("div", "note", "You confirmed it later."));
    var lines = flagLines(f);
    if (lines.length) {
      var w = el("div", "flagged");
      w.appendChild(el("span", "chip-warn", "⚠ Worth a look"));
      lines.forEach(function (t) { w.appendChild(el("div", "reason", t)); });
      card.appendChild(w);
    }
    card.appendChild(factDetails(f, run));
    return card;
  }
  function factGroups(facts, run) {
    var frag = document.createDocumentFragment();
    [["saved", "Saved"], ["confirmed", "Confirmed"], ["none", "Not saved"]].forEach(function (g) {
      var mine = facts.filter(function (f) { return resultOf(f).group === g[0]; });
      if (!mine.length) return;
      frag.appendChild(el("h3", "group-title", g[1] + " (" + mine.length + ")"));
      mine.forEach(function (f) { frag.appendChild(factCard(f, run)); });
    });
    return frag;
  }

  function confirmationsGroup(body) {
    var list = body.confirmations || [];
    if (!list.length) return null;
    var applied = appliedConfirmations(body), texts = {};
    (body.other_memory_writes || []).forEach(function (w) { if (w.memory_now) texts[w.target_id] = w.memory_now.text; });
    var frag = document.createDocumentFragment();
    frag.appendChild(el("h3", "group-title", "You confirmed (" + list.length + ")"));
    list.forEach(function (x) {
      var id = x.memory_id || x.id || "?";
      var ok = !!applied[id];
      var card = el("article", "fact " + (ok ? "confirmed" : "none"));
      if (x.memory_id) card.dataset.memoryId = x.memory_id;
      var top = el("div", "top");
      top.appendChild(el("span", "result", ok ? "✓ applied" : "– not applied"));
      card.appendChild(top);
      if (texts[id]) card.appendChild(el("div", "text", texts[id]));
      else { var t = el("div", "text"); t.appendChild(copyId(id)); card.appendChild(t); }
      if (x.quote) card.appendChild(el("div", "quote", "you said: “" + x.quote + "”"));
      frag.appendChild(card);
    });
    return frag;
  }

  function isLong(text) {
    var s = String(text || "");
    return s.length > 900 || s.split("\n").length > 10;
  }
  function turnEl(t) {
    var you = t.role === "user";
    var d = el("div", "turn " + (you ? "you" : "them"));
    var head = el("div", "head");
    head.appendChild(el("span", "who", you ? "You" : t.role === "assistant" ? "Assistant" : t.role));
    if (t.ts) { var ts = el("span", "ts", timeText(t.ts)); ts.title = when(t.ts); head.appendChild(ts); }
    d.appendChild(head);
    var text = el("div", "body", t.content);
    d.appendChild(text);
    if (isLong(t.content)) {
      // decided from the text, not the rendered height, so it also works while the section is collapsed
      text.classList.add("clamped");
      var more = el("button", "link", "Show more");
      more.type = "button";
      more.addEventListener("click", function () {
        var clamped = text.classList.toggle("clamped");
        more.textContent = clamped ? "Show more" : "Show less";
      });
      d.appendChild(more);
    }
    return d;
  }
  function messagesSection(body, forceOpen) {
    var turns = body.turns || [], context = body.context_turns || [];
    var d = el("details", "messages");
    d.open = forceOpen || messagesOpen;
    d.appendChild(el("summary", null, "Show the messages (" + turns.length + ")"));
    // a waiting entry forces it open without changing what you chose for the others
    if (!forceOpen) d.addEventListener("toggle", function () { messagesOpen = d.open; });
    if (context.length) {
      var ctx = el("details", "context");
      ctx.appendChild(el("summary", null, "Earlier messages (" + context.length +
                                          "), shown for context. rem doesn't take facts from these."));
      context.forEach(function (t) { ctx.appendChild(turnEl(t)); });
      d.appendChild(ctx);
    }
    if (!turns.length) d.appendChild(el("div", "none-text", "No messages."));
    turns.forEach(function (t) { d.appendChild(turnEl(t)); });
    return d;
  }

  function techSection(run, body) {
    var d = el("details", "tech");
    d.appendChild(el("summary", null, "Technical details"));
    d.appendChild(kv([["Run", "#" + run.id], ["Status", run.status], ["Model", run.model], ["Source", run.source],
                      ["Created", when(run.created_at)], ["Updated", when(run.updated_at)],
                      ["Messages read", run.n_turns]]));
    var other = body.other_memory_writes || [];
    // memory writes no fact accounts for: confirmations, and adds or corroborations not credited to a fact
    d.appendChild(el("h4", null, "Other memory writes"));
    if (!other.length) d.appendChild(el("div", "none-text", "none"));
    else {
      var mul = el("ul");
      other.forEach(function (w) {
        var now = w.memory_now;
        var text = now ? now.text : (w.new_value || w.target_id);
        var li = el("li", null, "memory " + w.event.toLowerCase() + ": " + text + (now ? " " + mark(now.status) : " (deleted since)"));
        li.title = "memory " + w.target_id;
        mul.appendChild(li);
      });
      d.appendChild(mul);
    }
    d.appendChild(el("h4", null, "Entity and relation writes"));
    var rows = (body.writes || []).filter(function (w) { return w.target_kind === "entity" || w.target_kind === "relation"; });
    if (!rows.length) d.appendChild(el("div", "none-text", "none"));
    else {
      var ul = el("ul");
      rows.forEach(function (w) {
        var li = el("li", null, w.target_kind + " " + w.event.toLowerCase() + ": " + (w.new_value || w.target_id));
        li.title = w.target_kind + " " + w.target_id;
        ul.appendChild(li);
      });
      d.appendChild(ul);
    }
    return d;
  }

  function flashFact(out, memoryId) {
    var cards = out.querySelectorAll("article.fact");
    for (var i = 0; i < cards.length; i++) {
      // compared as data, never built into a selector
      if (cards[i].dataset.memoryId !== memoryId) continue;
      var card = cards[i];
      card.scrollIntoView({ block: "center" });
      card.classList.add("flash");
      setTimeout(function () { card.classList.remove("flash"); }, 1600);
      return;
    }
  }

  function renderRun(body, memoryId) {
    var run = body.run, facts = body.facts || [];
    var out = $("run-body");
    clear(out);
    $("run-empty").hidden = true;
    out.appendChild(runHead(run));
    out.appendChild(runSummary(run, body, facts));
    if (facts.length) out.appendChild(factGroups(facts, run));
    var confirmed = confirmationsGroup(body);
    if (confirmed) out.appendChild(confirmed);
    out.appendChild(messagesSection(body, run.status === "pending" || run.status === "processing"));
    out.appendChild(techSection(run, body));
    $("run").scrollTop = 0;
    if (memoryId) flashFact(out, memoryId);
  }

  function wire() {
    $("project").addEventListener("change", function () {
      if (tab === "extraction") { loadRuns(); graphStale = true; sphereStale = true; }
      else { filtersChanged(); extStale = true; }
    });
    document.querySelectorAll("input.scope").forEach(function (c) { c.addEventListener("change", filtersChanged); });
    $("show-memories").addEventListener("change", update);   // Graph only: the Sphere always shows memories
    $("search").addEventListener("input", function () {
      if (tab === "sphere") { if (views.sphere) views.sphere.search($("search").value); } else applySearch();
    });
    $("search").addEventListener("keydown", function (e) {
      if (e.key !== "Enter") return;
      if (tab === "sphere") { if (views.sphere) views.sphere.searchEnter(); return; }
      var hits = applySearch();
      if (hits && hits.nonempty()) cy.animate({ center: { eles: hits[0] }, duration: 200 });
    });
    $("refresh").addEventListener("click", function () {
      if (tab !== "extraction") { refresh(); extStale = true; } else { loadRuns(); if (selectedRun !== null) selectRun(selectedRun); }
    });
    document.querySelectorAll("button.tab").forEach(function (b) {
      b.addEventListener("click", function () { switchTab(b.dataset.tab); });
    });
    $("run-status").addEventListener("change", function () { loadRuns(); });
    $("flagged-only").addEventListener("change", function () { loadRuns(); });
    $("load-more").addEventListener("click", function () { if (nextBefore !== null) loadRuns(nextBefore); });
    $("panel-close").addEventListener("click", function () { cy.elements().unselect(); closePanel(); });
    document.addEventListener("keydown", function (e) {
      if (e.key !== "Escape") return;
      if (tab === "sphere") { if (views.sphere) views.sphere.escape(); return; }
      cy.elements().unselect();
      closePanel();
    });
  }

  function start() {
    if (started || !window.pywebview || !window.pywebview.api) return;
    started = true;
    api = window.pywebview.api;
    document.body.dataset.tab = tab;
    makeCy();
    wire();
    setStatus("loading…");
    api.initial().then(function (payload) {
      setStatus("");
      load(payload);
      ready = true;
      switchTab(tab);
    }, fail);
  }

  // For sphere.js (an ES module that runs after this script): the shared API, filters, status and panel builders.
  window.remGraph = {
    api: function () { return api; },
    buildPanel: buildPanel,
    readFilters: readFilters,
    setStatus: setStatus,
    fail: fail,
    showRun: function (id, memoryId) { showRun(id, memoryId); },
    switchTab: function (name) { switchTab(name); },
    registerView: registerView,
    view: function (name) { return views[name]; }
  };

  window.addEventListener("pywebviewready", start);
  if (window.pywebview && window.pywebview.api) start();
})();
