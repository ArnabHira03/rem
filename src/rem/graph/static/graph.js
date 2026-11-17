/* rem graph window. Read-only: it only calls api.initial / api.view / api.refresh.
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

  function render(view) {
    info = new Map();
    view.nodes.forEach(function (n) { info.set(n.id, n); });
    view.edges.forEach(function (e) { info.set(e.id, e); });
    unhover();
    cy.batch(function () {
      cy.elements().remove();
      cy.add(toElements(view));
    });
    $("empty").hidden = view.nodes.some(function (n) { return n.kind === "entity"; });
    if (view.nodes.length) {
      cy.layout({ name: "cose", animate: false, randomize: true, nodeDimensionsIncludeLabels: true,
                  padding: 40, fit: true }).run();
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

  // ---------- search (JS only, over the current view) ----------
  function matches(n, q) {
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
  function copyId(id) {
    var c = el("span", "id", id);
    c.title = "Click to copy";
    c.addEventListener("click", function () {
      var done = function () { c.classList.add("copied"); setTimeout(function () { c.classList.remove("copied"); }, 600); };
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(id).then(done, function () {});
        }
      } catch (err) { /* the text stays selectable */ }
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
    li.appendChild(el("span", "meta", " · " + projectText(r)));
    if (!r.drawn) li.appendChild(el("span", "same", " (same node)"));
    li.title = r.provenance + ", " + r.scope;
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
                         ["project", projectText(e)], ["id", copyId(e.id.replace(/^rel:/, ""))]]));
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
  function refresh() {
    var mine = ++seq;
    setStatus("refreshing…");
    api.refresh(readFilters()).then(function (payload) {
      if (mine !== seq) return;
      setStatus("");
      load(payload);
    }, fail);
  }

  function wire() {
    $("project").addEventListener("change", update);
    document.querySelectorAll("input.scope").forEach(function (c) { c.addEventListener("change", update); });
    $("show-memories").addEventListener("change", update);
    $("search").addEventListener("input", function () { applySearch(); });
    $("search").addEventListener("keydown", function (e) {
      if (e.key !== "Enter") return;
      var hits = applySearch();
      if (hits && hits.nonempty()) cy.animate({ center: { eles: hits[0] }, duration: 200 });
    });
    $("refresh").addEventListener("click", refresh);
    $("panel-close").addEventListener("click", function () { cy.elements().unselect(); closePanel(); });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") { cy.elements().unselect(); closePanel(); }
    });
  }

  function start() {
    if (started || !window.pywebview || !window.pywebview.api) return;
    started = true;
    api = window.pywebview.api;
    makeCy();
    wire();
    setStatus("loading…");
    api.initial().then(function (payload) { setStatus(""); load(payload); }, fail);
  }

  window.addEventListener("pywebviewready", start);
  if (window.pywebview && window.pywebview.api) start();
})();
