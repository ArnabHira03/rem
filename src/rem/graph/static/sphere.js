/* rem graph window — the Sphere tab (an ES module; three.js comes from the page's import map).
 * Read-only: it only calls api.sphere. Every piece of text from data enters the page through textContent.
 * Deterministic: ids are hashed for every seed and offset; nothing here draws random numbers.
 *
 * Entities sit on the unit sphere (a force layout over the whole unfiltered graph), memories inside it at the
 * mean of the entities they link, and memories with no visible entity link on a ring at 1.5 radii. The full
 * view decides positions and sizes; the filtered view decides visibility, trust look, lines, arcs and the panel.
 */
import * as THREE from "three";

const G = window.remGraph;
const $ = (id) => document.getElementById(id);

// ---------- look ----------
const TYPE_COLORS = {
  person: "#e3a27c", tool: "#6fb2dc", project: "#a698e2", language: "#84c98c",
  org: "#d9bd68", place: "#df8a8f", concept: "#68c6ba",
};
const OTHER_COLOR = "#8790a6";
const BG = "#05070e";
const MEM_VERIFIED = "#c7d2ee";
const MEM_UNVERIFIED = "#56607c";
const LINE_COLOR = "#252c3d";
const ARC_VERIFIED = "#b9c7ee";
const ARC_UNVERIFIED = "#4a5470";
const RIM_COLOR = "#3a5cc0";
const HALO_COLOR = "#e8eefc";
const DIM = 0.1;

// ---------- geometry and motion ----------
const RING_R = 1.5;
const HOME_DIST = 4.4, MIN_DIST = 1.8, MAX_DIST = 9;
const FOV = 40;
const CAM_ELEV = 0.3;                   // camera height above the equator, radians
const CAM_DIR = new THREE.Vector3(0, Math.sin(CAM_ELEV), Math.cos(CAM_ELEV));
const SPIN = (2 * Math.PI) / 90;        // auto-rotate, rad/s
const RING_SPIN = (2 * Math.PI) / 75;   // ring drift, rad/s
const RESUME_MS = 1000;                 // after a pause ends, wait this long, then ease back in
const EASE_IN_MS = 900;                 // spin and ring drift ease back up to speed over about this long
const EASE_OUT_MS = 90;                 // and stop almost at once, so a hovered dot holds still
const SEPARATION = 0.3;                 // two entities' centres stay at least this x (size a + size b) apart
const FOCUS_MS = 600;
const REPULSE_COS = Math.cos((40 * Math.PI) / 180);
const REPULSE_CHORD = 2 * Math.sin((20 * Math.PI) / 180);
const ATTRACT = 0.04, REPULSE = 0.0005;
const FULL_ITERS = 300, WARM_ITERS = 60;
const FRAME_BUDGET_MS = 12;

// ---------- hashing (FNV-1a, 32 bit) ----------
function fnv(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}
const hash01 = (id, salt) => fnv(salt + "\u0000" + id) / 4294967296;
function hashDir(id, salt) {
  const z = 2 * hash01(id, salt + "z") - 1, t = 2 * Math.PI * hash01(id, salt + "t"), s = Math.sqrt(1 - z * z);
  return [s * Math.cos(t), z, s * Math.sin(t)];
}

// ---------- layout (pure, over the full view) ----------
// Pair weights: each memory linking k entities adds 1/(k-1) to each of its pairs; each relation adds 2.
function buildGraph(full) {
  const ents = full.nodes.filter((n) => n.kind === "entity").map((n) => n.id).sort();
  const index = new Map(ents.map((id, i) => [id, i]));
  const weights = new Map();
  const add = (a, b, w) => {
    if (a === b || !index.has(a) || !index.has(b)) return;
    const [x, y] = a < b ? [a, b] : [b, a];
    const key = x + "|" + y;
    const cur = weights.get(key);
    if (cur) cur.w += w; else weights.set(key, { a: index.get(x), b: index.get(y), w });
  };
  for (const n of full.nodes) {
    if (n.kind !== "memory") continue;
    const linked = [...new Set(n.entity_nodes)].sort();
    if (linked.length < 2) continue;
    const w = 1 / (linked.length - 1);
    for (let i = 0; i < linked.length; i++) for (let j = i + 1; j < linked.length; j++) add(linked[i], linked[j], w);
  }
  for (const e of full.edges) if (e.kind === "relation") add(e.source, e.target, 2);
  const keys = [...weights.keys()].sort();
  const pairs = keys.map((k) => weights.get(k));
  const nMem = new Map(full.nodes.filter((n) => n.kind === "entity").map((n) => [n.id, n.n_memories]));
  const sizes = ents.map((id) => entitySize(nMem.get(id) || 0));
  const print = fnv(ents.join("\n") + "\n#\n" + keys.map((k) => k + "|" + weights.get(k).w.toFixed(6)).join("\n"))
    .toString(16);
  return { ents, index, pairs, print, sizes };
}

// Dot size, from the number of linked memories in the full view.
function entitySize(nMemories) { return Math.min(0.24, 0.075 + 0.03 * Math.sqrt(nMemories)); }

// Pushes apart any two entities closer than SEPARATION x (size a + size b), so no two dots ever sit on one
// point. Deterministic, and it moves nothing when nothing overlaps, so an unchanged layout stays put.
function separate(g, pos) {
  const n = g.ents.length, P = new Float64Array(3 * n);
  g.ents.forEach((id, i) => P.set(pos.get(id), 3 * i));
  const maxNeed = 2 * SEPARATION * Math.max(...g.sizes, 0), minDot = 1 - (maxNeed * maxNeed) / 2;
  for (let pass = 0; pass < 40; pass++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dot = P[3 * i] * P[3 * j] + P[3 * i + 1] * P[3 * j + 1] + P[3 * i + 2] * P[3 * j + 2];
        if (dot < minDot) continue;
        const need = SEPARATION * (g.sizes[i] + g.sizes[j]);
        let dx = P[3 * i] - P[3 * j], dy = P[3 * i + 1] - P[3 * j + 1], dz = P[3 * i + 2] - P[3 * j + 2];
        const c = Math.hypot(dx, dy, dz);
        if (c >= need) continue;
        if (c < 1e-6) {
          const h = hashDir(g.ents[i] + "|" + g.ents[j], "split");
          dx = h[0]; dy = h[1]; dz = h[2];
        } else { dx /= c; dy /= c; dz /= c; }
        const push = (need - c) / 2 + 1e-4;
        P[3 * i] += dx * push; P[3 * i + 1] += dy * push; P[3 * i + 2] += dz * push;
        P[3 * j] -= dx * push; P[3 * j + 1] -= dy * push; P[3 * j + 2] -= dz * push;
        for (const k of [i, j]) {
          const len = Math.hypot(P[3 * k], P[3 * k + 1], P[3 * k + 2]);
          P[3 * k] /= len; P[3 * k + 1] /= len; P[3 * k + 2] /= len;
        }
        moved = true;
      }
    }
    if (!moved) break;
  }
  const out = new Map();
  g.ents.forEach((id, i) => out.set(id, [P[3 * i], P[3 * i + 1], P[3 * i + 2]]));
  return out;
}

function itersFor(n, base) {
  // a pass over all pairs is O(n^2); keep the whole layout inside roughly a second
  return Math.max(Math.round(base / 5), Math.min(base, Math.round(base * Math.pow(400 / Math.max(n, 400), 2))));
}

function seedPositions(g, prev) {
  const n = g.ents.length, P = new Float64Array(3 * n);
  const set = (i, v) => { P[3 * i] = v[0]; P[3 * i + 1] = v[1]; P[3 * i + 2] = v[2]; };
  if (!prev) {
    // Fibonacci spiral, ordered by a hash of the id
    const order = g.ents.map((id, i) => [fnv(id), id, i]).sort((x, y) => x[0] - y[0] || (x[1] < y[1] ? -1 : 1));
    const golden = Math.PI * (3 - Math.sqrt(5));
    order.forEach(([, , i], r) => {
      const y = 1 - (2 * (r + 0.5)) / n, s = Math.sqrt(1 - y * y), t = r * golden;
      set(i, [s * Math.cos(t), y, s * Math.sin(t)]);
    });
    return P;
  }
  // warm start: known nodes keep their place; new ones start at the mean of placed neighbours, else a hash seed
  const nbrs = g.ents.map(() => []);
  for (const p of g.pairs) { nbrs[p.a].push(p.b); nbrs[p.b].push(p.a); }
  g.ents.forEach((id, i) => {
    const known = prev.get(id);
    if (known) { set(i, known); return; }
    const m = [0, 0, 0];
    for (const j of nbrs[i]) {
      const q = prev.get(g.ents[j]);
      if (q) { m[0] += q[0]; m[1] += q[1]; m[2] += q[2]; }
    }
    const len = Math.hypot(m[0], m[1], m[2]);
    set(i, len > 1e-6 ? [m[0] / len, m[1] / len, m[2] / len] : hashDir(id, "seed"));
    // a tiny hashed nudge, so two new nodes with the same neighbours never start on the same point
    const j = hashDir(id, "nudge");
    for (let k = 0; k < 3; k++) P[3 * i + k] += 0.01 * j[k];
  });
  return P;
}

function layoutStep(g, P, F, temp) {
  const n = g.ents.length;
  F.fill(0);
  for (const { a, b, w } of g.pairs) {
    const dx = P[3 * b] - P[3 * a], dy = P[3 * b + 1] - P[3 * a + 1], dz = P[3 * b + 2] - P[3 * a + 2];
    const k = ATTRACT * w;
    F[3 * a] += dx * k; F[3 * a + 1] += dy * k; F[3 * a + 2] += dz * k;
    F[3 * b] -= dx * k; F[3 * b + 1] -= dy * k; F[3 * b + 2] -= dz * k;
  }
  for (let i = 0; i < n; i++) {
    const xi = P[3 * i], yi = P[3 * i + 1], zi = P[3 * i + 2];
    for (let j = i + 1; j < n; j++) {
      const xj = P[3 * j], yj = P[3 * j + 1], zj = P[3 * j + 2];
      if (xi * xj + yi * yj + zi * zj < REPULSE_COS) continue;
      let dx = xi - xj, dy = yi - yj, dz = zi - zj;
      let c = Math.hypot(dx, dy, dz);
      if (c < 1e-4) {  // coincident: push apart along a hashed direction
        const h = hashDir(g.ents[i] + "|" + g.ents[j], "split");
        dx = h[0]; dy = h[1]; dz = h[2]; c = 1e-4;
      } else { dx /= c; dy /= c; dz /= c; }
      const f = REPULSE * (1 / Math.max(c, 0.02) - 1 / REPULSE_CHORD);
      F[3 * i] += dx * f; F[3 * i + 1] += dy * f; F[3 * i + 2] += dz * f;
      F[3 * j] -= dx * f; F[3 * j + 1] -= dy * f; F[3 * j + 2] -= dz * f;
    }
  }
  for (let i = 0; i < n; i++) {
    const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
    let fx = F[3 * i], fy = F[3 * i + 1], fz = F[3 * i + 2];
    const radial = fx * x + fy * y + fz * z;
    fx -= radial * x; fy -= radial * y; fz -= radial * z;
    const m = Math.hypot(fx, fy, fz);
    if (m > temp) { fx *= temp / m; fy *= temp / m; fz *= temp / m; }
    const nx = x + fx, ny = y + fy, nz = z + fz, len = Math.hypot(nx, ny, nz) || 1;
    P[3 * i] = nx / len; P[3 * i + 1] = ny / len; P[3 * i + 2] = nz / len;
  }
}

// Runs in chunks across animation frames. Resolves to a Map id -> [x, y, z], or null when cancelled.
function runLayout(g, prev, isCancelled) {
  const warm = !!prev;
  const P = seedPositions(g, prev), F = new Float64Array(P.length);
  const iters = itersFor(g.ents.length, warm ? WARM_ITERS : FULL_ITERS);
  const t0 = warm ? 0.03 : 0.12;
  let it = 0;
  return new Promise((resolve) => {
    const chunk = () => {
      if (isCancelled()) { resolve(null); return; }
      const start = performance.now();
      while (it < iters && performance.now() - start < FRAME_BUDGET_MS) {
        layoutStep(g, P, F, t0 * (1 - it / iters) + 0.002);
        it++;
      }
      if (it < iters) { requestAnimationFrame(chunk); return; }
      const out = new Map();
      g.ents.forEach((id, i) => out.set(id, [P[3 * i], P[3 * i + 1], P[3 * i + 2]]));
      resolve(out);
    };
    chunk();
  });
}

// Direction the reset view faces: the centre of the largest connected component.
function largestCentre(g, pos) {
  const parent = g.ents.map((_, i) => i);
  const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  for (const p of g.pairs) { const a = find(p.a), b = find(p.b); if (a !== b) parent[Math.max(a, b)] = Math.min(a, b); }
  const groups = new Map();
  g.ents.forEach((_, i) => { const r = find(i); groups.set(r, (groups.get(r) || []).concat(i)); });
  let best = null;
  for (const members of groups.values()) if (!best || members.length > best.length) best = members;
  const m = new THREE.Vector3();
  for (const i of best || []) m.add(new THREE.Vector3(...pos.get(g.ents[i])));
  return m.lengthSq() > 1e-8 ? m.normalize() : new THREE.Vector3(0, 0, 1);
}

// Memory positions, from the full view's links only, so filters never move a memory inside the sphere.
function placeMemories(full, pos) {
  const out = new Map();
  const mems = full.nodes.filter((n) => n.kind === "memory");
  const sets = new Map();
  for (const m of mems) {
    const key = [...new Set(m.entity_nodes)].filter((id) => pos.has(id)).sort().join("\n");
    sets.set(key, (sets.get(key) || 0) + 1);
  }
  for (const m of mems) {
    const linked = [...new Set(m.entity_nodes)].filter((id) => pos.has(id)).sort();
    if (!linked.length) continue;
    let p;
    if (linked.length === 1) {
      // a small cap beneath the entity at r = 0.85, spread by the memory id
      const e = new THREE.Vector3(...pos.get(linked[0]));
      const t1 = new THREE.Vector3(0, 1, 0).cross(e);
      if (t1.lengthSq() < 1e-6) t1.set(1, 0, 0);
      t1.normalize();
      const t2 = new THREE.Vector3().crossVectors(e, t1);
      const ang = 0.05 + 0.09 * Math.sqrt(hash01(m.id, "cap-r")), az = 2 * Math.PI * hash01(m.id, "cap-a");
      p = e.multiplyScalar(Math.cos(ang))
        .addScaledVector(t1, Math.sin(ang) * Math.cos(az)).addScaledVector(t2, Math.sin(ang) * Math.sin(az))
        .multiplyScalar(0.85);
    } else {
      const mean = new THREE.Vector3();
      for (const id of linked) mean.add(new THREE.Vector3(...pos.get(id)));
      mean.divideScalar(linked.length);
      const len = mean.length();
      if (len < 0.08) p = new THREE.Vector3(...hashDir(m.id, "mem-dir")).multiplyScalar(0.9 * 0.08);
      else p = mean.multiplyScalar(0.9);
      if (sets.get(linked.join("\n")) > 1) p.add(new THREE.Vector3(...hashDir(m.id, "mem-jitter")).multiplyScalar(0.025));
    }
    out.set(m.id, p);
  }
  return out;
}

function ringAngleOf(id) { return 2 * Math.PI * hash01(id, "ring"); }
function ringSlot(id) {
  const a = ringAngleOf(id), r = RING_R + 0.05 * (hash01(id, "ring-r") - 0.5);
  return new THREE.Vector3(r * Math.cos(a), 0, r * Math.sin(a));
}

// ---------- textures ----------
const textures = new Map();
function dotTexture(color, filled) {
  const key = color + (filled ? ":f" : ":r");
  if (textures.has(key)) return textures.get(key);
  const size = 128, c = document.createElement("canvas");
  c.width = c.height = size;
  const x = c.getContext("2d"), h = size / 2;
  if (filled) {
    const glow = x.createRadialGradient(h, h, 0, h, h, h);
    glow.addColorStop(0, color + "66");
    glow.addColorStop(0.35, color + "33");
    glow.addColorStop(1, color + "00");
    x.fillStyle = glow;
    x.fillRect(0, 0, size, size);
    x.beginPath(); x.arc(h, h, size * 0.2, 0, 2 * Math.PI); x.fillStyle = color; x.fill();
  } else {
    x.beginPath(); x.arc(h, h, size * 0.18, 0, 2 * Math.PI);
    x.lineWidth = size * 0.055; x.strokeStyle = color; x.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  textures.set(key, t);
  return t;
}
function pointTexture() {
  if (textures.has("point")) return textures.get("point");
  const size = 64, c = document.createElement("canvas");
  c.width = c.height = size;
  const x = c.getContext("2d"), h = size / 2;
  const g = x.createRadialGradient(h, h, 0, h, h, h);
  g.addColorStop(0, "#ffffffff"); g.addColorStop(0.35, "#ffffffcc"); g.addColorStop(0.6, "#ffffff22"); g.addColorStop(1, "#ffffff00");
  x.fillStyle = g; x.fillRect(0, 0, size, size);
  const t = new THREE.CanvasTexture(c);
  textures.set("point", t);
  return t;
}
const typeColor = (type) => TYPE_COLORS[type] || OTHER_COLOR;
const lin = (hex) => new THREE.Color(hex);

// ---------- state ----------
const stage = $("sphere-stage");
const labelLayer = $("sphere-labels");
const tip = $("sphere-tip");
const panel = $("sphere-panel");
const panelBody = $("sphere-panel-body");
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");

let renderer = null, scene, camera, root, ringGroup, rim, halo;
let shown = false, raf = 0, lastT = 0;
let camDist = HOME_DIST;
let autoOn = !reduced.matches;
let resumeAt = 0;
let spinGain = 0, ringGain = 0;     // 0..1: how much of the auto-rotate / ring drift speed is applied
let pointer = null;                 // last pointer position over the canvas, for re-picking every frame
let ringAngle = 0;
const spinVel = { axis: new THREE.Vector3(0, 1, 0), w: 0 };   // drag inertia, rad/ms
let drag = null;
let anim = null;
let hoverId = null, selectedId = null, panelId = null;
let searchQuery = "", searchHits = null;

const cache = { pos: null, print: null, graph: null };   // never stored; cleared on window open and Reset layout
let seq = 0;
let world = null;   // the built scene's data: see buildScene

// ---------- scene ----------
function initRenderer() {
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  } catch (err) {
    renderer = null;
    return false;
  }
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setClearColor(lin(BG), 1);
  $("sphere-canvas").appendChild(renderer.domElement);
  scene = new THREE.Scene();
  scene.background = lin(BG);
  scene.fog = new THREE.Fog(lin(BG), 1, 10);
  camera = new THREE.PerspectiveCamera(FOV, 1, 0.05, 60);
  root = new THREE.Group();
  ringGroup = new THREE.Group();
  scene.add(root, ringGroup);
  rim = new THREE.Mesh(new THREE.SphereGeometry(1.0, 64, 48), new THREE.ShaderMaterial({
    uniforms: { color: { value: lin(RIM_COLOR) } },
    vertexShader: `varying vec3 vN; varying vec3 vV;
      void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal);
        vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `uniform vec3 color; varying vec3 vN; varying vec3 vV;
      void main() { float f = 1.0 - abs(dot(normalize(vN), normalize(vV)));
        gl_FragColor = vec4(color * (0.03 + 0.55 * pow(f, 3.0)), 1.0); }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.FrontSide,
  }));
  scene.add(rim);
  // a thin ring around the selected item, so a picked memory dot is easy to find
  halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: dotTexture(HALO_COLOR, false), transparent: true,
    depthWrite: false, depthTest: false, fog: false, opacity: 0.85 }));
  halo.visible = false;
  scene.add(halo);
  new ResizeObserver(resize).observe(stage);
  resize();
  wirePointer();
  return true;
}

function resize() {
  if (!renderer) return;
  const w = stage.clientWidth, h = stage.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = w + "px";
  renderer.domElement.style.height = h + "px";
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function setCamera() {
  camera.position.copy(CAM_DIR).multiplyScalar(camDist);
  camera.lookAt(0, 0, 0);
  scene.fog.near = camDist - 0.9;
  scene.fog.far = camDist + 2.4;
}

function disposeWorld() {
  if (!world) return;
  for (const obj of world.objects) {
    obj.parent && obj.parent.remove(obj);
    obj.geometry && obj.geometry !== world.spriteGeo && obj.geometry.dispose();
    obj.material && obj.material.dispose();
  }
  world = null;
}

function buildScene(res, g, pos, memPos) {
  disposeWorld();
  const full = res.full, filtered = res.filtered;
  const fullInfo = new Map(full.nodes.map((n) => [n.id, n]));
  const info = new Map();
  filtered.nodes.forEach((n) => info.set(n.id, n));
  filtered.edges.forEach((e) => info.set(e.id, e));
  const objects = [];

  // entities: every full-view entity gets a sprite; the filtered view decides which show and their trust look
  const ents = new Map();
  for (const id of g.ents) {
    const fn = fullInfo.get(id), vis = info.get(id);
    const color = typeColor(fn.type);
    const filled = vis ? vis.verified : fn.verified;
    const mat = new THREE.SpriteMaterial({ map: dotTexture(color, filled), transparent: true, depthWrite: false, fog: true });
    const sprite = new THREE.Sprite(mat);
    const p = new THREE.Vector3(...pos.get(id));
    const size = entitySize(fn.n_memories);
    sprite.position.copy(p);
    sprite.scale.setScalar(size);
    sprite.visible = !!vis;
    root.add(sprite);
    objects.push(sprite);
    ents.set(id, { id, pos: p, sprite, size, color, n: fn.n_memories, label: fn.label, visible: !!vis });
  }

  // memories: inside the sphere when they have a visible entity link, else on the ring
  const sphereMems = [], ringMems = [];
  for (const n of filtered.nodes) {
    if (n.kind !== "memory") continue;
    const links = n.entity_nodes.filter((id) => ents.has(id) && ents.get(id).visible);
    if (links.length && memPos.has(n.id)) sphereMems.push({ id: n.id, pos: memPos.get(n.id), verified: n.verified, links });
    else ringMems.push({ id: n.id, pos: ringSlot(n.id), angle: ringAngleOf(n.id), verified: n.verified });
  }
  const memPoints = makePoints(sphereMems, root, objects);
  const ringPoints = makePoints(ringMems, ringGroup, objects);

  // memory <-> entity lines
  const segs = [];
  for (const m of sphereMems) for (const id of m.links) segs.push({ mem: m.id, ent: id, a: m.pos, b: ents.get(id).pos });
  const lines = makeSegments(segs.map((s) => [s.a, s.b]), root, objects);

  // relations: great-circle arcs along the surface
  const arcs = [], arcPts = [];
  for (const e of filtered.edges) {
    if (e.kind !== "relation" || !ents.has(e.source) || !ents.has(e.target)) continue;
    const a = ents.get(e.source).pos, b = ents.get(e.target).pos;
    const from = arcs.length ? arcPts.length : 0;
    const steps = 24;
    for (let i = 0; i < steps; i++) arcPts.push([slerpDir(a, b, i / steps).multiplyScalar(1.002), slerpDir(a, b, (i + 1) / steps).multiplyScalar(1.002)]);
    arcs.push({ id: e.id, source: e.source, target: e.target, type: e.rel_type, verified: e.status === "verified",
                from, count: steps, mid: slerpDir(a, b, 0.5).multiplyScalar(1.03) });
  }
  const arcLines = makeSegments(arcPts, root, objects);

  // neighbourhoods, for hover and selection
  const nbrs = new Map();
  const link = (x, y) => { (nbrs.get(x) || nbrs.set(x, new Set()).get(x)).add(y); };
  for (const s of segs) { link(s.mem, s.ent); link(s.ent, s.mem); }
  for (const a of arcs) { link(a.source, a.target); link(a.target, a.source); }

  const topByMem = [...ents.values()].filter((e) => e.visible).sort((x, y) => y.n - x.n || (x.id < y.id ? -1 : 1));
  world = { info, ents, sphereMems, ringMems, memPoints, ringPoints, segs, lines, arcs, arcLines, nbrs, topByMem,
            objects, centre: largestCentre(g, pos),
            sphereIndex: new Map(sphereMems.map((m, i) => [m.id, i])), ringIndex: new Map(ringMems.map((m, i) => [m.id, i])) };
  if (selectedId && !info.has(selectedId)) { selectedId = null; closePanel(); }
  if (hoverId && !info.has(hoverId)) hoverId = null;
  if (panelId && info.has(panelId)) openPanel(panelId);   // rebuild with the new data
  computeSearch();
  applyLook();
}

function slerpDir(a, b, t) {
  const dot = Math.min(1, Math.max(-1, a.dot(b)));
  const th = Math.acos(dot);
  if (th < 1e-5) return a.clone();
  const s = Math.sin(th);
  return a.clone().multiplyScalar(Math.sin((1 - t) * th) / s).addScaledVector(b, Math.sin(t * th) / s).normalize();
}

function makePoints(items, parent, objects) {
  const geo = new THREE.BufferGeometry();
  const posArr = new Float32Array(items.length * 3), colArr = new Float32Array(items.length * 3);
  items.forEach((m, i) => { posArr.set([m.pos.x, m.pos.y, m.pos.z], 3 * i); });
  geo.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colArr, 3));
  const mat = new THREE.PointsMaterial({ size: 0.05, sizeAttenuation: true, vertexColors: true, map: pointTexture(),
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: true });
  const pts = new THREE.Points(geo, mat);
  parent.add(pts);
  objects.push(pts);
  return pts;
}

function makeSegments(pairs, parent, objects) {
  const geo = new THREE.BufferGeometry();
  const posArr = new Float32Array(pairs.length * 6), colArr = new Float32Array(pairs.length * 6);
  pairs.forEach(([a, b], i) => posArr.set([a.x, a.y, a.z, b.x, b.y, b.z], 6 * i));
  geo.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colArr, 3));
  const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, fog: true });
  const seg = new THREE.LineSegments(geo, mat);
  parent.add(seg);
  objects.push(seg);
  return seg;
}

// ---------- look: trust, hover, selection, search ----------
function focusId() { return hoverId || selectedId; }
function litSet() {
  const f = focusId();
  if (f && world) {
    const s = new Set([f]);
    (world.nbrs.get(f) || []).forEach((x) => s.add(x));
    return s;
  }
  return searchHits;
}

function applyLook() {
  if (!world) return;
  const f = focusId(), lit = litSet();
  const on = (id) => !lit || lit.has(id);
  for (const e of world.ents.values()) {
    if (!e.visible) continue;
    e.sprite.material.opacity = on(e.id) ? 1 : DIM;
    e.sprite.scale.setScalar(e.size * (e.id === f ? 1.35 : lit && lit.has(e.id) && f ? 1.12 : 1));
  }
  const vColor = lin(MEM_VERIFIED), uColor = lin(MEM_UNVERIFIED), tmp = new THREE.Color();
  const paintPoints = (pts, items) => {
    const col = pts.geometry.getAttribute("color");
    items.forEach((m, i) => {
      tmp.copy(m.verified ? vColor : uColor).multiplyScalar(on(m.id) ? (m.id === f ? 1.6 : 1) : DIM);
      col.setXYZ(i, tmp.r, tmp.g, tmp.b);
    });
    col.needsUpdate = true;
  };
  paintPoints(world.memPoints, world.sphereMems);
  paintPoints(world.ringPoints, world.ringMems);

  const lineColor = lin(LINE_COLOR), lc = world.lines.geometry.getAttribute("color");
  world.segs.forEach((s, i) => {
    if (f && (s.mem === f || s.ent === f)) tmp.copy(lin(world.ents.get(s.ent).color)).multiplyScalar(0.75);
    else tmp.copy(lineColor).multiplyScalar(lit ? DIM : 1);
    lc.setXYZ(2 * i, tmp.r, tmp.g, tmp.b); lc.setXYZ(2 * i + 1, tmp.r, tmp.g, tmp.b);
  });
  lc.needsUpdate = true;

  const ac = world.arcLines.geometry.getAttribute("color");
  for (const a of world.arcs) {
    const incident = f && (a.source === f || a.target === f);
    tmp.copy(lin(a.verified ? ARC_VERIFIED : ARC_UNVERIFIED))
      .multiplyScalar(incident ? 1.3 : lit && !(lit.has(a.source) && lit.has(a.target)) ? DIM : a.verified ? 0.7 : 1);
    for (let k = 0; k < a.count * 2; k++) ac.setXYZ(2 * a.from + k, tmp.r, tmp.g, tmp.b);
  }
  ac.needsUpdate = true;
}

// ---------- search: entity name or alias, over the filtered view ----------
function matches(n, q) {
  if (n.kind !== "entity") return false;
  if ((n.label || "").toLowerCase().includes(q)) return true;
  return (n.members || []).some((m) => (m.name || "").toLowerCase().includes(q) ||
    (m.aliases || []).some((a) => String(a).toLowerCase().includes(q)));
}
function computeSearch() {
  const q = searchQuery.trim().toLowerCase();
  if (!q || !world) { searchHits = null; return; }
  searchHits = new Set();
  for (const n of world.info.values()) if (n.kind === "entity" && matches(n, q)) searchHits.add(n.id);
}

// ---------- per-frame ----------
const v3 = new THREE.Vector3();
function worldPosOf(id) {
  if (!world) return null;
  const e = world.ents.get(id);
  if (e) return e.visible ? v3.copy(e.pos).applyMatrix4(root.matrixWorld) : null;
  let i = world.sphereIndex.get(id);
  if (i !== undefined) return v3.copy(world.sphereMems[i].pos).applyMatrix4(root.matrixWorld);
  i = world.ringIndex.get(id);
  if (i !== undefined) return v3.copy(world.ringMems[i].pos).applyMatrix4(ringGroup.matrixWorld);
  return null;
}
function toScreen(p) {
  const q = p.clone().project(camera);
  return { x: ((q.x + 1) / 2) * stage.clientWidth, y: ((1 - q.y) / 2) * stage.clientHeight };
}
function pxPerUnit(depth) {
  return stage.clientHeight / (2 * depth * Math.tan((FOV * Math.PI) / 360));
}

const labelPool = [];
function labelDiv(i) {
  while (labelPool.length <= i) {
    const d = document.createElement("div");
    d.className = "sphere-label";
    labelLayer.appendChild(d);
    labelPool.push(d);
  }
  return labelPool[i];
}

function updateLabels() {
  let used = 0;
  if (world) {
    const f = focusId(), lit = litSet();
    const forced = [];
    if (f) { if (world.ents.has(f)) forced.push(f); (world.nbrs.get(f) || []).forEach((x) => { if (world.ents.has(x)) forced.push(x); }); }
    else if (searchHits) forced.push(...searchHits);
    const frac = Math.min(1, 0.15 * Math.pow(HOME_DIST / camDist, 2.5));
    const top = world.topByMem.slice(0, Math.ceil(world.topByMem.length * frac)).map((e) => e.id);
    const seen = new Set(), boxes = [];
    const place = (text, p, isForced, cls) => {
      const s = toScreen(p), w = text.length * 6.4 + 10, h = 16;
      const box = [s.x + 6, s.y - h / 2, s.x + 6 + w, s.y + h / 2];
      if (!isForced && boxes.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) return;
      boxes.push(box);
      const d = labelDiv(used++);
      if (d.textContent !== text) d.textContent = text;
      d.className = "sphere-label" + (cls ? " " + cls : "");
      d.style.transform = `translate(${Math.round(box[0])}px, ${Math.round(box[1])}px)`;
      d.style.opacity = String(Math.max(0.25, Math.min(1, 0.55 + p.dot(CAM_DIR) * 0.6)) * (lit && !isForced ? DIM * 3 : 1));
      d.hidden = false;
    };
    for (const id of forced.concat(top)) {
      if (seen.has(id)) continue;
      seen.add(id);
      const e = world.ents.get(id);
      if (!e || !e.visible) continue;
      const p = worldPosOf(id).clone();
      const isForced = forced.includes(id);
      const front = p.dot(CAM_DIR);
      if (front <= 0.05 && !isForced) continue;   // front hemisphere only
      if (front <= -0.2 && isForced && id !== f) continue;
      place(e.label, p, isForced, isForced ? "strong" : "");
    }
    if (f) for (const a of world.arcs) {
      if (a.source !== f && a.target !== f) continue;
      place(a.type, a.mid.clone().applyMatrix4(root.matrixWorld), true, "rel");
    }
  }
  for (let i = used; i < labelPool.length; i++) labelPool[i].hidden = true;
}

function frame(t) {
  raf = 0;
  if (!shown || !renderer) return;
  const dt = lastT ? Math.min(100, t - lastT) : 16;
  lastT = t;
  const now = performance.now();

  if (anim) {
    const k = Math.min(1, (now - anim.t0) / FOCUS_MS), e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    if (anim.kind === "root") root.quaternion.slerpQuaternions(anim.from, anim.to, e);
    else ringAngle = anim.from + (anim.to - anim.from) * e;
    if (k >= 1) { const done = anim.done; anim = null; done && done(); }
  } else if (drag) {
    // rotation applied in the pointer handler
  } else if (spinVel.w > 1e-6) {
    root.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(spinVel.axis, spinVel.w * dt));
    spinVel.w *= Math.exp(-dt / 350);
  }
  // Auto-rotate pauses for the sphere's own items (hover, an open panel, drag, inertia, a focus turn), never
  // for the ring's. It eases out at once and back in after RESUME_MS.
  const spinOn = autoOn && !(drag && drag.moved) && !(anim && anim.kind === "root") && spinVel.w <= 1e-6 && !pausesSphere(hoverId) && !pausesSphere(panelId) &&
    now >= resumeAt;
  spinGain = ease(spinGain, spinOn ? 1 : 0, dt);
  if (spinGain > 1e-3) {
    root.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), (spinGain * SPIN * dt) / 1000));
  }
  // The ring stops while one of its dots is hovered or selected, while a panel is open, with auto-rotate off,
  // and under reduced motion; dragging the sphere doesn't stop it.
  const ringOn = !(anim && anim.kind === "ring") && autoOn && !reduced.matches && !panelId && !isRing(hoverId) &&
    !isRing(selectedId);
  ringGain = ease(ringGain, ringOn ? 1 : 0, dt);
  if (ringGain > 1e-3) ringAngle += (ringGain * RING_SPIN * dt) / 1000;
  ringGroup.rotation.y = ringAngle;
  setCamera();
  scene.updateMatrixWorld();
  const sel = selectedId && worldPosOf(selectedId);
  halo.visible = !!sel;
  if (sel) {
    halo.position.copy(sel);
    const e = world.ents.get(selectedId);
    halo.scale.setScalar(e ? e.size * 1.7 : 0.13);
  }
  renderer.render(scene, camera);
  updateLabels();
  // things drift under a resting pointer: re-pick so hovering doesn't need the mouse to move
  if (pointer && !drag && !anim) {
    const r = stage.getBoundingClientRect();
    setHover(pick(pointer.clientX - r.left, pointer.clientY - r.top), pointer);
  }
  raf = requestAnimationFrame(frame);
}
function isRing(id) { return !!(id && world && world.ringIndex.has(id)); }
function pausesSphere(id) { return !!id && !isRing(id); }
function ease(gain, target, dt) {
  const tau = target > gain ? EASE_IN_MS / 3 : EASE_OUT_MS / 3;
  const next = target + (gain - target) * Math.exp(-dt / tau);
  return Math.abs(next - target) < 1e-3 ? target : next;
}
function kick() { if (shown && renderer && !raf) { lastT = 0; raf = requestAnimationFrame(frame); } }

// ---------- picking (front-most hit within a few pixels) ----------
function pick(px, py) {
  if (!world) return null;
  let best = null, bestDepth = Infinity;
  const test = (id, p, radius) => {
    const s = toScreen(p);
    if (Math.hypot(s.x - px, s.y - py) > radius) return;
    const depth = camera.position.distanceTo(p);
    if (depth < bestDepth) { bestDepth = depth; best = id; }
  };
  for (const e of world.ents.values()) {
    if (!e.visible) continue;
    const p = e.pos.clone().applyMatrix4(root.matrixWorld);
    test(e.id, p, Math.max(6, 0.22 * e.size * pxPerUnit(camera.position.distanceTo(p))));
  }
  for (const m of world.sphereMems) test(m.id, m.pos.clone().applyMatrix4(root.matrixWorld), 5);
  for (const m of world.ringMems) test(m.id, m.pos.clone().applyMatrix4(ringGroup.matrixWorld), 9);
  return best;
}

function setHover(id, ev) {
  if (id !== hoverId) {
    if (pausesSphere(hoverId) && !pausesSphere(id)) resumeAt = performance.now() + RESUME_MS;
    hoverId = id;
    applyLook();
    stage.style.cursor = id ? "pointer" : "";
    showTip(id);
  }
  if (id && ev) placeTip(ev);
}
function showTip(id) {
  while (tip.firstChild) tip.removeChild(tip.firstChild);
  const n = id && world && world.info.get(id);
  if (!n) { tip.hidden = true; return; }
  const line = (cls, text) => { const d = document.createElement("div"); if (cls) d.className = cls; d.textContent = text; tip.appendChild(d); };
  if (n.kind === "entity") {
    line(null, n.label + " · " + n.type);
    line("sub", n.n_relations + " relations · " + n.n_memories + " memories");
  } else {
    line(null, n.text.length > 140 ? n.text.slice(0, 139) + "…" : n.text);
    line("sub", (n.verified ? "verified" : "unverified") + " memory" + (world.ringIndex.has(id) ? " · no visible entity" : ""));
  }
  tip.hidden = false;
}
function placeTip(ev) {
  if (tip.hidden) return;
  const r = stage.getBoundingClientRect();
  let x = ev.clientX - r.left + 14, y = ev.clientY - r.top + 14;
  x = Math.min(x, stage.clientWidth - tip.offsetWidth - 6);
  y = Math.min(y, stage.clientHeight - tip.offsetHeight - 6);
  tip.style.left = Math.max(4, x) + "px";
  tip.style.top = Math.max(4, y) + "px";
}

function wirePointer() {
  const el = renderer.domElement;
  el.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    try { el.setPointerCapture(ev.pointerId); } catch (err) { /* not an active pointer: drag without capture */ }
    drag = { x: ev.clientX, y: ev.clientY, x0: ev.clientX, y0: ev.clientY, t: performance.now(), moved: false };
    spinVel.w = 0;
    if (anim && anim.kind === "root") anim = null;
  });
  el.addEventListener("pointermove", (ev) => {
    if (drag) {
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      if (Math.hypot(ev.clientX - drag.x0, ev.clientY - drag.y0) > 4) drag.moved = true;
      if (!drag.moved) return;
      const len = Math.hypot(dx, dy);
      if (len > 0) {
        const ang = (len * Math.PI) / stage.clientHeight * (camDist / HOME_DIST) * 1.4;
        const axis = new THREE.Vector3(dy, dx, 0).normalize();
        root.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(axis, ang));
        const now = performance.now(), dtm = Math.max(1, now - drag.t);
        spinVel.axis.copy(axis);
        spinVel.w = ang / dtm;
        drag.t = now;
      }
      drag.x = ev.clientX; drag.y = ev.clientY;
      setHover(null);
      return;
    }
    pointer = { clientX: ev.clientX, clientY: ev.clientY };
    const r = stage.getBoundingClientRect();
    setHover(pick(ev.clientX - r.left, ev.clientY - r.top), ev);
  });
  const end = (ev) => {
    if (!drag) return;
    const d = drag;
    drag = null;
    if (d.moved) resumeAt = performance.now() + RESUME_MS;   // a click alone never paused the spin
    if (performance.now() - d.t > 80) spinVel.w = 0;   // released after holding still: no fling
    if (d.moved) return;
    spinVel.w = 0;
    const r = stage.getBoundingClientRect();
    const id = pick(ev.clientX - r.left, ev.clientY - r.top);
    if (id) focus(id); else { selectedId = null; closePanel(); applyLook(); }
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", () => { drag = null; });
  el.addEventListener("pointerleave", () => { pointer = null; if (!drag) setHover(null); });
  el.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    camDist = Math.min(MAX_DIST, Math.max(MIN_DIST, camDist * Math.exp(ev.deltaY * 0.001)));
  }, { passive: false });
}

// ---------- focus and panel ----------
function faceQuat(dir) {
  // yaw then pitch, so the picture stays upright: rotates `dir` onto +z, then up to the camera's direction
  const yaw = Math.atan2(dir.x, dir.z), pitch = Math.atan2(dir.y, Math.hypot(dir.x, dir.z));
  return new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), pitch - CAM_ELEV)
    .multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -yaw));
}

function focus(id) {
  if (!world || !world.info.has(id)) return;
  selectedId = id;
  spinVel.w = 0;
  applyLook();
  const after = () => openPanel(id);
  const ri = world.ringIndex.get(id);
  if (ri !== undefined) {
    // the ring turns the shortest way to bring the dot to the front centre; the sphere doesn't move
    const target = world.ringMems[ri].angle - Math.PI / 2;
    let delta = (target - ringAngle) % (2 * Math.PI);
    if (delta > Math.PI) delta -= 2 * Math.PI;
    if (delta < -Math.PI) delta += 2 * Math.PI;
    anim = { kind: "ring", from: ringAngle, to: ringAngle + delta, t0: performance.now(), done: after };
  } else {
    const p = worldPosOf(id);
    if (!p) return;
    const to = faceQuat(p.clone().normalize()).multiply(root.quaternion);
    anim = { kind: "root", from: root.quaternion.clone(), to, t0: performance.now(), done: after };
  }
  kick();
}

const nav = {
  exists: (id) => !!(world && world.info.has(id) && (!world.ents.has(id) || world.ents.get(id).visible)),
  select: (id) => focus(id),
  info: (id) => (world ? world.info.get(id) : undefined),
};

function openPanel(id) {
  const item = world && world.info.get(id);
  if (!item) return;
  panelId = id;
  while (panelBody.firstChild) panelBody.removeChild(panelBody.firstChild);
  G.buildPanel(panelBody, item, nav);
  panel.hidden = false;
  panel.scrollTop = 0;
}
function closePanel() {
  if (pausesSphere(panelId)) resumeAt = performance.now() + RESUME_MS;
  panelId = null;
  panel.hidden = true;
  while (panelBody.firstChild) panelBody.removeChild(panelBody.firstChild);
}

// ---------- data ----------
function resetView() {
  camDist = HOME_DIST;
  spinVel.w = 0;
  anim = null;
  ringAngle = 0;
  root.quaternion.copy(world ? faceQuat(world.centre) : new THREE.Quaternion());
}

async function accept(res, mine) {
  const g = buildGraph(res.full);
  let pos = cache.pos;
  const same = pos && cache.print === g.print && g.ents.every((id) => pos.has(id));
  if (!same) {
    G.setStatus("laying out…");
    pos = await runLayout(g, cache.pos, () => mine !== seq);
    if (!pos) return;   // a newer request took over
    G.setStatus("");
  }
  pos = separate(g, pos);
  const first = !cache.pos;
  cache.pos = pos;
  cache.print = g.print;
  buildScene(res, g, pos, placeMemories(res.full, pos));
  if (first) resetView();
  kick();
  return true;
}

function request() {
  if (!renderer) return Promise.resolve();
  const mine = ++seq;
  return G.api().sphere(G.readFilters()).then((res) => (mine === seq ? accept(res, mine) : null), G.fail);
}

function resetLayout() {
  cache.pos = null;
  cache.print = null;
  return request().then((ok) => { if (ok) G.setStatus("layout recomputed"); });
}

// ---------- the tab's hooks ----------
let loaded = false;
const view = {
  show() {
    if (!renderer && !initRenderer()) {
      $("sphere-nogl").hidden = false;
      G.switchTab("graph");
      G.setStatus("3D view needs WebGL — showing the 2D graph");
      return;
    }
    shown = true;
    resize();
    if (!loaded) { loaded = true; request(); }
    kick();
  },
  hide() {
    shown = false;
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    setHover(null);
  },
  filtersChanged() { if (renderer) request(); else loaded = false; },
  refreshed() { if (renderer) request(); else loaded = false; },
  search(q) { searchQuery = q || ""; computeSearch(); applyLook(); },
  searchEnter() {
    if (!searchHits || !searchHits.size || !world) return;
    const first = [...world.info.values()].find((n) => searchHits.has(n.id));
    if (first) focus(first.id);
  },
  escape() { selectedId = null; closePanel(); applyLook(); },
  focus,
  // Read-only state for checking the view from outside (screenshots and evaluate_js); changes nothing.
  inspect(id) {
    const out = {
      shown, autoOn, reduced: reduced.matches, camDist, ringAngle, hoverId, selectedId, panelId,
      rotation: root ? root.quaternion.toArray() : null, print: cache.print,
      entities: world ? [...world.ents.values()].filter((e) => e.visible).length : 0,
      sphereMemories: world ? world.sphereMems.length : 0, ringMemories: world ? world.ringMems.length : 0,
      ringIds: world ? world.ringMems.map((m) => m.id) : [],
      spinGain, ringGain,
    };
    if (id === "*closest" && world) {
      // the two entities whose dots come closest, as distance / (size a + size b)
      const es = [...world.ents.values()];
      let best = null;
      for (let i = 0; i < es.length; i++) for (let j = i + 1; j < es.length; j++) {
        const r = es[i].pos.distanceTo(es[j].pos) / (es[i].size + es[j].size);
        if (!best || r < best.ratio) best = { ratio: r, a: es[i].label, b: es[j].label };
      }
      out.closest = best;
    }
    if (id && world) {
      const p = worldPosOf(id);
      if (p) {
        const r = stage.getBoundingClientRect(), s = toScreen(p.clone());
        out.screen = { x: r.left + s.x, y: r.top + s.y, front: p.dot(CAM_DIR) };
        const e = world.ents.get(id);
        if (e) out.position = e.pos.toArray();
      }
    }
    return out;
  },
};

// The page shell (graph.html / graph.js) provides the stage and window.remGraph; without them, stay inert.
if (G && typeof G.registerView === "function" && stage) {
  $("sphere-autorotate").checked = autoOn;
  $("sphere-autorotate").addEventListener("change", (e) => { autoOn = e.target.checked; resumeAt = 0; });
  $("sphere-reset-view").addEventListener("click", () => resetView());
  $("sphere-reset-layout").addEventListener("click", () => resetLayout());
  $("sphere-panel-close").addEventListener("click", () => view.escape());
  reduced.addEventListener("change", () => {
    if (reduced.matches) { autoOn = false; $("sphere-autorotate").checked = false; }
  });
  G.registerView("sphere", view);
}
