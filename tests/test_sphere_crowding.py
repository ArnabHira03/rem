"""Edge cases for the Sphere tab's crowding: once entities fill the sphere, every dot shrinks by one factor.

These run the real layout code from sphere.js (the pure section between the geometry constants and runLayout)
in Node, so they skip where Node isn't installed.
"""
import json
import shutil
import subprocess
from importlib.resources import files

import pytest

STATIC = files("rem.graph") / "static"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs node to run sphere.js")

START = "// ---------- geometry and motion ----------"
END = "// Runs in chunks across animation frames."

# Builds a synthetic full view per case, lays it out as the page does (Fibonacci seed, force layout, separate),
# and reports the shrink factor, the sizes and the tightest pair as distance / (size a + size b).
HARNESS = r"""
const src = require("fs").readFileSync(0, "utf8");
const [body, cases] = JSON.parse(src);
const m = new Function(body.replace(/^const CAM_DIR.*$/m, "") +
  "\nreturn { buildGraph, separate, seedPositions, layoutStep, itersFor, entitySize, FULL_ITERS, SEPARATION," +
  " CROWD_MIN, CROWD_STEP };")();
const out = cases.map(({ counts, layout }) => {
  const nodes = counts.map((n, i) => ({ id: "e" + String(i).padStart(6, "0"), kind: "entity", n_memories: n }));
  // a chain of shared memories, so the force layout pulls neighbours into clusters
  for (let i = 0; i + 1 < counts.length; i++)
    nodes.push({ id: "m" + i, kind: "memory", entity_nodes: [nodes[i].id, nodes[(i * 7 + 1) % counts.length].id] });
  const g = m.buildGraph({ nodes, edges: [] });
  const raw = g.ents.map((id) => m.entitySize(counts[Number(id.slice(1))]));
  let worst = null;
  if (layout) {
    const P = m.seedPositions(g, null), F = new Float64Array(P.length), iters = m.itersFor(g.ents.length, m.FULL_ITERS);
    for (let it = 0; it < iters; it++) m.layoutStep(g, P, F, 0.12 * (1 - it / iters) + 0.002);
    const pos = new Map(g.ents.map((id, i) => [id, [P[3 * i], P[3 * i + 1], P[3 * i + 2]]]));
    const sep = m.separate(g, pos), Q = g.ents.map((id) => sep.get(id));
    worst = Infinity;
    for (let i = 0; i < Q.length; i++) for (let j = i + 1; j < Q.length; j++) {
      const d = Math.hypot(Q[i][0] - Q[j][0], Q[i][1] - Q[j][1], Q[i][2] - Q[j][2]);
      worst = Math.min(worst, d / (g.sizes[i] + g.sizes[j]));
    }
  }
  return { scale: g.scale, sizes: g.sizes, raw, worst };
});
process.stdout.write(JSON.stringify({ out, SEPARATION: m.SEPARATION, CROWD_MIN: m.CROWD_MIN, CROWD_STEP: m.CROWD_STEP }));
"""


def run(*cases):
    text = (STATIC / "sphere.js").read_text(encoding="utf-8")
    assert START in text and END in text, "sphere.js section markers moved; update this test"
    body = text[text.index(START):text.index(END)]
    proc = subprocess.run([NODE, "-e", HARNESS], input=json.dumps([body, list(cases)]),
                          capture_output=True, text=True, timeout=120, check=True)
    return json.loads(proc.stdout)


def case(counts, layout=False):
    return {"counts": list(counts), "layout": layout}


def test_empty_and_single_entity_keep_full_size():
    r = run(case([]), case([5]))
    assert [c["scale"] for c in r["out"]] == [1, 1]
    assert r["out"][1]["sizes"] == r["out"][1]["raw"]


def test_a_sparse_sphere_is_left_alone():
    r = run(case([3] * 200, layout=True))["out"][0]
    assert r["scale"] == 1
    assert r["sizes"] == r["raw"]
    assert r["worst"] >= 0.3 - 1e-6


def test_a_crowded_sphere_shrinks_dots_and_none_overlap():
    counts = [(i * 37) % 60 for i in range(1500)]
    r = run(case(counts, layout=True))
    c = r["out"][0]
    assert r["CROWD_MIN"] <= c["scale"] < 1
    assert c["worst"] >= r["SEPARATION"] - 1e-6, "two dots overlap"


def test_every_dot_shrinks_by_the_same_factor():
    counts = [(i * 13) % 80 for i in range(1500)]
    c = run(case(counts))["out"][0]
    for s, raw in zip(c["sizes"], c["raw"]):
        assert s == pytest.approx(raw * c["scale"])
    # so a dot with more memories is still at least as big
    order = sorted(range(len(counts)), key=lambda i: c["raw"][i])
    assert all(c["sizes"][a] <= c["sizes"][b] + 1e-12 for a, b in zip(order, order[1:]))


def test_the_factor_moves_in_steps_and_never_grows_with_more_entities():
    r = run(*(case([10] * n) for n in range(300, 3001, 150)))
    step = r["CROWD_STEP"]
    scales = [c["scale"] for c in r["out"]]
    for s in scales:
        assert s == 1 or abs(s / step - round(s / step)) < 1e-9
    assert scales == sorted(scales, reverse=True)


def test_one_more_entity_rarely_changes_the_factor():
    r = run(*(case([10] * n) for n in range(1000, 1100)))
    scales = [c["scale"] for c in r["out"]]
    changes = sum(a != b for a, b in zip(scales, scales[1:]))
    assert changes <= 2


def test_a_huge_graph_stops_at_the_minimum_factor():
    r = run(case([60] * 20000))
    assert r["out"][0]["scale"] == r["CROWD_MIN"]
