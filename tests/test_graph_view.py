import copy
from collections import Counter

from rem.graph.view import build_view, default_filters

ALL = ["global", "system", "project"]
PROJECTS = [{"id": 1, "name": "one", "root_path": "/r/one"}, {"id": 2, "name": "two", "root_path": "/r/two"}]


def snap(entities=(), relations=(), memories=(), projects=PROJECTS, current=None):
    return {
        "generated_at": "2026-11-17T10:00:00+00:00",
        "current_project_id": current,
        "projects": list(projects),
        "entities": list(entities),
        "relations": list(relations),
        "memories": list(memories),
    }


def ent(id, name, *, normalized=None, type="tool", scope="global", project_id=None, aliases=(), memory_ids=()):
    return {"id": id, "name": name, "normalized": normalized if normalized is not None else name.lower(),
            "type": type, "aliases": list(aliases), "scope": scope, "project_id": project_id,
            "memory_ids": list(memory_ids), "created_at": "t", "updated_at": "t"}


def rel(id, src, dst, *, type="uses", scope="global", project_id=None, status="verified", memory_ids=("m",)):
    return {"id": id, "source_entity_id": src, "target_entity_id": dst, "type": type,
            "sentence": f"{src} {type} {dst}", "memory_ids": list(memory_ids), "provenance": "user_asserted",
            "status": status, "scope": scope, "project_id": project_id, "created_at": "t", "updated_at": "t"}


def mem(id, text="some fact", *, scope="global", project_id=None, status="verified", entity_ids=(), job_id=None):
    names = {1: "one", 2: "two"}
    return {"id": id, "text": text, "scope": scope, "project_id": project_id,
            "project_name": names.get(project_id), "provenance": "user_asserted", "status": status,
            "agent_claim": False, "entity_ids": list(entity_ids), "source": "extraction",
            "job_id": job_id, "created_at": "t", "updated_at": "t"}


def view(s, project="all", scopes=ALL):
    return build_view(s, project=project, scopes=scopes)


def nodes_by_id(v):
    return {n["id"]: n for n in v["nodes"]}


def entity_nodes(v):
    return [n for n in v["nodes"] if n["kind"] == "entity"]


def all_members(v):
    return [m for n in entity_nodes(v) for m in n["members"]]


def test_groups_only_identical_normalized_names():
    s = snap(entities=[
        ent("e1", "uv"),
        ent("e2", "uv", scope="project", project_id=1),
        ent("e3", "UV-tool", normalized="uv-tool", aliases=["uv"]),
    ])
    v = view(s)
    ns = nodes_by_id(v)
    assert set(ns) == {"ent:uv", "ent:uv-tool"}
    assert {m["entity_id"] for m in ns["ent:uv"]["members"]} == {"e1", "e2"}
    assert [m["entity_id"] for m in ns["ent:uv-tool"]["members"]] == ["e3"]
    assert ns["ent:uv-tool"]["members"][0]["aliases"] == ["uv"]


def _rich_snapshot():
    return snap(
        entities=[
            ent("e1", "uv", memory_ids=["m1"]),
            ent("e2", "uv", scope="project", project_id=1, memory_ids=["m2"]),
            ent("e3", "qdrant", scope="project", project_id=1),
            ent("e4", "pip", scope="system"),
        ],
        relations=[
            rel("r1", "e2", "e3", scope="project", project_id=1),
            rel("r2", "e1", "e4"),
        ],
        memories=[
            mem("m1", "uses uv", entity_ids=["e1"]),
            mem("m2", "rem uses uv", scope="project", project_id=1, entity_ids=["e2", "e3"]),
        ],
    )


def test_members_preserve_ids_scope_project():
    s = _rich_snapshot()
    v = view(s)
    assert Counter(m["entity_id"] for m in all_members(v)) == Counter(e["id"] for e in s["entities"])
    by_id = {e["id"]: e for e in s["entities"]}
    for m in all_members(v):
        src = by_id[m["entity_id"]]
        assert m["scope"] == src["scope"] and m["project_id"] == src["project_id"]
        own_rels = {r["id"] for r in s["relations"] if src["id"] in (r["source_entity_id"], r["target_entity_id"])}
        assert {r["id"] for r in m["relations"]} == own_rels
        own_mems = {x["id"] for x in s["memories"] if src["id"] in x["entity_ids"] or x["id"] in src["memory_ids"]}
        assert {x["id"] for x in m["memories"]} == own_mems
    m2 = next(m for m in all_members(v) if m["entity_id"] == "e2")
    assert m2["project_name"] == "one"
    assert m2["relations"][0]["direction"] == "out" and m2["relations"][0]["other"] == "qdrant"
    assert m2["relations"][0]["other_node"] == "ent:qdrant" and m2["relations"][0]["project_name"] == "one"


def test_build_view_does_not_mutate_snapshot():
    s = _rich_snapshot()
    before = copy.deepcopy(s)
    v = view(s)
    assert s == before
    # the output holds no references into the input
    for m in all_members(v):
        m["aliases"].append("x")
    assert s == before


def test_filters_apply_before_grouping():
    s = snap(
        entities=[
            ent("a", "uv", scope="project", project_id=1, memory_ids=["m1"]),
            ent("b", "uv", scope="project", project_id=2, memory_ids=["m2"]),
            ent("c", "ruff", scope="project", project_id=2),
        ],
        relations=[rel("r2", "b", "c", scope="project", project_id=2)],
        memories=[mem("m1", scope="project", project_id=1, entity_ids=["a"]),
                  mem("m2", scope="project", project_id=2, entity_ids=["b"])],
    )
    v = view(s, project=1)
    ns = nodes_by_id(v)
    assert [m["entity_id"] for m in ns["ent:uv"]["members"]] == ["a"]
    assert "ent:ruff" not in ns
    assert v["edges"] == [{"id": "men:m1:uv", "kind": "mention", "source": "mem:m1", "target": "ent:uv"}]
    for n in v["nodes"]:
        assert n["project_id"] != 2 if n["kind"] == "memory" else True
        for m in n.get("members", []):
            assert m["project_id"] != 2
            assert all(r["project_id"] != 2 for r in m["relations"])
            assert all(x["project_id"] != 2 for x in m["memories"])
        assert all(r["project_id"] != 2 for r in n.get("relations", []))
        assert all(x["project_id"] != 2 for x in n.get("memories", []))
    assert len(nodes_by_id(view(s, project="all"))["ent:uv"]["members"]) == 2


def test_scope_checkboxes_hide_records():
    s = snap(
        entities=[
            ent("g", "uv", memory_ids=["mg"]),
            ent("s", "uv", scope="system"),
            ent("p", "uv", scope="project", project_id=1),
            ent("q", "pip", scope="project", project_id=1),
        ],
        relations=[rel("rg", "g", "s"), rel("rp", "p", "q", scope="project", project_id=1)],
        memories=[mem("mg", entity_ids=["g"]), mem("mp", scope="project", project_id=1, entity_ids=["p"])],
    )
    v = view(s, scopes=["project"])
    ns = nodes_by_id(v)
    assert [m["entity_id"] for m in ns["ent:uv"]["members"]] == ["p"]
    assert {r["id"] for r in ns["ent:uv"]["relations"]} == {"rp"}
    assert {x["id"] for x in ns["ent:uv"]["memories"]} == {"mp"}
    assert "mem:mg" not in ns and "mem:mp" in ns
    assert {e["id"] for e in v["edges"]} == {"rel:rp", "men:mp:uv"}


def _link_exists(s, mid, eid):
    m = next(x for x in s["memories"] if x["id"] == mid)
    e = next(x for x in s["entities"] if x["id"] == eid)
    return eid in m["entity_ids"] or mid in e["memory_ids"]


def _anti_inference_snapshots():
    cross_project = snap(
        entities=[ent("a", "uv", scope="project", project_id=1), ent("b", "uv", scope="project", project_id=2),
                  ent("c", "ruff", scope="project", project_id=1), ent("d", "pip", scope="project", project_id=2)],
        relations=[rel("r1", "a", "c", scope="project", project_id=1), rel("r2", "b", "d", scope="project", project_id=2)],
        memories=[mem("m1", scope="project", project_id=1, entity_ids=["a"])],
    )
    alias = snap(
        entities=[ent("x", "postgres", aliases=["postgresql"]), ent("y", "postgresql"), ent("z", "rem")],
        relations=[rel("r1", "z", "x")],
        memories=[mem("m1", entity_ids=["y"]), mem("m2", entity_ids=["x"])],
    )
    one_sided = snap(
        entities=[ent("e1", "uv", memory_ids=["m1"]), ent("e2", "pip")],
        relations=[],
        memories=[mem("m1"), mem("m2", entity_ids=["e2"]), mem("m3")],
    )
    in_group = snap(
        entities=[ent("e1", "uv"), ent("e2", "uv", normalized="uv", scope="system"), ent("e3", "pip")],
        relations=[rel("r1", "e1", "e2"), rel("r2", "e2", "e3", scope="system")],
        memories=[mem("m1", entity_ids=["e1", "e2"])],
    )
    return [cross_project, alias, one_sided, in_group]


def test_every_edge_is_stored():
    for s in _anti_inference_snapshots():
        for project in ("all", 1, 2):
            v = view(s, project=project)
            ns = nodes_by_id(v)
            rels = {r["id"]: r for r in s["relations"]}
            n_rel_edges = 0
            for edge in v["edges"]:
                if edge["kind"] == "relation":
                    n_rel_edges += 1
                    r = rels[edge["id"].removeprefix("rel:")]
                    src_ids = {m["entity_id"] for m in ns[edge["source"]]["members"]}
                    dst_ids = {m["entity_id"] for m in ns[edge["target"]]["members"]}
                    assert r["source_entity_id"] in src_ids and r["target_entity_id"] in dst_ids
                    assert edge["source"] != edge["target"]
                else:
                    assert edge["kind"] == "mention"
                    mid = edge["source"].removeprefix("mem:")
                    members = ns[edge["target"]]["members"]
                    assert any(_link_exists(s, mid, m["entity_id"]) for m in members)
            assert n_rel_edges <= len(s["relations"])


def test_self_loop_after_grouping_is_not_drawn_but_listed():
    s = snap(entities=[ent("e1", "uv"), ent("e2", "uv")], relations=[rel("r1", "e1", "e2")])
    v = view(s)
    assert v["edges"] == []
    node = nodes_by_id(v)["ent:uv"]
    assert node["n_relations"] == 1
    for m in node["members"]:
        assert [r["id"] for r in m["relations"]] == ["r1"]
        assert m["relations"][0]["drawn"] is False
    dirs = {m["entity_id"]: m["relations"][0]["direction"] for m in node["members"]}
    assert dirs == {"e1": "out", "e2": "in"}


def test_link_rule_either_side_one_edge_per_pair():
    s = snap(
        entities=[ent("e1", "uv", memory_ids=["M"]), ent("e2", "uv", scope="project", project_id=1)],
        memories=[mem("M", entity_ids=["e2"]), mem("lone")],
    )
    v = view(s)
    assert [e["id"] for e in v["edges"]] == ["men:M:uv"]
    ns = nodes_by_id(v)
    assert ns["mem:lone"]["entity_nodes"] == []
    assert ns["mem:M"]["entity_nodes"] == ["ent:uv"]
    for m in ns["ent:uv"]["members"]:
        assert [x["id"] for x in m["memories"]] == ["M"]
    assert ns["ent:uv"]["n_memories"] == 1


def test_dangling_ids_are_skipped():
    s = snap(
        entities=[ent("e1", "uv", memory_ids=["gone"])],
        relations=[rel("r1", "e1", "missing")],
        memories=[mem("m1", entity_ids=["absent"])],
    )
    v = view(s)
    assert v["edges"] == []
    node = nodes_by_id(v)["ent:uv"]
    assert node["relations"] == [] and node["memories"] == []
    assert node["members"][0]["relations"] == [] and node["members"][0]["memories"] == []
    assert nodes_by_id(v)["mem:m1"]["entity_nodes"] == []


def test_node_verified_derivation():
    s = snap(
        entities=[ent("e1", "uv"), ent("e2", "pip"), ent("e3", "ruff")],
        memories=[mem("m1", status="unverified", entity_ids=["e1", "e2"]), mem("m2", entity_ids=["e1"])],
    )
    ns = nodes_by_id(view(s))
    assert ns["ent:uv"]["verified"] is True
    assert ns["ent:pip"]["verified"] is False
    assert ns["ent:ruff"]["verified"] is False


def test_mixed_types_and_members_keep_their_own():
    s = snap(entities=[ent("e1", "rem", type="tool"), ent("e2", "rem", type="concept", scope="system")])
    node = nodes_by_id(view(s))["ent:rem"]
    assert node["type"] == "mixed"
    assert {m["entity_id"]: m["type"] for m in node["members"]} == {"e1": "tool", "e2": "concept"}
    same = nodes_by_id(view(snap(entities=[ent("e1", "rem"), ent("e2", "rem", scope="system")])))["ent:rem"]
    assert same["type"] == "tool"


def test_label_and_member_order_and_sorting():
    s = snap(entities=[ent("z", "Uv", normalized="uv", scope="project", project_id=2),
                       ent("y", "UV", normalized="uv", scope="global"),
                       ent("a", "aaa")],
             relations=[rel("r", "y", "a"), rel("q", "z", "a", scope="project", project_id=2)])
    v = view(s)
    node = nodes_by_id(v)["ent:uv"]
    assert [m["entity_id"] for m in node["members"]] == ["y", "z"]
    assert node["label"] == "UV"
    assert [n["id"] for n in v["nodes"]] == sorted(n["id"] for n in v["nodes"])
    assert [e["id"] for e in v["edges"]] == ["rel:q", "rel:r"]
    edge = v["edges"][1]
    assert edge == {"id": "rel:r", "kind": "relation", "source": "ent:uv", "target": "ent:aaa", "rel_type": "uses",
                    "status": "verified", "provenance": "user_asserted", "sentence": "y uses a", "evidence": 1,
                    "scope": "global", "project_name": None}


def test_default_filters():
    assert default_filters(snap(current=3)) == {"project": 3, "scopes": ["global", "system", "project"]}
    assert default_filters(snap(current=None))["project"] == "all"
    assert default_filters(snap(current=0))["project"] == 0


def test_empty_snapshot_gives_empty_view():
    assert view(snap()) == {"nodes": [], "edges": []}


def test_filtered_out_relation_between_kept_entities_is_dropped():
    s = snap(
        entities=[ent("e1", "uv"), ent("e2", "pip")],
        relations=[rel("rp", "e1", "e2", scope="project", project_id=2), rel("rs", "e1", "e2", scope="system")],
    )
    for v in (view(s, project=1, scopes=["global", "project"]), view(s, project=1, scopes=["global"])):
        assert v["edges"] == []
        for n in entity_nodes(v):
            assert n["relations"] == [] and n["n_relations"] == 0
            assert all(m["relations"] == [] for m in n["members"])
    # sanity: with both relations in view, both are drawn
    assert {e["id"] for e in view(s, project=2)["edges"]} == {"rel:rp", "rel:rs"}


def test_memory_project_name_falls_back_to_projects_lookup():
    m = mem("m1", scope="project", project_id=1, entity_ids=["e1"])
    m["project_name"] = None
    s = snap(entities=[ent("e1", "uv", scope="project", project_id=1)], memories=[m])
    v = view(s)
    assert nodes_by_id(v)["mem:m1"]["project_name"] == "one"
    assert nodes_by_id(v)["ent:uv"]["memories"][0]["project_name"] == "one"


def test_relation_identity_scope_and_other_entity_id():
    s = _rich_snapshot()
    v = view(s)
    rels = {r["id"]: r for r in s["relations"]}
    ents = {e["id"]: e for e in s["entities"]}
    for edge in (e for e in v["edges"] if e["kind"] == "relation"):
        assert edge["scope"] == rels[edge["id"].removeprefix("rel:")]["scope"]
    for m in all_members(v):
        for r in m["relations"]:
            stored = rels[r["id"]]
            want = stored["target_entity_id"] if r["direction"] == "out" else stored["source_entity_id"]
            assert r["other_entity_id"] == want
            assert r["other"] == ents[want]["name"]
            assert r["scope"] == stored["scope"]
    m2 = next(m for m in all_members(v) if m["entity_id"] == "e2")
    assert m2["relations"][0]["other_entity_id"] == "e3"
    assert {e["id"]: e["scope"] for e in v["edges"] if e["kind"] == "relation"} == {"rel:r1": "project", "rel:r2": "global"}


def test_memory_nodes_and_entries_carry_job_id():
    s = snap(entities=[ent("e1", "uv")],
             memories=[mem("m1", entity_ids=["e1"], job_id="job:7"), mem("m2", entity_ids=["e1"])])
    v = view(s)
    n = nodes_by_id(v)
    assert n["mem:m1"]["job_id"] == "job:7"
    assert n["mem:m2"]["job_id"] is None
    assert {m["id"]: m["job_id"] for m in n["ent:uv"]["memories"]} == {"m1": "job:7", "m2": None}
    assert {m["id"]: m["job_id"] for m in n["ent:uv"]["members"][0]["memories"]} == {"m1": "job:7", "m2": None}
