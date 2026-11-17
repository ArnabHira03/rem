"""Pure view builder for the graph window: filter, then group by normalized name, then stored edges only.

Display only. Records are grouped into one node iff their `normalized` names are identical; aliases and
similar names never merge. Every edge is a stored relation or a stored memory<->entity link. Nothing here
infers a relationship, and the snapshot is never mutated.
"""

from collections.abc import Collection
from typing import Literal

ALL_SCOPES = ["global", "system", "project"]


def default_filters(snapshot: dict) -> dict:
    current = snapshot.get("current_project_id")
    return {"project": "all" if current is None else current, "scopes": list(ALL_SCOPES), "show_memories": False}


def _keep(record: dict, project: int | Literal["all"], scopes: set[str]) -> bool:
    scope = record.get("scope")
    return scope in scopes and (scope != "project" or project == "all" or record.get("project_id") == project)


def _node_id(entity: dict) -> str:
    return f"ent:{entity['normalized']}"


def build_view(
    snapshot: dict,
    *,
    project: int | Literal["all"],
    scopes: Collection[str],
    show_memories: bool,
) -> dict:
    scope_set = set(scopes)
    project_names = {p["id"]: p["name"] for p in snapshot.get("projects", [])}

    def pname(record: dict) -> str | None:
        pid = record.get("project_id")
        return project_names.get(pid) if pid is not None else None

    # 1. filter each kind independently, before any grouping
    entities = {e["id"]: e for e in snapshot.get("entities", []) if _keep(e, project, scope_set)}
    relations = [r for r in snapshot.get("relations", []) if _keep(r, project, scope_set)]
    memories = {m["id"]: m for m in snapshot.get("memories", []) if _keep(m, project, scope_set)}

    # 2. group kept entities by normalized name only
    groups: dict[str, list[dict]] = {}
    for e in entities.values():
        groups.setdefault(_node_id(e), []).append(e)

    member_rels: dict[str, dict[str, dict]] = {eid: {} for eid in entities}
    member_mems: dict[str, dict[str, dict]] = {eid: {} for eid in entities}
    edges: list[dict] = []

    # 3. relations: both ends must be kept entities; edge only when the ends land on different nodes
    for r in relations:
        src, dst = entities.get(r["source_entity_id"]), entities.get(r["target_entity_id"])
        if src is None or dst is None:
            continue
        src_node, dst_node = _node_id(src), _node_id(dst)
        drawn = src_node != dst_node
        for direction, own, other, other_node in (("out", src, dst, dst_node), ("in", dst, src, src_node)):
            member_rels[own["id"]].setdefault(r["id"], {
                "id": r["id"], "direction": direction, "type": r["type"], "other": other["name"],
                "other_node": other_node, "status": r["status"], "provenance": r["provenance"],
                "scope": r["scope"], "project_id": r.get("project_id"), "project_name": pname(r), "drawn": drawn,
            })
        if drawn:
            edges.append({
                "id": f"rel:{r['id']}", "kind": "relation", "source": src_node, "target": dst_node,
                "rel_type": r["type"], "status": r["status"], "provenance": r["provenance"],
                "sentence": r.get("sentence"), "evidence": len(r.get("memory_ids", [])), "project_name": pname(r),
            })

    # 4. links (M, E): both kept, and stored on either side
    links: set[tuple[str, str]] = set()
    for m in memories.values():
        for eid in m.get("entity_ids", []):
            if eid in entities:
                links.add((m["id"], eid))
    for e in entities.values():
        for mid in e.get("memory_ids", []):
            if mid in memories:
                links.add((mid, e["id"]))

    def mem_entry(m: dict) -> dict:
        return {"id": m["id"], "text": m["text"], "status": m["status"], "provenance": m["provenance"],
                "scope": m["scope"], "project_id": m.get("project_id"),
                "project_name": m.get("project_name", pname(m))}

    mem_nodes: dict[str, set[str]] = {mid: set() for mid in memories}
    for mid, eid in links:
        member_mems[eid][mid] = mem_entry(memories[mid])
        mem_nodes[mid].add(_node_id(entities[eid]))

    nodes: list[dict] = []
    for node_id, group in groups.items():
        group = sorted(group, key=lambda e: (e["scope"], pname(e) or "", e["id"]))
        members = []
        node_rels: dict[str, dict] = {}
        node_mems: dict[str, dict] = {}
        for e in group:
            rels = [member_rels[e["id"]][k] for k in sorted(member_rels[e["id"]])]
            mems = [member_mems[e["id"]][k] for k in sorted(member_mems[e["id"]])]
            for r in rels:
                node_rels.setdefault(r["id"], r)
            for m in mems:
                node_mems.setdefault(m["id"], m)
            members.append({
                "entity_id": e["id"], "name": e["name"], "type": e["type"], "aliases": list(e.get("aliases", [])),
                "scope": e["scope"], "project_id": e.get("project_id"), "project_name": pname(e),
                "relations": [dict(r) for r in rels], "memories": [dict(m) for m in mems],
            })
        types = {e["type"] for e in group}
        node_rel_list = [dict(node_rels[k]) for k in sorted(node_rels)]
        node_mem_list = [dict(node_mems[k]) for k in sorted(node_mems)]
        nodes.append({
            "id": node_id, "kind": "entity", "label": group[0]["name"],
            "type": types.pop() if len(types) == 1 else "mixed",
            "verified": any(m["status"] == "verified" for m in node_mem_list),
            "n_relations": len(node_rel_list), "n_memories": len(node_mem_list),
            "members": members, "relations": node_rel_list, "memories": node_mem_list,
        })

    if show_memories:
        for mid, m in memories.items():
            nodes.append({
                "id": f"mem:{mid}", "kind": "memory", "label": m["text"][:80], "text": m["text"],
                "verified": m["status"] == "verified", "status": m["status"], "provenance": m["provenance"],
                "scope": m["scope"], "project_id": m.get("project_id"), "project_name": m.get("project_name", pname(m)),
                "entity_nodes": sorted(mem_nodes[mid]),
            })
        for mid, node_ids in mem_nodes.items():
            for node_id in node_ids:
                edges.append({"id": f"men:{mid}:{node_id.removeprefix('ent:')}", "kind": "mention",
                              "source": f"mem:{mid}", "target": node_id})

    nodes.sort(key=lambda n: n["id"])
    edges.sort(key=lambda e: e["id"])
    return {"nodes": nodes, "edges": edges}
