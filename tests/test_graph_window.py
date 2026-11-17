import importlib
import re
import sys
from importlib.resources import files

import pytest

STATIC = files("rem.graph") / "static"

PROJECTS = [{"id": 1, "name": "one", "root_path": "/r/one"}, {"id": 3, "name": "three", "root_path": "/r/three"}]


def ent(id, name, *, scope="global", project_id=None):
    return {"id": id, "name": name, "normalized": name.lower(), "type": "tool", "aliases": [], "scope": scope,
            "project_id": project_id, "memory_ids": [], "created_at": "t", "updated_at": "t"}


def mem(id, *, scope="global", project_id=None, entity_ids=()):
    return {"id": id, "text": f"fact {id}", "scope": scope, "project_id": project_id, "project_name": None,
            "provenance": "user_asserted", "status": "verified", "agent_claim": False,
            "entity_ids": list(entity_ids), "source": "extraction", "created_at": "t", "updated_at": "t"}


def snapshot(projects=PROJECTS, current=3):
    return {
        "generated_at": "2026-11-17T10:00:00+00:00",
        "current_project_id": current,
        "projects": list(projects),
        "entities": [ent("e1", "uv"), ent("e2", "qdrant", scope="project", project_id=1),
                     ent("e3", "pytest", scope="project", project_id=3)],
        "relations": [],
        "memories": [mem("m1", entity_ids=["e1"])],
    }


class FakeClient:
    def __init__(self, snaps):
        self.snaps = list(snaps)
        self.calls = []

    def graph(self, cwd=None):
        self.calls.append(cwd)
        return self.snaps[min(len(self.calls), len(self.snaps)) - 1]


def node_ids(view):
    return sorted(n["id"] for n in view["nodes"])


def test_render_html_is_self_contained():
    from rem.graph.window import render_html

    html = render_html()
    assert "<style>" in html
    assert (STATIC / "vendor" / "cytoscape.min.js").read_text(encoding="utf-8") in html
    assert (STATIC / "graph.js").read_text(encoding="utf-8") in html
    assert (STATIC / "graph.css").read_text(encoding="utf-8") in html
    for bad in ('src="http', 'href="http', "<script src="):
        assert bad not in html


def test_page_never_uses_innerhtml():
    assert "innerHTML" not in (STATIC / "graph.js").read_text(encoding="utf-8")
    assert "outerHTML" not in (STATIC / "graph.js").read_text(encoding="utf-8")
    assert "insertAdjacentHTML" not in (STATIC / "graph.js").read_text(encoding="utf-8")


def test_page_is_black_and_white():
    seen = 0
    for name in ("graph.css", "graph.js", "graph.html"):
        text = (STATIC / name).read_text(encoding="utf-8")
        for h in re.findall(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b", text):
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            r, g, b = (h[i:i + 2].lower() for i in (0, 2, 4))
            assert r == g == b, f"{name}: #{h} has a hue"
            seen += 1
        for r, g, b in re.findall(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text):
            assert r == g == b, f"{name}: rgb({r},{g},{b}) has a hue"
            seen += 1
        assert not re.search(r"\bhsla?\(", text), f"{name} uses hsl()"
    assert seen > 0


def test_api_initial_view_and_refresh():
    from rem.graph.window import Api

    gone = snapshot(projects=[PROJECTS[0]])
    client = FakeClient([snapshot(), snapshot(), gone])
    api = Api(client, cwd="/r/three")

    first = api.initial()
    assert set(first) == {"filters", "projects", "generated_at", "view"}
    assert first["filters"]["project"] == 3
    assert first["projects"] == PROJECTS
    assert first["generated_at"] == "2026-11-17T10:00:00+00:00"
    assert "ent:qdrant" not in node_ids(first["view"]) and "ent:pytest" in node_ids(first["view"])
    assert len(client.calls) == 1

    api.initial()
    assert len(client.calls) == 1  # cached

    with_mems = api.view({**first["filters"], "show_memories": True})
    assert "mem:m1" in node_ids(with_mems)
    assert len(client.calls) == 1

    refreshed = api.refresh(first["filters"])
    assert len(client.calls) == 2
    assert set(refreshed) == {"filters", "projects", "generated_at", "view"}
    assert refreshed["filters"]["project"] == 3

    fallback = api.refresh(first["filters"])
    assert len(client.calls) == 3
    assert client.calls == ["/r/three"] * 3
    assert fallback["filters"]["project"] == "all"
    assert "ent:qdrant" in node_ids(fallback["view"])


def test_api_coerces_project_strings_from_the_page():
    from rem.graph.window import Api

    api = Api(FakeClient([snapshot()]), cwd=None)
    api.initial()
    base = {"scopes": ["global", "system", "project"], "show_memories": False}
    assert node_ids(api.view({**base, "project": "3"})) == node_ids(api.view({**base, "project": 3}))
    assert "ent:qdrant" not in node_ids(api.view({**base, "project": "3"}))
    assert "ent:qdrant" in node_ids(api.view({**base, "project": "all"}))


def test_api_rejects_bad_filters():
    from rem.graph.window import Api

    api = Api(FakeClient([snapshot()]), cwd=None)
    api.initial()
    base = {"project": "all", "scopes": ["global"], "show_memories": False}
    for bad in ({**base, "extra": 1}, {"project": "all"}, {**base, "scopes": ["bogus"]},
                {**base, "project": "x"}, {**base, "show_memories": "yes"}, {**base, "project": True}):
        with pytest.raises(ValueError):
            api.view(bad)


def test_api_exposes_only_its_three_methods():
    from rem.graph.window import Api

    api = Api(FakeClient([snapshot()]), cwd=None)
    public = sorted(n for n in dir(api) if not n.startswith("_"))
    assert public == ["initial", "refresh", "view"]


def test_window_module_imports_without_qt():
    for name in [m for m in sys.modules if m == "rem.graph.window" or m.split(".")[0] in ("webview", "qtpy", "PySide6")]:
        del sys.modules[name]
    importlib.import_module("rem.graph.window")
    assert "webview" not in sys.modules
    assert "qtpy" not in sys.modules
    assert "PySide6" not in sys.modules


def test_run_window_without_display_fails_cleanly(monkeypatch):
    from rem.graph import window

    for v in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(window, "import_webview", lambda: pytest.fail("Qt must not be touched"))
    with pytest.raises(RuntimeError, match="no display"):
        window.run_window(object(), None)


def test_run_window_with_display_reaches_webview_import(monkeypatch):
    from rem.graph import window

    class Sentinel(Exception):
        pass

    def stop():
        raise Sentinel

    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(window, "import_webview", stop)
    with pytest.raises(Sentinel):
        window.run_window(object(), None)
