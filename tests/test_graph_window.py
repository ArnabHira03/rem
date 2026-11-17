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


def test_page_overrides_browser_blue_selection_focus_and_accent():
    css = (STATIC / "graph.css").read_text(encoding="utf-8")
    flat = re.sub(r"\s+", "", css)
    sel = re.search(r"::selection\{([^}]*)\}", flat)
    assert sel and "background:#000;" in sel.group(1) + ";" and "color:#fff;" in sel.group(1) + ";"
    focus = re.search(r":focus-visible\{([^}]*)\}", flat)
    assert focus and "outline:2pxsolid#000" in focus.group(1)
    for sel in ("input", "select"):
        rule = re.search(r"(?:^|[},])" + sel + r"(?:,[^{]*)?\{([^}]*)\}", flat)
        assert rule and "accent-color:#000" in rule.group(1), sel


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


def test_api_exposes_only_read_methods():
    from rem.graph.window import Api

    api = Api(FakeClient([snapshot()]), cwd=None)
    public = {n for n in dir(api) if not n.startswith("_")}
    assert public == {"initial", "view", "refresh", "jobs", "job"}


class FakeJobsClient:
    def __init__(self):
        self.calls = []

    def jobs(self, **kwargs):
        self.calls.append(("jobs", kwargs))
        return {"runs": [], "next_before": None}

    def job(self, id):
        self.calls.append(("job", id))
        return {"run": {"id": id}}


def test_api_jobs_and_job():
    from rem.graph.window import Api

    client = FakeJobsClient()
    api = Api(client, cwd=None)
    assert api.jobs({"project": "3", "status": "pending", "flagged": True, "before": None}) == \
        {"runs": [], "next_before": None}
    assert client.calls[-1] == ("jobs", {"project_id": 3, "status": "pending", "flagged": True, "before": None})
    api.jobs({"project": "all", "status": "all", "flagged": False, "before": 40})
    assert client.calls[-1] == ("jobs", {"project_id": None, "status": None, "flagged": False, "before": 40})
    assert api.job("12") == {"run": {"id": 12}}
    assert client.calls[-1] == ("job", 12)
    api.job(5)
    assert client.calls[-1] == ("job", 5)

    n = len(client.calls)
    base = {"project": "all", "status": "all", "flagged": False, "before": None}
    for bad in ({**base, "extra": 1}, {"project": "all"}, {**base, "status": "processing"}, {**base, "status": None},
                {**base, "flagged": "yes"}, {**base, "before": "7"}, {**base, "before": True},
                {**base, "project": "x"}, "nope"):
        with pytest.raises(ValueError):
            api.jobs(bad)
    for bad in ("x", "", None, True, 1.5, "-3"):
        with pytest.raises(ValueError):
            api.job(bad)
    assert len(client.calls) == n  # nothing invalid reaches the client


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
    monkeypatch.setattr(window, "_x11_reachable", lambda display: True)
    monkeypatch.setattr(window, "import_webview", stop)
    with pytest.raises(Sentinel):
        window.run_window(object(), None)


@pytest.fixture
def listening_unix(short_dir):
    """Bind and listen on a Unix socket at short_dir/<name>; yields the factory, closes everything after."""
    import socket

    socks = []

    def make(name):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(short_dir / name))
        s.listen(16)  # probes connect without being accepted: leave room in the backlog
        socks.append(s)
        return short_dir / name

    yield make
    for s in socks:
        s.close()


def _no_display_env(monkeypatch):
    for v in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"):
        monkeypatch.delenv(v, raising=False)


def test_dead_x_display_is_refused_before_qt(monkeypatch, short_dir):
    from rem.graph import window

    _no_display_env(monkeypatch)
    monkeypatch.setattr(window, "X11_SOCKET_DIR", short_dir)  # nothing listens at short_dir/X7
    monkeypatch.setenv("DISPLAY", ":7")
    monkeypatch.setattr(window, "import_webview", lambda: pytest.fail("Qt must not be touched"))
    with pytest.raises(RuntimeError, match=r"cannot reach the X display :7"):
        window.run_window(object(), None)


def test_x_display_reachable_over_unix_socket(monkeypatch, short_dir, listening_unix):
    from rem.graph import window

    monkeypatch.setattr(window, "X11_SOCKET_DIR", short_dir)
    listening_unix("X5")
    assert window._x11_reachable(":5")
    assert window._x11_reachable(":5.0")
    assert window._x11_reachable("unix:5")
    assert not window._x11_reachable(":6")


def test_x_display_reachable_over_tcp(monkeypatch):
    import socket

    from rem.graph import window

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        n = srv.getsockname()[1] - 6000
        assert window._x11_reachable(f"127.0.0.1:{n}")
    finally:
        srv.close()
    assert not window._x11_reachable(f"127.0.0.1:{n}")


def test_unparseable_display_is_let_through():
    from rem.graph import window

    assert window._x11_reachable("not-a-display")  # unknown form: let Qt decide


def test_wayland_display_checked_by_socket(monkeypatch, short_dir, listening_unix):
    from rem.graph import window

    _no_display_env(monkeypatch)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(short_dir))
    listening_unix("wl-test")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wl-test")
    assert window._display_problem() is None
    monkeypatch.setenv("WAYLAND_DISPLAY", "wl-gone")
    assert window._display_problem() == "cannot reach the Wayland display wl-gone"
    monkeypatch.setenv("WAYLAND_DISPLAY", str(short_dir / "wl-test"))  # absolute path form
    assert window._display_problem() is None


def test_dead_wayland_with_live_x_is_let_through(monkeypatch, short_dir, listening_unix):
    from rem.graph import window

    _no_display_env(monkeypatch)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(short_dir))
    monkeypatch.setenv("WAYLAND_DISPLAY", "wl-gone")
    monkeypatch.setattr(window, "X11_SOCKET_DIR", short_dir)
    listening_unix("X3")
    monkeypatch.setenv("DISPLAY", ":3")
    assert window._display_problem() is None  # Qt falls back to X (XWayland)


def test_qt_platform_override_skips_the_check(monkeypatch):
    from rem.graph import window

    _no_display_env(monkeypatch)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("DISPLAY", ":99")
    assert window._display_problem() is None
