"""The `rem graph` window: one self-contained HTML page in a pywebview (Qt) window, fed by a read-only `Api`.

Lazy imports only: importing this module never imports webview, qtpy or PySide6 (the optional `graph` extra).
The `Api` only reads (`RemClient.graph`, `jobs`, `job`) and builds views; nothing here writes to rem.
"""

import base64
import json
import os
import re
import socket
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from rem.graph.view import ALL_SCOPES, build_view, default_filters

TITLE = "rem — graph"
BACKGROUND = "#111111"  # the dark top bar's colour, so the window never flashes white while it loads
_FILTER_KEYS = {"project", "scopes", "show_memories"}
_JOB_FILTER_KEYS = {"project", "status", "flagged", "before"}
_JOB_STATUSES = ("all", "done", "failed", "pending")


def _static_text(*parts: str) -> str:
    node = files("rem.graph") / "static"
    for p in parts:
        node = node / p
    return node.read_text(encoding="utf-8")


def _script(source: str, name: str, *, module: bool = False) -> str:
    # An inlined script must not close its own <script> element early.
    if "</script" in source.lower():
        raise ValueError(f"{name} contains '</script' and cannot be inlined")
    attrs = ' type="module"' if module else ""
    return f"<script{attrs}>\n{source}\n</script>"


def _style(source: str, name: str) -> str:
    if "</style" in source.lower():
        raise ValueError(f"{name} contains '</style' and cannot be inlined")
    return f"<style>\n{source}\n</style>"


THREE_CORE_IMPORT = "'./three.core.js'"


def _data_url(source: str) -> str:
    return "data:text/javascript;base64," + base64.b64encode(source.encode("utf-8")).decode("ascii")


def three_module_source(module: str) -> str:
    """three.module.js with its two `./three.core.js` imports pointed at the import-map name `three/core`.
    Anything other than exactly two occurrences means a different three.js build: refuse it."""
    n = module.count(THREE_CORE_IMPORT)
    if n != 2:
        raise ValueError(f"three.module.js must import {THREE_CORE_IMPORT} exactly twice, found {n}")
    return module.replace(THREE_CORE_IMPORT, "'three/core'")


def importmap() -> str:
    """The import map that serves the vendored three.js from data: URLs, so the page never loads a file."""
    module = three_module_source(_static_text("vendor", "three", "three.module.js"))
    core = _static_text("vendor", "three", "three.core.js")
    imports = {"three": _data_url(module), "three/core": _data_url(core)}
    # base64 data: URLs can't contain '<', so the JSON can't close the <script> element
    return f'<script type="importmap">{json.dumps({"imports": imports})}</script>'


MARKERS = ("<!-- @importmap -->", "<!-- @style -->", "<!-- @scripts -->")


def render_html() -> str:
    """graph.html with its styles, the vendored cytoscape, the three.js import map, graph.js and sphere.js inlined:
    no network, no external files."""
    html = _static_text("graph.html")
    styles = "\n".join([_style(_static_text("graph.css"), "graph.css"), _style(_static_text("sphere.css"), "sphere.css")])
    scripts = "\n".join([
        _script(_static_text("vendor", "cytoscape.min.js"), "cytoscape.min.js"),
        _script(_static_text("graph.js"), "graph.js"),
        _script(_static_text("sphere.js"), "sphere.js", module=True),
    ])
    for marker in MARKERS:
        if html.count(marker) != 1:
            raise ValueError(f"graph.html must contain {marker} exactly once")
    # The scripts marker is split off first and filled last; the import map (base64 and JSON, which can't hold a
    # marker) goes in before the styles. So nothing inside the inlined code is ever treated as a marker.
    head, tail = html.split("<!-- @scripts -->")
    head = head.replace("<!-- @importmap -->", importmap())
    head = head.replace("<!-- @style -->", styles)
    return head + scripts + tail


def _project(value: Any) -> int | str:
    """`"all"`, or a project id as an int. A `<select>` hands ids over as strings."""
    if value == "all":
        return "all"
    if isinstance(value, bool):
        raise ValueError(f"invalid project: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise ValueError(f"invalid project: {value!r}")


def _filters(raw: Any) -> dict:
    if not isinstance(raw, dict) or set(raw) != _FILTER_KEYS:
        raise ValueError(f"filters must have exactly the keys {sorted(_FILTER_KEYS)}")
    scopes = raw["scopes"]
    if not isinstance(scopes, list) or not all(isinstance(s, str) and s in ALL_SCOPES for s in scopes):
        raise ValueError(f"invalid scopes: {scopes!r}")
    if not isinstance(raw["show_memories"], bool):
        raise ValueError(f"invalid show_memories: {raw['show_memories']!r}")
    return {"project": _project(raw["project"]), "scopes": list(scopes), "show_memories": raw["show_memories"]}


def _job_id(value: Any) -> int:
    """A job id as an int. The page may hand it over as a string."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise ValueError(f"invalid job id: {value!r}")


def _job_filters(raw: Any) -> dict:
    if not isinstance(raw, dict) or set(raw) != _JOB_FILTER_KEYS:
        raise ValueError(f"job filters must have exactly the keys {sorted(_JOB_FILTER_KEYS)}")
    if raw["status"] not in _JOB_STATUSES:
        raise ValueError(f"invalid status: {raw['status']!r}")
    if not isinstance(raw["flagged"], bool):
        raise ValueError(f"invalid flagged: {raw['flagged']!r}")
    before = raw["before"]
    if before is not None and (not isinstance(before, int) or isinstance(before, bool)):
        raise ValueError(f"invalid before: {before!r}")
    project = _project(raw["project"])
    return {"project_id": None if project == "all" else project,
            "status": None if raw["status"] == "all" else raw["status"],
            "flagged": raw["flagged"], "before": before}


class Api:
    """The js_api object: pywebview exposes its public attributes to the page, so only the read methods are public."""

    def __init__(self, client, cwd: Path | None):
        self._client = client
        self._cwd = cwd
        self._snap: dict | None = None

    def _fetch(self) -> dict:
        self._snap = self._client.graph(self._cwd)
        return self._snap

    def _payload(self, snap: dict, filters: dict) -> dict:
        return {
            "filters": filters,
            "projects": snap.get("projects", []),
            "generated_at": snap.get("generated_at"),
            "view": build_view(snap, **filters),
        }

    def initial(self) -> dict:
        snap = self._snap if self._snap is not None else self._fetch()
        return self._payload(snap, default_filters(snap))

    def view(self, filters: dict) -> dict:
        f = _filters(filters)
        snap = self._snap if self._snap is not None else self._fetch()
        return build_view(snap, **f)

    def refresh(self, filters: dict) -> dict:
        f = _filters(filters)
        snap = self._fetch()
        if f["project"] != "all" and f["project"] not in {p["id"] for p in snap.get("projects", [])}:
            f["project"] = "all"
        return self._payload(snap, f)

    def sphere(self, filters: dict) -> dict:
        """The Sphere tab's data: the full view (all projects and scopes, memories on), which decides positions and
        sizes, and the filtered view (memories always on), which decides what shows. Both come from one snapshot."""
        f = _filters(filters)
        snap = self._snap if self._snap is not None else self._fetch()
        return {
            "generated_at": snap.get("generated_at"),
            "full": build_view(snap, project="all", scopes=ALL_SCOPES, show_memories=True),
            "filtered": build_view(snap, **{**f, "show_memories": True}),
        }

    def jobs(self, filters: dict) -> dict:
        """One page of extraction runs (the `/jobs` body)."""
        return self._client.jobs(**_job_filters(filters))

    def job(self, job_id) -> dict:
        """One extraction run in full (the `/jobs/{id}` body)."""
        return self._client.job(_job_id(job_id))


def import_webview():
    """Import pywebview and its Qt binding; ImportError when the `graph` extra isn't installed."""
    import webview
    import qtpy  # noqa: F401  (the Qt backend; missing without the extra)

    return webview


X11_SOCKET_DIR = Path("/tmp/.X11-unix")
_PROBE_SECONDS = 1.0


def _connects(family: int, address) -> bool:
    s = socket.socket(family, socket.SOCK_STREAM)
    s.settimeout(_PROBE_SECONDS)
    try:
        s.connect(address)
        return True
    except OSError:
        return False
    finally:
        s.close()


def _x11_reachable(display: str) -> bool:
    """Whether an X server answers at `display` ("[host]:N[.S]"). An unknown form returns True: Qt decides."""
    m = re.fullmatch(r"([^:]*):(\d+)(?:\.\d+)?", display)
    if not m:
        return True
    host, n = m.group(1), int(m.group(2))
    if host in ("", "unix"):
        path = str(X11_SOCKET_DIR / f"X{n}")
        return _connects(socket.AF_UNIX, path) or _connects(socket.AF_UNIX, "\0" + path)  # abstract socket
    return _connects(socket.AF_INET, (host, 6000 + n))


def _wayland_reachable(name: str) -> bool:
    path = Path(name) if name.startswith("/") else Path(os.environ.get("XDG_RUNTIME_DIR", "")) / name
    return _connects(socket.AF_UNIX, str(path))


def _display_problem() -> str | None:
    """Why Qt would find no usable display, or None. Qt aborts the whole process (no exception) in that case.
    QT_QPA_PLATFORM set means the user chose a platform on purpose, so nothing is checked."""
    if os.environ.get("QT_QPA_PLATFORM"):
        return None
    wayland, x = os.environ.get("WAYLAND_DISPLAY"), os.environ.get("DISPLAY")
    if wayland and _wayland_reachable(wayland):
        return None
    if x:
        return None if _x11_reachable(x) else f"cannot reach the X display {x}"
    if wayland:
        return f"cannot reach the Wayland display {wayland}"
    return "no display (DISPLAY/WAYLAND_DISPLAY not set)"


def run_window(client, cwd: Path | None) -> None:
    """Open the graph window and block until it is closed."""
    if (problem := _display_problem()) is not None:
        raise RuntimeError(problem)
    webview = import_webview()
    html = render_html()
    # Qt's setHtml (pywebview's `html=`) can't show a page over 2 MB, and the embedded three.js makes this one
    # bigger. So the same self-contained page is written to a private temporary directory and loaded as a
    # file:// URL. pywebview starts no HTTP server for file:// URLs, and nothing is fetched from anywhere.
    with tempfile.TemporaryDirectory(prefix="rem-graph-") as tmp:
        page = Path(tmp) / "graph.html"
        page.write_text(html, encoding="utf-8")
        webview.create_window(TITLE, url=page.as_uri(), js_api=Api(client, cwd),
                              width=1200, height=800, background_color=BACKGROUND)
        webview.start(gui="qt")
