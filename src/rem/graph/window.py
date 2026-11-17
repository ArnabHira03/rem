"""The `rem graph` window: one self-contained HTML page in a pywebview (Qt) window, fed by a read-only `Api`.

Lazy imports only: importing this module never imports webview, qtpy or PySide6 (the optional `graph` extra).
The `Api` only reads (`RemClient.graph`) and builds views; nothing here writes to rem.
"""

import os
import re
import socket
from importlib.resources import files
from pathlib import Path
from typing import Any

from rem.graph.view import ALL_SCOPES, build_view, default_filters

TITLE = "rem — graph"
_FILTER_KEYS = {"project", "scopes", "show_memories"}


def _static_text(*parts: str) -> str:
    node = files("rem.graph") / "static"
    for p in parts:
        node = node / p
    return node.read_text(encoding="utf-8")


def _script(source: str, name: str) -> str:
    # An inlined script must not close its own <script> element early.
    if "</script" in source.lower():
        raise ValueError(f"{name} contains '</script' and cannot be inlined")
    return f"<script>\n{source}\n</script>"


def render_html() -> str:
    """graph.html with graph.css, the vendored cytoscape and graph.js inlined: no network, no external files."""
    html = _static_text("graph.html")
    css = _static_text("graph.css")
    if "</style" in css.lower():
        raise ValueError("graph.css contains '</style' and cannot be inlined")
    scripts = "\n".join([
        _script(_static_text("vendor", "cytoscape.min.js"), "cytoscape.min.js"),
        _script(_static_text("graph.js"), "graph.js"),
    ])
    for marker in ("<!-- @style -->", "<!-- @scripts -->"):
        if html.count(marker) != 1:
            raise ValueError(f"graph.html must contain {marker} exactly once")
    # Replace the scripts marker last, so nothing inside the inlined code is ever treated as a marker.
    head, tail = html.split("<!-- @scripts -->")
    head = head.replace("<!-- @style -->", f"<style>\n{css}\n</style>")
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


class Api:
    """The js_api object: pywebview exposes its public attributes to the page, so only the three methods are public."""

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
    webview.create_window(TITLE, html=render_html(), js_api=Api(client, cwd),
                          width=1200, height=800, background_color="#FFFFFF")
    webview.start(gui="qt")
