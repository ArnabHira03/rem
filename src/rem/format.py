"""Human-readable one-line formats for memories, search hits and history rows (CLI and MCP)."""


def memory_line(m: dict) -> str:
    where = f"{m['scope']}:{m['project_name']}" if m.get("project_name") else m["scope"]
    return f"{m['id']}  [{where}] {m['status']}  {m['text']}"


def search_line(m: dict) -> str:
    return f"{m['score']:.2f}  " + memory_line(m)


def history_line(h: dict) -> str:
    return f"{h['created_at']}  {h['event']:<11} {h['actor']:<9} {h['old_value']!r} → {h['new_value']!r}"
