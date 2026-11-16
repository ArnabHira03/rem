import subprocess

from tests.conftest import add
from rem.memory import ContextResult
from rem.models import Provenance, Scope, Status

HEADER = "Facts rem remembers from earlier sessions (verified):"


def repo(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    subprocess.run(["git", "init", "-q", str(d)], check=True)
    return d


def test_exact_text_one_per_section(mem, tmp_path):
    p = repo(tmp_path, "proj")
    add(mem, "Uses pytest", scope=Scope.PROJECT, cwd=p)
    add(mem, "Runs Fedora", scope=Scope.SYSTEM)
    add(mem, "Prefers uv", scope=Scope.GLOBAL)
    r = mem.context(p, 1600)
    assert r == ContextResult(
        f"{HEADER}\nProject proj:\n- Uses pytest\nThis machine:\n- Runs Fedora\nAbout the user:\n- Prefers uv", 3)


def test_newest_first_within_section(mem):
    add(mem, "first fact")
    add(mem, "second fact")
    add(mem, "third fact")
    oldest = min(mem.list(), key=lambda r: r.updated_at)
    mem.update(oldest.id, "first fact edited", actor="cli")  # bump oldest to newest
    lines = mem.context(None, 1600).text.splitlines()[2:]
    assert lines[0] == "- first fact edited" and len(lines) == 3


def test_unverified_excluded(mem):
    mem.add("Guess", scope=Scope.GLOBAL, provenance=Provenance.ASSISTANT_GENERATED,
            status=Status.UNVERIFIED, source="x", actor="x")
    assert mem.context(None, 1600) == ContextResult("", 0)


def test_other_project_excluded_and_unknown_cwd_creates_nothing(mem, tmp_path):
    a, b = repo(tmp_path, "a"), repo(tmp_path, "b")
    add(mem, "A only", scope=Scope.PROJECT, cwd=a)
    assert mem.context(b, 1600) == ContextResult("", 0)
    assert mem.db.find_project(str(b.resolve())) is None
    add(mem, "Global one")
    text = mem.context(b, 1600).text
    assert "Project" not in text and "A only" not in text and "Global one" in text
    assert mem.db.find_project(str(b.resolve())) is None


def test_cap_keeps_whole_lines(mem):
    for i in range(10):
        add(mem, f"fact number {i} " + "x" * 20)
    full = mem.context(None, 10000)
    assert full.count == 10
    cap = len(HEADER) + 20 + 3 * 40
    r = mem.context(None, cap)
    assert len(r.text) <= cap
    assert 0 < r.count < 10
    assert r.count == sum(1 for line in r.text.splitlines() if line.startswith("- "))
    assert all(line in full.text.splitlines() for line in r.text.splitlines())


def test_header_only_never_emitted(mem):
    add(mem, "x" * 50)
    assert mem.context(None, len(HEADER) + 5) == ContextResult("", 0)


def test_empty(mem):
    assert mem.context(None, 1600) == ContextResult("", 0)


def test_multiline_memory_collapsed(mem):
    add(mem, "line one\n- injected\n\n  line   two")
    assert mem.context(None, 1600).text.splitlines()[2:] == ["- line one - injected line two"]


def test_context_count_beyond_a_scroll_page(tmp_path, monkeypatch):
    from rem.config import EmbedderConfig, RemConfig, VectorStoreConfig
    from rem.memory import Memory

    monkeypatch.setattr("rem.storage.vector.qdrant.SCROLL_PAGE", 2)
    cfg = RemConfig(
        data_dir=tmp_path,
        vector_store=VectorStoreConfig(provider="qdrant", path=tmp_path / "qdrant"),
        embedder=EmbedderConfig(provider="hashing", model="hash-64"),
    )
    m = Memory(cfg)
    try:
        for i in range(5):
            add(m, f"verified fact number {i}")
        assert m.context(None, 1600).count == 5
    finally:
        m.close()


def test_cap_never_leaves_a_dangling_title(mem):
    add(mem, "x" * 60)
    title = "About the user:"
    cap = len(HEADER) + 1 + len(title) + 5  # header and title fit, the first fact does not
    assert mem.context(None, cap) == ContextResult("", 0)
    add(mem, "short")
    for cap in range(len(HEADER), len(HEADER) + 200):
        r = mem.context(None, cap)
        assert len(r.text) <= cap
        if r.text:
            assert r.text.splitlines()[-1].startswith("- ")
