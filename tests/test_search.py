import pytest

from rem.entities import ENTITY_BOOST, entity_id
from rem.models import EntityRecord, Scope, utcnow
from rem.storage.vector.base import VectorPoint
from tests.conftest import add


def test_current_project_ranks_first(mem, tmp_path):
    for p in ("a", "b"):
        (tmp_path / p / ".git").mkdir(parents=True)
    add(mem, "database is SQLite", scope=Scope.PROJECT, cwd=tmp_path / "a")
    add(mem, "database is Postgres", scope=Scope.PROJECT, cwd=tmp_path / "b")
    out = mem.search("which database", cwd=tmp_path / "a")
    assert out and out[0].record.project_name == "a"


def test_search_outside_known_project_does_not_create_one(mem, tmp_path):
    add(mem, "database is SQLite", scope=Scope.SYSTEM)
    mem.search("database", cwd=tmp_path / "unknown")
    assert mem.db.find_project(str((tmp_path / "unknown").resolve())) is None


def test_scope_filter_and_labels(mem):
    add(mem, "Prefers uv over pip")
    add(mem, "OS is Fedora 44", scope=Scope.SYSTEM)
    out = mem.search("Fedora OS", scope=Scope.SYSTEM)
    assert [r.record.scope for r in out] == ["system"] and out[0].record.provenance == "user_confirmed"


def test_empty_query_rejected(mem):
    with pytest.raises(ValueError, match="text must not be empty"):
        mem.search("  ")


def test_keyword_search_finds_exact_term_in_hybrid_mode(mem):
    add(mem, "the build uses tabs")
    add(mem, "the build uses spaces")
    out = mem.search("tabs")
    assert out and out[0].record.text == "the build uses tabs"


def link_entity(mem, query, memory_ids, name="thing"):
    """Test-only: an entity whose vector is the query's own, so it matches the query exactly."""
    now = utcnow()
    rec = EntityRecord(
        id=entity_id(Scope.GLOBAL, None, name), name=name, normalized=name, type="tool", aliases=[],
        scope=Scope.GLOBAL, project_id=None, memory_ids=list(memory_ids), created_at=now, updated_at=now,
    )
    mem.vectors.upsert(mem.entity_col, [VectorPoint(rec.id, mem.embedder.embed_query(query), rec.to_payload())])


def test_linked_memory_is_boosted_by_exactly_entity_boost(mem):
    a = add(mem, "the build uses tabs for indentation")
    b = add(mem, "the build uses spaces for indentation")
    before = {r.record.id: r.score for r in mem.search("build indentation")}
    assert a.id in before and b.id in before
    link_entity(mem, "build indentation", [a.id])
    after = {r.record.id: r.score for r in mem.search("build indentation")}
    assert after[a.id] == pytest.approx(before[a.id] + ENTITY_BOOST)
    assert after[b.id] == pytest.approx(before[b.id])


def test_entity_match_never_adds_a_candidate(mem):
    unrelated = add(mem, "zebra")
    assert mem.search("build indentation") == []
    link_entity(mem, "build indentation", [unrelated.id])
    assert mem.search("build indentation") == []


def test_search_works_when_the_entity_collection_is_gone(mem):
    add(mem, "the build uses tabs for indentation")
    mem.vectors.delete_col(mem.entity_col)
    assert mem.search("build indentation")
