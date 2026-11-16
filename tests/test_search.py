import pytest

from rem.entities import ENTITY_BOOST, entity_id, normalize_name
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


def link_entity(mem, memory_ids, name="thing", *, aliases=(), vector_of=None):
    """Test-only: an entity named `name` linked to memory_ids. Its vector is vector_of's embedding
    (default: the name's), which must not matter: the query matches entities by their words only."""
    now = utcnow()
    norm = normalize_name(name)
    rec = EntityRecord(
        id=entity_id(Scope.GLOBAL, None, norm), name=name, normalized=norm, type="tool", aliases=list(aliases),
        scope=Scope.GLOBAL, project_id=None, memory_ids=list(memory_ids), created_at=now, updated_at=now,
    )
    vec = mem.embedder.embed_query(vector_of if vector_of is not None else norm)
    mem.vectors.upsert(mem.entity_col, [VectorPoint(rec.id, vec, rec.to_payload())])


def scores(mem, query):
    return {r.record.id: r.score for r in mem.search(query)}


def test_linked_memory_is_boosted_by_exactly_entity_boost(mem):
    a = add(mem, "the build uses tabs for indentation")
    b = add(mem, "the build uses spaces for indentation")
    before = scores(mem, "build indentation")
    assert a.id in before and b.id in before
    link_entity(mem, [a.id], "indentation")
    after = scores(mem, "build indentation")
    assert after[a.id] == pytest.approx(before[a.id] + ENTITY_BOOST)
    assert after[b.id] == pytest.approx(before[b.id])


def test_entity_not_named_in_the_query_gives_no_boost_even_with_the_query_vector(mem):
    a = add(mem, "the build uses tabs for indentation")
    before = scores(mem, "build indentation")
    link_entity(mem, [a.id], "thing", vector_of="build indentation")  # cosine 1.0 with the query
    assert scores(mem, "build indentation") == pytest.approx(before)


def test_entity_name_matches_whole_words_only(mem):
    a = add(mem, "pins the uv version and the uvicorn version")
    named, other = scores(mem, "which version of uv"), scores(mem, "the uvicorn version")
    assert a.id in named and a.id in other
    link_entity(mem, [a.id], "uv")
    assert scores(mem, "which version of uv")[a.id] == pytest.approx(named[a.id] + ENTITY_BOOST)
    assert scores(mem, "the uvicorn version") == pytest.approx(other)  # "uv" is not a word of "uvicorn"


def test_entity_name_with_punctuation_matches(mem):
    a = add(mem, "I use node.js daily for scripts")
    before = scores(mem, "I use node.js daily")
    link_entity(mem, [a.id], "Node.js")
    assert scores(mem, "I use node.js daily")[a.id] == pytest.approx(before[a.id] + ENTITY_BOOST)


def test_entity_alias_in_the_query_boosts(mem):
    a = add(mem, "deploys scripts with nodejs runtime")
    before = scores(mem, "nodejs runtime")
    link_entity(mem, [a.id], "Node.js", aliases=["nodejs"])
    assert scores(mem, "nodejs runtime")[a.id] == pytest.approx(before[a.id] + ENTITY_BOOST)


def test_entity_match_never_adds_a_candidate(mem):
    unrelated = add(mem, "zebra")
    assert mem.search("build indentation") == []
    link_entity(mem, [unrelated.id], "indentation")
    assert mem.search("build indentation") == []


def test_search_works_when_the_entity_collection_is_gone(mem):
    add(mem, "the build uses tabs for indentation")
    mem.vectors.delete_col(mem.entity_col)
    assert mem.search("build indentation")
