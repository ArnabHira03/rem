import pytest

from rem.models import Scope
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
