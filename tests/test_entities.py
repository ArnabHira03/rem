import pytest

from rem.entities import bucket, entity_id, normalize_name
from rem.models import Scope


def test_normalize_collapses_case_and_space():
    assert normalize_name(" Python\n") == normalize_name("PYTHON") == "python"


def test_entity_id_is_deterministic_and_bucket_scoped():
    a = entity_id(Scope.PROJECT, 1, "python")
    assert a == entity_id(Scope.PROJECT, 1, "python")
    assert a != entity_id(Scope.PROJECT, 2, "python")
    assert entity_id(Scope.GLOBAL, None, "python") != a


def test_bucket_drops_project_id_outside_project_scope():
    assert bucket(Scope.GLOBAL, 7) == (Scope.GLOBAL, None)


from rem.entities import mentions_entity


@pytest.mark.parametrize("query,names,expected", [
    ("which version of uv", ["uv"], True),
    ("uvicorn settings", ["uv"], False),
    ("what package manager do i use", ["uv"], False),
    ("i use node.js daily", ["node.js"], True),
    ("node.jsx files", ["node.js"], False),
    ("is c++ installed?", ["c++"], True),
    ("abc++ thing", ["c++"], False),
    ("run qwen3.5:4b locally", ["qwen3.5:4b"], True),
    ("run qwen3.5:4bx", ["qwen3.5:4b"], False),
    ("uv.", ["uv"], True),
    ("my_uv var", ["uv"], True),  # "_" is not alphanumeric, so it is a boundary
    ("use node js here", ["node.js", "node js"], True),  # an alias matches
    ("über café tips", ["café"], True),
    ("cafés", ["café"], False),  # unicode letters are word characters
    ("anything", [""], False),
    ("anything", [], False),
])
def test_mentions_entity_matches_whole_token_sequences(query, names, expected):
    assert mentions_entity(normalize_name(query), names) is expected
