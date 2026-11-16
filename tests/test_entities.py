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
