import pytest

from rem.models import MemoryRecord, Provenance, Scope, Status, text_hash, utcnow
from rem.relations import RELATION_TYPES, derive_trust, normalize_type, relation_id, relation_sentence


def rec(prov, status, text="x"):
    now = utcnow()
    return MemoryRecord(
        id=text, text=text, hash=text_hash(text), scope=Scope.GLOBAL, project_id=None, project_name=None,
        provenance=prov, status=status, evidence_source=None, observed_at=None, source="t",
        session_id=None, job_id=None, created_at=now, updated_at=now,
    )


def test_types_include_fallback():
    assert "related_to" in RELATION_TYPES and len(RELATION_TYPES) == 7


def test_normalize_type():
    assert normalize_type(" Depends-On ") == "depends_on"
    assert normalize_type("runs on") == "runs_on"
    assert normalize_type("loves") == "related_to"


def test_relation_id_is_deterministic_and_directed():
    assert relation_id("a", "uses", "b") == relation_id("a", "uses", "b")
    assert relation_id("a", "uses", "b") != relation_id("b", "uses", "a")
    assert relation_id("a", "uses", "b") != relation_id("a", "depends_on", "b")


def test_relation_sentence():
    assert relation_sentence("uv", "preferred_over", "pip") == "uv preferred over pip"


def test_derive_trust_unverified_assistant_only():
    r = rec(Provenance.ASSISTANT_GENERATED, Status.UNVERIFIED)
    assert derive_trust([r]) == (Provenance.ASSISTANT_GENERATED, Status.UNVERIFIED)


def test_derive_trust_any_verified_wins():
    a = rec(Provenance.ASSISTANT_GENERATED, Status.UNVERIFIED, "a")
    u = rec(Provenance.USER_ASSERTED, Status.VERIFIED, "u")
    assert derive_trust([a, u]) == (Provenance.USER_ASSERTED, Status.VERIFIED)


def test_derive_trust_strongest_provenance():
    u = rec(Provenance.USER_ASSERTED, Status.VERIFIED, "u")
    c = rec(Provenance.USER_CONFIRMED, Status.VERIFIED, "c")
    assert derive_trust([u, c]) == (Provenance.USER_CONFIRMED, Status.VERIFIED)


def test_derive_trust_empty_raises():
    with pytest.raises(ValueError):
        derive_trust([])
