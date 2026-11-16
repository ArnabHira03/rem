import uuid

import pytest

from rem.models import (
    MemoryRecord,
    Provenance,
    Scope,
    Status,
    memory_id,
    normalize_text,
    text_hash,
    utcnow,
)


def test_normalize_and_hash_ignore_case_and_spacing():
    assert normalize_text("  Prefers   UV\n") == "prefers uv"
    assert text_hash("Prefers uv") == text_hash(" prefers  UV ")


def test_empty_text_rejected():
    with pytest.raises(ValueError, match="text must not be empty"):
        normalize_text("   ")


def test_memory_id_deterministic_and_scoped():
    a = memory_id("Prefers uv", Scope.GLOBAL, None)
    assert a == memory_id("prefers  UV", Scope.GLOBAL, None)
    assert a != memory_id("Prefers uv", Scope.PROJECT, 1)
    assert a != memory_id("Prefers uv", Scope.GLOBAL, None, job_id="job-1")
    uuid.UUID(a)


def test_payload_roundtrip_keeps_all_fields():
    rec = MemoryRecord(
        id="x", text="t", hash="h", scope=Scope.GLOBAL, project_id=None, project_name=None,
        provenance=Provenance.USER_CONFIRMED, status=Status.VERIFIED, evidence_source=None,
        observed_at=None, source="cli", session_id=None, job_id=None,
        created_at=utcnow(), updated_at=utcnow(),
    )
    p = rec.to_payload()
    assert set(p) == {
        "id", "text", "hash", "scope", "project_id", "project_name", "provenance", "status",
        "evidence_source", "observed_at", "source", "session_id", "job_id", "entity_ids",
        "corroborated_by", "confirm_requested_at",
        "created_at", "updated_at",
    }
    assert p["entity_ids"] == [] and p["scope"] == "global"
    assert MemoryRecord.from_payload(p) == rec


def test_payload_without_corroborated_by_loads():
    rec = MemoryRecord(
        id="x", text="t", hash="h", scope=Scope.GLOBAL, project_id=None, project_name=None,
        provenance=Provenance.USER_CONFIRMED, status=Status.VERIFIED, evidence_source=None,
        observed_at=None, source="cli", session_id=None, job_id=None,
        created_at=utcnow(), updated_at=utcnow(),
    )
    p = rec.to_payload()
    del p["corroborated_by"]
    assert MemoryRecord.from_payload(p).corroborated_by is None


def test_candidate_lives_in_models():
    from rem.extract import Candidate as C2
    from rem.models import Candidate

    assert Candidate is C2


def test_memory_does_not_import_extract():
    import subprocess
    import sys

    import rem.memory  # noqa: F401
    import rem.trust  # noqa: F401

    out = subprocess.run(
        [sys.executable, "-c", "import rem.memory, rem.trust, sys; print('rem.extract' in sys.modules)"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "False"
