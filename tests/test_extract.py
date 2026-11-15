import pytest

from rem.errors import LLMError
from rem.extract import EXTRACTION_SCHEMA, build_prompt, extract
from rem.llm.fake import FakeLLM
from rem.models import MemoryRecord, Provenance, Scope, Status, utcnow
from rem.storage.queue import Turn

T = [Turn("user", "x")]


def rec(text, scope=Scope.GLOBAL):
    return MemoryRecord(
        id="x", text=text, hash="h", scope=scope, project_id=None, project_name=None,
        provenance=Provenance.USER_ASSERTED, status=Status.VERIFIED, evidence_source=None,
        observed_at=None, source="t", session_id=None, job_id=None,
        created_at=utcnow(), updated_at=utcnow(),
    )


def test_prompt_numbers_related_and_orders_sections():
    s, u = build_prompt([Turn("user", "I use fish shell")], [Turn("assistant", "hi")], [rec("Prefers uv over pip")])
    assert "[0] (global, verified) Prefers uv over pip" in u
    assert u.index("## Existing memories") < u.index("## Earlier conversation") < u.index("## New messages")
    assert "user: I use fish shell" in u


def test_prompt_is_redacted_everywhere():
    secret = "sk-ant-" + "a" * 30
    _, u = build_prompt([Turn("user", "ok")], [Turn("user", f"key {secret}")], [rec(f"token {secret}")])
    assert secret not in u and "[REDACTED:anthropic_key]" in u


def test_schema_is_strict():
    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "object":
                assert n["additionalProperties"] is False
                assert set(n["required"]) == set(n["properties"])
            for v in n.values():
                walk(v)
    walk(EXTRACTION_SCHEMA)


def test_extract_parses_and_caps():
    facts = [{"text": f"fact {i}", "scope": "global", "origin": "user"} for i in range(25)]
    ex = extract(FakeLLM([{"facts": facts}]), [Turn("user", "x")], [], [], max_candidates=20)
    assert len(ex.candidates) == 20 and ex.model == "fake"


def test_extract_drops_invalid_items_but_keeps_valid():
    out = {"facts": [{"text": "  ", "scope": "global", "origin": "user"},
                     {"text": "Uses Fedora", "scope": "planet", "origin": "user"},
                     {"text": "Uses Fedora 44", "scope": "system", "origin": "user"}]}
    assert [c.text for c in extract(FakeLLM([out]), T, [], [], 20).candidates] == ["Uses Fedora 44"]


def test_extract_collapses_duplicate_texts():
    out = {"facts": [{"text": "Uses Fedora", "scope": "system", "origin": "user"},
                     {"text": " uses  fedora ", "scope": "system", "origin": "user"}]}
    assert len(extract(FakeLLM([out]), T, [], [], 20).candidates) == 1


def test_extract_missing_facts_key_raises():
    with pytest.raises(LLMError, match="facts"):
        extract(FakeLLM([{"nope": 1}]), T, [], [], 20)


def test_extract_propagates_llm_error():
    with pytest.raises(LLMError):
        extract(FakeLLM([LLMError("down")]), T, [], [], 20)


def test_prompt_marks_status_and_explains_unverified():
    unverified = rec("Uses Poetry").model_copy(update={"status": Status.UNVERIFIED})
    s, u = build_prompt(T, [], [unverified])
    assert "[0] (global, unverified) Uses Poetry" in u
    assert "unverified" in s and "verified" in s
