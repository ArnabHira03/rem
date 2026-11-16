import pytest

from rem.errors import LLMError
from rem.extract import EXTRACTION_SCHEMA, build_prompt, extract
from rem.llm.fake import FakeLLM
from rem.models import MemoryRecord, Provenance, Scope, Status, utcnow
from rem.storage.queue import Turn

T = [Turn("user", "x")]


def rec(text, scope=Scope.GLOBAL, id="x"):
    return MemoryRecord(
        id=id, text=text, hash="h", scope=scope, project_id=None, project_name=None,
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


def test_clip_turn_keeps_head_and_tail():
    from rem.extract import MAX_TURN_CHARS, clip_turn

    assert MAX_TURN_CHARS == 4000
    exact = "x" * 4000
    assert clip_turn(exact) is exact
    long = "A" * 2000 + "M" * 1234 + "Z" * 2000
    assert clip_turn(long) == "A" * 2000 + "\n[… 1234 chars omitted …]\n" + "Z" * 2000
    assert clip_turn("short") == "short"


def test_prompt_clips_new_and_context_turns_but_not_memories():
    big = "B" * 2000 + "middle" + "E" * 2000
    _, u = build_prompt([Turn("user", big)], [Turn("assistant", big)], [rec("M" * 5000)])
    assert u.count("[… 6 chars omitted …]") == 2 and "middle" not in u
    assert "M" * 5000 in u


def test_prompt_redacts_before_clipping():
    secret = "sk-ant-" + "a" * 30
    text = "x" * 1990 + f" {secret} " + "y" * 3000  # the secret straddles the 2000-char cut
    _, u = build_prompt([Turn("user", text)], [], [])
    assert "sk-ant-" not in u  # clipping first would leave a partial key the pattern no longer matches
    assert "[REDACTED" in u and "chars omitted" in u


def test_schema_is_strict():
    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "object":
                assert n["additionalProperties"] is False
                assert set(n["required"]) == set(n["properties"])
            for v in n.values():
                walk(v)
    walk(EXTRACTION_SCHEMA)
    item = EXTRACTION_SCHEMA["properties"]["facts"]["items"]
    assert item["properties"]["restates"] == {"type": ["integer", "null"]} and "restates" in item["required"]


def test_extract_parses_and_caps():
    facts = [{"text": f"fact {i}", "scope": "global", "origin": "user", "restates": None} for i in range(25)]
    ex = extract(FakeLLM([{"facts": facts}]), [Turn("user", "x")], [], [], max_candidates=20)
    assert len(ex.candidates) == 20 and ex.model == "fake"


def test_extract_drops_invalid_items_but_keeps_valid(caplog):
    out = {"facts": [{"text": "  ", "scope": "global", "origin": "user", "restates": None},
                     {"text": "Uses Fedora secretplanet", "scope": "planet", "origin": "user", "restates": None},
                     {"text": "Uses Fedora noscope", "origin": "user", "restates": None},
                     "Uses Fedora as a bare string",
                     {"text": "Uses Fedora 44", "scope": "system", "origin": "user", "restates": None}]}
    assert [c.text for c in extract(FakeLLM([out]), T, [], [], 20).candidates] == ["Uses Fedora 44"]
    assert caplog.text.count("dropping") == 4
    assert "secretplanet" not in caplog.text and "noscope" not in caplog.text and "bare string" not in caplog.text
    assert "24 chars, sha256" in caplog.text


def test_extract_missing_restates_key_means_null(caplog):
    related = [rec("Uses Poetry", id="m-0")]
    out = {"facts": [{"text": "Uses Poetry", "scope": "global", "origin": "user"}]}
    [c] = extract(FakeLLM([out]), T, [], related, 20).candidates
    assert c.text == "Uses Poetry" and c.restates is None
    assert "WARNING" not in caplog.text


def test_extract_keeps_duplicate_texts_for_apply_to_collapse():
    # extract must not dedup first-wins: apply_candidates keeps the more trusted of the two.
    out = {"facts": [{"text": "Uses Fedora", "scope": "system", "origin": "assistant", "restates": None},
                     {"text": " uses  fedora ", "scope": "system", "origin": "user", "restates": None}]}
    cands = extract(FakeLLM([out]), T, [], [], 20).candidates
    assert [c.origin for c in cands] == ["assistant", "user"]


def test_extract_resolves_restates_index_to_memory_id(caplog):
    related = [rec("Uses Poetry", id="m-0"), rec("Runs Fedora 44", id="m-1")]
    out = {"facts": [{"text": "Uses Poetry", "scope": "global", "origin": "user", "restates": 0},
                     {"text": "Runs Fedora 44", "scope": "global", "origin": "user", "restates": 1},
                     {"text": "Uses fish", "scope": "global", "origin": "user", "restates": None},
                     {"text": "Uses zsh", "scope": "global", "origin": "user", "restates": 2},
                     {"text": "Uses vim", "scope": "global", "origin": "user", "restates": -1},
                     {"text": "Uses tmux", "scope": "global", "origin": "user", "restates": "Uses tmux"},
                     {"text": "Uses git", "scope": "global", "origin": "user", "restates": True}]}
    cands = extract(FakeLLM([out]), T, [], related, 20).candidates
    assert [c.restates for c in cands] == ["m-0", "m-1", None, None, None, None, None]
    assert caplog.text.count("invalid restates index") == 4 and "Uses tmux" not in caplog.text


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


def test_prompt_defines_scopes_by_subject():
    s, _ = build_prompt([Turn("user", "x")], [], [])
    assert '"global": the person.' in s
    assert 'even when the tool is software' in s
    assert 'only a fact the user ties to this repository' in s
    assert '"system": facts true of this machine regardless of who asks' in s
    assert '"project": only the current repository' in s
    assert "If unsure between global and system for a preference, use global." in s
