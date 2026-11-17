import json
import subprocess
import sys

from rem.models import Scope, memory_id
from rem.review import (
    NARRATION_VERBS, RULES, RunRows, candidate_memory_id, parse_extraction, review_run, run_summary,
)


def cand(text, scope="global", origin="user", restates=None, entities=(), relations=()):
    return {"text": text, "scope": scope, "origin": origin, "restates": restates,
            "entities": list(entities), "relations": list(relations)}


def hist(event, target_id, kind="memory", n=[0]):
    n[0] += 1
    return {"id": n[0], "target_kind": kind, "event": event, "target_id": target_id, "old_value": None,
            "new_value": None, "actor": "extractor", "job_id": "job:7", "created_at": "t"}


def rows(cands=(), confirmations=(), **over):
    base = dict(
        job={"id": 7, "status": "done", "created_at": "a", "updated_at": "b", "last_error": None,
             "project_id": 3, "project_name": "rem", "source": "claude-code"},
        turns=[{"role": "user", "content": "I use tmux", "ts": "1"},
               {"role": "assistant", "content": "ok", "ts": "2"}],
        context_turns=[],
        extraction=("m", json.dumps({"candidates": list(cands), "confirmations": list(confirmations)})),
        history=[], memories_now={}, relations_by_memory={},
    )
    base.update(over)
    return RunRows(**base)


def rules_of(fact):
    return [f["rule"] for f in fact["flags"]]


def test_added_fact_with_memory_now():
    mid = memory_id("Uses tmux", Scope.GLOBAL, None, "job:7")
    now = {"id": mid, "text": "Uses tmux", "status": "verified", "provenance": "user_asserted"}
    out = review_run(rows([cand("Uses tmux")], history=[hist("ADD", mid)], memories_now={mid: now}))
    f = out["facts"][0]
    assert f["outcome"] == {"kind": "added", "memory_id": mid, "memory_now": now}
    assert f["flags"] == [] and f["origin_effective"] == "user"
    assert out["run"]["flags"] == []


def test_project_scope_id_uses_project():
    mid = memory_id("Uses uv", Scope.PROJECT, 3, "job:7")
    out = review_run(rows([cand("Uses uv", "project", "assistant")], history=[hist("ADD", mid)]))
    assert out["facts"][0]["outcome"]["kind"] == "added"


def test_downgraded_when_user_never_said_it():
    out = review_run(rows([cand("Prefers neovim")]))
    f = out["facts"][0]
    assert f["origin_proposed"] == "user" and f["origin_effective"] == "assistant"
    assert "downgraded" in rules_of(f)
    noturn = review_run(rows([cand("Uses tmux")], turns=[{"role": "assistant", "content": "I use tmux", "ts": "1"}]))
    assert noturn["facts"][0]["origin_effective"] == "assistant"
    assert "downgraded" in rules_of(noturn["facts"][0])


def test_duplicate_and_no_project_reasons():
    out = review_run(rows([cand("Uses tmux"), cand("uses   TMUX")]))
    assert out["facts"][1]["outcome"] == {"kind": "none", "reason": "duplicate"}
    out = review_run(rows([cand("Uses uv", "project")], job={**rows().job, "project_id": None}))
    assert out["facts"][0]["outcome"] == {"kind": "none", "reason": "no_project"}


def test_duplicate_keys_on_scope_too():
    out = review_run(rows([cand("Uses tmux", "global"), cand("Uses tmux", "system")]))
    assert out["facts"][1]["outcome"]["reason"] == "existing_or_skipped"


def test_corroborated_by_text_and_by_restates():
    now = {"m1": {"id": "m1", "text": "Uses  Tmux", "status": "verified", "provenance": "user_asserted"},
           "m2": {"id": "m2", "text": "other words", "status": "verified", "provenance": "user_asserted"}}
    out = review_run(rows([cand("Uses tmux"), cand("Likes vim", restates="m2")],
                          history=[hist("CORROBORATE", "m1"), hist("CORROBORATE", "m2")], memories_now=now,
                          turns=[{"role": "user", "content": "I use tmux. I like vim.", "ts": "1"}]))
    assert out["facts"][0]["outcome"] == {"kind": "corroborated", "memory_id": "m1", "memory_now": now["m1"]}
    assert out["facts"][1]["outcome"]["memory_id"] == "m2"


def test_one_corroborate_row_credits_one_fact():
    now = {"m1": {"id": "m1", "text": "uses tmux", "status": "verified", "provenance": "user_asserted"}}
    out = review_run(rows([cand("Uses tmux", "global"), cand("Uses tmux", "system")],
                          history=[hist("CORROBORATE", "m1")], memories_now=now))
    kinds = [f["outcome"]["kind"] for f in out["facts"]]
    assert kinds == ["corroborated", "none"]


def test_corroboration_not_credited_to_unverified_fact():
    now = {"m1": {"id": "m1", "text": "Uses tmux", "status": "verified", "provenance": "user_asserted"}}
    out = review_run(rows([cand("Likes screen", "global", "assistant", restates="m1"), cand("Uses tmux")],
                          history=[hist("CORROBORATE", "m1")], memories_now=now))
    assert out["facts"][0]["outcome"] == {"kind": "none", "reason": "existing_or_skipped"}
    assert out["facts"][1]["outcome"]["kind"] == "corroborated"


def test_existing_or_skipped():
    out = review_run(rows([cand("Uses tmux")]))
    assert out["facts"][0]["outcome"] == {"kind": "none", "reason": "existing_or_skipped"}


def test_added_then_deleted_memory():
    mid = memory_id("Uses tmux", Scope.GLOBAL, None, "job:7")
    out = review_run(rows([cand("Uses tmux")], history=[hist("ADD", mid)], memories_now={mid: None}))
    assert out["facts"][0]["outcome"]["kind"] == "added"
    assert out["facts"][0]["outcome"]["memory_now"] is None


def test_flag_rules():
    assert set(RULES) == {"downgraded", "invalid-type", "noise", "narration", "scope-hint",
                          "relation-not-written", "failed"}
    assert "fixed" in NARRATION_VERBS
    r = rows([
        cand("Uses git", entities=[{"name": "git", "type": "entity_type_commit"}, {"name": "x", "type": "tool"}]),
        cand("PR URL is https://x/pull/1"),
        cand("Commit ID is 82fd0fc"),
        cand("Fixed the layout", origin="assistant"),
        cand("Fixed the layout today", origin="user"),
        cand("rem uses Qdrant", "system", "assistant"),
        cand("This repo uses uv", "global", "assistant"),
        cand("Uses uvicorn and httpx 1000000", "global", "assistant"),
    ], turns=[{"role": "user", "content": "Fixed the layout today. I use git.", "ts": "1"}])
    f = review_run(r)["facts"]
    assert rules_of(f[0]) == ["invalid-type"]
    assert "noise" in rules_of(f[1]) and "noise" in rules_of(f[2])
    assert "narration" in rules_of(f[3])
    assert "narration" not in rules_of(f[4])
    assert "scope-hint" in rules_of(f[5]) and "scope-hint" in rules_of(f[6])
    assert rules_of(f[7]) == []
    assert all(x["reason"] for fact in f for x in fact["flags"])


def test_scope_hint_whole_word():
    f = review_run(rows([cand("Uses remote hosts", "global", "assistant")]))["facts"][0]
    assert "scope-hint" not in rules_of(f)


def test_relation_not_written():
    mid = memory_id("Uses tmux", Scope.GLOBAL, None, "job:7")
    now = {mid: {"id": mid, "text": "Uses tmux", "status": "verified", "provenance": "user_asserted"}}
    rel = [{"source": "a", "type": "uses", "target": "b"}]
    c = cand("Uses tmux", relations=rel)
    r = rows([c], history=[hist("ADD", mid)], memories_now=now)
    assert "relation-not-written" in rules_of(review_run(r)["facts"][0])
    ok = rows([c], history=[hist("ADD", mid)], memories_now=now, relations_by_memory={mid: ["r1"]})
    assert rules_of(review_run(ok)["facts"][0]) == []
    assert rules_of(review_run(rows([c]))["facts"][0]) == []


def test_relation_not_written_skipped_after_delete_or_edit():
    # rem delete and a text-changing update unlink the memory's relations: that is not a lost write
    mid = memory_id("Uses tmux", Scope.GLOBAL, None, "job:7")
    c = cand("Uses tmux", relations=[{"source": "a", "type": "uses", "target": "b"}])
    deleted = rows([c], history=[hist("ADD", mid)], memories_now={mid: None})
    assert rules_of(review_run(deleted)["facts"][0]) == []
    edited = rows([c], history=[hist("ADD", mid)], memories_now={
        mid: {"id": mid, "text": "Uses screen", "status": "verified", "provenance": "user_asserted"}})
    assert rules_of(review_run(edited)["facts"][0]) == []
    spacing = rows([c], history=[hist("ADD", mid)], memories_now={
        mid: {"id": mid, "text": "uses  TMUX ", "status": "verified", "provenance": "user_asserted"}})
    assert rules_of(review_run(spacing)["facts"][0]) == ["relation-not-written"]


def test_duplicate_credits_the_most_trusted_copy():
    # Memory._collapse_duplicates keeps the first verified copy of a (text, scope) group, at the
    # group's first position; review credits that copy and shows the others as duplicates
    now = {"m1": {"id": "m1", "text": "Uses tmux", "status": "verified", "provenance": "user_asserted"}}
    out = review_run(rows([cand("Uses tmux", origin="assistant"), cand("uses tmux", origin="user")],
                          history=[hist("CORROBORATE", "m1")], memories_now=now))
    f = out["facts"]
    assert f[0]["outcome"] == {"kind": "none", "reason": "duplicate"}
    assert f[1]["outcome"] == {"kind": "corroborated", "memory_id": "m1", "memory_now": now["m1"]}
    assert out["run"]["n_corroborated"] == 1
    # with no verified copy the first one stays the representative
    out = review_run(rows([cand("Prefers neovim"), cand("prefers neovim", origin="assistant")]))
    assert [x["outcome"]["reason"] for x in out["facts"]] == ["existing_or_skipped", "duplicate"]


def test_duplicate_group_pools_relations_for_the_rule():
    mid = memory_id("Uses tmux", Scope.GLOBAL, None, "job:7")
    now = {mid: {"id": mid, "text": "Uses tmux", "status": "verified", "provenance": "user_asserted"}}
    rel = [{"source": "a", "type": "uses", "target": "b"}]
    out = review_run(rows([cand("Uses tmux"), cand("uses tmux", relations=rel)],
                          history=[hist("ADD", mid)], memories_now=now))
    assert rules_of(out["facts"][0]) == ["relation-not-written"]


def test_add_falls_back_to_text_when_the_id_was_rederived():
    # Memory.add derives uuid5(mid) when mid is taken (an updated memory kept the id of its old text)
    taken = memory_id("Uses tmux", Scope.GLOBAL, None, "job:7")
    other = memory_id("Uses tmux", Scope.SYSTEM, None, "job:7")
    a = {**hist("ADD", "rederived"), "new_value": "Uses tmux"}
    b = {**hist("ADD", other), "new_value": "Uses tmux"}
    out = review_run(rows([cand("Uses tmux", "system"), cand("Uses tmux")], history=[b, a]))
    assert out["facts"][0]["outcome"]["memory_id"] == other
    assert out["facts"][1]["outcome"]["memory_id"] == "rederived"
    assert taken != "rederived"
    # each ADD row is credited at most once
    out = review_run(rows([cand("Uses tmux"), cand("Uses tmux", "system")], history=[a]))
    assert [x["outcome"]["kind"] for x in out["facts"]] == ["added", "none"]


def test_candidate_memory_id():
    assert candidate_memory_id({"text": "Uses uv", "scope": "project"}, 3, 7) == memory_id(
        "Uses uv", Scope.PROJECT, 3, "job:7")
    assert candidate_memory_id({"text": "Uses uv", "scope": "global"}, 3, 7) == memory_id(
        "Uses uv", Scope.GLOBAL, None, "job:7")


def test_failed_run_flag():
    r = rows([], job={**rows().job, "status": "failed", "last_error": "boom"})
    out = review_run(r)
    assert out["run"]["run_flags"] == [{"rule": "failed", "reason": "boom"}]
    assert out["run"]["flags"] == ["failed"] and out["run"]["error"] == "boom"


def test_run_summary_counts_and_flags():
    mid = memory_id("Uses tmux", Scope.GLOBAL, None, "job:7")
    now = {"m1": {"id": "m1", "text": "likes vim", "status": "verified", "provenance": "user_asserted"}}
    r = rows([cand("Uses tmux"), cand("Likes vim"), cand("Commit ID is 82fd0fc"), cand("Prefers emacs")],
             history=[hist("ADD", mid), hist("CORROBORATE", "m1")], memories_now=now,
             turns=[{"role": "user", "content": "I use tmux. I like vim.", "ts": "1"}])
    s = run_summary(r)
    assert (s["n_turns"], s["n_proposed"], s["n_added"], s["n_corroborated"], s["model"]) == (1, 4, 1, 1, "m")
    assert s["flags"] == ["downgraded", "noise"] and s["id"] == 7 and s["project_name"] == "rem"


def test_writes_and_confirmations():
    out = review_run(rows([], confirmations=[{"id": "x", "quote": "q"}], history=[hist("ADD", "e1", "entity")]))
    assert out["confirmations"] == [{"id": "x", "quote": "q"}]
    assert out["writes"][0]["target_kind"] == "entity" and out["writes"][0]["target_id"] == "e1"


def test_other_memory_writes():
    mid = memory_id("Uses tmux", Scope.GLOBAL, None, "job:7")
    now = {"m9": {"id": "m9", "text": "Likes vim", "status": "verified", "provenance": "user_confirmed"}}
    h = [hist("ADD", mid), hist("CONFIRM", "m9"), {**hist("ADD", "m5"), "new_value": "Uses zsh"},
         hist("CORROBORATE", "m6"), hist("ADD", "e1", "entity")]
    out = review_run(rows([cand("Uses tmux")], history=h, memories_now=now,
                          confirmations=[{"memory_id": "m9", "hash": "h", "quote": "yes, I like vim"}]))
    other = out["other_memory_writes"]
    assert [(w["event"], w["target_id"]) for w in other] == [("CONFIRM", "m9"), ("ADD", "m5"), ("CORROBORATE", "m6")]
    assert other[0]["memory_now"] == now["m9"] and other[1]["memory_now"] is None
    assert out["confirmations"] == [{"memory_id": "m9", "hash": "h", "quote": "yes, I like vim"}]


def test_run_without_extraction():
    out = review_run(rows(extraction=None))
    assert out["facts"] == [] and out["run"]["model"] is None


def test_parse_extraction_bare_list():
    assert parse_extraction('[{"text": "a"}]') == ([{"text": "a"}], [])
    assert parse_extraction('{"candidates": [{"text": "a"}], "confirmations": [{"id": "1"}]}') == (
        [{"text": "a"}], [{"id": "1"}])


def test_review_is_pure():
    code = ("import sys, rem.review; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('httpx', 'PySide6') "
            "or m.startswith(('rem.memory', 'rem.storage', 'rem.daemon'))]; "
            "print(bad); sys.exit(1 if bad else 0)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout
