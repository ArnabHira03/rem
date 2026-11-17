import pytest

from rem.cli import main
from rem.embed.hashing import HashingEmbedder
from rem.errors import LLMError
from rem.evals.extraction import DEFAULT_DATASET, load_cases, match_facts, run_eval
from rem.llm.fake import FakeLLM

FISH = {"text": "Uses fish shell", "scope": "global", "origin": "user"}
CASE_FISH = {
    "id": "fish", "turns": [{"role": "user", "content": "I use fish shell"}], "context": [],
    "existing": [], "expected": [FISH], "expect_corroborate": [],
}
CASE_EMPTY = {
    "id": "empty", "turns": [{"role": "user", "content": "thanks"}], "context": [],
    "existing": [], "expected": [], "expect_corroborate": [],
}
POETRY = "The project manages dependencies with Poetry"
CASE_DEDUP = {
    "id": "dedup", "turns": [{"role": "user", "content": "yes we use Poetry"}], "context": [],
    "existing": [{"text": POETRY, "scope": "project", "status": "unverified"}],
    "expected": [{"text": POETRY, "scope": "project", "origin": "user"}], "expect_corroborate": [POETRY],
}


def said(*facts, restates=None):
    """An LLM response: the given facts with the schema's required restates field."""
    return {"facts": [{**f, "restates": restates} for f in facts]}


@pytest.fixture
def hashing():
    return HashingEmbedder()


def test_match_facts_one_to_one(hashing):
    assert match_facts(["Uses fish shell"], ["Uses fish shell", "Uses zsh"], hashing) == [(0, 0)]


def test_perfect_fake_scores_one(cfg, hashing):
    llm = FakeLLM([said(FISH), said()])
    r = run_eval(llm, hashing, [CASE_FISH, CASE_EMPTY], cfg)
    assert (r.precision, r.recall, r.scope_acc, r.origin_acc, r.json_valid_rate) == (1, 1, 1, 1, 1)
    assert r.cases == 2 and r.failures == [] and r.dedup_acc is None  # no dedup cases: nothing to score


def test_invalid_json_counts_against_validity(cfg, hashing):
    r = run_eval(FakeLLM([LLMError("invalid JSON")]), hashing, [CASE_FISH], cfg)
    assert r.json_valid_rate == 0 and r.recall == 0 and r.failures[0]["id"] == CASE_FISH["id"]


def test_dedup_case_scores_corroboration(cfg, hashing):
    llm = FakeLLM([said(*CASE_DEDUP["expected"], restates=0)])
    assert run_eval(llm, hashing, [CASE_DEDUP], cfg).dedup_acc == 1


def test_dedup_case_fails_when_nothing_extracted(cfg, hashing):
    r = run_eval(FakeLLM([{"facts": []}]), hashing, [CASE_DEDUP], cfg)
    assert r.dedup_acc == 0 and r.failures[0]["id"] == "dedup"


def test_wrong_scope_lowers_scope_accuracy(cfg, hashing):
    wrong = {**FISH, "scope": "system"}
    r = run_eval(FakeLLM([said(wrong)]), hashing, [CASE_FISH], cfg)
    assert r.recall == 1 and r.scope_acc == 0 and r.origin_acc == 1


def test_eval_never_touches_the_real_store(cfg, hashing):
    run_eval(FakeLLM([said(FISH)]), hashing, [CASE_FISH], cfg)
    assert not cfg.db_path.exists()


def test_shipped_dataset_is_well_formed():
    cases = load_cases(DEFAULT_DATASET)
    assert len(cases) >= 30 and len({c["id"] for c in cases}) == len(cases)
    assert sum(1 for c in cases if not c["expected"]) >= 4
    assert sum(1 for c in cases if c["expect_corroborate"]) >= 4
    assert any("[REDACTED:" in t["content"] for c in cases for t in c["turns"])


def test_cli_eval_without_llm_exits_3(rem_home, capsys):
    assert main(["eval", "extraction"]) == 3
    assert "no [llm] configured" in capsys.readouterr().err


def test_cli_eval_bad_dataset_exits_2(tmp_path, monkeypatch, capsys):
    cfgfile = tmp_path / "c.toml"
    cfgfile.write_text('[llm]\nprovider = "ollama"\nmodel = "x"\n')
    monkeypatch.setenv("REM_HOME", str(tmp_path))
    monkeypatch.setenv("REM_CONFIG", str(cfgfile))
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    assert main(["eval", "extraction", "--dataset", str(bad)]) == 2
    assert main(["eval", "extraction", "--dataset", str(tmp_path / "missing.jsonl")]) == 2
    assert "cannot read dataset" in capsys.readouterr().err


def test_rates_without_a_denominator_are_none(cfg, hashing):
    r = run_eval(FakeLLM([said()]), hashing, [CASE_EMPTY], cfg)
    assert (r.precision, r.recall, r.scope_acc, r.origin_acc, r.dedup_acc) == (None,) * 5
    assert r.json_valid_rate == 1 and r.to_dict()["precision"] is None


def test_dedup_paraphrase_needs_restates_link(cfg, hashing):
    paraphrase = {"text": "Project manages dependencies with Poetry", "scope": "project", "origin": "user"}
    linked = run_eval(FakeLLM([said(paraphrase, restates=0)]), hashing, [CASE_DEDUP], cfg)
    unlinked = run_eval(FakeLLM([said(paraphrase)]), hashing, [CASE_DEDUP], cfg)
    assert linked.dedup_acc == 1 and unlinked.dedup_acc == 0


def test_eval_records_provider_exception_as_failure(cfg, hashing):
    from rem.config import LLMConfig
    from rem.llm.anthropic import AnthropicLLM
    from tests.test_llm_providers import StubAnthropic

    llm = AnthropicLLM(LLMConfig(provider="anthropic", model="m", allow_remote=True),
                       client=StubAnthropic(raises=RuntimeError("socket exploded")))
    r = run_eval(llm, hashing, [CASE_FISH], cfg)
    assert r.json_valid_rate == 0 and "socket exploded" in r.failures[0]["error"]


def _eval_env(tmp_path, monkeypatch, llm_toml):
    cfgfile = tmp_path / "c.toml"
    cfgfile.write_text('[embedder]\nprovider = "hashing"\nmodel = "hash-64"\n' + llm_toml)
    monkeypatch.setenv("REM_HOME", str(tmp_path))
    monkeypatch.setenv("REM_CONFIG", str(cfgfile))


def test_cli_eval_prints_na_for_empty_rates(tmp_path, monkeypatch, capsys):
    import json

    _eval_env(tmp_path, monkeypatch, '[llm]\nprovider = "ollama"\nmodel = "x"\n')
    monkeypatch.setattr("rem.llm.factory.create_llm", lambda cfg: FakeLLM([said(), said()]))
    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps(CASE_EMPTY) + "\n")
    assert main(["eval", "extraction", "--dataset", str(data)]) == 0
    out = capsys.readouterr().out
    assert "dedup accuracy         n/a" in out and "JSON validity         1.00" in out
    assert main(["eval", "extraction", "--dataset", str(data), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["dedup_acc"] is None


def test_cli_eval_without_api_key_exits_3(tmp_path, monkeypatch, capsys):
    _eval_env(tmp_path, monkeypatch, '[llm]\nprovider = "anthropic"\nmodel = "m"\nallow_remote = true\n')
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["eval", "extraction"]) == 3
    assert "set ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_eval_never_contacts_daemon(cfg, hashing, monkeypatch, tmp_path):
    import rem.client

    monkeypatch.setenv("REM_SOCKET", str(tmp_path / "nope.sock"))

    def boom(*a, **k):
        raise AssertionError("eval contacted the daemon")

    monkeypatch.setattr(rem.client.RemClient, "__init__", boom)
    r = run_eval(FakeLLM([said(FISH)]), hashing, [CASE_FISH], cfg)
    assert r.cases == 1


def lab(text, ents, rels=()):
    return {"text": text, "scope": "global", "origin": "user",
            "entities": [{"name": n, "type": t} for n, t in ents],
            "relations": [{"source": s, "type": ty, "target": d} for s, ty, d in rels]}


def _case(cid, *facts):
    return {"id": cid, "turns": [{"role": "user", "content": "x"}], "context": [], "existing": [],
            "expected": list(facts), "expect_corroborate": []}


UV_PIP = lab("Prefers uv over pip", [("uv", "tool"), ("pip", "tool")], [("uv", "preferred_over", "pip")])
LABEL_RATES = ("entity_precision", "entity_recall", "entity_type_acc", "relation_precision",
               "relation_recall", "relation_type_acc")


def test_perfect_labels_score_one(cfg, hashing):
    r = run_eval(FakeLLM([said(UV_PIP), said()]), hashing, [_case("uvpip", UV_PIP), CASE_EMPTY], cfg)
    assert all(getattr(r, k) == 1 for k in LABEL_RATES) and r.invalid_type_rate == 0
    assert r.labelled_cases == 2 and r.failures == []


def test_unlabelled_cases_skip_label_scores(cfg, hashing):
    fish = {**FISH, "entities": [{"name": "fish", "type": "tool"}], "relations": []}
    r = run_eval(FakeLLM([said(fish)]), hashing, [CASE_FISH], cfg)
    assert all(getattr(r, k) is None for k in LABEL_RATES)
    assert r.labelled_cases == 0 and r.precision == 1 and r.recall == 1 and r.invalid_type_rate == 0


def test_entity_matching_uses_normalized_names_and_aliases(cfg, hashing):
    want = lab("Uses PostgreSQL", [("x", "tool")])
    want["entities"] = [{"name": "PostgreSQL", "type": "tool", "aliases": ["Postgres"]}]
    got = lab("Uses PostgreSQL", [("postgres", "tool")])
    r = run_eval(FakeLLM([said(got)]), hashing, [_case("pg", want)], cfg)
    assert r.entity_recall == 1


def test_missing_and_extra_entities_reported(cfg, hashing):
    got = lab("Prefers uv over pip", [("uv", "tool"), ("npm", "tool")])
    r = run_eval(FakeLLM([said(got)]), hashing, [_case("c", lab("Prefers uv over pip", [("uv", "tool"), ("pip", "tool")]))], cfg)
    assert r.entity_precision == r.entity_recall == 0.5
    assert r.failures[0]["missing_entities"] == ["pip"] and r.failures[0]["extra_entities"] == ["npm"]


def test_relation_direction_matters(cfg, hashing):
    got = lab("Prefers uv over pip", [("uv", "tool"), ("pip", "tool")], [("pip", "preferred_over", "uv")])
    r = run_eval(FakeLLM([said(got)]), hashing, [_case("c", UV_PIP)], cfg)
    assert r.relation_precision == r.relation_recall == 0


def test_relation_type_scored_separately(cfg, hashing):
    ents = [("alpha", "project"), ("Qdrant", "tool")]
    want = lab("alpha depends on Qdrant", ents, [("alpha", "depends_on", "Qdrant")])
    got = lab("alpha depends on Qdrant", ents, [("alpha", "uses", "Qdrant")])
    r = run_eval(FakeLLM([said(got)]), hashing, [_case("c", want)], cfg)
    assert r.relation_recall == 1 and r.relation_type_acc == 0
    assert "wrong_relation_type" in r.failures[0]


def test_type_accuracy_and_invalid_rate_are_per_mention(cfg, hashing):
    a = lab("Uses uv daily", [("uv", "tool")])
    b = lab("Likes uv a lot", [("uv", "entity_type_x")])
    r = run_eval(FakeLLM([said(a, b)]), hashing, [_case("c", lab("Uses uv daily", [("uv", "tool")]))], cfg)
    assert r.entity_type_acc == 0.5 and r.invalid_type_rate == 0.5
    assert r.failures[0]["invalid_entity_type"] == [{"name": "uv", "type": "entity_type_x"}]


def test_failed_llm_case_counts_against_label_recall(cfg, hashing):
    r = run_eval(FakeLLM([LLMError("boom")]), hashing, [_case("c", UV_PIP)], cfg)
    assert r.entity_recall == 0 and r.relation_recall == 0 and r.entity_precision is None


def test_cli_eval_prints_label_scores(tmp_path, monkeypatch, capsys):
    import json

    _eval_env(tmp_path, monkeypatch, '[llm]\nprovider = "ollama"\nmodel = "x"\n')
    monkeypatch.setattr("rem.llm.factory.create_llm", lambda cfg: FakeLLM([said(UV_PIP)]))
    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps(_case("uvpip", UV_PIP)) + "\n")
    assert main(["eval", "extraction", "--dataset", str(data)]) == 0
    out = capsys.readouterr().out
    assert f"  {'entity precision':<20}{'1.00':>6}" in out and f"  {'relation recall':<20}{'1.00':>6}" in out
    assert f"  {'invalid types':<20}{'0.00':>6}" in out and "labelled cases" in out
    monkeypatch.setattr("rem.llm.factory.create_llm", lambda cfg: FakeLLM([said(FISH)]))
    data.write_text(json.dumps(CASE_FISH) + "\n")
    assert main(["eval", "extraction", "--dataset", str(data)]) == 0
    out = capsys.readouterr().out
    assert f"  {'entity precision':<20}{'n/a':>6}" in out and f"  {'relation recall':<20}{'n/a':>6}" in out
