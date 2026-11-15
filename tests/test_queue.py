import pytest

from rem.config import DaemonConfig
from rem.storage.queue import IngestStore, Turn
from rem.storage.sqlite import RemDB


class FakeClock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def q(tmp_path, clock):
    return IngestStore(RemDB(tmp_path / "rem.db"), DaemonConfig(debounce_seconds=5, lease_seconds=120, max_attempts=3), clock)


def T(role, content, mid=None):
    return Turn(role, content, mid)


def test_ingest_is_idempotent_and_debounced(q, clock):
    r1 = q.ingest("claude-code", "s1", None, [T("user", "I prefer uv", "m1"), T("assistant", "Noted", "m2")])
    assert (r1.accepted, r1.duplicates) == (2, 0) and r1.job_id is not None
    r2 = q.ingest("claude-code", "s1", None, [T("user", "I prefer uv", "m1")])
    assert (r2.accepted, r2.duplicates, r2.job_id) == (0, 1, None)
    assert q.counts() == {"pending": 1, "processing": 0, "done": 0, "failed": 0}
    assert q.claim() is None
    clock.t += 5
    job = q.claim()
    assert job.session_id == "s1" and job.source == "claude-code"
    assert [t.content for t in q.messages_for(job.id)] == ["I prefer uv", "Noted"]


def test_more_turns_while_pending_extend_debounce_not_jobs(q, clock):
    q.ingest("cli", "s", None, [T("user", "a", "1")])
    clock.t += 4
    q.ingest("cli", "s", None, [T("user", "b", "2")])
    clock.t += 2
    assert q.claim() is None
    clock.t += 3
    job = q.claim()
    assert [t.content for t in q.messages_for(job.id)] == ["a", "b"] and q.counts()["processing"] == 1


def test_missing_message_id_uses_content_hash(q):
    q.ingest("cli", "s", None, [T("user", "same")])
    assert q.ingest("cli", "s", None, [T("user", "same")]).duplicates == 1


def test_turns_are_redacted_before_storage(q, clock):
    q.ingest("cli", "s", None, [T("user", "key is sk-ant-api03-" + "x" * 40, "m1")])
    clock.t += 5
    assert "sk-ant" not in q.messages_for(q.claim().id)[0].content


def test_session_project_from_cwd(q, tmp_path):
    (tmp_path / "p" / ".git").mkdir(parents=True)
    r = q.ingest("cli", "s", tmp_path / "p", [T("user", "a", "1")])
    assert q.session_project(r.session_pk) is not None


def test_new_turns_while_processing_create_a_second_job_after(q, clock):
    q.ingest("cli", "s", None, [T("user", "a", "1")])
    clock.t += 5
    j1 = q.claim()
    q.ingest("cli", "s", None, [T("user", "b", "2")])
    clock.t += 5
    assert q.claim() is None
    q.complete(j1.id)
    j2 = q.claim()
    assert [t.content for t in q.messages_for(j2.id)] == ["b"]
    assert [t.content for t in q.context_for(j2.id)] == ["a"]


def test_expired_lease_is_reclaimed_with_same_messages(q, clock):
    q.ingest("cli", "s", None, [T("user", "a", "1")])
    clock.t += 5
    j = q.claim()
    clock.t += 121
    again = q.claim()
    assert again.id == j.id and [t.content for t in q.messages_for(again.id)] == ["a"]


def test_fail_backs_off_then_gives_up(q, clock):
    q.ingest("cli", "s", None, [T("user", "a", "1")])
    clock.t += 5
    j = q.claim()
    q.fail(j.id, "boom")
    assert q.claim() is None
    clock.t += 5
    j = q.claim()
    q.fail(j.id, "boom")
    clock.t += 10
    j = q.claim()
    q.fail(j.id, "boom")
    assert q.counts()["failed"] == 1 and q.claim() is None


def test_release_and_recover(q, clock):
    q.ingest("cli", "s", None, [T("user", "a", "1")])
    clock.t += 5
    j = q.claim()
    q.release(j.id)
    assert q.counts()["pending"] == 1
    q.claim()
    clock.t += 200
    assert q.recover() == 1 and q.counts()["pending"] == 1


def test_recover_at_startup_resets_unexpired_processing_jobs(q, clock):
    q.ingest("cli", "s", None, [T("user", "a", "1")])
    clock.t += 5
    q.claim()
    assert q.recover() == 1 and q.claim() is not None
