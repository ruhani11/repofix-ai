"""Atomic, account-scoped review log and compatible proposal DB migration."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
import sqlite3
from threading import Barrier
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.github_oauth import GitHubIdentity
from app.proposal_decision import ProposalDecisionError, ProposalDecisionInput
from app.proposal_review import ProposalDecisionRequest, ProposalReviewError
from app.proposal_store import ProposalStore
from test_proposal_store import record


OWNER = GitHubIdentity(id=123, login="demo-user", type="User")
OTHER = GitHubIdentity(id=999, login="other-user", type="User")


def decision(proposal, **changes):
    return ProposalDecisionInput(**{
        "proposal_id": proposal.proposal_id, "base_commit_sha": proposal.base_commit_sha,
        "patch_sha256": proposal.patch_sha256, "decision": "approve_for_validation", **changes,
    })


@pytest.fixture
def saved(tmp_path):
    path = tmp_path / "proposals.db"
    store = ProposalStore(path)
    proposal = record()
    store.save(OWNER.id, proposal)
    yield path, store, proposal
    store.close()


def test_approve_reject_reapprove_revisions_history_and_restart(saved):
    path, store, proposal = saved
    before = store.get(OWNER.id, proposal.proposal_id).model_dump_json()
    initial = store.get_review(OWNER.id, proposal.proposal_id)
    assert initial.status == "unreviewed" and initial.revision == 0 and initial.current_decision is None
    states = []
    for revision, action in enumerate(["approve_for_validation", "reject", "approve_for_validation"]):
        states.append(store.record_decision(OWNER, decision(proposal, decision=action), expected_revision=revision))
    assert [item.status for item in states] == ["approved_for_validation", "rejected", "approved_for_validation"]
    assert [item.revision for item in states] == [1, 2, 3]
    assert len({item.current_decision.decision_id for item in states}) == 3
    assert store.get(OWNER.id, proposal.proposal_id).model_dump_json() == before
    store.close()
    reopened = ProposalStore(path)
    assert reopened.get_review(OWNER.id, proposal.proposal_id) == states[-1]
    history = reopened.list_decisions(OWNER.id, proposal.proposal_id)
    assert [entry.revision for entry in history.decisions] == [3, 2, 1]
    assert [entry.decision for entry in history.decisions] == [item.current_decision for item in reversed(states)]
    reopened.close()


def test_stale_double_submit_and_old_approval_cannot_override_rejection(saved):
    _, store, proposal = saved
    payload = decision(proposal)
    store.record_decision(OWNER, payload, expected_revision=0)
    store.record_decision(OWNER, decision(proposal, decision="reject"), expected_revision=1)
    for revision in [0, 1, 3]:
        with pytest.raises(ProposalReviewError) as caught:
            store.record_decision(OWNER, payload, expected_revision=revision)
        assert caught.value.code == "revision_conflict"
    assert store.get_review(OWNER.id, proposal.proposal_id).status == "rejected"
    assert len(store.list_decisions(OWNER.id, proposal.proposal_id).decisions) == 2


@pytest.mark.parametrize("method", ["get", "list", "record"])
def test_foreign_and_missing_are_indistinguishable(saved, method):
    _, store, proposal = saved
    reasons = []
    for identity, identifier in [(OTHER, proposal.proposal_id), (OWNER, "0" * 32)]:
        with pytest.raises(ProposalReviewError) as caught:
            if method == "get": store.get_review(identity.id, identifier)
            elif method == "list": store.list_decisions(identity.id, identifier)
            else: store.record_decision(identity, decision(proposal, proposal_id=identifier), expected_revision=0)
        reasons.append((caught.value.code, str(caught.value)))
    assert reasons[0] == reasons[1] and reasons[0][0] == "not_found"


@pytest.mark.parametrize("field,value", [("base_commit_sha", "d" * 40), ("patch_sha256", "d" * 64)])
def test_changed_reviewed_bytes_do_not_save(saved, field, value):
    _, store, proposal = saved
    with pytest.raises(ProposalDecisionError) as caught:
        store.record_decision(OWNER, decision(proposal, **{field: value}), expected_revision=0)
    assert caught.value.code == "proposal_changed"
    assert store.get_review(OWNER.id, proposal.proposal_id).revision == 0


def test_transaction_rolls_back_if_session_recheck_fails(saved):
    _, store, proposal = saved
    authorize = Mock(side_effect=RuntimeError("PRIVATE_REVOKED_SESSION"))
    with pytest.raises(RuntimeError):
        store.record_decision(OWNER, decision(proposal), expected_revision=0, authorize=authorize)
    authorize.assert_called_once()
    assert store.get_review(OWNER.id, proposal.proposal_id).revision == 0
    assert store.list_decisions(OWNER.id, proposal.proposal_id).decisions == []


def test_database_insert_failure_does_not_claim_approval_and_can_recover(saved):
    path, store, proposal = saved
    with sqlite3.connect(path) as connection:
        connection.executescript("CREATE TRIGGER deny_review BEFORE INSERT ON fix_proposal_decisions BEGIN SELECT RAISE(ABORT, 'PRIVATE_DISK_ERROR'); END;")
    with pytest.raises(ProposalReviewError) as caught:
        store.record_decision(OWNER, decision(proposal), expected_revision=0)
    assert caught.value.code == "unavailable" and "PRIVATE" not in str(caught.value)
    assert store.get_review(OWNER.id, proposal.proposal_id).status == "unreviewed"
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER deny_review")
    assert store.record_decision(OWNER, decision(proposal), expected_revision=0).revision == 1


def test_two_independent_connections_cannot_both_accept_same_revision(saved):
    path, first, proposal = saved
    second = ProposalStore(path)
    barrier = Barrier(2)
    def attempt(store):
        observed = store.get_review(OWNER.id, proposal.proposal_id)
        assert observed.revision == 0
        barrier.wait(timeout=5)
        try:
            return store.record_decision(OWNER, decision(proposal), expected_revision=observed.revision).status
        except ProposalReviewError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, [first, second]))
    assert sorted(results) == ["approved_for_validation", "revision_conflict"]
    assert first.get_review(OWNER.id, proposal.proposal_id).revision == 1
    assert len(first.list_decisions(OWNER.id, proposal.proposal_id).decisions) == 1
    second.close()


def test_pagination_uses_revision_even_if_timestamps_tie(saved, monkeypatch):
    import app.proposal_store as module
    _, store, proposal = saved
    original = module.bind_proposal_decision
    instant = proposal.created_at + timedelta(seconds=1)
    monkeypatch.setattr(module, "bind_proposal_decision", lambda identity, loaded, payload: original(identity, loaded, payload, now=instant))
    for revision in range(3):
        store.record_decision(OWNER, decision(proposal), expected_revision=revision)
    first = store.list_decisions(OWNER.id, proposal.proposal_id, per_page=2)
    assert [entry.revision for entry in first.decisions] == [3, 2] and first.has_more and first.next_page == 2
    last = store.list_decisions(OWNER.id, proposal.proposal_id, page=2, per_page=2)
    assert [entry.revision for entry in last.decisions] == [1] and not last.has_more
    assert store.list_decisions(OWNER.id, proposal.proposal_id, page=10000).decisions == []


@pytest.mark.parametrize("field,value", [
    ("patch_sha256", "d" * 64), ("base_commit_sha", "d" * 40), ("tree_sha", "d" * 40),
    ("repository_id", 999), ("investigation_run_id", "d" * 32), ("account_id", 999),
    ("proposal_id", "d" * 32), ("decision", "publish"),
])
def test_corrupt_latest_record_fails_closed_instead_of_falling_back_to_old_approval(saved, field, value):
    path, store, proposal = saved
    store.record_decision(OWNER, decision(proposal), expected_revision=0)
    latest = store.record_decision(OWNER, decision(proposal, decision="reject"), expected_revision=1)
    data = latest.current_decision.model_dump(mode="json")
    data[field] = value
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE fix_proposal_decisions SET detail_json=? WHERE revision=2", (json.dumps(data),))
    for action in [lambda: store.get_review(OWNER.id, proposal.proposal_id), lambda: store.list_decisions(OWNER.id, proposal.proposal_id), lambda: store.record_decision(OWNER, decision(proposal), expected_revision=2)]:
        with pytest.raises(ProposalReviewError) as caught:
            action()
        assert caught.value.code == "unavailable"


def test_legacy_schema_v1_migrates_atomically_without_changing_existing_proposal(tmp_path):
    path = tmp_path / "legacy.db"
    proposal = record()
    encoded = proposal.model_dump_json()
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE fix_proposals (
                proposal_id TEXT PRIMARY KEY, account_id INTEGER NOT NULL CHECK(account_id > 0),
                investigation_run_id TEXT NOT NULL, created_at TEXT NOT NULL, repository_id INTEGER NOT NULL,
                repository_full_name TEXT NOT NULL, base_commit_sha TEXT NOT NULL, title TEXT NOT NULL,
                patch_sha256 TEXT NOT NULL, total_duration_ms INTEGER NOT NULL, detail_json TEXT NOT NULL
            );
            PRAGMA user_version=1;
        """)
        connection.execute("INSERT INTO fix_proposals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            proposal.proposal_id, OWNER.id, proposal.investigation_run_id, proposal.created_at.isoformat(),
            proposal.repository_id, proposal.repository_full_name, proposal.base_commit_sha,
            proposal.draft.title, proposal.patch_sha256, proposal.total_duration_ms, encoded,
        ))
    store = ProposalStore(path)
    assert store.get(OWNER.id, proposal.proposal_id) == proposal
    assert store.get_review(OWNER.id, proposal.proposal_id).revision == 0
    store.record_decision(OWNER, decision(proposal), expected_revision=0)
    store.close()
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert connection.execute("SELECT detail_json FROM fix_proposals").fetchone()[0] == encoded
        assert connection.execute("SELECT count(*) FROM fix_proposal_decisions").fetchone()[0] == 1


@pytest.mark.parametrize("revision", [True, -1, "0", 2**63 - 1])
def test_expected_revision_is_strict_and_required(saved, revision):
    _, store, proposal = saved
    with pytest.raises(ProposalReviewError) as caught:
        store.record_decision(OWNER, decision(proposal), expected_revision=revision)
    assert caught.value.code == "invalid_input"
    with pytest.raises(ValidationError):
        ProposalDecisionRequest(**{**decision(proposal).model_dump(), "expected_revision": revision})


@pytest.mark.parametrize("account,identifier", [(True, "a" * 32), (0, "a" * 32), (2**63, "a" * 32), (123, "' OR 1=1 --"), (123, "A" * 32)])
def test_invalid_review_key_rejected_before_disk(tmp_path, account, identifier):
    path = tmp_path / "db"
    store = ProposalStore(path)
    for action in [lambda: store.get_review(account, identifier), lambda: store.list_decisions(account, identifier)]:
        with pytest.raises(ProposalReviewError) as caught:
            action()
        assert caught.value.code == "invalid_input"
    assert not path.exists()


@pytest.mark.parametrize("paging", [{"page": 0}, {"page": 10001}, {"per_page": 51}, {"per_page": True}])
def test_decision_history_pagination_is_bounded(saved, paging):
    _, store, proposal = saved
    with pytest.raises(ProposalReviewError) as caught:
        store.list_decisions(OWNER.id, proposal.proposal_id, **paging)
    assert caught.value.code == "invalid_input"


def test_review_record_size_limit_and_missing_request_revision(saved, monkeypatch):
    import app.proposal_store as module
    _, store, proposal = saved
    monkeypatch.setattr(module, "MAX_DECISION_RECORD_BYTES", 1)
    with pytest.raises(ProposalReviewError) as caught:
        store.record_decision(OWNER, decision(proposal), expected_revision=0)
    assert caught.value.code == "invalid_input"
    assert store.get_review(OWNER.id, proposal.proposal_id).revision == 0
    with pytest.raises(ValidationError):
        ProposalDecisionRequest(**decision(proposal).model_dump())
