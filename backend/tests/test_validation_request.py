"""Offline approval-binding boundaries; no Docker, model calls or target code."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import subprocess

import httpx
import pytest
from pydantic import ValidationError

from app.github_oauth import GitHubIdentity
from app.investigation_runner import InvestigationEvent
from app.proposal_decision import ProposalDecisionInput, bind_proposal_decision
from app.proposal_review import ProposalReviewState
from app.proposal_store import ProposalStore
from app.validation_request import (
    ValidationBinding, ValidationInput, ValidationRequestError, bind_validation_request,
)
from test_proposal_store import record


OWNER = GitHubIdentity(id=123, login="demo-user", type="User")


@pytest.fixture(autouse=True)
def no_network_or_subprocess(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Validation request contracts must not use network or subprocesses.")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)


def action(proposal, *, decision="approve_for_validation"):
    return ProposalDecisionInput(
        proposal_id=proposal.proposal_id, base_commit_sha=proposal.base_commit_sha,
        patch_sha256=proposal.patch_sha256, decision=decision,
    )


def request(proposal, review, **changes):
    return ValidationInput(**{
        "proposal_id": proposal.proposal_id, "base_commit_sha": proposal.base_commit_sha,
        "patch_sha256": proposal.patch_sha256, "expected_revision": review.revision,
        "decision_id": review.current_decision.decision_id, **changes,
    })


@pytest.fixture
def approved():
    proposal = record().model_copy(update={"events": [
        InvestigationEvent("proposal_started", 0),
        InvestigationEvent("tool_completed", 1, "read_file", True),
        InvestigationEvent("proposal_source_validated", 2),
    ]})
    decision = bind_proposal_decision(
        OWNER, proposal, action(proposal), now=proposal.created_at + timedelta(seconds=1),
    )
    review = ProposalReviewState(
        proposal_id=proposal.proposal_id, base_commit_sha=proposal.base_commit_sha,
        patch_sha256=proposal.patch_sha256, revision=1,
        status="approved_for_validation", current_decision=decision,
    )
    return proposal, review, request(proposal, review)


def test_binding_retains_exact_server_provenance_without_mutating_records(approved):
    proposal, review, payload = approved
    before = [item.model_dump_json() for item in (proposal, review, payload)]
    instant = review.current_decision.created_at + timedelta(seconds=1)
    binding = bind_validation_request(OWNER, proposal, review, payload, now=instant)
    assert binding.account_id == OWNER.id and binding.repository_id == proposal.repository_id
    assert binding.repository_full_name == proposal.repository_full_name
    assert binding.proposal_id == proposal.proposal_id
    assert binding.investigation_run_id == proposal.investigation_run_id
    assert binding.base_commit_sha == proposal.base_commit_sha
    assert binding.tree_sha == proposal.tree_sha and binding.patch_sha256 == proposal.patch_sha256
    assert binding.review_revision == review.revision
    assert binding.decision_id == review.current_decision.decision_id
    assert binding.bound_at == instant
    assert [item.model_dump_json() for item in (proposal, review, payload)] == before
    assert ValidationBinding.model_validate_json(binding.model_dump_json()) == binding
    data = binding.model_dump()
    assert not {"changes", "unified_diff", "comment", "command", "status", "passed"} & data.keys()


@pytest.mark.parametrize("name,value", [
    ("command", "pytest"), ("image", "arbitrary:latest"), ("network", "host"),
    ("mounts", ["/:/host"]), ("environment", {"KEY": "PRIVATE"}),
    ("account_id", 999), ("repository_id", 5), ("tree_sha", "d" * 40),
    ("profile", "unrestricted"), ("timeout_seconds", 9999), ("publish", True),
])
def test_request_rejects_execution_controls_and_server_fields(approved, name, value):
    _, _, payload = approved
    with pytest.raises(ValidationError):
        ValidationInput(**{**payload.model_dump(), name: value})


@pytest.mark.parametrize("field", [
    "proposal_id", "base_commit_sha", "patch_sha256", "expected_revision", "decision_id",
])
def test_no_required_binding_value_is_defaulted(approved, field):
    data = approved[2].model_dump()
    del data[field]
    with pytest.raises(ValidationError):
        ValidationInput(**data)


@pytest.mark.parametrize("field,value", [
    ("expected_revision", 0), ("expected_revision", -1), ("expected_revision", True),
    ("expected_revision", "1"), ("expected_revision", 2**63),
    ("proposal_id", "A" * 32), ("proposal_id", "a" * 32 + "\n"),
    ("base_commit_sha", "A" * 40), ("base_commit_sha", "a" * 40 + "\n"),
    ("patch_sha256", "a" * 63), ("decision_id", "a" * 31), ("decision_id", 123),
])
def test_input_identifiers_and_revision_are_strict(approved, field, value):
    proposal, review, _ = approved
    with pytest.raises(ValidationError):
        request(proposal, review, **{field: value})


@pytest.mark.parametrize("field,value", [
    ("proposal_id", "d" * 32), ("base_commit_sha", "d" * 40), ("patch_sha256", "d" * 64),
])
def test_changed_requested_bytes_are_not_bound(approved, field, value):
    proposal, review, _ = approved
    with pytest.raises(ValidationRequestError) as caught:
        bind_validation_request(OWNER, proposal, review, request(proposal, review, **{field: value}))
    assert caught.value.code == "proposal_changed"


@pytest.mark.parametrize("changes", [
    {"expected_revision": 2}, {"decision_id": "d" * 32},
])
def test_request_must_name_current_revision_and_decision(approved, changes):
    proposal, review, _ = approved
    with pytest.raises(ValidationRequestError) as caught:
        bind_validation_request(OWNER, proposal, review, request(proposal, review, **changes))
    assert caught.value.code == "review_changed"


def test_unreviewed_and_rejected_snapshots_do_not_allow_binding(approved):
    proposal, review, payload = approved
    unreviewed = review.model_copy(update={"status": "unreviewed", "revision": 0, "current_decision": None})
    rejected = review.model_copy(update={
        "status": "rejected", "revision": 2,
        "current_decision": review.current_decision.model_copy(update={"decision": "reject"}),
    })
    for state in (unreviewed, rejected):
        with pytest.raises(ValidationRequestError) as caught:
            bind_validation_request(OWNER, proposal, state, payload)
        assert caught.value.code == "approval_required"


@pytest.mark.parametrize("field,value", [
    ("account_id", 999), ("repository_id", 999), ("repository_full_name", "other-user/example"),
    ("investigation_run_id", "d" * 32), ("tree_sha", "d" * 40),
])
def test_approval_provenance_must_match_account_and_proposal(approved, field, value):
    proposal, review, payload = approved
    state = review.model_copy(update={"current_decision": review.current_decision.model_copy(update={field: value})})
    with pytest.raises(ValidationRequestError) as caught:
        bind_validation_request(OWNER, proposal, state, payload)
    assert caught.value.code == "approval_mismatch"


def test_other_proposal_review_is_rejected_even_when_internally_consistent(approved):
    proposal, review, payload = approved
    other = review.model_copy(update={
        "proposal_id": "d" * 32,
        "current_decision": review.current_decision.model_copy(update={"proposal_id": "d" * 32}),
    })
    with pytest.raises(ValidationRequestError) as caught:
        bind_validation_request(OWNER, proposal, other, payload)
    assert caught.value.code == "approval_mismatch"


@pytest.mark.parametrize("changes,code", [
    ({"proposal_saved": False}, "not_saved"), ({"repository_id": True}, "invalid_proposal"),
    ({"unified_diff": "PRIVATE_FAKE_SOURCE"}, "invalid_proposal"),
    ({"model_calls": "2"}, "invalid_proposal"),
])
def test_saved_proposal_is_deeply_revalidated(approved, changes, code):
    proposal, review, payload = approved
    with pytest.raises(ValidationRequestError) as caught:
        bind_validation_request(OWNER, proposal.model_copy(update=changes), review, payload)
    assert caught.value.code == code and "PRIVATE" not in str(caught.value)


def test_nested_events_files_and_review_mutations_fail_closed(approved):
    proposal, review, payload = approved
    original = proposal.events[0]
    proposal.events[0] = replace(original, elapsed_ms="1")
    with pytest.raises(ValidationRequestError) as caught:
        bind_validation_request(OWNER, proposal, review, payload)
    assert caught.value.code == "invalid_proposal"
    proposal.events[0] = original
    changed = proposal.changes[0].model_copy(update={"proposed_content": "PRIVATE_FAKE_SOURCE"})
    damaged = proposal.model_copy(update={"changes": [changed, *proposal.changes[1:]]})
    with pytest.raises(ValidationRequestError) as caught:
        bind_validation_request(OWNER, damaged, review, payload)
    assert caught.value.code == "invalid_proposal"
    state = review.model_copy(update={"current_decision": review.current_decision.model_copy(update={"account_id": True})})
    with pytest.raises(ValidationRequestError) as caught:
        bind_validation_request(OWNER, proposal, state, payload)
    assert caught.value.code == "invalid_review"


@pytest.mark.parametrize("identity", [
    None, {"id": 123}, OWNER.model_copy(update={"id": True}),
    OWNER.model_copy(update={"id": 2**63}),
])
def test_invalid_identities_are_rejected(approved, identity):
    with pytest.raises(ValidationRequestError) as caught:
        bind_validation_request(identity, *approved)
    assert caught.value.code == "invalid_identity"


@pytest.mark.parametrize("index,code", [(0, "invalid_proposal"), (1, "invalid_review"), (2, "invalid_input")])
def test_unvalidated_dicts_are_not_trusted_server_objects(approved, index, code):
    values = list(approved)
    values[index] = values[index].model_dump()
    with pytest.raises(ValidationRequestError) as caught:
        bind_validation_request(OWNER, *values)
    assert caught.value.code == code


@pytest.mark.parametrize("now", [True, "2026-11-01T00:00:00Z", datetime(2026, 11, 1)])
def test_clock_requires_an_internal_aware_datetime(approved, now):
    with pytest.raises(ValidationRequestError) as caught:
        bind_validation_request(OWNER, *approved, now=now)
    assert caught.value.code == "invalid_time"


def test_approval_and_binding_time_order_and_utc_normalization(approved):
    proposal, review, payload = approved
    instant = review.current_decision.created_at
    with pytest.raises(ValidationRequestError) as caught:
        bind_validation_request(OWNER, *approved, now=instant - timedelta(microseconds=1))
    assert caught.value.code == "invalid_time"
    local = instant.astimezone(timezone(timedelta(hours=5, minutes=30)))
    assert bind_validation_request(OWNER, *approved, now=local).bound_at.utcoffset() == timedelta(0)
    damaged = review.model_copy(update={"current_decision": review.current_decision.model_copy(update={
        "created_at": proposal.created_at - timedelta(microseconds=1),
    })})
    with pytest.raises(ValidationRequestError) as caught:
        bind_validation_request(OWNER, proposal, damaged, payload)
    assert caught.value.code == "approval_mismatch"


def test_real_saved_review_revoke_reapprove_and_restart_rejects_old_request(tmp_path):
    path = tmp_path / "proposals.db"
    store = ProposalStore(path)
    proposal = record()
    store.save(OWNER.id, proposal)
    approved = store.record_decision(OWNER, action(proposal), expected_revision=0)
    old_request = request(proposal, approved)
    old_binding = bind_validation_request(OWNER, store.get(OWNER.id, proposal.proposal_id), approved, old_request)
    store.record_decision(OWNER, action(proposal, decision="reject"), expected_revision=1)
    with pytest.raises(ValidationRequestError) as caught:
        bind_validation_request(OWNER, proposal, store.get_review(OWNER.id, proposal.proposal_id), old_request)
    assert caught.value.code == "approval_required"
    store.record_decision(OWNER, action(proposal), expected_revision=2)
    store.close()
    reopened = ProposalStore(path)
    try:
        current = reopened.get_review(OWNER.id, proposal.proposal_id)
        assert current.revision == 3 and current.current_decision.decision_id != old_binding.decision_id
        with pytest.raises(ValidationRequestError) as caught:
            bind_validation_request(OWNER, proposal, current, old_request)
        assert caught.value.code == "review_changed"
        result = bind_validation_request(OWNER, proposal, current, request(proposal, current))
        assert result.review_revision == 3 and result.decision_id == current.current_decision.decision_id
        assert len(reopened.list_decisions(OWNER.id, proposal.proposal_id).decisions) == 3
        # Binding stores no approval capability. Admitting execution will still
        # need an atomic latest-revision check in the future service.
        assert old_binding.review_revision == 1
    finally:
        reopened.close()


def test_request_json_roundtrip_preserves_strict_revision(approved):
    payload = approved[2]
    assert ValidationInput.model_validate_json(payload.model_dump_json()) == payload
    data = payload.model_dump()
    data["expected_revision"] = True
    with pytest.raises(ValidationError):
        ValidationInput.model_validate_json(json.dumps(data))


def test_binding_cannot_be_mutated_in_place(approved):
    binding = bind_validation_request(
        OWNER, *approved, now=approved[1].current_decision.created_at,
    )
    with pytest.raises(ValidationError):
        binding.review_revision = 999
