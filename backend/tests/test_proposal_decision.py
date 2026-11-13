"""Offline human-decision contracts; no approval API or target execution."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import importlib
import importlib.util
import json

import httpx
import pytest
from pydantic import ValidationError

from app.github_oauth import GitHubIdentity
from app.investigation_runner import InvestigationEvent
from app.proposal_decision import (
    ProposalDecision, ProposalDecisionError, ProposalDecisionInput, bind_proposal_decision,
)
from test_proposal_store import record


OWNER = GitHubIdentity(id=123, login="demo-user", type="User")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Proposal decision contracts must not make network requests.")
    packages = [httpx]
    if importlib.util.find_spec("httpx2"):
        packages.append(importlib.import_module("httpx2"))
    for package in packages:
        monkeypatch.setattr(package.HTTPTransport, "handle_request", blocked)
        monkeypatch.setattr(package.AsyncHTTPTransport, "handle_async_request", blocked)


@pytest.fixture
def proposal():
    return record().model_copy(update={"events": [
        InvestigationEvent("proposal_started", 0),
        InvestigationEvent("tool_completed", 1, "read_file", True),
        InvestigationEvent("proposal_source_validated", 2),
    ]})


def payload(proposal, **changes):
    return ProposalDecisionInput(**{
        "proposal_id": proposal.proposal_id, "base_commit_sha": proposal.base_commit_sha,
        "patch_sha256": proposal.patch_sha256, "decision": "approve_for_validation", **changes,
    })


@pytest.mark.parametrize("decision", ["approve_for_validation", "reject"])
def test_review_event_binds_identity_proposal_and_exact_patch_without_changing_draft(proposal, decision):
    before = proposal.model_dump_json()
    instant = proposal.created_at + timedelta(seconds=1)
    result = bind_proposal_decision(OWNER, proposal, payload(proposal, decision=decision, comment="  Reviewed the boundary and proposed cases.  "), now=instant)
    assert result.decision == decision and result.account_id == OWNER.id
    assert result.proposal_id == proposal.proposal_id
    assert result.investigation_run_id == proposal.investigation_run_id
    assert result.repository_id == proposal.repository_id
    assert result.repository_full_name == proposal.repository_full_name
    assert result.base_commit_sha == proposal.base_commit_sha
    assert result.tree_sha == proposal.tree_sha and result.patch_sha256 == proposal.patch_sha256
    assert result.created_at == instant and len(result.decision_id) == 32
    assert result.comment == "Reviewed the boundary and proposed cases."
    assert proposal.model_dump_json() == before and proposal.status == "proposed"
    encoded = result.model_dump_json()
    assert "quantity >" not in encoded and "unified_diff" not in encoded
    assert ProposalDecision.model_validate_json(encoded) == result


@pytest.mark.parametrize("name,value", [
    ("account_id", 999), ("repository_id", 6), ("investigation_run_id", "d" * 32),
    ("tree_sha", "d" * 40), ("decision_id", "d" * 32), ("created_at", "2026-10-31T00:00:00Z"),
    ("status", "approved"), ("approved", True), ("command", "pytest"), ("scope", "github_write"),
])
def test_browser_cannot_supply_server_fields_or_extra_permissions(proposal, name, value):
    with pytest.raises(ValidationError):
        ProposalDecisionInput(**{**payload(proposal).model_dump(), name: value})


@pytest.mark.parametrize("name,value", [
    ("proposal_id", "A" * 32), ("proposal_id", "a" * 31), ("proposal_id", "a" * 32 + "\n"),
    ("proposal_id", 123), ("base_commit_sha", "A" * 40), ("base_commit_sha", "a" * 39),
    ("base_commit_sha", "a" * 40 + "\n"), ("patch_sha256", "F" * 64),
    ("patch_sha256", "a" * 63), ("patch_sha256", "a" * 64 + "\n"),
])
def test_reviewed_identifiers_are_strict_and_exact(proposal, name, value):
    with pytest.raises(ValidationError):
        payload(proposal, **{name: value})


@pytest.mark.parametrize("decision", ["approve", "applied", "publish", "", True, 1, None])
def test_decision_is_explicit_and_limited(proposal, decision):
    with pytest.raises(ValidationError):
        payload(proposal, decision=decision)


def test_no_automatic_decision_default(proposal):
    data = payload(proposal).model_dump()
    del data["decision"]
    with pytest.raises(ValidationError):
        ProposalDecisionInput(**data)


@pytest.mark.parametrize("name,value", [("proposal_id", "d" * 32), ("base_commit_sha", "d" * 40), ("patch_sha256", "d" * 64)])
@pytest.mark.parametrize("decision", ["approve_for_validation", "reject"])
def test_stale_or_other_reviewed_bytes_fail_for_both_decisions(proposal, name, value, decision):
    with pytest.raises(ProposalDecisionError) as caught:
        bind_proposal_decision(OWNER, proposal, payload(proposal, decision=decision, **{name: value}))
    assert caught.value.code == "proposal_changed"


def test_unsaved_draft_cannot_receive_a_bound_decision(proposal):
    with pytest.raises(ProposalDecisionError) as caught:
        bind_proposal_decision(OWNER, proposal.model_copy(update={"proposal_saved": False}), payload(proposal))
    assert caught.value.code == "not_saved"


@pytest.mark.parametrize("changes", [
    {"repository_id": True}, {"proposal_saved": 1}, {"status": "approved"},
    {"unified_diff": "PRIVATE_FAKE_PATCH"}, {"patch_sha256": "d" * 64},
    {"model_calls": "2"}, {"repository_stats": {"api_key": 1}},
])
def test_corrupt_mutated_proposals_are_revalidated(proposal, changes):
    with pytest.raises(ProposalDecisionError) as caught:
        bind_proposal_decision(OWNER, proposal.model_copy(update=changes), payload(proposal))
    assert caught.value.code == "invalid_proposal"
    assert "PRIVATE" not in str(caught.value)


def test_nested_event_and_draft_mutations_are_rejected(proposal):
    original = list(proposal.events)
    proposal.events[0] = replace(proposal.events[0], elapsed_ms="1")
    with pytest.raises(ProposalDecisionError, match="saved proposal"):
        bind_proposal_decision(OWNER, proposal, payload(proposal))
    proposal.events[:] = original
    proposal.draft.edits.append(proposal.draft.edits[0])
    with pytest.raises(ProposalDecisionError, match="saved proposal"):
        bind_proposal_decision(OWNER, proposal, payload(proposal))


@pytest.mark.parametrize("identity", [None, {"id": 123}, OWNER.model_copy(update={"id": True}), OWNER.model_copy(update={"id": 2**63}), OWNER.model_copy(update={"login": "../other"})])
def test_invalid_identity_cannot_bind(proposal, identity):
    with pytest.raises(ProposalDecisionError) as caught:
        bind_proposal_decision(identity, proposal, payload(proposal))
    assert caught.value.code == "invalid_identity"


@pytest.mark.parametrize("comment", ["bad\x00comment", "bad\x1fcomment", "bad\x7fcomment", "\ud800", "x" * 2001, 123])
def test_comments_are_bounded_text(proposal, comment):
    with pytest.raises(ValidationError):
        payload(proposal, comment=comment)


@pytest.mark.parametrize("comment", [None, "", " \n\t "])
def test_empty_comments_are_optional(proposal, comment):
    assert payload(proposal, comment=comment).comment is None


@pytest.mark.parametrize("now", ["2026-10-31T00:00:00Z", datetime(2026, 10, 31), True])
def test_clock_must_be_internal_aware_datetime(proposal, now):
    with pytest.raises(ProposalDecisionError) as caught:
        bind_proposal_decision(OWNER, proposal, payload(proposal), now=now)
    assert caught.value.code == "invalid_time"


def test_decision_cannot_predate_proposal_and_timezone_is_normalized(proposal):
    with pytest.raises(ProposalDecisionError) as caught:
        bind_proposal_decision(OWNER, proposal, payload(proposal), now=proposal.created_at - timedelta(microseconds=1))
    assert caught.value.code == "invalid_time"
    instant = (proposal.created_at + timedelta(seconds=1)).astimezone(timezone(timedelta(hours=5, minutes=30)))
    result = bind_proposal_decision(OWNER, proposal, payload(proposal), now=instant)
    assert result.created_at == instant and result.created_at.utcoffset() == timedelta(0)


def test_invalid_input_objects_and_mutated_decision_are_rejected(proposal):
    for invalid in [None, payload(proposal).model_dump(), payload(proposal).model_copy(update={"decision": "publish"})]:
        with pytest.raises(ProposalDecisionError) as caught:
            bind_proposal_decision(OWNER, proposal, invalid)
        assert caught.value.code == "invalid_input"
    with pytest.raises(ProposalDecisionError) as caught:
        bind_proposal_decision(OWNER, proposal.model_dump(), payload(proposal))
    assert caught.value.code == "invalid_proposal"


def test_each_bound_decision_gets_server_id_and_is_frozen(proposal):
    first = bind_proposal_decision(OWNER, proposal, payload(proposal))
    second = bind_proposal_decision(OWNER, proposal, payload(proposal, decision="reject"))
    assert first.decision_id != second.decision_id
    with pytest.raises(ValidationError):
        first.decision = "reject"
    assert "proposal_saved" not in json.loads(first.model_dump_json())
