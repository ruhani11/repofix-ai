"""Read-only validation preparation with real SQLite and mocked GitHub HTTP."""

import base64
from dataclasses import replace
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from app.github_oauth import GitHubIdentity
from app.github_tree import TreeEntry
from app.investigation_input import InvestigationInput
from app.investigation_runner import InvestigationEvent
from app.investigation_store import InvestigationStore, InvestigationStoreError
from app.proposal_decision import ProposalDecisionInput
from app.proposal_history import proposal_response
from app.proposal_review import ProposalReviewError
from app.proposal_store import ProposalStore, ProposalStoreError
from app.proposal_validation import _git_blob_sha
import app.validation_context as preparation
from app.validation_context import ValidationContextError, prepare_validation_context
from app.validation_request import ValidationInput
from test_proposal_context import metadata
from test_proposal_store import make_result


OWNER = GitHubIdentity(id=123, login="demo-user", type="User")
PREFIX = "/repos/demo-user/example"


@pytest.fixture(autouse=True)
def no_live_network_or_target_processes(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("Validation context must not use real network or target processes.")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", reject)
    monkeypatch.setattr(subprocess, "Popen", reject)


def action(proposal, decision="approve_for_validation"):
    return ProposalDecisionInput(
        proposal_id=proposal.proposal_id, base_commit_sha=proposal.base_commit_sha,
        patch_sha256=proposal.patch_sha256, decision=decision,
    )


@pytest.fixture
def saved(tmp_path):
    history, tree, result = make_result(content="def accept(quantity):\r\n    return quantity > 1\r\n")
    result = replace(result, events=(InvestigationEvent("proposal_started", 0), InvestigationEvent("proposal_source_validated", 1)))
    proposal = proposal_response(result, total_duration_ms=3)
    histories = InvestigationStore(tmp_path / "history.db")
    proposals = ProposalStore(tmp_path / "proposals.db")
    histories.save(OWNER.id, InvestigationInput(
        repository_id=history.repository_id, repository_name=history.repository_name,
        issue_description=history.issue_description,
    ), history.investigation.model_copy(update={"events": [
        InvestigationEvent("run_started", 0), InvestigationEvent("grounding_validated", 1),
    ]}))
    proposals.save(OWNER.id, proposal)
    review = proposals.record_decision(OWNER, action(proposal), expected_revision=0)
    payload = ValidationInput(
        proposal_id=proposal.proposal_id, base_commit_sha=proposal.base_commit_sha,
        patch_sha256=proposal.patch_sha256, expected_revision=review.revision,
        decision_id=review.current_decision.decision_id,
    )
    content = proposal.changes[0].original_content.encode()
    bundle = SimpleNamespace(histories=histories, proposals=proposals, proposal=proposal, tree=tree,
                             payload=payload, blobs={tree.entries[0].sha: content})
    yield bundle
    proposals.close()
    histories.close()


def transport_for(saved):
    paths = []
    state = {"metadata_count": 0, "metadata_changes": {}, "on_tree": None, "on_blob": None,
             "blob_damage": None, "status_at": None, "truncated": False, "tree_sha": saved.tree.tree_sha}

    def handler(request):
        assert request.method == "GET" and request.url.host == "api.github.com"
        assert "authorization" not in request.headers and "cookie" not in request.headers
        path = request.url.path
        paths.append(path)
        if state["status_at"] == path:
            return httpx.Response(503, json={"message": "PRIVATE_UPSTREAM_DETAIL"})
        if path == PREFIX:
            state["metadata_count"] += 1
            body = metadata(**state["metadata_changes"].get(state["metadata_count"], {}))
        elif path == f"{PREFIX}/git/commits/{saved.tree.commit_sha}":
            body = {"sha": saved.tree.commit_sha, "tree": {"sha": saved.tree.tree_sha}}
        elif path == f"{PREFIX}/git/trees/{saved.tree.tree_sha}":
            assert request.url.params["recursive"] == "1"
            body = {"sha": state["tree_sha"], "truncated": state["truncated"],
                    "tree": [entry.model_dump() for entry in saved.tree.entries]}
            if state["on_tree"]: state["on_tree"]()
        elif path.startswith(f"{PREFIX}/git/blobs/"):
            sha = path.rsplit("/", 1)[1]
            data = saved.blobs[sha]
            body = {"sha": sha, "size": len(data), "encoding": "base64",
                    "content": base64.b64encode(data).decode()}
            if state["blob_damage"]: body = {**body, **state["blob_damage"]}
            if state["on_blob"]: state["on_blob"]()
        else:
            pytest.fail(f"Unexpected mock path, including any HEAD/ref read: {path}")
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler), paths, state


def prepare(saved, transport, **changes):
    return prepare_validation_context(**{
        "identity": OWNER, "payload": saved.payload, "proposal_store": saved.proposals,
        "history_store": saved.histories, "transport": transport, **changes,
    })


def assert_error(saved, transport, expected, **changes):
    with pytest.raises(ValidationContextError) as caught:
        prepare(saved, transport, **changes)
    assert caught.value.code == expected
    assert "PRIVATE" not in str(caught.value)


def test_saved_approved_context_fetches_only_original_commit_and_verified_blobs(saved):
    transport, paths, _ = transport_for(saved)
    before = saved.proposals.get(OWNER.id, saved.proposal.proposal_id).model_dump_json()
    result = prepare(saved, transport)
    assert result.file_fetch_attempts == 1 and result.downloaded_bytes == len(next(iter(saved.blobs.values())))
    assert result.preparation_duration_ms >= 0 and result.workspace.binding.review_revision == 1
    assert result.workspace.binding.decision_id == saved.payload.decision_id
    assert result.workspace.binding.base_commit_sha == saved.proposal.base_commit_sha
    expected = [PREFIX, f"{PREFIX}/git/commits/{saved.tree.commit_sha}",
                f"{PREFIX}/git/trees/{saved.tree.tree_sha}", PREFIX,
                f"{PREFIX}/git/blobs/{saved.tree.entries[0].sha}", PREFIX]
    assert paths == expected
    baseline = json.loads(result.workspace.baseline_payload)
    source = next(item for item in baseline["files"] if item["path"] == "orders_api.py")
    assert source["content"].encode() == next(iter(saved.blobs.values()))
    assert "\r\n" in source["content"]
    assert saved.proposals.get(OWNER.id, saved.proposal.proposal_id).model_dump_json() == before
    assert len(saved.proposals.list_decisions(OWNER.id, saved.proposal.proposal_id).decisions) == 1


@pytest.mark.parametrize("kind", ["missing", "foreign"])
def test_missing_and_other_account_proposals_are_indistinguishable_before_github(saved, kind):
    transport, paths, _ = transport_for(saved)
    changes = {"payload": saved.payload.model_copy(update={"proposal_id": "d" * 32})} if kind == "missing" else {
        "identity": GitHubIdentity(id=999, login="other-user", type="User")}
    assert_error(saved, transport, "not_found", **changes)
    assert paths == []


@pytest.mark.parametrize("field,value,code", [
    ("base_commit_sha", "d" * 40, "proposal_changed"),
    ("patch_sha256", "d" * 64, "proposal_changed"),
    ("expected_revision", 2, "review_changed"), ("decision_id", "d" * 32, "review_changed"),
])
def test_exact_reviewed_request_is_checked_before_github(saved, field, value, code):
    transport, paths, _ = transport_for(saved)
    assert_error(saved, transport, code, payload=saved.payload.model_copy(update={field: value}))
    assert paths == []


def test_rejection_blocks_before_any_github_reads(saved):
    saved.proposals.record_decision(OWNER, action(saved.proposal, "reject"), expected_revision=1)
    transport, paths, _ = transport_for(saved)
    assert_error(saved, transport, "approval_required")
    assert paths == []


def test_unreviewed_saved_proposal_cannot_start_remote_preparation(saved):
    store = ProposalStore(":memory:")
    try:
        store.save(OWNER.id, saved.proposal)
        transport, paths, _ = transport_for(saved)
        assert_error(saved, transport, "approval_required", proposal_store=store)
        assert paths == []
    finally:
        store.close()


def test_supporting_python_files_are_fetched_and_non_python_or_blocked_paths_are_not(saved):
    data = b"VALUE = 7\n"
    sha = _git_blob_sha(data)
    saved.tree.entries.append(TreeEntry(path="helpers.py", type="blob", mode="100644", sha=sha, size=len(data)))
    saved.blobs[sha] = data
    for path in ["README.md", "requirements.txt", ".env", "secrets/private.py"]:
        saved.tree.entries.append(TreeEntry(path=path, type="blob", mode="100644", sha="d" * 40, size=10))
    transport, paths, _ = transport_for(saved)
    result = prepare(saved, transport)
    assert result.file_fetch_attempts == 2 and result.downloaded_bytes == sum(map(len, saved.blobs.values()))
    assert result.workspace.source_file_count == 2
    assert f"{PREFIX}/git/blobs/{sha}" in paths
    assert f"{PREFIX}/git/blobs/{'d' * 40}" not in paths


@pytest.mark.parametrize("when,reapprove", [("tree", False), ("blob", False), ("blob", True)])
def test_revocation_or_reapproval_during_reads_invalidates_the_old_request(saved, when, reapprove):
    transport, paths, state = transport_for(saved)
    def change_review():
        saved.proposals.record_decision(OWNER, action(saved.proposal, "reject"), expected_revision=1)
        if reapprove:
            saved.proposals.record_decision(OWNER, action(saved.proposal), expected_revision=2)
    state["on_tree" if when == "tree" else "on_blob"] = change_review
    assert_error(saved, transport, "review_changed" if reapprove else "approval_required")
    assert sum("/git/blobs/" in path for path in paths) == (0 if when == "tree" else 1)


@pytest.mark.parametrize("when", [1, 2, 3])
def test_current_repository_ownership_is_checked_before_after_tree_and_after_blobs(saved, when):
    transport, paths, state = transport_for(saved)
    state["metadata_changes"][when] = {"owner": {**OWNER.model_dump(), "id": 999}}
    assert_error(saved, transport, "selection_changed")
    assert sum("/git/blobs/" in path for path in paths) == (1 if when == 3 else 0)


def test_default_branch_changes_do_not_change_the_pinned_commit(saved):
    transport, paths, state = transport_for(saved)
    state["metadata_changes"] = {i: {"default_branch": "moved-branch"} for i in [1, 2, 3]}
    assert prepare(saved, transport).workspace.binding.base_commit_sha == saved.proposal.base_commit_sha
    assert not any("moved-branch" in path or "/refs/" in path or "/heads/" in path for path in paths)


@pytest.mark.parametrize("damage", [
    {"sha": "d" * 40}, {"size": 1}, {"content": base64.b64encode(b"PRIVATE_FAKE_SOURCE").decode()},
])
def test_bad_blob_identity_size_or_content_never_reaches_a_workspace(saved, damage):
    transport, _, state = transport_for(saved)
    state["blob_damage"] = damage
    assert_error(saved, transport, "github_unavailable")


@pytest.mark.parametrize("kind", ["truncated", "wrong_sha", "too_many_python_files", "oversized_python_file"])
def test_invalid_or_unsupported_tree_stops_before_blob_downloads(saved, kind):
    transport, paths, state = transport_for(saved)
    code = "github_unavailable"
    if kind == "truncated": state["truncated"] = True
    elif kind == "wrong_sha": state["tree_sha"] = "d" * 40
    elif kind == "too_many_python_files":
        saved.tree.entries.extend(saved.tree.entries[0].model_copy(update={"path": f"helper_{i}.py"}) for i in range(20))
        code = "source_limit"
    else:
        saved.tree.entries[0] = saved.tree.entries[0].model_copy(update={"size": 100 * 1024 + 1})
        code = "source_limit"
    assert_error(saved, transport, code)
    assert not any("/git/blobs/" in path for path in paths)


@pytest.mark.parametrize("stage", ["proposal", "review", "history"])
def test_storage_failures_are_safe_and_stop_before_github(saved, stage, monkeypatch):
    target = saved.proposals if stage != "history" else saved.histories
    method = "get_review" if stage == "review" else "get"
    error = ProposalStoreError() if stage == "proposal" else ProposalReviewError("unavailable") if stage == "review" else InvestigationStoreError("PRIVATE_DB")
    monkeypatch.setattr(target, method, Mock(side_effect=error))
    transport, paths, _ = transport_for(saved)
    assert_error(saved, transport, {"proposal": "proposal_unavailable", "review": "review_unavailable", "history": "history_unavailable"}[stage])
    assert paths == []


@pytest.mark.parametrize("kind", ["identity", "payload", "store", "corrupt_proposal", "missing_history"])
def test_invalid_untrusted_objects_and_missing_context_do_not_start_remote_reads(saved, kind, monkeypatch):
    changes = {}
    code = "invalid_input"
    if kind == "identity": changes["identity"] = OWNER.model_copy(update={"id": True}); code = "invalid_identity"
    elif kind == "payload": changes["payload"] = saved.payload.model_dump()
    elif kind == "store": changes["proposal_store"] = object(); code = "invalid_store"
    elif kind == "corrupt_proposal":
        monkeypatch.setattr(saved.proposals, "get", Mock(return_value=saved.proposal.model_dump()))
        code = "invalid_proposal"
    else:
        monkeypatch.setattr(saved.histories, "get", Mock(return_value=None)); code = "not_found"
    transport, paths, _ = transport_for(saved)
    assert_error(saved, transport, code, **changes)
    assert paths == []


def test_preparation_deadline_expires_after_a_completed_blob_without_running_work(saved, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(preparation, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    transport, paths, state = transport_for(saved)
    state["on_blob"] = lambda: clock.__setitem__(0, 61.0)
    assert_error(saved, transport, "timeout")
    assert state["metadata_count"] == 2


def test_upstream_failure_is_safe_and_preparation_does_not_append_a_review(saved):
    transport, _, state = transport_for(saved)
    state["status_at"] = f"{PREFIX}/git/commits/{saved.tree.commit_sha}"
    assert_error(saved, transport, "github_unavailable")
    assert saved.proposals.get_review(OWNER.id, saved.proposal.proposal_id).revision == 1
