"""Offline pinned-source packaging; source strings are never imported/executed."""

from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
import hashlib
import json
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from app.github_files import RepositoryFile
from app.github_oauth import GitHubIdentity
from app.github_tree import TreeEntry
from app.investigation_runner import InvestigationEvent
from app.proposal_decision import ProposalDecisionInput, bind_proposal_decision
from app.proposal_history import proposal_response
from app.proposal_review import ProposalReviewState
from app.proposal_validation import _git_blob_sha, validate_fix_proposal
from app.validation_request import ValidationInput
import app.validation_workspace as workspace
from test_proposal_store import make_result


OWNER = GitHubIdentity(id=123, login="demo-user", type="User")


@pytest.fixture(autouse=True)
def no_network_or_target_processes(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Workspace packaging must not use the network or launch processes.")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)


def make_bundle(*, content="def accept(quantity):\n    return quantity > 1\n", edit_path="orders_api.py", test_path="tests/test_orders_regression.py"):
    history, tree, result = make_result(content=content)
    tree.entries[0] = tree.entries[0].model_copy(update={"path": edit_path})
    original = RepositoryFile(
        repository_full_name=tree.repository_full_name, commit_sha=tree.commit_sha,
        path=edit_path, blob_sha=tree.entries[0].sha, size_bytes=len(content.encode()), content=content,
    )
    draft = result.validated.draft.model_copy(update={
        "edits": [result.validated.draft.edits[0].model_copy(update={"path": edit_path})],
        "tests": [result.validated.draft.tests[0].model_copy(update={"path": test_path})],
    })
    artifact = validate_fix_proposal(history, tree, [original], draft)
    result = replace(result, validated=artifact, events=(
        InvestigationEvent("proposal_started", 0),
        InvestigationEvent("proposal_source_validated", 1),
    ))
    proposal = proposal_response(result, total_duration_ms=3)
    decision = bind_proposal_decision(OWNER, proposal, ProposalDecisionInput(
        proposal_id=proposal.proposal_id, base_commit_sha=proposal.base_commit_sha,
        patch_sha256=proposal.patch_sha256, decision="approve_for_validation",
    ), now=proposal.created_at + timedelta(seconds=1))
    review = ProposalReviewState(
        proposal_id=proposal.proposal_id, base_commit_sha=proposal.base_commit_sha,
        patch_sha256=proposal.patch_sha256, revision=1, status="approved_for_validation",
        current_decision=decision,
    )
    payload = ValidationInput(
        proposal_id=proposal.proposal_id, base_commit_sha=proposal.base_commit_sha,
        patch_sha256=proposal.patch_sha256, expected_revision=1, decision_id=decision.decision_id,
    )
    return SimpleNamespace(proposal=proposal, tree=tree, sources=[original], review=review,
                           payload=payload, now=decision.created_at + timedelta(seconds=1))


@pytest.fixture
def bundle():
    return make_bundle()


def prepare(bundle, **changes):
    values = {"identity": OWNER, "proposal": bundle.proposal, "review": bundle.review,
              "payload": bundle.payload, "tree": bundle.tree, "source_files": bundle.sources,
              "now": bundle.now, **changes}
    return workspace.prepare_validation_workspace(**values)


def add_source(bundle, path, content="VALUE = 7\n"):
    data = content.encode()
    source = RepositoryFile(
        repository_full_name=bundle.tree.repository_full_name, commit_sha=bundle.tree.commit_sha,
        path=path, content=content, size_bytes=len(data), blob_sha=_git_blob_sha(data),
    )
    bundle.sources.append(source)
    bundle.tree.entries.append(TreeEntry(path=path, type="blob", mode="100644", sha=source.blob_sha, size=len(data)))
    return source


def files(payload):
    return {item["path"]: item["content"] for item in json.loads(payload)["files"]}


def test_original_and_patched_payloads_keep_identical_tests_and_exact_binding(bundle):
    helper = add_source(bundle, "helpers.py")
    before = (bundle.proposal.model_dump_json(), bundle.tree.model_dump_json(), [s.model_dump_json() for s in bundle.sources])
    result = prepare(bundle)
    baseline, patched = files(result.baseline_payload), files(result.patched_payload)
    assert baseline["orders_api.py"] == bundle.sources[0].content
    assert patched["orders_api.py"] == bundle.proposal.changes[0].proposed_content
    assert baseline["helpers.py"] == patched["helpers.py"] == helper.content
    test = bundle.proposal.draft.tests[0]
    assert baseline[test.path] == patched[test.path] == test.content
    assert result.test_paths == (test.path,) and result.source_file_count == 2
    assert result.source_bytes == sum(source.size_bytes for source in bundle.sources)
    assert result.baseline_bytes == sum(len(value.encode()) for value in baseline.values())
    assert result.patched_bytes == sum(len(value.encode()) for value in patched.values())
    for raw, phase in [(result.baseline_payload, "baseline"), (result.patched_payload, "patched")]:
        data = json.loads(raw)
        assert data["phase"] == phase and data["protocol_version"] == 1
        assert data["profile"] == "python-pytest-v1"
        assert data["binding"]["patch_sha256"] == bundle.proposal.patch_sha256
        assert data["binding"]["decision_id"] == bundle.review.current_decision.decision_id
        assert data["tests"] == [test.path]
        assert [item["path"] for item in data["files"]] == sorted(baseline)
        for item in data["files"]:
            assert item["sha256"] == hashlib.sha256(item["content"].encode()).hexdigest()
        assert not {"command", "environment", "image", "mounts", "comment", "verification_steps"} & data.keys()
    assert result.baseline_sha256 == hashlib.sha256(result.baseline_payload).hexdigest()
    assert result.patched_sha256 == hashlib.sha256(result.patched_payload).hexdigest()
    assert result.baseline_sha256 != result.patched_sha256
    assert before == (bundle.proposal.model_dump_json(), bundle.tree.model_dump_json(), [s.model_dump_json() for s in bundle.sources])


def test_order_is_canonical_and_returned_bytes_are_independent_of_later_mutation(bundle):
    add_source(bundle, "z/helper.py")
    first = prepare(bundle)
    bundle.tree.entries.reverse()
    bundle.sources.reverse()
    assert prepare(bundle) == first
    bundle.sources.clear()
    bundle.tree.entries.clear()
    assert files(first.baseline_payload)["z/helper.py"] == "VALUE = 7\n"
    assert first.binding.review_revision == 1
    with pytest.raises(FrozenInstanceError):
        first.baseline_payload = b"changed"


@pytest.mark.parametrize("content", [
    "def accept(quantity):\r\n    return quantity > 1\r\n",
    "def accept(quantity):\n    return quantity > 1",
    "# café 😀\ndef accept(quantity):\n    return quantity > 1\n",
])
def test_line_endings_unicode_and_missing_final_newline_are_preserved(content):
    b = make_bundle(content=content)
    result = prepare(b)
    assert files(result.baseline_payload)["orders_api.py"].encode() == content.encode()
    assert files(result.patched_payload)["orders_api.py"] == content.replace("quantity > 1", "quantity > 0")


def test_source_text_and_generated_test_text_remain_data_only():
    marker = "raise RuntimeError('PRIVATE_SOURCE_MUST_NEVER_EXECUTE')\n"
    b = make_bundle(content=marker + "def accept(quantity):\n    return quantity > 1\n")
    result = prepare(b)
    assert marker in files(result.baseline_payload)["orders_api.py"]
    assert "PRIVATE_SOURCE" not in repr(result)


def test_selection_excludes_non_python_and_blocked_paths_without_packing_their_bytes(bundle):
    for path in ["README.md", "requirements.txt", ".env", ".venv/private.py", "secrets/private.py", ".env.py"]:
        bundle.tree.entries.append(TreeEntry(path=path, type="blob", mode="100644", sha="d" * 40, size=10))
    assert workspace.select_validation_paths(bundle.tree) == ("orders_api.py",)
    result = prepare(bundle)
    assert set(files(result.baseline_payload)) == {"orders_api.py", bundle.proposal.draft.tests[0].path}
    assert b"requirements.txt" not in result.baseline_payload


@pytest.mark.parametrize("field,value", [
    ("commit_sha", "d" * 40), ("tree_sha", "d" * 40), ("repository_full_name", "other-user/example"),
])
def test_tree_provenance_cannot_change_saved_commit(bundle, field, value):
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        prepare(bundle, tree=bundle.tree.model_copy(update={field: value}))
    assert caught.value.code == "provenance_mismatch"


@pytest.mark.parametrize("field,value", [
    ("commit_sha", "d" * 40), ("repository_full_name", "other-user/example"),
    ("blob_sha", "d" * 40), ("size_bytes", 1), ("content", "PRIVATE_FAKE_SOURCE"),
    ("size_bytes", True),
])
def test_source_identity_sizes_and_actual_blob_hashes_are_checked(bundle, field, value):
    sources = [bundle.sources[0].model_copy(update={field: value})]
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        prepare(bundle, source_files=sources)
    assert caught.value.code in {"provenance_mismatch", "invalid_source"}
    assert "PRIVATE" not in str(caught.value)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "dict", "extra", "wrong_path", "iterator"])
def test_source_set_must_match_selection_exactly(bundle, kind):
    source = bundle.sources[0]
    if kind == "missing": sources = []
    elif kind == "duplicate":
        add_source(bundle, "helpers.py")
        sources = [source, source]
    elif kind == "dict": sources = [source.model_dump()]
    elif kind == "extra": sources = [source, source.model_copy(update={"path": "extra.py"})]
    elif kind == "wrong_path": sources = [source.model_copy(update={"path": "other.py"})]
    else: sources = iter([source])
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        prepare(bundle, source_files=sources)
    assert caught.value.code in {"missing_source", "invalid_source"}


@pytest.mark.parametrize("path,mode,kind", [("linked.py", "120000", "blob"), ("vendor", "160000", "commit")])
def test_symlinks_and_submodules_are_not_supported(bundle, path, mode, kind):
    bundle.tree.entries.append(TreeEntry(path=path, type=kind, mode=mode, sha="d" * 40, size=5))
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        workspace.select_validation_paths(bundle.tree)
    assert caught.value.code == "unsupported_tree"


@pytest.mark.parametrize("kind", ["duplicate", "case_alias", "file_ancestor", "parent_case_alias", "reserved_path"])
def test_ambiguous_and_nonportable_tree_paths_fail_before_fetching(bundle, kind):
    entry = bundle.tree.entries[0]
    if kind == "duplicate": bundle.tree.entries.append(entry)
    elif kind == "case_alias": bundle.tree.entries.append(entry.model_copy(update={"path": "ORDERS_API.py"}))
    elif kind == "file_ancestor": bundle.tree.entries.append(entry.model_copy(update={"path": "orders_api.py/helper.py"}))
    elif kind == "parent_case_alias":
        bundle.tree.entries.append(TreeEntry(path="Package", type="tree", mode="040000", sha="d" * 40))
        bundle.tree.entries.append(entry.model_copy(update={"path": "package/helper.py"}))
    else: bundle.tree.entries.append(entry.model_copy(update={"path": "CON.py"}))
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        workspace.select_validation_paths(bundle.tree)
    assert caught.value.code in {"ambiguous_tree", "unsupported_path"}


@pytest.mark.parametrize("entry_change", [{"size": None}, {"size": 100 * 1024 + 1}, {"size": True}])
def test_unknown_oversized_or_mutated_file_sizes_fail_before_fetching(bundle, entry_change):
    bundle.tree.entries[0] = bundle.tree.entries[0].model_copy(update=entry_change)
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        workspace.select_validation_paths(bundle.tree)
    assert caught.value.code in {"source_limit", "invalid_tree"}


def test_source_file_count_and_aggregate_bytes_are_bounded_before_fetching(bundle):
    entry = bundle.tree.entries[0]
    bundle.tree.entries.extend(entry.model_copy(update={"path": f"file_{i}.py"}) for i in range(workspace.MAX_WORKSPACE_SOURCE_FILES))
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        workspace.select_validation_paths(bundle.tree)
    assert caught.value.code == "source_limit"
    bundle.tree.entries[:] = [entry.model_copy(update={"path": f"file_{i}.py", "size": 100 * 1024}) for i in range(6)]
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        workspace.select_validation_paths(bundle.tree)
    assert caught.value.code == "workspace_limit"


def test_final_phase_and_serialized_payload_budgets_include_proposed_tests(bundle, monkeypatch):
    before = prepare(bundle)
    monkeypatch.setattr(workspace, "MAX_WORKSPACE_BYTES", before.source_bytes)
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        prepare(bundle)
    assert caught.value.code == "workspace_limit"
    monkeypatch.setattr(workspace, "MAX_WORKSPACE_BYTES", 512 * 1024)
    monkeypatch.setattr(workspace, "MAX_WORKSPACE_PAYLOAD_BYTES", len(before.baseline_payload) - 1)
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        prepare(bundle)
    assert caught.value.code == "payload_limit"


@pytest.mark.parametrize("kwargs", [{"edit_path": "orders_api.txt"}, {"test_path": "tests/checks.py"}, {"test_path": "tests/test_orders.js"}])
def test_broader_valid_proposal_targets_can_be_unsupported_by_python_profile(kwargs):
    bundle = make_bundle(**kwargs)
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        prepare(bundle)
    assert caught.value.code in {"source_limit", "unsupported_target"}


def test_new_test_cannot_replace_existing_unfetched_file_or_case_alias(bundle):
    test = bundle.proposal.draft.tests[0]
    bundle.tree.entries.append(TreeEntry(path=test.path.upper(), type="blob", mode="100644", sha="d" * 40, size=1))
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        prepare(bundle)
    assert caught.value.code == "proposal_source_mismatch"


@pytest.mark.parametrize("change", ["content", "mode"])
def test_saved_anchor_source_and_mode_must_match_new_pinned_source(bundle, change):
    source = bundle.sources[0]
    if change == "content":
        content = "# different source but individually valid\n" + source.content
        data = content.encode()
        source = source.model_copy(update={"content": content, "size_bytes": len(data), "blob_sha": _git_blob_sha(data)})
        bundle.tree.entries[0] = bundle.tree.entries[0].model_copy(update={"sha": source.blob_sha, "size": source.size_bytes})
    else:
        bundle.tree.entries[0] = bundle.tree.entries[0].model_copy(update={"mode": "100644"})
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        prepare(bundle, source_files=[source])
    assert caught.value.code == "proposal_source_mismatch"


@pytest.mark.parametrize("kind", ["rejected", "stale_revision", "stale_decision", "other_account", "unsaved", "corrupt_events"])
def test_approval_and_saved_record_guards_survive_packaging_boundary(bundle, kind):
    changes = {}
    if kind == "rejected": changes["review"] = bundle.review.model_copy(update={"status": "rejected", "current_decision": bundle.review.current_decision.model_copy(update={"decision": "reject"})})
    elif kind == "stale_revision": changes["payload"] = bundle.payload.model_copy(update={"expected_revision": 2})
    elif kind == "stale_decision": changes["payload"] = bundle.payload.model_copy(update={"decision_id": "d" * 32})
    elif kind == "other_account": changes["identity"] = OWNER.model_copy(update={"id": 999})
    elif kind == "unsaved": changes["proposal"] = bundle.proposal.model_copy(update={"proposal_saved": False})
    else: bundle.proposal.events[0] = replace(bundle.proposal.events[0], elapsed_ms="1")
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        prepare(bundle, **changes)
    assert caught.value.code in {"approval_required", "review_changed", "approval_mismatch", "not_saved", "invalid_proposal"}


def test_unchanged_non_target_content_is_also_checked_for_binary_controls(bundle):
    add_source(bundle, "helper.py", "PRIVATE\x00DATA")
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        prepare(bundle)
    assert caught.value.code == "invalid_source" and "PRIVATE" not in str(caught.value)


def test_tree_and_source_objects_are_required_not_unvalidated_dicts(bundle):
    with pytest.raises(workspace.ValidationWorkspaceError) as caught:
        prepare(bundle, tree=bundle.tree.model_dump())
    assert caught.value.code == "invalid_tree"
