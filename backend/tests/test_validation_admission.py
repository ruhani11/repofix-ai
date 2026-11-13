"""Atomic admission/lifecycle with real SQLite; no target/container execution."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import sqlite3
import subprocess
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from app.github_oauth import GitHubIdentity
from app.proposal_store import ProposalStore, ProposalStoreError
from app.validation_admission import ValidationAdmission, ValidationAdmissionError, ValidationAdmissionStore
from app.validation_request import ValidationInput
from app.validation_workspace import prepare_validation_workspace
from test_proposal_review_store import OWNER, decision
from test_validation_workspace import make_bundle


IMAGE_ID = "sha256:" + "a" * 64


@pytest.fixture(autouse=True)
def no_network_or_processes(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Admission must not use network, Docker or generated code.")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)


@pytest.fixture
def saved(tmp_path):
    path = tmp_path / "proposals.db"
    store = ProposalStore(path)
    source = make_bundle()
    store.save(OWNER.id, source.proposal)
    store.record_decision(OWNER, decision(source.proposal), expected_revision=0)
    item = SimpleNamespace(path=path, store=store, source=source, admissions=ValidationAdmissionStore(store))
    refresh(item)
    yield item
    store.close()


def refresh(saved):
    b = saved.source
    review = saved.store.get_review(OWNER.id, b.proposal.proposal_id)
    saved.payload = ValidationInput(
        proposal_id=b.proposal.proposal_id, base_commit_sha=b.proposal.base_commit_sha,
        patch_sha256=b.proposal.patch_sha256, expected_revision=review.revision,
        decision_id=review.current_decision.decision_id,
    )
    saved.workspace = prepare_validation_workspace(OWNER, b.proposal, review, saved.payload, b.tree, b.sources)


def admit(saved, **changes):
    return saved.admissions.admit(**{
        "identity": OWNER, "payload": saved.payload, "workspace": saved.workspace,
        "image_id": IMAGE_ID, "authorize": lambda: None, **changes,
    })


def test_admit_start_cleanup_close_and_restart_keep_immutable_proposal_and_review(saved):
    before = saved.store.get(OWNER.id, saved.payload.proposal_id).model_dump_json()
    row = admit(saved)
    assert row.state == "admitted" and row.started_at is None and row.closed_at is None
    assert row.binding == saved.workspace.binding and row.image_id == IMAGE_ID
    assert row.baseline_sha256 == saved.workspace.baseline_sha256
    assert row.patched_sha256 == saved.workspace.patched_sha256
    assert ValidationAdmission.model_validate_json(row.model_dump_json()) == row
    assert not {"source", "changes", "comment", "command", "passed", "outcome"} & row.model_dump().keys()
    running = saved.admissions.start(OWNER, row.validation_id, authorize=lambda: None)
    assert running.state == "running" and running.started_at >= row.created_at
    closed = saved.admissions.close(OWNER.id, row.validation_id)
    assert closed.state == "closed" and closed.closed_at >= running.started_at
    assert saved.admissions.close(OWNER.id, row.validation_id) == closed
    saved.store.close()
    reopened = ProposalStore(saved.path)
    try:
        assert ValidationAdmissionStore(reopened).get(OWNER.id, row.validation_id) == closed
        assert reopened.get(OWNER.id, saved.payload.proposal_id).model_dump_json() == before
        assert reopened.get_review(OWNER.id, saved.payload.proposal_id).revision == 1
    finally:
        reopened.close()


def test_restart_preserves_active_slot_without_expiry_or_automatic_resume(saved):
    row = admit(saved)
    saved.store.close()
    reopened = ProposalStore(saved.path)
    try:
        service = ValidationAdmissionStore(reopened)
        assert service.get(OWNER.id, row.validation_id).state == "admitted"
        with pytest.raises(ValidationAdmissionError) as caught:
            service.admit(OWNER, saved.payload, saved.workspace, IMAGE_ID, authorize=lambda: None)
        assert caught.value.code == "busy"
        service.close(OWNER.id, row.validation_id)
        assert service.admit(OWNER, saved.payload, saved.workspace, IMAGE_ID, authorize=lambda: None).validation_id != row.validation_id
    finally:
        reopened.close()


def test_active_slot_is_per_account_even_across_different_proposals(saved):
    first = admit(saved)
    b = make_bundle()
    saved.store.save(OWNER.id, b.proposal)
    review = saved.store.record_decision(OWNER, decision(b.proposal), expected_revision=0)
    payload = ValidationInput(proposal_id=b.proposal.proposal_id, base_commit_sha=b.proposal.base_commit_sha,
                              patch_sha256=b.proposal.patch_sha256, expected_revision=1,
                              decision_id=review.current_decision.decision_id)
    workspace = prepare_validation_workspace(OWNER, b.proposal, review, payload, b.tree, b.sources)
    with pytest.raises(ValidationAdmissionError) as caught:
        admit(saved, payload=payload, workspace=workspace)
    assert caught.value.code == "busy"
    saved.admissions.close(OWNER.id, first.validation_id)
    assert admit(saved, payload=payload, workspace=workspace).binding.proposal_id == b.proposal.proposal_id


def test_rejection_after_start_does_not_implicitly_cancel_or_close_active_work(saved):
    row = admit(saved)
    saved.admissions.start(OWNER, row.validation_id, authorize=lambda: None)
    saved.store.record_decision(OWNER, decision(saved.source.proposal, decision="reject"), expected_revision=1)
    assert saved.admissions.get(OWNER.id, row.validation_id).state == "running"
    assert saved.admissions.close(OWNER.id, row.validation_id).state == "closed"
    with pytest.raises(ValidationAdmissionError) as caught: admit(saved)
    assert caught.value.code == "approval_required"


@pytest.mark.parametrize("stage", ["admit", "start"])
@pytest.mark.parametrize("reapprove", [False, True])
def test_revocation_and_changed_revision_prevent_reservation_or_start(saved, stage, reapprove):
    row = admit(saved) if stage == "start" else None
    saved.store.record_decision(OWNER, decision(saved.source.proposal, decision="reject"), expected_revision=1)
    if reapprove:
        saved.store.record_decision(OWNER, decision(saved.source.proposal), expected_revision=2)
    with pytest.raises(ValidationAdmissionError) as caught:
        if row: saved.admissions.start(OWNER, row.validation_id, authorize=lambda: None)
        else: admit(saved)
    assert caught.value.code == ("review_changed" if reapprove else "approval_required")
    if row:
        assert saved.admissions.get(OWNER.id, row.validation_id).state == "admitted"
        saved.admissions.close(OWNER.id, row.validation_id)


def test_old_workspace_cannot_be_rebound_to_fresh_approval_without_preparation(saved):
    old = saved.workspace
    saved.store.record_decision(OWNER, decision(saved.source.proposal, decision="reject"), expected_revision=1)
    saved.store.record_decision(OWNER, decision(saved.source.proposal), expected_revision=2)
    refresh(saved)
    with pytest.raises(ValidationAdmissionError) as caught:
        admit(saved, workspace=old)
    assert caught.value.code == "workspace_changed"
    assert admit(saved).binding.review_revision == 3


@pytest.mark.parametrize("method", ["get", "start", "close"])
def test_foreign_and_missing_admission_ids_are_indistinguishable(saved, method):
    row = admit(saved)
    errors = []
    for identity, identifier in [(GitHubIdentity(id=999, login="other-user", type="User"), row.validation_id), (OWNER, "d" * 32)]:
        with pytest.raises(ValidationAdmissionError) as caught:
            if method == "get": saved.admissions.get(identity.id, identifier)
            elif method == "close": saved.admissions.close(identity.id, identifier)
            else: saved.admissions.start(identity, identifier, authorize=lambda: None)
        errors.append(caught.value.code)
    assert errors == ["not_found", "not_found"]


@pytest.mark.parametrize("phase", ["admit", "start"])
def test_session_failure_rolls_back_without_claiming_reservation_or_start(saved, phase):
    row = admit(saved) if phase == "start" else None
    authorize = Mock(side_effect=RuntimeError("PRIVATE_SESSION_REVOKED"))
    with pytest.raises(RuntimeError):
        if row: saved.admissions.start(OWNER, row.validation_id, authorize=authorize)
        else: admit(saved, authorize=authorize)
    authorize.assert_called_once()
    if row: assert saved.admissions.get(OWNER.id, row.validation_id).state == "admitted"
    else:
        with sqlite3.connect(saved.path) as connection:
            assert connection.execute("SELECT count(*) FROM fix_validation_admissions").fetchone()[0] == 0


@pytest.mark.parametrize("phase", ["admit", "start", "close"])
def test_database_failure_does_not_claim_a_saved_transition(saved, phase):
    row = admit(saved) if phase != "admit" else None
    if phase == "close": saved.admissions.start(OWNER, row.validation_id, authorize=lambda: None)
    operation = "INSERT" if phase == "admit" else "UPDATE"
    with sqlite3.connect(saved.path) as connection:
        connection.executescript(f"CREATE TRIGGER deny_admission BEFORE {operation} ON fix_validation_admissions BEGIN SELECT RAISE(ABORT, 'PRIVATE_DISK_ERROR'); END;")
    with pytest.raises(ValidationAdmissionError) as caught:
        if phase == "admit": admit(saved)
        elif phase == "start": saved.admissions.start(OWNER, row.validation_id, authorize=lambda: None)
        else: saved.admissions.close(OWNER.id, row.validation_id)
    assert caught.value.code == "unavailable" and "PRIVATE" not in str(caught.value)
    with sqlite3.connect(saved.path) as connection:
        connection.execute("DROP TRIGGER deny_admission")
    if row: assert saved.admissions.get(OWNER.id, row.validation_id).state == ("admitted" if phase == "start" else "running")
    else: assert admit(saved).state == "admitted"


@pytest.mark.parametrize("phase", ["admit", "start"])
def test_independent_connections_cannot_duplicate_an_active_job_or_start(saved, phase):
    row = admit(saved) if phase == "start" else None
    second = ProposalStore(saved.path)
    barrier = Barrier(2)
    def attempt(service):
        barrier.wait(timeout=5)
        try:
            if row: return service.start(OWNER, row.validation_id, authorize=lambda: None).state
            return service.admit(OWNER, saved.payload, saved.workspace, IMAGE_ID, authorize=lambda: None).state
        except ValidationAdmissionError as error:
            return error.code
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, [saved.admissions, ValidationAdmissionStore(second)]))
        assert sorted(results) == (["already_started", "running"] if row else ["admitted", "busy"])
    finally:
        second.close()


@pytest.mark.parametrize("change", [
    {"baseline_sha256": "d" * 64}, {"patched_payload": b"PRIVATE_INVALID_JSON"},
    {"source_file_count": True}, {"source_bytes": 0}, {"test_paths": ("different.py",)},
])
def test_mutated_workspace_metadata_or_bytes_are_rejected_before_persistence(saved, change):
    with pytest.raises(ValidationAdmissionError) as caught:
        admit(saved, workspace=replace(saved.workspace, **change))
    assert caught.value.code == "invalid_workspace"


def test_rehashed_unreviewed_target_change_is_not_admitted(saved):
    data = json.loads(saved.workspace.baseline_payload)
    target = next(item for item in data["files"] if item["path"] == "orders_api.py")
    target["content"] = target["content"].replace("quantity > 1", "quantity > 9")
    target["sha256"] = hashlib.sha256(target["content"].encode()).hexdigest()
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    changed = replace(saved.workspace, baseline_payload=encoded, baseline_sha256=hashlib.sha256(encoded).hexdigest())
    with pytest.raises(ValidationAdmissionError) as caught:
        admit(saved, workspace=changed)
    assert caught.value.code == "workspace_changed"


@pytest.mark.parametrize("field,value", [
    ("account_id", 999), ("patch_sha256", "d" * 64), ("review_revision", 2),
    ("repository_id", 999), ("tree_sha", "d" * 40), ("investigation_run_id", "d" * 32),
    ("repository_full_name", "other-user/example"),
])
def test_corrupted_admission_record_fails_closed(saved, field, value):
    row = admit(saved)
    data = row.model_dump(mode="json")
    data["binding"][field] = value
    with sqlite3.connect(saved.path) as connection:
        connection.execute("UPDATE fix_validation_admissions SET detail_json=? WHERE validation_id=?", (json.dumps(data), row.validation_id))
    # Account cross-reference and stale/corrupt binding are rejected at read or start.
    with pytest.raises(ValidationAdmissionError) as caught:
        saved.admissions.start(OWNER, row.validation_id, authorize=lambda: None)
    assert caught.value.code in {"unavailable", "proposal_changed", "review_changed"}


def test_v2_migration_preserves_original_proposal_and_decisions(saved):
    before = saved.store.get(OWNER.id, saved.payload.proposal_id).model_dump_json()
    review = saved.store.get_review(OWNER.id, saved.payload.proposal_id)
    saved.store.close()
    with sqlite3.connect(saved.path) as connection:
        connection.execute("DROP TABLE fix_validation_admissions")
        connection.execute("PRAGMA user_version=2")
    assert saved.store.get(OWNER.id, saved.payload.proposal_id).model_dump_json() == before
    assert saved.store.get_review(OWNER.id, saved.payload.proposal_id) == review
    assert admit(saved).state == "admitted"
    with sqlite3.connect(saved.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4


def test_failed_migration_rolls_back_version_and_keeps_saved_bytes(saved):
    saved.store.close()
    with sqlite3.connect(saved.path) as connection:
        before = connection.execute("SELECT detail_json FROM fix_proposals").fetchone()[0]
        connection.executescript("DROP TABLE fix_validation_admissions; CREATE TABLE fix_validation_admissions (wrong_column TEXT); PRAGMA user_version=2;")
    with pytest.raises(ProposalStoreError): saved.store.ensure_ready()
    with sqlite3.connect(saved.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("SELECT detail_json FROM fix_proposals").fetchone()[0] == before


@pytest.mark.parametrize("account,identifier", [(True, "a" * 32), (0, "a" * 32), (123, "A" * 32), (123, "a" * 32 + "\n")])
def test_invalid_keys_fail_before_disk(tmp_path, account, identifier):
    path = tmp_path / "missing.db"
    store = ProposalStore(path)
    service = ValidationAdmissionStore(store)
    with pytest.raises(ValidationAdmissionError) as caught: service.get(account, identifier)
    assert caught.value.code == "invalid_input" and not path.exists()


@pytest.mark.parametrize("image_id", [None, "image:latest", "sha256:" + "A" * 64, IMAGE_ID + "\n"])
def test_mutable_or_invalid_image_references_never_enter_admission_transaction(saved, image_id, monkeypatch):
    transaction = Mock(wraps=saved.store._validation_transaction)
    monkeypatch.setattr(saved.store, "_validation_transaction", transaction)
    with pytest.raises(ValidationAdmissionError) as caught: admit(saved, image_id=image_id)
    assert caught.value.code == "invalid_image"
    transaction.assert_not_called()


@pytest.mark.parametrize("identity", [None, OWNER.model_copy(update={"id": True}), OWNER.model_copy(update={"id": 2**63})])
def test_invalid_identity_fails_before_admission_transaction(saved, identity, monkeypatch):
    transaction = Mock(wraps=saved.store._validation_transaction)
    monkeypatch.setattr(saved.store, "_validation_transaction", transaction)
    with pytest.raises(ValidationAdmissionError) as caught: admit(saved, identity=identity)
    assert caught.value.code in {"invalid_identity", "invalid_input"}
    transaction.assert_not_called()
