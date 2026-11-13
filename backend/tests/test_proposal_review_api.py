"""Human review HTTP contracts, real SQLite, no AI or target operations."""

import importlib
import importlib.util
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.proposal_history import proposal_response
from app.proposal_review import ProposalReviewError
from app.main import create_app
from test_proposal_api import BASE, HEADERS, OWNER, OTHER, login, setup_app


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("Human review must not make a network request.")
    packages = [httpx]
    if importlib.util.find_spec("httpx2"):
        packages.append(importlib.import_module("httpx2"))
    for package in packages:
        monkeypatch.setattr(package.HTTPTransport, "handle_request", reject)
        monkeypatch.setattr(package.AsyncHTTPTransport, "handle_async_request", reject)


@pytest.fixture
def ready(tmp_path, monkeypatch):
    app, prepare, generate, result = setup_app(tmp_path, monkeypatch, configured=False)
    proposal = proposal_response(result, total_duration_ms=3)
    app.state.proposal_store.save(OWNER.id, proposal)
    yield app, proposal, prepare, generate
    prepare.assert_not_called()
    generate.assert_not_awaited()


def body(proposal, revision=0, **changes):
    return {
        "proposal_id": proposal.proposal_id, "base_commit_sha": proposal.base_commit_sha,
        "patch_sha256": proposal.patch_sha256, "decision": "approve_for_validation",
        "expected_revision": revision, **changes,
    }


def test_live_style_decision_lifecycle_history_and_restart_without_ai(ready):
    app, proposal, _, _ = ready
    current_path = f"/api/proposals/{proposal.proposal_id}/decision"
    history_path = f"/api/proposals/{proposal.proposal_id}/decisions"
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        initial = client.get(current_path)
        assert initial.status_code == 200 and initial.json()["status"] == "unreviewed"
        assert initial.json()["revision"] == 0 and initial.json()["current_decision"] is None
        assert client.get(history_path).json()["decisions"] == []
        approved = client.post("/api/proposals/decision", json=body(proposal, comment="Reviewed this exact diff."), headers=HEADERS)
        assert approved.status_code == 200, approved.text
        first = approved.json()
        assert first["status"] == "approved_for_validation" and first["revision"] == 1
        assert first["current_decision"]["account_id"] == OWNER.id
        assert first["current_decision"]["patch_sha256"] == proposal.patch_sha256
        rejected = client.post("/api/proposals/decision", json=body(proposal, 1, decision="reject"), headers=HEADERS)
        assert rejected.status_code == 200 and rejected.json()["status"] == "rejected"
        restored = client.post("/api/proposals/decision", json=body(proposal, 2), headers=HEADERS)
        assert restored.status_code == 200 and restored.json()["revision"] == 3
        final = restored.json()
        listing = client.get(history_path + "?per_page=2").json()
        assert [entry["revision"] for entry in listing["decisions"]] == [3, 2]
        assert listing["has_more"] and listing["next_page"] == 2
        last = client.get(history_path + "?page=2&per_page=2").json()
        assert [entry["revision"] for entry in last["decisions"]] == [1]
        # Proposal stays an immutable draft; review state is separate.
        assert client.get("/api/proposals/" + proposal.proposal_id).json() == proposal.model_dump(mode="json")
        for response in [initial, approved, rejected, restored, client.get(history_path)]:
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["pragma"] == "no-cache"
    reopened = create_app(app.state.settings)
    with TestClient(reopened, base_url=BASE) as client:
        assert client.get(current_path).status_code == 401
        login(client, reopened)
        assert client.get(current_path).json() == final


@pytest.mark.parametrize("action", ["current", "history", "post"])
def test_session_required_for_every_review_operation(ready, action):
    app, proposal, _, _ = ready
    with TestClient(app, base_url=BASE) as client:
        if action == "post": response = client.post("/api/proposals/decision", json=body(proposal), headers=HEADERS)
        else: response = client.get(f"/api/proposals/{proposal.proposal_id}/" + ("decision" if action == "current" else "decisions"))
        assert response.status_code == 401
    assert app.state.proposal_store.get_review(OWNER.id, proposal.proposal_id).revision == 0


@pytest.mark.parametrize("headers", [
    {}, {"Origin": BASE}, {"X-RepoFix-CSRF": "1"},
    {"Origin": "https://evil.example", "X-RepoFix-CSRF": "1"},
    {"Origin": BASE, "X-RepoFix-CSRF": "0"},
    [("Origin", BASE), ("Origin", BASE), ("X-RepoFix-CSRF", "1")],
    [("Origin", BASE), ("X-RepoFix-CSRF", "1"), ("X-RepoFix-CSRF", "1")],
])
def test_trusted_origin_and_one_csrf_header_required(ready, headers):
    app, proposal, _, _ = ready
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        response = client.post("/api/proposals/decision", json=body(proposal), headers=headers)
        assert response.status_code == 403
        assert client.get(f"/api/proposals/{proposal.proposal_id}/decision").json()["revision"] == 0


def test_missing_and_foreign_proposals_return_identical_404s(ready):
    app, proposal, _, _ = ready
    with TestClient(app, base_url=BASE) as client:
        login(client, app, OTHER)
        for suffix in ["decision", "decisions"]:
            foreign = client.get(f"/api/proposals/{proposal.proposal_id}/{suffix}")
            missing = client.get(f"/api/proposals/{'0' * 32}/{suffix}")
            assert foreign.status_code == missing.status_code == 404 and foreign.json() == missing.json()
        foreign = client.post("/api/proposals/decision", json=body(proposal), headers=HEADERS)
        missing = client.post("/api/proposals/decision", json=body(proposal, proposal_id="0" * 32), headers=HEADERS)
        assert foreign.status_code == missing.status_code == 404 and foreign.json() == missing.json()


@pytest.mark.parametrize("changes", [
    {"expected_revision": True}, {"expected_revision": -1}, {"expected_revision": "0"},
    {"account_id": OTHER.id}, {"created_at": "2026-10-31T00:00:00Z"},
    {"decision": "publish"}, {"command": "pytest"}, {"decision_id": "d" * 32},
])
def test_invalid_browser_fields_are_rejected(ready, changes):
    app, proposal, _, _ = ready
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        response = client.post("/api/proposals/decision", json=body(proposal, **changes), headers=HEADERS)
        assert response.status_code == 422
        assert client.get(f"/api/proposals/{proposal.proposal_id}/decision").json()["revision"] == 0


def test_old_clients_must_supply_observed_revision(ready):
    app, proposal, _, _ = ready
    request = body(proposal)
    del request["expected_revision"]
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        assert client.post("/api/proposals/decision", json=request, headers=HEADERS).status_code == 422


def test_stale_revision_and_hash_conflicts_do_not_overwrite_rejection(ready):
    app, proposal, _, _ = ready
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        assert client.post("/api/proposals/decision", json=body(proposal, decision="reject"), headers=HEADERS).status_code == 200
        stale = client.post("/api/proposals/decision", json=body(proposal), headers=HEADERS)
        changed = client.post("/api/proposals/decision", json=body(proposal, 1, patch_sha256="d" * 64), headers=HEADERS)
        assert stale.status_code == changed.status_code == 409
        current = client.get(f"/api/proposals/{proposal.proposal_id}/decision").json()
        assert current["status"] == "rejected" and current["revision"] == 1


def test_logout_after_database_wait_rolls_back_pending_review(ready, monkeypatch):
    app, proposal, _, _ = ready
    original = app.state.proposal_store.record_decision
    with TestClient(app, base_url=BASE) as client:
        token = login(client, app)
        def revoke_then_record(*args, **kwargs):
            app.state.auth_store.revoke_session(token)
            return original(*args, **kwargs)
        monkeypatch.setattr(app.state.proposal_store, "record_decision", revoke_then_record)
        response = client.post("/api/proposals/decision", json=body(proposal), headers=HEADERS)
        assert response.status_code == 401
    assert app.state.proposal_store.get_review(OWNER.id, proposal.proposal_id).revision == 0


def test_review_storage_failure_is_safe_and_does_not_infer_approval(ready, monkeypatch):
    app, proposal, _, _ = ready
    monkeypatch.setattr(app.state.proposal_store, "record_decision", Mock(side_effect=ProposalReviewError("unavailable")))
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        response = client.post("/api/proposals/decision", json=body(proposal), headers=HEADERS)
        assert response.status_code == 503 and "sqlite" not in response.text
        current = client.get(f"/api/proposals/{proposal.proposal_id}/decision").json()
        assert current["status"] == "unreviewed" and current["current_decision"] is None


@pytest.mark.parametrize("path", ["/api/proposals/bad/decision", "/api/proposals/bad/decisions", "?page=0", "?per_page=51"])
def test_read_path_and_page_bounds(ready, path):
    app, proposal, _, _ = ready
    if path.startswith("?"): path = f"/api/proposals/{proposal.proposal_id}/decisions" + path
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        assert client.get(path).status_code == 422


def test_review_works_during_ai_busy_slot_without_consuming_ai_attempts(ready):
    from app.api.investigations import _RunGate
    app, proposal, _, _ = ready
    app.state.investigation_run_gate = _RunGate()
    app.state.investigation_run_gate.reserve(OWNER.id)
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        assert client.post("/api/proposals/decision", json=body(proposal), headers=HEADERS).status_code == 200
    app.state.investigation_run_gate.release()


def test_openapi_exposes_review_routes_and_required_revision(ready):
    app, _, _, _ = ready
    schema = app.openapi()
    assert schema["paths"]["/api/proposals/decision"]["post"]
    assert schema["paths"]["/api/proposals/{proposal_id}/decision"]["get"]
    assert schema["paths"]["/api/proposals/{proposal_id}/decisions"]["get"]
    required = schema["components"]["schemas"]["ProposalDecisionRequest"]["required"]
    assert set(required) == {"proposal_id", "base_commit_sha", "patch_sha256", "decision", "expected_revision"}
