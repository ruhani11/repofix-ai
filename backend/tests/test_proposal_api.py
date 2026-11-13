"""Authenticated proposal HTTP lifecycle with isolated fake generation only."""

import asyncio
from dataclasses import replace
import importlib
import importlib.util
import os
from threading import Event
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import proposals as api
from app.api.investigations import _RunGate
from app.auth_store import AuthStore
from app.config import Settings
from app.github_client import RepositoryMetadata, RepositorySnapshot
from app.github_oauth import GitHubIdentity
from app.github_repositories import PublicRepository
from app.investigation_context import InvestigationContext
from app.investigation_input import InvestigationInput
from app.investigation_store import InvestigationStore
from app.main import create_app
from app.proposal_context import ProposalContext, ProposalContextError
from app.proposal_generator import ProposalGenerationError
from app.proposal_history import proposal_response
from app.proposal_store import ProposalStore, ProposalStoreError
from test_proposal_store import make_result


BASE = "http://127.0.0.1:8000"
HEADERS = {"Origin": BASE, "X-RepoFix-CSRF": "1"}
BODY = {"investigation_run_id": "c" * 32}
OWNER = GitHubIdentity(id=123, login="demo-user", type="User")
OTHER = GitHubIdentity(id=999, login="other-user", type="User")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("REPOFIX_"):
            monkeypatch.delenv(name, raising=False)
    def reject(*args, **kwargs):
        pytest.fail("Proposal API tests must not make real network calls.")
    packages = [httpx]
    if importlib.util.find_spec("httpx2"):
        packages.append(importlib.import_module("httpx2"))
    for package in packages:
        monkeypatch.setattr(package.HTTPTransport, "handle_request", reject)
        monkeypatch.setattr(package.AsyncHTTPTransport, "handle_async_request", reject)


def setup_app(tmp_path, monkeypatch, *, configured=True, turns=10):
    settings = Settings(
        _env_file=None, environment="test", history_database_path=tmp_path / "history.db",
        proposal_database_path=tmp_path / "proposals.db", ai_max_turns=turns,
        openai_api_key="offline-key" if configured else None,
        ai_model="offline-model" if configured else None,
    )
    app = create_app(settings)
    history, tree, result = make_result()
    payload = InvestigationInput(repository_id=history.repository_id, repository_name=history.repository_name, issue_description=history.issue_description)
    app.state.investigation_store.save(OWNER.id, payload, history.investigation)
    repository = PublicRepository(
        id=5, name="example", full_name="demo-user/example", owner=OWNER,
        private=False, visibility="public", default_branch="main", fork=False, archived=False, disabled=False,
    )
    def prepare(identity, body, store):
        saved = store.get(identity.id, body.investigation_run_id)
        if saved is None:
            raise ProposalContextError("not_found")
        return ProposalContext(saved, InvestigationContext(
            identity, payload, repository,
            RepositorySnapshot(repository=RepositoryMetadata(**repository.model_dump()), commit_sha="a" * 40), tree,
        ))
    preparer = Mock(side_effect=prepare)
    generator = AsyncMock(side_effect=lambda context, settings: replace(result, proposal_id=uuid4().hex))
    monkeypatch.setattr(api, "prepare_proposal_context", preparer)
    monkeypatch.setattr(api, "generate_fix_proposal", generator)
    return app, preparer, generator, result


def login(client, app, identity=OWNER):
    token = app.state.auth_store.create_session(identity)
    client.cookies.set("repofix_session", token)
    return token


def test_generate_save_list_detail_restart_and_relogin(tmp_path, monkeypatch):
    app, prepare, generate, result = setup_app(tmp_path, monkeypatch)
    assert not (tmp_path / "proposals.db").exists()
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        assert client.get("/api/proposals").json()["proposals"] == []
        response = client.post("/api/proposals/generate", json=BODY, headers=HEADERS)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "proposed" and data["proposal_saved"] is True
        assert data["investigation_run_id"] == BODY["investigation_run_id"]
        assert data["base_commit_sha"] == "a" * 40 and data["patch_sha256"] == result.validated.patch_sha256
        assert data["draft"]["status"] == "proposed" and "applied" not in data
        assert data["changes"][0]["original_content"] != data["changes"][0]["proposed_content"]
        listing = client.get("/api/proposals")
        assert listing.json()["proposals"][0]["proposal_id"] == data["proposal_id"]
        detail = client.get("/api/proposals/" + data["proposal_id"])
        assert detail.json() == data
        for reply in [response, listing, detail]:
            assert reply.headers["cache-control"] == "no-store"
            assert reply.headers["pragma"] == "no-cache"
            assert "offline-key" not in reply.text and "set-cookie" not in reply.headers
    reopened = create_app(app.state.settings.model_copy(update={"ai_model": None, "openai_api_key": None}))
    with TestClient(reopened, base_url=BASE) as client:
        assert client.get("/api/proposals").status_code == 401
        login(client, reopened)
        assert client.get("/api/proposals/" + data["proposal_id"]).json() == data
    assert prepare.call_count == 1 and generate.await_count == 1


@pytest.mark.parametrize("method,path", [("get", "/api/proposals"), ("get", "/api/proposals/" + "a" * 32), ("post", "/api/proposals/generate")])
def test_missing_session_stops_all_work(tmp_path, monkeypatch, method, path):
    app, prepare, generate, _ = setup_app(tmp_path, monkeypatch)
    with TestClient(app, base_url=BASE) as client:
        reply = getattr(client, method)(path, **({"json": BODY, "headers": HEADERS} if method == "post" else {}))
        assert reply.status_code == 401
    prepare.assert_not_called()
    generate.assert_not_awaited()
    assert not (tmp_path / "proposals.db").exists()


@pytest.mark.parametrize("headers", [
    {}, {"Origin": BASE}, {"X-RepoFix-CSRF": "1"},
    {"Origin": "https://evil.example", "X-RepoFix-CSRF": "1"},
    {"Origin": BASE, "X-RepoFix-CSRF": "0"},
    [("Origin", BASE), ("Origin", BASE), ("X-RepoFix-CSRF", "1")],
    [("Origin", BASE), ("X-RepoFix-CSRF", "1"), ("X-RepoFix-CSRF", "1")],
])
def test_csrf_and_origin_fail_before_storage_or_ai(tmp_path, monkeypatch, headers):
    app, prepare, generate, _ = setup_app(tmp_path, monkeypatch)
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        reply = client.post("/api/proposals/generate", json=BODY, headers=headers)
        assert reply.status_code == 403
    prepare.assert_not_called()
    generate.assert_not_awaited()
    assert not (tmp_path / "proposals.db").exists()


def test_vite_proxy_origin_is_accepted_in_development(tmp_path, monkeypatch):
    app, _, _, _ = setup_app(tmp_path, monkeypatch)
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        assert client.post("/api/proposals/generate", json=BODY, headers={**HEADERS, "Origin": "http://127.0.0.1:5173"}).status_code == 200


@pytest.mark.parametrize("configured,turns", [(False, 10), (True, 1)])
def test_configuration_stops_before_preparation(tmp_path, monkeypatch, configured, turns):
    app, prepare, generate, _ = setup_app(tmp_path, monkeypatch, configured=configured, turns=turns)
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        assert client.post("/api/proposals/generate", json=BODY, headers=HEADERS).status_code == 503
    prepare.assert_not_called()
    generate.assert_not_awaited()


@pytest.mark.parametrize("body", [{}, {"investigation_run_id": "x"}, {**BODY, "account_id": 999}, {**BODY, "base_commit_sha": "e" * 40}, {**BODY, "draft": {}}])
def test_input_only_accepts_saved_run_id(tmp_path, monkeypatch, body):
    app, prepare, generate, _ = setup_app(tmp_path, monkeypatch)
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        assert client.post("/api/proposals/generate", json=body, headers=HEADERS).status_code == 422
    prepare.assert_not_called()
    generate.assert_not_awaited()


def test_other_accounts_cannot_generate_or_retrieve_saved_artifacts(tmp_path, monkeypatch):
    app, _, generate, result = setup_app(tmp_path, monkeypatch)
    detail = proposal_response(result, total_duration_ms=3)
    app.state.proposal_store.save(OWNER.id, detail)
    with TestClient(app, base_url=BASE) as client:
        login(client, app, OTHER)
        assert client.get("/api/proposals").json()["proposals"] == []
        foreign = client.get("/api/proposals/" + detail.proposal_id)
        missing = client.get("/api/proposals/" + "0" * 32)
        assert foreign.status_code == missing.status_code == 404 and foreign.json() == missing.json()
        assert client.post("/api/proposals/generate", json=BODY, headers=HEADERS).status_code == 404
    generate.assert_not_awaited()


@pytest.mark.parametrize("code,status", [("ineligible_investigation", 409), ("selection_changed", 409), ("github_unavailable", 502), ("history_unavailable", 503), ("invalid_history", 502)])
def test_context_errors_stop_generation(tmp_path, monkeypatch, code, status):
    app, prepare, generate, _ = setup_app(tmp_path, monkeypatch)
    prepare.side_effect = ProposalContextError(code)
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        assert client.post("/api/proposals/generate", json=BODY, headers=HEADERS).status_code == status
        assert client.get("/api/proposals").json()["proposals"] == []
    generate.assert_not_awaited()


@pytest.mark.parametrize("code,status", [
    ("timeout", 504), ("turn_limit", 409), ("tool_limit", 409), ("event_limit", 409),
    ("invalid_proposal", 502), ("unobserved_source", 502), ("invalid_proposal_source", 502),
    ("provider_daily_quota", 429), ("provider_rate_limit", 429), ("provider_limit", 429),
    ("provider_unavailable", 503), ("provider_timeout", 504), ("agent_error", 502),
])
def test_generation_errors_safe_unsaved_and_release_slot(tmp_path, monkeypatch, code, status):
    app, _, generate, _ = setup_app(tmp_path, monkeypatch)
    generate.side_effect = ProposalGenerationError(code, retry_after=60)
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        reply = client.post("/api/proposals/generate", json=BODY, headers=HEADERS)
        assert reply.status_code == status
        assert (reply.headers.get("retry-after") == "60") == (code in api._PROVIDER_WAIT_ERRORS)
        assert reply.headers["cache-control"] == "no-store"
        assert client.get("/api/proposals").json()["proposals"] == []
        gate = app.state.investigation_run_gate
        gate.reserve(999)
        gate.release()


def test_storage_preflight_failure_stops_paid_call_and_can_recover(tmp_path, monkeypatch):
    app, prepare, generate, _ = setup_app(tmp_path, monkeypatch)
    original = app.state.proposal_store.ensure_ready
    monkeypatch.setattr(app.state.proposal_store, "ensure_ready", Mock(side_effect=ProposalStoreError()))
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        assert client.post("/api/proposals/generate", json=BODY, headers=HEADERS).status_code == 503
        prepare.assert_not_called()
        generate.assert_not_awaited()
        monkeypatch.setattr(app.state.proposal_store, "ensure_ready", original)
        assert client.post("/api/proposals/generate", json=BODY, headers=HEADERS).json()["proposal_saved"] is True


def test_late_save_failure_preserves_paid_draft_without_claiming_saved(tmp_path, monkeypatch):
    app, _, generate, _ = setup_app(tmp_path, monkeypatch)
    monkeypatch.setattr(app.state.proposal_store, "save", Mock(side_effect=ProposalStoreError()))
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        reply = client.post("/api/proposals/generate", json=BODY, headers=HEADERS)
        assert reply.status_code == 200 and reply.json()["proposal_saved"] is False
        assert reply.json()["draft"]["edits"] and reply.json()["unified_diff"]
        assert client.get("/api/proposals").json()["proposals"] == []
    assert generate.await_count == 1


def test_revoked_session_during_preparation_stops_paid_call(tmp_path, monkeypatch):
    app, prepare, generate, _ = setup_app(tmp_path, monkeypatch)
    original = prepare.side_effect
    with TestClient(app, base_url=BASE) as client:
        token = login(client, app)
        def revoke(*args):
            app.state.auth_store.revoke_session(token)
            return original(*args)
        prepare.side_effect = revoke
        assert client.post("/api/proposals/generate", json=BODY, headers=HEADERS).status_code == 401
    generate.assert_not_awaited()


@pytest.mark.parametrize("field,value", [("investigation_run_id", "d" * 32), ("repository_id", 999), ("repository_full_name", "other/example"), ("base_commit_sha", "d" * 40), ("tree_sha", "d" * 40)])
def test_generation_provenance_mismatch_is_not_saved(tmp_path, monkeypatch, field, value):
    app, _, generate, result = setup_app(tmp_path, monkeypatch)
    generate.side_effect = None
    generate.return_value = replace(result, validated=replace(result.validated, **{field: value}))
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        reply = client.post("/api/proposals/generate", json=BODY, headers=HEADERS)
        assert reply.status_code == 502
        assert client.get("/api/proposals").json()["proposals"] == []


def test_bad_diff_from_internal_generator_is_rejected(tmp_path, monkeypatch):
    app, _, generate, result = setup_app(tmp_path, monkeypatch)
    generate.side_effect = None
    generate.return_value = replace(result, validated=replace(result.validated, unified_diff="PRIVATE_FAKE_DIFF"))
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        reply = client.post("/api/proposals/generate", json=BODY, headers=HEADERS)
        assert reply.status_code == 502 and "PRIVATE_FAKE_DIFF" not in reply.text
        assert client.get("/api/proposals").json()["proposals"] == []


def test_shared_active_slot_and_hourly_budget(tmp_path, monkeypatch):
    app, prepare, generate, _ = setup_app(tmp_path, monkeypatch)
    gate = _RunGate()
    app.state.investigation_run_gate = gate
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        gate.reserve(999)
        busy = client.post("/api/proposals/generate", json=BODY, headers=HEADERS)
        assert busy.status_code == 429 and busy.headers["retry-after"] == "5"
        gate.release()
        for _ in range(6):
            gate.reserve(OWNER.id)
            gate.release()
        limited = client.post("/api/proposals/generate", json=BODY, headers=HEADERS)
        assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1
    prepare.assert_not_called()
    generate.assert_not_awaited()


def test_preparation_timeout_keeps_slot_until_sync_worker_finishes(tmp_path, monkeypatch):
    app, prepare, generate, _ = setup_app(tmp_path, monkeypatch)
    started, finish, done = Event(), Event(), Event()
    def blocked(*args):
        started.set()
        assert finish.wait(5)
        done.set()
        raise ProposalContextError("github_unavailable")
    prepare.side_effect = blocked
    monkeypatch.setattr(api, "PREPARATION_TIMEOUT_SECONDS", 0.05)
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        try:
            first = client.post("/api/proposals/generate", json=BODY, headers=HEADERS)
            assert started.is_set() and first.status_code == 504
            assert client.post("/api/proposals/generate", json=BODY, headers=HEADERS).status_code == 429
        finally:
            finish.set()
            assert done.wait(5)
    generate.assert_not_awaited()


def test_save_timeout_returns_unsaved_flag_holds_slot_and_does_not_repeat_ai(tmp_path, monkeypatch):
    app, _, generate, _ = setup_app(tmp_path, monkeypatch)
    finish, done = Event(), Event()
    original = app.state.proposal_store.save
    def blocked(*args):
        try:
            assert finish.wait(5)
            original(*args)
        finally:
            done.set()
    monkeypatch.setattr(app.state.proposal_store, "save", blocked)
    monkeypatch.setattr(api, "SAVE_TIMEOUT_SECONDS", 0.05)
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        try:
            reply = client.post("/api/proposals/generate", json=BODY, headers=HEADERS)
            assert reply.status_code == 200 and reply.json()["proposal_saved"] is False
            assert client.post("/api/proposals/generate", json=BODY, headers=HEADERS).status_code == 429
        finally:
            finish.set()
            assert done.wait(5)
        # The timed-out save can still complete; clients should refresh, not retry AI.
        assert client.get("/api/proposals").json()["proposals"][0]["proposal_id"] == reply.json()["proposal_id"]
    assert generate.await_count == 1


@pytest.mark.parametrize("path", ["/api/proposals?page=0", "/api/proposals?per_page=51", "/api/proposals/not-an-id"])
def test_http_bounds(tmp_path, monkeypatch, path):
    app, _, _, _ = setup_app(tmp_path, monkeypatch)
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        assert client.get(path).status_code == 422


def test_missing_storage_and_unknown_internal_error_do_not_expose_text(tmp_path, monkeypatch):
    app, _, generate, _ = setup_app(tmp_path, monkeypatch)
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        store = app.state.proposal_store
        app.state.proposal_store = None
        try:
            assert client.get("/api/proposals").status_code == 503
            assert client.post("/api/proposals/generate", json=BODY, headers=HEADERS).status_code == 503
        finally:
            app.state.proposal_store = store
        generate.side_effect = RuntimeError("PRIVATE_KEY_SOURCE_ERROR")
        reply = client.post("/api/proposals/generate", json=BODY, headers=HEADERS)
        assert reply.status_code == 500 and "PRIVATE_KEY_SOURCE_ERROR" not in reply.text


def test_app_injection_shutdown_and_openapi_contract(tmp_path, monkeypatch):
    app, _, _, _ = setup_app(tmp_path, monkeypatch)
    store = ProposalStore(":memory:")
    close = Mock(wraps=store.close)
    monkeypatch.setattr(store, "close", close)
    injected = create_app(app.state.settings, proposal_store=store)
    assert injected.state.proposal_store is store
    with TestClient(injected, base_url=BASE):
        paths = injected.openapi()["paths"]
        assert set(paths["/api/proposals/generate"]) == {"post"}
        assert paths["/api/proposals"]["get"]["responses"]["200"]
        assert paths["/api/proposals/{proposal_id}"]["get"]["responses"]["404"]
    close.assert_called_once()
    with pytest.raises(ValueError):
        create_app(app.state.settings, proposal_store=object())
