"""Authenticated history API and storage-failure contracts, entirely offline."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import investigations as api
from app.config import Settings
from app.github_client import RepositoryMetadata, RepositorySnapshot
from app.github_oauth import GitHubIdentity
from app.github_repositories import PublicRepository
from app.github_tree import RepositoryTree
from app.investigation_context import InvestigationContext
from app.investigation_input import InvestigationInput
from app.investigation_report import InvestigationReport
from app.investigation_runner import InvestigationRunError, InvestigationRunResult
from app.investigation_store import InvestigationStore, InvestigationStoreError
from app.main import create_app


BASE = "http://127.0.0.1:8000"
HEADERS = {"Origin": BASE, "X-RepoFix-CSRF": "1"}
BODY = {"repository_id": 5, "repository_name": "example", "issue_description": "  Reproduce a synthetic issue.  "}
OWNER = GitHubIdentity(id=123, login="demo-user", type="User")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("A history test attempted a real network request.")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", reject)


def make_app(monkeypatch, path, *, ready=True):
    config = Settings(
        _env_file=None, environment="test", history_database_path=path,
        openai_api_key="offline-key" if ready else None,
        ai_model="offline-model" if ready else None,
    )
    app = create_app(config)
    repository = PublicRepository(
        id=5, name="example", full_name="demo-user/example", owner=OWNER,
        private=False, visibility="public", default_branch="main",
        fork=False, archived=False, disabled=False,
    )
    context = InvestigationContext(
        identity=OWNER, request=InvestigationInput(**BODY), repository=repository,
        snapshot=RepositorySnapshot(
            repository=RepositoryMetadata(**repository.model_dump()), commit_sha="a" * 40,
        ),
        tree=RepositoryTree(
            repository_full_name=repository.full_name, commit_sha="a" * 40,
            tree_sha="b" * 40, entries=[],
        ),
    )
    prepare = Mock(side_effect=lambda identity, payload: replace(context, identity=identity, request=payload))

    async def completed(_context, _settings):
        return InvestigationRunResult(
            run_id=uuid4().hex, repository_full_name=repository.full_name, commit_sha="a" * 40,
            report=InvestigationReport(
                status="insufficient_evidence", issue_summary="Synthetic issue.", issue_category="other",
                root_cause=None, evidence=[], next_steps=["Collect reproduction details."],
                limitations=["Offline fixture."],
            ),
            duration_ms=0, model_calls=1, tool_attempts=0,
            repository_stats={"tool_calls": 0}, events=(),
        )

    run = AsyncMock(side_effect=completed)
    monkeypatch.setattr(api, "prepare_investigation_context", prepare)
    monkeypatch.setattr(api, "run_investigation", run)
    return app, prepare, run


def login(client, app, identity=OWNER):
    token = app.state.auth_store.create_session(identity)
    client.cookies.set("repofix_session", token)
    return token


def test_run_saves_issue_report_and_metadata_without_secrets(tmp_path, monkeypatch):
    path = tmp_path / "history.db"
    app, prepare, run = make_app(monkeypatch, path)
    assert not path.exists()
    with TestClient(app, base_url=BASE) as client:
        token = login(client, app)
        response = client.post("/api/investigations/run", json=BODY, headers=HEADERS)
        assert response.status_code == 200
        completed = response.json()
        assert completed["history_saved"] is True
        listing = client.get("/api/investigations")
        assert listing.status_code == 200
        assert listing.headers["cache-control"] == "no-store"
        item = listing.json()["investigations"][0]
        assert item["run_id"] == completed["run_id"]
        assert item["report_status"] == "insufficient_evidence"
        assert "issue_description" not in item and "report" not in item
        detail = client.get(f"/api/investigations/{item['run_id']}")
        assert detail.status_code == 200
        assert detail.json()["issue_description"] == BODY["issue_description"].strip()
        assert detail.json()["investigation"] == completed
        assert detail.json()["created_at"].endswith("Z")
        assert token not in detail.text and "offline-key" not in detail.text
        assert str(path) not in detail.text and "account_id" not in detail.text
        assert run.await_count == prepare.call_count == 1


@pytest.mark.parametrize("endpoint", ["/api/investigations", "/api/investigations/" + "a" * 32])
def test_history_requires_login_before_database_work(tmp_path, monkeypatch, endpoint):
    path = tmp_path / "history.db"
    app, prepare, run = make_app(monkeypatch, path)
    with TestClient(app, base_url=BASE) as client:
        response = client.get(endpoint)
        assert response.status_code == 401
        assert response.headers["cache-control"] == "no-store"
    assert not path.exists()
    run.assert_not_awaited()
    prepare.assert_not_called()


def test_another_account_sees_neither_list_entry_nor_detail(tmp_path, monkeypatch):
    app, _, _ = make_app(monkeypatch, tmp_path / "history.db")
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        result = client.post("/api/investigations/run", json=BODY, headers=HEADERS).json()
        login(client, app, GitHubIdentity(id=456, login="other-user", type="User"))
        assert client.get("/api/investigations").json()["investigations"] == []
        hidden = client.get(f"/api/investigations/{result['run_id']}")
        missing = client.get("/api/investigations/" + "0" * 32)
        assert hidden.status_code == missing.status_code == 404
        assert hidden.json() == missing.json()


def test_restart_requires_relogin_but_preserves_history_without_ai_key(tmp_path, monkeypatch):
    path = tmp_path / "history.db"
    first, _, _ = make_app(monkeypatch, path)
    with TestClient(first, base_url=BASE) as client:
        old_token = login(client, first)
        result = client.post("/api/investigations/run", json=BODY, headers=HEADERS).json()
    second, prepare, run = make_app(monkeypatch, path, ready=False)
    with TestClient(second, base_url=BASE) as client:
        client.cookies.set("repofix_session", old_token)
        assert client.get("/api/investigations").status_code == 401
        # A GitHub rename keeps the same numeric account owner.
        login(client, second, GitHubIdentity(id=123, login="renamed-user", type="User"))
        detail = client.get(f"/api/investigations/{result['run_id']}")
        assert detail.status_code == 200
        assert detail.json()["investigation"] == result
        assert len(client.get("/api/investigations").json()["investigations"]) == 1
    prepare.assert_not_called()
    run.assert_not_awaited()


def test_storage_failure_stops_preparation_and_ai_and_releases_gate(tmp_path, monkeypatch):
    app, prepare, run = make_app(monkeypatch, tmp_path / "history.db")
    store = app.state.investigation_store
    original = store.ensure_ready
    monkeypatch.setattr(store, "ensure_ready", Mock(side_effect=InvestigationStoreError("PRIVATE_SQLITE_PATH")))
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        result = client.post("/api/investigations/run", json=BODY, headers=HEADERS)
        assert result.status_code == 503
        assert "PRIVATE_SQLITE_PATH" not in result.text
        prepare.assert_not_called()
        run.assert_not_awaited()
        monkeypatch.setattr(store, "ensure_ready", original)
        recovered = client.post("/api/investigations/run", json=BODY, headers=HEADERS)
        assert recovered.status_code == 200 and recovered.json()["history_saved"] is True


def test_save_failure_returns_completed_report_without_claiming_it_was_saved(tmp_path, monkeypatch):
    app, _, run = make_app(monkeypatch, tmp_path / "history.db")
    monkeypatch.setattr(app.state.investigation_store, "save", Mock(side_effect=InvestigationStoreError("PRIVATE_DISK_ERROR")))
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        response = client.post("/api/investigations/run", json=BODY, headers=HEADERS)
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert response.json()["history_saved"] is False
        assert response.json()["report"]["status"] == "insufficient_evidence"
        assert "PRIVATE_DISK_ERROR" not in response.text
        assert client.get("/api/investigations").json()["investigations"] == []
    assert run.await_count == 1


def test_failed_investigation_is_not_saved(tmp_path, monkeypatch):
    app, _, run = make_app(monkeypatch, tmp_path / "history.db")
    run.side_effect = InvestigationRunError("ungrounded_report", "Unverified source.")
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        assert client.post("/api/investigations/run", json=BODY, headers=HEADERS).status_code == 502
        assert client.get("/api/investigations").json()["investigations"] == []


def test_provenance_mismatch_is_not_saved(tmp_path, monkeypatch):
    app, _, run = make_app(monkeypatch, tmp_path / "history.db")
    original = run.side_effect

    async def changed(context, settings):
        return replace(await original(context, settings), commit_sha="c" * 40)

    run.side_effect = changed
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        assert client.post("/api/investigations/run", json=BODY, headers=HEADERS).status_code == 502
        assert client.get("/api/investigations").json()["investigations"] == []


def test_http_history_pagination_and_empty_history(tmp_path, monkeypatch):
    app, _, run = make_app(monkeypatch, tmp_path / "history.db")
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        assert client.get("/api/investigations").json()["investigations"] == []
        ids = [client.post("/api/investigations/run", json=BODY, headers=HEADERS).json()["run_id"] for _ in range(3)]
        first = client.get("/api/investigations?per_page=2").json()
        second = client.get("/api/investigations?page=2&per_page=2").json()
        assert [r["run_id"] for r in first["investigations"] + second["investigations"]] == ids[::-1]
        assert first["next_page"] == 2 and first["has_more"] is True
        assert second["has_more"] is False and second["next_page"] is None
    assert run.await_count == 3


@pytest.mark.parametrize("endpoint", [
    "/api/investigations?page=0", "/api/investigations?per_page=51",
    "/api/investigations?page=10001", "/api/investigations/not-a-run-id",
])
def test_history_input_bounds(tmp_path, monkeypatch, endpoint):
    app, _, run = make_app(monkeypatch, tmp_path / "history.db")
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        assert client.get(endpoint).status_code == 422
    run.assert_not_awaited()


def test_uninitialized_history_is_controlled(tmp_path, monkeypatch):
    app, _, run = make_app(monkeypatch, tmp_path / "history.db")
    app.state.investigation_store = None
    with TestClient(app, base_url=BASE) as client:
        login(client, app)
        response = client.get("/api/investigations")
        assert response.status_code == 503
        assert response.headers["cache-control"] == "no-store"
    run.assert_not_awaited()


@pytest.mark.parametrize("path", ["", "   ", ":memory:", "bad\npath"])
def test_settings_require_a_real_history_file_path(path):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, history_database_path=path)


def test_relative_history_path_is_anchored_to_backend(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = Settings(_env_file=None, history_database_path="data/custom.db")
    assert config.history_database_path == Path(__file__).resolve().parents[1] / "data/custom.db"


def test_injected_store_type_is_validated():
    with pytest.raises(ValueError, match="InvestigationStore"):
        create_app(Settings(_env_file=None), history_store=object())
