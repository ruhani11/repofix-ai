"""Offline HTTP tests for authenticated investigation preparation."""

from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import investigations
from app.auth_store import AuthStore, SESSION_IDLE_SECONDS
from app.config import Settings
from app.github_client import (
    GitHubClientError, RepositoryMetadata, RepositorySnapshot,
)
from app.github_oauth import GitHubIdentity
from app.github_repositories import PublicRepository
from app.github_selection import RepositorySelectionError
from app.github_tree import RepositoryTree
from app.investigation_context import InvestigationContext
from app.investigation_input import InvestigationInput
from app.main import create_app


BASE_URL = "http://127.0.0.1:8000"
PREPARATION_URL = "/api/investigations/prepare"
VALID_INPUT = {
    "repository_id": 5,
    "repository_name": "example",
    "issue_description": "  Login fails.  ",
}


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    for key in (
        "REPOFIX_APP_NAME",
        "REPOFIX_ENVIRONMENT",
        "REPOFIX_GITHUB_CLIENT_ID",
        "REPOFIX_GITHUB_CLIENT_SECRET",
        "REPOFIX_GITHUB_REDIRECT_URI",
    ):
        monkeypatch.delenv(key, raising=False)

    def block_network(*args, **kwargs):
        pytest.fail("A preparation API test attempted a real network request.")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", block_network)


@pytest.fixture
def preparation_api(monkeypatch):
    identity = GitHubIdentity(id=123, login="demo-user", type="User")
    repository = PublicRepository(
        id=5,
        name="example",
        full_name="demo-user/example",
        owner=identity,
        private=False,
        visibility="public",
        default_branch="main",
        fork=False,
        archived=False,
        disabled=False,
    )
    snapshot = RepositorySnapshot(
        repository=RepositoryMetadata(**repository.model_dump()),
        commit_sha="a" * 40,
    )
    tree = RepositoryTree(
        repository_full_name=repository.full_name,
        commit_sha=snapshot.commit_sha,
        tree_sha="b" * 40,
        entries=[{
            "path": "README.md",
            "type": "blob",
            "mode": "100644",
            "sha": "c" * 40,
            "size": 20,
        }],
    )
    context = InvestigationContext(
        identity=identity,
        request=InvestigationInput(**VALID_INPUT),
        repository=repository,
        snapshot=snapshot,
        tree=tree,
    )
    service = Mock(return_value=context)
    monkeypatch.setattr(investigations, "prepare_investigation_context", service)

    clock = [0.0]
    application = create_app(Settings(_env_file=None, environment="test"))
    application.state.auth_store = AuthStore(clock=lambda: clock[0])

    with TestClient(application, base_url=BASE_URL) as client:
        yield application, client, identity, service, clock


def connect_session(preparation_api):
    application, client, identity, _, _ = preparation_api
    token = application.state.auth_store.create_session(identity)
    client.cookies.set("repofix_session", token)
    return token


def test_success_returns_summary_using_session_identity(preparation_api):
    _, client, identity, service, _ = preparation_api
    token = connect_session(preparation_api)

    response = client.post(PREPARATION_URL, json=VALID_INPUT)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "status", "repository", "commit_sha", "tree_sha",
        "entry_count", "issue_description",
    }
    assert body["status"] == "prepared"
    assert body["repository"]["id"] == 5
    assert body["repository"]["owner"]["id"] == identity.id
    assert body["commit_sha"] == "a" * 40
    assert body["tree_sha"] == "b" * 40
    assert body["entry_count"] == 1
    assert body["issue_description"] == "Login fails."
    assert token not in response.text
    assert response.headers["cache-control"] == "no-store"
    service.assert_called_once_with(identity, InvestigationInput(**VALID_INPUT))


@pytest.mark.parametrize("cookie", [None, "x" * 43])
def test_missing_or_unknown_session_stops_preparation(preparation_api, cookie):
    _, client, _, service, _ = preparation_api
    if cookie is not None:
        client.cookies.set("repofix_session", cookie)

    response = client.post(PREPARATION_URL, json=VALID_INPUT)

    assert response.status_code == 401
    service.assert_not_called()


@pytest.mark.parametrize("payload", [
    {},
    {"repository_id": 5, "repository_name": "example"},
    {**VALID_INPUT, "issue_description": ""},
    {**VALID_INPUT, "issue_description": "x" * 8001},
    {**VALID_INPUT, "repository_id": True},
    {**VALID_INPUT, "repository_name": ".."},
    {**VALID_INPUT, "approved": True},
    {**VALID_INPUT, "owner": "other-user"},
])
def test_invalid_input_stops_preparation(preparation_api, payload):
    _, client, _, service, _ = preparation_api
    connect_session(preparation_api)

    response = client.post(PREPARATION_URL, json=payload)

    assert response.status_code == 422
    service.assert_not_called()


@pytest.mark.parametrize("error,status", [
    (RepositorySelectionError("dummy-upstream-secret"), 400),
    (GitHubClientError("dummy-upstream-secret"), 502),
])
def test_service_failures_are_sanitized(preparation_api, error, status):
    _, client, _, service, _ = preparation_api
    connect_session(preparation_api)
    service.side_effect = error

    response = client.post(PREPARATION_URL, json=VALID_INPUT)

    assert response.status_code == status
    assert "dummy-upstream-secret" not in response.text
    assert isinstance(response.json()["detail"], str)
    assert response.headers["cache-control"] == "no-store"
    service.assert_called_once()


def test_expired_session_stops_preparation(preparation_api):
    _, client, _, service, clock = preparation_api
    connect_session(preparation_api)
    clock[0] = SESSION_IDLE_SECONDS

    assert client.post(PREPARATION_URL, json=VALID_INPUT).status_code == 401
    service.assert_not_called()


def test_logout_stops_subsequent_preparation(preparation_api):
    _, client, _, service, _ = preparation_api
    connect_session(preparation_api)

    response = client.post(
        "/api/auth/logout",
        headers={"Origin": BASE_URL, "X-RepoFix-CSRF": "1"},
    )

    assert response.status_code == 204
    assert client.post(PREPARATION_URL, json=VALID_INPUT).status_code == 401
    service.assert_not_called()


def test_missing_auth_store_returns_controlled_error(preparation_api):
    application, client, _, service, _ = preparation_api
    del application.state.auth_store

    response = client.post(PREPARATION_URL, json=VALID_INPUT)

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    service.assert_not_called()


def test_preparation_is_registered_in_served_openapi(preparation_api):
    _, client, _, service, _ = preparation_api

    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    operation = schema["paths"][PREPARATION_URL]["post"]
    assert {"200", "400", "401", "422", "502", "503"} <= set(
        operation["responses"]
    )
    model = schema["components"]["schemas"]["InvestigationInput"]
    assert set(model["properties"]) == {
        "repository_id", "repository_name", "issue_description",
    }
    assert model["additionalProperties"] is False
    service.assert_not_called()
