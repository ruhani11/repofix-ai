"""Offline HTTP tests for session-protected repository selection."""

from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import repositories
from app.auth_store import AuthStore, SESSION_IDLE_SECONDS
from app.config import Settings
from app.github_client import GitHubClientError
from app.github_oauth import GitHubIdentity
from app.github_repositories import PublicRepository
from app.github_selection import RepositorySelectionError, RepositorySelectionInput
from app.main import create_app


BASE_URL = "http://127.0.0.1:8000"
SELECTION_URL = "/api/repositories/select"
VALID_SELECTION = {"repository_id": 5, "repository_name": "example"}


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
        pytest.fail("A selection API test attempted a real network request.")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", block_network)


@pytest.fixture
def selection_api(monkeypatch):
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
    tool = Mock(return_value=repository)
    monkeypatch.setattr(repositories, "fetch_selected_repository", tool)

    clock = [0.0]
    application = create_app(Settings(_env_file=None, environment="test"))
    application.state.auth_store = AuthStore(clock=lambda: clock[0])

    with TestClient(application, base_url=BASE_URL) as client:
        yield application, client, identity, tool, clock


def connect_session(selection_api):
    application, client, identity, _, _ = selection_api
    token = application.state.auth_store.create_session(identity)
    client.cookies.set("repofix_session", token)
    return token


@pytest.mark.parametrize("cookie", [None, "x" * 43])
def test_missing_or_unknown_session_never_calls_github(selection_api, cookie):
    _, client, _, tool, _ = selection_api
    if cookie is not None:
        client.cookies.set("repofix_session", cookie)

    response = client.post(SELECTION_URL, json=VALID_SELECTION)

    assert response.status_code == 401
    assert response.json() == {"detail": "Connect your GitHub account first."}
    assert response.headers["cache-control"] == "no-store"
    tool.assert_not_called()


def test_valid_selection_uses_session_identity(selection_api):
    _, client, identity, tool, _ = selection_api
    token = connect_session(selection_api)

    response = client.post(SELECTION_URL, json=VALID_SELECTION)

    assert response.status_code == 200
    assert response.json()["id"] == 5
    assert response.json()["full_name"] == "demo-user/example"
    assert response.json()["owner"]["id"] == identity.id
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert token not in response.text
    tool.assert_called_once_with(
        identity, RepositorySelectionInput(**VALID_SELECTION)
    )


@pytest.mark.parametrize("payload", [
    {},
    {"repository_id": 0, "repository_name": "example"},
    {"repository_id": True, "repository_name": "example"},
    {"repository_id": "5", "repository_name": "example"},
    {"repository_id": 5, "repository_name": ".."},
    {"repository_id": 5, "repository_name": "../other"},
    {**VALID_SELECTION, "owner": "other-user"},
    {**VALID_SELECTION, "repository_url": "https://example.com"},
])
def test_invalid_body_stops_before_github(selection_api, payload):
    _, client, _, tool, _ = selection_api
    connect_session(selection_api)

    response = client.post(SELECTION_URL, json=payload)

    assert response.status_code == 422
    tool.assert_not_called()


@pytest.mark.parametrize("error,status,detail", [
    (
        RepositorySelectionError("dummy-upstream-secret"),
        400,
        "Repository selection is unavailable or changed. "
        "Refresh the repository list and select again.",
    ),
    (
        GitHubClientError("dummy-upstream-secret"),
        502,
        "Could not verify the repository with GitHub. Try later.",
    ),
])
def test_known_failures_are_sanitized(selection_api, error, status, detail):
    _, client, _, tool, _ = selection_api
    connect_session(selection_api)
    tool.side_effect = error

    response = client.post(SELECTION_URL, json=VALID_SELECTION)

    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert "dummy-upstream-secret" not in response.text
    assert response.headers["cache-control"] == "no-store"
    tool.assert_called_once()


def test_expired_session_blocks_selection(selection_api):
    _, client, _, tool, clock = selection_api
    connect_session(selection_api)
    clock[0] = SESSION_IDLE_SECONDS

    response = client.post(SELECTION_URL, json=VALID_SELECTION)

    assert response.status_code == 401
    tool.assert_not_called()


def test_logout_blocks_subsequent_selection(selection_api):
    application, client, _, tool, _ = selection_api
    token = connect_session(selection_api)

    response = client.post(
        "/api/auth/logout",
        headers={"Origin": BASE_URL, "X-RepoFix-CSRF": "1"},
    )

    assert response.status_code == 204
    assert application.state.auth_store.get_identity(token) is None
    assert client.post(SELECTION_URL, json=VALID_SELECTION).status_code == 401
    tool.assert_not_called()


def test_missing_auth_store_returns_controlled_error(selection_api):
    application, client, _, tool, _ = selection_api
    del application.state.auth_store

    response = client.post(SELECTION_URL, json=VALID_SELECTION)

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    tool.assert_not_called()


def test_selection_is_registered_in_openapi(selection_api):
    _, client, _, tool, _ = selection_api

    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    operation = schema["paths"][SELECTION_URL]["post"]
    assert {"200", "400", "401", "422", "502", "503"} <= set(
        operation["responses"]
    )
    model = schema["components"]["schemas"]["RepositorySelectionInput"]
    assert set(model["properties"]) == {"repository_id", "repository_name"}
    assert model["additionalProperties"] is False
    tool.assert_not_called()
