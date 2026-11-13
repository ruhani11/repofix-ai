"""Offline HTTP tests for the connected account's repository listing."""

from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import repositories
from app.auth_store import AuthStore, SESSION_IDLE_SECONDS
from app.config import Settings
from app.github_client import GitHubClientError
from app.github_oauth import GitHubIdentity
from app.github_repositories import PublicRepository, RepositoryPage
from app.main import create_app


BASE_URL = "http://127.0.0.1:8000"
LISTING_URL = "/api/repositories"


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
        pytest.fail("A listing API test attempted a real network request.")

    monkeypatch.setattr(
        httpx.HTTPTransport,
        "handle_request",
        block_network,
    )


@pytest.fixture
def listing_api(monkeypatch):
    identity = GitHubIdentity(
        id=123,
        login="demo-user",
        type="User",
    )

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

    page = RepositoryPage(
        account_login=identity.login,
        page=1,
        per_page=20,
        repositories=[repository],
        may_have_more=False,
        next_page=None,
        limit_reached=False,
    )

    tool = Mock(return_value=page)
    monkeypatch.setattr(
        repositories,
        "fetch_public_repositories",
        tool,
    )

    clock = [0.0]
    application = create_app(
        Settings(_env_file=None, environment="test")
    )
    application.state.auth_store = AuthStore(
        clock=lambda: clock[0]
    )

    with TestClient(application, base_url=BASE_URL) as client:
        yield application, client, identity, tool, clock


def connect_session(listing_api):
    application, client, identity, _, _ = listing_api
    token = application.state.auth_store.create_session(identity)
    client.cookies.set("repofix_session", token)
    return token


@pytest.mark.parametrize("cookie", [None, "x" * 43])
def test_missing_or_unknown_session_never_calls_github(
    listing_api, cookie
):
    _, client, _, tool, _ = listing_api

    if cookie is not None:
        client.cookies.set("repofix_session", cookie)

    response = client.get(LISTING_URL)

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Connect your GitHub account first."
    }
    assert response.headers["cache-control"] == "no-store"
    tool.assert_not_called()


def test_session_identity_and_default_pagination(listing_api):
    _, client, identity, tool, _ = listing_api
    token = connect_session(listing_api)

    response = client.get(LISTING_URL)

    assert response.status_code == 200
    body = response.json()

    assert body["account_login"] == identity.login
    assert (
        body["repositories"][0]["full_name"]
        == "demo-user/example"
    )
    assert (
        body["repositories"][0]["html_url"]
        == "https://github.com/demo-user/example"
    )
    assert body["next_page"] is None
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert token not in response.text

    tool.assert_called_once_with(
        identity,
        page=1,
        per_page=20,
    )


@pytest.mark.parametrize(
    "page,per_page",
    [(2, 5), (50, 20), (1, 1)],
)
def test_valid_pagination_is_forwarded(
    listing_api, page, per_page
):
    _, client, identity, tool, _ = listing_api
    connect_session(listing_api)

    tool.return_value = RepositoryPage(
        account_login=identity.login,
        page=page,
        per_page=per_page,
        repositories=[],
        may_have_more=False,
        next_page=None,
        limit_reached=False,
    )

    response = client.get(
        LISTING_URL,
        params={"page": page, "per_page": per_page},
    )

    assert response.status_code == 200
    assert response.json()["repositories"] == []
    assert response.json()["page"] == page
    assert response.json()["per_page"] == per_page

    tool.assert_called_once_with(
        identity,
        page=page,
        per_page=per_page,
    )


@pytest.mark.parametrize(
    "params",
    [
        {"page": 0},
        {"page": 51},
        {"page": "true"},
        {"per_page": 0},
        {"per_page": 21},
        {"per_page": "invalid"},
        {"username": "other-user"},
        {"account_login": "other-user"},
        {
            "repository_url":
            "https://github.com/other-user/example"
        },
    ],
)
def test_invalid_or_extra_query_parameters_stop_before_github(
    listing_api, params
):
    _, client, _, tool, _ = listing_api
    connect_session(listing_api)

    response = client.get(LISTING_URL, params=params)

    assert response.status_code == 422
    tool.assert_not_called()


def test_github_failure_is_sanitized(listing_api):
    _, client, _, tool, _ = listing_api
    connect_session(listing_api)
    tool.side_effect = GitHubClientError(
        "dummy-upstream-secret"
    )

    response = client.get(LISTING_URL)

    assert response.status_code == 502
    assert response.json() == {
        "detail":
        "Could not list public repositories from GitHub. Try later."
    }
    assert "dummy-upstream-secret" not in response.text
    assert response.headers["cache-control"] == "no-store"
    tool.assert_called_once()


def test_expired_session_blocks_listing(listing_api):
    _, client, _, tool, clock = listing_api
    connect_session(listing_api)
    clock[0] = SESSION_IDLE_SECONDS

    assert client.get(LISTING_URL).status_code == 401
    tool.assert_not_called()


def test_logout_blocks_subsequent_listing(listing_api):
    application, client, _, tool, _ = listing_api
    token = connect_session(listing_api)

    assert client.get(LISTING_URL).status_code == 200
    tool.reset_mock()

    response = client.post(
        "/api/auth/logout",
        headers={
            "Origin": BASE_URL,
            "X-RepoFix-CSRF": "1",
        },
    )

    assert response.status_code == 204
    assert (
        application.state.auth_store.get_identity(token)
        is None
    )
    assert client.get(LISTING_URL).status_code == 401
    tool.assert_not_called()


def test_missing_auth_store_returns_controlled_error(listing_api):
    application, client, _, tool, _ = listing_api
    del application.state.auth_store

    response = client.get(LISTING_URL)

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    tool.assert_not_called()


def test_openapi_exposes_only_pagination_inputs(listing_api):
    _, client, _, tool, _ = listing_api
    response = client.get("/openapi.json")

    assert response.status_code == 200
    operation = response.json()["paths"][LISTING_URL]["get"]

    assert {
        parameter["name"]
        for parameter in operation["parameters"]
    } == {"page", "per_page"}

    assert {
        "200", "401", "422", "502", "503"
    }.issubset(operation["responses"])

    tool.assert_not_called()