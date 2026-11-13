"""Offline HTTP tests: dummy settings, mocked GitHub, no live network."""

from urllib.parse import parse_qs, urlsplit
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import auth as auth_api
from app.auth_store import (
    LOGIN_TTL_SECONDS,
    SESSION_IDLE_SECONDS,
    AuthStore,
    AuthStoreError,
)
from app.config import Settings
from app.github_oauth import GitHubIdentity, GitHubOAuthError
from app.main import create_app


BASE_URL = "http://127.0.0.1:8000"
LOGIN = "/api/auth/github/login"
CALLBACK = "/api/auth/github/callback"
ME = "/api/auth/me"
LOGOUT = "/api/auth/logout"


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
        pytest.fail("An auth HTTP test attempted a real network request.")

    monkeypatch.setattr(
        httpx.HTTPTransport, "handle_request", block_network
    )
    monkeypatch.setattr(
        auth_api, "verify_github_login", block_network
    )


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        environment="test",
        github_client_id="dummy-client-id",
        github_client_secret="dummy-client-secret",
    )


@pytest.fixture
def identity():
    return GitHubIdentity(
        id=123,
        login="demo-user",
        type="User",
        name="Demo User",
    )


@pytest.fixture
def exchange(monkeypatch, identity):
    mocked = Mock(return_value=identity)
    monkeypatch.setattr(auth_api, "verify_github_login", mocked)
    return mocked


@pytest.fixture
def clock():
    return [0.0]


@pytest.fixture
def application(settings, clock):
    app = create_app(settings)
    app.state.auth_store = AuthStore(clock=lambda: clock[0])
    return app


@pytest.fixture
def client(application):
    with TestClient(
        application,
        base_url=BASE_URL,
        follow_redirects=False,
    ) as browser:
        yield browser


def start_login(client):
    response = client.get(LOGIN)
    assert response.status_code == 302

    location = urlsplit(response.headers["location"])
    assert location.scheme == "https"
    assert location.netloc == "github.com"
    assert location.path == "/login/oauth/authorize"

    return parse_qs(location.query)["state"][0]


def finish_login(client):
    state = start_login(client)
    response = client.get(
        CALLBACK,
        params={"state": state, "code": "dummy-code"},
    )
    assert response.status_code == 303
    assert response.headers["location"] == ME
    return response


def test_app_registers_auth_and_owns_its_store(application, settings):
    paths = application.openapi()["paths"]

    assert {LOGIN, CALLBACK, ME, LOGOUT}.issubset(paths)
    assert "/api/health" in paths
    assert "/api/repositories/inspect" in paths

    another_app = create_app(settings)
    assert application.state.auth_store is not another_app.state.auth_store


def test_unconfigured_login_returns_503():
    app = create_app(
        Settings(_env_file=None, environment="test")
    )
    with TestClient(app, base_url=BASE_URL) as browser:
        assert browser.get(LOGIN).status_code == 503


@pytest.mark.parametrize(
    "headers",
    [
        {"Sec-Fetch-Site": "cross-site"},
        {"Origin": "https://attacker.example"},
        [("Origin", BASE_URL), ("Origin", BASE_URL)],
    ],
)
def test_untrusted_login_is_blocked(client, headers):
    response = client.get(LOGIN, headers=headers)

    assert response.status_code == 403
    assert "repofix_login" not in client.cookies


def test_login_callback_and_identity(
    client, application, settings, identity, exchange
):
    state = start_login(client)
    browser_token = client.cookies["repofix_login"]

    def verify_after_state_consumption(
        config, *, code, code_verifier
    ):
        assert config is settings
        assert code == "dummy-code"
        assert len(code_verifier) >= 43

        # The callback must consume state before calling GitHub.
        with pytest.raises(AuthStoreError):
            application.state.auth_store.consume_login(
                state=state,
                browser_token=browser_token,
            )

        return identity

    exchange.side_effect = verify_after_state_consumption

    response = client.get(
        CALLBACK,
        params={"state": state, "code": "dummy-code"},
    )

    assert response.status_code == 303
    assert response.headers["location"] == ME
    exchange.assert_called_once()
    assert "repofix_login" not in client.cookies

    session_token = client.cookies["repofix_session"]
    cookie = response.headers.get_list("set-cookie")[0]

    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Path=/" in cookie

    response = client.get(ME)

    assert response.status_code == 200
    assert response.json() == identity.model_dump()
    assert session_token not in response.text
    assert "dummy-client-secret" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"


@pytest.mark.parametrize(
    "case",
    [
        "missing-state",
        "wrong-state",
        "missing-cookie",
        "expired",
    ],
)
def test_invalid_callback_never_exchanges_code(client, clock, case):
    state = start_login(client)

    if case == "missing-state":
        params = {"code": "dummy-code"}
    else:
        params = {
            "state": "x" * 43 if case == "wrong-state" else state,
            "code": "dummy-code",
        }

    if case == "missing-cookie":
        client.cookies.clear()

    if case == "expired":
        clock[0] = LOGIN_TTL_SECONDS

    response = client.get(CALLBACK, params=params)

    assert response.status_code == 400
    assert "repofix_session" not in client.cookies

    if case == "wrong-state":
        assert "repofix_login" in client.cookies


@pytest.mark.parametrize("field", ["state", "code", "error"])
def test_duplicate_callback_parameters_are_rejected(client, field):
    state = start_login(client)
    params = [
        ("state", state),
        ("code", "dummy-code"),
    ]

    if field == "error":
        params.append(("error", "access_denied"))

    params.append((field, "duplicate"))

    assert client.get(CALLBACK, params=params).status_code == 400


def test_oversized_callback_is_rejected(client):
    response = client.get(
        CALLBACK,
        params={"state": "x" * 4097},
    )
    assert response.status_code == 400


@pytest.mark.parametrize(
    "params",
    [{"error": "access_denied"}, {}],
)
def test_cancelled_or_incomplete_login_clears_binding(client, params):
    state = start_login(client)

    response = client.get(
        CALLBACK,
        params={"state": state, **params},
    )

    assert response.status_code == 400
    assert "repofix_login" not in client.cookies
    assert "repofix_session" not in client.cookies


def test_callback_replay_is_rejected(client, exchange):
    state = start_login(client)
    params = {"state": state, "code": "dummy-code"}

    assert client.get(CALLBACK, params=params).status_code == 303
    assert client.get(CALLBACK, params=params).status_code == 400
    exchange.assert_called_once()


@pytest.mark.parametrize("failure", ["github", "session"])
def test_failed_relogin_preserves_session_and_hides_errors(
    client, application, exchange, monkeypatch, failure
):
    finish_login(client)
    old_token = client.cookies["repofix_session"]
    state = start_login(client)

    if failure == "github":
        exchange.side_effect = GitHubOAuthError(
            "dummy-upstream-secret"
        )
        expected = 502
    else:
        monkeypatch.setattr(
            application.state.auth_store,
            "create_session",
            Mock(
                side_effect=AuthStoreError(
                    "dummy-upstream-secret"
                )
            ),
        )
        expected = 503

    response = client.get(
        CALLBACK,
        params={"state": state, "code": "dummy-code"},
    )

    assert response.status_code == expected
    assert "dummy-upstream-secret" not in response.text
    assert "repofix_login" not in client.cookies
    assert client.cookies["repofix_session"] == old_token
    assert client.get(ME).status_code == 200


def test_successful_relogin_rotates_session(
    client, application, exchange
):
    finish_login(client)
    old_token = client.cookies["repofix_session"]

    finish_login(client)

    assert client.cookies["repofix_session"] != old_token
    assert application.state.auth_store.get_identity(old_token) is None
    assert client.get(ME).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Origin": BASE_URL},
        {"X-RepoFix-CSRF": "1"},
        {"Origin": BASE_URL, "X-RepoFix-CSRF": "wrong"},
        {
            "Origin": "https://attacker.example",
            "X-RepoFix-CSRF": "1",
        },
        [
            ("Origin", BASE_URL),
            ("Origin", BASE_URL),
            ("X-RepoFix-CSRF", "1"),
        ],
        [
            ("Origin", BASE_URL),
            ("X-RepoFix-CSRF", "1"),
            ("X-RepoFix-CSRF", "1"),
        ],
    ],
)
def test_invalid_logout_does_not_revoke_session(
    client, exchange, headers
):
    finish_login(client)

    assert client.post(LOGOUT, headers=headers).status_code == 403
    assert client.get(ME).status_code == 200


@pytest.mark.parametrize(
    "origin",
    [BASE_URL, "http://127.0.0.1:5173"],
)
def test_logout_revokes_session_and_is_idempotent(
    client, application, exchange, origin
):
    finish_login(client)
    old_token = client.cookies["repofix_session"]
    headers = {
        "Origin": origin,
        "X-RepoFix-CSRF": "1",
    }

    response = client.post(LOGOUT, headers=headers)

    assert response.status_code == 204
    assert response.content == b""
    assert "repofix_session" not in client.cookies
    assert application.state.auth_store.get_identity(old_token) is None
    assert client.get(ME).status_code == 401
    assert client.post(LOGOUT, headers=headers).status_code == 204
    assert client.get(LOGOUT).status_code == 405


def test_session_idle_expiry_requires_login(client, clock, exchange):
    finish_login(client)
    clock[0] = SESSION_IDLE_SECONDS

    response = client.get(ME)

    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"


def test_production_uses_secure_host_cookies_and_exact_origin(exchange):
    origin = "https://repofix.example"
    settings = Settings(
        _env_file=None,
        environment="production",
        github_client_id="dummy-client-id",
        github_client_secret="dummy-client-secret",
        github_redirect_uri=origin + CALLBACK,
    )

    with TestClient(
        create_app(settings),
        base_url=origin,
        follow_redirects=False,
    ) as browser:
        state = start_login(browser)
        assert "__Host-repofix_login" in browser.cookies

        response = browser.get(
            CALLBACK,
            params={"state": state, "code": "dummy-code"},
        )

        assert response.status_code == 303
        cookie = response.headers.get_list("set-cookie")[0]

        assert cookie.startswith("__Host-repofix_session=")
        assert "Secure" in cookie
        assert "HttpOnly" in cookie
        assert "Path=/" in cookie
        assert "Domain=" not in cookie
        assert browser.get(ME).status_code == 200

        response = browser.post(
            LOGOUT,
            headers={
                "Origin": "http://127.0.0.1:5173",
                "X-RepoFix-CSRF": "1",
            },
        )
        assert response.status_code == 403

        response = browser.post(
            LOGOUT,
            headers={
                "Origin": origin,
                "X-RepoFix-CSRF": "1",
            },
        )
        assert response.status_code == 204