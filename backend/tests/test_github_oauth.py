"""Offline tests for GitHub OAuth using dummy credentials and MockTransport."""

from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.github_oauth import (
    AUTHORIZE_URL,
    MAX_OAUTH_RESPONSE_BYTES,
    TOKEN_URL,
    USER_URL,
    GitHubOAuthError,
    build_authorization_url,
    verify_github_login,
)


SECRET = "dummy-client-secret"
TOKEN = "dummy_access_token"
CODE = "dummy-code"
STATE = "s" * 43
VERIFIER = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"

TOKEN_DATA = {
    "access_token": TOKEN,
    "token_type": "bearer",
    "scope": "",
}

USER_DATA = {
    "id": 123,
    "login": "demo-user",
    "type": "User",
    "name": None,
}


@pytest.fixture(autouse=True)
def isolate_environment_and_block_real_network(monkeypatch):
    for name in (
        "REPOFIX_APP_NAME",
        "REPOFIX_ENVIRONMENT",
        "REPOFIX_GITHUB_CLIENT_ID",
        "REPOFIX_GITHUB_CLIENT_SECRET",
        "REPOFIX_GITHUB_REDIRECT_URI",
    ):
        monkeypatch.delenv(name, raising=False)

    def blocked(*args, **kwargs):
        pytest.fail("OAuth unit tests must not use real network requests.")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        environment="test",
        github_client_id="dummy-client-id",
        github_client_secret=SECRET,
    )


def mock_login(settings, *, stage=None, response=None, calls=None):
    requests = calls if calls is not None else []

    def handler(request):
        requests.append(request)

        if str(request.url) == TOKEN_URL:
            return (
                response
                if stage == "token"
                else httpx.Response(200, json=TOKEN_DATA)
            )

        if str(request.url) == USER_URL:
            return (
                response
                if stage == "user"
                else httpx.Response(200, json=USER_DATA)
            )

        pytest.fail("Unexpected OAuth request destination.")

    return verify_github_login(
        settings,
        code=CODE,
        code_verifier=VERIFIER,
        transport=httpx.MockTransport(handler),
    )


def test_authorization_url_uses_pkce_and_no_repository_permissions(settings):
    url = build_authorization_url(
        settings,
        state=STATE,
        code_verifier=VERIFIER,
    )

    parsed = urlsplit(url)
    query = parse_qs(parsed.query, keep_blank_values=True)

    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == AUTHORIZE_URL
    assert query["client_id"] == [settings.github_client_id]
    assert query["redirect_uri"] == [settings.github_redirect_uri]
    assert query["scope"] == [""]
    assert query["state"] == [STATE]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == [
        "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    ]

    assert SECRET not in url
    assert VERIFIER not in url


def test_login_exchanges_code_and_returns_only_identity(settings):
    calls = []

    response = httpx.Response(
        200,
        json={
            **TOKEN_DATA,
            "refresh_token": "dummy-refresh-token",
        },
    )

    identity = mock_login(
        settings,
        stage="token",
        response=response,
        calls=calls,
    )

    assert len(calls) == 2
    token_request, user_request = calls

    assert str(token_request.url) == (
        "https://github.com/login/oauth/access_token"
    )
    assert token_request.method == "POST"
    assert token_request.url.query == b""
    assert "authorization" not in token_request.headers

    assert parse_qs(token_request.content.decode()) == {
        "client_id": [settings.github_client_id],
        "client_secret": [SECRET],
        "code": [CODE],
        "redirect_uri": [settings.github_redirect_uri],
        "code_verifier": [VERIFIER],
    }

    assert str(user_request.url) == "https://api.github.com/user"
    assert user_request.method == "GET"
    assert user_request.headers["authorization"] == f"Bearer {TOKEN}"

    assert identity.model_dump() == USER_DATA
    assert TOKEN not in identity.model_dump_json()
    assert "dummy-refresh-token" not in identity.model_dump_json()

    with pytest.raises(ValidationError):
        identity.login = "another-user"


def test_unconfigured_oauth_is_rejected_before_network():
    settings = Settings(_env_file=None)

    with pytest.raises(GitHubOAuthError, match="not configured"):
        build_authorization_url(
            settings,
            state=STATE,
            code_verifier=VERIFIER,
        )

    with pytest.raises(GitHubOAuthError, match="not configured"):
        verify_github_login(
            settings,
            code=CODE,
            code_verifier=VERIFIER,
        )


@pytest.mark.parametrize("state", ["short", "s" * 129])
def test_invalid_state_is_rejected(settings, state):
    with pytest.raises(GitHubOAuthError, match="state"):
        build_authorization_url(
            settings,
            state=state,
            code_verifier=VERIFIER,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"code": ""},
        {"code": "bad\ncode"},
        {"code_verifier": "short"},
        {"code_verifier": "v" * 129},
    ],
)
def test_invalid_exchange_input_is_rejected_before_network(settings, overrides):
    values = {
        "code": CODE,
        "code_verifier": VERIFIER,
        **overrides,
    }

    with pytest.raises(GitHubOAuthError):
        verify_github_login(settings, **values)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "error": "bad_verification_code",
            "error_description": SECRET,
        },
        {**TOKEN_DATA, "scope": "repo"},
        {"access_token": TOKEN, "token_type": "bearer"},
        {**TOKEN_DATA, "token_type": "unsupported"},
        {**TOKEN_DATA, "access_token": ""},
        {**TOKEN_DATA, "access_token": "bad\r\ntoken"},
        {**TOKEN_DATA, "access_token": "a" * 2049},
    ],
)
def test_bad_token_response_stops_before_user_request(settings, payload):
    calls = []

    with pytest.raises(GitHubOAuthError) as error:
        mock_login(
            settings,
            stage="token",
            response=httpx.Response(200, json=payload),
            calls=calls,
        )

    assert len(calls) == 1
    assert SECRET not in str(error.value)
    assert TOKEN not in str(error.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": True},
        {"id": 0},
        {"login": "../user"},
        {"type": "Organization"},
    ],
)
def test_invalid_identity_is_rejected(settings, overrides):
    with pytest.raises(GitHubOAuthError, match="identity"):
        mock_login(
            settings,
            stage="user",
            response=httpx.Response(
                200,
                json={**USER_DATA, **overrides},
            ),
        )


@pytest.mark.parametrize("stage", ["token", "user"])
@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_http_failures_and_redirects_are_rejected(settings, stage, status):
    calls = []

    response = httpx.Response(
        status,
        text=SECRET,
        headers={"Location": "https://example.com"},
    )

    with pytest.raises(GitHubOAuthError) as error:
        mock_login(
            settings,
            stage=stage,
            response=response,
            calls=calls,
        )

    assert len(calls) == (1 if stage == "token" else 2)
    assert SECRET not in str(error.value)


@pytest.mark.parametrize("stage", ["token", "user"])
@pytest.mark.parametrize("body", ["not-json", "[]"])
def test_invalid_json_is_rejected(settings, stage, body):
    with pytest.raises(
        GitHubOAuthError,
        match="invalid authentication data",
    ):
        mock_login(
            settings,
            stage=stage,
            response=httpx.Response(200, text=body),
        )


@pytest.mark.parametrize("stage", ["token", "user"])
def test_oversized_response_is_rejected(settings, stage):
    response = httpx.Response(
        200,
        content=b"x" * (MAX_OAUTH_RESPONSE_BYTES + 1),
    )

    with pytest.raises(GitHubOAuthError, match="too large"):
        mock_login(
            settings,
            stage=stage,
            response=response,
        )


@pytest.mark.parametrize("stage", ["token", "user"])
@pytest.mark.parametrize(
    "exception",
    [httpx.ReadTimeout, httpx.ConnectError],
)
def test_network_errors_do_not_expose_credentials(settings, stage, exception):
    def handler(request):
        if stage == "user" and str(request.url) == TOKEN_URL:
            return httpx.Response(200, json=TOKEN_DATA)

        raise exception(SECRET, request=request)

    with pytest.raises(GitHubOAuthError) as error:
        verify_github_login(
            settings,
            code=CODE,
            code_verifier=VERIFIER,
            transport=httpx.MockTransport(handler),
        )

    assert SECRET not in str(error.value)
    assert TOKEN not in str(error.value)