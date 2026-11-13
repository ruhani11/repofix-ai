"""Bounded GitHub OAuth requests; return identity without persisting tokens."""

import base64
import hashlib
import json
import re
from typing import Literal
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from app.config import Settings


AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
TOKEN_URL = "https://github.com/login/oauth/access_token"
USER_URL = "https://api.github.com/user"
MAX_OAUTH_RESPONSE_BYTES = 64 * 1024


class GitHubOAuthError(RuntimeError):
    """Safe authentication error without upstream bodies or credentials."""


class GitHubIdentity(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)

    id: int = Field(gt=0)
    login: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}$")
    type: Literal["User"]
    name: str | None = Field(default=None, max_length=256)


class _TokenResponse(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="ignore",
        hide_input_in_errors=True,
    )

    access_token: SecretStr = Field(exclude=True, repr=False)
    token_type: Literal["bearer"]
    scope: Literal[""]


def _require_configured(settings: Settings) -> None:
    if not settings.github_auth_configured:
        raise GitHubOAuthError("GitHub OAuth is not configured.")


def _validate_verifier(code_verifier: str) -> None:
    if not isinstance(code_verifier, str) or re.fullmatch(
        r"[A-Za-z0-9._~-]{43,128}", code_verifier
    ) is None:
        raise GitHubOAuthError("Invalid OAuth PKCE verifier.")


def build_authorization_url(
    settings: Settings, *, state: str, code_verifier: str
) -> str:
    """Caller must generate and store state/verifier in a browser-bound login."""
    _require_configured(settings)
    _validate_verifier(code_verifier)

    if not isinstance(state, str) or re.fullmatch(
        r"[A-Za-z0-9_-]{32,128}", state
    ) is None:
        raise GitHubOAuthError("Invalid OAuth state.")

    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")

    parameters = {
        "client_id": settings.github_client_id,
        "redirect_uri": settings.github_redirect_uri,
        "scope": "",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "prompt": "select_account",
    }

    return f"{AUTHORIZE_URL}?{urlencode(parameters)}"


def _request_json(
    method: Literal["GET", "POST"],
    url: str,
    *,
    transport: httpx.BaseTransport | None,
    headers: dict[str, str] | None = None,
    data: dict[str, str] | None = None,
) -> dict:
    request_headers = {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "User-Agent": "RepoFix-AI/0.1.0",
    }
    request_headers.update(headers or {})

    try:
        with httpx.Client(
            timeout=httpx.Timeout(15.0, connect=5.0),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        ) as client:
            with client.stream(
                method,
                url,
                headers=request_headers,
                data=data,
            ) as response:
                if response.status_code != 200:
                    raise GitHubOAuthError(
                        "GitHub authentication request failed."
                    )

                body = bytearray()

                for chunk in response.iter_bytes(chunk_size=4096):
                    if len(body) + len(chunk) > MAX_OAUTH_RESPONSE_BYTES:
                        raise GitHubOAuthError(
                            "GitHub authentication response is too large."
                        )

                    body.extend(chunk)

    except httpx.TimeoutException:
        raise GitHubOAuthError(
            "GitHub authentication timed out. Try again."
        ) from None

    except httpx.RequestError:
        raise GitHubOAuthError(
            "Could not connect to GitHub for authentication."
        ) from None

    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError, RecursionError):
        raise GitHubOAuthError(
            "GitHub returned invalid authentication data."
        ) from None

    if not isinstance(payload, dict):
        raise GitHubOAuthError(
            "GitHub returned invalid authentication data."
        )

    return payload


def verify_github_login(
    settings: Settings,
    *,
    code: str,
    code_verifier: str,
    transport: httpx.BaseTransport | None = None,
) -> GitHubIdentity:
    """Call only AFTER consuming a valid, unexpired, browser-bound OAuth state.

    Access/refresh tokens are not returned or persisted. This function alone
    does not validate state, establish a session, or authorize repository writes.
    """
    _require_configured(settings)
    _validate_verifier(code_verifier)

    if not isinstance(code, str) or re.fullmatch(
        r"[A-Za-z0-9_-]{1,256}", code
    ) is None:
        raise GitHubOAuthError("Invalid GitHub authorization code.")

    payload = _request_json(
        "POST",
        TOKEN_URL,
        transport=transport,
        data={
            "client_id": settings.github_client_id,
            "client_secret": settings.github_client_secret.get_secret_value(),
            "code": code,
            "redirect_uri": settings.github_redirect_uri,
            "code_verifier": code_verifier,
        },
    )

    if "error" in payload:
        raise GitHubOAuthError(
            "GitHub rejected the login. Start login again."
        )

    try:
        token_response = _TokenResponse.model_validate(payload)
    except ValidationError:
        raise GitHubOAuthError(
            "GitHub returned an unsupported OAuth token response."
        ) from None

    token = token_response.access_token.get_secret_value()

    if re.fullmatch(r"[A-Za-z0-9._~+/=-]{1,2048}", token) is None:
        raise GitHubOAuthError("GitHub returned an invalid access token.")

    profile = _request_json(
        "GET",
        USER_URL,
        transport=transport,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )

    try:
        return GitHubIdentity.model_validate(profile)
    except ValidationError:
        raise GitHubOAuthError(
            "GitHub returned an invalid account identity."
        ) from None