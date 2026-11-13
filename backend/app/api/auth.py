"""GitHub OAuth routes; require an app-owned AuthStore before use."""

from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

from app.auth_store import LOGIN_TTL_SECONDS, SESSION_TTL_SECONDS, AuthStore, AuthStoreError
from app.config import Settings
from app.github_oauth import (
    GitHubIdentity,
    GitHubOAuthError,
    build_authorization_url,
    verify_github_login,
)


router = APIRouter(prefix="/api/auth", tags=["auth"])
AUTH_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


def _context(request: Request) -> tuple[Settings, AuthStore]:
    settings = getattr(request.app.state, "settings", None)
    store = getattr(request.app.state, "auth_store", None)
    if not isinstance(settings, Settings) or not isinstance(store, AuthStore):
        raise HTTPException(
            status_code=503,
            detail="Authentication services are not initialized.",
            headers=AUTH_HEADERS,
        )
    return settings, store


def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status, headers=AUTH_HEADERS)


def _secure(settings: Settings) -> bool:
    return urlsplit(settings.github_redirect_uri).scheme == "https"


def _cookie_name(settings: Settings, kind: str) -> str:
    prefix = "__Host-" if _secure(settings) else ""
    return f"{prefix}repofix_{kind}"


def _set_cookie(response: Response, settings: Settings, kind: str, token: str, ttl: int) -> None:
    response.set_cookie(
        key=_cookie_name(settings, kind),
        value=token,
        max_age=ttl,
        path="/",
        secure=_secure(settings),
        httponly=True,
        samesite="lax",
    )


def _delete_cookie(response: Response, settings: Settings, kind: str) -> None:
    response.delete_cookie(
        key=_cookie_name(settings, kind),
        path="/",
        secure=_secure(settings),
        httponly=True,
        samesite="lax",
    )


def _origin_key(value: str) -> tuple[str, str, int] | None:
    if len(value) > 2048 or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        return None
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            return None
        port = parsed.port
        if port is not None and port <= 0:
            return None
    except ValueError:
        return None
    return parsed.scheme, parsed.hostname.lower(), port or (443 if parsed.scheme == "https" else 80)


def _trusted_origin(origin: str, settings: Settings) -> bool:
    callback = urlsplit(settings.github_redirect_uri)
    configured = _origin_key(f"{callback.scheme}://{callback.netloc}")
    supplied = _origin_key(origin)
    if supplied is None or configured is None:
        return False
    if supplied == configured:
        return True
    # Vite proxy during local development; same host/scheme, port 5173 only.
    return (
        settings.environment in {"development", "test"}
        and configured[1] in {"127.0.0.1", "localhost", "::1"}
        and supplied == (configured[0], configured[1], 5173)
    )


def _workbench_url(settings: Settings) -> str | None:
    """Local frontend only; never accept a caller-supplied redirect URL."""
    callback = urlsplit(settings.github_redirect_uri)
    if (
        settings.environment not in {"development", "test"} or callback.scheme != "http"
        or callback.hostname not in {"127.0.0.1", "localhost", "::1"}
    ):
        return None
    host = "[::1]" if callback.hostname == "::1" else callback.hostname
    return f"http://{host}:5173/"


def require_github_identity(request: Request) -> GitHubIdentity:
    """Reusable dependency for future repository/investigation endpoints."""
    settings, store = _context(request)
    token = request.cookies.get(_cookie_name(settings, "session"))
    identity = store.get_identity(token)
    if identity is None:
        raise HTTPException(
            status_code=401,
            detail="Connect your GitHub account first.",
            headers=AUTH_HEADERS,
        )
    return identity


@router.get(
    "/github/login",
    response_class=RedirectResponse,
    status_code=302,
    summary="Connect GitHub — open in browser",
    description=(
        "Open [Connect your GitHub account](/api/auth/github/login) "
        "as a browser navigation to start login. "
        "This endpoint redirects to GitHub; Swagger's Try it out / Execute "
        "uses fetch and cannot complete the external OAuth navigation. "
        "After login, test GET /api/auth/me in the same browser. "
        "The local frontend uses ?flow=workbench to return to the app."
    ),
    responses={
        403: {"description": "Untrusted login origin."},
        503: {"description": "OAuth is unconfigured or temporarily unavailable."},
    },
)
def github_login(request: Request) -> Response:
    settings, store = _context(request)
    if not settings.github_auth_configured:
        return _error(503, "GitHub OAuth is not configured.")
    if request.headers.get("sec-fetch-site") == "cross-site":
        return _error(403, "Start login from RepoFix AI.")
    origins = request.headers.getlist("origin")
    if origins and (len(origins) != 1 or not _trusted_origin(origins[0], settings)):
        return _error(403, "Start login from RepoFix AI.")
    flow = request.query_params.getlist("flow")
    if flow and (flow != ["workbench"] or _workbench_url(settings) is None):
        return _error(400, "Invalid GitHub login flow.")

    try:
        attempt = store.start_login(return_to_workbench=bool(flow))
        url = build_authorization_url(
            settings, state=attempt.state, code_verifier=attempt.code_verifier
        )
    except (AuthStoreError, GitHubOAuthError):
        return _error(503, "Could not start GitHub login. Try later.")

    response = RedirectResponse(url, status_code=302, headers=AUTH_HEADERS)
    _set_cookie(response, settings, "login", attempt.browser_token, LOGIN_TTL_SECONDS)
    return response


@router.get("/github/callback")
def github_callback(request: Request) -> Response:
    settings, store = _context(request)
    if not settings.github_auth_configured:
        return _error(503, "GitHub OAuth is not configured.")
    if len(request.scope.get("query_string", b"")) > 4096:
        return _error(400, "Invalid GitHub callback.")
    params = request.query_params
    if any(len(params.getlist(name)) > 1 for name in ("state", "code", "error")):
        return _error(400, "Invalid GitHub callback.")

    try:
        attempt = store.consume_login_attempt(
            state=params.get("state"),
            browser_token=request.cookies.get(_cookie_name(settings, "login")),
        )
    except AuthStoreError:
        # Do not clear a legitimate pending login on an unrelated bad callback.
        return _error(400, "Invalid or expired login. Start login again.")

    workbench = _workbench_url(settings) if attempt.return_to_workbench else None

    def login_error(status: int, detail: str, outcome: str) -> Response:
        if workbench is not None:
            return RedirectResponse(f"{workbench}?github_login={outcome}", status_code=303, headers=AUTH_HEADERS)
        return _error(status, detail)

    if "error" in params or not params.get("code"):
        response = login_error(400, "GitHub login was cancelled or incomplete. Start again.", "cancelled")
        _delete_cookie(response, settings, "login")
        return response

    try:
        identity = verify_github_login(
            settings, code=params["code"], code_verifier=attempt.code_verifier
        )
        token = store.create_session(
            identity,
            previous_token=request.cookies.get(_cookie_name(settings, "session")),
        )
    except GitHubOAuthError:
        response = login_error(502, "Could not verify GitHub login. Start login again.", "failed")
    except AuthStoreError:
        response = login_error(503, "Could not create a session. Try later.", "failed")
    else:
        response = RedirectResponse(workbench or "/api/auth/me", status_code=303, headers=AUTH_HEADERS)
        _set_cookie(response, settings, "session", token, SESSION_TTL_SECONDS)

    _delete_cookie(response, settings, "login")
    return response


@router.get("/me", response_model=GitHubIdentity)
def current_user(request: Request, response: Response) -> GitHubIdentity:
    response.headers.update(AUTH_HEADERS)
    return require_github_identity(request)


@router.post("/logout", status_code=204)
def logout(
    request: Request,
    csrf_header: Annotated[str | None, Header(alias="X-RepoFix-CSRF")] = None,
) -> Response:
    settings, store = _context(request)
    origins = request.headers.getlist("origin")
    csrf_headers = request.headers.getlist("x-repofix-csrf")
    if (
        len(origins) != 1
        or not _trusted_origin(origins[0], settings)
        or len(csrf_headers) != 1
        or csrf_header != "1"
    ):
        return _error(403, "Logout requires a trusted Origin and X-RepoFix-CSRF: 1.")

    store.revoke_session(request.cookies.get(_cookie_name(settings, "session")))
    # Clear the pending-login cookie too; its server record expires at its own TTL.
    response = Response(status_code=204, headers=AUTH_HEADERS)
    _delete_cookie(response, settings, "session")
    _delete_cookie(response, settings, "login")
    return response
