"""Bounded in-memory OAuth/session store for one backend process."""

import hashlib
import hmac
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock

from app.github_oauth import GitHubIdentity


LOGIN_TTL_SECONDS = 600
SESSION_TTL_SECONDS = 3600
SESSION_IDLE_SECONDS = 900
MAX_PENDING_LOGINS = 1000
MAX_SESSIONS = 1000


class AuthStoreError(RuntimeError):
    """Safe error without login or session credentials."""


@dataclass(frozen=True)
class LoginAttempt:
    state: str = field(repr=False)
    browser_token: str = field(repr=False)
    code_verifier: str = field(repr=False)


@dataclass(frozen=True)
class _PendingLogin:
    browser_digest: str
    code_verifier: str = field(repr=False)
    expires_at: float
    return_to_workbench: bool


@dataclass
class _Session:
    identity: GitHubIdentity
    expires_at: float
    last_seen: float


def _token_digest(token: str | None) -> str | None:
    if not isinstance(token, str) or re.fullmatch(r"[A-Za-z0-9_-]{43}", token) is None:
        return None
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _new_token(records: dict) -> tuple[str, str]:
    for _ in range(5):
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        if digest not in records:
            return token, digest
    raise AuthStoreError("Could not create authentication credentials. Try again.")


class AuthStore:
    """Share one instance within an app. Restart clears all logins/sessions.

    Routes must supply tokens from cookies, consume login state before OAuth
    exchange, and create sessions only after GitHub identity verification.
    Cookie flags, CSRF checks and request rate limits belong to the HTTP layer.
    This store cannot be shared across worker processes or server instances.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = Lock()
        self._pending: dict[str, _PendingLogin] = {}
        self._sessions: dict[str, _Session] = {}

    def _prune(self, now: float) -> None:
        """Caller holds the lock; capacity limits keep these scans bounded."""
        for key, login in list(self._pending.items()):
            if now >= login.expires_at:
                del self._pending[key]

        for key, session in list(self._sessions.items()):
            if (
                now >= session.expires_at
                or now - session.last_seen >= SESSION_IDLE_SECONDS
            ):
                del self._sessions[key]

    def start_login(self, *, return_to_workbench: bool = False) -> LoginAttempt:
        with self._lock:
            now = self._clock()
            self._prune(now)
            if len(self._pending) >= MAX_PENDING_LOGINS:
                raise AuthStoreError("Too many pending logins. Try later.")

            state, state_digest = _new_token(self._pending)
            browser_token = secrets.token_urlsafe(32)
            code_verifier = secrets.token_urlsafe(32)
            browser_digest = _token_digest(browser_token)
            self._pending[state_digest] = _PendingLogin(
                browser_digest=browser_digest,
                code_verifier=code_verifier,
                expires_at=now + LOGIN_TTL_SECONDS,
                return_to_workbench=return_to_workbench,
            )
            return LoginAttempt(state, browser_token, code_verifier)

    def consume_login(self, *, state: str | None, browser_token: str | None) -> str:
        """Return verifier once; a wrong browser cannot consume valid state."""
        return self.consume_login_attempt(state=state, browser_token=browser_token).code_verifier

    def consume_login_attempt(self, *, state: str | None, browser_token: str | None) -> _PendingLogin:
        """Consume the verifier and return preference bound to this login."""
        state_digest = _token_digest(state)
        browser_digest = _token_digest(browser_token)
        if state_digest is None or browser_digest is None:
            raise AuthStoreError("Invalid or expired login. Start login again.")

        with self._lock:
            self._prune(self._clock())
            login = self._pending.get(state_digest)
            if login is None or not hmac.compare_digest(
                login.browser_digest, browser_digest
            ):
                raise AuthStoreError("Invalid or expired login. Start login again.")

            del self._pending[state_digest]
            return login

    def create_session(
        self, identity: GitHubIdentity, *, previous_token: str | None = None
    ) -> str:
        """Issue a fresh session and replace the browser's old session atomically."""
        if not isinstance(identity, GitHubIdentity):
            raise AuthStoreError("A verified GitHub identity is required.")

        with self._lock:
            now = self._clock()
            self._prune(now)
            previous_digest = _token_digest(previous_token)
            replacing_existing = previous_digest in self._sessions
            if len(self._sessions) >= MAX_SESSIONS and not replacing_existing:
                raise AuthStoreError("Too many active sessions. Try later.")

            token, digest = _new_token(self._sessions)
            if previous_digest is not None:
                self._sessions.pop(previous_digest, None)
            self._sessions[digest] = _Session(
                identity=identity,
                expires_at=now + SESSION_TTL_SECONDS,
                last_seen=now,
            )
            return token

    def get_identity(self, session_token: str | None) -> GitHubIdentity | None:
        digest = _token_digest(session_token)
        if digest is None:
            return None

        with self._lock:
            now = self._clock()
            self._prune(now)
            session = self._sessions.get(digest)
            if session is None:
                return None
            session.last_seen = now
            return session.identity

    def revoke_session(self, session_token: str | None) -> None:
        digest = _token_digest(session_token)
        if digest is not None:
            with self._lock:
                self._prune(self._clock())
                self._sessions.pop(digest, None)
