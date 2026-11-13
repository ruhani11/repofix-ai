"""Offline auth-store tests with a fake clock; no sleeps or GitHub requests."""

import re
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

import pytest

from app import auth_store as store_module
from app.auth_store import (
    LOGIN_TTL_SECONDS,
    SESSION_IDLE_SECONDS,
    SESSION_TTL_SECONDS,
    AuthStore,
    AuthStoreError,
)
from app.github_oauth import GitHubIdentity


@pytest.fixture
def clock():
    return [0.0]


@pytest.fixture
def store(clock):
    return AuthStore(clock=lambda: clock[0])


@pytest.fixture
def identity():
    return GitHubIdentity(id=123, login="demo-user", type="User")


def test_login_credentials_are_distinct_and_hidden_from_repr(store):
    attempt = store.start_login()
    values = (
        attempt.state,
        attempt.browser_token,
        attempt.code_verifier,
    )

    assert len(set(values)) == 3
    assert all(
        re.fullmatch(r"[A-Za-z0-9_-]{43}", value)
        for value in values
    )
    assert all(value not in repr(attempt) for value in values)
    assert all(value not in repr(store._pending) for value in values)


def test_login_can_be_consumed_only_once(store):
    attempt = store.start_login()
    inputs = {
        "state": attempt.state,
        "browser_token": attempt.browser_token,
    }

    assert store.consume_login(**inputs) == attempt.code_verifier

    with pytest.raises(AuthStoreError, match="Invalid or expired"):
        store.consume_login(**inputs)


def test_other_browser_cannot_consume_valid_login(store):
    first = store.start_login()
    second = store.start_login()

    with pytest.raises(AuthStoreError) as error:
        store.consume_login(
            state=first.state,
            browser_token=second.browser_token,
        )

    assert first.state not in str(error.value)
    assert second.browser_token not in str(error.value)

    assert store.consume_login(
        state=first.state,
        browser_token=first.browser_token,
    ) == first.code_verifier

    assert store.consume_login(
        state=second.state,
        browser_token=second.browser_token,
    ) == second.code_verifier


@pytest.mark.parametrize("value", [None, "", "invalid", "x" * 43])
def test_missing_malformed_or_unknown_state_is_rejected(store, value):
    attempt = store.start_login()

    with pytest.raises(AuthStoreError):
        store.consume_login(
            state=value,
            browser_token=attempt.browser_token,
        )


@pytest.mark.parametrize("value", [None, "", "invalid"])
def test_missing_or_malformed_browser_token_is_rejected(store, value):
    attempt = store.start_login()

    with pytest.raises(AuthStoreError):
        store.consume_login(
            state=attempt.state,
            browser_token=value,
        )


@pytest.mark.parametrize("expired", [False, True])
def test_login_expiry_boundary(store, clock, expired):
    attempt = store.start_login()
    clock[0] = (
        LOGIN_TTL_SECONDS
        if expired
        else LOGIN_TTL_SECONDS - 1
    )

    if expired:
        with pytest.raises(AuthStoreError):
            store.consume_login(
                state=attempt.state,
                browser_token=attempt.browser_token,
            )
    else:
        assert store.consume_login(
            state=attempt.state,
            browser_token=attempt.browser_token,
        ) == attempt.code_verifier


def test_concurrent_callbacks_have_exactly_one_success(store):
    attempt = store.start_login()
    barrier = Barrier(8)

    def consume(_):
        barrier.wait(timeout=10)

        try:
            store.consume_login(
                state=attempt.state,
                browser_token=attempt.browser_token,
            )
            return True
        except AuthStoreError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(consume, range(8)))

    assert sum(results) == 1


def test_session_returns_identity_without_storing_raw_token(store, identity):
    token = store.create_session(identity)

    assert store.get_identity(token) == identity
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", token)
    assert token not in repr(store._sessions)


@pytest.mark.parametrize("token", [None, "", "invalid", "x" * 43])
def test_missing_malformed_or_unknown_session_has_no_identity(store, token):
    assert store.get_identity(token) is None
    store.revoke_session(token)


@pytest.mark.parametrize(
    "value",
    [None, {"id": 123, "login": "demo-user"}],
)
def test_session_rejects_unvalidated_identity(store, value):
    with pytest.raises(
        AuthStoreError,
        match="verified GitHub identity",
    ):
        store.create_session(value)


def test_rotation_revokes_previous_session_only(store, identity):
    previous = store.create_session(identity)
    unrelated = store.create_session(identity)
    current = store.create_session(
        identity,
        previous_token=previous,
    )

    assert current != previous
    assert store.get_identity(previous) is None
    assert store.get_identity(current) == identity
    assert store.get_identity(unrelated) == identity


@pytest.mark.parametrize("expired", [False, True])
def test_session_idle_expiry_boundary(store, clock, identity, expired):
    token = store.create_session(identity)
    clock[0] = (
        SESSION_IDLE_SECONDS
        if expired
        else SESSION_IDLE_SECONDS - 1
    )

    assert store.get_identity(token) == (
        None if expired else identity
    )


def test_valid_access_refreshes_idle_activity(store, clock, identity):
    token = store.create_session(identity)

    clock[0] = SESSION_IDLE_SECONDS - 1
    assert store.get_identity(token) == identity

    clock[0] += SESSION_IDLE_SECONDS - 1
    assert store.get_identity(token) == identity


def test_activity_cannot_extend_absolute_expiry(store, clock, identity):
    token = store.create_session(identity)

    for instant in range(600, SESSION_TTL_SECONDS, 600):
        clock[0] = instant
        assert store.get_identity(token) == identity

    clock[0] = SESSION_TTL_SECONDS
    assert store.get_identity(token) is None


def test_logout_is_idempotent_and_preserves_other_sessions(store, identity):
    token = store.create_session(identity)
    unrelated = store.create_session(identity)

    store.revoke_session(token)
    store.revoke_session(token)

    assert store.get_identity(token) is None
    assert store.get_identity(unrelated) == identity


def test_pending_capacity_is_reclaimed_after_expiry(
    store,
    clock,
    monkeypatch,
):
    monkeypatch.setattr(store_module, "MAX_PENDING_LOGINS", 1)
    previous = store.start_login()

    with pytest.raises(AuthStoreError, match="Too many pending"):
        store.start_login()

    clock[0] = LOGIN_TTL_SECONDS
    current = store.start_login()

    assert current.state != previous.state
    assert store.consume_login(
        state=current.state,
        browser_token=current.browser_token,
    ) == current.code_verifier


def test_session_capacity_is_reclaimed_after_expiry(
    store,
    clock,
    identity,
    monkeypatch,
):
    monkeypatch.setattr(store_module, "MAX_SESSIONS", 1)
    previous = store.create_session(identity)

    with pytest.raises(AuthStoreError, match="Too many active"):
        store.create_session(identity)

    clock[0] = SESSION_IDLE_SECONDS
    current = store.create_session(identity)

    assert store.get_identity(previous) is None
    assert store.get_identity(current) == identity


def test_session_can_rotate_at_capacity(store, identity, monkeypatch):
    monkeypatch.setattr(store_module, "MAX_SESSIONS", 1)
    previous = store.create_session(identity)

    current = store.create_session(
        identity,
        previous_token=previous,
    )

    assert store.get_identity(previous) is None
    assert store.get_identity(current) == identity


def test_collision_retries_are_bounded_and_preserve_old_session(
    store,
    identity,
    monkeypatch,
):
    previous = store.create_session(identity)
    generator = Mock(return_value=previous)

    monkeypatch.setattr(
        store_module.secrets,
        "token_urlsafe",
        generator,
    )

    with pytest.raises(AuthStoreError, match="Could not create"):
        store.create_session(
            identity,
            previous_token=previous,
        )

    assert generator.call_count == 5
    assert store.get_identity(previous) == identity


def test_separate_store_does_not_share_logins_or_sessions(store, identity):
    attempt = store.start_login()
    token = store.create_session(identity)
    other = AuthStore()

    assert other.get_identity(token) is None

    with pytest.raises(AuthStoreError):
        other.consume_login(
            state=attempt.state,
            browser_token=attempt.browser_token,
        )