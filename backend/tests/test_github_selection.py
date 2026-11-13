"""Offline checks for selecting the connected account's public repository."""

import httpx
import pytest
from pydantic import ValidationError

from app.github_client import GitHubClientError
from app.github_oauth import GitHubIdentity
from app.github_selection import (
    RepositorySelectionError,
    RepositorySelectionInput,
    fetch_selected_repository,
)


@pytest.fixture(autouse=True)
def block_live_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("A selection test attempted a real request.")

    monkeypatch.setattr(
        httpx.HTTPTransport,
        "handle_request",
        blocked,
    )


@pytest.fixture
def identity():
    return GitHubIdentity(
        id=123,
        login="demo-user",
        type="User",
    )


@pytest.fixture
def selection():
    return RepositorySelectionInput(
        repository_id=5,
        repository_name="example",
    )


@pytest.fixture
def repository():
    return {
        "id": 5,
        "name": "example",
        "full_name": "demo-user/example",
        "owner": {
            "id": 123,
            "login": "demo-user",
            "type": "User",
        },
        "private": False,
        "visibility": "public",
        "default_branch": "main",
        "fork": False,
        "archived": False,
        "disabled": False,
    }


def fetch_json(identity, selection, payload):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=payload)
    )
    return fetch_selected_repository(
        identity,
        selection,
        transport=transport,
    )


def test_selection_uses_session_owner_and_one_fixed_get(
    identity, selection, repository
):
    requests = []

    def handler(request):
        requests.append(request)

        assert request.method == "GET"
        assert (
            str(request.url)
            == "https://api.github.com/repos/demo-user/example"
        )
        assert "Authorization" not in request.headers

        return httpx.Response(200, json=repository)

    result = fetch_selected_repository(
        identity,
        selection,
        transport=httpx.MockTransport(handler),
    )

    assert len(requests) == 1
    assert result.id == selection.repository_id
    assert result.owner.id == identity.id
    assert result.html_url == "https://github.com/demo-user/example"


@pytest.mark.parametrize(
    "payload",
    [
        {"repository_id": 0, "repository_name": "example"},
        {"repository_id": True, "repository_name": "example"},
        {"repository_id": "5", "repository_name": "example"},
        {"repository_id": 2**63, "repository_name": "example"},
        {"repository_id": 5, "repository_name": ".."},
        {"repository_id": 5, "repository_name": "../other"},
        {"repository_id": 5, "repository_name": ""},
        {
            "repository_id": 5,
            "repository_name": "example",
            "owner": "other-user",
        },
        {
            "repository_id": 5,
            "repository_name": "example",
            "repository_url": "https://example.com",
        },
    ],
)
def test_invalid_selection_input_is_rejected(payload):
    with pytest.raises(ValidationError):
        RepositorySelectionInput.model_validate(payload)


@pytest.mark.parametrize(
    "case",
    [
        "identity-dict",
        "identity-bypass",
        "selection-dict",
        "selection-bypass",
    ],
)
def test_invalid_objects_never_reach_github(
    identity, selection, case
):
    if case == "identity-dict":
        identity = identity.model_dump()

    elif case == "identity-bypass":
        identity = GitHubIdentity.model_construct(
            id=123,
            login="../other",
            type="User",
            name=None,
        )

    elif case == "selection-dict":
        selection = selection.model_dump()

    else:
        selection = RepositorySelectionInput.model_construct(
            repository_id=5,
            repository_name="../other",
        )

    def unexpected(request):
        pytest.fail("Invalid input reached GitHub.")

    with pytest.raises(ValueError):
        fetch_selected_repository(
            identity,
            selection,
            transport=httpx.MockTransport(unexpected),
        )


@pytest.mark.parametrize(
    "field,value",
    [("id", 6), ("name", "other")],
)
def test_changed_repository_selection_is_rejected(
    identity, selection, repository, field, value
):
    repository[field] = value

    if field == "name":
        repository["full_name"] = "demo-user/other"

    with pytest.raises(
        RepositorySelectionError,
        match="selection changed",
    ):
        fetch_json(identity, selection, repository)


@pytest.mark.parametrize(
    "owner",
    [
        {"id": 999, "login": "demo-user", "type": "User"},
        {"id": 123, "login": "other-user", "type": "User"},
    ],
)
def test_different_owner_is_rejected(
    identity, selection, repository, owner
):
    repository["owner"] = owner
    repository["full_name"] = f"{owner['login']}/example"

    with pytest.raises(
        RepositorySelectionError,
        match="connected account",
    ):
        fetch_json(identity, selection, repository)


def test_disabled_repository_is_rejected(
    identity, selection, repository
):
    repository["disabled"] = True

    with pytest.raises(
        RepositorySelectionError,
        match="disabled",
    ):
        fetch_json(identity, selection, repository)


def test_archived_repository_and_case_variations_are_readable(
    identity, selection, repository
):
    repository.update(
        name="Example",
        full_name="Demo-User/Example",
        archived=True,
    )
    repository["owner"]["login"] = "Demo-User"

    result = fetch_json(identity, selection, repository)

    assert result.id == selection.repository_id
    assert result.archived


@pytest.mark.parametrize(
    "field,value",
    [
        ("private", True),
        ("visibility", "private"),
        ("full_name", "other-user/example"),
        ("id", "5"),
    ],
)
def test_unsupported_metadata_is_rejected(
    identity, selection, repository, field, value
):
    repository[field] = value

    with pytest.raises(
        GitHubClientError,
        match="unsupported public repository",
    ):
        fetch_json(identity, selection, repository)


@pytest.mark.parametrize(
    "body",
    [b"not JSON", b"[]", b"null"],
)
def test_invalid_response_body_is_sanitized(
    identity, selection, body
):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=body)
    )

    with pytest.raises(
        GitHubClientError,
        match="unsupported public repository",
    ):
        fetch_selected_repository(
            identity,
            selection,
            transport=transport,
        )


@pytest.mark.parametrize("status", [302, 404, 429])
def test_http_failure_is_safe_and_redirect_is_not_followed(
    identity, selection, status
):
    requests = []

    def handler(request):
        requests.append(request)

        return httpx.Response(
            status,
            headers={
                "Location": "https://attacker.example"
            },
            json={
                "secret": "dummy-upstream-secret"
            },
        )

    with pytest.raises(GitHubClientError) as error:
        fetch_selected_repository(
            identity,
            selection,
            transport=httpx.MockTransport(handler),
        )

    assert len(requests) == 1
    assert "dummy-upstream-secret" not in str(error.value)