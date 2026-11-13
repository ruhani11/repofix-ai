"""Offline repository-listing tests; no real GitHub requests."""

from urllib.parse import parse_qs

import httpx
import pytest

from app.github_client import GitHubClientError
from app.github_oauth import GitHubIdentity
from app.github_repositories import (
    MAX_REPOSITORIES_PER_PAGE,
    MAX_REPOSITORY_PAGES,
    fetch_public_repositories,
)


@pytest.fixture(autouse=True)
def block_live_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("A repository-listing test attempted a real request.")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)


@pytest.fixture
def identity():
    return GitHubIdentity(id=123, login="demo-user", type="User")


@pytest.fixture
def repository():
    return {
        "id": 5,
        "name": "example",
        "full_name": "demo-user/example",
        "owner": {"id": 123, "login": "demo-user", "type": "User"},
        "private": False,
        "visibility": "public",
        "description": None,
        "language": None,
        "default_branch": "main",
        "fork": False,
        "archived": False,
        "disabled": False,
        "html_url": "https://attacker.example/untrusted",
    }


def fetch_json(identity, payload, **kwargs):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=payload)
    )
    return fetch_public_repositories(
        identity, transport=transport, **kwargs
    )


def test_fixed_read_only_request_and_safe_result(identity, repository):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "GET"
        assert request.url.scheme == "https"
        assert request.url.host == "api.github.com"
        assert request.url.path == "/users/demo-user/repos"
        assert parse_qs(request.url.query.decode()) == {
            "type": ["owner"],
            "sort": ["full_name"],
            "direction": ["asc"],
            "per_page": ["20"],
            "page": ["1"],
        }
        assert "Authorization" not in request.headers
        return httpx.Response(200, json=[repository])

    result = fetch_public_repositories(
        identity, transport=httpx.MockTransport(handler)
    )

    assert len(requests) == 1
    assert result.account_login == identity.login
    assert result.repositories[0].id == 5
    assert (
        result.repositories[0].html_url
        == "https://github.com/demo-user/example"
    )
    assert "attacker.example" not in result.model_dump_json()
    assert result.next_page is None
    assert not result.may_have_more
    assert not result.limit_reached


def test_empty_page_is_valid(identity):
    result = fetch_json(identity, [])
    assert result.repositories == []
    assert result.next_page is None
    assert not result.may_have_more


@pytest.mark.parametrize("page", [1, MAX_REPOSITORY_PAGES])
def test_full_page_has_bounded_next_probe(identity, repository, page):
    result = fetch_json(
        identity, [repository], page=page, per_page=1
    )
    assert result.may_have_more

    if page == MAX_REPOSITORY_PAGES:
        assert result.next_page is None
        assert result.limit_reached
    else:
        assert result.next_page == page + 1
        assert not result.limit_reached


@pytest.mark.parametrize(
    "kwargs",
    [
        {"page": 0},
        {"page": MAX_REPOSITORY_PAGES + 1},
        {"page": True},
        {"page": "1"},
        {"per_page": 0},
        {"per_page": MAX_REPOSITORIES_PER_PAGE + 1},
        {"per_page": True},
        {"per_page": "20"},
    ],
)
def test_invalid_pagination_never_requests(identity, kwargs):
    def unexpected(request):
        pytest.fail("Invalid pagination reached GitHub.")

    with pytest.raises(ValueError):
        fetch_public_repositories(
            identity,
            transport=httpx.MockTransport(unexpected),
            **kwargs,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"id": 123, "login": "demo-user", "type": "User"},
        GitHubIdentity.model_construct(
            id=123,
            login="../other",
            type="User",
            name=None,
        ),
    ],
)
def test_invalid_identity_never_requests(value):
    def unexpected(request):
        pytest.fail("Invalid identity reached GitHub.")

    with pytest.raises(ValueError):
        fetch_public_repositories(
            value, transport=httpx.MockTransport(unexpected)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("private", True),
        ("private", 0),
        ("visibility", "private"),
        ("full_name", "other/example"),
        ("name", ".."),
        ("name", "folder/repo"),
        ("id", "5"),
        ("default_branch", ""),
    ],
)
def test_invalid_repository_rejected(
    identity, repository, field, value
):
    repository[field] = value

    with pytest.raises(
        GitHubClientError, match="invalid public repositories"
    ):
        fetch_json(identity, [repository])


@pytest.mark.parametrize(
    "owner",
    [
        {"id": 999, "login": "demo-user", "type": "User"},
        {"id": 123, "login": "other-user", "type": "User"},
    ],
)
def test_different_owner_rejected(identity, repository, owner):
    repository["owner"] = owner
    repository["full_name"] = f"{owner['login']}/example"

    with pytest.raises(
        GitHubClientError, match="different account"
    ):
        fetch_json(identity, [repository])


@pytest.mark.parametrize("duplicate", ["id", "full_name"])
def test_duplicate_repositories_rejected(
    identity, repository, duplicate
):
    if duplicate == "id":
        second = dict(
            repository,
            name="second",
            full_name="demo-user/second",
        )
    else:
        second = dict(repository, id=6)

    with pytest.raises(
        GitHubClientError, match="duplicate repositories"
    ):
        fetch_json(identity, [repository, second])


@pytest.mark.parametrize(
    "payload", [None, {}, "not a repository list"]
)
def test_wrong_payload_shape_rejected(identity, payload):
    with pytest.raises(
        GitHubClientError, match="invalid public repositories"
    ):
        fetch_json(identity, payload)


def test_malformed_json_rejected(identity):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=b"not JSON"
        )
    )

    with pytest.raises(
        GitHubClientError, match="invalid public repositories"
    ):
        fetch_public_repositories(identity, transport=transport)


def test_requested_page_size_enforced(identity, repository):
    second = dict(
        repository,
        id=6,
        name="second",
        full_name="demo-user/second",
    )

    with pytest.raises(
        GitHubClientError,
        match="more repositories than requested",
    ):
        fetch_json(
            identity, [repository, second], per_page=1
        )


def test_maximum_page_size_enforced(identity, repository):
    payload = [
        dict(
            repository,
            id=100 + i,
            name=f"repo-{i}",
            full_name=f"demo-user/repo-{i}",
        )
        for i in range(MAX_REPOSITORIES_PER_PAGE + 1)
    ]

    with pytest.raises(
        GitHubClientError, match="invalid public repositories"
    ):
        fetch_json(identity, payload)


@pytest.mark.parametrize(
    "status,headers,message",
    [
        (
            302,
            {"Location": "https://attacker.example"},
            "Redirect blocked",
        ),
        (403, {}, "denied access"),
        (
            403,
            {"x-ratelimit-remaining": "0"},
            "rate limit",
        ),
        (404, {}, "not found or not public"),
        (429, {}, "rate limit"),
        (500, {}, "HTTP 500"),
    ],
)
def test_http_failures_are_safe(
    identity, status, headers, message
):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            status,
            headers=headers,
            json={"secret": "dummy-upstream-secret"},
        )

    with pytest.raises(
        GitHubClientError, match=message
    ) as error:
        fetch_public_repositories(
            identity, transport=httpx.MockTransport(handler)
        )

    assert len(requests) == 1
    assert "dummy-upstream-secret" not in str(error.value)


def test_listing_preserves_flags_without_granting_permissions(
    identity, repository
):
    repository.update(
        fork=True,
        archived=True,
        disabled=True,
        permissions={"push": True},
    )

    result = fetch_json(identity, [repository])
    selected = result.repositories[0]

    assert selected.fork and selected.archived and selected.disabled
    assert "permissions" not in selected.model_dump()


def test_owner_login_comparison_is_case_insensitive(
    identity, repository
):
    repository["owner"]["login"] = "Demo-User"
    repository["full_name"] = "Demo-User/example"

    result = fetch_json(identity, [repository])

    assert result.repositories[0].owner.id == identity.id