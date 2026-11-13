"""Test the GitHub client without real network requests."""

import httpx
import pytest

from app.github_client import (
    MAX_RESPONSE_BYTES,
    GitHubClientError,
    fetch_repository_metadata,
)


REPOSITORY_URL = "https://github.com/owner/repo"


@pytest.fixture
def public_metadata():
    return {
        "full_name": "owner/repo",
        "default_branch": "main",
        "private": False,
        "visibility": "public",
        "description": None,
        "language": None,
    }


def test_fetch_public_metadata(public_metadata):
    def handler(request):
        assert request.method == "GET"
        assert str(request.url) == "https://api.github.com/repos/owner/repo"
        assert request.headers["Accept"] == "application/vnd.github+json"
        assert "Authorization" not in request.headers
        return httpx.Response(200, json=public_metadata)

    metadata = fetch_repository_metadata(
        REPOSITORY_URL, transport=httpx.MockTransport(handler)
    )

    assert metadata.full_name == "owner/repo"
    assert metadata.default_branch == "main"
    assert metadata.private is False
    assert metadata.visibility == "public"
    assert metadata.language is None


@pytest.mark.parametrize(
    "status, headers, message",
    [
        (404, {}, "not found or not public"),
        (403, {}, "denied access"),
        (403, {"x-ratelimit-remaining": "0"}, "rate limit"),
        (403, {"retry-after": "30"}, "rate limit"),
        (429, {}, "rate limit"),
        (301, {"Location": "https://example.com"}, "Redirect blocked"),
        (500, {}, "HTTP 500"),
    ],
)
def test_http_errors_are_handled(status, headers, message):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, headers=headers)

    with pytest.raises(GitHubClientError, match=message):
        fetch_repository_metadata(
            REPOSITORY_URL, transport=httpx.MockTransport(handler)
        )

    assert len(requests) == 1


def test_invalid_json_is_rejected():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"not JSON")
    )

    with pytest.raises(GitHubClientError, match="invalid repository metadata"):
        fetch_repository_metadata(REPOSITORY_URL, transport=transport)


@pytest.mark.parametrize(
    "field, value",
    [("private", "false"), ("default_branch", ""), ("visibility", None)],
)
def test_invalid_metadata_is_rejected(public_metadata, field, value):
    public_metadata[field] = value
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=public_metadata)
    )

    with pytest.raises(GitHubClientError, match="invalid repository metadata"):
        fetch_repository_metadata(REPOSITORY_URL, transport=transport)


@pytest.mark.parametrize(
    "private, visibility",
    [(True, "private"), (True, "public"), (False, "internal")],
)
def test_nonpublic_metadata_is_rejected(public_metadata, private, visibility):
    public_metadata.update(private=private, visibility=visibility)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=public_metadata)
    )

    with pytest.raises(GitHubClientError, match="Only public repositories"):
        fetch_repository_metadata(REPOSITORY_URL, transport=transport)


def test_different_repository_is_rejected(public_metadata):
    public_metadata["full_name"] = "someone/another-repo"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=public_metadata)
    )

    with pytest.raises(GitHubClientError, match="different repository"):
        fetch_repository_metadata(REPOSITORY_URL, transport=transport)


def test_oversized_response_is_rejected():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=b"x" * (MAX_RESPONSE_BYTES + 1)
        )
    )

    with pytest.raises(GitHubClientError, match="too large"):
        fetch_repository_metadata(REPOSITORY_URL, transport=transport)


@pytest.mark.parametrize(
    "error, message",
    [
        (httpx.ReadTimeout("test timeout"), "timed out"),
        (httpx.ConnectError("test offline"), "Could not connect"),
    ],
)
def test_network_errors_are_handled(error, message):
    def handler(request):
        raise error

    with pytest.raises(GitHubClientError, match=message):
        fetch_repository_metadata(
            REPOSITORY_URL, transport=httpx.MockTransport(handler)
        )


def test_invalid_url_is_rejected_before_network_access():
    def handler(request):
        pytest.fail("An invalid URL must not trigger a network request.")

    with pytest.raises(ValueError):
        fetch_repository_metadata(
            "https://example.com/owner/repo",
            transport=httpx.MockTransport(handler),
        )