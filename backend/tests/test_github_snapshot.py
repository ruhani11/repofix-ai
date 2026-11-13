"""Verify fixed-commit capture without live GitHub requests."""

import httpx
import pytest

from app.github_client import GitHubClientError, fetch_repository_snapshot


REPOSITORY_URL = "https://github.com/owner/repo"
METADATA_URL = "https://api.github.com/repos/owner/repo"
REFERENCE_URL = f"{METADATA_URL}/git/ref/heads/main"
COMMIT_SHA = "a" * 40  # Synthetic test fixture, not a real repository result.


@pytest.fixture
def metadata():
    return {
        "full_name": "owner/repo",
        "default_branch": "main",
        "private": False,
        "visibility": "public",
    }


def make_transport(metadata, reference, *, reference_url=REFERENCE_URL, status=200):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "GET"
        assert "Authorization" not in request.headers

        if str(request.url) == METADATA_URL:
            return httpx.Response(200, json=metadata)

        assert str(request.url) == reference_url
        return httpx.Response(status, json=reference)

    return httpx.MockTransport(handler), requests


@pytest.mark.parametrize(
    "branch, reference_url",
    [
        ("main", REFERENCE_URL),
        ("feature/fix", f"{METADATA_URL}/git/ref/heads/feature%2Ffix"),
    ],
)
def test_snapshot_captures_default_branch_commit(metadata, branch, reference_url):
    metadata["default_branch"] = branch
    reference = {
        "ref": f"refs/heads/{branch}",
        "object": {"type": "commit", "sha": COMMIT_SHA},
    }
    transport, requests = make_transport(
        metadata, reference, reference_url=reference_url
    )

    snapshot = fetch_repository_snapshot(REPOSITORY_URL, transport=transport)

    assert snapshot.repository.full_name == "owner/repo"
    assert snapshot.repository.default_branch == branch
    assert snapshot.commit_sha == COMMIT_SHA
    assert len(requests) == 2


@pytest.mark.parametrize(
    "reference",
    [
        {},
        [],
        {"ref": "refs/heads/main"},
        {
            "ref": "refs/heads/main",
            "object": {"type": "commit", "sha": "invalid"},
        },
        {
            "ref": "refs/heads/main",
            "object": {"type": "tag", "sha": COMMIT_SHA},
        },
        {
            "ref": "refs/heads/main",
            "object": {"type": "commit", "sha": "A" * 40},
        },
    ],
)
def test_invalid_commit_reference_is_rejected(metadata, reference):
    transport, _ = make_transport(metadata, reference)

    with pytest.raises(GitHubClientError, match="invalid commit reference"):
        fetch_repository_snapshot(REPOSITORY_URL, transport=transport)


def test_different_branch_reference_is_rejected(metadata):
    reference = {
        "ref": "refs/heads/another-branch",
        "object": {"type": "commit", "sha": COMMIT_SHA},
    }
    transport, _ = make_transport(metadata, reference)

    with pytest.raises(GitHubClientError, match="different branch"):
        fetch_repository_snapshot(REPOSITORY_URL, transport=transport)


@pytest.mark.parametrize("status", [404, 409])
def test_unavailable_reference_is_handled(metadata, status):
    transport, requests = make_transport(metadata, {}, status=status)

    with pytest.raises(GitHubClientError):
        fetch_repository_snapshot(REPOSITORY_URL, transport=transport)

    assert len(requests) == 2


def test_nonpublic_repository_stops_before_reference_request(metadata):
    metadata.update(private=True, visibility="private")
    transport, requests = make_transport(metadata, {})

    with pytest.raises(GitHubClientError, match="Only public repositories"):
        fetch_repository_snapshot(REPOSITORY_URL, transport=transport)

    assert len(requests) == 1
