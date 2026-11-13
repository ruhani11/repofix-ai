"""Test bounded, commit-pinned repository structure reading offline."""

import httpx
import pytest

from app.github_client import (
    GitHubClientError, RepositoryMetadata, RepositorySnapshot,
)
from app.github_tree import MAX_TREE_ENTRIES, fetch_repository_tree


COMMIT_SHA = "a" * 40
TREE_SHA = "b" * 40
BLOB_SHA = "c" * 40
API_PREFIX = "https://api.github.com/repos/owner/repo/git"


@pytest.fixture
def snapshot():
    return RepositorySnapshot(
        repository=RepositoryMetadata(
            full_name="owner/repo",
            default_branch="main",
            private=False,
            visibility="public",
        ),
        commit_sha=COMMIT_SHA,
    )


@pytest.fixture
def commit():
    return {"sha": COMMIT_SHA, "tree": {"sha": TREE_SHA}}


@pytest.fixture
def entry():
    return {
        "path": "src/main.py",
        "type": "blob",
        "mode": "100644",
        "sha": BLOB_SHA,
        "size": 42,
    }


@pytest.fixture
def tree(entry):
    return {"sha": TREE_SHA, "truncated": False, "tree": [entry]}


def make_transport(commit, tree):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "GET"
        assert "Authorization" not in request.headers

        if str(request.url) == f"{API_PREFIX}/commits/{COMMIT_SHA}":
            return httpx.Response(200, json=commit)

        assert str(request.url) == f"{API_PREFIX}/trees/{TREE_SHA}?recursive=1"
        return httpx.Response(200, json=tree)

    return httpx.MockTransport(handler), requests


def test_tree_uses_captured_commit(snapshot, commit, tree):
    transport, requests = make_transport(commit, tree)

    result = fetch_repository_tree(snapshot, transport=transport)

    assert result.repository_full_name == "owner/repo"
    assert result.commit_sha == COMMIT_SHA
    assert result.tree_sha == TREE_SHA
    assert result.entries[0].path == "src/main.py"
    assert result.entries[0].size == 42
    assert len(requests) == 2


def test_links_and_submodules_are_listed_without_following(snapshot, commit, tree, entry):
    tree["tree"] = [
        entry,
        {"path": "src", "type": "tree", "mode": "040000", "sha": TREE_SHA},
        {"path": "link", "type": "blob", "mode": "120000", "sha": BLOB_SHA},
        {"path": "vendor", "type": "commit", "mode": "160000", "sha": COMMIT_SHA},
    ]
    transport, requests = make_transport(commit, tree)

    result = fetch_repository_tree(snapshot, transport=transport)

    assert len(result.entries) == 4
    assert len(requests) == 2


@pytest.mark.parametrize("bad_commit", [{}, [], {"sha": COMMIT_SHA}])
def test_invalid_commit_is_rejected(snapshot, tree, bad_commit):
    transport, requests = make_transport(bad_commit, tree)

    with pytest.raises(GitHubClientError, match="invalid Git commit"):
        fetch_repository_tree(snapshot, transport=transport)

    assert len(requests) == 1


def test_wrong_commit_stops_before_tree_request(snapshot, commit, tree):
    commit["sha"] = "d" * 40
    transport, requests = make_transport(commit, tree)

    with pytest.raises(GitHubClientError, match="different commit"):
        fetch_repository_tree(snapshot, transport=transport)

    assert len(requests) == 1


@pytest.mark.parametrize(
    "bad_tree",
    [{}, [], {"sha": TREE_SHA, "tree": [], "truncated": "false"}],
)
def test_invalid_tree_is_rejected(snapshot, commit, bad_tree):
    transport, _ = make_transport(commit, bad_tree)

    with pytest.raises(GitHubClientError, match="Invalid Git tree"):
        fetch_repository_tree(snapshot, transport=transport)


def test_wrong_tree_identity_is_rejected(snapshot, commit, tree):
    tree["sha"] = "d" * 40
    transport, _ = make_transport(commit, tree)

    with pytest.raises(GitHubClientError, match="different tree"):
        fetch_repository_tree(snapshot, transport=transport)


def test_truncated_listing_is_rejected(snapshot, commit, tree):
    tree["truncated"] = True
    transport, _ = make_transport(commit, tree)

    with pytest.raises(GitHubClientError, match="incomplete tree listing"):
        fetch_repository_tree(snapshot, transport=transport)


def test_duplicate_paths_are_rejected(snapshot, commit, tree, entry):
    tree["tree"] = [entry, dict(entry)]
    transport, _ = make_transport(commit, tree)

    with pytest.raises(GitHubClientError, match="duplicate tree paths"):
        fetch_repository_tree(snapshot, transport=transport)


@pytest.mark.parametrize(
    "path",
    [
        "../escape.py", "/absolute.py", "src//main.py", "src/./main.py",
        "src/../main.py", "src\\main.py", "src/.git/config", "src/\nmain.py",
    ],
)
def test_invalid_paths_are_rejected(snapshot, commit, tree, entry, path):
    tree["tree"] = [entry | {"path": path}]
    transport, _ = make_transport(commit, tree)

    with pytest.raises(GitHubClientError, match="Invalid Git tree"):
        fetch_repository_tree(snapshot, transport=transport)


@pytest.mark.parametrize(
    "object_type, mode",
    [("blob", "040000"), ("tree", "100644"), ("commit", "100755"), ("blob", "000000")],
)
def test_invalid_modes_are_rejected(snapshot, commit, tree, entry, object_type, mode):
    tree["tree"] = [entry | {"type": object_type, "mode": mode}]
    transport, _ = make_transport(commit, tree)

    with pytest.raises(GitHubClientError, match="Invalid Git tree"):
        fetch_repository_tree(snapshot, transport=transport)


def test_invalid_entry_sha_is_rejected(snapshot, commit, tree, entry):
    tree["tree"] = [entry | {"sha": "invalid"}]
    transport, _ = make_transport(commit, tree)

    with pytest.raises(GitHubClientError, match="Invalid Git tree"):
        fetch_repository_tree(snapshot, transport=transport)


@pytest.mark.parametrize("count", [MAX_TREE_ENTRIES, MAX_TREE_ENTRIES + 1])
def test_entry_limit_boundary(snapshot, commit, tree, entry, count):
    tree["tree"] = [entry | {"path": f"file_{index}.py"} for index in range(count)]
    transport, _ = make_transport(commit, tree)

    if count > MAX_TREE_ENTRIES:
        with pytest.raises(GitHubClientError, match="1000-entry limit"):
            fetch_repository_tree(snapshot, transport=transport)
    else:
        result = fetch_repository_tree(snapshot, transport=transport)
        assert len(result.entries) == count


def test_invalid_snapshot_stops_before_network(snapshot, commit, tree):
    snapshot.commit_sha = "../invalid"
    transport, requests = make_transport(commit, tree)

    with pytest.raises(GitHubClientError, match="Invalid repository snapshot"):
        fetch_repository_tree(snapshot, transport=transport)

    assert requests == []


def test_nonpublic_snapshot_stops_before_network(snapshot, commit, tree):
    snapshot.repository.private = True
    snapshot.repository.visibility = "private"
    transport, requests = make_transport(commit, tree)

    with pytest.raises(GitHubClientError, match="Only public repositories"):
        fetch_repository_tree(snapshot, transport=transport)

    assert requests == []
