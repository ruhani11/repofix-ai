"""Offline integration tests for fixed-commit investigation preparation."""

import httpx
import pytest

from app.github_client import GitHubClientError
from app.github_oauth import GitHubIdentity
from app.github_selection import RepositorySelectionError
from app.investigation_context import prepare_investigation_context
from app.investigation_input import InvestigationInput


IDENTITY = GitHubIdentity(id=123, login="demo-user", type="User")
INPUT = InvestigationInput(
    repository_id=5,
    repository_name="example",
    issue_description="Login fails after submitting valid credentials.",
)
COMMIT_SHA = "a" * 40
TREE_SHA = "b" * 40
REPOSITORY_PATH = "/repos/demo-user/example"


@pytest.fixture(autouse=True)
def block_real_network(monkeypatch):
    def reject_request(*args, **kwargs):
        pytest.fail("A context test attempted a real network request.")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject_request)


def repository_metadata(**changes):
    return {
        "id": 5,
        "name": "example",
        "full_name": "demo-user/example",
        "owner": IDENTITY.model_dump(),
        "private": False,
        "visibility": "public",
        "default_branch": "main",
        "fork": False,
        "archived": False,
        "disabled": False,
        **changes,
    }


def make_github_transport(
    *,
    first_metadata=None,
    final_metadata=None,
    failed_path=None,
    wrong_commit=False,
    truncated_tree=False,
):
    paths = []
    metadata_reads = 0

    def handler(request):
        nonlocal metadata_reads
        assert request.method == "GET"
        assert request.url.host == "api.github.com"
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        path = request.url.path
        paths.append(path)

        if path == failed_path:
            return httpx.Response(429)

        if path == REPOSITORY_PATH:
            metadata_reads += 1
            body = repository_metadata()
            if metadata_reads == 1 and first_metadata is not None:
                body = first_metadata
            if metadata_reads == 3 and final_metadata is not None:
                body = final_metadata
        elif path == f"{REPOSITORY_PATH}/git/ref/heads/main":
            body = {
                "ref": "refs/heads/main",
                "object": {"type": "commit", "sha": COMMIT_SHA},
            }
        elif path == f"{REPOSITORY_PATH}/git/commits/{COMMIT_SHA}":
            body = {
                "sha": "c" * 40 if wrong_commit else COMMIT_SHA,
                "tree": {"sha": TREE_SHA},
            }
        elif path == f"{REPOSITORY_PATH}/git/trees/{TREE_SHA}":
            assert request.url.params["recursive"] == "1"
            body = {
                "sha": TREE_SHA,
                "truncated": truncated_tree,
                "tree": [{
                    "path": "README.md",
                    "type": "blob",
                    "mode": "100644",
                    "sha": "d" * 40,
                    "size": 20,
                }],
            }
        else:
            pytest.fail(f"Unexpected GitHub request: {path}")

        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler), paths


def test_context_binds_issue_owner_repository_and_fixed_commit():
    transport, paths = make_github_transport()

    context = prepare_investigation_context(IDENTITY, INPUT, transport=transport)

    assert context.identity == IDENTITY
    assert context.request == INPUT
    assert context.repository.id == INPUT.repository_id
    assert context.repository.owner.id == IDENTITY.id
    assert context.snapshot.commit_sha == context.tree.commit_sha == COMMIT_SHA
    assert context.tree.repository_full_name == context.repository.full_name
    assert context.tree.tree_sha == TREE_SHA
    assert [entry.path for entry in context.tree.entries] == ["README.md"]
    assert paths.count(REPOSITORY_PATH) == 3
    assert not any("/blobs/" in path for path in paths)


@pytest.mark.parametrize("changed_metadata", [
    repository_metadata(id=6),
    repository_metadata(owner={**IDENTITY.model_dump(), "id": 456}),
    repository_metadata(default_branch="develop"),
    repository_metadata(disabled=True),
])
def test_changed_selection_at_final_check_is_rejected(changed_metadata):
    transport, _ = make_github_transport(final_metadata=changed_metadata)

    with pytest.raises(RepositorySelectionError):
        prepare_investigation_context(IDENTITY, INPUT, transport=transport)


def test_invalid_initial_selection_stops_before_commit_or_tree_reads():
    transport, paths = make_github_transport(
        first_metadata=repository_metadata(id=6)
    )

    with pytest.raises(RepositorySelectionError):
        prepare_investigation_context(IDENTITY, INPUT, transport=transport)

    assert paths == [REPOSITORY_PATH]


@pytest.mark.parametrize("options", [
    {"wrong_commit": True},
    {"truncated_tree": True},
    {"failed_path": REPOSITORY_PATH},
])
def test_upstream_failure_or_invalid_tree_cannot_return_context(options):
    transport, _ = make_github_transport(**options)

    with pytest.raises(GitHubClientError):
        prepare_investigation_context(IDENTITY, INPUT, transport=transport)


@pytest.mark.parametrize("identity,payload", [
    (None, INPUT),
    (IDENTITY, None),
    (IDENTITY, INPUT.model_copy(update={"repository_id": True})),
    (IDENTITY, INPUT.model_copy(update={"issue_description": ""})),
    (IDENTITY.model_copy(update={"login": "../other"}), INPUT),
])
def test_invalid_internal_input_stops_before_network(identity, payload):
    transport, paths = make_github_transport()

    with pytest.raises(ValueError):
        prepare_investigation_context(identity, payload, transport=transport)

    assert paths == []
