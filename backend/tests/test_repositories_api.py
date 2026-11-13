"""Verify the registered inspection API without accessing GitHub."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api import repositories
from app.config import Settings
from app.github_client import GitHubClientError, RepositoryMetadata, RepositorySnapshot
from app.github_files import MAX_FILE_BYTES, RepositoryFile
from app.github_tree import RepositoryTree, TreeEntry
from app.main import create_app


URL = "https://github.com/owner/repo"


@pytest.fixture
def inspection_api(monkeypatch):
    """Fake tool results at the HTTP boundary; lower-level tests verify GitHub."""
    snapshot = RepositorySnapshot(
        repository=RepositoryMetadata(
            full_name="owner/repo", default_branch="main", private=False,
            visibility="public", description=None, language="Python",
        ),
        commit_sha="a" * 40,
    )
    tree = RepositoryTree(
        repository_full_name="owner/repo", commit_sha=snapshot.commit_sha,
        tree_sha="b" * 40,
        entries=[TreeEntry(
            path="README.md", type="blob", mode="100644", sha="c" * 40, size=6,
        )],
    )

    def source_for(selected_tree, path):
        entry = next(item for item in selected_tree.entries if item.path == path)
        return RepositoryFile(
            repository_full_name=selected_tree.repository_full_name,
            commit_sha=selected_tree.commit_sha, path=path,
            blob_sha=entry.sha, size_bytes=6, content="hello\n",
        )

    snapshot_tool = Mock(return_value=snapshot)
    tree_tool = Mock(return_value=tree)
    file_tool = Mock(side_effect=source_for)
    monkeypatch.setattr(repositories, "fetch_repository_snapshot", snapshot_tool)
    monkeypatch.setattr(repositories, "fetch_repository_tree", tree_tool)
    monkeypatch.setattr(repositories, "read_repository_file", file_tool)

    application = create_app(Settings(_env_file=None, environment="test"))
    with TestClient(application) as client:
        yield client, tree, snapshot_tool, tree_tool, file_tool


def test_registered_api_returns_selected_file_with_provenance(inspection_api):
    client, tree, snapshot_tool, tree_tool, file_tool = inspection_api
    response = client.post("/api/repositories/inspect", json={
        "repository_url": URL + ".git", "selected_paths": ["README.md"],
    })

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"snapshot", "tree", "files"}
    assert body["snapshot"]["repository"]["full_name"] == "owner/repo"
    assert body["snapshot"]["commit_sha"] == body["tree"]["commit_sha"]
    assert len(body["files"]) == 1
    assert body["files"][0]["commit_sha"] == body["snapshot"]["commit_sha"]
    assert body["files"][0]["content"] == "hello\n"
    snapshot_tool.assert_called_once_with(URL)
    tree_tool.assert_called_once_with(snapshot_tool.return_value)
    file_tool.assert_called_once_with(tree, "README.md")


def test_omitted_selection_does_not_download_files(inspection_api):
    client, _, snapshot_tool, tree_tool, file_tool = inspection_api
    response = client.post("/api/repositories/inspect", json={"repository_url": URL})

    assert response.status_code == 200
    assert response.json()["files"] == []
    snapshot_tool.assert_called_once()
    tree_tool.assert_called_once()
    file_tool.assert_not_called()


def test_three_selected_files_are_supported_in_request_order(inspection_api):
    client, tree, _, _, file_tool = inspection_api
    paths = ["README.md", "app.py", "tests/test_app.py"]
    tree.entries.extend([
        TreeEntry(path=path, type="blob", mode="100644", sha="d" * 40, size=6)
        for path in paths[1:]
    ])
    response = client.post("/api/repositories/inspect", json={
        "repository_url": URL, "selected_paths": paths,
    })

    assert response.status_code == 200
    assert [source["path"] for source in response.json()["files"]] == paths
    assert [call.args[1] for call in file_tool.call_args_list] == paths


@pytest.mark.parametrize("payload", [
    {},
    {"repository_url": "https://example.com/owner/repo"},
    {"repository_url": "http://github.com/owner/repo"},
    {"repository_url": 123},
    {"repository_url": URL, "selected_paths": "README.md"},
    {"repository_url": URL, "selected_paths": [123]},
    {"repository_url": URL, "selected_paths": ["README.md", "README.md"]},
    {"repository_url": URL, "selected_paths": ["a.py", "b.py", "c.py", "d.py"]},
    {"repository_url": URL, "selected_paths": ["../app.py"]},
    {"repository_url": URL, "selected_paths": [".env"]},
    {"repository_url": URL, "selected_paths": ["node_modules/package/index.js"]},
    {"repository_url": URL, "selected_paths": ["image.png"]},
    {"repository_url": URL, "snapshot": {}},
    {"repository_url": URL, "tree": {}},
])
def test_invalid_input_is_rejected_before_repository_tools(inspection_api, payload):
    client, _, snapshot_tool, tree_tool, file_tool = inspection_api
    response = client.post("/api/repositories/inspect", json=payload)

    assert response.status_code == 422
    snapshot_tool.assert_not_called()
    tree_tool.assert_not_called()
    file_tool.assert_not_called()


def test_all_selections_are_checked_before_any_file_download(inspection_api):
    client, _, _, _, file_tool = inspection_api
    response = client.post("/api/repositories/inspect", json={
        "repository_url": URL, "selected_paths": ["README.md", "missing.py"],
    })

    assert response.status_code == 400
    assert response.json()["detail"] == "Selected file is not in the tree."
    file_tool.assert_not_called()


@pytest.mark.parametrize("object_type, mode, size", [
    ("tree", "040000", None),
    ("blob", "120000", 6),
    ("commit", "160000", None),
    ("blob", "100644", None),
    ("blob", "100644", MAX_FILE_BYTES + 1),
])
def test_unavailable_file_entry_stops_content_download(
    inspection_api, object_type, mode, size,
):
    client, tree, _, _, file_tool = inspection_api
    tree.entries[0] = TreeEntry(
        path="README.md", type=object_type, mode=mode, sha="c" * 40, size=size,
    )
    response = client.post("/api/repositories/inspect", json={
        "repository_url": URL, "selected_paths": ["README.md"],
    })

    assert response.status_code == 400
    file_tool.assert_not_called()


@pytest.mark.parametrize("stage", ["snapshot", "tree", "file"])
def test_github_failures_return_controlled_error_and_stop(inspection_api, stage):
    client, _, snapshot_tool, tree_tool, file_tool = inspection_api
    tools = {"snapshot": snapshot_tool, "tree": tree_tool, "file": file_tool}
    tools[stage].side_effect = GitHubClientError("GitHub request timed out.")

    response = client.post("/api/repositories/inspect", json={
        "repository_url": URL, "selected_paths": ["README.md"],
    })

    assert response.status_code == 502
    assert response.json() == {"detail": "GitHub request timed out."}
    assert "files" not in response.json()
    if stage == "snapshot":
        tree_tool.assert_not_called()
    if stage != "file":
        file_tool.assert_not_called()


def test_openapi_exposes_inspection_and_health(inspection_api):
    client, _, snapshot_tool, tree_tool, file_tool = inspection_api
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "post" in paths["/api/repositories/inspect"]
    assert "get" in paths["/api/health"]
    assert "400" in paths["/api/repositories/inspect"]["post"]["responses"]
    assert "502" in paths["/api/repositories/inspect"]["post"]["responses"]
    snapshot_tool.assert_not_called()
    tree_tool.assert_not_called()
    file_tool.assert_not_called()
