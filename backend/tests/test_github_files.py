"""Verify bounded text reading, content integrity, and file policy offline."""

import base64
import hashlib

import httpx
import pytest

from app.github_client import GitHubClientError
from app.github_files import MAX_FILE_BYTES, read_repository_file
from app.github_tree import RepositoryTree, TreeEntry


def make_file(data=b"hello\n", *, path="README.md", mode="100644", object_type="blob"):
    header = f"blob {len(data)}\0".encode("ascii")
    sha = hashlib.sha1(header + data, usedforsecurity=False).hexdigest()
    tree = RepositoryTree(
        repository_full_name="owner/repo",
        commit_sha="a" * 40,
        tree_sha="b" * 40,
        entries=[TreeEntry(
            path=path, type=object_type, mode=mode, sha=sha, size=len(data)
        )],
    )
    blob = {
        "sha": sha,
        "size": len(data),
        "encoding": "base64",
        "content": base64.b64encode(data).decode("ascii"),
    }
    return tree, blob


def make_transport(tree, blob):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "GET"
        assert "Authorization" not in request.headers
        expected_url = (
            "https://api.github.com/repos/owner/repo/git/blobs/"
            + tree.entries[0].sha
        )
        assert str(request.url) == expected_url
        return httpx.Response(200, json=blob)

    return httpx.MockTransport(handler), requests


def test_utf8_content_and_provenance():
    text = "# Demo\nनमस्ते 🌍\n"
    data = text.encode("utf-8")
    tree, blob = make_file(data)
    transport, requests = make_transport(tree, blob)

    result = read_repository_file(tree, "README.md", transport=transport)

    assert result.content == text
    assert result.repository_full_name == "owner/repo"
    assert result.commit_sha == tree.commit_sha
    assert result.path == "README.md"
    assert result.blob_sha == tree.entries[0].sha
    assert result.size_bytes == len(data)
    assert result.size_bytes > len(text)
    assert len(requests) == 1


def test_empty_file_has_known_git_blob_hash():
    tree, blob = make_file(b"")
    transport, _ = make_transport(tree, blob)

    result = read_repository_file(tree, "README.md", transport=transport)

    assert result.content == ""
    assert result.size_bytes == 0
    assert result.blob_sha == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_wrapped_base64_is_supported():
    tree, blob = make_file()
    encoded = blob["content"]
    blob["content"] = encoded[:4] + "\n" + encoded[4:] + "\n"
    transport, _ = make_transport(tree, blob)

    result = read_repository_file(tree, "README.md", transport=transport)

    assert result.content == "hello\n"


def test_missing_path_stops_before_network():
    tree, blob = make_file()
    transport, requests = make_transport(tree, blob)

    with pytest.raises(GitHubClientError, match="missing or ambiguous"):
        read_repository_file(tree, "missing.py", transport=transport)

    assert requests == []


def test_ambiguous_path_stops_before_network():
    tree, blob = make_file()
    tree.entries.append(tree.entries[0].model_copy())
    transport, requests = make_transport(tree, blob)

    with pytest.raises(GitHubClientError, match="missing or ambiguous"):
        read_repository_file(tree, "README.md", transport=transport)

    assert requests == []


def test_invalid_tree_stops_before_network():
    tree, blob = make_file()
    tree.commit_sha = "invalid"
    transport, requests = make_transport(tree, blob)

    with pytest.raises(GitHubClientError, match="Invalid repository tree"):
        read_repository_file(tree, "README.md", transport=transport)

    assert requests == []


@pytest.mark.parametrize(
    "object_type, mode",
    [("tree", "040000"), ("blob", "120000"), ("commit", "160000")],
)
def test_nonregular_files_stopped_before_network(object_type, mode):
    tree, blob = make_file(mode=mode, object_type=object_type)
    transport, requests = make_transport(tree, blob)

    with pytest.raises(GitHubClientError, match="Only regular files"):
        read_repository_file(tree, "README.md", transport=transport)

    assert requests == []


@pytest.mark.parametrize(
    "path",
    [".env", ".env.example", "credentials.json", "secrets/config.json", "node_modules/lib.js"],
)
def test_blocked_paths_stopped_before_network(path):
    tree, blob = make_file(path=path)
    transport, requests = make_transport(tree, blob)

    with pytest.raises(GitHubClientError, match="blocked by file policy"):
        read_repository_file(tree, path, transport=transport)

    assert requests == []


def test_unsupported_file_type_stopped_before_network():
    tree, blob = make_file(path="image.png")
    transport, requests = make_transport(tree, blob)

    with pytest.raises(GitHubClientError, match="file type is not supported"):
        read_repository_file(tree, "image.png", transport=transport)

    assert requests == []


def test_unknown_file_size_stopped_before_network():
    tree, blob = make_file()
    tree.entries[0].size = None
    transport, requests = make_transport(tree, blob)

    with pytest.raises(GitHubClientError, match="File size is unknown"):
        read_repository_file(tree, "README.md", transport=transport)

    assert requests == []


@pytest.mark.parametrize("size", [MAX_FILE_BYTES, MAX_FILE_BYTES + 1])
def test_file_size_boundary(size):
    tree, blob = make_file(b"x" * size)
    transport, requests = make_transport(tree, blob)

    if size > MAX_FILE_BYTES:
        with pytest.raises(GitHubClientError, match="100 KiB limit"):
            read_repository_file(tree, "README.md", transport=transport)
        assert requests == []
    else:
        result = read_repository_file(tree, "README.md", transport=transport)
        assert result.size_bytes == size


@pytest.mark.parametrize(
    "changes",
    [{"encoding": "utf-8"}, {"sha": "invalid"}, {"size": "6"}, {"content": None}],
)
def test_invalid_blob_metadata_is_rejected(changes):
    tree, blob = make_file()
    blob.update(changes)
    transport, _ = make_transport(tree, blob)

    with pytest.raises(GitHubClientError, match="invalid or oversized blob"):
        read_repository_file(tree, "README.md", transport=transport)


def test_wrong_blob_identity_is_rejected():
    tree, blob = make_file()
    blob["sha"] = "d" * 40
    transport, _ = make_transport(tree, blob)

    with pytest.raises(GitHubClientError, match="identity or size does not match"):
        read_repository_file(tree, "README.md", transport=transport)


def test_wrong_declared_size_is_rejected():
    tree, blob = make_file()
    blob["size"] += 1
    transport, _ = make_transport(tree, blob)

    with pytest.raises(GitHubClientError, match="identity or size does not match"):
        read_repository_file(tree, "README.md", transport=transport)


@pytest.mark.parametrize("encoded", ["!bad!", "非ascii"])
def test_invalid_base64_is_rejected(encoded):
    tree, blob = make_file()
    blob["content"] = encoded
    transport, _ = make_transport(tree, blob)

    with pytest.raises(GitHubClientError, match="invalid Base64"):
        read_repository_file(tree, "README.md", transport=transport)


def test_wrong_decoded_size_is_rejected():
    tree, blob = make_file()
    blob["content"] = base64.b64encode(b"longer content").decode("ascii")
    transport, _ = make_transport(tree, blob)

    with pytest.raises(GitHubClientError, match="Decoded file size"):
        read_repository_file(tree, "README.md", transport=transport)


def test_same_size_content_tampering_is_rejected():
    tree, blob = make_file(b"hello\n")
    blob["content"] = base64.b64encode(b"world\n").decode("ascii")
    transport, _ = make_transport(tree, blob)

    with pytest.raises(GitHubClientError, match="expected Git blob"):
        read_repository_file(tree, "README.md", transport=transport)


@pytest.mark.parametrize(
    "data, message",
    [(b"\xff", "Only UTF-8"), (b"hello\0binary", "control characters")],
)
def test_unsupported_text_content_is_rejected(data, message):
    tree, blob = make_file(data)
    transport, _ = make_transport(tree, blob)

    with pytest.raises(GitHubClientError, match=message):
        read_repository_file(tree, "README.md", transport=transport)


def test_oversized_encoded_content_is_rejected():
    tree, blob = make_file()
    blob["content"] = "A" * (4 * ((MAX_FILE_BYTES + 2) // 3) + 4)
    transport, _ = make_transport(tree, blob)

    with pytest.raises(GitHubClientError, match="Encoded file exceeds"):
        read_repository_file(tree, "README.md", transport=transport)
