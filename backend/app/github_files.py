"""Read bounded UTF-8 text files identified by a verified Git tree."""

import base64
import binascii
import hashlib
from pathlib import PurePosixPath
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from app.github_client import SHA_PATTERN, GitHubClientError, _fetch_response_body
from app.github_tree import RepositoryTree
from app.github_url import parse_github_repository_url


MAX_FILE_BYTES = 100 * 1024
ALLOWED_TEXT_SUFFIXES = frozenset({
    ".py", ".md", ".txt", ".rst", ".toml", ".json", ".yaml", ".yml",
    ".ini", ".cfg", ".js", ".jsx", ".ts", ".tsx", ".html", ".css",
})
ALLOWED_TEXT_NAMES = frozenset({"dockerfile", "makefile", ".gitignore"})
BLOCKED_DIRECTORIES = frozenset({
    ".git", ".ssh", ".venv", "venv", "node_modules", "secrets", "credentials",
})
BLOCKED_NAMES = frozenset({
    "credentials.json", "service-account.json", "id_rsa", "id_ed25519",
})


class RepositoryFile(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    repository_full_name: str
    commit_sha: str = Field(pattern=SHA_PATTERN)
    path: str
    blob_sha: str = Field(pattern=SHA_PATTERN)
    size_bytes: int = Field(ge=0, le=MAX_FILE_BYTES)
    content: str


class _GitBlob(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    sha: str = Field(pattern=SHA_PATTERN)
    size: int = Field(ge=0, le=MAX_FILE_BYTES)
    encoding: Literal["base64"]
    content: str


def _check_file_policy(path: str) -> None:
    parts = [part.casefold() for part in path.split("/")]
    if (
        any(part in BLOCKED_DIRECTORIES for part in parts[:-1])
        or any(part.startswith(".env") for part in parts)
        or parts[-1] in BLOCKED_NAMES
    ):
        raise GitHubClientError("Reading this path is blocked by file policy.")

    name = parts[-1]
    if (
        PurePosixPath(name).suffix not in ALLOWED_TEXT_SUFFIXES
        and name not in ALLOWED_TEXT_NAMES
    ):
        raise GitHubClientError("This file type is not supported for text reading.")


def read_repository_file(
    tree: RepositoryTree,
    path: str,
    *,
    transport: httpx.BaseTransport | None = None,
    token: SecretStr | None = None,
) -> RepositoryFile:
    """Read one permitted regular file without executing or writing its content."""
    try:
        tree = RepositoryTree.model_validate(tree.model_dump())
    except ValidationError:
        raise GitHubClientError("Invalid repository tree.") from None

    matches = [entry for entry in tree.entries if entry.path == path]
    if len(matches) != 1:
        raise GitHubClientError("File path is missing or ambiguous in the tree.")

    entry = matches[0]
    if entry.type != "blob" or entry.mode not in {"100644", "100755"}:
        raise GitHubClientError("Only regular files can be read.")

    _check_file_policy(entry.path)
    if entry.size is None or entry.size > MAX_FILE_BYTES:
        raise GitHubClientError("File size is unknown or exceeds the 100 KiB limit.")

    repository = parse_github_repository_url(
        f"https://github.com/{tree.repository_full_name}"
    )
    body = _fetch_response_body(
        f"/repos/{repository.owner}/{repository.name}/git/blobs/{entry.sha}",
        transport=transport, token=token,
    )

    try:
        blob = _GitBlob.model_validate_json(body)
    except ValidationError:
        raise GitHubClientError("GitHub returned an invalid or oversized blob.") from None

    if blob.sha != entry.sha or blob.size != entry.size:
        raise GitHubClientError("GitHub blob identity or size does not match the tree.")

    encoded = "".join(blob.content.split())
    if len(encoded) > 4 * ((MAX_FILE_BYTES + 2) // 3):
        raise GitHubClientError("Encoded file exceeds the reading limit.")

    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise GitHubClientError("GitHub returned invalid Base64 content.") from None

    if len(data) != blob.size or len(data) > MAX_FILE_BYTES:
        raise GitHubClientError("Decoded file size does not match its metadata.")

    header = f"blob {len(data)}\0".encode("ascii")
    actual_sha = hashlib.sha1(header + data, usedforsecurity=False).hexdigest()
    if actual_sha != entry.sha:
        raise GitHubClientError("File content does not match the expected Git blob.")

    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        raise GitHubClientError("Only UTF-8 text files are supported.") from None

    if any(
        (ord(char) < 32 and char not in "\n\r\t") or ord(char) == 127
        for char in content
    ):
        raise GitHubClientError("Binary or unsupported control characters detected.")

    return RepositoryFile(
        repository_full_name=tree.repository_full_name,
        commit_sha=tree.commit_sha,
        path=entry.path,
        blob_sha=entry.sha,
        size_bytes=len(data),
        content=content,
    )
