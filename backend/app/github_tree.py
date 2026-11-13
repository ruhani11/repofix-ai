"""Read a bounded repository tree from a captured commit."""

from typing import Literal

import httpx
from pydantic import (
    BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator, model_validator,
)

from app.github_client import (
    SHA_PATTERN, GitHubClientError, RepositorySnapshot, _fetch_response_body,
)
from app.github_url import parse_github_repository_url


MAX_TREE_ENTRIES = 1000


class TreeEntry(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    path: str = Field(min_length=1, max_length=1024)
    type: Literal["blob", "tree", "commit"]
    mode: Literal["100644", "100755", "040000", "120000", "160000"]
    sha: str = Field(pattern=SHA_PATTERN)
    size: int | None = Field(default=None, ge=0)

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if "\\" in value or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("Invalid repository path characters.")
        if any(part in {"", ".", "..", ".git"} for part in value.split("/")):
            raise ValueError("Expected a safe relative repository path.")
        return value

    @model_validator(mode="after")
    def validate_object_mode(self) -> "TreeEntry":
        modes = {
            "blob": {"100644", "100755", "120000"},
            "tree": {"040000"},
            "commit": {"160000"},
        }
        if self.mode not in modes[self.type]:
            raise ValueError("Git object type and mode do not match.")
        return self


class RepositoryTree(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    repository_full_name: str
    commit_sha: str = Field(pattern=SHA_PATTERN)
    tree_sha: str = Field(pattern=SHA_PATTERN)
    entries: list[TreeEntry] = Field(max_length=MAX_TREE_ENTRIES)


class _TreePointer(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    sha: str = Field(pattern=SHA_PATTERN)


class _GitCommit(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    sha: str = Field(pattern=SHA_PATTERN)
    tree: _TreePointer


class _GitTree(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    sha: str = Field(pattern=SHA_PATTERN)
    truncated: bool
    tree: list[TreeEntry] = Field(max_length=MAX_TREE_ENTRIES)


def fetch_repository_tree(
    snapshot: RepositorySnapshot,
    *,
    transport: httpx.BaseTransport | None = None,
    token: SecretStr | None = None,
) -> RepositoryTree:
    """Read tree metadata only; do not read files or follow links/submodules."""
    try:
        snapshot = RepositorySnapshot.model_validate(snapshot.model_dump())
    except ValidationError:
        raise GitHubClientError("Invalid repository snapshot.") from None

    if token is None and (snapshot.repository.private or snapshot.repository.visibility != "public"):
        raise GitHubClientError("Only public repositories are supported.")

    repository = parse_github_repository_url(
        f"https://github.com/{snapshot.repository.full_name}"
    )
    prefix = f"/repos/{repository.owner}/{repository.name}/git"
    body = _fetch_response_body(
        f"{prefix}/commits/{snapshot.commit_sha}", transport=transport, token=token
    )

    try:
        commit = _GitCommit.model_validate_json(body)
    except ValidationError:
        raise GitHubClientError("GitHub returned an invalid Git commit.") from None

    if commit.sha != snapshot.commit_sha:
        raise GitHubClientError("GitHub returned a different commit.")

    body = _fetch_response_body(
        f"{prefix}/trees/{commit.tree.sha}?recursive=1", transport=transport, token=token
    )
    try:
        tree = _GitTree.model_validate_json(body)
    except ValidationError:
        raise GitHubClientError(
            "Invalid Git tree or the 1000-entry limit was exceeded."
        ) from None

    if tree.sha != commit.tree.sha:
        raise GitHubClientError("GitHub returned a different tree.")
    if tree.truncated:
        raise GitHubClientError("GitHub returned an incomplete tree listing.")
    if len({entry.path for entry in tree.tree}) != len(tree.tree):
        raise GitHubClientError("GitHub returned duplicate tree paths.")

    return RepositoryTree(
        repository_full_name=snapshot.repository.full_name,
        commit_sha=snapshot.commit_sha,
        tree_sha=tree.sha,
        entries=tree.tree,
    )
