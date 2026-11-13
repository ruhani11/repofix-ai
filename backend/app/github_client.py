"""Read public repository metadata and capture a fixed commit."""

from typing import Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from app.github_url import parse_github_repository_url


MAX_RESPONSE_BYTES = 256 * 1024
SHA_PATTERN = r"^[0-9a-f]{40}$"


class GitHubClientError(RuntimeError):
    """A GitHub request failed or returned an unsupported response."""


class RepositoryMetadata(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    full_name: str = Field(min_length=3, max_length=140)
    default_branch: str = Field(min_length=1)
    private: bool
    visibility: str
    description: str | None = None
    language: str | None = None


class RepositorySnapshot(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    repository: RepositoryMetadata
    commit_sha: str = Field(pattern=SHA_PATTERN)


class _CommitReference(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    type: Literal["commit"]
    sha: str = Field(pattern=SHA_PATTERN)


class _GitReference(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    ref: str
    object: _CommitReference


def _fetch_response_body(
    api_path: str,
    *,
    transport: httpx.BaseTransport | None = None,
    token: SecretStr | None = None,
) -> bytes:
    """Internal GET helper; callers construct paths from validated inputs."""
    headers = {
        "Accept": "application/vnd.github+json",
        "Accept-Encoding": "identity",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "RepoFix-AI/0.1.0",
    }

    if token is not None:
        # A credential may only be sent to the fixed GitHub API origin.
        if not api_path.startswith("/") or api_path.startswith("//") or "\\" in api_path:
            raise GitHubClientError("Invalid GitHub API path.")
        headers["Authorization"] = f"Bearer {token.get_secret_value()}"

    try:
        with httpx.Client(
            base_url="https://api.github.com",
            headers=headers,
            timeout=httpx.Timeout(15.0, connect=5.0),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        ) as client:
            with client.stream("GET", api_path) as response:
                status = response.status_code

                if status == 404:
                    if token is not None:
                        raise GitHubClientError("GitHub resource not found or access is unavailable.")
                    raise GitHubClientError("GitHub resource not found or not public.")

                if status == 429 or (
                    status == 403
                    and (
                        response.headers.get("x-ratelimit-remaining") == "0"
                        or "retry-after" in response.headers
                    )
                ):
                    raise GitHubClientError("GitHub rate limit reached. Try later.")

                if status == 403:
                    raise GitHubClientError("GitHub denied access to this repository.")

                if status == 409:
                    raise GitHubClientError(
                        "Repository has no usable Git history or a reference conflict."
                    )

                if 300 <= status < 400:
                    raise GitHubClientError(
                        "Redirect blocked. Use the repository's current GitHub URL."
                    )

                if status != 200:
                    raise GitHubClientError(f"GitHub returned HTTP {status}.")

                body = bytearray()
                for chunk in response.iter_bytes(chunk_size=4096):
                    if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                        raise GitHubClientError("GitHub response is too large.")
                    body.extend(chunk)

                return bytes(body)

    except httpx.TimeoutException:
        raise GitHubClientError("GitHub request timed out. Try again.") from None
    except httpx.RequestError:
        raise GitHubClientError("Could not connect to GitHub.") from None


def fetch_repository_metadata(
    repository_url: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> RepositoryMetadata:
    """Fetch validated metadata for a public repository."""
    repository = parse_github_repository_url(repository_url)
    api_path = f"/repos/{repository.owner}/{repository.name}"
    body = _fetch_response_body(api_path, transport=transport)

    try:
        metadata = RepositoryMetadata.model_validate_json(body)
    except ValidationError:
        raise GitHubClientError("GitHub returned invalid repository metadata.") from None

    if metadata.private or metadata.visibility != "public":
        raise GitHubClientError("Only public repositories are supported.")

    expected_name = f"{repository.owner}/{repository.name}"
    if metadata.full_name.lower() != expected_name.lower():
        raise GitHubClientError("GitHub returned metadata for a different repository.")

    return metadata


def fetch_repository_snapshot(
    repository_url: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> RepositorySnapshot:
    """Resolve the public repository's default branch to an exact commit SHA."""
    metadata = fetch_repository_metadata(repository_url, transport=transport)
    return _snapshot_from_metadata(metadata, transport=transport)


def _snapshot_from_metadata(
    metadata: RepositoryMetadata,
    *,
    transport: httpx.BaseTransport | None = None,
    token: SecretStr | None = None,
) -> RepositorySnapshot:
    """Resolve a default branch after the caller has verified repository metadata."""
    repository = parse_github_repository_url(f"https://github.com/{metadata.full_name}")
    branch = quote(metadata.default_branch, safe="")
    api_path = f"/repos/{repository.owner}/{repository.name}/git/ref/heads/{branch}"
    body = _fetch_response_body(api_path, transport=transport, token=token)

    try:
        reference = _GitReference.model_validate_json(body)
    except ValidationError:
        raise GitHubClientError("GitHub returned an invalid commit reference.") from None

    if reference.ref != f"refs/heads/{metadata.default_branch}":
        raise GitHubClientError("GitHub returned a reference for a different branch.")

    return RepositorySnapshot(repository=metadata, commit_sha=reference.object.sha)
