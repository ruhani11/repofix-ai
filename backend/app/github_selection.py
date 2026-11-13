"""Revalidate a public repository selection against the connected identity."""

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from app.github_client import GitHubClientError, _fetch_response_body
from app.github_oauth import GitHubIdentity
from app.github_repositories import PublicRepository


class RepositorySelectionError(RuntimeError):
    """The requested selection is stale or unsupported for this account."""


class RepositorySelectionInput(BaseModel):
    """Frontend sends the ID and name from a listing, never an owner or URL."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
    )

    repository_id: int = Field(gt=0, le=2**63 - 1)
    repository_name: str = Field(
        pattern=r"^[A-Za-z0-9_.-]{1,100}$"
    )

    @field_validator("repository_name")
    @classmethod
    def reject_reserved_names(cls, value: str) -> str:
        if value in {".", ".."}:
            raise ValueError("Invalid repository name.")
        return value


def fetch_selected_repository(
    identity: GitHubIdentity,
    selection: RepositorySelectionInput,
    *,
    transport: httpx.BaseTransport | None = None,
) -> PublicRepository:
    """Check current metadata using session-derived identity.

    This read does not persist a selection, capture a commit, or authorize
    repository writes. Revalidate at investigation start; names and ownership
    can change after a listing. Archived repos remain eligible for reading.
    """
    if not isinstance(identity, GitHubIdentity):
        raise ValueError("A verified GitHub identity is required.")

    if not isinstance(selection, RepositorySelectionInput):
        raise ValueError("A validated repository selection is required.")

    # Protect callers that bypass validation with model_construct.
    try:
        identity = GitHubIdentity.model_validate(
            identity.model_dump()
        )
        selection = RepositorySelectionInput.model_validate(
            selection.model_dump()
        )
    except ValidationError:
        raise ValueError(
            "Invalid identity or repository selection."
        ) from None

    body = _fetch_response_body(
        f"/repos/{identity.login}/{selection.repository_name}",
        transport=transport,
    )

    try:
        repository = PublicRepository.model_validate_json(body)
    except ValidationError:
        raise GitHubClientError(
            "GitHub returned unsupported public repository metadata."
        ) from None

    if (
        repository.owner.id != identity.id
        or repository.owner.login.lower() != identity.login.lower()
    ):
        raise RepositorySelectionError(
            "Select a public repository owned by your connected account."
        )

    if (
        repository.id != selection.repository_id
        or repository.name.lower()
        != selection.repository_name.lower()
    ):
        raise RepositorySelectionError(
            "Repository selection changed. Refresh the repository list."
        )

    if repository.disabled:
        raise RepositorySelectionError(
            "This repository is disabled and cannot be selected."
        )

    return repository