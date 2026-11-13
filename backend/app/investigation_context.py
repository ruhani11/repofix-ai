"""Prepare read-only repository context for the future Investigator agent."""

from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from app.github_client import (
    GitHubClientError,
    RepositorySnapshot,
    fetch_repository_snapshot,
)
from app.github_oauth import GitHubIdentity
from app.github_repositories import PublicRepository
from app.github_selection import (
    RepositorySelectionError,
    RepositorySelectionInput,
    fetch_selected_repository,
)
from app.github_tree import RepositoryTree, fetch_repository_tree
from app.investigation_input import InvestigationInput


@dataclass(frozen=True)
class InvestigationContext:
    """Server-owned context; contains input and evidence, not an AI diagnosis."""

    identity: GitHubIdentity
    request: InvestigationInput
    repository: PublicRepository
    snapshot: RepositorySnapshot
    tree: RepositoryTree


def prepare_investigation_context(
    identity: GitHubIdentity,
    payload: InvestigationInput,
    *,
    transport: httpx.BaseTransport | None = None,
) -> InvestigationContext:
    """The API caller must supply identity from its authenticated session.

    Recheck selection, capture a commit, load its bounded tree, then check
    selection again. These reads are not an atomic GitHub transaction.
    No files are downloaded, code executed, or repository changes approved.
    """
    if not isinstance(identity, GitHubIdentity):
        raise ValueError("A verified GitHub identity is required.")
    if not isinstance(payload, InvestigationInput):
        raise ValueError("A validated investigation input is required.")

    # Protect internal callers that bypass validation with model_construct.
    try:
        identity = GitHubIdentity.model_validate(identity.model_dump())
        payload = InvestigationInput.model_validate(payload.model_dump())
    except ValidationError:
        raise ValueError("Invalid identity or investigation input.") from None

    selection = RepositorySelectionInput(
        repository_id=payload.repository_id,
        repository_name=payload.repository_name,
    )
    repository = fetch_selected_repository(
        identity, selection, transport=transport
    )
    snapshot = fetch_repository_snapshot(
        repository.html_url, transport=transport
    )

    if snapshot.repository.full_name.lower() != repository.full_name.lower():
        raise GitHubClientError("Repository snapshot does not match the selection.")

    tree = fetch_repository_tree(snapshot, transport=transport)
    if (
        tree.repository_full_name.lower() != snapshot.repository.full_name.lower()
        or tree.commit_sha != snapshot.commit_sha
    ):
        raise GitHubClientError("Repository tree does not match the snapshot.")

    current_repository = fetch_selected_repository(
        identity, selection, transport=transport
    )
    if (
        current_repository.full_name.lower()
        != snapshot.repository.full_name.lower()
        or current_repository.default_branch != snapshot.repository.default_branch
    ):
        raise RepositorySelectionError(
            "Repository changed during preparation. Select again."
        )

    return InvestigationContext(
        identity=identity,
        request=payload,
        repository=current_repository,
        snapshot=snapshot,
        tree=tree,
    )
