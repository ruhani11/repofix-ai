"""Load an account's saved investigation and prepare its original commit tree."""

from dataclasses import dataclass

import httpx
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from app.fix_proposal import FixProposalInput
from app.github_client import GitHubClientError, RepositoryMetadata, RepositorySnapshot
from app.github_oauth import GitHubIdentity
from app.github_selection import (
    RepositorySelectionError, RepositorySelectionInput, fetch_selected_repository,
)
from app.github_tree import fetch_repository_tree
from app.investigation_context import InvestigationContext
from app.investigation_history import InvestigationHistoryDetail
from app.investigation_input import InvestigationInput
from app.investigation_store import InvestigationStore, InvestigationStoreError
from app.proposal_validation import ProposalValidationError, _revalidate


class ProposalContextError(RuntimeError):
    """Controlled internal reason, without account/source/provider payloads."""

    def __init__(self, code: str):
        self.code = code
        super().__init__("Could not prepare the saved investigation for a fix proposal.")


@dataclass(frozen=True)
class ProposalContext:
    """History is account scoped; existing bounded readers can reuse the context.

    This object prepares source metadata, not a new diagnosis or approved fix.
    The future HTTP layer must derive identity from its session and recheck that
    session after synchronous preparation and before any paid model work.
    """

    history: InvestigationHistoryDetail
    repository_context: InvestigationContext


def prepare_proposal_context(
    identity: GitHubIdentity,
    payload: FixProposalInput,
    history_store: InvestigationStore,
    *,
    transport: httpx.BaseTransport | None = None,
) -> ProposalContext:
    """Use the saved SHA directly; never recapture a default-branch head.

    Missing and another account's records are indistinguishable and stop before
    any GitHub request. Current public ownership/selection is checked before and
    after commit/tree reads. These checks are not an atomic GitHub transaction.
    Only metadata/tree GETs occur; source file reads and AI generation are later.
    """
    if (
        not isinstance(identity, GitHubIdentity)
        or not isinstance(payload, FixProposalInput)
        or not isinstance(history_store, InvestigationStore)
    ):
        raise ProposalContextError("invalid_input")
    try:
        identity = GitHubIdentity.model_validate(identity.model_dump(warnings="error"))
        payload = FixProposalInput.model_validate(payload.model_dump(warnings="error"))
    except (ValidationError, PydanticSerializationError, ValueError, TypeError):
        raise ProposalContextError("invalid_input") from None

    try:
        history = history_store.get(identity.id, payload.investigation_run_id)
    except InvestigationStoreError:
        raise ProposalContextError("history_unavailable") from None
    if history is None:
        raise ProposalContextError("not_found")
    if not isinstance(history, InvestigationHistoryDetail):
        raise ProposalContextError("invalid_history")
    try:
        history = _revalidate(history, InvestigationHistoryDetail)
        selection = RepositorySelectionInput(
            repository_id=history.repository_id, repository_name=history.repository_name,
        )
        request = InvestigationInput(
            repository_id=selection.repository_id, repository_name=selection.repository_name,
            issue_description=history.issue_description,
        )
    except (ProposalValidationError, ValidationError, PydanticSerializationError, ValueError, TypeError):
        raise ProposalContextError("invalid_history") from None

    investigation = history.investigation
    if investigation.run_id != payload.investigation_run_id:
        raise ProposalContextError("invalid_history")
    if not investigation.history_saved or investigation.report.status != "likely_cause":
        raise ProposalContextError("ineligible_investigation")
    expected_full_name = investigation.repository_full_name.casefold()

    try:
        repository = fetch_selected_repository(identity, selection, transport=transport)
        if repository.full_name.casefold() != expected_full_name:
            raise ProposalContextError("selection_changed")
        # Crucially, do not call fetch_repository_snapshot (which would read HEAD).
        snapshot = RepositorySnapshot(
            repository=RepositoryMetadata.model_validate(repository.model_dump()),
            commit_sha=investigation.commit_sha,
        )
        tree = fetch_repository_tree(snapshot, transport=transport)
        if tree.repository_full_name.casefold() != expected_full_name or tree.commit_sha != investigation.commit_sha:
            raise ProposalContextError("provenance_mismatch")

        current_repository = fetch_selected_repository(identity, selection, transport=transport)
        if current_repository.full_name.casefold() != expected_full_name:
            raise ProposalContextError("selection_changed")
    except RepositorySelectionError:
        raise ProposalContextError("selection_changed") from None
    except GitHubClientError:
        raise ProposalContextError("github_unavailable") from None

    # A moved branch or a changed default branch cannot rebind the saved SHA.
    context = InvestigationContext(
        identity=identity, request=request, repository=current_repository,
        snapshot=snapshot, tree=tree,
    )
    return ProposalContext(history=history, repository_context=context)
