"""Fetch a saved proposal's pinned Python sources for later sandbox admission.

Read-only preparation only: no target checkout, Docker, model call or execution.
The future API derives identity from its session, applies CSRF/admission limits
and must atomically recheck current review before starting a validation job.
"""

from dataclasses import dataclass
import time

import httpx

from app.fix_proposal import FixProposalInput
from app.github_client import GitHubClientError
from app.github_files import read_repository_file
from app.github_oauth import GitHubIdentity
from app.github_selection import RepositorySelectionError, RepositorySelectionInput, fetch_selected_repository
from app.investigation_store import InvestigationStore
from app.proposal_context import ProposalContextError, prepare_proposal_context
from app.proposal_history import ProposalResponse
from app.proposal_review import ProposalReviewError
from app.proposal_store import ProposalStore, ProposalStoreError
from app.validation_request import ValidationInput, ValidationRequestError, _validated, bind_validation_request
from app.validation_workspace import (
    ValidationWorkspace, ValidationWorkspaceError, prepare_validation_workspace, select_validation_paths,
)


VALIDATION_PREPARATION_TIMEOUT_SECONDS = 60


class ValidationContextError(RuntimeError):
    """Safe internal reason without source, store paths or provider payloads."""

    def __init__(self, code: str):
        self.code = code
        super().__init__("Could not prepare the approved proposal for sandbox validation.")


@dataclass(frozen=True)
class ValidationContext:
    """Bounded in-memory preparation result; never proof of active execution consent."""

    workspace: ValidationWorkspace
    preparation_duration_ms: int
    file_fetch_attempts: int
    downloaded_bytes: int


def prepare_validation_context(
    identity: GitHubIdentity,
    payload: ValidationInput,
    proposal_store: ProposalStore,
    history_store: InvestigationStore,
    *,
    transport: httpx.BaseTransport | None = None,
) -> ValidationContext:
    """Own saved records, current public ownership and original commit only.

    Reject missing/foreign/unapproved/stale requests before GitHub access.
    Recheck the latest review before each blob and after the final ownership
    check. These are point-in-time checks, not atomic job admission. The deadline
    covers completed operations but cannot terminate an in-flight synchronous
    read or SQLite wait; the existing reader/store also have their own timeouts.
    transport is an internal test seam, not an HTTP/browser input.
    """
    started = time.monotonic()
    deadline = started + VALIDATION_PREPARATION_TIMEOUT_SECONDS

    def check_time():
        if time.monotonic() >= deadline:
            raise ValidationContextError("timeout")

    try:
        identity = _validated(identity, GitHubIdentity, "invalid_identity")
        payload = _validated(payload, ValidationInput, "invalid_input")
    except ValidationRequestError as error:
        raise ValidationContextError(error.code) from None
    if not 0 < identity.id <= 2**63 - 1:
        raise ValidationContextError("invalid_identity")
    if not isinstance(proposal_store, ProposalStore) or not isinstance(history_store, InvestigationStore):
        raise ValidationContextError("invalid_store")
    check_time()
    try:
        proposal = proposal_store.get(identity.id, payload.proposal_id)
    except ProposalStoreError:
        raise ValidationContextError("proposal_unavailable") from None
    if proposal is None:
        raise ValidationContextError("not_found")
    try:
        proposal = _validated(proposal, ProposalResponse, "invalid_proposal")
    except ValidationRequestError as error:
        raise ValidationContextError(error.code) from None

    def current_review():
        check_time()
        try:
            review = proposal_store.get_review(identity.id, proposal.proposal_id)
        except ProposalReviewError as error:
            raise ValidationContextError("not_found" if error.code == "not_found" else "review_unavailable") from None
        try:
            bind_validation_request(identity, proposal, review, payload)
        except ValidationRequestError as error:
            raise ValidationContextError(error.code) from None
        check_time()
        return review

    current_review()  # includes exact requested proposal/base/patch/revision/decision
    check_time()
    try:
        prepared = prepare_proposal_context(
            identity, FixProposalInput(investigation_run_id=proposal.investigation_run_id),
            history_store, transport=transport,
        )
    except ProposalContextError as error:
        raise ValidationContextError(error.code) from None
    check_time()
    context = prepared.repository_context
    if (
        prepared.history.repository_id != proposal.repository_id
        or context.repository.id != proposal.repository_id
        or context.tree.repository_full_name != proposal.repository_full_name
        or context.tree.commit_sha != proposal.base_commit_sha
        or context.tree.tree_sha != proposal.tree_sha
    ):
        raise ValidationContextError("provenance_mismatch")
    try:
        paths = select_validation_paths(context.tree)
    except ValidationWorkspaceError as error:
        raise ValidationContextError(error.code) from None
    sources = []
    attempts = downloaded = 0
    for path in paths:
        current_review()
        attempts += 1
        try:
            source = read_repository_file(context.tree, path, transport=transport)
        except GitHubClientError:
            raise ValidationContextError("github_unavailable") from None
        check_time()
        sources.append(source)
        downloaded += source.size_bytes
    # Existing context preparation checks metadata around tree reads. Check
    # current public ownership once more after the bounded blob reads.
    check_time()
    try:
        current = fetch_selected_repository(identity, RepositorySelectionInput(
            repository_id=proposal.repository_id, repository_name=context.repository.name,
        ), transport=transport)
    except RepositorySelectionError:
        raise ValidationContextError("selection_changed") from None
    except GitHubClientError:
        raise ValidationContextError("github_unavailable") from None
    if current.full_name != proposal.repository_full_name:
        raise ValidationContextError("selection_changed")
    review = current_review()
    try:
        workspace = prepare_validation_workspace(identity, proposal, review, payload, context.tree, sources)
    except ValidationWorkspaceError as error:
        raise ValidationContextError(error.code) from None
    check_time()
    return ValidationContext(
        workspace=workspace, preparation_duration_ms=max(0, int((time.monotonic() - started) * 1000)),
        file_fetch_attempts=attempts, downloaded_bytes=downloaded,
    )
