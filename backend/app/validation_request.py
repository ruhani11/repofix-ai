"""Bind a validation request to a saved proposal and supplied review snapshot.

This pure module starts no Docker process and grants no execution capability.
The future authenticated service must load both records account-scoped and
atomically recheck current approval when admitting work. A previously returned
binding cannot establish that an approval is still active.
"""

from datetime import datetime, timezone
import json
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator
from pydantic_core import PydanticSerializationError

from app.github_client import SHA_PATTERN
from app.github_oauth import GitHubIdentity
from app.investigation_runner import InvestigationEvent
from app.proposal_history import Digest, Identifier, ProposalResponse
from app.proposal_review import ProposalReviewState


_Model = TypeVar("_Model", bound=BaseModel)
_EVENTS = TypeAdapter(list[InvestigationEvent])
_DETAILS = {
    "invalid_identity": "A valid authenticated GitHub identity is required.",
    "invalid_input": "The validation request is invalid.",
    "invalid_proposal": "The saved proposal could not be verified for validation.",
    "invalid_review": "The proposal review could not be verified for validation.",
    "not_saved": "Save the proposal before requesting validation.",
    "proposal_changed": "The requested proposal, base commit or patch hash does not match.",
    "approval_required": "The proposal needs current approval for validation.",
    "review_changed": "The review decision changed. Refresh its current revision and review again.",
    "approval_mismatch": "The approval does not match this account and saved proposal.",
    "invalid_time": "A valid validation request time at or after approval is required.",
}


class ValidationRequestError(RuntimeError):
    """Stable, safe reasons without source, comments, secrets or raw model errors."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(_DETAILS.get(code, "The validation request could not be prepared."))


class _ValidationModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, hide_input_in_errors=True)


class ValidationInput(_ValidationModel):
    """The exact reviewed draft and approval; never client-selected commands.

    All fields are required. A client cannot choose image, mount, network,
    environment, resource limits or publishing permissions through this input.
    Runtime selection belongs to a later server-owned restricted profile.
    """

    proposal_id: Identifier
    base_commit_sha: str = Field(pattern=SHA_PATTERN)
    patch_sha256: Digest
    expected_revision: int = Field(ge=1, le=2**63 - 1)
    decision_id: Identifier


class ValidationBinding(_ValidationModel):
    """Metadata-only snapshot, not admission, test success or publication consent."""

    proposal_id: Identifier
    investigation_run_id: Identifier
    account_id: int = Field(gt=0, le=2**63 - 1)
    repository_id: int = Field(gt=0, le=2**63 - 1)
    repository_full_name: str = Field(min_length=3, max_length=201)
    base_commit_sha: str = Field(pattern=SHA_PATTERN)
    tree_sha: str = Field(pattern=SHA_PATTERN)
    patch_sha256: Digest
    review_revision: int = Field(ge=1, le=2**63 - 1)
    decision_id: Identifier
    bound_at: datetime

    @field_validator("bound_at")
    @classmethod
    def require_aware_time(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("A timezone-aware binding time is required.")
        return value.astimezone(timezone.utc)


def _validated(value: _Model, model: type[_Model], code: str) -> _Model:
    try:
        if not isinstance(value, model):
            raise ValueError()
        data = value.model_dump(warnings="error")
        if model is ProposalResponse:
            # Reconstruct only dataclass events; preserve strict Python types
            # everywhere else, including bool/int distinctions in stored data.
            data["events"] = _EVENTS.validate_json(json.dumps(data.get("events")), strict=True)
        return model.model_validate(data)
    except (ValidationError, PydanticSerializationError, ValueError, TypeError):
        raise ValidationRequestError(code) from None


def bind_validation_request(
    identity: GitHubIdentity,
    proposal: ProposalResponse,
    review: ProposalReviewState,
    payload: ValidationInput,
    *,
    now: datetime | None = None,
) -> ValidationBinding:
    """Check supplied records without I/O, mutation, AI or target execution.

    Caller supplies an authenticated identity, its saved proposal and its latest
    review from the store. Ownership lookup, CSRF, session expiry, atomic current
    revision admission, pinned source retrieval and sandbox lifecycle remain the
    service's responsibility. now is an internal clock seam, never HTTP input.
    """
    identity = _validated(identity, GitHubIdentity, "invalid_identity")
    if not 0 < identity.id <= 2**63 - 1:
        raise ValidationRequestError("invalid_identity")
    payload = _validated(payload, ValidationInput, "invalid_input")
    proposal = _validated(proposal, ProposalResponse, "invalid_proposal")
    review = _validated(review, ProposalReviewState, "invalid_review")
    if not proposal.proposal_saved:
        raise ValidationRequestError("not_saved")
    if (
        payload.proposal_id != proposal.proposal_id
        or payload.base_commit_sha != proposal.base_commit_sha
        or payload.patch_sha256 != proposal.patch_sha256
    ):
        raise ValidationRequestError("proposal_changed")
    if (
        review.proposal_id != proposal.proposal_id
        or review.base_commit_sha != proposal.base_commit_sha
        or review.patch_sha256 != proposal.patch_sha256
    ):
        raise ValidationRequestError("approval_mismatch")
    decision = review.current_decision
    if review.status != "approved_for_validation" or decision is None:
        raise ValidationRequestError("approval_required")
    if (
        payload.expected_revision != review.revision
        or payload.decision_id != decision.decision_id
    ):
        raise ValidationRequestError("review_changed")
    if (
        decision.decision != "approve_for_validation"
        or decision.account_id != identity.id
        or decision.repository_id != proposal.repository_id
        or decision.repository_full_name != proposal.repository_full_name
        or decision.investigation_run_id != proposal.investigation_run_id
        or decision.tree_sha != proposal.tree_sha
        or decision.created_at < proposal.created_at
    ):
        raise ValidationRequestError("approval_mismatch")
    instant = datetime.now(timezone.utc) if now is None else now
    if (
        not isinstance(instant, datetime) or instant.tzinfo is None
        or instant.utcoffset() is None or instant < decision.created_at
    ):
        raise ValidationRequestError("invalid_time")
    return ValidationBinding(
        proposal_id=proposal.proposal_id, investigation_run_id=proposal.investigation_run_id,
        account_id=identity.id, repository_id=proposal.repository_id,
        repository_full_name=proposal.repository_full_name, base_commit_sha=proposal.base_commit_sha,
        tree_sha=proposal.tree_sha, patch_sha256=proposal.patch_sha256,
        review_revision=review.revision, decision_id=decision.decision_id, bound_at=instant,
    )
