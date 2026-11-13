"""Revisioned human review HTTP contracts; no sandbox or publication action."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.github_client import SHA_PATTERN
from app.proposal_decision import ProposalDecision, ProposalDecisionInput
from app.proposal_history import Digest, Identifier


MAX_DECISION_RECORD_BYTES = 16 * 1024


class ProposalReviewError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__("The proposal review could not be completed.")


class ProposalDecisionRequest(ProposalDecisionInput):
    # Required, not silently assumed to be the first decision.
    expected_revision: int = Field(ge=0, le=2**63 - 2)

    def decision_input(self) -> ProposalDecisionInput:
        return ProposalDecisionInput.model_validate(self.model_dump(exclude={"expected_revision"}))


class _ReviewModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, hide_input_in_errors=True)


class ProposalReviewState(_ReviewModel):
    proposal_id: Identifier
    base_commit_sha: str = Field(pattern=SHA_PATTERN)
    patch_sha256: Digest
    revision: int = Field(ge=0, le=2**63 - 1)
    status: Literal["unreviewed", "approved_for_validation", "rejected"]
    current_decision: ProposalDecision | None

    @model_validator(mode="after")
    def consistent_state(self):
        decision = self.current_decision
        if decision is None:
            if self.revision != 0 or self.status != "unreviewed":
                raise ValueError("Invalid unreviewed state.")
        elif (
            self.revision < 1
            or self.status != ("approved_for_validation" if decision.decision == "approve_for_validation" else "rejected")
            or decision.proposal_id != self.proposal_id
            or decision.base_commit_sha != self.base_commit_sha
            or decision.patch_sha256 != self.patch_sha256
        ):
            raise ValueError("Invalid proposal review state.")
        return self


class ProposalDecisionEntry(_ReviewModel):
    revision: int = Field(ge=1, le=2**63 - 1)
    decision: ProposalDecision


class ProposalDecisionPage(_ReviewModel):
    proposal_id: Identifier
    page: int = Field(ge=1, le=10000)
    per_page: int = Field(ge=1, le=50)
    decisions: list[ProposalDecisionEntry] = Field(max_length=50)
    has_more: bool
    next_page: int | None = Field(default=None, ge=1, le=10000)
    limit_reached: bool = False
