"""Completed investigation response and account-scoped history contracts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.github_client import SHA_PATTERN
from app.investigation_report import InvestigationReport
from app.investigation_runner import MAX_EVENTS, InvestigationEvent


class InvestigationRunResponse(BaseModel):
    """Validated report with measured metadata and explicit storage outcome."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    status: Literal["completed"] = "completed"
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    repository_full_name: str = Field(min_length=3, max_length=201)
    commit_sha: str = Field(pattern=SHA_PATTERN)
    report: InvestigationReport
    total_duration_ms: int = Field(ge=0)
    agent_duration_ms: int = Field(ge=0)
    model_calls: int = Field(ge=1, le=20)
    tool_attempts: int = Field(ge=0, le=20)
    repository_stats: dict[str, int]
    events: list[InvestigationEvent] = Field(max_length=MAX_EVENTS)
    history_saved: bool = Field(
        default=False,
        description="Whether this completed report was saved to local history.",
    )


class InvestigationHistoryItem(BaseModel):
    """Small listing entry; quotes and original issue text belong in detail."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    created_at: datetime
    repository_id: int = Field(gt=0)
    repository_name: str = Field(min_length=1, max_length=100)
    repository_full_name: str = Field(min_length=3, max_length=201)
    commit_sha: str = Field(pattern=SHA_PATTERN)
    report_status: Literal["likely_cause", "insufficient_evidence"]
    issue_summary: str = Field(min_length=1, max_length=2000)
    total_duration_ms: int = Field(ge=0)


class InvestigationHistoryDetail(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    created_at: datetime
    repository_id: int = Field(gt=0)
    repository_name: str = Field(min_length=1, max_length=100)
    issue_description: str = Field(min_length=1, max_length=8000)
    investigation: InvestigationRunResponse


class InvestigationHistoryPage(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    page: int = Field(ge=1, le=10000)
    per_page: int = Field(ge=1, le=50)
    investigations: list[InvestigationHistoryItem] = Field(max_length=50)
    has_more: bool
    next_page: int | None = Field(default=None, ge=1, le=10000)
    limit_reached: bool = False
