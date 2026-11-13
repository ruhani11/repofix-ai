"""Persistable, internally consistent proposal artifacts; never an approval."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.fix_proposal import FixProposal, MAX_EDITS, MAX_TEST_FILES, _repository_path
from app.github_client import SHA_PATTERN
from app.investigation_runner import MAX_EVENTS, InvestigationEvent
from app.proposal_validation import (
    MAX_DIFF_BYTES, MAX_PROPOSED_BYTES, MAX_SOURCE_BYTES, ValidatedFileChange,
    _encode_source, _file_diff, _git_blob_sha,
)


Identifier = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Record(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, hide_input_in_errors=True)


class ProposalFileChange(_Record):
    path: str = Field(min_length=1, max_length=1024)
    action: Literal["modify", "create"]
    mode: Literal["100644", "100755"]
    original_blob_sha: str | None = Field(pattern=SHA_PATTERN)
    proposed_blob_sha: str = Field(pattern=SHA_PATTERN)
    original_content: str | None = Field(max_length=100 * 1024)
    proposed_content: str = Field(max_length=100 * 1024)

    @field_validator("path")
    @classmethod
    def path_policy(cls, value):
        return _repository_path(value)


class ProposalArtifact(_Record):
    investigation_run_id: Identifier
    repository_id: int = Field(gt=0, le=2**63 - 1)
    repository_full_name: str = Field(min_length=3, max_length=201)
    base_commit_sha: str = Field(pattern=SHA_PATTERN)
    tree_sha: str = Field(pattern=SHA_PATTERN)
    draft: FixProposal
    changes: list[ProposalFileChange] = Field(min_length=2, max_length=MAX_EDITS + MAX_TEST_FILES)
    unified_diff: str = Field(min_length=1, max_length=MAX_DIFF_BYTES)
    patch_sha256: Digest

    @model_validator(mode="after")
    def consistent_artifact(self):
        """Check stored bytes, anchors and diff; no new remote/source attestation."""
        try:
            paths = [change.path for change in self.changes]
            edits = {edit.path: edit for edit in self.draft.edits}
            tests = {test.path: test for test in self.draft.tests}
            if paths != sorted(set(paths)) or set(paths) != set(edits) | set(tests):
                raise ValueError("Invalid proposal targets.")
            original_bytes = proposed_bytes = 0
            rendered = []
            for change in self.changes:
                data = _encode_source(change.proposed_content)
                proposed_bytes += len(data)
                if _git_blob_sha(data) != change.proposed_blob_sha:
                    raise ValueError("Invalid proposed blob.")
                if change.path in edits:
                    if change.action != "modify" or change.original_content is None:
                        raise ValueError("Invalid original file.")
                    original = _encode_source(change.original_content)
                    original_bytes += len(original)
                    if _git_blob_sha(original) != change.original_blob_sha:
                        raise ValueError("Invalid original blob.")
                    edit = edits[change.path]
                    first = change.original_content.find(edit.before)
                    if first < 0 or change.original_content.find(edit.before, first + 1) >= 0:
                        raise ValueError("Invalid source anchor.")
                    expected = change.original_content[:first] + edit.after + change.original_content[first + len(edit.before):]
                    if expected != change.proposed_content:
                        raise ValueError("Invalid replacement.")
                elif (
                    change.action != "create" or change.mode != "100644"
                    or change.original_content is not None or change.original_blob_sha is not None
                    or change.proposed_content != tests[change.path].content
                ):
                    raise ValueError("Invalid new test file.")
                rendered.append(_file_diff(ValidatedFileChange(**change.model_dump())))
            patch = "".join(rendered)
            encoded = patch.encode("utf-8")
            if (
                original_bytes > MAX_SOURCE_BYTES or proposed_bytes > MAX_PROPOSED_BYTES
                or len(encoded) > MAX_DIFF_BYTES or patch != self.unified_diff
                or hashlib.sha256(encoded).hexdigest() != self.patch_sha256
            ):
                raise ValueError("Invalid proposal diff.")
        except Exception:
            raise ValueError("Proposal artifact consistency check failed.") from None
        return self


class ProposalResponse(ProposalArtifact):
    status: Literal["proposed"] = "proposed"
    proposal_id: Identifier
    created_at: datetime
    total_duration_ms: int = Field(ge=0)
    agent_duration_ms: int = Field(ge=0)
    model_calls: int = Field(ge=1, le=20)
    tool_attempts: int = Field(ge=0, le=20)
    repository_stats: dict[str, int]
    events: list[InvestigationEvent] = Field(max_length=MAX_EVENTS)
    proposal_saved: bool = False

    @field_validator("created_at")
    @classmethod
    def utc_time(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("A UTC creation time is required.")
        return value.astimezone(timezone.utc)

    @field_validator("repository_stats")
    @classmethod
    def metadata_only(cls, value):
        if set(value) - {"tool_calls", "file_fetch_attempts", "downloaded_bytes"} or any(
            type(count) is not int or count < 0 for count in value.values()
        ):
            raise ValueError("Invalid repository counters.")
        return value


class ProposalHistoryItem(_Record):
    status: Literal["proposed"] = "proposed"
    proposal_id: Identifier
    investigation_run_id: Identifier
    created_at: datetime
    repository_id: int = Field(gt=0)
    repository_full_name: str = Field(min_length=3, max_length=201)
    base_commit_sha: str = Field(pattern=SHA_PATTERN)
    title: str = Field(min_length=1, max_length=2000)
    patch_sha256: Digest
    total_duration_ms: int = Field(ge=0)


class ProposalHistoryPage(_Record):
    page: int = Field(ge=1, le=10000)
    per_page: int = Field(ge=1, le=50)
    proposals: list[ProposalHistoryItem] = Field(max_length=50)
    has_more: bool
    next_page: int | None = Field(default=None, ge=1, le=10000)
    limit_reached: bool = False


def proposal_response(result, *, total_duration_ms: int) -> ProposalResponse:
    """Deep revalidation through JSON also rejects mutated nested draft models."""
    from app.proposal_generator import ProposalGenerationResult

    if not isinstance(result, ProposalGenerationResult):
        raise ValueError("Invalid proposal generation result.")
    data = asdict(result.validated)
    data.update(
        proposal_id=result.proposal_id, created_at=datetime.now(timezone.utc).isoformat(),
        total_duration_ms=total_duration_ms, agent_duration_ms=result.duration_ms,
        model_calls=result.model_calls, tool_attempts=result.tool_attempts,
        repository_stats=result.repository_stats, events=[asdict(e) for e in result.events],
        proposal_saved=True,
    )
    return ProposalResponse.model_validate_json(json.dumps(data, default=lambda model: model.model_dump()))
