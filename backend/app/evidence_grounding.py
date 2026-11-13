"""Match report citations to server-observed reads from one pinned repository."""

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.github_client import SHA_PATTERN, GitHubClientError, RepositorySnapshot
from app.github_files import MAX_FILE_BYTES, _check_file_policy
from app.github_repositories import PublicRepository
from app.github_tree import RepositoryTree
from app.investigation_context import InvestigationContext
from app.investigation_report import EvidenceReference, InvestigationReport
from app.repository_tools import (
    MAX_EXCERPT_CHARACTERS,
    MAX_EXCERPT_LINES,
    MAX_TOOL_CALLS,
)


class EvidenceGroundingError(RuntimeError):
    """The report or observed source evidence could not be verified."""


class _ObservedRead(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, hide_input_in_errors=True
    )

    repository_full_name: str = Field(min_length=1, max_length=201)
    commit_sha: str = Field(pattern=SHA_PATTERN)
    blob_sha: str = Field(pattern=SHA_PATTERN)
    path: str = Field(min_length=1, max_length=1024)
    start_line: int = Field(ge=1)
    end_line: int | None = Field(ge=1)
    total_lines: int = Field(ge=0, le=MAX_FILE_BYTES)
    content: str = Field(max_length=MAX_EXCERPT_CHARACTERS)
    truncated: bool
    last_line_may_be_partial: bool

    @model_validator(mode="after")
    def validate_excerpt(self) -> "_ObservedRead":
        if self.end_line is None:
            if (
                self.total_lines != 0 or self.content or self.start_line != 1
                or self.truncated or self.last_line_may_be_partial
            ):
                raise ValueError("Invalid empty-file excerpt.")
            return self

        span = self.end_line - self.start_line + 1
        if (
            span < 1 or span > MAX_EXCERPT_LINES
            or self.end_line > self.total_lines
            or len(self.content.split("\n")) != span
            or self.truncated != (
                self.last_line_may_be_partial or self.end_line < self.total_lines
            )
            or (
                self.last_line_may_be_partial
                and len(self.content) != MAX_EXCERPT_CHARACTERS
            )
        ):
            raise ValueError("Invalid source excerpt metadata.")
        return self


def _matches(citation: EvidenceReference, observed: _ObservedRead) -> bool:
    if citation.path != observed.path or observed.end_line is None:
        return False

    # Conservatively exclude the last line when its text may be incomplete.
    complete_end = observed.end_line - int(observed.last_line_may_be_partial)
    if citation.start_line < observed.start_line or citation.end_line > complete_end:
        return False

    lines = observed.content.split("\n")
    start = citation.start_line - observed.start_line
    end = citation.end_line - observed.start_line + 1
    return citation.quote in "\n".join(lines[start:end])


def validate_report_grounding(
    report: InvestigationReport,
    context: InvestigationContext,
    observed_reads: list[dict[str, object]],
) -> InvestigationReport:
    """Use successful read_file data captured by the server during THIS run.

    Never accept observed_reads from client input or model-authored citations.
    Each citation must fit one complete observed excerpt and quote it exactly.
    This validates source grounding, not root-cause correctness. No network IO.
    """
    if (
        not isinstance(report, InvestigationReport)
        or not isinstance(context, InvestigationContext)
        or not isinstance(observed_reads, list)
        or len(observed_reads) > MAX_TOOL_CALLS
    ):
        raise EvidenceGroundingError("Invalid grounding inputs.")

    try:
        checked = InvestigationReport.model_validate(report.model_dump())
        repository = PublicRepository.model_validate(context.repository.model_dump())
        snapshot = RepositorySnapshot.model_validate(context.snapshot.model_dump())
        tree = RepositoryTree.model_validate(context.tree.model_dump())
        reads = [_ObservedRead.model_validate(item) for item in observed_reads]
    except (ValidationError, AttributeError, TypeError):
        raise EvidenceGroundingError("Invalid report or observed source data.") from None

    repository_name = snapshot.repository.full_name
    if (
        repository_name.casefold() != repository.full_name.casefold()
        or tree.repository_full_name.casefold() != repository_name.casefold()
        or tree.commit_sha != snapshot.commit_sha
        or len({entry.path for entry in tree.entries}) != len(tree.entries)
    ):
        raise EvidenceGroundingError("Repository context is inconsistent.")

    entries = {entry.path: entry for entry in tree.entries}
    for observed in reads:
        entry = entries.get(observed.path)
        if (
            observed.repository_full_name.casefold() != repository_name.casefold()
            or observed.commit_sha != snapshot.commit_sha
            or entry is None or entry.type != "blob"
            or entry.mode not in {"100644", "100755"}
            or entry.sha != observed.blob_sha
            or entry.size is None or entry.size > MAX_FILE_BYTES
        ):
            raise EvidenceGroundingError("Observed read does not match this repository snapshot.")
        try:
            _check_file_policy(observed.path)
        except GitHubClientError:
            raise EvidenceGroundingError("Observed file is outside the read policy.") from None

    for citation in checked.evidence:
        if not any(_matches(citation, observed) for observed in reads):
            raise EvidenceGroundingError(
                "A citation does not match complete source lines observed during this run."
            )

    return checked
