"""Bounded fix/test drafts; source matching, diff generation and execution are separate."""

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    AfterValidator, BaseModel, ConfigDict, Field, StringConstraints,
    field_validator, model_validator,
)

from app.github_client import GitHubClientError
from app.github_files import _check_file_policy


MAX_EDITS = 5
MAX_TEST_FILES = 3
MAX_EDIT_CHARACTERS = 12000
MAX_TEST_CHARACTERS = 24000
MAX_PROPOSAL_BYTES = 96 * 1024

EditBefore = Annotated[str, StringConstraints(min_length=1, max_length=MAX_EDIT_CHARACTERS)]
EditAfter = Annotated[str, StringConstraints(max_length=MAX_EDIT_CHARACTERS)]
TestCode = Annotated[str, StringConstraints(min_length=1, max_length=MAX_TEST_CHARACTERS)]

_WINDOWS_RESERVED = frozenset({
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
})


def _repository_path(value: str) -> str:
    parts = value.split("/")
    if (
        any(part in {"", ".", ".."} for part in parts)
        or any(char in '\\:<>"|?*' or ord(char) < 32 or ord(char) == 127 for char in value)
        or any(part != part.strip() or part.endswith(".") for part in parts)
        or any(part.split(".")[0].casefold() in _WINDOWS_RESERVED for part in parts)
    ):
        raise ValueError("Use a portable relative repository file path.")
    _source_text(value)
    try:
        _check_file_policy(value)
    except GitHubClientError:
        raise ValueError("Proposal path is blocked or its file type is unsupported.") from None
    return value


def _source_text(value: str) -> str:
    """Keep indentation and line endings exactly; reject non-text/control payloads."""
    if any((ord(char) < 32 and char not in "\t\n\r") or ord(char) == 127 for char in value):
        raise ValueError("Source text contains unsupported control characters.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("Source text must be valid UTF-8.") from None
    return value


ProposalText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
    AfterValidator(_source_text),
]


class _ProposalModel(BaseModel):
    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, hide_input_in_errors=True,
    )


class FixProposalInput(_ProposalModel):
    """Future API input: load ownership, repository and base commit on the server."""

    investigation_run_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class ProposedEdit(_ProposalModel):
    """One exact source replacement in an existing file; never fuzzy matching."""

    path: str = Field(min_length=1, max_length=1024)
    before: EditBefore
    after: EditAfter
    reason: ProposalText

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _repository_path(value)

    @field_validator("before", "after")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _source_text(value)

    @model_validator(mode="after")
    def validate_change(self) -> "ProposedEdit":
        if not self.before.strip():
            raise ValueError("An edit needs a nonblank original source anchor.")
        if self.before == self.after:
            raise ValueError("An edit must change source text.")
        if self.after and not self.after.strip():
            raise ValueError("Use an empty replacement for a deletion.")
        return self


class ProposedTestFile(_ProposalModel):
    """Draft content for a new regression-test file; not an execution result."""

    path: str = Field(min_length=1, max_length=1024)
    content: TestCode
    purpose: ProposalText

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _repository_path(value)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A proposed test file needs nonblank source text.")
        return _source_text(value)


class FixProposal(_ProposalModel):
    """Agent output contract only; does not assert approval or a verified fix.

    The future service must bind this draft to the authenticated saved run and
    pinned commit, check exact unique anchors and tree/file policy, verify new
    test paths do not exist, and construct the review diff deterministically.
    A valid schema alone proves none of those source or semantic properties.
    """

    status: Literal["proposed"]
    title: ProposalText
    summary: ProposalText
    edits: list[ProposedEdit] = Field(min_length=1, max_length=MAX_EDITS)
    tests: list[ProposedTestFile] = Field(min_length=1, max_length=MAX_TEST_FILES)
    verification_steps: list[ProposalText] = Field(min_length=1, max_length=5)
    risks: list[ProposalText] = Field(max_length=5)
    limitations: list[ProposalText] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def validate_targets_and_size(self) -> "FixProposal":
        paths = [item.path.casefold() for item in [*self.edits, *self.tests]]
        # The first contract permits one contiguous replacement per target file.
        if len(set(paths)) != len(paths):
            raise ValueError("Proposal target paths must be distinct, including case.")
        for path in paths:
            if any(parent.as_posix() in paths for parent in PurePosixPath(path).parents if parent.as_posix() != "."):
                raise ValueError("Proposal targets cannot be file/directory ancestors.")
        if len(self.model_dump_json().encode("utf-8")) > MAX_PROPOSAL_BYTES:
            raise ValueError("Proposal exceeds the serialized size budget.")
        return self
