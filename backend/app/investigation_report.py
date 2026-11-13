"""Structured investigation output; source grounding is validated separately."""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


ReportText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=2000,
    ),
]
SourceQuote = Annotated[
    str,
    StringConstraints(
        strip_whitespace=False,
        min_length=1,
        max_length=2000,
    ),
]


class EvidenceReference(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        hide_input_in_errors=True,
    )

    path: str = Field(min_length=1, max_length=1024)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    quote: SourceQuote
    explanation: ReportText

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if (
            value.startswith("/")
            or "\\" in value
            or ":" in value.split("/")[0]
            or any(
                part in {"", ".", ".."}
                for part in value.split("/")
            )
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in value
            )
        ):
            raise ValueError(
                "Evidence must use a relative repository file path."
            )

        return value

    @field_validator("quote")
    @classmethod
    def validate_nonblank_quote(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "Evidence quote must contain source text."
            )

        return value

    @model_validator(mode="after")
    def validate_line_range(self) -> "EvidenceReference":
        if self.end_line < self.start_line:
            raise ValueError(
                "Evidence end_line must be at least start_line."
            )

        return self


class InvestigationReport(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        hide_input_in_errors=True,
    )

    status: Literal["likely_cause", "insufficient_evidence"]
    issue_summary: ReportText
    issue_category: Literal[
        "authentication",
        "validation",
        "api",
        "logic",
        "configuration",
        "other",
    ]
    root_cause: ReportText | None
    evidence: list[EvidenceReference] = Field(max_length=10)
    next_steps: list[ReportText] = Field(min_length=1, max_length=5)
    limitations: list[ReportText] = Field(max_length=5)

    @model_validator(mode="after")
    def validate_report_consistency(self) -> "InvestigationReport":
        if self.status == "likely_cause":
            if self.root_cause is None or not self.evidence:
                raise ValueError(
                    "A likely cause requires a hypothesis "
                    "and source evidence."
                )
        else:
            if self.root_cause is not None:
                raise ValueError(
                    "Use a null root_cause when evidence is insufficient."
                )

            if not self.limitations:
                raise ValueError(
                    "Explain why the available evidence is insufficient."
                )

        return self