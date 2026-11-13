"""Validated bug input for the connected-account investigation workflow."""

from pydantic import Field, field_validator

from app.github_selection import RepositorySelectionInput


class InvestigationInput(RepositorySelectionInput):
    """Reuse strict ID/name validation; owner comes from the login session.

    This model validates input only. Repository ownership must be checked
    again when starting an investigation. It grants no write approval.
    """

    issue_description: str = Field(min_length=1, max_length=8000)

    @field_validator("issue_description", mode="before")
    @classmethod
    def trim_issue_description(cls, value: object) -> object:
        """Trim surrounding whitespace while preserving logs and inner lines."""
        if isinstance(value, str):
            return value.strip()
        return value
