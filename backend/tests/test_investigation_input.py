"""Validate the connected-account bug input without network or model calls."""

import pytest
from pydantic import ValidationError

from app.investigation_input import InvestigationInput


VALID_INPUT = {
    "repository_id": 5,
    "repository_name": "example",
    "issue_description": "Login fails after submitting valid credentials.",
}


def test_valid_input_trims_outer_whitespace_and_preserves_logs():
    description = "Login fails.\nExpected: dashboard.\nActual: त्रुटि"

    result = InvestigationInput(
        **{**VALID_INPUT, "issue_description": f"  {description}\n  "}
    )

    assert result.repository_id == 5
    assert result.repository_name == "example"
    assert result.issue_description == description


@pytest.mark.parametrize("description", [
    "",
    " \t\n ",
    "x" * 8001,
    None,
    123,
    True,
    [],
])
def test_invalid_description_is_rejected(description):
    with pytest.raises(ValidationError) as error:
        InvestigationInput(**{**VALID_INPUT, "issue_description": description})

    assert any(
        item["loc"] == ("issue_description",)
        for item in error.value.errors()
    )


@pytest.mark.parametrize("field", [
    "repository_id",
    "repository_name",
    "issue_description",
])
def test_all_input_fields_are_required(field):
    payload = VALID_INPUT.copy()
    del payload[field]

    with pytest.raises(ValidationError) as error:
        InvestigationInput(**payload)

    assert any(
        item["loc"] == (field,) and item["type"] == "missing"
        for item in error.value.errors()
    )


@pytest.mark.parametrize("patch,field,error_type", [
    ({"repository_id": True}, "repository_id", "int_type"),
    ({"repository_id": "5"}, "repository_id", "int_type"),
    ({"repository_id": 0}, "repository_id", "greater_than"),
    ({"repository_name": "../other"}, "repository_name", "string_pattern_mismatch"),
    ({"repository_name": ".."}, "repository_name", "value_error"),
    ({"owner": "other-user"}, "owner", "extra_forbidden"),
    ({"repository_url": "https://example.com"}, "repository_url", "extra_forbidden"),
    ({"approved": True}, "approved", "extra_forbidden"),
])
def test_invalid_selection_and_extra_fields_are_rejected(patch, field, error_type):
    with pytest.raises(ValidationError) as error:
        InvestigationInput(**{**VALID_INPUT, **patch})

    assert any(
        item["loc"] == (field,) and item["type"] == error_type
        for item in error.value.errors()
    )


def test_exact_description_length_limit_is_accepted():
    result = InvestigationInput(
        **{**VALID_INPUT, "issue_description": "x" * 8000}
    )

    assert len(result.issue_description) == 8000


def test_validated_input_cannot_be_modified():
    result = InvestigationInput(**VALID_INPUT)

    with pytest.raises(ValidationError) as error:
        result.repository_id = 6

    assert error.value.errors()[0]["type"] == "frozen_instance"
    assert result.repository_id == 5
