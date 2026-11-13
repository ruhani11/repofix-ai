"""Offline output-contract tests; these do not verify real repository evidence."""

import pytest
from agents import AgentOutputSchema
from pydantic import ValidationError

from app.investigation_report import EvidenceReference, InvestigationReport


def evidence_data(**overrides):
    values = {
        "path": "app/auth.py",
        "start_line": 8,
        "end_line": 8,
        "quote": "    return user.id\n",
        "explanation": (
            "The code accesses a user attribute without checking for None."
        ),
    }
    values.update(overrides)
    return values


def report_data(**overrides):
    values = {
        "status": "likely_cause",
        "issue_summary": "Login crashes when no user is found.",
        "issue_category": "authentication",
        "root_cause": (
            "A missing user is accessed before checking for None."
        ),
        "evidence": [evidence_data()],
        "next_steps": [
            "Reproduce the missing-user case with a regression test."
        ],
        "limitations": [
            "The hypothesis has not been reproduced or tested."
        ],
    }
    values.update(overrides)
    return values


def test_likely_cause_supports_strict_sdk_output_and_preserves_quotes():
    report = InvestigationReport.model_validate(report_data())
    schema = AgentOutputSchema(InvestigationReport)

    assert report.evidence[0].quote == "    return user.id\n"
    assert schema.is_strict_json_schema() is True
    assert schema.json_schema()["additionalProperties"] is False
    assert set(schema.json_schema()["required"]) == set(
        InvestigationReport.model_fields
    )
    assert schema.validate_json(report.model_dump_json()) == report


def test_insufficient_evidence_keeps_cause_unknown():
    report = InvestigationReport.model_validate(
        report_data(
            status="insufficient_evidence",
            root_cause=None,
            evidence=[],
            limitations=[
                "Relevant source files have not been inspected."
            ],
        )
    )

    assert report.root_cause is None
    assert report.evidence == []
    assert AgentOutputSchema(InvestigationReport).validate_json(
        report.model_dump_json()
    ) == report


@pytest.mark.parametrize(
    "overrides",
    [
        {"root_cause": None},
        {"evidence": []},
        {"status": "confirmed"},
        {"issue_summary": "   "},
        {"issue_category": "invented-category"},
        {"confidence": 0.99},
        {"next_steps": []},
    ],
)
def test_invalid_or_unsupported_report_claims_are_rejected(overrides):
    with pytest.raises(ValidationError):
        InvestigationReport.model_validate(report_data(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"path": "/etc/passwd"},
        {"path": "../auth.py"},
        {"path": "https://example.com/auth.py"},
        {"path": "C:\\auth.py"},
        {"start_line": 0},
        {"start_line": True},
        {"start_line": "8"},
        {"end_line": 7},
        {"quote": "   "},
        {"quote": "x" * 2001},
    ],
)
def test_invalid_evidence_shape_is_rejected(overrides):
    with pytest.raises(ValidationError):
        EvidenceReference.model_validate(evidence_data(**overrides))


@pytest.mark.parametrize(
    "field,count",
    [
        ("evidence", 11),
        ("next_steps", 6),
        ("limitations", 6),
    ],
)
def test_oversized_report_lists_are_rejected(field, count):
    values = report_data()
    item = (
        evidence_data()
        if field == "evidence"
        else "Further inspection needed."
    )
    values[field] = [item] * count

    with pytest.raises(ValidationError):
        InvestigationReport.model_validate(values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"root_cause": "An asserted cause without enough evidence."},
        {"limitations": []},
    ],
)
def test_insufficient_evidence_cannot_assert_a_cause_or_hide_limits(
    overrides,
):
    values = report_data(
        status="insufficient_evidence",
        root_cause=None,
        evidence=[],
    )
    values.update(overrides)

    with pytest.raises(ValidationError):
        InvestigationReport.model_validate(values)


@pytest.mark.parametrize("step", ["", "   "])
def test_blank_next_steps_are_rejected(step):
    with pytest.raises(ValidationError):
        InvestigationReport.model_validate(
            report_data(next_steps=[step])
        )