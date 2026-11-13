"""Offline draft-contract checks; no proposal is generated, applied or executed."""

from copy import deepcopy

import pytest
from agents import AgentOutputSchema
from pydantic import ValidationError

from app.fix_proposal import (
    MAX_EDIT_CHARACTERS, MAX_PROPOSAL_BYTES, MAX_TEST_CHARACTERS,
    FixProposal, FixProposalInput, ProposedEdit, ProposedTestFile,
)


def proposal_data():
    return {
        "status": "proposed",
        "title": "Accept the smallest positive quantity",
        "summary": "Correct the lower boundary and propose regression coverage.",
        "edits": [{
            "path": "orders_api.py",
            "before": "    if order.quantity <= 1:\n",
            "after": "    if order.quantity <= 0:\n",
            "reason": "A quantity of one satisfies the positive-integer requirement.",
        }],
        "tests": [{
            "path": "tests/test_orders_regression.py",
            "content": "def test_positive_boundary():\n    assert 1 > 0\n",
            "purpose": "Placeholder contract fixture; not an implemented HTTP regression test.",
        }],
        "verification_steps": ["Run the proposed regression tests in a later isolated validation stage."],
        "risks": [],
        "limitations": ["Source matching and test execution have not been performed."],
    }


def test_strict_sdk_roundtrip_preserves_exact_source_and_requires_all_fields():
    data = proposal_data()
    data["title"] = "  Accept the smallest positive quantity  "
    data["edits"][0]["before"] = "\tif quantity <= 1:  \r\n"
    data["edits"][0]["after"] = "\tif quantity <= 0:  \r\n"
    proposal = FixProposal.model_validate(data)
    sdk = AgentOutputSchema(FixProposal)

    assert proposal.title == data["title"].strip()
    assert proposal.edits[0].before == data["edits"][0]["before"]
    assert proposal.edits[0].after == data["edits"][0]["after"]
    assert sdk.is_strict_json_schema() is True
    schema = sdk.json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(FixProposal.model_fields)
    assert sdk.validate_json(proposal.model_dump_json()) == proposal
    for definition in schema["$defs"].values():
        assert definition["additionalProperties"] is False
        assert set(definition["required"]) == set(definition["properties"])


def test_request_accepts_only_saved_run_id_and_not_caller_owned_provenance():
    run_id = "a" * 32
    request = FixProposalInput(investigation_run_id=run_id)
    assert request.investigation_run_id == run_id
    assert AgentOutputSchema(FixProposalInput).validate_json(request.model_dump_json()) == request
    for extra in ("account_id", "repository_full_name", "commit_sha", "approved"):
        with pytest.raises(ValidationError):
            FixProposalInput.model_validate({"investigation_run_id": run_id, extra: "caller-value"})


@pytest.mark.parametrize("run_id", ["", "a" * 31, "a" * 33, "A" * 32, "g" * 32, 1, True])
def test_invalid_run_identifiers_are_rejected(run_id):
    with pytest.raises(ValidationError):
        FixProposalInput(investigation_run_id=run_id)


@pytest.mark.parametrize("path", [
    "/tmp/file.py", "../file.py", "app/../file.py", "app//file.py",
    "./file.py", "app\\file.py", "C:/file.py", "app/file.py:stream",
    "app/file.py\n", "app/file?.py", "app/file.py ", "app/dir./file.py",
    "app/CON.py", "app/lpt1/test.py", ".git/config.py", "secrets/token.py",
    ".env", "config/.env.local", "credentials.json", "image.png", "file\ud800.py",
])
def test_invalid_or_blocked_paths_are_rejected_for_edits_and_test_files(path):
    data = proposal_data()
    for model, values in ((ProposedEdit, data["edits"][0]), (ProposedTestFile, data["tests"][0])):
        with pytest.raises(ValidationError):
            model.model_validate({**values, "path": path})


@pytest.mark.parametrize("before,after", [
    ("", "replacement"), ("  \n", "replacement"), ("same", "same"),
    ("source", "  \t"), ("source\x00", "replacement"),
    ("source", "replacement\x1b"), ("source", "\ud800"),
    ("a" * (MAX_EDIT_CHARACTERS + 1), "replacement"),
    ("source", "b" * (MAX_EDIT_CHARACTERS + 1)),
])
def test_invalid_source_replacements_are_rejected(before, after):
    with pytest.raises(ValidationError):
        ProposedEdit.model_validate({**proposal_data()["edits"][0], "before": before, "after": after})


def test_exact_deletion_is_allowed_without_stripping_whitespace():
    edit = ProposedEdit.model_validate({**proposal_data()["edits"][0], "after": ""})
    assert edit.before == "    if order.quantity <= 1:\n"
    assert edit.after == ""


@pytest.mark.parametrize("content", ["", "  \n", "x\x00", "\ud800", "a" * (MAX_TEST_CHARACTERS + 1)])
def test_invalid_test_content_is_rejected(content):
    with pytest.raises(ValidationError):
        ProposedTestFile.model_validate({**proposal_data()["tests"][0], "content": content})


@pytest.mark.parametrize("change", [
    {"status": "applied"}, {"status": "verified"}, {"approved": True},
    {"diff": "an unchecked model-authored patch"}, {"tests_passed": True},
    {"commit_sha": "a" * 40}, {"edits": []}, {"tests": []},
    {"verification_steps": []}, {"limitations": []}, {"title": "  "},
    {"summary": 5}, {"risks": "none"}, {"title": "\ud800"},
])
def test_draft_cannot_claim_other_workflow_states_or_skip_required_content(change):
    with pytest.raises(ValidationError):
        FixProposal.model_validate({**proposal_data(), **change})


@pytest.mark.parametrize("collision", ["duplicate_edit", "case_edit", "test_collision", "case_test", "ancestor"])
def test_conflicting_target_paths_are_rejected(collision):
    data = proposal_data()
    if collision in {"duplicate_edit", "case_edit"}:
        second = deepcopy(data["edits"][0])
        if collision == "case_edit":
            second["path"] = "Orders_API.py"
        data["edits"].append(second)
    elif collision == "test_collision":
        data["tests"][0]["path"] = data["edits"][0]["path"]
    elif collision == "case_test":
        second = deepcopy(data["tests"][0])
        second["path"] = second["path"].upper()
        data["tests"].append(second)
    else:
        data["tests"][0]["path"] = "orders_api.py/test_orders.py"
    with pytest.raises(ValidationError):
        FixProposal.model_validate(data)


@pytest.mark.parametrize("field,count", [("edits", 6), ("tests", 4), ("verification_steps", 6), ("risks", 6), ("limitations", 6)])
def test_per_draft_list_budgets_are_enforced(field, count):
    data = proposal_data()
    if field in {"edits", "tests"}:
        original = data[field][0]
        data[field] = [{**original, "path": f"tests/file_{i}.py"} for i in range(count)]
    else:
        data[field] = ["Description."] * count
    with pytest.raises(ValidationError):
        FixProposal.model_validate(data)


def test_nested_extra_fields_and_wrong_types_are_not_silently_coerced():
    for field, extra in (("edits", {"before": 1}), ("tests", {"content": True}), ("edits", {"applied": True})):
        data = proposal_data()
        data[field][0].update(extra)
        with pytest.raises(ValidationError):
            FixProposal.model_validate(data)


def test_serialized_utf8_budget_counts_multibyte_source_text():
    data = proposal_data()
    data["edits"] = [{
        **data["edits"][0], "path": f"app/file_{i}.py",
        "before": "字" * 10000, "after": "新" * 10000,
    } for i in range(2)]
    assert sum(len(x["before"]) + len(x["after"]) for x in data["edits"]) < MAX_PROPOSAL_BYTES
    with pytest.raises(ValidationError, match="serialized size budget"):
        FixProposal.model_validate(data)


def test_syntactically_valid_draft_is_not_a_claim_of_source_matching_or_executability():
    data = proposal_data()
    data["edits"][0]["before"] = "an anchor not checked against any repository"
    data["tests"][0]["content"] = "syntactically invalid code is still draft text ("
    proposal = FixProposal.model_validate(data)
    assert proposal.status == "proposed"
    assert proposal.limitations
