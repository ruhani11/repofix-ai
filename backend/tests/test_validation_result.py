import json

import pytest

from app.sandbox_profile import PROFILE_ID, RUNTIME_REVISION
from app.validation_result import (
    MAX_REPORT_BYTES,
    ValidationResultError,
    compare_validation_phases,
    parse_validation_phase,
)


def report(phase, **metrics):
    return {
        "status": "completed", "profile": PROFILE_ID,
        "runtime_revision": RUNTIME_REVISION, "protocol_version": 1,
        "phase": phase, "payload_sha256": "a" * 64 if phase == "baseline" else "b" * 64,
        "worker_output_bytes": 223,
        "result": {
            "collected": 2, "passed": int(phase == "baseline") + int(phase == "patched") * 2,
            "failed": int(phase == "baseline"), "exit_code": int(phase == "baseline"),
            "skipped": 0, "errors": 0, "nodeids_sha256": "c" * 64, **metrics,
        },
    }


def parse(record):
    return parse_validation_phase(
        json.dumps(record).encode(), phase=record["phase"], payload_sha256=record["payload_sha256"],
    )


@pytest.mark.parametrize(("before", "after", "expected"), [
    ({}, {}, "verified"),
    ({"passed": 2, "failed": 0, "exit_code": 0}, {}, "baseline_not_reproduced"),
    ({}, {"passed": 1, "failed": 1, "exit_code": 1}, "patched_tests_failed"),
    ({}, {"nodeids_sha256": "d" * 64}, "different_tests"),
    ({}, {"collected": 3, "passed": 3}, "different_tests"),
    ({}, {"passed": 1, "skipped": 1}, "incomplete"),
    ({"passed": 1, "failed": 0, "errors": 1}, {}, "incomplete"),
    ({"collected": 0, "passed": 0, "failed": 0, "exit_code": 0}, {}, "incomplete"),
])
def test_compare_requires_a_complete_reproduced_failure_and_patched_pass(before, after, expected):
    assert compare_validation_phases(parse(report("baseline", **before)), parse(report("patched", **after))) == expected


@pytest.mark.parametrize(("field", "value"), [
    ("phase", "patched"), ("payload_sha256", "b" * 64), ("profile", "other"),
    ("runtime_revision", 1), ("protocol_version", True), ("worker_output_bytes", 65537),
    ("worker_output_bytes", True), ("logs", "must not be returned"),
])
def test_reject_wrong_or_unexpected_report_metadata(field, value):
    record = report("baseline")
    record[field] = value
    with pytest.raises(ValidationResultError, match="could not be accepted"):
        parse_validation_phase(json.dumps(record).encode(), phase="baseline", payload_sha256="a" * 64)


@pytest.mark.parametrize("metrics", [
    {"passed": True}, {"failed": -1}, {"collected": 101}, {"exit_code": 0},
    {"exit_code": 1.0}, {"collected": 3}, {"skipped": 1}, {"nodeids_sha256": "invalid"},
])
def test_reject_bad_worker_counters(metrics):
    with pytest.raises(ValidationResultError) as error:
        parse(report("baseline", **metrics))
    assert error.value.code == "invalid_report"


@pytest.mark.parametrize("raw", [
    b"", b" " * (MAX_REPORT_BYTES + 1), b"\xff", b"[null]", b"[" * 1500 + b"]" * 1500,
    b'{"status":"completed","status":"error","reason":"timeout"}',
    b'{"status":"error","reason":"private raw output"}',
])
def test_malformed_oversized_and_unrecognized_output_are_rejected(raw):
    with pytest.raises(ValidationResultError) as error:
        parse_validation_phase(raw, phase="baseline", payload_sha256="a" * 64)
    assert error.value.code == "invalid_report"


def test_known_runtime_error_is_never_a_completed_phase():
    with pytest.raises(ValidationResultError) as error:
        parse_validation_phase(b'{"status":"error","reason":"timeout"}', phase="baseline", payload_sha256="a" * 64)
    assert error.value.code == "timeout"


def test_comparison_rejects_swapped_phases():
    with pytest.raises(ValidationResultError) as error:
        compare_validation_phases(parse(report("patched")), parse(report("baseline")))
    assert error.value.code == "invalid_phase_pair"
