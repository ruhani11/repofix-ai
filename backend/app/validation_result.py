"""Validate bounded runtime reports and compare observations for the same tests."""

import json
import re
from dataclasses import dataclass
from typing import Literal

from app.sandbox_profile import PROFILE_ID, RUNTIME_REVISION


MAX_REPORT_BYTES = 16 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_COUNTS = ("collected", "passed", "failed", "skipped", "errors")
_METRICS = {*_COUNTS, "exit_code", "nodeids_sha256"}
_REPORT = {
    "status", "profile", "runtime_revision", "protocol_version", "phase",
    "payload_sha256", "result", "worker_output_bytes",
}
_ERRORS = {
    "runtime_not_ready", "payload_invalid", "output_limit", "worker_io_error",
    "timeout", "worker_exit_mismatch", "phase_failed",
}

Phase = Literal["baseline", "patched"]
Verdict = Literal[
    "verified", "baseline_not_reproduced", "patched_tests_failed",
    "different_tests", "incomplete",
]


class ValidationResultError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__("The Docker validation result could not be accepted.")


@dataclass(frozen=True)
class ValidationPhaseResult:
    phase: Phase
    payload_sha256: str
    collected: int
    passed: int
    failed: int
    skipped: int
    errors: int
    exit_code: int
    nodeids_sha256: str
    worker_output_bytes: int


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate report field")
        result[key] = value
    return result


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _digest(value):
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def parse_validation_phase(
    raw: bytes, *, phase: Phase, payload_sha256: str, expected_revision: int = RUNTIME_REVISION,
) -> ValidationPhaseResult:
    """Decode a bounded report for a phase and hash supplied by the host.

    The caller must bound the process stream before collecting it and confirm
    successful Docker exit and cleanup before accepting a completed report.
    This parser does not authorize execution or attest to arbitrary test code.
    """
    if phase not in ("baseline", "patched") or not _digest(payload_sha256):
        raise ValidationResultError("invalid_expectation")
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_REPORT_BYTES:
        raise ValidationResultError("invalid_report")
    try:
        record = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_fields)
    except (ValueError, RecursionError):
        raise ValidationResultError("invalid_report") from None
    if type(record) is not dict:
        raise ValidationResultError("invalid_report")
    reason = record.get("reason")
    if set(record) == {"status", "reason"} and record["status"] == "error":
        if type(reason) is str and reason in _ERRORS:
            raise ValidationResultError(reason)
        raise ValidationResultError("invalid_report")
    if (
        set(record) != _REPORT or record["status"] != "completed"
        or record["profile"] != PROFILE_ID
        or not _integer(record["runtime_revision"], expected_revision, expected_revision)
        or not _integer(record["protocol_version"], 1, 1)
        or record["phase"] != phase or record["payload_sha256"] != payload_sha256
        or not _integer(record["worker_output_bytes"], 0, 64 * 1024)
    ):
        raise ValidationResultError("invalid_report")

    result = record["result"]
    if (
        type(result) is not dict or set(result) != _METRICS
        or not all(_integer(result[key], 0, 1000) for key in _COUNTS)
        or result["collected"] > 100 or not _integer(result["exit_code"], 0, 1)
        or not _digest(result["nodeids_sha256"])
    ):
        raise ValidationResultError("invalid_report")
    counted = result["passed"] + result["failed"] + result["skipped"]
    has_failures = bool(result["failed"] or result["errors"])
    if (
        counted > result["collected"]
        or (not result["errors"] and counted != result["collected"])
        or result["exit_code"] != int(has_failures)
    ):
        raise ValidationResultError("invalid_report")
    return ValidationPhaseResult(
        phase=phase, payload_sha256=payload_sha256,
        worker_output_bytes=record["worker_output_bytes"], **result,
    )


def compare_validation_phases(
    baseline: ValidationPhaseResult, patched: ValidationPhaseResult,
) -> Verdict:
    """Verified means the same complete test set failed before and passed after."""
    if baseline.phase != "baseline" or patched.phase != "patched":
        raise ValidationResultError("invalid_phase_pair")
    if any(not r.collected or r.errors or r.skipped for r in (baseline, patched)):
        return "incomplete"
    if (
        baseline.collected != patched.collected
        or baseline.nodeids_sha256 != patched.nodeids_sha256
    ):
        return "different_tests"
    if not baseline.failed:
        return "baseline_not_reproduced"
    if patched.failed:
        return "patched_tests_failed"
    return "verified"
