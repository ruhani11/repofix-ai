"""Durable completed comparisons scoped to the proposal's GitHub account."""

from dataclasses import asdict
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from app.proposal_store import ProposalStore
from app.sandbox_profile import RUNTIME_REVISION
from app.validation_admission import ValidationAdmission, ValidationAdmissionError, ValidationAdmissionStore
from app.validation_result import ValidationPhaseResult, Verdict, compare_validation_phases, parse_validation_phase


MAX_RESULT_RECORD_BYTES = 16 * 1024
MAX_RESULTS_PER_PROPOSAL = 20


class ValidationRunResult(BaseModel):
    """Saved observations, with no test source, console output or publish permission."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, hide_input_in_errors=True)
    profile: Literal["python-pytest-v1"] = "python-pytest-v1"
    runtime_revision: Literal[2, 3] = RUNTIME_REVISION
    admission: ValidationAdmission
    baseline: ValidationPhaseResult
    patched: ValidationPhaseResult
    verdict: Verdict

    @model_validator(mode="after")
    def consistent_result(self):
        if self.admission.state != "closed" or self.admission.started_at is None:
            raise ValueError("A completed validation requires a started and closed admission.")
        for phase in ("baseline", "patched"):
            details = asdict(getattr(self, phase))
            report = {
                "status": "completed", "profile": self.profile, "runtime_revision": self.runtime_revision,
                "protocol_version": 1, "phase": details.pop("phase"), "payload_sha256": details.pop("payload_sha256"),
                "worker_output_bytes": details.pop("worker_output_bytes"), "result": details,
            }
            parse_validation_phase(
                json.dumps(report).encode(), phase=phase,
                payload_sha256=getattr(self.admission, f"{phase}_sha256"),
                expected_revision=self.runtime_revision,
            )
        if compare_validation_phases(self.baseline, self.patched) != self.verdict:
            raise ValueError("Saved verdict does not match the two phases.")
        return self


class ValidationResultPage(BaseModel):
    results: list[ValidationRunResult]
    has_more: bool
    next_page: int | None


class ValidationResultStore:
    def __init__(self, store: ProposalStore):
        self._store = store
        self._admissions = ValidationAdmissionStore(store)

    def complete(self, account_id, validation_id, baseline, patched):
        """Close and save together. Call only after both containers are gone."""
        self._admissions._key(account_id, validation_id)

        def save(connection):
            record = self._admissions._load(connection, account_id, validation_id)
            if record.state != "running":
                raise ValidationAdmissionError("not_running")
            closed = self._admissions._mark_closed(connection, record)
            result = ValidationRunResult(
                admission=closed, baseline=baseline, patched=patched,
                verdict=compare_validation_phases(baseline, patched),
            )
            encoded = result.model_dump_json(warnings="error")
            if len(encoded.encode()) > MAX_RESULT_RECORD_BYTES:
                raise ValueError("Validation result is too large.")
            connection.execute(
                "INSERT INTO fix_validation_results VALUES (?, ?, ?, ?, ?)",
                (validation_id, account_id, closed.binding.proposal_id, closed.closed_at.isoformat(), encoded),
            )
            return result

        return self._admissions._transaction(save)

    def _decode(self, connection, row, account_id):
        encoded = row["detail_json"]
        if not isinstance(encoded, str) or len(encoded.encode()) > MAX_RESULT_RECORD_BYTES:
            raise ValueError("Invalid saved validation result.")
        result = ValidationRunResult.model_validate_json(encoded)
        admission = self._admissions._load(connection, account_id, row["validation_id"])
        if (
            result.admission != admission or admission.binding.proposal_id != row["proposal_id"]
            or admission.closed_at is None or admission.closed_at.isoformat() != row["completed_at"]
        ):
            raise ValueError("Invalid saved validation provenance.")
        return result

    def get(self, account_id, validation_id):
        self._admissions._key(account_id, validation_id)

        def read(connection):
            row = connection.execute(
                "SELECT * FROM fix_validation_results WHERE account_id=? AND validation_id=?",
                (account_id, validation_id),
            ).fetchone()
            if row is None:
                raise ValidationAdmissionError("not_found")
            return self._decode(connection, row, account_id)

        return self._admissions._transaction(read)

    def list(self, account_id, proposal_id, *, page=1, per_page=10):
        self._admissions._key(account_id, proposal_id)
        if type(page) is not int or not 1 <= page <= 10000 or type(per_page) is not int or not 1 <= per_page <= MAX_RESULTS_PER_PROPOSAL:
            raise ValidationAdmissionError("invalid_input")

        def read(connection):
            self._store._owned_proposal(connection, account_id, proposal_id)
            rows = connection.execute(
                "SELECT * FROM fix_validation_results WHERE account_id=? AND proposal_id=? "
                "ORDER BY completed_at DESC, validation_id DESC LIMIT ? OFFSET ?",
                (account_id, proposal_id, per_page + 1, (page - 1) * per_page),
            ).fetchall()
            has_more = len(rows) > per_page and page < 10000
            return ValidationResultPage(
                results=[self._decode(connection, row, account_id) for row in rows[:per_page]],
                has_more=has_more, next_page=page + 1 if has_more else None,
            )

        return self._admissions._transaction(read)
