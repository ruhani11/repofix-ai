"""Durable atomic validation admission; never Docker or target execution.

The future authenticated runner supplies a freshly prepared workspace and an
operator-inspected immutable image ID. It must call start exactly once before
Docker, and close only after all containers/processes are confirmed removed.
Reservations survive restart and never silently expire or resume work.
"""

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Literal
from uuid import uuid4
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from app.fix_proposal import _repository_path
from app.proposal_history import Digest, Identifier
from app.proposal_review import ProposalReviewError
from app.proposal_store import ProposalStore, ProposalStoreError
from app.proposal_validation import _encode_source
from app.sandbox_profile import PROFILE_ID
from app.validation_request import ValidationBinding, ValidationInput, ValidationRequestError, _validated, bind_validation_request
from app.github_oauth import GitHubIdentity
from app.validation_workspace import (
    MAX_WORKSPACE_BYTES, MAX_WORKSPACE_FILES, MAX_WORKSPACE_PAYLOAD_BYTES,
    MAX_WORKSPACE_SOURCE_FILES, WORKSPACE_PROTOCOL_VERSION, ValidationWorkspace,
)


MAX_ADMISSION_RECORD_BYTES = 16 * 1024
_IDENTIFIER = TypeAdapter(Identifier)
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ValidationAdmissionError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__("The validation job reservation could not be completed.")


class ValidationAdmission(BaseModel):
    """A reservation/lifecycle record; closed does not mean tests passed."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, hide_input_in_errors=True)
    validation_id: Identifier
    binding: ValidationBinding
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    baseline_sha256: Digest
    patched_sha256: Digest
    state: Literal["admitted", "running", "closed"]
    created_at: datetime
    started_at: datetime | None
    closed_at: datetime | None

    @field_validator("created_at", "started_at", "closed_at")
    @classmethod
    def utc_times(cls, value):
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("An aware lifecycle time is required.")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def consistent_state(self):
        if self.created_at < self.binding.bound_at:
            raise ValueError("Admission cannot predate preparation.")
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("Start cannot predate admission.")
        if self.closed_at is not None and self.closed_at < (self.started_at or self.created_at):
            raise ValueError("Close cannot predate active work.")
        if (
            (self.state == "admitted" and (self.started_at is not None or self.closed_at is not None))
            or (self.state == "running" and (self.started_at is None or self.closed_at is not None))
            or (self.state == "closed" and self.closed_at is None)
        ):
            raise ValueError("Inconsistent validation lifecycle.")
        return self


def _clock():
    return datetime.now(timezone.utc)


def _request(binding):
    return ValidationInput(
        proposal_id=binding.proposal_id, base_commit_sha=binding.base_commit_sha,
        patch_sha256=binding.patch_sha256, expected_revision=binding.review_revision,
        decision_id=binding.decision_id,
    )


def _workspace_copy(workspace):
    """Consistency only; trusted pinned source preparation is still required."""
    try:
        if not isinstance(workspace, ValidationWorkspace):
            raise ValueError()
        binding = _validated(workspace.binding, ValidationBinding, "invalid_workspace")
        snapshots = []
        for raw, phase, expected in [
            (workspace.baseline_payload, "baseline", workspace.baseline_sha256),
            (workspace.patched_payload, "patched", workspace.patched_sha256),
        ]:
            if type(raw) is not bytes or len(raw) > MAX_WORKSPACE_PAYLOAD_BYTES or hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError()
            data = json.loads(raw)
            if (
                set(data) != {"protocol_version", "profile", "phase", "binding", "tests", "files"}
                or type(data["protocol_version"]) is not int or data["protocol_version"] != WORKSPACE_PROTOCOL_VERSION
                or data["profile"] != PROFILE_ID or data["phase"] != phase
                or data["binding"] != binding.model_dump(mode="json")
                or type(data["tests"]) is not list or tuple(data["tests"]) != workspace.test_paths
                or type(data["files"]) is not list or not 1 <= len(data["files"]) <= MAX_WORKSPACE_FILES
            ):
                raise ValueError()
            if json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() != raw:
                raise ValueError()  # includes duplicate/noncanonical JSON fields
            files = {}
            for item in data["files"]:
                if type(item) is not dict or set(item) != {"path", "content", "sha256"}:
                    raise ValueError()
                if type(item["path"]) is not str or not 1 <= len(item["path"]) <= 1024:
                    raise ValueError()
                path = _repository_path(item["path"])
                if not path.endswith(".py") or path.casefold() in {name.casefold() for name in files}:
                    raise ValueError()
                content = _encode_source(item["content"])
                if hashlib.sha256(content).hexdigest() != item["sha256"]:
                    raise ValueError()
                files[path] = item["content"]
            if list(files) != sorted(files) or sum(len(value.encode()) for value in files.values()) > MAX_WORKSPACE_BYTES:
                raise ValueError()
            prefixes = {}
            for path in files:
                for parent in [PurePosixPath(path), *PurePosixPath(path).parents]:
                    name = parent.as_posix()
                    if name == ".":
                        continue
                    folded = name.casefold()
                    if folded in prefixes and prefixes[folded] != name:
                        raise ValueError()
                    prefixes[folded] = name
                if any(parent.as_posix() in files for parent in PurePosixPath(path).parents):
                    raise ValueError()
            snapshots.append(files)
        before, after = snapshots
        if set(before) != set(after) or type(workspace.test_paths) is not tuple:
            raise ValueError()
        counts = {
            "source_file_count": len(before) - len(workspace.test_paths),
            "source_bytes": sum(len(value.encode()) for path, value in before.items() if path not in workspace.test_paths),
            "baseline_bytes": sum(len(value.encode()) for value in before.values()),
            "patched_bytes": sum(len(value.encode()) for value in after.values()),
        }
        if not 1 <= counts["source_file_count"] <= MAX_WORKSPACE_SOURCE_FILES or any(
            type(getattr(workspace, name)) is not int or getattr(workspace, name) != count for name, count in counts.items()
        ):
            raise ValueError()
        return binding, before, after
    except Exception:
        raise ValidationAdmissionError("invalid_workspace") from None


class ValidationAdmissionStore:
    """Shares the proposal database so review/admission serialize together.

    Internal service only. Browser fields never choose workspace bytes, image,
    account, lifecycle state or cleanup. authorize is a required no-I/O session
    callback at admit/start, after the SQLite write lock has been acquired.
    """

    def __init__(self, proposal_store: ProposalStore):
        if not isinstance(proposal_store, ProposalStore):
            raise ValidationAdmissionError("invalid_store")
        self._store = proposal_store

    def _transaction(self, callback):
        try:
            return self._store._validation_transaction(callback)
        except ProposalStoreError:
            raise ValidationAdmissionError("unavailable") from None
        except ProposalReviewError as error:
            raise ValidationAdmissionError("not_found" if error.code == "not_found" else "unavailable") from None
        except ValidationRequestError as error:
            raise ValidationAdmissionError(error.code) from None

    @staticmethod
    def _key(account_id, validation_id):
        try:
            if type(account_id) is not int or not 0 < account_id <= 2**63 - 1:
                raise ValueError()
            # Validate exact IDs without inventing a separate regex policy.
            _IDENTIFIER.validate_python(validation_id, strict=True)
        except Exception:
            raise ValidationAdmissionError("invalid_input") from None

    @staticmethod
    def _encoded(record):
        encoded = record.model_dump_json(warnings="error")
        if len(encoded.encode()) > MAX_ADMISSION_RECORD_BYTES:
            raise ValidationAdmissionError("invalid_input")
        return encoded

    def _load(self, connection, account_id, validation_id):
        row = connection.execute(
            "SELECT * FROM fix_validation_admissions WHERE account_id=? AND validation_id=?",
            (account_id, validation_id),
        ).fetchone()
        if row is None:
            raise ValidationAdmissionError("not_found")
        if len(row["detail_json"].encode()) > MAX_ADMISSION_RECORD_BYTES:
            raise ValueError()
        result = ValidationAdmission.model_validate_json(row["detail_json"])
        if (
            result.validation_id != row["validation_id"] or result.binding.account_id != account_id
            or result.binding.proposal_id != row["proposal_id"] or result.state != row["state"]
        ):
            raise ValueError()
        proposal = self._store._owned_proposal(connection, account_id, row["proposal_id"])
        decision_row = connection.execute(
            "SELECT * FROM fix_proposal_decisions WHERE proposal_id=? AND revision=?",
            (proposal.proposal_id, result.binding.review_revision),
        ).fetchone()
        if decision_row is None:
            raise ValueError()
        decision = self._store._decision_entry(decision_row, account_id, proposal).decision
        if (
            decision.decision != "approve_for_validation" or decision.decision_id != result.binding.decision_id
            or decision.created_at > result.binding.bound_at
            or any(getattr(decision, name) != getattr(result.binding, name) for name in (
                "account_id", "proposal_id", "investigation_run_id", "repository_id", "repository_full_name",
                "base_commit_sha", "tree_sha", "patch_sha256",
            ))
        ):
            raise ValueError()
        return result

    def _current(self, connection, identity, payload):
        proposal = self._store._owned_proposal(connection, identity.id, payload.proposal_id)
        review = self._store._review_state(connection, identity.id, proposal)
        binding = bind_validation_request(identity, proposal, review, payload)
        return proposal, binding

    def admit(self, identity, payload, workspace, image_id, *, authorize):
        try:
            identity = _validated(identity, GitHubIdentity, "invalid_identity")
            payload = _validated(payload, ValidationInput, "invalid_input")
        except ValidationRequestError as error:
            raise ValidationAdmissionError(error.code) from None
        self._key(identity.id, payload.proposal_id)
        if type(image_id) is not str or _IMAGE_ID.fullmatch(image_id) is None:
            raise ValidationAdmissionError("invalid_image")
        if not callable(authorize):
            raise ValidationAdmissionError("invalid_input")
        binding, before, after = _workspace_copy(workspace)

        def reserve(connection):
            proposal, current = self._current(connection, identity, payload)
            if current.model_dump(exclude={"bound_at"}) != binding.model_dump(exclude={"bound_at"}):
                raise ValidationAdmissionError("workspace_changed")
            tests = {test.path: test.content for test in proposal.draft.tests}
            changes = {change.path: change for change in proposal.changes}
            if tuple(sorted(tests)) != workspace.test_paths:
                raise ValidationAdmissionError("workspace_changed")
            for path in before:
                change = changes.get(path)
                if change is not None:
                    original = change.original_content if change.action == "modify" else change.proposed_content
                    if before[path] != original or after[path] != change.proposed_content:
                        raise ValidationAdmissionError("workspace_changed")
                elif before[path] != after[path]:
                    raise ValidationAdmissionError("workspace_changed")
            if not set(changes) <= set(before):
                raise ValidationAdmissionError("workspace_changed")
            if connection.execute("SELECT 1 FROM fix_validation_admissions WHERE account_id=? AND state!='closed'", (identity.id,)).fetchone():
                raise ValidationAdmissionError("busy")
            record = ValidationAdmission(
                validation_id=uuid4().hex, binding=binding, image_id=image_id,
                baseline_sha256=workspace.baseline_sha256, patched_sha256=workspace.patched_sha256,
                state="admitted", created_at=_clock(), started_at=None, closed_at=None,
            )
            encoded = self._encoded(record)
            authorize()
            connection.execute("INSERT INTO fix_validation_admissions VALUES (?, ?, ?, ?, ?)", (
                record.validation_id, identity.id, proposal.proposal_id, record.state, encoded,
            ))
            return record
        return self._transaction(reserve)

    def get(self, account_id, validation_id):
        self._key(account_id, validation_id)
        return self._transaction(lambda connection: self._load(connection, account_id, validation_id))

    def start(self, identity, validation_id, *, authorize):
        try:
            identity = _validated(identity, GitHubIdentity, "invalid_identity")
        except ValidationRequestError as error:
            raise ValidationAdmissionError(error.code) from None
        self._key(identity.id, validation_id)
        if not callable(authorize):
            raise ValidationAdmissionError("invalid_input")
        def begin(connection):
            record = self._load(connection, identity.id, validation_id)
            if record.state != "admitted":
                raise ValidationAdmissionError("already_started")
            _, current = self._current(connection, identity, _request(record.binding))
            if current.model_dump(exclude={"bound_at"}) != record.binding.model_dump(exclude={"bound_at"}):
                raise ValidationAdmissionError("workspace_changed")
            authorize()
            result = ValidationAdmission.model_validate({**record.model_dump(), "state": "running", "started_at": _clock()})
            connection.execute("UPDATE fix_validation_admissions SET state=?, detail_json=? WHERE validation_id=?", (result.state, self._encoded(result), validation_id))
            return result
        return self._transaction(begin)

    def close(self, account_id, validation_id):
        """Internal cleanup acknowledgment; caller must confirm all work is gone.

        No expiry, automatic recovery or user HTTP endpoint releases this slot.
        Closing does not publish a result or claim tests passed. Session expiry
        must not prevent a worker from completing verified cleanup.
        """
        return self._release(account_id, validation_id, admitted_only=False)

    def cancel_unstarted(self, account_id, validation_id):
        """Release a reservation only if start has not committed."""
        return self._release(account_id, validation_id, admitted_only=True)

    def _release(self, account_id, validation_id, *, admitted_only):
        self._key(account_id, validation_id)
        def release(connection):
            record = self._load(connection, account_id, validation_id)
            if admitted_only and record.started_at is not None:
                raise ValidationAdmissionError("already_started")
            if record.state == "closed":
                return record
            return self._mark_closed(connection, record)
        return self._transaction(release)

    def _mark_closed(self, connection, record):
        """Used only inside a write transaction after cleanup or canceled start."""
        result = ValidationAdmission.model_validate({**record.model_dump(), "state": "closed", "closed_at": _clock()})
        connection.execute(
            "UPDATE fix_validation_admissions SET state=?, detail_json=? WHERE validation_id=?",
            (result.state, self._encoded(result), record.validation_id),
        )
        return result
