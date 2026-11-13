"""Run the exact approved Python workspace in two restricted Docker phases.

Internal service. The caller supplies the account's session check and existing
stores. Results are returned after container removal and an atomic commit of
the comparison and closed reservation. Interrupted-run recovery is separate.
"""

import hashlib
import re
import shutil
import subprocess
import time
from threading import Event, Thread
from typing import Callable

from app.github_oauth import GitHubIdentity
from app.investigation_store import InvestigationStore
from app.proposal_store import ProposalStore
from app.sandbox_profile import build_sandbox_phase_create_args, check_sandbox_runtime
from app.validation_admission import ValidationAdmission, ValidationAdmissionError, ValidationAdmissionStore
from app.validation_context import prepare_validation_context
from app.validation_history import ValidationResultStore, ValidationRunResult
from app.validation_request import ValidationInput
from app.validation_result import (
    MAX_REPORT_BYTES, Phase, ValidationPhaseResult, ValidationResultError, parse_validation_phase,
)


_RUN_LABEL = "ai.repofix.validation-run"
_CONTAINER_ID = re.compile(rb"[0-9a-f]{64}\Z")
_COMMAND_TIMEOUT = 15
_PHASE_TIMEOUT = 60
_STOP_GRACE = 2


class ValidationRunnerError(RuntimeError):
    def __init__(self, code: str, *, validation_id: str | None = None, cleanup_confirmed: bool = True):
        self.code = code
        self.validation_id = validation_id
        self.cleanup_confirmed = cleanup_confirmed
        super().__init__("The approved proposal could not finish Docker validation.")


def _command(
    docker: str, args: tuple[str, ...], *, timeout: float, stdin: bytes | None = None,
) -> tuple[int, bytes]:
    """Capture at most one bounded report and discard stderr without buffering it."""
    deadline = time.monotonic() + timeout
    try:
        process = subprocess.Popen(
            (docker, *args), shell=False, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    except OSError:
        raise ValidationRunnerError("docker_unavailable") from None

    output = bytearray()
    overflow = Event()
    io_failed = Event()

    def drain():
        try:
            with process.stdout as source:
                while chunk := source.read1(4096):
                    if len(output) + len(chunk) > MAX_REPORT_BYTES:
                        overflow.set()
                        return
                    output.extend(chunk)
        except (OSError, ValueError):
            io_failed.set()

    def supply():
        try:
            with process.stdin as sink:
                sink.write(stdin)
        except (OSError, ValueError):
            io_failed.set()

    threads = [Thread(target=drain, daemon=True)]
    if stdin is not None:
        threads.append(Thread(target=supply, daemon=True))
    started = []
    try:
        for thread in threads:
            thread.start()
            started.append(thread)
        while True:
            if overflow.is_set():
                raise ValidationRunnerError("output_limit")
            if io_failed.is_set():
                raise ValidationRunnerError("docker_io_error")
            if time.monotonic() >= deadline:
                raise ValidationRunnerError("timeout")
            code = process.poll()
            if code is not None and not any(thread.is_alive() for thread in started):
                # Readers may have set a failure flag since the first check.
                if overflow.is_set():
                    raise ValidationRunnerError("output_limit")
                if io_failed.is_set():
                    raise ValidationRunnerError("docker_io_error")
                return code, bytes(output)
            time.sleep(0.01)
    finally:
        until = time.monotonic() + _STOP_GRACE
        try:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=max(0, until - time.monotonic()))
            for thread in started:
                thread.join(timeout=max(0, until - time.monotonic()))
            if any(thread.is_alive() for thread in started):
                raise TimeoutError()
        except (OSError, subprocess.TimeoutExpired, TimeoutError):
            raise ValidationRunnerError("docker_cli_stop_unconfirmed", cleanup_confirmed=False) from None
        finally:
            # No thread owns an unopened pipe if thread startup itself failed.
            if not started:
                process.stdout.close()
            if stdin is not None and len(started) < 2:
                process.stdin.close()


def _recover_container_id(docker: str, name: str, admission: ValidationAdmission) -> str | None:
    # If create might have reached Docker, remove only a positively identified
    # container. A missing name alone cannot disprove a delayed create.
    fmt = '{{.Id}} {{.Image}} {{.Name}} {{index .Config.Labels "' + _RUN_LABEL + '"}}'
    code, output = _command(docker, ("container", "inspect", "--format", fmt, name), timeout=_COMMAND_TIMEOUT)
    fields = output.split()
    expected = [admission.image_id.encode(), f"/{name}".encode(), admission.validation_id.encode()]
    if code != 0 or len(fields) != 4 or fields[1:] != expected or not _CONTAINER_ID.fullmatch(fields[0]):
        return None
    return fields[0].decode("ascii")


def _phase(
    docker: str, admission: ValidationAdmission, phase: Phase, payload: bytes,
) -> ValidationPhaseResult:
    digest = admission.baseline_sha256 if phase == "baseline" else admission.patched_sha256
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValidationRunnerError("workspace_changed", validation_id=admission.validation_id)

    args = build_sandbox_phase_create_args(admission.image_id, admission.validation_id, phase)
    name = args[args.index("--name") + 1]
    args = (*args[:1], "--label", f"{_RUN_LABEL}={admission.validation_id}", *args[1:])
    container_id = None
    create_attempted = True
    try:
        try:
            code, output = _command(docker, args, timeout=_COMMAND_TIMEOUT)
        except ValidationRunnerError as error:
            if error.code == "docker_unavailable":
                create_attempted = False
            raise
        if code != 0 or not _CONTAINER_ID.fullmatch(output.strip()):
            raise ValidationRunnerError("docker_create_failed", validation_id=admission.validation_id)
        container_id = output.strip().decode("ascii")

        code, output = _command(
            docker, ("start", "--attach", "--interactive", container_id),
            timeout=_PHASE_TIMEOUT, stdin=payload + b"\n",
        )
        try:
            report = parse_validation_phase(output, phase=phase, payload_sha256=digest)
        except ValidationResultError as error:
            reason = "docker_phase_failed" if code and error.code == "invalid_report" else error.code
            raise ValidationRunnerError(reason, validation_id=admission.validation_id) from None
        if code != 0:
            raise ValidationRunnerError("docker_phase_failed", validation_id=admission.validation_id)
        return report
    finally:
        if create_attempted:
            try:
                owned = container_id or _recover_container_id(docker, name, admission)
                if owned is None:
                    raise ValidationRunnerError("container_identity_unconfirmed")
                code, _ = _command(docker, ("rm", "--force", owned), timeout=_COMMAND_TIMEOUT)
                if code != 0:
                    raise ValidationRunnerError("docker_remove_failed")
            except BaseException:
                raise ValidationRunnerError(
                    "cleanup_unconfirmed", validation_id=admission.validation_id, cleanup_confirmed=False,
                ) from None


def run_approved_validation(
    identity: GitHubIdentity, request: ValidationInput, proposal_store: ProposalStore,
    history_store: InvestigationStore, *, authorize: Callable[[], None],
) -> ValidationRunResult:
    """Prepare, atomically admit/start, execute and acknowledge verified cleanup."""
    if not callable(authorize):
        raise ValidationRunnerError("invalid_authorizer")
    authorize()
    context = prepare_validation_context(identity, request, proposal_store, history_store)
    docker = shutil.which("docker")
    if docker is None:
        raise ValidationRunnerError("docker_unavailable")
    image_id = check_sandbox_runtime()["image_id"]
    workspace = context.workspace
    admissions = ValidationAdmissionStore(proposal_store)
    reservation = admissions.admit(identity, request, workspace, image_id, authorize=authorize)
    try:
        running = admissions.start(identity, reservation.validation_id, authorize=authorize)
    except BaseException:
        try:
            admissions.cancel_unstarted(identity.id, reservation.validation_id)
        except ValidationAdmissionError:
            raise ValidationRunnerError(
                "reservation_retained", validation_id=reservation.validation_id, cleanup_confirmed=False,
            ) from None
        raise

    def release():
        try:
            return admissions.close(identity.id, running.validation_id)
        except ValidationAdmissionError:
            raise ValidationRunnerError("reservation_retained", validation_id=running.validation_id) from None

    try:
        before = _phase(docker, running, "baseline", workspace.baseline_payload)
        after = _phase(docker, running, "patched", workspace.patched_payload)
    except ValidationRunnerError as error:
        if error.cleanup_confirmed:
            release()
        raise ValidationRunnerError(
            error.code, validation_id=running.validation_id, cleanup_confirmed=error.cleanup_confirmed,
        ) from None
    try:
        return ValidationResultStore(proposal_store).complete(identity.id, running.validation_id, before, after)
    except ValidationAdmissionError:
        raise ValidationRunnerError("result_save_failed", validation_id=running.validation_id) from None
