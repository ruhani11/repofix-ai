"""Exercise job ordering with real admission state and a fake Docker daemon."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from app import validation_runner as runner
from app.sandbox_profile import RUNTIME_REVISION
from app.validation_admission import ValidationAdmissionError
from test_proposal_review_store import OWNER, decision
from test_validation_admission import IMAGE_ID, admit, saved


@pytest.fixture
def job(monkeypatch, saved):
    events = []
    phases = {}
    state = SimpleNamespace(events=events, phases=phases, create_error=None, start_error=None,
                            rm_error=False, recover=True, patched_exit=0)
    monkeypatch.setattr(runner, "prepare_validation_context", lambda *a: SimpleNamespace(workspace=saved.workspace))
    monkeypatch.setattr(runner.shutil, "which", lambda name: "docker")
    monkeypatch.setattr(runner, "check_sandbox_runtime", lambda: {"image_id": IMAGE_ID})

    def command(docker, args, *, timeout, stdin=None):
        assert docker == "docker" and timeout > 0
        command = args[0]
        events.append(command)
        if command == "create":
            name = args[args.index("--name") + 1]
            run_id, phase = name.removeprefix("repofix-validation-").split("-")
            state.run_id = run_id
            assert saved.admissions.get(OWNER.id, run_id).state == "running"
            assert IMAGE_ID in args and args[-1] == "--run-phase"
            assert args[args.index("--network") + 1] == "none"
            assert "--read-only" in args and "--interactive" in args
            assert f"ai.repofix.validation-run={run_id}" in args
            container = "c" * 64 if phase == "baseline" else "d" * 64
            phases[container] = (name, phase)
            if state.create_error:
                raise runner.ValidationRunnerError(state.create_error)
            return 0, (container + "\n").encode()
        if command == "container":
            assert args[1] == "inspect"
            container = next(k for k, (name, _) in phases.items() if name == args[-1])
            if not state.recover:
                return 1, b""
            return 0, f"{container} {IMAGE_ID} /{args[-1]} {state.run_id}\n".encode()
        if command == "rm":
            assert args == ("rm", "--force", args[-1]) and args[-1] in phases
            if state.rm_error:
                return 1, b""
            del phases[args[-1]]
            return 0, b""
        assert command == "start"
        name, phase = phases[args[-1]]
        assert args[1:3] == ("--attach", "--interactive")
        assert stdin == getattr(saved.workspace, f"{phase}_payload") + b"\n"
        if state.start_error:
            raise runner.ValidationRunnerError(state.start_error)
        failed = int(phase == "baseline")
        return (state.patched_exit if phase == "patched" else 0), json.dumps({
            "status": "completed", "profile": "python-pytest-v1", "runtime_revision": RUNTIME_REVISION,
            "protocol_version": 1, "phase": phase, "payload_sha256": hashlib.sha256(stdin[:-1]).hexdigest(),
            "result": {"collected": 1, "passed": 1 - failed, "failed": failed, "skipped": 0,
                       "errors": 0, "exit_code": failed, "nodeids_sha256": "e" * 64},
            "worker_output_bytes": 140,
        }).encode()

    monkeypatch.setattr(runner, "_command", command)
    state.run = lambda authorize=lambda: None: runner.run_approved_validation(
        OWNER, saved.payload, saved.store, object(), authorize=authorize,
    )
    return state


def test_starts_only_after_approval_and_closes_after_both_removals(job, saved):
    result = job.run(lambda: job.events.append("authorize"))
    assert job.events == ["authorize"] * 3 + ["create", "start", "rm"] * 2
    assert not job.phases
    assert result.verdict == "verified"
    assert result.runtime_revision == RUNTIME_REVISION
    previous = type(result).model_validate_json(json.dumps({**json.loads(result.model_dump_json()), "runtime_revision": 2}))
    assert previous.verdict == "verified" and previous.runtime_revision == 2
    assert result.admission.state == "closed"
    assert saved.admissions.get(OWNER.id, job.run_id) == result.admission


def test_rejected_start_cancels_only_unstarted_reservation(job, saved, monkeypatch):
    start = runner.ValidationAdmissionStore.start

    def revoke_before_start(store, identity, validation_id, **kwargs):
        job.run_id = validation_id
        saved.store.record_decision(OWNER, decision(saved.source.proposal, decision="reject"), expected_revision=1)
        return start(store, identity, validation_id, **kwargs)

    monkeypatch.setattr(runner.ValidationAdmissionStore, "start", revoke_before_start)
    with pytest.raises(ValidationAdmissionError) as error:
        job.run()
    assert error.value.code == "approval_required"
    assert not job.events and not job.phases
    closed = saved.admissions.get(OWNER.id, job.run_id)
    assert closed.state == "closed" and closed.started_at is None


def test_cancel_unstarted_cannot_close_running_reservation(saved):
    record = admit(saved)
    saved.admissions.start(OWNER, record.validation_id, authorize=lambda: None)
    with pytest.raises(ValidationAdmissionError) as error:
        saved.admissions.cancel_unstarted(OWNER.id, record.validation_id)
    assert error.value.code == "already_started"
    assert saved.admissions.get(OWNER.id, record.validation_id).state == "running"


@pytest.mark.parametrize(("settings", "reason", "events", "closed"), [
    ({"start_error": "timeout"}, "timeout", ["create", "start", "rm"], True),
    ({"start_error": "output_limit"}, "output_limit", ["create", "start", "rm"], True),
    ({"patched_exit": 1}, "docker_phase_failed", ["create", "start", "rm"] * 2, True),
    ({"create_error": "timeout"}, "timeout", ["create", "container", "rm"], True),
    ({"create_error": "timeout", "recover": False}, "cleanup_unconfirmed", ["create", "container"], False),
    ({"rm_error": True}, "cleanup_unconfirmed", ["create", "start", "rm"], False),
    ({"start_error": "docker_cli_stop_unconfirmed", "rm_error": True}, "cleanup_unconfirmed", ["create", "start", "rm"], False),
])
def test_failures_never_return_success_and_retain_uncertain_cleanup(job, saved, settings, reason, events, closed):
    for key, value in settings.items():
        setattr(job, key, value)
    with pytest.raises(runner.ValidationRunnerError) as error:
        job.run()
    assert error.value.code == reason and error.value.validation_id == job.run_id
    assert job.events == events
    assert error.value.cleanup_confirmed == closed
    assert saved.admissions.get(OWNER.id, job.run_id).state == ("closed" if closed else "running")


def test_cli_streams_stdin_and_stdout_without_merging_stderr():
    code, stdout = runner._command(sys.executable, ("-c", "import sys; sys.stderr.write('ignored'); sys.stdout.buffer.write(sys.stdin.buffer.read())"),
                                   timeout=10, stdin=b'{"test":true}\n')
    assert code == 0 and stdout == b'{"test":true}\n'


@pytest.mark.parametrize(("script", "timeout", "reason"), [
    ("import sys; sys.stdout.buffer.write(b'x' * 100000)", 10, "output_limit"),
    ("import time; time.sleep(30)", 0.2, "timeout"),
])
def test_cli_terminates_on_output_limit_or_timeout(script, timeout, reason):
    with pytest.raises(runner.ValidationRunnerError) as error:
        runner._command(sys.executable, ("-c", script), timeout=timeout)
    assert error.value.code == reason and error.value.cleanup_confirmed
