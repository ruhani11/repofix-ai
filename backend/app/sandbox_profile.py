"""Server-owned Python Docker profile and trusted-image health check.

This CLI checks only our runtime image; it never loads a target repository,
proposal or generated test. Do not reuse its subprocess helper for untrusted
test output. Target execution needs a separate bounded-output runner and atomic
current-approval admission, neither of which is implemented here.
"""

import json
import re
import shutil
import subprocess
import time
from uuid import uuid4


PROFILE_ID = "python-pytest-v1"
PROFILE_IMAGE = "repofix-python-validation:v1"
PROFILE_LABEL = "ai.repofix.validation-profile"
RUNTIME_LABEL = "ai.repofix.validation-runtime"
RUNTIME_REVISION = 3
HEALTH_TIMEOUT_SECONDS = 60
MEMORY_BYTES = 512 * 1024 * 1024
PIDS_LIMIT = 64
_IMAGE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID = re.compile(r"[0-9a-f]{32}\Z")
_CHECKS = frozenset({
    "non_root", "isolated_python", "no_capabilities", "no_new_privileges",
    "seccomp_enabled", "root_read_only", "network_disabled", "memory_limited",
    "swap_disabled", "cpu_limited", "pids_limited", "tmpfs_restricted",
    "scratch_writable", "dependencies_ready",
})


class SandboxProfileError(RuntimeError):
    def __init__(self, code: str, *, checks: dict | None = None):
        self.code = code
        self.checks = checks
        super().__init__("The restricted Python runtime is not ready.")


def sandbox_container_name(validation_id: str) -> str:
    if type(validation_id) is not str or _ID.fullmatch(validation_id) is None:
        raise SandboxProfileError("invalid_id")
    return "repofix-validation-" + validation_id


def build_sandbox_create_args(image_id: str, validation_id: str) -> tuple[str, ...]:
    """Fixed CLI arguments, excluding executable; no shell or host mounts.

    Only a full inspected local image ID is accepted, never a mutable tag, a
    browser-provided command, a repository Dockerfile or client runtime options.
    This builder retains the trusted runtime self-check command.
    """
    if type(image_id) is not str or _IMAGE.fullmatch(image_id) is None:
        raise SandboxProfileError("invalid_image")
    name = sandbox_container_name(validation_id)
    return (
        "create", "--name", name, "--pull", "never", "--init",
        "--label", f"{PROFILE_LABEL}={PROFILE_ID}",
        "--network", "none", "--read-only", "--user", "10001:10001",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true",
        "--memory", "512m", "--memory-swap", "512m", "--cpus", "1",
        "--pids-limit", "64", "--ulimit", "nofile=128:128", "--ulimit", "core=0:0",
        "--shm-size", "16m", "--restart", "no", "--log-driver", "none",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=67108864,mode=1777",
        "--tmpfs", "/workspace:rw,noexec,nosuid,nodev,size=16777216,uid=10001,gid=10001,mode=0700",
        "--workdir", "/workspace", "--env", "HOME=/tmp",
        "--env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONUNBUFFERED=1",
        "--entrypoint", "/usr/local/bin/python", image_id,
        "-I", "-B", "/opt/repofix/sandbox_runtime.py", "--self-check",
    )


def build_sandbox_phase_create_args(image_id: str, validation_id: str, phase: str) -> tuple[str, ...]:
    """Same fixed restrictions with a private stdin-fed phase worker.

    Internal runner only: this grants no approval and must follow atomic
    admission/start for target work. Diagnostics use fixed operator-owned data.
    """
    if type(phase) is not str or phase not in {"baseline", "patched"}:
        raise SandboxProfileError("invalid_phase")
    args = list(build_sandbox_create_args(image_id, validation_id))
    args[args.index("--name") + 1] += "-" + phase
    args.insert(1, "--interactive")
    args[-1] = "--run-phase"
    return tuple(args)


def _health_command(
    docker: str, args: tuple[str, ...], deadline: float, *, allow_nonzero: bool = False,
) -> str:
    """Trusted diagnostic commands only; target logs need a different runner."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SandboxProfileError("timeout")
    try:
        result = subprocess.run(
            [docker, *args], shell=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=remaining, check=False,
        )
    except subprocess.TimeoutExpired:
        raise SandboxProfileError("timeout") from None
    except OSError:
        raise SandboxProfileError("docker_unavailable") from None
    if len(result.stdout.encode("utf-8")) > 16 * 1024:
        raise SandboxProfileError("invalid_runtime_response")
    if result.returncode != 0:
        if allow_nonzero:
            try:
                report = json.loads(result.stdout)
                checks = report["checks"]
                if type(checks) is dict and set(checks) == _CHECKS and all(type(value) is bool for value in checks.values()):
                    raise SandboxProfileError("runtime_checks_failed", checks=checks)
            except (KeyError, TypeError, ValueError):
                pass
        raise SandboxProfileError("docker_command_failed")
    return result.stdout.strip()


def check_sandbox_runtime() -> dict:
    """Operator-invoked local health check; creates/removes one trusted container.

    The image must be built explicitly beforehand. No implicit image pull/build,
    host folder, Docker socket, API key or target code is supplied to the image.
    Docker and the locally operator-built image are trusted prerequisites.
    """
    docker = shutil.which("docker")
    if docker is None:
        raise SandboxProfileError("docker_unavailable")
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    if _health_command(docker, ("info", "--format", "{{.OSType}}"), deadline) != "linux":
        raise SandboxProfileError("linux_required")
    try:
        image = json.loads(_health_command(
            docker, ("image", "inspect", PROFILE_IMAGE, "--format", "{{json .}}"), deadline,
        ))
        image_id = image["Id"]
        if (
            image["Os"] != "linux" or image["Config"]["Labels"][PROFILE_LABEL] != PROFILE_ID
            or image["Config"]["Labels"][RUNTIME_LABEL] != str(RUNTIME_REVISION)
        ):
            raise ValueError()
        validation_id = uuid4().hex
        create_args = build_sandbox_create_args(image_id, validation_id)
    except (KeyError, TypeError, ValueError):
        raise SandboxProfileError("invalid_image") from None
    name = sandbox_container_name(validation_id)
    created = False
    create_attempted = False
    try:
        create_attempted = True
        _health_command(docker, create_args, deadline)
        created = True
        raw = _health_command(docker, ("start", "--attach", name), deadline, allow_nonzero=True)
        try:
            result = json.loads(raw)
            checks = result["checks"]
            if (
                result["profile"] != PROFILE_ID
                or type(result.get("runtime_revision")) is not int or result["runtime_revision"] != RUNTIME_REVISION
                or type(checks) is not dict or set(checks) != _CHECKS
                or any(type(value) is not bool for value in checks.values())
            ):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise SandboxProfileError("runtime_checks_failed") from None
        if result.get("status") != "ready" or not all(checks.values()):
            raise SandboxProfileError("runtime_checks_failed", checks=checks)
        return {"status": "ready", "profile": PROFILE_ID, "runtime_revision": RUNTIME_REVISION, "image_id": image_id, "checks": checks}
    finally:
        if created or create_attempted:
            # A timed-out create can have reached the daemon. The name is a
            # fresh server-generated UUID, never a client-selected container.
            try:
                _health_command(docker, ("rm", "--force", name), time.monotonic() + 10)
            except SandboxProfileError:
                # Missing after a failed create needs no cleanup. Do not hide a
                # cleanup failure after a confirmed successful create.
                if created:
                    raise SandboxProfileError("cleanup_failed") from None


def main() -> int:
    try:
        result = check_sandbox_runtime()
    except SandboxProfileError as error:
        failure = {"status": "unavailable", "reason": error.code}
        if error.checks is not None:
            failure["checks"] = error.checks
        print(json.dumps(failure, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
