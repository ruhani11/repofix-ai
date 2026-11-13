"""Offline profile/health-check tests; Docker and target execution are mocked."""

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

import app.sandbox_profile as profile


IMAGE_ID = "sha256:" + "a" * 64
RUN_ID = "b" * 32
RUNTIME_PATH = Path(__file__).resolve().parents[1] / "sandbox" / "python" / "sandbox_runtime.py"


@pytest.fixture(autouse=True)
def no_real_network_or_docker(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Profile tests must not start Docker or use the network.")
    monkeypatch.setattr(subprocess, "Popen", blocked)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked)


def test_profile_disallows_host_access_and_sets_bounded_container_controls():
    args = profile.build_sandbox_create_args(IMAGE_ID, RUN_ID)
    assert isinstance(args, tuple) and args[0] == "create"
    assert args[-5:] == (IMAGE_ID, "-I", "-B", "/opt/repofix/sandbox_runtime.py", "--self-check")
    assert args[args.index("--name") + 1] == "repofix-validation-" + RUN_ID
    required = {
        "--pull": "never", "--network": "none", "--user": "10001:10001",
        "--cap-drop": "ALL", "--security-opt": "no-new-privileges=true",
        "--memory": "512m", "--memory-swap": "512m", "--cpus": "1",
        "--pids-limit": "64", "--shm-size": "16m", "--restart": "no",
        "--log-driver": "none", "--workdir": "/workspace",
        "--entrypoint": "/usr/local/bin/python",
    }
    for flag, value in required.items():
        assert args[args.index(flag) + 1] == value
    assert "--read-only" in args and "--init" in args
    assert not {"--privileged", "--mount", "--volume", "-v", "--env-file", "--publish", "-p", "--pid", "--device", "--use-api-socket"} & set(args)
    env = [args[i + 1] for i, flag in enumerate(args) if flag == "--env"]
    assert set(env) == {"HOME=/tmp", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1", "PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1"}
    tmpfs = [args[i + 1] for i, flag in enumerate(args) if flag == "--tmpfs"]
    assert len(tmpfs) == 2
    for mount in tmpfs:
        assert {"rw", "noexec", "nosuid", "nodev"} <= set(mount.split(":", 1)[1].split(","))
        assert "size=" in mount


@pytest.mark.parametrize("image_id", [None, True, "latest", profile.PROFILE_IMAGE, "sha256:" + "A" * 64, IMAGE_ID + "\n", IMAGE_ID + " --privileged", "sha256:" + "a" * 63])
def test_only_full_local_image_ids_are_accepted(image_id):
    with pytest.raises(profile.SandboxProfileError) as caught:
        profile.build_sandbox_create_args(image_id, RUN_ID)
    assert caught.value.code == "invalid_image"


@pytest.mark.parametrize("run_id", [None, True, "", "A" * 32, RUN_ID + "\n", "../host", "a" * 31])
def test_names_cannot_inject_cli_options_or_other_containers(run_id):
    with pytest.raises(profile.SandboxProfileError) as caught:
        profile.build_sandbox_create_args(IMAGE_ID, run_id)
    assert caught.value.code == "invalid_id"


@pytest.fixture
def fake_docker(monkeypatch):
    calls = []
    image = {"Id": IMAGE_ID, "Os": "linux", "Config": {"Labels": {profile.PROFILE_LABEL: profile.PROFILE_ID, profile.RUNTIME_LABEL: str(profile.RUNTIME_REVISION)}}}
    report = {"profile": profile.PROFILE_ID, "runtime_revision": profile.RUNTIME_REVISION, "status": "ready", "checks": {key: True for key in profile._CHECKS}}
    state = {"image": image, "report": report, "os": "linux", "fail": None, "start_exit": 0}
    monkeypatch.setattr(profile.shutil, "which", lambda name: "C:/Docker Path/docker.exe")

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[0] == "C:/Docker Path/docker.exe" and kwargs["shell"] is False
        assert kwargs["timeout"] > 0 and "env" not in kwargs and "input" not in kwargs
        cmd = argv[1]
        if state["fail"] == cmd:
            raise subprocess.TimeoutExpired(argv, 1)
        if cmd == "info": out = state["os"]
        elif cmd == "image": out = json.dumps(state["image"])
        elif cmd == "create": out = "c" * 64
        elif cmd == "start": out = json.dumps(state["report"])
        elif cmd == "rm": out = argv[-1]
        else: pytest.fail("Unexpected Docker command.")
        return SimpleNamespace(returncode=state["start_exit"] if cmd == "start" else 0, stdout=out, stderr="PRIVATE_UNUSED_DIAGNOSTIC")

    monkeypatch.setattr(profile.subprocess, "run", run)
    return state, calls


def test_health_uses_inspected_id_checks_controls_and_cleans_up(fake_docker):
    _, calls = fake_docker
    result = profile.check_sandbox_runtime()
    assert result["status"] == "ready" and result["image_id"] == IMAGE_ID
    assert all(result["checks"].values())
    assert [call[1] for call in calls] == ["info", "image", "create", "start", "rm"]
    assert IMAGE_ID in calls[2] and profile.PROFILE_IMAGE not in calls[2]
    assert calls[3][-1] == calls[4][-1] == calls[2][calls[2].index("--name") + 1]
    assert len(calls[4][-1]) == len("repofix-validation-") + 32


@pytest.mark.parametrize("change", ["wrong_os", "missing_label", "wrong_label", "mutable_image"])
def test_untrusted_or_wrong_image_fails_before_container_creation(fake_docker, change):
    state, calls = fake_docker
    if change == "wrong_os": state["image"]["Os"] = "windows"
    elif change == "missing_label": state["image"]["Config"]["Labels"] = {}
    elif change == "wrong_label": state["image"]["Config"]["Labels"][profile.PROFILE_LABEL] = "different"
    else: state["image"]["Id"] = "arbitrary:latest"
    with pytest.raises(profile.SandboxProfileError) as caught:
        profile.check_sandbox_runtime()
    assert caught.value.code == "invalid_image"
    assert not any(call[1] == "create" for call in calls)


@pytest.mark.parametrize("damage", ["failed_check", "missing_check", "coerced_check", "wrong_profile", "wrong_status"])
def test_bad_health_reports_cannot_claim_readiness_and_still_cleanup(fake_docker, damage):
    state, calls = fake_docker
    if damage == "failed_check": state["report"]["checks"]["non_root"] = False
    elif damage == "missing_check": del state["report"]["checks"]["non_root"]
    elif damage == "coerced_check": state["report"]["checks"]["non_root"] = 1
    elif damage == "wrong_profile": state["report"]["profile"] = "other"
    else: state["report"]["status"] = "unavailable"
    with pytest.raises(profile.SandboxProfileError) as caught:
        profile.check_sandbox_runtime()
    assert caught.value.code == "runtime_checks_failed" and calls[-1][1] == "rm"


@pytest.mark.parametrize("command,expected", [("create", "timeout"), ("start", "timeout"), ("rm", "cleanup_failed")])
def test_timeout_cleanup_and_failed_removal_are_not_reported_ready(fake_docker, command, expected):
    state, calls = fake_docker
    state["fail"] = command
    with pytest.raises(profile.SandboxProfileError) as caught:
        profile.check_sandbox_runtime()
    assert caught.value.code == expected and calls[-1][1] == "rm"


def test_missing_cli_and_non_linux_engine_fail_closed(fake_docker, monkeypatch):
    state, calls = fake_docker
    state["os"] = "windows"
    with pytest.raises(profile.SandboxProfileError) as caught:
        profile.check_sandbox_runtime()
    assert caught.value.code == "linux_required" and len(calls) == 1
    monkeypatch.setattr(profile.shutil, "which", lambda _: None)
    with pytest.raises(profile.SandboxProfileError) as caught:
        profile.check_sandbox_runtime()
    assert caught.value.code == "docker_unavailable"


def test_cli_prints_only_safe_reason_on_failure(monkeypatch, capsys):
    def fail(): raise profile.SandboxProfileError("docker_unavailable")
    monkeypatch.setattr(profile, "check_sandbox_runtime", fail)
    assert profile.main() == 1
    assert json.loads(capsys.readouterr().out) == {"status": "unavailable", "reason": "docker_unavailable"}


def test_failed_container_check_reports_only_boolean_control_metadata(fake_docker, capsys):
    state, calls = fake_docker
    state["report"]["status"] = "unavailable"
    state["report"]["checks"]["memory_limited"] = False
    assert profile.main() == 1
    result = json.loads(capsys.readouterr().out)
    assert result["reason"] == "runtime_checks_failed"
    assert result["checks"]["memory_limited"] is False
    assert "PRIVATE" not in json.dumps(result) and calls[-1][1] == "rm"


def test_nonzero_process_exit_cannot_be_overridden_by_ready_json(fake_docker):
    state, calls = fake_docker
    state["start_exit"] = 1
    with pytest.raises(profile.SandboxProfileError) as caught:
        profile.check_sandbox_runtime()
    assert caught.value.code == "runtime_checks_failed" and calls[-1][1] == "rm"


@pytest.fixture
def runtime(monkeypatch):
    spec = importlib.util.spec_from_file_location("repofix_trusted_runtime", RUNTIME_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = {
        "/proc/self/status": "CapEff:\t0000000000000000\nNoNewPrivs:\t1\nSeccomp:\t2\n",
        "/proc/self/mountinfo": "1 0 0:1 / / ro - overlay overlay ro\n2 1 0:2 / /tmp rw,nosuid,nodev,noexec - tmpfs tmpfs rw\n3 1 0:3 / /workspace rw,nosuid,nodev,noexec - tmpfs tmpfs rw\n",
        "/proc/net/dev": "Inter-| Receive\n lo: 0 0 0\n",
        "/sys/fs/cgroup/memory.max": "536870912\n",
        "/sys/fs/cgroup/memory.swap.max": "0\n",
        "/sys/fs/cgroup/pids.max": "64\n", "/sys/fs/cgroup/cpu.max": "100000 100000\n",
    }
    monkeypatch.setattr(module, "_read", lambda path: files[path])
    monkeypatch.setattr(module, "os", SimpleNamespace(getuid=lambda: 10001, getgid=lambda: 10001))
    monkeypatch.setattr(module, "sys", SimpleNamespace(flags=SimpleNamespace(isolated=1), dont_write_bytecode=True, argv=["runtime", "--self-check"]))
    monkeypatch.setattr(module, "version", lambda name: module.EXPECTED[name])
    writable = Mock(return_value=True)
    monkeypatch.setattr(module, "_scratch_writable", writable)
    return module, files, writable


def test_runtime_checks_observed_linux_controls_and_dependencies(runtime):
    module, _, writable = runtime
    result = module.inspect_runtime()
    assert result["status"] == "ready" and set(result["checks"]) == profile._CHECKS
    assert all(result["checks"].values()) and writable.call_count == 1


@pytest.mark.parametrize("path,value,key", [
    ("/proc/self/status", "CapEff:\t1\nNoNewPrivs:\t1\nSeccomp:\t2\n", "no_capabilities"),
    ("/proc/self/status", "CapEff:\t0\nNoNewPrivs:\t0\nSeccomp:\t2\n", "no_new_privileges"),
    ("/proc/self/status", "CapEff:\t0\nNoNewPrivs:\t1\nSeccomp:\t0\n", "seccomp_enabled"),
    ("/proc/net/dev", "lo: 0\neth0: 1\n", "network_disabled"),
    ("/sys/fs/cgroup/memory.max", "max\n", "memory_limited"),
    ("/sys/fs/cgroup/memory.swap.max", "1024\n", "swap_disabled"),
    ("/sys/fs/cgroup/pids.max", "max\n", "pids_limited"),
    ("/sys/fs/cgroup/cpu.max", "200000 100000\n", "cpu_limited"),
])
def test_weakened_linux_controls_fail_without_scratch_writes(runtime, path, value, key):
    module, files, writable = runtime
    files[path] = value
    report = module.inspect_runtime()
    assert report["status"] == "unavailable" and report["checks"][key] is False
    writable.assert_not_called()


@pytest.mark.parametrize("change", ["writable_root", "executable_tmp", "wrong_user", "missing_cgroup", "wrong_dependency", "not_isolated"])
def test_incomplete_or_wrong_runtime_never_passes(runtime, change):
    module, files, writable = runtime
    if change == "writable_root": files["/proc/self/mountinfo"] = files["/proc/self/mountinfo"].replace("/ / ro", "/ / rw")
    elif change == "executable_tmp": files["/proc/self/mountinfo"] = files["/proc/self/mountinfo"].replace("rw,nosuid,nodev,noexec", "rw,nosuid,nodev")
    elif change == "wrong_user": module.os.getuid = lambda: 0
    elif change == "missing_cgroup": del files["/sys/fs/cgroup/memory.max"]
    elif change == "wrong_dependency": module.version = lambda _: "different"
    else: module.sys.flags.isolated = 0
    assert module.inspect_runtime()["status"] == "unavailable"
    writable.assert_not_called()


def test_runtime_refuses_target_commands_and_emits_safe_metadata(runtime, capsys):
    module, _, _ = runtime
    module.sys.argv = ["runtime", "pytest", "untrusted.py"]
    assert module.main() == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "unsupported_mode"
    module.sys.argv = ["runtime", "--self-check"]
    assert module.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"


def test_image_definition_contains_only_trusted_runtime_assets():
    directory = RUNTIME_PATH.parent
    dockerfile = (directory / "Dockerfile").read_text()
    copy_lines = [line for line in dockerfile.splitlines() if line.startswith("COPY ")]
    assert copy_lines == [
        "COPY sandbox-runtime-requirements.txt /opt/repofix/sandbox-runtime-requirements.txt",
        "COPY sandbox_runtime.py /opt/repofix/sandbox_runtime.py",
        "COPY pytest.ini /opt/repofix/pytest.ini",
    ]
    assert "USER 10001:10001" in dockerfile and "--only-binary=:all:" in dockerfile
    assert "CMD [\"--self-check\"]" in dockerfile
    requirements = (directory / "sandbox-runtime-requirements.txt").read_text()
    for name, value in {"fastapi": "0.141.1", "httpx": "0.28.1", "pytest": "9.2.0"}.items():
        assert f"{name}=={value}" in requirements
