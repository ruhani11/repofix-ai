"""Runtime boundary tests; target execution and Docker are always mocked here."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

import app.sandbox_profile as profile
import app.sandbox_runtime_check as diagnostic
from test_sandbox_profile import fake_docker


RUNTIME = Path(__file__).resolve().parents[1] / "sandbox" / "python" / "sandbox_runtime.py"


@pytest.fixture(autouse=True)
def no_live_network_or_target_code(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Runtime unit checks must not launch Docker/target code or use network.")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)


@pytest.fixture
def runtime():
    spec = importlib.util.spec_from_file_location("repofix_test_runtime_entrypoint", RUNTIME)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoded(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def change_file(data, content=None, path=None):
    item = data["files"][0]
    if content is not None:
        item["content"] = content
        item["sha256"] = hashlib.sha256(content.encode()).hexdigest()
    if path is not None: item["path"] = path


def test_fixed_diagnostic_payloads_match_wire_protocol_with_same_tests(runtime):
    payloads = diagnostic._diagnostic_payloads()
    before, after = [runtime.validate_payload(payloads[phase]) for phase in ["baseline", "patched"]]
    assert before["binding"] == after["binding"] and before["tests"] == after["tests"]
    originals = {item["path"]: item["content"] for item in before["files"]}
    patched = {item["path"]: item["content"] for item in after["files"]}
    assert '<= 1:' in originals["orders_api.py"] and '<= 0:' in patched["orders_api.py"]
    assert originals["test_orders_api.py"] == patched["test_orders_api.py"]


@pytest.mark.parametrize("path", ["../bad.py", "/bad.py", "C:/bad.py", "bad\\x.py", "secrets/private.py", ".env.py", "CON.py", "bad./x.py", "x.txt", "a/../b.py"])
def test_runtime_rejects_unsafe_nonportable_or_blocked_paths(runtime, path):
    data = json.loads(diagnostic._diagnostic_payloads()["baseline"])
    change_file(data, path=path)
    with pytest.raises(runtime.RuntimeInputError): runtime.validate_payload(encoded(data))


@pytest.mark.parametrize("change", [
    {"command": "pytest --help"}, {"profile": "arbitrary"}, {"protocol_version": True},
    {"phase": "publish"}, {"tests": ["orders_api.py"]}, {"tests": []},
])
def test_unknown_execution_controls_and_invalid_protocol_are_rejected(runtime, change):
    data = json.loads(diagnostic._diagnostic_payloads()["baseline"])
    with pytest.raises(runtime.RuntimeInputError): runtime.validate_payload(encoded({**data, **change}))


@pytest.mark.parametrize("field,value", [("account_id", True), ("repository_id", "1"), ("review_revision", 0), ("decision_id", "d" * 31), ("patch_sha256", "d" * 63), ("bound_at", "2026-11-01"), ("repository_full_name", "repo/..")])
def test_binding_metadata_is_strict_but_is_not_an_authorization_token(runtime, field, value):
    data = json.loads(diagnostic._diagnostic_payloads()["baseline"])
    data["binding"][field] = value
    with pytest.raises(runtime.RuntimeInputError): runtime.validate_payload(encoded(data))


@pytest.mark.parametrize("kind", ["hash", "duplicate", "ancestor", "case_alias", "control", "large_file", "large_payload", "noncanonical", "duplicate_json_key"])
def test_byte_budgets_integrity_and_path_conflicts_fail_before_writes(runtime, kind):
    data = json.loads(diagnostic._diagnostic_payloads()["baseline"])
    if kind == "hash": data["files"][0]["sha256"] = "d" * 64
    elif kind == "duplicate": data["files"].append(data["files"][0])
    elif kind == "ancestor":
        extra = dict(data["files"][0], path="orders_api.py/helper.py")
        data["files"].append(extra); data["files"].sort(key=lambda item:item["path"])
    elif kind == "case_alias":
        data["files"].append(dict(data["files"][0], path="ORDERS_API.py")); data["files"].sort(key=lambda item:item["path"])
    elif kind == "control": change_file(data, content="PRIVATE\x00CODE")
    elif kind == "large_file": change_file(data, content="x" * (100 * 1024 + 1))
    raw = encoded(data)
    if kind == "large_payload": raw = b" " * (2 * 1024 * 1024 + 1)
    elif kind == "noncanonical": raw = json.dumps(data, indent=2).encode()
    elif kind == "duplicate_json_key": raw = raw[:-1] + b',"phase":"baseline"}'
    with pytest.raises(runtime.RuntimeInputError): runtime.validate_payload(raw)


def test_source_bytes_are_written_exactly_only_into_empty_private_workspace(runtime, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "WORKSPACE", tmp_path)
    content = "# café\r\nraise RuntimeError('DATA_ONLY')"
    runtime._write_workspace([{"path": "pkg/source.py", "content": content}])
    assert (tmp_path / "pkg/source.py").read_bytes() == content.encode()
    with pytest.raises(runtime.RuntimeInputError): runtime._write_workspace([])


def test_runtime_checks_total_source_and_file_count_budgets_independently(runtime):
    data = json.loads(diagnostic._diagnostic_payloads()["baseline"])
    item = data["files"][0]
    data["files"] = [dict(item, path=f"source_{i}.py") for i in range(21)] + [data["files"][1]]
    data["files"].sort(key=lambda item:item["path"])
    with pytest.raises(runtime.RuntimeInputError): runtime.validate_payload(encoded(data))
    content = "x" * (100 * 1024)
    template = {"content": content, "sha256": hashlib.sha256(content.encode()).hexdigest()}
    data["files"] = [dict(template, path=f"source_{i}.py") for i in range(6)] + [data["files"][-1]]
    data["files"].sort(key=lambda item:item["path"])
    with pytest.raises(runtime.RuntimeInputError): runtime.validate_payload(encoded(data))


@pytest.mark.parametrize("change", [{"passed": True}, {"exit_code": 2}, {"collected": 0}, {"errors": 1, "exit_code": 0}, {"nodeids_sha256": "bad"}])
def test_worker_result_file_requires_strict_consistent_counters(runtime, tmp_path, monkeypatch, change):
    path = tmp_path / "report.json"
    monkeypatch.setattr(runtime, "RESULT_FILE", path)
    data = {"collected": 1, "passed": 1, "failed": 0, "skipped": 0, "errors": 0, "exit_code": 0, "nodeids_sha256": "a" * 64}
    path.write_bytes(encoded(data))
    assert runtime._read_worker_report() == data
    path.write_bytes(encoded({**data, **change}))
    with pytest.raises(runtime.RuntimeInputError): runtime._read_worker_report()


def test_phase_stdin_uses_a_single_bounded_line_frame(runtime, monkeypatch, capsys):
    raw = diagnostic._diagnostic_payloads()["baseline"]
    fake_sys = SimpleNamespace(argv=["runtime", "--run-phase"], stdin=SimpleNamespace(buffer=io.BytesIO(raw+b"\n")))
    monkeypatch.setattr(runtime, "sys", fake_sys)
    phase = Mock(return_value={"status": "completed"})
    monkeypatch.setattr(runtime, "run_phase", phase)
    assert runtime.main() == 0
    phase.assert_called_once_with(raw)
    capsys.readouterr()


def test_runtime_guard_prevents_any_writes_or_worker_start_on_host(runtime, monkeypatch):
    writer, execute = Mock(), Mock()
    monkeypatch.setattr(runtime, "inspect_runtime", lambda: {"status": "unavailable"})
    monkeypatch.setattr(runtime, "_write_workspace", writer)
    monkeypatch.setattr(runtime, "_execute_tests", execute)
    assert runtime.run_phase(diagnostic._diagnostic_payloads()["baseline"])["reason"] == "runtime_not_ready"
    writer.assert_not_called(); execute.assert_not_called()


def test_valid_phase_uses_fixed_selected_tests_and_returns_only_metadata(runtime, monkeypatch):
    raw = diagnostic._diagnostic_payloads()["baseline"]
    monkeypatch.setattr(runtime, "inspect_runtime", lambda: {"status": "ready"})
    writer = Mock()
    metrics = {"collected": 1, "passed": 0, "failed": 1, "skipped": 0, "errors": 0, "exit_code": 1, "nodeids_sha256": "a" * 64}
    execute = Mock(return_value=(metrics, 123))
    monkeypatch.setattr(runtime, "_write_workspace", writer)
    monkeypatch.setattr(runtime, "_execute_tests", execute)
    result = runtime.run_phase(raw)
    assert result["status"] == "completed" and result["result"] == metrics
    assert result["payload_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["runtime_revision"] == profile.RUNTIME_REVISION and result["worker_output_bytes"] == 123
    execute.assert_called_once_with(["test_orders_api.py"])
    assert not {"logs", "source", "binding", "files"} & result.keys()


@pytest.mark.parametrize("reason", ["timeout", "output_limit", "worker_io_error"])
def test_controlled_worker_failures_never_claim_a_completed_phase(runtime, monkeypatch, reason):
    monkeypatch.setattr(runtime, "inspect_runtime", lambda: {"status": "ready"})
    monkeypatch.setattr(runtime, "_write_workspace", Mock())
    monkeypatch.setattr(runtime, "_execute_tests", Mock(side_effect=runtime.RuntimeInputError(reason)))
    assert runtime.run_phase(diagnostic._diagnostic_payloads()["baseline"]) == {"status": "error", "reason": reason}


def test_pytest_flags_ignore_repository_configuration_and_plugin_environment(runtime, monkeypatch):
    observed = {}
    def fake_main(args, plugins):
        observed["args"], observed["reporter"] = args, plugins[0]
        reporter = plugins[0]
        reporter.pytest_collection_modifyitems(None, None, [SimpleNamespace(nodeid="test_orders_api.py::test_one")])
        reporter.pytest_runtest_logreport(SimpleNamespace(nodeid="test_orders_api.py::test_one", skipped=False, when="call", passed=True, failed=False))
        return 0
    monkeypatch.setattr(pytest, "main", fake_main)
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(path=[]))
    monkeypatch.setenv("PYTEST_ADDOPTS", "--dangerous-option")
    monkeypatch.setenv("PYTEST_PLUGINS", "arbitrary_plugin")
    # Restore environment removals as well as additions at fixture teardown.
    for key in ["PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONPATH", "PYTHONSTARTUP", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"]:
        monkeypatch.setenv(key, runtime.os.environ.get(key, ""))
    result = runtime._pytest_report(["test_orders_api.py"])
    assert result["passed"] == 1 and result["collected"] == 1 and result["exit_code"] == 0
    args = observed["args"]
    assert {"--capture=no", "--noconftest", "--disable-plugin-autoload", "--import-mode=importlib"} <= set(args)
    assert args[args.index("-c")+1] == str(runtime.CONFIG_FILE)
    assert args[-1] == str(runtime.WORKSPACE / "test_orders_api.py")
    assert "PYTEST_ADDOPTS" not in runtime.os.environ and "PYTEST_PLUGINS" not in runtime.os.environ


@pytest.mark.parametrize("threshold,expected", [(0, (1, 0, 0)), (-1, (0, 1, 1))])
def test_unittest_subtests_count_once_and_preserve_failure(runtime, tmp_path, monkeypatch, threshold, expected):
    # Run a fixed, local synthetic test through the pinned pytest reporter.
    test_name = f"test_quantity_{threshold + 1}.py"
    (tmp_path / test_name).write_text(
        "import unittest\n"
        "class Quantity(unittest.TestCase):\n"
        "    def test_nonpositive(self):\n"
        "        for quantity in (0, -1):\n"
        "            with self.subTest(quantity=quantity):\n"
        f"                self.assertLessEqual(quantity, {threshold})\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "WORKSPACE", tmp_path)
    monkeypatch.setattr(runtime.sys, "path", runtime.sys.path.copy())
    for key in ["PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONPATH", "PYTHONSTARTUP", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"]:
        monkeypatch.setenv(key, runtime.os.environ.get(key, ""))
    result = runtime._pytest_report([test_name])
    assert result["collected"] == 1 and result["errors"] == result["skipped"] == 0
    assert (result["passed"], result["failed"], result["exit_code"]) == expected


@pytest.mark.parametrize("mode", ["normal", "overflow", "timeout"])
def test_worker_supervisor_counts_discarded_bytes_and_stops_on_limits(runtime, monkeypatch, mode):
    class Process:
        pid = 12345
        def __init__(self):
            self.returncode = 0 if mode == "normal" else None
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(b"x" * (65537 if mode == "overflow" else 10))
        def poll(self): return self.returncode
        def kill(self): self.returncode = -9
        def wait(self, timeout): return self.returncode
    process = Process()
    popen = Mock(return_value=process)
    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    monkeypatch.setattr(runtime.os, "killpg", lambda *args: process.kill(), raising=False)
    monkeypatch.setattr(runtime, "signal", SimpleNamespace(SIGKILL=9))
    if mode == "timeout":
        values = iter([0, 46])
        monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: next(values), sleep=lambda _:None))
    monkeypatch.setattr(runtime, "_read_worker_report", lambda: {"exit_code": 0})
    if mode == "normal":
        assert runtime._execute_tests(["test_orders_api.py"]) == ({"exit_code": 0}, 10)
    else:
        with pytest.raises(runtime.RuntimeInputError) as caught: runtime._execute_tests(["test_orders_api.py"])
        assert str(caught.value) == ("output_limit" if mode == "overflow" else "timeout")
        assert process.returncode == -9
    assert popen.call_args.kwargs["shell"] is False and popen.call_args.kwargs["start_new_session"] is True


def test_phase_args_preserve_restrictions_and_only_change_stdin_command_and_name():
    image, job = "sha256:" + "a" * 64, "b" * 32
    base = list(profile.build_sandbox_create_args(image, job))
    for phase in ["baseline", "patched"]:
        args = list(profile.build_sandbox_phase_create_args(image, job, phase))
        assert args.pop(1) == "--interactive"
        args[args.index("--name") + 1] = base[base.index("--name") + 1]
        assert args[-1] == "--run-phase"
        args[-1] = "--self-check"
        assert args == base
    with pytest.raises(profile.SandboxProfileError): profile.build_sandbox_phase_create_args(image, job, "publish")


@pytest.mark.parametrize("where", ["label", "response"])
def test_old_runtime_revision_is_rejected(fake_docker, where):
    state, calls = fake_docker
    if where == "label": state["image"]["Config"]["Labels"][profile.RUNTIME_LABEL] = "1"
    else: state["report"]["runtime_revision"] = 1
    with pytest.raises(profile.SandboxProfileError): profile.check_sandbox_runtime()
    if where == "label": assert not any(call[1] == "create" for call in calls)


@pytest.fixture
def fake_diagnostic(monkeypatch):
    calls = []
    state = {"damage": None, "cleanup_failed": False}
    monkeypatch.setattr(diagnostic, "check_sandbox_runtime", lambda: {"image_id": "sha256:" + "a" * 64})
    monkeypatch.setattr(diagnostic.shutil, "which", lambda _: "docker")
    def health(docker, args, deadline):
        calls.append(args)
        if state["cleanup_failed"] and args[0] == "rm": raise profile.SandboxProfileError("cleanup_failed")
        return "ok"
    monkeypatch.setattr(diagnostic, "_health_command", health)
    def run(args, **kwargs):
        assert kwargs["shell"] is False and kwargs["timeout"] == 60
        assert kwargs["input"].endswith(b"\n")
        packet = kwargs["input"][:-1]
        data = json.loads(packet)
        baseline = data["phase"] == "baseline"
        result = {"collected": 1, "passed": 0 if baseline else 1, "failed": 1 if baseline else 0,
                  "skipped": 0, "errors": 0, "exit_code": 1 if baseline else 0, "nodeids_sha256": "d" * 64}
        body = {"status": "completed", "profile": profile.PROFILE_ID, "runtime_revision": profile.RUNTIME_REVISION,
                "protocol_version": 1, "phase": data["phase"], "payload_sha256": hashlib.sha256(packet).hexdigest(),
                "result": result, "worker_output_bytes": 10}
        if state["damage"] == "skip": result["skipped"] = 1
        if state["damage"] == "digest": body["payload_sha256"] = "f" * 64
        if state["damage"] == "tests" and not baseline: result["nodeids_sha256"] = "f" * 64
        return SimpleNamespace(returncode=0, stdout=encoded(body), stderr=b"PRIVATE_NOT_LOGGED")
    monkeypatch.setattr(diagnostic.subprocess, "run", run)
    return state, calls


def test_diagnostic_runs_only_fixed_fixture_and_cleans_each_phase(fake_diagnostic):
    _, calls = fake_diagnostic
    result = diagnostic.check_test_entrypoint()
    assert result["status"] == "runtime_self_test_passed" and result["diagnostic_only"] is True
    assert result["baseline"]["failed"] == result["patched"]["passed"] == 1
    assert [args[0] for args in calls] == ["create", "rm", "create", "rm"]
    assert calls[0][calls[0].index("--name")+1].endswith("-baseline")
    assert calls[2][calls[2].index("--name")+1].endswith("-patched")


@pytest.mark.parametrize("damage", ["skip", "digest", "tests", "cleanup"])
def test_diagnostic_never_passes_with_skips_mixed_payloads_tests_or_cleanup_failure(fake_diagnostic, damage):
    state, _ = fake_diagnostic
    if damage == "cleanup": state["cleanup_failed"] = True
    else: state["damage"] = damage
    with pytest.raises(profile.SandboxProfileError): diagnostic.check_test_entrypoint()
