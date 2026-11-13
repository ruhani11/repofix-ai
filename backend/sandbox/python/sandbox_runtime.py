"""Trusted container entrypoint. Host admission/cleanup remain runner duties.

Payload bindings are provenance metadata, not authorization tokens. Target code
runs only in the isolated pytest worker after all runtime/payload guards pass.
Observed pytest results are not cryptographic attestation of untrusted test code.
"""

from importlib.metadata import PackageNotFoundError, version
import json
import os
import hashlib
from datetime import datetime
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
from threading import Event, Thread
import time


PROFILE_ID = "python-pytest-v1"
RUNTIME_REVISION = 3
EXPECTED = {"fastapi": "0.141.1", "httpx": "0.28.1", "pytest": "9.2.0"}
MAX_PAYLOAD = 2 * 1024 * 1024
MAX_SOURCE = 512 * 1024
MAX_FILE = 100 * 1024
MAX_TEST_CASES = 100
MAX_WORKER_OUTPUT = 64 * 1024
TEST_TIMEOUT_SECONDS = 45
WORKSPACE = Path("/workspace")
RESULT_FILE = Path("/tmp/repofix-pytest-result.json")
CONFIG_FILE = Path(__file__).with_name("pytest.ini")
_BLOCKED = {".git", ".ssh", ".venv", "venv", "node_modules", "secrets", "credentials"}
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


class RuntimeInputError(ValueError):
    pass


def _hex(value, size):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(size) + r"}", value) is not None


def _safe_path(value):
    if type(value) is not str or not 1 <= len(value) <= 1024 or not value.endswith(".py"):
        raise RuntimeInputError()
    parts = value.split("/")
    if (
        any(part in {"", ".", ".."} or part != part.strip() or part.endswith(".") for part in parts)
        or any(c in '\\:<>"|?*' or ord(c) < 32 or ord(c) == 127 for c in value)
        or any(part.casefold() in _BLOCKED or part.casefold().startswith(".env") for part in parts)
        or any(part.split(".")[0].casefold() in _RESERVED for part in parts)
    ):
        raise RuntimeInputError()
    value.encode("utf-8")
    return value


def _binding(data):
    if type(data) is not dict or set(data) != {
        "proposal_id", "investigation_run_id", "account_id", "repository_id", "repository_full_name",
        "base_commit_sha", "tree_sha", "patch_sha256", "review_revision", "decision_id", "bound_at",
    }:
        raise RuntimeInputError()
    for key in ("proposal_id", "investigation_run_id", "decision_id"):
        if not _hex(data[key], 32): raise RuntimeInputError()
    for key in ("base_commit_sha", "tree_sha"):
        if not _hex(data[key], 40): raise RuntimeInputError()
    if not _hex(data["patch_sha256"], 64): raise RuntimeInputError()
    for key in ("account_id", "repository_id", "review_revision"):
        if type(data[key]) is not int or not 1 <= data[key] < 2**63: raise RuntimeInputError()
    name = data["repository_full_name"]
    if type(name) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}", name) is None:
        raise RuntimeInputError()
    if name.rsplit("/", 1)[1] in {".", ".."}: raise RuntimeInputError()
    if type(data["bound_at"]) is not str or len(data["bound_at"]) > 40: raise RuntimeInputError()
    instant = datetime.fromisoformat(data["bound_at"])
    if instant.tzinfo is None or instant.utcoffset() is None: raise RuntimeInputError()


def validate_payload(raw):
    """Independent strict wire/byte/path checks, without importing source."""
    try:
        if type(raw) is not bytes or len(raw) > MAX_PAYLOAD: raise RuntimeInputError()
        data = json.loads(raw)
        if type(data) is not dict or set(data) != {"protocol_version", "profile", "phase", "binding", "tests", "files"}:
            raise RuntimeInputError()
        if type(data["protocol_version"]) is not int or data["protocol_version"] != 1 or data["profile"] != PROFILE_ID or data["phase"] not in {"baseline", "patched"}:
            raise RuntimeInputError()
        _binding(data["binding"])
        if json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() != raw:
            raise RuntimeInputError()
        tests = data["tests"]
        if type(tests) is not list or not 1 <= len(tests) <= 3: raise RuntimeInputError()
        for path in tests:
            _safe_path(path)
            name = PurePosixPath(path).name
            if not (name.startswith("test_") or name.endswith("_test.py")): raise RuntimeInputError()
        if tests != sorted(set(tests)): raise RuntimeInputError()
        items = data["files"]
        if type(items) is not list or not 1 <= len(items) - len(tests) <= 20 or len(items) > 23:
            raise RuntimeInputError()
        files, aliases, total = {}, {}, 0
        for item in items:
            if type(item) is not dict or set(item) != {"path", "content", "sha256"}: raise RuntimeInputError()
            path = _safe_path(item["path"])
            if path in files or type(item["content"]) is not str: raise RuntimeInputError()
            content = item["content"]
            if any((ord(c) < 32 and c not in "\t\n\r") or ord(c) == 127 for c in content): raise RuntimeInputError()
            encoded = content.encode()
            if len(encoded) > MAX_FILE or hashlib.sha256(encoded).hexdigest() != item["sha256"]: raise RuntimeInputError()
            files[path] = content
            total += len(encoded)
            for parent in [PurePosixPath(path), *PurePosixPath(path).parents]:
                name = parent.as_posix()
                if name == ".": continue
                if name.casefold() in aliases and aliases[name.casefold()] != name: raise RuntimeInputError()
                aliases[name.casefold()] = name
        if total > MAX_SOURCE or list(files) != sorted(files) or not set(tests) <= set(files): raise RuntimeInputError()
        if any(not files[path].strip() for path in tests): raise RuntimeInputError()
        if any(parent.as_posix() in files for path in files for parent in PurePosixPath(path).parents): raise RuntimeInputError()
        return data
    except Exception:
        raise RuntimeInputError("Invalid workspace payload.") from None


def _write_workspace(items):
    # No repository process exists yet. A fresh private tmpfs must be empty.
    if WORKSPACE.is_symlink() or not WORKSPACE.is_dir() or any(WORKSPACE.iterdir()):
        raise RuntimeInputError()
    for item in items:
        path = WORKSPACE.joinpath(*item["path"].split("/"))
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags, 0o600), "wb") as output:
            output.write(item["content"].encode("utf-8"))


def _pytest_report(tests):
    """Called only by the container worker; tests never choose pytest options."""
    import pytest  # trusted installed package loaded before adding repository path
    from _pytest.subtests import SubtestReport  # provided by the pinned pytest version

    class Reporter:
        priority = {"passed": 1, "skipped": 2, "failed": 3}

        def __init__(self):
            self.counts = {"collected": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
            self.fingerprint = hashlib.sha256(b"[]").hexdigest()
            self.outcomes = {}

        def pytest_collection_modifyitems(self, session, config, items):
            if len(items) > MAX_TEST_CASES: pytest.exit("Test-case limit exceeded.", returncode=4)
            self.counts["collected"] = len(items)
            self.fingerprint = hashlib.sha256(json.dumps(sorted(item.nodeid for item in items), ensure_ascii=False).encode()).hexdigest()

        def pytest_collectreport(self, report):
            if report.failed: self.counts["errors"] += 1

        def pytest_runtest_logreport(self, report):
            # Subtest reports share the parent node ID and are not collected
            # items. Preserve any failure or skip without adding another test.
            if hasattr(report, "wasxfail") or report.skipped: outcome = "skipped"
            elif report.when == "call":
                if report.failed: outcome = "failed"
                elif report.passed and not isinstance(report, SubtestReport): outcome = "passed"
                else: return
            else:
                if report.failed: self.counts["errors"] += 1
                return
            previous = self.outcomes.get(report.nodeid)
            if previous is None or self.priority[outcome] > self.priority[previous]:
                self.outcomes[report.nodeid] = outcome

    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONPATH", "PYTHONSTARTUP"):
        os.environ.pop(name, None)
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    sys.path.insert(0, str(WORKSPACE))
    reporter = Reporter()
    args = ["-q", "--tb=no", "--capture=no", "--disable-warnings", "--noconftest", "--disable-plugin-autoload",
            "-p", "no:cacheprovider", "--import-mode=importlib", "--rootdir=" + str(WORKSPACE),
            "-c", str(CONFIG_FILE), *[str(WORKSPACE / path) for path in tests]]
    code = int(pytest.main(args, plugins=[reporter]))
    for outcome in reporter.outcomes.values():
        reporter.counts[outcome] += 1
    return {**reporter.counts, "nodeids_sha256": reporter.fingerprint, "exit_code": code}


def _read_worker_report():
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(RESULT_FILE, flags), "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 16384: raise RuntimeInputError()
        data = json.loads(source.read(16385))
    if type(data) is not dict or set(data) != {"collected", "passed", "failed", "skipped", "errors", "nodeids_sha256", "exit_code"}:
        raise RuntimeInputError()
    for key in ("collected", "passed", "failed", "skipped", "errors", "exit_code"):
        if type(data[key]) is not int or not 0 <= data[key] <= 1000: raise RuntimeInputError()
    if data["collected"] > MAX_TEST_CASES or data["exit_code"] not in {0, 1} or not _hex(data["nodeids_sha256"], 64): raise RuntimeInputError()
    if data["passed"] + data["failed"] + data["skipped"] > data["collected"]: raise RuntimeInputError()
    if data["errors"] == 0 and data["passed"] + data["failed"] + data["skipped"] != data["collected"]: raise RuntimeInputError()
    if data["exit_code"] == 0 and (data["failed"] or data["errors"]): raise RuntimeInputError()
    if data["exit_code"] == 1 and not (data["failed"] or data["errors"]): raise RuntimeInputError()
    return data


def _execute_tests(tests):
    """Separate worker, hard byte counter, deadline and no raw-log retention."""
    overflow, writer_failed = Event(), Event()
    count = [0]
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--pytest-worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=False, start_new_session=True,
    )
    def drain():
        try:
            while True:
                chunk = process.stdout.read(4096)
                if not chunk: break
                count[0] += len(chunk)
                if count[0] > MAX_WORKER_OUTPUT: overflow.set()
        except OSError:
            writer_failed.set()
    def supply():
        try:
            process.stdin.write(json.dumps({"tests": tests}).encode())
            process.stdin.close()
        except (OSError, ValueError):
            writer_failed.set()
    reader = Thread(target=drain, daemon=True)
    writer = Thread(target=supply, daemon=True)
    reader.start(); writer.start()
    deadline = time.monotonic() + TEST_TIMEOUT_SECONDS
    reason = None
    try:
        while process.poll() is None:
            if overflow.is_set(): reason = "output_limit"; break
            if writer_failed.is_set(): reason = "worker_io_error"; break
            if time.monotonic() >= deadline: reason = "timeout"; break
            time.sleep(0.02)
    finally:
        if process.poll() is None:
            try: os.killpg(process.pid, signal.SIGKILL)
            except OSError: process.kill()
        process.wait(timeout=2)
        reader.join(timeout=1); writer.join(timeout=1)
        if not reader.is_alive(): process.stdout.close()
    if overflow.is_set(): reason = "output_limit"
    if reason is not None: raise RuntimeInputError(reason)
    if reader.is_alive() or writer.is_alive() or writer_failed.is_set(): raise RuntimeInputError("worker_io_error")
    result = _read_worker_report()
    if result["exit_code"] != process.returncode: raise RuntimeInputError("worker_exit_mismatch")
    return result, count[0]


def _read(path: str) -> str:
    return Path(path).read_text(encoding="ascii")


def _scratch_writable() -> bool:
    for directory in ("/tmp", "/workspace"):
        path = Path(directory) / f".repofix-runtime-check-{os.getpid()}"
        try:
            with path.open("xb") as target:
                target.write(b"ok")
        finally:
            path.unlink(missing_ok=True)
    return True


def inspect_runtime() -> dict:
    checks = {
        "non_root": False, "isolated_python": False, "no_capabilities": False,
        "no_new_privileges": False, "seccomp_enabled": False,
        "root_read_only": False, "network_disabled": False, "memory_limited": False,
        "swap_disabled": False, "cpu_limited": False, "pids_limited": False,
        "tmpfs_restricted": False, "scratch_writable": False, "dependencies_ready": False,
    }
    try:
        checks["non_root"] = os.getuid() == 10001 and os.getgid() == 10001
        checks["isolated_python"] = sys.flags.isolated == 1 and sys.dont_write_bytecode
        status = dict(line.split(":", 1) for line in _read("/proc/self/status").splitlines() if ":" in line)
        checks["no_capabilities"] = int(status["CapEff"].strip(), 16) == 0
        checks["no_new_privileges"] = status["NoNewPrivs"].strip() == "1"
        checks["seccomp_enabled"] = status["Seccomp"].strip() == "2"
        mounts = {}
        for line in _read("/proc/self/mountinfo").splitlines():
            first, second = line.split(" - ", 1)
            fields = first.split()
            mounts[fields[4]] = (set(fields[5].split(",")), second.split()[0])
        checks["root_read_only"] = "ro" in mounts["/"][0]
        checks["tmpfs_restricted"] = all(
            mounts[path][1] == "tmpfs" and {"rw", "noexec", "nosuid", "nodev"} <= mounts[path][0]
            for path in ("/tmp", "/workspace")
        )
        interfaces = {line.split(":", 1)[0].strip() for line in _read("/proc/net/dev").splitlines() if ":" in line}
        checks["network_disabled"] = interfaces == {"lo"}
        checks["memory_limited"] = _read("/sys/fs/cgroup/memory.max").strip() == "536870912"
        checks["swap_disabled"] = _read("/sys/fs/cgroup/memory.swap.max").strip() == "0"
        checks["pids_limited"] = _read("/sys/fs/cgroup/pids.max").strip() == "64"
        quota, period = map(int, _read("/sys/fs/cgroup/cpu.max").split())
        checks["cpu_limited"] = 0 < quota <= period
        checks["dependencies_ready"] = all(version(name) == expected for name, expected in EXPECTED.items())
        # Never probe writable directories unless container-specific guards pass.
        if all(value for key, value in checks.items() if key != "scratch_writable"):
            checks["scratch_writable"] = _scratch_writable()
    except (OSError, KeyError, ValueError, IndexError, PackageNotFoundError):
        pass
    return {
        "status": "ready" if all(checks.values()) else "unavailable",
        "profile": PROFILE_ID, "runtime_revision": RUNTIME_REVISION, "checks": checks,
    }


def run_phase(raw):
    if inspect_runtime()["status"] != "ready":
        return {"status": "error", "reason": "runtime_not_ready"}
    try:
        data = validate_payload(raw)
    except RuntimeInputError:
        return {"status": "error", "reason": "payload_invalid"}
    try:
        _write_workspace(data["files"])
        result, output_bytes = _execute_tests(data["tests"])
    except Exception as error:
        reason = str(error) if isinstance(error, RuntimeInputError) and str(error) in {
            "output_limit", "worker_io_error", "timeout", "worker_exit_mismatch",
        } else "phase_failed"
        return {"status": "error", "reason": reason}
    return {
        "status": "completed", "profile": PROFILE_ID, "runtime_revision": RUNTIME_REVISION,
        "protocol_version": 1, "phase": data["phase"],
        "payload_sha256": hashlib.sha256(raw).hexdigest(), "result": result,
        "worker_output_bytes": output_bytes,
    }


def _worker():
    if inspect_runtime()["status"] != "ready": return 2
    try:
        raw = sys.stdin.buffer.read(16385)
        if len(raw) > 16384: return 2
        data = json.loads(raw)
        if type(data) is not dict or set(data) != {"tests"} or type(data["tests"]) is not list or not 1 <= len(data["tests"]) <= 3:
            return 2
        tests = data["tests"]
        if tests != sorted(set(tests)): return 2
        for path in tests:
            _safe_path(path)
            name = PurePosixPath(path).name
            if not (name.startswith("test_") or name.endswith("_test.py")): return 2
        result = _pytest_report(tests)
        encoded = json.dumps(result, sort_keys=True).encode()
        if len(encoded) > 16384: return 2
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(RESULT_FILE, flags, 0o600), "wb") as report:
            report.write(encoded)
        return result["exit_code"]
    except Exception:
        return 2


def main() -> int:
    mode = sys.argv[1:]
    if mode == ["--pytest-worker"]:
        return _worker()
    if mode == ["--self-check"]:
        result = inspect_runtime()
    elif mode == ["--run-phase"]:
        # Canonical JSON has escaped source newlines. One LF frames the packet
        # without relying on Docker's interactive-stdin EOF behavior.
        raw = sys.stdin.buffer.readline(MAX_PAYLOAD + 2)
        result = run_phase(raw[:-1]) if raw.endswith(b"\n") else {"status": "error", "reason": "payload_invalid"}
    else:
        print(json.dumps({"status": "unavailable", "reason": "unsupported_mode"}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"ready", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
