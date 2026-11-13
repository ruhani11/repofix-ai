"""Operator diagnostic using only fixed trusted fixture code, never a proposal.

No GitHub, model, private settings or admission DB is read. This diagnostic's
capture_output helper must not be reused as the untrusted target-job runner.
"""

from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from uuid import uuid4

from app.sandbox_profile import (
    PROFILE_ID, RUNTIME_REVISION, SandboxProfileError, _health_command,
    build_sandbox_phase_create_args, check_sandbox_runtime,
)


_SOURCE = '''from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/orders", status_code=201)
def create_order(body: dict):
    if body["quantity"] <= 1:
        raise HTTPException(400, "Quantity must be positive.")
    return {"status": "accepted"}
'''
_TEST = '''from fastapi.testclient import TestClient
from orders_api import app

def test_quantity_one_is_accepted():
    response = TestClient(app).post("/orders", json={"quantity": 1})
    assert response.status_code == 201
'''


def _diagnostic_payloads():
    # Synthetic provenance only. These IDs never enter application history and
    # do not pretend to be a saved proposal or a user's approval.
    binding = {
        "proposal_id": "1" * 32, "investigation_run_id": "2" * 32,
        "account_id": 1, "repository_id": 1, "repository_full_name": "repofix/runtime-self-test",
        "base_commit_sha": "a" * 40, "tree_sha": "b" * 40,
        "patch_sha256": "c" * 64, "review_revision": 1, "decision_id": "3" * 32,
        "bound_at": datetime.now(timezone.utc).isoformat(),
    }
    results = {}
    for phase in ("baseline", "patched"):
        source = _SOURCE if phase == "baseline" else _SOURCE.replace('<= 1:', '<= 0:')
        files = {"orders_api.py": source, "test_orders_api.py": _TEST}
        data = {
            "protocol_version": 1, "profile": PROFILE_ID, "phase": phase, "binding": binding,
            "tests": ["test_orders_api.py"], "files": [
                {"path": path, "content": content, "sha256": hashlib.sha256(content.encode()).hexdigest()}
                for path, content in sorted(files.items())
            ],
        }
        results[phase] = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return results


def _diagnostic_phase(docker, image_id, run_id, phase, payload):
    args = build_sandbox_phase_create_args(image_id, run_id, phase)
    name = args[args.index("--name") + 1]
    attempted = False
    try:
        attempted = True
        _health_command(docker, args, time.monotonic() + 30)
        try:
            response = subprocess.run(
                [docker, "start", "--attach", "--interactive", name], input=payload + b"\n",
                capture_output=True, timeout=60, check=False, shell=False,
            )
        except subprocess.TimeoutExpired:
            raise SandboxProfileError("timeout") from None
        except OSError:
            raise SandboxProfileError("docker_unavailable") from None
        if response.returncode != 0 or len(response.stdout) > 16384:
            raise SandboxProfileError("test_phase_failed")
        try:
            result = json.loads(response.stdout)
            metrics = result["result"]
            if (
                result["status"] != "completed" or result["profile"] != PROFILE_ID
                or type(result["runtime_revision"]) is not int or result["runtime_revision"] != RUNTIME_REVISION
                or type(result["protocol_version"]) is not int or result["protocol_version"] != 1
                or result["phase"] != phase or result["payload_sha256"] != hashlib.sha256(payload).hexdigest()
                or type(metrics) is not dict or set(metrics) != {"collected", "passed", "failed", "skipped", "errors", "nodeids_sha256", "exit_code"}
            ):
                raise ValueError()
            for key in ("collected", "passed", "failed", "skipped", "errors", "exit_code"):
                if type(metrics[key]) is not int or not 0 <= metrics[key] <= 1000: raise ValueError()
            if type(metrics["nodeids_sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", metrics["nodeids_sha256"]) is None:
                raise ValueError()
            return metrics
        except (KeyError, TypeError, ValueError):
            raise SandboxProfileError("invalid_test_result") from None
    finally:
        if attempted:
            try:
                _health_command(docker, ("rm", "--force", name), time.monotonic() + 10)
            except SandboxProfileError:
                raise SandboxProfileError("cleanup_failed") from None


def check_test_entrypoint():
    """No arguments, source paths or target proposal input are accepted."""
    ready = check_sandbox_runtime()
    docker = shutil.which("docker")
    if docker is None: raise SandboxProfileError("docker_unavailable")
    run_id = uuid4().hex
    phases = {
        phase: _diagnostic_phase(docker, ready["image_id"], run_id, phase, data)
        for phase, data in _diagnostic_payloads().items()
    }
    before, after = phases["baseline"], phases["patched"]
    if not (
        before["exit_code"] == 1 and before["collected"] == before["failed"] == 1
        and before["passed"] == before["skipped"] == before["errors"] == 0
        and after["exit_code"] == 0 and after["collected"] == after["passed"] == 1
        and after["failed"] == after["skipped"] == after["errors"] == 0
        and before["nodeids_sha256"] == after["nodeids_sha256"]
    ):
        raise SandboxProfileError("unexpected_diagnostic_outcome")
    return {
        "status": "runtime_self_test_passed", "diagnostic_only": True,
        "profile": PROFILE_ID, "runtime_revision": RUNTIME_REVISION,
        "image_id": ready["image_id"], "same_collected_tests": True, **phases,
    }


def main():
    try:
        if len(sys.argv) != 1: raise SandboxProfileError("diagnostic_takes_no_arguments")
        result = check_test_entrypoint()
    except SandboxProfileError as error:
        print(json.dumps({"status": "unavailable", "diagnostic_only": True, "reason": error.code}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
