"""Offline HTTP contracts, limits and HTTP-to-SDK integration for /run."""

import asyncio
import os
import base64
import hashlib
import importlib.util
import json
from dataclasses import replace
from threading import Event
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import investigations as api
from app.config import Settings
from app.github_client import GitHubClientError
from app.github_selection import RepositorySelectionError
from app.investigation_report import InvestigationReport
from app.investigation_store import InvestigationStore
from app.investigation_runner import (
    InvestigationRunError,
    InvestigationRunResult,
    run_investigation,
)
from app.main import create_app
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider
from agents import Model, ModelProvider, ModelResponse, Usage
from openai import APIStatusError, APITimeoutError
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from app.github_client import RepositoryMetadata, RepositorySnapshot
from app.github_oauth import GitHubIdentity
from app.github_repositories import PublicRepository
from app.github_tree import RepositoryTree
from app.investigation_context import InvestigationContext
from app.investigation_input import InvestigationInput

openai_http = (
    importlib.import_module("httpx2") if importlib.util.find_spec("httpx2") is not None else httpx
)

BASE = "http://127.0.0.1:8000"
PATH = "/api/investigations/run"
HEADERS = {"Origin": BASE, "X-RepoFix-CSRF": "1"}
BODY = {
    "repository_id": 5,
    "repository_name": "example",
    "issue_description": "  Synthetic issue description.  ",
}


def make_fixture(content=None):
    text = (
        content if content is not None else "\n".join(f"line {number}" for number in range(1, 101))
    )
    data = text.encode("utf-8")
    blob_sha = hashlib.sha1(
        f"blob {len(data)}\0".encode("ascii") + data,
        usedforsecurity=False,
    ).hexdigest()
    identity = GitHubIdentity(id=123, login="demo-user", type="User")
    repository = PublicRepository(
        id=5,
        name="example",
        full_name="demo-user/example",
        owner=identity,
        private=False,
        visibility="public",
        default_branch="main",
        fork=False,
        archived=False,
        disabled=False,
    )
    snapshot = RepositorySnapshot(
        repository=RepositoryMetadata(**repository.model_dump()),
        commit_sha="a" * 40,
    )
    tree = RepositoryTree(
        repository_full_name=repository.full_name,
        commit_sha=snapshot.commit_sha,
        tree_sha="b" * 40,
        entries=[
            {"path": path, "type": "blob", "mode": mode, "sha": blob_sha, "size": len(data)}
            for path, mode in (
                ("app/auth.py", "100644"),
                (".env", "100644"),
                ("link.py", "120000"),
            )
        ],
    )
    context = InvestigationContext(
        identity=identity,
        request=InvestigationInput(
            repository_id=5,
            repository_name="example",
            issue_description="Synthetic grounding test.",
        ),
        repository=repository,
        snapshot=snapshot,
        tree=tree,
    )
    requests = []

    def handler(request):
        assert request.method == "GET"
        assert request.url.host == "api.github.com"
        assert request.url.path == f"/repos/demo-user/example/git/blobs/{blob_sha}"
        requests.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "sha": blob_sha,
                "size": len(data),
                "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii"),
            },
        )

    return context, httpx.MockTransport(handler), requests


def tool(name="read_file", arguments=None, call_id="call-1"):
    return ResponseFunctionToolCall(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=json.dumps(arguments or {"path": "app/auth.py", "start_line": 1}),
    )


def final(quote="line 10", insufficient=False):
    report = dict(
        status="likely_cause",
        issue_summary="Synthetic issue.",
        issue_category="logic",
        root_cause="Synthetic hypothesis.",
        evidence=[
            dict(
                path="app/auth.py",
                start_line=10,
                end_line=10,
                quote=quote,
                explanation="Synthetic evidence.",
            )
        ],
        next_steps=["Reproduce."],
        limitations=["No tests executed."],
    )
    if insufficient:
        report.update(status="insufficient_evidence", root_cause=None, evidence=[])
    return ResponseOutputMessage(
        id="msg-test",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", annotations=[], text=json.dumps(report))],
    )


class FakeModel(Model):
    def __init__(self, outputs, delay=0):
        self.outputs = iter(outputs)
        self.calls = []
        self.delay = delay

    async def get_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
        **kwargs,
    ):
        self.calls.append((system_instructions, input, model_settings, tools, tracing))
        await asyncio.sleep(self.delay)
        output = next(self.outputs)
        if isinstance(output, BaseException):
            raise output
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    async def stream_response(self, *args, **kwargs):
        raise AssertionError("Unexpected streaming request.")
        yield


class Provider(ModelProvider):
    def __init__(self, model):
        self.model = model

    def get_model(self, name):
        assert name == "offline-model"
        return self.model


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("REPOFIX_"):
            monkeypatch.delenv(key, raising=False)

    def reject(*args, **kwargs):
        pytest.fail("Live network is forbidden.")

    for package in {httpx, openai_http}:
        monkeypatch.setattr(package.HTTPTransport, "handle_request", reject)
        monkeypatch.setattr(package.AsyncHTTPTransport, "handle_async_request", reject)
    monkeypatch.setattr(tracing_setup, "GLOBAL_TRACE_PROVIDER", DefaultTraceProvider())


def make_app(monkeypatch, ready=True):
    context, transport, requests = make_fixture()
    config = Settings(
        _env_file=None,
        environment="test",
        openai_api_key="offline-key" if ready else None,
        ai_model="offline-model" if ready else None,
    )
    app = create_app(config, history_store=InvestigationStore(":memory:"))
    token = app.state.auth_store.create_session(context.identity)
    prepare = Mock(
        side_effect=lambda identity, payload: replace(context, identity=identity, request=payload)
    )
    report = InvestigationReport(
        status="insufficient_evidence",
        issue_summary="Synthetic issue.",
        issue_category="other",
        root_cause=None,
        evidence=[],
        next_steps=["Collect evidence."],
        limitations=["Offline API test."],
    )
    result = InvestigationRunResult(
        run_id="e" * 32,
        repository_full_name=context.repository.full_name,
        commit_sha=context.snapshot.commit_sha,
        report=report,
        duration_ms=0,
        model_calls=1,
        tool_attempts=0,
        repository_stats={"tool_calls": 0, "file_fetch_attempts": 0, "downloaded_bytes": 0},
        events=(),
    )
    run = AsyncMock(return_value=result)
    monkeypatch.setattr(api, "prepare_investigation_context", prepare)
    monkeypatch.setattr(api, "run_investigation", run)
    return app, token, prepare, run, context, transport


def test_http_route_to_real_sdk_and_grounding(monkeypatch):
    app, token, prepare, run, context, transport = make_app(monkeypatch)
    model = FakeModel([[tool()], [final()]])

    async def actual(context, settings):
        return await run_investigation(
            context, settings, model_provider=Provider(model), repository_transport=transport
        )

    monkeypatch.setattr(api, "run_investigation", actual)
    with TestClient(app, base_url=BASE) as client:
        client.cookies.set("repofix_session", token)
        response = client.post(PATH, json=BODY, headers=HEADERS)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "completed" and body["report"]["status"] == "likely_cause"
        assert body["history_saved"] is True
        saved = client.get(f"/api/investigations/{body['run_id']}")
        assert saved.status_code == 200
        assert saved.json()["investigation"] == body
        assert body["model_calls"] == 2 and body["tool_attempts"] == 1
        assert body["commit_sha"] == context.snapshot.commit_sha
        assert body["repository_stats"]["tool_calls"] == 1
        assert body["events"][-2]["kind"] == "grounding_validated"
        assert response.headers["cache-control"] == "no-store"
        assert token not in response.text
        assert prepare.call_args.args[0].id == context.identity.id
        assert prepare.call_args.args[1].issue_description == "Synthetic issue description."


def test_missing_session_stops_work(monkeypatch):
    app, token, prepare, run, *_ = make_app(monkeypatch)
    with TestClient(app, base_url=BASE) as client:
        response = client.post(PATH, json=BODY, headers=HEADERS)
        assert response.status_code == 401
        prepare.assert_not_called()
        run.assert_not_called()


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Origin": BASE},
        {"X-RepoFix-CSRF": "1"},
        {"Origin": "https://evil.example", "X-RepoFix-CSRF": "1"},
        {"Origin": BASE, "X-RepoFix-CSRF": "0"},
        [("Origin", BASE), ("Origin", BASE), ("X-RepoFix-CSRF", "1")],
        [("Origin", BASE), ("X-RepoFix-CSRF", "1"), ("X-RepoFix-CSRF", "1")],
    ],
)
def test_origin_and_csrf_rejections(monkeypatch, headers):
    app, token, prepare, run, *_ = make_app(monkeypatch)
    with TestClient(app, base_url=BASE) as client:
        client.cookies.set("repofix_session", token)
        response = client.post(PATH, json=BODY, headers=headers)
        assert response.status_code == 403
        assert response.headers["cache-control"] == "no-store"
        prepare.assert_not_called()
        run.assert_not_called()


@pytest.mark.parametrize(
    "extra",
    [
        {"owner": "other"},
        {"repository_url": "https://evil.example"},
        {"commit_sha": "a" * 40},
        {"model": "other"},
        {"approved": True},
    ],
)
def test_client_cannot_override_scope_or_settings(monkeypatch, extra):
    app, token, prepare, run, *_ = make_app(monkeypatch)
    with TestClient(app, base_url=BASE) as client:
        client.cookies.set("repofix_session", token)
        response = client.post(PATH, json={**BODY, **extra}, headers=HEADERS)
        assert response.status_code == 422
        prepare.assert_not_called()
        run.assert_not_called()


def test_ai_configuration_checked_before_preparation(monkeypatch):
    app, token, prepare, run, *_ = make_app(monkeypatch, ready=False)
    with TestClient(app, base_url=BASE) as client:
        client.cookies.set("repofix_session", token)
        assert client.post(PATH, json=BODY, headers=HEADERS).status_code == 503
        prepare.assert_not_called()
        run.assert_not_called()


@pytest.mark.parametrize(
    "code,status,retry_after",
    [
        ("not_configured", 503, None),
        ("timeout", 504, None),
        ("turn_limit", 409, None),
        ("tool_limit", 409, None),
        ("event_limit", 409, None),
        ("ungrounded_report", 502, None),
        ("model_error", 502, None),
        ("unknown", 502, None),
        ("provider_daily_quota", 429, 28217),
        ("provider_rate_limit", 429, 51),
        ("provider_limit", 429, None),
        ("provider_unavailable", 503, 5),
        ("provider_timeout", 504, None),
    ],
)
def test_controlled_failure_mapping_and_redaction(monkeypatch, code, status, retry_after):
    app, token, prepare, run, *_ = make_app(monkeypatch)
    run.side_effect = InvestigationRunError(
        code, "secret-provider-payload", retry_after=retry_after,
    )
    with TestClient(app, base_url=BASE) as client:
        client.cookies.set("repofix_session", token)
        response = client.post(PATH, json=BODY, headers=HEADERS)
        assert response.status_code == status
        assert response.headers.get("Retry-After") == (
            str(retry_after) if retry_after is not None else None
        )
        assert response.headers["Cache-Control"] == "no-store"
        assert "secret-provider-payload" not in response.text
        # A handled failure must release the slot so another attempt can start.
        assert client.post(PATH, json=BODY, headers=HEADERS).status_code == status


@pytest.mark.parametrize("raw", [True, 0, -1, 86401, "50", "x\r\nBad: 1"])
def test_invalid_provider_wait_cannot_enter_http_headers(monkeypatch, raw):
    app, token, prepare, run, *_ = make_app(monkeypatch)
    failure = InvestigationRunError("provider_daily_quota", "Safe message.")
    # Simulate internal metadata corruption after constructor validation.
    failure.retry_after = raw
    run.side_effect = failure
    with TestClient(app, base_url=BASE) as client:
        client.cookies.set("repofix_session", token)
        response = client.post(PATH, json=BODY, headers=HEADERS)
    assert response.status_code == 429
    assert "Retry-After" not in response.headers


@pytest.mark.parametrize("code", ["model_error", "unknown-private-code", "timeout", "provider_timeout"])
def test_wait_hint_only_forwarded_for_eligible_error_codes(monkeypatch, code):
    app, token, prepare, run, *_ = make_app(monkeypatch)
    run.side_effect = InvestigationRunError(code, "secret-provider-payload", retry_after=50)
    with TestClient(app, base_url=BASE) as client:
        client.cookies.set("repofix_session", token)
        response = client.post(PATH, json=BODY, headers=HEADERS)
    assert response.status_code == (504 if code in {"timeout", "provider_timeout"} else 502)
    assert "Retry-After" not in response.headers
    assert "secret-provider-payload" not in response.text
    assert "unknown-private-code" not in response.text


def sdk_provider_error(kind):
    """Synthetic SDK errors shaped like observed Google responses."""
    request = httpx.Request("POST", "https://provider.invalid")
    if kind == "timeout":
        return APITimeoutError(request=request)

    status = 503 if kind == "unavailable" else 429
    details = []
    if kind in {"daily", "minute"}:
        quota_id = (
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
            if kind == "daily"
            else "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
        )
        details = [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": quota_id}],
            },
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": "28217s" if kind == "daily" else "50.1s",
            },
        ]
    return APIStatusError(
        "secret-provider-payload",
        response=httpx.Response(status, request=request),
        body=[{"error": {"message": "secret-provider-payload", "details": details}}],
    )


@pytest.mark.parametrize("kind,status,wait,phrase", [
    ("daily", 429, "28217", "daily quota"),
    ("minute", 429, "51", "rate limit"),
    ("unknown-limit", 429, None, "quota or rate limit"),
    ("unavailable", 503, None, "temporarily unavailable"),
    ("timeout", 504, None, "provider request timed out"),
])
def test_sdk_error_through_actual_runner_to_http(monkeypatch, kind, status, wait, phrase):
    app, token, prepare, stub, context, transport = make_app(monkeypatch)
    model = FakeModel([sdk_provider_error(kind), [final(insufficient=True)]])

    async def actual(prepared, settings):
        return await run_investigation(
            prepared, settings, model_provider=Provider(model), repository_transport=transport,
        )

    monkeypatch.setattr(api, "run_investigation", actual)
    with TestClient(app, base_url=BASE) as client:
        client.cookies.set("repofix_session", token)
        response = client.post(PATH, json=BODY, headers=HEADERS)
        assert response.status_code == status, response.text
        assert response.headers.get("Retry-After") == wait
        assert phrase in response.json()["detail"]
        assert "secret-provider-payload" not in response.text
        assert "provider.invalid" not in response.text

        # After the actual runner fails, a new run can complete normally.
        recovered = client.post(PATH, json=BODY, headers=HEADERS)
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["report"]["status"] == "insufficient_evidence"


def test_openapi_documents_provider_errors_and_wait_header(monkeypatch):
    app, *_ = make_app(monkeypatch)
    responses = app.openapi()["paths"][PATH]["post"]["responses"]
    for status in ["429", "503"]:
        schema = responses[status]["headers"]["Retry-After"]["schema"]
        assert schema["minimum"] == 1
        assert schema["maximum"] == 86400
    assert "provider request" in responses["504"]["description"]


@pytest.mark.parametrize(
    "failure,status",
    [
        (RepositorySelectionError("private"), 400),
        (GitHubClientError("private"), 502),
        (RuntimeError("private"), 500),
    ],
)
def test_preparation_error_is_sanitized(monkeypatch, failure, status):
    app, token, prepare, run, *_ = make_app(monkeypatch)
    prepare.side_effect = failure
    with TestClient(app, base_url=BASE) as client:
        client.cookies.set("repofix_session", token)
        response = client.post(PATH, json=BODY, headers=HEADERS)
        assert response.status_code == status and "private" not in response.text
        run.assert_not_called()
        assert client.post(PATH, json=BODY, headers=HEADERS).status_code == status


def test_logout_during_preparation_prevents_model_call(monkeypatch):
    app, token, prepare, run, context, *_ = make_app(monkeypatch)

    def revoked(identity, payload):
        app.state.auth_store.revoke_session(token)
        return replace(context, request=payload)

    prepare.side_effect = revoked
    with TestClient(app, base_url=BASE) as client:
        client.cookies.set("repofix_session", token)
        assert client.post(PATH, json=BODY, headers=HEADERS).status_code == 401
        run.assert_not_called()


def test_returned_provenance_must_match_context(monkeypatch):
    app, token, prepare, run, *_ = make_app(monkeypatch)
    run.return_value = replace(run.return_value, commit_sha="d" * 40)
    with TestClient(app, base_url=BASE) as client:
        client.cookies.set("repofix_session", token)
        assert client.post(PATH, json=BODY, headers=HEADERS).status_code == 502


@pytest.mark.parametrize("preparation_fails", [False, True])
def test_attempt_throttle_expires_and_does_not_rerun_model(monkeypatch, preparation_fails):
    app, token, prepare, run, *_ = make_app(monkeypatch)
    if preparation_fails:
        prepare.side_effect = RepositorySelectionError("synthetic failure")
    expected_status = 400 if preparation_fails else 200
    now = [0.0]
    monkeypatch.setattr(api, "monotonic", lambda: now[0])
    with TestClient(app, base_url=BASE) as client:
        client.cookies.set("repofix_session", token)
        for _ in range(api.MAX_RUN_ATTEMPTS_PER_HOUR):
            assert client.post(PATH, json=BODY, headers=HEADERS).status_code == expected_status
        response = client.post(PATH, json=BODY, headers=HEADERS)
        assert response.status_code == 429 and response.headers["retry-after"] == "3600"
        assert prepare.call_count == api.MAX_RUN_ATTEMPTS_PER_HOUR
        assert run.await_count == (0 if preparation_fails else api.MAX_RUN_ATTEMPTS_PER_HOUR)
        now[0] = api.RUN_WINDOW_SECONDS
        assert client.post(PATH, json=BODY, headers=HEADERS).status_code == expected_status


def test_busy_run_is_rejected_without_queueing(monkeypatch):
    app, token, prepare, run, *_ = make_app(monkeypatch)

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def wait(context, settings):
            entered.set()
            await release.wait()
            return run.return_value

        monkeypatch.setattr(api, "run_investigation", wait)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=BASE,
            cookies={"repofix_session": token},
        ) as client:
            first = asyncio.create_task(client.post(PATH, json=BODY, headers=HEADERS))
            await asyncio.wait_for(entered.wait(), 2)
            second = await client.post(PATH, json=BODY, headers=HEADERS)
            assert second.status_code == 429
            assert prepare.call_count == 1
            release.set()
            assert (await first).status_code == 200

    asyncio.run(scenario())


def test_preparation_timeout_retains_slot_until_worker_finishes(monkeypatch):
    app, token, prepare, run, context, *_ = make_app(monkeypatch)
    release = Event()
    finished = Event()

    def wait(identity, payload):
        try:
            assert release.wait(2)
            return replace(context, request=payload)
        finally:
            finished.set()

    prepare.side_effect = wait
    # A short deadline is injected only for this test.
    original_deadline = api.PREPARATION_TIMEOUT_SECONDS
    monkeypatch.setattr(api, "PREPARATION_TIMEOUT_SECONDS", 0.02)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=BASE,
            cookies={"repofix_session": token},
        ) as client:
            try:
                first = await client.post(PATH, json=BODY, headers=HEADERS)
                assert first.status_code == 504
                assert (await client.post(PATH, json=BODY, headers=HEADERS)).status_code == 429
                run.assert_not_called()
            finally:
                release.set()
            assert await asyncio.to_thread(finished.wait, 2)
            monkeypatch.setattr(api, "PREPARATION_TIMEOUT_SECONDS", original_deadline)
            for _ in range(100):
                response = await client.post(PATH, json=BODY, headers=HEADERS)
                if response.status_code != 429:
                    break
                await asyncio.sleep(0.001)
            assert response.status_code == 200

    asyncio.run(scenario())
