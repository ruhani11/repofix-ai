"""Offline Gemini HTTP/SDK regression tests; no real keys, network or waits."""

import asyncio
import importlib
import importlib.util
import json
import logging
import os
from collections import deque
from types import SimpleNamespace

import httpx
import pytest
from agents import (
    Agent, AgentOutputSchema, ModelSettings, ModelTracing, RunConfig, Runner,
    function_tool,
)
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider
from openai import APIStatusError

from app import gemini_provider as module
from app.config import Settings
from app.investigation_report import InvestigationReport


# Match the installed OpenAI client's HTTP implementation without requiring
# HTTPX2 in older environments. Neither transport can access the network here.
sdk_http = (
    importlib.import_module("httpx2")
    if importlib.util.find_spec("httpx2") is not None
    else httpx
)
MODEL = "gemini-offline-fixture"
PRIVATE_KEY = "dummy-private-fixture-key"
REPORT = {
    "status": "insufficient_evidence",
    "issue_summary": "Synthetic provider integration scenario.",
    "issue_category": "other",
    "root_cause": None,
    "evidence": [],
    "next_steps": ["Investigate a real issue separately."],
    "limitations": ["Only synthetic fixture data was available."],
}


def settings(**overrides):
    values = dict(
        _env_file=None, ai_provider="gemini", ai_model=MODEL,
        gemini_api_key=PRIVATE_KEY, ai_timeout_seconds=180,
    )
    values.update(overrides)
    return Settings(**values)


def quota_error(delay="50s", *, daily=False):
    quota = (
        "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        if daily else "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
    )
    details = [{
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [{"quotaId": quota}],
    }]
    if delay is not None:
        details.append({
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": delay,
        })
    return [{"error": {
        "code": 429, "message": "PRIVATE_PROVIDER_ERROR_BODY",
        "status": "RESOURCE_EXHAUSTED", "details": details,
    }}]


def completion(*, tool_call=None):
    message = {"role": "assistant", "content": json.dumps(REPORT)}
    if tool_call is not None:
        message.update(content=None, tool_calls=[tool_call])
    return {
        "id": "fixture-completion", "object": "chat.completion",
        "created": 0, "model": MODEL,
        "choices": [{
            "index": 0, "message": message,
            "finish_reason": "tool_calls" if tool_call else "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture(autouse=True)
def isolate_environment_network_and_tracing(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("REPOFIX_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        tracing_setup, "GLOBAL_TRACE_PROVIDER", DefaultTraceProvider(),
    )

    def reject(*args, **kwargs):
        pytest.fail("A Gemini test attempted real network access.")

    for package in {httpx, sdk_http}:
        monkeypatch.setattr(package.HTTPTransport, "handle_request", reject)
        monkeypatch.setattr(package.AsyncHTTPTransport, "handle_async_request", reject)
    monkeypatch.setattr(module.random, "uniform", lambda low, high: 0.1)


@pytest.fixture
def provider_http(monkeypatch):
    fixture = SimpleNamespace(
        replies=deque(), requests=[], clients=[], sleeps=[], client_options=[],
    )

    def handle(request):
        assert request.method == "POST"
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.url.path == "/v1beta/openai/chat/completions"
        fixture.requests.append(json.loads(request.content))
        assert fixture.replies, "Unexpected extra model HTTP request."
        status, body, headers = fixture.replies.popleft()
        return sdk_http.Response(status, json=body, headers=headers)

    original_client = module.DefaultAsyncHttpxClient

    def make_client(**kwargs):
        fixture.client_options.append(kwargs)
        client = original_client(transport=sdk_http.MockTransport(handle), **kwargs)
        fixture.clients.append(client)
        return client

    async def no_wait(seconds):
        fixture.sleeps.append(seconds)

    monkeypatch.setattr(module, "DefaultAsyncHttpxClient", make_client)
    monkeypatch.setattr(module.asyncio, "sleep", no_wait)
    yield fixture
    assert all(client.is_closed for client in fixture.clients)


async def probe(provider):
    return await provider.get_model(MODEL).get_response(
        system_instructions="Synthetic diagnostic prompt.",
        input="No real repository was inspected.",
        model_settings=ModelSettings(store=None, max_tokens=128),
        tools=[], output_schema=AgentOutputSchema(InvestigationReport),
        handoffs=[], tracing=ModelTracing.DISABLED,
    )


@pytest.mark.parametrize("status,body,headers,wait", [
    (429, quota_error(), {}, 50.1),
    (429, quota_error()[0], {}, 50.1),
    (429, {"error": {"message": "wait"}}, {"Retry-After": "3"}, 3.1),
    (429, quota_error("5s"), {"Retry-After": "3"}, 5.1),
    (429, quota_error("60s"), {}, 60.1),
    (429, quota_error("120s"), {}, None),
    (429, quota_error("28217s", daily=True), {}, None),
    (429, quota_error("5s", daily=True), {}, None),
    (429, quota_error(None), {}, None),
    (429, quota_error("invalid"), {}, None),
    (429, quota_error(), {"Retry-After": "nan"}, None),
    (429, quota_error(), {"Retry-After": "-1"}, None),
    (429, quota_error(), {"Retry-After": "Thu, 01 Jan 2099 00:00:00 GMT"}, None),
    (400, quota_error(), {}, None),
    (401, quota_error(), {}, None),
    (403, quota_error(), {}, None),
    (503, {"error": {"message": "unavailable"}}, {}, 2.1),
    (503, quota_error("5s"), {}, 5.1),
])
def test_http_retry_policy(provider_http, status, body, headers, wait):
    provider_http.replies.append((status, body, headers))
    if wait is not None:
        provider_http.replies.append((200, completion(), {}))

    async def run():
        async with module.open_gemini_provider(settings()) as provider:
            if wait is None:
                with pytest.raises(APIStatusError) as caught:
                    await probe(provider)
                assert caught.value.status_code == status
            else:
                await probe(provider)

    asyncio.run(run())
    assert provider_http.sleeps == ([] if wait is None else [wait])
    assert len(provider_http.requests) == (1 if wait is None else 2)
    if wait is not None:
        assert provider_http.requests[0] == provider_http.requests[1]


def test_retry_budget_shared_across_model_calls(provider_http):
    for status in [503, 200, 503, 200, 503]:
        provider_http.replies.append((status, completion(), {}))

    async def run():
        async with module.open_gemini_provider(settings()) as provider:
            await probe(provider)
            await probe(provider)
            with pytest.raises(APIStatusError) as caught:
                await probe(provider)
            assert caught.value.status_code == 503

    asyncio.run(run())
    assert provider_http.sleeps == [2.1, 4.1]
    assert len(provider_http.requests) == 5


def test_new_run_has_its_own_retry_budget(provider_http):
    for status in [503, 200, 503, 200]:
        provider_http.replies.append((status, completion(), {}))

    async def run():
        for _ in range(2):
            async with module.open_gemini_provider(settings()) as provider:
                await probe(provider)

    asyncio.run(run())
    assert provider_http.sleeps == [2.1, 2.1]
    assert len(provider_http.requests) == 4
    assert len(provider_http.clients) == 2


def test_sdk_tool_result_preserved_when_next_model_call_is_retried(provider_http):
    reads = []

    @function_tool
    def read_fixture(path: str) -> str:
        """Read synthetic fixture data only."""
        reads.append(path)
        return "Synthetic fixture content; no real repository access."

    call = {"id": "fixture-call-1", "type": "function", "function": {
        "name": "read_fixture", "arguments": json.dumps({"path": "fixture.py"}),
    }}
    provider_http.replies.extend([
        (200, completion(tool_call=call), {}),
        (429, quota_error(), {}),
        (200, completion(), {}),
    ])

    async def run():
        async with module.open_gemini_provider(settings()) as provider:
            agent = Agent(
                name="Offline provider fixture", model=MODEL,
                tools=[read_fixture], output_type=InvestigationReport,
                model_settings=ModelSettings(store=None, parallel_tool_calls=False),
            )
            result = await Runner.run(
                agent, "Synthetic scenario only.", max_turns=4,
                run_config=RunConfig(model_provider=provider, tracing_disabled=True),
            )
            assert isinstance(result.final_output, InvestigationReport)
            assert result.final_output.model_dump() == REPORT

    asyncio.run(run())
    assert reads == ["fixture.py"]
    assert len(provider_http.requests) == 3
    assert provider_http.requests[1] == provider_http.requests[2]
    assert any(item["role"] == "tool" for item in provider_http.requests[2]["messages"])
    assert all("store" not in body for body in provider_http.requests)
    assert provider_http.sleeps == [50.1]


def test_outer_timeout_cancels_retry_wait(provider_http, monkeypatch):
    provider_http.replies.append((429, quota_error(), {}))
    wait_started = asyncio.Event()

    async def block_wait(seconds):
        wait_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(module.asyncio, "sleep", block_wait)

    async def run():
        async with module.open_gemini_provider(settings()) as provider:
            task = asyncio.create_task(probe(provider))
            try:
                # Start the timeout only after entering the retry wait; slow
                # Windows client construction must not affect this assertion.
                await asyncio.wait_for(wait_started.wait(), timeout=2)
                with pytest.raises(TimeoutError):
                    async with asyncio.timeout(0.02):
                        await task
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    assert len(provider_http.requests) == 1


def test_retry_logs_contain_metadata_without_payloads(provider_http, caplog):
    provider_http.replies.extend([(429, quota_error(), {}), (200, completion(), {})])

    async def run():
        async with module.open_gemini_provider(settings()) as provider:
            await probe(provider)

    with caplog.at_level(logging.WARNING, logger=module.__name__):
        asyncio.run(run())
    assert "Gemini retry 1/2: HTTP 429; waiting 50.10s" in caplog.text
    assert PRIVATE_KEY not in caplog.text
    assert "PRIVATE_PROVIDER_ERROR_BODY" not in caplog.text
    assert "Synthetic diagnostic prompt" not in caplog.text


def test_entering_context_performs_no_request_and_closes_client(provider_http):
    async def run():
        async with module.open_gemini_provider(settings()):
            assert provider_http.requests == []
            assert not provider_http.clients[0].is_closed

    asyncio.run(run())
    assert provider_http.client_options == [{"trust_env": False, "follow_redirects": False}]


@pytest.mark.parametrize("invalid", [None, "wrong-settings-type"])
def test_invalid_settings_rejected_before_client_creation(provider_http, invalid):
    async def run():
        with pytest.raises(module.GeminiProviderError):
            async with module.open_gemini_provider(invalid):
                pytest.fail("Invalid settings were accepted.")

    asyncio.run(run())
    assert provider_http.clients == []


def test_missing_key_rejected_before_client_creation(provider_http):
    async def run():
        with pytest.raises(module.GeminiProviderError):
            async with module.open_gemini_provider(settings(gemini_api_key=None)):
                pytest.fail("Missing credentials were accepted.")

    asyncio.run(run())
    assert provider_http.clients == []
