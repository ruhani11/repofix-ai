"""Offline actual-SDK proposal loops; no live providers or target execution."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
import base64
import hashlib
import importlib
import importlib.util
import json
import os

import httpx
import pytest
from agents import Model, ModelProvider, ModelResponse, Usage
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider
from openai import RateLimitError
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from app.config import Settings
from app.github_client import RepositoryMetadata, RepositorySnapshot
from app.github_oauth import GitHubIdentity
from app.github_repositories import PublicRepository
from app.github_tree import RepositoryTree
from app.investigation_context import InvestigationContext
from app.investigation_history import InvestigationHistoryDetail, InvestigationRunResponse
from app.investigation_input import InvestigationInput
from app.investigation_report import InvestigationReport
from app.investigation_runner import InvestigationEvent
from app.proposal_context import ProposalContext
from app.proposal_generator import ProposalGenerationError, generate_fix_proposal
import app.proposal_generator as module


openai_http = importlib.import_module("httpx2") if importlib.util.find_spec("httpx2") else httpx
SOURCE = "def accept(quantity):\n    if quantity <= 1:\n        return False\n    return True\n"
BEFORE = "    if quantity <= 1:\n"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("REPOFIX_"):
            monkeypatch.delenv(name, raising=False)
    def reject(*args, **kwargs):
        pytest.fail("A proposal test attempted a real provider or GitHub request.")
    for package in {httpx, openai_http}:
        monkeypatch.setattr(package.HTTPTransport, "handle_request", reject)
        monkeypatch.setattr(package.AsyncHTTPTransport, "handle_async_request", reject)
    monkeypatch.setattr(tracing_setup, "GLOBAL_TRACE_PROVIDER", DefaultTraceProvider())


def make_fixture(content=SOURCE):
    identity = GitHubIdentity(id=123, login="demo-user", type="User")
    repo = PublicRepository(
        id=5, name="example", full_name="demo-user/example", owner=identity,
        private=False, visibility="public", default_branch="main",
        fork=False, archived=False, disabled=False,
    )
    data = content.encode("utf-8")
    sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()
    tree = RepositoryTree(
        repository_full_name=repo.full_name, commit_sha="a" * 40, tree_sha="b" * 40,
        entries=[{"path": path, "type": "blob", "mode": mode, "sha": sha, "size": len(data)}
                 for path, mode in (("orders_api.py", "100644"), ("test_existing.py", "100644"),
                                    (".env", "100644"), ("link.py", "120000"))],
    )
    request = InvestigationInput(repository_id=5, repository_name="example", issue_description="A quantity of one is rejected.")
    source_context = InvestigationContext(
        identity, request, repo,
        RepositorySnapshot(repository=RepositoryMetadata(**repo.model_dump()), commit_sha="a" * 40), tree,
    )
    report = InvestigationReport(
        status="likely_cause", issue_summary=request.issue_description, issue_category="logic",
        root_cause="The lower boundary excludes one.", evidence=[{
            "path": "orders_api.py", "start_line": 2, "end_line": 2, "quote": BEFORE.rstrip("\n"),
            "explanation": "The source comparison excludes one.",
        }], next_steps=["Draft a minimal boundary correction."], limitations=["Not reproduced by the agent."],
    )
    history = InvestigationHistoryDetail(
        created_at=datetime(2026, 10, 23, tzinfo=timezone.utc), repository_id=5, repository_name="example",
        issue_description=request.issue_description,
        investigation=InvestigationRunResponse(
            run_id="c" * 32, repository_full_name=repo.full_name, commit_sha="a" * 40,
            report=report, total_duration_ms=1, agent_duration_ms=1, model_calls=1,
            tool_attempts=1, repository_stats={}, events=[
                InvestigationEvent("run_started", 0),
                InvestigationEvent("tool_completed", 1, "read_file", True),
                InvestigationEvent("grounding_validated", 1),
            ], history_saved=True,
        ),
    )
    paths = []
    def handle(http_request):
        assert http_request.method == "GET"
        assert http_request.url.host == "api.github.com"
        assert http_request.url.path == f"/repos/demo-user/example/git/blobs/{sha}"
        assert "authorization" not in http_request.headers and "cookie" not in http_request.headers
        paths.append(http_request.url.path)
        return httpx.Response(200, json={"sha": sha, "size": len(data), "encoding": "base64", "content": base64.b64encode(data).decode()})
    return ProposalContext(history, source_context), httpx.MockTransport(handle), paths


def settings(**changes):
    return Settings(_env_file=None, openai_api_key="offline-key", ai_model="offline-model", **changes)


def draft_data(**changes):
    data = {
        "status": "proposed", "title": "Accept quantity one", "summary": "Correct the lower boundary.",
        "edits": [{"path": "orders_api.py", "before": BEFORE, "after": BEFORE.replace("<= 1", "<= 0"), "reason": "One is positive."}],
        "tests": [{"path": "tests/test_orders_regression.py", "content": "def test_one():\n    assert accept(1) is True\n", "purpose": "Draft regression code; not executed."}],
        "verification_steps": ["Validate the proposed test later in isolation."],
        "risks": [], "limitations": ["Changes are unapproved and tests have not run."],
    }
    data.update(changes)
    return data


def tool(name="read_file", arguments=None, call_id="read-1"):
    return ResponseFunctionToolCall(
        type="function_call", name=name, call_id=call_id,
        arguments=json.dumps(arguments if arguments is not None else {"path": "orders_api.py", "start_line": 1}),
    )


def final(data=None, *, raw=None):
    return ResponseOutputMessage(
        id="msg-proposal", type="message", role="assistant", status="completed",
        content=[ResponseOutputText(type="output_text", annotations=[], text=raw if raw is not None else json.dumps(data or draft_data()))],
    )


class FakeModel(Model):
    def __init__(self, outputs, *, delay=0, before_response=None):
        self.outputs = iter(outputs)
        self.delay = delay
        self.before_response = before_response
        self.calls = []

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing, **kwargs):
        self.calls.append((system_instructions, input, model_settings, tools, tracing))
        if self.before_response:
            self.before_response(len(self.calls))
        await asyncio.sleep(self.delay)
        output = next(self.outputs)
        if isinstance(output, BaseException):
            raise output
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    async def stream_response(self, *args, **kwargs):
        raise AssertionError("Unexpected streaming request")
        yield


class Provider(ModelProvider):
    def __init__(self, model, name="offline-model"):
        self.model, self.name, self.requests, self.closed = model, name, 0, False

    def get_model(self, model_name):
        assert model_name == self.name
        self.requests += 1
        return self.model

    async def aclose(self):
        self.closed = True


def run(outputs, *, config=None, content=SOURCE):
    context, transport, paths = make_fixture(content)
    model = FakeModel(outputs)
    provider = Provider(model)
    result = asyncio.run(generate_fix_proposal(context, config or settings(), model_provider=provider, repository_transport=transport))
    return result, model, provider, paths


def test_actual_sdk_read_then_draft_is_source_validated_with_metadata_only_events():
    result, model, provider, paths = run([[tool()], [final()]])
    assert len(result.proposal_id) == 32
    assert result.validated.investigation_run_id == "c" * 32
    assert result.validated.base_commit_sha == "a" * 40
    assert result.validated.draft.status == "proposed"
    assert "-    if quantity <= 1:\n+    if quantity <= 0:\n" in result.validated.unified_diff
    assert result.model_calls == 2 and result.tool_attempts == 1
    assert result.repository_stats == {"tool_calls": 1, "file_fetch_attempts": 1, "downloaded_bytes": len(SOURCE.encode())}
    assert len(paths) == 1 and result.duration_ms >= 0 and provider.closed is False
    assert result.events[-2].kind == "proposal_source_validated"
    assert result.events[-1].kind == "proposal_completed"
    assert BEFORE not in repr(result.events) and "offline-key" not in repr(result)
    assert "A quantity of one is rejected." not in model.calls[0][0]
    assert "A quantity of one is rejected." in json.dumps(model.calls[0][1])
    assert {item.name for item in model.calls[0][3]} == {"find_files", "search_code", "read_file"}
    assert model.calls[0][-1].is_disabled()


def test_http_generation_from_saved_eventful_history_through_real_context_and_sdk(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.api import proposals as api
    from app.fix_proposal import FixProposalInput
    from app.main import create_app
    from app.proposal_context import prepare_proposal_context

    prepared, blob_transport, _ = make_fixture()
    source = prepared.repository_context
    prefix = "/repos/demo-user/example"
    paths = []

    def github(request):
        paths.append(request.url.path)
        if request.url.path == prefix:
            return httpx.Response(200, json=source.repository.model_dump())
        if request.url.path == prefix + "/git/commits/" + source.snapshot.commit_sha:
            return httpx.Response(200, json={"sha": source.snapshot.commit_sha, "tree": {"sha": source.tree.tree_sha}})
        if request.url.path == prefix + "/git/trees/" + source.tree.tree_sha:
            return httpx.Response(200, json={"sha": source.tree.tree_sha, "truncated": False, "tree": [entry.model_dump() for entry in source.tree.entries]})
        return blob_transport.handle_request(request)

    transport = httpx.MockTransport(github)
    model = FakeModel([[tool()], [final()]])
    provider = Provider(model)
    config = settings(history_database_path=tmp_path / "history.db", proposal_database_path=tmp_path / "proposals.db")
    app = create_app(config)
    app.state.investigation_store.save(source.identity.id, source.request, prepared.history.investigation)
    payload = FixProposalInput(investigation_run_id=prepared.history.investigation.run_id)

    def actual_prepare(identity, body, store):
        assert body == payload
        return prepare_proposal_context(identity, body, store, transport=transport)

    async def actual_generate(context, selected_settings):
        return await generate_fix_proposal(context, selected_settings, model_provider=provider, repository_transport=transport)

    monkeypatch.setattr(api, "prepare_proposal_context", actual_prepare)
    monkeypatch.setattr(api, "generate_fix_proposal", actual_generate)
    base = "http://127.0.0.1:8000"
    with TestClient(app, base_url=base) as client:
        token = app.state.auth_store.create_session(source.identity)
        client.cookies.set("repofix_session", token)
        saved = client.get("/api/investigations/" + payload.investigation_run_id)
        assert saved.status_code == 200 and len(saved.json()["investigation"]["events"]) == 3
        generated = client.post("/api/proposals/generate", json=payload.model_dump(), headers={"Origin": base, "X-RepoFix-CSRF": "1"})
        assert generated.status_code == 200, generated.text
        data = generated.json()
        assert data["proposal_saved"] is True and data["status"] == "proposed"
        assert data["base_commit_sha"] == source.snapshot.commit_sha
        assert "quantity <= 0" in data["unified_diff"]
        assert client.get("/api/proposals/" + data["proposal_id"]).json() == data
        assert client.get("/api/proposals").json()["proposals"][0]["proposal_id"] == data["proposal_id"]
    assert len(model.calls) == 2 and len(paths) == 5
    assert model.calls[0][2].store is False


def test_crlf_anchors_are_preserved_from_tool_read_to_final_patch():
    content = SOURCE.replace("\n", "\r\n")
    data = draft_data()
    data["edits"][0].update(before=BEFORE.replace("\n", "\r\n"), after=BEFORE.replace("<= 1", "<= 0").replace("\n", "\r\n"))
    result, model, _, _ = run([[tool()], [final(data)]], content=content)
    change = next(item for item in result.validated.changes if item.action == "modify")
    assert change.proposed_content == content.replace("<= 1", "<= 0")
    observed = next(item["output"] for item in model.calls[1][1] if item.get("type") == "function_call_output")
    assert json.loads(observed)["data"]["content"] == content


@pytest.mark.parametrize("outputs", [
    [[final()]],
    [[tool("search_code", {"query": "quantity <= 1", "paths": ["orders_api.py"]})], [final()]],
    [[tool(arguments={"path": ".env", "start_line": 1})], [final()]],
    [[tool(arguments={"path": "link.py", "start_line": 1})], [final()]],
])
def test_prior_citations_search_previews_and_denied_reads_cannot_authorize_edits(outputs):
    with pytest.raises(ProposalGenerationError) as caught:
        run(outputs)
    assert caught.value.code == "unobserved_source"


def test_cached_but_unobserved_source_cannot_be_edited():
    content = SOURCE + "# extra line\n" * 100
    with pytest.raises(ProposalGenerationError) as caught:
        run([[tool(arguments={"path": "orders_api.py", "start_line": 81})], [final()]], content=content)
    assert caught.value.code == "unobserved_source"


@pytest.mark.parametrize("failure", ["ambiguous", "existing_test"])
def test_source_validation_rejects_bad_draft_without_an_automatic_paid_repair(failure):
    data = draft_data()
    if failure == "ambiguous":
        data["edits"][0].update(before="return", after="yield")
    else:
        data["tests"][0]["path"] = "test_existing.py"
    context, transport, _ = make_fixture()
    model = FakeModel([[tool()], [final(data)]])
    with pytest.raises(ProposalGenerationError) as caught:
        asyncio.run(generate_fix_proposal(context, settings(), model_provider=Provider(model), repository_transport=transport))
    assert caught.value.code == "invalid_proposal_source" and len(model.calls) == 2


def test_final_budget_turn_has_no_tools_and_still_validates_the_draft():
    result, model, _, _ = run([[tool()], [final()]], config=settings(ai_max_turns=2))
    assert result.model_calls == 2
    assert model.calls[-1][3] == [] and model.calls[-1][2].tool_choice == "none"
    assert "Final turn" in model.calls[-1][0] and "FixProposal" in model.calls[-1][0]
    assert "function_call_output" in json.dumps(model.calls[-1][1])


def test_unsupported_final_turn_tool_is_rejected_before_execution():
    with pytest.raises(ProposalGenerationError) as caught:
        run([[tool()], [tool("find_files", {"keyword": "orders"})]], config=settings(ai_max_turns=2))
    assert caught.value.code == "turn_limit"


def test_tool_batch_cannot_exceed_the_existing_attempt_budget():
    calls = [tool("find_files", {"keyword": "orders"}, f"find-{n}") for n in range(21)]
    with pytest.raises(ProposalGenerationError) as caught:
        run([calls, [final()]])
    assert caught.value.code == "tool_limit"


def test_strict_tool_arguments_do_not_coerce_string_line_numbers():
    with pytest.raises(ProposalGenerationError) as caught:
        run([[tool(arguments={"path": "orders_api.py", "start_line": "1"})], [final()]])
    assert caught.value.code == "unobserved_source"


@pytest.mark.parametrize("raw", ["not JSON", "{}", json.dumps({**draft_data(), "status": "applied"})])
def test_invalid_or_applied_model_output_never_returns_a_success(raw):
    with pytest.raises(ProposalGenerationError) as caught:
        run([[tool()], [final(raw=raw)]])
    assert caught.value.code == "agent_error"


@pytest.mark.parametrize("invalid", ["missing_key", "one_turn", "wrong_provider", "invalid_settings", "wrong_context", "wrong_owner", "wrong_commit", "wrong_issue", "not_saved", "insufficient"])
def test_invalid_preflight_context_or_configuration_stops_before_model_and_source_reads(invalid):
    context, transport, paths = make_fixture()
    config = settings()
    model = FakeModel([])
    provider = Provider(model)
    code = "invalid_context"
    if invalid == "missing_key":
        config = config.model_copy(update={"openai_api_key": None})
        code = "not_configured"
    elif invalid == "one_turn":
        config = settings(ai_max_turns=1)
        code = "invalid_settings"
    elif invalid == "wrong_provider":
        provider = object()
        code = "invalid_provider"
    elif invalid == "invalid_settings":
        config = config.model_copy(update={"ai_max_turns": True})
        # Settings uses the existing non-strict numeric settings contract;
        # bool is parsed as one and then fails the proposal minimum of two.
        code = "invalid_settings"
    elif invalid == "wrong_context":
        context = object()
    elif invalid == "wrong_owner":
        context = replace(context, repository_context=replace(context.repository_context, identity=GitHubIdentity(id=456, login="other-user", type="User")))
    elif invalid == "wrong_commit":
        context = replace(context, repository_context=replace(context.repository_context, tree=context.repository_context.tree.model_copy(update={"commit_sha": "f" * 40})))
    elif invalid == "wrong_issue":
        context = replace(context, repository_context=replace(context.repository_context, request=context.repository_context.request.model_copy(update={"issue_description": "A different unrelated issue."})))
    else:
        record = context.history.model_dump()
        record["investigation"]["events"] = list(context.history.investigation.events)
        if invalid == "not_saved":
            record["investigation"]["history_saved"] = False
        else:
            record["investigation"]["report"].update(status="insufficient_evidence", root_cause=None)
        context = replace(context, history=InvestigationHistoryDetail.model_validate(record))
    with pytest.raises(ProposalGenerationError) as caught:
        asyncio.run(generate_fix_proposal(context, config, model_provider=provider, repository_transport=transport))
    assert caught.value.code == code and model.calls == [] and paths == []


def test_a_new_run_cannot_reuse_read_observations_or_the_file_cache():
    context, transport, paths = make_fixture()
    model = FakeModel([[tool()], [final()], [final()]])
    provider = Provider(model)
    first = asyncio.run(generate_fix_proposal(context, settings(), model_provider=provider, repository_transport=transport))
    assert first.validated.draft.status == "proposed"
    with pytest.raises(ProposalGenerationError) as caught:
        asyncio.run(generate_fix_proposal(context, settings(), model_provider=provider, repository_transport=transport))
    assert caught.value.code == "unobserved_source" and len(paths) == 1


def test_caller_mutation_after_first_await_cannot_rebind_context_or_issue():
    context, transport, _ = make_fixture()
    def mutate(call):
        if call == 1:
            context.repository_context.tree.entries.clear()
            context.history.investigation.report.evidence.clear()
    model = FakeModel([[tool()], [final()]], before_response=mutate)
    result = asyncio.run(generate_fix_proposal(context, settings(), model_provider=Provider(model), repository_transport=transport))
    assert result.validated.base_commit_sha == "a" * 40
    assert result.validated.investigation_run_id == "c" * 32


def test_timeout_has_no_success_and_does_not_start_a_second_run(monkeypatch):
    original = asyncio.timeout
    monkeypatch.setattr(module.asyncio, "timeout", lambda seconds: original(0.01))
    context, transport, paths = make_fixture()
    model = FakeModel([[tool()]], delay=0.1)
    with pytest.raises(ProposalGenerationError) as caught:
        asyncio.run(generate_fix_proposal(context, settings(), model_provider=Provider(model), repository_transport=transport))
    # The deadline may expire during SDK setup or during its first model call.
    assert caught.value.code == "timeout" and len(model.calls) <= 1 and paths == []


def test_external_cancellation_propagates():
    context, transport, _ = make_fixture()
    model = FakeModel([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(generate_fix_proposal(context, settings(), model_provider=Provider(model), repository_transport=transport))


def test_raw_provider_or_source_failure_details_are_not_exposed():
    with pytest.raises(ProposalGenerationError) as caught:
        run([RuntimeError("PRIVATE_MODEL_PAYLOAD")])
    assert caught.value.code == "run_error" and "PRIVATE" not in str(caught.value)


def test_provider_quota_reason_and_retry_hint_are_classified_without_payloads():
    response = openai_http.Response(429, request=openai_http.Request("POST", "https://api.openai.com/v1/responses"), headers={"retry-after": "60"})
    error = RateLimitError("PRIVATE_PROVIDER_PAYLOAD", response=response, body={"error": {"message": "PRIVATE_PROVIDER_PAYLOAD"}})
    with pytest.raises(ProposalGenerationError) as caught:
        run([error])
    assert caught.value.code == "provider_limit" and caught.value.retry_after == 60
    assert "PRIVATE" not in str(caught.value)


@pytest.mark.parametrize("fails", [False, True])
def test_owned_openai_provider_configuration_and_cleanup(monkeypatch, fails):
    outputs = [RuntimeError("PRIVATE")] if fails else [[tool()], [final()]]
    model = FakeModel(outputs)
    state = {}
    class Client:
        def __init__(self, **kwargs): state.update(kwargs)
        async def __aenter__(self): return self
        async def __aexit__(self, *args): state["client_closed"] = True
    class Owned(Provider):
        def __init__(self, **kwargs):
            assert kwargs["use_responses"] is True and kwargs["use_responses_websocket"] is False
            super().__init__(model)
        async def aclose(self): state["provider_closed"] = True
    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    monkeypatch.setattr(module, "OpenAIProvider", Owned)
    context, transport, _ = make_fixture()
    if fails:
        with pytest.raises(ProposalGenerationError):
            asyncio.run(generate_fix_proposal(context, settings(), repository_transport=transport))
    else:
        asyncio.run(generate_fix_proposal(context, settings(), repository_transport=transport))
    assert state["api_key"] == "offline-key" and state["base_url"] == "https://api.openai.com/v1"
    assert state["max_retries"] == 0 and state["client_closed"] and state["provider_closed"]
    assert state["http_client"].is_closed


@pytest.mark.parametrize("fails", [False, True])
def test_gemini_uses_existing_scoped_provider_without_openai_fallback(monkeypatch, fails):
    model = FakeModel([RuntimeError("PRIVATE_GEMINI_FAILURE")] if fails else [[tool()], [final()]])
    state = []
    @asynccontextmanager
    async def gemini(config):
        state.append("opened")
        try: yield Provider(model, "gemini-offline")
        finally: state.append("closed")
    monkeypatch.setattr(module, "open_gemini_provider", gemini)
    monkeypatch.setattr(module, "AsyncOpenAI", lambda **kwargs: pytest.fail("Unexpected OpenAI fallback"))
    config = Settings(_env_file=None, ai_provider="gemini", ai_model="gemini-offline", gemini_api_key="offline-gemini-key")
    context, transport, _ = make_fixture()
    if fails:
        with pytest.raises(ProposalGenerationError):
            asyncio.run(generate_fix_proposal(context, config, repository_transport=transport))
    else:
        result = asyncio.run(generate_fix_proposal(context, config, repository_transport=transport))
        assert result.model_calls == 2
    assert state == ["opened", "closed"]
    assert all(call[2].store is None for call in model.calls)
