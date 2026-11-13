"""Offline SDK-loop integration tests; no credentials or live requests."""

import asyncio
import base64
import hashlib
import importlib.util
import json
import os

import httpx
import pytest
from agents import Model, ModelProvider, ModelResponse, Usage
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from openai import APIConnectionError

from app.config import Settings
from app.investigation_runner import InvestigationRunError, run_investigation
import app.investigation_runner as module
from app.github_client import RepositoryMetadata, RepositorySnapshot
from app.github_oauth import GitHubIdentity
from app.github_repositories import PublicRepository
from app.github_tree import RepositoryTree
from app.investigation_context import InvestigationContext
from app.investigation_input import InvestigationInput
from app import gemini_provider


# Match the HTTP implementation used by the installed OpenAI dependency.
# HTTPX2 is transitive in newer SDK environments; no new package is required.
openai_http = (
    importlib.import_module("httpx2")
    if importlib.util.find_spec("httpx2") is not None
    else httpx
)


def make_fixture(content=None):
    text = content if content is not None else "\n".join(
        f"line {number}" for number in range(1, 101)
    )
    data = text.encode("utf-8")
    blob_sha = hashlib.sha1(
        f"blob {len(data)}\0".encode("ascii") + data,
        usedforsecurity=False,
    ).hexdigest()
    identity = GitHubIdentity(id=123, login="demo-user", type="User")
    repository = PublicRepository(
        id=5, name="example", full_name="demo-user/example", owner=identity,
        private=False, visibility="public", default_branch="main",
        fork=False, archived=False, disabled=False,
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
            {"path": path, "type": "blob", "mode": mode,
             "sha": blob_sha, "size": len(data)}
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
            repository_id=5, repository_name="example",
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
        return httpx.Response(200, json={
            "sha": blob_sha,
            "size": len(data),
            "encoding": "base64",
            "content": base64.b64encode(data).decode("ascii"),
        })

    return context, httpx.MockTransport(handler), requests


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    # Keep actual process settings/credentials out of synthetic test fixtures.
    for name in tuple(os.environ):
        if name.startswith("REPOFIX_"):
            monkeypatch.delenv(name, raising=False)
    def reject(*args, **kwargs):
        pytest.fail('A runner test attempted a real network request.')
    for package in {httpx, openai_http}:
        monkeypatch.setattr(package.HTTPTransport, 'handle_request', reject)
        monkeypatch.setattr(package.AsyncHTTPTransport, 'handle_async_request', reject)
    monkeypatch.setattr(tracing_setup, 'GLOBAL_TRACE_PROVIDER', DefaultTraceProvider())


def settings(**kwargs):
    return Settings(_env_file=None, openai_api_key='offline-key', ai_model='offline-model', **kwargs)


def tool(name='read_file', arguments=None, call_id='call-1'):
    return ResponseFunctionToolCall(type='function_call', name=name, call_id=call_id,
        arguments=json.dumps(arguments or {'path': 'app/auth.py', 'start_line': 1}))


def final(quote='line 10', insufficient=False):
    report = dict(status='likely_cause', issue_summary='Synthetic issue.', issue_category='logic',
        root_cause='Synthetic hypothesis.', evidence=[dict(path='app/auth.py', start_line=10,
        end_line=10, quote=quote, explanation='Synthetic evidence.')], next_steps=['Reproduce.'],
        limitations=['No tests executed.'])
    if insufficient:
        report.update(status='insufficient_evidence', root_cause=None, evidence=[])
    return ResponseOutputMessage(id='msg-test', type='message', role='assistant', status='completed',
        content=[ResponseOutputText(type='output_text', annotations=[], text=json.dumps(report))])


class FakeModel(Model):
    def __init__(self, outputs, delay=0):
        self.outputs = iter(outputs)
        self.calls = []
        self.delay = delay
    async def get_response(self, system_instructions, input, model_settings, tools, output_schema,
            handoffs, tracing, **kwargs):
        self.calls.append((system_instructions, input, model_settings, tools, tracing))
        await asyncio.sleep(self.delay)
        output = next(self.outputs)
        if isinstance(output, BaseException):
            raise output
        return ModelResponse(output=output, usage=Usage(), response_id=None)
    async def stream_response(self, *args, **kwargs):
        raise AssertionError('Unexpected streaming request.')
        yield


class Provider(ModelProvider):
    def __init__(self, model):
        self.model = model
    def get_model(self, name):
        assert name == 'offline-model'
        return self.model


def run(outputs, **kwargs):
    context, transport, requests = make_fixture()
    model = FakeModel(outputs)
    result = asyncio.run(run_investigation(context, settings(), model_provider=Provider(model),
        repository_transport=transport, **kwargs))
    return result, model, requests


def test_actual_sdk_read_then_grounded_report():
    result, model, requests = run([[tool()], [final()]])
    assert len(requests) == 1
    assert result.model_calls == 2 and result.tool_attempts == 1
    assert result.repository_stats['tool_calls'] == 1
    assert result.repository_stats['file_fetch_attempts'] == 1
    assert result.repository_stats['downloaded_bytes'] > 0
    assert result.report.evidence[0].quote == 'line 10'
    assert result.commit_sha == 'a' * 40 and result.duration_ms >= 0
    assert result.events[-2].kind == 'grounding_validated'
    assert 'line 10' not in repr(result.events)
    assert 'Synthetic grounding test.' not in model.calls[0][0]
    assert 'Synthetic grounding test.' in json.dumps(model.calls[0][1])
    assert model.calls[0][-1].is_disabled()


@pytest.mark.parametrize('outputs', [[[final()]], [[tool()], [final('fabricated quote')]]])
def test_unobserved_or_fabricated_evidence_rejected(outputs):
    with pytest.raises(InvestigationRunError) as caught:
        run(outputs)
    assert caught.value.code == 'ungrounded_report'


def test_denied_read_can_end_with_insufficient_evidence():
    result, model, requests = run([[tool(arguments={'path': '.env', 'start_line': 1})], [final(insufficient=True)]])
    assert result.report.status == 'insufficient_evidence' and requests == []
    assert any(e.kind == 'tool_completed' and e.succeeded is False for e in result.events)


def test_invalid_sdk_arguments_are_counted_as_attempts():
    result, model, requests = run([[tool(arguments={'path': 'app/auth.py', 'start_line': 'bad'})], [final(insufficient=True)]])
    assert result.tool_attempts == 1 and result.repository_stats['tool_calls'] == 0
    assert requests == []


def test_tool_batch_cannot_bypass_attempt_budget():
    calls = [tool('find_files', {'keyword': ''}, f'call-{i}') for i in range(21)]
    with pytest.raises(InvestigationRunError) as caught:
        run([calls, [final(insufficient=True)]])
    assert caught.value.code == 'tool_limit'


def test_sdk_max_turn_limit():
    context, transport, requests = make_fixture()
    model = FakeModel([[tool('find_files', {'keyword': ''})]])
    with pytest.raises(InvestigationRunError) as caught:
        asyncio.run(run_investigation(context, settings(ai_max_turns=1), model_provider=Provider(model), repository_transport=transport))
    assert caught.value.code == 'turn_limit' and len(model.calls) == 1


def test_timeout_and_no_success_report(monkeypatch):
    context, transport, requests = make_fixture()
    # Shorten only the test deadline; production settings retain their validation.
    original = asyncio.timeout
    monkeypatch.setattr(module.asyncio, 'timeout', lambda seconds: original(0.01))
    model = FakeModel([[final(insufficient=True)]], delay=0.1)
    with pytest.raises(InvestigationRunError) as caught:
        asyncio.run(run_investigation(context, settings(), model_provider=Provider(model), repository_transport=transport))
    assert caught.value.code == 'timeout' and requests == []


def test_unexpected_error_details_are_redacted():
    with pytest.raises(InvestigationRunError) as caught:
        run([RuntimeError('secret-provider-payload')])
    assert caught.value.code == 'run_error' and 'secret-provider-payload' not in str(caught.value)


def test_external_cancellation_propagates():
    with pytest.raises(asyncio.CancelledError):
        run([asyncio.CancelledError()])


def test_missing_key_prevents_run():
    context, transport, requests = make_fixture()
    config = Settings(_env_file=None, ai_model='offline-model')
    with pytest.raises(InvestigationRunError) as caught:
        asyncio.run(run_investigation(context, config, repository_transport=transport))
    assert caught.value.code == 'not_configured' and requests == []


def test_provider_error_is_sanitized():
    error = APIConnectionError(message='secret-provider-payload', request=openai_http.Request('POST', 'https://api.openai.com/v1/responses'))
    with pytest.raises(InvestigationRunError) as caught:
        run([error])
    assert caught.value.code == 'model_error'
    assert 'secret-provider-payload' not in str(caught.value)


def test_structurally_invalid_report_is_rejected_by_sdk():
    message = ResponseOutputMessage(id='msg-test', type='message', role='assistant', status='completed',
        content=[ResponseOutputText(type='output_text', annotations=[], text='{}')])
    with pytest.raises(InvestigationRunError) as caught:
        run([[message]])
    assert caught.value.code == 'agent_error'


def test_observations_do_not_carry_into_another_run():
    context, transport, requests = make_fixture()
    model = FakeModel([[tool()], [final()], [final()]])
    provider = Provider(model)
    first = asyncio.run(run_investigation(context, settings(), model_provider=provider, repository_transport=transport))
    assert first.report.status == 'likely_cause'
    with pytest.raises(InvestigationRunError) as caught:
        asyncio.run(run_investigation(context, settings(), model_provider=provider, repository_transport=transport))
    assert caught.value.code == 'ungrounded_report' and len(requests) == 1


@pytest.mark.parametrize('provider_fails', [False, True])
def test_owned_client_configuration_and_cleanup(monkeypatch, provider_fails):
    output = RuntimeError('private failure') if provider_fails else [final(insufficient=True)]
    model = FakeModel([output])
    received = {}
    class Client:
        def __init__(self, **kwargs):
            received.update(kwargs)
        async def __aenter__(self): return self
        async def __aexit__(self, *args): received['closed'] = True
    class LiveProvider(Provider):
        def __init__(self, **kwargs):
            assert kwargs['use_responses'] is True and kwargs['use_responses_websocket'] is False
            super().__init__(model)
        async def aclose(self): received['provider_closed'] = True
    monkeypatch.setattr(module, 'AsyncOpenAI', Client)
    monkeypatch.setattr(module, 'OpenAIProvider', LiveProvider)
    context, transport, requests = make_fixture()
    if provider_fails:
        with pytest.raises(InvestigationRunError):
            asyncio.run(run_investigation(context, settings(), repository_transport=transport))
    else:
        result = asyncio.run(run_investigation(context, settings(), repository_transport=transport))
        assert result.report.status == 'insufficient_evidence'
    assert received['base_url'] == 'https://api.openai.com/v1'
    assert received['max_retries'] == 0 and received['api_key'] == 'offline-key'
    assert received['closed'] and received['provider_closed']
    assert received['http_client'].is_closed


@pytest.mark.parametrize('limit', [1, 2, 3, 10, 20])
def test_reserved_last_turn_disables_tools_and_preserves_history(limit):
    context, transport, requests = make_fixture()
    outputs = [
        [tool('find_files', {'keyword': 'auth'}, f'find-{number}')]
        for number in range(limit - 1)
    ] + [[final(insufficient=True)]]
    model = FakeModel(outputs)
    result = asyncio.run(run_investigation(
        context, settings(ai_max_turns=limit), model_provider=Provider(model),
        repository_transport=transport,
    ))
    assert result.model_calls == limit
    assert result.tool_attempts == limit - 1 and requests == []
    assert model.calls[-1][3] == []
    assert model.calls[-1][2].tool_choice == 'none'
    assert 'This is the final turn' in model.calls[-1][0]
    assert all(
        {item.name for item in call[3]} == {'find_files', 'read_file', 'search_code'}
        for call in model.calls[:-1]
    )
    assert all(call[2].tool_choice is None for call in model.calls[:-1])
    assert 'Synthetic grounding test.' in json.dumps(model.calls[-1][1])
    if limit > 1:
        assert 'function_call_output' in json.dumps(model.calls[-1][1])


def test_reserved_report_can_cite_previously_observed_source():
    context, transport, requests = make_fixture()
    model = FakeModel([[tool()], [final()]])
    result = asyncio.run(run_investigation(
        context, settings(ai_max_turns=2), model_provider=Provider(model),
        repository_transport=transport,
    ))
    assert result.report.status == 'likely_cause'
    assert result.report.evidence[0].quote == 'line 10'
    assert result.model_calls == 2 and result.tool_attempts == 1
    assert len(requests) == 1
    assert 'line 10' in json.dumps(model.calls[-1][1])
    assert model.calls[-1][3] == []


@pytest.mark.parametrize('with_read', [False, True])
def test_reserved_report_cannot_bypass_source_grounding(with_read):
    context, transport, requests = make_fixture()
    outputs = ([[tool()]] if with_read else []) + [[final('invented quote')]]
    model = FakeModel(outputs)
    with pytest.raises(InvestigationRunError) as caught:
        asyncio.run(run_investigation(
            context, settings(ai_max_turns=len(outputs)),
            model_provider=Provider(model), repository_transport=transport,
        ))
    assert caught.value.code == 'ungrounded_report'
    assert model.calls[-1][3] == []


@pytest.mark.parametrize('initial_read', [False, True])
def test_reserved_turn_rejects_unrequested_tools_before_execution(initial_read):
    context, transport, requests = make_fixture()
    outputs = ([[tool()]] if initial_read else []) + [[tool(call_id='denied-last')]]
    model = FakeModel(outputs)
    with pytest.raises(InvestigationRunError) as caught:
        asyncio.run(run_investigation(
            context, settings(ai_max_turns=len(outputs)),
            model_provider=Provider(model), repository_transport=transport,
        ))
    assert caught.value.code == 'turn_limit'
    assert len(requests) == int(initial_read)


def test_early_report_does_not_consume_remaining_reserved_budget():
    context, transport, requests = make_fixture()
    model = FakeModel([[final(insufficient=True)]])
    result = asyncio.run(run_investigation(
        context, settings(), model_provider=Provider(model),
        repository_transport=transport,
    ))
    assert result.model_calls == 1 and result.tool_attempts == 0
    assert {item.name for item in model.calls[0][3]} == {
        'find_files', 'read_file', 'search_code',
    }
    assert model.calls[0][2].tool_choice is None


def test_reused_provider_does_not_reuse_previous_run_turn_budget():
    context, transport, requests = make_fixture()
    model = FakeModel([[final(insufficient=True)], [final(insufficient=True)]])
    provider = Provider(model)
    for _ in range(2):
        result = asyncio.run(run_investigation(
            context, settings(ai_max_turns=1), model_provider=provider,
            repository_transport=transport,
        ))
        assert result.model_calls == 1
    assert len(model.calls) == 2
    assert all('logical model turn 1 of 1' in call[0] for call in model.calls)


@pytest.fixture
def report_turn_http(monkeypatch):
    """Mock the owned Gemini HTTP client; never use a live key or real sleep."""
    state = {'replies': [], 'requests': [], 'sleeps': [], 'clients': []}

    def handle(request):
        assert request.method == 'POST'
        assert request.url.host == 'generativelanguage.googleapis.com'
        assert request.url.path == '/v1beta/openai/chat/completions'
        state['requests'].append(json.loads(request.content))
        assert state['replies'], 'Unexpected extra provider HTTP request.'
        status, body = state['replies'].pop(0)
        return openai_http.Response(status, json=body)

    original_client = gemini_provider.DefaultAsyncHttpxClient

    def make_client(**kwargs):
        client = original_client(transport=openai_http.MockTransport(handle), **kwargs)
        state['clients'].append(client)
        return client

    async def no_wait(seconds):
        state['sleeps'].append(seconds)

    monkeypatch.setattr(gemini_provider, 'DefaultAsyncHttpxClient', make_client)
    monkeypatch.setattr(gemini_provider.random, 'uniform', lambda low, high: 0.1)
    monkeypatch.setattr(gemini_provider.asyncio, 'sleep', no_wait)
    yield state
    assert all(client.is_closed for client in state['clients'])


def report_turn_settings(limit):
    return Settings(
        _env_file=None, ai_provider='gemini', ai_model='gemini-offline-fixture',
        gemini_api_key='dummy-private-fixture-key', ai_max_turns=limit,
    )


def report_turn_completion(*, read=False):
    message = {'role': 'assistant', 'content': final(insufficient=True).content[0].text}
    if read:
        message.update(content=None, tool_calls=[{
            'id': 'read-1', 'type': 'function', 'function': {
                'name': 'read_file',
                'arguments': json.dumps({'path': 'app/auth.py', 'start_line': 1}),
            },
        }])
    return {
        'id': 'fixture-completion', 'object': 'chat.completion', 'created': 0,
        'model': 'gemini-offline-fixture',
        'choices': [{'index': 0, 'message': message,
                     'finish_reason': 'tool_calls' if read else 'stop'}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    }


def report_turn_quota(*, daily=False):
    quota = ('GenerateRequestsPerDayPerProjectPerModel-FreeTier' if daily
             else 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier')
    return [{'error': {
        'code': 429, 'message': 'PRIVATE_PROVIDER_ERROR_BODY',
        'status': 'RESOURCE_EXHAUSTED', 'details': [
            {'@type': 'type.googleapis.com/google.rpc.QuotaFailure',
             'violations': [{'quotaId': quota}]},
            {'@type': 'type.googleapis.com/google.rpc.RetryInfo',
             'retryDelay': '28217s' if daily else '5s'},
        ],
    }}]


@pytest.mark.parametrize('status', [429, 503])
def test_reserved_gemini_report_retry_keeps_history_schema_and_logical_budget(
    report_turn_http, status,
):
    context, transport, requests = make_fixture()
    error = report_turn_quota() if status == 429 else {'error': {'message': 'temporary'}}
    report_turn_http['replies'].extend([
        (200, report_turn_completion(read=True)),
        (status, error),
        (200, report_turn_completion()),
    ])
    result = asyncio.run(run_investigation(
        context, report_turn_settings(2), repository_transport=transport,
    ))
    assert result.model_calls == 2 and result.tool_attempts == 1
    assert len(requests) == 1
    assert len(report_turn_http['requests']) == 3
    first, last, retry = report_turn_http['requests']
    assert {item['function']['name'] for item in first['tools']} == {
        'find_files', 'read_file', 'search_code',
    }
    assert not last.get('tools') and last['tool_choice'] == 'none'
    assert last == retry
    assert any(item['role'] == 'tool' for item in last['messages'])
    assert 'line 10' in json.dumps(last['messages'])
    assert all('store' not in body for body in report_turn_http['requests'])
    assert last['response_format'] == first['response_format']
    assert report_turn_http['sleeps'] == ([5.1] if status == 429 else [2.1])


def test_reserved_gemini_report_does_not_hide_or_retry_daily_quota(report_turn_http):
    context, transport, requests = make_fixture()
    report_turn_http['replies'].append((429, report_turn_quota(daily=True)))
    with pytest.raises(InvestigationRunError) as caught:
        asyncio.run(run_investigation(
            context, report_turn_settings(1), repository_transport=transport,
        ))
    assert caught.value.code == 'provider_daily_quota'
    assert caught.value.retry_after == 28217
    assert 'PRIVATE_PROVIDER_ERROR_BODY' not in str(caught.value)
    assert len(report_turn_http['requests']) == 1
    assert report_turn_http['sleeps'] == [] and requests == []
    assert not report_turn_http['requests'][0].get('tools')


def search_tool(arguments=None, call_id='search-1'):
    return tool('search_code', arguments if arguments is not None else {
        'query': 'line 10', 'paths': ['app/auth.py'],
    }, call_id)


def test_search_read_report_uses_actual_sdk_callbacks_and_only_read_evidence(monkeypatch):
    context, transport, requests = make_fixture()
    model = FakeModel([
        [tool('find_files', {'keyword': 'auth'}, 'find-1')],
        [search_tool()],
        [tool(call_id='read-1')],
        [final()],
    ])
    observed = []
    validate = module.validate_report_grounding

    def capture(report, prepared, reads):
        observed.extend(reads)
        return validate(report, prepared, reads)

    monkeypatch.setattr(module, 'validate_report_grounding', capture)
    result = asyncio.run(run_investigation(
        context, settings(ai_max_turns=4), model_provider=Provider(model),
        repository_transport=transport,
    ))

    assert result.report.status == 'likely_cause'
    assert result.report.evidence[0].quote == 'line 10'
    assert result.model_calls == 4 and result.tool_attempts == 3
    assert result.repository_stats['tool_calls'] == 3
    assert result.repository_stats['file_fetch_attempts'] == len(requests) == 1
    assert len(observed) == 1
    assert observed[0]['content'].startswith('line 1\n')
    assert 'matches' not in observed[0] and 'source_lines' not in observed[0]
    assert 'search_code' in model.calls[0][0]
    assert 'discovery_only' in json.dumps(model.calls[2][1])
    assert 'source_lines' in json.dumps(model.calls[-1][1])
    assert model.calls[-1][3] == []
    starts = [e.tool_name for e in result.events if e.kind == 'tool_started']
    assert starts == ['find_files', 'search_code', 'read_file']
    assert result.events[-2].kind == 'grounding_validated'
    assert 'line 10' not in repr(result.events)


def test_exact_search_preview_cannot_replace_a_read_for_grounding():
    context, transport, requests = make_fixture()
    model = FakeModel([[search_tool()], [final()]])
    with pytest.raises(InvestigationRunError) as caught:
        asyncio.run(run_investigation(
            context, settings(ai_max_turns=2), model_provider=Provider(model),
            repository_transport=transport,
        ))
    assert caught.value.code == 'ungrounded_report'
    assert len(requests) == 1
    assert model.calls[-1][3] == []


def test_search_only_can_finish_with_insufficient_evidence():
    result, model, requests = run([[search_tool()], [final(insufficient=True)]])
    assert result.report.status == 'insufficient_evidence'
    assert result.report.root_cause is None and result.report.evidence == []
    assert result.tool_attempts == result.repository_stats['tool_calls'] == 1
    assert len(requests) == 1
    assert any(e.tool_name == 'search_code' and e.succeeded for e in result.events)


@pytest.mark.parametrize('arguments,reader_calls', [
    ({'query': True, 'paths': ['app/auth.py']}, 0),
    ({'query': 'line', 'paths': []}, 0),
    ({'query': 'line', 'paths': ['app/auth.py'], 'repository_url': 'PRIVATE_URL'}, 0),
    ({'query': 'line', 'paths': ['app/auth.py', '.env']}, 1),
])
def test_bad_search_arguments_retain_runner_attempt_accounting(arguments, reader_calls):
    result, model, requests = run([
        [search_tool(arguments)], [final(insufficient=True)],
    ])
    assert result.tool_attempts == 1
    assert result.repository_stats['tool_calls'] == reader_calls
    assert result.repository_stats['file_fetch_attempts'] == 0
    assert requests == []
    assert any(e.tool_name == 'search_code' and e.succeeded is False for e in result.events)
    assert 'PRIVATE_URL' not in repr(result)


def test_search_batch_cannot_bypass_the_runner_attempt_budget():
    context, transport, requests = make_fixture()
    calls = [search_tool(call_id=f'search-{number}') for number in range(21)]
    model = FakeModel([calls, [final(insufficient=True)]])
    with pytest.raises(InvestigationRunError) as caught:
        asyncio.run(run_investigation(
            context, settings(), model_provider=Provider(model),
            repository_transport=transport,
        ))
    assert caught.value.code == 'tool_limit'
    assert len(requests) <= 1


def test_reserved_final_turn_rejects_search_before_any_file_fetch():
    context, transport, requests = make_fixture()
    model = FakeModel([[search_tool()]])
    with pytest.raises(InvestigationRunError) as caught:
        asyncio.run(run_investigation(
            context, settings(ai_max_turns=1), model_provider=Provider(model),
            repository_transport=transport,
        ))
    assert caught.value.code == 'turn_limit'
    assert model.calls[0][3] == []
    assert model.calls[0][2].tool_choice == 'none'
    assert requests == []


def test_search_failure_stays_partial_and_cannot_ground_source(monkeypatch):
    from app import repository_tools as reader_module
    from app.github_client import GitHubClientError

    def fail_read(*args, **kwargs):
        raise GitHubClientError('PRIVATE_UPSTREAM')

    monkeypatch.setattr(reader_module, 'read_repository_file', fail_read)
    result, model, requests = run([[search_tool()], [final(insufficient=True)]])
    history = json.dumps(model.calls[-1][1])
    assert 'file_unavailable' in history and 'coverage_complete' in history
    assert 'PRIVATE_UPSTREAM' not in history
    assert result.report.evidence == []
    assert result.repository_stats['file_fetch_attempts'] == 1
    assert requests == []


def test_gemini_http_search_read_and_grounded_reserved_report(report_turn_http):
    context, transport, requests = make_fixture()
    search = report_turn_completion(read=True)
    function = search['choices'][0]['message']['tool_calls'][0]['function']
    function.update(name='search_code', arguments=json.dumps({
        'query': 'line 10', 'paths': ['app/auth.py'],
    }))
    search['choices'][0]['message']['tool_calls'][0]['id'] = 'search-1'
    report = report_turn_completion()
    report['choices'][0]['message']['content'] = final().content[0].text
    report_turn_http['replies'].extend([
        (200, search), (200, report_turn_completion(read=True)), (200, report),
    ])
    result = asyncio.run(run_investigation(
        context, report_turn_settings(3), repository_transport=transport,
    ))
    assert result.report.status == 'likely_cause'
    assert result.report.evidence[0].quote == 'line 10'
    assert result.model_calls == 3 and result.tool_attempts == 2
    assert len(requests) == result.repository_stats['file_fetch_attempts'] == 1
    bodies = report_turn_http['requests']
    assert len(bodies) == 3
    assert {item['function']['name'] for item in bodies[0]['tools']} == {
        'find_files', 'read_file', 'search_code',
    }
    assert 'discovery_only' in json.dumps(bodies[1]['messages'])
    assert 'source_lines' in json.dumps(bodies[2]['messages'])
    assert not bodies[-1].get('tools') and bodies[-1]['tool_choice'] == 'none'
    assert all('store' not in body for body in bodies)
    assert report_turn_http['sleeps'] == []
