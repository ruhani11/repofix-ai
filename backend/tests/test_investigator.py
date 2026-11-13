"""Offline agent capability and investigation-isolation checks."""

import asyncio
import base64
import hashlib
import json

import httpx
import pytest
from agents import AgentOutputSchema, FunctionTool
from agents.tool_context import ToolContext
from agents.tracing import setup as tracing_setup
from agents.tracing.provider import DefaultTraceProvider

from app.github_client import RepositoryMetadata, RepositorySnapshot
from app.github_oauth import GitHubIdentity
from app.github_repositories import PublicRepository
from app.github_tree import RepositoryTree
from app.investigation_context import InvestigationContext
from app.investigation_input import InvestigationInput
from app.investigation_report import InvestigationReport
from app.investigator import INVESTIGATOR_INSTRUCTIONS, build_investigator_agent
from app.repository_tools import RepositoryTools


@pytest.fixture(autouse=True)
def isolate_network_and_tracing(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("An Investigator test attempted a real network request.")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", reject)
    monkeypatch.setattr(
        tracing_setup, "GLOBAL_TRACE_PROVIDER", DefaultTraceProvider()
    )


def make_reader(name="alpha", commit_sha="a" * 40):
    data = f"value = '{name}'".encode("utf-8")
    blob_sha = hashlib.sha1(
        f"blob {len(data)}\0".encode("ascii") + data,
        usedforsecurity=False,
    ).hexdigest()
    repository_id = 5 if name == "alpha" else 6
    identity = GitHubIdentity(id=123, login="demo-user", type="User")
    repository = PublicRepository(
        id=repository_id, name=name, full_name=f"demo-user/{name}", owner=identity,
        private=False, visibility="public", default_branch="main",
        fork=False, archived=False, disabled=False,
    )
    snapshot = RepositorySnapshot(
        repository=RepositoryMetadata(**repository.model_dump()),
        commit_sha=commit_sha,
    )
    tree = RepositoryTree(
        repository_full_name=repository.full_name,
        commit_sha=commit_sha,
        tree_sha="c" * 40,
        entries=[{
            "path": "app/main.py", "type": "blob", "mode": "100644",
            "sha": blob_sha, "size": len(data),
        }],
    )
    context = InvestigationContext(
        identity=identity,
        request=InvestigationInput(
            repository_id=repository_id, repository_name=name,
            issue_description="Synthetic isolation test.",
        ),
        repository=repository,
        snapshot=snapshot,
        tree=tree,
    )
    requests = []

    def handler(request):
        assert request.method == "GET"
        assert request.url.host == "api.github.com"
        assert request.url.path == f"/repos/demo-user/{name}/git/blobs/{blob_sha}"
        requests.append(request.url.path)
        return httpx.Response(200, json={
            "sha": blob_sha, "size": len(data), "encoding": "base64",
            "content": base64.b64encode(data).decode("ascii"),
        })

    return RepositoryTools(context, transport=httpx.MockTransport(handler)), requests


def invoke_read(agent, context_data=None):
    tool = next(item for item in agent.tools if item.name == "read_file")
    payload = json.dumps({"path": "app/main.py", "start_line": 1})
    context = ToolContext(
        context=context_data, tool_name=tool.name,
        tool_call_id="call-test", tool_arguments=payload,
    )
    return json.loads(asyncio.run(tool.on_invoke_tool(context, payload)))


def test_construction_is_read_only_structured_and_performs_no_reads():
    reader, requests = make_reader()
    agent = build_investigator_agent(reader, model=" offline-test-model ")

    assert agent.model == "offline-test-model"
    assert {tool.name for tool in agent.tools} == {"find_files", "read_file"}
    assert all(isinstance(tool, FunctionTool) for tool in agent.tools)
    assert agent.handoffs == []
    assert agent.mcp_servers == []
    assert agent.output_type is InvestigationReport
    assert AgentOutputSchema(agent.output_type).is_strict_json_schema() is True
    assert agent.model_settings.store is False
    assert agent.model_settings.parallel_tool_calls is False
    assert agent.model_settings.max_tokens is not None
    assert agent.model_settings.max_tokens > 0
    assert reader.stats["tool_calls"] == 0
    assert requests == []


def test_repository_scope_and_cache_do_not_mix_between_investigations():
    alpha_reader, alpha_requests = make_reader()
    beta_reader, beta_requests = make_reader("beta", "b" * 40)
    alpha = build_investigator_agent(alpha_reader, model="offline-test-model")
    beta = build_investigator_agent(beta_reader, model="offline-test-model")

    # SDK local context cannot replace the repository bound by the factory.
    first = invoke_read(alpha, {"repository_full_name": "demo-user/beta"})
    assert beta_reader.stats["tool_calls"] == 0
    second = invoke_read(beta, {"repository_full_name": "demo-user/alpha"})
    cached = invoke_read(alpha)

    assert first["ok"] is True and second["ok"] is True
    assert first["data"]["repository_full_name"] == "demo-user/alpha"
    assert second["data"]["repository_full_name"] == "demo-user/beta"
    assert first["data"]["commit_sha"] == "a" * 40
    assert second["data"]["commit_sha"] == "b" * 40
    assert first["data"]["content"] == "value = 'alpha'"
    assert second["data"]["content"] == "value = 'beta'"
    assert cached == first
    assert len(alpha_requests) == len(beta_requests) == 1
    assert alpha_reader.stats["tool_calls"] == 2
    assert beta_reader.stats["tool_calls"] == 1


@pytest.mark.parametrize("model", [
    None, 123, True, "", "   ", "https://example.com/model", "two models", "x" * 101,
])
def test_invalid_model_input_is_rejected_without_reads(model):
    reader, requests = make_reader()

    with pytest.raises(ValueError, match="model identifier"):
        build_investigator_agent(reader, model=model)

    assert reader.stats["tool_calls"] == 0
    assert requests == []


def test_prepared_reader_is_required():
    with pytest.raises(ValueError, match="prepared RepositoryTools"):
        build_investigator_agent(None, model="offline-test-model")


def invoke_search(agent, context_data=None):
    tool = next(item for item in agent.tools if item.name == "search_code")
    payload = json.dumps({"query": "value", "paths": ["app/main.py"]})
    context = ToolContext(
        context=context_data, tool_name=tool.name,
        tool_call_id="call-search-test", tool_arguments=payload,
    )
    return json.loads(asyncio.run(tool.on_invoke_tool(context, payload)))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("model", ["offline-test-model", "gemini-offline-test"])
def test_optional_search_keeps_capabilities_and_guidance_aligned(enabled, model):
    reader, requests = make_reader()
    agent = build_investigator_agent(
        reader, model=model, include_code_search=enabled,
    )

    expected_tools = {"find_files", "read_file"}
    if enabled:
        expected_tools.add("search_code")
        assert agent.instructions.startswith(INVESTIGATOR_INSTRUCTIONS)
        assert "search_code" in agent.instructions
        assert "Never construct a report quote from a search preview." in agent.instructions
        assert "coverage_complete refers only to selected files." in agent.instructions
    else:
        assert agent.instructions == INVESTIGATOR_INSTRUCTIONS
        assert "search_code" not in agent.instructions

    assert {tool.name for tool in agent.tools} == expected_tools
    assert all(isinstance(tool, FunctionTool) for tool in agent.tools)
    assert all(tool.strict_json_schema for tool in agent.tools)
    assert agent.model == model
    assert agent.output_type is InvestigationReport
    assert AgentOutputSchema(agent.output_type).is_strict_json_schema() is True
    assert agent.model_settings.parallel_tool_calls is False
    assert agent.model_settings.max_tokens == 4096
    expected_store = None if model.startswith("gemini-") else False
    assert agent.model_settings.store is expected_store
    assert agent.handoffs == agent.mcp_servers == []
    assert reader.stats == {
        "tool_calls": 0, "file_fetch_attempts": 0, "downloaded_bytes": 0,
    }
    assert requests == []


@pytest.mark.parametrize("flag", [None, 1, "true"])
def test_optional_search_rejects_nonboolean_flags_before_reader_work(flag):
    reader, requests = make_reader()

    with pytest.raises(ValueError, match="boolean"):
        build_investigator_agent(
            reader, model="offline-test-model", include_code_search=flag,
        )

    assert reader.stats == {
        "tool_calls": 0, "file_fetch_attempts": 0, "downloaded_bytes": 0,
    }
    assert requests == []


def test_search_and_reads_keep_repository_scope_and_cache_isolated():
    alpha_reader, alpha_requests = make_reader()
    beta_reader, beta_requests = make_reader("beta", "b" * 40)
    alpha = build_investigator_agent(
        alpha_reader, model="offline-test-model", include_code_search=True,
    )
    beta = build_investigator_agent(
        beta_reader, model="offline-test-model", include_code_search=True,
    )

    # Local SDK context cannot change the server-prepared repository or commit.
    first = invoke_search(alpha, {
        "repository_full_name": "demo-user/beta", "commit_sha": "b" * 40,
    })
    assert beta_reader.stats["tool_calls"] == 0
    second = invoke_search(beta, {
        "repository_full_name": "demo-user/alpha", "commit_sha": "a" * 40,
    })

    assert first["ok"] is second["ok"] is True
    assert first["source_kind"] == second["source_kind"] == "untrusted_repository_data"
    assert first["data"]["repository_full_name"] == "demo-user/alpha"
    assert second["data"]["repository_full_name"] == "demo-user/beta"
    assert first["data"]["commit_sha"] == "a" * 40
    assert second["data"]["commit_sha"] == "b" * 40
    assert first["data"]["result_kind"] == second["data"]["result_kind"] == "discovery_only"
    assert first["data"]["matches"][0]["preview"] == "value = 'alpha'"
    assert second["data"]["matches"][0]["preview"] == "value = 'beta'"

    read = invoke_read(alpha)
    assert read["data"]["blob_sha"] == first["data"]["matches"][0]["blob_sha"]
    assert read["data"]["content"] == "value = 'alpha'"
    assert len(alpha_requests) == len(beta_requests) == 1
    assert alpha_reader.stats["file_fetch_attempts"] == beta_reader.stats["file_fetch_attempts"] == 1
    assert alpha_reader.stats["tool_calls"] == 2
    assert beta_reader.stats["tool_calls"] == 1


def test_enabling_search_does_not_mutate_other_agents_or_global_instructions():
    reader, requests = make_reader()
    default = build_investigator_agent(reader, model="offline-test-model")
    enabled = build_investigator_agent(
        reader, model="offline-test-model", include_code_search=True,
    )
    another = build_investigator_agent(reader, model="offline-test-model")

    assert default.instructions == another.instructions == INVESTIGATOR_INSTRUCTIONS
    assert "search_code" not in INVESTIGATOR_INSTRUCTIONS
    assert {tool.name for tool in default.tools} == {"find_files", "read_file"}
    assert {tool.name for tool in another.tools} == {"find_files", "read_file"}
    assert {tool.name for tool in enabled.tools} == {"find_files", "read_file", "search_code"}
    assert default.tools[0] is not enabled.tools[0]
    assert enabled.tools[0] is not another.tools[0]
    assert reader.stats["tool_calls"] == 0
    assert requests == []
