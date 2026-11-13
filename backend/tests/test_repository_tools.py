"""Offline checks for pinned reads, output limits, caching and tool budgets."""

import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from app import repository_tools as module
from app.github_client import (
    GitHubClientError, RepositoryMetadata, RepositorySnapshot,
)
from app.github_oauth import GitHubIdentity
from app.github_repositories import PublicRepository
from app.github_tree import RepositoryTree
from app.investigation_context import InvestigationContext
from app.investigation_input import InvestigationInput
from app.repository_tools import RepositoryToolError, RepositoryTools


@pytest.fixture(autouse=True)
def block_real_network(monkeypatch):
    def reject_request(*args, **kwargs):
        pytest.fail("A repository tool test attempted a real network request.")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject_request)


def make_fixture(content=None, *, fail=False):
    text = content if content is not None else "\n".join(
        f"line {number}" for number in range(1, 101)
    )
    data = text.encode("utf-8")
    header = f"blob {len(data)}\0".encode("ascii")
    blob_sha = hashlib.sha1(header + data, usedforsecurity=False).hexdigest()
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
    entries = [
        {"path": path, "type": "blob", "mode": mode,
         "sha": blob_sha, "size": len(data)}
        for path, mode in (
            ("app/auth.py", "100644"),
            (".env", "100644"),
            ("link.py", "120000"),
        )
    ]
    tree = RepositoryTree(
        repository_full_name=repository.full_name,
        commit_sha=snapshot.commit_sha,
        tree_sha="b" * 40,
        entries=entries,
    )
    context = InvestigationContext(
        identity=identity,
        request=InvestigationInput(
            repository_id=5, repository_name="example",
            issue_description="Login fails.",
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
        if fail:
            return httpx.Response(429)
        return httpx.Response(200, json={
            "sha": blob_sha,
            "size": len(data),
            "encoding": "base64",
            "content": base64.b64encode(data).decode("ascii"),
        })

    return context, httpx.MockTransport(handler), requests, len(data)


def test_file_discovery_excerpts_and_cache_share_fixed_commit():
    context, transport, requests, size = make_fixture()
    tools = RepositoryTools(context, transport=transport)

    matches = tools.find_files("AUTH")
    assert [item["path"] for item in matches["files"]] == ["app/auth.py"]

    first = tools.read_file("app/auth.py")
    second = tools.read_file("app/auth.py", 81)

    assert first["start_line"] == 1 and first["end_line"] == 80
    assert first["truncated"] is True
    assert second["content"].startswith("line 81")
    assert second["end_line"] == 100 and second["truncated"] is False
    assert first["commit_sha"] == second["commit_sha"] == context.snapshot.commit_sha
    assert len(requests) == 1
    assert tools.stats == {
        "tool_calls": 3, "file_fetch_attempts": 1, "downloaded_bytes": size,
    }


@pytest.mark.parametrize("path", [
    ".env", "link.py", "../auth.py", "https://example.com", "missing.py",
])
def test_blocked_paths_never_fetch(path):
    context, transport, requests, _ = make_fixture()
    tools = RepositoryTools(context, transport=transport)

    with pytest.raises(RepositoryToolError):
        tools.read_file(path)

    assert requests == []


@pytest.mark.parametrize("line", [0, True, "1"])
def test_invalid_start_line_never_fetches(line):
    context, transport, requests, _ = make_fixture()
    tools = RepositoryTools(context, transport=transport)

    with pytest.raises(RepositoryToolError):
        tools.read_file("app/auth.py", line)

    assert requests == []


def test_invalid_calls_consume_call_budget():
    context, transport, requests, _ = make_fixture()
    tools = RepositoryTools(context, transport=transport)

    for _ in range(module.MAX_TOOL_CALLS):
        with pytest.raises(RepositoryToolError):
            tools.find_files(None)

    with pytest.raises(RepositoryToolError, match="call limit"):
        tools.find_files()

    assert requests == []


def test_byte_budget_is_checked_before_download(monkeypatch):
    context, transport, requests, size = make_fixture()
    tools = RepositoryTools(context, transport=transport)
    monkeypatch.setattr(module, "MAX_TOTAL_FILE_BYTES", size - 1)

    with pytest.raises(RepositoryToolError, match="download budget"):
        tools.read_file("app/auth.py")

    assert requests == []


def test_failed_downloads_consume_fetch_attempt_budget():
    context, transport, requests, _ = make_fixture(fail=True)
    tools = RepositoryTools(context, transport=transport)

    for _ in range(module.MAX_FILE_FETCH_ATTEMPTS):
        with pytest.raises(GitHubClientError):
            tools.read_file("app/auth.py")

    with pytest.raises(RepositoryToolError, match="fetch attempt"):
        tools.read_file("app/auth.py")

    assert len(requests) == module.MAX_FILE_FETCH_ATTEMPTS
    assert tools.stats["downloaded_bytes"] == 0


def test_long_line_is_clipped_and_marked_partial():
    context, transport, _, _ = make_fixture("x" * 15000)
    response = RepositoryTools(context, transport=transport).read_file("app/auth.py")

    assert len(response["content"]) == module.MAX_EXCERPT_CHARACTERS
    assert response["last_line_may_be_partial"] is True
    assert response["truncated"] is True
    assert response["end_line"] == 1


def test_empty_file_returns_empty_excerpt():
    context, transport, _, _ = make_fixture("")
    response = RepositoryTools(context, transport=transport).read_file("app/auth.py")

    assert response["end_line"] is None and response["total_lines"] == 0
    assert response["content"] == "" and response["truncated"] is False


def test_context_tree_changes_cannot_rebind_existing_tools():
    context, transport, _, _ = make_fixture()
    tools = RepositoryTools(context, transport=transport)
    context.tree.commit_sha = "e" * 40
    context.tree.entries.clear()

    assert tools.find_files()["commit_sha"] == "a" * 40
    assert tools.read_file("app/auth.py")["commit_sha"] == "a" * 40


def test_file_matching_output_is_limited():
    context, transport, _, _ = make_fixture()
    template = context.tree.entries[0]
    context.tree.entries = [
        template.model_copy(update={"path": f"file{number:03}.py"})
        for number in range(55)
    ]
    response = RepositoryTools(context, transport=transport).find_files()

    assert len(response["files"]) == module.MAX_FILE_MATCHES
    assert response["matching_count"] == 55 and response["truncated"] is True


@pytest.mark.parametrize("problem", ["commit", "duplicate_path"])
def test_inconsistent_context_is_rejected(problem):
    context, transport, requests, _ = make_fixture()
    if problem == "commit":
        context.tree.commit_sha = "e" * 40
    else:
        context.tree.entries.append(context.tree.entries[0])

    with pytest.raises(ValueError):
        RepositoryTools(context, transport=transport)

    assert requests == []


def test_line_beyond_end_of_file_is_rejected():
    context, transport, _, _ = make_fixture()
    tools = RepositoryTools(context, transport=transport)

    with pytest.raises(RepositoryToolError, match="line count"):
        tools.read_file("app/auth.py", 101)


def test_parallel_calls_cannot_exceed_shared_budget():
    context, transport, requests, _ = make_fixture()
    tools = RepositoryTools(context, transport=transport)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda _: tools.find_files(), range(module.MAX_TOOL_CALLS)
        ))

    assert len(results) == module.MAX_TOOL_CALLS
    assert tools.stats["tool_calls"] == module.MAX_TOOL_CALLS
    with pytest.raises(RepositoryToolError, match="call limit"):
        tools.find_files()
    assert requests == []


def make_search_tools(content=None, paths=("a.py", "b.py"), *, fail=False):
    """Reuse the synthetic pinned blob across explicitly selected file paths."""
    context, transport, requests, size = make_fixture(content, fail=fail)
    template = context.tree.entries[0]
    context.tree.entries = [
        template.model_copy(update={"path": path}) for path in paths
    ]
    return RepositoryTools(context, transport=transport), requests, size


def test_search_returns_absolute_lines_and_shares_the_read_cache():
    text = "\n".join(f"row_{number}" for number in range(1, 1700))
    text += "\n  Validate_Submission_Answers()"
    tools, requests, size = make_search_tools(text)
    found = tools.search_code("validate_submission_answers", ["b.py"])

    assert found["coverage_complete"] is True
    assert found["truncated"] is False
    assert found["matches"][0]["line_number"] == 1700
    assert found["matches"][0]["preview"] == "  Validate_Submission_Answers()"
    assert found["commit_sha"] == "a" * 40
    assert found["result_kind"] == "discovery_only"
    assert len(found["matches"][0]["blob_sha"]) == 40
    read = tools.read_file("b.py", 1700)
    assert read["content"] == found["matches"][0]["preview"]
    assert read["blob_sha"] == found["matches"][0]["blob_sha"]
    assert tools.stats == {
        "tool_calls": 2, "file_fetch_attempts": 1, "downloaded_bytes": size,
    }
    assert len(requests) == 1


def test_search_uses_literal_casefold_matching_and_counts_lines():
    tools, _, _ = make_search_tools("x.*[a-z] x.*[a-z]\nxyz\nSTRASSE")
    result = tools.search_code(".*[a-z]", ["a.py"])

    assert result["matching_line_count"] == 1
    assert result["matches"][0]["line_number"] == 1
    unicode_result = tools.search_code("straße", ["a.py"])
    assert unicode_result["matches"][0]["line_number"] == 3


@pytest.mark.parametrize("query", [
    None, True, "", "  ", "x" * 121, "x\ny", "x\ty", "x\x7fy",
])
def test_invalid_search_query_is_rejected_before_any_fetch(query):
    tools, requests, _ = make_search_tools()

    with pytest.raises(RepositoryToolError):
        tools.search_code(query, ["a.py"])
    assert requests == []
    assert tools.stats["tool_calls"] == 1
    assert tools.stats["file_fetch_attempts"] == 0


@pytest.mark.parametrize("paths", [
    None, (), [], ["a.py", "a.py"], ["missing.py"], ["a.py", {}],
    ["a.py", ".env"], ["../a.py"], ["https://example.com/a.py"],
    ["a.py", "b.py", "c.py", "d.py", "e.py", "f.py"],
])
def test_invalid_or_blocked_search_scope_is_rejected_before_any_fetch(paths):
    tools, requests, _ = make_search_tools(paths=("a.py", "b.py", ".env"))

    with pytest.raises(RepositoryToolError):
        tools.search_code("line", paths)
    assert requests == []
    assert tools.stats["file_fetch_attempts"] == 0


def test_search_match_limit_preserves_selected_scope_count_and_coverage():
    tools, requests, _ = make_search_tools(
        "\n".join("needle needle" for _ in range(30))
    )
    result = tools.search_code("needle", ["b.py", "a.py"])

    assert len(result["matches"]) == module.MAX_SEARCH_MATCHES
    assert result["matching_line_count"] == 60
    assert result["truncated"] is True
    assert result["coverage_complete"] is True
    assert result["searched_paths"] == ["b.py", "a.py"]
    assert result["unsearched_paths"] == []
    assert len(requests) == 2


def test_search_long_line_preview_does_not_claim_to_show_the_whole_line():
    tools, _, _ = make_search_tools("x" * 1000 + "needle")
    result = tools.search_code("needle", ["a.py"])
    match = result["matches"][0]

    assert match["line_number"] == 1
    assert match["preview_truncated"] is True
    assert len(match["preview"]) == module.MAX_SEARCH_PREVIEW_CHARACTERS
    assert result["coverage_complete"] is True


def test_search_empty_file_has_complete_selected_coverage_and_zero_matches():
    tools, requests, _ = make_search_tools("")
    result = tools.search_code("needle", ["a.py"])

    assert result["matching_line_count"] == 0
    assert result["matches"] == []
    assert result["coverage_complete"] is True
    assert result["truncated"] is False
    assert len(requests) == 1


def test_search_byte_budget_keeps_partial_matches_and_cached_reads(monkeypatch):
    tools, requests, size = make_search_tools("needle")
    monkeypatch.setattr(module, "MAX_TOTAL_FILE_BYTES", size)
    result = tools.search_code("needle", ["a.py", "b.py"])

    assert result["searched_paths"] == ["a.py"]
    assert result["unsearched_paths"] == ["b.py"]
    assert result["coverage_complete"] is False
    assert result["truncated"] is True
    assert result["errors"] == [{"path": "b.py", "code": "read_budget_exhausted"}]
    assert result["matching_line_count"] == 1
    assert tools.read_file("a.py")["content"] == "needle"
    assert len(requests) == 1
    with pytest.raises(RepositoryToolError, match="download budget"):
        tools.read_file("b.py")


def test_search_fetch_failures_cannot_masquerade_as_exhaustive_no_matches():
    tools, requests, _ = make_search_tools("needle", fail=True)
    result = tools.search_code("needle", ["a.py", "b.py"])

    assert result["searched_paths"] == []
    assert result["unsearched_paths"] == ["a.py", "b.py"]
    assert result["coverage_complete"] is False
    assert result["truncated"] is True
    assert all(error["code"] == "file_unavailable" for error in result["errors"])
    assert len(requests) == 2
    assert tools.stats["file_fetch_attempts"] == 2
    assert tools.stats["downloaded_bytes"] == 0


def test_search_and_read_share_failed_fetch_attempt_budget(monkeypatch):
    tools, requests, _ = make_search_tools("needle", fail=True)
    monkeypatch.setattr(module, "MAX_FILE_FETCH_ATTEMPTS", 1)
    result = tools.search_code("needle", ["a.py", "b.py"])

    assert len(requests) == 1
    assert result["coverage_complete"] is False
    assert [error["code"] for error in result["errors"]] == [
        "file_unavailable", "read_budget_exhausted",
    ]
    with pytest.raises(RepositoryToolError, match="fetch attempt"):
        tools.read_file("b.py")


def test_search_one_failed_file_keeps_later_matches_and_redacts_provider_text():
    tools, _, size = make_search_tools("needle")
    original_transport = tools._transport
    attempts = []

    def handler(request):
        attempts.append(request.url.path)
        if len(attempts) == 1:
            return httpx.Response(503, json={"message": "PRIVATE_PROVIDER_PAYLOAD"})
        return original_transport.handle_request(request)

    tools._transport = httpx.MockTransport(handler)
    result = tools.search_code("needle", ["a.py", "b.py"])

    assert result["searched_paths"] == ["b.py"]
    assert result["unsearched_paths"] == ["a.py"]
    assert result["matches"][0]["path"] == "b.py"
    assert result["matching_line_count"] == 1
    assert result["coverage_complete"] is False
    assert result["truncated"] is True
    assert result["errors"] == [{"path": "a.py", "code": "file_unavailable"}]
    assert "PRIVATE_" not in json.dumps(result)
    assert tools.stats["file_fetch_attempts"] == 2
    assert tools.stats["downloaded_bytes"] == size


def test_parallel_search_calls_share_call_budget_and_one_cached_blob():
    tools, requests, _ = make_search_tools("needle")
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda _: tools.search_code("needle", ["a.py"]),
            range(module.MAX_TOOL_CALLS),
        ))

    assert len(results) == module.MAX_TOOL_CALLS
    assert len(requests) == 1
    with pytest.raises(RepositoryToolError, match="call limit"):
        tools.find_files()


def test_search_escaped_paths_and_previews_fit_runner_response_bound():
    from app.investigation_runner import MAX_TOOL_RESULT_CHARACTERS

    paths = ['"' * 1000 + str(number) + ".py" for number in range(5)]
    content = "\n".join('needle' + '"\\' * 300 for _ in range(20))
    tools, _, _ = make_search_tools(content, paths=paths)
    result = tools.search_code("needle", paths)

    assert len(result["matches"]) == module.MAX_SEARCH_MATCHES
    assert len(json.dumps(result, ensure_ascii=False)) < MAX_TOOL_RESULT_CHARACTERS
