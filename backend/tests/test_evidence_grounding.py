"""Offline grounding checks using real reader output and mock GitHub blobs."""

import base64
import hashlib
from dataclasses import replace

import httpx
import pytest

from app.evidence_grounding import EvidenceGroundingError, validate_report_grounding
from app.investigation_report import InvestigationReport
from app.github_client import RepositoryMetadata, RepositorySnapshot
from app.github_oauth import GitHubIdentity
from app.github_repositories import PublicRepository
from app.github_tree import RepositoryTree
from app.investigation_context import InvestigationContext
from app.investigation_input import InvestigationInput
from app.repository_tools import MAX_TOOL_CALLS, RepositoryTools

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
def no_network(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("Live network is forbidden in grounding tests.")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", reject)


def setup(content=None, citation_line=10, quote='line 10'):
    context, transport, requests = make_fixture(content)
    reader = RepositoryTools(context, transport=transport)
    read = reader.read_file('app/auth.py')
    report = InvestigationReport(
        status='likely_cause', issue_summary='Synthetic test issue.',
        issue_category='logic', root_cause='A test hypothesis, not a verified diagnosis.',
        evidence=[dict(path='app/auth.py', start_line=citation_line,
                       end_line=citation_line, quote=quote, explanation='Synthetic citation.')],
        next_steps=['Reproduce the reported issue.'], limitations=['Not tested.'],
    )
    return report, context, read, reader, requests


def test_real_reader_quote_and_snapshot_pass():
    report, context, read, _, requests = setup()
    assert validate_report_grounding(report, context, [read]) == report
    assert len(requests) == 1


@pytest.mark.parametrize('changes', [
    {'path':'missing.py'}, {'start_line':81,'end_line':81,'quote':'line 81'},
    {'quote':'line 99'}, {'quote':'LINE 10'}, {'quote':'line 10\n'},
])
def test_unobserved_paths_lines_and_quotes_fail(changes):
    report, context, read, _, _ = setup()
    citation = report.evidence[0].model_dump()
    citation.update(changes)
    changed = InvestigationReport.model_validate({**report.model_dump(), 'evidence':[citation]})
    with pytest.raises(EvidenceGroundingError):
        validate_report_grounding(changed, context, [read])


@pytest.mark.parametrize('changes', [
    {'repository_full_name':'someone/else'}, {'commit_sha':'e'*40},
    {'blob_sha':'f'*40}, {'path':'missing.py'}, {'path':'.env'},
    {'path':'link.py'}, {'start_line':True}, {'end_line':79},
    {'truncated':False}, {'unexpected':'value'},
])
def test_inconsistent_source_records_fail(changes):
    report, context, read, _, _ = setup()
    with pytest.raises(EvidenceGroundingError):
        validate_report_grounding(report, context, [{**read, **changes}])


def test_no_source_read_cannot_ground_a_cause():
    report, context, _, _, _ = setup()
    with pytest.raises(EvidenceGroundingError):
        validate_report_grounding(report, context, [])


def test_insufficient_evidence_can_have_no_reads():
    report, context, _, _, _ = setup()
    report = InvestigationReport.model_validate({**report.model_dump(),
        'status':'insufficient_evidence', 'root_cause':None, 'evidence':[]})
    assert validate_report_grounding(report, context, []) == report


def test_partial_last_line_cannot_be_cited():
    report, context, read, _, _ = setup('head\n' + 'x'*15000,
                                      citation_line=2, quote='x')
    assert read['last_line_may_be_partial'] is True
    with pytest.raises(EvidenceGroundingError):
        validate_report_grounding(report, context, [read])


def test_complete_line_before_partial_tail_can_be_cited():
    report, context, read, _, _ = setup('head\n' + 'x'*15000,
                                      citation_line=1, quote='head')
    assert validate_report_grounding(report, context, [read]) == report


def test_empty_file_is_valid_observation_without_citations():
    report, context, read, _, _ = setup('', citation_line=1, quote='unused')
    report = InvestigationReport.model_validate({**report.model_dump(),
        'status':'insufficient_evidence', 'root_cause':None, 'evidence':[]})
    assert read['end_line'] is None
    assert validate_report_grounding(report, context, [read]) == report


def test_cached_continuation_is_valid():
    report, context, first, reader, requests = setup(citation_line=81, quote='line 81')
    second = reader.read_file('app/auth.py', 81)
    assert validate_report_grounding(report, context, [first, second]) == report
    assert len(requests) == 1


def test_context_tree_commit_mismatch_fails():
    report, context, read, _, _ = setup()
    context.tree.commit_sha = 'e'*40
    with pytest.raises(EvidenceGroundingError):
        validate_report_grounding(report, context, [read])


def test_malformed_repository_context_fails_safely():
    report, context, read, _, _ = setup()
    with pytest.raises(EvidenceGroundingError):
        validate_report_grounding(report, replace(context, repository=None), [read])


def test_mutated_report_is_revalidated():
    report, context, read, _, _ = setup()
    report.evidence.clear()
    with pytest.raises(EvidenceGroundingError):
        validate_report_grounding(report, context, [read])


def test_observation_count_is_bounded():
    report, context, read, _, _ = setup()
    with pytest.raises(EvidenceGroundingError):
        validate_report_grounding(report, context, [read]*(MAX_TOOL_CALLS+1))
