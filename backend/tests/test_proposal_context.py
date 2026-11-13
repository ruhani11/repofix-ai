"""Account-scoped saved-SHA preparation with SQLite and mocked GitHub HTTP."""

import base64
import hashlib
from unittest.mock import Mock

import httpx
import pytest

from app.fix_proposal import FixProposalInput
from app.github_oauth import GitHubIdentity
from app.investigation_history import InvestigationRunResponse
from app.investigation_input import InvestigationInput
from app.investigation_report import InvestigationReport
from app.investigation_runner import InvestigationEvent
from app.investigation_store import InvestigationStore, InvestigationStoreError
from app import proposal_context as preparation
from app.proposal_context import ProposalContextError, prepare_proposal_context
from app.repository_tools import RepositoryTools


IDENTITY = GitHubIdentity(id=123, login="demo-user", type="User")
RUN_ID = "a" * 32
COMMIT_SHA = "b" * 40
TREE_SHA = "c" * 40
SOURCE = "def accept(quantity):\n    return quantity > 0\n"
BLOB_SHA = hashlib.sha1(
    f"blob {len(SOURCE.encode())}\0".encode() + SOURCE.encode(), usedforsecurity=False,
).hexdigest()
PREFIX = "/repos/demo-user/example"
PAYLOAD = FixProposalInput(investigation_run_id=RUN_ID)
ISSUE = InvestigationInput(repository_id=5, repository_name="example", issue_description="Quantity one is rejected.")


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("A proposal-context test attempted a real network request.")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", reject)


def saved_response(*, status="likely_cause"):
    return InvestigationRunResponse(
        run_id=RUN_ID, repository_full_name="demo-user/example", commit_sha=COMMIT_SHA,
        report=InvestigationReport(
            status=status, issue_summary="Quantity one is rejected.", issue_category="logic",
            root_cause="The lower bound excludes one." if status == "likely_cause" else None,
            evidence=[{
                "path": "src/orders.py", "start_line": 2, "end_line": 2,
                "quote": "    return quantity > 0", "explanation": "A synthetic source fixture.",
            }] if status == "likely_cause" else [],
            next_steps=["Prepare a candidate and regression tests."],
            limitations=["No runtime check; this is a synthetic fixture."],
        ),
        total_duration_ms=1, agent_duration_ms=1, model_calls=1, tool_attempts=1,
        repository_stats={}, events=[
            InvestigationEvent("run_started", 0),
            InvestigationEvent("tool_completed", 1, "read_file", True),
            InvestigationEvent("grounding_validated", 1),
        ], history_saved=True,
    )


@pytest.fixture
def store():
    instance = InvestigationStore(":memory:")
    instance.save(IDENTITY.id, ISSUE, saved_response())
    yield instance
    instance.close()


def metadata(**changes):
    return {
        "id": 5, "name": "example", "full_name": "demo-user/example",
        "owner": IDENTITY.model_dump(), "private": False, "visibility": "public",
        "default_branch": "main", "fork": False, "archived": False, "disabled": False,
        **changes,
    }


def github_transport(*, first=None, final=None, commit_sha=COMMIT_SHA, tree_sha=TREE_SHA,
                     truncated=False, duplicate=False, status_at=None, identity=IDENTITY):
    paths = []
    reads = 0
    requested_prefix = f"/repos/{identity.login}/example"

    def handler(request):
        nonlocal reads
        assert request.method == "GET"
        assert request.url.host == "api.github.com"
        assert "authorization" not in request.headers and "cookie" not in request.headers
        path = request.url.path
        paths.append(path)
        if status_at is not None and path.casefold() == status_at[0].casefold():
            return httpx.Response(status_at[1], json={"message": "PRIVATE_UPSTREAM_DETAIL"})
        if path.casefold() == requested_prefix.casefold():
            reads += 1
            body = (first if reads == 1 else final) or metadata()
        elif path.casefold() == f"{PREFIX}/git/commits/{COMMIT_SHA}".casefold():
            body = {"sha": commit_sha, "tree": {"sha": TREE_SHA}}
        elif path.casefold() == f"{PREFIX}/git/trees/{TREE_SHA}".casefold():
            assert request.url.params["recursive"] == "1"
            entry = {"path": "src/orders.py", "type": "blob", "mode": "100644", "sha": BLOB_SHA, "size": len(SOURCE.encode())}
            body = {"sha": tree_sha, "truncated": truncated, "tree": [entry, entry] if duplicate else [entry]}
        elif path.casefold() == f"{PREFIX}/git/blobs/{BLOB_SHA}".casefold():
            body = {"sha": BLOB_SHA, "size": len(SOURCE.encode()), "encoding": "base64", "content": base64.b64encode(SOURCE.encode()).decode()}
        else:
            pytest.fail(f"Unexpected mock request (including any HEAD/ref read): {path}")
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler), paths


def assert_error(store, transport, code, *, identity=IDENTITY, payload=PAYLOAD):
    with pytest.raises(ProposalContextError) as caught:
        prepare_proposal_context(identity, payload, store, transport=transport)
    assert caught.value.code == code
    assert str(caught.value) == "Could not prepare the saved investigation for a fix proposal."


def test_preparation_loads_by_session_account_and_uses_saved_sha_without_head_reads(store, monkeypatch):
    transport, paths = github_transport()
    get = Mock(wraps=store.get)
    monkeypatch.setattr(store, "get", get)
    context = prepare_proposal_context(IDENTITY, PAYLOAD, store, transport=transport)
    get.assert_called_once_with(IDENTITY.id, RUN_ID)
    source = context.repository_context
    assert context.history.investigation.run_id == RUN_ID
    assert source.identity == IDENTITY and source.request == ISSUE
    assert source.repository.id == context.history.repository_id == 5
    assert source.snapshot.commit_sha == source.tree.commit_sha == COMMIT_SHA
    assert source.tree.tree_sha == TREE_SHA
    assert paths == [PREFIX, f"{PREFIX}/git/commits/{COMMIT_SHA}", f"{PREFIX}/git/trees/{TREE_SHA}", PREFIX]
    assert not any("/blobs/" in path or "/ref/" in path or "/heads/" in path for path in paths)


def test_prepared_context_reuses_existing_bounded_reader_on_the_pinned_tree(store):
    transport, paths = github_transport()
    context = prepare_proposal_context(IDENTITY, PAYLOAD, store, transport=transport)
    reader = RepositoryTools(context.repository_context, transport=transport)
    matches = reader.find_files("orders")["files"]
    assert matches == [{"path": "src/orders.py", "size_bytes": len(SOURCE.encode())}]
    result = reader.read_file("src/orders.py")
    assert result["commit_sha"] == COMMIT_SHA
    assert result["content"].startswith("def accept(quantity):")
    assert paths[-1] == f"{PREFIX}/git/blobs/{BLOB_SHA}"


def test_saved_history_event_dataclasses_survive_deep_revalidation(store):
    stored = store.get(IDENTITY.id, RUN_ID)
    assert len(stored.investigation.events) == 3
    transport, _ = github_transport()
    prepared = prepare_proposal_context(IDENTITY, PAYLOAD, store, transport=transport)
    assert prepared.history == stored
    assert all(isinstance(event, InvestigationEvent) for event in prepared.history.investigation.events)
    assert prepared.history.created_at == stored.created_at
    assert prepared.history.investigation.events[1].succeeded is True


@pytest.mark.parametrize("missing", ["own_missing", "other_account"])
def test_missing_and_other_accounts_runs_are_indistinguishable_before_github(store, missing):
    transport, paths = github_transport()
    identity = IDENTITY if missing == "own_missing" else GitHubIdentity(id=456, login="other-user", type="User")
    payload = FixProposalInput(investigation_run_id="f" * 32) if missing == "own_missing" else PAYLOAD
    assert_error(store, transport, "not_found", identity=identity, payload=payload)
    assert paths == []


def test_history_storage_failure_is_safe_and_stops_before_github(store, monkeypatch):
    transport, paths = github_transport()
    monkeypatch.setattr(store, "get", Mock(side_effect=InvestigationStoreError("PRIVATE_DATABASE_PATH")))
    assert_error(store, transport, "history_unavailable")
    assert paths == []


def test_insufficient_evidence_cannot_start_proposal_source_preparation():
    store = InvestigationStore(":memory:")
    try:
        store.save(IDENTITY.id, ISSUE, saved_response(status="insufficient_evidence"))
        transport, paths = github_transport()
        assert_error(store, transport, "ineligible_investigation")
        assert paths == []
    finally:
        store.close()


@pytest.mark.parametrize("identity,payload", [
    (None, PAYLOAD), (IDENTITY, None),
    (IDENTITY.model_copy(update={"id": True}), PAYLOAD),
    (IDENTITY.model_copy(update={"login": "../other"}), PAYLOAD),
    (IDENTITY, PAYLOAD.model_copy(update={"investigation_run_id": "g" * 32})),
])
def test_invalid_identity_or_request_stops_before_storage_and_network(store, monkeypatch, identity, payload):
    get = Mock(wraps=store.get)
    monkeypatch.setattr(store, "get", get)
    transport, paths = github_transport()
    assert_error(store, transport, "invalid_input", identity=identity, payload=payload)
    get.assert_not_called()
    assert paths == []


def test_non_store_input_is_rejected_before_network():
    transport, paths = github_transport()
    assert_error(object(), transport, "invalid_input")
    assert paths == []


@pytest.mark.parametrize("invalid", ["wrong_type", "wrong_run", "bad_repository_name", "bad_repository_id", "not_saved"])
def test_returned_history_is_revalidated_before_network(store, monkeypatch, invalid):
    record = store.get(IDENTITY.id, RUN_ID)
    expected = "invalid_history"
    if invalid == "wrong_type":
        record = record.model_dump()
    elif invalid == "wrong_run":
        record = record.model_copy(update={"investigation": record.investigation.model_copy(update={"run_id": "f" * 32})})
    elif invalid == "bad_repository_name":
        record = record.model_copy(update={"repository_name": "../private"})
    elif invalid == "bad_repository_id":
        record = record.model_copy(update={"repository_id": True})
    else:
        record = record.model_copy(update={"investigation": record.investigation.model_copy(update={"history_saved": False})})
        expected = "ineligible_investigation"
    monkeypatch.setattr(store, "get", Mock(return_value=record))
    transport, paths = github_transport()
    assert_error(store, transport, expected)
    assert paths == []


@pytest.mark.parametrize("when", ["first", "final"])
@pytest.mark.parametrize("change", [
    {"id": 6}, {"owner": {**IDENTITY.model_dump(), "id": 456}},
    {"disabled": True}, {"name": "renamed", "full_name": "demo-user/renamed"},
])
def test_ownership_and_repository_identity_are_rechecked_around_tree_reads(store, when, change):
    transport, paths = github_transport(**{when: metadata(**change)})
    assert_error(store, transport, "selection_changed")
    assert len(paths) == (1 if when == "first" else 4)


@pytest.mark.parametrize("changes", [{"private": True}, {"visibility": "private"}])
def test_private_or_unsupported_repository_metadata_is_rejected(store, changes):
    transport, paths = github_transport(first=metadata(**changes))
    assert_error(store, transport, "github_unavailable")
    assert paths == [PREFIX]


@pytest.mark.parametrize("path,status", [
    (PREFIX, 404), (PREFIX, 429),
    (f"{PREFIX}/git/commits/{COMMIT_SHA}", 503),
    (f"{PREFIX}/git/trees/{TREE_SHA}", 429),
])
def test_upstream_failures_are_controlled_without_provider_payloads(store, path, status):
    transport, _ = github_transport(status_at=(path, status))
    assert_error(store, transport, "github_unavailable")


@pytest.mark.parametrize("changes", [
    {"commit_sha": "f" * 40}, {"tree_sha": "f" * 40}, {"truncated": True}, {"duplicate": True},
])
def test_wrong_commit_or_tree_and_incomplete_or_duplicate_tree_cannot_prepare(store, changes):
    transport, paths = github_transport(**changes)
    assert_error(store, transport, "github_unavailable")
    assert paths.count(PREFIX) == 1


def test_default_branch_changes_do_not_move_the_saved_commit(store):
    transport, paths = github_transport(final=metadata(default_branch="develop"))
    context = prepare_proposal_context(IDENTITY, PAYLOAD, store, transport=transport)
    assert context.repository_context.repository.default_branch == "develop"
    assert context.repository_context.snapshot.commit_sha == COMMIT_SHA
    assert context.repository_context.tree.commit_sha == COMMIT_SHA
    assert not any("/heads/" in path for path in paths)


@pytest.mark.parametrize("field,value", [("commit_sha", "f" * 40), ("repository_full_name", "other/repo")])
def test_internal_tree_provenance_mismatch_stops_before_final_selection(store, monkeypatch, field, value):
    transport, paths = github_transport()
    original = preparation.fetch_repository_tree

    def changed(snapshot, *, transport):
        return original(snapshot, transport=transport).model_copy(update={field: value})

    monkeypatch.setattr(preparation, "fetch_repository_tree", changed)
    assert_error(store, transport, "provenance_mismatch")
    assert paths.count(PREFIX) == 1


def test_archived_repository_remains_readable_for_a_proposal(store):
    transport, _ = github_transport(first=metadata(archived=True), final=metadata(archived=True))
    context = prepare_proposal_context(IDENTITY, PAYLOAD, store, transport=transport)
    assert context.repository_context.repository.archived is True


def test_repository_name_and_owner_casing_do_not_change_identity(store):
    upper = metadata(name="EXAMPLE", full_name="DEMO-USER/EXAMPLE", owner={**IDENTITY.model_dump(), "login": "DEMO-USER"})
    transport, _ = github_transport(first=upper, final=upper)
    context = prepare_proposal_context(IDENTITY, PAYLOAD, store, transport=transport)
    assert context.repository_context.repository.id == 5
    assert context.repository_context.tree.commit_sha == COMMIT_SHA


def test_renamed_account_history_remains_readable_but_cannot_silently_rebind_source(store):
    renamed = GitHubIdentity(id=IDENTITY.id, login="renamed-user", type="User")
    changed = metadata(owner=renamed.model_dump(), full_name="renamed-user/example")
    transport, paths = github_transport(first=changed, identity=renamed)
    assert store.get(renamed.id, RUN_ID) is not None
    assert_error(store, transport, "selection_changed", identity=renamed)
    assert paths == ["/repos/renamed-user/example"]


def test_preparation_does_not_add_a_new_history_run(store):
    before = store.list(IDENTITY.id).model_dump()
    transport, _ = github_transport()
    prepare_proposal_context(IDENTITY, PAYLOAD, store, transport=transport)
    assert store.list(IDENTITY.id).model_dump() == before
