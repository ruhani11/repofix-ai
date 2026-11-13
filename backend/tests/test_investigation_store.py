"""Temporary SQLite files only; no GitHub, model, or private settings access."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from app.investigation_history import InvestigationRunResponse
from app.investigation_input import InvestigationInput
from app.investigation_report import InvestigationReport
from app.investigation_store import InvestigationStore, InvestigationStoreError
import app.investigation_store as module


def record(run_id=None):
    request = InvestigationInput(
        repository_id=5, repository_name="example", issue_description="  Unicode issue ✓  ",
    )
    result = InvestigationRunResponse(
        run_id=run_id or uuid4().hex, repository_full_name="demo/example",
        commit_sha="a" * 40,
        report=InvestigationReport(
            status="insufficient_evidence", issue_summary="A synthetic issue.",
            issue_category="other", root_cause=None, evidence=[],
            next_steps=["Collect reproduction details."], limitations=["No target tests run."],
        ),
        total_duration_ms=20, agent_duration_ms=10, model_calls=1, tool_attempts=0,
        repository_stats={"tool_calls": 0}, events=[], history_saved=True,
    )
    return request, result


def test_file_round_trip_survives_close_and_new_store(tmp_path):
    path = tmp_path / "nested" / "history.db"
    store = InvestigationStore(path)
    assert not path.exists()
    request, result = record()
    store.ensure_ready()
    store.save(123, request, result)
    store.close()
    reopened = InvestigationStore(path)
    try:
        saved = reopened.get(123, result.run_id)
        assert saved.issue_description == "Unicode issue ✓"
        assert saved.repository_id == 5 and saved.repository_name == "example"
        assert saved.investigation == result
        assert saved.created_at.utcoffset().total_seconds() == 0
        assert reopened.list(123).investigations[0].run_id == result.run_id
    finally:
        reopened.close()


def test_owner_filter_applies_to_both_list_and_detail(tmp_path):
    store = InvestigationStore(tmp_path / "history.db")
    request, result = record()
    try:
        store.save(123, request, result)
        assert store.list(456).investigations == []
        assert store.get(456, result.run_id) is None
        assert store.get(123, "0" * 32) is None
        summary = store.list(123).model_dump(mode="json")
        assert "issue_description" not in json.dumps(summary)
        assert all("evidence" not in item and "report" not in item for item in summary["investigations"])
        assert "account_id" not in json.dumps(summary)
    finally:
        store.close()


def test_pagination_is_newest_first_and_exhausts(tmp_path):
    store = InvestigationStore(tmp_path / "history.db")
    ids = []
    try:
        for _ in range(5):
            request, result = record()
            store.save(123, request, result)
            ids.append(result.run_id)
        pages = [store.list(123, page=n, per_page=2) for n in (1, 2, 3, 4)]
        assert [item.run_id for page in pages for item in page.investigations] == ids[::-1]
        assert [page.has_more for page in pages] == [True, True, False, False]
        assert [page.next_page for page in pages] == [2, 3, None, None]
    finally:
        store.close()


@pytest.mark.parametrize("second_owner", [123, 456])
def test_duplicate_run_id_cannot_overwrite_saved_report(tmp_path, second_owner):
    store = InvestigationStore(tmp_path / "history.db")
    request, result = record()
    try:
        store.save(123, request, result)
        changed = request.model_copy(update={"issue_description": "Replacement attempt"})
        with pytest.raises(InvestigationStoreError):
            store.save(second_owner, changed, result)
        assert store.get(123, result.run_id).issue_description == request.issue_description
        assert len(store.list(123).investigations) == 1
        assert store.get(456, result.run_id) is None
    finally:
        store.close()


@pytest.mark.parametrize("account", [True, 0, -1, "123", 2**63])
def test_invalid_account_never_opens_database(tmp_path, account):
    path = tmp_path / "history.db"
    store = InvestigationStore(path)
    with pytest.raises(InvestigationStoreError):
        store.list(account)
    assert not path.exists()


@pytest.mark.parametrize("page,per_page", [(True, 20), (0, 20), (10001, 20), (1, 0), (1, 51)])
def test_invalid_pagination_is_bounded_before_io(tmp_path, page, per_page):
    path = tmp_path / "history.db"
    store = InvestigationStore(path)
    with pytest.raises(InvestigationStoreError):
        store.list(123, page=page, per_page=per_page)
    assert not path.exists()


@pytest.mark.parametrize("run_id", ["../history.db", "' OR 1=1 --", "a" * 31])
def test_bad_identifiers_are_rejected_before_io(tmp_path, run_id):
    path = tmp_path / "history.db"
    with pytest.raises(InvestigationStoreError):
        InvestigationStore(path).get(123, run_id)
    assert not path.exists()


@pytest.mark.parametrize("bad_json", ['{"PRIVATE_CORRUPTION": true}', "not-json-PRIVATE_CORRUPTION"])
def test_corrupt_detail_is_a_controlled_failure(tmp_path, bad_json):
    path = tmp_path / "history.db"
    store = InvestigationStore(path)
    request, result = record()
    try:
        store.save(123, request, result)
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE investigation_history SET detail_json=?", (bad_json,))
        with pytest.raises(InvestigationStoreError) as caught:
            store.get(123, result.run_id)
        assert "PRIVATE_CORRUPTION" not in str(caught.value)
        assert str(path) not in str(caught.value)
    finally:
        store.close()


def test_unavailable_parent_does_not_expose_path(tmp_path):
    blocker = tmp_path / "PRIVATE_FILENAME"
    blocker.write_text("not a directory")
    store = InvestigationStore(blocker / "history.db")
    with pytest.raises(InvestigationStoreError) as caught:
        store.ensure_ready()
    assert "PRIVATE_FILENAME" not in str(caught.value)


def test_future_schema_is_not_overwritten(tmp_path):
    path = tmp_path / "history.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(InvestigationStoreError):
        InvestigationStore(path).list(123)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 99


def test_record_size_limit_prevents_insert(tmp_path, monkeypatch):
    path = tmp_path / "history.db"
    store = InvestigationStore(path)
    request, result = record()
    monkeypatch.setattr(module, "MAX_HISTORY_RECORD_BYTES", 10)
    with pytest.raises(InvestigationStoreError, match="too large"):
        store.save(123, request, result)
    assert not path.exists()


def test_parallel_saves_share_connection_safely(tmp_path):
    store = InvestigationStore(tmp_path / "history.db")
    records = [record() for _ in range(12)]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda pair: store.save(123, *pair), records))
        saved = store.list(123)
        assert {item.run_id for item in saved.investigations} == {r.run_id for _, r in records}
        assert all(store.get(123, r.run_id).investigation == r for _, r in records)
    finally:
        store.close()
