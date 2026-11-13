"""Offline proposal persistence and byte consistency; no generated code runs."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.fix_proposal import FixProposal
from app.github_files import RepositoryFile
from app.github_tree import RepositoryTree
from app.investigation_history import InvestigationHistoryDetail, InvestigationRunResponse
from app.investigation_report import InvestigationReport
from app.proposal_generator import ProposalGenerationResult
from app.proposal_history import ProposalResponse, proposal_response
from app.proposal_store import ProposalStore, ProposalStoreError
from app.proposal_validation import validate_fix_proposal


def make_result(*, content="def accept(quantity):\n    return quantity > 1\n", run_id="c" * 32):
    history = InvestigationHistoryDetail(
        created_at=datetime(2026, 10, 23, tzinfo=timezone.utc), repository_id=5,
        repository_name="example", issue_description="A quantity of one is rejected.",
        investigation=InvestigationRunResponse(
            run_id=run_id, repository_full_name="demo-user/example", commit_sha="a" * 40,
            report=InvestigationReport(
                status="likely_cause", issue_summary="One is rejected.", issue_category="logic",
                root_cause="The boundary excludes one.", evidence=[{
                    "path": "orders_api.py", "start_line": 2, "end_line": 2,
                    "quote": "    return quantity > 1", "explanation": "The comparison excludes one.",
                }], next_steps=["Draft a fix."], limitations=["Not reproduced."],
            ), total_duration_ms=1, agent_duration_ms=1, model_calls=1, tool_attempts=1,
            repository_stats={}, events=[], history_saved=True,
        ),
    )
    raw = content.encode()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw, usedforsecurity=False).hexdigest()
    tree = RepositoryTree(
        repository_full_name="demo-user/example", commit_sha="a" * 40, tree_sha="b" * 40,
        entries=[{"path": "orders_api.py", "type": "blob", "mode": "100755", "sha": blob, "size": len(raw)}],
    )
    draft = FixProposal(
        status="proposed", title="Accept quantity one", summary="Correct the lower boundary.",
        edits=[{"path": "orders_api.py", "before": "quantity > 1", "after": "quantity > 0", "reason": "One is positive."}],
        tests=[{"path": "tests/test_orders_regression.py", "content": "def test_one():\n    assert accept(1) is True\n", "purpose": "Draft only; not run."}],
        verification_steps=["Run the proposed test in an approved sandbox later."],
        risks=[], limitations=["Not applied, approved or executed."],
    )
    source = RepositoryFile(
        repository_full_name=tree.repository_full_name, commit_sha=tree.commit_sha,
        path="orders_api.py", blob_sha=blob, size_bytes=len(raw), content=content,
    )
    result = ProposalGenerationResult(
        proposal_id=uuid4().hex, validated=validate_fix_proposal(history, tree, [source], draft),
        duration_ms=2, model_calls=2, tool_attempts=1,
        repository_stats={"tool_calls": 1, "file_fetch_attempts": 1, "downloaded_bytes": len(raw)}, events=(),
    )
    return history, tree, result


def record(**kwargs):
    return proposal_response(make_result(**kwargs)[2], total_duration_ms=3)


def test_sqlite_restart_roundtrip_keeps_exact_bytes_and_account_scope(tmp_path):
    path = tmp_path / "nested" / "proposals.db"
    store = ProposalStore(path)
    assert not path.exists()
    store.ensure_ready()
    detail = record(content="def accept(quantity):\r\n    return quantity > 1\r\n")
    store.save(123, detail)
    store.close()
    reopened = ProposalStore(path)
    loaded = reopened.get(123, detail.proposal_id)
    assert loaded == detail and loaded.proposal_saved is True
    assert loaded.changes[0].mode == "100755"
    assert "\r\n" in loaded.changes[0].original_content
    assert loaded.created_at.utcoffset().total_seconds() == 0
    assert reopened.get(999, detail.proposal_id) is None
    assert reopened.get(123, "0" * 32) is None
    listing = reopened.list(123).model_dump()
    assert listing["proposals"][0]["proposal_id"] == detail.proposal_id
    assert not {"draft", "changes", "unified_diff"} & set(listing["proposals"][0])
    assert reopened.list(999).proposals == []
    reopened.close()


def test_order_pagination_and_empty_pages(tmp_path):
    store = ProposalStore(tmp_path / "db")
    for index in [3, 1, 2]:
        detail = record().model_copy(update={"proposal_id": f"{index:032x}", "created_at": datetime(2026, 10, 23, tzinfo=timezone.utc)})
        store.save(123, detail)
    first = store.list(123, per_page=2)
    assert [int(x.proposal_id, 16) for x in first.proposals] == [3, 2]
    assert first.has_more and first.next_page == 2
    last = store.list(123, page=2, per_page=2)
    assert [int(x.proposal_id, 16) for x in last.proposals] == [1]
    assert not last.has_more and last.next_page is None
    assert store.list(123, page=10000).proposals == []
    store.close()


@pytest.mark.parametrize("account", [None, True, 0, -1, 2**63, "123"])
def test_invalid_accounts_rejected_without_disk(tmp_path, account):
    path = tmp_path / "db"
    store = ProposalStore(path)
    for action in [lambda: store.get(account, "a" * 32), lambda: store.list(account), lambda: store.save(account, record())]:
        with pytest.raises(ProposalStoreError):
            action()
    assert not path.exists()


@pytest.mark.parametrize("identifier", [None, "a" * 31, "A" * 32, "a" * 32 + "\n", "' OR 1=1 --", 123])
def test_invalid_identifiers_rejected(tmp_path, identifier):
    store = ProposalStore(tmp_path / "db")
    with pytest.raises(ProposalStoreError):
        store.get(123, identifier)


@pytest.mark.parametrize("paging", [{"page": 0}, {"page": 10001}, {"page": True}, {"per_page": 0}, {"per_page": 51}, {"per_page": "1"}])
def test_invalid_pagination(tmp_path, paging):
    with pytest.raises(ProposalStoreError):
        ProposalStore(tmp_path / "db").list(123, **paging)


def test_duplicate_proposal_cannot_overwrite_or_reassign_owner(tmp_path):
    store = ProposalStore(tmp_path / "db")
    detail = record()
    store.save(123, detail)
    with pytest.raises(ProposalStoreError):
        store.save(999, detail)
    assert store.get(123, detail.proposal_id) == detail
    assert store.get(999, detail.proposal_id) is None
    store.close()


@pytest.mark.parametrize("field,value", [
    ("proposal_saved", False), ("status", "approved"), ("patch_sha256", "0" * 64),
    ("unified_diff", "fake patch"), ("model_calls", 0), ("repository_stats", {"api_key": 5}),
    ("created_at", datetime(2026, 1, 1)),
])
def test_save_revalidates_mutated_models_without_disk(tmp_path, field, value):
    path = tmp_path / "db"
    bad = record().model_copy(update={field: value})
    with pytest.raises(ProposalStoreError):
        ProposalStore(path).save(123, bad)
    assert not path.exists()


@pytest.mark.parametrize("tamper", ["patch", "anchor", "original", "proposed", "mode", "missing_change", "test_content", "id", "saved", "garbage"])
def test_corrupted_stored_artifacts_fail_safely(tmp_path, tamper):
    path = tmp_path / "db"
    store = ProposalStore(path)
    detail = record()
    store.save(123, detail)
    data = detail.model_dump(mode="json")
    if tamper == "patch": data["unified_diff"] += "PRIVATE_BAD_PATCH"
    if tamper == "anchor": data["draft"]["edits"][0]["before"] = "unseen anchor"
    if tamper == "original": data["changes"][0]["original_content"] += "extra"
    if tamper == "proposed": data["changes"][0]["proposed_content"] += "extra"
    if tamper == "mode": data["changes"][1]["mode"] = "100755"
    if tamper == "missing_change": data["changes"].pop()
    if tamper == "test_content": data["draft"]["tests"][0]["content"] += "extra"
    if tamper == "id": data["proposal_id"] = "d" * 32
    if tamper == "saved": data["proposal_saved"] = False
    encoded = "PRIVATE_BAD_JSON" if tamper == "garbage" else json.dumps(data)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE fix_proposals SET detail_json=?", (encoded,))
    with pytest.raises(ProposalStoreError) as error:
        store.get(123, detail.proposal_id)
    assert str(error.value) == "Fix proposal storage is unavailable."
    store.close()


def test_schema_version_and_path_errors_are_safe(tmp_path):
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=99")
    for store in [ProposalStore(path), ProposalStore(tmp_path)]:
        with pytest.raises(ProposalStoreError) as error:
            store.ensure_ready()
        assert str(tmp_path) not in str(error.value)


def test_concurrent_inserts_and_readback(tmp_path):
    store = ProposalStore(tmp_path / "db")
    items = [record() for _ in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: store.save(123, item), items))
        loaded = list(pool.map(lambda item: store.get(123, item.proposal_id), items))
    assert loaded == items and len(store.list(123).proposals) == 8
    store.close()


def test_size_limit_is_enforced_before_save(tmp_path, monkeypatch):
    import app.proposal_store as module
    monkeypatch.setattr(module, "MAX_PROPOSAL_RECORD_BYTES", 10)
    with pytest.raises(ProposalStoreError):
        ProposalStore(tmp_path / "db").save(123, record())


@pytest.mark.parametrize("path", ["", " ", ":memory:", "bad\npath", "bad\x00path"])
def test_settings_require_persistent_proposal_path(path):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, proposal_database_path=path)


def test_settings_paths_are_stable_distinct_and_env_configurable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, proposal_database_path="data/review.db")
    assert settings.proposal_database_path == Path(__file__).resolve().parents[1] / "data" / "review.db"
    monkeypatch.setenv("REPOFIX_PROPOSAL_DATABASE_PATH", str(tmp_path / "chosen.db"))
    assert Settings(_env_file=None).proposal_database_path == tmp_path / "chosen.db"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, history_database_path=tmp_path / "same", proposal_database_path=tmp_path / "same")


def test_response_factory_rejects_mutable_nested_draft_corruption():
    _, _, result = make_result()
    result.validated.draft.edits[0] = result.validated.draft.edits[0].model_copy(update={"after": "forged"})
    with pytest.raises(ValidationError):
        proposal_response(result, total_duration_ms=3)


def test_saving_a_dict_or_foreign_object_is_not_allowed(tmp_path):
    for value in [record().model_dump(), object(), None]:
        with pytest.raises(ProposalStoreError):
            ProposalStore(tmp_path / "db").save(123, value)
