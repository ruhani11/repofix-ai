"""Synthetic pinned-file checks and local Git patch roundtrips; no target execution."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import shutil
import subprocess

import pytest

from app.fix_proposal import FixProposal
from app.github_files import MAX_FILE_BYTES, RepositoryFile
from app.github_tree import RepositoryTree, TreeEntry
from app.investigation_history import InvestigationHistoryDetail
from app.investigation_runner import InvestigationEvent
from app.proposal_validation import (
    ProposalValidationError, validate_fix_proposal,
)


def blob_sha(content):
    data = content.encode("utf-8")
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


def make_case(content=None, *, path="src/orders.py", test_path="tests/test_orders.py", mode="100644"):
    content = content if content is not None else (
        "def accept(quantity):\n    if quantity <= 1:\n        return False\n    return True\n"
    )
    before = "    if quantity <= 1:\n"
    sha = blob_sha(content)
    file = RepositoryFile(
        repository_full_name="demo-user/example", commit_sha="a" * 40,
        path=path, blob_sha=sha, size_bytes=len(content.encode()), content=content,
    )
    tree = RepositoryTree(
        repository_full_name=file.repository_full_name, commit_sha=file.commit_sha,
        tree_sha="b" * 40, entries=[
            TreeEntry(path="src", type="tree", mode="040000", sha="c" * 40),
            TreeEntry(path="tests", type="tree", mode="040000", sha="d" * 40),
            TreeEntry(path=path, type="blob", mode=mode, sha=sha, size=file.size_bytes),
        ],
    )
    history = InvestigationHistoryDetail(
        created_at=datetime(2026, 10, 22, tzinfo=timezone.utc),
        repository_id=5, repository_name="example", issue_description="Quantity one is rejected.",
        investigation={
            "status": "completed", "run_id": "e" * 32,
            "repository_full_name": file.repository_full_name, "commit_sha": file.commit_sha,
            "report": {
                "status": "likely_cause", "issue_summary": "Quantity one is rejected.",
                "issue_category": "logic", "root_cause": "The lower boundary excludes one.",
                "evidence": [{"path": path, "start_line": 2, "end_line": 2,
                              "quote": before.rstrip("\n"), "explanation": "The comparison excludes one."}],
                "next_steps": ["Propose the boundary correction and regression tests."],
                "limitations": ["Synthetic source fixture; no target tests run."],
            },
            "total_duration_ms": 1, "agent_duration_ms": 1, "model_calls": 1,
            "tool_attempts": 1, "repository_stats": {}, "events": [
                InvestigationEvent("run_started", 0),
                InvestigationEvent("tool_completed", 1, "read_file", True),
                InvestigationEvent("grounding_validated", 1),
            ], "history_saved": True,
        },
    )
    draft = FixProposal(
        status="proposed", title="Accept positive quantities", summary="Correct the lower boundary.",
        edits=[{"path": path, "before": before, "after": before.replace("<= 1", "<= 0"),
                "reason": "One is positive."}],
        tests=[{"path": test_path, "content": "def test_one():\n    assert accept(1) is True\n",
                "purpose": "Draft code, deliberately not executed by this test module."}],
        verification_steps=["Run the proposed test in a later isolated environment."],
        risks=[], limitations=["Not executed or approved."],
    )
    return history, tree, [file], draft


def assert_failure(case, code):
    with pytest.raises(ProposalValidationError) as caught:
        validate_fix_proposal(*case)
    assert caught.value.code == code
    assert str(caught.value) == "The fix proposal could not be validated."


def test_source_validation_accepts_history_with_events_loaded_from_json():
    history, tree, files, draft = make_case()
    restored = InvestigationHistoryDetail.model_validate_json(history.model_dump_json())
    assert len(restored.investigation.events) == 3
    result = validate_fix_proposal(restored, tree, files, draft)
    assert result.base_commit_sha == restored.investigation.commit_sha
    assert result.investigation_run_id == restored.investigation.run_id
    assert result.changes[0].original_content == files[0].content
    assert restored.investigation.events == history.investigation.events


@pytest.mark.parametrize("changes", [{"kind": True}, {"elapsed_ms": True}, {"elapsed_ms": "1"}])
def test_mutated_event_fields_still_fail_strict_revalidation(changes):
    history, tree, files, draft = make_case()
    history.investigation.events[0] = replace(history.investigation.events[0], **changes)
    assert_failure((history, tree, files, draft), "invalid_input")


def add_source(case, path, content, before="1", after="0"):
    history, tree, files, draft = case
    file = files[0].model_copy(update={
        "path": path, "content": content, "blob_sha": blob_sha(content),
        "size_bytes": len(content.encode()),
    })
    tree.entries.append(TreeEntry(
        path=path, type="blob", mode="100644", sha=file.blob_sha, size=file.size_bytes,
    ))
    files.append(file)
    draft.edits.append(draft.edits[0].model_copy(update={
        "path": path, "before": before, "after": after,
    }))
    return history, tree, files, draft


def test_source_verified_artifact_binds_server_history_and_generates_deterministic_diff():
    case = make_case()
    history, tree, files, draft = case
    original = deepcopy(case)
    result = validate_fix_proposal(*case)

    assert result.investigation_run_id == history.investigation.run_id
    assert result.repository_id == history.repository_id
    assert result.repository_full_name == history.investigation.repository_full_name
    assert result.base_commit_sha == history.investigation.commit_sha
    assert result.tree_sha == tree.tree_sha
    assert result.draft.status == "proposed"
    assert result.patch_sha256 == hashlib.sha256(result.unified_diff.encode()).hexdigest()
    assert result == validate_fix_proposal(*case)
    changed = next(item for item in result.changes if item.action == "modify")
    assert changed.original_blob_sha == files[0].blob_sha
    assert changed.proposed_blob_sha == blob_sha(changed.proposed_content)
    assert changed.original_content == files[0].content
    assert "    if quantity <= 0:\n" in changed.proposed_content
    assert "-    if quantity <= 1:\n+    if quantity <= 0:\n" in result.unified_diff
    assert "--- /dev/null\n+++ b/tests/test_orders.py\n" in result.unified_diff
    assert case == original


@pytest.mark.parametrize("which,field,value", [
    ("tree", "commit_sha", "f" * 40), ("tree", "repository_full_name", "other/repo"),
    ("file", "commit_sha", "f" * 40), ("file", "repository_full_name", "other/repo"),
])
def test_mixed_repository_or_commit_provenance_is_rejected(which, field, value):
    history, tree, files, draft = make_case()
    if which == "tree":
        tree = tree.model_copy(update={field: value})
    else:
        files[0] = files[0].model_copy(update={field: value})
    assert_failure((history, tree, files, draft), "provenance_mismatch")


def test_github_repository_identity_comparison_is_case_insensitive():
    history, tree, files, draft = make_case()
    tree.repository_full_name = "Demo-User/Example"
    files[0].repository_full_name = "DEMO-USER/EXAMPLE"
    result = validate_fix_proposal(history, tree, files, draft)
    assert result.repository_full_name == "demo-user/example"


@pytest.mark.parametrize("issue", ["not_saved", "insufficient_evidence"])
def test_only_saved_likely_cause_reports_are_eligible(issue):
    history, tree, files, draft = make_case()
    values = history.model_dump()
    values["investigation"]["events"] = list(history.investigation.events)
    if issue == "not_saved":
        values["investigation"]["history_saved"] = False
    else:
        values["investigation"]["report"].update(
            status="insufficient_evidence", root_cause=None,
        )
    history = InvestigationHistoryDetail.model_validate(values)
    assert_failure((history, tree, files, draft), "ineligible_investigation")


@pytest.mark.parametrize("field,value", [
    ("blob_sha", "f" * 40), ("size_bytes", 0),
    ("content", "Unrelated complete text\n"),
])
def test_blob_identity_size_and_actual_content_must_all_match_tree(field, value):
    history, tree, files, draft = make_case()
    files[0] = files[0].model_copy(update={field: value})
    assert_failure((history, tree, files, draft), "invalid_file")


def test_returning_only_an_excerpt_is_not_accepted_as_a_verified_complete_file():
    history, tree, files, draft = make_case()
    snippet = draft.edits[0].before
    files[0] = files[0].model_copy(update={
        "content": snippet, "size_bytes": len(snippet.encode()), "blob_sha": blob_sha(snippet),
    })
    assert_failure((history, tree, files, draft), "invalid_file")


@pytest.mark.parametrize("before,content,expected", [
    ("missing source anchor", None, "missing_anchor"),
    ("return", None, "ambiguous_anchor"),
    ("aa", "aaa", "ambiguous_anchor"),
])
def test_missing_multiple_and_overlapping_anchor_matches_are_rejected(before, content, expected):
    history, tree, files, draft = make_case(content)
    draft.edits[0] = draft.edits[0].model_copy(update={"before": before, "after": "replacement"})
    assert_failure((history, tree, files, draft), expected)


@pytest.mark.parametrize("kind", ["none", "duplicate", "extra", "wrong_path", "iterator"])
def test_the_source_set_must_equal_the_draft_edit_targets(kind):
    history, tree, files, draft = make_case()
    if kind == "none":
        files = []
    elif kind == "duplicate":
        files *= 2
    elif kind == "extra":
        files.append(files[0].model_copy(update={"path": "src/other.py"}))
    elif kind == "wrong_path":
        files[0] = files[0].model_copy(update={"path": "src/other.py"})
    else:
        files = iter(files)
    assert_failure((history, tree, files, draft), "missing_source")


def test_file_edit_target_cannot_also_have_tree_descendants():
    history, tree, files, draft = make_case()
    tree.entries.append(TreeEntry(
        path=files[0].path + "/child.py", type="blob", mode="100644", sha="f" * 40, size=0,
    ))
    assert_failure((history, tree, files, draft), "path_conflict")


@pytest.mark.parametrize("kind", ["missing", "symlink", "submodule", "size_unknown", "directory", "case_alias", "duplicate"])
def test_edit_targets_must_be_unambiguous_regular_files(kind):
    history, tree, files, draft = make_case()
    entry = tree.entries[-1]
    expected = "invalid_file"
    if kind == "missing":
        tree.entries.pop()
        expected = "path_conflict"
    elif kind == "symlink":
        tree.entries[-1] = entry.model_copy(update={"mode": "120000"})
    elif kind == "submodule":
        tree.entries[-1] = entry.model_copy(update={"type": "commit", "mode": "160000"})
    elif kind == "size_unknown":
        tree.entries[-1] = entry.model_copy(update={"size": None})
    elif kind == "directory":
        tree.entries[-1] = entry.model_copy(update={"type": "tree", "mode": "040000"})
    elif kind == "case_alias":
        tree.entries.append(entry.model_copy(update={"path": "src/Orders.py"}))
        expected = "path_conflict"
    else:
        tree.entries.append(entry)
        expected = "invalid_tree"
    assert_failure((history, tree, files, draft), expected)


@pytest.mark.parametrize("kind", ["existing_file", "existing_directory", "case_alias", "descendant", "parent_file", "parent_symlink", "parent_submodule", "parent_case", "implicit_parent_case"])
def test_new_test_files_cannot_overwrite_or_traverse_conflicting_tree_paths(kind):
    history, tree, files, draft = make_case()
    test_path = draft.tests[0].path
    if kind in {"existing_file", "case_alias"}:
        tree.entries.append(TreeEntry(
            path=test_path if kind == "existing_file" else test_path.upper(),
            type="blob", mode="100644", sha="f" * 40, size=0,
        ))
    elif kind == "existing_directory":
        tree.entries.append(TreeEntry(path=test_path, type="tree", mode="040000", sha="f" * 40))
    elif kind == "descendant":
        tree.entries.append(TreeEntry(path=test_path + "/child.py", type="blob", mode="100644", sha="f" * 40, size=0))
    elif kind == "implicit_parent_case":
        tree.entries.pop(1)
        tree.entries.append(TreeEntry(path="Tests/other.py", type="blob", mode="100644", sha="f" * 40, size=0))
    else:
        changes = {
            "parent_file": {"type": "blob", "mode": "100644"},
            "parent_symlink": {"type": "blob", "mode": "120000"},
            "parent_submodule": {"type": "commit", "mode": "160000"},
            "parent_case": {"path": "Tests"},
        }
        tree.entries[1] = tree.entries[1].model_copy(update=changes[kind])
    assert_failure((history, tree, files, draft), "path_conflict")


@pytest.mark.parametrize("changed", ["draft_list", "draft_construct", "history", "tree", "file", "wrong_type"])
def test_mutated_or_unvalidated_input_models_are_revalidated(changed):
    history, tree, files, draft = make_case()
    if changed == "draft_list":
        draft.edits.append(draft.edits[0])
    elif changed == "draft_construct":
        draft = FixProposal.model_construct(**{**draft.model_dump(), "status": "applied"})
    elif changed == "history":
        history = history.model_copy(update={"repository_id": True})
    elif changed == "tree":
        tree.entries[-1] = tree.entries[-1].model_copy(update={"size": True})
    elif changed == "file":
        files[0] = files[0].model_copy(update={"size_bytes": "12"})
    else:
        draft = draft.model_dump()
    assert_failure((history, tree, files, draft), "invalid_input")


def test_file_size_limit_counts_utf8_bytes_not_characters():
    history, tree, files, draft = make_case()
    content = "字" * (MAX_FILE_BYTES // 3 + 1)
    files[0] = files[0].model_copy(update={"content": content})
    assert len(content) < MAX_FILE_BYTES
    assert_failure((history, tree, files, draft), "file_limit")


def test_an_edit_cannot_expand_a_supported_file_beyond_its_size_limit():
    content = "#" + "x" * (MAX_FILE_BYTES - 4) + "1\n"
    history, tree, files, draft = make_case(content)
    draft.edits[0] = draft.edits[0].model_copy(update={"before": "1", "after": "0123456789"})
    assert_failure((history, tree, files, draft), "file_limit")


def test_total_source_budget_is_enforced_across_individually_valid_files():
    content = "#" + "x" * 90000 + "1\n"
    case = make_case(content)
    case[3].edits[0] = case[3].edits[0].model_copy(update={"before": "1", "after": "0"})
    for name in ("src/second.py", "src/third.py"):
        case = add_source(case, name, content)
    assert_failure(case, "source_limit")


def test_total_proposed_budget_also_counts_new_test_files():
    content = "#" + "x" * 85000 + "1\n"
    case = make_case(content)
    case[3].edits[0] = case[3].edits[0].model_copy(update={"before": "1", "after": "0"})
    for name in ("src/second.py", "src/third.py"):
        case = add_source(case, name, content)
    case[3].tests[0] = case[3].tests[0].model_copy(update={"content": "#" + "x" * 20000})
    assert_failure(case, "proposed_limit")


def test_long_line_diff_expansion_has_a_separate_output_budget():
    content = "#" + "x" * 70000 + "1\n"
    case = make_case(content)
    case[3].edits[0] = case[3].edits[0].model_copy(update={"before": "1", "after": "0"})
    case = add_source(case, "src/second.py", content)
    assert_failure(case, "diff_limit")


def test_patch_order_and_digest_do_not_depend_on_model_file_order():
    case = add_source(make_case(), "src/second.py", "VALUE = 1\n")
    first = validate_fix_proposal(*case)
    case[2].reverse()
    case[3].edits.reverse()
    second = validate_fix_proposal(*case)
    assert first.unified_diff == second.unified_diff
    assert first.patch_sha256 == second.patch_sha256


def test_exact_deletion_is_a_modification_to_an_empty_file_not_a_file_removal():
    history, tree, files, draft = make_case("remove me")
    draft.edits[0] = draft.edits[0].model_copy(update={"before": "remove me", "after": ""})
    result = validate_fix_proposal(history, tree, files, draft)
    change = next(item for item in result.changes if item.action == "modify")
    assert change.proposed_content == ""
    assert "deleted file mode" not in result.unified_diff
    assert change.proposed_blob_sha == blob_sha("")


def test_schema_and_source_validation_do_not_claim_syntax_or_test_success():
    history, tree, files, draft = make_case()
    draft.edits[0] = draft.edits[0].model_copy(update={"after": "syntactically invalid ("})
    draft.tests[0] = draft.tests[0].model_copy(update={"content": "not valid Python ("})
    result = validate_fix_proposal(history, tree, files, draft)
    assert result.draft.status == "proposed"
    assert result.draft.limitations == ["Not executed or approved."]


@pytest.mark.parametrize("variant", ["normal", "crlf", "no_final_newline", "unicode_separator", "quoted_paths", "executable", "delete_all"])
def test_generated_patch_roundtrips_through_git_on_synthetic_files(tmp_path, variant):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git CLI is unavailable; deterministic source/diff checks still run.")
    content = None
    options = {}
    if variant == "crlf":
        content = "def accept(quantity):\r\n    if quantity <= 1:\r\n        return False\r\n"
    elif variant == "no_final_newline":
        content = "def accept(quantity):\n    if quantity <= 1:\n        return False"
    elif variant == "unicode_separator":
        content = "# Keep Unicode separators inside a Git line: a\u2028b\u0085c\n    if quantity <= 1:\n"
    elif variant == "quoted_paths":
        options = {"path": "src/order café.py", "test_path": "tests/test order café.py"}
    elif variant == "executable":
        options = {"mode": "100755"}
    elif variant == "delete_all":
        content = "remove me"
    history, tree, files, draft = make_case(content, **options)
    if variant == "crlf":
        draft.edits[0] = draft.edits[0].model_copy(update={
            "before": "    if quantity <= 1:\r\n", "after": "    if quantity <= 0:\r\n",
        })
    elif variant == "delete_all":
        draft.edits[0] = draft.edits[0].model_copy(update={"before": "remove me", "after": ""})
    if variant == "no_final_newline":
        draft.tests[0] = draft.tests[0].model_copy(update={"content": "# test without final newline"})
    result = validate_fix_proposal(history, tree, files, draft)
    for file in files:
        path = tmp_path / file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file.content.encode("utf-8"))
        if variant == "executable":
            path.chmod(0o755)
    for args in (["--check", "-"], ["--whitespace=nowarn", "-"]):
        completed = subprocess.run(
            [git, "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "apply", *args],
            input=result.unified_diff.encode("utf-8"),
            cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
        )
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    for change in result.changes:
        assert (tmp_path / change.path).read_bytes() == change.proposed_content.encode("utf-8")
