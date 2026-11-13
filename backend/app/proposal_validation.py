"""Verify draft edits against supplied pinned source and build a review patch.

This module performs no network, filesystem, provider or shell operations. The
future service must obtain history through an authenticated account-scoped read
and fetch the tree/files through the verified GitHub readers. A validated patch
is still only a proposal, not an approved or tested fix.
"""

from dataclasses import dataclass
from difflib import unified_diff
import hashlib
import json
from typing import Literal, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError

from app.fix_proposal import MAX_EDITS, FixProposal, _source_text
from app.github_files import MAX_FILE_BYTES, RepositoryFile
from app.github_tree import RepositoryTree, TreeEntry
from app.investigation_history import InvestigationHistoryDetail
from app.investigation_runner import InvestigationEvent


MAX_SOURCE_BYTES = 256 * 1024
MAX_PROPOSED_BYTES = 256 * 1024
MAX_DIFF_BYTES = 256 * 1024
_Model = TypeVar("_Model", bound=BaseModel)
_EVENTS = TypeAdapter(list[InvestigationEvent])


class ProposalValidationError(RuntimeError):
    """Stable internal reason; never includes source text or input payloads."""

    def __init__(self, code: str):
        self.code = code
        super().__init__("The fix proposal could not be validated.")


@dataclass(frozen=True)
class ValidatedFileChange:
    path: str
    action: Literal["modify", "create"]
    mode: Literal["100644", "100755"]
    original_blob_sha: str | None
    proposed_blob_sha: str
    original_content: str | None
    proposed_content: str


@dataclass(frozen=True)
class ValidatedFixProposal:
    """Server-built artifact. patch_sha256 covers patch bytes, not an approval."""

    investigation_run_id: str
    repository_id: int
    repository_full_name: str
    base_commit_sha: str
    tree_sha: str
    draft: FixProposal
    changes: tuple[ValidatedFileChange, ...]
    unified_diff: str
    patch_sha256: str


def _revalidate(value: _Model, model: type[_Model]) -> _Model:
    if not isinstance(value, model):
        raise ProposalValidationError("invalid_input")
    try:
        data = value.model_dump(warnings="error")
        if model is InvestigationHistoryDetail:
            # Only dataclass events need JSON reconstruction. Keep every other
            # value in Python form so bool/string mutations cannot be converted
            # into valid strict integers by Pydantic's model JSON serializer.
            run = data.get("investigation")
            if not isinstance(run, dict) or "events" not in run:
                raise ValueError("Invalid saved investigation.")
            run["events"] = _EVENTS.validate_json(
                json.dumps(run["events"]), strict=True,
            )
        return model.model_validate(data)
    except (ValidationError, PydanticSerializationError, ValueError, TypeError):
        raise ProposalValidationError("invalid_input") from None


def _git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _encode_source(content: str) -> bytes:
    try:
        _source_text(content)
        encoded = content.encode("utf-8")
    except (ValueError, UnicodeEncodeError, TypeError):
        raise ProposalValidationError("invalid_file") from None
    if len(encoded) > MAX_FILE_BYTES:
        raise ProposalValidationError("file_limit")
    return encoded


def _tree_index(tree: RepositoryTree) -> dict[str, TreeEntry]:
    index: dict[str, TreeEntry] = {}
    for entry in tree.entries:
        if entry.path in index:
            raise ProposalValidationError("invalid_tree")
        index[entry.path] = entry
    return index


def _check_target(path: str, index: dict[str, TreeEntry], *, create: bool) -> TreeEntry | None:
    parts = path.split("/")
    for name, entry in index.items():
        # Compare each entry's shared parent segments once. This also handles
        # implicit directories without repeatedly scanning the tree per prefix.
        existing = name.split("/")
        common = 0
        for supplied, actual in zip(parts[:-1], existing):
            if supplied.casefold() != actual.casefold():
                break
            if supplied != actual:
                raise ProposalValidationError("path_conflict")
            common += 1
        if common == len(existing) and common < len(parts):
            if entry.type != "tree" or entry.mode != "040000":
                raise ProposalValidationError("path_conflict")
    matches = [entry for name, entry in index.items() if name.casefold() == path.casefold()]
    has_descendant = any(name.casefold().startswith(path.casefold() + "/") for name in index)
    if has_descendant:
        raise ProposalValidationError("path_conflict")
    if create:
        if matches:
            raise ProposalValidationError("path_conflict")
        return None
    if len(matches) != 1 or matches[0].path != path:
        raise ProposalValidationError("path_conflict")
    entry = matches[0]
    if entry.type != "blob" or entry.mode not in {"100644", "100755"} or entry.size is None:
        raise ProposalValidationError("invalid_file")
    return entry


def _git_path(path: str) -> str:
    data = path.encode("utf-8")
    if all(33 <= byte <= 126 and byte not in {34, 92} for byte in data):
        return path
    # Git's quoted paths use UTF-8 bytes with C-style octal escapes, not JSON \u.
    return '"' + "".join(
        chr(byte) if 32 <= byte <= 126 and byte not in {34, 92} else f"\\{byte:03o}"
        for byte in data
    ) + '"'


def _git_lines(text: str) -> list[str]:
    # Git lines end only at LF. str.splitlines would also split Unicode separators.
    parts = text.split("\n")
    return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _file_diff(change: ValidatedFileChange) -> str:
    old_path, new_path = _git_path("a/" + change.path), _git_path("b/" + change.path)
    header = f"diff --git {old_path} {new_path}\n"
    if change.action == "create":
        header += "new file mode 100644\n"
        header += f"index {'0' * 40}..{change.proposed_blob_sha}\n"
    else:
        header += f"index {change.original_blob_sha}..{change.proposed_blob_sha} {change.mode}\n"
    lines = unified_diff(
        _git_lines(change.original_content or ""), _git_lines(change.proposed_content),
        fromfile="/dev/null" if change.action == "create" else old_path,
        tofile=new_path, n=3, lineterm="\n",
    )
    body = "".join(
        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
        for line in lines
    )
    return header + body


def validate_fix_proposal(
    history: InvestigationHistoryDetail,
    tree: RepositoryTree,
    source_files: list[RepositoryFile] | tuple[RepositoryFile, ...],
    draft: FixProposal,
) -> ValidatedFixProposal:
    """Match exact unique anchors in full files, then generate a bounded patch.

    Provide exactly the original files targeted by draft.edits, not excerpts.
    New test files are checked against the complete bounded tree. This function
    verifies consistency with supplied provenance; the caller is responsible
    for authenticated history lookup and trusted commit/tree acquisition.
    """
    history = _revalidate(history, InvestigationHistoryDetail)
    tree = _revalidate(tree, RepositoryTree)
    draft = _revalidate(draft, FixProposal)
    investigation = history.investigation
    if not investigation.history_saved or investigation.report.status != "likely_cause":
        raise ProposalValidationError("ineligible_investigation")
    if (
        tree.repository_full_name.casefold() != investigation.repository_full_name.casefold()
        or tree.commit_sha != investigation.commit_sha
    ):
        raise ProposalValidationError("provenance_mismatch")
    index = _tree_index(tree)
    if not isinstance(source_files, (list, tuple)) or len(source_files) > MAX_EDITS:
        raise ProposalValidationError("missing_source")
    files: dict[str, RepositoryFile] = {}
    source_bytes = 0
    for raw in source_files:
        file = _revalidate(raw, RepositoryFile)
        if file.path in files:
            raise ProposalValidationError("missing_source")
        if (
            file.repository_full_name.casefold() != investigation.repository_full_name.casefold()
            or file.commit_sha != investigation.commit_sha
        ):
            raise ProposalValidationError("provenance_mismatch")
        files[file.path] = file
        source_bytes += len(_encode_source(file.content))
    if source_bytes > MAX_SOURCE_BYTES:
        raise ProposalValidationError("source_limit")
    if set(files) != {edit.path for edit in draft.edits}:
        raise ProposalValidationError("missing_source")
    changes: list[ValidatedFileChange] = []
    proposed_bytes = 0
    for edit in draft.edits:
        entry = _check_target(edit.path, index, create=False)
        assert entry is not None
        file = files[edit.path]
        original = _encode_source(file.content)
        if (
            file.blob_sha != entry.sha or file.size_bytes != entry.size
            or len(original) != entry.size or _git_blob_sha(original) != entry.sha
        ):
            raise ProposalValidationError("invalid_file")
        first = file.content.find(edit.before)
        if first < 0:
            raise ProposalValidationError("missing_anchor")
        if file.content.find(edit.before, first + 1) >= 0:
            raise ProposalValidationError("ambiguous_anchor")
        content = file.content[:first] + edit.after + file.content[first + len(edit.before):]
        encoded = _encode_source(content)
        proposed_bytes += len(encoded)
        changes.append(ValidatedFileChange(
            path=edit.path, action="modify", mode=entry.mode,
            original_blob_sha=entry.sha, proposed_blob_sha=_git_blob_sha(encoded),
            original_content=file.content, proposed_content=content,
        ))
    for test in draft.tests:
        _check_target(test.path, index, create=True)
        encoded = _encode_source(test.content)
        proposed_bytes += len(encoded)
        changes.append(ValidatedFileChange(
            path=test.path, action="create", mode="100644",
            original_blob_sha=None, proposed_blob_sha=_git_blob_sha(encoded),
            original_content=None, proposed_content=test.content,
        ))
    if proposed_bytes > MAX_PROPOSED_BYTES:
        raise ProposalValidationError("proposed_limit")
    ordered = tuple(sorted(changes, key=lambda change: change.path))
    patch = "".join(_file_diff(change) for change in ordered)
    encoded_patch = patch.encode("utf-8")
    if len(encoded_patch) > MAX_DIFF_BYTES:
        raise ProposalValidationError("diff_limit")
    return ValidatedFixProposal(
        investigation_run_id=investigation.run_id,
        repository_id=history.repository_id,
        repository_full_name=investigation.repository_full_name,
        base_commit_sha=investigation.commit_sha, tree_sha=tree.tree_sha,
        draft=draft, changes=ordered, unified_diff=patch,
        patch_sha256=hashlib.sha256(encoded_patch).hexdigest(),
    )
