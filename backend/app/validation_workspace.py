"""Prepare bounded original/patched Python file payloads without executing code.

Inputs must come from account-scoped saved records and the verified pinned Git
tree/blob readers. This pure builder performs no I/O and does not admit a job.
The Docker runner must still atomically check active approval and validate the
wire payload before writing regular files inside its private tmpfs workspace.
"""

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import PurePosixPath

from app.fix_proposal import _repository_path
from app.github_client import GitHubClientError
from app.github_files import MAX_FILE_BYTES, RepositoryFile, _check_file_policy
from app.github_oauth import GitHubIdentity
from app.github_tree import RepositoryTree
from app.proposal_history import ProposalResponse
from app.proposal_review import ProposalReviewState
from app.proposal_validation import (
    ProposalValidationError, _check_target, _encode_source, _git_blob_sha, _revalidate,
)
from app.sandbox_profile import PROFILE_ID
from app.validation_request import (
    ValidationBinding, ValidationInput, ValidationRequestError, _validated,
    bind_validation_request,
)


MAX_WORKSPACE_SOURCE_FILES = 20
MAX_WORKSPACE_FILES = 23  # twenty existing Python files plus three proposed tests
MAX_WORKSPACE_BYTES = 512 * 1024
MAX_WORKSPACE_PAYLOAD_BYTES = 2 * 1024 * 1024
WORKSPACE_PROTOCOL_VERSION = 1


class ValidationWorkspaceError(RuntimeError):
    """Safe coded failure; never includes paths, source or raw model errors."""

    def __init__(self, code: str):
        self.code = code
        super().__init__("The approved proposal's Python workspace could not be prepared.")


@dataclass(frozen=True)
class ValidationWorkspace:
    """Internal immutable payload bytes, not an execution or approval capability.

    Do not deserialize browser data into this class. The service must build it
    from trusted records, recheck admission and pass exact payload bytes to the
    future runner. A payload hash is an identifier, not proof of approval.
    """

    binding: ValidationBinding
    test_paths: tuple[str, ...]
    source_file_count: int
    source_bytes: int
    baseline_bytes: int
    patched_bytes: int
    baseline_sha256: str
    patched_sha256: str
    baseline_payload: bytes = field(repr=False)
    patched_payload: bytes = field(repr=False)


def _tree_copy(tree: RepositoryTree) -> RepositoryTree:
    try:
        return _revalidate(tree, RepositoryTree)
    except ProposalValidationError:
        raise ValidationWorkspaceError("invalid_tree") from None


def _selected(tree: RepositoryTree) -> tuple[str, ...]:
    index = {entry.path: entry for entry in tree.entries}
    if len(index) != len(tree.entries) or len({path.casefold() for path in index}) != len(index):
        raise ValidationWorkspaceError("ambiguous_tree")
    # A narrow first profile deliberately excludes linked/submodule worktrees.
    if any(entry.mode in {"120000", "160000"} for entry in tree.entries):
        raise ValidationWorkspaceError("unsupported_tree")
    paths = []
    for entry in tree.entries:
        if entry.type != "blob" or not entry.path.endswith(".py"):
            continue
        try:
            _check_file_policy(entry.path)
        except GitHubClientError:
            continue  # never fetch or package blocked Python files
        if entry.size is None or entry.size > MAX_FILE_BYTES:
            raise ValidationWorkspaceError("source_limit")
        try:
            _repository_path(entry.path)
            _check_target(entry.path, index, create=False)
        except (ValueError, ProposalValidationError):
            raise ValidationWorkspaceError("unsupported_path") from None
        paths.append(entry.path)
    if not paths or len(paths) > MAX_WORKSPACE_SOURCE_FILES:
        raise ValidationWorkspaceError("source_limit")
    if sum(index[path].size for path in paths) > MAX_WORKSPACE_BYTES:
        raise ValidationWorkspaceError("workspace_limit")
    return tuple(sorted(paths))


def select_validation_paths(tree: RepositoryTree) -> tuple[str, ...]:
    """Select all permitted lowercase-.py regular files before bounded fetching.

    This is a filtered Python workspace, not a full checkout. Non-Python assets,
    dependency manifests and blocked paths are omitted. Missing runtime/data
    dependencies must be reported as unsupported/error, never a passing fix.
    """
    return _selected(_tree_copy(tree))


def _payload(binding: ValidationBinding, phase: str, files: dict[str, str], tests: tuple[str, ...]) -> tuple[bytes, int]:
    total = sum(len(_encode_source(content)) for content in files.values())
    if len(files) > MAX_WORKSPACE_FILES or total > MAX_WORKSPACE_BYTES:
        raise ValidationWorkspaceError("workspace_limit")
    data = {
        "protocol_version": WORKSPACE_PROTOCOL_VERSION, "profile": PROFILE_ID,
        "phase": phase, "binding": binding.model_dump(mode="json"), "tests": list(tests),
        "files": [{"path": path, "content": content, "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
                  for path, content in sorted(files.items())],
    }
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_WORKSPACE_PAYLOAD_BYTES:
        raise ValidationWorkspaceError("payload_limit")
    return encoded, total


def prepare_validation_workspace(
    identity: GitHubIdentity,
    proposal: ProposalResponse,
    review: ProposalReviewState,
    payload: ValidationInput,
    tree: RepositoryTree,
    source_files: list[RepositoryFile] | tuple[RepositoryFile, ...],
    *,
    now: datetime | None = None,
) -> ValidationWorkspace:
    """Build original+new-tests and patched+the-same-tests byte payloads.

    The service supplies the account's saved immutable proposal, latest review,
    trusted tree at base_commit_sha and exactly the files from
    select_validation_paths. No HEAD capture, fuzzy edit, local target checkout,
    dependency install or generated-code execution occurs here. now is internal.
    """
    try:
        proposal = _validated(proposal, ProposalResponse, "invalid_proposal")
        binding = bind_validation_request(identity, proposal, review, payload, now=now)
    except ValidationRequestError as error:
        raise ValidationWorkspaceError(error.code) from None
    tree = _tree_copy(tree)
    if (
        tree.repository_full_name != binding.repository_full_name
        or tree.commit_sha != binding.base_commit_sha or tree.tree_sha != binding.tree_sha
    ):
        raise ValidationWorkspaceError("provenance_mismatch")
    paths = _selected(tree)
    if not isinstance(source_files, (list, tuple)) or len(source_files) != len(paths):
        raise ValidationWorkspaceError("missing_source")
    index = {entry.path: entry for entry in tree.entries}
    originals: dict[str, str] = {}
    try:
        for raw in source_files:
            source = _revalidate(raw, RepositoryFile)
            if source.path not in paths or source.path in originals:
                raise ValidationWorkspaceError("missing_source")
            if source.repository_full_name != binding.repository_full_name or source.commit_sha != binding.base_commit_sha:
                raise ValidationWorkspaceError("provenance_mismatch")
            entry = index[source.path]
            data = _encode_source(source.content)
            if (
                len(data) != source.size_bytes or source.size_bytes != entry.size
                or source.blob_sha != entry.sha or _git_blob_sha(data) != entry.sha
            ):
                raise ValidationWorkspaceError("invalid_source")
            originals[source.path] = source.content
    except ProposalValidationError:
        raise ValidationWorkspaceError("invalid_source") from None
    if set(originals) != set(paths):
        raise ValidationWorkspaceError("missing_source")
    source_bytes = sum(len(content.encode("utf-8")) for content in originals.values())
    baseline = dict(originals)
    patched = dict(originals)
    tests = tuple(sorted(test.path for test in proposal.draft.tests))
    try:
        for change in proposal.changes:
            if not change.path.endswith(".py"):
                raise ValidationWorkspaceError("unsupported_target")
            if change.action == "modify":
                entry = _check_target(change.path, index, create=False)
                if (
                    change.path not in originals or originals[change.path] != change.original_content
                    or entry.sha != change.original_blob_sha or entry.mode != change.mode
                ):
                    raise ValidationWorkspaceError("proposal_source_mismatch")
                patched[change.path] = change.proposed_content
            else:
                name = PurePosixPath(change.path).name
                if not (name.startswith("test_") or name.endswith("_test.py")):
                    raise ValidationWorkspaceError("unsupported_target")
                _check_target(change.path, index, create=True)
                baseline[change.path] = patched[change.path] = change.proposed_content
    except ProposalValidationError:
        raise ValidationWorkspaceError("proposal_source_mismatch") from None
    try:
        before, baseline_bytes = _payload(binding, "baseline", baseline, tests)
        after, patched_bytes = _payload(binding, "patched", patched, tests)
    except ProposalValidationError:
        raise ValidationWorkspaceError("invalid_source") from None
    return ValidationWorkspace(
        binding=binding, test_paths=tests, source_file_count=len(originals), source_bytes=source_bytes,
        baseline_bytes=baseline_bytes, patched_bytes=patched_bytes,
        baseline_sha256=hashlib.sha256(before).hexdigest(), patched_sha256=hashlib.sha256(after).hexdigest(),
        baseline_payload=before, patched_payload=after,
    )
