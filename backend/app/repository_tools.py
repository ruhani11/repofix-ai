"""Bounded pinned repository reads and literal content discovery."""

from threading import Lock

import httpx

from app.github_client import GitHubClientError
from app.github_files import (
    MAX_FILE_BYTES, RepositoryFile, _check_file_policy, read_repository_file,
)
from app.github_tree import RepositoryTree
from app.investigation_context import InvestigationContext


MAX_TOOL_CALLS = 20
MAX_FILE_FETCH_ATTEMPTS = 10
MAX_TOTAL_FILE_BYTES = 256 * 1024
MAX_FILE_MATCHES = 50
MAX_EXCERPT_LINES = 80
MAX_EXCERPT_CHARACTERS = 12000
MAX_SEARCH_FILES = 5
MAX_SEARCH_QUERY_CHARACTERS = 120
MAX_SEARCH_MATCHES = 10
MAX_SEARCH_PREVIEW_CHARACTERS = 300


class RepositoryToolError(RuntimeError):
    """Invalid tool input or a tool budget limit was reached."""


class RepositoryTools:
    """One instance per investigation, constructed from server-prepared context.

    The API must authenticate before preparing that context. These tools accept
    paths and keywords only, never repository URLs, owners or shell commands.
    """

    def __init__(
        self,
        context: InvestigationContext,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not isinstance(context, InvestigationContext):
            raise ValueError("A prepared investigation context is required.")

        # Copy nested tree models so later changes to context cannot rebind tools.
        self._tree = RepositoryTree.model_validate(context.tree.model_dump())
        if (
            self._tree.commit_sha != context.snapshot.commit_sha
            or self._tree.repository_full_name.lower()
            != context.repository.full_name.lower()
            or len({entry.path for entry in self._tree.entries})
            != len(self._tree.entries)
        ):
            raise ValueError("Investigation context has inconsistent repository evidence.")

        self._files = {}
        for entry in sorted(self._tree.entries, key=lambda item: item.path):
            if (
                entry.type != "blob"
                or entry.mode not in {"100644", "100755"}
                or entry.size is None
                or entry.size > MAX_FILE_BYTES
            ):
                continue
            try:
                _check_file_policy(entry.path)
            except GitHubClientError:
                continue
            self._files[entry.path] = entry

        self._transport = transport
        self._cache: dict[str, RepositoryFile] = {}
        self._tool_calls = 0
        self._fetch_attempts = 0
        self._downloaded_bytes = 0
        self._lock = Lock()

    def _consume_call(self) -> None:
        if self._tool_calls >= MAX_TOOL_CALLS:
            raise RepositoryToolError("Repository tool call limit reached.")
        # Invalid inputs and failed reads also consume the call budget.
        self._tool_calls += 1

    def _load_file(self, path: str) -> RepositoryFile:
        """Share verified file cache and budgets; caller must hold self._lock."""
        source = self._cache.get(path)
        if source is not None:
            return source
        if self._fetch_attempts >= MAX_FILE_FETCH_ATTEMPTS:
            raise RepositoryToolError("File fetch attempt limit reached.")
        if self._downloaded_bytes + self._files[path].size > MAX_TOTAL_FILE_BYTES:
            raise RepositoryToolError("Total file download budget reached.")

        self._fetch_attempts += 1
        source = read_repository_file(self._tree, path, transport=self._transport)
        self._downloaded_bytes += source.size_bytes
        self._cache[path] = source
        return source

    def find_files(self, keyword: str = "") -> dict[str, object]:
        """Find supported files by case-insensitive path substring, not content."""
        with self._lock:
            self._consume_call()
            if not isinstance(keyword, str) or len(keyword) > 120:
                raise RepositoryToolError("Use a filename keyword of at most 120 characters.")

            query = keyword.strip().casefold()
            matches = [
                {"path": path, "size_bytes": entry.size}
                for path, entry in self._files.items()
                if query in path.casefold()
            ]
            return {
                "repository_full_name": self._tree.repository_full_name,
                "commit_sha": self._tree.commit_sha,
                "files": matches[:MAX_FILE_MATCHES],
                "matching_count": len(matches),
                "truncated": len(matches) > MAX_FILE_MATCHES,
            }

    def read_file(self, path: str, start_line: int = 1) -> dict[str, object]:
        """Read a bounded excerpt with 1-based line numbers from a pinned blob."""
        with self._lock:
            self._consume_call()
            if not isinstance(path, str) or path not in self._files:
                raise RepositoryToolError("Choose a supported regular file from find_files.")
            if type(start_line) is not int or start_line < 1:
                raise RepositoryToolError("start_line must be a positive integer.")

            source = self._load_file(path)

            lines = source.content.splitlines()
            if start_line > max(1, len(lines)):
                raise RepositoryToolError("start_line exceeds the file's line count.")

            window = lines[start_line - 1:start_line - 1 + MAX_EXCERPT_LINES]
            text = "\n".join(window)
            clipped = len(text) > MAX_EXCERPT_CHARACTERS
            content = text[:MAX_EXCERPT_CHARACTERS]
            visible_count = len(content.split("\n")) if window else 0
            end_line = start_line + visible_count - 1 if visible_count else None

            return {
                "repository_full_name": source.repository_full_name,
                "commit_sha": source.commit_sha,
                "blob_sha": source.blob_sha,
                "path": source.path,
                "start_line": start_line,
                "end_line": end_line,
                "total_lines": len(lines),
                "content": content,
                "truncated": clipped or (end_line or 0) < len(lines),
                "last_line_may_be_partial": clipped,
            }

    def search_code(self, query: str, paths: list[str]) -> dict[str, object]:
        """Find literal text in up to five selected files, without executing code.

        Matching is case-insensitive and counts matching lines, not occurrences.
        Previews are line prefixes for discovery only. Use read_file around a
        returned line number before citing source; search results are not reads.
        Coverage and omitted matches are explicit, including fetch/budget errors.
        """
        with self._lock:
            self._consume_call()
            if (
                not isinstance(query, str)
                or not query.strip()
                or len(query) > MAX_SEARCH_QUERY_CHARACTERS
                or any(ord(char) < 32 or ord(char) == 127 for char in query)
            ):
                raise RepositoryToolError(
                    "Use nonblank literal text of at most 120 characters without controls."
                )
            if (
                not isinstance(paths, list)
                or not 1 <= len(paths) <= MAX_SEARCH_FILES
                or any(
                    not isinstance(path, str)
                    or len(path) > 1024
                    or path not in self._files
                    for path in paths
                )
                or len(set(paths)) != len(paths)
            ):
                raise RepositoryToolError(
                    "Choose one to five distinct supported file paths from find_files."
                )

            selected_paths = list(paths)
            needle = query.strip().casefold()
            matches = []
            searched_paths = []
            errors = []
            matching_lines = 0
            for path in selected_paths:
                try:
                    source = self._load_file(path)
                except RepositoryToolError:
                    # No partial result may masquerade as an exhaustive search.
                    errors.append({"path": path, "code": "read_budget_exhausted"})
                    break
                except GitHubClientError:
                    errors.append({"path": path, "code": "file_unavailable"})
                    continue

                for line_number, line in enumerate(source.content.splitlines(), start=1):
                    if needle not in line.casefold():
                        continue
                    matching_lines += 1
                    if len(matches) < MAX_SEARCH_MATCHES:
                        matches.append({
                            "path": path,
                            "blob_sha": source.blob_sha,
                            "line_number": line_number,
                            "preview": line[:MAX_SEARCH_PREVIEW_CHARACTERS],
                            "preview_truncated": len(line) > MAX_SEARCH_PREVIEW_CHARACTERS,
                        })
                searched_paths.append(path)

            unsearched_paths = [path for path in selected_paths if path not in searched_paths]
            return {
                "repository_full_name": self._tree.repository_full_name,
                "commit_sha": self._tree.commit_sha,
                "result_kind": "discovery_only",
                "query": query.strip(),
                "matches": matches,
                "matching_line_count": matching_lines,
                "searched_paths": searched_paths,
                "unsearched_paths": unsearched_paths,
                "errors": errors,
                "coverage_complete": not unsearched_paths,
                "truncated": bool(unsearched_paths) or matching_lines > len(matches),
            }

    @property
    def stats(self) -> dict[str, int]:
        """Local counters for later tracing; cache hits still count as tool calls."""
        with self._lock:
            return {
                "tool_calls": self._tool_calls,
                "file_fetch_attempts": self._fetch_attempts,
                "downloaded_bytes": self._downloaded_bytes,
            }
