"""Publish one exact, verified proposal to a new GitHub draft branch."""

import json
import re
from typing import Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from app.github_oauth import GitHubIdentity
from app.github_repositories import PublicRepository
from app.github_tree import TreeEntry
from app.github_url import parse_github_repository_url
from app.proposal_history import ProposalResponse
from app.validation_history import ValidationRunResult

_SHA = re.compile(r"[0-9a-f]{40}")
_MAX_RESPONSE = 512 * 1024


class DraftPullRequestError(RuntimeError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


class DraftPullRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    number: int = Field(gt=0)
    url: str
    branch: str
    draft: bool
    state: Literal["open", "closed"]


def _sha(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("Invalid Git object ID.")
    return value


def _request(client, method, path, *, body=None, missing=False):
    expected = 200 if method == "GET" else 201
    try:
        with client.stream(method, path, json=body) as response:
            if missing and response.status_code == 404:
                return None
            if response.status_code != expected:
                if response.status_code in (401, 403):
                    raise DraftPullRequestError(502, "GitHub refused the draft PR request. Check the local token and repository permissions.")
                raise DraftPullRequestError(
                    502, "GitHub could not complete the draft PR request. Check GitHub before trying again; a branch may already exist.",
                )
            data = bytearray()
            for chunk in response.iter_bytes(chunk_size=8192):
                if len(data) + len(chunk) > _MAX_RESPONSE:
                    raise ValueError("GitHub response too large.")
                data.extend(chunk)
            decoded = json.loads(data)
            if not isinstance(decoded, (dict, list)):
                raise ValueError("Unexpected GitHub response.")
            return decoded
    except httpx.RequestError:
        raise DraftPullRequestError(
            502, "GitHub did not confirm the request. Check GitHub before trying again; a branch or draft PR may already exist.",
        ) from None


def _tree(client, prefix, sha):
    data = _request(client, "GET", f"{prefix}/git/trees/{sha}?recursive=1")
    if data["sha"] != sha or data["truncated"] is not False or not isinstance(data["tree"], list):
        raise ValueError("Unexpected Git tree.")
    if len(data["tree"]) > 1000:
        raise ValueError("Git tree exceeds the supported size.")
    entries = [TreeEntry.model_validate(item) for item in data["tree"]]
    if len({item.path.casefold() for item in entries}) != len(entries):
        raise ValueError("Ambiguous Git tree.")
    return entries


def _expected_tree(entries, changes):
    def entry_value(item):
        return (item.type, item.mode, None if item.type == "tree" else item.sha)

    expected = {item.path: entry_value(item) for item in entries}
    for change in changes:
        if any(part.casefold() in {".github", ".git", ".gitmodules"} for part in change.path.split("/")):
            raise DraftPullRequestError(409, "Draft PRs cannot modify GitHub workflows or Git metadata.")
        previous = expected.get(change.path)
        if change.action == "modify" and previous != ("blob", change.mode, change.original_blob_sha):
            raise DraftPullRequestError(409, "The original file no longer matches the verified patch.")
        if change.action == "create" and previous is not None:
            raise DraftPullRequestError(409, "A file the patch would create already exists.")
        expected[change.path] = ("blob", change.mode, change.proposed_blob_sha)
        parts = change.path.split("/")
        for length in range(1, len(parts)):
            directory = "/".join(parts[:length])
            if expected.setdefault(directory, ("tree", "040000", None))[0] != "tree":
                raise ValueError("Conflicting file and directory.")
    return expected


def publish_draft_pr(
    identity: GitHubIdentity, proposal: ProposalResponse, validation: ValidationRunResult,
    token: SecretStr, *, authorize, transport: httpx.BaseTransport | None = None,
) -> DraftPullRequest:
    repo = parse_github_repository_url("https://github.com/" + proposal.repository_full_name)
    if f"{repo.owner}/{repo.name}" != proposal.repository_full_name or repo.owner.casefold() != identity.login.casefold():
        raise DraftPullRequestError(409, "The verified repository no longer matches the connected account.")
    prefix = f"/repos/{repo.owner}/{repo.name}"
    branch = f"repofix/{proposal.proposal_id}"
    ref_path = f"{prefix}/git/ref/heads/{quote(branch, safe='/')}"
    title = "RepoFix: verified patch " + proposal.proposal_id[:8]
    body = (
        "Draft patch from RepoFix. Please review the diff and the added tests before merging.\n\n"
        f"- Original: {validation.baseline.passed} passed, {validation.baseline.failed} failed\n"
        f"- Patched: {validation.patched.passed} passed, {validation.patched.failed} failed\n"
        f"- Validated commit: `{proposal.base_commit_sha}`\n"
        f"- Validation ID: `{validation.admission.validation_id}`"
    )
    with httpx.Client(
        base_url="https://api.github.com", follow_redirects=False, trust_env=False,
        timeout=httpx.Timeout(12, connect=5), transport=transport,
        headers={"Authorization": f"Bearer {token.get_secret_value()}",
                 "Accept": "application/vnd.github+json", "Accept-Encoding": "identity",
                 "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "RepoFix-AI/0.1.0"},
    ) as client:
        try:
            actor = GitHubIdentity.model_validate(_request(client, "GET", "/user"))
            if actor.id != identity.id or actor.login.casefold() != identity.login.casefold():
                raise DraftPullRequestError(403, "The local GitHub draft PR token belongs to a different account.")
            metadata = PublicRepository.model_validate(_request(client, "GET", prefix))
            if (
                metadata.id != proposal.repository_id or metadata.full_name != proposal.repository_full_name
                or metadata.owner.id != identity.id or metadata.archived or metadata.disabled
            ):
                raise DraftPullRequestError(409, "This public repository is no longer eligible for draft PR creation.")
            base = metadata.default_branch
            expected = _expected_tree(_tree(client, prefix, proposal.tree_sha), proposal.changes)

            def verify_tree(sha):
                actual = _expected_tree(_tree(client, prefix, sha), ())
                if expected != actual:
                    raise DraftPullRequestError(409, "The GitHub branch does not contain exactly the verified changes.")

            def verify_commit(sha):
                commit = _request(client, "GET", f"{prefix}/git/commits/{sha}")
                if commit["sha"] != sha or [parent["sha"] for parent in commit["parents"]] != [proposal.base_commit_sha]:
                    raise DraftPullRequestError(409, "The GitHub branch does not start from the verified commit.")
                verify_tree(_sha(commit["tree"]["sha"]))

            def require_current_base():
                current = _request(client, "GET", f"{prefix}/git/ref/heads/{quote(base, safe='/')}")
                if (current["ref"] != f"refs/heads/{base}" or current["object"]["type"] != "commit"
                    or current["object"]["sha"] != proposal.base_commit_sha):
                    raise DraftPullRequestError(409, "The default branch moved since validation. Generate and validate a new proposal.")

            def parse_pull(item, *, must_be_draft=False):
                number, draft, state = item["number"], item["draft"], item["state"]
                if (
                    type(number) is not int or number < 1 or type(draft) is not bool or state not in ("open", "closed")
                    or (must_be_draft and (not draft or state != "open"))
                    or item["head"]["ref"] != branch or item["base"]["ref"] != base
                    or any(item[key]["repo"]["id"] != metadata.id for key in ("base", "head"))
                ):
                    raise ValueError("Unexpected pull request.")
                return DraftPullRequest(number=number, url=f"{repo.canonical_url}/pull/{number}",
                                        branch=branch, draft=draft, state=state)

            pulls = _request(client, "GET", f"{prefix}/pulls?state=all&head={quote(repo.owner + ':' + branch, safe='')}&per_page=2")
            if not isinstance(pulls, list) or len(pulls) > 1:
                raise DraftPullRequestError(409, "More than one pull request matches this proposal branch. Check GitHub.")
            if pulls:
                existing = parse_pull(pulls[0])
                verify_commit(_sha(pulls[0]["head"]["sha"]))
                authorize()
                return existing

            require_current_base()
            ref = _request(client, "GET", ref_path, missing=True)
            if ref is None:
                authorize()
                tree = _request(client, "POST", f"{prefix}/git/trees", body={
                    "base_tree": proposal.tree_sha,
                    "tree": [{"path": change.path, "mode": change.mode, "type": "blob",
                              "content": change.proposed_content} for change in proposal.changes],
                })
                tree_sha = _sha(tree["sha"])
                verify_tree(tree_sha)
                authorize()
                commit = _request(client, "POST", f"{prefix}/git/commits", body={
                    "message": title, "tree": tree_sha, "parents": [proposal.base_commit_sha],
                })
                commit_sha = _sha(commit["sha"])
                if commit["tree"]["sha"] != tree_sha or [parent["sha"] for parent in commit["parents"]] != [proposal.base_commit_sha]:
                    raise ValueError("Unexpected created commit.")
                require_current_base()
                authorize()
                ref = _request(client, "POST", f"{prefix}/git/refs", body={
                    "ref": f"refs/heads/{branch}", "sha": commit_sha,
                })
                if ref["object"]["sha"] != commit_sha:
                    raise ValueError("Unexpected created reference.")
            else:
                commit_sha = _sha(ref["object"]["sha"])
                verify_commit(commit_sha)

            if ref["ref"] != f"refs/heads/{branch}" or ref["object"]["type"] != "commit":
                raise ValueError("Unexpected Git reference.")
            require_current_base()
            current_ref = _request(client, "GET", ref_path)
            if current_ref != ref and (
                current_ref["ref"] != f"refs/heads/{branch}" or current_ref["object"]["type"] != "commit"
                or current_ref["object"]["sha"] != commit_sha
            ):
                raise DraftPullRequestError(409, "The proposal branch changed. Check GitHub before continuing.")
            authorize()
            pull = _request(client, "POST", f"{prefix}/pulls", body={
                "title": title, "body": body, "head": branch, "base": base, "draft": True,
            })
            result = parse_pull(pull, must_be_draft=True)
            if pull["head"]["sha"] != commit_sha or pull["base"]["sha"] != proposal.base_commit_sha:
                raise DraftPullRequestError(409, "The branch changed during publication. Review the draft PR on GitHub.")
            return result
        except (KeyError, TypeError, ValueError, ValidationError, AttributeError, RecursionError):
            raise DraftPullRequestError(502, "GitHub returned an unexpected response. Check the proposal branch and pull requests before trying again.") from None
