"""HTTP interfaces for public repository listing, selection and inspection."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from app.api.auth import AUTH_HEADERS, _context, _workbench_url, require_github_identity
from app.github_client import (
    GitHubClientError, RepositoryMetadata, RepositorySnapshot, _fetch_response_body,
    _snapshot_from_metadata, fetch_repository_snapshot,
)
from app.github_files import (
    MAX_FILE_BYTES, RepositoryFile, _check_file_policy, read_repository_file,
)
from app.github_oauth import GitHubIdentity
from app.github_repositories import (
    AccountRepository,
    InspectorRepositoryPage,
    MAX_REPOSITORIES_PER_PAGE,
    MAX_REPOSITORY_PAGES,
    PublicRepository,
    RepositoryPage,
    fetch_inspector_repositories,
    fetch_public_repositories,
)
from app.github_selection import (
    RepositorySelectionError,
    RepositorySelectionInput,
    fetch_selected_repository,
)
from app.github_tree import RepositoryTree, TreeEntry, fetch_repository_tree
from app.github_url import parse_github_repository_url


MAX_SELECTED_FILES = 3
RepositoryPath = Annotated[str, Field(min_length=1, max_length=1024)]
router = APIRouter(prefix="/api/repositories", tags=["repositories"])


class RepositoryListingQuery(BaseModel):
    """Only pagination comes from the caller; identity comes from the session."""

    model_config = ConfigDict(extra="forbid")

    page: int = Field(default=1, ge=1, le=MAX_REPOSITORY_PAGES)
    per_page: int = Field(
        default=MAX_REPOSITORIES_PER_PAGE,
        ge=1,
        le=MAX_REPOSITORIES_PER_PAGE,
    )


@router.get(
    "",
    response_model=RepositoryPage,
    summary="List connected account's public repositories",
    description=(
        "Connect GitHub using the browser login link first. "
        "This endpoint lists public repositories owned by the connected account. "
        "The account is taken from the session, never a supplied username. "
        "Keep per_page unchanged when advancing pages. "
        "A full page may have a next page; that next page can be empty. "
        "Listing does not approve a repository change or start an investigation."
    ),
    responses={
        401: {"description": "Connect GitHub first; session missing or expired."},
        502: {"description": "GitHub listing or response validation failed."},
        503: {"description": "Authentication services are unavailable."},
    },
)
def list_public_repositories(
    response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
    pagination: Annotated[RepositoryListingQuery, Query()],
) -> RepositoryPage:
    response.headers.update(AUTH_HEADERS)

    try:
        return fetch_public_repositories(
            identity,
            page=pagination.page,
            per_page=pagination.per_page,
        )
    except GitHubClientError:
        raise HTTPException(
            status_code=502,
            detail="Could not list public repositories from GitHub. Try later.",
            headers=AUTH_HEADERS,
        ) from None


@router.post(
    "/select",
    response_model=PublicRepository,
    summary="Validate a repository selection for the connected account",
    description=(
        "Send the repository ID and name from the public repository listing. "
        "The owner is taken from your session. Current GitHub metadata is "
        "checked again before returning the repository. "
        "This endpoint does not store a selection or start an investigation. "
        "It does not authorize repository changes."
    ),
    responses={
        400: {"description": "Repository selection changed or is unsupported."},
        401: {"description": "Connect GitHub first; session missing or expired."},
        502: {"description": "GitHub access or metadata validation failed."},
        503: {"description": "Authentication services are unavailable."},
    },
)
def select_repository(
    payload: RepositorySelectionInput,
    response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
) -> PublicRepository:
    response.headers.update(AUTH_HEADERS)

    try:
        return fetch_selected_repository(identity, payload)
    except RepositorySelectionError:
        raise HTTPException(
            status_code=400,
            detail=(
                "Repository selection is unavailable or changed. "
                "Refresh the repository list and select again."
            ),
            headers=AUTH_HEADERS,
        ) from None
    except GitHubClientError:
        raise HTTPException(
            status_code=502,
            detail="Could not verify the repository with GitHub. Try later.",
            headers=AUTH_HEADERS,
        ) from None


def _validate_inspection_paths(paths: list[str]) -> list[str]:
    if len(paths) != len(set(paths)):
        raise ValueError("Selected file paths must be unique.")
    for path in paths:
        TreeEntry.validate_relative_path(path)
        try:
            _check_file_policy(path)
        except GitHubClientError as error:
            raise ValueError(str(error)) from None
    return paths


class RepositoryInspectionRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    repository_url: str = Field(min_length=1, max_length=300)
    selected_paths: list[RepositoryPath] = Field(
        default_factory=list, max_length=MAX_SELECTED_FILES
    )

    @field_validator("repository_url")
    @classmethod
    def validate_repository_url(cls, value: str) -> str:
        return parse_github_repository_url(value).canonical_url

    _validate_paths = field_validator("selected_paths")(_validate_inspection_paths)


class RepositoryInspectionResponse(BaseModel):
    snapshot: RepositorySnapshot
    tree: RepositoryTree
    files: list[RepositoryFile] = Field(max_length=MAX_SELECTED_FILES)


def _validate_selected_entries(tree: RepositoryTree, paths: list[str]) -> None:
    entries = {entry.path: entry for entry in tree.entries}
    for path in paths:
        entry = entries.get(path)
        if entry is None:
            raise HTTPException(status_code=400, detail="Selected file is not in the tree.")
        if entry.type != "blob" or entry.mode not in {"100644", "100755"}:
            raise HTTPException(status_code=400, detail="Select regular text files only.")
        if entry.size is None or entry.size > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=400, detail="Selected file size is unknown or exceeds 100 KiB."
            )


@router.post(
    "/inspect",
    response_model=RepositoryInspectionResponse,
    responses={
        400: {"description": "Selected file is unavailable or unsupported."},
        502: {"description": "GitHub access or response validation failed."},
    },
)
def inspect_repository(
    payload: RepositoryInspectionRequest,
) -> RepositoryInspectionResponse:
    try:
        snapshot = fetch_repository_snapshot(payload.repository_url)
        tree = fetch_repository_tree(snapshot)
    except GitHubClientError as error:
        raise HTTPException(status_code=502, detail=str(error)) from None

    # Validate every selection before downloading any selected file content.
    _validate_selected_entries(tree, payload.selected_paths)

    try:
        files = [read_repository_file(tree, path) for path in payload.selected_paths]
    except GitHubClientError as error:
        raise HTTPException(status_code=502, detail=str(error)) from None

    return RepositoryInspectionResponse(snapshot=snapshot, tree=tree, files=files)


class InspectorSelection(RepositorySelectionInput):
    selected_paths: list[RepositoryPath] = Field(
        default_factory=list, max_length=MAX_SELECTED_FILES
    )

    _validate_paths = field_validator("selected_paths")(_validate_inspection_paths)


def _local_inspector_token(request: Request, identity: GitHubIdentity) -> SecretStr | None:
    settings, _ = _context(request)
    token = settings.github_inspector_token
    if token is None or _workbench_url(settings) is None:
        return None
    # Verify the token belongs to this cookie's account before every private operation.
    try:
        account = GitHubIdentity.model_validate_json(_fetch_response_body("/user", token=token))
    except (GitHubClientError, ValidationError):
        raise HTTPException(
            status_code=502,
            detail="Could not verify the local GitHub read token. Check its configuration and expiration.",
            headers=AUTH_HEADERS,
        ) from None
    if account.id != identity.id or account.login.casefold() != identity.login.casefold():
        return None
    return token


@router.get("/inspector", response_model=InspectorRepositoryPage)
def list_inspector_repositories(
    request: Request,
    response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
    pagination: Annotated[RepositoryListingQuery, Query()],
) -> InspectorRepositoryPage:
    response.headers.update(AUTH_HEADERS)
    token = _local_inspector_token(request, identity)
    try:
        return fetch_inspector_repositories(
            identity, page=pagination.page, per_page=pagination.per_page, token=token
        )
    except GitHubClientError:
        raise HTTPException(
            status_code=502, detail="Could not load your GitHub repositories. Try again.",
            headers=AUTH_HEADERS,
        ) from None


@router.post("/inspector/inspect", response_model=RepositoryInspectionResponse)
def inspect_selected_repository(
    payload: InspectorSelection,
    request: Request,
    response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
) -> RepositoryInspectionResponse:
    response.headers.update(AUTH_HEADERS)
    token = _local_inspector_token(request, identity)
    try:
        body = _fetch_response_body(f"/repos/{identity.login}/{payload.repository_name}", token=token)
        repository = AccountRepository.model_validate_json(body)
    except (GitHubClientError, ValidationError):
        raise HTTPException(
            status_code=502, detail="Could not read the selected repository from GitHub. Check access and refresh the list.",
            headers=AUTH_HEADERS,
        ) from None
    if (
        repository.id != payload.repository_id
        or repository.name.casefold() != payload.repository_name.casefold()
        or repository.owner.id != identity.id
        or repository.owner.login.casefold() != identity.login.casefold()
        or repository.disabled
        or (repository.private and token is None)
    ):
        raise HTTPException(
            status_code=400, detail="The repository is unavailable or has changed. Refresh the list.",
            headers=AUTH_HEADERS,
        )
    try:
        metadata = RepositoryMetadata.model_validate(repository.model_dump())
        snapshot = _snapshot_from_metadata(metadata, token=token)
        tree = fetch_repository_tree(snapshot, token=token)
        _validate_selected_entries(tree, payload.selected_paths)
        files = [read_repository_file(tree, path, token=token) for path in payload.selected_paths]
    except GitHubClientError as error:
        raise HTTPException(status_code=502, detail=str(error), headers=AUTH_HEADERS) from None
    except HTTPException as error:
        error.headers = AUTH_HEADERS
        raise
    return RepositoryInspectionResponse(snapshot=snapshot, tree=tree, files=files)
