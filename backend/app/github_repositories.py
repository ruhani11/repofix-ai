"""Read bounded pages of public repositories owned by a verified identity."""

from typing import Annotated, Literal
from urllib.parse import urlencode

import httpx
from pydantic import (
    BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError,
    computed_field, model_validator,
)

from app.github_client import GitHubClientError, _fetch_response_body
from app.github_oauth import GitHubIdentity


MAX_REPOSITORIES_PER_PAGE = 20
MAX_REPOSITORY_PAGES = 50


class AccountRepository(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)

    id: int = Field(gt=0)
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    full_name: str = Field(min_length=3, max_length=140)
    owner: GitHubIdentity
    private: bool
    visibility: Literal["public", "private"]
    description: str | None = Field(default=None, max_length=4096)
    language: str | None = Field(default=None, max_length=100)
    default_branch: str = Field(min_length=1, max_length=255)
    fork: bool
    archived: bool
    disabled: bool

    @model_validator(mode="after")
    def validate_repository(self) -> "AccountRepository":
        if self.private != (self.visibility == "private"):
            raise ValueError("Repository privacy and visibility do not match.")
        if self.name in {".", ".."}:
            raise ValueError("Invalid repository name.")
        expected = f"{self.owner.login}/{self.name}"
        if self.full_name != expected:
            raise ValueError("Repository owner and full name do not match.")
        return self

    @computed_field
    @property
    def html_url(self) -> str:
        """Construct the link locally; never trust a returned upstream URL."""
        return f"https://github.com/{self.full_name}"


class PublicRepository(AccountRepository):
    visibility: Literal["public"]


class RepositoryPage(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    account_login: str
    page: int = Field(ge=1, le=MAX_REPOSITORY_PAGES)
    per_page: int = Field(ge=1, le=MAX_REPOSITORIES_PER_PAGE)
    repositories: list[PublicRepository] = Field(
        max_length=MAX_REPOSITORIES_PER_PAGE
    )
    may_have_more: bool
    next_page: int | None = Field(
        default=None, ge=1, le=MAX_REPOSITORY_PAGES
    )
    limit_reached: bool


class InspectorRepositoryPage(RepositoryPage):
    repositories: list[AccountRepository] = Field(max_length=MAX_REPOSITORIES_PER_PAGE)
    private_access: bool


_ACCOUNT_REPOSITORY_LIST = TypeAdapter(
    Annotated[list[AccountRepository], Field(max_length=MAX_REPOSITORIES_PER_PAGE)]
)


_REPOSITORY_LIST = TypeAdapter(
    Annotated[
        list[PublicRepository],
        Field(max_length=MAX_REPOSITORIES_PER_PAGE),
    ]
)


def fetch_public_repositories(
    identity: GitHubIdentity,
    *,
    page: int = 1,
    per_page: int = MAX_REPOSITORIES_PER_PAGE,
    transport: httpx.BaseTransport | None = None,
) -> RepositoryPage:
    """Fetch one page; the API caller must obtain identity from its session.

    A full page permits a bounded next-page probe, which may return empty.
    No total count or guaranteed extra page is inferred from a full page.
    Keep per_page unchanged when advancing pages. No URLs are auto-followed.
    """
    return _fetch_repository_page(identity, page=page, per_page=per_page, transport=transport)


def fetch_inspector_repositories(
    identity: GitHubIdentity,
    *,
    page: int,
    per_page: int,
    token: SecretStr | None,
) -> InspectorRepositoryPage:
    listing = _fetch_repository_page(identity, page=page, per_page=per_page, token=token)
    if isinstance(listing, InspectorRepositoryPage):
        return listing
    return InspectorRepositoryPage(**listing.model_dump(), private_access=False)


def _fetch_repository_page(
    identity: GitHubIdentity,
    *,
    page: int,
    per_page: int,
    token: SecretStr | None = None,
    transport: httpx.BaseTransport | None = None,
) -> RepositoryPage:
    if not isinstance(identity, GitHubIdentity):
        raise ValueError("A verified GitHub identity is required.")
    try:
        identity = GitHubIdentity.model_validate(identity.model_dump())
    except ValidationError:
        raise ValueError("Invalid GitHub identity.") from None

    if type(page) is not int or not 1 <= page <= MAX_REPOSITORY_PAGES:
        raise ValueError(f"page must be an integer from 1 to {MAX_REPOSITORY_PAGES}.")
    if (
        type(per_page) is not int
        or not 1 <= per_page <= MAX_REPOSITORIES_PER_PAGE
    ):
        raise ValueError(
            f"per_page must be an integer from 1 to {MAX_REPOSITORIES_PER_PAGE}."
        )

    scope = {"type": "owner"} if token is None else {"affiliation": "owner", "visibility": "all"}
    query = urlencode({
        **scope,
        "sort": "full_name",
        "direction": "asc",
        "per_page": per_page,
        "page": page,
    })
    endpoint = f"/users/{identity.login}/repos" if token is None else "/user/repos"
    body = _fetch_response_body(f"{endpoint}?{query}", transport=transport, token=token)
    adapter = _REPOSITORY_LIST if token is None else _ACCOUNT_REPOSITORY_LIST
    try:
        repositories = adapter.validate_json(body)
    except ValidationError:
        kind = "public repositories" if token is None else "account repositories"
        raise GitHubClientError(f"GitHub returned invalid {kind}.") from None

    if len(repositories) > per_page:
        raise GitHubClientError("GitHub returned more repositories than requested.")
    if any(
        repo.owner.id != identity.id
        or repo.owner.login.lower() != identity.login.lower()
        for repo in repositories
    ):
        raise GitHubClientError("GitHub returned repositories for a different account.")
    if (
        len({repo.id for repo in repositories}) != len(repositories)
        or len({repo.full_name.lower() for repo in repositories}) != len(repositories)
    ):
        raise GitHubClientError("GitHub returned duplicate repositories.")

    may_have_more = len(repositories) == per_page
    limit_reached = may_have_more and page == MAX_REPOSITORY_PAGES
    page_model = RepositoryPage if token is None else InspectorRepositoryPage
    return page_model(
        **({"private_access": True} if token is not None else {}),
        account_login=identity.login,
        page=page,
        per_page=per_page,
        repositories=repositories,
        may_have_more=may_have_more,
        next_page=page + 1 if may_have_more and not limit_reached else None,
        limit_reached=limit_reached,
    )
