"""Session-scoped results and explicitly enabled local validation execution."""

from threading import Lock
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Request, Response
from pydantic import field_validator

from app.api.auth import AUTH_HEADERS, _trusted_origin, require_github_identity
from app.api.investigations import _history_store, _http_error
from app.api.proposals import _proposal_store
from app.config import Settings
from app.github_draft_pr import DraftPullRequest, DraftPullRequestError, publish_draft_pr
from app.github_oauth import GitHubIdentity
from app.proposal_history import Identifier
from app.proposal_review import ProposalReviewError
from app.proposal_store import ProposalStoreError
from app.sandbox_profile import SandboxProfileError
from app.validation_admission import ValidationAdmissionError
from app.validation_context import ValidationContextError
from app.validation_history import MAX_RESULTS_PER_PROPOSAL, ValidationResultPage, ValidationResultStore, ValidationRunResult
from app.validation_request import ValidationInput, ValidationRequestError, bind_validation_request
from app.validation_runner import ValidationRunnerError, run_approved_validation


router = APIRouter(prefix="/api/validations", tags=["validations"])
_ID = r"^[0-9a-f]{32}$"
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_RUN_LOCK = Lock()
_DRAFT_LOCK = Lock()
_ERRORS = {
    "not_found": (404, "Saved proposal or validation result not found."),
    "approval_required": (409, "Approve the exact proposal for validation first."),
    "review_changed": (409, "The approval changed. Refresh the proposal review."),
    "proposal_changed": (409, "The saved proposal changed. Refresh and review it again."),
    "workspace_changed": (409, "The saved validation inputs changed."),
    "selection_changed": (409, "The saved repository is unavailable or changed."),
    "busy": (409, "This account has an active validation reservation."),
    "timeout": (504, "Validation timed out."),
    "github_unavailable": (502, "The saved repository could not be read from GitHub."),
    "docker_unavailable": (503, "Start Docker Desktop and check the local validation image."),
    "cleanup_unconfirmed": (503, "Container cleanup could not be confirmed. The reservation is still active."),
    "docker_cli_stop_unconfirmed": (503, "The Docker command could not be confirmed stopped. The reservation is still active."),
    "reservation_retained": (503, "The reservation could not be released. Do not repeat the run."),
    "result_save_failed": (503, "Validation result could not be saved. The reservation was retained."),
    "invalid_identity": (401, "Connect your GitHub account again."),
}


def _validation_error(error):
    status, detail = _ERRORS.get(error.code, (503, "Validation could not be completed."))
    return _http_error(status, detail)


class DraftPullRequestInput(ValidationInput):
    validation_id: Identifier
    confirm_publication: bool

    @field_validator("confirm_publication")
    @classmethod
    def require_confirmation(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("Confirm the GitHub draft PR publication.")
        return value


@router.post("/draft-pr", response_model=DraftPullRequest, summary="Create a GitHub draft PR for a verified patch")
def draft_pull_request(
    payload: DraftPullRequestInput, request: Request, response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
    csrf_header: Annotated[str | None, Header(alias="X-RepoFix-CSRF")] = None,
) -> DraftPullRequest:
    response.headers.update(AUTH_HEADERS)
    settings = getattr(request.app.state, "settings", None)
    if (
        not isinstance(settings, Settings) or settings.environment != "development"
        or request.client is None or request.client.host not in _LOOPBACK
        or request.url.hostname not in _LOOPBACK
    ):
        raise _http_error(403, "Draft PR creation is available only in local development.")
    origins = request.headers.getlist("origin")
    if (
        len(origins) != 1 or not _trusted_origin(origins[0], settings)
        or urlsplit(origins[0]).hostname not in _LOOPBACK
        or csrf_header != "1" or request.headers.getlist("x-repofix-csrf") != ["1"]
    ):
        raise _http_error(403, "Draft PR creation requires a trusted local Origin and X-RepoFix-CSRF: 1.")
    if settings.github_draft_pr_token is None:
        raise _http_error(503, "Configure REPOFIX_GITHUB_DRAFT_PR_TOKEN in the local backend first.")
    store = _proposal_store(request)
    binding_input = ValidationInput.model_validate(
        payload.model_dump(exclude={"validation_id", "confirm_publication"})
    )

    def authorize():
        if require_github_identity(request).id != identity.id:
            raise _http_error(401, "Connect your GitHub account again.")
        review = store.get_review(identity.id, proposal.proposal_id)
        current = bind_validation_request(identity, proposal, review, binding_input)
        if current.model_dump(exclude={"bound_at"}) != validation.admission.binding.model_dump(exclude={"bound_at"}):
            raise _http_error(409, "The verified result does not match the current approval and patch.")

    if not _DRAFT_LOCK.acquire(blocking=False):
        raise _http_error(429, "A draft PR request is already running.")
    try:
        proposal = store.get(identity.id, payload.proposal_id)
        if proposal is None:
            raise _http_error(404, "Saved proposal or validation result not found.")
        validation = ValidationResultStore(store).get(identity.id, payload.validation_id)
        if validation.verdict != "verified":
            raise _http_error(409, "Choose a saved result marked Patch verified.")
        authorize()
        return publish_draft_pr(identity, proposal, validation, settings.github_draft_pr_token, authorize=authorize)
    except DraftPullRequestError as error:
        raise _http_error(error.status, str(error)) from None
    except (ProposalReviewError, ValidationRequestError) as error:
        raise _http_error(409, "The approval no longer matches. Refresh the proposal review.") from None
    except ValidationAdmissionError as error:
        raise _validation_error(error) from None
    except ProposalStoreError:
        raise _http_error(503, "The saved proposal could not be read.") from None
    finally:
        _DRAFT_LOCK.release()


@router.post(
    "", status_code=201, response_model=ValidationRunResult,
    summary="Validate a saved, approved Python proposal locally",
    description=(
        "Requires local development opt-in, a connected GitHub account, a trusted local Origin "
        "and X-RepoFix-CSRF: 1. Uses the exact approved revision and local Docker image. "
        "Waits for both phases and cleanup, then saves their result. No AI call or GitHub write occurs."
    ),
)
def validate_proposal(
    payload: ValidationInput, request: Request, response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
    csrf_header: Annotated[str | None, Header(alias="X-RepoFix-CSRF")] = None,
) -> ValidationRunResult:
    response.headers.update(AUTH_HEADERS)
    settings = getattr(request.app.state, "settings", None)
    if (
        not isinstance(settings, Settings) or settings.environment != "development"
        or not settings.enable_local_validation or request.client is None
        or request.client.host not in _LOOPBACK or request.url.hostname not in _LOOPBACK
    ):
        raise _http_error(403, "Starting validations is available only in enabled local development.")
    origins = request.headers.getlist("origin")
    if (
        len(origins) != 1 or not _trusted_origin(origins[0], settings)
        or urlsplit(origins[0]).hostname not in _LOOPBACK
        or csrf_header != "1" or request.headers.getlist("x-repofix-csrf") != ["1"]
    ):
        raise _http_error(403, "Validation requires a trusted local Origin and X-RepoFix-CSRF: 1.")
    proposal_store, history = _proposal_store(request), _history_store(request)

    def authorize():
        if require_github_identity(request).id != identity.id:
            raise _http_error(401, "Connect your GitHub account again.")

    if not _RUN_LOCK.acquire(blocking=False):
        raise _http_error(429, "A local validation is already in progress.")
    try:
        return run_approved_validation(identity, payload, proposal_store, history, authorize=authorize)
    except HTTPException:
        raise
    except (ValidationAdmissionError, ValidationContextError, ValidationRunnerError) as error:
        raise _validation_error(error) from None
    except SandboxProfileError:
        raise _http_error(503, "The restricted Docker runtime is unavailable or failed its check.") from None
    except Exception:
        raise _http_error(500, "Validation could not be completed.") from None
    finally:
        _RUN_LOCK.release()


@router.get("", response_model=ValidationResultPage, summary="List saved validation results for a proposal")
def list_validations(
    request: Request, response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
    proposal_id: Annotated[str, Query(pattern=_ID)],
    page: Annotated[int, Query(ge=1, le=10000)] = 1,
    per_page: Annotated[int, Query(ge=1, le=MAX_RESULTS_PER_PROPOSAL)] = 10,
) -> ValidationResultPage:
    response.headers.update(AUTH_HEADERS)
    try:
        return ValidationResultStore(_proposal_store(request)).list(identity.id, proposal_id, page=page, per_page=per_page)
    except ValidationAdmissionError as error:
        raise _validation_error(error) from None


@router.get("/{validation_id}", response_model=ValidationRunResult, summary="Open a saved validation result")
def get_validation(
    request: Request, response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
    validation_id: Annotated[str, Path(pattern=_ID)],
) -> ValidationRunResult:
    response.headers.update(AUTH_HEADERS)
    try:
        return ValidationResultStore(_proposal_store(request)).get(identity.id, validation_id)
    except ValidationAdmissionError as error:
        raise _validation_error(error) from None
