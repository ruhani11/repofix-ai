"""Authenticated draft generation and saved proposal review; no target writes."""

import asyncio
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Request, Response
from pydantic import ValidationError

from app.api.auth import AUTH_HEADERS, _trusted_origin, require_github_identity
from app.api.investigations import (
    PREPARATION_TIMEOUT_SECONDS, _history_store, _http_error, _run_gate,
    _PROVIDER_WAIT_ERRORS, _RETRY_AFTER_HEADER, _RUN_ERRORS,
)
from app.config import Settings
from app.fix_proposal import FixProposalInput
from app.github_oauth import GitHubIdentity
from app.proposal_context import ProposalContextError, prepare_proposal_context
from app.proposal_decision import ProposalDecisionError
from app.proposal_generator import ProposalGenerationError, generate_fix_proposal
from app.proposal_history import ProposalHistoryPage, ProposalResponse, proposal_response
from app.proposal_review import (
    ProposalDecisionPage, ProposalDecisionRequest, ProposalReviewError, ProposalReviewState,
)
from app.proposal_store import ProposalStore, ProposalStoreError


router = APIRouter(prefix="/api/proposals", tags=["proposals"])
SAVE_TIMEOUT_SECONDS = 5
_CONTEXT_ERRORS = {
    "not_found": (404, "Saved investigation not found."),
    "history_unavailable": (503, "Investigation history is unavailable."),
    "ineligible_investigation": (409, "A saved investigation with a likely cause is required."),
    "selection_changed": (409, "The saved repository is unavailable or changed. Investigate it again."),
    "github_unavailable": (502, "Could not prepare the saved repository commit from GitHub."),
}
_GENERATION_ERRORS = {
    "not_configured": (503, "Configure the AI API key and model first."),
    "invalid_settings": (503, "Proposal generation requires at least two model turns."),
    "timeout": (504, "AI proposal generation timed out."),
    "turn_limit": (409, "Proposal generation reached its model turn limit."),
    "tool_limit": (409, "Proposal generation reached its tool attempt limit."),
    "event_limit": (409, "Proposal generation reached its event limit."),
    "invalid_proposal": (502, "The model did not return a valid fix proposal."),
    "unobserved_source": (502, "Proposed edits did not use source read during this run."),
    "invalid_proposal_source": (502, "Proposed changes could not be verified against the saved commit."),
    **{code: _RUN_ERRORS[code] for code in [*_PROVIDER_WAIT_ERRORS, "provider_timeout"]},
}


def _proposal_store(request):
    store = getattr(request.app.state, "proposal_store", None)
    if not isinstance(store, ProposalStore):
        raise _http_error(503, "Fix proposal storage is unavailable.")
    return store


@router.get(
    "", response_model=ProposalHistoryPage, summary="List your saved fix proposals",
    description="Account-scoped draft summaries, newest first. No AI or repository request occurs.",
    responses={401: {"description": "Connect GitHub first."}, 503: {"description": "Storage unavailable."}},
)
def list_proposals(
    request: Request, response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
    page: Annotated[int, Query(ge=1, le=10000)] = 1,
    per_page: Annotated[int, Query(ge=1, le=50)] = 20,
) -> ProposalHistoryPage:
    response.headers.update(AUTH_HEADERS)
    try:
        return _proposal_store(request).list(identity.id, page=page, per_page=per_page)
    except ProposalStoreError:
        raise _http_error(503, "Fix proposal storage is unavailable.") from None


@router.post(
    "/generate", response_model=ProposalResponse, summary="Draft a fix for a saved investigation",
    description=(
        "Send only investigation_run_id from your saved likely-cause report. "
        "Trusted Origin and X-RepoFix-CSRF: 1 are required. This sends issue/report "
        "text and selected source to the configured AI provider and may incur API charges. "
        "The repository is rechecked, the saved commit is used, and exact edits are "
        "source-validated before a server-built diff is returned. The same app-process "
        "slot and six-attempts/hour/account budget are shared with investigations; "
        "these limits reset on restart and are not a spending cap. "
        "Proposals are saved for later review; check proposal_saved. A late storage "
        "failure returns the draft with proposal_saved=false. A patch hash identifies "
        "bytes, not approval. No patch is applied, no target tests run, and nothing is "
        "committed or published. Multiple explicit generations create separate drafts."
    ),
    responses={
        401: {"description": "Connect GitHub first."},
        403: {"description": "Trusted Origin and CSRF header required."},
        404: {"description": "Saved investigation not found."},
        409: {"description": "Ineligible investigation, repository changed or generation budget reached."},
        429: {"description": "Shared busy/hourly limit or provider quota/rate limit.", "headers": _RETRY_AFTER_HEADER},
        502: {"description": "Repository/provider/draft validation failed."},
        503: {"description": "Configuration, storage or provider unavailable.", "headers": _RETRY_AFTER_HEADER},
        504: {"description": "Preparation or generation timed out."},
    },
)
async def generate_proposal(
    payload: FixProposalInput, request: Request, response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
    csrf_header: Annotated[str | None, Header(alias="X-RepoFix-CSRF")] = None,
) -> ProposalResponse:
    response.headers.update(AUTH_HEADERS)
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, Settings):
        raise _http_error(503, "Proposal services are unavailable.")
    origins = request.headers.getlist("origin")
    if (
        len(origins) != 1 or not _trusted_origin(origins[0], settings)
        or len(request.headers.getlist("x-repofix-csrf")) != 1 or csrf_header != "1"
    ):
        raise _http_error(403, "Proposal generation requires a trusted Origin and X-RepoFix-CSRF: 1.")
    if not settings.ai_configured:
        raise _http_error(503, "Configure the AI API key and model first.")
    if settings.ai_max_turns < 2:
        raise _http_error(503, "Proposal generation requires at least two model turns.")
    history = _history_store(request)
    store = _proposal_store(request)
    gate = _run_gate(request)
    gate.reserve(identity.id)
    started = perf_counter()
    worker = None

    def prepare():
        store.ensure_ready()
        return prepare_proposal_context(identity, payload, history)

    try:
        worker = asyncio.create_task(asyncio.to_thread(prepare))
        async with asyncio.timeout(PREPARATION_TIMEOUT_SECONDS):
            context = await asyncio.shield(worker)
        # Logging out/expiring while GitHub I/O runs must stop the paid call.
        if require_github_identity(request).id != identity.id:
            raise _http_error(401, "Connect your GitHub account again.")
        result = await generate_fix_proposal(context, settings)
        artifact = result.validated
        saved = context.history
        if (
            artifact.investigation_run_id != payload.investigation_run_id
            or artifact.repository_id != saved.repository_id
            or artifact.repository_full_name != saved.investigation.repository_full_name
            or artifact.base_commit_sha != saved.investigation.commit_sha
            or artifact.tree_sha != context.repository_context.tree.tree_sha
            or context.repository_context.identity.id != identity.id
        ):
            raise _http_error(502, "Proposal provenance could not be verified.")
        completed = proposal_response(result, total_duration_ms=round((perf_counter() - started) * 1000))
        worker = asyncio.create_task(asyncio.to_thread(store.save, identity.id, completed))
        try:
            async with asyncio.timeout(SAVE_TIMEOUT_SECONDS):
                await asyncio.shield(worker)
        except (ProposalStoreError, TimeoutError):
            # Timeout can mean a pending write; don't automatically repeat paid work.
            return completed.model_copy(update={"proposal_saved": False})
        return completed
    except HTTPException:
        raise
    except ProposalContextError as error:
        status, detail = _CONTEXT_ERRORS.get(error.code, (502, "The saved proposal context could not be verified."))
        raise _http_error(status, detail) from None
    except ProposalGenerationError as error:
        status, detail = _GENERATION_ERRORS.get(error.code, (502, "The AI fix proposal could not be completed."))
        hint = error.retry_after if error.code in _PROVIDER_WAIT_ERRORS else None
        raise _http_error(status, detail, hint) from None
    except ProposalStoreError:
        raise _http_error(503, "Fix proposal storage is unavailable.") from None
    except TimeoutError:
        raise _http_error(504, "Saved repository preparation timed out.") from None
    except (ValidationError, ValueError):
        raise _http_error(502, "The generated proposal artifact could not be verified.") from None
    except Exception:
        raise _http_error(500, "The fix proposal could not be completed.") from None
    finally:
        if worker is not None and not worker.done():
            gate.release_when_preparation_finishes(worker)
        else:
            gate.release()


@router.get(
    "/{proposal_id}", response_model=ProposalResponse, summary="Open a saved fix proposal",
    description="Returns the saved draft, exact original/proposed files, review diff and patch hash. No AI call, approval or code execution occurs.",
    responses={401: {"description": "Connect GitHub first."}, 404: {"description": "Proposal not found."}, 503: {"description": "Storage unavailable."}},
)
def get_proposal(
    proposal_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")],
    request: Request, response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
) -> ProposalResponse:
    response.headers.update(AUTH_HEADERS)
    try:
        detail = _proposal_store(request).get(identity.id, proposal_id)
    except ProposalStoreError:
        raise _http_error(503, "Fix proposal storage is unavailable.") from None
    if detail is None:
        raise _http_error(404, "Saved proposal not found.")
    return detail


def _review_http_error(error):
    status, detail = {
        "not_found": (404, "Saved proposal not found."),
        "revision_conflict": (409, "The review decision changed. Refresh its current revision and review again."),
        "proposal_changed": (409, "The reviewed proposal, base commit or patch hash changed. Refresh and review again."),
        "not_saved": (409, "Save the proposal before recording a decision."),
        "invalid_time": (409, "The review time is invalid. Check the server clock."),
        "invalid_identity": (401, "Connect your GitHub account again."),
        "invalid_input": (400, "The proposal review decision is invalid."),
    }.get(error.code, (503, "Proposal review storage is unavailable."))
    return _http_error(status, detail)


@router.get(
    "/{proposal_id}/decision", response_model=ProposalReviewState,
    summary="Read the current human review decision",
    description=(
        "Returns unreviewed at revision 0 until a human records a decision. The latest "
        "revision determines approved_for_validation or rejected. This is a saved "
        "point-in-time review state, not an execution token or test result. No AI, "
        "repository request, patch or test execution occurs."
    ),
    responses={401: {"description": "Connect GitHub first."}, 404: {"description": "Proposal not found."}, 503: {"description": "Review storage unavailable."}},
)
def get_proposal_decision(
    proposal_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")],
    request: Request, response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
) -> ProposalReviewState:
    response.headers.update(AUTH_HEADERS)
    try:
        return _proposal_store(request).get_review(identity.id, proposal_id)
    except (ProposalReviewError, ProposalDecisionError) as error:
        raise _review_http_error(error) from None


@router.get(
    "/{proposal_id}/decisions", response_model=ProposalDecisionPage,
    summary="List your proposal's human decision history",
    description="Immutable review events, newest revision first, scoped to the connected account. No AI or execution occurs.",
    responses={401: {"description": "Connect GitHub first."}, 404: {"description": "Proposal not found."}, 503: {"description": "Review storage unavailable."}},
)
def list_proposal_decisions(
    proposal_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")],
    request: Request, response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
    page: Annotated[int, Query(ge=1, le=10000)] = 1,
    per_page: Annotated[int, Query(ge=1, le=50)] = 20,
) -> ProposalDecisionPage:
    response.headers.update(AUTH_HEADERS)
    try:
        return _proposal_store(request).list_decisions(identity.id, proposal_id, page=page, per_page=per_page)
    except (ProposalReviewError, ProposalDecisionError) as error:
        raise _review_http_error(error) from None


@router.post(
    "/decision", response_model=ProposalReviewState,
    summary="Record your review decision for an exact saved proposal",
    description=(
        "Read GET /api/proposals/{proposal_id}/decision first. Send its current revision "
        "as expected_revision and the exact proposal ID, base commit and patch hash "
        "you reviewed. Explicitly choose approve_for_validation or reject. Trusted "
        "Origin and X-RepoFix-CSRF: 1 are required. The service atomically checks "
        "ownership, bytes and revision, rechecks your session and appends a human "
        "decision. A later reject revokes approval for future validation; a fresh "
        "explicit approval can restore it. Conflicting/stale requests return 409 and "
        "must refresh state. History and proposals are not overwritten. Approval "
        "is limited to future restricted validation and starts no work. No AI, "
        "target patch/test execution, GitHub write or publication occurs."
    ),
    responses={
        400: {"description": "Invalid internal review input."},
        401: {"description": "Session missing, expired or revoked."},
        403: {"description": "Trusted Origin and CSRF header required."},
        404: {"description": "Proposal not found."},
        409: {"description": "Stale revision/proposal or invalid server time."},
        503: {"description": "Review storage unavailable; read current state before retrying."},
    },
)
def record_proposal_decision(
    payload: ProposalDecisionRequest, request: Request, response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
    csrf_header: Annotated[str | None, Header(alias="X-RepoFix-CSRF")] = None,
) -> ProposalReviewState:
    response.headers.update(AUTH_HEADERS)
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, Settings):
        raise _http_error(503, "Proposal review services are unavailable.")
    origins = request.headers.getlist("origin")
    if (
        len(origins) != 1 or not _trusted_origin(origins[0], settings)
        or len(request.headers.getlist("x-repofix-csrf")) != 1 or csrf_header != "1"
    ):
        raise _http_error(403, "Proposal review requires a trusted Origin and X-RepoFix-CSRF: 1.")

    def recheck_session():
        if require_github_identity(request).id != identity.id:
            raise _http_error(401, "Connect your GitHub account again.")

    try:
        return _proposal_store(request).record_decision(
            identity, payload.decision_input(), expected_revision=payload.expected_revision,
            authorize=recheck_session,
        )
    except (ProposalReviewError, ProposalDecisionError) as error:
        raise _review_http_error(error) from None
