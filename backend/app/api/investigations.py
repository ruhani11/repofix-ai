"""Authenticated preparation and bounded read-only investigation APIs."""

import asyncio
from math import ceil
from threading import Lock
from time import monotonic, perf_counter
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from app.api.auth import AUTH_HEADERS, _trusted_origin, require_github_identity
from app.config import Settings
from app.github_client import SHA_PATTERN, GitHubClientError
from app.github_oauth import GitHubIdentity
from app.github_repositories import PublicRepository
from app.github_selection import RepositorySelectionError
from app.github_tree import MAX_TREE_ENTRIES
from app.investigation_context import prepare_investigation_context
from app.investigation_input import InvestigationInput
from app.investigation_history import (
    InvestigationHistoryDetail, InvestigationHistoryPage, InvestigationRunResponse,
)
from app.investigation_store import InvestigationStore, InvestigationStoreError
from app.investigation_runner import (
    MAX_RETRY_AFTER_SECONDS, InvestigationRunError, run_investigation,
)


router = APIRouter(prefix="/api/investigations", tags=["investigations"])

PREPARATION_TIMEOUT_SECONDS = 30
RUN_WINDOW_SECONDS = 3600
MAX_RUN_ATTEMPTS_PER_HOUR = 6
MAX_TRACKED_ACCOUNTS = 1000
_GATE_INIT_LOCK = Lock()
_RUN_ERRORS = {
    "not_configured": (503, "Configure the AI API key and model first."),
    "timeout": (504, "AI investigation timed out. Try a more focused issue."),
    "turn_limit": (409, "Investigation reached its model turn limit."),
    "tool_limit": (409, "Investigation reached its tool attempt limit."),
    "event_limit": (409, "Investigation reached its event limit."),
    "ungrounded_report": (502, "The report's source citations could not be verified."),
    "invalid_report": (502, "The model did not return a valid investigation report."),
    "model_error": (502, "The AI provider could not complete this investigation."),
    "provider_daily_quota": (
        429, "AI provider daily quota is exhausted. Check provider usage and retry after reset.",
    ),
    "provider_rate_limit": (
        429, "AI provider request rate limit reached. Wait before trying again.",
    ),
    "provider_limit": (
        429, "AI provider quota or rate limit reached. Check provider usage before retrying.",
    ),
    "provider_unavailable": (
        503, "The AI provider is temporarily unavailable. Try again later.",
    ),
    "provider_timeout": (
        504, "The AI provider request timed out. Try again later.",
    ),
}
_PROVIDER_WAIT_ERRORS = frozenset({
    "provider_daily_quota", "provider_rate_limit", "provider_limit",
    "provider_unavailable",
})
_RETRY_AFTER_HEADER = {
    "Retry-After": {
        "description": (
            "Suggested wait in seconds, when known. Uses the local throttle "
            "delay or a validated provider hint; availability is not guaranteed."
        ),
        "schema": {"type": "integer", "minimum": 1, "maximum": MAX_RETRY_AFTER_SECONDS},
    },
}


def _http_error(status: int, detail: str, retry_after: int | None = None):
    headers = dict(AUTH_HEADERS)
    if type(retry_after) is int and 1 <= retry_after <= MAX_RETRY_AFTER_SECONDS:
        headers["Retry-After"] = str(retry_after)
    return HTTPException(status_code=status, detail=detail, headers=headers)


class _RunGate:
    """App-owned, single-process concurrency and attempt throttle."""

    def __init__(self):
        self._lock = Lock()
        self._active = False
        self._history: dict[int, list[float]] = {}
        self._pending_preparation: asyncio.Task | None = None

    def reserve(self, account_id: int):
        now = monotonic()
        with self._lock:
            if self._active:
                raise _http_error(429, "An investigation is already running. Try later.", 5)
            recent = {
                account: [t for t in times if now - t < RUN_WINDOW_SECONDS]
                for account, times in self._history.items()
            }
            self._history = {account: times for account, times in recent.items() if times}
            times = self._history.get(account_id, [])
            if len(times) >= MAX_RUN_ATTEMPTS_PER_HOUR:
                retry = ceil(times[0] + RUN_WINDOW_SECONDS - now)
                raise _http_error(429, "Hourly investigation attempt limit reached.", retry)
            if account_id not in self._history and len(self._history) >= MAX_TRACKED_ACCOUNTS:
                raise _http_error(503, "Investigation capacity is temporarily unavailable.", 60)
            self._history[account_id] = [*times, now]
            self._active = True

    def release(self):
        with self._lock:
            self._active = False
            self._pending_preparation = None

    def release_when_preparation_finishes(self, task: asyncio.Task):
        # Keep a strong reference and the slot while the shielded worker finishes.
        self._pending_preparation = task

        def finished(completed):
            try:
                if not completed.cancelled():
                    completed.exception()  # Consume errors without exposing their text.
            finally:
                self.release()

        task.add_done_callback(finished)


def _run_gate(request: Request) -> _RunGate:
    with _GATE_INIT_LOCK:
        gate = getattr(request.app.state, "investigation_run_gate", None)
        if gate is None:
            gate = _RunGate()
            request.app.state.investigation_run_gate = gate
        if not isinstance(gate, _RunGate):
            raise _http_error(503, "Investigation services are unavailable.")
        return gate


def _history_store(request: Request) -> InvestigationStore:
    store = getattr(request.app.state, "investigation_store", None)
    if not isinstance(store, InvestigationStore):
        raise _http_error(503, "Investigation history is unavailable.")
    return store


class InvestigationPreparationResponse(BaseModel):
    """Preparation summary, without an agent diagnosis or persisted run ID."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    status: Literal["prepared"] = "prepared"
    repository: PublicRepository
    commit_sha: str = Field(pattern=SHA_PATTERN)
    tree_sha: str = Field(pattern=SHA_PATTERN)
    entry_count: int = Field(ge=0, le=MAX_TREE_ENTRIES)
    issue_description: str = Field(min_length=1, max_length=8000)


@router.get(
    "", response_model=InvestigationHistoryPage,
    summary="List your saved investigations",
    description="Lists completed reports for the connected GitHub account, newest first. No AI or GitHub repository request is made.",
    responses={401: {"description": "Connect GitHub first."}, 503: {"description": "History is unavailable."}},
)
def list_investigations(
    request: Request, response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
    page: Annotated[int, Query(ge=1, le=10000)] = 1,
    per_page: Annotated[int, Query(ge=1, le=50)] = 20,
) -> InvestigationHistoryPage:
    response.headers.update(AUTH_HEADERS)
    try:
        return _history_store(request).list(identity.id, page=page, per_page=per_page)
    except InvestigationStoreError:
        raise _http_error(503, "Investigation history is unavailable.") from None


@router.get(
    "/{run_id}", response_model=InvestigationHistoryDetail,
    summary="Open one of your saved investigations",
    description="Returns the saved issue, report, pinned commit and metadata. Reconnect GitHub after a server restart to access your history.",
    responses={401: {"description": "Connect GitHub first."}, 404: {"description": "Saved investigation not found."}, 503: {"description": "History is unavailable."}},
)
def get_investigation(
    run_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")],
    request: Request, response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
) -> InvestigationHistoryDetail:
    response.headers.update(AUTH_HEADERS)
    try:
        detail = _history_store(request).get(identity.id, run_id)
    except InvestigationStoreError:
        raise _http_error(503, "Investigation history is unavailable.") from None
    if detail is None:
        # Missing and another account's run are deliberately indistinguishable.
        raise _http_error(404, "Saved investigation not found.")
    return detail


@router.post(
    "/prepare",
    response_model=InvestigationPreparationResponse,
    summary="Prepare a selected repository for bug investigation",
    description=(
        "Connect GitHub first. Send repository ID/name from your public listing "
        "and describe the bug. Owner identity comes from the login session. "
        "The service rechecks selection, captures a commit and reads its bounded "
        "tree. entry_count includes files, directories and other Git entries. "
        "The response is a preparation summary; no run is persisted and no "
        "AI diagnosis, source-text download, test execution or repository "
        "change occurs. Investigation runs prepare their own fresh context."
    ),
    responses={
        400: {"description": "Repository selection changed or is unsupported."},
        401: {"description": "Connect GitHub first; session missing or expired."},
        502: {"description": "GitHub reads or repository evidence validation failed."},
        503: {"description": "Authentication services are unavailable."},
    },
)
def prepare_investigation(
    payload: InvestigationInput,
    response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
) -> InvestigationPreparationResponse:
    response.headers.update(AUTH_HEADERS)

    try:
        context = prepare_investigation_context(identity, payload)
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
            detail="Could not prepare repository evidence from GitHub. Try later.",
            headers=AUTH_HEADERS,
        ) from None

    return InvestigationPreparationResponse(
        repository=context.repository,
        commit_sha=context.snapshot.commit_sha,
        tree_sha=context.tree.tree_sha,
        entry_count=len(context.tree.entries),
        issue_description=context.request.issue_description,
    )


@router.post(
    "/run",
    response_model=InvestigationRunResponse,
    summary="Investigate a selected repository issue with AI",
    description=(
        "Connect GitHub first. Use repository ID/name from the public listing "
        "and describe the issue. Set X-RepoFix-CSRF to 1; a trusted browser "
        "Origin is required. This action sends issue text and selected source "
        "excerpts to the configured AI provider and may incur API charges. "
        "The service prepares fresh commit-pinned context and verifies report "
        "citations. completed means the workflow finished; inspect report.status "
        "for likely_cause or insufficient_evidence. No patch, test execution or "
        "GitHub change occurs. Local limits: one active run per app process and "
        "six attempts per hour per account, including failed preparations. "
        "Limits reset on restart and are not a monetary spending cap. "
        "Provider quota/rate limits return 429, provider unavailability returns "
        "503, and provider/workflow timeouts return 504. A valid provider wait "
        "hint is returned in Retry-After when available."
        " Completed reports are saved to your local history. Check history_saved; "
        "if saving fails after the model completes, the report is still returned "
        "with history_saved=false. History availability is checked before AI work."
    ),
    responses={
        400: {"description": "Repository selection changed or is unsupported."},
        401: {"description": "Connect GitHub first; session missing or expired."},
        403: {"description": "Trusted Origin and CSRF header required."},
        409: {"description": "Investigation turn or tool limit reached."},
        429: {
            "description": "Investigation busy, hourly attempt limit or provider quota/rate limit reached.",
            "headers": _RETRY_AFTER_HEADER,
        },
        502: {"description": "GitHub, AI provider or report validation failed."},
        503: {
            "description": "AI configuration, provider or investigation services unavailable.",
            "headers": _RETRY_AFTER_HEADER,
        },
        504: {"description": "Repository preparation, provider request or AI investigation timed out."},
    },
)
async def investigate_issue(
    payload: InvestigationInput,
    request: Request,
    response: Response,
    identity: Annotated[GitHubIdentity, Depends(require_github_identity)],
    csrf_header: Annotated[str | None, Header(alias="X-RepoFix-CSRF")] = None,
) -> InvestigationRunResponse:
    response.headers.update(AUTH_HEADERS)
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, Settings):
        raise _http_error(503, "Investigation services are unavailable.")
    origins = request.headers.getlist("origin")
    csrf_headers = request.headers.getlist("x-repofix-csrf")
    if (
        len(origins) != 1
        or not _trusted_origin(origins[0], settings)
        or len(csrf_headers) != 1
        or csrf_header != "1"
    ):
        raise _http_error(403, "Investigation requires a trusted Origin and X-RepoFix-CSRF: 1.")
    if not settings.ai_configured:
        raise _http_error(503, "Configure the AI API key and model first.")
    store = _history_store(request)

    gate = _run_gate(request)
    gate.reserve(identity.id)
    started = perf_counter()
    preparation = None
    try:
        await asyncio.to_thread(store.ensure_ready)
        preparation = asyncio.create_task(asyncio.to_thread(
            prepare_investigation_context, identity, payload,
        ))
        async with asyncio.timeout(PREPARATION_TIMEOUT_SECONDS):
            context = await asyncio.shield(preparation)

        # A logout/expiry during preparation must stop a new paid model run.
        current_identity = require_github_identity(request)
        if current_identity.id != identity.id:
            raise _http_error(401, "Connect your GitHub account again.")

        result = await run_investigation(context, settings)
        if (
            result.repository_full_name != context.repository.full_name
            or result.commit_sha != context.snapshot.commit_sha
        ):
            raise _http_error(502, "Investigation provenance could not be verified.")
        completed = InvestigationRunResponse(
            run_id=result.run_id,
            repository_full_name=result.repository_full_name,
            commit_sha=result.commit_sha,
            report=result.report,
            total_duration_ms=round((perf_counter() - started) * 1000),
            agent_duration_ms=result.duration_ms,
            model_calls=result.model_calls,
            tool_attempts=result.tool_attempts,
            repository_stats=result.repository_stats,
            events=list(result.events),
            history_saved=True,
        )
        try:
            await asyncio.to_thread(store.save, identity.id, context.request, completed)
        except InvestigationStoreError:
            # Preserve a completed paid report if disk access changed during AI work.
            return completed.model_copy(update={"history_saved": False})
        return completed
    except HTTPException:
        raise
    except RepositorySelectionError:
        raise _http_error(400, "Repository selection changed. Refresh the list and select again.") from None
    except GitHubClientError:
        raise _http_error(502, "Could not prepare repository evidence from GitHub. Try later.") from None
    except InvestigationStoreError:
        raise _http_error(503, "Investigation history is unavailable. Try later.") from None
    except TimeoutError:
        raise _http_error(504, "Repository preparation timed out. Try later.") from None
    except InvestigationRunError as error:
        status, detail = _RUN_ERRORS.get(
            error.code, (502, "The AI investigation could not be completed."),
        )
        retry_after = (
            getattr(error, "retry_after", None)
            if error.code in _PROVIDER_WAIT_ERRORS else None
        )
        raise _http_error(status, detail, retry_after) from None
    except Exception:
        raise _http_error(500, "The investigation could not be completed.") from None
    finally:
        if preparation is not None and not preparation.done():
            # Cancellation cannot kill sync GitHub I/O; retain the slot until done.
            gate.release_when_preparation_finishes(preparation)
        else:
            gate.release()
