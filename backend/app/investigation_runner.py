"""Run one bounded, read-only investigation and validate its source citations."""

import asyncio
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass, replace
from email.utils import parsedate_to_datetime
from time import perf_counter
from time import time as unix_time
from typing import NoReturn
from uuid import uuid4

import httpx
from agents import (
    Model, ModelProvider, RunConfig, RunHooks, Runner, ToolExecutionConfig,
)
from agents.exceptions import AgentsException, MaxTurnsExceeded
from agents.models.openai_provider import OpenAIProvider
from openai import APIError, APITimeoutError, AsyncOpenAI, DefaultAsyncHttpxClient

from app.config import Settings
from app.evidence_grounding import (
    EvidenceGroundingError, validate_report_grounding,
)
from app.gemini_provider import GeminiProviderError, open_gemini_provider
from app.investigation_context import InvestigationContext
from app.investigation_report import InvestigationReport
from app.investigator import build_investigator_agent
from app.repository_tools import MAX_TOOL_CALLS, RepositoryTools


MAX_EVENTS = 128
MAX_TOOL_RESULT_CHARACTERS = 65536
MAX_RETRY_AFTER_SECONDS = 86400
_RETRY_DURATION = re.compile(r"([0-9]+(?:\.[0-9]{1,9})?)s")


class InvestigationRunError(RuntimeError):
    """Controlled failure; detail never contains provider or repository payloads."""

    def __init__(
        self, code: str, detail: str, *, retry_after: int | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.retry_after = (
            retry_after
            if type(retry_after) is int and 1 <= retry_after <= MAX_RETRY_AFTER_SECONDS
            else None
        )
        super().__init__(detail)


class _ReportBudgetModel(Model):
    """Reserve the final logical model turn for reporting, within one SDK run.

    The wrapped provider still owns its clients and HTTP retries. A Gemini retry
    stays inside the same logical turn. Earlier tool history is passed through;
    final reports still go through the normal schema and grounding validators.
    """

    def __init__(self, wrapped: Model, max_turns: int) -> None:
        self.wrapped = wrapped
        self.max_turns = max_turns
        self.calls = 0

    async def get_response(
        self, system_instructions, input, model_settings, tools,
        output_schema, handoffs, tracing, **kwargs,
    ):
        if self.calls >= self.max_turns:
            raise InvestigationRunError(
                "turn_limit", "Investigation reached its model turn limit."
            )

        self.calls += 1
        report_only = self.calls == self.max_turns
        budget_note = (
            f"Server-owned run budget: logical model turn {self.calls} "
            f"of {self.max_turns}. The final turn is reserved for reporting."
        )
        if report_only:
            budget_note += (
                " This is the final turn. No further tools are available."
                " Return the structured InvestigationReport now, using only"
                " source excerpts already observed in this run. If no probable"
                " cause is supported, return insufficient_evidence with"
                " root_cause null and explain the missing evidence. State any"
                " budget-related coverage limitation. Never invent citations,"
                " test results or a confirmed bug."
            )
            tools = []
            handoffs = []
            model_settings = replace(model_settings, tool_choice="none")

        response = await self.wrapped.get_response(
            system_instructions=f"{system_instructions or ''}\n\n{budget_note}",
            input=input,
            model_settings=model_settings,
            tools=tools,
            output_schema=output_schema,
            handoffs=handoffs,
            tracing=tracing,
            **kwargs,
        )
        if report_only and any(
            getattr(item, "type", None) not in {"message", "reasoning"}
            for item in response.output
        ):
            # A provider ignoring tool_choice must not execute another tool.
            raise InvestigationRunError(
                "turn_limit", "Investigation reached its model turn limit."
            )
        return response

    async def stream_response(self, *args, **kwargs):
        # /run is non-streaming; do not expose an unbounded alternative path.
        raise InvestigationRunError(
            "run_error", "Streaming investigation is not supported."
        )
        yield  # Keep the SDK's async-iterator interface.


def _classify_provider_error(error: APIError) -> InvestigationRunError:
    """Extract typed metadata only; never copy provider messages into errors."""
    if isinstance(error, APITimeoutError):
        return InvestigationRunError(
            "provider_timeout", "The AI provider request timed out. Try again later.",
        )

    status = getattr(error, "status_code", None)
    if status not in {429, 503}:
        return InvestigationRunError(
            "model_error", "The AI provider could not complete this investigation.",
        )

    hints: list[float] = []
    response = getattr(error, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header is not None:
        try:
            hints.append(float(header))
        except (ValueError, OverflowError):
            try:
                parsed = parsedate_to_datetime(header)
                if parsed.tzinfo is not None:
                    hints.append(max(0.0, parsed.timestamp() - unix_time()))
            except (ValueError, TypeError, OverflowError, OSError):
                pass

    quota_ids: list[str] = []
    bodies = error.body if isinstance(error.body, list) else [error.body]
    for body in bodies:
        if not isinstance(body, dict):
            continue
        metadata = body.get("error", body)
        if not isinstance(metadata, dict):
            continue
        details = metadata.get("details", [])
        if not isinstance(details, list):
            continue
        for item in details:
            if not isinstance(item, dict):
                continue
            kind = item.get("@type")
            if kind == "type.googleapis.com/google.rpc.RetryInfo":
                match = _RETRY_DURATION.fullmatch(str(item.get("retryDelay", "")))
                if match is not None:
                    hints.append(float(match.group(1)))
            if kind == "type.googleapis.com/google.rpc.QuotaFailure":
                violations = item.get("violations", [])
                if isinstance(violations, list):
                    for violation in violations:
                        if isinstance(violation, dict):
                            quota_id = violation.get("quotaId")
                            if isinstance(quota_id, str):
                                quota_ids.append(quota_id.lower())

    retry_after = None
    if hints and all(
        math.isfinite(hint) and 0 <= hint <= MAX_RETRY_AFTER_SECONDS
        for hint in hints
    ):
        retry_after = max(1, math.ceil(max(hints)))

    if status == 503:
        code = "provider_unavailable"
        detail = "The AI provider is temporarily unavailable. Try again later."
    elif any("perday" in quota_id for quota_id in quota_ids):
        code = "provider_daily_quota"
        detail = "The AI provider's daily quota is exhausted. Check usage and retry after reset."
    elif quota_ids and all("perminute" in quota_id for quota_id in quota_ids):
        code = "provider_rate_limit"
        detail = "The AI provider is temporarily rate limited. Try again later."
    else:
        # A 429 alone does not establish whether daily/billing quota is exhausted.
        code = "provider_limit"
        detail = "The AI provider's quota or rate limit was reached. Check usage before retrying."

    return InvestigationRunError(code, detail, retry_after=retry_after)


@dataclass(frozen=True)
class InvestigationEvent:
    """Local metadata only; excludes prompts, arguments, source and reasoning."""

    kind: str
    elapsed_ms: int
    tool_name: str | None = None
    succeeded: bool | None = None


@dataclass(frozen=True)
class InvestigationRunResult:
    run_id: str
    repository_full_name: str
    commit_sha: str
    report: InvestigationReport
    duration_ms: int
    model_calls: int
    tool_attempts: int
    repository_stats: dict[str, int]
    events: tuple[InvestigationEvent, ...]


class _RunObserver(RunHooks):
    def __init__(self, started: float) -> None:
        self.started = started
        self.model_calls = 0
        self.tool_attempts = 0
        self.reads: list[dict] = []
        self.events: list[InvestigationEvent] = []
        self.failure: InvestigationRunError | None = None

    def fail(self, code: str, detail: str) -> NoReturn:
        # The SDK can wrap hook exceptions; retain our controlled reason.
        self.failure = InvestigationRunError(code, detail)
        raise self.failure

    def record(
        self,
        kind: str,
        tool_name: str | None = None,
        succeeded: bool | None = None,
    ) -> None:
        if len(self.events) >= MAX_EVENTS:
            self.fail(
                "event_limit", "Investigation event limit reached."
            )
        self.events.append(InvestigationEvent(
            kind=kind,
            elapsed_ms=round((perf_counter() - self.started) * 1000),
            tool_name=tool_name,
            succeeded=succeeded,
        ))

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        self.model_calls += 1
        self.record("model_started")

    async def on_llm_end(self, context, agent, response):
        self.record("model_completed")

    async def on_tool_start(self, context, agent, tool):
        if tool.name not in {"find_files", "read_file", "search_code"}:
            self.fail(
                "tool_denied", "The investigation requested an unavailable tool."
            )
        if self.tool_attempts >= MAX_TOOL_CALLS:
            self.fail(
                "tool_limit", "Investigation tool attempt limit reached."
            )
        self.tool_attempts += 1
        self.record("tool_started", tool.name)

    async def on_tool_end(self, context, agent, tool, result):
        if not isinstance(result, str) or len(result) > MAX_TOOL_RESULT_CHARACTERS:
            self.fail(
                "invalid_tool_result", "Invalid repository tool response."
            )
        try:
            payload = json.loads(result)
        except (ValueError, RecursionError):
            self.fail(
                "invalid_tool_result", "Invalid repository tool response."
            )
        if not isinstance(payload, dict) or type(payload.get("ok")) is not bool:
            self.fail(
                "invalid_tool_result", "Invalid repository tool response."
            )
        succeeded = payload["ok"]
        if succeeded:
            if (
                payload.get("source_kind") != "untrusted_repository_data"
                or not isinstance(payload.get("data"), dict)
            ):
                self.fail(
                    "invalid_tool_result", "Invalid repository tool response."
                )
            if tool.name == "read_file":
                # Only read excerpts can ground citations. Search previews are
                # discovery hints, even when they contain exact source text.
                self.reads.append(deepcopy(payload["data"]))
        self.record("tool_completed", tool.name, succeeded)


async def run_investigation(
    context: InvestigationContext,
    settings: Settings,
    *,
    model_provider: ModelProvider | None = None,
    repository_transport: httpx.BaseTransport | None = None,
) -> InvestigationRunResult:
    """Use server-prepared context; authenticate and prepare it in the API first.

    Provider/transport overrides are internal offline-test seams, never HTTP
    input. No browser-supplied observations, URLs or commit overrides are used.
    A supplied provider remains owned by its caller.

    Timeout cancels the awaited workflow. It cannot terminate a synchronous
    GitHub read already running in an SDK worker thread.
    """
    if not isinstance(settings, Settings):
        raise InvestigationRunError("invalid_settings", "Invalid AI settings.")
    settings = settings.model_copy(deep=True)
    # Older settings have no provider field; preserve their OpenAI behavior.
    provider_name = getattr(settings, "ai_provider", "openai")
    if provider_name not in {"openai", "gemini"}:
        raise InvestigationRunError("invalid_settings", "Invalid AI provider.")
    if not settings.ai_configured:
        raise InvestigationRunError(
            "not_configured", "Configure the AI API key and model first."
        )
    if not isinstance(context, InvestigationContext):
        raise InvestigationRunError(
            "invalid_context", "Prepare the repository investigation first."
        )
    if model_provider is not None and not isinstance(model_provider, ModelProvider):
        raise InvestigationRunError("invalid_provider", "Invalid model provider.")

    if model_provider is None:
        is_gemini_model = (settings.ai_model or "").startswith("gemini-")
        if is_gemini_model != (provider_name == "gemini"):
            raise InvestigationRunError(
                "invalid_settings", "The selected AI model does not match its provider."
            )

    started = perf_counter()
    run_id = uuid4().hex
    observer = _RunObserver(started)

    async def execute(provider: ModelProvider) -> InvestigationRunResult:
        # Freeze our copy before the first await; tools have an isolated cache.
        prepared = deepcopy(context)
        reader = RepositoryTools(prepared, transport=repository_transport)
        agent = build_investigator_agent(
            reader, model=settings.ai_model, include_code_search=True,
        )
        agent.model = _ReportBudgetModel(
            provider.get_model(settings.ai_model), settings.ai_max_turns,
        )
        user_input = json.dumps({
            "repository_full_name": prepared.repository.full_name,
            "commit_sha": prepared.snapshot.commit_sha,
            "issue_description": prepared.request.issue_description,
        }, ensure_ascii=False)
        observer.record("run_started")
        result = await Runner.run(
            agent,
            user_input,
            max_turns=settings.ai_max_turns,
            hooks=observer,
            run_config=RunConfig(
                model_provider=provider,
                tracing_disabled=True,
                trace_include_sensitive_data=False,
                tool_execution=ToolExecutionConfig(
                    max_function_tool_concurrency=1,
                ),
            ),
        )
        if observer.failure is not None:
            raise observer.failure
        if not isinstance(result.final_output, InvestigationReport):
            raise InvestigationRunError(
                "invalid_report", "The model did not return a valid investigation report."
            )
        report = validate_report_grounding(
            result.final_output, prepared, observer.reads,
        )
        observer.record("grounding_validated")
        stats = reader.stats
        observer.record("run_completed")
        return InvestigationRunResult(
            run_id=run_id,
            repository_full_name=prepared.repository.full_name,
            commit_sha=prepared.snapshot.commit_sha,
            report=report,
            duration_ms=round((perf_counter() - started) * 1000),
            model_calls=observer.model_calls,
            tool_attempts=observer.tool_attempts,
            repository_stats=stats,
            events=tuple(observer.events),
        )

    try:
        async with asyncio.timeout(settings.ai_timeout_seconds):
            if model_provider is not None:
                return await execute(model_provider)

            if provider_name == "gemini":
                async with open_gemini_provider(settings) as provider:
                    return await execute(provider)

            # Use the configured secret explicitly; do not rely on SDK globals.
            # The default SDK HTTP client matches the installed OpenAI version.
            async with DefaultAsyncHttpxClient(trust_env=False) as http_client:
                async with AsyncOpenAI(
                    api_key=settings.openai_api_key.get_secret_value(),
                    base_url="https://api.openai.com/v1",
                    max_retries=0,
                    timeout=settings.ai_timeout_seconds,
                    http_client=http_client,
                ) as client:
                    provider = OpenAIProvider(
                        openai_client=client,
                        use_responses=True,
                        use_responses_websocket=False,
                    )
                    try:
                        return await execute(provider)
                    finally:
                        await provider.aclose()
    except InvestigationRunError:
        raise
    except GeminiProviderError:
        raise InvestigationRunError(
            "not_configured", "Configure the Gemini API key and model first."
        ) from None
    except TimeoutError:
        raise InvestigationRunError(
            "timeout", "Investigation timed out. Try a more focused issue description."
        ) from None
    except MaxTurnsExceeded:
        raise InvestigationRunError(
            "turn_limit", "Investigation reached its model turn limit."
        ) from None
    except EvidenceGroundingError:
        raise InvestigationRunError(
            "ungrounded_report", "The report cites source evidence that could not be verified."
        ) from None
    except APIError as error:
        raise _classify_provider_error(error) from None
    except AgentsException:
        if observer.failure is not None:
            raise observer.failure from None
        raise InvestigationRunError(
            "agent_error", "The agent could not complete a valid investigation."
        ) from None
    except Exception:
        # Do not expose SDK exception bodies, credentials or repository text.
        # CancelledError is a BaseException and deliberately propagates.
        raise InvestigationRunError(
            "run_error", "The investigation could not be completed."
        ) from None
