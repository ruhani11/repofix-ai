"""Create a run-scoped Gemini provider through Chat Completions compatibility."""

import asyncio
import logging
import math
import random
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from agents import Model
from agents.models.openai_provider import OpenAIProvider
from openai import APIStatusError, AsyncOpenAI, DefaultAsyncHttpxClient

from app.config import Settings


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_RUN_RETRIES = 2
MAX_RETRY_DELAY_SECONDS = 60
_DURATION = re.compile(r"([0-9]+(?:\.[0-9]{1,9})?)s")
logger = logging.getLogger(__name__)


class GeminiProviderError(RuntimeError):
    """Configuration failure safe to handle without exposing credentials."""


def _retry_delay(error: APIStatusError, retry_number: int) -> float | None:
    """Retry brief provider failures, never infer a wait for exhausted quota."""
    if error.status_code not in {429, 503}:
        return None

    delays = []
    header = error.response.headers.get("retry-after")
    if header is not None:
        try:
            delays.append(float(header))
        except ValueError:
            # An unsupported date-form header must not become an early retry.
            return None

    bodies = error.body if isinstance(error.body, list) else [error.body]
    for body in bodies:
        if not isinstance(body, dict):
            continue
        detail = body.get("error", body)
        if not isinstance(detail, dict):
            continue
        details = detail.get("details", [])
        if not isinstance(details, list):
            continue
        for item in details:
            if not isinstance(item, dict):
                continue
            kind = item.get("@type")
            if kind == "type.googleapis.com/google.rpc.QuotaFailure":
                violations = item.get("violations", [])
                if isinstance(violations, list):
                    for violation in violations:
                        if isinstance(violation, dict) and "perday" in str(
                            violation.get("quotaId", "")
                        ).lower():
                            return None
            if kind == "type.googleapis.com/google.rpc.RetryInfo":
                match = _DURATION.fullmatch(str(item.get("retryDelay", "")))
                if match is None:
                    return None
                delays.append(float(match.group(1)))

    if any(not math.isfinite(delay) or delay < 0 for delay in delays):
        return None
    if delays:
        delay = max(delays)
    elif error.status_code == 503:
        delay = float(2 ** retry_number)
    else:
        # A 429 without a usable provider hint may be daily/billing exhaustion.
        return None
    if delay > MAX_RETRY_DELAY_SECONDS:
        return None
    return delay + random.uniform(0.05, 0.25)


@dataclass
class _RetryBudget:
    used: int = 0


class _RetryingGeminiModel(Model):
    """Retry only the failed non-streaming call; preserve prior tool results."""

    def __init__(self, wrapped: Model, budget: _RetryBudget):
        self._wrapped = wrapped
        self._budget = budget

    async def get_response(self, *args, **kwargs):
        while True:
            try:
                return await self._wrapped.get_response(*args, **kwargs)
            except APIStatusError as error:
                if self._budget.used >= MAX_RUN_RETRIES:
                    raise
                delay = _retry_delay(error, self._budget.used + 1)
                if delay is None:
                    raise
                self._budget.used += 1
                # Metadata only: never log credentials, prompts or error bodies.
                logger.warning(
                    "Gemini retry %d/%d: HTTP %d; waiting %.2fs",
                    self._budget.used, MAX_RUN_RETRIES, error.status_code, delay,
                )
                # The runner's overall deadline also cancels this wait.
                await asyncio.sleep(delay)

    async def stream_response(self, *args, **kwargs):
        # Streaming is unused by /run; never replay a partially emitted stream.
        async for event in self._wrapped.stream_response(*args, **kwargs):
            yield event


class _RetryingGeminiProvider(OpenAIProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._retry_budget = _RetryBudget()

    def get_model(self, model_name: str | None) -> Model:
        return _RetryingGeminiModel(
            super().get_model(model_name), self._retry_budget,
        )


@asynccontextmanager
async def open_gemini_provider(
    settings: Settings,
) -> AsyncIterator[OpenAIProvider]:
    """Own the provider and HTTP client for exactly one workflow.

    Entering this context does not make a model request. The caller must select
    an explicit Gemini model, disable external tracing, and enforce its overall
    workflow deadline. At most two retries are shared by all model calls in
    this context. SDK model_calls count logical calls; retries are separately
    logged HTTP attempts. There is no OpenAI fallback or global SDK mutation.
    """
    if not isinstance(settings, Settings):
        raise GeminiProviderError("Invalid Gemini provider settings.")

    key = settings.gemini_api_key
    if key is None:
        raise GeminiProviderError("Configure the Gemini API key first.")

    async with DefaultAsyncHttpxClient(
        trust_env=False,
        follow_redirects=False,
    ) as http_client:
        async with AsyncOpenAI(
            api_key=key.get_secret_value(),
            base_url=GEMINI_BASE_URL,
            max_retries=0,
            timeout=settings.ai_timeout_seconds,
            http_client=http_client,
        ) as client:
            provider = _RetryingGeminiProvider(
                openai_client=client,
                use_responses=False,
                use_responses_websocket=False,
            )
            try:
                yield provider
            finally:
                await provider.aclose()
