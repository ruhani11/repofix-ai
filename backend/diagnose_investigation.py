"""Temporary local diagnostics. Run from backend: python -u diagnose_investigation.py.

Starting the server makes no AI request. Diagnostics print metadata only; they
do not save reports, prompts, source quotes, keys or repository file contents.
All production validation remains authoritative and failures are re-raised.
Supports OpenAI Responses and Gemini Chat Completions. Token counts are the
SDK-reported usage of each completed response, not a billing-cost estimate.
"""

import asyncio
import json
from time import perf_counter

from pydantic import ValidationError

from app import evidence_grounding as grounding
from app.investigation_report import InvestigationReport
from app.repository_tools import MAX_TOOL_CALLS


_REJECTION_CODES = {
    "Invalid grounding inputs.": "invalid_inputs",
    "Invalid report or observed source data.": "invalid_schema",
    "Repository context is inconsistent.": "context_mismatch",
    "Observed read does not match this repository snapshot.": "snapshot_mismatch",
    "Observed file is outside the read policy.": "read_policy",
    "A citation does not match complete source lines observed during this run.":
        "citation_mismatch",
}


def grounding_summary(report, observed_reads, error):
    """Explain match failures without revealing model/source text or paths."""
    summary = {
        "reason": _REJECTION_CODES.get(str(error), "grounding_rejected"),
        "observed_read_count": len(observed_reads) if isinstance(observed_reads, list) else None,
        "invalid_observation_indices": [],
        "citations": [],
    }
    valid_reads = []
    if isinstance(observed_reads, list):
        for index, data in enumerate(observed_reads[:MAX_TOOL_CALLS], start=1):
            try:
                valid_reads.append(grounding._ObservedRead.model_validate(data))
            except (ValidationError, TypeError, ValueError):
                summary["invalid_observation_indices"].append(index)

    try:
        checked = InvestigationReport.model_validate(report.model_dump())
    except (ValidationError, AttributeError, TypeError, ValueError):
        summary["report_schema_valid"] = False
        return summary

    summary["report_schema_valid"] = True
    for index, citation in enumerate(checked.evidence, start=1):
        same_file = [read for read in valid_reads if read.path == citation.path]
        windows = []
        quote_locations = []
        for read in same_file:
            if read.end_line is None:
                continue
            end = read.end_line - int(read.last_line_may_be_partial)
            windows.append([read.start_line, end])
            lines = read.content.split("\n")
            if read.last_line_may_be_partial:
                lines = lines[:-1]
            complete_text = "\n".join(lines)
            offset = complete_text.find(citation.quote)
            if offset >= 0:
                start = read.start_line + complete_text[:offset].count("\n")
                location = [start, start + citation.quote.count("\n")]
                if location not in quote_locations:
                    quote_locations.append(location)

        summary["citations"].append({
            "index": index,
            "cited_lines": [citation.start_line, citation.end_line],
            "path_was_read": bool(same_file),
            "complete_observed_windows": windows,
            "range_covered": any(
                citation.start_line >= start and citation.end_line <= end
                for start, end in windows
            ),
            "exact_match": any(grounding._matches(citation, read) for read in same_file),
            "quote_found_at_observed_lines": quote_locations,
        })
    return summary


def token_usage_metadata(result):
    """Return numeric SDK usage only; never dump the response or raw usage."""
    usage = getattr(result, "usage", None)
    fields = ("input_tokens", "output_tokens", "total_tokens")
    values = {field: getattr(usage, field, None) for field in fields}
    if any(type(value) is not int or value < 0 for value in values.values()):
        return None
    return values


def main():
    import uvicorn
    from agents import set_tracing_disabled
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    from agents.models.openai_responses import OpenAIResponsesModel
    from agents.tracing import set_trace_provider
    from agents.tracing.provider import DefaultTraceProvider

    # Local diagnostic process only; do not alter application source or sessions.
    set_trace_provider(DefaultTraceProvider())
    set_tracing_disabled(True)

    from app.config import get_settings
    from app import investigation_runner as runner

    settings = get_settings()
    if not settings.ai_configured:
        raise SystemExit("Configure the selected AI provider key and model first.")

    model_class = (
        OpenAIResponsesModel
        if settings.ai_provider == "openai"
        else OpenAIChatCompletionsModel
    )

    original_budget = runner._ReportBudgetModel.get_response
    original_response = model_class.get_response
    original_tool_start = runner._RunObserver.on_tool_start
    original_tool_end = runner._RunObserver.on_tool_end
    original_grounding = runner.validate_report_grounding
    request_number = 0

    async def diagnostic_budget(self, *args, **kwargs):
        turn = self.calls + 1
        phase = "REPORT ONLY" if turn == self.max_turns else "INVESTIGATION"
        print(f"Logical turn {turn}/{self.max_turns}: {phase}", flush=True)
        return await original_budget(self, *args, **kwargs)

    async def diagnostic_response(self, *args, **kwargs):
        nonlocal request_number
        request_number += 1
        current = request_number
        started = perf_counter()
        print(f"HTTP model request {current}: started", flush=True)
        try:
            result = await original_response(self, *args, **kwargs)
            print(
                f"HTTP model request {current}: completed "
                f"({perf_counter() - started:.2f}s)", flush=True,
            )
            usage = token_usage_metadata(result)
            print(
                f"Request {current} SDK token usage:",
                json.dumps(usage) if usage is not None else "unavailable",
                flush=True,
            )
            return result
        except asyncio.CancelledError:
            print(f"HTTP model request {current}: cancelled", flush=True)
            raise
        except Exception as error:
            print(
                f"HTTP model request {current}: failed; "
                f"class={type(error).__name__}; "
                f"status={getattr(error, 'status_code', None)}", flush=True,
            )
            raise

    async def diagnostic_tool_start(self, context, agent, tool):
        await original_tool_start(self, context, agent, tool)
        print(f"Tool {self.tool_attempts}: {tool.name} started", flush=True)

    async def diagnostic_tool_end(self, context, agent, tool, result):
        await original_tool_end(self, context, agent, tool, result)
        payload = json.loads(result)
        print(f"Tool completed: {tool.name}; success={payload['ok']}", flush=True)

    def diagnostic_grounding(report, context, observed_reads):
        try:
            result = original_grounding(report, context, observed_reads)
        except grounding.EvidenceGroundingError as error:
            try:
                summary = grounding_summary(report, observed_reads, error)
                print("GROUNDING DIAGNOSTIC:", json.dumps(summary), flush=True)
            except Exception:
                # Diagnostic failure must not hide or replace the original error.
                print("GROUNDING DIAGNOSTIC: metadata unavailable", flush=True)
            raise
        print("Grounding validation: PASSED", flush=True)
        return result

    runner._ReportBudgetModel.get_response = diagnostic_budget
    model_class.get_response = diagnostic_response
    runner._RunObserver.on_tool_start = diagnostic_tool_start
    runner._RunObserver.on_tool_end = diagnostic_tool_end
    runner.validate_report_grounding = diagnostic_grounding

    print("Provider:", settings.ai_provider, flush=True)
    print("Model:", settings.ai_model, flush=True)
    print("API format:", "Responses" if settings.ai_provider == "openai" else "Chat Completions", flush=True)
    print("Max turns:", settings.ai_max_turns, flush=True)
    print("Timeout seconds:", settings.ai_timeout_seconds, flush=True)
    print("Grounding diagnostics enabled; source/quotes/keys are not logged.", flush=True)
    print("Request numbers span this server process; token usage is per completed response.", flush=True)
    try:
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000, access_log=False)
    finally:
        runner._ReportBudgetModel.get_response = original_budget
        model_class.get_response = original_response
        runner._RunObserver.on_tool_start = original_tool_start
        runner._RunObserver.on_tool_end = original_tool_end
        runner.validate_report_grounding = original_grounding


if __name__ == "__main__":
    main()
