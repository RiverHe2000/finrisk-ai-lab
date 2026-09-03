"""Tool-using extraction agent.

Instead of the fixed retrieve-then-extract loop in :mod:`report_rag.pipeline`,
the agent lets the model drive retrieval: it can issue several searches with
its own query wording, inspect results, and submit metrics when confident.
Submissions pass through the same :class:`GroundingValidator`, so the agent
cannot bypass the evidence requirement.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from report_rag.catalogue import RISK_METRICS
from report_rag.config import Settings
from report_rag.extraction.grounding import GroundingValidator
from report_rag.llm.base import (
    AssistantTurn,
    Conversation,
    ToolCall,
    ToolCallingLLM,
    ToolResult,
    ToolSpec,
)
from report_rag.pipeline import RiskExtractionPipeline
from report_rag.retrieval.hybrid import Retriever
from report_rag.schemas import (
    Chunk,
    Document,
    ExtractedMetric,
    ExtractionReport,
    GroundingStatus,
    MetricSpec,
    ValidatedMetric,
)

log = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """\
You are a prudential-reporting analyst. Extract the requested risk metrics from a bank's
annual report using the tools provided.

Workflow:
- Call `search_report` with focused queries (metric names, regulatory terms, table headings).
  Reformulate and search again if the first results do not contain the number.
- When you have located a metric, call `submit_metric` with the value exactly as printed,
  the unit, the period, a VERBATIM sentence from one excerpt as `evidence_quote`, and that
  excerpt's `chunk_id`. Submit `found=false` if the report does not disclose the metric.
- Never guess. Every submitted value must be visible in the quoted sentence.
- Submit each metric at most once. Stop when every metric has been submitted.
"""

SEARCH_TOOL = ToolSpec(
    name="search_report",
    description=(
        "Hybrid keyword + semantic search over the annual report. Returns excerpts with ids."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)


def submit_tool(metric_ids: Sequence[str]) -> ToolSpec:
    """Build the submit tool with the metric id enum baked in."""
    schema = ExtractedMetric.model_json_schema()
    schema["properties"]["metric_id"]["enum"] = list(metric_ids)
    schema["additionalProperties"] = False
    schema.pop("title", None)
    return ToolSpec(
        name="submit_metric",
        description="Submit one extracted metric (or found=false) with verbatim evidence.",
        input_schema=schema,
    )


@dataclass(slots=True)
class AgentTrace:
    """Observability: what the agent searched and submitted."""

    searches: list[dict[str, Any]] = field(default_factory=list)
    submissions: list[ExtractedMetric] = field(default_factory=list)
    turns: int = 0
    stopped_because: str = ""


class ExtractionAgent:
    """Run a bounded tool-calling loop over one document.

    Args:
        llm: A :class:`ToolCallingLLM` (Anthropic or a scripted stub).
        settings: Retrieval settings (chunking, top_k).
        metrics: Metrics to extract.
        max_turns: Hard cap on model turns to bound cost.
    """

    name = "agent"

    def __init__(
        self,
        llm: ToolCallingLLM,
        settings: Settings | None = None,
        metrics: Sequence[MetricSpec] = RISK_METRICS,
        max_turns: int = 30,
    ) -> None:
        self._llm = llm
        self.settings = settings or Settings()
        self.metrics = tuple(metrics)
        self.max_turns = max_turns
        self.validator = GroundingValidator()

    def run(self, doc: Document) -> tuple[ExtractionReport, AgentTrace]:
        """Drive the loop to completion and return validated results plus a trace."""
        # Reuse the pipeline's indexing so agent and RAG modes share one retriever.
        indexer = RiskExtractionPipeline(extractor=_NoopExtractor(), settings=self.settings)
        _, retriever = indexer.build_retriever(doc)

        seen_chunks: dict[str, Chunk] = {}
        submissions: dict[str, ExtractedMetric] = {}
        trace = AgentTrace()
        tools = [SEARCH_TOOL, submit_tool([m.metric_id for m in self.metrics])]
        conversation = Conversation(system=AGENT_SYSTEM_PROMPT)
        conversation.add_user(self._task_message())

        for _ in range(self.max_turns):
            trace.turns += 1
            turn = self._llm.next_turn(conversation, tools)
            conversation.add_assistant(turn)
            if not turn.wants_tools:
                trace.stopped_because = f"model ended turn ({turn.stop_reason})"
                break
            results = [
                self._dispatch(call, retriever, seen_chunks, submissions, trace)
                for call in turn.tool_calls
            ]
            conversation.add_tool_results(results)
            if len(submissions) == len(self.metrics):
                trace.stopped_because = "all metrics submitted"
                break
        else:
            trace.stopped_because = f"max_turns={self.max_turns} reached"

        validated = [
            self._validate(spec, submissions.get(spec.metric_id), seen_chunks)
            for spec in self.metrics
        ]
        return ExtractionReport(doc_id=doc.doc_id, extractor="agent", results=validated), trace

    # ------------------------------------------------------------------ internals
    def _task_message(self) -> str:
        lines = ["Extract the following metrics:"]
        lines.extend(
            f"- {m.metric_id}: {m.name} ({m.unit.value}) - {m.description}" for m in self.metrics
        )
        return "\n".join(lines)

    def _dispatch(
        self,
        call: ToolCall,
        retriever: Retriever,
        seen_chunks: dict[str, Chunk],
        submissions: dict[str, ExtractedMetric],
        trace: AgentTrace,
    ) -> ToolResult:
        if call.name == SEARCH_TOOL.name:
            return self._do_search(call, retriever, seen_chunks, trace)
        if call.name == "submit_metric":
            return self._do_submit(call, submissions, trace)
        return ToolResult(call.call_id, f"Unknown tool '{call.name}'", is_error=True)

    def _do_search(
        self,
        call: ToolCall,
        retriever: Retriever,
        seen_chunks: dict[str, Chunk],
        trace: AgentTrace,
    ) -> ToolResult:
        query = str(call.arguments.get("query", "")).strip()
        if not query:
            return ToolResult(call.call_id, "query must be a non-empty string", is_error=True)
        top_k = int(call.arguments.get("top_k", self.settings.top_k))
        hits = retriever.retrieve(query, top_k=max(1, min(top_k, 10)))
        trace.searches.append({"query": query, "chunk_ids": [h.chunk.chunk_id for h in hits]})
        for h in hits:
            seen_chunks[h.chunk.chunk_id] = h.chunk
        payload = [
            {"chunk_id": h.chunk.chunk_id, "section": h.chunk.section, "text": h.chunk.text}
            for h in hits
        ]
        return ToolResult(call.call_id, json.dumps(payload, ensure_ascii=False))

    def _do_submit(
        self, call: ToolCall, submissions: dict[str, ExtractedMetric], trace: AgentTrace
    ) -> ToolResult:
        try:
            metric = ExtractedMetric.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResult(call.call_id, f"Invalid submission: {exc}", is_error=True)
        if metric.metric_id in submissions:
            return ToolResult(
                call.call_id, f"{metric.metric_id} already submitted; ignored.", is_error=True
            )
        submissions[metric.metric_id] = metric
        trace.submissions.append(metric)
        remaining = len(self.metrics) - len(submissions)
        return ToolResult(call.call_id, f"Recorded {metric.metric_id}. {remaining} remaining.")

    def _validate(
        self, spec: MetricSpec, metric: ExtractedMetric | None, seen: dict[str, Chunk]
    ) -> ValidatedMetric:
        if metric is None:
            return ValidatedMetric(
                metric=ExtractedMetric(metric_id=spec.metric_id, found=False),
                grounding=GroundingStatus.NOT_FOUND,
                retrieved_chunk_ids=list(seen),
                accepted=False,
            )
        return self.validator.validate(spec, metric, seen)


class _NoopExtractor:
    """Placeholder so the pipeline can be used purely for indexing."""

    name = "rules"

    def extract(self, spec: MetricSpec, chunks: Sequence[Chunk]) -> ExtractedMetric:
        return ExtractedMetric(metric_id=spec.metric_id, found=False)


__all__ = ["AGENT_SYSTEM_PROMPT", "AgentTrace", "AssistantTurn", "ExtractionAgent"]
