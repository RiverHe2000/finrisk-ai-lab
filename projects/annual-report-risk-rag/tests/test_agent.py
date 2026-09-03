from __future__ import annotations

import json

from report_rag.agent import SEARCH_TOOL, ExtractionAgent, submit_tool
from report_rag.catalogue import get_metric
from report_rag.config import Settings
from report_rag.llm.base import AssistantTurn, Conversation, ToolCall, ToolResult
from report_rag.llm.stub import ScriptedToolLLM
from report_rag.schemas import Document, GroundingStatus


def _last_tool_results(conversation: Conversation) -> list[ToolResult]:
    role, payload = conversation.turns[-1]
    assert role == "tool_results"
    return payload  # type: ignore[no-any-return]


def _submit_from_search(conversation: Conversation) -> AssistantTurn:
    """Craft a grounded submission from whatever the search tool returned."""
    results = _last_tool_results(conversation)
    excerpts = json.loads(results[0].content)
    for ex in excerpts:
        for sentence in ex["text"].split(". "):
            if "12.4%" in sentence:
                return AssistantTurn(
                    tool_calls=[
                        ToolCall(
                            "t2",
                            "submit_metric",
                            {
                                "metric_id": "cet1_ratio",
                                "found": True,
                                "value": 12.4,
                                "unit": "percent",
                                "period": "FY2025",
                                "evidence_quote": sentence,
                                "chunk_id": ex["chunk_id"],
                                "confidence": 0.9,
                            },
                        )
                    ]
                )
    msg = "expected 12.4% in search results"
    raise AssertionError(msg)


def test_agent_searches_then_submits_grounded_metric(
    southern_cross: Document, settings: Settings
) -> None:
    llm = ScriptedToolLLM(
        [
            AssistantTurn(
                text="Searching",
                tool_calls=[ToolCall("t1", "search_report", {"query": "CET1 ratio", "top_k": 3})],
            ),
            _submit_from_search,
        ]
    )
    agent = ExtractionAgent(llm, settings, metrics=[get_metric("cet1_ratio")])
    report, trace = agent.run(southern_cross)
    assert report.extractor == "agent"
    assert report.results[0].accepted
    assert report.results[0].metric.value == 12.4
    assert trace.searches[0]["query"] == "CET1 ratio"
    assert trace.stopped_because == "all metrics submitted"
    assert trace.turns == 2
    assert llm.seen_tools[0] == ["search_report", "submit_metric"]


def test_agent_handles_bad_tool_calls_and_duplicates(
    southern_cross: Document, settings: Settings
) -> None:
    good = {
        "metric_id": "cet1_ratio",
        "found": False,
        "confidence": 0.1,
    }
    llm = ScriptedToolLLM(
        [
            AssistantTurn(
                tool_calls=[
                    ToolCall("a", "search_report", {"query": "   "}),
                    ToolCall("b", "unknown_tool", {}),
                    ToolCall("c", "submit_metric", {"metric_id": "cet1_ratio", "found": "maybe"}),
                ]
            ),
            AssistantTurn(
                tool_calls=[
                    ToolCall("d", "submit_metric", good),
                    ToolCall("e", "submit_metric", good),
                ]
            ),
        ]
    )
    agent = ExtractionAgent(llm, settings, metrics=[get_metric("cet1_ratio")])
    report, trace = agent.run(southern_cross)
    assert report.results[0].grounding is GroundingStatus.NOT_FOUND
    assert len(trace.submissions) == 1
    assert trace.stopped_because == "all metrics submitted"


def test_agent_stops_when_model_ends_turn(southern_cross: Document, settings: Settings) -> None:
    llm = ScriptedToolLLM([AssistantTurn(text="I give up", stop_reason="end_turn")])
    agent = ExtractionAgent(llm, settings, metrics=[get_metric("lcr")])
    report, trace = agent.run(southern_cross)
    assert not report.results[0].accepted
    assert "model ended turn" in trace.stopped_because


def test_agent_respects_max_turns(southern_cross: Document, settings: Settings) -> None:
    forever = [
        AssistantTurn(tool_calls=[ToolCall(f"s{i}", "search_report", {"query": "LCR"})])
        for i in range(5)
    ]
    agent = ExtractionAgent(
        ScriptedToolLLM(forever), settings, metrics=[get_metric("lcr")], max_turns=2
    )
    _, trace = agent.run(southern_cross)
    assert trace.turns == 2
    assert trace.stopped_because == "max_turns=2 reached"


def test_submit_tool_schema_has_enum_and_no_extra_props() -> None:
    tool = submit_tool(["cet1_ratio", "lcr"])
    assert tool.input_schema["properties"]["metric_id"]["enum"] == ["cet1_ratio", "lcr"]
    assert tool.input_schema["additionalProperties"] is False
    assert SEARCH_TOOL.input_schema["required"] == ["query"]
