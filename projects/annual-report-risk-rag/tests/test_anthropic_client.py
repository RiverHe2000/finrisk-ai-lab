"""Exercise the SDK adapter with a fake client - no network, no credentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from anthropic.types import TextBlock, ToolUseBlock
from pydantic import BaseModel

from report_rag.config import Settings
from report_rag.llm.anthropic_client import AnthropicLLM, _to_message_params
from report_rag.llm.base import AssistantTurn, Conversation, ToolCall, ToolResult, ToolSpec


class Payload(BaseModel):
    answer: str


@dataclass
class _FakeParsed:
    parsed_output: Any
    stop_reason: str = "end_turn"


@dataclass
class _FakeCreated:
    content: list[Any]
    stop_reason: str | None = "tool_use"


@dataclass
class _FakeMessages:
    parse_result: Any = None
    create_result: Any = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append({"parse": kwargs})
        return self.parse_result

    def create(self, **kwargs: Any) -> Any:
        self.calls.append({"create": kwargs})
        return self.create_result


@dataclass
class _FakeClient:
    messages: _FakeMessages


def test_complete_structured_passes_model_and_effort() -> None:
    fake = _FakeClient(_FakeMessages(parse_result=_FakeParsed(Payload(answer="42"))))
    llm = AnthropicLLM(Settings(model="claude-opus-5", effort="low"), client=fake)  # type: ignore[arg-type]
    out = llm.complete_structured(system="sys", user="q", output_model=Payload)
    assert out.answer == "42"
    kwargs = fake.messages.calls[0]["parse"]
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["output_format"] is Payload
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["messages"] == [{"role": "user", "content": "q"}]


def test_complete_structured_raises_on_refusal_or_missing_output() -> None:
    fake = _FakeClient(_FakeMessages(parse_result=_FakeParsed(None, stop_reason="refusal")))
    llm = AnthropicLLM(Settings(), client=fake)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="refused"):
        llm.complete_structured(system="s", user="u", output_model=Payload)
    fake.messages.parse_result = _FakeParsed(None, stop_reason="max_tokens")
    with pytest.raises(RuntimeError, match="No structured output"):
        llm.complete_structured(system="s", user="u", output_model=Payload)


def test_next_turn_maps_tool_use_blocks() -> None:
    blocks = [
        TextBlock(type="text", text="Let me search."),
        ToolUseBlock(type="tool_use", id="tu_1", name="search_report", input={"query": "LCR"}),
    ]
    fake = _FakeClient(_FakeMessages(create_result=_FakeCreated(blocks)))
    llm = AnthropicLLM(Settings(), client=fake)  # type: ignore[arg-type]
    conv = Conversation(system="sys")
    conv.add_user("go")
    turn = llm.next_turn(conv, [ToolSpec("search_report", "d", {"type": "object"})])
    assert turn.text == "Let me search."
    assert turn.tool_calls == [ToolCall("tu_1", "search_report", {"query": "LCR"})]
    assert turn.stop_reason == "tool_use"
    sent = fake.messages.calls[0]["create"]
    assert sent["tools"][0]["name"] == "search_report"
    assert sent["system"] == "sys"


def test_transcript_translation_round_trips_all_roles() -> None:
    conv = Conversation(system="s")
    conv.add_user("hello")
    conv.add_assistant(AssistantTurn(text="hi", tool_calls=[ToolCall("1", "t", {"a": 1})]))
    conv.add_tool_results([ToolResult("1", "ok"), ToolResult("2", "bad", is_error=True)])
    conv.add_assistant(AssistantTurn(text="", tool_calls=[ToolCall("3", "t", {})]))
    msgs = _to_message_params(conv)
    assert msgs[0] == {"role": "user", "content": "hello"}
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0] == {"type": "text", "text": "hi"}  # type: ignore[index]
    assert msgs[1]["content"][1]["type"] == "tool_use"  # type: ignore[index]
    results = msgs[2]["content"]
    assert results[1]["is_error"] is True  # type: ignore[index]
    assert msgs[3]["content"] == [{"type": "tool_use", "id": "3", "name": "t", "input": {}}]
