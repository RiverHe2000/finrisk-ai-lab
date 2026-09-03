"""Anthropic SDK implementation of the LLM protocols."""

from __future__ import annotations

import json
from typing import Any, cast

import anthropic
from anthropic.types import MessageParam, TextBlock, ToolParam, ToolUseBlock

from report_rag.config import Settings
from report_rag.llm.base import (
    AssistantTurn,
    Conversation,
    ModelT,
    ToolCall,
    ToolResult,
    ToolSpec,
)


class AnthropicLLM:
    """Structured extraction and tool-calling on top of ``anthropic.Anthropic``.

    The client resolves credentials from ``ANTHROPIC_API_KEY`` (or an
    ``ant auth login`` profile), so nothing is hard-coded here.
    """

    def __init__(
        self, settings: Settings | None = None, client: anthropic.Anthropic | None = None
    ) -> None:
        self.settings = settings or Settings()
        self._client = client or anthropic.Anthropic()

    # ------------------------------------------------------------------ StructuredLLM
    def complete_structured(self, *, system: str, user: str, output_model: type[ModelT]) -> ModelT:
        """One structured-output call validated against ``output_model``."""
        response = self._client.messages.parse(
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=output_model,
            output_config={"effort": cast(Any, self.settings.effort)},
        )
        if response.stop_reason == "refusal":
            msg = "Model refused the extraction request"
            raise RuntimeError(msg)
        parsed = response.parsed_output
        if parsed is None:
            msg = f"No structured output returned (stop_reason={response.stop_reason})"
            raise RuntimeError(msg)
        return parsed

    # ------------------------------------------------------------------ ToolCallingLLM
    def next_turn(self, conversation: Conversation, tools: list[ToolSpec]) -> AssistantTurn:
        """Ask the model for its next turn given the transcript and tools."""
        response = self._client.messages.create(
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
            system=conversation.system,
            messages=_to_message_params(conversation),
            tools=[_to_tool_param(t) for t in tools],
            output_config={"effort": cast(Any, self.settings.effort)},
        )
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                raw = block.input
                arguments = raw if isinstance(raw, dict) else json.loads(json.dumps(raw))
                calls.append(ToolCall(call_id=block.id, name=block.name, arguments=arguments))
        return AssistantTurn(
            text="\n".join(text_parts),
            tool_calls=calls,
            stop_reason=response.stop_reason or "end_turn",
        )


def _to_tool_param(tool: ToolSpec) -> ToolParam:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _to_message_params(conversation: Conversation) -> list[MessageParam]:
    """Translate the neutral transcript into Anthropic message params."""
    messages: list[MessageParam] = []
    for role, payload in conversation.turns:
        if role == "user":
            messages.append({"role": "user", "content": str(payload)})
        elif role == "assistant":
            turn = cast(AssistantTurn, payload)
            content: list[Any] = []
            if turn.text:
                content.append({"type": "text", "text": turn.text})
            content.extend(
                {"type": "tool_use", "id": c.call_id, "name": c.name, "input": c.arguments}
                for c in turn.tool_calls
            )
            messages.append({"role": "assistant", "content": content})
        elif role == "tool_results":
            results = cast(list[ToolResult], payload)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r.call_id,
                            "content": r.content,
                            "is_error": r.is_error,
                        }
                        for r in results
                    ],
                }
            )
    return messages
