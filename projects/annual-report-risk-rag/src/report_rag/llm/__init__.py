"""LLM client abstractions: a structured-output protocol, the Anthropic backend, and test stubs."""

from report_rag.llm.base import AssistantTurn, StructuredLLM, ToolCall, ToolCallingLLM, ToolSpec
from report_rag.llm.stub import ScriptedToolLLM, StubStructuredLLM

__all__ = [
    "AssistantTurn",
    "ScriptedToolLLM",
    "StructuredLLM",
    "StubStructuredLLM",
    "ToolCall",
    "ToolCallingLLM",
    "ToolSpec",
]
