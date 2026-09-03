"""Provider-agnostic LLM interfaces.

Two narrow protocols keep the rest of the code base independent of the SDK:

* :class:`StructuredLLM` - one call, one validated Pydantic object (used by the RAG extractor).
* :class:`ToolCallingLLM` - one assistant turn that may request tool calls (used by the agent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredLLM(Protocol):
    """Return an instance of ``output_model`` for a system + user prompt."""

    def complete_structured(self, *, system: str, user: str, output_model: type[ModelT]) -> ModelT:
        """Run one structured-output completion."""
        ...


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the agent exposes to the model (JSON-schema based)."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool invocation requested by the model."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of executing a :class:`ToolCall`."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass(slots=True)
class AssistantTurn:
    """What the model produced in one turn."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"

    @property
    def wants_tools(self) -> bool:
        """True when the model asked for at least one tool to run."""
        return bool(self.tool_calls)


@dataclass(slots=True)
class Conversation:
    """Provider-neutral transcript the agent loop appends to."""

    system: str
    turns: list[tuple[str, Any]] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        """Append a user text message."""
        self.turns.append(("user", text))

    def add_assistant(self, turn: AssistantTurn) -> None:
        """Append the assistant's turn (text + tool calls)."""
        self.turns.append(("assistant", turn))

    def add_tool_results(self, results: list[ToolResult]) -> None:
        """Append tool results as a single user message (keeps parallel calls intact)."""
        self.turns.append(("tool_results", results))


class ToolCallingLLM(Protocol):
    """Produce the next assistant turn for a conversation and a tool set."""

    def next_turn(self, conversation: Conversation, tools: list[ToolSpec]) -> AssistantTurn:
        """Return the assistant's next turn."""
        ...
