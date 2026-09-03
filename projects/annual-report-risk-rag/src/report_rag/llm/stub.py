"""Deterministic LLM doubles for tests and offline demos."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel

from report_rag.llm.base import AssistantTurn, Conversation, ModelT, ToolSpec


class StubStructuredLLM:
    """Returns pre-baked payloads, in order, validated against the requested model.

    Also records every prompt so tests can assert on prompt construction.
    """

    def __init__(self, payloads: Iterable[dict[str, Any]]) -> None:
        self._payloads = deque(payloads)
        self.calls: list[dict[str, str]] = []

    def complete_structured(self, *, system: str, user: str, output_model: type[ModelT]) -> ModelT:
        """Pop the next payload and validate it."""
        self.calls.append({"system": system, "user": user})
        if not self._payloads:
            msg = "StubStructuredLLM ran out of payloads"
            raise RuntimeError(msg)
        return output_model.model_validate(self._payloads.popleft())


class ScriptedToolLLM:
    """Replays a script of assistant turns; optionally reacts to the last tool result.

    ``script`` is a list of :class:`AssistantTurn` or callables that receive the
    conversation and return a turn, which lets a test craft a submission from
    what ``search_report`` actually returned.
    """

    def __init__(
        self, script: Iterable[AssistantTurn | Callable[[Conversation], AssistantTurn]]
    ) -> None:
        self._script = deque(script)
        self.seen_tools: list[list[str]] = []

    def next_turn(self, conversation: Conversation, tools: list[ToolSpec]) -> AssistantTurn:
        """Return the next scripted turn."""
        self.seen_tools.append([t.name for t in tools])
        if not self._script:
            return AssistantTurn(text="done", stop_reason="end_turn")
        step = self._script.popleft()
        return step(conversation) if callable(step) else step


def payload_from(model: BaseModel) -> dict[str, Any]:
    """Convenience: turn a Pydantic instance into a stub payload."""
    return model.model_dump(mode="json")
