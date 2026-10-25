"""A deterministic stand-in for the model: returns pre-written responses in order.

Used by the test suite and by failure-injection experiments that must not depend on the live API.
"""

from __future__ import annotations
import json
import itertools
from typing import Any
from ares.errors import AgentError
from collections.abc import Callable, Iterable
from ares.llm.base import LLMRequest, LLMResponse, ToolCall, Usage

Step = LLMResponse | AgentError | Callable[[LLMRequest], "LLMResponse | AgentError"]

_ids = itertools.count(1)


class ScriptedLLM:
    model = "scripted"

    def __init__(self, steps: Iterable[Step], *, repeat_last: bool = False):
        self.steps = list(steps)
        self.repeat_last = repeat_last
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.steps:
            raise AssertionError(f"ScriptedLLM ran out of steps after {len(self.requests) - 1} calls")
        step = self.steps[0] if (self.repeat_last and len(self.steps) == 1) else self.steps.pop(0)
        result = step(request) if callable(step) else step
        if isinstance(result, AgentError):
            raise result
        result = result.model_copy(deep=True)
        # report a prompt size proportional to the request, like a real server, so token-based logic
        # (budgets, proactive compaction) behaves realistically offline
        size = len(json.dumps(request.messages)) + len(json.dumps(request.tools))
        result.usage.prompt_tokens = int(size / 3.5)
        return result


def call(name: str, /, **args: Any) -> LLMResponse:
    """A response containing one tool call with the given arguments."""
    return raw_call(name, json.dumps(args))


def raw_call(name: str, arguments: str) -> LLMResponse:
    """A tool call whose argument text is used verbatim (for malformed-JSON cases)."""
    return LLMResponse(
        tool_calls=[ToolCall(id=f"call_{next(_ids)}", name=name, arguments=arguments)],
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=100, completion_tokens=20),
    )


def calls(*pairs: tuple[str, dict[str, Any]]) -> LLMResponse:
    """A response with several parallel tool calls."""
    return LLMResponse(
        tool_calls=[ToolCall(id=f"call_{next(_ids)}", name=n, arguments=json.dumps(a)) for n, a in pairs],
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=100, completion_tokens=20 * len(pairs)),
    )


def reply(text: str) -> LLMResponse:
    """A plain-text response with no tool call."""
    return LLMResponse(content=text, finish_reason="stop", usage=Usage(prompt_tokens=100, completion_tokens=20))
