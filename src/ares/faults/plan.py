"""Fault plans: which failures to inject, where, and how often.

Every injection decision is a pure function of (plan seed, fault index, target, call number), so the same
plan and seed inject the same faults at the same calls, independent of anything else in the run.
"""
from __future__ import annotations
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field


class FaultKind(StrEnum):
    TOOL_ERROR = "tool_error"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_MALFORMED = "tool_malformed"
    TOOL_EMPTY = "tool_empty"
    TOOL_CONFLICT = "tool_conflict"
    TOOL_OVERSIZED = "tool_oversized"
    LLM_RATE_LIMIT = "llm_rate_limit"
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_TIMEOUT = "llm_timeout"
    LLM_GARBLED = "llm_garbled"
    LLM_NO_TOOL = "llm_no_tool"
    LLM_CONTEXT_LIMIT = "llm_context_limit"


LLM_KINDS = frozenset({FaultKind.LLM_RATE_LIMIT, FaultKind.LLM_UNAVAILABLE, FaultKind.LLM_TIMEOUT,
                       FaultKind.LLM_GARBLED, FaultKind.LLM_NO_TOOL, FaultKind.LLM_CONTEXT_LIMIT})


class FaultSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: FaultKind
    target: str = "*"  # tool name, "*" for every tool; ignored for LLM faults
    rate: float = Field(1.0, ge=0.0, le=1.0)
    on_calls: list[int] | None = None  # inject exactly on these call numbers instead of by rate
    max_injections: int | None = None
    transient: bool = True
    after_effect: bool = False  # the side effect happened even though the call reported failure
    retry_after_s: float | None = 2.0
    limit_tokens: int = 8000
    oversize_chars: int = 150000

    @property
    def is_llm(self) -> bool:
        return self.kind in LLM_KINDS

    def matches(self, target: str) -> bool:
        if self.is_llm:
            return target == "llm"
        return self.target in ("*", target)


class FaultPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    faults: list[FaultSpec] = Field(default_factory=list)
    description: str = ""


def _plan(name: str, description: str, *faults: FaultSpec) -> FaultPlan:
    return FaultPlan(name=name, description=description, faults=list(faults))


READ_TOOLS = ("search_web", "fetch_page")

PRESETS: dict[str, FaultPlan] = {p.name: p for p in [
    _plan("none", "No injected faults."),
    _plan("tool_errors", "30% of search/fetch calls fail with a transient upstream error.",
          *(FaultSpec(kind=FaultKind.TOOL_ERROR, target=t, rate=0.3) for t in READ_TOOLS)),
    _plan("tool_errors_permanent", "30% of fetches fail with a permanent error (like a 404).",
          FaultSpec(kind=FaultKind.TOOL_ERROR, target="fetch_page", rate=0.3, transient=False)),
    _plan("tool_timeouts", "30% of search/fetch calls time out.",
          *(FaultSpec(kind=FaultKind.TOOL_TIMEOUT, target=t, rate=0.3) for t in READ_TOOLS)),
    _plan("malformed_outputs", "30% of fetches return output that violates the contract.",
          FaultSpec(kind=FaultKind.TOOL_MALFORMED, target="fetch_page", rate=0.3)),
    _plan("empty_outputs", "30% of searches return no results.",
          FaultSpec(kind=FaultKind.TOOL_EMPTY, target="search_web", rate=0.3)),
    _plan("conflicts", "50% of fetched pages carry altered numbers.",
          FaultSpec(kind=FaultKind.TOOL_CONFLICT, target="fetch_page", rate=0.5)),
    _plan("oversized_outputs", "Every fetch returns ~150k extra characters.",
          FaultSpec(kind=FaultKind.TOOL_OVERSIZED, target="fetch_page")),
    _plan("llm_rate_limits", "20% of model calls are rate limited (Retry-After 2s).",
          FaultSpec(kind=FaultKind.LLM_RATE_LIMIT, rate=0.2)),
    _plan("llm_unavailable", "20% of model calls get a 503.",
          FaultSpec(kind=FaultKind.LLM_UNAVAILABLE, rate=0.2)),
    _plan("llm_garbled", "20% of model responses have cut-off tool-call JSON.",
          FaultSpec(kind=FaultKind.LLM_GARBLED, rate=0.2)),
    _plan("llm_no_tool", "20% of model responses are prose instead of a tool call.",
          FaultSpec(kind=FaultKind.LLM_NO_TOOL, rate=0.2)),
    _plan("context_limit", "A 3,000-token context window.",
          FaultSpec(kind=FaultKind.LLM_CONTEXT_LIMIT, limit_tokens=3000)),
    _plan("mixed", "15% tool errors, 15% timeouts, 15% malformed fetches, 10% rate limits, 10% garbled.",
          *(FaultSpec(kind=FaultKind.TOOL_ERROR, target=t, rate=0.15) for t in READ_TOOLS),
          *(FaultSpec(kind=FaultKind.TOOL_TIMEOUT, target=t, rate=0.15) for t in READ_TOOLS),
          FaultSpec(kind=FaultKind.TOOL_MALFORMED, target="fetch_page", rate=0.15),
          FaultSpec(kind=FaultKind.LLM_RATE_LIMIT, rate=0.1),
          FaultSpec(kind=FaultKind.LLM_GARBLED, rate=0.1)),
]}
