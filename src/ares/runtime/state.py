"""Explicit, serializable agent state and the finite-state machine's transition table."""
from __future__ import annotations
from typing import Any
from enum import StrEnum
from pydantic import BaseModel, Field
from ares.errors import ErrorInfo, ErrorKind


class Phase(StrEnum):
    DECIDE = "decide"  # ask the model for the next action
    VALIDATE = "validate"  # check the next proposed tool call against its contract
    AUTHORIZE = "authorize"  # least-privilege check
    AWAIT_APPROVAL = "await_approval"  # human approval gate for risky tools
    EXECUTE = "execute"  # run the tool under a timeout
    OBSERVE = "observe"  # give the tool result back to the model
    FINALIZE = "finalize"  # validate and ground the final report
    RECOVER = "recover"  # apply the recovery policy to the last error
    DONE = "done"
    FAILED = "failed"
    ESCALATED = "escalated"  # stopped and handed to a human


TERMINAL = frozenset({Phase.DONE, Phase.FAILED, Phase.ESCALATED})

_EDGES: dict[Phase, set[Phase]] = {
    Phase.DECIDE: {Phase.VALIDATE, Phase.RECOVER},
    Phase.VALIDATE: {Phase.AUTHORIZE, Phase.FINALIZE, Phase.RECOVER},
    Phase.AUTHORIZE: {Phase.EXECUTE, Phase.AWAIT_APPROVAL, Phase.RECOVER},
    Phase.AWAIT_APPROVAL: {Phase.EXECUTE, Phase.RECOVER, Phase.ESCALATED},
    Phase.EXECUTE: {Phase.OBSERVE, Phase.RECOVER},
    Phase.OBSERVE: {Phase.VALIDATE, Phase.DECIDE},
    Phase.FINALIZE: {Phase.DONE, Phase.RECOVER},
    Phase.RECOVER: {Phase.DECIDE, Phase.VALIDATE, Phase.EXECUTE, Phase.ESCALATED},
}
# Any non-terminal phase may fail (budget exhausted, unrecoverable error).
TRANSITIONS: dict[Phase, frozenset[Phase]] = {src: frozenset(dst | {Phase.FAILED}) for src, dst in _EDGES.items()}


class IllegalTransition(RuntimeError):
    pass


class PendingCall(BaseModel):
    """A tool call proposed by the model, waiting to be validated and executed."""

    id: str
    name: str
    arguments: str  # raw JSON text from the model
    args: dict[str, Any] | None = None  # parsed, once VALIDATE has run
    invalid: bool = False  # failed validation at least once (counted once for argument validity)


class ToolStep(BaseModel):
    """One proposed tool call and what became of it; the trajectory used for tool-selection metrics."""

    name: str
    call_id: str
    outcome: str  # "ok", "deduplicated", "submitted", or an ErrorKind value


class Counters(BaseModel):
    transitions: int = 0
    llm_calls: int = 0
    proposed_calls: int = 0  # tool calls the model proposed
    invalid_calls: int = 0  # of those, rejected by the input contract or naming an unknown tool
    tool_executions: int = 0
    tool_errors: int = 0
    deduplicated: int = 0
    retries: int = 0
    repairs: int = 0
    compactions: int = 0
    repeated_calls: int = 0  # exact repeats of an already-successful read-only call (wasted steps)
    perturbations: int = 0  # repairs sampled at repair_temperature after an identical repeated failure
    recoveries: int = 0  # error episodes followed by successful progress
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    llm_latency_s: float = 0.0
    tool_latency_s: float = 0.0
    backoff_s: float = 0.0
    injected_faults: int = 0
    injections_detected: int = 0  # untrusted outputs in which the heuristic found injected instructions
    blocked_actions: int = 0  # egress / DLP blocks
    errors: dict[str, int] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def bump_error(self, kind: ErrorKind) -> None:
        self.errors[kind.value] = self.errors.get(kind.value, 0) + 1


class AgentState(BaseModel):
    """Everything the run knows. Serializable at every transition, so a run can be inspected or resumed."""

    run_id: str
    task_id: str
    phase: Phase = Phase.DECIDE
    messages: list[dict[str, Any]] = Field(default_factory=list)
    pending: list[PendingCall] = Field(default_factory=list)
    current: PendingCall | None = None
    observation: Any = None
    answered: list[str] = Field(default_factory=list)  # tool_call ids that already have a tool message
    last_error: ErrorInfo | None = None
    error_streak: dict[str, int] = Field(default_factory=dict)  # error signature -> attempts since progress
    failed_actions: dict[str, int] = Field(default_factory=dict)  # hash of a failed model action -> count
    perturb_next: bool = False
    completed_actions: list[str] = Field(default_factory=list)  # hashes of successful read-only calls
    wrapup_sent: bool = False
    last_prompt_tokens: int = 0  # server-counted prompt size of the last successful call
    chars_at_last_call: int = 0  # transcript size (chars) when that call was made
    overflow_at: int | None = None  # smallest estimated prompt the server rejected as too long
    compaction_level: int = 1  # rises with each overflow; never falls within a run
    fetched_urls: list[str] = Field(default_factory=list)
    source_texts: dict[str, str] = Field(default_factory=dict)  # url -> what the model was shown (post-screening)
    shown_texts: list[str] = Field(default_factory=list)  # every external tool result as shown to the model
    known_urls: list[str] = Field(default_factory=list)  # URLs that external tools returned (fetch provenance)
    trajectory: list[ToolStep] = Field(default_factory=list)
    final: dict[str, Any] | None = None
    reason: str | None = None
    counters: Counters = Field(default_factory=Counters)
