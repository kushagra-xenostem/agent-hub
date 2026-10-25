"""Trace schema (versioned).

A run is one agent episode. Each run has an ordered list of events; together they are the audit trail
and contain everything needed to replay the run: every LLM request fingerprint and response, every tool
call with arguments and output, every validation, permission and approval decision, every error and the
recovery action taken for it, and every state transition.
"""
from __future__ import annotations
from typing import Any
from enum import StrEnum
from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class EventKind(StrEnum):
    RUN_START = "run_start"
    TRANSITION = "transition"  # src -> dst phase
    LLM_CALL = "llm_call"  # request fingerprint, new messages, response, usage, latency
    VALIDATION = "validation"  # a tool call passed or failed its input contract
    PERMISSION = "permission"  # least-privilege decision
    APPROVAL = "approval"  # human approval gate decision
    TOOL_CALL = "tool_call"  # execution: args, output or error, duration, deduplicated
    ERROR = "error"  # classified ErrorInfo
    RECOVERY = "recovery"  # action chosen by the recovery policy
    COMPACTION = "compaction"  # context was shrunk to fit
    SECURITY = "security"  # injection detected/redacted, egress or DLP block
    FAULT = "fault"  # a failure injected by the fault harness (target, kind, call number)
    WRAPUP = "wrapup"  # the model was told to finish because the step budget is nearly spent
    RUN_END = "run_end"


class TraceEvent(BaseModel):
    run_id: str
    seq: int
    ts: float  # unix time
    kind: EventKind
    phase: str  # FSM phase when the event happened
    duration_ms: float | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    run_id: str
    task_id: str
    variant: str
    model: str
    started_at: float
    ended_at: float | None = None
    status: str = "running"
    reason: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    counters: dict[str, Any] = Field(default_factory=dict)
    final: dict[str, Any] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
