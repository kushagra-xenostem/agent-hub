"""Run configuration (the experiment knobs) and the task definition."""
from __future__ import annotations

from enum import StrEnum

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ares.tools.contracts import Scope


class Strategy(StrEnum):
    FAIL_FAST = "fail_fast"  # stop at the first error
    NAIVE_RETRY = "naive_retry"  # repeat the failed step immediately, whatever the error
    BACKOFF_RETRY = "backoff_retry"  # retry transient errors with exponential backoff; stop on the rest
    REPAIR = "repair"  # backoff for transient errors; feed other errors back to the model to fix


class AgentConfig(BaseModel):
    """Every reliability mechanism is a switch, so experiments can ablate them one at a time."""

    model_config = ConfigDict(extra="forbid")

    variant: str = "ares"  # label used in traces and reports

    # sampling
    temperature: float = 0.0
    seed: int | None = 0
    max_tokens: int = 2048  # a full report is ~1k tokens; 1024 truncated it in the first live run

    # bounds
    max_llm_calls: int = 15
    max_tool_executions: int = 25
    max_transitions: int = 300
    max_total_tokens: int = 300_000
    max_wall_s: float = 1200.0
    max_retries: int = 3  # per error signature, reset by progress
    max_observation_chars: int | None = 4000  # None: pass tool output through uncapped

    # context management (the project #1 server runs a 65,536-token window)
    context_window: int = 65_536
    compact_at: float | None = 0.75  # compact before a call whose estimated size exceeds this fraction
    keep_recent_observations: int = 2  # newest tool results that compaction never shrinks
    chars_per_token: float = 3.5  # only for text added since the server's last exact count

    # step discipline
    loop_guard: bool = True  # flag exact repeats of read-only calls in the observation
    wrapup_at_remaining: int | None = 4  # with this many LLM calls left, tell the model to submit (leaves room to repair the report)

    # recovery
    recovery: Strategy = Strategy.REPAIR
    backoff_base_s: float = 1.0
    backoff_max_s: float = 30.0
    # If the model repeats an identical failed action, the next call samples at this temperature (seeded,
    # so still reproducible). Greedy decoding otherwise reproduces the same mistake forever. None: off.
    repair_temperature: float | None = 0.7

    # validation
    validate_args: bool = True
    validate_output: bool = True
    check_grounding: bool = True  # report may only cite pages fetched in this run
    check_claims: bool = True  # figures in findings/summary must appear in what the agent read (or follow from it)
    url_provenance: bool = True  # fetch only URLs that a tool returned in this session (no invented URLs)

    # security
    enforce_permissions: bool = True
    hide_unpermitted_tools: bool = True  # least privilege: don't even offer tools outside the task's scopes
    require_approval: bool = True  # human approval gate for tools marked requires_approval
    egress_allowlist: bool = True  # outbound tools may only address the task's allowed_recipients
    dlp: bool = True  # block outbound calls and reports that contain secrets or the system canary
    mark_untrusted: bool = True  # wrap external content and tell the model it is data, not instructions
    injection_filter: Literal["off", "flag", "redact"] = "redact"  # heuristic scan of untrusted output


DEFAULT_SCOPES = frozenset({Scope.WEB_READ, Scope.NOTES_READ, Scope.NOTES_WRITE})


class Task(BaseModel):
    id: str
    prompt: str
    scopes: frozenset[Scope] = Field(default=DEFAULT_SCOPES)
    allowed_recipients: frozenset[str] = frozenset()  # domains outbound tools may address
    secrets: list[str] = Field(default_factory=list)  # confidential strings that must never leave the system
