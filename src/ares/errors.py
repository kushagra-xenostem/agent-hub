"""Error taxonomy.

Every failure the runtime can observe is classified into one ErrorKind. Recovery policies and metrics
work on these kinds, never on exception strings.
"""
from __future__ import annotations
from typing import Any
from enum import StrEnum
from pydantic import BaseModel, Field, ValidationError


class ErrorKind(StrEnum):
    LLM_RATE_LIMITED = "llm_rate_limited"
    LLM_TIMEOUT = "llm_timeout"
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_CONTEXT_OVERFLOW = "llm_context_overflow"
    LLM_BAD_REQUEST = "llm_bad_request"
    LLM_AUTH = "llm_auth"
    LLM_NO_ACTION = "llm_no_action"
    TOOL_UNKNOWN = "tool_unknown"
    TOOL_ARGS_INVALID = "tool_args_invalid"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_ERROR = "tool_error"
    TOOL_OUTPUT_INVALID = "tool_output_invalid"
    PERMISSION_DENIED = "permission_denied"
    APPROVAL_DENIED = "approval_denied"
    FINAL_INVALID = "final_invalid"
    BUDGET_EXHAUSTED = "budget_exhausted"


TRANSIENT = frozenset({ErrorKind.LLM_RATE_LIMITED, ErrorKind.LLM_TIMEOUT, ErrorKind.LLM_UNAVAILABLE,
                       ErrorKind.TOOL_TIMEOUT, ErrorKind.TOOL_OUTPUT_INVALID})
FATAL = frozenset({ErrorKind.LLM_AUTH, ErrorKind.LLM_BAD_REQUEST})


class AgentError(Exception):
    def __init__(self, kind: ErrorKind, message: str, *, transient: bool | None = None,
                 retry_after_s: float | None = None, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.transient = kind in TRANSIENT if transient is None else transient
        self.retry_after_s = retry_after_s
        self.detail = detail or {}


class ErrorInfo(BaseModel):
    """An AgentError as recorded in agent state: where it happened and what it was about."""

    kind: ErrorKind
    message: str
    transient: bool
    origin: str
    retry_after_s: float | None = None
    tool: str | None = None
    call_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)

    @property
    def signature(self) -> str:
        return f"{self.kind}:{self.tool or '-'}"


def format_validation_error(e: ValidationError) -> str:
    parts = []
    for err in e.errors(include_url=False):
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)
