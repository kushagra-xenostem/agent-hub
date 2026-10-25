"""Recovery policies: given a classified error and how often it has happened, choose what to do.

Bounded by design: every strategy gives up after max_retries attempts at the same error signature.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel
from ares.errors import FATAL, ErrorInfo, ErrorKind
from ares.runtime.config import AgentConfig, Strategy

ActionKind = Literal["retry", "repair", "compact", "fail"]


class RecoveryAction(BaseModel):
    kind: ActionKind
    delay_s: float = 0.0
    reason: str


def backoff_delay(cfg: AgentConfig, err: ErrorInfo, attempt: int) -> float:
    # Deterministic (no jitter) so runs are reproducible; a server's Retry-After is a floor.
    delay = min(cfg.backoff_max_s, cfg.backoff_base_s * 2 ** (attempt - 1))
    if err.retry_after_s:
        delay = max(delay, err.retry_after_s)
    return delay


def choose(cfg: AgentConfig, err: ErrorInfo, attempt: int) -> RecoveryAction:
    """attempt: 1 for the first occurrence of this error signature since the last progress."""
    if err.kind in FATAL:
        return RecoveryAction(kind="fail", reason="not recoverable: configuration error")
    if attempt > cfg.max_retries:
        return RecoveryAction(kind="fail", reason=f"gave up after {cfg.max_retries} recovery attempts")

    if cfg.recovery == Strategy.FAIL_FAST:
        return RecoveryAction(kind="fail", reason="fail-fast policy")

    if cfg.recovery == Strategy.NAIVE_RETRY:
        return RecoveryAction(kind="retry", reason="naive retry")

    if err.transient:
        delay = backoff_delay(cfg, err, attempt)
        return RecoveryAction(kind="retry", delay_s=delay, reason=f"transient; retry after {delay:g}s")

    if cfg.recovery == Strategy.BACKOFF_RETRY:
        return RecoveryAction(kind="fail", reason="not transient; backoff policy does not retry it")

    # REPAIR
    if err.kind == ErrorKind.LLM_CONTEXT_OVERFLOW:
        return RecoveryAction(kind="compact", reason="shrink old observations and ask again")
    return RecoveryAction(kind="repair", reason="return the error to the model so it can correct itself")
