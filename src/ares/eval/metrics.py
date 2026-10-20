"""Aggregate metrics over groups of runs."""
from __future__ import annotations

import math
import statistics
from collections.abc import Iterable

from pydantic import BaseModel

from ares.runtime.agent import RunResult
from ares.runtime.state import Phase


class Scored(BaseModel):
    """One run plus its grades."""

    result: RunResult
    faithfulness: float
    correct: bool  # finished, and every finding is supported by the true source it cites


class Summary(BaseModel):
    n: int
    success_rate: float  # reached DONE
    success_ci: tuple[float, float]  # 95% Wilson interval
    correct_rate: float  # DONE and fully faithful
    recovery_rate: float | None  # among runs that hit an error, the share that still reached DONE
    mean_llm_calls: float
    mean_tool_runs: float
    mean_retries: float
    mean_tokens: float
    mean_latency_s: float  # model time + tool time + backoff waits
    p95_latency_s: float
    arg_validity: float | None  # valid proposed calls / proposed calls
    injected_per_run: float


def summarize(rows: Iterable[Scored]) -> Summary:
    rows = list(rows)
    if not rows:
        raise ValueError("no runs to summarize")
    res = [r.result for r in rows]
    errored = [r for r in rows if r.result.counters.errors]
    latency = sorted(r.counters.llm_latency_s + r.counters.tool_latency_s + r.counters.backoff_s for r in res)
    proposed = sum(r.counters.proposed_calls for r in res)
    return Summary(
        n=len(rows),
        success_rate=_mean(r.status == Phase.DONE for r in res),
        success_ci=wilson(sum(r.status == Phase.DONE for r in res), len(res)),
        correct_rate=_mean(r.correct for r in rows),
        recovery_rate=_mean(r.result.status == Phase.DONE for r in errored) if errored else None,
        mean_llm_calls=_mean(r.counters.llm_calls for r in res),
        mean_tool_runs=_mean(r.counters.tool_executions for r in res),
        mean_retries=_mean(r.counters.retries for r in res),
        mean_tokens=_mean(r.counters.total_tokens for r in res),
        mean_latency_s=_mean(latency),
        p95_latency_s=latency[min(len(latency) - 1, int(0.95 * len(latency)))],
        arg_validity=1 - sum(r.counters.invalid_calls for r in res) / proposed if proposed else None,
        injected_per_run=_mean(r.counters.injected_faults for r in res),
    )


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def _mean(values: Iterable[float | bool]) -> float:
    values = [float(v) for v in values]
    return round(statistics.fmean(values), 4) if values else 0.0
