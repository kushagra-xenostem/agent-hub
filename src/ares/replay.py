"""Replay debugging: re-run the current code against a recorded run's model responses.

The trace stores every LLM request fingerprint and response, the configuration, the task, the fault plan
and seed, and every approval decision. Replay rebuilds the agent from that, serves the recorded responses
in order (no model calls), and checks each new request against the recorded fingerprint.

- Unchanged code: every fingerprint matches and the final state is identical -> the run is reproducible.
- Changed code (e.g. a bug fix): replay shows the first step where the new code's behaviour diverges.
"""
from __future__ import annotations

from pydantic import BaseModel

from ares.errors import AgentError, ErrorKind
from ares.faults import FaultInjector, FaultPlan
from ares.llm.base import LLMRequest, LLMResponse
from ares.runtime import Agent, AgentConfig, Task
from ares.runtime.agent import RunResult
from ares.tools import Corpus, default_registry
from ares.tracing import EventKind, MemoryTracer, SQLiteTraceStore, TraceEvent


class Divergence(BaseModel):
    call_index: int  # 1-based model call where the request differed
    recorded_fingerprint: str
    replayed_fingerprint: str
    recorded_last_message: dict | None
    replayed_last_message: dict | None


class ReplayExhausted(RuntimeError):
    """The new code asked the model more often than the recording did."""


class RecordedLLM:
    def __init__(self, events: list[TraceEvent], model: str, *, strict: bool = True):
        self.model = model
        self.strict = strict
        self.calls: list[tuple[TraceEvent, TraceEvent | None]] = []  # (llm_call, following error event)
        for i, ev in enumerate(events):
            if ev.kind == EventKind.LLM_CALL:
                err = next((e for e in events[i + 1:] if e.kind == EventKind.ERROR), None) if not ev.data["ok"] else None
                self.calls.append((ev, err))
        self.n = 0
        self.divergences: list[Divergence] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.n >= len(self.calls):
            raise ReplayExhausted(f"the recording has only {len(self.calls)} model calls")
        ev, err = self.calls[self.n]
        self.n += 1
        fp = request.fingerprint()
        if fp != ev.data["fingerprint"]:
            recorded_last = (ev.data.get("new_messages") or [None])[-1]
            self.divergences.append(Divergence(call_index=self.n, recorded_fingerprint=ev.data["fingerprint"],
                                               replayed_fingerprint=fp, recorded_last_message=recorded_last,
                                               replayed_last_message=request.messages[-1] if request.messages else None))
            if self.strict:
                raise ReplayExhausted(f"diverged at model call {self.n}")
        if not ev.data["ok"]:
            d = err.data if err else {}
            raise AgentError(ErrorKind(ev.data["error"]), d.get("message", "recorded error"),
                             transient=d.get("transient"), retry_after_s=d.get("retry_after_s"), detail=d.get("detail"))
        return LLMResponse.model_validate(ev.data["response"])


class ReplayReport(BaseModel):
    run_id: str
    recorded_status: str
    replayed_status: str
    model_calls_recorded: int
    model_calls_replayed: int
    divergences: list[Divergence]
    same_final: bool
    same_trajectory: bool
    verified: bool  # no divergence, same status, same report, same trajectory
    error: str | None = None


def _corpus_for(fingerprint: str | None, task_id: str) -> Corpus:
    from ares.security.attacks import ATTACKS, attack_corpus

    from ares.tools.corpus import SNAPSHOTS

    if fingerprint and (SNAPSHOTS / f"{fingerprint}.jsonl").exists():  # the exact corpus the run used
        return Corpus.load(SNAPSHOTS / f"{fingerprint}.jsonl")
    base = Corpus.load()
    if fingerprint is None:  # runs recorded before traces carried the corpus fingerprint
        return attack_corpus(base) if task_id in {a.id for a in ATTACKS} else base
    if base.fingerprint() == fingerprint:
        return base
    attacked = attack_corpus(base)
    if attacked.fingerprint() == fingerprint:
        return attacked
    raise ValueError("the corpus this run used is not available (its fingerprint matches no known corpus)")


def replay(store: SQLiteTraceStore, run_id: str, *, strict: bool = False, tracer: MemoryTracer | None = None,
           corpus: Corpus | None = None, config_overrides: dict | None = None,
           system_prompt: str | None = None) -> tuple[ReplayReport, RunResult | None]:
    """config_overrides / system_prompt let you replay a recording under changed behaviour (what-if)."""
    record = store.get_run(run_id)
    if record is None:
        raise KeyError(run_id)
    events = store.events(record.run_id)
    start = next(e for e in events if e.kind == EventKind.RUN_START)
    approvals = [e.data["decision"] for e in events if e.kind == EventKind.APPROVAL]

    llm = RecordedLLM(events, record.model, strict=strict)
    registry = default_registry()
    injector = None
    if start.data.get("fault_plan"):
        injector = FaultInjector(FaultPlan.model_validate(start.data["fault_plan"]), start.data["fault_seed"])
        registry = injector.wrap_registry(registry)
        wrapped = injector.wrap_llm(llm)
        llm_for_agent = wrapped
    else:
        llm_for_agent = llm
    decisions = iter(approvals)
    config = AgentConfig.model_validate({**record.config, **(config_overrides or {})})
    extra = {"system_prompt": system_prompt} if system_prompt is not None else {}
    agent = Agent(llm_for_agent, registry, corpus or _corpus_for(start.data.get("corpus"), record.task_id), config=config,
                  tracer=tracer or MemoryTracer(), approver=lambda call, spec: next(decisions, "deny"),
                  injector=injector, sleep=lambda s: None, **extra)
    task = Task.model_validate(start.data["task"])
    result, error = None, None
    try:
        result = agent.run(task, run_id=f"{record.run_id}-replay")
    except ReplayExhausted as e:
        error = str(e)
    recorded_traj = [t["name"] + ":" + t["outcome"] for t in record.meta.get("trajectory", [])]
    replayed_traj = [f"{t.name}:{t.outcome}" for t in result.trajectory] if result else []
    same_final = result is not None and result.final == record.final
    same_traj = recorded_traj == replayed_traj
    status = result.status.value if result else "diverged"
    return ReplayReport(
        run_id=record.run_id, recorded_status=record.status, replayed_status=status,
        model_calls_recorded=len(llm.calls), model_calls_replayed=llm.n, divergences=llm.divergences,
        same_final=same_final, same_trajectory=same_traj, error=error,
        verified=not llm.divergences and status == record.status and same_final and same_traj
        and llm.n == len(llm.calls),
    ), result
