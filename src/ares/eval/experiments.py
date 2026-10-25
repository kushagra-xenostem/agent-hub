"""Experiment grids: (fault plan x agent configuration x seed x task) -> summarized cells."""

from __future__ import annotations
import time
from pydantic import BaseModel
from ares.llm.base import LLMClient
from ares.eval.graders import faithfulness
from collections.abc import Callable, Iterable
from ares.faults import FaultInjector, FaultPlan
from ares.tracing.store import NullTracer, Tracer
from ares.eval.metrics import Scored, Summary, summarize
from ares.runtime import Agent, AgentConfig, Phase, Task
from ares.tools import Corpus, ToolRegistry, default_registry

ModelFactory = Callable[[int], LLMClient]  # seed -> fresh model client for one run

SIM_TASKS = [
    Task(id="ssb", prompt="What is the outlook for the solid-state battery market?"),
    Task(id="riscv", prompt="How big is the RISC-V edge AI chip market and who leads it?"),
    Task(id="vfarm", prompt="What is the state of the vertical farming market and its economics?"),
]


def build_agent(llm: LLMClient, corpus: Corpus, config: AgentConfig, *, plan: FaultPlan | None = None,
                seed: int | str = 0, registry: ToolRegistry | None = None, tracer: Tracer | None = None,
                sleep: Callable[[float], None] = time.sleep, **agent_kw) -> Agent:
    registry = registry or default_registry()
    injector = None
    if plan is not None and plan.faults:
        injector = FaultInjector(plan, seed)
        registry = injector.wrap_registry(registry)
        llm = injector.wrap_llm(llm)
    return Agent(llm, registry, corpus, config=config, tracer=tracer or NullTracer(), injector=injector,
                 sleep=sleep, **agent_kw)


def score(agent: Agent, task: Task, corpus: Corpus, *, run_id: str | None = None) -> Scored:
    res = agent.run(task, run_id=run_id)
    faith = faithfulness(res.final, corpus)
    return Scored(result=res, faithfulness=faith, correct=res.status == Phase.DONE and faith == 1.0)


class Cell(BaseModel):
    plan: str
    variant: str
    summary: Summary


def grid(plans: Iterable[FaultPlan], variants: dict[str, AgentConfig], seeds: Iterable[int], *,
         model_factory: ModelFactory, corpus: Corpus, tasks: list[Task] = SIM_TASKS,
         sleep: Callable[[float], None] = lambda s: None, tracer: Tracer | None = None) -> list[Cell]:
    seeds = list(seeds)
    cells = []
    for plan in plans:
        for name, config in variants.items():
            rows = []
            for seed in seeds:
                for task in tasks:
                    cfg = config.model_copy(update={"variant": name, "seed": seed})
                    # faults are drawn independently per (seed, task); the model seed stays the seed
                    agent = build_agent(model_factory(seed), corpus, cfg, plan=plan, seed=f"{seed}:{task.id}",
                                        tracer=tracer, sleep=sleep)
                    rows.append(score(agent, task, corpus))
            cells.append(Cell(plan=plan.name, variant=name, summary=summarize(rows)))
    return cells


def markdown_table(cells: list[Cell]) -> str:
    head = ("| fault plan | variant | success (95% CI) | correct | recovery | LLM calls | tool runs | retries | "
            "latency s (mean / p95) | arg validity | faults/run |")
    lines = [head, "|" + "---|" * 11]
    for c in cells:
        s = c.summary
        rec = "–" if s.recovery_rate is None else f"{s.recovery_rate:.0%}"
        av = "–" if s.arg_validity is None else f"{s.arg_validity:.0%}"
        ci = f" ({s.success_ci[0]:.0%}–{s.success_ci[1]:.0%})"
        lines.append(f"| {c.plan} | {c.variant} | {s.success_rate:.0%}{ci} | {s.correct_rate:.0%} | {rec} | "
                     f"{s.mean_llm_calls:.1f} | {s.mean_tool_runs:.1f} | {s.mean_retries:.1f} | "
                     f"{s.mean_latency_s:.1f} / {s.p95_latency_s:.1f} | {av} | {s.injected_per_run:.1f} |")
    return "\n".join(lines)
