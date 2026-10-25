"""Evaluation runner: (tasks x variants x fault plans x seeds x repeats), resumable, graded, summarized.

Every finished run is appended to a JSONL results file immediately, so an interrupted live evaluation
resumes where it stopped. Traces go to the SQLite store as usual.
"""

from __future__ import annotations
import json
import time
import statistics
from pathlib import Path
from ares.tools import Corpus
from pydantic import BaseModel
from ares.faults import PRESETS
from ares.llm.base import LLMClient
from collections import defaultdict
from ares.eval.metrics import wilson
from collections.abc import Callable
from ares.eval.experiments import build_agent
from ares.tracing.store import NullTracer, Tracer
from ares.eval.dataset import EvalTask, Grade, grade
from ares.runtime.config import AgentConfig, Strategy

# The ablation ladder: each step adds one family of mechanisms.
_OFF = dict(validate_args=False, validate_output=False, check_grounding=False, check_claims=False, url_provenance=False,
            enforce_permissions=False, hide_unpermitted_tools=False, require_approval=False, egress_allowlist=False,
            dlp=False, mark_untrusted=False, injection_filter="off", repair_temperature=None)
VARIANTS: dict[str, AgentConfig] = {
    "L1_fsm": AgentConfig(variant="L1_fsm", recovery=Strategy.FAIL_FAST, **_OFF),
    "L2_validation": AgentConfig(variant="L2_validation", recovery=Strategy.FAIL_FAST,
                                 **{**_OFF, "validate_args": True, "validate_output": True, "check_grounding": True,
                                    "check_claims": True, "url_provenance": True}),
    "L3_recovery": AgentConfig(variant="L3_recovery", recovery=Strategy.REPAIR,
                               **{**_OFF, "validate_args": True, "validate_output": True, "check_grounding": True,
                                  "check_claims": True, "url_provenance": True, "repair_temperature": 0.7}),
    "L4_full": AgentConfig(variant="L4_full"),
}


class EvalRecord(BaseModel):
    task_id: str
    variant: str
    plan: str
    seed: int
    repeat: int
    run_id: str
    model: str
    grade: Grade
    final: dict | None
    trajectory: list[str]
    started_at: float

    @property
    def key(self) -> tuple:
        return (self.task_id, self.variant, self.plan, self.seed, self.repeat)


def load_records(path: Path) -> list[EvalRecord]:
    if not path.exists():
        return []
    return [EvalRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_eval(tasks: list[EvalTask], variants: list[str], plans: list[str], seeds: list[int], *,
             model_factory: Callable[[int], LLMClient], corpus: Corpus, results: Path, repeats: int = 1,
             tracer: Tracer | None = None, sleep: Callable[[float], None] = time.sleep,
             on_record: Callable[[EvalRecord], None] | None = None) -> list[EvalRecord]:
    results.parent.mkdir(parents=True, exist_ok=True)
    done = {r.key for r in load_records(results)}
    new = []
    for plan_name in plans:
        for variant in variants:
            for seed in seeds:
                for repeat in range(repeats):
                    for task in tasks:
                        key = (task.id, variant, plan_name, seed, repeat)
                        if key in done:
                            continue
                        cfg = VARIANTS[variant].model_copy(update={"seed": seed})
                        llm = model_factory(seed)
                        agent = build_agent(llm, corpus, cfg, plan=PRESETS[plan_name], seed=f"{seed}:{task.id}",
                                            tracer=tracer or NullTracer(), sleep=sleep)
                        started = time.time()
                        res = agent.run(task.as_task())
                        rec = EvalRecord(task_id=task.id, variant=variant, plan=plan_name, seed=seed, repeat=repeat,
                                         run_id=res.run_id, model=llm.model, grade=grade(task, res, corpus),
                                         final=res.final, trajectory=[t.name for t in res.trajectory],
                                         started_at=started)
                        with results.open("a", encoding="utf-8") as f:
                            f.write(rec.model_dump_json() + "\n")
                        new.append(rec)
                        if on_record:
                            on_record(rec)
    return new


# --- aggregation ----------------------------------------------------------------------------------
class EvalSummary(BaseModel):
    variant: str
    plan: str
    n: int
    success: float
    success_ci: tuple[float, float]
    fact_recall: float | None
    faithfulness: float
    tool_selection_accuracy: float | None
    arg_validity: float | None
    recovery_rate: float | None
    unnecessary_llm_calls: float
    mean_llm_calls: float
    mean_tokens: float
    mean_latency_s: float
    p95_latency_s: float
    reproducibility: float | None  # identical report + trajectory across repeats of the same (task, seed)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def summarize_eval(records: list[EvalRecord]) -> list[EvalSummary]:
    groups: dict[tuple[str, str], list[EvalRecord]] = defaultdict(list)
    for r in records:
        groups[(r.variant, r.plan)].append(r)
    out = []
    for (variant, plan), rs in groups.items():
        g = [r.grade for r in rs]
        errored = [x for x in g if x.errors]
        lat = sorted(x.latency_s for x in g)
        by_case: dict[tuple, list[EvalRecord]] = defaultdict(list)
        for r in rs:
            by_case[(r.task_id, r.seed)].append(r)
        pairs = [v for v in by_case.values() if len(v) > 1]
        repro = None
        if pairs:
            same = [all(json.dumps(x.final, sort_keys=True) == json.dumps(v[0].final, sort_keys=True)
                        and x.trajectory == v[0].trajectory for x in v[1:]) for v in pairs]
            repro = round(sum(same) / len(same), 4)
        out.append(EvalSummary(
            variant=variant, plan=plan, n=len(g), success=_mean([x.success for x in g]),
            success_ci=wilson(sum(x.success for x in g), len(g)),
            fact_recall=_mean([x.fact_recall for x in g]), faithfulness=_mean([x.faithfulness for x in g]),
            tool_selection_accuracy=_mean([x.tool_selection_accuracy for x in g]),
            arg_validity=_mean([x.arg_validity for x in g]),
            recovery_rate=_mean([x.status == "done" for x in errored]) if errored else None,
            unnecessary_llm_calls=_mean([x.unnecessary_llm_calls for x in g]),
            mean_llm_calls=_mean([x.llm_calls for x in g]), mean_tokens=_mean([x.tokens for x in g]),
            mean_latency_s=_mean([x.latency_s for x in g]),
            p95_latency_s=lat[min(len(lat) - 1, int(0.95 * len(lat)))], reproducibility=repro,
        ))
    return out


def _pct(x):
    return "–" if x is None else f"{x:.0%}"


def eval_markdown(summaries: list[EvalSummary]) -> str:
    lines = ["| variant | faults | n | **success** (95% CI) | fact recall | faithfulness | tool selection | "
             "arg validity | recovery | unnecessary LLM calls | LLM calls | tokens | latency s (mean / p95) | "
             "reproducibility |", "|" + "---|" * 14]
    for s in summaries:
        lines.append(
            f"| {s.variant} | {s.plan} | {s.n} | **{_pct(s.success)}** ({s.success_ci[0]:.0%}–{s.success_ci[1]:.0%}) | "
            f"{_pct(s.fact_recall)} | {_pct(s.faithfulness)} | {_pct(s.tool_selection_accuracy)} | "
            f"{_pct(s.arg_validity)} | {_pct(s.recovery_rate)} | {s.unnecessary_llm_calls:.1f} | "
            f"{s.mean_llm_calls:.1f} | {s.mean_tokens:,.0f} | {s.mean_latency_s:.1f} / {s.p95_latency_s:.1f} | "
            f"{_pct(s.reproducibility)} |")
    return "\n".join(lines)


def category_markdown(records: list[EvalRecord], variant: str, plan: str) -> str:
    rs = [r for r in records if r.variant == variant and r.plan == plan]
    by: dict[str, list[EvalRecord]] = defaultdict(list)
    for r in rs:
        by[r.grade.category].append(r)
    lines = ["| category | n | success | fact recall | faithfulness | failed tasks |", "|---|---|---|---|---|---|"]
    for cat, xs in sorted(by.items()):
        failed = ", ".join(sorted({r.task_id for r in xs if not r.grade.success})) or "–"
        lines.append(f"| {cat} | {len(xs)} | {_pct(_mean([r.grade.success for r in xs]))} | "
                     f"{_pct(_mean([r.grade.fact_recall for r in xs]))} | {_pct(_mean([r.grade.faithfulness for r in xs]))} "
                     f"| {failed} |")
    return "\n".join(lines)


# --- regrading from stored traces (no model calls) ------------------------------------------------
def rebuild_result(store, run_id: str):
    """Reconstruct a RunResult from the trace store, for regrading with an updated grader."""
    import json as _json

    from ares.runtime.agent import RunResult
    from ares.runtime.state import Counters, Phase, ToolStep
    from ares.tracing import EventKind

    rec = store.get_run(run_id)
    if rec is None:
        raise KeyError(run_id)
    if rec.meta.get("tool_log") is not None:  # CrewAI baseline: rebuild calls + outputs from its tool log
        messages = []
        for i, c in enumerate(rec.meta["tool_log"]):
            cid = f"{run_id}-{i}"
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": c["name"], "arguments": _json.dumps(c["args"])}}]})
            messages.append({"role": "tool", "tool_call_id": cid,
                             "content": _json.dumps(c.get("output", {"error": c.get("error")}), ensure_ascii=False)})
        trajectory = [ToolStep(name=c["name"], call_id=f"{run_id}-{i}", outcome="ok" if c["ok"] else "tool_error")
                      for i, c in enumerate(rec.meta["tool_log"])]
    else:
        end = next(e for e in reversed(store.events(run_id)) if e.kind == EventKind.RUN_END)
        messages = end.data["messages"]
        trajectory = [ToolStep(**t) for t in rec.meta.get("trajectory", [])]
    return RunResult(run_id=run_id, task_id=rec.task_id, variant=rec.variant, status=Phase(rec.status),
                     reason=rec.reason, final=rec.final, counters=Counters(**rec.counters),
                     duration_s=round((rec.ended_at or rec.started_at) - rec.started_at, 2),
                     fetched_urls=rec.meta.get("fetched_urls", []), trajectory=trajectory,
                     outbox=rec.meta.get("outbox", []), notes=rec.meta.get("notes", {}), messages=messages)


def regrade(results: Path, store, tasks: list[EvalTask], corpus: Corpus) -> list[EvalRecord]:
    """Re-apply the current graders to every stored run; keeps a backup of the old results file."""
    by_id = {t.id: t for t in tasks}
    records = load_records(results)
    backup = results.with_suffix(".before-regrade.jsonl")
    if not backup.exists():
        backup.write_text(results.read_text(encoding="utf-8"), encoding="utf-8")
    out = []
    for r in records:
        res = rebuild_result(store, r.run_id)
        out.append(r.model_copy(update={"grade": grade(by_id[r.task_id], res, corpus)}))
    results.write_text("".join(x.model_dump_json() + "\n" for x in out), encoding="utf-8")
    return out
