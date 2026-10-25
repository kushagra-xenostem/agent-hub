"""Command line: check the API, run a task, list runs, inspect a run's timeline."""
from __future__ import annotations
import json
import time
import typer
import datetime
from pathlib import Path
from rich.table import Table
from rich.markup import escape
from ares.faults import PRESETS
from rich.console import Console
from ares.errors import AgentError
from ares.config import PROJECT_ROOT
from ares.eval.experiments import build_agent
from ares.tools import Corpus, default_registry
from ares.llm import LLMRequest, OpenAICompatClient
from ares.tracing.render import ConsoleTracer, describe
from ares.runtime import Agent, AgentConfig, Strategy, Task
from ares.tracing import EventKind, MultiTracer, SQLiteTraceStore

app = typer.Typer(no_args_is_help=True, add_completion=False, help="ARES agent runtime")
console = Console()
DEFAULT_DB = PROJECT_ROOT / "runs" / "traces.sqlite"


def _snap(corpus: Corpus) -> Corpus:
    corpus.snapshot()  # keep recorded runs replayable even after the corpus changes
    return corpus


def _select_tasks(spec: str):
    from ares.eval.dataset import load_tasks

    tasks = load_tasks()
    if spec == "all":
        return tasks
    if spec == "core":  # 2 per category: the subset used for the more expensive live grids
        picked, seen = [], {}
        for t in tasks:
            if seen.get(t.category, 0) < 2:
                picked.append(t)
                seen[t.category] = seen.get(t.category, 0) + 1
        return picked
    if spec.startswith("category:"):
        return [t for t in tasks if t.category == spec.split(":", 1)[1]]
    ids = set(spec.split(","))
    return [t for t in tasks if t.id in ids]


@app.command("eval")
def eval_cmd(
    name: str = typer.Argument(..., help="Results name: runs/eval/<name>.jsonl (resumable)."),
    model: str = typer.Option("live", help="live (the API in .env) or sim (simulated researcher)."),
    variants: str = typer.Option("L4_full", help="Comma-separated: L1_fsm, L2_validation, L3_recovery, L4_full."),
    plans: str = typer.Option("none", help="Comma-separated fault plans."),
    tasks: str = typer.Option("all", help="all | core | category:<name> | comma-separated ids."),
    seeds: int = typer.Option(1),
    repeats: int = typer.Option(1, help="Repeat each (task, seed) to measure reproducibility."),
    db: Path = typer.Option(DEFAULT_DB),
) -> None:
    """Run the evaluation dataset and grade every run (resumes if interrupted)."""
    from ares.eval.runner import category_markdown, eval_markdown, load_records, run_eval, summarize_eval
    from ares.llm.simulated import SimulatedResearcher

    chosen = _select_tasks(tasks)
    results = PROJECT_ROOT / "runs" / "eval" / f"{name}.jsonl"
    if model == "live":
        client = OpenAICompatClient.from_env()
        factory, tracer, sleep = (lambda s: client), SQLiteTraceStore(db), time.sleep
    else:
        factory, tracer, sleep = (lambda s: SimulatedResearcher(seed=s, p_bad_args=0.1)), None, (lambda s: None)

    def progress(r):
        g = r.grade
        console.print(f"{r.task_id:<24} {r.variant:<14} {r.plan:<8} seed={r.seed} rep={r.repeat} "
                      f"{'OK  ' if g.success else 'FAIL'} status={g.status:<6} facts={g.fact_recall} "
                      f"llm={g.llm_calls} {g.latency_s:.0f}s {r.run_id}", highlight=False, markup=False)

    run_eval(chosen, variants.split(","), plans.split(","), list(range(seeds)), model_factory=factory,
             corpus=_snap(Corpus.load()), results=results, repeats=repeats, tracer=tracer, sleep=sleep, on_record=progress)
    records = load_records(results)
    console.print(eval_markdown(summarize_eval(records)), markup=False, highlight=False)
    for v in dict.fromkeys(r.variant for r in records):
        console.print(f"\n{v}:", highlight=False)
        console.print(category_markdown(records, v, "none"), markup=False, highlight=False)


@app.command()
def baseline(
    name: str = typer.Argument(..., help="Results name: runs/eval/<name>.jsonl (shared with `ares eval`)."),
    tasks: str = typer.Option("all", help="all | core | category:<name> | comma-separated ids."),
    plans: str = typer.Option("none", help="Comma-separated tool-fault plans."),
    seeds: int = typer.Option(1),
    attacks: bool = typer.Option(False, help="Run the security attacks against the crew instead."),
    db: Path = typer.Option(DEFAULT_DB),
) -> None:
    """Run the original CrewAI crew (needs .venv-baseline) on the eval tasks or the attacks."""
    from ares.baseline.crew import run_baseline_attack, run_baseline_eval

    store = SQLiteTraceStore(db)
    if attacks:
        from ares.security.attacks import ATTACKS, attack_corpus
        from ares.security.suite import to_json

        corpus, rows = _snap(attack_corpus()), []
        out = PROJECT_ROOT / "docs" / "results" / "security_live_baseline.json"
        for attack in ATTACKS:
            row, _, record = run_baseline_attack(attack, corpus)
            store.start_run(record)
            store.end_run(record)
            rows.append(row)
            console.print(f"{row.attack} B0_crewai delivered={row.delivered} seen={row.injection_seen} "
                          f"attempted={row.attempted} SUCCEEDED={row.succeeded} status={row.status} {row.run_id}",
                          highlight=False, markup=False)
            out.write_text(to_json(rows), encoding="utf-8")
        return

    def progress(r):
        g = r.grade
        console.print(f"{r.task_id:<24} B0_crewai {r.plan:<8} {'OK  ' if g.success else 'FAIL'} status={g.status} "
                      f"facts={g.fact_recall} unsupported={len(g.unsupported_numbers)} tools={g.tool_runs} "
                      f"{g.latency_s:.0f}s {r.run_id}", highlight=False, markup=False)

    run_baseline_eval(_select_tasks(tasks), plans.split(","), list(range(seeds)), corpus=_snap(Corpus.load()),
                      results=PROJECT_ROOT / "runs" / "eval" / f"{name}.jsonl", store=store, on_record=progress)


@app.command()
def report(port: int = 8000, host: str = "127.0.0.1", db: Path = typer.Option(DEFAULT_DB)) -> None:
    """Serve the replay/report web view."""
    import uvicorn

    from ares.report.app import create_app

    console.print(f"ARES report on http://{host}:{port}")
    uvicorn.run(create_app(db=db), host=host, port=port, log_level="warning")


@app.command()
def replay(run_id: str, db: Path = typer.Option(DEFAULT_DB), strict: bool = False) -> None:
    """Re-run the current code against a recorded run (zero model calls): verify it, or find the divergence."""
    from ares.replay import replay as do_replay

    rep, _ = do_replay(SQLiteTraceStore(db), run_id, strict=strict)
    console.print_json(rep.model_dump_json())
    console.print("[green]VERIFIED[/]" if rep.verified else "[red]NOT REPRODUCED[/]")


@app.command()
def otel(run_id: str, endpoint: str = typer.Option("", help="OTLP/HTTP collector, e.g. http://localhost:4318"),
         out: Path = typer.Option(PROJECT_ROOT / "runs" / "otel" / "spans.jsonl"),
         db: Path = typer.Option(DEFAULT_DB)) -> None:
    """Export a stored run as OpenTelemetry spans (to a collector, or a JSON-lines file)."""
    from ares.tracing.otel import JsonFileExporter, export_run, otlp_exporter

    store = SQLiteTraceStore(db)
    record = store.get_run(run_id)
    if record is None:
        raise typer.BadParameter(f"no run {run_id}")
    exporter = otlp_exporter(endpoint) if endpoint else JsonFileExporter(out)
    n = export_run(record, store.events(record.run_id), exporter)
    console.print(f"exported {n} spans for {record.run_id} -> {endpoint or out}")


@app.command()
def check() -> None:
    """Check the LLM API: reachable, correct sanity answer, structured tool calls."""
    client = OpenAICompatClient.from_env()
    try:
        console.print(f"models: {client.list_models()}")
        r = client.complete(LLMRequest(messages=[{"role": "user", "content": "What is 17*23? Reply with only the number."}],
                                       max_tokens=10))
        ok = (r.content or "").strip() == "391"
        console.print(f"sanity 17*23 -> {r.content!r} {'[green]OK' if ok else '[red]WRONG'}[/] ({r.latency_s:.1f}s)")
        tools = default_registry().schemas(["search_web"])
        r = client.complete(LLMRequest(messages=[{"role": "user", "content": "Search for RISC-V market news."}],
                                       tools=tools, max_tokens=200))
        tc = r.tool_calls[0] if r.tool_calls else None
        shown = escape(f"{tc.name} {tc.arguments}") if tc else "[red]none[/]"
        console.print(f"tool call -> {shown} ({r.latency_s:.1f}s)")
    except AgentError as e:
        console.print(f"[red]{e.kind}: {e.message}")
        raise typer.Exit(1)


@app.command()
def run(
    prompt: str = typer.Argument(..., help="The research question."),
    task_id: str = typer.Option("adhoc", help="Task id stored with the trace."),
    strategy: Strategy = typer.Option(Strategy.REPAIR, help="Recovery strategy."),
    seed: int = typer.Option(0),
    temperature: float = typer.Option(0.0),
    db: Path = typer.Option(DEFAULT_DB, help="SQLite trace database."),
    transitions: bool = typer.Option(False, help="Also print every state transition."),
    faults: str = typer.Option("none", help=f"Fault plan: {', '.join(PRESETS)}."),
    variant: str = typer.Option("ares", help="Label stored with the trace."),
) -> None:
    """Run the agent on one question, printing each step and storing the trace."""
    if faults not in PRESETS:
        raise typer.BadParameter(f"unknown fault plan {faults!r}")
    store = SQLiteTraceStore(db)
    agent = build_agent(OpenAICompatClient.from_env(), _snap(Corpus.load()),
                        AgentConfig(recovery=strategy, seed=seed, temperature=temperature, variant=variant),
                        plan=PRESETS[faults], seed=seed,
                        tracer=MultiTracer(store, ConsoleTracer(console, show_transitions=transitions)))
    res = agent.run(Task(id=task_id, prompt=prompt))
    c = res.counters
    console.print(f"\n[bold]{res.status.upper()}[/] {escape(str(res.reason))}  |  {res.duration_s:.1f}s, "
                  f"{c.llm_calls} LLM calls, {c.tool_executions} tool runs, {c.total_tokens} tokens, "
                  f"errors={escape(str(c.errors or 'none'))}")
    if res.final:
        console.print(f"\n[bold]{escape(res.final['title'])}[/]\n{escape(res.final['summary'])}")
        for f in res.final["findings"]:
            console.print(f" - {escape(f['claim'])}  [dim]{escape(f['source_url'])}[/]")
    console.print(f"\ntrace: ares show {res.run_id}")


def sim_sized(plan):
    """The simulated researcher's prompts peak around 2k tokens (the real model's at 5-10k), so the
    context-limit plan gets a window that it actually exceeds."""
    if plan.name != "context_limit":
        return plan
    faults = [f.model_copy(update={"limit_tokens": 1600}) for f in plan.faults]
    return plan.model_copy(update={"faults": faults, "description": "A 1,600-token context window (sim-sized)."})


@app.command()
def experiment(
    name: str = typer.Argument("recovery", help="recovery | validation"),
    seeds: int = typer.Option(30, help="Seeds per cell (x3 tasks)."),
    out: Path = typer.Option(PROJECT_ROOT / "docs" / "results", help="Where to write the markdown/JSON results."),
) -> None:
    """Offline experiment grids with the simulated researcher (fast, deterministic)."""
    from ares.eval.experiments import grid, markdown_table
    from ares.llm.simulated import SimulatedResearcher

    corpus = _snap(Corpus.load())
    if name == "recovery":
        plans = [PRESETS[p] for p in ("none", "tool_errors", "tool_errors_permanent", "tool_timeouts",
                                      "malformed_outputs", "empty_outputs", "llm_rate_limits", "llm_unavailable",
                                      "llm_garbled", "llm_no_tool", "context_limit", "mixed")]
        plans = [sim_sized(p) for p in plans]
        variants = {s.value: AgentConfig(recovery=s) for s in Strategy}
        factory = lambda s: SimulatedResearcher(seed=s, p_bad_args=0.1)
    elif name == "validation":
        plans = [PRESETS[p] for p in ("none", "malformed_outputs", "conflicts", "oversized_outputs")]
        variants = {
            "all_checks": AgentConfig(),
            "no_output_validation": AgentConfig(validate_output=False),
            "no_arg_validation": AgentConfig(validate_args=False),
            "no_observation_cap": AgentConfig(max_observation_chars=None, compact_at=None),
        }
        factory = lambda s: SimulatedResearcher(seed=s, p_bad_args=0.1)
    else:
        raise typer.BadParameter("experiment must be 'recovery' or 'validation'")
    t0 = time.perf_counter()
    cells = grid(plans, variants, range(seeds), model_factory=factory, corpus=corpus)
    elapsed = time.perf_counter() - t0
    table = markdown_table(cells)
    out.mkdir(parents=True, exist_ok=True)
    header = (f"# Experiment: {name}\n\nSimulated researcher (p_bad_args=0.1), {seeds} seeds x 3 tasks per cell, "
              f"{len(cells) * seeds * 3} runs in {elapsed:.1f}s. Latency is simulated model/tool time plus "
              f"backoff waits.\n\n")
    (out / f"{name}.md").write_text(header + table + "\n", encoding="utf-8")
    (out / f"{name}.json").write_text(json.dumps([c.model_dump() for c in cells], indent=1), encoding="utf-8")
    console.print(table, highlight=False, markup=False)
    console.print(f"\n{len(cells) * seeds * 3} runs in {elapsed:.1f}s -> {out / (name + '.md')}")


@app.command()
def security(
    live: bool = typer.Option(False, help="Also attack the real model (defended vs undefended)."),
    attacks: str = typer.Option("all", help="Comma-separated attack ids for --live, e.g. A1,A3."),
    db: Path = typer.Option(DEFAULT_DB, help="SQLite trace database for live runs."),
    out: Path = typer.Option(PROJECT_ROOT / "docs" / "results"),
) -> None:
    """Security suite: worst-case guarantee matrix (offline) and, with --live, real-model attack runs."""
    from ares.security.attacks import ATTACKS, attack_corpus
    from ares.security.suite import (DEFENDED, UNDEFENDED, guarantee_matrix, live_markdown, matrix_markdown,
                                     run_attack, to_json)

    out.mkdir(parents=True, exist_ok=True)
    rows = guarantee_matrix()
    table = matrix_markdown(rows)
    intro = ("# Security guarantees under a fully compromised model\n\nEach attack is run with a scripted model "
             "that obeys the injection. **LEAK** = the attack's effect happened. Columns remove one control at "
             "a time.\n\n")
    (out / "security_matrix.md").write_text(intro + table + "\n", encoding="utf-8")
    console.print(table, markup=False, highlight=False)
    if not live:
        return
    chosen = [a for a in ATTACKS if attacks == "all" or a.id in attacks.split(",")]
    corpus, store, llm = _snap(attack_corpus()), SQLiteTraceStore(db), OpenAICompatClient.from_env()
    live_rows = []
    for attack in chosen:
        for config in (DEFENDED, UNDEFENDED):
            row, _ = run_attack(attack, config, llm, corpus, tracer=store)
            live_rows.append(row)
            console.print(f"{row.attack} {row.profile:<10} attempted={row.attempted!s:<5} "
                          f"SUCCEEDED={row.succeeded!s:<5} status={row.status} blocked={row.blocked} "
                          f"detected={row.injections_detected} llm_calls={row.llm_calls} {row.run_id}", highlight=False)
            (out / "security_live.json").write_text(to_json(live_rows), encoding="utf-8")
    (out / "security_live.md").write_text("# Live attacks on Qwen3.6-35B-A3B\n\n" + live_markdown(live_rows) + "\n",
                                          encoding="utf-8")


@app.command()
def runs(db: Path = typer.Option(DEFAULT_DB), limit: int = 20) -> None:
    """List recent runs."""
    table = Table("run_id", "started", "task", "variant", "status", "llm", "tools", "tokens", "errors")
    for r in SQLiteTraceStore(db).list_runs(limit):
        c = r.counters
        table.add_row(r.run_id, datetime.datetime.fromtimestamp(r.started_at).strftime("%m-%d %H:%M:%S"), r.task_id,
                      r.variant, r.status, str(c.get("llm_calls", "")), str(c.get("tool_executions", "")),
                      str(c.get("prompt_tokens", 0) + c.get("completion_tokens", 0)),
                      json.dumps(c.get("errors") or {}))
    console.print(table)


@app.command()
def show(run_id: str, db: Path = typer.Option(DEFAULT_DB), transitions: bool = False) -> None:
    """Print one run's timeline from the trace store."""
    store = SQLiteTraceStore(db)
    record = store.get_run(run_id)
    if record is None:
        console.print(f"[red]no run {run_id}")
        raise typer.Exit(1)
    console.print(f"[bold]{record.run_id}[/] task={record.task_id} variant={record.variant} model={record.model} "
                  f"status={record.status}")
    for ev in store.events(record.run_id):
        if ev.kind == EventKind.TRANSITION and not transitions:
            continue
        style, text = describe(ev)
        console.print(f"[dim]{ev.seq:>3} {ev.phase:<14}[/] [{style}]{escape(text)}[/]", highlight=False)
    if record.final:
        console.print_json(data=record.final)


if __name__ == "__main__":
    app()
