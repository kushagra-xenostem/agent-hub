"""Replay / report view: a small server-rendered FastAPI app over the trace store and results files.

  ares report            -> http://127.0.0.1:8000
"""
from __future__ import annotations
import re
import html
import json
import datetime
from pathlib import Path
from collections import defaultdict
from ares.config import PROJECT_ROOT
from ares.tracing.render import describe
from fastapi import FastAPI, HTTPException, Query
from ares.tracing import EventKind, SQLiteTraceStore
from fastapi.responses import HTMLResponse, JSONResponse

KIND_CLASS = {
    "llm_call": "llm", "tool_call": "tool", "error": "err", "recovery": "rec", "fault": "fault",
    "security": "sec", "transition": "dim", "validation": "dim", "permission": "dim", "approval": "rec",
    "compaction": "rec", "wrapup": "rec", "run_start": "dim", "run_end": "end",
}

CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--line:#e7e5e4;--card:#fff;--acc:#2563eb;--ok:#15803d;--bad:#b91c1c;
--warn:#a16207;--vio:#7c3aed;--pink:#be185d}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--fg:#e7e5e4;--mut:#a8a29e;--line:#292524;--card:#1c1917;
--acc:#60a5fa;--ok:#4ade80;--bad:#f87171;--warn:#facc15;--vio:#a78bfa;--pink:#f472b6}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
header{display:flex;gap:20px;align-items:center;padding:12px 24px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header b{font-size:16px}header a{color:var(--mut);text-decoration:none}header a:hover{color:var(--fg)}
main{padding:20px 24px;max-width:1400px;margin:0 auto}a{color:var(--acc)}
h1{font-size:20px;margin:0 0 12px}h2{font-size:16px;margin:24px 0 8px}h3{font-size:14px;margin:16px 0 6px}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{color:var(--mut);font-weight:600;white-space:nowrap}tr:hover td{background:color-mix(in srgb,var(--acc) 5%,transparent)}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 16px}.card{background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:8px 12px;min-width:120px}.card .k{color:var(--mut);font-size:12px}.card .v{font-size:18px;font-weight:600}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600}
.done{background:color-mix(in srgb,var(--ok) 15%,transparent);color:var(--ok)}
.failed,.diverged{background:color-mix(in srgb,var(--bad) 15%,transparent);color:var(--bad)}
.escalated,.running{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
.llm{color:var(--acc)}.tool{color:var(--ok)}.err{color:var(--warn)}.rec{color:var(--vio)}.fault{color:var(--bad);font-weight:600}
.sec{color:var(--pink);font-weight:600}.dim{color:var(--mut)}.end{font-weight:700}
details summary{cursor:pointer}pre{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px;
overflow-x:auto;font-size:12px;max-height:420px}code{font-size:12px}
form.filters{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}input,select,button{font:inherit;padding:4px 8px;
border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
button,.btn{background:var(--acc);color:#fff;border:0;cursor:pointer;text-decoration:none;padding:5px 10px;border-radius:6px}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}.mut{color:var(--mut)}
"""


def esc(x) -> str:
    return html.escape(str(x))


def page(title: str, body: str) -> HTMLResponse:
    nav = ('<header><b>ARES</b><a href="/">Runs</a><a href="/eval">Evaluation</a><a href="/security">Security</a>'
           '<a href="/docs">Results docs</a></header>')
    return HTMLResponse(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
                        f"content='width=device-width,initial-scale=1'><title>{esc(title)} · ARES</title>"
                        f"<style>{CSS}</style></head><body>{nav}<main>{body}</main></body></html>")


def badge(status: str) -> str:
    return f'<span class="badge {esc(status)}">{esc(status)}</span>'


def table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="wrap"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def pct(x) -> str:
    return "–" if x is None else f"{x:.0%}"


_INLINE = [(re.compile(r"\*\*(.+?)\*\*"), r"<b>\1</b>"), (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
           (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r'<a href="\2">\1</a>')]


_LIST_ITEM = re.compile(r"^\s*([-*]|\d+\.)\s+")


def md_inline(text: str) -> str:
    out = esc(text)
    for rx, rep in _INLINE:
        out = rx.sub(rep, out)
    return out


def markdown(text: str) -> str:
    """Enough Markdown for the results docs: headings, tables, lists, code blocks, paragraphs."""
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append(f"<pre>{esc(chr(10).join(lines[i + 1:j]))}</pre>")
            i = j + 1
            continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block if not re.match(r"^\|[\s\-:|]+\|$", r)]
            out.append(table([md_inline(c) for c in cells[0]], [[md_inline(c) for c in r] for r in cells[1:]]))
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            level = min(3, len(m.group(1)))
            out.append(f"<h{level}>{md_inline(m.group(2))}</h{level}>")
        elif re.match(r"^\s*([-*]|\d+\.)\s+", line):
            items = []
            while i < len(lines) and _LIST_ITEM.match(lines[i]):
                items.append(f"<li>{md_inline(_LIST_ITEM.sub('', lines[i]))}</li>")
                i += 1
            out.append(f"<ul>{''.join(items)}</ul>")
            continue
        elif line.strip().startswith(">"):
            out.append(f"<p class='mut'>{md_inline(line.strip().lstrip('> '))}</p>")
        elif line.strip():
            out.append(f"<p>{md_inline(line)}</p>")
        i += 1
    return "\n".join(out)


def create_app(db: Path | None = None, results_dir: Path | None = None, eval_dir: Path | None = None,
               docs_dir: Path | None = None) -> FastAPI:
    db = db or PROJECT_ROOT / "runs" / "traces.sqlite"
    results_dir = results_dir or PROJECT_ROOT / "docs" / "results"
    eval_dir = eval_dir or PROJECT_ROOT / "runs" / "eval"
    docs_dir = docs_dir or PROJECT_ROOT / "docs"
    store = SQLiteTraceStore(db)
    app = FastAPI(title="ARES report")

    @app.get("/", response_class=HTMLResponse)
    def runs(task: str = "", variant: str = "", status: str = "", limit: int = Query(200, le=2000)):
        rows = []
        for r in store.list_runs(limit * 3):
            if (task and task not in r.task_id) or (variant and variant != r.variant) or (status and status != r.status):
                continue
            c = r.counters
            rows.append([
                f'<a href="/runs/{esc(r.run_id)}">{esc(r.run_id)}</a>',
                datetime.datetime.fromtimestamp(r.started_at).strftime("%m-%d %H:%M"), esc(r.task_id), esc(r.variant),
                badge(r.status), str(c.get("llm_calls", "")), str(c.get("tool_executions", "")),
                f"{c.get('prompt_tokens', 0) + c.get('completion_tokens', 0):,}",
                esc(", ".join(f"{k}×{v}" for k, v in (c.get("errors") or {}).items())) or "–",
                str(c.get("injected_faults", 0) or "–"), str(c.get("blocked_actions", 0) or "–")])
            if len(rows) >= limit:
                break
        filters = (f'<form class="filters"><input name="task" placeholder="task" value="{esc(task)}">'
                   f'<input name="variant" placeholder="variant" value="{esc(variant)}">'
                   f'<select name="status"><option value="">any status</option>'
                   + "".join(f'<option {"selected" if status == s else ""}>{s}</option>'
                             for s in ("done", "failed", "escalated", "running"))
                   + '</select><button>Filter</button></form>')
        return page("Runs", f"<h1>Runs</h1>{filters}" + table(
            ["run", "started", "task", "variant", "status", "LLM", "tools", "tokens", "errors", "faults", "blocked"], rows))

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_view(run_id: str, transitions: bool = False):
        r = store.get_run(run_id)
        if r is None:
            raise HTTPException(404, "no such run")
        events = store.events(r.run_id)
        c = r.counters
        dur = (r.ended_at or r.started_at) - r.started_at
        cards = "".join(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in [
            ("status", badge(r.status)), ("duration", f"{dur:.0f}s"), ("LLM calls", c.get("llm_calls", 0)),
            ("tool runs", c.get("tool_executions", 0)), ("tokens", f"{c.get('prompt_tokens', 0) + c.get('completion_tokens', 0):,}"),
            ("retries / repairs", f"{c.get('retries', 0)} / {c.get('repairs', 0)}"),
            ("faults injected", c.get("injected_faults", 0)), ("security blocks", c.get("blocked_actions", 0))])
        rows = []
        for ev in events:
            if ev.kind == EventKind.TRANSITION and not transitions:
                continue
            _, text = describe(ev)
            detail = ""
            if ev.kind in (EventKind.LLM_CALL, EventKind.TOOL_CALL, EventKind.ERROR, EventKind.SECURITY, EventKind.FAULT,
                           EventKind.RUN_START, EventKind.RUN_END):
                detail = f"<details><summary class='mut'>data</summary><pre>{esc(json.dumps(ev.data, indent=1, ensure_ascii=False)[:20000])}</pre></details>"
            ms = f"{ev.duration_ms:.0f}" if ev.duration_ms else ""
            rows.append([str(ev.seq), f"<span class='mut'>{esc(ev.phase)}</span>",
                         f"<span class='{KIND_CLASS.get(ev.kind.value, '')}'>{esc(ev.kind.value)}</span>",
                         f"<span class='{KIND_CLASS.get(ev.kind.value, '')}'>{esc(text)}</span>{detail}", ms])
        final = ""
        if r.final:
            findings = "".join(f"<li>{esc(f.get('claim', ''))} <a class='mut' href='{esc(f.get('source_url', ''))}'>"
                               f"{esc(f.get('source_url', ''))}</a></li>" for f in r.final.get("findings") or []
                               if isinstance(f, dict))
            final = (f"<h2>Final report</h2><h3>{esc(r.final.get('title', ''))}</h3>"
                     f"<p style='white-space:pre-wrap'>{esc(r.final.get('summary', ''))}</p><ul>{findings}</ul>")
        tool_log = ""
        if r.meta.get("tool_log"):  # baseline runs carry their tool log instead of events
            tool_log = "<h2>Tool log</h2>" + table(["#", "tool", "args", "ok"], [
                [str(i), esc(t["name"]), f"<code>{esc(json.dumps(t['args'])[:200])}</code>", "yes" if t["ok"] else
                 f"<span class='bad'>{esc(t.get('error', ''))[:200]}</span>"] for i, t in enumerate(r.meta["tool_log"])])
        actions = (f'<p><a class="btn" href="/runs/{esc(r.run_id)}/replay">Replay this run</a> '
                   f'<a href="/runs/{esc(r.run_id)}/otel.json">OpenTelemetry spans (JSON)</a> · '
                   f'<a href="?transitions={"false" if transitions else "true"}">{"hide" if transitions else "show"} transitions</a></p>')
        head = (f"<h1>{esc(r.run_id)}</h1><p class='mut'>task <b>{esc(r.task_id)}</b> · variant <b>{esc(r.variant)}</b> · "
                f"model {esc(r.model)} · {esc(r.reason or '')}</p>")
        timeline = ("<h2>Timeline</h2>" + table(["#", "phase", "kind", "what happened", "ms"], rows)) if rows else ""
        return page(r.run_id, head + cards + (actions if events else "") + final + timeline + tool_log)

    @app.get("/runs/{run_id}/otel.json")
    def run_otel(run_id: str):
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from ares.tracing.otel import export_run

        r = store.get_run(run_id)
        if r is None:
            raise HTTPException(404, "no such run")
        exp = InMemorySpanExporter()
        export_run(r, store.events(r.run_id), exp)
        return JSONResponse([json.loads(s.to_json()) for s in exp.get_finished_spans()])

    @app.get("/runs/{run_id}/replay", response_class=HTMLResponse)
    def run_replay(run_id: str):
        from ares.replay import replay

        r = store.get_run(run_id)
        if r is None:
            raise HTTPException(404, "no such run")
        try:
            rep, _ = replay(store, r.run_id)
        except Exception as e:  # noqa: BLE001 - shown to the user
            return page("Replay", f"<h1>Replay of {esc(r.run_id)}</h1><p class='bad'>Cannot replay: {esc(e)}</p>")
        verdict = ("<p class='ok'>Verified: the current code reproduces this run exactly, with zero model calls.</p>"
                   if rep.verified else "<p class='bad'>Not reproduced by the current code.</p>")
        rows = [[k, esc(v)] for k, v in [("recorded status", rep.recorded_status), ("replayed status", rep.replayed_status),
                                        ("model calls recorded / replayed", f"{rep.model_calls_recorded} / {rep.model_calls_replayed}"),
                                        ("same final report", rep.same_final), ("same trajectory", rep.same_trajectory),
                                        ("error", rep.error or "–")]]
        div = ""
        if rep.divergences:
            d = rep.divergences[0]
            div = (f"<h2>First divergence: model call {d.call_index}</h2><div class='wrap'><table><tr><th>recorded request ended with</th>"
                   f"<th>current code sent</th></tr><tr><td><pre>{esc(json.dumps(d.recorded_last_message, indent=1)[:4000])}</pre></td>"
                   f"<td><pre>{esc(json.dumps(d.replayed_last_message, indent=1)[:4000])}</pre></td></tr></table></div>")
        return page("Replay", f"<h1>Replay of <a href='/runs/{esc(r.run_id)}'>{esc(r.run_id)}</a></h1>{verdict}"
                    + table(["", ""], rows) + div)

    @app.get("/eval", response_class=HTMLResponse)
    def eval_view(name: str = "main"):
        from ares.eval.runner import category_markdown, load_records, summarize_eval

        files = sorted(p.stem for p in eval_dir.glob("*.jsonl")) if eval_dir.exists() else []
        picker = " · ".join(f'<a href="/eval?name={esc(f)}">{"<b>" + esc(f) + "</b>" if f == name else esc(f)}</a>'
                            for f in files) or "no results yet"
        records = load_records(eval_dir / f"{name}.jsonl")
        if not records:
            return page("Evaluation", f"<h1>Evaluation</h1><p>{picker}</p><p class='mut'>No records in {esc(name)}.</p>")
        summ = summarize_eval(records)
        summary_rows = [[esc(s.variant), esc(s.plan), str(s.n),
                         f"<b>{pct(s.success)}</b> <span class='mut'>({s.success_ci[0]:.0%}–{s.success_ci[1]:.0%})</span>",
                         pct(s.fact_recall), pct(s.faithfulness), pct(s.tool_selection_accuracy), pct(s.arg_validity),
                         pct(s.recovery_rate), f"{s.unnecessary_llm_calls:.1f}", f"{s.mean_llm_calls:.1f}",
                         f"{s.mean_tokens:,.0f}", f"{s.mean_latency_s:.0f} / {s.p95_latency_s:.0f}", pct(s.reproducibility)]
                        for s in sorted(summ, key=lambda s: (s.plan, s.variant))]
        grid: dict[str, dict[str, list]] = defaultdict(dict)
        variants = sorted({(r.variant, r.plan) for r in records})
        for r in records:
            grid[r.task_id].setdefault(f"{r.variant}|{r.plan}", []).append(r)
        grid_rows = []
        for task_id in sorted(grid):
            cells = [esc(task_id)]
            for v, p in variants:
                rs = grid[task_id].get(f"{v}|{p}", [])
                cells.append(" ".join(f'<a class="{"ok" if x.grade.success else "bad"}" href="/runs/{esc(x.run_id)}">'
                                      f'{"✓" if x.grade.success else "✗"}</a>' for x in rs) or "<span class='mut'>·</span>")
            grid_rows.append(cells)
        cats = "".join(f"<h3>{esc(v)}</h3>" + markdown(category_markdown(records, v, "none"))
                       for v in dict.fromkeys(r.variant for r in records if r.plan == "none"))
        return page("Evaluation", f"<h1>Evaluation: {esc(name)}</h1><p>{picker}</p><h2>Summary</h2>"
                    + table(["variant", "faults", "n", "success (95% CI)", "fact recall", "faithfulness", "tool selection",
                             "arg validity", "recovery", "unnecessary LLM calls", "LLM calls", "tokens", "latency s (mean / p95)",
                             "reproducibility"], summary_rows)
                    + "<h2>By category (no faults)</h2>" + cats
                    + "<h2>Per task</h2><p class='mut'>✓ success · ✗ failure · click to open the run</p>"
                    + table(["task"] + [f"{esc(v)}<br><span class='mut'>{esc(p)}</span>" for v, p in variants], grid_rows))

    @app.get("/security", response_class=HTMLResponse)
    def security_view():
        parts = ["<h1>Security</h1>"]
        matrix = results_dir / "security_matrix.md"
        if matrix.exists():
            parts.append(markdown(matrix.read_text(encoding="utf-8")))
        for fname, title in [("security_live.json", "Live attacks: ARES (defended vs undefended)"),
                             ("security_live_baseline.json", "Live attacks: CrewAI baseline")]:
            f = results_dir / fname
            if not f.exists():
                continue
            rows = json.loads(f.read_text(encoding="utf-8"))
            yn = lambda b: "<span class='bad'>YES</span>" if b else "no"  # noqa: E731
            parts.append(f"<h2>{title}</h2>" + table(
                ["attack", "profile", "delivered", "injection seen", "attempted", "succeeded", "blocked", "detected", "status", "run"],
                [[f"{esc(r['attack'])} {esc(r['name'])}", esc(r["profile"]), "yes" if r.get("delivered") else "no",
                  "yes" if r.get("injection_seen") else "no", "yes" if r["attempted"] else "no", yn(r["succeeded"]),
                  str(r["blocked"]), str(r["injections_detected"]), badge(r["status"]),
                  f'<a href="/runs/{esc(r["run_id"])}">{esc(r["run_id"])}</a>'] for r in rows]))
        return page("Security", "".join(parts))

    @app.get("/docs", response_class=HTMLResponse)
    def docs_index():
        files = sorted(p.relative_to(docs_dir).as_posix() for p in docs_dir.rglob("*.md"))
        return page("Docs", "<h1>Results and docs</h1><ul>" + "".join(
            f'<li><a href="/docs/{esc(f)}">{esc(f)}</a></li>' for f in files) + "</ul>")

    @app.get("/docs/{path:path}", response_class=HTMLResponse)
    def doc(path: str):
        target = (docs_dir / path).resolve()
        if docs_dir.resolve() not in target.parents or not target.exists() or target.suffix != ".md":
            raise HTTPException(404, "no such doc")
        return page(path, markdown(target.read_text(encoding="utf-8")))

    @app.get("/api/runs")
    def api_runs(limit: int = Query(100, le=2000)):
        return [r.model_dump() for r in store.list_runs(limit)]

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: str):
        r = store.get_run(run_id)
        if r is None:
            raise HTTPException(404, "no such run")
        return {"run": r.model_dump(), "events": [e.model_dump() for e in store.events(r.run_id)]}

    return app
