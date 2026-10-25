"""One-line, human-readable descriptions of trace events (CLI timeline and live view)."""
from __future__ import annotations
import json
from rich.markup import escape
from rich.console import Console
from ares.tracing.schema import EventKind, RunRecord, TraceEvent


def _short(value, limit: int = 90) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def describe(ev: TraceEvent) -> tuple[str, str]:
    """(style, text) for one event."""
    d = ev.data
    match ev.kind:
        case EventKind.RUN_START:
            return "bold", f"start task={d['task']['id']} tools={','.join(d['offered_tools'])}"
        case EventKind.TRANSITION:
            return "dim", f"{d['src']} -> {d['dst']}"
        case EventKind.LLM_CALL:
            if not d["ok"]:
                return "red", f"LLM call failed: {d['error']}"
            r = d["response"]
            u = r["usage"]
            what = ", ".join(f"{tc['name']}({_short(tc['arguments'], 60)})" for tc in r["tool_calls"]) or \
                f"text: {_short(r.get('content') or '', 60)}"
            return "cyan", f"LLM {r['latency_s']:.1f}s in={u['prompt_tokens']} (cached {u['cached_tokens']}) " \
                           f"out={u['completion_tokens']} -> {what}"
        case EventKind.VALIDATION:
            return ("green" if d["ok"] else "yellow"), f"args {'valid' if d['ok'] else 'INVALID'} for {d['tool']}"
        case EventKind.PERMISSION:
            return ("dim" if d["allowed"] else "red"), f"permission {d['scope']} for {d['tool']}: " \
                                                       f"{'granted' if d['allowed'] else 'DENIED'}"
        case EventKind.APPROVAL:
            return "magenta", f"approval for {d['tool']}: {d['decision']}"
        case EventKind.TOOL_CALL:
            if not d["ok"]:
                return "red", f"tool {d['tool']} failed: {d['error']}: {_short(d['message'])}"
            dedup = " (deduplicated)" if d.get("deduplicated") else ""
            return "green", f"tool {d['tool']} {ev.duration_ms:.0f}ms{dedup} -> {_short(d['output'])}"
        case EventKind.ERROR:
            return "yellow", f"error {d['kind']}: {_short(d['message'], 120)}"
        case EventKind.RECOVERY:
            delay = f" after {d['delay_s']:g}s" if d["delay_s"] else ""
            return "magenta", f"recover #{d['attempt']} {d['signature']}: {d['action']}{delay} ({d['reason']})"
        case EventKind.COMPACTION:
            return "magenta", f"compacted {d['shrunk_observations']} old observations (-{d['removed_chars']} chars): {d['reason']}"
        case EventKind.SECURITY:
            rest = {k: v for k, v in d.items() if k != "check"}
            return "bold magenta", f"SECURITY {d['check']}: {_short(rest, 100)}"
        case EventKind.FAULT:
            return "bold red", f"INJECTED {d['kind']} into {d['target']} (call {d['call']}) {_short(d['detail'], 60)}"
        case EventKind.WRAPUP:
            return "magenta", f"wrap-up nudge: {d['remaining_llm_calls']} LLM calls left"
        case EventKind.RUN_END:
            style = "bold green" if d["status"] == "done" else "bold red"
            return style, f"end {d['status']} in {d['duration_s']:.1f}s: {d['reason']}"
    return "", ev.kind


class ConsoleTracer:
    """Prints events as they happen."""

    def __init__(self, console: Console | None = None, *, show_transitions: bool = False):
        self.console = console or Console()
        self.show_transitions = show_transitions

    def start_run(self, run: RunRecord) -> None:
        self.console.print(f"[bold]run {run.run_id}[/] variant={run.variant} model={run.model}")

    def event(self, ev: TraceEvent) -> None:
        if ev.kind == EventKind.TRANSITION and not self.show_transitions:
            return
        style, text = describe(ev)
        self.console.print(f"[dim]{ev.seq:>3} {ev.phase:<14}[/] [{style}]{escape(text)}[/]", highlight=False)

    def end_run(self, run: RunRecord) -> None:
        pass
