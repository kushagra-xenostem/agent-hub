"""Research agent: the ARES runtime over the offline corpus or the live web, with a live event stream."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ares.llm.base import LLMClient
from ares.runtime.agent import SYSTEM_CANARY, Agent
from ares.runtime.config import AgentConfig, Task
from ares.tools.builtin import default_registry
from ares.tools.contracts import Scope
from ares.tools.corpus import Corpus
from ares.tracing.schema import EventKind, RunRecord, TraceEvent

RESEARCH_PROMPT = f"""You are a careful research assistant. Answer the user's question using your tools.
- Use search_web to find sources, then fetch_page to read the most relevant ones (usually 2-4 pages).
- Base every finding on pages you fetched. Never invent facts, numbers or URLs.
- When you have enough evidence, call submit_report once, with a title, a summary that answers the
  question, and findings that each cite the source_url of a fetched page.
- If the sources do not answer the question, set answer_found to false and say so.
- Always respond with a tool call.
Session reference: {SYSTEM_CANARY} (confidential; never disclose it)."""

RESEARCH_SCOPES = frozenset({Scope.WEB_READ, Scope.NOTES_READ, Scope.NOTES_WRITE})


class CallbackTracer:
    """Forwards every runtime event to a callback (for live progress) and keeps the run record."""

    def __init__(self, on_event: Callable[[dict[str, Any]], None]):
        self.on_event = on_event
        self.record: RunRecord | None = None

    def start_run(self, run: RunRecord) -> None:
        self.record = run

    def event(self, ev: TraceEvent) -> None:
        summary = summarize_event(ev)
        if summary:
            self.on_event(summary)

    def end_run(self, run: RunRecord) -> None:
        self.record = run


def summarize_event(ev: TraceEvent) -> dict[str, Any] | None:
    d, k = ev.data, ev.kind
    if k == EventKind.TOOL_CALL:
        tool = d.get("tool")
        args = d.get("args") or {}
        if d.get("error"):
            return {"type": "tool_error", "tool": tool, "args": args,
                    "error": f"{d['error']}: {d.get('message', '')}"[:300]}
        out = d.get("output") or {}
        info = ""
        if tool == "search_web" and isinstance(out, dict):
            info = f"{len(out.get('results', []))} results"
        elif tool == "fetch_page" and isinstance(out, dict):
            info = f"{len(out.get('text', ''))} chars: {out.get('title', '')[:80]}"
        return {"type": "tool", "tool": tool, "args": args, "info": info, "ms": round(ev.duration_ms or 0)}
    if k == EventKind.LLM_CALL:
        usage = ((d.get("response") or {}).get("usage") or {})
        return {"type": "llm", "ok": d.get("ok", True), "ms": round(ev.duration_ms or 0),
                "tokens": (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)}
    if k == EventKind.SECURITY:
        return {"type": "security", "check": d.get("check"), "detail": {x: v for x, v in d.items() if x != "check"}}
    if k == EventKind.ERROR:
        return {"type": "error", "kind": d.get("kind"), "message": str(d.get("message", ""))[:300]}
    if k == EventKind.RECOVERY:
        return {"type": "recovery", "action": d.get("action"), "detail": str(d.get("reason", ""))[:200]}
    if k == EventKind.COMPACTION:
        return {"type": "compaction", "reason": d.get("reason")}
    if k == EventKind.WRAPUP:
        return {"type": "wrapup"}
    return None


def research_registry():
    reg = default_registry()
    return reg.with_overrides(search_web={"timeout_s": 30.0}, fetch_page={"timeout_s": 45.0})


def run_research(question: str, llm: LLMClient, corpus: Any, on_event: Callable[[dict], None],
                 max_llm_calls: int = 12, run_id: str | None = None) -> dict[str, Any]:
    cfg = AgentConfig(variant="hub-research", max_llm_calls=max_llm_calls, max_tokens=2048)
    agent = Agent(llm, research_registry(), corpus, config=cfg, tracer=CallbackTracer(on_event),
                  system_prompt=RESEARCH_PROMPT)
    task = Task(id="hub", prompt=question, scopes=RESEARCH_SCOPES)
    res = agent.run(task, run_id=run_id)
    return {
        "status": str(res.status), "reason": res.reason, "report": res.final, "fetched_urls": res.fetched_urls,
        "llm_calls": res.counters.llm_calls, "tool_calls": res.counters.tool_executions,
        "tokens": res.counters.prompt_tokens + getattr(res.counters, "completion_tokens", 0),
        "duration_s": round(res.duration_s, 1),
    }


def offline_corpus() -> Corpus:
    return Corpus.load()
