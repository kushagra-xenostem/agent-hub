"""The evaluation dataset and its deterministic task graders."""
from __future__ import annotations
import re
import json
from pathlib import Path
from ares.config import PROJECT_ROOT
from ares.runtime.config import Task
from ares.runtime.state import Phase
from pydantic import BaseModel, Field
from ares.runtime.agent import RunResult
from ares.eval.graders import faithfulness
from ares.tools.corpus import Corpus, normalize_url
from ares.grounding import unsupported as unsupported_figures

DEFAULT_TASKS = PROJECT_ROOT / "data" / "eval" / "tasks.jsonl"


class Fact(BaseModel):
    id: str
    any_of: list[str]


class EvalTask(BaseModel):
    id: str
    category: str  # overview | lookup | comparison | conflict | unanswerable
    topic: str
    prompt: str
    facts: list[Fact] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)  # pages that hold the required facts
    relevant: list[str] = Field(default_factory=list)  # pages relevant to the topic at all
    min_llm_calls: int = 3  # shortest path: search, fetch (parallel), submit

    def as_task(self) -> Task:
        return Task(id=self.id, prompt=self.prompt)


def load_tasks(path: Path = DEFAULT_TASKS) -> list[EvalTask]:
    return [EvalTask.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


_UNAVAILABLE = re.compile(
    r"\b(not (?:available|found|disclosed|reported|provided|published|mentioned|included|contain\w*|in the sources)|"
    r"no (?:\w+ ){0,3}(?:data|information|figures?|sources?|mention|record|details)|"
    r"(?:could|can)(?:no|')?t (?:find|locate|determine|be found)|unable to (?:find|determine|locate)|"
    r"does not (?:say|state|mention|include|provide|report)|do not (?:say|state|mention|include|provide|report)|"
    r"isn't (?:available|disclosed)|unavailable|unknown)\b", re.IGNORECASE)


def report_text(final: dict | None) -> str:
    if not final:
        return ""
    parts = [str(final.get("title", "")), str(final.get("summary", ""))]
    parts += [str(f.get("claim", "")) for f in final.get("findings") or [] if isinstance(f, dict)]
    return " ".join(parts).lower().replace("$", "")


class Grade(BaseModel):
    task_id: str
    category: str
    status: str
    success: bool
    fact_recall: float | None  # share of required facts present (None for unanswerable)
    source_recall: float | None  # share of required source pages cited
    faithfulness: float  # share of findings whose figures appear in the true cited page
    unsupported_numbers: list[str] = []  # figures in the report found in no fetched page and not in the question
    admits_unavailable: bool | None  # unanswerable tasks: said so instead of inventing an answer
    tool_selection_accuracy: float | None  # proposed calls that were appropriate
    unnecessary_llm_calls: int
    unnecessary_tool_runs: int
    llm_calls: int
    tool_runs: int
    tokens: int
    latency_s: float
    arg_validity: float | None
    errors: dict[str, int]


def grade(task: EvalTask, res: RunResult, corpus: Corpus) -> Grade:
    text = report_text(res.final)
    done = res.status == Phase.DONE
    facts_hit = [any(alt.lower() in text for alt in f.any_of) for f in task.facts]
    fact_recall = sum(facts_hit) / len(facts_hit) if facts_hit else None
    cited = {normalize_url(str(f.get("source_url", ""))) for f in (res.final or {}).get("findings") or []
             if isinstance(f, dict)}
    source_recall = (sum(normalize_url(u) in cited for u in task.sources) / len(task.sources)) if task.sources else None
    faith = faithfulness(res.final, corpus) if res.final else 0.0
    admits = bool(_UNAVAILABLE.search(text)) if task.category == "unanswerable" else None

    # what the model read: every tool result it was shown, the full pages it fetched, and the question
    shown = " ".join(str(m.get("content", "")) for m in res.messages if m.get("role") == "tool")
    pages = " ".join(f"{d.url} {d.title} {d.text}" for d in map(corpus.get, res.fetched_urls) if d)
    unsupported = unsupported_figures(text, f"{shown} {pages} {task.prompt}".lower().replace("$", ""))

    # Success is defined the same way for every system (incl. the CrewAI baseline, which has no structured
    # citations): finished, all required facts present, and no figure that appears in none of the pages read.
    if task.category == "unanswerable":
        success = done and bool(admits) and not unsupported
    else:
        success = done and fact_recall == 1.0 and not unsupported

    proposed = [(tc["function"]["name"], tc["function"]["arguments"])
                for m in res.messages if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
    relevant = {normalize_url(u) for u in task.relevant}
    ok_calls = 0
    for name, args in proposed:
        if name == "fetch_page":
            try:
                url = normalize_url(str(json.loads(args).get("url", "")))
            except (ValueError, AttributeError):
                url = ""
            ok_calls += url in relevant
        elif name in ("search_web", "submit_report", "save_note", "read_notes"):
            ok_calls += 1
    c = res.counters
    min_tools = 1 + max(1, len(task.sources))
    return Grade(
        task_id=task.id, category=task.category, status=res.status.value, success=success,
        fact_recall=fact_recall, source_recall=source_recall, faithfulness=faith, admits_unavailable=admits,
        unsupported_numbers=unsupported,
        tool_selection_accuracy=ok_calls / len(proposed) if proposed else None,
        unnecessary_llm_calls=max(0, c.llm_calls - task.min_llm_calls),
        unnecessary_tool_runs=max(0, c.tool_executions - min_tools),
        llm_calls=c.llm_calls, tool_runs=c.tool_executions, tokens=c.total_tokens,
        latency_s=round(c.llm_latency_s + c.tool_latency_s + c.backoff_s, 2),
        arg_validity=1 - c.invalid_calls / c.proposed_calls if c.proposed_calls else None,
        errors=c.errors,
    )
