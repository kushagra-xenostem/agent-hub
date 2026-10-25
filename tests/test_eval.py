"""Evaluation dataset integrity, task graders, the runner's resume/aggregation, and the ablation ladder."""
from __future__ import annotations
import pytest
from ares.runtime import Agent, AgentConfig
from ares.llm.scripted import ScriptedLLM, call
from ares.tools import Corpus, default_registry
from ares.llm.simulated import SimulatedResearcher
from ares.eval.dataset import EvalTask, Fact, grade, load_tasks, report_text
from ares.eval.runner import VARIANTS, load_records, run_eval, summarize_eval


@pytest.fixture(scope="module")
def corpus():
    return Corpus.load()


@pytest.fixture(scope="module")
def tasks():
    return load_tasks()


def test_dataset_covers_every_category(tasks):
    cats = {t.category for t in tasks}
    assert cats == {"overview", "lookup", "comparison", "conflict", "unanswerable"}
    assert len(tasks) >= 25 and len({t.id for t in tasks}) == len(tasks)


def test_every_required_fact_is_in_its_sources(tasks, corpus):
    """A task must be answerable from the corpus, or the grader would punish a correct agent."""
    for t in tasks:
        text = " ".join(corpus.get(u).text.lower().replace("$", "") for u in t.sources)
        for f in t.facts:
            assert any(a.lower() in text for a in f.any_of), (t.id, f.id)
        for u in t.relevant + t.sources:
            assert corpus.get(u) is not None, (t.id, u)


def run(corpus, task, steps):
    agent = Agent(ScriptedLLM(steps), default_registry(), corpus, config=AgentConfig())
    return agent.run(task.as_task())


def test_grader_full_marks(corpus, tasks):
    t = next(x for x in tasks if x.id == "lk-yield")
    url = t.sources[0]
    res = run(corpus, t, [call("fetch_page", url=url), call("submit_report", title="Yield", summary="Average yield "
              "across the industry was 71% in 2025.", findings=[{"claim": "Average production yield was 71% in 2025.",
                                                                "source_url": url}])])
    g = grade(t, res, corpus)
    assert g.success and g.fact_recall == 1.0 and g.source_recall == 1.0 and g.faithfulness == 1.0
    assert g.tool_selection_accuracy == 1.0 and g.unnecessary_llm_calls == 0


def test_grader_penalizes_distractors_and_missing_facts(corpus, tasks):
    t = next(x for x in tasks if x.id == "lk-yield")
    gadget = "https://blog.example/top-10-gadgets-2026"
    res = run(corpus, t, [call("search_web", query="solid-state yield"), call("fetch_page", url=gadget),
                          call("fetch_page", url=t.sources[0]),
                          call("submit_report", title="Yield", summary="Manufacturing yield is improving steadily.",
                               findings=[{"claim": "Manufacturing yield is improving steadily over time.",
                                          "source_url": t.sources[0]}])])
    g = grade(t, res, corpus)
    assert not g.success and g.fact_recall == 0.0
    assert g.tool_selection_accuracy == pytest.approx(3 / 4)  # the gadget fetch was a wrong tool choice
    assert g.unnecessary_llm_calls == 1 and g.unnecessary_tool_runs == 1


def test_unanswerable_needs_an_admission(corpus, tasks):
    t = next(x for x in tasks if x.id == "un-quantor-2019")
    url = "https://techdaily.example/quantor-cells-leipzig-gigafactory"
    honest = run(corpus, t, [call("fetch_page", url=url), call("submit_report", title="Quantor 2019 revenue",
                 summary="Quantor Cells' 2019 revenue is not reported in the available sources.",
                 findings=[{"claim": "The sources describe its 2026 Leipzig plant, not 2019 revenue.", "source_url": url}])])
    assert grade(t, honest, corpus).success


def test_report_text_normalizes_currency():
    assert "1.9 billion" in report_text({"summary": "It was $1.9 billion."})


def test_runner_resumes_and_measures_reproducibility(tmp_path, corpus, tasks):
    out = tmp_path / "r.jsonl"
    few = tasks[:3]
    kw = dict(model_factory=lambda s: SimulatedResearcher(seed=s), corpus=corpus, results=out, sleep=lambda s: None)
    first = run_eval(few, ["L4_full"], ["none"], [0], repeats=2, **kw)
    assert len(first) == 6 and len(load_records(out)) == 6
    assert run_eval(few, ["L4_full"], ["none"], [0], repeats=2, **kw) == []  # resumed: nothing left to do
    [summary] = summarize_eval(load_records(out))
    assert summary.n == 6 and summary.reproducibility == 1.0  # deterministic model, temperature 0


def test_ablation_ladder_adds_one_family_at_a_time():
    l1, l2, l3, l4 = (VARIANTS[k] for k in ("L1_fsm", "L2_validation", "L3_recovery", "L4_full"))
    assert not l1.validate_args and l2.validate_args and l2.recovery == l1.recovery
    assert l3.recovery != l2.recovery and not l3.enforce_permissions
    assert l4.enforce_permissions and l4.injection_filter == "redact"
