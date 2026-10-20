"""Failure-injection harness: each fault kind, determinism, tracing, and the agent's response to it."""
from __future__ import annotations

import pytest

from ares.errors import ErrorKind
from ares.eval.experiments import SIM_TASKS, build_agent, grid, score
from ares.faults import PRESETS, FaultInjector, FaultKind, FaultPlan, FaultSpec
from ares.llm.scripted import call
from ares.llm.simulated import SimulatedResearcher
from ares.runtime import AgentConfig, Phase, Strategy, Task, approve_all
from ares.tools import Corpus, Scope
from ares.tracing import EventKind, MemoryTracer
from tests.conftest import SSB, assert_transcript_valid, report


def plan(*faults: FaultSpec) -> FaultPlan:
    return FaultPlan(name="test", faults=list(faults))


@pytest.fixture(scope="module")
def real_corpus() -> Corpus:
    return Corpus.load()


def run_scripted(harness, task, steps, fault_plan, seed=0, **cfg):
    harness.llm = None
    from ares.llm.scripted import ScriptedLLM
    llm = ScriptedLLM(steps)
    harness.llm = llm
    agent = build_agent(llm, harness.corpus, AgentConfig(**cfg), plan=fault_plan, seed=seed,
                        tracer=harness.tracer, sleep=harness.sleeps.append)
    return agent.run(task)


# --- determinism ------------------------------------------------------------------------------
def test_same_seed_same_injections():
    p = plan(FaultSpec(kind=FaultKind.TOOL_ERROR, target="fetch_page", rate=0.5))
    a, b, c = FaultInjector(p, seed=1), FaultInjector(p, seed=1), FaultInjector(p, seed=2)
    pick = lambda inj: [inj.decide("fetch_page") is not None for _ in range(40)]
    assert pick(a) == pick(b)
    assert pick(FaultInjector(p, seed=1)) != pick(c)


def test_rate_is_respected_statistically():
    inj = FaultInjector(plan(FaultSpec(kind=FaultKind.TOOL_ERROR, rate=0.3)), seed=0)
    hits = sum(inj.decide("search_web") is not None for _ in range(2000))
    assert 0.26 < hits / 2000 < 0.34


def test_on_calls_and_max_injections():
    inj = FaultInjector(plan(FaultSpec(kind=FaultKind.TOOL_ERROR, target="fetch_page", on_calls=[2, 3],
                                       max_injections=1)), seed=0)
    assert [inj.decide("fetch_page") is not None for _ in range(4)] == [False, True, False, False]
    assert inj.decide("search_web") is None  # other targets untouched


# --- each fault kind reaches the agent as the right classified error ----------------------------
@pytest.mark.parametrize("spec,expected", [
    (FaultSpec(kind=FaultKind.TOOL_ERROR, target="fetch_page", on_calls=[1]), "tool_error"),
    (FaultSpec(kind=FaultKind.TOOL_TIMEOUT, target="fetch_page", on_calls=[1]), "tool_timeout"),
    (FaultSpec(kind=FaultKind.TOOL_MALFORMED, target="fetch_page", on_calls=[1]), "tool_output_invalid"),
    (FaultSpec(kind=FaultKind.LLM_RATE_LIMIT, on_calls=[1]), "llm_rate_limited"),
    (FaultSpec(kind=FaultKind.LLM_UNAVAILABLE, on_calls=[1]), "llm_unavailable"),
    (FaultSpec(kind=FaultKind.LLM_TIMEOUT, on_calls=[1]), "llm_timeout"),
])
def test_fault_is_classified_and_recovered(harness, task, spec, expected):
    res = run_scripted(harness, task, [call("fetch_page", url=SSB), call("submit_report", **report(SSB))], plan(spec))
    assert res.status == Phase.DONE
    assert res.counters.errors == {expected: 1}
    assert res.counters.injected_faults == 1 and res.counters.recoveries == 1
    fault_events = harness.tracer.of_kind(EventKind.FAULT)
    assert [e.data["kind"] for e in fault_events] == [spec.kind.value]
    assert_transcript_valid(res.messages)


def test_rate_limit_backoff_honors_injected_retry_after(harness, task):
    spec = FaultSpec(kind=FaultKind.LLM_RATE_LIMIT, on_calls=[1, 2], retry_after_s=3.0)
    res = run_scripted(harness, task, [call("fetch_page", url=SSB), call("submit_report", **report(SSB))], plan(spec))
    assert res.status == Phase.DONE and harness.sleeps == [3.0, 3.0]


def test_garbled_arguments_are_repaired(harness, task):
    spec = FaultSpec(kind=FaultKind.LLM_GARBLED, on_calls=[1])
    res = run_scripted(harness, task, [call("fetch_page", url=SSB), call("fetch_page", url=SSB),
                                       call("submit_report", **report(SSB))], plan(spec))
    assert res.status == Phase.DONE and res.counters.errors == {"tool_args_invalid": 1}
    assert_transcript_valid(res.messages)


def test_no_tool_reply_is_nudged(harness, task):
    spec = FaultSpec(kind=FaultKind.LLM_NO_TOOL, on_calls=[1])
    res = run_scripted(harness, task, [call("fetch_page", url=SSB), call("fetch_page", url=SSB),
                                       call("submit_report", **report(SSB))], plan(spec))
    assert res.status == Phase.DONE and res.counters.errors == {"llm_no_action": 1}


def test_permanent_tool_error_is_not_retried_by_backoff(harness, task):
    spec = FaultSpec(kind=FaultKind.TOOL_ERROR, target="fetch_page", transient=False)
    res = run_scripted(harness, task, [call("fetch_page", url=SSB)], plan(spec), recovery=Strategy.BACKOFF_RETRY)
    assert res.status == Phase.FAILED and "not transient" in res.reason


def test_timeout_after_side_effect_does_not_duplicate_it(harness):
    """The email was sent but the reply was lost; the retry must not send it again."""
    task = Task(id="mail", prompt="Email it.", scopes=frozenset({Scope.WEB_READ, Scope.EXTERNAL_SEND}),
                allowed_recipients=frozenset({"company.example"}))
    spec = FaultSpec(kind=FaultKind.TOOL_TIMEOUT, target="send_email", on_calls=[1], after_effect=True)
    email = {"to": "team@company.example", "subject": "report", "body": "summary"}
    from ares.llm.scripted import ScriptedLLM
    llm = ScriptedLLM([call("send_email", **email), call("fetch_page", url=SSB), call("submit_report", **report(SSB))])
    agent = build_agent(llm, harness.corpus, AgentConfig(), plan=plan(spec), approver=approve_all,
                        sleep=harness.sleeps.append)
    res = agent.run(task)
    assert res.status == Phase.DONE
    assert len(res.outbox) == 1  # sent once, despite the timeout and the retry
    assert res.counters.errors == {"tool_timeout": 1} and res.counters.retries == 1


def test_empty_and_conflicting_outputs_are_valid_but_wrong(real_corpus):
    inj = FaultInjector(plan(FaultSpec(kind=FaultKind.TOOL_CONFLICT, target="fetch_page")), seed=0)
    reg = inj.wrap_registry(__import__("ares.tools", fromlist=["default_registry"]).default_registry())
    from ares.tools import ToolContext
    from ares.tools.builtin import FetchInput, FetchOutput
    url = "https://marketwatch.example/solid-state-battery-outlook-2026"
    out = reg.get("fetch_page").handler(FetchInput(url=url), ToolContext(run_id="r", corpus=real_corpus))
    FetchOutput.model_validate(out)  # passes the contract...
    assert "1.9 billion" not in out["text"] and "2025" in out["text"]  # ...but the figures are wrong; years kept


def test_oversized_output_is_capped_by_observation_limit(harness, task):
    spec = FaultSpec(kind=FaultKind.TOOL_OVERSIZED, target="fetch_page")
    res = run_scripted(harness, task, [call("fetch_page", url=SSB), call("submit_report", **report(SSB))], plan(spec))
    assert res.status == Phase.DONE
    obs = next(m for m in res.messages if m["role"] == "tool")
    assert len(obs["content"]) < 4200 and "truncated" in obs["content"]


def test_context_limit_triggers_overflow_and_compaction(harness, task):
    spec = FaultSpec(kind=FaultKind.LLM_CONTEXT_LIMIT, limit_tokens=1200)
    res = run_scripted(harness, task, [call("fetch_page", url=SSB), call("search_web", query="solid-state battery"),
                                       call("search_web", query="battery cost"), call("submit_report", **report(SSB))],
                       plan(spec), compact_at=None, keep_recent_observations=1)
    assert ErrorKind.LLM_CONTEXT_OVERFLOW.value in res.counters.errors
    assert res.counters.compactions >= 1


def test_all_presets_are_valid():
    assert {"none", "tool_errors", "tool_timeouts", "malformed_outputs", "conflicts", "context_limit",
            "llm_rate_limits", "mixed"} <= set(PRESETS)


# --- the simulated researcher -------------------------------------------------------------------
def test_simulated_researcher_completes_cleanly(real_corpus):
    for t in SIM_TASKS:
        s = score(build_agent(SimulatedResearcher(seed=0), real_corpus, AgentConfig()), t, real_corpus)
        assert s.result.status == Phase.DONE and s.correct, (t.id, s.result.reason)
        assert s.result.counters.errors == {}


def test_simulated_runs_are_reproducible(real_corpus):
    def once():
        a = build_agent(SimulatedResearcher(seed=3, p_bad_args=0.3), real_corpus, AgentConfig(seed=3),
                        plan=PRESETS["mixed"], seed=3, sleep=lambda s: None)
        r = a.run(SIM_TASKS[0], run_id="fixed")
        return r.status, r.counters.model_dump(exclude={"llm_latency_s", "tool_latency_s"}), r.final
    assert once() == once()


def test_conflicting_pages_make_reports_unfaithful(real_corpus):
    cells = grid([PRESETS["none"], PRESETS["conflicts"]], {"ares": AgentConfig()}, range(3),
                 model_factory=lambda s: SimulatedResearcher(seed=s), corpus=real_corpus)
    clean, conflict = cells[0].summary, cells[1].summary
    assert clean.correct_rate == 1.0
    assert conflict.success_rate == 1.0 and conflict.correct_rate < 1.0  # finished, but wrong: a silent failure
