"""The state machine on the happy path, its invariants, and its budgets."""
from __future__ import annotations

import pytest

from ares.llm.scripted import call, calls
from ares.runtime import TRANSITIONS, Phase
from ares.runtime.state import IllegalTransition
from ares.tracing import EventKind
from tests.conftest import SSB, SSB_CHALLENGES, assert_transcript_valid, report


def test_happy_path(harness, task):
    agent = harness.agent([
        call("search_web", query="solid-state battery market"),
        call("fetch_page", url=SSB),
        call("submit_report", **report(SSB)),
    ])
    res = agent.run(task)

    assert res.status == Phase.DONE
    assert res.final["findings"][0]["source_url"] == SSB
    assert res.fetched_urls == [SSB]
    assert [t.name for t in res.trajectory] == ["search_web", "fetch_page", "submit_report"]
    assert res.counters.llm_calls == 3
    assert res.counters.tool_executions == 2
    assert res.counters.errors == {}
    assert_transcript_valid(res.messages)

    path = [e.data["dst"] for e in harness.tracer.of_kind(EventKind.TRANSITION)]
    assert path == ["validate", "authorize", "execute", "observe", "decide",
                    "validate", "authorize", "execute", "observe", "decide",
                    "validate", "finalize", "done"]


def test_parallel_calls_in_one_turn(harness, task):
    agent = harness.agent([
        calls(("fetch_page", {"url": SSB}), ("fetch_page", {"url": SSB_CHALLENGES})),
        call("submit_report", **report(SSB, SSB_CHALLENGES)),
    ])
    res = agent.run(task)
    assert res.status == Phase.DONE
    assert res.fetched_urls == [SSB, SSB_CHALLENGES]
    assert res.counters.llm_calls == 2
    assert_transcript_valid(res.messages)


def test_calls_after_submit_are_skipped_but_answered(harness, task):
    agent = harness.agent([
        call("fetch_page", url=SSB),
        calls(("submit_report", report(SSB)), ("search_web", {"query": "more"})),
    ])
    res = agent.run(task)
    assert res.status == Phase.DONE
    assert res.counters.tool_executions == 1
    assert_transcript_valid(res.messages)


def test_every_traced_transition_is_legal(harness, task):
    agent = harness.agent([
        call("search_web", query="x"),  # invalid: too short -> repair
        call("search_web", query="solid-state"),
        call("fetch_page", url=SSB),
        call("submit_report", **report(SSB)),
    ])
    agent.run(task)
    for e in harness.tracer.of_kind(EventKind.TRANSITION):
        assert Phase(e.data["dst"]) in TRANSITIONS[Phase(e.data["src"])]


def test_illegal_transition_is_rejected(harness, task):
    agent = harness.agent([call("fetch_page", url=SSB)])
    from ares.runtime.agent import _Run

    run = _Run(agent, task, "run-x")
    with pytest.raises(IllegalTransition):
        run.goto(Phase.DONE)  # DECIDE -> DONE skips validation and finalization


def test_llm_call_budget_stops_a_looping_agent(harness, task):
    agent = harness.agent([call("search_web", query="solid-state battery")], repeat_last=True, max_llm_calls=4)
    res = agent.run(task)
    assert res.status == Phase.FAILED
    assert "LLM call budget" in res.reason
    assert res.counters.llm_calls == 4
    assert_transcript_valid(res.messages)


def test_trace_contains_replay_material(harness, task):
    agent = harness.agent([call("fetch_page", url=SSB), call("submit_report", **report(SSB))])
    res = agent.run(task)
    llm_events = harness.tracer.of_kind(EventKind.LLM_CALL)
    assert len(llm_events) == 2
    assert all(e.data["fingerprint"] and e.data["response"]["tool_calls"] for e in llm_events)
    # new_messages across calls reconstruct everything the model was sent
    sent = [m for e in llm_events for m in e.data["new_messages"]]
    assert sent == res.messages[: len(sent)]
    run = harness.tracer.runs[res.run_id]
    assert run.status == "done" and run.final == res.final and run.counters["llm_calls"] == 2


def test_loop_guard_flags_repeated_reads(harness, task):
    agent = harness.agent([call("fetch_page", url=SSB), call("fetch_page", url=SSB), call("submit_report", **report(SSB))])
    res = agent.run(task)
    assert res.status == Phase.DONE
    assert res.counters.repeated_calls == 1
    assert [t.outcome for t in res.trajectory] == ["ok", "repeated", "submitted"]
    assert "already made this exact call" in harness.llm.requests[2].messages[-1]["content"]


def test_wrapup_nudge_before_budget_runs_out(harness, task):
    agent = harness.agent([call("search_web", query="solid-state battery")] * 3 + [call("fetch_page", url=SSB),
                          call("submit_report", **report(SSB))], max_llm_calls=5, wrapup_at_remaining=1)
    res = agent.run(task)
    assert res.status == Phase.DONE
    last_request = harness.llm.requests[-1].messages
    assert last_request[-1]["role"] == "user" and "1 steps left" in last_request[-1]["content"]
    assert len(harness.tracer.of_kind(EventKind.WRAPUP)) == 1
