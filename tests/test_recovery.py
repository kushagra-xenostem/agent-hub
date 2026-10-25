"""Recovery strategies against classified errors."""
from __future__ import annotations
import time
from ares.tools import ToolFailure
from ares.runtime import Phase, Strategy
from ares.errors import AgentError, ErrorKind
from ares.llm.scripted import call, raw_call, reply
from tests.conftest import SSB, assert_transcript_valid, report


def rate_limited(retry_after=4.0):
    return AgentError(ErrorKind.LLM_RATE_LIMITED, "rate limited", retry_after_s=retry_after)


def test_repair_fixes_invalid_arguments(harness, task):
    agent = harness.agent([
        call("search_web", query="solid-state", max_results=50),  # violates le=10
        call("search_web", query="solid-state", max_results=5),
        call("fetch_page", url=SSB),
        call("submit_report", **report(SSB)),
    ])
    res = agent.run(task)
    assert res.status == Phase.DONE
    assert res.counters.errors == {"tool_args_invalid": 1}
    assert res.counters.invalid_calls == 1 and res.counters.repairs == 1 and res.counters.recoveries == 1
    # the model saw the contract violation
    feedback = harness.llm.requests[1].messages[-1]
    assert feedback["role"] == "tool" and "max_results" in feedback["content"]
    assert_transcript_valid(res.messages)


def test_malformed_json_arguments(harness, task):
    agent = harness.agent([
        raw_call("fetch_page", '{"url": "https://test.example/ssb-market"'),  # truncated JSON
        call("fetch_page", url=SSB),
        call("submit_report", **report(SSB)),
    ])
    res = agent.run(task)
    assert res.status == Phase.DONE
    assert res.counters.errors == {"tool_args_invalid": 1}


def test_unknown_tool_is_repaired(harness, task):
    agent = harness.agent([call("browse", url=SSB), call("fetch_page", url=SSB), call("submit_report", **report(SSB))])
    res = agent.run(task)
    assert res.status == Phase.DONE
    assert res.counters.errors == {"tool_unknown": 1}


def test_fail_fast_stops_on_first_error(harness, task):
    agent = harness.agent([call("search_web", query="x")], recovery=Strategy.FAIL_FAST)
    res = agent.run(task)
    assert res.status == Phase.FAILED
    assert res.reason.startswith("tool_args_invalid")
    assert_transcript_valid(res.messages)


def test_naive_retry_cannot_fix_a_deterministic_error(harness, task):
    # retrying the same invalid call reproduces the same error until the retry bound
    agent = harness.agent([call("search_web", query="x")], recovery=Strategy.NAIVE_RETRY, max_retries=3)
    res = agent.run(task)
    assert res.status == Phase.FAILED
    assert res.counters.errors == {"tool_args_invalid": 4}
    assert res.counters.retries == 3 and res.counters.llm_calls == 1


def test_backoff_honors_retry_after(harness, task):
    agent = harness.agent([rate_limited(4.0), rate_limited(4.0), call("fetch_page", url=SSB),
                           call("submit_report", **report(SSB))],
                          recovery=Strategy.BACKOFF_RETRY, backoff_base_s=1.0)
    res = agent.run(task)
    assert res.status == Phase.DONE
    assert harness.sleeps == [4.0, 4.0]  # Retry-After is a floor over 1s, 2s
    assert res.counters.retries == 2 and res.counters.recoveries == 1


def test_backoff_is_exponential_and_bounded(harness, task):
    agent = harness.agent([AgentError(ErrorKind.LLM_UNAVAILABLE, "503")] * 4,
                          recovery=Strategy.BACKOFF_RETRY, backoff_base_s=1.0, max_retries=3)
    res = agent.run(task)
    assert res.status == Phase.FAILED
    assert harness.sleeps == [1.0, 2.0, 4.0]
    assert "gave up after 3" in res.reason


def test_backoff_does_not_retry_non_transient(harness, task):
    agent = harness.agent([call("search_web", query="x")], recovery=Strategy.BACKOFF_RETRY)
    res = agent.run(task)
    assert res.status == Phase.FAILED and "not transient" in res.reason


def test_auth_errors_are_never_retried(harness, task):
    agent = harness.agent([AgentError(ErrorKind.LLM_AUTH, "401")], recovery=Strategy.NAIVE_RETRY)
    res = agent.run(task)
    assert res.status == Phase.FAILED and res.counters.retries == 0


def test_text_reply_is_nudged_back_to_tools(harness, task):
    agent = harness.agent([reply("The market is big."), call("fetch_page", url=SSB), call("submit_report", **report(SSB))])
    res = agent.run(task)
    assert res.status == Phase.DONE
    nudge = harness.llm.requests[1].messages
    assert nudge[-2] == {"role": "assistant", "content": "The market is big."}
    assert nudge[-1]["role"] == "user" and "tool call" in nudge[-1]["content"]


def test_transient_tool_error_is_retried(harness, task):
    flaky = {"n": 0}

    def fetch(args, ctx):
        flaky["n"] += 1
        if flaky["n"] == 1:
            raise ToolFailure("503 upstream", transient=True)
        doc = ctx.corpus.get(args.url)
        return {"url": doc.url, "title": doc.title, "text": doc.text}

    reg = harness.registry.with_overrides(fetch_page={"handler": fetch})
    agent = harness.agent([call("fetch_page", url=SSB), call("submit_report", **report(SSB))], registry=reg)
    res = agent.run(task)
    assert res.status == Phase.DONE
    assert res.counters.retries == 1 and res.counters.llm_calls == 2  # retried the tool, not the model


def test_tool_timeout(harness, task):
    def slow(args, ctx):
        time.sleep(0.5)

    reg = harness.registry.with_overrides(search_web={"handler": slow, "timeout_s": 0.05})
    agent = harness.agent([call("search_web", query="solid-state")], registry=reg, recovery=Strategy.FAIL_FAST)
    res = agent.run(task)
    assert res.status == Phase.FAILED and res.reason.startswith("tool_timeout")


def test_malformed_tool_output_is_caught(harness, task):
    reg = harness.registry.with_overrides(fetch_page={"handler": lambda a, c: {"url": SSB, "body": "oops"}})
    agent = harness.agent([call("fetch_page", url=SSB)], registry=reg, recovery=Strategy.FAIL_FAST)
    res = agent.run(task)
    assert res.status == Phase.FAILED and res.reason.startswith("tool_output_invalid")


def test_malformed_tool_output_passes_through_without_output_validation(harness, task):
    reg = harness.registry.with_overrides(fetch_page={"handler": lambda a, c: {"url": SSB, "body": "oops"}})
    agent = harness.agent([call("fetch_page", url=SSB), call("submit_report", **report(SSB))], registry=reg,
                          validate_output=False)
    res = agent.run(task)
    assert res.status == Phase.DONE  # nothing noticed: this is what the ablation measures


def long_pages(harness):
    """fetch_page returning realistically large pages (the test corpus pages are tiny)."""
    def fetch(args, ctx):
        return {"url": args.url, "title": "Long page", "text": "Solid-state battery analysis. " * 120}
    return harness.registry.with_overrides(fetch_page={"handler": fetch})


def test_context_overflow_compacts_and_retries(harness, task):
    agent = harness.agent([
        call("fetch_page", url="https://test.example/a"),
        call("fetch_page", url="https://test.example/b"),
        call("fetch_page", url=SSB),
        AgentError(ErrorKind.LLM_CONTEXT_OVERFLOW, "too long"),
        call("submit_report", **report(SSB)),
    ], registry=long_pages(harness), compact_at=None)  # reactive path only
    res = agent.run(task)
    assert res.status == Phase.DONE
    assert res.counters.compactions == 1
    tool_msgs = [m for m in harness.llm.requests[-1].messages if m["role"] == "tool"]
    assert '"compacted": true' in tool_msgs[0]["content"]  # oldest shrunk
    assert '"compacted"' not in tool_msgs[-1]["content"]  # newest kept


def test_proactive_compaction_before_the_window_fills(harness, task):
    agent = harness.agent([
        call("fetch_page", url="https://test.example/a"),
        call("fetch_page", url="https://test.example/b"),
        call("fetch_page", url=SSB),
        call("submit_report", **report(SSB)),
    ], registry=long_pages(harness), context_window=4000, compact_at=0.75, max_tokens=200)
    res = agent.run(task)
    assert res.status == Phase.DONE
    assert res.counters.compactions >= 1
    assert res.counters.errors == {}  # no overflow error was ever needed
    from ares.tracing import EventKind
    ev = harness.tracer.of_kind(EventKind.COMPACTION)[0]
    assert "estimated" in ev.data["reason"] and ev.data["removed_chars"] > 0


def test_regression_truncated_report_is_repaired_without_poisoning_history(harness, task):
    """First live run: the report hit max_tokens, its JSON was cut off, and sending that JSON back made
    the server reject every later request. The history must carry valid JSON and the model gets a hint."""
    from ares.llm.base import LLMResponse

    truncated = raw_call("submit_report", '{"title": "SSB", "summary": "The market is growing, see https')
    truncated = LLMResponse(**{**truncated.model_dump(), "finish_reason": "length"})
    agent = harness.agent([call("fetch_page", url=SSB), truncated, call("submit_report", **report(SSB))])
    res = agent.run(task)

    assert res.status == Phase.DONE
    retry_request = harness.llm.requests[2].messages
    sent_args = [tc["function"]["arguments"] for m in retry_request for tc in m.get("tool_calls", [])]
    assert all(__import__("json").loads(a) is not None for a in sent_args)  # every argument string parses
    assert "output limit" in retry_request[-1]["content"]
    assert_transcript_valid(res.messages)


def test_identical_repeated_failure_triggers_perturbed_sampling(harness, task):
    """Second live run: the model resent the same invalid report verbatim; greedy repair never escaped."""
    no_title = {k: v for k, v in report(SSB).items() if k != "title"}
    agent = harness.agent([call("fetch_page", url=SSB), call("submit_report", **no_title),
                           call("submit_report", **no_title), call("submit_report", **report(SSB))])
    res = agent.run(task)
    assert res.status == Phase.DONE
    temps = [r.temperature for r in harness.llm.requests]
    assert temps == [0.0, 0.0, 0.0, 0.7]  # only the call after the identical repeat is perturbed
    assert res.counters.perturbations == 1
    assert "Required fields: title, summary" in harness.llm.requests[2].messages[-1]["content"]


def test_perturbation_can_be_disabled(harness, task):
    no_title = {k: v for k, v in report(SSB).items() if k != "title"}
    agent = harness.agent([call("fetch_page", url=SSB)] + [call("submit_report", **no_title)] * 4,
                          repair_temperature=None)
    res = agent.run(task)
    assert res.status == Phase.FAILED
    assert {r.temperature for r in harness.llm.requests} == {0.0} and res.counters.perturbations == 0


def test_truncated_observation_stays_valid_json(harness, task):
    """Phase 2 finding: cutting serialized JSON gave the model a broken document."""
    import json
    from ares.runtime.agent import render_observation
    obs = {"url": SSB, "title": "t", "text": "x" * 10_000}
    out = render_observation(obs, 4000)
    parsed = json.loads(out)  # still valid JSON
    assert parsed["url"] == SSB and "truncated" in parsed["text"] and len(out) <= 4000 + 100
    assert render_observation({"a": "short"}, 4000) == '{"a": "short"}'


def test_arg_validity_is_measured_even_when_not_enforced(harness, task):
    agent = harness.agent([call("search_web", query="x"), call("fetch_page", url=SSB), call("submit_report", **report(SSB))],
                          validate_args=False)
    res = agent.run(task)
    assert res.counters.invalid_calls == 1  # measured...
    assert "tool_args_invalid" not in res.counters.errors  # ...but not enforced
