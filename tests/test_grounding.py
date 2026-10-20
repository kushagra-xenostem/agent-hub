"""Numeric grounding (shared by runtime and graders), URL provenance and summary-level claim checks."""
from __future__ import annotations

from ares.grounding import derivable, numbers, unsupported
from ares.llm.scripted import call
from ares.runtime import Phase
from tests.conftest import SSB, report

SAT = "Orbitlink leads with 68% of subscribers, Skyweave 14%. Plans cost $99 and $79. Orders: 2,100 and 1,350."


def test_numbers_are_normalized():
    assert numbers("$1,350 and 8.40 and 71%") == {"1350", "8.4", "71"}


def test_one_step_derivations_with_their_inputs_are_grounded():
    assert unsupported("Orbitlink ($99) costs $20 more than Skyweave ($79).", SAT) == []  # 99 - 79
    assert unsupported("With 2,100 and 1,350 orders they hold 3,450 together.", SAT) == []
    assert unsupported("68% plus 14% is a combined 82%.", SAT) == []
    assert derivable("4.9", numbers(SAT))  # 68 / 14 ratio, 1 decimal


def test_derivations_must_show_their_inputs():
    """Regression: with dozens of figures read, *some* pair sums to almost anything; a bare 13.4% CAGR
    passed by coincidence until derivations had to use figures the text itself states."""
    assert unsupported("Orbitlink costs $20 more than Skyweave.", SAT) == ["20"]
    read = "Market $6.3 billion in 2025, $11.8 billion by 2030. Energy 38%. Lettuce $3.90 vs $1.60. 9-hectare farm."
    assert unsupported("From $6.3 billion to $11.8 billion: a 13.4% CAGR.", read) == ["13.4"]
    # the live case: 1.6 + 11.8 = 13.4 by coincidence, across distant sentences and different units
    report = ("Lettuce costs $3.90 against $1.60 per kg. Energy is 38% of costs. "
              "The market reaches $11.8 billion, a 13.4% CAGR.")
    assert unsupported(report, read) == ["13.4"]
    # known residual risk: an *adjacent* sentence can still supply a coincidental pair
    adjacent = "Lettuce costs $3.90 against $1.60 per kg. The market reaches $11.8 billion, a 13.4% CAGR."
    assert unsupported(adjacent, read) == []


def test_unsupported_figures_are_reported():
    assert unsupported("Revenue was $5.5 billion and CAGR 13.4%.", SAT) == ["13.4", "5.5"]
    assert unsupported("There are 3 key challenges.", SAT) == []  # small counts are not figures


def test_url_provenance_rejects_invented_urls(harness, task):
    """Live finding: the model invented reuters/statista/wikipedia URLs and burned its retries on 404s."""
    agent = harness.agent([
        call("fetch_page", url="https://www.reuters.com/technology/satellite/"),  # never returned by a tool
        call("search_web", query="solid-state battery market"),
        call("fetch_page", url=SSB),
        call("submit_report", **report(SSB)),
    ])
    res = agent.run(task)
    assert res.status == Phase.DONE
    assert res.counters.errors == {"tool_args_invalid": 1} and res.counters.tool_executions == 2  # no 404 fetch
    feedback = harness.llm.requests[1].messages[-1]["content"]
    assert "did not appear in any search result" in feedback


def test_url_provenance_blocks_attacker_urls_from_injected_text(harness, task):
    agent = harness.agent([call("search_web", query="solid-state battery market"),
                           call("fetch_page", url="https://verified-data.example/vertical-farming"),
                           call("fetch_page", url=SSB), call("submit_report", **report(SSB))])
    res = agent.run(task)
    assert res.status == Phase.DONE and "https://verified-data.example/vertical-farming" not in res.fetched_urls


def test_summary_figures_must_be_grounded_too(harness, task):
    """Live finding: a computed 13.4% CAGR sat in the summary, where only findings were checked."""
    bad = report(SSB, summary="The market grows at a 13.4% CAGR from $1.9 billion.")
    agent = harness.agent([call("fetch_page", url=SSB), call("submit_report", **bad), call("submit_report", **report(SSB))])
    res = agent.run(task)
    assert res.status == Phase.DONE and res.counters.errors == {"final_invalid": 1}
    assert "13.4" in harness.llm.requests[2].messages[-1]["content"]


def test_derivation_may_use_the_previous_sentence():
    read = "Latticeguard holds 26% and Q-Shield 18% of the market."
    assert unsupported("Latticeguard holds 26%, Q-Shield 18%. Together they hold 44% of the market.", read) == []
