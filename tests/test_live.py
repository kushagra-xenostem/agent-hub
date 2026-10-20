"""End-to-end against the real model (project #1 server). Run with: pytest -m live"""
from __future__ import annotations

import pytest

from ares.llm import OpenAICompatClient
from ares.runtime import Agent, AgentConfig, Phase, Task
from ares.tools import Corpus, default_registry
from ares.tracing import MemoryTracer
from tests.conftest import assert_transcript_valid

pytestmark = pytest.mark.live


def test_live_research_task_is_grounded():
    tracer = MemoryTracer()
    agent = Agent(OpenAICompatClient.from_env(), default_registry(), Corpus.load(), config=AgentConfig(),
                  tracer=tracer)
    res = agent.run(Task(id="live-ssb", prompt="What is the outlook for the solid-state battery market? "
                                               "Include market size and growth, the key players, and the main challenges."))
    assert res.status == Phase.DONE, res.reason
    assert_transcript_valid(res.messages)
    cited = {f["source_url"] for f in res.final["findings"]}
    assert cited <= set(res.fetched_urls)  # grounding
    text = (res.final["summary"] + " ".join(f["claim"] for f in res.final["findings"])).lower()
    assert "1.9 billion" in text and "8.4 billion" in text and "quantor" in text  # facts only the corpus has
