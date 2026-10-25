from __future__ import annotations
import pytest
from ares.tracing import MemoryTracer
from ares.llm.scripted import ScriptedLLM
from ares.runtime import Agent, AgentConfig, Task
from ares.tools import Corpus, Document, default_registry

SSB = "https://test.example/ssb-market"
SSB_CHALLENGES = "https://test.example/ssb-challenges"


@pytest.fixture(autouse=True)
def _unit_tests_isolate_mechanisms(request, monkeypatch):
    """Scripted unit tests fetch known URLs directly to exercise one mechanism at a time; URL provenance
    (on by default everywhere else) is tested on its own in test_url_provenance_*."""
    if "provenance" in request.node.name:
        yield
        return
    from ares.runtime import config as config_mod

    field = config_mod.AgentConfig.model_fields["url_provenance"]
    monkeypatch.setattr(field, "default", False)
    config_mod.AgentConfig.model_rebuild(force=True)
    yield
    monkeypatch.undo()
    config_mod.AgentConfig.model_rebuild(force=True)


@pytest.fixture
def corpus() -> Corpus:
    return Corpus([
        Document(url=SSB, title="Solid-state battery market",
                 text="The solid-state battery market reached $1.9 billion in 2025 and may reach $8.4 billion by 2030."),
        Document(url=SSB_CHALLENGES, title="Solid-state battery challenges",
                 text="Dendrites, cost of $210 per kWh and a 71% manufacturing yield slow adoption."),
        Document(url="https://test.example/gadgets", title="Gadgets", text="A foldable e-reader."),
    ])


@pytest.fixture
def task() -> Task:
    return Task(id="ssb", prompt="What is the outlook for the solid-state battery market?")


class Harness:
    """Builds an agent around a scripted model and records its trace in memory."""

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.registry = default_registry()
        self.tracer = MemoryTracer()
        self.sleeps: list[float] = []
        self.llm: ScriptedLLM | None = None

    def agent(self, steps, *, registry=None, approver=None, repeat_last=False, **config) -> Agent:
        self.llm = ScriptedLLM(steps, repeat_last=repeat_last)
        kw = {"approver": approver} if approver else {}
        return Agent(self.llm, registry or self.registry, self.corpus, config=AgentConfig(**config),
                     tracer=self.tracer, sleep=self.sleeps.append, **kw)


@pytest.fixture
def harness(corpus) -> Harness:
    return Harness(corpus)


def report(*urls: str, summary: str = "The market is growing fast but costs remain high.") -> dict:
    return {
        "title": "Solid-state battery outlook",
        "summary": summary,
        "findings": [{"claim": f"Finding from {u}", "source_url": u} for u in urls],
    }


def assert_transcript_valid(messages: list[dict]) -> None:
    """Every tool call the model made is answered by exactly one tool message, after it."""
    open_ids: list[str] = []
    for m in messages:
        if m["role"] == "assistant":
            open_ids += [tc["id"] for tc in m.get("tool_calls", [])]
        elif m["role"] == "tool":
            assert m["tool_call_id"] in open_ids, f"tool message for unknown/answered id {m['tool_call_id']}"
            open_ids.remove(m["tool_call_id"])
    assert not open_ids, f"unanswered tool calls: {open_ids}"
