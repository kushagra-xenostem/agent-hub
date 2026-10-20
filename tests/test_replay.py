"""Replay debugging: recorded runs re-execute with zero model calls, verify, or show where they diverge."""
from __future__ import annotations

from ares.errors import AgentError, ErrorKind
from ares.eval.experiments import build_agent
from ares.faults import PRESETS
from ares.llm.scripted import ScriptedLLM, call
from ares.llm.simulated import SimulatedResearcher
from ares.replay import replay
from ares.runtime import AgentConfig
from ares.tracing import SQLiteTraceStore
from tests.conftest import SSB, report


def record(tmp_path, corpus, task, llm, **kw):
    store = SQLiteTraceStore(tmp_path / "t.sqlite")
    res = build_agent(llm, corpus, AgentConfig(), tracer=store, sleep=lambda s: None, **kw).run(task)
    return store, res


def test_replay_of_unchanged_code_is_verified(tmp_path, corpus, task):
    steps = [call("search_web", query="solid-state battery"), call("search_web", query="x"),  # invalid -> repair
             AgentError(ErrorKind.LLM_RATE_LIMITED, "429", retry_after_s=2.0),
             call("fetch_page", url=SSB), call("submit_report", **report(SSB))]
    store, res = record(tmp_path, corpus, task, ScriptedLLM(steps))
    rep, replayed = replay(store, res.run_id, corpus=corpus)
    assert rep.verified, rep
    assert rep.model_calls_replayed == rep.model_calls_recorded == 5  # incl. the recorded 429
    assert replayed.final == res.final and replayed.counters.errors == res.counters.errors


def test_replay_reproduces_injected_faults(tmp_path, task):
    from ares.tools import Corpus
    corpus = Corpus.load()
    store, res = record(tmp_path, corpus, task, SimulatedResearcher(seed=4, p_bad_args=0.2),
                        plan=PRESETS["mixed"], seed="4:t")
    assert res.counters.injected_faults > 0
    rep, replayed = replay(store, res.run_id)
    assert rep.verified and replayed.counters.injected_faults == res.counters.injected_faults


def test_replay_shows_where_changed_behaviour_diverges(tmp_path, corpus, task):
    store, res = record(tmp_path, corpus, task, ScriptedLLM([call("fetch_page", url=SSB),
                                                               call("submit_report", **report(SSB))]))
    rep, _ = replay(store, res.run_id, corpus=corpus, system_prompt="You are a different agent.")
    assert not rep.verified and rep.divergences[0].call_index == 1  # the very first request changed


def test_strict_replay_stops_at_the_divergence(tmp_path, corpus, task):
    store, res = record(tmp_path, corpus, task, ScriptedLLM([call("fetch_page", url=SSB),
                                                               call("submit_report", **report(SSB))]))
    rep, result = replay(store, res.run_id, corpus=corpus, strict=True, config_overrides={"max_tokens": 99})
    assert result is None and rep.replayed_status == "diverged" and "diverged at model call 1" in rep.error


def test_snapshot_keeps_old_runs_replayable_after_the_corpus_changes(tmp_path, monkeypatch, task):
    import ares.tools.corpus as corpus_mod
    from ares.tools import Corpus, Document

    monkeypatch.setattr(corpus_mod, "SNAPSHOTS", tmp_path / "corpora")
    original = Corpus.load()
    original.snapshot(tmp_path / "corpora")
    store, res = record(tmp_path, original, task, SimulatedResearcher(seed=1))

    # the corpus evolves: a new, highly relevant page now ranks first for the same query
    changed = Corpus(list(original._docs.values()) + [Document(
        url="https://new.example/solid-state-battery-outlook", title="Solid-state battery outlook market",
        text="Solid-state battery outlook and market commentary.")])
    monkeypatch.setattr(Corpus, "load", classmethod(lambda cls, path=None: changed if path is None else
                                                    Corpus([Document.model_validate_json(x) for x in
                                                            open(path, encoding="utf-8").read().splitlines() if x])))
    rep, _ = replay(store, res.run_id)
    assert rep.verified  # restored from the snapshot, not the changed corpus
