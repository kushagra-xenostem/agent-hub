"""OpenTelemetry export of stored runs."""
from __future__ import annotations
from ares.llm.scripted import call
from tests.conftest import SSB, report
from ares.tracing.otel import export_run
from ares.tracing import SQLiteTraceStore
from ares.errors import AgentError, ErrorKind
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_run_exports_as_a_span_tree(tmp_path, harness, task):
    store = SQLiteTraceStore(tmp_path / "t.sqlite")
    harness.tracer = store
    res = harness.agent([AgentError(ErrorKind.LLM_UNAVAILABLE, "503"), call("fetch_page", url=SSB),
                         call("submit_report", **report(SSB))]).run(task)
    exporter = InMemorySpanExporter()
    n = export_run(store.get_run(res.run_id), store.events(res.run_id), exporter)
    spans = exporter.get_finished_spans()
    assert n == len(spans)
    root = next(s for s in spans if s.name == "agent.run")
    llm = [s for s in spans if s.name == "gen_ai.chat"]
    tools = [s for s in spans if s.name.startswith("tool.")]
    assert len(llm) == 3 and len(tools) == 1
    assert all(s.parent.span_id == root.context.span_id for s in llm + tools)  # one tree
    assert all(s.context.trace_id == root.context.trace_id for s in spans)
    assert llm[1].attributes["gen_ai.usage.input_tokens"] > 0
    assert not llm[0].status.is_ok  # the 503
    names = {e.name for e in root.events}
    assert {"transition", "error", "recovery"} <= names
    assert root.attributes["ares.status"] == "done"


def test_trace_ids_are_stable_per_run(tmp_path, harness, task):
    store = SQLiteTraceStore(tmp_path / "t.sqlite")
    harness.tracer = store
    res = harness.agent([call("fetch_page", url=SSB), call("submit_report", **report(SSB))]).run(task)
    ids = []
    for _ in range(2):
        exp = InMemorySpanExporter()
        export_run(store.get_run(res.run_id), store.events(res.run_id), exp)
        ids.append(sorted((s.context.trace_id, s.context.span_id) for s in exp.get_finished_spans()))
    assert ids[0] == ids[1]
