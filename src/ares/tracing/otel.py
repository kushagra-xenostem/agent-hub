"""OpenTelemetry export of stored runs.

One root span per run (`agent.run`), a child span per model call (`gen_ai.chat`, GenAI semantic-convention
attributes) and per tool call (`tool.<name>`); state transitions, errors, recoveries, injected faults and
security decisions become span events. Trace ids are derived from the run id, so re-exporting is stable and
traces can be joined with the model server's logs by run id.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace.id_generator import IdGenerator
from ares.tracing.schema import EventKind, RunRecord, TraceEvent
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

_EVENT_KINDS = {EventKind.TRANSITION, EventKind.ERROR, EventKind.RECOVERY, EventKind.FAULT, EventKind.SECURITY,
                EventKind.VALIDATION, EventKind.PERMISSION, EventKind.APPROVAL, EventKind.COMPACTION,
                EventKind.WRAPUP}


class _RunIds(IdGenerator):
    """Deterministic ids: trace id from the run id, span ids from (run id, sequence)."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.n = 0

    def generate_trace_id(self) -> int:
        return int(hashlib.sha256(self.run_id.encode()).hexdigest()[:32], 16)

    def generate_span_id(self) -> int:
        self.n += 1
        return int(hashlib.sha256(f"{self.run_id}:{self.n}".encode()).hexdigest()[:16], 16) or 1


class JsonFileExporter(SpanExporter):
    def __init__(self, path: Path):
        self.path = Path(path)

    def export(self, spans) -> SpanExportResult:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for span in spans:
                f.write(json.dumps(json.loads(span.to_json())) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def _ns(seconds: float) -> int:
    return int(seconds * 1e9)


def _attr(value):
    if isinstance(value, (str, bool, int, float)):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)[:4000]


def export_run(record: RunRecord, events: list[TraceEvent], exporter: SpanExporter) -> int:
    """Export one run; returns the number of spans."""
    provider = TracerProvider(resource=Resource.create({"service.name": "ares-agent", "ares.variant": record.variant}),
                              id_generator=_RunIds(record.run_id))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("ares")
    end = record.ended_at or (events[-1].ts if events else record.started_at)
    count = 0
    root = tracer.start_span("agent.run", start_time=_ns(record.started_at), attributes={
        "ares.run_id": record.run_id, "ares.task_id": record.task_id, "ares.variant": record.variant,
        "gen_ai.request.model": record.model, "ares.status": record.status, "ares.reason": record.reason or "",
        **{f"ares.counters.{k}": _attr(v) for k, v in record.counters.items()},
    })
    ctx = trace.set_span_in_context(root)
    for ev in events:
        d = ev.data
        if ev.kind == EventKind.LLM_CALL:
            dur = (ev.duration_ms or 0) / 1000
            attrs = {"gen_ai.system": "openai", "gen_ai.operation.name": "chat", "gen_ai.request.model": record.model,
                     "ares.fingerprint": d.get("fingerprint", ""), "ares.n_messages": d.get("n_messages", 0)}
            if d.get("ok"):
                r = d["response"]
                attrs.update({"gen_ai.usage.input_tokens": r["usage"]["prompt_tokens"],
                              "gen_ai.usage.output_tokens": r["usage"]["completion_tokens"],
                              "ares.usage.cached_tokens": r["usage"]["cached_tokens"],
                              "gen_ai.response.finish_reasons": [r.get("finish_reason") or ""],
                              "ares.tool_calls": [tc["name"] for tc in r["tool_calls"]]})
            span = tracer.start_span("gen_ai.chat", context=ctx, start_time=_ns(ev.ts - dur), attributes=attrs)
            if not d.get("ok"):
                span.set_status(Status(StatusCode.ERROR, str(d.get("error"))))
            span.end(end_time=_ns(ev.ts))
            count += 1
        elif ev.kind == EventKind.TOOL_CALL:
            dur = (ev.duration_ms or 0) / 1000
            span = tracer.start_span(f"tool.{d['tool']}", context=ctx, start_time=_ns(ev.ts - dur), attributes={
                "ares.tool.name": d["tool"], "ares.tool.call_id": d.get("call_id", ""),
                "ares.tool.args": _attr(d.get("args")), "ares.tool.ok": bool(d.get("ok")),
                "ares.tool.deduplicated": bool(d.get("deduplicated", False))})
            if not d.get("ok"):
                span.set_status(Status(StatusCode.ERROR, f"{d.get('error')}: {d.get('message', '')}"[:500]))
            span.end(end_time=_ns(ev.ts))
            count += 1
        elif ev.kind in _EVENT_KINDS:
            root.add_event(ev.kind.value, timestamp=_ns(ev.ts),
                           attributes={"ares.phase": ev.phase, **{f"ares.{k}": _attr(v) for k, v in d.items()}})
    if record.status != "done":
        root.set_status(Status(StatusCode.ERROR, record.reason or record.status))
    root.end(end_time=_ns(end))
    provider.shutdown()
    return count + 1


def otlp_exporter(endpoint: str) -> SpanExporter:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")
