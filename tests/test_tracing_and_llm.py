"""SQLite trace store round-trip and the HTTP client's error classification."""
from __future__ import annotations

import httpx
import pytest

from ares.config import LLMSettings
from ares.errors import AgentError, ErrorKind
from ares.llm import LLMRequest, OpenAICompatClient
from ares.llm.scripted import call
from ares.runtime import Agent, AgentConfig
from ares.tracing import EventKind, SQLiteTraceStore
from tests.conftest import SSB, report


def test_sqlite_store_round_trip(tmp_path, harness, task):
    store = SQLiteTraceStore(tmp_path / "t.sqlite")
    harness.tracer = store
    agent = harness.agent([call("fetch_page", url=SSB), call("submit_report", **report(SSB))])
    res = agent.run(task)

    run = store.get_run(res.run_id)
    assert run.status == "done" and run.final == res.final and run.meta["fetched_urls"] == [SSB]
    assert store.get_run(res.run_id[:10]).run_id == res.run_id  # prefix lookup for the CLI
    events = store.events(res.run_id)
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert events[0].kind == EventKind.RUN_START and events[-1].kind == EventKind.RUN_END
    assert store.list_runs()[0].run_id == res.run_id


def client_for(handler) -> OpenAICompatClient:
    settings = LLMSettings(base_url="http://test/v1", api_key="k", model="m")
    http = httpx.Client(base_url=settings.base_url, transport=httpx.MockTransport(handler))
    return OpenAICompatClient(settings, http=http)


def test_client_parses_tool_calls_and_usage():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search_web", "arguments": '{"query": "q"}'}}]}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 40}},
            "timings": {"predicted_per_second": 17.0},
        })

    resp = client_for(handler).complete(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
    assert resp.tool_calls[0].name == "search_web" and resp.tool_calls[0].arguments == '{"query": "q"}'
    assert resp.usage.cached_tokens == 40 and resp.server_timings["predicted_per_second"] == 17.0


@pytest.mark.parametrize("status,body,headers,kind,retry_after", [
    (401, {"error": {"message": "Invalid or missing API key."}}, {}, ErrorKind.LLM_AUTH, None),
    (429, {"error": {"message": "Rate limit exceeded"}}, {"Retry-After": "4"}, ErrorKind.LLM_RATE_LIMITED, 4.0),
    (400, {"error": {"message": "the request exceeds the available context size"}}, {}, ErrorKind.LLM_CONTEXT_OVERFLOW, None),
    (400, {"error": {"message": "bad field"}}, {}, ErrorKind.LLM_BAD_REQUEST, None),
    (503, {"error": {"message": "loading model"}}, {}, ErrorKind.LLM_UNAVAILABLE, None),
    (504, {"error": {"message": "gateway timeout"}}, {}, ErrorKind.LLM_TIMEOUT, None),
])
def test_client_classifies_http_errors(status, body, headers, kind, retry_after):
    client = client_for(lambda r: httpx.Response(status, json=body, headers=headers))
    with pytest.raises(AgentError) as ei:
        client.complete(LLMRequest(messages=[]))
    assert ei.value.kind == kind and ei.value.retry_after_s == retry_after


def test_client_classifies_network_failures():
    def boom(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(AgentError) as ei:
        client_for(boom).complete(LLMRequest(messages=[]))
    assert ei.value.kind == ErrorKind.LLM_UNAVAILABLE and ei.value.transient


def test_request_sends_sampling_and_seed():
    seen = {}

    def handler(request):
        import json
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client_for(handler).complete(LLMRequest(messages=[], temperature=0.0, seed=7, max_tokens=99))
    assert seen["temperature"] == 0.0 and seen["seed"] == 7 and seen["max_tokens"] == 99 and seen["model"] == "m"


def test_agent_end_to_end_through_http_client(corpus, task):
    """The real client inside the real agent, against a mocked server: catches protocol mismatches."""
    script = iter([
        {"id": "a", "type": "function", "function": {"name": "fetch_page", "arguments": f'{{"url": "{SSB}"}}'}},
        {"id": "b", "type": "function", "function": {"name": "submit_report", "arguments": __import__("json").dumps(report(SSB))}},
    ])

    def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls",
                                                       "message": {"content": "", "tool_calls": [next(script)]}}]})

    from ares.tools import default_registry
    res = Agent(client_for(handler), default_registry(), corpus, config=AgentConfig()).run(task)
    assert res.status == "done"


def test_history_parse_error_is_not_transient():
    msg = "Failed to parse tool call arguments as JSON: [json.exception.parse_error.101] parse error"
    client = client_for(lambda r: httpx.Response(500, json={"error": {"message": msg}}))
    with pytest.raises(AgentError) as ei:
        client.complete(LLMRequest(messages=[]))
    assert ei.value.kind == ErrorKind.LLM_BAD_REQUEST and not ei.value.transient
