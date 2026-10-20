"""Least privilege, the approval gate, idempotency, grounding and tool contracts."""
from __future__ import annotations

from ares.llm.scripted import call
from ares.runtime import Phase, Task, approve_all
from ares.tools import Scope, default_registry
from ares.tracing import EventKind
from tests.conftest import SSB, assert_transcript_valid, report

EMAIL = {"to": "team@company.example", "subject": "data", "body": "notes"}
MAIL_SCOPES = frozenset({Scope.WEB_READ, Scope.EXTERNAL_SEND})
ALLOWED = frozenset({"company.example"})


def test_unpermitted_tools_are_not_offered(harness, task):
    agent = harness.agent([call("fetch_page", url=SSB), call("submit_report", **report(SSB))])
    agent.run(task)
    offered = {t["function"]["name"] for t in harness.llm.requests[0].tools}
    assert "send_email" not in offered and "fetch_page" in offered


def test_permission_denied_even_if_model_calls_hidden_tool(harness, task):
    agent = harness.agent([call("send_email", **EMAIL), call("fetch_page", url=SSB), call("submit_report", **report(SSB))])
    res = agent.run(task)
    assert res.status == Phase.DONE
    assert res.outbox == []
    assert res.counters.errors == {"permission_denied": 1}
    denied = [e for e in harness.tracer.of_kind(EventKind.PERMISSION) if not e.data["allowed"]]
    assert denied[0].data["tool"] == "send_email"  # audit trail


def test_without_permission_enforcement_the_email_is_sent(harness, task):
    agent = harness.agent([call("send_email", **EMAIL), call("fetch_page", url=SSB), call("submit_report", **report(SSB))],
                          enforce_permissions=False, require_approval=False, egress_allowlist=False)
    res = agent.run(task)
    assert len(res.outbox) == 1  # the security failure the controls exist to prevent


def test_approval_gate_denies_by_default(harness):
    task = Task(id="mail", prompt="Email the summary.", scopes=MAIL_SCOPES, allowed_recipients=ALLOWED)
    agent = harness.agent([call("send_email", **EMAIL), call("fetch_page", url=SSB), call("submit_report", **report(SSB))])
    res = agent.run(task)
    assert res.outbox == [] and res.counters.errors == {"approval_denied": 1}
    assert harness.tracer.of_kind(EventKind.APPROVAL)[0].data["decision"] == "deny"


def test_approval_gate_approves(harness):
    task = Task(id="mail", prompt="Email the summary.", scopes=MAIL_SCOPES, allowed_recipients=ALLOWED)
    agent = harness.agent([call("send_email", **EMAIL), call("fetch_page", url=SSB), call("submit_report", **report(SSB))],
                          approver=approve_all)
    res = agent.run(task)
    assert len(res.outbox) == 1


def test_escalation_ends_the_run(harness):
    task = Task(id="mail", prompt="Email the summary.", scopes=MAIL_SCOPES, allowed_recipients=ALLOWED)
    agent = harness.agent([call("send_email", **EMAIL)], approver=lambda c, s: "escalate")
    res = agent.run(task)
    assert res.status == Phase.ESCALATED and res.outbox == []
    assert_transcript_valid(res.messages)


def test_side_effect_retries_are_idempotent(harness):
    task = Task(id="mail", prompt="Email the summary.", scopes=MAIL_SCOPES, allowed_recipients=ALLOWED)
    agent = harness.agent([call("send_email", **EMAIL), call("send_email", **EMAIL), call("fetch_page", url=SSB),
                           call("submit_report", **report(SSB))], approver=approve_all)
    res = agent.run(task)
    assert len(res.outbox) == 1
    assert res.counters.deduplicated == 1


def test_report_must_cite_fetched_pages(harness, task):
    agent = harness.agent([
        call("search_web", query="solid-state battery"),
        call("submit_report", **report(SSB)),  # seen in search results, never fetched
        call("fetch_page", url=SSB),
        call("submit_report", **report(SSB)),
    ])
    res = agent.run(task)
    assert res.status == Phase.DONE
    assert res.counters.errors == {"final_invalid": 1}


def test_hallucinated_citation_is_rejected(harness, task):
    agent = harness.agent([call("fetch_page", url=SSB), call("submit_report", **report("https://made.up/source"))],
                          max_retries=0)
    res = agent.run(task)
    assert res.status == Phase.FAILED and "not fetched" in res.reason


def test_invalid_report_schema(harness, task):
    bad = {"title": "x", "summary": "short", "findings": []}
    agent = harness.agent([call("fetch_page", url=SSB), call("submit_report", **bad), call("submit_report", **report(SSB))])
    res = agent.run(task)
    assert res.status == Phase.DONE and res.counters.errors == {"final_invalid": 1}


def test_tool_schemas_are_strict_contracts():
    reg = default_registry()
    search = next(s for s in reg.schemas() if s["function"]["name"] == "search_web")["function"]["parameters"]
    assert search["required"] == ["query"]
    assert search["properties"]["max_results"]["maximum"] == 10
    assert search["additionalProperties"] is False
    assert "title" not in search["properties"]["query"]


def test_extra_arguments_are_rejected(harness, task):
    agent = harness.agent([call("fetch_page", url=SSB, render_js=True), call("fetch_page", url=SSB),
                           call("submit_report", **report(SSB))])
    res = agent.run(task)
    assert res.counters.errors == {"tool_args_invalid": 1}


def _objects(schema: dict):
    """Every object schema inside a tool schema, including $defs."""
    if schema.get("type") == "object":
        yield schema
    for sub in (schema.get("$defs") or {}).values():
        yield from _objects(sub)
    for sub in (schema.get("properties") or {}).values():
        yield from _objects(sub)
        if "items" in sub:
            yield from _objects(sub["items"])


def test_every_required_field_is_visible_to_the_model():
    """Regression: schema cleanup once deleted the report's `title` property while `required` still
    listed it, so the model could never produce a valid report."""
    for tool in default_registry().schemas():
        for obj in _objects(tool["function"]["parameters"]):
            missing = set(obj.get("required", [])) - set(obj.get("properties", {}))
            assert not missing, f"{tool['function']['name']}: required but not described: {missing}"
    report_schema = next(t for t in default_registry().schemas() if t["function"]["name"] == "submit_report")
    assert "title" in report_schema["function"]["parameters"]["properties"]
