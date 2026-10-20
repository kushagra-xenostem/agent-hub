"""The replay/report web view renders every page from a real trace store and results files."""
from __future__ import annotations

from fastapi.testclient import TestClient

from ares.errors import AgentError, ErrorKind
from ares.eval.dataset import load_tasks
from ares.eval.runner import run_eval
from ares.llm.scripted import call
from ares.llm.simulated import SimulatedResearcher
from ares.report.app import create_app, markdown
from ares.tools import Corpus
from ares.tracing import SQLiteTraceStore
from tests.conftest import SSB, report


def make_client(tmp_path, harness, task):
    store = SQLiteTraceStore(tmp_path / "t.sqlite")
    harness.tracer = store
    res = harness.agent([AgentError(ErrorKind.LLM_RATE_LIMITED, "429", retry_after_s=1.0), call("fetch_page", url=SSB),
                         call("submit_report", **report(SSB))]).run(task)
    eval_dir = tmp_path / "eval"
    run_eval(load_tasks()[:2], ["L4_full"], ["none"], [0], model_factory=lambda s: SimulatedResearcher(seed=s),
             corpus=Corpus.load(), results=eval_dir / "main.jsonl", sleep=lambda s: None)
    results = tmp_path / "results"
    results.mkdir()
    (results / "security_matrix.md").write_text("# M\n\n| a | b |\n|---|---|\n| A1 | **LEAK** |\n", encoding="utf-8")
    app = create_app(db=tmp_path / "t.sqlite", results_dir=results, eval_dir=eval_dir, docs_dir=tmp_path)
    return TestClient(app), res


def test_pages_render(tmp_path, harness, task):
    client, res = make_client(tmp_path, harness, task)
    home = client.get("/")
    assert home.status_code == 200 and res.run_id in home.text
    run = client.get(f"/runs/{res.run_id}")
    assert run.status_code == 200 and "llm_rate_limited" in run.text and "Final report" in run.text
    assert client.get(f"/runs/{res.run_id}?transitions=true").status_code == 200
    assert client.get("/eval").status_code == 200 and "Summary" in client.get("/eval").text
    sec = client.get("/security")
    assert sec.status_code == 200 and "LEAK" in sec.text
    assert client.get("/runs/nope").status_code == 404


def test_replay_and_otel_endpoints(tmp_path, harness, task):
    client, res = make_client(tmp_path, harness, task)
    spans = client.get(f"/runs/{res.run_id}/otel.json").json()
    assert any(s["name"] == "agent.run" for s in spans)
    # this run used the test corpus, which replay cannot locate by fingerprint: the page explains instead of crashing
    page = client.get(f"/runs/{res.run_id}/replay")
    assert page.status_code == 200 and "Replay of" in page.text


def test_api_and_docs(tmp_path, harness, task):
    client, res = make_client(tmp_path, harness, task)
    assert client.get("/api/runs").json()[0]["run_id"]
    assert len(client.get(f"/api/runs/{res.run_id}").json()["events"]) > 5
    (tmp_path / "x.md").write_text("# Title\n\n- a **b**\n", encoding="utf-8")
    assert "<b>b</b>" in client.get("/docs/x.md").text
    assert client.get("/docs/../pyproject.toml").status_code == 404  # no path traversal


def test_markdown_renderer_handles_tables_and_code():
    out = markdown("# H\n\n| a | b |\n|---|---|\n| 1 | `x` |\n\n```\n<raw>\n```\n")
    assert "<table>" in out and "<code>x</code>" in out and "&lt;raw&gt;" in out
