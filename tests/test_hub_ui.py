"""Renders the hub UI headlessly against an in-process backend with scripted models."""
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

ROOT = Path(__file__).resolve().parents[1]


class Chat:
    def __init__(self):
        self.replies = []

    def __call__(self, messages, max_tokens=800, temperature=0.0):
        return self.replies.pop(0) if self.replies else '{"sql": "SELECT count(*) AS n FROM customers", "explanation": "count"}'


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    from ares.hub.api import create_app
    from ares.hub.research import offline_corpus

    chat = Chat()
    app = create_app(tmp_path_factory.mktemp("hubui"), chat_factory=lambda: chat,
                     corpus_factory=lambda s, e: offline_corpus())
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(f"{url}/health")
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    mp = pytest.MonkeyPatch()
    mp.setenv("BACKEND_URL", url)
    yield url, chat
    mp.undo()
    server.should_exit = True


def test_ui_tabs_and_database_question(backend):
    from streamlit.testing.v1 import AppTest

    url, chat = backend
    at = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Research", "Database", "Browser", "Security (add-on)",
                                          "Evaluation (add-on)", "History"]
    chat.replies = ['{"sql": "SELECT count(*) AS n FROM customers", "explanation": "count"}', "There are 120 customers."]
    q = next(t for t in at.text_input if t.label == "Ask a question about the data")
    q.set_value("How many customers?")
    next(b for b in at.button if b.label == "Ask").click().run()
    assert not at.exception
    rid = at.session_state.db_run
    for _ in range(100):
        if httpx.get(f"{url}/runs/{rid}").json()["status"] != "running":
            break
        time.sleep(0.05)
    at.run()
    assert not at.exception
    assert any("There are 120 customers." in m.value for m in at.markdown)
