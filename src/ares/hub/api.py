"""Multi-Agent Hub API: research, database and browser agents, plus security and evaluation add-ons."""

from __future__ import annotations
import os
import re
import uuid
import shutil
import sqlite3
from typing import Any
from pathlib import Path
from ares.hub.chat import Chat
from ares.hub.runs import RunManager
from collections.abc import Callable
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from ares.config import PROJECT_ROOT, LLMSettings
from fastapi import FastAPI, File, HTTPException, UploadFile
from ares.hub.database import UnsafeSQL, csv_to_sqlite, run_query, schema

SAMPLE_DB = PROJECT_ROOT / "data" / "sample_db" / "shop.db"
SAMPLE_EVAL = PROJECT_ROOT / "data" / "sample_db" / "eval.jsonl"


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    source: str = Field("live", pattern="^(live|offline)$")
    extra_urls: list[str] = Field(default_factory=list, max_length=5)
    max_llm_calls: int = Field(12, ge=3, le=25)


class AskDbRequest(BaseModel):
    db_id: str = "sample"
    question: str = Field(min_length=3, max_length=1000)


class SqlRequest(BaseModel):
    db_id: str = "sample"
    sql: str = Field(min_length=1, max_length=5000)


class BrowserRequest(BaseModel):
    task: str = Field(min_length=5, max_length=1000)
    start_url: str | None = None
    allowed_domains: list[str] = Field(default_factory=list)
    max_steps: int = Field(15, ge=1, le=40)
    headless: bool = True


class LiveAttackRequest(BaseModel):
    attack_id: str


def create_app(data_dir: Path | None = None, *, chat_factory: Callable[[], Callable] | None = None,
               llm_factory: Callable[[], Any] | None = None, corpus_factory: Callable[[str, list[str]], Any] | None = None,
               browser_launcher=None, allow_private_browsing: bool = False) -> FastAPI:
    data_dir = data_dir or Path(os.getenv("HUB_DATA", str(PROJECT_ROOT / "data" / "hub")))
    runs = RunManager(data_dir / "hub.db", data_dir)
    db_dir = data_dir / "databases"
    app = FastAPI(title="Multi-Agent Hub", version="1.0.0")
    app.state.runs = runs

    def settings() -> LLMSettings:
        try:
            return LLMSettings.from_env()
        except RuntimeError as e:
            raise HTTPException(503, f"LLM not configured: {e}") from e

    def get_chat():
        return chat_factory() if chat_factory else Chat(settings())

    def get_llm():
        if llm_factory:
            return llm_factory()
        from ares.llm.openai_compat import OpenAICompatClient
        return OpenAICompatClient(settings())

    def get_corpus(source: str, extra: list[str]):
        if corpus_factory:
            return corpus_factory(source, extra)
        if source == "offline":
            from ares.hub.research import offline_corpus
            return offline_corpus()
        from ares.hub.live_web import LiveWeb
        return LiveWeb(extra_urls=extra)

    def db_path(db_id: str) -> Path:
        if db_id == "sample":
            return SAMPLE_DB
        if not re.fullmatch(r"[a-z0-9_-]{1,40}", db_id):
            raise HTTPException(422, "invalid database id")
        p = db_dir / f"{db_id}.db"
        if not p.exists():
            raise HTTPException(404, "database not found")
        return p

    # ---------- health ----------
    @app.get("/health")
    def health(check_llm: bool = False) -> dict:
        out = {"status": "ok", "sample_db": SAMPLE_DB.exists()}
        try:
            import playwright  # noqa: F401
            out["browser"] = "playwright"
        except ImportError:
            out["browser"] = "unavailable"
        if check_llm:
            try:
                s = LLMSettings.from_env()
                from ares.llm.openai_compat import OpenAICompatClient
                out["llm"] = bool(OpenAICompatClient(s).list_models())
            except Exception:
                out["llm"] = False
        return out

    # ---------- runs ----------
    @app.get("/runs")
    def list_runs(kind: str | None = None) -> list[dict]:
        return runs.list(kind)

    @app.get("/runs/{rid}")
    def get_run(rid: str) -> dict:
        r = runs.get(rid)
        if not r:
            raise HTTPException(404, "run not found")
        return r

    @app.post("/runs/{rid}/stop")
    def stop_run(rid: str) -> dict:
        if not runs.stop(rid):
            raise HTTPException(409, "run is not stoppable")
        return {"stopping": rid}

    @app.delete("/runs/{rid}")
    def delete_run(rid: str) -> dict:
        if not runs.delete(rid):
            raise HTTPException(404, "run not found")
        shutil.rmtree(runs.run_dir(rid), ignore_errors=True)
        return {"deleted": rid}

    @app.get("/runs/{rid}/screenshots/{step}")
    def screenshot(rid: str, step: int) -> FileResponse:
        p = runs.run_dir(rid) / f"{step:02d}.png"
        if not p.exists():
            raise HTTPException(404, "no screenshot")
        return FileResponse(p, media_type="image/png")

    # ---------- research ----------
    @app.post("/research")
    def research(req: ResearchRequest) -> dict:
        llm = get_llm()
        corpus = get_corpus(req.source, req.extra_urls)

        def work(rid, emit):
            from ares.hub.research import run_research
            out = run_research(req.question, llm, corpus, emit, req.max_llm_calls, run_id=rid)
            if hasattr(corpus, "stats"):
                out["web"] = corpus.stats
            return out

        return {"run_id": runs.start("research", req.question[:120], req.model_dump(), work)}

    # ---------- databases ----------
    @app.get("/databases")
    def databases() -> list[dict]:
        out = [{"id": "sample", "name": "Sample shop (customers, products, orders)"}] if SAMPLE_DB.exists() else []
        if db_dir.exists():
            out += [{"id": p.stem, "name": p.stem} for p in sorted(db_dir.glob("*.db"))]
        return out

    @app.post("/databases/upload")
    async def upload_db(files: list[UploadFile] = File(...)) -> dict:
        db_dir.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^a-z0-9_-]", "_", Path(files[0].filename or "data").stem.lower())[:30] + "-" + uuid.uuid4().hex[:4]
        dest = db_dir / f"{name}.db"
        data = [(f.filename or "data.csv", await f.read()) for f in files]
        if all(n.lower().endswith((".db", ".sqlite", ".sqlite3")) for n, _ in data) and len(data) == 1:
            if not data[0][1].startswith(b"SQLite format 3"):
                raise HTTPException(422, "not a SQLite database file")
            dest.write_bytes(data[0][1])
            try:
                schema(dest)
            except sqlite3.DatabaseError as e:
                dest.unlink(missing_ok=True)
                raise HTTPException(422, f"unreadable database: {e}") from e
            return {"id": name, "tables": [t["table"] for t in schema(dest)]}
        if all(n.lower().endswith(".csv") for n, _ in data):
            try:
                tables = csv_to_sqlite(data, dest)
            except (ValueError, UnicodeDecodeError, sqlite3.DatabaseError) as e:
                dest.unlink(missing_ok=True)
                raise HTTPException(422, str(e)) from e
            return {"id": name, "tables": tables}
        raise HTTPException(422, "upload one .db/.sqlite file or one or more .csv files")

    @app.delete("/databases/{db_id}")
    def delete_db(db_id: str) -> dict:
        if db_id == "sample":
            raise HTTPException(409, "the sample database cannot be deleted")
        db_path(db_id).unlink()
        return {"deleted": db_id}

    @app.get("/databases/{db_id}/schema")
    def db_schema(db_id: str) -> list[dict]:
        return schema(db_path(db_id))

    @app.post("/databases/query")
    def manual_query(req: SqlRequest) -> dict:
        try:
            return run_query(db_path(req.db_id), req.sql)
        except UnsafeSQL as e:
            raise HTTPException(422, f"Blocked: {e}") from e
        except sqlite3.DatabaseError as e:
            raise HTTPException(422, f"SQL error: {e}") from e

    @app.post("/databases/ask")
    def ask(req: AskDbRequest) -> dict:
        path, chat = db_path(req.db_id), get_chat()

        def work(rid, emit):
            from ares.hub.database import ask_database
            return ask_database(path, req.question, chat, emit)

        return {"run_id": runs.start("database", req.question[:120], req.model_dump(), work)}

    # ---------- browser ----------
    @app.post("/browser")
    def browser(req: BrowserRequest) -> dict:
        chat = get_chat()

        def work(rid, emit):
            from ares.hub.browser import BrowserAgent, launch_browser
            agent = BrowserAgent(chat, allow_private=allow_private_browsing, allowed_domains=req.allowed_domains,
                                 max_steps=req.max_steps, headless=req.headless, shots_dir=runs.run_dir(rid),
                                 on_event=emit, launcher=browser_launcher or launch_browser)
            runs.cancel_hooks[rid] = lambda: setattr(agent, "stop_requested", True)
            out = agent.run(req.task, req.start_url)
            for s in out["steps"]:
                s["screenshot"] = s["step"] if s.get("screenshot") else None
            return out

        return {"run_id": runs.start("browser", req.task[:120], req.model_dump(), work)}

    # ---------- security add-on ----------
    @app.get("/security/attacks")
    def attacks() -> list[dict]:
        from ares.security.attacks import ATTACKS
        return [{"id": a.id, "name": a.name, "owasp": a.owasp, "goal": a.goal} for a in ATTACKS]

    @app.post("/security/matrix")
    def security_matrix() -> dict:
        def work(rid, emit):
            from ares.security.suite import guarantee_matrix
            rows = guarantee_matrix()
            emit({"type": "info", "message": f"{len(rows)} attack x defence combinations evaluated"})
            return {"rows": [r.model_dump() for r in rows]}

        return {"run_id": runs.start("security", "Guarantee matrix (worst-case model)", {}, work)}

    @app.post("/security/live")
    def security_live(req: LiveAttackRequest) -> dict:
        from ares.security.attacks import ATTACKS
        attack = next((a for a in ATTACKS if a.id == req.attack_id), None)
        if not attack:
            raise HTTPException(404, "unknown attack")
        llm = get_llm()

        def work(rid, emit):
            from ares.hub.research import CallbackTracer
            from ares.security.attacks import attack_corpus
            from ares.security.suite import DEFENDED, run_attack
            row, res = run_attack(attack, DEFENDED, llm, attack_corpus(), tracer=CallbackTracer(emit))
            return {"row": row.model_dump(), "final": res.final}

        return {"run_id": runs.start("security", f"Live attack {attack.id}: {attack.name}", req.model_dump(), work)}

    # ---------- evaluation add-on ----------
    @app.post("/eval/sql")
    def eval_sql() -> dict:
        chat = get_chat()

        def work(rid, emit):
            from ares.hub.sql_eval import evaluate_sql
            return evaluate_sql(SAMPLE_DB, SAMPLE_EVAL, chat, emit)

        return {"run_id": runs.start("eval", "Text-to-SQL execution accuracy (sample shop)", {}, work)}

    return app
