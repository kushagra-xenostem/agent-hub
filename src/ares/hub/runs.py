"""Background runs (research, database, browser, security, evaluation) with live events and history."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT, params TEXT, status TEXT NOT NULL, result TEXT,
    events TEXT, error TEXT, created REAL NOT NULL, finished REAL
);
"""


class RunManager:
    def __init__(self, db_path: Path, data_dir: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.execute("UPDATE runs SET status='failed', error='Interrupted by restart' WHERE status='running'")
        self._db.commit()
        self._lock = threading.RLock()
        self.live: dict[str, dict[str, Any]] = {}
        self.cancel_hooks: dict[str, Callable[[], None]] = {}

    def run_dir(self, run_id: str) -> Path:
        return self.data_dir / "runs" / run_id

    def start(self, kind: str, title: str, params: dict, fn: Callable[[str, Callable[[dict], None]], dict]) -> str:
        rid = f"{kind[:3]}-{uuid.uuid4().hex[:10]}"
        state = {"id": rid, "kind": kind, "title": title, "params": params, "status": "running", "result": None,
                 "events": [], "error": None, "created": time.time(), "finished": None}
        with self._lock:
            self.live[rid] = state
            self._db.execute("INSERT INTO runs (id, kind, title, params, status, events, created) VALUES (?,?,?,?,?,?,?)",
                             (rid, kind, title, json.dumps(params), "running", "[]", state["created"]))
            self._db.commit()

        def on_event(ev: dict) -> None:
            with self._lock:
                state["events"].append({"ts": time.time(), **ev})

        def target():
            try:
                result = fn(rid, on_event)
                status, err = (result or {}).get("run_status", "done"), None
            except Exception as e:
                result, status, err = None, "failed", f"{type(e).__name__}: {e}"
                on_event({"type": "error", "message": err, "trace": traceback.format_exc()[-1500:]})
            with self._lock:
                state.update(status=status, result=result, error=err, finished=time.time())
                self._db.execute("UPDATE runs SET status=?, result=?, events=?, error=?, finished=? WHERE id=?",
                                 (status, json.dumps(result, default=str), json.dumps(state["events"], default=str),
                                  err, state["finished"], rid))
                self._db.commit()
                self.cancel_hooks.pop(rid, None)

        threading.Thread(target=target, daemon=True, name=f"run-{rid}").start()
        return rid

    def get(self, rid: str) -> dict | None:
        with self._lock:
            if rid in self.live:
                s = self.live[rid]
                return {**s, "events": list(s["events"])}
            r = self._db.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
        if not r:
            return None
        d = dict(r)
        for k in ("params", "result", "events"):
            d[k] = json.loads(d[k]) if d[k] else ([] if k == "events" else None)
        return d

    def list(self, kind: str | None = None, limit: int = 100) -> list[dict]:
        with self._lock:
            sql, args = "SELECT id, kind, title, status, error, created, finished FROM runs", []
            if kind:
                sql += " WHERE kind=?"
                args.append(kind)
            rows = self._db.execute(sql + " ORDER BY created DESC LIMIT ?", (*args, limit)).fetchall()
        return [dict(r) for r in rows]

    def delete(self, rid: str) -> bool:
        with self._lock:
            self.live.pop(rid, None)
            n = self._db.execute("DELETE FROM runs WHERE id=?", (rid,)).rowcount
            self._db.commit()
        return n > 0

    def stop(self, rid: str) -> bool:
        hook = self.cancel_hooks.get(rid)
        if hook:
            hook()
            return True
        return False
