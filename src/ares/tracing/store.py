"""Trace sinks: in-memory (tests), SQLite (persistent), and fan-out to several."""
from __future__ import annotations
import json
import sqlite3
import threading
from pathlib import Path
from typing import Protocol
from ares.tracing.schema import EventKind, RunRecord, TraceEvent


class Tracer(Protocol):
    def start_run(self, run: RunRecord) -> None: ...
    def event(self, ev: TraceEvent) -> None: ...
    def end_run(self, run: RunRecord) -> None: ...


class NullTracer:
    def start_run(self, run: RunRecord) -> None:
        pass

    def event(self, ev: TraceEvent) -> None:
        pass

    def end_run(self, run: RunRecord) -> None:
        pass


class MemoryTracer:
    def __init__(self) -> None:
        self.runs: dict[str, RunRecord] = {}
        self.events: list[TraceEvent] = []

    def start_run(self, run: RunRecord) -> None:
        self.runs[run.run_id] = run

    def event(self, ev: TraceEvent) -> None:
        self.events.append(ev)

    def end_run(self, run: RunRecord) -> None:
        self.runs[run.run_id] = run

    def of_kind(self, kind: EventKind, run_id: str | None = None) -> list[TraceEvent]:
        return [e for e in self.events if e.kind == kind and (run_id is None or e.run_id == run_id)]


class MultiTracer:
    def __init__(self, *tracers: Tracer):
        self.tracers = tracers

    def start_run(self, run: RunRecord) -> None:
        for t in self.tracers:
            t.start_run(run)

    def event(self, ev: TraceEvent) -> None:
        for t in self.tracers:
            t.event(ev)

    def end_run(self, run: RunRecord) -> None:
        for t in self.tracers:
            t.end_run(run)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    variant TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    config TEXT NOT NULL,
    counters TEXT NOT NULL,
    final TEXT,
    meta TEXT NOT NULL,
    schema_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    phase TEXT NOT NULL,
    duration_ms REAL,
    data TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE INDEX IF NOT EXISTS events_kind ON events (run_id, kind);
CREATE INDEX IF NOT EXISTS runs_task ON runs (task_id, variant);
"""


class SQLiteTraceStore:
    """Persists runs and events. Events are committed as they happen, so a crashed run keeps its trace."""

    def __init__(self, path: Path | str):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)

    def start_run(self, run: RunRecord) -> None:
        self._upsert(run)

    def end_run(self, run: RunRecord) -> None:
        self._upsert(run)

    def event(self, ev: TraceEvent) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                (ev.run_id, ev.seq, ev.ts, ev.kind, ev.phase, ev.duration_ms, json.dumps(ev.data, ensure_ascii=False)),
            )
            self._db.commit()

    def _upsert(self, r: RunRecord) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r.run_id, r.task_id, r.variant, r.model, r.status, r.reason, r.started_at, r.ended_at,
                 json.dumps(r.config), json.dumps(r.counters),
                 json.dumps(r.final) if r.final is not None else None, json.dumps(r.meta), r.schema_version),
            )
            self._db.commit()

    def list_runs(self, limit: int = 50, task_id: str | None = None) -> list[RunRecord]:
        sql, args = "SELECT * FROM runs", []
        if task_id:
            sql += " WHERE task_id = ?"
            args.append(task_id)
        sql += " ORDER BY started_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [self._run(row) for row in rows]

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM runs WHERE run_id = ? OR run_id LIKE ?",
                                   (run_id, run_id + "%")).fetchone()
        return self._run(row) if row else None

    def events(self, run_id: str) -> list[TraceEvent]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM events WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall()
        return [
            TraceEvent(run_id=r["run_id"], seq=r["seq"], ts=r["ts"], kind=r["kind"], phase=r["phase"],
                       duration_ms=r["duration_ms"], data=json.loads(r["data"]))
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @staticmethod
    def _run(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"], task_id=row["task_id"], variant=row["variant"], model=row["model"],
            status=row["status"], reason=row["reason"], started_at=row["started_at"], ended_at=row["ended_at"],
            config=json.loads(row["config"]), counters=json.loads(row["counters"]),
            final=json.loads(row["final"]) if row["final"] else None, meta=json.loads(row["meta"]),
            schema_version=row["schema_version"],
        )
