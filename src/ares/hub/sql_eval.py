"""Execution-accuracy evaluation for the database agent on the sample shop database."""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from ares.hub.database import ask_database, run_query


def _norm_rows(rows: list[list]) -> list[tuple]:
    out = []
    for r in rows:
        out.append(tuple(round(v, 2) if isinstance(v, float) else (str(v).strip().lower() if isinstance(v, str) else v)
                         for v in r))
    return out


def results_match(pred: list[list], gold: list[list]) -> bool:
    """Same multiset of rows; a single-column gold result also matches if the prediction contains that column."""
    p, g = _norm_rows(pred), _norm_rows(gold)
    if sorted(map(repr, p)) == sorted(map(repr, g)):
        return True
    if g and all(len(r) == 1 for r in g) and p and len(p) == len(g):
        gv = sorted(repr(r[0]) for r in g)
        for i in range(len(p[0])):
            if sorted(repr(r[i]) for r in p) == gv:
                return True
    return False


def evaluate_sql(db: Path, eval_file: Path, chat: Callable, on_event: Callable[[dict], None] | None = None) -> dict:
    emit = on_event or (lambda e: None)
    items = [json.loads(x) for x in eval_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = []
    for n, it in enumerate(items, 1):
        t0 = time.time()
        res = ask_database(db, it["question"], chat)
        gold = run_query(db, it["gold_sql"])["rows"]
        ok = res["status"] == "ok" and results_match(res["result"]["rows"], gold)
        row = {"question": it["question"], "correct": ok, "sql": res.get("sql"), "repaired": res.get("repaired", False),
               "status": res["status"], "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        emit({"type": "eval_item", "n": n, "total": len(items), **row})
    acc = sum(r["correct"] for r in rows) / len(rows)
    return {"execution_accuracy": round(acc, 3), "correct": sum(r["correct"] for r in rows), "total": len(rows),
            "repaired": sum(r["repaired"] for r in rows), "avg_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 1),
            "items": rows}
