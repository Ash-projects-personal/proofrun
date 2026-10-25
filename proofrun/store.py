"""SQLite persistence for runs and rollouts (stdlib sqlite3)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  started_at TEXT,
  suite TEXT,
  agent TEXT,
  n INTEGER,
  status TEXT
);
CREATE TABLE IF NOT EXISTS rollouts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT,
  task_id TEXT,
  kind TEXT,
  rollout INTEGER,
  passed INTEGER,
  failure_class TEXT,
  duration_ms REAL,
  final_answer TEXT,
  trace TEXT,
  FOREIGN KEY (run_id) REFERENCES runs(id)
);
"""


def _connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def init_db(db_path: str | Path) -> None:
    _connect(db_path).close()


def create_run(db_path: str | Path, run_id: str, suite: str, agent: str, n: int,
               started_at: str) -> None:
    conn = _connect(db_path)
    conn.execute(
        "INSERT INTO runs (id, started_at, suite, agent, n, status) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, started_at, suite, agent, n, "running"),
    )
    conn.commit()
    conn.close()


def save_rollout(db_path: str | Path, run_id: str, result: dict) -> None:
    conn = _connect(db_path)
    conn.execute(
        """INSERT INTO rollouts
           (run_id, task_id, kind, rollout, passed, failure_class,
            duration_ms, final_answer, trace)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, result["task_id"], result["kind"], result["rollout"],
         1 if result["passed"] else 0, result["failure_class"],
         result["duration_ms"], result["final_answer"],
         json.dumps(result["trace"])),
    )
    conn.commit()
    conn.close()


def finish_run(db_path: str | Path, run_id: str, status: str) -> None:
    conn = _connect(db_path)
    conn.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))
    conn.commit()
    conn.close()


def get_run(db_path: str | Path, run_id: str) -> dict | None:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        conn.close()
        return None
    run = dict(row)
    rows = conn.execute(
        "SELECT * FROM rollouts WHERE run_id = ? ORDER BY task_id, rollout",
        (run_id,),
    ).fetchall()
    conn.close()
    run["rollouts"] = []
    for r in rows:
        d = dict(r)
        d["passed"] = bool(d["passed"])
        d["trace"] = json.loads(d["trace"])
        run["rollouts"].append(d)
    return run


def list_runs(db_path: str | Path) -> list[dict]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]
