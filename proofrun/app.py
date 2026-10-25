"""FastAPI backend: run list, full reports, JSON export, dashboard page."""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from . import store as store_mod
from .report import report_from_db
from .runner import run_suite

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "runs" / "proofrun.db"
TASKS_PATH = BASE / "tasks" / "starter.json"
CORPUS_DIR = BASE / "data" / "corpus"
STORE_FILE = BASE / "data" / "store.json"
RUNS_ROOT = BASE / "runs"

app = FastAPI(title="ProofRun")
_jobs: dict[str, str] = {}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/runs")
def list_runs():
    return store_mod.list_runs(DB_PATH)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = store_mod.get_run(DB_PATH, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return report_from_db(run)


@app.get("/api/runs/{run_id}/export")
def export_run(run_id: str):
    run = store_mod.get_run(DB_PATH, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    report = report_from_db(run)
    return JSONResponse(
        content=report,
        headers={"Content-Disposition": f'attachment; filename="proofrun-{run_id}.json"'},
    )


@app.post("/api/runs")
def start_run(body: dict):
    agent = body.get("agent", "demo-v1")
    n = int(body.get("n", 8))
    if agent not in ("demo-v1", "demo-v2"):
        raise HTTPException(status_code=400, detail="agent must be demo-v1 or demo-v2")
    if not 1 <= n <= 32:
        raise HTTPException(status_code=400, detail="n must be between 1 and 32")
    import uuid

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    _jobs[run_id] = "running"

    def _work():
        try:
            run_suite(TASKS_PATH, agent, n, RUNS_ROOT, DB_PATH, CORPUS_DIR, STORE_FILE,
                      run_id=run_id)
            _jobs[run_id] = "done"
        except Exception as e:  # noqa: BLE001 - record the failure, don't crash the server
            _jobs[run_id] = f"error: {e}"

    threading.Thread(target=_work, daemon=True).start()
    return {"run_id": run_id, "status": "running"}


@app.get("/api/runs/{run_id}/status")
def run_status(run_id: str):
    return {"run_id": run_id, "status": _jobs.get(run_id, "unknown")}


@app.get("/")
def dashboard():
    return FileResponse(BASE / "proofrun" / "static" / "dashboard.html")
