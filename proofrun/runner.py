"""Suite runner: a LangGraph graph that walks (task, rollout) pairs.

dispatch -> execute -> dispatch ... -> summarize. Each rollout gets its own
isolated workdir, its own seeded brain, and its own trace. The runner
evaluates the checks, classifies the failure, and persists the rollout
before moving on, so a crash mid-run never loses completed work.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from . import scoring, store as store_mod
from .agent import run_agent
from .brain import CONFIGS, SimulatedBrain, seed_for
from .report import build_report
from .tasks import evaluate_checks, load_suite
from .tools import Toolbelt


class RunState(TypedDict):
    run_id: str
    seed_base: str
    suite_name: str
    tasks: list
    config_name: str
    n: int
    queue: list
    results: list
    store_path: str
    runs_root: str
    corpus_dir: str
    store_file: str
    report: dict | None


def _dispatch(state: RunState) -> RunState:
    return state


def _route_dispatch(state: RunState) -> str:
    return "execute" if state["queue"] else "summarize"


def _execute(state: RunState) -> dict:
    task, rollout = state["queue"].pop(0)
    run_id = state["run_id"]
    seed = seed_for(state.get("seed_base") or run_id, task.id, rollout)
    config = CONFIGS[state["config_name"]]
    brain = SimulatedBrain(config, seed)
    workdir = Path(state["runs_root"]) / run_id / task.id / f"r{rollout}"
    workdir.mkdir(parents=True, exist_ok=True)
    for name, content in task.setup_files.items():
        (workdir / name).write_text(content)
    tools = Toolbelt(workdir, Path(state["corpus_dir"]), Path(state["store_file"]))
    start = time.monotonic()
    final = run_agent(task, brain, tools, workdir)
    duration_ms = (time.monotonic() - start) * 1000
    passed, failed = evaluate_checks(task, final["trace"], workdir)
    crashed = bool(final["error"])
    policy_hit = any(
        s.get("policy_violation") for s in final["trace"] if s.get("type") == "tool_call"
    )
    if policy_hit:
        passed = False
        failed.append("policy violation: agent attempted an action outside the working directory")
    failure = scoring.classify(task, final["trace"], passed, final["timed_out"], crashed)
    result = {
        "task_id": task.id, "kind": task.kind, "rollout": rollout,
        "passed": passed, "failure_class": failure, "failed_checks": failed,
        "duration_ms": round(duration_ms, 1),
        "final_answer": final["final_answer"], "error": final["error"],
        "timed_out": final["timed_out"], "trace": final["trace"],
    }
    state["results"].append(result)
    store_mod.save_rollout(state["store_path"], run_id, result)
    return {"queue": state["queue"], "results": state["results"]}


def _summarize(state: RunState) -> dict:
    report = build_report(
        run_id=state["run_id"], suite_name=state["suite_name"],
        agent_name=state["config_name"], n=state["n"],
        tasks=state["tasks"], results=state["results"],
    )
    store_mod.finish_run(state["store_path"], state["run_id"], "done")
    out = Path(state["runs_root"]) / state["run_id"] / "report.json"
    out.write_text(json.dumps(report, indent=2))
    return {"report": report}


def run_suite(tasks_path: str | Path, agent_name: str, n: int, runs_root: str | Path,
              db_path: str | Path, corpus_dir: str | Path, store_file: str | Path,
              run_id: str | None = None, seed_base: str | None = None) -> tuple[str, dict]:
    run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
    suite_name, tasks = load_suite(tasks_path)
    started_at = datetime.now(timezone.utc).isoformat()
    store_mod.init_db(db_path)
    store_mod.create_run(db_path, run_id, suite_name, agent_name, n, started_at)
    queue = [(t, i) for t in tasks for i in range(n)]
    g = StateGraph(RunState)
    g.add_node("dispatch", _dispatch)
    g.add_node("execute", _execute)
    g.add_node("summarize", _summarize)
    g.add_conditional_edges("dispatch", _route_dispatch,
                            {"execute": "execute", "summarize": "summarize"})
    g.add_edge("execute", "dispatch")
    g.set_entry_point("dispatch")
    graph = g.compile()
    init: RunState = {
        "run_id": run_id, "seed_base": seed_base or run_id,
        "suite_name": suite_name, "tasks": tasks,
        "config_name": agent_name, "n": n, "queue": queue, "results": [],
        "store_path": str(db_path), "runs_root": str(runs_root),
        "corpus_dir": str(corpus_dir), "store_file": str(store_file),
        "report": None,
    }
    final = graph.invoke(init)
    return run_id, final["report"]
