"""Report assembly: per-task pass^k, reliability curve, failure breakdown."""
from __future__ import annotations

from .scoring import FAILURE_CLASSES, pass_k


def build_report(run_id: str, suite_name: str, agent_name: str, n: int,
                 tasks: list, results: list[dict]) -> dict:
    per_task: dict[str, dict] = {}
    for task in tasks:
        rs = [r for r in results if r["task_id"] == task.id]
        passes = sum(1 for r in rs if r["passed"])
        count = len(rs)
        per_task[task.id] = {
            "task_id": task.id,
            "kind": task.kind,
            "prompt": task.prompt,
            "n": count,
            "passes": passes,
            "pass_rate": round(passes / count, 4) if count else 0.0,
            "pass_k": {f"pass^{k}": round(pass_k(passes, count, k), 4)
                       for k in range(1, count + 1)},
            "failures": {cls: sum(1 for r in rs if r["failure_class"] == cls)
                         for cls in FAILURE_CLASSES
                         if any(r["failure_class"] == cls for r in rs)},
            "rollouts": rs,
        }
    total = len(results)
    total_pass = sum(1 for r in results if r["passed"])
    curve = {}
    for k in range(1, n + 1):
        vals = [pass_k(per_task[t.id]["passes"], per_task[t.id]["n"], k) for t in tasks]
        curve[f"pass^{k}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return {
        "run_id": run_id,
        "suite": suite_name,
        "agent": agent_name,
        "n": n,
        "tasks": len(tasks),
        "total_rollouts": total,
        "total_passes": total_pass,
        "overall_pass_rate": round(total_pass / total, 4) if total else 0.0,
        "reliability_curve": curve,
        "failure_totals": {cls: sum(1 for r in results if r["failure_class"] == cls)
                           for cls in FAILURE_CLASSES
                           if any(r["failure_class"] == cls for r in results)},
        "per_task": per_task,
    }


def report_from_db(run: dict) -> dict:
    """Rebuild a report from a stored run row (rollouts carry parsed traces)."""
    tasks = []
    seen = set()
    for r in run["rollouts"]:
        if r["task_id"] not in seen:
            seen.add(r["task_id"])
            tasks.append({"id": r["task_id"], "kind": r["kind"], "prompt": ""})
    # Minimal task stand-ins; prompts are not stored per rollout.
    task_objs = [type("T", (), t) for t in tasks]
    results = []
    for r in run["rollouts"]:
        results.append({
            "task_id": r["task_id"], "kind": r["kind"], "rollout": r["rollout"],
            "passed": r["passed"], "failure_class": r["failure_class"],
            "failed_checks": [], "duration_ms": r["duration_ms"],
            "final_answer": r["final_answer"], "error": "",
            "timed_out": False, "trace": r["trace"],
        })
    return build_report(run_id=run["id"], suite_name=run["suite"],
                        agent_name=run["agent"], n=run["n"],
                        tasks=task_objs, results=results)
