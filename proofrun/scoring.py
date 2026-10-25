"""Scoring: pass^k reliability and the failure taxonomy.

pass^k is the tau-bench / ResolveBench estimator: the probability that all
k independent rollouts pass, estimated as C(passes, k) / C(N, k). With N=8,
pass^8 is 1 for a task only when all 8 rollouts pass.

Failure taxonomy (rule-based, from the trace only):
  timeout               the run exceeded its per-run timeout
  crash                 the agent raised an exception
  policy_violation      the agent tried to act outside the working directory
  hallucinated_success  claimed done without producing the checked artifact
  wrong_tool            never called the tool the task needs
  bad_arguments         called the needed tool with the wrong arguments
  unknown               failed, and no rule above explains it
"""
from __future__ import annotations

import math

from .brain import expected_first_call

REQUIRED_TOOL = {
    "research": "search_docs",
    "data-calc": "calculator",
    "data-code": "run_python",
    "tool-api": "mock_api_get",
}

FAILURE_CLASSES = [
    "wrong_tool", "bad_arguments", "hallucinated_success",
    "policy_violation", "timeout", "crash", "unknown",
]


def pass_k(passes: int, n: int, k: int) -> float:
    if k > n or passes < k:
        return 0.0
    return math.comb(passes, k) / math.comb(n, k)


def _args_mismatch(task, req_calls: list[dict]) -> bool:
    expected = expected_first_call(task)
    if expected is None:
        return False
    _, args = expected
    if task.kind == "data-code":
        for call in req_calls:
            code = call.get("args", {}).get("code", "")
            op = args.get("operation")
            if op == "average" and "/ len(" in code and args.get("file") in code:
                return False
            if op == "sum" and "/ len(" not in code and args.get("file") in code:
                return False
        return True
    for call in req_calls:
        cargs = call.get("args", {})
        if all(cargs.get(k) == v for k, v in args.items()):
            return False
    return True


def classify(task, trace: list[dict], passed: bool, timed_out: bool, crashed: bool):
    if timed_out:
        return "timeout"
    if crashed:
        return "crash"
    if any(s.get("policy_violation") for s in trace if s.get("type") == "tool_call"):
        return "policy_violation"
    if passed:
        return None
    calls = [s for s in trace if s.get("type") == "tool_call"]
    if not calls:
        return "hallucinated_success"
    expected_paths = {c.path for c in task.checks if c.type in ("file_contains", "file_equals")}
    wrote_artifact = any(
        c.get("tool") == "write_file" and c.get("args", {}).get("path") in expected_paths
        for c in calls
    )
    if not wrote_artifact:
        # Wrote nothing at all: claimed done without doing the work.
        # Wrote to some other path: the filename argument was wrong.
        if any(c.get("tool") == "write_file" for c in calls):
            return "bad_arguments"
        return "hallucinated_success"
    req = REQUIRED_TOOL.get(task.kind)
    req_calls = [c for c in calls if c.get("tool") == req]
    if not req_calls:
        return "wrong_tool"
    if _args_mismatch(task, req_calls):
        return "bad_arguments"
    for c in req_calls:
        result = str(c.get("result", ""))
        if result.startswith("Error") or "No matching documents" in result:
            return "bad_arguments"
    return "unknown"
