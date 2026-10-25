"""Task suite format and check evaluation.

A task suite is a JSON file: {"name": ..., "tasks": [Task, ...]}.
A Task has a natural-language prompt, a kind, and a list of checks.
Every check is machine-checkable: no LLM judge anywhere in the loop.

Check types:
  file_contains  the file exists and contains the given text
  file_equals    the file exists and its stripped content equals the text
  tool_called    the tool was called with the given args (subset match)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Check:
    type: str  # file_contains | file_equals | tool_called
    path: str = ""  # relative to the run workdir (file checks)
    text: str = ""  # expected substring / exact content
    tool: str = ""  # tool name (tool_called checks)
    args: dict = field(default_factory=dict)  # args that must match (subset)

    @classmethod
    def from_dict(cls, d: dict) -> "Check":
        return cls(
            type=d["type"],
            path=d.get("path", ""),
            text=d.get("text", ""),
            tool=d.get("tool", ""),
            args=d.get("args", {}),
        )


@dataclass
class Task:
    id: str
    kind: str  # research | data-calc | data-code | tool-api
    prompt: str
    checks: list[Check]
    timeout_s: float = 30.0
    max_steps: int = 12
    setup_files: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            id=d["id"],
            kind=d["kind"],
            prompt=d["prompt"],
            checks=[Check.from_dict(c) for c in d["checks"]],
            timeout_s=d.get("timeout_s", 30.0),
            max_steps=d.get("max_steps", 12),
            setup_files=d.get("setup_files", {}),
        )


def load_suite(path: str | Path) -> tuple[str, list[Task]]:
    with open(path) as f:
        raw = json.load(f)
    return raw.get("name", Path(path).stem), [Task.from_dict(t) for t in raw["tasks"]]


def evaluate_checks(task: Task, trace: list[dict], workdir: Path) -> tuple[bool, list[str]]:
    """Run every check against the workdir and the trace.

    Returns (passed, failed check descriptions).
    """
    failed: list[str] = []
    tool_calls = [s for s in trace if s.get("type") == "tool_call"]
    for check in task.checks:
        if check.type == "file_contains":
            p = workdir / check.path
            if not p.exists():
                failed.append(f"file_contains: {check.path} does not exist")
            elif check.text not in p.read_text(errors="replace"):
                failed.append(f"file_contains: {check.path} does not contain {check.text!r}")
        elif check.type == "file_equals":
            p = workdir / check.path
            if not p.exists():
                failed.append(f"file_equals: {check.path} does not exist")
            elif p.read_text(errors="replace").strip() != check.text.strip():
                failed.append(f"file_equals: {check.path} != {check.text!r}")
        elif check.type == "tool_called":
            ok = False
            for call in tool_calls:
                if call.get("tool") != check.tool:
                    continue
                args = call.get("args", {})
                if all(args.get(k) == v for k, v in check.args.items()):
                    ok = True
                    break
            if not ok:
                failed.append(f"tool_called: {check.tool} with args {check.args} never called")
        else:
            failed.append(f"unknown check type: {check.type}")
    return (len(failed) == 0), failed
