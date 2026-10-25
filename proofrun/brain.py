"""Demo agent brain: a deterministic, seeded stand-in for a real LLM agent.

No LLM API key is available in this environment, so the demo agent does not
call a model. It parses the task prompt with fixed rules, builds a plan of
tool calls, and executes it. One failure mode is drawn per rollout from a
seeded RNG, so the harness measures real failures instead of staged ones.

The brain is a stand-in. The harness is model-agnostic: any LangGraph agent
that follows the same next-action interface can be run through it. The two
demo configs differ in a real way: demo-v2 adds a verify-and-retry step
(read the output file back, and if it does not match what was written, run
the plan once more). The harness measures whether that change helps.
"""
from __future__ import annotations

import hashlib
import json
import random
import re


class BrainConfig:
    def __init__(self, name: str, probs: dict[str, float], verify: bool = False):
        self.name = name
        self.probs = probs
        self.verify = verify
        assert abs(sum(probs.values()) - 1.0) < 1e-9, "probabilities must sum to 1"


DEMO_V1 = BrainConfig("demo-v1", {
    "clean": 0.55, "wrong_tool": 0.10, "bad_args": 0.12,
    "hallucinate": 0.10, "policy": 0.08, "crash": 0.05,
})
DEMO_V2 = BrainConfig("demo-v2", {
    "clean": 0.80, "wrong_tool": 0.05, "bad_args": 0.05,
    "hallucinate": 0.04, "policy": 0.03, "crash": 0.03,
}, verify=True)

CONFIGS = {"demo-v1": DEMO_V1, "demo-v2": DEMO_V2}

WRONG_TOOL = {
    "search_docs": ("calculator", {"expression": "2 + 2"}),
    "calculator": ("search_docs", {"query": "the"}),
    "run_python": ("calculator", {"expression": "2 + 2"}),
    "mock_api_get": ("search_docs", {"query": "the"}),
}


def seed_for(seed_base: str, task_id: str, rollout: int) -> int:
    h = hashlib.sha256(f"{seed_base}|{task_id}|{rollout}".encode()).hexdigest()
    return int(h[:8], 16)


def _draw_mode(rng: random.Random, probs: dict[str, float]) -> str:
    roll = rng.random()
    acc = 0.0
    for mode, p in probs.items():
        acc += p
        if roll < acc:
            return mode
    return "clean"


# -- prompt parsing -------------------------------------------------------
# The demo prompts follow fixed patterns; the stand-in parses them with
# regexes. A real LLM agent would read the same prompts directly.

def _clean_filename(s: str) -> str:
    # Prompts end in a period ("Write the answer to answer.txt."),
    # and (\S+) grabs it. Strip sentence punctuation, not the extension.
    return s.rstrip(".,;:")


def parse_research(prompt: str):
    m = re.search(r"Search the documents for (.+?)\.\s*Write the answer to (\S+)", prompt)
    if not m:
        return None
    return {"query": m.group(1).strip(), "outfile": _clean_filename(m.group(2).strip())}


def parse_calc(prompt: str):
    m = re.search(r"Use the calculator to compute (.+?)\.\s*Write the result to (\S+)", prompt)
    if not m:
        return None
    return {"expression": m.group(1).strip(), "outfile": _clean_filename(m.group(2).strip())}


def parse_code(prompt: str):
    m = re.search(
        r"Run Python code to compute the (\w+) of .*? in (\S+)\.\s*Write the result to (\S+)",
        prompt,
    )
    if not m:
        return None
    return {"operation": m.group(1).strip(), "file": m.group(2).strip(),
            "outfile": _clean_filename(m.group(3).strip())}


def parse_api(prompt: str):
    m = re.search(
        r"Use the store API to get the (\w+) of product (\S+)\.\s*Write the result to (\S+)",
        prompt,
    )
    if m:
        return {"kind": "product", "field": m.group(1).strip(),
                "product": m.group(2).strip(),
                "outfile": _clean_filename(m.group(3).strip())}
    m = re.search(
        r"Use the store API to count the orders for user (\S+)\.\s*Write the result to (\S+)",
        prompt,
    )
    if m:
        return {"kind": "orders", "user": m.group(1).strip(),
                "outfile": _clean_filename(m.group(2).strip())}
    return None


def expected_first_call(task) -> tuple[str, dict] | None:
    """The tool call a correct agent makes first, derived from the prompt.

    Used by the failure classifier, not by the agent.
    """
    if task.kind == "research":
        p = parse_research(task.prompt)
        return ("search_docs", {"query": p["query"]}) if p else None
    if task.kind == "data-calc":
        p = parse_calc(task.prompt)
        return ("calculator", {"expression": p["expression"]}) if p else None
    if task.kind == "data-code":
        p = parse_code(task.prompt)
        return ("run_python", {"operation": p["operation"], "file": p["file"]}) if p else None
    if task.kind == "tool-api":
        p = parse_api(task.prompt)
        if not p:
            return None
        if p["kind"] == "product":
            return ("mock_api_get", {"endpoint": "/product", "params": {"id": p["product"]}})
        return ("mock_api_get", {"endpoint": "/orders", "params": {"user": p["user"]}})
    return None


def code_for(operation: str, filename: str) -> str:
    if operation == "average":
        return (
            f"nums = [float(x) for x in open('{filename}').read().split()]\n"
            "print(sum(nums) / len(nums))"
        )
    if operation == "sum":
        return (
            f"nums = [float(x) for x in open('{filename}').read().split()]\n"
            "print(sum(nums))"
        )
    if operation == "maximum":
        return (
            f"nums = [float(x) for x in open('{filename}').read().split()]\n"
            "print(max(nums))"
        )
    return f"print('unknown operation {operation}')"


class SimulatedBrain:
    """Parses the prompt, builds a plan, executes it with one seeded failure mode."""

    def __init__(self, config: BrainConfig, seed: int):
        self.config = config
        self.rng = random.Random(seed)
        self.mode = _draw_mode(self.rng, config.probs)
        self.plan: list[dict] = []
        self.idx = 0
        self.probed_policy = False
        self.crashed = False
        self.verified = False
        self.expecting_verify = False
        self.retried = False
        self.last_result: str | None = None
        self.written_content: str | None = None
        self.outfile: str | None = None
        self._plan_built = False

    # -- plan -----------------------------------------------------------
    def build_plan(self, task) -> None:
        if task.kind == "research":
            p = parse_research(task.prompt)
            self.outfile = p["outfile"]
            self.plan = [
                {"tool": "search_docs", "args": {"query": p["query"]}},
                {"kind": "write", "path": p["outfile"]},
            ]
        elif task.kind == "data-calc":
            p = parse_calc(task.prompt)
            self.outfile = p["outfile"]
            self.plan = [
                {"tool": "calculator", "args": {"expression": p["expression"]}},
                {"kind": "write", "path": p["outfile"]},
            ]
        elif task.kind == "data-code":
            p = parse_code(task.prompt)
            self.outfile = p["outfile"]
            self.plan = [
                {"tool": "run_python", "args": {"code": code_for(p["operation"], p["file"])}},
                {"kind": "write", "path": p["outfile"]},
            ]
        elif task.kind == "tool-api":
            p = parse_api(task.prompt)
            self.outfile = p["outfile"]
            if p["kind"] == "product":
                self.plan = [
                    {"tool": "mock_api_get",
                     "args": {"endpoint": "/product", "params": {"id": p["product"]}}},
                    {"kind": "write", "path": p["outfile"], "field": p["field"]},
                ]
            else:
                self.plan = [
                    {"tool": "mock_api_get",
                     "args": {"endpoint": "/orders", "params": {"user": p["user"]}}},
                    {"kind": "write", "path": p["outfile"], "field": "count"},
                ]
        self._plan_built = True
        self._apply_mode_to_plan()

    def _apply_mode_to_plan(self) -> None:
        """Corrupt the plan according to the drawn failure mode."""
        if self.mode == "wrong_tool":
            first = self.plan[0]
            wrong, args = WRONG_TOOL[first["tool"]]
            self.plan[0] = {"tool": wrong, "args": args}
        elif self.mode == "bad_args":
            first = self.plan[0]
            if first["tool"] == "search_docs":
                first["args"] = {"query": first["args"]["query"] + " (wrong)"}
            elif first["tool"] == "calculator":
                first["args"] = {"expression": first["args"]["expression"] + " + 1"}
            elif first["tool"] == "run_python":
                code = first["args"]["code"]
                if "/ len(" in code:
                    code = code.replace("sum(nums) / len(nums)", "sum(nums)")
                else:
                    code = code.replace("print(sum(nums))", "print(sum(nums) / len(nums))")
                first["args"] = {"code": code}
            elif first["tool"] == "mock_api_get":
                params = dict(first["args"]["params"])
                if "id" in params:
                    params["id"] = "P-999"
                elif "user" in params:
                    params["user"] = "U-999"
                first["args"] = {"endpoint": first["args"]["endpoint"], "params": params}

    # -- execution ------------------------------------------------------
    def _refresh_last_result(self, trace: list[dict]) -> None:
        for step in reversed(trace):
            if step.get("type") == "tool_call":
                self.last_result = step.get("result")
                return

    def _write_content(self, step: dict) -> str:
        raw = self.last_result or ""
        field = step.get("field")
        if field:
            try:
                data = json.loads(raw)
                return str(data.get(field, raw))
            except (json.JSONDecodeError, AttributeError):
                return raw
        return raw

    def _retry(self, task) -> None:
        self.retried = True
        self.mode = _draw_mode(self.rng, self.config.probs)
        self.idx = 0
        self.verified = False
        self.probed_policy = False
        self.crashed = False
        self.written_content = None
        self.expecting_verify = False
        self.build_plan(task)

    def _verify_read(self) -> dict:
        self.verified = True
        self.expecting_verify = True
        return {"type": "tool", "tool": "read_file", "args": {"path": self.outfile},
                "thought": "Verifying the output file before finishing."}

    def next_action(self, state: dict) -> dict:
        """Return a tool action or a final answer. May raise RuntimeError (crash mode)."""
        task = state["task"]
        trace = state["trace"]
        if not self._plan_built:
            self.build_plan(task)
        if self.mode == "crash" and not self.crashed:
            self.crashed = True
            raise RuntimeError("simulated agent crash")
        if self.mode == "policy" and not self.probed_policy:
            self.probed_policy = True
            return {"type": "tool", "tool": "read_file", "args": {"path": "/etc/passwd"},
                    "thought": "Let me check the system file first."}
        self._refresh_last_result(trace)
        if self.expecting_verify:
            self.expecting_verify = False
            got = (self.last_result or "").strip()
            want = (self.written_content or "").strip()
            if got != want:
                if not self.retried:
                    self._retry(task)
                    return self.next_action(state)
            return {"type": "answer", "text": "Done, I completed the task."}
        if self.mode == "hallucinate" and self.idx == 0:
            # Never did the work. demo-v2 catches this with verification.
            if self.config.verify and not self.verified:
                return self._verify_read()
            return {"type": "answer", "text": "Done, I completed the task."}
        if self.idx < len(self.plan):
            step = self.plan[self.idx]
            self.idx += 1
            if step.get("kind") == "write":
                content = self._write_content(step)
                self.written_content = content
                return {"type": "tool", "tool": "write_file",
                        "args": {"path": step["path"], "content": content},
                        "thought": f"Writing the result to {step['path']}."}
            return {"type": "tool", "tool": step["tool"], "args": dict(step["args"]),
                    "thought": f"Calling {step['tool']}."}
        # Plan finished.
        if self.config.verify and not self.verified:
            return self._verify_read()
        return {"type": "answer", "text": "Done, I completed the task."}
