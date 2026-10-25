"""Demo agent: a small LangGraph ReAct-style loop around a pluggable brain.

reason -> act -> reason ... until the brain answers, errors, times out, or
hits the step limit. The full trace (thoughts, tool calls, args, results,
errors) is kept in state and returned to the runner.
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    task: Any
    brain: Any
    tools: Any
    workdir: Any
    trace: list
    steps: int
    done: bool
    final_answer: str
    error: str
    timed_out: bool
    started_at: float
    pending: dict | None


def _reason(state: AgentState) -> AgentState:
    if state["done"]:
        return state
    elapsed = time.monotonic() - state["started_at"]
    if elapsed > state["task"].timeout_s:
        state["timed_out"] = True
        state["done"] = True
        state["trace"].append({
            "step": len(state["trace"]) + 1, "type": "error",
            "error": f"run timed out after {state['task'].timeout_s}s",
        })
        return state
    if state["steps"] >= state["task"].max_steps:
        state["done"] = True
        state["error"] = "max steps reached"
        state["trace"].append({
            "step": len(state["trace"]) + 1, "type": "error",
            "error": "max steps reached",
        })
        return state
    try:
        action = state["brain"].next_action(state)
    except Exception as e:  # noqa: BLE001 - a brain crash is a measured outcome
        state["done"] = True
        state["error"] = f"agent crashed: {e}"
        state["trace"].append({
            "step": len(state["trace"]) + 1, "type": "error", "error": str(e),
        })
        return state
    if action["type"] == "answer":
        state["done"] = True
        state["final_answer"] = action["text"]
        state["trace"].append({
            "step": len(state["trace"]) + 1, "type": "answer", "text": action["text"],
        })
    else:
        state["pending"] = action
        state["trace"].append({
            "step": len(state["trace"]) + 1, "type": "thought",
            "text": action.get("thought", ""),
        })
    return state


def _route(state: AgentState) -> str:
    if state["done"] or state["pending"] is None:
        return "end"
    return "act"


def _act(state: AgentState) -> AgentState:
    action = state["pending"]
    start = time.monotonic()
    result, violated = state["tools"].call(action["tool"], action["args"])
    dur_ms = (time.monotonic() - start) * 1000
    state["trace"].append({
        "step": len(state["trace"]) + 1, "type": "tool_call",
        "tool": action["tool"], "args": action["args"],
        "result": result[:2000], "duration_ms": round(dur_ms, 1),
        "policy_violation": violated,
    })
    state["steps"] += 1
    state["pending"] = None
    return state


def build_agent_graph():
    g = StateGraph(AgentState)
    g.add_node("reason", _reason)
    g.add_node("act", _act)
    g.add_conditional_edges("reason", _route, {"act": "act", "end": END})
    g.add_edge("act", "reason")
    g.set_entry_point("reason")
    return g.compile()


def run_agent(task, brain, tools, workdir) -> dict:
    graph = build_agent_graph()
    init: AgentState = {
        "task": task, "brain": brain, "tools": tools, "workdir": workdir,
        "trace": [], "steps": 0, "done": False, "final_answer": "",
        "error": "", "timed_out": False, "started_at": time.monotonic(),
        "pending": None,
    }
    return graph.invoke(init)
