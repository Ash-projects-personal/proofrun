# ProofRun

ResolveBench for your own agent.

I kept watching agents pass a demo once and fail in front of a customer. Public
benchmarks test the model, not my agent, my tools, or my prompts. So I built
the tool I wanted: point it at your agent and your tasks, it runs the agent
N times per task, and it tells you whether the agent holds up, where it
breaks, and whether your fix actually helped.

## What it does

- **Runs your agent N times per task.** Default N is 8. Each rollout gets its
  own isolated working directory, its own timeout, and a full trace of every
  tool call, every argument, and every error.
- **Scores pass^k reliability.** pass^k is the probability that all k
  independent rollouts pass, estimated the tau-bench / ResolveBench way.
  pass^8 is 1 for a task only when all 8 rollouts pass.
- **Classifies every failure.** wrong tool, bad arguments, hallucinated
  success (claimed done, did nothing), policy violation, timeout, crash.
  If no rule explains a failure, it says unknown instead of guessing.
- **Shows the work.** A dashboard with the reliability curve per task, the
  failure breakdown, and the worst failing traces, each expandable to the
  full step list. Every report exports as a single JSON file.
- **Closes the loop.** Change the agent, re-run, and see the pass rate move.
  That is the whole point.

## Measured results

I ran the starter suite (10 tasks, 8 rollouts each, 80 rollouts total) through
two versions of the demo agent, both driven by the same seed base so the same
random draws drove both runs. These numbers come from `runs/paired-v1.json`
and `runs/paired-v2.json`. Nothing here is typed by hand.

| | demo-v1 | demo-v2 |
|---|---|---|
| Overall pass rate | 45/80 (56.25%) | 68/80 (85.00%) |
| pass^8 (all 8 pass) | 0% (0 of 10 tasks) | 20% (2 of 10 tasks) |
| pass^4 | 8.71% | 50.00% |
| Hallucinated success | 9 | 0 |
| Wrong tool | 9 | 4 |
| Bad arguments | 10 | 5 |
| Policy violation | 6 | 2 |
| Crash | 1 | 1 |

demo-v2 differs from demo-v1 in two ways: it fails less often to begin with,
and it adds a verify-and-retry step (it reads the output file back, and if
the file does not match what was written, it runs the plan once more). The
gain above is the combined effect. The part I find most useful: verification
alone killed the hallucinated-success class entirely, 9 to 0. It cannot fix
wrong reasoning, and the numbers show that too: wrong tool and bad arguments
only fell by about half.

Per-task pass rates for demo-v2: r1 8/8, r2 7/8, r3 8/8, d1 5/8, d2 7/8,
d3 6/8, d4 7/8, a1 7/8, a2 7/8, a3 6/8. The data-code tasks are the weakest
for this agent, which is useful to know.

## The demo agent

The demo agent is a deterministic stand-in, not a real LLM agent. There is
no model API key in this environment, so the demo brain parses the task
prompt with fixed rules, builds a plan of tool calls, and executes it. One
failure mode is drawn per rollout from a seeded RNG, so the harness measures
real failures instead of staged ones. The pass rates above are real
measurements of that stand-in, not of a frontier model.

The harness itself is model-agnostic. It runs any LangGraph agent through
the same interface: a brain that returns the next tool call or a final
answer. Swap in a real LLM-backed agent and the same reports come out.

## How to run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run the starter suite (10 tasks, 8 rollouts each)
.venv/bin/python -m proofrun run --agent demo-v1 --n 8

# Run the fixed agent on the same draws, for a paired comparison
.venv/bin/python -m proofrun run --agent demo-v2 --n 8 --seed-base paired-2026-10-25

# Serve the dashboard (port 8100; port 8000 is taken by another local app here)
.venv/bin/python -m proofrun serve --port 8100
```

The dashboard is at http://127.0.0.1:8100. Reports land in `runs/<run-id>/report.json`
and in SQLite at `runs/proofrun.db`. The API also works directly:
`GET /api/runs`, `GET /api/runs/<id>`, `GET /api/runs/<id>/export`,
`POST /api/runs` with `{"agent": "demo-v1", "n": 8}`.

## Write your own tasks

A suite is one JSON file. Each task has a prompt, a kind, and checks. Every
check is machine-checkable, no LLM judge.

```json
{
  "id": "r1",
  "kind": "research",
  "prompt": "Search the documents for the capital of France. Write the answer to answer.txt.",
  "checks": [
    {"type": "file_contains", "path": "answer.txt", "text": "Paris"},
    {"type": "tool_called", "tool": "search_docs", "args": {"query": "the capital of France"}}
  ]
}
```

Check types: `file_contains`, `file_equals`, `tool_called` (args are a subset
match). Task kinds: `research`, `data-calc`, `data-code`, `tool-api`. A task
can ship setup files that are copied into the rollout workdir before the
agent starts. See `tasks/starter.json` for all 10 starter tasks.

The demo toolbelt: `search_docs`, `calculator`, `run_python`, `write_file`,
`read_file`, `mock_api_get`, `list_files`. All file access stays inside the
run workdir; a path that escapes it is blocked and recorded as a policy
violation.

## Limits, stated plainly

- It measures reliability on your tasks, not on the world. A high pass rate
  on 10 easy tasks says nothing about production.
- It does not fix agents. It finds where they break.
- The demo agent is a stand-in. The numbers above describe it, not a real
  model. Plug in a real agent before quoting any pass rate to anyone.
- The workdir boundary is a working-directory rule, not a security sandbox.
  Do not run untrusted agent code through `run_python` and call it safe.
- pass^k with N=8 is a coarse estimate. Eight rollouts per task is the
  minimum for a pass^8 reading, not a lot of evidence.

## Credits

pass^k comes from tau-bench (Sierra) and ResolveBench, who defined the
metric and the failure-mechanism taxonomy it is paired with here. The
failure classes in this tool (wrong tool, bad arguments, hallucinated
success, timeout) follow the same shape as ResolveBench's four mechanisms.
ProofRun is not affiliated with either project. It takes their benchmark
idea and turns it into a tool for your agent on your tasks.
