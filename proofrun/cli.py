"""CLI: run a suite from the terminal, or serve the dashboard.

Usage:
  python -m proofrun run --agent demo-v1 --n 8
  python -m proofrun run --agent demo-v2 --n 8 --out runs/report-v2.json
  python -m proofrun serve --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def _cmd_run(args) -> int:
    from .runner import run_suite

    run_id, report = run_suite(
        tasks_path=BASE / "tasks" / "starter.json",
        agent_name=args.agent,
        n=args.n,
        runs_root=BASE / "runs",
        db_path=BASE / "runs" / "proofrun.db",
        corpus_dir=BASE / "data" / "corpus",
        store_file=BASE / "data" / "store.json",
        seed_base=args.seed_base,
    )
    out = Path(args.out) if args.out else BASE / "runs" / run_id / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"run: {run_id}")
    print(f"agent: {args.agent}  tasks: {report['tasks']}  n: {args.n}")
    print(f"overall pass rate: {report['overall_pass_rate']:.2%} "
          f"({report['total_passes']}/{report['total_rollouts']})")
    print("reliability curve: " + ", ".join(
        f"{k}={v:.2%}" for k, v in report["reliability_curve"].items()))
    print(f"failures: {report['failure_totals']}")
    print(f"report: {out}")
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("proofrun.app:app", host="127.0.0.1", port=args.port, log_level="info")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="proofrun")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run the starter suite through the harness")
    p_run.add_argument("--agent", default="demo-v1", choices=["demo-v1", "demo-v2"])
    p_run.add_argument("--n", type=int, default=8)
    p_run.add_argument("--out", default=None, help="copy the report JSON here too")
    p_run.add_argument("--seed-base", default=None,
                       help="seed base for reproducible rollouts; same base + same "
                            "suite = same failure-mode draws across agent configs")
    p_run.set_defaults(func=_cmd_run)
    p_serve = sub.add_parser("serve", help="serve the dashboard")
    p_serve.add_argument("--port", type=int, default=8100,
                         help="default 8100; port 8000 is taken by another local app here")
    p_serve.set_defaults(func=_cmd_serve)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
