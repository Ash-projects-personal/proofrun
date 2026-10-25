"""Sandboxed toolbelt for the demo agent.

Every tool reads and writes only inside the run's workdir. A path that
escapes the workdir is blocked and recorded as a policy violation. This is
a working-directory boundary, not a security sandbox; the README says so.
"""
from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path


class Toolbelt:
    def __init__(self, workdir: Path, corpus_dir: Path, store_path: Path):
        self.workdir = workdir
        self.corpus_dir = corpus_dir
        self.store_path = store_path
        self._store = json.loads(store_path.read_text())

    # -- policy ---------------------------------------------------------
    def _safe_path(self, rel: str) -> Path | None:
        p = (self.workdir / rel).resolve()
        try:
            p.relative_to(self.workdir.resolve())
        except ValueError:
            return None
        return p

    # -- tools ----------------------------------------------------------
    def search_docs(self, query: str) -> str:
        hits = []
        for f in sorted(self.corpus_dir.glob("*.txt")):
            for line in f.read_text().splitlines():
                if all(w.lower() in line.lower() for w in query.split()):
                    hits.append(line.strip())
        if not hits:
            return "No matching documents found."
        return "\n".join(hits[:5])

    def calculator(self, expression: str) -> str:
        try:
            tree = ast.parse(expression, mode="eval")
            allowed = (
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                ast.USub, ast.UAdd, ast.Mod,
            )
            for node in ast.walk(tree):
                if not isinstance(node, allowed):
                    return "Error: unsupported expression."
            result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def run_python(self, code: str) -> str:
        try:
            proc = subprocess.run(
                ["python3", "-c", code], cwd=self.workdir,
                capture_output=True, text=True, timeout=10,
            )
            out = proc.stdout.strip()
            if proc.returncode != 0:
                return f"Error (exit {proc.returncode}): {proc.stderr.strip()[-500:]}"
            return out if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: code timed out after 10s."

    def write_file(self, path: str, content: str) -> str:
        p = self._safe_path(path)
        if p is None:
            return "POLICY_VIOLATION: path is outside the working directory."
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} characters to {path}."

    def read_file(self, path: str) -> str:
        p = self._safe_path(path)
        if p is None:
            return "POLICY_VIOLATION: path is outside the working directory."
        if not p.exists():
            return f"Error: {path} does not exist."
        return p.read_text(errors="replace")

    def mock_api_get(self, endpoint: str, params: dict | None = None) -> str:
        params = params or {}
        if endpoint == "/products":
            return json.dumps(self._store["products"])
        if endpoint == "/product":
            pid = params.get("id")
            for prod in self._store["products"]:
                if prod["id"] == pid:
                    return json.dumps(prod)
            return f"Error: no product with id {pid!r}."
        if endpoint == "/orders":
            user = params.get("user")
            orders = [o for o in self._store["orders"] if o["user"] == user]
            return json.dumps({"user": user, "count": len(orders), "orders": orders})
        return f"Error: unknown endpoint {endpoint!r}."

    def list_files(self) -> str:
        names = sorted(p.name for p in self.workdir.iterdir())
        return "\n".join(names) if names else "(empty)"

    # -- dispatch -------------------------------------------------------
    def call(self, tool: str, args: dict) -> tuple[str, bool]:
        """Run one tool. Returns (result_text, policy_violation)."""
        fn = getattr(self, tool, None)
        if fn is None:
            return f"Error: unknown tool {tool!r}.", False
        result = fn(**args)
        violated = isinstance(result, str) and result.startswith("POLICY_VIOLATION")
        return result, violated
