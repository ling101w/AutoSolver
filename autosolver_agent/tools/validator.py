"""Static and runtime validation tool."""

from __future__ import annotations

import ast
from typing import Any, Dict, List

from autosolver_agent.caseio import score_answer
from autosolver_agent.models import Case, ParsedCase, ValidationResult
from autosolver_agent.runtime import SAFE_IMPORT_ROOTS, run_candidate

FORBIDDEN_IMPORT_ROOTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "paramiko",
    "pathlib",
    "shutil",
    "glob",
    "importlib",
}

FORBIDDEN_CALLS = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
}

FORBIDDEN_NAMES = {
    "__builtins__",
    "__debug__",
    "__file__",
    "__loader__",
    "__package__",
    "__spec__",
    "globals",
    "locals",
    "vars",
}

ALLOWED_IMPORT_ROOTS = set(SAFE_IMPORT_ROOTS)


class Validator:
    def __init__(self, smoke_timeout: float) -> None:
        self.smoke_timeout = smoke_timeout

    def validate_static(self, code: str) -> ValidationResult:
        errors: List[Dict[str, Any]] = []
        if not code or "def solve" not in code:
            errors.append({"type": "missing_contract", "message": "missing solve(input_text) function"})
            return ValidationResult(valid=False, stage="static", errors=errors)
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            errors.append(
                {
                    "type": "syntax_error",
                    "message": str(exc),
                    "line": exc.lineno,
                    "offset": exc.offset,
                }
            )
            return ValidationResult(valid=False, stage="static", errors=errors)
        solve_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve"]
        if not solve_defs:
            errors.append({"type": "missing_contract", "message": "missing top-level solve function"})
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.level:
                    errors.append(
                        {
                            "type": "forbidden_import",
                            "message": "relative imports are not allowed",
                            "line": getattr(node, "lineno", None),
                        }
                    )
                    continue
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for name in names:
                    root = name.split(".")[0]
                    if root in FORBIDDEN_IMPORT_ROOTS or root not in ALLOWED_IMPORT_ROOTS:
                        errors.append(
                            {
                                "type": "forbidden_import",
                                "message": f"forbidden import: {name}",
                                "line": getattr(node, "lineno", None),
                            }
                        )
            elif isinstance(node, ast.Call):
                call_name = self._call_name(node.func)
                if call_name in FORBIDDEN_CALLS:
                    errors.append(
                        {
                            "type": "forbidden_call",
                            "message": f"forbidden call: {call_name}",
                            "line": getattr(node, "lineno", None),
                        }
                    )
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in FORBIDDEN_NAMES:
                    errors.append(
                        {
                            "type": "forbidden_name",
                            "message": f"forbidden name: {node.id}",
                            "line": getattr(node, "lineno", None),
                        }
                    )
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    errors.append(
                        {
                            "type": "forbidden_attribute",
                            "message": f"dunder attribute access is not allowed: {node.attr}",
                            "line": getattr(node, "lineno", None),
                        }
                    )
        try:
            compile(code, "<candidate_static_validation>", "exec")
        except Exception as exc:
            errors.append({"type": "compile_error", "message": str(exc)})
        return ValidationResult(valid=not errors, stage="static", errors=errors)

    def validate_runtime(
        self,
        code: str,
        cases: List[Case],
        parsed_cases: List[ParsedCase],
        max_cases: int = 1,
    ) -> ValidationResult:
        errors: List[Dict[str, Any]] = []
        total_runtime = 0.0
        last_answer: Any = None
        for case, parsed in list(zip(cases, parsed_cases))[:max_cases]:
            run = run_candidate(code, case.text, self.smoke_timeout)
            total_runtime += run.get("runtime", 0.0)
            if run["status"] != "ok":
                errors.append(
                    {
                        "type": "runtime_error",
                        "case": case.name,
                        "status": run["status"],
                        "message": run.get("error"),
                    }
                )
                continue
            last_answer = run["answer"]
            scored = score_answer(parsed, run["answer"])
            if not scored["valid"]:
                errors.append(
                    {
                        "type": "invalid_output",
                        "case": case.name,
                        "message": scored.get("error"),
                        "covered": scored.get("covered"),
                    }
                )
        return ValidationResult(
            valid=not errors,
            stage="runtime",
            errors=errors,
            runtime=total_runtime,
            answer=last_answer,
        )

    def validate(self, code: str, cases: List[Case], parsed_cases: List[ParsedCase]) -> ValidationResult:
        static = self.validate_static(code)
        if not static.valid:
            return static
        return self.validate_runtime(code, cases, parsed_cases)

    def _call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts = []
            cur: ast.AST = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            return ".".join(reversed(parts))
        return ""
