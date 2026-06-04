"""Subprocess candidate execution."""

from __future__ import annotations

import builtins
import multiprocessing
import os
import signal
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

SAFE_IMPORT_ROOTS = {
    "__future__",
    "bisect",
    "collections",
    "copy",
    "dataclasses",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "random",
    "statistics",
    "time",
    "typing",
}

SAFE_BUILTIN_NAMES = {
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "BaseException",
    "BufferError",
    "BytesWarning",
    "DeprecationWarning",
    "EOFError",
    "Ellipsis",
    "Exception",
    "False",
    "FloatingPointError",
    "FutureWarning",
    "GeneratorExit",
    "ImportError",
    "ImportWarning",
    "IndexError",
    "KeyError",
    "KeyboardInterrupt",
    "LookupError",
    "MemoryError",
    "NameError",
    "None",
    "NotImplemented",
    "NotImplementedError",
    "OSError",
    "OverflowError",
    "PendingDeprecationWarning",
    "ReferenceError",
    "ResourceWarning",
    "RuntimeError",
    "RuntimeWarning",
    "StopAsyncIteration",
    "StopIteration",
    "SyntaxError",
    "SyntaxWarning",
    "SystemError",
    "SystemExit",
    "TabError",
    "TimeoutError",
    "True",
    "TypeError",
    "UnboundLocalError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "UnicodeError",
    "UnicodeTranslateError",
    "UnicodeWarning",
    "UserWarning",
    "ValueError",
    "Warning",
    "ZeroDivisionError",
    "__build_class__",
    "abs",
    "all",
    "any",
    "bin",
    "bool",
    "bytes",
    "callable",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hash",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "oct",
    "ord",
    "pow",
    "print",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "classmethod",
    "staticmethod",
    "str",
    "sum",
    "tuple",
    "zip",
}


@dataclass(frozen=True)
class ExecutionPolicy:
    allowed_import_roots: Set[str] = field(default_factory=lambda: set(SAFE_IMPORT_ROOTS))
    max_memory_mb: int = 256
    cpu_grace_seconds: int = 0
    kill_grace_seconds: float = 0.2


def _candidate_worker(
    code: str,
    case_text: str,
    queue: "multiprocessing.Queue",
    policy: ExecutionPolicy,
) -> None:
    try:
        _apply_resource_limits(policy)
        namespace: Dict[str, Any] = {
            "__builtins__": _safe_builtins(policy),
            "__name__": "candidate_solver",
        }
        exec(compile(code, "<candidate_solver>", "exec"), namespace, namespace)
        solve = namespace.get("solve")
        if solve is None:
            queue.put(("error", None, 0.0, "missing solve function"))
            return
        start = time.time()
        answer = solve(case_text)
        elapsed = time.time() - start
        queue.put(("ok", answer, elapsed, None))
    except Exception:
        queue.put(("error", None, 0.0, traceback.format_exc()))


def run_candidate(
    code: str,
    case_text: str,
    timeout: float,
    policy: Optional[ExecutionPolicy] = None,
) -> Dict[str, Any]:
    if policy is None:
        policy = ExecutionPolicy(cpu_grace_seconds=max(1, int(timeout) + 1))
    queue: "multiprocessing.Queue" = multiprocessing.Queue()
    process = multiprocessing.Process(target=_candidate_worker, args=(code, case_text, queue, policy))
    process.daemon = True
    start = time.time()
    process.start()
    process.join(timeout)
    runtime = time.time() - start
    if process.is_alive():
        process.terminate()
        process.join(policy.kill_grace_seconds)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(policy.kill_grace_seconds)
        return {"status": "timeout", "answer": None, "runtime": runtime, "error": "timeout"}
    if queue.empty():
        return {"status": "error", "answer": None, "runtime": runtime, "error": "empty result"}
    status, answer, child_runtime, error = queue.get()
    return {"status": status, "answer": answer, "runtime": child_runtime or runtime, "error": error}


def _safe_builtins(policy: ExecutionPolicy) -> Dict[str, Any]:
    safe = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES if hasattr(builtins, name)}
    safe["__import__"] = _safe_import(policy)
    return safe


def _safe_import(policy: ExecutionPolicy):
    def guarded_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):  # type: ignore[no-untyped-def]
        root = name.split(".")[0]
        if level != 0 or root not in policy.allowed_import_roots:
            raise ImportError(f"import not allowed: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    return guarded_import


def _apply_resource_limits(policy: ExecutionPolicy) -> None:
    if os.name != "posix":
        return
    try:
        import resource

        cpu_limit = max(1, int(policy.cpu_grace_seconds))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
        memory_bytes = max(16, int(policy.max_memory_mb)) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        if hasattr(resource, "RLIMIT_FSIZE"):
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    except Exception:
        return
    try:
        signal.signal(signal.SIGXCPU, lambda *_: raise_timeout())
    except Exception:
        return


def raise_timeout() -> None:
    raise TimeoutError("candidate exceeded CPU limit")
