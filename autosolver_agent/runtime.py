"""Subprocess candidate execution."""

from __future__ import annotations

import multiprocessing
import time
import traceback
from typing import Any, Dict


def _candidate_worker(code: str, case_text: str, queue: "multiprocessing.Queue") -> None:
    try:
        namespace: Dict[str, Any] = {}
        exec(compile(code, "<candidate_solver>", "exec"), namespace)
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


def run_candidate(code: str, case_text: str, timeout: float) -> Dict[str, Any]:
    queue: "multiprocessing.Queue" = multiprocessing.Queue()
    process = multiprocessing.Process(target=_candidate_worker, args=(code, case_text, queue))
    process.daemon = True
    start = time.time()
    process.start()
    process.join(timeout)
    runtime = time.time() - start
    if process.is_alive():
        process.terminate()
        process.join(0.2)
        return {"status": "timeout", "answer": None, "runtime": runtime, "error": "timeout"}
    if queue.empty():
        return {"status": "error", "answer": None, "runtime": runtime, "error": "empty result"}
    status, answer, child_runtime, error = queue.get()
    return {"status": status, "answer": answer, "runtime": child_runtime or runtime, "error": error}
