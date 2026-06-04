"""Structured JSONL event recording for AutoSolver runs."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class EventRecorder:
    def __init__(self, path: str, run_id: Optional[str] = None, truncate: bool = True) -> None:
        self.path = os.path.abspath(path)
        self.run_id = run_id or uuid.uuid4().hex
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if truncate:
            with open(self.path, "w", encoding="utf-8"):
                pass

    def record(
        self,
        event: str,
        *,
        phase: str,
        iteration: Optional[int] = None,
        message: str = "",
        candidate: Optional[str] = None,
        candidate_hash: Optional[str] = None,
        elapsed: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        item = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "iteration": iteration,
            "event": event,
            "message": message,
            "candidate": candidate,
            "candidate_hash": candidate_hash,
            "elapsed": round(elapsed, 6) if elapsed is not None else None,
            "context": context or {},
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        return item


class PhaseTimer:
    def __init__(self) -> None:
        self.timings: Dict[str, float] = {}

    def mark(self, phase: str, elapsed: float) -> None:
        self.timings[phase] = round(self.timings.get(phase, 0.0) + elapsed, 6)


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def now_monotonic() -> float:
    return time.time()
