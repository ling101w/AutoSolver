"""Short-term and long-term JSON memory."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from autosolver_agent.models import ScoreResult, ValidationResult


class MemoryStore:
    def __init__(self, memory_dir: str) -> None:
        self.memory_dir = os.path.abspath(memory_dir)
        self.long_term_path = os.path.join(self.memory_dir, "long_term_memory.json")
        self.short_term: Dict[str, Any] = {
            "run_started_at": _now(),
            "iterations": [],
            "errors": [],
            "impact_analysis": [],
        }
        os.makedirs(self.memory_dir, exist_ok=True)
        self.long_term = self._load_long_term()

    def _load_long_term(self) -> Dict[str, Any]:
        try:
            with open(self.long_term_path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            if isinstance(value, dict):
                value.setdefault("strategy_history", [])
                value.setdefault("feature_strategy_effects", [])
                value.setdefault("updated_at", _now())
                return value
        except FileNotFoundError:
            pass
        except Exception:
            corrupt_path = self.long_term_path + ".corrupt"
            try:
                os.replace(self.long_term_path, corrupt_path)
            except Exception:
                pass
        return {
            "created_at": _now(),
            "updated_at": _now(),
            "strategy_history": [],
            "feature_strategy_effects": [],
        }

    def digest(self, limit: int = 8) -> Dict[str, Any]:
        return {
            "strategy_history": self.long_term.get("strategy_history", [])[-limit:],
            "feature_strategy_effects": self.long_term.get("feature_strategy_effects", [])[-limit:],
            "short_term_recent": self.short_term.get("iterations", [])[-limit:],
            "recent_errors": self.short_term.get("errors", [])[-limit:],
        }

    def record_candidate(
        self,
        iteration: int,
        candidate_name: str,
        rationale: Dict[str, Any],
        selected_features: Dict[str, Any],
    ) -> None:
        self.short_term["iterations"].append(
            {
                "iteration": iteration,
                "candidate": candidate_name,
                "rationale": rationale,
                "selected_features": selected_features,
                "created_at": _now(),
            }
        )

    def record_validation(self, iteration: int, validation: ValidationResult) -> None:
        item = {
            "iteration": iteration,
            "stage": validation.stage,
            "valid": validation.valid,
            "runtime": round(validation.runtime, 6),
            "errors": validation.errors,
            "created_at": _now(),
        }
        if not validation.valid:
            self.short_term["errors"].append(item)
        self.short_term.setdefault("validation", []).append(item)

    def record_score(self, iteration: int, score: ScoreResult, impact: Dict[str, Any]) -> None:
        item = {
            "iteration": iteration,
            "name": score.name,
            "rank": list(score.rank),
            "covered": score.total_covered,
            "tasks": score.total_tasks,
            "penalty": round(score.total_penalty, 6),
            "runtime": round(score.total_runtime, 6),
            "failures": score.failures,
            "convergence": score.convergence,
            "created_at": _now(),
        }
        self.short_term.setdefault("scores", []).append(item)
        self.short_term["impact_analysis"].append(impact)
        self.long_term["strategy_history"].append(item)
        self.long_term["feature_strategy_effects"].append(
            {
                "iteration": iteration,
                "name": score.name,
                "impact": impact,
                "created_at": _now(),
            }
        )

    def save(self, short_term_path: Optional[str] = None) -> None:
        self.long_term["updated_at"] = _now()
        _write_json_atomic(self.long_term_path, self.long_term)
        if short_term_path:
            _write_json_atomic(short_term_path, self.short_term)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: str, value: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)
