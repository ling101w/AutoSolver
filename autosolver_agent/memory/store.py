"""Short-term memory, long-term experiment memory, and bandit recommendations."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from autosolver_agent.models import ScoreResult, ValidationResult


MEMORY_SCHEMA_VERSION = 2
DEFAULT_MAX_LONG_TERM_ITEMS = 1000
_CAPPED_LONG_TERM_KEYS = ("strategy_history", "feature_strategy_effects", "experiments")


class MemoryStore:
    def __init__(self, memory_dir: str, max_long_term_items: Optional[int] = None) -> None:
        self.memory_dir = os.path.abspath(memory_dir)
        self.long_term_path = os.path.join(self.memory_dir, "long_term_memory.json")
        self.lock_path = self.long_term_path + ".lock"
        self.max_long_term_items = _positive_int(
            max_long_term_items,
            env_name="AUTOSOLVER_MEMORY_MAX_ITEMS",
            default=DEFAULT_MAX_LONG_TERM_ITEMS,
        )
        self.short_term: Dict[str, Any] = {
            "run_started_at": _now(),
            "iterations": [],
            "errors": [],
            "impact_analysis": [],
            "experiments": [],
        }
        os.makedirs(self.memory_dir, exist_ok=True)
        self.long_term = self._load_long_term()
        self._loaded_counts = self._list_counts(self.long_term)

    def _load_long_term(self) -> Dict[str, Any]:
        with _FileLock(self.lock_path):
            value = self._read_long_term_unlocked()
        if isinstance(value, dict):
            return self._ensure_schema(value)
        return self._new_long_term()

    def _read_long_term_unlocked(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.long_term_path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            if isinstance(value, dict):
                return value
        except FileNotFoundError:
            pass
        except Exception:
            corrupt_path = self.long_term_path + ".corrupt"
            try:
                os.replace(self.long_term_path, corrupt_path)
            except Exception:
                pass
        return None

    def _new_long_term(self) -> Dict[str, Any]:
        value = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "created_at": _now(),
            "updated_at": _now(),
            "strategy_history": [],
            "feature_strategy_effects": [],
            "experiments": [],
            "bandit_arms": {},
            "metadata": {"retention": {"max_items_per_list": self.max_long_term_items}},
        }
        return self._ensure_schema(value)

    def _ensure_schema(self, value: Dict[str, Any]) -> Dict[str, Any]:
        previous_version = _schema_version(value)
        value.setdefault("created_at", _now())
        value.setdefault("updated_at", _now())
        value["schema_version"] = MEMORY_SCHEMA_VERSION
        for key in _CAPPED_LONG_TERM_KEYS:
            if not isinstance(value.get(key), list):
                value[key] = []
            else:
                value[key] = _as_list(value[key])
        if not isinstance(value.get("bandit_arms"), dict):
            value["bandit_arms"] = {}
        if not value["bandit_arms"] and value.get("experiments"):
            value["bandit_arms"] = _build_bandit_arms(value.get("experiments", []))
        metadata = value.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        retention = metadata.get("retention")
        if not isinstance(retention, dict):
            retention = {}
        retention["max_items_per_list"] = self.max_long_term_items
        metadata["retention"] = retention
        if previous_version != MEMORY_SCHEMA_VERSION:
            migrations = metadata.setdefault("migrations", [])
            if isinstance(migrations, list):
                migrations.append(
                    {
                        "from_schema_version": previous_version,
                        "to_schema_version": MEMORY_SCHEMA_VERSION,
                        "migrated_at": _now(),
                    }
                )
        value["metadata"] = metadata
        self._trim_long_term(value)
        return value

    def digest(
        self,
        limit: int = 8,
        features: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        exploration: float = 1.4,
    ) -> Dict[str, Any]:
        candidate_arms = []
        if features:
            candidate_arms = list(features.get("recommended_focus", []) or features.get("tags", []))
        return {
            "strategy_history": self.long_term.get("strategy_history", [])[-limit:],
            "feature_strategy_effects": self.long_term.get("feature_strategy_effects", [])[-limit:],
            "short_term_recent": self.short_term.get("iterations", [])[-limit:],
            "recent_errors": self.short_term.get("errors", [])[-limit:],
            "similar_experiments": self.retrieve_similar(features or {}, top_k=top_k) if features else [],
            "bandit_recommendations": self.bandit_recommendations(candidate_arms, exploration=exploration, limit=top_k),
            "best_experiment": self.best_experiment_summary(),
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

    def record_experiment(
        self,
        iteration: int,
        candidate_name: str,
        features: Dict[str, Any],
        strategy: Sequence[str],
        params: Dict[str, Any],
        score: Optional[ScoreResult] = None,
        validation: Optional[ValidationResult] = None,
        artifact_paths: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        reward = _reward(score, validation, failure_reason)
        record = {
            "id": f"{_now()}::{iteration}::{candidate_name}",
            "created_at": _now(),
            "iteration": iteration,
            "candidate": candidate_name,
            "features": _compact_features(features),
            "tags": list(features.get("tags", [])),
            "strategy": list(strategy) or ["unknown"],
            "params": params,
            "score": _score_summary(score),
            "validation": _validation_summary(validation),
            "failure_reason": failure_reason,
            "artifact_paths": artifact_paths or {},
            "reward": reward,
        }
        self.short_term["experiments"].append(record)
        self.long_term["experiments"].append(record)
        self._update_bandit(record)
        return record

    def retrieve_similar(self, features: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        if not features:
            return []
        target = _compact_features(features)
        target_tags = set(features.get("tags", []))
        items = []
        for record in self.long_term.get("experiments", []):
            distance = _feature_distance(target, record.get("features", {}))
            tags = set(record.get("tags", []))
            if target_tags or tags:
                overlap = len(target_tags & tags)
                union = len(target_tags | tags) or 1
                distance += 1.0 - overlap / union
            item = {
                "distance": round(distance, 6),
                "candidate": record.get("candidate"),
                "strategy": record.get("strategy", []),
                "params": record.get("params", {}),
                "score": record.get("score"),
                "failure_reason": record.get("failure_reason"),
                "reward": record.get("reward"),
                "artifact_paths": record.get("artifact_paths", {}),
            }
            items.append(item)
        items.sort(key=lambda item: (item["distance"], -(item.get("reward") or -1e18)))
        return items[: max(0, top_k)]

    def bandit_recommendations(
        self,
        candidate_arms: Optional[Sequence[str]] = None,
        exploration: float = 1.4,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        arms = dict(self.long_term.get("bandit_arms", {}))
        for arm in candidate_arms or []:
            key = _arm_key([str(arm)])
            arms.setdefault(key, {"count": 0, "total_reward": 0.0, "mean_reward": 0.0, "failures": 0})
        total = sum(int(item.get("count", 0)) for item in arms.values()) + 1
        recommendations = []
        for arm, stats in arms.items():
            count = int(stats.get("count", 0))
            mean = float(stats.get("mean_reward", 0.0))
            if count == 0:
                ucb = float("inf")
                mode = "explore_cold_start"
            else:
                ucb = mean + exploration * math.sqrt(math.log(total + 1.0) / count)
                mode = "exploit" if count >= 2 else "explore"
            recommendations.append(
                {
                    "arm": arm,
                    "count": count,
                    "mean_reward": round(mean, 6),
                    "ucb": "inf" if math.isinf(ucb) else round(ucb, 6),
                    "mode": mode,
                    "failures": int(stats.get("failures", 0)),
                }
            )
        recommendations.sort(key=lambda item: (item["ucb"] == "inf", item["ucb"] if item["ucb"] != "inf" else 1e18), reverse=True)
        return recommendations[: max(0, limit)]

    def best_experiment_summary(self) -> Dict[str, Any]:
        valid = [
            record
            for record in self.long_term.get("experiments", [])
            if record.get("score") and record.get("score", {}).get("failures", 999) == 0
        ]
        if not valid:
            return {}
        valid.sort(
            key=lambda record: (
                record["score"].get("failures", 999),
                -record["score"].get("covered", 0),
                record["score"].get("penalty", 1e18),
                record["score"].get("runtime", 1e18),
            )
        )
        best = valid[0]
        return {
            "candidate": best.get("candidate"),
            "strategy": best.get("strategy", []),
            "params": best.get("params", {}),
            "score": best.get("score"),
            "artifact_paths": best.get("artifact_paths", {}),
            "reward": best.get("reward"),
        }

    def save(self, short_term_path: Optional[str] = None) -> None:
        with _FileLock(self.lock_path):
            latest = self._read_long_term_unlocked()
            if isinstance(latest, dict):
                latest = self._ensure_schema(latest)
            else:
                latest = self._new_long_term()
            merged = self._merge_long_term(latest)
            merged["updated_at"] = _now()
            self._trim_long_term(merged)
            _write_json_atomic(self.long_term_path, merged)
            if short_term_path:
                _write_json_atomic(short_term_path, self.short_term)
        self.long_term = merged
        self._loaded_counts = self._list_counts(self.long_term)

    def _merge_long_term(self, latest: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(latest)
        appended: Dict[str, List[Dict[str, Any]]] = {}
        for key in _CAPPED_LONG_TERM_KEYS:
            current_items = _as_list(self.long_term.get(key))
            offset = min(self._loaded_counts.get(key, 0), len(current_items))
            delta = current_items[offset:]
            appended[key] = delta
            merged[key] = _as_list(merged.get(key)) + delta
        arms = dict(merged.get("bandit_arms", {}))
        for record in appended.get("experiments", []):
            _apply_bandit_record(arms, record)
        merged["bandit_arms"] = arms
        metadata = dict(merged.get("metadata", {}))
        metadata["retention"] = {"max_items_per_list": self.max_long_term_items}
        merged["metadata"] = metadata
        merged["schema_version"] = MEMORY_SCHEMA_VERSION
        return merged

    def _trim_long_term(self, value: Dict[str, Any]) -> None:
        for key in _CAPPED_LONG_TERM_KEYS:
            items = value.get(key)
            if isinstance(items, list) and len(items) > self.max_long_term_items:
                value[key] = items[-self.max_long_term_items :]

    def _list_counts(self, value: Dict[str, Any]) -> Dict[str, int]:
        return {key: len(_as_list(value.get(key))) for key in _CAPPED_LONG_TERM_KEYS}

    def _update_bandit(self, record: Dict[str, Any]) -> None:
        arms = self.long_term.setdefault("bandit_arms", {})
        _apply_bandit_record(arms, record)


class _FileLock:
    def __init__(self, path: str) -> None:
        self.path = path
        self._handle = None

    def __enter__(self) -> "_FileLock":
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._handle = open(self.path, "a+", encoding="utf-8")
        self._handle.seek(0)
        if not self._handle.read(1):
            self._handle.write("0")
            self._handle.flush()
        self._lock()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self._unlock()
        finally:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def _lock(self) -> None:
        if self._handle is None:
            return
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)

    def _unlock(self) -> None:
        if self._handle is None:
            return
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_int(value: Optional[int], env_name: str, default: int) -> int:
    if value is None:
        raw = os.environ.get(env_name)
        if raw:
            try:
                value = int(raw)
            except ValueError:
                value = None
    try:
        parsed = int(value if value is not None else default)
    except (TypeError, ValueError):
        parsed = default
    return max(1, parsed)


def _schema_version(value: Dict[str, Any]) -> int:
    try:
        return int(value.get("schema_version", 0))
    except (TypeError, ValueError):
        return 0


def _as_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _build_bandit_arms(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    arms: Dict[str, Dict[str, Any]] = {}
    for record in records:
        _apply_bandit_record(arms, record)
    return arms


def _apply_bandit_record(arms: Dict[str, Dict[str, Any]], record: Dict[str, Any]) -> None:
    arm = _arm_key(record.get("strategy", []))
    stats = arms.setdefault(arm, {"count": 0, "total_reward": 0.0, "mean_reward": 0.0, "failures": 0})
    stats["count"] = int(stats.get("count", 0)) + 1
    stats["total_reward"] = float(stats.get("total_reward", 0.0)) + float(record.get("reward", 0.0))
    stats["mean_reward"] = stats["total_reward"] / max(1, stats["count"])
    if record.get("failure_reason"):
        stats["failures"] = int(stats.get("failures", 0)) + 1


def _write_json_atomic(path: str, value: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)


def _compact_features(features: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "row_count",
        "task_count",
        "courier_count",
        "pair_ratio",
        "avg_willingness",
        "avg_score",
        "high_willingness_ratio",
        "capacity_ratio",
        "low_capacity",
    ]
    compact = {}
    for key in keys:
        if key in features:
            value = features[key]
            if isinstance(value, bool):
                compact[key] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                compact[key] = float(value)
    return compact


def _feature_distance(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    total = 0.0
    for key in keys:
        lv = float(left.get(key, 0.0) or 0.0)
        rv = float(right.get(key, 0.0) or 0.0)
        scale = max(1.0, abs(lv), abs(rv))
        total += abs(lv - rv) / scale
    return total / len(keys)


def _score_summary(score: Optional[ScoreResult]) -> Optional[Dict[str, Any]]:
    if score is None:
        return None
    return {
        "name": score.name,
        "rank": list(score.rank),
        "covered": score.total_covered,
        "tasks": score.total_tasks,
        "penalty": round(score.total_penalty, 6),
        "runtime": round(score.total_runtime, 6),
        "failures": score.failures,
        "convergence": score.convergence,
    }


def _validation_summary(validation: Optional[ValidationResult]) -> Optional[Dict[str, Any]]:
    if validation is None:
        return None
    return {
        "valid": validation.valid,
        "stage": validation.stage,
        "runtime": round(validation.runtime, 6),
        "errors": validation.errors,
    }


def _reward(score: Optional[ScoreResult], validation: Optional[ValidationResult], failure_reason: Optional[str]) -> float:
    if failure_reason or (validation is not None and not validation.valid):
        return -1000.0
    if score is None:
        return -100.0
    coverage_ratio = score.total_covered / max(1, score.total_tasks)
    avg_penalty = score.total_penalty / max(1, score.total_tasks)
    return round(coverage_ratio * 100.0 - avg_penalty - score.failures * 100.0 - score.total_runtime * 0.05, 6)


def _arm_key(strategy: Sequence[str]) -> str:
    values = [str(item) for item in strategy if str(item).strip()]
    return "+".join(sorted(dict.fromkeys(values))) or "unknown"
