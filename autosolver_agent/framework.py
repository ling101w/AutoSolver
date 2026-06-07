"""LLM-maintained solver framework schemas and persistence."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FRAMEWORK_SCHEMA_VERSION = 1
_MAX_HISTORY_ITEMS = 200
_DANGEROUS_FRAGMENTS = (
    "import os",
    "import sys",
    "import subprocess",
    "subprocess.",
    "subprocess(",
    "socket.",
    "requests.",
    "urllib.",
    "open(",
    "eval(",
    "exec(",
    "__import__",
    "compile(",
)


class FrameworkValidationError(RuntimeError):
    """Raised when an LLM-maintained framework payload is unsafe or invalid."""


class FeatureDimension(BaseModel):
    """A feature interpretation dimension proposed and maintained by the LLM."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=1200)
    signals: List[str] = Field(default_factory=list, max_length=20)
    interpretation_notes: List[str] = Field(default_factory=list, max_length=20)
    status: str = Field(default="active", max_length=40)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("name", mode="after")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        return _clean_name(value, "feature_dimension")


class StrategyKnowledge(BaseModel):
    """A solver strategy proposed and maintained by the LLM."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=1600)
    applicable_tags: List[str] = Field(default_factory=list, max_length=30)
    feature_signals: List[str] = Field(default_factory=list, max_length=30)
    implementation_notes: str = Field(default="", max_length=2400)
    recommended_parameters: Dict[str, Any] = Field(default_factory=dict)
    risks: List[str] = Field(default_factory=list, max_length=20)
    status: str = Field(default="active", max_length=40)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("name", mode="after")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        return _clean_name(value, "strategy")


class SkillKnowledge(BaseModel):
    """Reusable implementation guidance proposed and maintained by the LLM."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, max_length=80)
    strategy_names: List[str] = Field(default_factory=list, max_length=30)
    construction_notes: str = Field(default="", max_length=2400)
    code_contract: str = Field(default="", max_length=1600)
    constraints: List[str] = Field(default_factory=list, max_length=30)
    examples: List[str] = Field(default_factory=list, max_length=20)
    status: str = Field(default="active", max_length=40)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("name", mode="after")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        return _clean_name(value, "skill")

    @field_validator("strategy_names", mode="after")
    @classmethod
    def _clean_strategy_names(cls, value: List[str]) -> List[str]:
        return [_clean_name(item, "strategy") for item in value if str(item).strip()]


class SolverFramework(BaseModel):
    """The persisted feature/strategy/skill framework owned by the LLM."""

    model_config = ConfigDict(extra="allow")

    schema_version: int = FRAMEWORK_SCHEMA_VERSION
    feature_dimensions: List[FeatureDimension] = Field(default_factory=list, max_length=80)
    strategies: List[StrategyKnowledge] = Field(default_factory=list, max_length=120)
    skills: List[SkillKnowledge] = Field(default_factory=list, max_length=120)
    source: str = Field(default="llm", max_length=80)
    updated_at: Optional[str] = None

    @model_validator(mode="after")
    def _validate_framework(self) -> "SolverFramework":
        if self.schema_version != FRAMEWORK_SCHEMA_VERSION:
            raise ValueError(f"unsupported framework schema_version {self.schema_version}")
        _ensure_unique("feature dimension", [item.name for item in self.feature_dimensions])
        _ensure_unique("strategy", [item.name for item in self.strategies])
        _ensure_unique("skill", [item.name for item in self.skills])
        strategy_names = {item.name for item in self.strategies if item.status != "retired"}
        missing = sorted(
            {
                strategy
                for skill in self.skills
                if skill.status != "retired"
                for strategy in skill.strategy_names
                if strategy not in strategy_names
            }
        )
        if missing:
            raise ValueError(f"skill references unknown strategies: {', '.join(missing)}")
        return self


class FrameworkUpdate(BaseModel):
    """A partial framework update produced after candidate evaluation."""

    model_config = ConfigDict(extra="allow")

    update_reason: str = Field(default="", max_length=1600)
    source_experiments: List[str] = Field(default_factory=list, max_length=30)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    feature_dimensions: List[FeatureDimension] = Field(default_factory=list, max_length=40)
    strategies: List[StrategyKnowledge] = Field(default_factory=list, max_length=60)
    skills: List[SkillKnowledge] = Field(default_factory=list, max_length=60)
    retire_feature_names: List[str] = Field(default_factory=list, max_length=40)
    retire_strategy_names: List[str] = Field(default_factory=list, max_length=40)
    retire_skill_names: List[str] = Field(default_factory=list, max_length=40)

    @field_validator("retire_feature_names", "retire_strategy_names", "retire_skill_names", mode="after")
    @classmethod
    def _clean_retire_names(cls, value: List[str]) -> List[str]:
        return [_clean_name(item, "framework_item") for item in value if str(item).strip()]


class InstanceInterpretation(BaseModel):
    """Dynamic LLM interpretation of objective case features."""

    model_config = ConfigDict(extra="allow")

    tags: List[str] = Field(default_factory=list, max_length=40)
    opportunities: List[str] = Field(default_factory=list, max_length=30)
    risks: List[str] = Field(default_factory=list, max_length=30)
    recommended_focus: List[str] = Field(default_factory=list, max_length=30)
    feature_notes: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = Field(default="", max_length=2400)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("tags", "recommended_focus", mode="after")
    @classmethod
    def _clean_names(cls, value: List[str]) -> List[str]:
        return [_clean_name(item, "tag") for item in value if str(item).strip()]


class FrameworkStore:
    """Persistent framework memory separate from experiment memory."""

    def __init__(self, memory_dir: str) -> None:
        self.memory_dir = os.path.abspath(memory_dir)
        self.path = os.path.join(self.memory_dir, "framework_memory.json")
        self.lock_path = self.path + ".lock"
        os.makedirs(self.memory_dir, exist_ok=True)
        self.document = self._load_document()

    @property
    def framework(self) -> SolverFramework:
        return SolverFramework.model_validate(self.document["framework"])

    def is_empty(self) -> bool:
        framework = self.framework
        return not (framework.feature_dimensions or framework.strategies or framework.skills)

    def reload(self) -> SolverFramework:
        self.document = self._load_document()
        return self.framework

    def bootstrap(self, framework: SolverFramework, *, source: str) -> Dict[str, Any]:
        _validate_safe_payload(framework.model_dump(mode="json"))
        with _FileLock(self.lock_path):
            latest = self._read_document_unlocked() or self._new_document()
            latest = self._ensure_document(latest)
            current = SolverFramework.model_validate(latest["framework"])
            if current.feature_dimensions or current.strategies or current.skills:
                self.document = latest
                return {"action": "bootstrap_skipped_existing_framework", "framework_counts": self.counts()}
            latest["framework"] = framework.model_copy(update={"updated_at": _now(), "source": source}).model_dump(mode="json")
            latest["updated_at"] = _now()
            latest["history"] = _trim_history(
                list(latest.get("history", []))
                + [{"created_at": _now(), "action": "bootstrap", "source": source, "counts": _framework_counts(framework)}]
            )
            _write_json_atomic(self.path, latest)
            self.document = latest
            return {"action": "bootstrap_applied", "framework_counts": self.counts()}

    def apply_update(self, update: FrameworkUpdate, *, source: str, iteration: int) -> Dict[str, Any]:
        _validate_safe_payload(update.model_dump(mode="json"))
        with _FileLock(self.lock_path):
            latest = self._read_document_unlocked() or self._new_document()
            latest = self._ensure_document(latest)
            framework = self._merge_update(SolverFramework.model_validate(latest["framework"]), update)
            latest["framework"] = framework.model_copy(update={"updated_at": _now(), "source": source}).model_dump(mode="json")
            latest["updated_at"] = _now()
            latest["history"] = _trim_history(
                list(latest.get("history", []))
                + [
                    {
                        "created_at": _now(),
                        "action": "update",
                        "iteration": iteration,
                        "source": source,
                        "reason": update.update_reason,
                        "confidence": update.confidence,
                        "counts": _framework_counts(framework),
                    }
                ]
            )
            _write_json_atomic(self.path, latest)
            self.document = latest
            return {"action": "update_applied", "iteration": iteration, "framework_counts": self.counts(), "reason": update.update_reason}

    def sanitize_update(self, update: FrameworkUpdate) -> Tuple[FrameworkUpdate, Dict[str, Any]]:
        """Drop or trim framework update fragments that cannot reference active strategies."""

        current = self.framework
        retired = set(update.retire_strategy_names)
        available_strategies = {
            item.name for item in current.strategies if item.status != "retired" and item.name not in retired
        }
        available_strategies.update(item.name for item in update.strategies if item.status != "retired")

        filtered_skills = []
        dropped_skills = []
        filtered_strategy_names: Dict[str, Dict[str, List[str]]] = {}
        for skill in update.skills:
            if skill.status == "retired":
                filtered_skills.append(skill)
                continue
            allowed = [name for name in skill.strategy_names if name in available_strategies]
            removed = [name for name in skill.strategy_names if name not in available_strategies]
            if removed:
                filtered_strategy_names[skill.name] = {"kept": allowed, "removed": removed}
            if skill.strategy_names and not allowed:
                dropped_skills.append(skill.name)
                continue
            filtered_skills.append(skill.model_copy(update={"strategy_names": allowed}))

        changed = bool(dropped_skills or filtered_strategy_names)
        sanitized = update.model_copy(update={"skills": filtered_skills}) if changed else update
        return sanitized, {
            "changed": changed,
            "dropped_skills": dropped_skills,
            "filtered_strategy_names": filtered_strategy_names,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "schema_version": FRAMEWORK_SCHEMA_VERSION,
            "framework": self.framework.model_dump(mode="json"),
            "history": list(self.document.get("history", []))[-10:],
            "counts": self.counts(),
        }

    def prompt_context(self) -> str:
        return json.dumps(
            {
                "solver_framework": self.framework.model_dump(mode="json"),
                "maintenance_policy": {
                    "owner": "LLM",
                    "hardcoded_seed": False,
                    "update_mode": "after_each_evaluated_iteration",
                    "guardrails": [
                        "Generated solver code must still obey solve(input_text: str) -> list.",
                        (
                            "Framework entries may propose strategies and implementation guidance, "
                            "but cannot relax validator, scorer, runtime, or parser constraints."
                        ),
                    ],
                },
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    def digest(self) -> Dict[str, Any]:
        framework = self.framework
        return {
            "counts": self.counts(),
            "strategy_names": [item.name for item in framework.strategies if item.status != "retired"][:30],
            "feature_dimension_names": [item.name for item in framework.feature_dimensions if item.status != "retired"][:30],
            "skill_names": [item.name for item in framework.skills if item.status != "retired"][:30],
            "recent_updates": list(self.document.get("history", []))[-5:],
        }

    def candidate_strategy_names(self, limit: int = 20) -> List[str]:
        names = [item.name for item in self.framework.strategies if item.status != "retired"]
        return names[: max(0, limit)]

    def counts(self) -> Dict[str, int]:
        return _framework_counts(self.framework)

    def _load_document(self) -> Dict[str, Any]:
        with _FileLock(self.lock_path):
            value = self._read_document_unlocked()
        return self._ensure_document(value) if value is not None else self._new_document()

    def _read_document_unlocked(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise FrameworkValidationError(f"framework memory must contain a JSON object: {self.path}")
            return value
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            raise FrameworkValidationError(f"framework memory is not valid JSON: {self.path}") from exc

    def _new_document(self) -> Dict[str, Any]:
        now = _now()
        return {
            "schema_version": FRAMEWORK_SCHEMA_VERSION,
            "created_at": now,
            "updated_at": now,
            "framework": SolverFramework(updated_at=now).model_dump(mode="json"),
            "history": [],
        }

    def _ensure_document(self, value: Dict[str, Any]) -> Dict[str, Any]:
        if int(value.get("schema_version", 0)) != FRAMEWORK_SCHEMA_VERSION:
            raise FrameworkValidationError(
                f"unsupported framework schema_version {value.get('schema_version')}; expected {FRAMEWORK_SCHEMA_VERSION}"
            )
        framework = SolverFramework.model_validate(value.get("framework", {}))
        _validate_safe_payload(framework.model_dump(mode="json"))
        history = value.get("history", [])
        if not isinstance(history, list):
            raise FrameworkValidationError("framework history must be a list")
        value["framework"] = framework.model_dump(mode="json")
        value["history"] = _trim_history([item for item in history if isinstance(item, dict)])
        value.setdefault("created_at", _now())
        value.setdefault("updated_at", _now())
        return value

    def _merge_update(self, framework: SolverFramework, update: FrameworkUpdate) -> SolverFramework:
        features = {item.name: item for item in framework.feature_dimensions if item.name not in set(update.retire_feature_names)}
        strategies = {item.name: item for item in framework.strategies if item.name not in set(update.retire_strategy_names)}
        skills = {item.name: item for item in framework.skills if item.name not in set(update.retire_skill_names)}
        for feature_item in update.feature_dimensions:
            features[feature_item.name] = feature_item
        for strategy_item in update.strategies:
            strategies[strategy_item.name] = strategy_item
        for skill_item in update.skills:
            skills[skill_item.name] = skill_item
        return SolverFramework(
            feature_dimensions=list(features.values()),
            strategies=list(strategies.values()),
            skills=list(skills.values()),
            updated_at=_now(),
        )


def parse_solver_framework(text_or_value: Any) -> SolverFramework:
    value = _load_json_like(text_or_value)
    if not isinstance(value, dict):
        raise FrameworkValidationError("framework response is not a JSON object")
    framework = SolverFramework.model_validate(_sanitize_llm_payload(value.get("solver_framework", value)))
    _validate_safe_payload(framework.model_dump(mode="json"))
    return framework


def parse_instance_interpretation(text_or_value: Any) -> InstanceInterpretation:
    value = _load_json_like(text_or_value)
    if not isinstance(value, dict):
        raise FrameworkValidationError("instance interpretation response is not a JSON object")
    interpretation = InstanceInterpretation.model_validate(_sanitize_llm_payload(value))
    _validate_safe_payload(interpretation.model_dump(mode="json"))
    return interpretation


def parse_framework_update(text_or_value: Any) -> FrameworkUpdate:
    value = _load_json_like(text_or_value)
    if not isinstance(value, dict):
        raise FrameworkValidationError("framework update response is not a JSON object")
    update = FrameworkUpdate.model_validate(_sanitize_llm_payload(value))
    _validate_safe_payload(update.model_dump(mode="json"))
    return update


def solver_framework_schema_text() -> str:
    return json.dumps(SolverFramework.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)


def instance_interpretation_schema_text() -> str:
    return json.dumps(InstanceInterpretation.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)


def framework_update_schema_text() -> str:
    return json.dumps(FrameworkUpdate.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)


def _load_json_like(text_or_value: Any) -> Any:
    if isinstance(text_or_value, (dict, list)):
        return text_or_value
    text = str(text_or_value).strip()
    try:
        return json.loads(text)
    except Exception as exc:
        raise FrameworkValidationError(f"response is not valid JSON: {exc}") from exc


def _clean_name(value: str, default: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._-")
    return (cleaned or default)[:80]


def _ensure_unique(kind: str, names: List[str]) -> None:
    seen = set()
    duplicates = []
    for name in names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"duplicate {kind} names: {', '.join(sorted(set(duplicates)))}")


def _validate_safe_payload(value: Any) -> None:
    if isinstance(value, str):
        lowered = value.lower()
        for fragment in _DANGEROUS_FRAGMENTS:
            if fragment in lowered:
                raise FrameworkValidationError(f"framework payload contains forbidden fragment: {fragment}")
        return
    if isinstance(value, list):
        for item in value:
            _validate_safe_payload(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _validate_safe_payload(item)


def _sanitize_llm_payload(value: Any) -> Any:
    """Clean unsafe-looking text snippets from LLM-owned framework metadata."""

    if isinstance(value, str):
        cleaned = value
        for fragment in _DANGEROUS_FRAGMENTS:
            cleaned = re.sub(re.escape(fragment), "[unsafe_reference]", cleaned, flags=re.IGNORECASE)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_llm_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_llm_payload(item) for key, item in value.items()}
    return value


def _framework_counts(framework: SolverFramework) -> Dict[str, int]:
    return {
        "feature_dimensions": len([item for item in framework.feature_dimensions if item.status != "retired"]),
        "strategies": len([item for item in framework.strategies if item.status != "retired"]),
        "skills": len([item for item in framework.skills if item.status != "retired"]),
    }


def _trim_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return history[-_MAX_HISTORY_ITEMS:]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _FileLock:
    def __init__(self, path: str) -> None:
        self.path = path
        self._handle: Any = None

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

            getattr(msvcrt, "locking")(self._handle.fileno(), getattr(msvcrt, "LK_LOCK"), 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)

    def _unlock(self) -> None:
        if self._handle is None:
            return
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt

            getattr(msvcrt, "locking")(self._handle.fileno(), getattr(msvcrt, "LK_UNLCK"), 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)


def _write_json_atomic(path: str, value: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)
