"""Structured LLM contracts for AutoSolver generation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SolverPlan(BaseModel):
    """High-level plan produced before code generation."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, max_length=80)
    strategy_combination: List[str]
    parameter_changes: Dict[str, Any]
    exploration_mode: str
    reasoning: str
    risk_control: str
    generation_directives: List[str]

    @field_validator("strategy_combination", mode="after")
    @classmethod
    def _strategy_not_empty(cls, value: List[str]) -> List[str]:
        strategies = [str(item) for item in value if str(item).strip()]
        if not strategies:
            raise ValueError("strategy_combination must contain at least one strategy")
        return strategies


class CandidateRationale(BaseModel):
    """Rationale metadata for one generated solver candidate."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, max_length=80)
    idea: str
    strategy_combination: List[str]
    parameter_changes: Dict[str, Any]
    expected_effect: str
    risk_control: str
    plan_summary: Optional[str] = None

    @field_validator("name", mode="after")
    @classmethod
    def _sanitize_name(cls, value: str) -> str:
        import re

        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
        return (cleaned or "llm_candidate")[:80]

    @field_validator("strategy_combination", mode="after")
    @classmethod
    def _strategy_not_empty(cls, value: List[str]) -> List[str]:
        strategies = [str(item) for item in value if str(item).strip()]
        if not strategies:
            raise ValueError("strategy_combination must contain at least one strategy")
        return strategies


class CandidateEnvelope(BaseModel):
    """Structured LLM output containing rationale and full solver code."""

    rationale: CandidateRationale
    code: str = Field(min_length=20)

    @field_validator("code", mode="after")
    @classmethod
    def _code_has_contract(cls, value: str) -> str:
        code = value.strip()
        if "def solve" not in code:
            raise ValueError("code must define solve(input_text: str) -> list")
        return code


class StructuredOutputError(RuntimeError):
    """Raised when an LLM response cannot be converted to the expected schema."""

    def __init__(self, message: str, raw_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response


def parse_solver_plan(text_or_value: Any) -> SolverPlan:
    value = _load_json_like(text_or_value)
    if not isinstance(value, dict):
        raise StructuredOutputError("solver plan response is not a JSON object", str(text_or_value)[:4000])
    return SolverPlan.model_validate(value)


def parse_candidate_envelope(text_or_value: Any) -> CandidateEnvelope:
    """Parse the required JSON envelope."""

    raw = text_or_value if isinstance(text_or_value, str) else json.dumps(text_or_value, ensure_ascii=False)
    value = _load_json_like(text_or_value)
    if not isinstance(value, dict):
        raise StructuredOutputError("candidate response is not a JSON object", raw[:8000])
    if "rationale" not in value or "code" not in value:
        raise StructuredOutputError("candidate response must contain rationale and code", raw[:8000])
    try:
        return CandidateEnvelope.model_validate(value)
    except Exception as exc:
        raise StructuredOutputError(f"candidate response did not match CandidateEnvelope schema: {exc}", raw[:8000]) from exc


def candidate_schema_text() -> str:
    return json.dumps(CandidateEnvelope.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)


def plan_schema_text() -> str:
    return json.dumps(SolverPlan.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)


def model_dump(value: BaseModel) -> Dict[str, Any]:
    return value.model_dump(mode="json")


def _load_json_like(text_or_value: Any) -> Any:
    if isinstance(text_or_value, (dict, list)):
        return text_or_value
    text = str(text_or_value).strip()
    try:
        return json.loads(text)
    except Exception as exc:
        raise StructuredOutputError(f"response is not valid JSON: {exc}", text[:4000]) from exc
