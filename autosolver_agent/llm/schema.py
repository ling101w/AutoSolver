"""Structured LLM contracts for AutoSolver generation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SolverPlan(BaseModel):
    """High-level plan produced before code generation."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(default="solver_plan", min_length=1, max_length=80)
    strategy_combination: List[str] = Field(default_factory=list)
    parameter_changes: Dict[str, Any] = Field(default_factory=dict)
    exploration_mode: str = Field(default="balanced")
    reasoning: str = Field(default="")
    risk_control: str = Field(default="")
    generation_directives: List[str] = Field(default_factory=list)

    @field_validator("strategy_combination", mode="after")
    @classmethod
    def _strategy_not_empty(cls, value: List[str]) -> List[str]:
        return [str(item) for item in value if str(item).strip()] or ["expected_greedy", "local_search_repair"]


class CandidateRationale(BaseModel):
    """Rationale metadata for one generated solver candidate."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, max_length=80)
    idea: str = Field(default="")
    strategy_combination: List[str] = Field(default_factory=list)
    parameter_changes: Dict[str, Any] = Field(default_factory=dict)
    expected_effect: str = Field(default="")
    risk_control: str = Field(default="")
    plan_summary: Optional[str] = None

    @field_validator("name", mode="after")
    @classmethod
    def _sanitize_name(cls, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
        return (cleaned or "llm_candidate")[:80]

    @field_validator("strategy_combination", mode="after")
    @classmethod
    def _strategy_not_empty(cls, value: List[str]) -> List[str]:
        return [str(item) for item in value if str(item).strip()] or ["expected_greedy"]


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
    if "plan" in value and isinstance(value["plan"], dict):
        value = value["plan"]
    return SolverPlan.model_validate(value)


def parse_candidate_envelope(text_or_value: Any) -> CandidateEnvelope:
    """Parse the preferred JSON envelope, with legacy code-block fallback."""

    raw = text_or_value if isinstance(text_or_value, str) else json.dumps(text_or_value, ensure_ascii=False)
    try:
        value = _load_json_like(text_or_value)
        if isinstance(value, dict):
            if "rationale" in value and "code" in value:
                return CandidateEnvelope.model_validate(value)
            if "code" in value:
                rationale = {key: value.get(key) for key in _RATIONALE_KEYS if key in value}
                return CandidateEnvelope.model_validate({"rationale": rationale, "code": value["code"]})
    except Exception as exc:
        first_error = exc
    else:
        first_error = None

    legacy = _legacy_candidate(raw)
    if legacy is not None:
        return CandidateEnvelope.model_validate(legacy)
    message = "candidate response did not match CandidateEnvelope schema"
    if first_error is not None:
        message += f": {first_error}"
    raise StructuredOutputError(message, raw[:8000])


def candidate_schema_text() -> str:
    return json.dumps(CandidateEnvelope.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)


def plan_schema_text() -> str:
    return json.dumps(SolverPlan.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)


def model_dump(value: BaseModel) -> Dict[str, Any]:
    return value.model_dump(mode="json")


_RATIONALE_KEYS = {
    "name",
    "idea",
    "strategy_combination",
    "parameter_changes",
    "expected_effect",
    "risk_control",
    "plan_summary",
}


def _load_json_like(text_or_value: Any) -> Any:
    if isinstance(text_or_value, (dict, list)):
        return text_or_value
    text = str(text_or_value).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for raw in re.findall(r"```json\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        try:
            return json.loads(raw)
        except Exception:
            continue
    return None


def _legacy_candidate(text: str) -> Optional[Dict[str, Any]]:
    metadata: Dict[str, Any] = {}
    for raw in re.findall(r"```json\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        try:
            value = json.loads(raw)
        except Exception:
            continue
        if isinstance(value, dict):
            metadata = value
            break
    code_blocks = re.findall(r"```(?:python|py)\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    code = code_blocks[-1].strip() if code_blocks else ""
    if not code and "def solve" in text:
        code = text[text.index("def solve") :].strip()
    if not code:
        return None
    rationale = {key: metadata.get(key) for key in _RATIONALE_KEYS if key in metadata}
    if not rationale.get("name"):
        rationale["name"] = "legacy_candidate"
    if not rationale.get("strategy_combination"):
        rationale["strategy_combination"] = ["expected_greedy"]
    return {"rationale": rationale, "code": code}
