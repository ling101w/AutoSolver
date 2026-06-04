"""Internal workflow services and run state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from autosolver_agent.events import EventRecorder, PhaseTimer
from autosolver_agent.models import Candidate


@dataclass
class WorkflowConfig:
    iterations: int
    deadline: float
    per_case_timeout: float
    search_per_case_timeout: float
    output_path: str
    finalize_top_k: int
    max_repair_attempts: int
    memory_top_k: int
    bandit_exploration: float
    summary_output_path: Optional[str] = None


@dataclass
class WorkflowRunState:
    run_id: str
    event_log_path: str
    case_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    timings: PhaseTimer = field(default_factory=PhaseTimer)
    candidate_hashes: Dict[str, str] = field(default_factory=dict)


class GenerationService:
    def __init__(self, workflow: Any) -> None:
        self.workflow = workflow

    def generate(self, state: Any) -> Dict[str, Any]:
        return self.workflow._generate_candidate(state)


class EvaluationService:
    def __init__(self, workflow: Any) -> None:
        self.workflow = workflow

    def validate_and_score(self, state: Any) -> Dict[str, Any]:
        return self.workflow._validate_and_score_candidate(state)


class RepairService:
    def __init__(self, workflow: Any) -> None:
        self.workflow = workflow

    def repair_schema_failure(self, *args: Any, **kwargs: Any) -> Candidate:
        return self.workflow._repair_schema_failure(*args, **kwargs)

    def repair_validation_failure(self, *args: Any, **kwargs: Any) -> Candidate:
        return self.workflow._repair_validation_failure(*args, **kwargs)


class FinalizationService:
    def __init__(self, workflow: Any) -> None:
        self.workflow = workflow

    def finalize(self, state: Any) -> Dict[str, Any]:
        return self.workflow._finalize_run(state)


class ReportBuilder:
    def __init__(self, workflow: Any) -> None:
        self.workflow = workflow

    def build(self) -> Dict[str, Any]:
        return self.workflow._report_payload()


def build_event_recorder(path: str, run_id: str) -> EventRecorder:
    return EventRecorder(path=path, run_id=run_id, truncate=True)
