"""Shared data models for the modular AutoSolver Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


Rank = Tuple[int, int, float, float]


@dataclass
class Case:
    name: str
    text: str
    path: Optional[str] = None


@dataclass
class ParsedCase:
    rows: List[Tuple[Tuple[str, ...], str, str, float, float]]
    by_key: Dict[str, Dict[str, Tuple[float, float]]]
    key_tasks: Dict[str, Tuple[str, ...]]
    all_tasks: List[str]
    all_couriers: List[str]


@dataclass
class StrategySpec:
    name: str
    description: str
    implementation_notes: str
    suitable_features: List[str]
    example_signals: Dict[str, Any]
    risks: List[str] = field(default_factory=list)
    recommended_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverSkill:
    name: str
    strategy_names: List[str]
    construction_notes: str
    code_contract: str
    constraints: List[str]
    examples: List[str] = field(default_factory=list)


@dataclass
class Candidate:
    name: str
    code: str
    rationale: Dict[str, Any]
    iteration: int
    source: str = "llm"


@dataclass
class ValidationResult:
    valid: bool
    stage: str
    errors: List[Dict[str, Any]] = field(default_factory=list)
    runtime: float = 0.0
    answer: Any = None


@dataclass
class ScoreResult:
    name: str
    rank: Rank
    total_covered: int
    total_tasks: int
    total_penalty: float
    total_runtime: float
    failures: int
    cases: List[Dict[str, Any]]
    convergence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationArtifact:
    iteration: int
    candidate_name: str
    code_path: str
    rationale_path: str
    validation_path: str
    score_path: Optional[str] = None
    impact_path: Optional[str] = None
