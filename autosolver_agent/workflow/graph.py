"""Workflow orchestration for the modular AutoSolver Agent."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, TypedDict

from autosolver_agent.artifacts import ArtifactStore, serialize
from autosolver_agent.llm import LLMCodeGenerator
from autosolver_agent.memory import MemoryStore
from autosolver_agent.models import Candidate, Case, ParsedCase, ScoreResult
from autosolver_agent.skills import SolverSkillLibrary, StrategyLibrary
from autosolver_agent.tools import InstanceClassifier, Scorer, Validator

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - exercised only when dependency missing
    END = "__end__"
    StateGraph = None


class WorkflowState(TypedDict, total=False):
    phase: str
    iteration: int
    candidate: Candidate
    stop_reason: str


class AutoSolverWorkflow:
    def __init__(
        self,
        cases: List[Case],
        parsed_cases: List[ParsedCase],
        iterations: int,
        deadline: float,
        per_case_timeout: float,
        search_per_case_timeout: float,
        output_path: str,
        memory: MemoryStore,
        artifacts: ArtifactStore,
        llm: LLMCodeGenerator,
        verbose: bool = True,
        finalize_top_k: int = 3,
    ) -> None:
        self.cases = cases
        self.parsed_cases = parsed_cases
        self.iterations = max(1, iterations)
        self.deadline = deadline
        self.per_case_timeout = per_case_timeout
        self.search_per_case_timeout = search_per_case_timeout
        self.output_path = output_path
        self.memory = memory
        self.artifacts = artifacts
        self.llm = llm
        self.verbose = verbose
        self.finalize_top_k = max(1, finalize_top_k)
        self.classifier = InstanceClassifier()
        self.validator = Validator(smoke_timeout=min(2.0, search_per_case_timeout))
        self.scorer = Scorer(per_case_timeout=search_per_case_timeout)
        self.strategy_library = StrategyLibrary()
        self.solver_library = SolverSkillLibrary()
        self.instance_features: Dict[str, Any] = {}
        self.best_candidate: Optional[Candidate] = None
        self.best_score: Optional[ScoreResult] = None
        self.candidates: List[Candidate] = []
        self.scores: List[ScoreResult] = []
        self.validation_errors: List[Dict[str, Any]] = []
        self.impact_analysis: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        self.final_solver_path: Optional[str] = None

    def run(self) -> Dict[str, Any]:
        graph = self._build_graph()
        if graph is not None:
            graph.invoke({"phase": "classify", "iteration": 0})
        else:
            self._run_sequential()
        return self.report()

    def _build_graph(self) -> Any:
        if StateGraph is None:
            self.log("langgraph unavailable; running the same workflow sequentially")
            return None
        builder = StateGraph(WorkflowState)
        builder.add_node("classify", self._node_classify)
        builder.add_node("generate", self._node_generate)
        builder.add_node("validate_and_score", self._node_validate_and_score)
        builder.add_node("finalize", self._node_finalize)
        builder.set_entry_point("classify")
        builder.add_edge("classify", "generate")
        builder.add_edge("generate", "validate_and_score")
        builder.add_conditional_edges(
            "validate_and_score",
            lambda state: state.get("phase", "generate"),
            {"generate": "generate", "finalize": "finalize"},
        )
        builder.add_edge("finalize", END)
        return builder.compile()

    def _run_sequential(self) -> None:
        state: WorkflowState = {"phase": "classify", "iteration": 0}
        state.update(self._node_classify(state))
        while True:
            state.update(self._node_generate(state))
            state.update(self._node_validate_and_score(state))
            if state.get("phase") == "finalize":
                break
        state.update(self._node_finalize(state))

    def _node_classify(self, state: WorkflowState) -> WorkflowState:
        self.instance_features = self.classifier.classify(self.cases, self.parsed_cases)
        self.log(
            "classified instances: focus="
            + ",".join(self.instance_features.get("aggregate", {}).get("recommended_focus", []))
        )
        return {"phase": "generate", "iteration": 1}

    def _node_generate(self, state: WorkflowState) -> WorkflowState:
        iteration = int(state.get("iteration", 1))
        if self._time_exhausted():
            return {"phase": "finalize", "iteration": iteration, "stop_reason": "budget exhausted before generation"}
        aggregate_features = self.instance_features.get("aggregate", {})
        candidate = self.llm.generate(
            iteration=iteration,
            instance_features=self.instance_features,
            strategy_context=self.strategy_library.as_prompt_context(aggregate_features),
            solver_context=self.solver_library.as_prompt_context(),
            memory_digest=self.memory.digest(),
            disk_results=self.artifacts.disk_results(),
            previous_impact=self.impact_analysis,
            case_samples=self._case_samples(),
            per_case_timeout=self.per_case_timeout,
        )
        self.candidates.append(candidate)
        artifact = self.artifacts.save_candidate(candidate)
        candidate.rationale["_artifact_validation_path"] = artifact.validation_path
        self.memory.record_candidate(iteration, candidate.name, candidate.rationale, aggregate_features)
        self.log(f"iteration {iteration}: generated {candidate.name}")
        return {"phase": "validate", "iteration": iteration, "candidate": candidate}

    def _node_validate_and_score(self, state: WorkflowState) -> WorkflowState:
        iteration = int(state.get("iteration", 1))
        candidate = state.get("candidate")
        if candidate is None:
            return {"phase": "finalize", "iteration": iteration, "stop_reason": "missing candidate"}
        artifact = self.artifacts.artifacts[-1]
        validation = self.validator.validate(candidate.code, self.cases, self.parsed_cases)
        self.artifacts.save_validation(artifact, validation)
        self.memory.record_validation(iteration, validation)
        if not validation.valid:
            error_item = {"iteration": iteration, "candidate": candidate.name, "errors": validation.errors}
            self.validation_errors.append(error_item)
            self.log(f"iteration {iteration}: validation failed at {validation.stage}")
            return self._next_or_finalize(iteration, "validation failed")
        score = self.scorer.score(
            candidate=candidate,
            cases=self.cases,
            parsed_cases=self.parsed_cases,
            best=self.best_score,
            timeout=self.search_per_case_timeout,
        )
        self.scores.append(score)
        self.artifacts.save_score(artifact, score)
        impact = self._impact(candidate, score)
        self.impact_analysis.append(impact)
        self.artifacts.save_impact(artifact, impact)
        self.memory.record_score(iteration, score, impact)
        if self.best_score is None or score.rank < self.best_score.rank:
            self.best_score = score
            self.best_candidate = candidate
            self.log(
                f"iteration {iteration}: new best {candidate.name} "
                f"covered={score.total_covered}/{score.total_tasks} penalty={score.total_penalty:.4f}"
            )
        else:
            self.log(f"iteration {iteration}: no improvement ({candidate.name})")
        return self._next_or_finalize(iteration, "iteration limit reached")

    def _node_finalize(self, state: WorkflowState) -> WorkflowState:
        if not self.candidates:
            raise RuntimeError("No LLM candidate was generated; cannot finalize.")
        chosen = self._finalize_recheck()
        if chosen is None or self.best_candidate is None:
            raise RuntimeError("No scored candidate survived validation; cannot finalize.")
        output_path = self.output_path if os.path.isabs(self.output_path) else os.path.abspath(self.output_path)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(self.best_candidate.code)
        self.final_solver_path = output_path
        self.memory.save(short_term_path=os.path.join(self.memory.memory_dir, "short_term_last_run.json"))
        self.log(f"finalized {self.best_candidate.name} -> {output_path}")
        return {"phase": "done", "iteration": int(state.get("iteration", self.iterations))}

    def _next_or_finalize(self, iteration: int, reason: str) -> WorkflowState:
        next_iteration = iteration + 1
        if next_iteration > self.iterations:
            return {"phase": "finalize", "iteration": iteration, "stop_reason": reason}
        if self._time_exhausted():
            return {"phase": "finalize", "iteration": iteration, "stop_reason": "budget exhausted"}
        return {"phase": "generate", "iteration": next_iteration}

    def _finalize_recheck(self) -> Optional[ScoreResult]:
        scored = sorted(self.scores, key=lambda score: score.rank)[: self.finalize_top_k]
        if not scored:
            return None
        candidates_by_name = {candidate.name: candidate for candidate in self.candidates}
        best_pair: Optional[tuple[ScoreResult, Candidate]] = None
        final_scorer = Scorer(per_case_timeout=self.per_case_timeout)
        for prior_score in scored:
            candidate = candidates_by_name.get(prior_score.name)
            if candidate is None:
                continue
            self.log(f"finalize recheck: {candidate.name}")
            final_score = final_scorer.score(
                candidate=candidate,
                cases=self.cases,
                parsed_cases=self.parsed_cases,
                best=best_pair[0] if best_pair else None,
                timeout=self.per_case_timeout,
            )
            if best_pair is None or _final_rank(final_score) < _final_rank(best_pair[0]):
                best_pair = (final_score, candidate)
        if best_pair is None:
            return None
        self.best_score, self.best_candidate = best_pair
        return self.best_score

    def _impact(self, candidate: Candidate, score: ScoreResult) -> Dict[str, Any]:
        rationale = candidate.rationale or {}
        return {
            "iteration": candidate.iteration,
            "candidate": candidate.name,
            "strategy_combination": rationale.get("strategy_combination"),
            "parameter_changes": rationale.get("parameter_changes"),
            "expected_effect": rationale.get("expected_effect"),
            "actual_rank": list(score.rank),
            "actual_penalty": round(score.total_penalty, 6),
            "actual_covered": score.total_covered,
            "convergence": score.convergence,
            "analysis": (
                "improved global best"
                if score.convergence.get("is_improved")
                else "did not improve global best"
            ),
        }

    def _case_samples(self) -> List[str]:
        samples = []
        for case in self.cases[:3]:
            lines = case.text.splitlines()
            samples.append(case.name + "\n" + "\n".join(lines[:12]))
        return samples

    def _time_exhausted(self) -> bool:
        return time.time() + self.search_per_case_timeout * max(1, len(self.cases)) + 0.5 >= self.deadline

    def report(self) -> Dict[str, Any]:
        return {
            "output_path": self.final_solver_path or os.path.abspath(self.output_path),
            "cases": [case.name for case in self.cases],
            "iterations_requested": self.iterations,
            "iterations_completed": len(self.candidates),
            "best": serialize(self.best_score) if self.best_score else None,
            "instance_features": self.instance_features,
            "short_term_memory": self.memory.short_term,
            "long_term_memory_digest": self.memory.digest(),
            "artifacts": self.artifacts.summary(),
            "validation_errors": self.validation_errors,
            "convergence": [score.convergence for score in self.scores],
            "impact_analysis": self.impact_analysis,
            "notes": self.notes,
        }

    def log(self, message: str) -> None:
        self.notes.append(message)
        if self.verbose:
            print(f"[agent] {message}", flush=True)


def _final_rank(score: ScoreResult) -> tuple:
    return (score.failures, -score.total_covered, score.total_penalty)
