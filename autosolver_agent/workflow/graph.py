"""Workflow orchestration for the modular AutoSolver Agent."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from autosolver_agent.artifacts import ArtifactStore, serialize
from autosolver_agent.events import code_hash, now_monotonic
from autosolver_agent.llm import LLMCodeGenerator
from autosolver_agent.llm.schema import SolverPlan
from autosolver_agent.memory import MemoryStore
from autosolver_agent.models import Candidate, Case, IterationArtifact, ParsedCase, ScoreResult, ValidationResult
from autosolver_agent.skills import SolverSkillLibrary, StrategyLibrary
from autosolver_agent.tools import InstanceClassifier, PlannerToolbox, Scorer, Validator
from autosolver_agent.workflow.services import (
    EvaluationService,
    FinalizationService,
    GenerationService,
    RepairService,
    ReportBuilder,
    WorkflowConfig,
    WorkflowRunState,
    build_event_recorder,
)

LANGGRAPH_END: Any = "__end__"
LangGraphStateGraph: Any = None
try:
    from langgraph.graph import END as _LANGGRAPH_END
    from langgraph.graph import StateGraph as _LANGGRAPH_STATE_GRAPH

    LANGGRAPH_END = _LANGGRAPH_END
    LangGraphStateGraph = _LANGGRAPH_STATE_GRAPH
except Exception:  # pragma: no cover - exercised only when dependency missing
    pass


class WorkflowState(TypedDict, total=False):
    phase: str
    iteration: int
    candidate: Candidate
    plan: SolverPlan
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
        max_repair_attempts: int = 2,
        memory_top_k: int = 5,
        bandit_exploration: float = 1.4,
        summary_output_path: Optional[str] = None,
        case_diagnostics: Optional[List[Dict[str, Any]]] = None,
        event_log_path: Optional[str] = None,
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
        self.max_repair_attempts = max(0, max_repair_attempts)
        self.memory_top_k = max(1, memory_top_k)
        self.bandit_exploration = bandit_exploration
        self.summary_output_path = summary_output_path
        self.config = WorkflowConfig(
            iterations=self.iterations,
            deadline=self.deadline,
            per_case_timeout=self.per_case_timeout,
            search_per_case_timeout=self.search_per_case_timeout,
            output_path=self.output_path,
            finalize_top_k=self.finalize_top_k,
            max_repair_attempts=self.max_repair_attempts,
            memory_top_k=self.memory_top_k,
            bandit_exploration=self.bandit_exploration,
            summary_output_path=self.summary_output_path,
        )
        run_id = uuid.uuid4().hex
        resolved_event_log = event_log_path or os.path.join(self.artifacts.artifact_dir, "events.jsonl")
        self.run_state = WorkflowRunState(
            run_id=run_id,
            event_log_path=os.path.abspath(resolved_event_log),
            case_diagnostics=case_diagnostics or [],
        )
        self.events = build_event_recorder(self.run_state.event_log_path, run_id)
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
        self.planner_trace: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.repair_history: List[Dict[str, Any]] = []
        self.memory_retrieval: List[Dict[str, Any]] = []
        self.bandit_trace: List[Dict[str, Any]] = []
        self.experiment_records: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        self.final_solver_path: Optional[str] = None
        self.generation_service = GenerationService(self)
        self.evaluation_service = EvaluationService(self)
        self.repair_service = RepairService(self)
        self.finalization_service = FinalizationService(self)
        self.report_builder = ReportBuilder(self)

    def run(self) -> Dict[str, Any]:
        graph = self._build_graph()
        if graph is not None:
            graph.invoke({"phase": "classify", "iteration": 0})
        else:
            self._run_sequential()
        return self.report()

    def _build_graph(self) -> Any:
        if LangGraphStateGraph is None:
            self.log("langgraph unavailable; running the same workflow sequentially")
            return None
        builder = LangGraphStateGraph(WorkflowState)
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
        builder.add_edge("finalize", LANGGRAPH_END)
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
        return self._timed_node("classify", int(state.get("iteration", 0)), lambda: self._classify_instances(state))

    def _node_generate(self, state: WorkflowState) -> WorkflowState:
        return self._timed_node(
            "generate",
            int(state.get("iteration", 1)),
            lambda: self.generation_service.generate(state),
        )

    def _node_validate_and_score(self, state: WorkflowState) -> WorkflowState:
        return self._timed_node(
            "validate_and_score",
            int(state.get("iteration", 1)),
            lambda: self.evaluation_service.validate_and_score(state),
        )

    def _node_finalize(self, state: WorkflowState) -> WorkflowState:
        return self._timed_node(
            "finalize",
            int(state.get("iteration", self.iterations)),
            lambda: self.finalization_service.finalize(state),
        )

    def _classify_instances(self, state: WorkflowState) -> WorkflowState:
        self.instance_features = self.classifier.classify(self.cases, self.parsed_cases)
        self.log(
            "classified instances: focus="
            + ",".join(self.instance_features.get("aggregate", {}).get("recommended_focus", []))
        )
        self._record_event(
            "instances_classified",
            phase="classify",
            iteration=int(state.get("iteration", 0)),
            context={"case_count": len(self.cases), "diagnostics": len(self.run_state.case_diagnostics)},
        )
        return {"phase": "generate", "iteration": 1}

    def _generate_candidate(self, state: WorkflowState) -> WorkflowState:
        iteration = int(state.get("iteration", 1))
        if self._time_exhausted():
            return {"phase": "finalize", "iteration": iteration, "stop_reason": "budget exhausted before generation"}

        aggregate_features = self.instance_features.get("aggregate", {})
        strategy_context = self.strategy_library.as_prompt_context(aggregate_features)
        solver_context = self.solver_library.as_prompt_context()
        memory_digest = self.memory.digest(
            features=aggregate_features,
            top_k=self.memory_top_k,
            exploration=self.bandit_exploration,
        )
        best_summary = self._best_code_summary()
        toolbox = PlannerToolbox(
            instance_features=self.instance_features,
            strategy_context=strategy_context,
            memory=self.memory,
            artifacts=self.artifacts,
            feature_query=aggregate_features,
            memory_top_k=self.memory_top_k,
            bandit_exploration=self.bandit_exploration,
            best_summary=best_summary,
        )
        plan = self.llm.plan(
            iteration=iteration,
            instance_features=self.instance_features,
            strategy_context=strategy_context,
            memory_digest=memory_digest,
            previous_impact=self.impact_analysis,
            toolbox=toolbox,
        )
        tool_context = toolbox.snapshot()
        self.planner_trace.append(
            {"iteration": iteration, **(self.llm.last_planner_trace[-1] if self.llm.last_planner_trace else {})}
        )
        self.tool_calls.append({"iteration": iteration, "calls": self.llm.last_tool_calls or toolbox.trace})
        self.memory_retrieval.append(
            {"iteration": iteration, "similar_experiments": tool_context.get("similar_experiments", [])}
        )
        self.bandit_trace.append({"iteration": iteration, "recommendations": tool_context.get("bandit_recommendations", [])})

        try:
            candidate = self.llm.generate_from_plan(
                iteration=iteration,
                plan=plan,
                instance_features=self.instance_features,
                solver_context=solver_context,
                memory_digest=memory_digest,
                disk_results=self.artifacts.disk_results(),
                previous_impact=self.impact_analysis,
                case_samples=self._case_samples(),
                per_case_timeout=self.per_case_timeout,
                tool_context=tool_context,
            )
        except Exception as exc:
            if self.max_repair_attempts <= 0:
                raise
            candidate = self.repair_service.repair_schema_failure(
                iteration=iteration,
                plan=plan,
                error=str(exc),
                raw_response=getattr(exc, "raw_response", ""),
                memory_digest=memory_digest,
                best_summary=best_summary,
            )

        self.candidates.append(candidate)
        candidate_hash = self._remember_candidate_hash(candidate)
        artifact = self.artifacts.save_candidate(candidate)
        candidate.rationale["_artifact_validation_path"] = artifact.validation_path
        self.memory.record_candidate(iteration, candidate.name, candidate.rationale, aggregate_features)
        self._record_event(
            "candidate_generated",
            phase="generate",
            iteration=iteration,
            candidate=candidate.name,
            candidate_hash=candidate_hash,
            context={"plan": plan.name, "source": candidate.source},
        )
        self.log(f"iteration {iteration}: planned {plan.name} -> generated {candidate.name}")
        return {"phase": "validate", "iteration": iteration, "candidate": candidate, "plan": plan}

    def _validate_and_score_candidate(self, state: WorkflowState) -> WorkflowState:
        iteration = int(state.get("iteration", 1))
        candidate = state.get("candidate")
        if candidate is None:
            return {"phase": "finalize", "iteration": iteration, "stop_reason": "missing candidate"}

        artifact = self.artifacts.artifacts[-1]
        validation = self.validator.validate(candidate.code, self.cases, self.parsed_cases)
        if not validation.valid and self.max_repair_attempts > 0:
            repaired = self.repair_service.repair_validation_failure(iteration, state, candidate, validation)
            if repaired is not candidate:
                candidate = repaired
                state["candidate"] = candidate
                artifact = self.artifacts.artifacts[-1]
                validation = self.validator.validate(candidate.code, self.cases, self.parsed_cases)

        self.artifacts.save_validation(artifact, validation)
        self.memory.record_validation(iteration, validation)
        if not validation.valid:
            error_item = {"iteration": iteration, "candidate": candidate.name, "errors": validation.errors}
            self.validation_errors.append(error_item)
            self._record_experiment(iteration, candidate, validation=validation, artifact=artifact, failure_reason="validation failed")
            self._record_event(
                "validation_failed",
                phase="validate_and_score",
                iteration=iteration,
                candidate=candidate.name,
                candidate_hash=self.run_state.candidate_hashes.get(candidate.name),
                context={"stage": validation.stage, "errors": validation.errors},
            )
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
        self._record_experiment(iteration, candidate, score=score, validation=validation, artifact=artifact)
        self._record_event(
            "candidate_scored",
            phase="validate_and_score",
            iteration=iteration,
            candidate=candidate.name,
            candidate_hash=self.run_state.candidate_hashes.get(candidate.name),
            context={"rank": list(score.rank), "covered": score.total_covered, "penalty": score.total_penalty},
        )
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

    def _finalize_run(self, state: WorkflowState) -> WorkflowState:
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
        if self.summary_output_path:
            from autosolver_agent.artifacts import write_json

            write_json(self.summary_output_path, self._summary())
        self._record_event(
            "solver_finalized",
            phase="finalize",
            iteration=int(state.get("iteration", self.iterations)),
            candidate=self.best_candidate.name,
            candidate_hash=self.run_state.candidate_hashes.get(self.best_candidate.name),
            context={"output_path": output_path},
        )
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
        return self.report_builder.build()

    def _report_payload(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_state.run_id,
            "output_path": self.final_solver_path or os.path.abspath(self.output_path),
            "event_log_path": self.run_state.event_log_path,
            "case_diagnostics": self.run_state.case_diagnostics,
            "timings": self.run_state.timings.timings,
            "candidate_hashes": self.run_state.candidate_hashes,
            "cases": [case.name for case in self.cases],
            "iterations_requested": self.iterations,
            "iterations_completed": len({candidate.iteration for candidate in self.candidates}),
            "best": serialize(self.best_score) if self.best_score else None,
            "instance_features": self.instance_features,
            "short_term_memory": self.memory.short_term,
            "long_term_memory_digest": self.memory.digest(),
            "planner_trace": self.planner_trace,
            "tool_calls": self.tool_calls,
            "repair_history": self.repair_history,
            "memory_retrieval": self.memory_retrieval,
            "bandit": self.bandit_trace,
            "experiments": self.experiment_records,
            "artifacts": self.artifacts.summary(),
            "validation_errors": self.validation_errors,
            "convergence": [score.convergence for score in self.scores],
            "impact_analysis": self.impact_analysis,
            "summary": self._summary(),
            "notes": self.notes,
        }

    def log(self, message: str) -> None:
        self.notes.append(message)
        self._record_event("log", phase="log", message=message)
        if self.verbose:
            print(f"[agent] {message}", flush=True)

    def _timed_node(self, phase: str, iteration: int, func: Any) -> WorkflowState:
        started = now_monotonic()
        self._record_event("phase_started", phase=phase, iteration=iteration)
        try:
            result = func()
        except Exception as exc:
            elapsed = now_monotonic() - started
            self.run_state.timings.mark(phase, elapsed)
            self._record_event(
                "phase_failed",
                phase=phase,
                iteration=iteration,
                elapsed=elapsed,
                context={"error": str(exc)},
            )
            raise
        elapsed = now_monotonic() - started
        self.run_state.timings.mark(phase, elapsed)
        self._record_event("phase_completed", phase=phase, iteration=iteration, elapsed=elapsed)
        return result

    def _record_event(
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
    ) -> None:
        self.events.record(
            event,
            phase=phase,
            iteration=iteration,
            message=message,
            candidate=candidate,
            candidate_hash=candidate_hash,
            elapsed=elapsed,
            context=context,
        )

    def _remember_candidate_hash(self, candidate: Candidate) -> str:
        value = code_hash(candidate.code)
        self.run_state.candidate_hashes[candidate.name] = value
        return value

    def _repair_schema_failure(
        self,
        iteration: int,
        plan: SolverPlan,
        error: str,
        raw_response: str,
        memory_digest: Dict[str, Any],
        best_summary: Dict[str, Any],
    ) -> Candidate:
        last_error = error
        for attempt in range(1, self.max_repair_attempts + 1):
            try:
                candidate = self.llm.repair(
                    iteration=iteration,
                    plan=plan,
                    errors=[{"type": "schema_error", "message": last_error}],
                    instance_features=self.instance_features,
                    solver_context=self.solver_library.as_prompt_context(),
                    memory_digest=memory_digest,
                    case_samples=self._case_samples(),
                    per_case_timeout=self.per_case_timeout,
                    raw_response=raw_response,
                    best_summary=best_summary,
                    attempt=attempt,
                )
                self.repair_history.append(
                    {"iteration": iteration, "attempt": attempt, "reason": "schema_error", "candidate": candidate.name}
                )
                return candidate
            except Exception as exc:
                last_error = str(exc)
                self.repair_history.append(
                    {"iteration": iteration, "attempt": attempt, "reason": "schema_error", "error": last_error}
                )
                self._record_event(
                    "repair_failed",
                    phase="generate",
                    iteration=iteration,
                    context={"attempt": attempt, "reason": "schema_error", "error": last_error},
                )
        raise RuntimeError(f"LLM structured generation failed after repair attempts: {last_error}")

    def _repair_validation_failure(
        self,
        iteration: int,
        state: WorkflowState,
        candidate: Candidate,
        validation: ValidationResult,
    ) -> Candidate:
        plan = state.get("plan")
        if plan is None:
            plan = self.llm._fallback_plan(iteration, self.instance_features, self.memory.digest())
        aggregate_features = self.instance_features.get("aggregate", {})
        memory_digest = self.memory.digest(
            features=aggregate_features,
            top_k=self.memory_top_k,
            exploration=self.bandit_exploration,
        )
        best_summary = self._best_code_summary()
        current = candidate
        current_validation = validation
        for attempt in range(1, self.max_repair_attempts + 1):
            try:
                repaired = self.llm.repair(
                    iteration=iteration,
                    plan=plan,
                    errors=current_validation.errors,
                    instance_features=self.instance_features,
                    solver_context=self.solver_library.as_prompt_context(),
                    memory_digest=memory_digest,
                    case_samples=self._case_samples(),
                    per_case_timeout=self.per_case_timeout,
                    failed_code=current.code,
                    failed_rationale=current.rationale,
                    best_summary=best_summary,
                    score_delta=self._last_score_delta(),
                    attempt=attempt,
                )
            except Exception as exc:
                self.repair_history.append(
                    {
                        "iteration": iteration,
                        "attempt": attempt,
                        "reason": "validation_error",
                        "from": current.name,
                        "error": str(exc),
                    }
                )
                self._record_event(
                    "repair_failed",
                    phase="validate_and_score",
                    iteration=iteration,
                    candidate=current.name,
                    candidate_hash=self.run_state.candidate_hashes.get(current.name),
                    context={"attempt": attempt, "reason": "validation_error", "error": str(exc)},
                )
                continue
            self.candidates.append(repaired)
            repaired_hash = self._remember_candidate_hash(repaired)
            artifact = self.artifacts.save_candidate(repaired)
            repaired.rationale["_artifact_validation_path"] = artifact.validation_path
            self.memory.record_candidate(iteration, repaired.name, repaired.rationale, aggregate_features)
            self.repair_history.append(
                {
                    "iteration": iteration,
                    "attempt": attempt,
                    "reason": "validation_error",
                    "from": current.name,
                    "candidate": repaired.name,
                    "errors": current_validation.errors,
                }
            )
            self._record_event(
                "candidate_repaired",
                phase="validate_and_score",
                iteration=iteration,
                candidate=repaired.name,
                candidate_hash=repaired_hash,
                context={"attempt": attempt, "from": current.name},
            )
            current_validation = self.validator.validate(repaired.code, self.cases, self.parsed_cases)
            if current_validation.valid:
                return repaired
            self.artifacts.save_validation(artifact, current_validation)
            self.memory.record_validation(iteration, current_validation)
            self._record_experiment(
                iteration,
                repaired,
                validation=current_validation,
                artifact=artifact,
                failure_reason="repair validation failed",
            )
            current = repaired
        return current

    def _record_experiment(
        self,
        iteration: int,
        candidate: Candidate,
        score: Optional[ScoreResult] = None,
        validation: Optional[ValidationResult] = None,
        artifact: Optional[IterationArtifact] = None,
        failure_reason: Optional[str] = None,
    ) -> None:
        rationale = candidate.rationale or {}
        features = self.instance_features.get("aggregate", {})
        artifact_paths = serialize(artifact) if artifact is not None else {}
        record = self.memory.record_experiment(
            iteration=iteration,
            candidate_name=candidate.name,
            features=features,
            strategy=rationale.get("strategy_combination") or rationale.get("plan", {}).get("strategy_combination", []),
            params=rationale.get("parameter_changes") or rationale.get("plan", {}).get("parameter_changes", {}),
            score=score,
            validation=validation,
            artifact_paths=artifact_paths,
            failure_reason=failure_reason,
        )
        self.experiment_records.append(record)

    def _best_code_summary(self) -> Dict[str, Any]:
        summary = self.memory.best_experiment_summary()
        if self.best_candidate is not None:
            summary = dict(summary)
            summary.update(
                {
                    "current_best": self.best_candidate.name,
                    "score": serialize(self.best_score) if self.best_score else None,
                    "code_excerpt": self.best_candidate.code[:1200],
                }
            )
        return summary

    def _last_score_delta(self) -> Dict[str, Any]:
        if not self.scores:
            return {}
        last = self.scores[-1]
        return {
            "name": last.name,
            "convergence": last.convergence,
            "rank": list(last.rank),
            "penalty": last.total_penalty,
            "covered": last.total_covered,
        }

    def _summary(self) -> Dict[str, Any]:
        return {
            "iterations_requested": self.iterations,
            "iterations_completed": len({candidate.iteration for candidate in self.candidates}),
            "candidates_generated": len(self.candidates),
            "repairs_attempted": len(self.repair_history),
            "valid_scores": len(self.scores),
            "validation_failures": len(self.validation_errors),
            "best_candidate": self.best_candidate.name if self.best_candidate else None,
            "best_rank": list(self.best_score.rank) if self.best_score else None,
            "best_penalty": round(self.best_score.total_penalty, 6) if self.best_score else None,
            "best_covered": self.best_score.total_covered if self.best_score else None,
            "best_tasks": self.best_score.total_tasks if self.best_score else None,
            "bandit_arms": self.memory.long_term.get("bandit_arms", {}),
        }


def _final_rank(score: ScoreResult) -> tuple:
    return (score.failures, -score.total_covered, score.total_penalty)
