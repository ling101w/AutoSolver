"""Workflow orchestration for the modular AutoSolver Agent."""

from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END as LANGGRAPH_END
from langgraph.graph import StateGraph as LangGraphStateGraph

from autosolver_agent.artifacts import ArtifactStore, serialize
from autosolver_agent.caseio import aggregate_features, dataset_features
from autosolver_agent.events import code_hash, now_monotonic
from autosolver_agent.framework import FrameworkStore
from autosolver_agent.llm import LLMCodeGenerator
from autosolver_agent.llm.generator import sanitize_name
from autosolver_agent.llm.schema import SolverPlan, model_dump
from autosolver_agent.memory import MemoryStore
from autosolver_agent.models import Candidate, Case, IterationArtifact, ParsedCase, ScoreResult, ValidationResult
from autosolver_agent.tools import PlannerToolbox, Scorer, Validator
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


class WorkflowState(TypedDict, total=False):
    phase: str
    iteration: int
    candidate: Candidate
    candidates: List[Candidate]
    plan: SolverPlan
    plans: List[SolverPlan]
    stop_reason: str


class CandidateEvaluationItem(TypedDict):
    candidate: Candidate
    validation: ValidationResult
    score: Optional[ScoreResult]
    impact: Optional[Dict[str, Any]]


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
        strategy_workers: int = 1,
        summary_output_path: Optional[str] = None,
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
        self.strategy_workers = max(1, int(strategy_workers or 1))
        self.iteration_stride = 1
        self.summary_output_path = summary_output_path
        self.config = WorkflowConfig(
            iterations=self.iterations,
            deadline=self.deadline,
            per_case_timeout=self.per_case_timeout,
            search_per_case_timeout=self.search_per_case_timeout,
            strategy_workers=self.strategy_workers,
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
        )
        self.events = build_event_recorder(self.run_state.event_log_path, run_id)
        self.framework_store = FrameworkStore(self.memory.memory_dir)
        self.validator = Validator(smoke_timeout=min(2.0, search_per_case_timeout))
        self.scorer = Scorer(per_case_timeout=search_per_case_timeout)
        self.instance_features: Dict[str, Any] = {}
        self.objective_features: Dict[str, Any] = {}
        self.best_candidate: Optional[Candidate] = None
        self.best_score: Optional[ScoreResult] = None
        self.candidates: List[Candidate] = []
        self.candidate_artifacts: Dict[str, IterationArtifact] = {}
        self.scores: List[ScoreResult] = []
        self.validation_errors: List[Dict[str, Any]] = []
        self.impact_analysis: List[Dict[str, Any]] = []
        self.planner_trace: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.repair_history: List[Dict[str, Any]] = []
        self.memory_retrieval: List[Dict[str, Any]] = []
        self.bandit_trace: List[Dict[str, Any]] = []
        self.experiment_records: List[Dict[str, Any]] = []
        self.framework_updates: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        self.final_solver_path: Optional[str] = None
        self.generation_service = GenerationService(self)
        self.evaluation_service = EvaluationService(self)
        self.repair_service = RepairService(self)
        self.finalization_service = FinalizationService(self)
        self.report_builder = ReportBuilder(self)

    def run(self) -> Dict[str, Any]:
        graph = self._build_graph()
        graph.invoke({"phase": "classify", "iteration": 0})
        return self.report()

    def run_worker_loop(
        self,
        worker_id: int = 0,
        first_iteration: int = 1,
        iteration_counter: Any = None,
        iteration_lock: Any = None,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.iteration_stride = 1
        self._record_event("worker_started", phase="worker", iteration=first_iteration, context={"worker_id": worker_id})
        state: WorkflowState = self._prepare_worker_loop(first_iteration)
        while True:
            state.update(self._node_generate(state))
            state.update(self._node_validate_and_score(state))
            if state.get("phase") == "finalize":
                break
            claimed = _claim_worker_iteration(iteration_counter, iteration_lock, max_iterations, self.deadline)
            if claimed is None:
                state = {"phase": "finalize", "iteration": int(state.get("iteration", first_iteration)), "stop_reason": "global iteration limit reached"}
                break
            state = {"phase": "generate", "iteration": claimed}
        return self.report()

    def _build_graph(self) -> Any:
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

    def _prepare_worker_loop(self, start_iteration: int) -> WorkflowState:
        state: WorkflowState = {"phase": "classify", "iteration": 0}
        state.update(self._node_classify(state))
        return {"phase": "generate", "iteration": max(1, int(start_iteration))}

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
        iteration = int(state.get("iteration", 0))
        self.objective_features = self._objective_feature_payload()
        objective_aggregate = self.objective_features.get("aggregate", {})
        memory_digest = self.memory.digest(
            features=objective_aggregate,
            top_k=self.memory_top_k,
            exploration=self.bandit_exploration,
        )
        memory_digest["solver_framework"] = self.framework_store.digest()
        if self.framework_store.is_empty():
            framework = self.llm.bootstrap_framework(
                objective_features=self.objective_features,
                memory_digest=memory_digest,
                case_samples=self._case_samples(),
            )
            applied = self.framework_store.bootstrap(framework, source="llm_bootstrap")
            self.framework_updates.append({"iteration": iteration, **applied})
            self._record_event("framework_bootstrapped", phase="classify", iteration=iteration, context=applied)

        self.framework_store.reload()
        framework_context = self.framework_store.prompt_context()
        interpretation = self.llm.interpret_instances(
            iteration=iteration,
            objective_features=self.objective_features,
            solver_framework_context=framework_context,
            memory_digest=memory_digest,
            case_samples=self._case_samples(),
        )
        aggregate = dict(objective_aggregate)
        aggregate["tags"] = list(interpretation.tags)
        aggregate["recommended_focus"] = list(interpretation.recommended_focus)
        aggregate["framework_confidence"] = interpretation.confidence
        self.instance_features = {
            "aggregate": aggregate,
            "cases": self.objective_features.get("cases", []),
            "objective_features": self.objective_features,
            "interpretation": interpretation.model_dump(mode="json"),
            "solver_framework": self.framework_store.snapshot(),
        }
        self.log("interpreted instances: focus=" + ",".join(aggregate.get("recommended_focus", [])))
        self._record_event(
            "instances_interpreted",
            phase="classify",
            iteration=iteration,
            context={"case_count": len(self.cases), "tags": aggregate.get("tags", []), "recommended_focus": aggregate.get("recommended_focus", [])},
        )
        return {"phase": "generate", "iteration": 1}

    def _generate_candidate(self, state: WorkflowState) -> WorkflowState:
        iteration = int(state.get("iteration", 1))
        if self._time_exhausted():
            return {"phase": "finalize", "iteration": iteration, "stop_reason": "budget exhausted before generation"}

        self.framework_store.reload()
        aggregate_features = self.instance_features.get("aggregate", {})
        framework_context = self.framework_store.prompt_context()
        solver_context = framework_context
        memory_digest = self.memory.digest(
            features=aggregate_features,
            top_k=self.memory_top_k,
            exploration=self.bandit_exploration,
        )
        memory_digest["solver_framework"] = self.framework_store.digest()
        best_summary = self._best_code_summary()
        candidate_arms = _unique_strings(
            list(aggregate_features.get("recommended_focus", []))
            + list(aggregate_features.get("tags", []))
            + self.framework_store.candidate_strategy_names()
        )
        toolbox = PlannerToolbox(
            instance_features=self.instance_features,
            solver_framework=self.framework_store.snapshot(),
            memory=self.memory,
            artifacts=self.artifacts,
            feature_query=aggregate_features,
            memory_top_k=self.memory_top_k,
            bandit_exploration=self.bandit_exploration,
            best_summary=best_summary,
            candidate_arms=candidate_arms,
        )
        plan = self.llm.plan(
            iteration=iteration,
            instance_features=self.instance_features,
            solver_framework_context=framework_context,
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

        plans = self._strategy_plan_batch(plan, aggregate_features, memory_digest, tool_context)
        generated = self._generate_candidates_from_plans(
            iteration=iteration,
            plans=plans,
            solver_context=solver_context,
            memory_digest=memory_digest,
            tool_context=tool_context,
            best_summary=best_summary,
        )
        candidates: List[Candidate] = []
        used_names = set(self.run_state.candidate_hashes)
        for index, variant_plan, candidate in generated:
            self._normalize_candidate_name(candidate, iteration, index, variant_plan, used_names)
            self.candidates.append(candidate)
            candidate_hash = self._remember_candidate_hash(candidate)
            artifact = self.artifacts.save_candidate(candidate)
            self._remember_candidate_artifact(candidate, artifact)
            candidate.rationale["_artifact_validation_path"] = artifact.validation_path
            self.memory.record_candidate(iteration, candidate.name, candidate.rationale, aggregate_features)
            self._record_event(
                "candidate_generated",
                phase="generate",
                iteration=iteration,
                candidate=candidate.name,
                candidate_hash=candidate_hash,
                context={
                    "plan": variant_plan.name,
                    "source": candidate.source,
                    "strategy_combination": list(variant_plan.strategy_combination),
                    "strategy_batch_size": len(plans),
                },
            )
            candidates.append(candidate)

        if len(candidates) == 1:
            self.log(f"iteration {iteration}: planned {plans[0].name} -> generated {candidates[0].name}")
        else:
            self.log(
                f"iteration {iteration}: generated {len(candidates)} strategy candidates "
                f"with {self._strategy_worker_count(len(candidates))} strategy workers"
            )
        return {
            "phase": "validate",
            "iteration": iteration,
            "candidate": candidates[0],
            "candidates": candidates,
            "plan": plans[0],
            "plans": plans,
        }

    def _generate_candidates_from_plans(
        self,
        *,
        iteration: int,
        plans: List[SolverPlan],
        solver_context: str,
        memory_digest: Dict[str, Any],
        tool_context: Dict[str, Any],
        best_summary: Dict[str, Any],
    ) -> List[tuple[int, SolverPlan, Candidate]]:
        disk_results = self.artifacts.disk_results()
        previous_impact = list(self.impact_analysis)
        case_samples = self._case_samples()

        def generate_one(item: tuple[int, SolverPlan]) -> tuple[int, SolverPlan, Candidate]:
            index, variant_plan = item
            try:
                candidate = self.llm.generate_from_plan(
                    iteration=iteration,
                    plan=variant_plan,
                    instance_features=self.instance_features,
                    solver_context=solver_context,
                    memory_digest=memory_digest,
                    disk_results=disk_results,
                    previous_impact=previous_impact,
                    case_samples=case_samples,
                    per_case_timeout=self.per_case_timeout,
                    tool_context=tool_context,
                )
            except Exception as exc:
                if self.max_repair_attempts <= 0:
                    raise
                candidate = self.repair_service.repair_schema_failure(
                    iteration=iteration,
                    plan=variant_plan,
                    error=str(exc),
                    raw_response=getattr(exc, "raw_response", ""),
                    memory_digest=memory_digest,
                    best_summary=best_summary,
                )
            return index, variant_plan, candidate

        indexed_plans = list(enumerate(plans, start=1))
        if len(indexed_plans) == 1:
            return [generate_one(indexed_plans[0])]
        with ThreadPoolExecutor(max_workers=self._strategy_worker_count(len(indexed_plans))) as executor:
            return list(executor.map(generate_one, indexed_plans))

    def _validate_and_score_candidate(self, state: WorkflowState) -> WorkflowState:
        iteration = int(state.get("iteration", 1))
        candidates = list(state.get("candidates") or ([state["candidate"]] if state.get("candidate") is not None else []))
        if not candidates:
            return {"phase": "finalize", "iteration": iteration, "stop_reason": "missing candidate"}

        validations = self._validate_candidates(candidates)
        final_items: List[CandidateEvaluationItem] = []
        for candidate, validation in zip(candidates, validations):
            if not validation.valid and self.max_repair_attempts > 0:
                repaired = self.repair_service.repair_validation_failure(iteration, state, candidate, validation)
                if repaired is not candidate:
                    candidate = repaired
                    validation = self.validator.validate(candidate.code, self.cases, self.parsed_cases)
            final_items.append({"candidate": candidate, "validation": validation, "score": None, "impact": None})

        valid_items = [item for item in final_items if item["validation"].valid]
        if valid_items:
            scores = self._score_candidate_batch(
                [item["candidate"] for item in valid_items],
                best=self.best_score,
            )
            for item, score in zip(valid_items, scores):
                item["score"] = score
                item["impact"] = self._impact(item["candidate"], score)

        valid_count = 0
        for item in final_items:
            candidate = item["candidate"]
            validation = item["validation"]
            artifact = self.candidate_artifacts.get(candidate.name)
            if artifact is None:
                artifact = self.artifacts.save_candidate(candidate)
                self._remember_candidate_artifact(candidate, artifact)
            self.artifacts.save_validation(artifact, validation)
            self.memory.record_validation(iteration, validation)
            if not validation.valid:
                error_item = {"iteration": iteration, "candidate": candidate.name, "errors": validation.errors}
                self.validation_errors.append(error_item)
                self._record_experiment(
                    iteration,
                    candidate,
                    validation=validation,
                    artifact=artifact,
                    failure_reason="validation failed",
                )
                self._record_event(
                    "validation_failed",
                    phase="validate_and_score",
                    iteration=iteration,
                    candidate=candidate.name,
                    candidate_hash=self.run_state.candidate_hashes.get(candidate.name),
                    context={"stage": validation.stage, "errors": validation.errors},
                )
                self.log(f"iteration {iteration}: validation failed for {candidate.name} at {validation.stage}")
                continue

            valid_count += 1
            score = item["score"]
            impact = item["impact"]
            if score is None or impact is None:
                raise RuntimeError(f"missing score for valid candidate {candidate.name}")
            self.scores.append(score)
            self.artifacts.save_score(artifact, score)
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

        self._reflect_framework(iteration, state, final_items)
        reason = "iteration limit reached" if valid_count else "validation failed"
        return self._next_or_finalize(iteration, reason)

    def _strategy_plan_batch(
        self,
        base_plan: SolverPlan,
        aggregate_features: Dict[str, Any],
        memory_digest: Dict[str, Any],
        tool_context: Dict[str, Any],
    ) -> List[SolverPlan]:
        if self.strategy_workers <= 1:
            return [base_plan]

        bandit = [
            str(item.get("arm"))
            for item in list(memory_digest.get("bandit_recommendations", []))
            + list(tool_context.get("bandit_recommendations", []))
            if item.get("arm")
        ]
        selected = list(tool_context.get("solver_framework", {}).get("framework", {}).get("strategies", []))
        selected_names = [str(item.get("name")) for item in selected if isinstance(item, dict) and item.get("name")]
        interpreted_focus = list(aggregate_features.get("recommended_focus", []))
        primary_names = _unique_strings(list(base_plan.strategy_combination) + bandit + interpreted_focus + selected_names)
        if not primary_names:
            return [base_plan]

        plans = [base_plan]
        for index, primary in enumerate(primary_names, start=1):
            if len(plans) >= self.strategy_workers:
                break
            combo = _unique_strings([primary] + list(base_plan.strategy_combination))[:5]
            if combo == list(base_plan.strategy_combination):
                continue
            directives = list(base_plan.generation_directives)
            directives.append(f"Treat {primary} as the primary strategy for this parallel candidate.")
            plans.append(
                base_plan.model_copy(
                    update={
                        "name": sanitize_name(
                            f"{base_plan.name}_{index:02d}_{primary}",
                            f"parallel_plan_{index:02d}",
                        ),
                        "strategy_combination": combo,
                        "exploration_mode": "parallel_strategy",
                        "reasoning": (
                            (base_plan.reasoning or "").strip()
                            + f" Parallel candidate with primary focus on {primary}."
                        ).strip(),
                        "generation_directives": directives,
                    }
                )
            )
        return plans

    def _validate_candidates(self, candidates: List[Candidate]) -> List[ValidationResult]:
        if len(candidates) <= 1:
            return [self.validator.validate(candidates[0].code, self.cases, self.parsed_cases)]
        workers = self._strategy_worker_count(len(candidates))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    lambda candidate: self.validator.validate(candidate.code, self.cases, self.parsed_cases),
                    candidates,
                )
            )

    def _score_candidate_batch(
        self,
        candidates: List[Candidate],
        best: Optional[ScoreResult],
    ) -> List[ScoreResult]:
        if len(candidates) <= 1:
            return [
                self.scorer.score(
                    candidate=candidates[0],
                    cases=self.cases,
                    parsed_cases=self.parsed_cases,
                    best=best,
                    timeout=self.search_per_case_timeout,
                )
            ]

        workers = self._strategy_worker_count(len(candidates))

        def score_one(candidate: Candidate) -> ScoreResult:
            scorer = Scorer(per_case_timeout=self.search_per_case_timeout)
            return scorer.score(
                candidate=candidate,
                cases=self.cases,
                parsed_cases=self.parsed_cases,
                best=best,
                timeout=self.search_per_case_timeout,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(score_one, candidates))

    def _strategy_worker_count(self, candidate_count: int) -> int:
        return max(1, min(self.strategy_workers, max(1, candidate_count)))

    def _normalize_candidate_name(
        self,
        candidate: Candidate,
        iteration: int,
        index: int,
        plan: SolverPlan,
        used_names: set[str],
    ) -> None:
        primary = plan.strategy_combination[0] if plan.strategy_combination else f"strategy_{index}"
        default_name = f"llm_iter_{iteration:03d}_{index:02d}"
        base = sanitize_name(candidate.name, default_name)
        name = base
        if name in used_names:
            name = sanitize_name(f"{base}_{primary}", default_name)
        counter = 2
        while name in used_names:
            name = sanitize_name(f"{base}_{primary}_{counter}", default_name)
            counter += 1
        candidate.name = name
        candidate.rationale["name"] = name
        candidate.rationale["parallel_strategy_plan"] = plan.name
        candidate.rationale["primary_strategy"] = primary
        used_names.add(name)

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
        next_iteration = iteration + max(1, self.iteration_stride)
        if iteration >= self.iterations or next_iteration > self.iterations:
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
            "timings": self.run_state.timings.timings,
            "candidate_hashes": self.run_state.candidate_hashes,
            "cases": [case.name for case in self.cases],
            "iterations_requested": self.iterations,
            "iterations_completed": len({candidate.iteration for candidate in self.candidates}),
            "strategy_workers": self.strategy_workers,
            "best": serialize(self.best_score) if self.best_score else None,
            "instance_features": self.instance_features,
            "solver_framework": self.framework_store.snapshot(),
            "framework_updates": self.framework_updates,
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

    def _remember_candidate_artifact(self, candidate: Candidate, artifact: IterationArtifact) -> None:
        self.candidate_artifacts[candidate.name] = artifact

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
                    solver_context=self.framework_store.prompt_context(),
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
        plan = None
        plan_payload = (candidate.rationale or {}).get("plan")
        if isinstance(plan_payload, dict):
            try:
                plan = SolverPlan.model_validate(plan_payload)
            except Exception:
                plan = None
        if plan is None:
            plan = state.get("plan")
        if plan is None:
            raise RuntimeError("validation repair requires the original SolverPlan; no generated plan is available")
        aggregate_features = self.instance_features.get("aggregate", {})
        memory_digest = self.memory.digest(
            features=aggregate_features,
            top_k=self.memory_top_k,
            exploration=self.bandit_exploration,
        )
        memory_digest["solver_framework"] = self.framework_store.digest()
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
                    solver_context=self.framework_store.prompt_context(),
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
            self._normalize_candidate_name(
                repaired,
                iteration,
                attempt,
                plan,
                set(self.run_state.candidate_hashes),
            )
            self.candidates.append(repaired)
            repaired_hash = self._remember_candidate_hash(repaired)
            artifact = self.artifacts.save_candidate(repaired)
            self._remember_candidate_artifact(repaired, artifact)
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

    def _reflect_framework(
        self,
        iteration: int,
        state: WorkflowState,
        final_items: List[CandidateEvaluationItem],
    ) -> None:
        try:
            plans = [model_dump(plan) for plan in list(state.get("plans") or ([state["plan"]] if state.get("plan") is not None else []))]
            evaluations = [self._evaluation_summary(item) for item in final_items]
            update = self.llm.reflect_framework(
                iteration=iteration,
                solver_framework_context=self.framework_store.prompt_context(),
                instance_features=self.instance_features,
                plans=plans,
                evaluations=evaluations,
                experiments=self.experiment_records,
                previous_impact=self.impact_analysis,
            )
            applied = self.framework_store.apply_update(update, source="llm_reflection", iteration=iteration)
            self.framework_updates.append({"iteration": iteration, **applied})
            self.instance_features["solver_framework"] = self.framework_store.snapshot()
            self._record_event("framework_updated", phase="validate_and_score", iteration=iteration, context=applied)
        except Exception as exc:
            failure = {"iteration": iteration, "action": "framework_update_rejected", "error": str(exc)}
            self.framework_updates.append(failure)
            self._record_event("framework_update_rejected", phase="validate_and_score", iteration=iteration, context=failure)

    def _evaluation_summary(self, item: CandidateEvaluationItem) -> Dict[str, Any]:
        candidate = item["candidate"]
        return {
            "candidate": candidate.name,
            "rationale": candidate.rationale,
            "validation": serialize(item["validation"]),
            "score": serialize(item["score"]) if item.get("score") is not None else None,
            "impact": item.get("impact"),
        }

    def _objective_feature_payload(self) -> Dict[str, Any]:
        per_case = []
        features = []
        for case, parsed in zip(self.cases, self.parsed_cases):
            item = dataset_features(parsed)
            item["name"] = case.name
            features.append({key: value for key, value in item.items() if key != "name"})
            per_case.append(item)
        return {
            "aggregate": aggregate_features(features),
            "cases": per_case,
            "interpretation_owner": "llm_framework",
        }

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
            "framework": self.framework_store.digest(),
        }


def _final_rank(score: ScoreResult) -> tuple:
    return (score.failures, -score.total_covered, score.total_penalty)


def _unique_strings(values: List[Any]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _claim_worker_iteration(
    iteration_counter: Any,
    iteration_lock: Any,
    max_iterations: Optional[int],
    deadline: float,
) -> Optional[int]:
    if iteration_counter is None or iteration_lock is None or max_iterations is None:
        return None
    if time.time() >= deadline:
        return None
    with iteration_lock:
        if iteration_counter.value >= max_iterations:
            return None
        iteration_counter.value += 1
        return int(iteration_counter.value)
