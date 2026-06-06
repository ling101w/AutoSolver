"""Multi-process worker orchestration for AutoSolver workflows."""

from __future__ import annotations

import multiprocessing
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from autosolver_agent.artifacts import ArtifactStore, write_json
from autosolver_agent.framework import FrameworkStore
from autosolver_agent.llm import LLMCodeGenerator
from autosolver_agent.memory import MemoryStore
from autosolver_agent.models import Candidate, Case, ParsedCase, ScoreResult
from autosolver_agent.tools import Scorer
from autosolver_agent.workflow.graph import AutoSolverWorkflow, _final_rank


@dataclass
class ParallelRunConfig:
    iterations: int
    deadline: float
    per_case_timeout: float
    search_per_case_timeout: float
    output_path: str
    memory_dir: str
    artifact_dir: str
    llm_model: Optional[str]
    llm_base_url: Optional[str]
    verbose: bool
    finalize_top_k: int
    max_repair_attempts: int
    memory_top_k: int
    bandit_exploration: float
    strategy_workers: int
    summary_output_path: Optional[str]
    event_log_path: Optional[str]


class ParallelAutoSolverRunner:
    def __init__(
        self,
        cases: List[Case],
        parsed_cases: List[ParsedCase],
        config: ParallelRunConfig,
    ) -> None:
        self.cases = cases
        self.parsed_cases = parsed_cases
        self.config = config

    def run(self) -> Dict[str, Any]:
        if self.config.strategy_workers < 2:
            raise RuntimeError("ParallelAutoSolverRunner requires strategy_workers >= 2.")
        if self.config.iterations < 1:
            raise RuntimeError("ParallelAutoSolverRunner requires iterations >= 1.")
        worker_count = self.config.strategy_workers
        iteration_counter = multiprocessing.Value("i", 0)
        iteration_lock = multiprocessing.Lock()
        queue: multiprocessing.Queue = multiprocessing.Queue()
        processes = []
        for worker_id in range(worker_count):
            process = multiprocessing.Process(
                target=_worker_entry,
                args=(
                    worker_id,
                    iteration_counter,
                    iteration_lock,
                    self.cases,
                    self.parsed_cases,
                    self.config,
                    queue,
                ),
            )
            process.start()
            processes.append(process)

        worker_reports: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for _ in processes:
            kind, payload = queue.get()
            if kind == "ok":
                worker_reports.append(payload)
            else:
                errors.append(payload)

        for process in processes:
            process.join()

        if errors:
            first = errors[0]
            raise RuntimeError(f"parallel worker {first.get('worker_id')} failed: {first.get('error')}")

        return self._finalize(worker_count, worker_reports, errors)

    def _finalize(
        self,
        worker_count: int,
        worker_reports: List[Dict[str, Any]],
        errors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        candidates = _candidate_refs(worker_reports, self.config.finalize_top_k)
        best_pair: Optional[tuple[ScoreResult, str, Dict[str, Any]]] = None
        scorer = Scorer(per_case_timeout=self.config.per_case_timeout)
        for item in candidates:
            code_path = item.get("code_path")
            if not code_path:
                raise RuntimeError(f"worker candidate is missing code_path: {item}")
            with open(code_path, "r", encoding="utf-8") as handle:
                code = handle.read()
            candidate = Candidate(
                name=str(item.get("candidate") or item.get("name") or os.path.basename(code_path)),
                code=code,
                rationale=dict(item.get("rationale") or {}),
                iteration=int(item.get("iteration") or 0),
                source="parallel_finalize",
            )
            score = scorer.score(
                candidate=candidate,
                cases=self.cases,
                parsed_cases=self.parsed_cases,
                best=best_pair[0] if best_pair else None,
                timeout=self.config.per_case_timeout,
            )
            if best_pair is None or _final_rank(score) < _final_rank(best_pair[0]):
                best_pair = (score, code, item)

        if best_pair is None:
            raise RuntimeError("no scored candidate survived parallel worker finalization")

        output_path = self.config.output_path if os.path.isabs(self.config.output_path) else os.path.abspath(self.config.output_path)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(best_pair[1])

        memory = MemoryStore(self.config.memory_dir)
        memory.save(short_term_path=os.path.join(memory.memory_dir, "short_term_last_run.json"))
        framework = FrameworkStore(self.config.memory_dir)

        report = {
            "run_mode": "parallel_workers",
            "output_path": output_path,
            "parallel_workers": worker_count,
            "strategy_workers_requested": self.config.strategy_workers,
            "iterations_requested": self.config.iterations,
            "iterations_completed": sum(int(report.get("iterations_completed", 0)) for report in worker_reports),
            "best": _score_payload(best_pair[0]),
            "best_worker_candidate": best_pair[2],
            "worker_reports": worker_reports,
            "worker_errors": errors,
            "long_term_memory_digest": memory.digest(),
            "solver_framework": framework.snapshot(),
            "summary": {
                "best_candidate": best_pair[0].name,
                "best_rank": list(best_pair[0].rank),
                "best_penalty": round(best_pair[0].total_penalty, 6),
                "best_covered": best_pair[0].total_covered,
                "best_tasks": best_pair[0].total_tasks,
                "workers": worker_count,
            },
        }
        if self.config.summary_output_path:
            write_json(self.config.summary_output_path, report["summary"])
        return report


def _worker_entry(
    worker_id: int,
    iteration_counter: Any,
    iteration_lock: Any,
    cases: List[Case],
    parsed_cases: List[ParsedCase],
    config: ParallelRunConfig,
    queue: multiprocessing.Queue,
) -> None:
    try:
        first_iteration = _claim_iteration(iteration_counter, iteration_lock, config.iterations, config.deadline)
        if first_iteration is None:
            queue.put(
                (
                    "ok",
                    {
                        "worker_id": worker_id,
                        "iterations_completed": 0,
                        "artifacts": [],
                        "experiments": [],
                        "summary": {"candidates_generated": 0, "valid_scores": 0},
                    },
                )
            )
            return
        artifact_dir = os.path.join(config.artifact_dir, f"worker_{worker_id:02d}")
        event_log_path = (
            os.path.join(artifact_dir, "events.jsonl")
            if config.event_log_path is None
            else f"{os.path.splitext(config.event_log_path)[0]}.worker_{worker_id:02d}.jsonl"
        )
        memory = MemoryStore(config.memory_dir)
        artifacts = ArtifactStore(artifact_dir)
        llm = LLMCodeGenerator(model=config.llm_model, base_url=config.llm_base_url)
        workflow = AutoSolverWorkflow(
            cases=cases,
            parsed_cases=parsed_cases,
            iterations=config.iterations,
            deadline=config.deadline,
            per_case_timeout=config.per_case_timeout,
            search_per_case_timeout=config.search_per_case_timeout,
            output_path=os.path.join(artifact_dir, "worker_best_solver.py"),
            memory=memory,
            artifacts=artifacts,
            llm=llm,
            verbose=config.verbose,
            finalize_top_k=config.finalize_top_k,
            max_repair_attempts=config.max_repair_attempts,
            memory_top_k=config.memory_top_k,
            bandit_exploration=config.bandit_exploration,
            strategy_workers=1,
            summary_output_path=None,
            event_log_path=event_log_path,
        )
        report = workflow.run_worker_loop(
            worker_id=worker_id,
            first_iteration=first_iteration,
            iteration_counter=iteration_counter,
            iteration_lock=iteration_lock,
            max_iterations=config.iterations,
        )
        report["worker_id"] = worker_id
        queue.put(("ok", report))
    except Exception as exc:
        queue.put(
            (
                "error",
                {
                    "worker_id": worker_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        )


def _claim_iteration(
    iteration_counter: Any,
    iteration_lock: Any,
    max_iterations: int,
    deadline: float,
) -> Optional[int]:
    if time.time() >= deadline:
        return None
    with iteration_lock:
        if iteration_counter.value >= max_iterations:
            return None
        iteration_counter.value += 1
        return int(iteration_counter.value)


def _candidate_refs(worker_reports: List[Dict[str, Any]], limit_per_worker: int) -> List[Dict[str, Any]]:
    refs = []
    for report in worker_reports:
        worker_id = report.get("worker_id")
        artifacts = {item.get("candidate_name"): item for item in report.get("artifacts", [])}
        experiments = sorted(
            report.get("experiments", []),
            key=lambda item: _experiment_rank(item),
        )
        for experiment in experiments[: max(1, limit_per_worker)]:
            artifact = artifacts.get(experiment.get("candidate")) or {}
            score = experiment.get("score") or {}
            refs.append(
                {
                    "worker_id": worker_id,
                    "candidate": experiment.get("candidate"),
                    "iteration": experiment.get("iteration"),
                    "strategy": experiment.get("strategy", []),
                    "params": experiment.get("params", {}),
                    "score": score,
                    "code_path": artifact.get("code_path"),
                    "rationale": {"strategy_combination": experiment.get("strategy", [])},
                }
            )
    refs.sort(key=lambda item: _score_rank(item.get("score") or {}))
    return refs


def _experiment_rank(item: Dict[str, Any]) -> tuple:
    score = item.get("score")
    if not isinstance(score, dict):
        return (999, 0, 1e18, 1e18)
    return _score_rank(score)


def _score_rank(score: Dict[str, Any]) -> tuple:
    rank = score.get("rank")
    if isinstance(rank, list) and len(rank) >= 4:
        return tuple(rank[:4])
    return (
        int(score.get("failures", 999)),
        -int(score.get("covered", 0)),
        float(score.get("penalty", 1e18)),
        float(score.get("runtime", 1e18)),
    )


def _score_payload(score: ScoreResult) -> Dict[str, Any]:
    return {
        "name": score.name,
        "rank": list(score.rank),
        "total_covered": score.total_covered,
        "total_tasks": score.total_tasks,
        "total_penalty": score.total_penalty,
        "total_runtime": score.total_runtime,
        "failures": score.failures,
        "cases": score.cases,
        "convergence": score.convergence,
    }
