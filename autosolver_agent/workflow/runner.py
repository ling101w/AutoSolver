"""Worker orchestration for AutoSolver workflows."""

from __future__ import annotations

import multiprocessing
import os
import queue as queue_module
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from autosolver_agent.artifacts import ArtifactStore, write_json
from autosolver_agent.framework import FrameworkStore
from autosolver_agent.llm import LLMCodeGenerator
from autosolver_agent.memory import MemoryStore
from autosolver_agent.models import Candidate, Case, ParsedCase, ScoreResult
from autosolver_agent.tools import Scorer
from autosolver_agent.workflow.graph import AutoSolverWorkflow, _final_rank, _format_progress_line


@dataclass
class AutoSolverRunConfig:
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
    baseline_solver_paths: List[str] = field(default_factory=list)
    llm: Any = None


class AutoSolverRunner:
    def __init__(
        self,
        cases: List[Case],
        parsed_cases: List[ParsedCase],
        config: AutoSolverRunConfig,
    ) -> None:
        self.cases = cases
        self.parsed_cases = parsed_cases
        self.config = config

    def run(self) -> Dict[str, Any]:
        if self.config.strategy_workers < 1:
            raise RuntimeError("AutoSolverRunner requires strategy_workers >= 1.")
        if self.config.iterations < 1:
            raise RuntimeError("AutoSolverRunner requires iterations >= 1.")
        if self.config.llm is None:
            LLMCodeGenerator.validate_environment()
        if self._runs_in_current_process():
            return self._run_current_process_workflow()

        return self._run_worker_processes()

    def _runs_in_current_process(self) -> bool:
        return self.config.strategy_workers == 1 or self.config.llm is not None

    def _run_current_process_workflow(self) -> Dict[str, Any]:
        memory = MemoryStore(self.config.memory_dir)
        artifacts = ArtifactStore(self.config.artifact_dir)
        llm = LLMCodeGenerator(model=self.config.llm_model, base_url=self.config.llm_base_url, llm=self.config.llm)
        workflow = AutoSolverWorkflow(
            cases=self.cases,
            parsed_cases=self.parsed_cases,
            iterations=self.config.iterations,
            deadline=self.config.deadline,
            per_case_timeout=self.config.per_case_timeout,
            search_per_case_timeout=self.config.search_per_case_timeout,
            output_path=self.config.output_path,
            memory=memory,
            artifacts=artifacts,
            llm=llm,
            verbose=self.config.verbose,
            finalize_top_k=self.config.finalize_top_k,
            max_repair_attempts=self.config.max_repair_attempts,
            memory_top_k=self.config.memory_top_k,
            bandit_exploration=self.config.bandit_exploration,
            baseline_solver_paths=self.config.baseline_solver_paths,
            strategy_workers=self.config.strategy_workers,
            summary_output_path=self.config.summary_output_path,
            event_log_path=self.config.event_log_path,
        )
        return workflow.run()

    def _run_worker_processes(self) -> Dict[str, Any]:
        worker_count = self.config.strategy_workers
        iteration_counter = multiprocessing.Value("i", 0)
        iteration_lock = multiprocessing.Lock()
        queue: multiprocessing.Queue = multiprocessing.Queue()
        processes = []
        if self.config.verbose:
            print(f"[agent] starting {worker_count} worker processes", flush=True)
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
        pending = {worker_id: process for worker_id, process in enumerate(processes)}
        dead_since: Dict[int, float] = {}
        while pending:
            remaining = self.config.deadline - time.time()
            if remaining <= 0:
                for worker_id, process in list(pending.items()):
                    _terminate_process(process)
                    errors.append(
                        {
                            "worker_id": worker_id,
                            "error": "worker exceeded run deadline without reporting",
                            "exitcode": process.exitcode,
                        }
                    )
                    pending.pop(worker_id, None)
                break

            try:
                kind, payload = queue.get(timeout=min(0.25, max(0.01, remaining)))
                _handle_worker_message(kind, payload, worker_reports, errors, pending, self.config.verbose)
                if kind != "progress":
                    dead_since.pop(int(payload.get("worker_id", -1)) if isinstance(payload, dict) else -1, None)
            except queue_module.Empty:
                pass

            _drain_worker_queue(queue, worker_reports, errors, pending, dead_since, self.config.verbose)
            now = time.time()
            for worker_id, process in list(pending.items()):
                if process.is_alive() or process.exitcode is None:
                    continue
                first_seen = dead_since.setdefault(worker_id, now)
                if now - first_seen < 0.25:
                    continue
                errors.append(
                    {
                        "worker_id": worker_id,
                        "error": f"worker exited without reporting (exitcode={process.exitcode})",
                        "exitcode": process.exitcode,
                    }
                )
                pending.pop(worker_id, None)

        for worker_id, process in enumerate(processes):
            process.join(0.5)
            if process.is_alive():
                _terminate_process(process)
                errors.append(
                    {
                        "worker_id": worker_id,
                        "error": "worker did not exit after reporting",
                        "exitcode": process.exitcode,
                    }
                )

        if errors:
            first = errors[0]
            raise RuntimeError(f"worker {first.get('worker_id')} failed: {first.get('error')}")

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
        if self.config.verbose:
            print(f"[agent] finalizing global best from {len(candidates)} worker candidates", flush=True)
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
                source="worker_finalize",
            )
            score = scorer.score(
                candidate=candidate,
                cases=self.cases,
                parsed_cases=self.parsed_cases,
                best=best_pair[0] if best_pair else None,
                timeout=self.config.per_case_timeout,
            )
            if self.config.verbose:
                print(
                    f"[agent] finalize recheck worker={item.get('worker_id')} candidate={candidate.name} "
                    f"covered={score.total_covered}/{score.total_tasks} penalty={score.total_penalty:.4f}",
                    flush=True,
                )
            if best_pair is None or _final_rank(score) < _final_rank(best_pair[0]):
                best_pair = (score, code, item)

        if best_pair is None:
            raise RuntimeError("no scored candidate survived worker finalization")

        output_path = self.config.output_path if os.path.isabs(self.config.output_path) else os.path.abspath(self.config.output_path)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(best_pair[1])

        memory = MemoryStore(self.config.memory_dir)
        memory.save(short_term_path=os.path.join(memory.memory_dir, "short_term_last_run.json"))
        framework = FrameworkStore(self.config.memory_dir)

        report = {
            "run_mode": "worker_processes",
            "output_path": output_path,
            "worker_count": worker_count,
            "strategy_workers_requested": self.config.strategy_workers,
            "iterations_requested": self.config.iterations,
            "iterations_completed": sum(int(report.get("iterations_completed", 0)) for report in worker_reports),
            "best": _score_payload(best_pair[0]),
            "best_worker_candidate": best_pair[2],
            "worker_reports": worker_reports,
            "worker_errors": errors,
            "baseline_solver_paths": list(self.config.baseline_solver_paths),
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


def _record_worker_result(
    kind: Any,
    payload: Any,
    worker_reports: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
    pending: Dict[int, multiprocessing.Process],
) -> None:
    if not isinstance(payload, dict):
        errors.append({"worker_id": None, "error": f"worker returned non-dict payload: {payload!r}"})
        return
    worker_id = payload.get("worker_id")
    if isinstance(worker_id, int):
        pending.pop(worker_id, None)
    if kind == "ok":
        worker_reports.append(payload)
    else:
        errors.append(payload)


def _handle_worker_message(
    kind: Any,
    payload: Any,
    worker_reports: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
    pending: Dict[int, multiprocessing.Process],
    verbose: bool,
) -> None:
    if kind == "progress":
        if verbose and isinstance(payload, dict):
            print(_format_progress_line(payload), flush=True)
        return
    _record_worker_result(kind, payload, worker_reports, errors, pending)


def _drain_worker_queue(
    result_queue: multiprocessing.Queue,
    worker_reports: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
    pending: Dict[int, multiprocessing.Process],
    dead_since: Dict[int, float],
    verbose: bool,
) -> None:
    while True:
        try:
            kind, payload = result_queue.get_nowait()
        except queue_module.Empty:
            return
        _handle_worker_message(kind, payload, worker_reports, errors, pending, verbose)
        if kind != "progress" and isinstance(payload, dict) and isinstance(payload.get("worker_id"), int):
            dead_since.pop(payload["worker_id"], None)


def _terminate_process(process: multiprocessing.Process) -> None:
    if not process.is_alive():
        return
    process.terminate()
    process.join(0.5)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(0.5)


def _worker_entry(
    worker_id: int,
    iteration_counter: Any,
    iteration_lock: Any,
    cases: List[Case],
    parsed_cases: List[ParsedCase],
    config: AutoSolverRunConfig,
    queue: multiprocessing.Queue,
) -> None:
    try:
        progress_callback = _worker_progress_callback(queue, worker_id) if config.verbose else None
        first_iteration = _claim_iteration(iteration_counter, iteration_lock, config.iterations, config.deadline)
        if first_iteration is None:
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "completed",
                        "phase": "worker",
                        "iteration": None,
                        "message": "no iteration available",
                        "summary": {"iterations_completed": 0, "candidates_generated": 0, "valid_scores": 0},
                        "time_left": round(max(0.0, config.deadline - time.time()), 1),
                    }
                )
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
            baseline_solver_paths=config.baseline_solver_paths,
            strategy_workers=1,
            summary_output_path=None,
            event_log_path=event_log_path,
            progress_callback=progress_callback,
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


def _worker_progress_callback(result_queue: multiprocessing.Queue, worker_id: int) -> Any:
    def emit(payload: Dict[str, Any]) -> None:
        item = dict(payload)
        item["worker_id"] = worker_id
        result_queue.put(("progress", item))

    return emit


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
