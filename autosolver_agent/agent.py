"""Top-level modular AutoSolver LangChain Agent."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from autosolver_agent.artifacts import ArtifactStore, write_json
from autosolver_agent.caseio import discover_case_paths, load_cases, parse_case
from autosolver_agent.llm import LLMCodeGenerator
from autosolver_agent.memory import MemoryStore
from autosolver_agent.workflow import AutoSolverWorkflow
from autosolver_agent.workflow.parallel import ParallelAutoSolverRunner, ParallelRunConfig


class AutoSolverLangChainAgent:
    def __init__(
        self,
        case_paths: Optional[List[str]] = None,
        output_path: str = "generated_submit_solution.py",
        budget_seconds: float = 90.0,
        per_case_timeout: float = 10.0,
        search_per_case_timeout: Optional[float] = None,
        iterations: int = 3,
        memory_dir: str = "runs/autosolver_memory",
        artifact_dir: str = "runs/autosolver_artifacts",
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm: Any = None,
        max_cases: int = 3,
        verbose: bool = True,
        finalize_top_k: int = 3,
        max_repair_attempts: int = 2,
        memory_top_k: int = 5,
        bandit_exploration: float = 1.4,
        strategy_workers: int = 5,
        summary_output_path: Optional[str] = None,
        event_log_path: Optional[str] = None,
    ) -> None:
        self.case_paths = case_paths or []
        self.output_path = output_path
        self.budget_seconds = budget_seconds
        self.per_case_timeout = per_case_timeout
        self.search_per_case_timeout = search_per_case_timeout or per_case_timeout
        self.iterations = iterations
        self.memory_dir = memory_dir
        self.artifact_dir = artifact_dir
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm = llm
        self.max_cases = max_cases
        self.verbose = verbose
        self.finalize_top_k = finalize_top_k
        self.max_repair_attempts = max_repair_attempts
        self.memory_top_k = memory_top_k
        self.bandit_exploration = bandit_exploration
        self.strategy_workers = int(strategy_workers)
        self.summary_output_path = summary_output_path
        self.event_log_path = event_log_path

    def run(self) -> Dict[str, Any]:
        if self.strategy_workers < 1:
            raise RuntimeError("AutoSolver Agent requires strategy_workers >= 1.")
        paths = self.case_paths or discover_case_paths(os.getcwd())
        cases = load_cases(paths, self.max_cases)
        if not cases:
            raise RuntimeError("No valid case files found.")
        parsed_cases = [parse_case(case.text, case_name=case.name, path=case.path) for case in cases]

        deadline = time.time() + self.budget_seconds
        if self.llm is None and self.strategy_workers >= 2:
            LLMCodeGenerator.validate_environment()
            runner = ParallelAutoSolverRunner(
                cases=cases,
                parsed_cases=parsed_cases,
                config=ParallelRunConfig(
                    iterations=self.iterations,
                    deadline=deadline,
                    per_case_timeout=self.per_case_timeout,
                    search_per_case_timeout=self.search_per_case_timeout,
                    output_path=self.output_path,
                    memory_dir=self.memory_dir,
                    artifact_dir=self.artifact_dir,
                    llm_model=self.llm_model,
                    llm_base_url=self.llm_base_url,
                    verbose=self.verbose,
                    finalize_top_k=self.finalize_top_k,
                    max_repair_attempts=self.max_repair_attempts,
                    memory_top_k=self.memory_top_k,
                    bandit_exploration=self.bandit_exploration,
                    strategy_workers=self.strategy_workers,
                    summary_output_path=self.summary_output_path,
                    event_log_path=self.event_log_path,
                ),
            )
            report = runner.run()
            report_path = self.output_path + ".report.json"
            write_json(report_path, report)
            return report

        if self.llm is None:
            LLMCodeGenerator.validate_environment()
        return self._run_single_workflow(cases, parsed_cases, deadline)

    def _run_single_workflow(
        self,
        cases: List[Any],
        parsed_cases: List[Any],
        deadline: float,
    ) -> Dict[str, Any]:
        memory = MemoryStore(self.memory_dir)
        artifacts = ArtifactStore(self.artifact_dir)
        llm = LLMCodeGenerator(model=self.llm_model, base_url=self.llm_base_url, llm=self.llm)
        workflow = AutoSolverWorkflow(
            cases=cases,
            parsed_cases=parsed_cases,
            iterations=self.iterations,
            deadline=deadline,
            per_case_timeout=self.per_case_timeout,
            search_per_case_timeout=self.search_per_case_timeout,
            output_path=self.output_path,
            memory=memory,
            artifacts=artifacts,
            llm=llm,
            verbose=self.verbose,
            finalize_top_k=self.finalize_top_k,
            max_repair_attempts=self.max_repair_attempts,
            memory_top_k=self.memory_top_k,
            bandit_exploration=self.bandit_exploration,
            strategy_workers=self.strategy_workers,
            summary_output_path=self.summary_output_path,
            event_log_path=self.event_log_path,
        )
        report = workflow.run()
        report_path = self.output_path + ".report.json"
        write_json(report_path, report)
        return report

    def run_json(self) -> str:
        return json.dumps(self.run(), indent=2, ensure_ascii=False, sort_keys=True)
