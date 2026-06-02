"""Top-level modular AutoSolver LangChain Agent."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from autosolver_agent.artifacts import write_json
from autosolver_agent.caseio import discover_case_paths, load_cases, parse_case
from autosolver_agent.llm import LLMCodeGenerator
from autosolver_agent.memory import MemoryStore
from autosolver_agent.workflow import AutoSolverWorkflow


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
        summary_output_path: Optional[str] = None,
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
        self.summary_output_path = summary_output_path

    def run(self) -> Dict[str, Any]:
        paths = self.case_paths or discover_case_paths(os.getcwd())
        cases = load_cases(paths, self.max_cases)
        if not cases:
            raise RuntimeError("No valid case files found.")
        parsed_cases = [parse_case(case.text) for case in cases]
        memory = MemoryStore(self.memory_dir)
        from autosolver_agent.artifacts import ArtifactStore

        artifacts = ArtifactStore(self.artifact_dir)
        llm = LLMCodeGenerator(model=self.llm_model, base_url=self.llm_base_url, llm=self.llm)
        workflow = AutoSolverWorkflow(
            cases=cases,
            parsed_cases=parsed_cases,
            iterations=self.iterations,
            deadline=time.time() + self.budget_seconds,
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
            summary_output_path=self.summary_output_path,
        )
        report = workflow.run()
        report_path = self.output_path + ".report.json"
        write_json(report_path, report)
        return report

    def run_json(self) -> str:
        return json.dumps(self.run(), indent=2, ensure_ascii=False, sort_keys=True)
