"""Scoring tool."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from autosolver_agent.caseio import score_answer
from autosolver_agent.models import Candidate, Case, ParsedCase, ScoreResult
from autosolver_agent.runtime import run_candidate


class Scorer:
    def __init__(self, per_case_timeout: float) -> None:
        self.per_case_timeout = per_case_timeout

    def score(
        self,
        candidate: Candidate,
        cases: List[Case],
        parsed_cases: List[ParsedCase],
        best: Optional[ScoreResult] = None,
        timeout: Optional[float] = None,
    ) -> ScoreResult:
        per_case = timeout if timeout is not None else self.per_case_timeout
        case_results = []
        total_covered = 0
        total_tasks = 0
        total_penalty = 0.0
        total_runtime = 0.0
        failures = 0
        for case, parsed in zip(cases, parsed_cases):
            run = run_candidate(candidate.code, case.text, per_case)
            total_tasks += len(parsed.all_tasks)
            if run["status"] != "ok":
                failures += 1
                penalty = 1_000_000.0 + 100.0 * len(parsed.all_tasks)
                case_results.append(
                    {
                        "case": case.name,
                        "status": run["status"],
                        "covered": 0,
                        "tasks": len(parsed.all_tasks),
                        "penalty": penalty,
                        "runtime": run.get("runtime", 0.0),
                        "error": run.get("error"),
                    }
                )
                total_penalty += penalty
                total_runtime += run.get("runtime", 0.0)
                continue
            scored = score_answer(parsed, run["answer"])
            if not scored["valid"]:
                failures += 1
            total_covered += scored["covered"]
            total_penalty += scored["penalty"]
            total_runtime += run["runtime"]
            case_results.append(
                {
                    "case": case.name,
                    "status": "ok" if scored["valid"] else "invalid",
                    "covered": scored["covered"],
                    "tasks": len(parsed.all_tasks),
                    "penalty": scored["penalty"],
                    "runtime": run["runtime"],
                    "error": scored.get("error"),
                }
            )
        rank = (failures, -total_covered, total_penalty, total_runtime)
        convergence = self._convergence(rank, best)
        return ScoreResult(
            name=candidate.name,
            rank=rank,
            total_covered=total_covered,
            total_tasks=total_tasks,
            total_penalty=total_penalty,
            total_runtime=total_runtime,
            failures=failures,
            cases=case_results,
            convergence=convergence,
        )

    def _convergence(self, rank: tuple, best: Optional[ScoreResult]) -> Dict[str, Any]:
        if best is None:
            return {"is_improved": True, "delta_penalty": None, "delta_covered": None}
        return {
            "is_improved": rank < best.rank,
            "delta_penalty": round(rank[2] - best.total_penalty, 6),
            "delta_covered": -rank[1] - best.total_covered,
            "previous_best": best.name,
        }
