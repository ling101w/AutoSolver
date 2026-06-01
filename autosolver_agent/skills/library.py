"""Strategy and solver skill knowledge base."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from autosolver_agent.models import SolverSkill, StrategySpec


class StrategyLibrary:
    """Describes available search strategies and where they tend to work."""

    def __init__(self) -> None:
        self._strategies = [
            StrategySpec(
                name="expected_greedy",
                description="Rank task-courier rows by expected penalty and assign disjoint task groups greedily.",
                implementation_notes=(
                    "Parse TSV rows, compute expected_one = willingness * score + "
                    "(1 - willingness) * 100 * task_count, then choose low expected rows while "
                    "preventing duplicate tasks and couriers."
                ),
                suitable_features=["general", "medium_willingness", "many_couriers"],
                example_signals={"pair_ratio_max": 0.25, "capacity_ratio_min": 1.0},
                risks=["Can miss useful bundled orders when pair rows are important."],
                recommended_parameters={"coverage_weight": 8.0, "willingness_weight": 8.0},
            ),
            StrategySpec(
                name="bundle_first",
                description="Prefer multi-task rows early when合单 rows are dense and courier capacity is tight.",
                implementation_notes=(
                    "Bias ranking toward len(tasks) > 1 rows, then repair uncovered single tasks and add extra "
                    "couriers only when they reduce expected penalty."
                ),
                suitable_features=["high_pair_ratio", "scarce_couriers"],
                example_signals={"pair_ratio_min": 0.2, "capacity_ratio_max": 1.6},
                risks=["Overusing bundles may block better single-task assignments."],
                recommended_parameters={"pair_weight": 45.0, "coverage_weight": 12.0},
            ),
            StrategySpec(
                name="willingness_weighted",
                description="Bias toward high acceptance probability, then add secondary couriers to reduce reject risk.",
                implementation_notes=(
                    "Use willingness as a large negative ranking term. After constructing a valid base solution, "
                    "iteratively add unused couriers to existing task groups if penalty decreases."
                ),
                suitable_features=["low_willingness", "high_reject_risk"],
                example_signals={"avg_willingness_max": 0.35},
                risks=["May pay higher score when score values vary widely."],
                recommended_parameters={"willingness_weight": 35.0, "extra_limit": 120},
            ),
            StrategySpec(
                name="flow_single_initial",
                description="Use min-cost one-to-one assignment for single tasks as a stable initial solution.",
                implementation_notes=(
                    "When courier_count >= task_count, solve a standard-library min-cost flow over single-task rows, "
                    "then improve with local swaps and extra couriers."
                ),
                suitable_features=["many_couriers", "low_pair_ratio"],
                example_signals={"capacity_ratio_min": 1.0, "pair_ratio_max": 0.35},
                risks=["Single-task flow ignores bundles in the initial phase."],
                recommended_parameters={"use_flow": True},
            ),
            StrategySpec(
                name="beam_cover",
                description="Beam search over compact task masks to maximize coverage before local improvement.",
                implementation_notes=(
                    "Keep the best few courier rows per task group, beam over task masks and courier masks, "
                    "then normalize and locally improve."
                ),
                suitable_features=["small_task_count", "high_pair_ratio"],
                example_signals={"task_count_max": 45},
                risks=["Mask search should be disabled for large task counts to avoid timeout."],
                recommended_parameters={"beam_width": 160, "beam_task_limit": 42},
            ),
            StrategySpec(
                name="local_search_repair",
                description="Improve a valid base solution by moving, swapping, dropping, and repairing couriers/groups.",
                implementation_notes=(
                    "Repeatedly scan high-penalty groups for courier moves/swaps. Periodically destroy a few groups "
                    "and greedily repair uncovered tasks."
                ),
                suitable_features=["general", "needs_penalty_optimization"],
                example_signals={"always": True},
                risks=["Local search must respect a strict internal deadline."],
                recommended_parameters={"local_rounds": 3, "loop_local_rounds": 1},
            ),
        ]

    def all(self) -> List[StrategySpec]:
        return list(self._strategies)

    def select_for_features(self, features: Dict[str, Any]) -> List[StrategySpec]:
        pair_ratio = float(features.get("pair_ratio", 0.0) or 0.0)
        avg_willingness = float(features.get("avg_willingness", 0.0) or 0.0)
        capacity_ratio = float(features.get("capacity_ratio", 0.0) or 0.0)
        task_count = int(features.get("task_count", 0) or 0)
        names = {"expected_greedy", "local_search_repair"}
        if pair_ratio >= 0.2 or capacity_ratio < 1.4:
            names.add("bundle_first")
            names.add("beam_cover")
        if avg_willingness <= 0.35:
            names.add("willingness_weighted")
        if capacity_ratio >= 1.0 and pair_ratio <= 0.35:
            names.add("flow_single_initial")
        if task_count <= 45:
            names.add("beam_cover")
        return [strategy for strategy in self._strategies if strategy.name in names]

    def as_prompt_context(self, features: Dict[str, Any]) -> str:
        selected = self.select_for_features(features)
        payload = {
            "selected_strategy_names": [item.name for item in selected],
            "strategies": [asdict(item) for item in self._strategies],
        }
        return _compact_json(payload)


class SolverSkillLibrary:
    """Reusable implementation guidance for LLM generated solvers."""

    def __init__(self) -> None:
        self._skills = [
            SolverSkill(
                name="standard_library_solver",
                strategy_names=[
                    "expected_greedy",
                    "bundle_first",
                    "willingness_weighted",
                    "flow_single_initial",
                    "beam_cover",
                    "local_search_repair",
                ],
                construction_notes=(
                    "Generate one complete Python file using only the standard library. It must expose "
                    "solve(input_text: str) -> list. The function should parse TSV text, build a valid disjoint "
                    "assignment, optimize within an internal deadline below the judge timeout, and return "
                    "[(task_key, [courier_id, ...]), ...]."
                ),
                code_contract=(
                    "No file IO, network IO, subprocesses, dynamic imports, eval, exec, or compile. "
                    "Do not depend on LangChain, numpy, scipy, pandas, networkx, or any non-standard package."
                ),
                constraints=[
                    "A task may appear in at most one output task group.",
                    "A courier may appear at most once globally.",
                    "Every output courier must exist for that task_key in the input.",
                    "Return a Python list, not JSON text.",
                ],
                examples=[
                    "For high pair_ratio, try bundle-biased greedy plus repair.",
                    "For low willingness, add unused eligible couriers when expected penalty decreases.",
                ],
            )
        ]

    def all(self) -> List[SolverSkill]:
        return list(self._skills)

    def as_prompt_context(self) -> str:
        return _compact_json({"solver_skills": [asdict(item) for item in self._skills]})


def _compact_json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
