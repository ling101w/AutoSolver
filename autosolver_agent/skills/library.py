"""Strategy and solver skill knowledge base."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from autosolver_agent.models import SolverExample, SolverSkill, StrategySpec


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
                reference_examples=[
                    "basic_seed_solver_pack",
                    "task_first_greedy_repair_reference",
                    "multi_start_hybrid_reference",
                ],
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
                reference_examples=[
                    "basic_seed_solver_pack",
                    "task_first_greedy_repair_reference",
                    "multi_start_hybrid_reference",
                ],
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
                reference_examples=[
                    "basic_seed_solver_pack",
                    "task_first_greedy_repair_reference",
                    "multi_start_hybrid_reference",
                ],
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
                reference_examples=["basic_seed_solver_pack", "multi_start_hybrid_reference"],
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
                reference_examples=[
                    "basic_seed_solver_pack",
                    "task_first_greedy_repair_reference",
                    "multi_start_hybrid_reference",
                ],
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
                reference_examples=[
                    "basic_seed_solver_pack",
                    "task_first_greedy_repair_reference",
                    "multi_start_hybrid_reference",
                ],
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
                    "Use solver_examples as reference architectures; adapt the patterns instead of copying every routine.",
                ],
            )
        ]
        self._examples = [
            SolverExample(
                name="basic_seed_solver_pack",
                source_file="solvers/seed_solvers.py",
                strategy_names=[
                    "expected_greedy",
                    "bundle_first",
                    "willingness_weighted",
                    "flow_single_initial",
                    "beam_cover",
                    "local_search_repair",
                ],
                summary=(
                    "A directly callable pack of complete baseline seed solvers. Each strategy has its own solve_* "
                    "entry point plus a default solve() that runs the local_search_repair seed."
                ),
                applicable_features=[
                    "general",
                    "small_task_count",
                    "many_couriers",
                    "scarce_couriers",
                    "high_pair_ratio",
                    "low_willingness",
                    "low_pair_ratio",
                ],
                entry_points=[
                    "solve_expected_greedy(input_text)",
                    "solve_bundle_first(input_text)",
                    "solve_willingness_weighted(input_text)",
                    "solve_flow_single_initial(input_text)",
                    "solve_beam_cover(input_text)",
                    "solve_local_search_repair(input_text)",
                    "SEED_SOLVERS[strategy_name](input_text)",
                ],
                reusable_patterns=[
                    "Use one shared parser and penalty model, then vary the group/courier ranking function per strategy.",
                    "Run a coverage repair pass after every seed construction to keep output valid.",
                    "Use min-cost flow only for the single-task initial solution, then repair missing tasks with bundle rows.",
                    "Use beam search only on bounded task counts and fall back to bundle-first greedy for larger cases.",
                    "Use local replacement search to swap conflicting active groups for better bundle or penalty choices.",
                ],
                implementation_guardrails=[
                    "Every public solver returns the official list-of-tuples answer shape.",
                    "The module uses only Python standard library imports.",
                    "The default solve() remains callable by the existing candidate runtime and validator.",
                ],
                prompt_excerpt=(
                    "Reference seed pack: call solve_expected_greedy, solve_bundle_first, solve_willingness_weighted, "
                    "solve_flow_single_initial, solve_beam_cover, or solve_local_search_repair. "
                    "Each function parses TSV, constructs a valid disjoint seed, repairs missing coverage, "
                    "adds improving extra couriers, and formats the answer."
                ),
            ),
            SolverExample(
                name="task_first_greedy_repair_reference",
                source_file="solvers/solver.py",
                strategy_names=[
                    "expected_greedy",
                    "bundle_first",
                    "willingness_weighted",
                    "beam_cover",
                    "local_search_repair",
                ],
                summary=(
                    "Deterministic task-first construction that parses TSV rows into compact task-group and courier ids, "
                    "builds one feasible cover, greedily adds useful extra couriers, then repairs and polishes."
                ),
                applicable_features=[
                    "general",
                    "small_task_count",
                    "high_pair_ratio",
                    "low_willingness",
                    "needs_penalty_optimization",
                ],
                entry_points=[
                    "solve: parse_input -> construct_task_first_greedy_solution -> repair_task_coverage -> format_solution",
                    "construct_task_first_greedy_solution: choose cover groups, seed variants, pair replacement, three-cycle polish",
                    "polish_courier_assignment: relocate, swap, then add remaining couriers by penalty gain",
                    "repair_task_coverage: direct missing-group add or replacement of conflicting active groups",
                ],
                reusable_patterns=[
                    "Represent each task group as a bitmask so coverage/conflict checks are O(1).",
                    "Track owner[courier], active groups, assigned couriers, covered task mask, and incremental penalty stats.",
                    "Seed one courier per group using regret, singleton penalty, willingness, name order, and hash order variants.",
                    "Only add unused couriers when penalty_after_add improves the active group penalty.",
                    "Run local repair in bounded passes: relocate courier, swap couriers, add extras, replace pair groups, repair coverage.",
                ],
                implementation_guardrails=[
                    "Keep all output task groups disjoint and every courier globally unique.",
                    "After any bundle replacement or destroy step, call coverage repair before formatting the answer.",
                    "Use deterministic tie-breakers for stable artifacts and easier scorer comparisons.",
                ],
                prompt_excerpt=(
                    "Reference outline from solver.py: parse into ProblemData/State; choose_cover_groups; "
                    "seed_groups_by_regret or ordered/hash seeds; allocate_remaining_couriers_by_gain; "
                    "polish by relocate_couriers_by_gain and swap_couriers_by_gain; "
                    "repair uncovered tasks with direct or replacement coverage moves."
                ),
            ),
            SolverExample(
                name="multi_start_hybrid_reference",
                source_file="solvers/solver_70433_best_E1.py",
                strategy_names=[
                    "expected_greedy",
                    "bundle_first",
                    "willingness_weighted",
                    "flow_single_initial",
                    "beam_cover",
                    "local_search_repair",
                ],
                summary=(
                    "Fused multi-start solver that keeps the same incremental State model, tries multiple initial "
                    "constructors under a hard deadline, and iterates with perturbation, tabu-style swaps, and repair."
                ),
                applicable_features=[
                    "many_couriers",
                    "low_pair_ratio",
                    "scarce_couriers",
                    "low_willingness",
                    "large_task_count",
                    "needs_penalty_optimization",
                ],
                entry_points=[
                    "solve: deterministic seeds -> structured initial solutions -> iterative local search until deadline",
                    "init_min_cost_flow_single: single-task min-cost assignment for many-courier low-pair cases",
                    "init_min_weight_matching: scarce-courier one-row matching seed",
                    "init_shuffled_greedy: randomized greedy with temperature, pair_bias, and willingness_bias",
                    "destroy_repair / kick_state / perturb_extras / tabu_confchange: diversification and local improvement",
                ],
                reusable_patterns=[
                    "Classify the instance inside solve with avg_willingness, courier/task ratio, scarce_case, low_case, and hard_case.",
                    "Consider every seed through a single better_state gate and clone the best state to avoid accidental mutation.",
                    "Use deadline checks before expensive starts; scale optional work by remaining time and instance size.",
                    "For low willingness, increase willingness_bias; for scarce cases, increase pair_bias to explore bundled cover.",
                    "Alternate structured seeds with perturb/destroy-repair loops, then polish and re-evaluate the best state.",
                ],
                implementation_guardrails=[
                    "Keep an internal safety margin below the judge timeout.",
                    "Do not introduce non-standard packages; matching/flow/LP-style routines must be standard-library implementations.",
                    "Disable expensive mask/partition or structured starts on large instances unless time_left is clearly sufficient.",
                ],
                prompt_excerpt=(
                    "Reference outline from solver_70433_best_E1.py: build deterministic task-first and hash seeds; "
                    "optionally try init_min_cost_flow_single and init_min_weight_matching seeds; "
                    "for suitable cases try compact bundle starts; "
                    "loop until deadline with destroy_repair, kick_state, perturb_extras, randomized greedy, tabu_confchange, "
                    "and final polish; return format_solution(best)."
                ),
            ),
        ]

    def all(self) -> List[SolverSkill]:
        return list(self._skills)

    def examples(self) -> List[SolverExample]:
        return list(self._examples)

    def as_prompt_context(self) -> str:
        return _compact_json(
            {
                "solver_skills": [asdict(item) for item in self._skills],
                "solver_examples": [asdict(item) for item in self._examples],
            }
        )


def _compact_json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
