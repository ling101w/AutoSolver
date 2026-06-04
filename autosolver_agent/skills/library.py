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
            StrategySpec(
                name="task_first_regret_seed",
                description="Build a full task cover first, seeding one courier per group by regret between best choices.",
                implementation_notes=(
                    "Use bitmasks for task groups, choose disjoint cover groups, then assign the group with the largest "
                    "gap between its best and second-best available courier. Finish with extra-courier gain filling."
                ),
                suitable_features=["general", "single_cover_available", "sparse_candidate_groups", "tight_capacity"],
                example_signals={"single_task_group_coverage_min": 0.7, "avg_candidates_per_group_min": 1.5},
                risks=["Regret seed is deterministic and may overfit tie order without hash/order variants."],
                recommended_parameters={"regret_weight": 1.0, "seed_order_variants": 4},
                reference_examples=[
                    "task_first_greedy_repair_reference",
                    "template_task_first_reference",
                ],
            ),
            StrategySpec(
                name="hash_multi_start",
                description="Run deterministic hash-order seed variants to diversify without random instability.",
                implementation_notes=(
                    "Sort selected cover groups by several stable integer hash seeds, construct ordered seed states, "
                    "polish each one, and keep the best by coverage then penalty."
                ),
                suitable_features=["high_score_variance", "large_task_count", "needs_tie_diversification"],
                example_signals={"score_cv_min": 0.5, "task_count_min": 40},
                risks=["Too many hash starts can spend time before deeper local improvement."],
                recommended_parameters={"hash_seeds": [289, 245, 173, 95, 242], "max_hash_starts": 5},
                reference_examples=[
                    "task_first_greedy_repair_reference",
                    "multi_start_hybrid_reference",
                    "template_task_first_reference",
                    "template_hybrid_metaheuristic_reference",
                ],
            ),
            StrategySpec(
                name="pair_replacement_polish",
                description="Replace two active singleton groups with a compatible pair group when it lowers penalty.",
                implementation_notes=(
                    "Shortlist cheap pair groups, simulate replacing their active singleton components, then apply the "
                    "best improving replacement followed by courier-gain polish."
                ),
                suitable_features=["bundle_rich", "high_pair_ratio", "small_task_count", "medium_task_count"],
                example_signals={"bundle_ratio_min": 0.15, "single_task_group_coverage_min": 0.7},
                risks=["Pair replacement should reject overlapping or uncovered-task regressions."],
                recommended_parameters={"candidate_limit": 120, "shortlist": 20, "rounds": 4},
                reference_examples=[
                    "task_first_greedy_repair_reference",
                    "template_task_first_reference",
                    "template_hybrid_metaheuristic_reference",
                ],
            ),
            StrategySpec(
                name="courier_relocation_swap",
                description="Move or swap assigned couriers between active groups using incremental penalty deltas.",
                implementation_notes=(
                    "Track each courier owner and each group's penalty statistics. Repeatedly relocate one courier or "
                    "swap two couriers when the combined delta is negative, then refill improving extras."
                ),
                suitable_features=["dense_candidate_groups", "mobile_couriers", "needs_penalty_optimization"],
                example_signals={"avg_groups_per_courier_min": 2.0, "avg_candidates_per_group_min": 3.0},
                risks=["Must keep at least one courier per active group and preserve global courier uniqueness."],
                recommended_parameters={"local_passes": 6, "delta_epsilon": 1e-12},
                reference_examples=[
                    "task_first_greedy_repair_reference",
                    "multi_start_hybrid_reference",
                    "template_task_first_reference",
                ],
            ),
            StrategySpec(
                name="three_cycle_polish",
                description="Search three active groups for cyclic courier rotations that reduce total penalty.",
                implementation_notes=(
                    "For compact active sets, test A->B->C->A and A->C->B->A courier rotations and apply a few "
                    "negative-delta cycles, with local polish after each application."
                ),
                suitable_features=["dense_candidate_groups", "mobile_couriers", "small_task_count"],
                example_signals={"active_group_max": 45, "avg_groups_per_courier_min": 2.5},
                risks=["Cubic scans must be disabled on large active sets."],
                recommended_parameters={"move_limit": 5, "max_active_groups": 45},
                reference_examples=[
                    "task_first_greedy_repair_reference",
                    "template_task_first_reference",
                ],
            ),
            StrategySpec(
                name="randomized_shuffled_greedy",
                description="Use seeded randomized row ordering with temperature, pair bias, and willingness bias.",
                implementation_notes=(
                    "Score each group-courier row by singleton penalty per task, bundle bias, willingness bias, and "
                    "small seeded random noise. Construct disjoint seeds, repair missing tasks, then polish."
                ),
                suitable_features=["high_score_variance", "very_low_willingness_tail", "scarce_couriers", "large_task_count"],
                example_signals={"score_cv_min": 0.5, "avg_willingness_max": 0.35},
                risks=["Needs deterministic RNG seeding for reproducible artifacts."],
                recommended_parameters={"temperature_schedule": [1.5, 4.0, 9.0, 18.0], "pair_bias_max": 140.0},
                reference_examples=[
                    "multi_start_hybrid_reference",
                    "template_hybrid_metaheuristic_reference",
                ],
            ),
            StrategySpec(
                name="min_weight_matching_seed",
                description="Construct a one-row-per-task/group matching seed for scarce-courier cases.",
                implementation_notes=(
                    "When couriers are tight, choose non-overlapping group-courier rows with a matching-style objective, "
                    "favoring coverage first and expected singleton penalty second."
                ),
                suitable_features=["scarce_couriers", "tight_capacity", "single_cover_available"],
                example_signals={"capacity_ratio_max": 1.35},
                risks=["A pure row matching seed may underuse beneficial secondary couriers until later polish."],
                recommended_parameters={"coverage_bonus": 10000.0, "scarce_pair_bias": 60.0},
                reference_examples=[
                    "multi_start_hybrid_reference",
                    "template_hybrid_metaheuristic_reference",
                ],
            ),
            StrategySpec(
                name="tabu_confchange",
                description="Run bounded tabu-style move and swap search with conflict-change gating.",
                implementation_notes=(
                    "Maintain short tabu tenures for courier/group moves, allow aspiration if a move improves the best "
                    "state, and use conflict-change flags to skip stale neighborhoods."
                ),
                suitable_features=["large_task_count", "dense_candidate_groups", "needs_penalty_optimization"],
                example_signals={"task_count_min": 80, "avg_candidates_per_group_min": 3.0},
                risks=["Must use deadline checks and cap sampled swaps to avoid timeout."],
                recommended_parameters={"default_steps": 18, "swap_trials": 80, "tenure_base": 4},
                reference_examples=[
                    "multi_start_hybrid_reference",
                    "template_hybrid_metaheuristic_reference",
                ],
            ),
            StrategySpec(
                name="destroy_repair_ils",
                description="Iteratively perturb the current state by dropping groups or extras, then greedily repair.",
                implementation_notes=(
                    "Alternate perturb_extras, kick_state, and destroy_repair against the current best clone. After each "
                    "perturbation, repair coverage, refill improving couriers, and apply local polish."
                ),
                suitable_features=["large_task_count", "high_reject_risk", "high_score_variance"],
                example_signals={"task_count_min": 60, "low_willingness_ratio_min": 0.35},
                risks=["Destroy size should grow slowly and always leave enough time to repair coverage."],
                recommended_parameters={"max_drop_count": 4, "kick_strength_max": 8, "no_improve_reset": 4},
                reference_examples=[
                    "multi_start_hybrid_reference",
                    "template_hybrid_metaheuristic_reference",
                ],
            ),
            StrategySpec(
                name="repartition_small_union",
                description="Repartition tasks across two active groups when the union is small enough to enumerate.",
                implementation_notes=(
                    "Focus on high-penalty active pairs whose union has at most a few tasks and couriers. Enumerate "
                    "existing singleton/pair group partitions and brute-force courier assignment within that small union."
                ),
                suitable_features=["bundle_rich", "small_task_count", "medium_task_count", "dense_candidate_groups"],
                example_signals={"max_union_tasks": 4, "max_union_couriers": 6},
                risks=["Only run on tiny unions and high-penalty focus sets."],
                recommended_parameters={"max_tasks": 4, "max_couriers": 6, "focus_fraction": 0.6},
                reference_examples=[
                    "multi_start_hybrid_reference",
                    "template_hybrid_metaheuristic_reference",
                ],
            ),
        ]

    def all(self) -> List[StrategySpec]:
        return list(self._strategies)

    def select_for_features(self, features: Dict[str, Any]) -> List[StrategySpec]:
        pair_ratio = float(features.get("pair_ratio", 0.0) or 0.0)
        bundle_ratio = float(features.get("bundle_ratio", 0.0) or 0.0)
        avg_willingness = float(features.get("avg_willingness", 0.0) or 0.0)
        low_willingness_ratio = float(features.get("low_willingness_ratio", 0.0) or 0.0)
        capacity_ratio = float(features.get("capacity_ratio", 0.0) or 0.0)
        task_count = int(features.get("task_count", 0) or 0)
        avg_candidates = float(features.get("avg_candidates_per_group", 0.0) or 0.0)
        avg_groups_per_courier = float(features.get("avg_groups_per_courier", 0.0) or 0.0)
        score_cv = float(features.get("score_cv", 0.0) or 0.0)
        single_cover = float(features.get("single_task_group_coverage", 0.0) or 0.0)
        tags = set(features.get("tags", []))
        names = {"expected_greedy", "local_search_repair", "task_first_regret_seed", "courier_relocation_swap"}
        if pair_ratio >= 0.2 or bundle_ratio >= 0.15 or capacity_ratio < 1.4 or "bundle_rich" in tags:
            names.add("bundle_first")
            names.add("beam_cover")
            names.add("pair_replacement_polish")
        if avg_willingness <= 0.35:
            names.add("willingness_weighted")
        if low_willingness_ratio >= 0.35:
            names.add("destroy_repair_ils")
        if capacity_ratio >= 1.0 and pair_ratio <= 0.35 and single_cover >= 0.8:
            names.add("flow_single_initial")
        if capacity_ratio <= 1.35:
            names.add("min_weight_matching_seed")
        if task_count <= 45:
            names.add("beam_cover")
            names.add("three_cycle_polish")
            if bundle_ratio >= 0.1:
                names.add("repartition_small_union")
        if avg_candidates >= 3.0 or avg_groups_per_courier >= 2.5:
            names.add("three_cycle_polish")
            names.add("tabu_confchange")
        if task_count >= 80 or score_cv >= 0.5:
            names.add("hash_multi_start")
            names.add("randomized_shuffled_greedy")
        if task_count >= 100 or score_cv >= 0.8:
            names.add("destroy_repair_ils")
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
                    "task_first_regret_seed",
                    "hash_multi_start",
                    "pair_replacement_polish",
                    "courier_relocation_swap",
                    "three_cycle_polish",
                    "randomized_shuffled_greedy",
                    "min_weight_matching_seed",
                    "tabu_confchange",
                    "destroy_repair_ils",
                    "repartition_small_union",
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
            ),
            SolverSkill(
                name="incremental_bitmask_state",
                strategy_names=[
                    "task_first_regret_seed",
                    "pair_replacement_polish",
                    "courier_relocation_swap",
                    "three_cycle_polish",
                    "repartition_small_union",
                    "tabu_confchange",
                ],
                construction_notes=(
                    "Represent each task group by an integer mask and maintain State with active groups, assigned couriers, "
                    "owner[courier], covered mask, incremental willingness sums, reject product, and group penalty."
                ),
                code_contract="Keep all state mutation in add_courier/remove_courier/remove_group helpers with invariant-preserving updates.",
                constraints=[
                    "A courier owner must be -1 or exactly one active group.",
                    "An active group must have at least one assigned courier before formatting output.",
                    "covered_count must be derived from the task mask after group activation/removal.",
                ],
                examples=[
                    "Use penalty_after_add and penalty_after_remove to evaluate moves without mutating.",
                    "Clone and restore states before evaluating destructive pair or repartition moves.",
                ],
            ),
            SolverSkill(
                name="deterministic_seed_constructors",
                strategy_names=[
                    "expected_greedy",
                    "bundle_first",
                    "task_first_regret_seed",
                    "hash_multi_start",
                    "min_weight_matching_seed",
                    "flow_single_initial",
                ],
                construction_notes=(
                    "Generate several valid initial covers: task-first regret, ordered by name, ordered by best penalty, "
                    "ordered by willingness, stable hash-order starts, and optional flow/matching starts."
                ),
                code_contract="Every constructor should return a valid State or None and must be passed through the same better_state gate.",
                constraints=[
                    "Prefer full coverage before optimizing penalty.",
                    "Use deterministic tie-breakers so artifacts are reproducible.",
                    "Disable expensive constructors when feature thresholds or time_left checks fail.",
                ],
                examples=[
                    "For scarce couriers, try min_weight_matching_seed before randomized starts.",
                    "For single_cover_available and low_pair_ratio, try flow_single_initial as a stable baseline.",
                ],
            ),
            SolverSkill(
                name="local_move_polisher",
                strategy_names=[
                    "local_search_repair",
                    "courier_relocation_swap",
                    "three_cycle_polish",
                    "pair_replacement_polish",
                    "repartition_small_union",
                ],
                construction_notes=(
                    "Improve a valid state with bounded negative-delta moves: add improving extras, relocate couriers, "
                    "swap couriers between groups, replace singleton pairs by bundle groups, and apply tiny three-cycles."
                ),
                code_contract="Each move must be evaluated with exact penalty deltas and applied only if it preserves validity.",
                constraints=[
                    "Never leave an active group empty unless removing the whole group.",
                    "Run coverage repair after any group replacement or repartition step.",
                    "Cap cubic cycle scans by active group count.",
                ],
                examples=[
                    "Run relocate -> swap -> add extras in repeated polish passes.",
                    "For bundle_rich cases, shortlist cheap pair groups before detailed replacement simulation.",
                ],
            ),
            SolverSkill(
                name="metaheuristic_loop_control",
                strategy_names=[
                    "randomized_shuffled_greedy",
                    "destroy_repair_ils",
                    "tabu_confchange",
                    "hash_multi_start",
                ],
                construction_notes=(
                    "Use a deadline-aware multi-start loop that alternates deterministic starts, randomized greedy starts, "
                    "destroy-repair perturbations, kick moves, extra-courier perturbation, and tabu/conf-change search."
                ),
                code_contract="Seed RNG deterministically from instance size and stop all optional work before the internal safety margin.",
                constraints=[
                    "Always keep the current best clone separate from the working state.",
                    "Jump back to best after several non-improving rounds.",
                    "Scale temperature, pair_bias, and willingness_bias from feature tags such as scarce or low willingness.",
                ],
                examples=[
                    "Use larger pair_bias for scarce or bundle-rich cases.",
                    "Use larger willingness_bias for very_low_willingness_tail cases.",
                ],
            ),
            SolverSkill(
                name="coverage_repair_guardrails",
                strategy_names=[
                    "bundle_first",
                    "beam_cover",
                    "destroy_repair_ils",
                    "repartition_small_union",
                    "local_search_repair",
                ],
                construction_notes=(
                    "After any partial construction, fill uncovered tasks by direct non-conflicting groups first, then by "
                    "replacing conflicting active groups only when all removed tasks remain covered by the new group."
                ),
                code_contract="Repair functions must return whether full coverage was restored and must leave the state valid even on failure.",
                constraints=[
                    "Do not emit duplicate tasks.",
                    "Do not steal a courier from a singleton group unless that group is also removed or still has another courier.",
                    "Prefer direct repairs over replacement repairs for simplicity and validity.",
                ],
                examples=[
                    "best_direct_coverage_repair handles missing task masks with unused couriers.",
                    "best_replacement_coverage_repair can activate a bundle group when it exactly covers conflicting singleton groups.",
                ],
            ),
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
            SolverExample(
                name="template_task_first_reference",
                source_file="examples/solver_template_2.py",
                strategy_names=[
                    "task_first_regret_seed",
                    "hash_multi_start",
                    "pair_replacement_polish",
                    "courier_relocation_swap",
                    "three_cycle_polish",
                    "local_search_repair",
                ],
                summary=(
                    "Compact deterministic template centered on bitmask State, task-first cover selection, regret seeding, "
                    "hash-order variants, pair replacement, three-courier cycle polish, and explicit coverage repair."
                ),
                applicable_features=[
                    "small_task_count",
                    "medium_task_count",
                    "single_cover_available",
                    "bundle_rich",
                    "dense_candidate_groups",
                    "compact_mask_search_candidate",
                ],
                entry_points=[
                    "solve -> parse_input -> construct_task_first_greedy_solution -> repair_task_coverage -> format_solution",
                    "initial_single_task_states: regret seed plus name, penalty, willingness, and hash-order starts",
                    "improve_by_pair_group_replacements and polish_three_courier_cycles for compact local improvement",
                ],
                reusable_patterns=[
                    "Keep a single incremental State model and never optimize on raw output tuples.",
                    "Use choose_cover_groups to separate task coverage from courier assignment.",
                    "Use exact penalty_after_add/remove deltas for extras, relocation, swaps, and cycle moves.",
                    "Call repair_task_coverage after any replacement that may uncover tasks.",
                ],
                implementation_guardrails=[
                    "Do not run three-cycle scans when active groups exceed the configured cap.",
                    "Pair replacement should only replace active singleton groups that exactly match the pair mask.",
                    "Every formatting step sorts stable names for deterministic output.",
                ],
                prompt_excerpt=(
                    "Template 2 useful outline: bitmask ProblemData/State; construct_task_first_greedy_solution; "
                    "initial_single_task_states with regret and hash orders; allocate_remaining_couriers_by_gain; "
                    "relocate/swap polish; pair replacement; coverage repair; three-cycle polish."
                ),
            ),
            SolverExample(
                name="template_hybrid_metaheuristic_reference",
                source_file="examples/solver_template_1.py",
                strategy_names=[
                    "task_first_regret_seed",
                    "hash_multi_start",
                    "randomized_shuffled_greedy",
                    "min_weight_matching_seed",
                    "flow_single_initial",
                    "tabu_confchange",
                    "destroy_repair_ils",
                    "repartition_small_union",
                    "local_search_repair",
                ],
                summary=(
                    "Full hybrid template that adds instance-adaptive multi-start construction, matching/flow seeds, "
                    "randomized greedy schedules, tabu/conf-change moves, destroy-repair, kicks, perturbation, and repartition."
                ),
                applicable_features=[
                    "large_task_count",
                    "scarce_couriers",
                    "tight_capacity",
                    "very_low_willingness_tail",
                    "high_reject_risk",
                    "high_score_variance",
                    "dense_candidate_groups",
                ],
                entry_points=[
                    "solve: deterministic phase -> optional structured starts -> ILS loop until deadline",
                    "init_shuffled_greedy with temperature, pair_bias, willingness_bias schedules",
                    "tabu_confchange, perturb_extras, kick_state, destroy_repair, repartition_state",
                ],
                reusable_patterns=[
                    "Classify inside solve with scarce_case, low_case, very_low_case, and hard_case booleans.",
                    "Gate expensive structured starts by time_left and instance size thresholds.",
                    "Use best/current-work separation and restore best after several non-improving iterations.",
                    "Scale pair_bias upward for scarce cases and willingness_bias upward for low acceptance tails.",
                ],
                implementation_guardrails=[
                    "Reserve a safety margin and check time_left before every expensive optional phase.",
                    "Keep randomized behavior deterministic with an instance-size-based seed.",
                    "Do not enumerate repartitions unless task and courier union caps are satisfied.",
                ],
                prompt_excerpt=(
                    "Template 1 useful outline: start from deterministic task-first seed; try hash seeds, flow and matching "
                    "seeds when suitable; optionally use compact bundle starts; then run deadline-bounded ILS with shuffled "
                    "greedy, tabu_confchange, perturb_extras, kick_state, destroy_repair, repartition, and final polish."
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
