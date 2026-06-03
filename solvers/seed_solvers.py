"""Directly callable seed solvers for the strategy library.

Each public solve_* function accepts TSV text and returns:
    [(task_id_list, [courier_id, ...]), ...]

The implementations are intentionally compact reference baselines. They prefer
valid coverage first, then reduce expected penalty with simple local repairs.
"""

from __future__ import annotations

from heapq import heappop, heappush
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


EPS = 1e-12
FALLBACK_PER_TASK = 100.0
BEAM_WIDTH = 160
BEAM_GROUP_LIMIT = 220

Problem = Dict[str, Any]
Answer = List[Tuple[str, List[str]]]
CourierRanker = Callable[[Problem, str, str], Tuple[Any, ...]]
GroupRanker = Callable[[Problem, str, str], Tuple[Any, ...]]


def solve_expected_greedy(input_text: str) -> list:
    """Rank task-group/courier rows by expected penalty and greedily cover tasks."""

    problem = _parse_problem(input_text)
    if not problem["key_tasks"]:
        return []
    answer = _greedy_construct(problem, _expected_group_rank, _expected_courier_rank)
    return _format_answer(problem, answer)


def solve_bundle_first(input_text: str) -> list:
    """Prefer multi-task groups before repairing uncovered single tasks."""

    problem = _parse_problem(input_text)
    if not problem["key_tasks"]:
        return []
    answer = _greedy_construct(problem, _bundle_group_rank, _expected_courier_rank)
    answer = _local_replacement_search(problem, answer, max_rounds=2, prefer_bundles=True)
    return _format_answer(problem, answer)


def solve_willingness_weighted(input_text: str) -> list:
    """Bias seed choices toward high acceptance probability, then add useful extras."""

    problem = _parse_problem(input_text)
    if not problem["key_tasks"]:
        return []
    answer = _greedy_construct(problem, _willingness_group_rank, _willingness_courier_rank)
    _add_improving_extras(problem, answer, max_passes=4, willingness_floor=0.0)
    return _format_answer(problem, answer)


def solve_flow_single_initial(input_text: str) -> list:
    """Build a single-task min-cost assignment seed, then repair with bundle rows."""

    problem = _parse_problem(input_text)
    if not problem["key_tasks"]:
        return []
    answer = _min_cost_single_task_assignment(problem)
    if not answer:
        answer = _greedy_construct(problem, _expected_group_rank, _expected_courier_rank)
    else:
        used_tasks, used_couriers = _used_sets(problem, answer)
        _repair_missing(problem, answer, used_tasks, used_couriers, _expected_courier_rank)
        _add_improving_extras(problem, answer, max_passes=3)
    return _format_answer(problem, answer)


def solve_beam_cover(input_text: str) -> list:
    """Beam search over compact task masks for small/medium task-count cases."""

    problem = _parse_problem(input_text)
    if not problem["key_tasks"]:
        return []
    if len(problem["all_tasks"]) > 48:
        return solve_bundle_first(input_text)

    answer = _beam_construct(problem)
    used_tasks, used_couriers = _used_sets(problem, answer)
    _repair_missing(problem, answer, used_tasks, used_couriers, _expected_courier_rank)
    _add_improving_extras(problem, answer, max_passes=3)
    return _format_answer(problem, answer)


def solve_local_search_repair(input_text: str) -> list:
    """Start from expected greedy, then apply simple replacement and repair passes."""

    problem = _parse_problem(input_text)
    if not problem["key_tasks"]:
        return []
    answer = _greedy_construct(problem, _expected_group_rank, _expected_courier_rank)
    answer = _local_replacement_search(problem, answer, max_rounds=5, prefer_bundles=False)
    _add_improving_extras(problem, answer, max_passes=4)
    return _format_answer(problem, answer)


SEED_SOLVERS: Dict[str, Callable[[str], list]] = {
    "expected_greedy": solve_expected_greedy,
    "bundle_first": solve_bundle_first,
    "willingness_weighted": solve_willingness_weighted,
    "flow_single_initial": solve_flow_single_initial,
    "beam_cover": solve_beam_cover,
    "local_search_repair": solve_local_search_repair,
}


def solve(input_text: str) -> list:
    """Default seed solver: valid greedy seed plus local repair."""

    return solve_local_search_repair(input_text)


def _parse_problem(input_text: str) -> Problem:
    by_key: Dict[str, Dict[str, Tuple[float, float]]] = {}
    key_tasks: Dict[str, Tuple[str, ...]] = {}
    task_order: List[str] = []
    courier_order: List[str] = []
    seen_tasks = set()
    seen_couriers = set()

    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0
    for line in lines[start:]:
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        raw_key, courier, score_raw, willingness_raw = parts[:4]
        tasks = tuple(task.strip() for task in raw_key.split(",") if task.strip())
        courier = courier.strip()
        if not tasks or not courier:
            continue
        try:
            score = float(score_raw)
            willingness = float(willingness_raw)
        except ValueError:
            continue

        key = ",".join(tasks)
        key_tasks[key] = tasks
        if key not in by_key:
            by_key[key] = {}
        old = by_key[key].get(courier)
        if old is None or _singleton_penalty(len(tasks), score, willingness) < _singleton_penalty(
            len(tasks), old[0], old[1]
        ):
            by_key[key][courier] = (score, willingness)

        for task in tasks:
            if task not in seen_tasks:
                seen_tasks.add(task)
                task_order.append(task)
        if courier not in seen_couriers:
            seen_couriers.add(courier)
            courier_order.append(courier)

    single_key_by_task = {}
    for key, tasks in key_tasks.items():
        if len(tasks) == 1:
            single_key_by_task[tasks[0]] = key

    return {
        "by_key": by_key,
        "key_tasks": key_tasks,
        "all_tasks": task_order,
        "all_couriers": courier_order,
        "single_key_by_task": single_key_by_task,
    }


def _singleton_penalty(task_count: int, score: float, willingness: float) -> float:
    fallback = FALLBACK_PER_TASK * task_count
    return (1.0 - willingness) * fallback + willingness * score


def _group_penalty(problem: Problem, key: str, couriers: Sequence[str]) -> float:
    tasks = problem["key_tasks"][key]
    fallback = FALLBACK_PER_TASK * len(tasks)
    data = problem["by_key"][key]
    reject_prob = 1.0
    weighted_score = 0.0
    weight = 0.0
    for courier in couriers:
        score, willingness = data[courier]
        reject_prob *= 1.0 - willingness
        weighted_score += willingness * score
        weight += willingness
    if weight <= EPS:
        return fallback
    return reject_prob * fallback + (1.0 - reject_prob) * weighted_score / weight


def _expected_courier_rank(problem: Problem, key: str, courier: str) -> Tuple[Any, ...]:
    score, willingness = problem["by_key"][key][courier]
    penalty = _singleton_penalty(len(problem["key_tasks"][key]), score, willingness)
    return (penalty, score, -willingness, courier)


def _willingness_courier_rank(problem: Problem, key: str, courier: str) -> Tuple[Any, ...]:
    score, willingness = problem["by_key"][key][courier]
    penalty = _singleton_penalty(len(problem["key_tasks"][key]), score, willingness)
    return (-willingness, penalty, score, courier)


def _best_available_courier(
    problem: Problem,
    key: str,
    used_couriers: set,
    courier_ranker: CourierRanker,
) -> Optional[str]:
    candidates = [
        (courier_ranker(problem, key, courier), courier)
        for courier in problem["by_key"][key]
        if courier not in used_couriers
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _expected_group_rank(problem: Problem, key: str, courier: str) -> Tuple[Any, ...]:
    tasks = problem["key_tasks"][key]
    penalty = _group_penalty(problem, key, [courier])
    return (penalty / len(tasks), penalty, -len(tasks), key, courier)


def _bundle_group_rank(problem: Problem, key: str, courier: str) -> Tuple[Any, ...]:
    tasks = problem["key_tasks"][key]
    penalty = _group_penalty(problem, key, [courier])
    bundle_priority = 0 if len(tasks) > 1 else 1
    return (bundle_priority, penalty / len(tasks), penalty, -len(tasks), key, courier)


def _willingness_group_rank(problem: Problem, key: str, courier: str) -> Tuple[Any, ...]:
    score, willingness = problem["by_key"][key][courier]
    tasks = problem["key_tasks"][key]
    penalty = _group_penalty(problem, key, [courier])
    return (-willingness, penalty / len(tasks), score, -len(tasks), key, courier)


def _ranked_group_candidates(
    problem: Problem,
    group_ranker: GroupRanker,
    courier_ranker: CourierRanker,
) -> List[Tuple[str, str]]:
    items = []
    for key in problem["by_key"]:
        courier = _best_available_courier(problem, key, set(), courier_ranker)
        if courier is None:
            continue
        items.append((group_ranker(problem, key, courier), key, courier))
    items.sort(key=lambda item: item[0])
    return [(key, courier) for _, key, courier in items]


def _greedy_construct(
    problem: Problem,
    group_ranker: GroupRanker,
    courier_ranker: CourierRanker,
) -> Answer:
    answer: Answer = []
    used_tasks = set()
    used_couriers = set()

    for key, courier in _ranked_group_candidates(problem, group_ranker, courier_ranker):
        tasks = problem["key_tasks"][key]
        if courier in used_couriers or any(task in used_tasks for task in tasks):
            continue
        answer.append((key, [courier]))
        used_couriers.add(courier)
        used_tasks.update(tasks)

    _repair_missing(problem, answer, used_tasks, used_couriers, courier_ranker)
    _add_improving_extras(problem, answer, max_passes=3)
    return answer


def _repair_missing(
    problem: Problem,
    answer: Answer,
    used_tasks: set,
    used_couriers: set,
    courier_ranker: CourierRanker,
) -> None:
    all_tasks = set(problem["all_tasks"])
    while used_tasks != all_tasks:
        missing = all_tasks - used_tasks
        best = None
        for key, tasks in problem["key_tasks"].items():
            task_set = set(tasks)
            if task_set & used_tasks:
                continue
            newly_covered = len(task_set & missing)
            if newly_covered <= 0:
                continue
            courier = _best_available_courier(problem, key, used_couriers, courier_ranker)
            if courier is None:
                continue
            penalty = _group_penalty(problem, key, [courier])
            rank = (-newly_covered, penalty / newly_covered, penalty, key, courier)
            if best is None or rank < best[0]:
                best = (rank, key, courier)
        if best is None:
            return
        _, key, courier = best
        answer.append((key, [courier]))
        used_couriers.add(courier)
        used_tasks.update(problem["key_tasks"][key])


def _add_improving_extras(
    problem: Problem,
    answer: Answer,
    max_passes: int = 3,
    willingness_floor: Optional[float] = None,
) -> None:
    _, used_couriers = _used_sets(problem, answer)
    for _ in range(max_passes):
        best = None
        for index, (key, couriers) in enumerate(answer):
            old_penalty = _group_penalty(problem, key, couriers)
            for courier, (_, willingness) in problem["by_key"][key].items():
                if courier in used_couriers:
                    continue
                if willingness_floor is not None and willingness < willingness_floor:
                    continue
                new_penalty = _group_penalty(problem, key, couriers + [courier])
                delta = new_penalty - old_penalty
                rank = (delta, -willingness, courier)
                if delta < -EPS and (best is None or rank < best[0]):
                    best = (rank, index, courier)
        if best is None:
            return
        _, index, courier = best
        answer[index][1].append(courier)
        used_couriers.add(courier)


def _min_cost_single_task_assignment(problem: Problem) -> Answer:
    tasks = list(problem["all_tasks"])
    couriers = list(problem["all_couriers"])
    if not tasks or not couriers:
        return []

    source = 0
    task_offset = 1
    courier_offset = task_offset + len(tasks)
    sink = courier_offset + len(couriers)
    graph = [[] for _ in range(sink + 1)]

    def add_edge(left: int, right: int, cap: int, cost: int) -> None:
        graph[left].append([right, cap, cost, len(graph[right])])
        graph[right].append([left, 0, -cost, len(graph[left]) - 1])

    for i, task in enumerate(tasks):
        add_edge(source, task_offset + i, 1, 0)
        key = problem["single_key_by_task"].get(task)
        if key is None:
            continue
        for j, courier in enumerate(couriers):
            if courier not in problem["by_key"][key]:
                continue
            score, willingness = problem["by_key"][key][courier]
            cost = int(round(_singleton_penalty(1, score, willingness) * 1000))
            add_edge(task_offset + i, courier_offset + j, 1, cost)

    for j, _ in enumerate(couriers):
        add_edge(courier_offset + j, sink, 1, 0)

    potential = [0] * len(graph)
    flow = 0
    while flow < len(tasks):
        dist = [10**18] * len(graph)
        parent: List[Optional[Tuple[int, int]]] = [None] * len(graph)
        dist[source] = 0
        heap = [(0, source)]
        while heap:
            current, node = heappop(heap)
            if current != dist[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                to_node, cap, cost, _ = edge
                if cap <= 0:
                    continue
                next_dist = current + cost + potential[node] - potential[to_node]
                if next_dist < dist[to_node]:
                    dist[to_node] = next_dist
                    parent[to_node] = (node, edge_index)
                    heappush(heap, (next_dist, to_node))
        if parent[sink] is None:
            break
        for node, value in enumerate(dist):
            if value < 10**18:
                potential[node] += value
        node = sink
        while node != source:
            prev, edge_index = parent[node]
            edge = graph[prev][edge_index]
            reverse = edge[3]
            edge[1] -= 1
            graph[node][reverse][1] += 1
            node = prev
        flow += 1

    answer = []
    for i, task in enumerate(tasks):
        key = problem["single_key_by_task"].get(task)
        if key is None:
            continue
        task_node = task_offset + i
        for edge in graph[task_node]:
            to_node, cap, _, _ = edge
            if courier_offset <= to_node < sink and cap == 0:
                courier = couriers[to_node - courier_offset]
                answer.append((key, [courier]))
                break
    return answer


def _beam_construct(problem: Problem) -> Answer:
    task_to_bit = {task: 1 << index for index, task in enumerate(problem["all_tasks"])}
    full_mask = 0
    for bit in task_to_bit.values():
        full_mask |= bit

    ranked = _ranked_group_candidates(problem, _bundle_group_rank, _expected_courier_rank)
    ranked = ranked[:BEAM_GROUP_LIMIT]
    candidates = []
    for key, courier in ranked:
        mask = 0
        for task in problem["key_tasks"][key]:
            mask |= task_to_bit[task]
        penalty = _group_penalty(problem, key, [courier])
        candidates.append((key, courier, mask, penalty))

    beam = [(0, 0.0, tuple(), [])]
    for key, courier, group_mask, penalty in candidates:
        expanded = list(beam)
        for mask, total_penalty, used_tuple, selected in beam:
            if mask & group_mask or courier in used_tuple:
                continue
            next_used = tuple(sorted(used_tuple + (courier,)))
            next_selected = selected + [(key, [courier])]
            expanded.append((mask | group_mask, total_penalty + penalty, next_used, next_selected))
        expanded.sort(key=lambda state: _beam_state_rank(problem, full_mask, state))
        beam = expanded[:BEAM_WIDTH]
        if beam and beam[0][0] == full_mask:
            break

    best = min(beam, key=lambda state: _beam_state_rank(problem, full_mask, state))
    return [(key, list(couriers)) for key, couriers in best[3]]


def _beam_state_rank(problem: Problem, full_mask: int, state: Tuple[int, float, Tuple[str, ...], Answer]) -> Tuple[Any, ...]:
    mask, total_penalty, _, selected = state
    missing = _popcount(full_mask & ~mask)
    covered = len(problem["all_tasks"]) - missing
    objective = total_penalty + FALLBACK_PER_TASK * missing
    return (missing, objective, -covered, len(selected))


def _local_replacement_search(
    problem: Problem,
    answer: Answer,
    max_rounds: int,
    prefer_bundles: bool,
) -> Answer:
    current = [(key, list(couriers)) for key, couriers in answer]
    current = _repair_copy(problem, current)
    current_obj = _objective(problem, current)

    for _ in range(max_rounds):
        best_answer = None
        best_obj = current_obj
        for key in problem["by_key"]:
            if any(key == active_key for active_key, _ in current):
                continue
            tasks = set(problem["key_tasks"][key])
            if prefer_bundles and len(tasks) == 1:
                continue
            for courier in _top_couriers(problem, key, limit=3):
                trial = []
                removed_indices = set()
                for index, (active_key, active_couriers) in enumerate(current):
                    active_tasks = set(problem["key_tasks"][active_key])
                    if active_tasks & tasks or courier in active_couriers:
                        removed_indices.add(index)
                    else:
                        trial.append((active_key, list(active_couriers)))
                used_tasks, used_couriers = _used_sets(problem, trial)
                if tasks & used_tasks or courier in used_couriers:
                    continue
                trial.append((key, [courier]))
                trial = _repair_copy(problem, trial)
                obj = _objective(problem, trial)
                if obj + EPS < best_obj:
                    best_obj = obj
                    best_answer = trial
                elif removed_indices and len(tasks) > 1 and obj <= best_obj + EPS and prefer_bundles:
                    best_obj = obj
                    best_answer = trial
        if best_answer is None:
            break
        current = best_answer
        current_obj = best_obj
    return current


def _repair_copy(problem: Problem, answer: Answer) -> Answer:
    repaired = [(key, list(couriers)) for key, couriers in answer]
    used_tasks, used_couriers = _used_sets(problem, repaired)
    _repair_missing(problem, repaired, used_tasks, used_couriers, _expected_courier_rank)
    _add_improving_extras(problem, repaired, max_passes=2)
    return repaired


def _top_couriers(problem: Problem, key: str, limit: int) -> List[str]:
    items = [(_expected_courier_rank(problem, key, courier), courier) for courier in problem["by_key"][key]]
    items.sort(key=lambda item: item[0])
    return [courier for _, courier in items[:limit]]


def _objective(problem: Problem, answer: Answer) -> float:
    used_tasks, _ = _used_sets(problem, answer)
    total = 0.0
    for key, couriers in answer:
        if couriers:
            total += _group_penalty(problem, key, couriers)
    missing = len(set(problem["all_tasks"]) - used_tasks)
    return total + FALLBACK_PER_TASK * missing


def _used_sets(problem: Problem, answer: Answer) -> Tuple[set, set]:
    used_tasks = set()
    used_couriers = set()
    for key, couriers in answer:
        used_tasks.update(problem["key_tasks"].get(key, ()))
        used_couriers.update(couriers)
    return used_tasks, used_couriers


def _format_answer(problem: Problem, answer: Answer) -> list:
    normalized = []
    used_tasks = set()
    used_couriers = set()
    for key, couriers in answer:
        if key not in problem["key_tasks"]:
            continue
        tasks = problem["key_tasks"][key]
        if any(task in used_tasks for task in tasks):
            continue
        unique_couriers = []
        for courier in couriers:
            if courier in used_couriers or courier not in problem["by_key"][key]:
                continue
            unique_couriers.append(courier)
            used_couriers.add(courier)
        if not unique_couriers:
            continue
        unique_couriers.sort(key=lambda item: _expected_courier_rank(problem, key, item))
        normalized.append((key, unique_couriers))
        used_tasks.update(tasks)
    normalized.sort(key=lambda item: item[0])
    return normalized


def _popcount(value: int) -> int:
    return value.bit_count() if hasattr(value, "bit_count") else bin(value).count("1")

