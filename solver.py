BIG = 10_000.0
TASK_FALLBACK_PER_TASK = 100.0
EPS = 1e-12

LOCAL_POLISH_PASSES = 8
PAIR_REPLACEMENT_ROUNDS = 5
PAIR_REPLACEMENT_CANDIDATE_LIMIT = 120
PAIR_REPLACEMENT_SHORTLIST = 20
HASH_START_SEEDS = (289, 245, 173, 95, 242)
THREE_CYCLE_MOVE_LIMIT = 5
THREE_CYCLE_MAX_ACTIVE_GROUPS = 45


def popcount(mask: int) -> int:
    return bin(mask).count("1")


class Candidate:
    __slots__ = ("courier", "score", "willingness", "singleton_penalty")

    def __init__(
        self,
        courier: int,
        score: float,
        willingness: float,
        singleton_penalty: float,
    ) -> None:
        self.courier = courier
        self.score = score
        self.willingness = willingness
        self.singleton_penalty = singleton_penalty


class ProblemData:
    def __init__(self) -> None:
        self.group_names = []
        self.group_masks = []
        self.group_task_counts = []
        self.group_fallbacks = []
        self.group_candidates = []
        self.cand_by_group = []
        self.groups_by_courier = []
        self.courier_names = []
        self.total_tasks = 0
        self.task_full_mask = 0


class State:
    def __init__(self, data: ProblemData) -> None:
        n_groups = len(data.group_names)
        n_couriers = len(data.courier_names)
        self.data = data
        self.active = set()
        self.assigned = [set() for _ in range(n_groups)]
        self.owner = [-1] * n_couriers

        self.sum_w = [0.0] * n_groups
        self.sum_ws = [0.0] * n_groups
        self.reject_prod = [1.0] * n_groups
        self.zero_rejects = [0] * n_groups
        self.group_penalty = [0.0] * n_groups

        self.task_mask = 0
        self.covered_count = 0
        self.total_penalty = 0.0

    @property
    def energy(self) -> float:
        return self.total_penalty + BIG * (self.data.total_tasks - self.covered_count)


def solve(input_text: str) -> list:
    data = parse_input(input_text)
    if not data.group_names:
        return []

    state = construct_task_first_greedy_solution(data)
    return format_solution(state)


def parse_input(input_text: str) -> ProblemData:
    raw_lines = input_text.strip().splitlines()
    if not raw_lines:
        return ProblemData()

    start = 1 if raw_lines[0].startswith("task_id_list") else 0

    task_to_id = {}
    courier_to_id = {}
    grouped = {}

    for line in raw_lines[start:]:
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue

        group_name = parts[0].strip()
        courier_name = parts[1].strip()
        try:
            score = float(parts[2])
            willingness = float(parts[3])
        except ValueError:
            continue

        if not group_name or not courier_name:
            continue

        if courier_name not in courier_to_id:
            courier_to_id[courier_name] = len(courier_to_id)
        courier = courier_to_id[courier_name]

        for task_name in group_name.split(","):
            task_name = task_name.strip()
            if task_name and task_name not in task_to_id:
                task_to_id[task_name] = len(task_to_id)

        grouped.setdefault(group_name, []).append((courier, score, willingness))

    data = ProblemData()
    data.total_tasks = len(task_to_id)
    data.task_full_mask = (1 << data.total_tasks) - 1
    data.courier_names = [""] * len(courier_to_id)
    for name, idx in courier_to_id.items():
        data.courier_names[idx] = name
    data.groups_by_courier = [[] for _ in data.courier_names]

    for group_name, rows in grouped.items():
        mask = 0
        for task_name in group_name.split(","):
            task_name = task_name.strip()
            if task_name:
                mask |= 1 << task_to_id[task_name]

        task_count = popcount(mask)
        fallback = TASK_FALLBACK_PER_TASK * task_count
        best_by_courier = {}
        for courier, score, willingness in rows:
            old = best_by_courier.get(courier)
            new_penalty = singleton_penalty(fallback, score, willingness)
            if old is None or new_penalty < singleton_penalty(fallback, old[0], old[1]):
                best_by_courier[courier] = (score, willingness)

        group_id = len(data.group_names)
        candidates = [
            Candidate(
                courier=courier,
                score=score,
                willingness=willingness,
                singleton_penalty=singleton_penalty(fallback, score, willingness),
            )
            for courier, (score, willingness) in best_by_courier.items()
        ]
        candidates.sort(key=lambda x: (x.singleton_penalty, x.score, -x.willingness))

        data.group_names.append(group_name)
        data.group_masks.append(mask)
        data.group_task_counts.append(task_count)
        data.group_fallbacks.append(fallback)
        data.group_candidates.append(candidates)
        data.cand_by_group.append({c.courier: (c.score, c.willingness) for c in candidates})
        for cand in candidates:
            data.groups_by_courier[cand.courier].append(group_id)

    return data


def construct_task_first_greedy_solution(data: ProblemData) -> State:
    groups = choose_cover_groups(data)
    best = None

    for state in initial_single_task_states(data, groups):
        if better_state(state, best):
            best = state

    improve_by_pair_group_replacements(best)
    polish_three_courier_cycles(best)
    return best


def initial_single_task_states(data: ProblemData, groups: list):
    state = State(data)
    seed_groups_by_regret(state, groups)
    allocate_remaining_couriers_by_gain(state)
    polish_courier_assignment(state)
    yield state

    orders = [
        sorted(groups, key=lambda g: data.group_names[g], reverse=True),
        sorted(groups, key=lambda g: data.group_candidates[g][0].singleton_penalty if data.group_candidates[g] else BIG),
        sorted(groups, key=lambda g: -data.group_candidates[g][0].willingness if data.group_candidates[g] else 0.0),
    ]
    for order in orders:
        state = construct_ordered_seed_state(data, order)
        yield state

    for seed in HASH_START_SEEDS:
        order = sorted(groups, key=lambda g: hash_start_key(data, g, seed))
        state = construct_ordered_seed_state(data, order)
        yield state


def construct_ordered_seed_state(data: ProblemData, ordered_groups: list) -> State:
    state = State(data)
    for group in ordered_groups:
        courier = best_available_seed_courier(state, group)
        if courier is not None:
            add_courier(state, group, courier)
    allocate_remaining_couriers_by_gain(state)
    polish_courier_assignment(state)
    return state


def better_state(a: State, b) -> bool:
    if b is None:
        return True
    if a.covered_count != b.covered_count:
        return a.covered_count > b.covered_count
    return a.total_penalty + EPS < b.total_penalty


def hash_start_key(data: ProblemData, group: int, seed: int) -> int:
    value = group_order_number(data, group) + seed * 0x9E3779B1
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value


def group_order_number(data: ProblemData, group: int) -> int:
    name = data.group_names[group].split(",")[0]
    if len(name) > 1 and name[0] == "T" and name[1:].isdigit():
        return int(name[1:])
    return group


def choose_cover_groups(data: ProblemData) -> list:
    selected = []
    covered_mask = 0

    for group in range(len(data.group_names)):
        if data.group_task_counts[group] == 1:
            selected.append(group)
            covered_mask |= data.group_masks[group]

    if covered_mask == data.task_full_mask:
        return selected

    selected_set = set(selected)
    while covered_mask != data.task_full_mask:
        missing_mask = data.task_full_mask & ~covered_mask
        best_group = -1
        best_key = None

        for group in range(len(data.group_names)):
            if group in selected_set:
                continue
            group_mask = data.group_masks[group]
            if group_mask & covered_mask:
                continue
            newly_covered = popcount(group_mask & missing_mask)
            if newly_covered <= 0:
                continue
            seed_penalty = best_seed_penalty(data, group)
            key = (
                seed_penalty / newly_covered,
                -newly_covered,
                data.group_names[group],
            )
            if best_key is None or key < best_key:
                best_key = key
                best_group = group

        if best_group == -1:
            break

        selected.append(best_group)
        selected_set.add(best_group)
        covered_mask |= data.group_masks[best_group]

    return selected


def seed_groups_by_regret(state: State, groups: list) -> None:
    remaining = set(groups)

    while remaining:
        best_group = -1
        best_key = None

        for group in remaining:
            penalties = available_seed_penalties(state, group)
            if not penalties:
                continue
            regret = penalties[1] - penalties[0] if len(penalties) > 1 else BIG
            key = (-regret, -penalties[0], state.data.group_names[group])
            if best_key is None or key < best_key:
                best_key = key
                best_group = group

        if best_group == -1:
            break

        courier = best_available_seed_courier(state, best_group)
        if courier is not None:
            add_courier(state, best_group, courier)
        remaining.remove(best_group)


def available_seed_penalties(state: State, group: int) -> list:
    penalties = [
        cand.singleton_penalty
        for cand in state.data.group_candidates[group]
        if state.owner[cand.courier] == -1
    ]
    penalties.sort()
    return penalties


def best_available_seed_courier(state: State, group: int):
    for cand in state.data.group_candidates[group]:
        if state.owner[cand.courier] == -1:
            return cand.courier
    return None


def best_seed_penalty(data: ProblemData, group: int) -> float:
    if not data.group_candidates[group]:
        return BIG
    return data.group_candidates[group][0].singleton_penalty


def allocate_remaining_couriers_by_gain(state: State) -> int:
    moves = 0
    while True:
        best_group = -1
        best_courier = -1
        best_delta = -EPS

        for group in sorted(state.active, key=lambda g: state.data.group_names[g]):
            for cand in state.data.group_candidates[group]:
                courier = cand.courier
                if state.owner[courier] != -1:
                    continue
                delta = penalty_after_add(state, group, courier) - state.group_penalty[group]
                if delta < best_delta:
                    best_delta = delta
                    best_group = group
                    best_courier = courier

        if best_group == -1:
            return moves

        add_courier(state, best_group, best_courier)
        moves += 1


def polish_courier_assignment(state: State) -> None:
    for _ in range(LOCAL_POLISH_PASSES):
        moves = 0
        moves += relocate_couriers_by_gain(state)
        moves += swap_couriers_by_gain(state)
        moves += allocate_remaining_couriers_by_gain(state)
        if moves == 0:
            break


def relocate_couriers_by_gain(state: State) -> int:
    moves = 0
    while True:
        best_from = -1
        best_to = -1
        best_courier = -1
        best_delta = -EPS

        active = sorted(state.active, key=lambda g: state.data.group_names[g])
        for from_group in active:
            if len(state.assigned[from_group]) <= 1:
                continue

            for courier in sorted(state.assigned[from_group]):
                remove_delta = (
                    penalty_after_remove(state, from_group, courier)
                    - state.group_penalty[from_group]
                )

                for to_group in active:
                    if to_group == from_group:
                        continue
                    if courier in state.assigned[to_group]:
                        continue
                    if courier not in state.data.cand_by_group[to_group]:
                        continue

                    add_delta = (
                        penalty_after_add(state, to_group, courier)
                        - state.group_penalty[to_group]
                    )
                    delta = remove_delta + add_delta
                    if delta < best_delta:
                        best_delta = delta
                        best_from = from_group
                        best_to = to_group
                        best_courier = courier

        if best_courier == -1:
            return moves

        remove_courier(state, best_from, best_courier)
        add_courier(state, best_to, best_courier)
        moves += 1


def swap_couriers_by_gain(state: State) -> int:
    moves = 0
    while True:
        best_a = -1
        best_b = -1
        best_ca = -1
        best_cb = -1
        best_delta = -EPS

        active = sorted(state.active, key=lambda g: state.data.group_names[g])
        for i, group_a in enumerate(active):
            for group_b in active[i + 1 :]:
                old_penalty = state.group_penalty[group_a] + state.group_penalty[group_b]

                for courier_a in sorted(state.assigned[group_a]):
                    if courier_a not in state.data.cand_by_group[group_b]:
                        continue
                    for courier_b in sorted(state.assigned[group_b]):
                        if courier_b not in state.data.cand_by_group[group_a]:
                            continue

                        new_a = (state.assigned[group_a] - {courier_a}) | {courier_b}
                        new_b = (state.assigned[group_b] - {courier_b}) | {courier_a}
                        new_penalty = penalty_for_courier_set(
                            state.data,
                            group_a,
                            new_a,
                        ) + penalty_for_courier_set(state.data, group_b, new_b)
                        delta = new_penalty - old_penalty
                        if delta < best_delta:
                            best_delta = delta
                            best_a = group_a
                            best_b = group_b
                            best_ca = courier_a
                            best_cb = courier_b

        if best_ca == -1:
            return moves

        remove_courier(state, best_a, best_ca)
        remove_courier(state, best_b, best_cb)
        add_courier(state, best_a, best_cb)
        add_courier(state, best_b, best_ca)
        moves += 1


def improve_by_pair_group_replacements(state: State) -> None:
    data = state.data
    single_group_by_mask = {
        data.group_masks[group]: group
        for group in range(len(data.group_names))
        if data.group_task_counts[group] == 1
    }
    pair_groups = [
        group
        for group in range(len(data.group_names))
        if data.group_task_counts[group] == 2 and data.group_candidates[group]
    ]
    pair_groups.sort(key=lambda g: (data.group_candidates[g][0].singleton_penalty, data.group_names[g]))
    pair_groups = pair_groups[:PAIR_REPLACEMENT_CANDIDATE_LIMIT]

    for _ in range(PAIR_REPLACEMENT_ROUNDS):
        approximate = []
        for pair_group in pair_groups:
            if pair_group in state.active:
                continue
            candidate = pair_replacement_state(
                state,
                pair_group,
                single_group_by_mask,
                do_polish=False,
            )
            if candidate is not None:
                approximate.append((candidate.total_penalty - state.total_penalty, pair_group))

        if not approximate:
            break

        approximate.sort()
        best = None
        best_delta = -EPS
        for _, pair_group in approximate[:PAIR_REPLACEMENT_SHORTLIST]:
            candidate = pair_replacement_state(
                state,
                pair_group,
                single_group_by_mask,
                do_polish=True,
            )
            if candidate is None:
                continue
            delta = candidate.total_penalty - state.total_penalty
            if delta < best_delta:
                best_delta = delta
                best = candidate

        if best is None:
            break

        restore_state(state, best)


def pair_replacement_state(
    state: State,
    pair_group: int,
    single_group_by_mask: dict,
    do_polish: bool,
):
    groups_to_remove = active_single_groups_for_pair(
        state,
        pair_group,
        single_group_by_mask,
    )
    if groups_to_remove is None:
        return None

    candidate = clone_state(state)
    for group in groups_to_remove:
        remove_group(candidate, group)

    courier = best_available_seed_courier(candidate, pair_group)
    if courier is None:
        return None

    add_courier(candidate, pair_group, courier)
    allocate_remaining_couriers_by_gain(candidate)
    if do_polish:
        polish_courier_assignment(candidate)
    return candidate


def active_single_groups_for_pair(
    state: State,
    pair_group: int,
    single_group_by_mask: dict,
):
    data = state.data
    groups = []
    bit = 1
    pair_mask = data.group_masks[pair_group]

    while bit <= data.task_full_mask:
        if pair_mask & bit:
            group = single_group_by_mask.get(bit)
            if group is None or group not in state.active:
                return None
            groups.append(group)
        bit <<= 1

    if len(groups) != 2:
        return None
    return groups


def clone_state(state: State) -> State:
    clone = State(state.data)
    for group in sorted(state.active):
        for courier in sorted(state.assigned[group]):
            add_courier(clone, group, courier)
    return clone


def restore_state(target: State, source: State) -> None:
    target.active = set(source.active)
    target.assigned = [couriers.copy() for couriers in source.assigned]
    target.owner = source.owner[:]
    target.sum_w = source.sum_w[:]
    target.sum_ws = source.sum_ws[:]
    target.reject_prod = source.reject_prod[:]
    target.zero_rejects = source.zero_rejects[:]
    target.group_penalty = source.group_penalty[:]
    target.task_mask = source.task_mask
    target.covered_count = source.covered_count
    target.total_penalty = source.total_penalty


def remove_group(state: State, group: int) -> None:
    for courier in list(state.assigned[group]):
        remove_courier(state, group, courier)


def polish_three_courier_cycles(state: State) -> int:
    if len(state.active) > THREE_CYCLE_MAX_ACTIVE_GROUPS:
        return 0

    moves = 0
    for _ in range(THREE_CYCLE_MOVE_LIMIT):
        move = best_three_courier_cycle(state)
        if move is None:
            return moves
        apply_three_courier_cycle(state, move)
        polish_courier_assignment(state)
        moves += 1
    return moves


def best_three_courier_cycle(state: State):
    data = state.data
    active = sorted(state.active, key=lambda g: data.group_names[g])
    best_delta = -EPS
    best_move = None

    for i, group_a in enumerate(active):
        for j in range(i + 1, len(active)):
            group_b = active[j]
            for group_c in active[j + 1 :]:
                old_penalty = (
                    state.group_penalty[group_a]
                    + state.group_penalty[group_b]
                    + state.group_penalty[group_c]
                )

                for courier_a in sorted(state.assigned[group_a]):
                    for courier_b in sorted(state.assigned[group_b]):
                        for courier_c in sorted(state.assigned[group_c]):
                            if (
                                courier_a in data.cand_by_group[group_b]
                                and courier_b in data.cand_by_group[group_c]
                                and courier_c in data.cand_by_group[group_a]
                            ):
                                delta = three_cycle_delta(
                                    state,
                                    group_a,
                                    group_b,
                                    group_c,
                                    courier_a,
                                    courier_b,
                                    courier_c,
                                    old_penalty,
                                    0,
                                )
                                if delta < best_delta:
                                    best_delta = delta
                                    best_move = (
                                        group_a,
                                        group_b,
                                        group_c,
                                        courier_a,
                                        courier_b,
                                        courier_c,
                                        0,
                                    )

                            if (
                                courier_a in data.cand_by_group[group_c]
                                and courier_c in data.cand_by_group[group_b]
                                and courier_b in data.cand_by_group[group_a]
                            ):
                                delta = three_cycle_delta(
                                    state,
                                    group_a,
                                    group_b,
                                    group_c,
                                    courier_a,
                                    courier_b,
                                    courier_c,
                                    old_penalty,
                                    1,
                                )
                                if delta < best_delta:
                                    best_delta = delta
                                    best_move = (
                                        group_a,
                                        group_b,
                                        group_c,
                                        courier_a,
                                        courier_b,
                                        courier_c,
                                        1,
                                    )

    return best_move


def three_cycle_delta(
    state: State,
    group_a: int,
    group_b: int,
    group_c: int,
    courier_a: int,
    courier_b: int,
    courier_c: int,
    old_penalty: float,
    mode: int,
) -> float:
    if mode == 0:
        new_a = (state.assigned[group_a] - {courier_a}) | {courier_c}
        new_b = (state.assigned[group_b] - {courier_b}) | {courier_a}
        new_c = (state.assigned[group_c] - {courier_c}) | {courier_b}
    else:
        new_a = (state.assigned[group_a] - {courier_a}) | {courier_b}
        new_b = (state.assigned[group_b] - {courier_b}) | {courier_c}
        new_c = (state.assigned[group_c] - {courier_c}) | {courier_a}

    new_penalty = (
        penalty_for_courier_set(state.data, group_a, new_a)
        + penalty_for_courier_set(state.data, group_b, new_b)
        + penalty_for_courier_set(state.data, group_c, new_c)
    )
    return new_penalty - old_penalty


def apply_three_courier_cycle(state: State, move) -> None:
    group_a, group_b, group_c, courier_a, courier_b, courier_c, mode = move

    remove_courier(state, group_a, courier_a)
    remove_courier(state, group_b, courier_b)
    remove_courier(state, group_c, courier_c)

    if mode == 0:
        add_courier(state, group_a, courier_c)
        add_courier(state, group_b, courier_a)
        add_courier(state, group_c, courier_b)
    else:
        add_courier(state, group_a, courier_b)
        add_courier(state, group_b, courier_c)
        add_courier(state, group_c, courier_a)


def singleton_penalty(fallback: float, score: float, willingness: float) -> float:
    return (1.0 - willingness) * fallback + willingness * score


def penalty_from_stats(
    fallback: float,
    sum_w: float,
    sum_ws: float,
    reject_prod: float,
    zero_rejects: int,
) -> float:
    if sum_w <= EPS:
        return fallback
    reject_prob = 0.0 if zero_rejects > 0 else reject_prod
    accepted_score = sum_ws / sum_w
    return reject_prob * fallback + (1.0 - reject_prob) * accepted_score


def group_penalty_from_stats(
    data: ProblemData,
    group: int,
    sum_w: float,
    sum_ws: float,
    reject_prod: float,
    zero_rejects: int,
) -> float:
    return penalty_from_stats(
        data.group_fallbacks[group],
        sum_w,
        sum_ws,
        reject_prod,
        zero_rejects,
    )


def penalty_for_courier_set(data: ProblemData, group: int, couriers) -> float:
    sum_w = 0.0
    sum_ws = 0.0
    reject_prod = 1.0
    zero_rejects = 0

    for courier in couriers:
        score, willingness = data.cand_by_group[group][courier]
        sum_w += willingness
        sum_ws += willingness * score
        reject_factor = 1.0 - willingness
        if reject_factor <= EPS:
            zero_rejects += 1
        else:
            reject_prod *= reject_factor

    return group_penalty_from_stats(data, group, sum_w, sum_ws, reject_prod, zero_rejects)


def penalty_after_add(state: State, group: int, courier: int) -> float:
    score, willingness = state.data.cand_by_group[group][courier]
    sum_w = state.sum_w[group] + willingness
    sum_ws = state.sum_ws[group] + willingness * score
    zero_rejects = state.zero_rejects[group]
    reject_prod = state.reject_prod[group]
    reject_factor = 1.0 - willingness
    if reject_factor <= EPS:
        zero_rejects += 1
    else:
        reject_prod *= reject_factor
    return group_penalty_from_stats(
        state.data,
        group,
        sum_w,
        sum_ws,
        reject_prod,
        zero_rejects,
    )


def penalty_after_remove(state: State, group: int, courier: int) -> float:
    score, willingness = state.data.cand_by_group[group][courier]
    sum_w = state.sum_w[group] - willingness
    sum_ws = state.sum_ws[group] - willingness * score
    zero_rejects = state.zero_rejects[group]
    reject_prod = state.reject_prod[group]
    reject_factor = 1.0 - willingness
    if reject_factor <= EPS:
        zero_rejects -= 1
    else:
        reject_prod /= reject_factor
    return group_penalty_from_stats(
        state.data,
        group,
        sum_w,
        sum_ws,
        reject_prod,
        zero_rejects,
    )


def add_courier(state: State, group: int, courier: int) -> None:
    data = state.data
    if group not in state.active:
        state.active.add(group)
        state.task_mask |= data.group_masks[group]
        state.covered_count = popcount(state.task_mask)

    old_penalty = state.group_penalty[group]
    score, willingness = data.cand_by_group[group][courier]
    state.assigned[group].add(courier)
    state.owner[courier] = group
    state.sum_w[group] += willingness
    state.sum_ws[group] += willingness * score
    reject_factor = 1.0 - willingness
    if reject_factor <= EPS:
        state.zero_rejects[group] += 1
    else:
        state.reject_prod[group] *= reject_factor

    new_penalty = group_penalty_from_stats(
        data,
        group,
        state.sum_w[group],
        state.sum_ws[group],
        state.reject_prod[group],
        state.zero_rejects[group],
    )
    state.group_penalty[group] = new_penalty
    state.total_penalty += new_penalty - old_penalty


def remove_courier(state: State, group: int, courier: int) -> None:
    data = state.data
    old_penalty = state.group_penalty[group]
    score, willingness = data.cand_by_group[group][courier]
    state.assigned[group].remove(courier)
    state.owner[courier] = -1
    state.sum_w[group] -= willingness
    state.sum_ws[group] -= willingness * score
    reject_factor = 1.0 - willingness
    if reject_factor <= EPS:
        state.zero_rejects[group] -= 1
    else:
        state.reject_prod[group] /= reject_factor

    if not state.assigned[group]:
        state.active.remove(group)
        state.task_mask &= ~data.group_masks[group]
        state.covered_count = popcount(state.task_mask)
        state.sum_w[group] = 0.0
        state.sum_ws[group] = 0.0
        state.reject_prod[group] = 1.0
        state.zero_rejects[group] = 0
        state.group_penalty[group] = 0.0
        state.total_penalty -= old_penalty
        return

    new_penalty = group_penalty_from_stats(
        data,
        group,
        state.sum_w[group],
        state.sum_ws[group],
        state.reject_prod[group],
        state.zero_rejects[group],
    )
    state.group_penalty[group] = new_penalty
    state.total_penalty += new_penalty - old_penalty


def format_solution(state: State) -> list:
    result = []
    for group in sorted(state.active, key=lambda g: state.data.group_names[g]):
        couriers = sorted(
            (state.data.courier_names[c] for c in state.assigned[group]),
            key=lambda name: name,
        )
        if couriers:
            result.append((state.data.group_names[group], couriers))
    return result
