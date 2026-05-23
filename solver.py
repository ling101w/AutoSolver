import math
import random
import time


LOCAL_STALL_LIMIT = 2000
DEEPOPT_STALL_LIMIT = 8000
RESTART_STALL_LIMIT = 20000

BIG = 1_000_000.0
TIME_LIMIT_SECONDS = 8.75
EPS = 1e-12


def popcount(mask: int) -> int:
    return bin(mask).count("1")


class Candidate:
    __slots__ = ("courier", "score", "willingness", "singleton_penalty")

    def __init__(
        self, courier: int, score: float, willingness: float, singleton_penalty: float
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
        self.conflict_groups = []
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

    def copy(self) -> "State":
        other = State(self.data)
        other.active = set(self.active)
        other.assigned = [s.copy() for s in self.assigned]
        other.owner = self.owner[:]
        other.sum_w = self.sum_w[:]
        other.sum_ws = self.sum_ws[:]
        other.reject_prod = self.reject_prod[:]
        other.zero_rejects = self.zero_rejects[:]
        other.group_penalty = self.group_penalty[:]
        other.task_mask = self.task_mask
        other.covered_count = self.covered_count
        other.total_penalty = self.total_penalty
        return other

    @property
    def energy(self) -> float:
        return self.total_penalty + BIG * (self.data.total_tasks - self.covered_count)


class TabuTable:
    def __init__(self) -> None:
        self.courier_group = {}
        self.group_add = {}
        self.group_remove = {}

    def clear(self) -> None:
        self.courier_group.clear()
        self.group_add.clear()
        self.group_remove.clear()


class ConfChange:
    def __init__(self, data: ProblemData) -> None:
        self.data = data
        self.group = [True] * len(data.group_names)
        self.courier = [True] * len(data.courier_names)

    def reset(self) -> None:
        self.group = [True] * len(self.data.group_names)
        self.courier = [True] * len(self.data.courier_names)

    def touch(self, groups, couriers) -> None:
        for g in groups:
            self.group[g] = False
            for ng in self.data.conflict_groups[g]:
                self.group[ng] = True
        for c in couriers:
            self.courier[c] = False
            for g in self.data.groups_by_courier[c]:
                self.group[g] = True


class MoveBandit:
    def __init__(self, arms) -> None:
        self.arms = arms
        self.counts = [0] * len(arms)
        self.values = [0.0] * len(arms)
        self.total = 0

    def select(self, rng: random.Random) -> str:
        for i, cnt in enumerate(self.counts):
            if cnt == 0:
                return self.arms[i]

        self.total += 1
        log_total = math.log(max(2, self.total))
        c = 0.65
        best_i = 0
        best_score = -1.0
        for i, value in enumerate(self.values):
            score = value + c * math.sqrt(log_total / self.counts[i])
            if score > best_score:
                best_score = score
                best_i = i
        if rng.random() < 0.08:
            return rng.choice(self.arms)
        return self.arms[best_i]

    def update(self, arm: str, reward: float) -> None:
        i = self.arms.index(arm)
        self.counts[i] += 1
        reward = min(10_000.0, max(0.0, reward))
        self.values[i] += (reward - self.values[i]) / self.counts[i]

    def soft_reset(self) -> None:
        self.counts = [max(0, c // 2) for c in self.counts]
        self.values = [v * 0.5 for v in self.values]
        self.total = sum(self.counts)


def solve(input_text: str) -> list:
    data = parse_input(input_text)
    if not data.group_names:
        return []

    deadline = time.perf_counter() + TIME_LIMIT_SECONDS
    rng = random.Random(301_2026)

    restart_id = 0
    current = construct_initial_solution(data, rng, restart_id)
    greedy_warmup(current, rng, move_limit=260, deadline=deadline)
    best = current.copy()

    tabu = TabuTable()
    confchange = ConfChange(data)
    bandit = MoveBandit(
        ["add", "remove", "replace_courier", "activate", "replace_group"]
    )

    iter_id = 0
    last_improve_iter = 0
    last_best_iter = 0
    last_deepopt_iter = 0

    while time.perf_counter() < deadline:
        iter_id += 1
        arm = bandit.select(rng)
        move = generate_move(current, best, arm, tabu, confchange, iter_id, rng)

        if move is not None:
            old_energy = current.energy
            old_best_key = state_key(best)
            apply_move(current, move)
            update_tabu(tabu, move, iter_id, rng)
            update_confchange(confchange, move)

            reward = max(0.0, old_energy - current.energy)
            if state_key(current) < old_best_key:
                reward += 5000.0
            bandit.update(arm, reward)

            if current.energy + EPS < old_energy:
                last_improve_iter = iter_id

            if better(current, best):
                best = current.copy()
                last_best_iter = iter_id
                last_deepopt_iter = iter_id
        else:
            bandit.update(arm, 0.0)

        if iter_id - last_improve_iter >= LOCAL_STALL_LIMIT:
            before = current.energy
            ruin_recreate(current, rng, deadline)
            last_improve_iter = iter_id
            if current.energy + EPS < before:
                last_improve_iter = iter_id
            if better(current, best):
                best = current.copy()
                last_best_iter = iter_id
                last_deepopt_iter = iter_id

        if (
            iter_id - last_best_iter >= DEEPOPT_STALL_LIMIT
            and iter_id - last_deepopt_iter >= DEEPOPT_STALL_LIMIT
            and time.perf_counter() + 0.08 < deadline
        ):
            deepopt(current, rng, deadline)
            greedy_warmup(current, rng, move_limit=90, deadline=deadline)
            last_improve_iter = iter_id
            last_deepopt_iter = iter_id
            if better(current, best):
                best = current.copy()
                last_best_iter = iter_id

        if iter_id - last_best_iter >= RESTART_STALL_LIMIT:
            restart_id += 1
            current = construct_initial_solution(data, rng, restart_id)
            greedy_warmup(current, rng, move_limit=260, deadline=deadline)
            tabu.clear()
            confchange.reset()
            bandit.soft_reset()
            last_improve_iter = iter_id
            last_best_iter = iter_id
            last_deepopt_iter = iter_id
            if better(current, best):
                best = current.copy()

    return format_solution(best)


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
        fallback = 100.0 * task_count
        best_by_courier = {}
        for courier, score, willingness in rows:
            old = best_by_courier.get(courier)
            if old is None or singleton_penalty(fallback, score, willingness) < singleton_penalty(fallback, old[0], old[1]):
                best_by_courier[courier] = (score, willingness)

        group_id = len(data.group_names)
        candidates = [
            Candidate(
                courier=c,
                score=score,
                willingness=w,
                singleton_penalty=singleton_penalty(fallback, score, w),
            )
            for c, (score, w) in best_by_courier.items()
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

    n_groups = len(data.group_names)
    data.conflict_groups = [[] for _ in range(n_groups)]
    for i in range(n_groups):
        mi = data.group_masks[i]
        for j in range(n_groups):
            if i != j and (mi & data.group_masks[j]):
                data.conflict_groups[i].append(j)

    return data


def singleton_penalty(fallback: float, score: float, willingness: float) -> float:
    return (1.0 - willingness) * fallback + willingness * score


def state_key(state):
    return (-state.covered_count, state.total_penalty)


def better(a: State, b) -> bool:
    if b is None:
        return True
    if a.covered_count != b.covered_count:
        return a.covered_count > b.covered_count
    return a.total_penalty + EPS < b.total_penalty


def group_penalty_from_stats(
    data: ProblemData,
    group: int,
    sum_w: float,
    sum_ws: float,
    reject_prod: float,
    zero_rejects: int,
) -> float:
    if sum_w <= EPS:
        return data.group_fallbacks[group]
    reject_prob = 0.0 if zero_rejects > 0 else reject_prod
    accepted_score = sum_ws / sum_w
    return reject_prob * data.group_fallbacks[group] + (1.0 - reject_prob) * accepted_score


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
        state.data, group, sum_w, sum_ws, reject_prod, zero_rejects
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
        state.data, group, sum_w, sum_ws, reject_prod, zero_rejects
    )


def penalty_after_replace(state: State, group: int, old_c: int, new_c: int) -> float:
    score_old, w_old = state.data.cand_by_group[group][old_c]
    score_new, w_new = state.data.cand_by_group[group][new_c]
    sum_w = state.sum_w[group] - w_old + w_new
    sum_ws = state.sum_ws[group] - w_old * score_old + w_new * score_new
    zero_rejects = state.zero_rejects[group]
    reject_prod = state.reject_prod[group]

    old_factor = 1.0 - w_old
    if old_factor <= EPS:
        zero_rejects -= 1
    else:
        reject_prod /= old_factor

    new_factor = 1.0 - w_new
    if new_factor <= EPS:
        zero_rejects += 1
    else:
        reject_prod *= new_factor

    return group_penalty_from_stats(
        state.data, group, sum_w, sum_ws, reject_prod, zero_rejects
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


def replace_courier(state: State, group: int, old_c: int, new_c: int) -> None:
    remove_courier(state, group, old_c)
    add_courier(state, group, new_c)


def remove_group(state: State, group: int):
    couriers = list(state.assigned[group])
    for courier in couriers:
        remove_courier(state, group, courier)
    return couriers


def construct_initial_solution(
    data: ProblemData, rng: random.Random, restart_id: int
) -> State:
    state = State(data)
    groups = list(range(len(data.group_names)))

    def group_key(g: int) -> float:
        best = data.group_candidates[g][0].singleton_penalty
        ratio = best / max(1, data.group_task_counts[g])
        noise = rng.random() * (2.0 + 0.35 * (restart_id % 7))
        return ratio - 0.02 * data.group_task_counts[g] + noise

    groups.sort(key=group_key)

    for group in groups:
        if state.task_mask & data.group_masks[group]:
            continue
        courier = choose_initial_courier(state, group, rng)
        if courier is None:
            continue
        add_courier(state, group, courier)

    return state


def choose_initial_courier(
    state: State, group: int, rng: random.Random, pool=None
):
    options = []
    for cand in state.data.group_candidates[group][:10]:
        c = cand.courier
        if state.owner[c] == -1 and (pool is None or c in pool):
            options.append(c)
    if not options:
        for cand in state.data.group_candidates[group]:
            c = cand.courier
            if state.owner[c] == -1 and (pool is None or c in pool):
                return c
        return None

    if rng.random() < 0.82 or len(options) == 1:
        return options[0]
    return rng.choice(options[: min(4, len(options))])


def greedy_warmup(
    state: State, rng: random.Random, move_limit: int, deadline: float
) -> None:
    moves = 0
    while moves < move_limit and time.perf_counter() < deadline:
        best_move = None
        best_delta = -EPS

        active_groups = list(state.active)
        rng.shuffle(active_groups)
        for group in active_groups[: min(len(active_groups), 48)]:
            move = best_add_move_for_group(state, group, limit=16)
            if move is not None and move[0] < best_delta:
                best_delta = move[0]
                best_move = ("add", group, move[1])

            move = best_replace_move_for_group(state, group, limit=14)
            if move is not None and move[0] < best_delta:
                best_delta = move[0]
                best_move = ("replace_courier", group, move[1], move[2])

        if best_move is None:
            move = best_activate_move(state, rng, sample=80)
            if move is not None:
                best_move = ("activate", move[1], move[2])

        if best_move is None:
            break

        apply_move(state, best_move)
        moves += 1


def best_add_move_for_group(
    state: State, group: int, limit: int
):
    best_delta = 0.0
    best_c = -1
    checked = 0
    for cand in state.data.group_candidates[group]:
        c = cand.courier
        if state.owner[c] != -1 or c in state.assigned[group]:
            continue
        new_penalty = penalty_after_add(state, group, c)
        delta = new_penalty - state.group_penalty[group]
        if delta < best_delta:
            best_delta = delta
            best_c = c
        checked += 1
        if checked >= limit:
            break
    if best_c == -1:
        return None
    return best_delta, best_c


def best_replace_move_for_group(
    state: State, group: int, limit: int
):
    if len(state.assigned[group]) <= 0:
        return None
    best_delta = 0.0
    best_old = -1
    best_new = -1
    free_checked = 0
    free_candidates = []
    for cand in state.data.group_candidates[group]:
        c = cand.courier
        if state.owner[c] == -1 and c not in state.assigned[group]:
            free_candidates.append(c)
            free_checked += 1
            if free_checked >= limit:
                break
    if not free_candidates:
        return None
    for old_c in list(state.assigned[group]):
        for new_c in free_candidates:
            new_penalty = penalty_after_replace(state, group, old_c, new_c)
            delta = new_penalty - state.group_penalty[group]
            if delta < best_delta:
                best_delta = delta
                best_old = old_c
                best_new = new_c
    if best_old == -1:
        return None
    return best_delta, best_old, best_new


def best_activate_move(
    state: State, rng: random.Random, sample: int
):
    data = state.data
    best_delta = 0.0
    best_group = -1
    best_courier = -1
    n = len(data.group_names)
    for _ in range(min(sample, n)):
        group = rng.randrange(n)
        if group in state.active or (state.task_mask & data.group_masks[group]):
            continue
        courier = choose_initial_courier(state, group, rng)
        if courier is None:
            continue
        new_penalty = penalty_after_add_to_empty(state, group, courier)
        delta = new_penalty - BIG * data.group_task_counts[group]
        if delta < best_delta:
            best_delta = delta
            best_group = group
            best_courier = courier
    if best_group == -1:
        return None
    return best_delta, best_group, best_courier


def penalty_after_add_to_empty(state: State, group: int, courier: int) -> float:
    score, willingness = state.data.cand_by_group[group][courier]
    reject_factor = 1.0 - willingness
    zero_rejects = 1 if reject_factor <= EPS else 0
    reject_prod = 1.0 if zero_rejects else reject_factor
    return group_penalty_from_stats(
        state.data,
        group,
        willingness,
        willingness * score,
        reject_prod,
        zero_rejects,
    )


def generate_move(
    state: State,
    best: State,
    arm: str,
    tabu: TabuTable,
    confchange: ConfChange,
    iter_id: int,
    rng: random.Random,
):
    generators = {
        "add": generate_add_move,
        "remove": generate_remove_move,
        "replace_courier": generate_replace_courier_move,
        "activate": generate_activate_move,
        "replace_group": generate_replace_group_move,
    }
    return generators[arm](state, best, tabu, confchange, iter_id, rng)


def generate_add_move(state, best, tabu, confchange, iter_id, rng):
    best_move = None
    best_delta = float("inf")
    groups = list(state.active)
    if not groups:
        return None
    for _ in range(72):
        group = rng.choice(groups)
        if not confchange.group[group] and rng.random() < 0.65:
            continue
        candidates = state.data.group_candidates[group]
        if not candidates:
            continue
        cand = candidates[rng.randrange(min(len(candidates), 24))]
        c = cand.courier
        if state.owner[c] != -1 or c in state.assigned[group]:
            continue
        delta = penalty_after_add(state, group, c) - state.group_penalty[group]
        move = ("add", group, c)
        if is_tabu(move, tabu, iter_id) and not aspiration(state, best, delta, 0):
            continue
        if delta < best_delta:
            best_delta = delta
            best_move = move
    return best_move


def generate_remove_move(state, best, tabu, confchange, iter_id, rng):
    groups = [g for g in state.active if len(state.assigned[g]) > 1]
    if not groups:
        return None
    best_move = None
    best_delta = float("inf")
    for _ in range(56):
        group = rng.choice(groups)
        if not confchange.group[group] and rng.random() < 0.65:
            continue
        c = rng.choice(tuple(state.assigned[group]))
        delta = penalty_after_remove(state, group, c) - state.group_penalty[group]
        move = ("remove", group, c)
        if is_tabu(move, tabu, iter_id) and not aspiration(state, best, delta, 0):
            continue
        if delta < best_delta:
            best_delta = delta
            best_move = move
    return best_move


def generate_replace_courier_move(state, best, tabu, confchange, iter_id, rng):
    groups = list(state.active)
    if not groups:
        return None
    best_move = None
    best_delta = float("inf")
    for _ in range(72):
        group = rng.choice(groups)
        if not state.assigned[group]:
            continue
        if not confchange.group[group] and rng.random() < 0.65:
            continue
        old_c = rng.choice(tuple(state.assigned[group]))
        candidates = state.data.group_candidates[group]
        for _ in range(3):
            cand = candidates[rng.randrange(min(len(candidates), 28))]
            new_c = cand.courier
            if state.owner[new_c] != -1 or new_c in state.assigned[group]:
                continue
            delta = penalty_after_replace(state, group, old_c, new_c) - state.group_penalty[group]
            move = ("replace_courier", group, old_c, new_c)
            if is_tabu(move, tabu, iter_id) and not aspiration(state, best, delta, 0):
                continue
            if delta < best_delta:
                best_delta = delta
                best_move = move
    return best_move


def generate_activate_move(state, best, tabu, confchange, iter_id, rng):
    data = state.data
    best_move = None
    best_delta = float("inf")
    for _ in range(88):
        group = rng.randrange(len(data.group_names))
        if group in state.active or (state.task_mask & data.group_masks[group]):
            continue
        if not confchange.group[group] and rng.random() < 0.55:
            continue
        courier = choose_initial_courier(state, group, rng)
        if courier is None:
            continue
        penalty = penalty_after_add_to_empty(state, group, courier)
        delta = penalty - BIG * data.group_task_counts[group]
        move = ("activate", group, courier)
        if is_tabu(move, tabu, iter_id) and not aspiration(
            state, best, delta, data.group_task_counts[group]
        ):
            continue
        if delta < best_delta:
            best_delta = delta
            best_move = move
    return best_move


def generate_replace_group_move(state, best, tabu, confchange, iter_id, rng):
    data = state.data
    best_move = None
    best_delta = float("inf")
    for _ in range(80):
        group = rng.randrange(len(data.group_names))
        if group in state.active:
            continue
        if not confchange.group[group] and rng.random() < 0.55:
            continue
        conflicts = [g for g in state.active if data.group_masks[g] & data.group_masks[group]]
        if not conflicts:
            continue

        freed = set()
        removed_penalty = 0.0
        removed_mask = 0
        for old_g in conflicts:
            freed.update(state.assigned[old_g])
            removed_penalty += state.group_penalty[old_g]
            removed_mask |= data.group_masks[old_g]

        courier = choose_initial_courier_after_removal(state, group, freed)
        if courier is None:
            continue
        penalty = penalty_after_add_to_empty_for_courier(data, group, courier)
        old_covered = state.covered_count
        new_mask = (state.task_mask & ~removed_mask) | data.group_masks[group]
        delta_covered = popcount(new_mask) - old_covered
        delta = penalty - removed_penalty - BIG * delta_covered
        move = ("replace_group", group, courier, tuple(conflicts))
        if is_tabu(move, tabu, iter_id) and not aspiration(state, best, delta, delta_covered):
            continue
        if delta < best_delta:
            best_delta = delta
            best_move = move
    return best_move


def choose_initial_courier_after_removal(
    state: State, group: int, freed
):
    for cand in state.data.group_candidates[group]:
        c = cand.courier
        if state.owner[c] == -1 or c in freed:
            return c
    return None


def penalty_after_add_to_empty_for_courier(
    data: ProblemData, group: int, courier: int
) -> float:
    score, willingness = data.cand_by_group[group][courier]
    reject_factor = 1.0 - willingness
    zero_rejects = 1 if reject_factor <= EPS else 0
    reject_prod = 1.0 if zero_rejects else reject_factor
    return group_penalty_from_stats(
        data,
        group,
        willingness,
        willingness * score,
        reject_prod,
        zero_rejects,
    )


def is_tabu(move, tabu: TabuTable, iter_id: int) -> bool:
    kind = move[0]
    if kind in ("add", "activate"):
        group, c = move[1], move[2]
        return (
            tabu.courier_group.get((c, group), -1) > iter_id
            or tabu.group_add.get(group, -1) > iter_id
        )
    if kind == "replace_courier":
        group, _, new_c = move[1], move[2], move[3]
        return tabu.courier_group.get((new_c, group), -1) > iter_id
    if kind == "remove":
        group = move[1]
        return tabu.group_remove.get(group, -1) > iter_id
    if kind == "replace_group":
        group = move[1]
        return tabu.group_add.get(group, -1) > iter_id
    return False


def aspiration(state: State, best: State, delta_energy: float, delta_covered: int) -> bool:
    new_covered = state.covered_count + delta_covered
    new_penalty = state.total_penalty + delta_energy + BIG * delta_covered
    if new_covered != best.covered_count:
        return new_covered > best.covered_count
    return new_penalty + EPS < best.total_penalty


def apply_move(state: State, move) -> None:
    kind = move[0]
    if kind == "add":
        _, group, c = move
        add_courier(state, group, c)
    elif kind == "remove":
        _, group, c = move
        remove_courier(state, group, c)
    elif kind == "replace_courier":
        _, group, old_c, new_c = move
        replace_courier(state, group, old_c, new_c)
    elif kind == "activate":
        _, group, c = move
        add_courier(state, group, c)
    elif kind == "replace_group":
        _, group, c, conflicts = move
        for old_g in conflicts:
            if old_g in state.active:
                remove_group(state, old_g)
        add_courier(state, group, c)


def update_tabu(tabu: TabuTable, move, iter_id: int, rng: random.Random) -> None:
    tenure = 7 + rng.randrange(11)
    kind = move[0]
    if kind == "add":
        _, group, c = move
        tabu.group_remove[group] = iter_id + tenure
    elif kind == "activate":
        _, group, c = move
        tabu.group_remove[group] = iter_id + tenure
    elif kind == "remove":
        _, group, c = move
        tabu.courier_group[(c, group)] = iter_id + tenure
    elif kind == "replace_courier":
        _, group, old_c, _ = move
        tabu.courier_group[(old_c, group)] = iter_id + tenure
    elif kind == "replace_group":
        _, group, c, conflicts = move
        tabu.group_remove[group] = iter_id + tenure
        for old_g in conflicts:
            tabu.group_add[old_g] = iter_id + tenure


def update_confchange(confchange: ConfChange, move) -> None:
    kind = move[0]
    if kind in ("add", "remove", "activate"):
        _, group, c = move
        confchange.touch([group], [c])
    elif kind == "replace_courier":
        _, group, old_c, new_c = move
        confchange.touch([group], [old_c, new_c])
    elif kind == "replace_group":
        _, group, c, conflicts = move
        confchange.touch([group] + list(conflicts), [c])


def ruin_recreate(state: State, rng: random.Random, deadline: float) -> None:
    data = state.data
    old_energy = state.energy
    snapshot = state.copy()

    if state.active:
        seed_group = rng.choice(tuple(state.active))
        region_mask = data.group_masks[seed_group]
    else:
        region_mask = 0

    task_ids = list(range(data.total_tasks))
    rng.shuffle(task_ids)
    for t in task_ids[: rng.randrange(4, min(9, max(5, data.total_tasks + 1)))]:
        region_mask |= 1 << t

    to_remove = [g for g in list(state.active) if data.group_masks[g] & region_mask]
    rng.shuffle(to_remove)
    for group in to_remove[:10]:
        remove_group(state, group)

    candidates = [
        g
        for g in range(len(data.group_names))
        if g not in state.active
        and (data.group_masks[g] & region_mask)
        and not (state.task_mask & data.group_masks[g])
    ]
    candidates.sort(
        key=lambda g: (
            data.group_candidates[g][0].singleton_penalty
            / max(1, data.group_task_counts[g])
            + rng.random() * 3.0
        )
    )

    for group in candidates[:80]:
        if time.perf_counter() >= deadline:
            break
        if state.task_mask & data.group_masks[group]:
            continue
        c = choose_initial_courier(state, group, rng)
        if c is not None:
            add_courier(state, group, c)

    greedy_warmup(state, rng, move_limit=90, deadline=deadline)

    if state.energy > old_energy + 250_000.0 and rng.random() < 0.55:
        restore_state(state, snapshot)


def deepopt(state: State, rng: random.Random, deadline: float) -> None:
    data = state.data
    old = state.copy()
    old_energy = state.energy

    if state.active:
        active_groups = list(state.active)
        active_groups.sort(
            key=lambda g: state.group_penalty[g] / max(1, data.group_task_counts[g]),
            reverse=True,
        )
        region_mask = 0
        for g in active_groups[: rng.randrange(2, min(6, len(active_groups) + 1))]:
            region_mask |= data.group_masks[g]
    else:
        region_mask = 0

    all_tasks = list(range(data.total_tasks))
    rng.shuffle(all_tasks)
    for t in all_tasks[:6]:
        region_mask |= 1 << t

    removed_groups = [g for g in list(state.active) if data.group_masks[g] & region_mask]
    freed_couriers = set()
    freed_mask = 0
    for group in removed_groups:
        freed_couriers.update(state.assigned[group])
        freed_mask |= data.group_masks[group]
        remove_group(state, group)

    outside_mask = state.task_mask
    available_couriers = {c for c, owner in enumerate(state.owner) if owner == -1}
    available_couriers.update(freed_couriers)

    candidate_groups = []
    for group in range(len(data.group_names)):
        mask = data.group_masks[group]
        if mask & outside_mask:
            continue
        if not (mask & freed_mask):
            continue
        if mask | freed_mask != freed_mask:
            continue
        c = first_available_candidate(data, group, available_couriers)
        if c is None:
            continue
        penalty = penalty_after_add_to_empty_for_courier(data, group, c)
        candidate_groups.append((penalty - BIG * data.group_task_counts[group], group))

    candidate_groups.sort(key=lambda x: x[0])
    candidate_groups = candidate_groups[:90]

    beam = [(0.0, 0, 0, [])]  # local energy, task mask, courier mask, [(g, c)]
    width = 80
    for _, group in candidate_groups:
        if time.perf_counter() + 0.015 >= deadline:
            break
        next_beam = beam[:]
        gmask = data.group_masks[group]
        for energy, used_mask, used_cmask, chosen in beam:
            if used_mask & gmask:
                continue
            c = first_available_candidate_excluding(
                data, group, available_couriers, used_cmask
            )
            if c is None:
                continue
            cmask = 1 << c
            penalty = penalty_after_add_to_empty_for_courier(data, group, c)
            local_energy = energy + penalty - BIG * data.group_task_counts[group]
            next_beam.append(
                (local_energy, used_mask | gmask, used_cmask | cmask, chosen + [(group, c)])
            )
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:width]

    best_local = min(beam, key=lambda x: x[0], default=None)
    if best_local is not None:
        for group, c in best_local[3]:
            if group not in state.active and not (state.task_mask & data.group_masks[group]):
                if state.owner[c] == -1:
                    add_courier(state, group, c)

    greedy_warmup(state, rng, move_limit=120, deadline=deadline)

    if state.energy > old_energy + EPS:
        restore_state(state, old)


def first_available_candidate(
    data: ProblemData, group: int, available
):
    for cand in data.group_candidates[group]:
        if cand.courier in available:
            return cand.courier
    return None


def first_available_candidate_excluding(
    data: ProblemData, group: int, available, used_cmask: int
):
    for cand in data.group_candidates[group]:
        c = cand.courier
        if c in available and not (used_cmask & (1 << c)):
            return c
    return None


def restore_state(target: State, source: State) -> None:
    target.active = set(source.active)
    target.assigned = [s.copy() for s in source.assigned]
    target.owner = source.owner[:]
    target.sum_w = source.sum_w[:]
    target.sum_ws = source.sum_ws[:]
    target.reject_prod = source.reject_prod[:]
    target.zero_rejects = source.zero_rejects[:]
    target.group_penalty = source.group_penalty[:]
    target.task_mask = source.task_mask
    target.covered_count = source.covered_count
    target.total_penalty = source.total_penalty


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
