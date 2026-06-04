"""
Fused solver for the courier assignment problem.

Combines the incremental bitmask-based state from the deterministic solver
with the multi-source initial constructions and metaheuristics from the
submission solver. Runs under a hard time budget.

Public API kept compatible with score_solution.py:
    parse_input(text) -> ProblemData
    State(data)
    add_courier(state, group, courier)
    solve(input_text) -> list[(task_id_list_str, [courier_name, ...])]
"""

from collections import defaultdict, deque
import random
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIME_BUDGET = 8.85
SAFETY_MARGIN = 0.15

EPS = 1e-12
BIG = 10_000.0
TASK_FALLBACK_PER_TASK = 100.0

LOCAL_POLISH_PASSES = 6
PAIR_REPLACEMENT_ROUNDS = 4
PAIR_REPLACEMENT_CANDIDATE_LIMIT = 120
PAIR_REPLACEMENT_SHORTLIST = 20
HASH_START_SEEDS = (289, 245, 173, 95, 242)
THREE_CYCLE_MOVE_LIMIT = 3
THREE_CYCLE_MAX_ACTIVE_GROUPS = 35

REPARTITION_MAX_TASKS = 4
REPARTITION_MAX_COURIERS = 6
REPARTITION_FOCUS_FRACTION = 0.6

TABU_SWAP_TRIALS = 80
TABU_DEFAULT_STEPS = 18

VERY_LOW_WILLINGNESS = 0.16


def popcount(mask: int) -> int:
    return bin(mask).count("1")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class Candidate:
    __slots__ = ("courier", "score", "willingness", "singleton_penalty")

    def __init__(self, courier, score, willingness, singleton_penalty):
        self.courier = courier
        self.score = score
        self.willingness = willingness
        self.singleton_penalty = singleton_penalty


class ProblemData:
    def __init__(self):
        self.group_names = []
        self.group_masks = []
        self.group_task_counts = []
        self.group_fallbacks = []
        self.group_candidates = []      # list[list[Candidate]]
        self.cand_by_group = []         # list[dict[courier_id, (score, willingness)]]
        self.groups_by_courier = []     # list[list[group_id]]
        self.courier_names = []
        self.task_names = []
        self.total_tasks = 0
        self.task_full_mask = 0
        self.single_group_by_mask = {}
        self.mask_to_group = {}
        self.avg_willingness = 0.0


class State:
    def __init__(self, data: ProblemData):
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
    def energy(self):
        return self.total_penalty + BIG * (self.data.total_tasks - self.covered_count)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_input(input_text: str) -> ProblemData:
    raw_lines = input_text.strip().splitlines()
    if not raw_lines:
        return ProblemData()

    start = 1 if raw_lines[0].startswith("task_id_list") else 0

    task_to_id = {}
    courier_to_id = {}
    grouped = {}
    willingness_total = 0.0
    willingness_count = 0

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
        willingness_total += willingness
        willingness_count += 1

    data = ProblemData()
    data.total_tasks = len(task_to_id)
    data.task_full_mask = (1 << data.total_tasks) - 1
    data.courier_names = [""] * len(courier_to_id)
    for name, idx in courier_to_id.items():
        data.courier_names[idx] = name
    data.task_names = [""] * len(task_to_id)
    for name, idx in task_to_id.items():
        data.task_names[idx] = name
    data.groups_by_courier = [[] for _ in data.courier_names]
    data.avg_willingness = (willingness_total / willingness_count) if willingness_count else 0.0

    for group_name, gen_rows in grouped.items():
        mask = 0
        for task_name in group_name.split(","):
            task_name = task_name.strip()
            if task_name:
                mask |= 1 << task_to_id[task_name]
        task_count = popcount(mask)
        fallback = TASK_FALLBACK_PER_TASK * task_count
        best_by_courier = {}
        for courier, score, willingness in gen_rows:
            old = best_by_courier.get(courier)
            new_p = singleton_penalty(fallback, score, willingness)
            if old is None or new_p < singleton_penalty(fallback, old[0], old[1]):
                best_by_courier[courier] = (score, willingness)
        candidates = [
            Candidate(c, s, w, singleton_penalty(fallback, s, w))
            for c, (s, w) in best_by_courier.items()
        ]
        candidates.sort(key=lambda x: (x.singleton_penalty, x.score, -x.willingness))

        gid = len(data.group_names)
        data.group_names.append(group_name)
        data.group_masks.append(mask)
        data.group_task_counts.append(task_count)
        data.group_fallbacks.append(fallback)
        data.group_candidates.append(candidates)
        data.cand_by_group.append({c.courier: (c.score, c.willingness) for c in candidates})
        data.mask_to_group[mask] = gid
        if task_count == 1:
            data.single_group_by_mask[mask] = gid
        for cand in candidates:
            data.groups_by_courier[cand.courier].append(gid)

    return data


# ---------------------------------------------------------------------------
# Penalty computation (incremental + bulk)
# ---------------------------------------------------------------------------
def singleton_penalty(fallback, score, willingness):
    return (1.0 - willingness) * fallback + willingness * score


def penalty_from_stats(fallback, sum_w, sum_ws, reject_prod, zero_rejects):
    if sum_w <= EPS:
        return fallback
    reject_prob = 0.0 if zero_rejects > 0 else reject_prod
    accepted_score = sum_ws / sum_w
    return reject_prob * fallback + (1.0 - reject_prob) * accepted_score


def group_penalty_from_stats(data, group, sum_w, sum_ws, reject_prod, zero_rejects):
    return penalty_from_stats(
        data.group_fallbacks[group],
        sum_w, sum_ws, reject_prod, zero_rejects,
    )


def penalty_for_courier_set(data, group, couriers):
    sum_w = 0.0
    sum_ws = 0.0
    reject_prod = 1.0
    zero_rejects = 0
    cand = data.cand_by_group[group]
    for courier in couriers:
        s, w = cand[courier]
        sum_w += w
        sum_ws += w * s
        rf = 1.0 - w
        if rf <= EPS:
            zero_rejects += 1
        else:
            reject_prod *= rf
    return group_penalty_from_stats(data, group, sum_w, sum_ws, reject_prod, zero_rejects)


def penalty_after_add(state, group, courier):
    s, w = state.data.cand_by_group[group][courier]
    sum_w = state.sum_w[group] + w
    sum_ws = state.sum_ws[group] + w * s
    zero_rejects = state.zero_rejects[group]
    reject_prod = state.reject_prod[group]
    rf = 1.0 - w
    if rf <= EPS:
        zero_rejects += 1
    else:
        reject_prod *= rf
    return group_penalty_from_stats(state.data, group, sum_w, sum_ws, reject_prod, zero_rejects)


def penalty_after_remove(state, group, courier):
    s, w = state.data.cand_by_group[group][courier]
    sum_w = state.sum_w[group] - w
    sum_ws = state.sum_ws[group] - w * s
    zero_rejects = state.zero_rejects[group]
    reject_prod = state.reject_prod[group]
    rf = 1.0 - w
    if rf <= EPS:
        zero_rejects -= 1
    else:
        if rf > EPS:
            reject_prod /= rf
    return group_penalty_from_stats(state.data, group, sum_w, sum_ws, reject_prod, zero_rejects)


# ---------------------------------------------------------------------------
# State mutators
# ---------------------------------------------------------------------------
def add_courier(state, group, courier):
    data = state.data
    if courier in state.assigned[group]:
        return
    if state.owner[courier] != -1 and state.owner[courier] != group:
        # courier already assigned elsewhere -> caller bug
        raise ValueError("courier already owned by another group")

    if group not in state.active:
        state.active.add(group)
        state.task_mask |= data.group_masks[group]
        state.covered_count = popcount(state.task_mask)

    old_p = state.group_penalty[group]
    s, w = data.cand_by_group[group][courier]
    state.assigned[group].add(courier)
    state.owner[courier] = group
    state.sum_w[group] += w
    state.sum_ws[group] += w * s
    rf = 1.0 - w
    if rf <= EPS:
        state.zero_rejects[group] += 1
    else:
        state.reject_prod[group] *= rf

    new_p = group_penalty_from_stats(
        data, group,
        state.sum_w[group], state.sum_ws[group],
        state.reject_prod[group], state.zero_rejects[group],
    )
    state.group_penalty[group] = new_p
    state.total_penalty += new_p - old_p


def remove_courier(state, group, courier):
    data = state.data
    if courier not in state.assigned[group]:
        return
    old_p = state.group_penalty[group]
    s, w = data.cand_by_group[group][courier]
    state.assigned[group].remove(courier)
    state.owner[courier] = -1
    state.sum_w[group] -= w
    state.sum_ws[group] -= w * s
    rf = 1.0 - w
    if rf <= EPS:
        state.zero_rejects[group] -= 1
    else:
        if rf > EPS:
            state.reject_prod[group] /= rf

    if not state.assigned[group]:
        state.active.discard(group)
        state.task_mask &= ~data.group_masks[group]
        state.covered_count = popcount(state.task_mask)
        state.sum_w[group] = 0.0
        state.sum_ws[group] = 0.0
        state.reject_prod[group] = 1.0
        state.zero_rejects[group] = 0
        state.group_penalty[group] = 0.0
        state.total_penalty -= old_p
        return

    new_p = group_penalty_from_stats(
        data, group,
        state.sum_w[group], state.sum_ws[group],
        state.reject_prod[group], state.zero_rejects[group],
    )
    state.group_penalty[group] = new_p
    state.total_penalty += new_p - old_p


def remove_group(state, group):
    for courier in list(state.assigned[group]):
        remove_courier(state, group, courier)


def clone_state(state):
    clone = State(state.data)
    for g in state.active:
        for c in state.assigned[g]:
            add_courier(clone, g, c)
    return clone


def restore_state(target, source):
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


def better_state(a, b):
    if b is None:
        return True
    if a.covered_count != b.covered_count:
        return a.covered_count > b.covered_count
    return a.total_penalty + EPS < b.total_penalty


# ---------------------------------------------------------------------------
# Helpers used during construction
# ---------------------------------------------------------------------------
def best_available_seed_courier(state, group):
    for cand in state.data.group_candidates[group]:
        if state.owner[cand.courier] == -1:
            return cand.courier
    return None


def available_seed_penalties(state, group):
    ps = [c.singleton_penalty for c in state.data.group_candidates[group]
          if state.owner[c.courier] == -1]
    ps.sort()
    return ps


def best_seed_penalty(data, group):
    if not data.group_candidates[group]:
        return BIG
    return data.group_candidates[group][0].singleton_penalty


def hash_start_key(data, group, seed):
    name = data.group_names[group].split(",")[0]
    if len(name) > 1 and name[0] == "T" and name[1:].isdigit():
        v = int(name[1:])
    else:
        v = group
    v = (v + seed * 0x9E3779B1) & 0xFFFFFFFF
    v ^= v >> 16
    v = (v * 0x7FEB352D) & 0xFFFFFFFF
    v ^= v >> 15
    v = (v * 0x846CA68B) & 0xFFFFFFFF
    v ^= v >> 16
    return v


# ---------------------------------------------------------------------------
# Cover-group selection
# ---------------------------------------------------------------------------
def choose_cover_groups(data):
    selected = []
    covered_mask = 0
    for g in range(len(data.group_names)):
        if data.group_task_counts[g] == 1:
            selected.append(g)
            covered_mask |= data.group_masks[g]
    if covered_mask == data.task_full_mask:
        return selected

    selected_set = set(selected)
    while covered_mask != data.task_full_mask:
        missing = data.task_full_mask & ~covered_mask
        best_group = -1
        best_key = None
        for g in range(len(data.group_names)):
            if g in selected_set:
                continue
            mask = data.group_masks[g]
            if mask & covered_mask:
                continue
            newly = popcount(mask & missing)
            if newly <= 0:
                continue
            seed_p = best_seed_penalty(data, g)
            key = (seed_p / newly, -newly, data.group_names[g])
            if best_key is None or key < best_key:
                best_key = key
                best_group = g
        if best_group == -1:
            break
        selected.append(best_group)
        selected_set.add(best_group)
        covered_mask |= data.group_masks[best_group]
    return selected


# ---------------------------------------------------------------------------
# Initial constructions
# ---------------------------------------------------------------------------
def seed_groups_by_regret(state, groups):
    remaining = set(groups)
    while remaining:
        best_group = -1
        best_key = None
        for g in remaining:
            ps = available_seed_penalties(state, g)
            if not ps:
                continue
            regret = ps[1] - ps[0] if len(ps) > 1 else BIG
            key = (-regret, -ps[0], state.data.group_names[g])
            if best_key is None or key < best_key:
                best_key = key
                best_group = g
        if best_group == -1:
            break
        c = best_available_seed_courier(state, best_group)
        if c is not None:
            add_courier(state, best_group, c)
        remaining.discard(best_group)


def allocate_remaining_couriers_by_gain(state):
    """Assign every still-free courier to the active group where it lowers
    penalty the most. Greedy until no improving move exists."""
    moves = 0
    while True:
        best_group = -1
        best_courier = -1
        best_delta = -EPS
        for g in state.active:
            cand = state.data.cand_by_group[g]
            for c in state.data.group_candidates[g]:
                courier = c.courier
                if state.owner[courier] != -1:
                    continue
                delta = penalty_after_add(state, g, courier) - state.group_penalty[g]
                if delta < best_delta:
                    best_delta = delta
                    best_group = g
                    best_courier = courier
        if best_group == -1:
            return moves
        add_courier(state, best_group, best_courier)
        moves += 1


def construct_ordered_seed_state(data, ordered_groups):
    state = State(data)
    for g in ordered_groups:
        c = best_available_seed_courier(state, g)
        if c is not None:
            add_courier(state, g, c)
    allocate_remaining_couriers_by_gain(state)
    polish_courier_assignment(state)
    return state


def init_task_first_greedy(data):
    groups = choose_cover_groups(data)
    state = State(data)
    seed_groups_by_regret(state, groups)
    allocate_remaining_couriers_by_gain(state)
    polish_courier_assignment(state)
    return state


def init_hash_seed_states(data):
    groups = choose_cover_groups(data)
    out = []
    orders = [
        sorted(groups, key=lambda g: data.group_names[g], reverse=True),
        sorted(groups, key=lambda g: best_seed_penalty(data, g)),
        sorted(groups, key=lambda g: -(data.group_candidates[g][0].willingness
                                       if data.group_candidates[g] else 0.0)),
    ]
    for order in orders:
        out.append(construct_ordered_seed_state(data, order))
    for seed in HASH_START_SEEDS:
        order = sorted(groups, key=lambda g: hash_start_key(data, g, seed))
        out.append(construct_ordered_seed_state(data, order))
    return out


def init_shuffled_greedy(data, rng, temperature, pair_bias, willingness_bias):
    """Randomized one-row-per-pick greedy in courier+group space."""
    state = State(data)

    items = []
    for gid, candidates in enumerate(data.group_candidates):
        tc = data.group_task_counts[gid]
        for cand in candidates:
            base = cand.singleton_penalty / tc
            base -= pair_bias * (tc - 1)
            base -= willingness_bias * cand.willingness
            base += temperature * rng.random()
            items.append((base, gid, cand.courier))
    items.sort()

    used_tasks_mask = 0
    for _, gid, courier in items:
        if state.owner[courier] != -1:
            continue
        gmask = data.group_masks[gid]
        if used_tasks_mask & gmask:
            continue
        if gid in state.active:
            continue  # one seed per group; further couriers come from fill-in
        add_courier(state, gid, courier)
        used_tasks_mask |= gmask

    # Try to cover any still-missing tasks with single-task groups.
    if used_tasks_mask != data.task_full_mask:
        missing = data.task_full_mask & ~used_tasks_mask
        bit = 1
        while bit <= data.task_full_mask:
            if missing & bit:
                gid = data.single_group_by_mask.get(bit)
                if gid is not None:
                    c = best_available_seed_courier(state, gid)
                    if c is not None:
                        add_courier(state, gid, c)
                        used_tasks_mask |= bit
            bit <<= 1

    allocate_remaining_couriers_by_gain(state)
    return state


def init_min_cost_flow_single(data):
    """Min-cost matching of singleton groups to couriers.

    Only useful when there is at least one single-task group per task.
    """
    if data.total_tasks == 0:
        return None
    # Need a single-task group for every task to produce a full cover.
    full_cover = True
    task_to_group = [-1] * data.total_tasks
    for tid in range(data.total_tasks):
        gid = data.single_group_by_mask.get(1 << tid)
        if gid is None:
            full_cover = False
            break
        task_to_group[tid] = gid
    if not full_cover:
        return None

    n_tasks = data.total_tasks
    n_couriers = len(data.courier_names)
    if n_couriers < n_tasks:
        return None

    source = n_tasks + n_couriers
    sink = source + 1
    n_nodes = sink + 1
    graph = [[] for _ in range(n_nodes)]

    def add_edge(u, v, cap, cost):
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])

    for tid in range(n_tasks):
        add_edge(source, tid, 1, 0.0)
    for cid in range(n_couriers):
        add_edge(n_tasks + cid, sink, 1, 0.0)
    for tid in range(n_tasks):
        gid = task_to_group[tid]
        for cand in data.group_candidates[gid]:
            add_edge(tid, n_tasks + cand.courier, 1, cand.singleton_penalty)

    INF = 1e100
    potential = [0.0] * n_nodes
    flow = 0
    while flow < n_tasks:
        dist = [INF] * n_nodes
        prev_node = [-1] * n_nodes
        prev_edge = [-1] * n_nodes
        dist[source] = 0.0
        # Dijkstra with reduced costs
        visited = [False] * n_nodes
        # No heap needed; n is small enough for O(n^2)
        for _ in range(n_nodes):
            u = -1
            best = INF
            for i in range(n_nodes):
                if not visited[i] and dist[i] < best:
                    best = dist[i]
                    u = i
            if u == -1 or u == sink:
                break
            visited[u] = True
            for ei, edge in enumerate(graph[u]):
                v, cap, cost, _ = edge
                if cap <= 0:
                    continue
                nd = dist[u] + cost + potential[u] - potential[v]
                if nd + 1e-12 < dist[v]:
                    dist[v] = nd
                    prev_node[v] = u
                    prev_edge[v] = ei
        if prev_node[sink] == -1:
            return None
        for i in range(n_nodes):
            if dist[i] < INF / 2:
                potential[i] += dist[i]
        node = sink
        while node != source:
            u = prev_node[node]
            ei = prev_edge[node]
            graph[u][ei][1] -= 1
            graph[node][graph[u][ei][3]][1] += 1
            node = u
        flow += 1

    state = State(data)
    for tid in range(n_tasks):
        for v, cap, _, _ in graph[tid]:
            if n_tasks <= v < n_tasks + n_couriers and cap == 0:
                cid = v - n_tasks
                gid = task_to_group[tid]
                if cid in data.cand_by_group[gid]:
                    add_courier(state, gid, cid)
                break
    allocate_remaining_couriers_by_gain(state)
    return state


def init_min_weight_matching(data):
    """Match tasks pairwise (using available 2-task groups) plus singletons.

    Optional: requires networkx. Returns None if unavailable.
    """
    try:
        import networkx as nx
    except Exception:
        return None
    if data.total_tasks < 2:
        return None

    n = data.total_tasks
    # collect best singleton penalty for each task and best pair penalty for
    # each task pair
    single_p = [None] * n
    for tid in range(n):
        gid = data.single_group_by_mask.get(1 << tid)
        if gid is None:
            continue
        single_p[tid] = (best_seed_penalty(data, gid), gid)

    pair_best = {}
    for gid in range(len(data.group_names)):
        if data.group_task_counts[gid] != 2:
            continue
        mask = data.group_masks[gid]
        # extract two task ids
        bit_low = mask & -mask
        rest = mask ^ bit_low
        a = bit_low.bit_length() - 1
        b = rest.bit_length() - 1
        p = best_seed_penalty(data, gid)
        prev = pair_best.get((a, b))
        if prev is None or p < prev[0]:
            pair_best[(a, b)] = (p, gid)

    if not single_p[0] and not pair_best:
        return None

    g = nx.Graph()
    task_nodes = [f"T{tid}" for tid in range(n)]
    dummy_nodes = [f"D{i}" for i in range(n)]
    # Force minimum weight by negating weights via large constant when needed.
    # min_weight_matching takes "weight" attribute and tries to minimize.
    have_edge = False
    for tid in range(n):
        if single_p[tid] is not None:
            for d in dummy_nodes:
                g.add_edge(task_nodes[tid], d, weight=single_p[tid][0])
            have_edge = True
    for (a, b), (p, _) in pair_best.items():
        g.add_edge(task_nodes[a], task_nodes[b], weight=p)
        have_edge = True
    if not have_edge:
        return None

    try:
        matching = nx.algorithms.matching.min_weight_matching(g, weight="weight")
    except Exception:
        return None
    if not matching:
        return None

    state = State(data)
    used_tasks = 0
    for u, v in matching:
        if u.startswith("T") and v.startswith("T"):
            a = int(u[1:])
            b = int(v[1:])
            if a > b:
                a, b = b, a
            entry = pair_best.get((a, b))
            if entry is None:
                continue
            mask = (1 << a) | (1 << b)
            if used_tasks & mask:
                continue
            gid = entry[1]
            c = best_available_seed_courier(state, gid)
            if c is None:
                continue
            add_courier(state, gid, c)
            used_tasks |= mask
        elif u.startswith("T") or v.startswith("T"):
            t_node = u if u.startswith("T") else v
            tid = int(t_node[1:])
            if used_tasks & (1 << tid):
                continue
            entry = single_p[tid]
            if entry is None:
                continue
            gid = entry[1]
            c = best_available_seed_courier(state, gid)
            if c is None:
                continue
            add_courier(state, gid, c)
            used_tasks |= 1 << tid

    # cover missing tasks with singletons if possible
    missing = data.task_full_mask & ~used_tasks
    bit = 1
    while bit <= data.task_full_mask:
        if missing & bit:
            gid = data.single_group_by_mask.get(bit)
            if gid is not None:
                c = best_available_seed_courier(state, gid)
                if c is not None:
                    add_courier(state, gid, c)
        bit <<= 1

    allocate_remaining_couriers_by_gain(state)
    return state


def init_milp_single_bundle(data, deadline):
    """Optional MILP that covers every task with size-1..3 courier bundles per
    single-task group. Returns None if scipy is missing or the instance is too
    large."""
    if data.total_tasks == 0 or data.total_tasks > 45:
        return None
    if len(data.courier_names) < data.total_tasks:
        return None
    remaining = deadline - time.time()
    if remaining < 2.0:
        return None
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception:
        return None

    # One single-task group per task is required.
    single_groups = []
    for tid in range(data.total_tasks):
        gid = data.single_group_by_mask.get(1 << tid)
        if gid is None:
            return None
        single_groups.append(gid)

    from itertools import combinations
    pool_size = min(40, len(data.courier_names))
    keep_per_task = 600
    bundles = []
    for tid, gid in enumerate(single_groups):
        cands = data.group_candidates[gid]
        if not cands:
            return None
        pool = [c.courier for c in cands[:pool_size]]
        scored = []
        for size in (1, 2, 3):
            if len(pool) < size:
                continue
            for subset in combinations(pool, size):
                p = penalty_for_courier_set(data, gid, subset)
                scored.append((p, tid, gid, subset))
        scored.sort(key=lambda x: x[0])
        bundles.extend(scored[:keep_per_task])
    if not bundles:
        return None

    n_tasks = data.total_tasks
    n_couriers = len(data.courier_names)
    rows_count = n_tasks + n_couriers
    var_count = len(bundles)
    matrix = lil_matrix((rows_count, var_count), dtype=float)
    costs = np.zeros(var_count, dtype=float)
    for col, (p, tid, gid, subset) in enumerate(bundles):
        matrix[tid, col] = 1.0
        for c in subset:
            matrix[n_tasks + c, col] = 1.0
        costs[col] = p

    lower = np.r_[np.ones(n_tasks), np.zeros(n_couriers)]
    upper = np.ones(rows_count, dtype=float)

    time_limit = max(0.5, min(2.2, deadline - time.time() - 0.5))
    try:
        result = milp(
            c=costs,
            integrality=np.ones(var_count),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(matrix.tocsr(), lower, upper),
            options={"time_limit": time_limit, "mip_rel_gap": 0.0},
        )
    except Exception:
        return None
    if result is None or result.x is None:
        return None

    state = State(data)
    used_couriers = set()
    for idx, value in enumerate(result.x):
        if value > 0.5:
            _, _, gid, subset = bundles[idx]
            for c in subset:
                if c in used_couriers:
                    return None
                add_courier(state, gid, c)
                used_couriers.add(c)
    if state.covered_count != data.total_tasks:
        return None
    return state


def init_milp_bundle(data, deadline):
    """Two-phase MILP: maximize coverage, then minimize penalty.

    Each variable is (group, courier-subset); each task and each courier may
    be used at most once across selected variables. Returns None when
    unavailable.
    """
    if data.total_tasks == 0 or data.total_tasks > 45:
        return None
    remaining = deadline - time.time()
    if remaining < 2.0:
        return None
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception:
        return None

    from itertools import combinations
    n_tasks = data.total_tasks
    n_couriers = len(data.courier_names)
    bundles = []  # (gid, subset, value, task_count)
    seen = set()
    for gid in range(len(data.group_names)):
        cands = data.group_candidates[gid]
        if not cands:
            continue
        tc = data.group_task_counts[gid]
        pool_size = 10 if tc == 1 else 6
        pool = []
        seen_couriers = set()
        for ordered in (
            sorted(cands, key=lambda c: c.singleton_penalty),
            sorted(cands, key=lambda c: (-c.willingness, c.score)),
            sorted(cands, key=lambda c: c.score),
        ):
            for cand in ordered:
                if cand.courier in seen_couriers:
                    continue
                seen_couriers.add(cand.courier)
                pool.append(cand.courier)
                if len(pool) >= pool_size:
                    break
            if len(pool) >= pool_size:
                break
        subsets = [(c,) for c in pool]
        subsets.extend(tuple(p) for p in combinations(pool, 2))
        if tc == 1 and n_couriers >= n_tasks * 1.6:
            subsets.extend(tuple(t) for t in combinations(pool[:6], 3))
        keep = 50 if tc == 1 else 8
        scored = []
        for sub in subsets:
            key = (gid, sub)
            if key in seen:
                continue
            seen.add(key)
            p = penalty_for_courier_set(data, gid, sub)
            scored.append((p / tc, p, sub))
        for _, p, sub in sorted(scored)[:keep]:
            bundles.append((gid, sub, p, tc))
    if not bundles:
        return None

    var_count = len(bundles)
    rows_count = n_tasks + n_couriers
    matrix = lil_matrix((rows_count, var_count), dtype=float)
    coverage = np.zeros(var_count, dtype=float)
    costs = np.zeros(var_count, dtype=float)
    for col, (gid, sub, value, tc) in enumerate(bundles):
        mask = data.group_masks[gid]
        mm = mask
        while mm:
            bit = mm & -mm
            matrix[bit.bit_length() - 1, col] = 1.0
            mm -= bit
        for c in sub:
            matrix[n_tasks + c, col] = 1.0
        coverage[col] = tc
        costs[col] = value

    lower = np.zeros(rows_count, dtype=float)
    upper = np.ones(rows_count, dtype=float)

    t1 = max(0.3, min(0.6, (deadline - time.time()) * 0.15))
    try:
        first = milp(
            c=-coverage,
            integrality=np.ones(var_count),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(matrix.tocsr(), lower, upper),
            options={"time_limit": t1, "mip_rel_gap": 0.0},
        )
    except Exception:
        return None
    if first is None or first.x is None:
        return None
    max_covered = int(round(-float(first.fun))) if first.fun is not None else int(round(
        sum(coverage[i] for i, v in enumerate(first.x) if v > 0.5)
    ))
    if max_covered <= 0:
        return None

    cover_row = lil_matrix((1, var_count), dtype=float)
    cover_row[0, :] = coverage
    combined = lil_matrix((rows_count + 1, var_count), dtype=float)
    combined[:rows_count, :] = matrix
    combined[rows_count:, :] = cover_row

    second_lower = np.r_[lower, max_covered - 1e-7]
    second_upper = np.r_[upper, max_covered + 1e-7]

    t2 = max(0.4, min(1.0, deadline - time.time() - 0.5))
    try:
        second = milp(
            c=costs,
            integrality=np.ones(var_count),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(combined.tocsr(), second_lower, second_upper),
            options={"time_limit": t2, "mip_rel_gap": 0.01},
        )
    except Exception:
        return None
    if second is None or second.x is None:
        return None

    state = State(data)
    used_couriers = set()
    for idx, val in enumerate(second.x):
        if val > 0.5:
            gid, sub, _, _ = bundles[idx]
            for c in sub:
                if c in used_couriers:
                    return None
                add_courier(state, gid, c)
                used_couriers.add(c)
    return state


# ---------------------------------------------------------------------------
# Local search neighborhoods
# ---------------------------------------------------------------------------
def relocate_couriers_by_gain(state):
    moves = 0
    data = state.data
    while True:
        best_from = -1
        best_to = -1
        best_courier = -1
        best_delta = -EPS
        active = sorted(state.active)
        for from_g in active:
            if len(state.assigned[from_g]) <= 1:
                continue
            old_from = state.group_penalty[from_g]
            for courier in list(state.assigned[from_g]):
                rem_p = penalty_after_remove(state, from_g, courier)
                rem_delta = rem_p - old_from
                for to_g in active:
                    if to_g == from_g:
                        continue
                    if courier in state.assigned[to_g]:
                        continue
                    if courier not in data.cand_by_group[to_g]:
                        continue
                    add_p = penalty_after_add(state, to_g, courier)
                    add_delta = add_p - state.group_penalty[to_g]
                    delta = rem_delta + add_delta
                    if delta < best_delta:
                        best_delta = delta
                        best_from = from_g
                        best_to = to_g
                        best_courier = courier
        if best_courier == -1:
            return moves
        remove_courier(state, best_from, best_courier)
        add_courier(state, best_to, best_courier)
        moves += 1


def swap_couriers_by_gain(state):
    moves = 0
    data = state.data
    while True:
        best_a = -1
        best_b = -1
        best_ca = -1
        best_cb = -1
        best_delta = -EPS
        active = sorted(state.active)
        for i, ga in enumerate(active):
            cand_ga = data.cand_by_group[ga]
            assigned_ga = state.assigned[ga]
            old_pa = state.group_penalty[ga]
            fallback_a = data.group_fallbacks[ga]
            sw_a = state.sum_w[ga]
            sws_a = state.sum_ws[ga]
            rp_a = state.reject_prod[ga]
            zr_a = state.zero_rejects[ga]
            for gb in active[i + 1:]:
                cand_gb = data.cand_by_group[gb]
                assigned_gb = state.assigned[gb]
                old_pb = state.group_penalty[gb]
                old_p = old_pa + old_pb
                fallback_b = data.group_fallbacks[gb]
                sw_b = state.sum_w[gb]
                sws_b = state.sum_ws[gb]
                rp_b = state.reject_prod[gb]
                zr_b = state.zero_rejects[gb]
                for ca in assigned_ga:
                    if ca not in cand_gb:
                        continue
                    # Stats for ga after removing ca
                    sa, wa = cand_ga[ca]
                    sw_a_no = sw_a - wa
                    sws_a_no = sws_a - wa * sa
                    rf_ca = 1.0 - wa
                    if rf_ca <= EPS:
                        zr_a_no = zr_a - 1
                        rp_a_no = rp_a
                    else:
                        zr_a_no = zr_a
                        rp_a_no = rp_a / rf_ca if rf_ca > EPS else rp_a
                    for cb in assigned_gb:
                        if cb not in cand_ga:
                            continue
                        # ga: remove ca, add cb
                        sb_in_a, wb_in_a = cand_ga[cb]
                        new_sw_a = sw_a_no + wb_in_a
                        new_sws_a = sws_a_no + wb_in_a * sb_in_a
                        rf_cb_a = 1.0 - wb_in_a
                        new_zr_a = zr_a_no
                        new_rp_a = rp_a_no
                        if rf_cb_a <= EPS:
                            new_zr_a += 1
                        else:
                            new_rp_a *= rf_cb_a
                        pa = penalty_from_stats(fallback_a, new_sw_a, new_sws_a, new_rp_a, new_zr_a)

                        # gb: remove cb, add ca
                        sb_val, wb_val = cand_gb[cb]
                        sw_b_no = sw_b - wb_val
                        sws_b_no = sws_b - wb_val * sb_val
                        rf_cb = 1.0 - wb_val
                        if rf_cb <= EPS:
                            zr_b_no = zr_b - 1
                            rp_b_no = rp_b
                        else:
                            zr_b_no = zr_b
                            rp_b_no = rp_b / rf_cb if rf_cb > EPS else rp_b

                        sa_in_b, wa_in_b = cand_gb[ca]
                        new_sw_b = sw_b_no + wa_in_b
                        new_sws_b = sws_b_no + wa_in_b * sa_in_b
                        rf_ca_b = 1.0 - wa_in_b
                        new_zr_b = zr_b_no
                        new_rp_b = rp_b_no
                        if rf_ca_b <= EPS:
                            new_zr_b += 1
                        else:
                            new_rp_b *= rf_ca_b
                        pb = penalty_from_stats(fallback_b, new_sw_b, new_sws_b, new_rp_b, new_zr_b)

                        delta = pa + pb - old_p
                        if delta < best_delta:
                            best_delta = delta
                            best_a = ga
                            best_b = gb
                            best_ca = ca
                            best_cb = cb
        if best_ca == -1:
            return moves
        remove_courier(state, best_a, best_ca)
        remove_courier(state, best_b, best_cb)
        add_courier(state, best_a, best_cb)
        add_courier(state, best_b, best_ca)
        moves += 1


def polish_courier_assignment(state, passes=LOCAL_POLISH_PASSES):
    for _ in range(passes):
        moves = 0
        moves += relocate_couriers_by_gain(state)
        moves += swap_couriers_by_gain(state)
        moves += allocate_remaining_couriers_by_gain(state)
        if moves == 0:
            break


def polish_three_courier_cycles(state, deadline_t=None):
    if len(state.active) > THREE_CYCLE_MAX_ACTIVE_GROUPS:
        return 0
    moves = 0
    for _ in range(THREE_CYCLE_MOVE_LIMIT):
        if deadline_t is not None and time.time() >= deadline_t:
            return moves
        move = best_three_courier_cycle(state)
        if move is None:
            return moves
        apply_three_courier_cycle(state, move)
        polish_courier_assignment(state, passes=2)
        moves += 1
    return moves


def best_three_courier_cycle(state):
    data = state.data
    active = sorted(state.active)
    best_delta = -EPS
    best_move = None
    for i, ga in enumerate(active):
        for j in range(i + 1, len(active)):
            gb = active[j]
            for gc in active[j + 1:]:
                old_p = (
                    state.group_penalty[ga]
                    + state.group_penalty[gb]
                    + state.group_penalty[gc]
                )
                for ca in list(state.assigned[ga]):
                    cand_b_ok = ca in data.cand_by_group[gb]
                    cand_c_ok = ca in data.cand_by_group[gc]
                    for cb in list(state.assigned[gb]):
                        cb_in_a = cb in data.cand_by_group[ga]
                        cb_in_c = cb in data.cand_by_group[gc]
                        for cc in list(state.assigned[gc]):
                            cc_in_a = cc in data.cand_by_group[ga]
                            cc_in_b = cc in data.cand_by_group[gb]
                            if cand_b_ok and cb_in_c and cc_in_a:
                                d = three_cycle_delta(state, ga, gb, gc, ca, cb, cc, old_p, 0)
                                if d < best_delta:
                                    best_delta = d
                                    best_move = (ga, gb, gc, ca, cb, cc, 0)
                            if cand_c_ok and cc_in_b and cb_in_a:
                                d = three_cycle_delta(state, ga, gb, gc, ca, cb, cc, old_p, 1)
                                if d < best_delta:
                                    best_delta = d
                                    best_move = (ga, gb, gc, ca, cb, cc, 1)
    return best_move


def three_cycle_delta(state, ga, gb, gc, ca, cb, cc, old_p, mode):
    if mode == 0:
        new_a = (state.assigned[ga] - {ca}) | {cc}
        new_b = (state.assigned[gb] - {cb}) | {ca}
        new_c = (state.assigned[gc] - {cc}) | {cb}
    else:
        new_a = (state.assigned[ga] - {ca}) | {cb}
        new_b = (state.assigned[gb] - {cb}) | {cc}
        new_c = (state.assigned[gc] - {cc}) | {ca}
    new_p = (
        penalty_for_courier_set(state.data, ga, new_a)
        + penalty_for_courier_set(state.data, gb, new_b)
        + penalty_for_courier_set(state.data, gc, new_c)
    )
    return new_p - old_p


def apply_three_courier_cycle(state, move):
    ga, gb, gc, ca, cb, cc, mode = move
    remove_courier(state, ga, ca)
    remove_courier(state, gb, cb)
    remove_courier(state, gc, cc)
    if mode == 0:
        add_courier(state, ga, cc)
        add_courier(state, gb, ca)
        add_courier(state, gc, cb)
    else:
        add_courier(state, ga, cb)
        add_courier(state, gb, cc)
        add_courier(state, gc, ca)


def improve_by_pair_group_replacements(state, deadline_t=None):
    data = state.data
    pair_groups = [
        g for g in range(len(data.group_names))
        if data.group_task_counts[g] == 2 and data.group_candidates[g]
    ]
    pair_groups.sort(key=lambda g: (best_seed_penalty(data, g), data.group_names[g]))
    pair_groups = pair_groups[:PAIR_REPLACEMENT_CANDIDATE_LIMIT]

    for _ in range(PAIR_REPLACEMENT_ROUNDS):
        if deadline_t is not None and time.time() >= deadline_t:
            return
        approximate = []
        for pg in pair_groups:
            if pg in state.active:
                continue
            if deadline_t is not None and time.time() >= deadline_t:
                break
            cand = pair_replacement_state(state, pg, do_polish=False)
            if cand is not None:
                approximate.append((cand.total_penalty - state.total_penalty, pg))
        if not approximate:
            break
        approximate.sort()
        best = None
        best_delta = -EPS
        for _, pg in approximate[:PAIR_REPLACEMENT_SHORTLIST]:
            if deadline_t is not None and time.time() >= deadline_t:
                break
            cand = pair_replacement_state(state, pg, do_polish=True)
            if cand is None:
                continue
            delta = cand.total_penalty - state.total_penalty
            if delta < best_delta and cand.covered_count >= state.covered_count:
                best_delta = delta
                best = cand
        if best is None:
            break
        restore_state(state, best)


def pair_replacement_state(state, pair_group, do_polish):
    data = state.data
    pair_mask = data.group_masks[pair_group]
    bit = 1
    groups_to_remove = []
    while bit <= data.task_full_mask:
        if pair_mask & bit:
            sg = data.single_group_by_mask.get(bit)
            if sg is None or sg not in state.active:
                return None
            groups_to_remove.append(sg)
        bit <<= 1
    if len(groups_to_remove) != 2:
        return None
    cand = clone_state(state)
    for g in groups_to_remove:
        remove_group(cand, g)
    c = best_available_seed_courier(cand, pair_group)
    if c is None:
        return None
    add_courier(cand, pair_group, c)
    allocate_remaining_couriers_by_gain(cand)
    if do_polish:
        polish_courier_assignment(cand, passes=2)
    return cand


# ---------------------------------------------------------------------------
# Repartition (regroup task partitions across two adjacent groups)
# ---------------------------------------------------------------------------
def repartition_state(state, rng, max_pairs=8):
    data = state.data
    active = sorted(state.active, key=lambda g: -state.group_penalty[g])
    if len(active) < 2:
        return False
    focus_count = max(2, int(len(active) * REPARTITION_FOCUS_FRACTION))
    focus = active[:focus_count]
    if len(focus) < 2:
        focus = active

    pairs_seen = set()
    pairs = []
    max_unique = len(focus) * (len(focus) - 1) // 2
    target = min(max_pairs, max_unique)
    attempts = 0
    while len(pairs) < target and attempts < target * 4:
        attempts += 1
        a, b = rng.sample(focus, 2)
        key = (a, b) if a < b else (b, a)
        if key in pairs_seen:
            continue
        pairs_seen.add(key)
        pairs.append(key)

    improved = False
    for ga, gb in pairs:
        if ga not in state.active or gb not in state.active:
            continue
        union_mask = data.group_masks[ga] | data.group_masks[gb]
        if popcount(union_mask) > REPARTITION_MAX_TASKS:
            continue
        couriers = list(state.assigned[ga] | state.assigned[gb])
        if len(couriers) > REPARTITION_MAX_COURIERS:
            continue
        partitions = list(_enumerate_mask_partitions(data, union_mask))
        if not partitions:
            continue
        old_p = state.group_penalty[ga] + state.group_penalty[gb]
        best_partition = None
        best_assignment = None
        best_p = old_p - EPS
        for part in partitions:
            if set(part) == {ga, gb}:
                continue
            assignment = _best_assignment_for_partition(data, part, couriers)
            if assignment is None:
                continue
            p, buckets = assignment
            if p < best_p:
                best_p = p
                best_partition = part
                best_assignment = buckets
        if best_partition is None:
            continue
        # Apply: remove both source groups, add the new ones.
        remove_group(state, ga)
        remove_group(state, gb)
        for gid, bucket in zip(best_partition, best_assignment):
            for c in bucket:
                add_courier(state, gid, c)
        improved = True
    return improved


def _enumerate_mask_partitions(data, mask):
    """All partitions of `mask` into existing groups (singleton or pair)."""
    bits = []
    mm = mask
    while mm:
        bit = mm & -mm
        bits.append(bit)
        mm ^= bit
    out = []

    def rec(remaining_mask, current):
        if remaining_mask == 0:
            out.append(tuple(current))
            return
        # Pick the lowest remaining bit.
        bit = remaining_mask & -remaining_mask
        # Single-task group containing this task
        sg = data.single_group_by_mask.get(bit)
        if sg is not None:
            current.append(sg)
            rec(remaining_mask ^ bit, current)
            current.pop()
        # Pair groups containing this task (existing only)
        # Search through remaining bits for a paired group.
        rem = remaining_mask ^ bit
        mm = rem
        while mm:
            other = mm & -mm
            pair_mask = bit | other
            pg = data.mask_to_group.get(pair_mask)
            if pg is not None and data.group_task_counts[pg] == 2:
                current.append(pg)
                rec(remaining_mask ^ pair_mask, current)
                current.pop()
            mm ^= other

    rec(mask, [])
    return out


def _best_assignment_for_partition(data, partition, couriers):
    """Bruteforce assign each courier to one of the partition's groups (or
    leave unused), minimizing the sum of penalties with the constraint that
    each group has at least one courier."""
    n_parts = len(partition)
    assigned = [[] for _ in range(n_parts)]
    best = [None]
    cand_for = [data.cand_by_group[g] for g in partition]

    def rec(index):
        if index == len(couriers):
            for bucket in assigned:
                if not bucket:
                    return
            value = sum(
                penalty_for_courier_set(data, partition[i], assigned[i])
                for i in range(n_parts)
            )
            if best[0] is None or value < best[0][0]:
                best[0] = (value, [b[:] for b in assigned])
            return
        c = couriers[index]
        rec(index + 1)  # leave courier unused
        for i in range(n_parts):
            if c in cand_for[i]:
                assigned[i].append(c)
                rec(index + 1)
                assigned[i].pop()

    rec(0)
    return best[0]


# ---------------------------------------------------------------------------
# Tabu search over state moves
# ---------------------------------------------------------------------------
def tabu_confchange(state, rng, deadline_t, max_steps=TABU_DEFAULT_STEPS):
    data = state.data
    if not state.active:
        return state

    best_state = clone_state(state)
    best_value = state.total_penalty
    best_covered = state.covered_count
    tabu = {}
    conf_change = {g: True for g in state.active}

    def touch(group, courier=None):
        conf_change[group] = False
        if courier is not None:
            for other in state.active:
                if other != group and courier in data.cand_by_group[other]:
                    conf_change[other] = True

    for step in range(max_steps):
        if time.time() >= deadline_t:
            break
        active = sorted(state.active)
        for g in active:
            conf_change.setdefault(g, True)

        tenure = 4 + (step % 5)
        progress = step / max(1, max_steps)
        uphill_limit = 2.0 + 12.0 * (1.0 - progress)
        best_move = None

        for ga in active:
            if not conf_change.get(ga, True):
                continue
            old_a = state.group_penalty[ga]
            assigned_a = list(state.assigned[ga])
            for courier in assigned_a:
                if len(state.assigned[ga]) <= 1:
                    drop_allowed = False
                else:
                    drop_allowed = True
                if drop_allowed:
                    new_a = penalty_after_remove(state, ga, courier)
                    delta = new_a - old_a
                    aspiration = state.total_penalty + delta + EPS < best_value
                    if (step >= tabu.get((courier, ga), -1) or aspiration) and delta <= uphill_limit:
                        score = delta + rng.random() * 0.02
                        if best_move is None or score < best_move[0]:
                            best_move = (score, delta, ("drop", ga, courier))
                # move to another group
                for gb in active:
                    if gb == ga:
                        continue
                    if not (conf_change.get(ga, True) or conf_change.get(gb, True)):
                        continue
                    if courier in state.assigned[gb]:
                        continue
                    if courier not in data.cand_by_group[gb]:
                        continue
                    old_b = state.group_penalty[gb]
                    new_a = penalty_after_remove(state, ga, courier)
                    new_b = penalty_after_add(state, gb, courier)
                    delta = (new_a - old_a) + (new_b - old_b)
                    aspiration = state.total_penalty + delta + EPS < best_value
                    if step < tabu.get((courier, gb), -1) and not aspiration:
                        continue
                    if delta > uphill_limit:
                        continue
                    score = delta + rng.random() * 0.02
                    if best_move is None or score < best_move[0]:
                        best_move = (score, delta, ("move", ga, gb, courier))
                # add free courier from anywhere to gb is covered by greedy fill;
                # we don't search "add" here.

        # Sample some swaps
        if len(active) >= 2:
            trials = min(TABU_SWAP_TRIALS, 4 * len(active))
            for _ in range(trials):
                ga = rng.choice(active)
                gb = rng.choice(active)
                if ga == gb:
                    continue
                if not state.assigned[ga] or not state.assigned[gb]:
                    continue
                ca = rng.choice(tuple(state.assigned[ga]))
                cb = rng.choice(tuple(state.assigned[gb]))
                if ca == cb:
                    continue
                if ca not in data.cand_by_group[gb]:
                    continue
                if cb not in data.cand_by_group[ga]:
                    continue
                new_a = (state.assigned[ga] - {ca}) | {cb}
                new_b = (state.assigned[gb] - {cb}) | {ca}
                old_a = state.group_penalty[ga]
                old_b = state.group_penalty[gb]
                pa = penalty_for_courier_set(data, ga, new_a)
                pb = penalty_for_courier_set(data, gb, new_b)
                delta = pa + pb - old_a - old_b
                aspiration = state.total_penalty + delta + EPS < best_value
                if (step < tabu.get((ca, gb), -1) or step < tabu.get((cb, ga), -1)) and not aspiration:
                    continue
                if delta > uphill_limit:
                    continue
                score = delta + rng.random() * 0.02
                if best_move is None or score < best_move[0]:
                    best_move = (score, delta, ("swap", ga, gb, ca, cb))

        if best_move is None:
            for g in state.active:
                conf_change[g] = True
            continue

        _, delta, op = best_move
        if op[0] == "drop":
            _, ga, courier = op
            remove_courier(state, ga, courier)
            tabu[(courier, ga)] = step + tenure
            touch(ga, courier)
        elif op[0] == "move":
            _, ga, gb, courier = op
            remove_courier(state, ga, courier)
            add_courier(state, gb, courier)
            tabu[(courier, ga)] = step + tenure
            touch(ga, courier)
            touch(gb, courier)
        else:
            _, ga, gb, ca, cb = op
            remove_courier(state, ga, ca)
            remove_courier(state, gb, cb)
            add_courier(state, ga, cb)
            add_courier(state, gb, ca)
            tabu[(ca, ga)] = step + tenure
            tabu[(cb, gb)] = step + tenure
            touch(ga, ca)
            touch(ga, cb)
            touch(gb, ca)
            touch(gb, cb)

        if state.covered_count > best_covered or (
            state.covered_count == best_covered and state.total_penalty + EPS < best_value
        ):
            best_value = state.total_penalty
            best_covered = state.covered_count
            best_state = clone_state(state)

    if best_covered > state.covered_count or (
        best_covered == state.covered_count and best_value + EPS < state.total_penalty
    ):
        restore_state(state, best_state)
    return state


# ---------------------------------------------------------------------------
# Perturbation / kick / destroy-repair
# ---------------------------------------------------------------------------
def perturb_extras(state, rng):
    removable = []
    for g in state.active:
        if len(state.assigned[g]) > 1:
            for c in state.assigned[g]:
                removable.append((g, c))
    if not removable:
        return False
    rng.shuffle(removable)
    n = 1 + rng.randrange(min(8, len(removable)))
    removed = 0
    for g, c in removable:
        if len(state.assigned[g]) > 1 and c in state.assigned[g]:
            remove_courier(state, g, c)
            removed += 1
            if removed >= n:
                break
    allocate_remaining_couriers_by_gain(state)
    return removed > 0


def kick_state(state, rng, strength):
    data = state.data
    moved = 0
    active = list(state.active)
    if not active:
        return False
    for _ in range(strength):
        active = list(state.active)
        if not active:
            break
        src = rng.choice(active)
        if not state.assigned[src]:
            continue
        courier = rng.choice(tuple(state.assigned[src]))
        if rng.random() < 0.65:
            targets = [g for g in active if g != src and courier in data.cand_by_group[g]]
            if targets:
                dst = rng.choice(targets)
                if courier not in state.assigned[dst]:
                    remove_courier(state, src, courier)
                    add_courier(state, dst, courier)
                    moved += 1
                    continue
        if len(state.assigned[src]) > 1:
            remove_courier(state, src, courier)
            moved += 1
    allocate_remaining_couriers_by_gain(state)
    return moved > 0


def destroy_repair(state, rng, drop_count):
    data = state.data
    active = list(state.active)
    if not active:
        return False
    rng.shuffle(active)
    for g in active[:min(drop_count, len(active))]:
        remove_group(state, g)
    # Try to restore coverage with single-task groups first.
    missing = data.task_full_mask & ~state.task_mask
    bit = 1
    while bit <= data.task_full_mask:
        if missing & bit:
            sg = data.single_group_by_mask.get(bit)
            if sg is not None and sg not in state.active:
                c = best_available_seed_courier(state, sg)
                if c is not None:
                    add_courier(state, sg, c)
        bit <<= 1
    allocate_remaining_couriers_by_gain(state)
    return True


# ---------------------------------------------------------------------------
# Main solve loop
# ---------------------------------------------------------------------------
def solve(input_text: str) -> list:
    start = time.time()
    deadline = start + TIME_BUDGET

    data = parse_input(input_text)
    if not data.group_names:
        return []

    rng = random.Random(
        len(data.group_names) * 1000003
        + len(data.courier_names) * 1009
        + data.total_tasks * 917
    )

    avg_w = data.avg_willingness
    scarce_case = len(data.courier_names) <= 1.35 * data.total_tasks
    low_case = avg_w < 0.22
    very_low_case = avg_w < VERY_LOW_WILLINGNESS
    hard_case = scarce_case or low_case

    best = None

    def consider(state):
        nonlocal best
        if state is None:
            return
        if better_state(state, best):
            best = clone_state(state)

    def time_left():
        return deadline - time.time()

    # ---- Phase 1: deterministic seeds ----
    seed = init_task_first_greedy(data)
    polish_courier_assignment(seed)
    improve_by_pair_group_replacements(seed, deadline_t=deadline - SAFETY_MARGIN)
    polish_three_courier_cycles(seed, deadline_t=deadline - SAFETY_MARGIN)
    polish_courier_assignment(seed)
    consider(seed)

    if time_left() > 0.6:
        for s in init_hash_seed_states(data):
            polish_courier_assignment(s, passes=2)
            consider(s)
            if time_left() < 0.6:
                break

    # ---- Phase 2: optional structured initial solutions ----
    if time_left() > 1.0 and data.total_tasks <= 60:
        s = init_min_cost_flow_single(data)
        if s is not None:
            polish_courier_assignment(s)
            consider(s)

    if time_left() > 1.5 and scarce_case:
        s = init_min_weight_matching(data)
        if s is not None:
            polish_courier_assignment(s)
            consider(s)

    if time_left() > 2.0 and not hard_case and len(data.courier_names) >= data.total_tasks * 1.6:
        s = init_milp_single_bundle(data, deadline - 0.3)
        if s is not None:
            polish_courier_assignment(s)
            consider(s)
    elif time_left() > 2.0 and not hard_case:
        s = init_milp_bundle(data, deadline - 0.3)
        if s is not None:
            polish_courier_assignment(s)
            consider(s)
    elif time_left() > 2.0 and hard_case:
        s = init_milp_bundle(data, deadline - 0.5)
        if s is not None:
            polish_courier_assignment(s)
            consider(s)

    # ---- Phase 3: ILS with tabu / perturbations ----
    if best is None:
        # Fall back to a minimal greedy if everything failed.
        best = init_task_first_greedy(data)

    work = clone_state(best)
    round_no = 0
    no_improve = 0
    last_improve_time = time.time()
    while True:
        remaining = time_left()
        if remaining <= SAFETY_MARGIN + 0.05:
            break
        # Early stop: if no improvement in last 2.5s and we're past 88% of budget,
        # stop early to reduce variance from unproductive tail iterations.
        if (time.time() - last_improve_time > 2.5
                and time.time() - start > TIME_BUDGET * 0.88):
            break
        round_no += 1

        # Periodic deeper polish on the global best.
        if round_no % 7 == 0 and time_left() > 0.5:
            tmp = clone_state(best)
            improve_by_pair_group_replacements(tmp, deadline_t=deadline - SAFETY_MARGIN)
            if time_left() > 0.4:
                repartition_state(tmp, rng, max_pairs=10)
            polish_courier_assignment(tmp)
            consider(tmp)
            work = clone_state(best)

        # Choose next seed for the iteration.
        if round_no % 11 == 0:
            destroy_repair(work, rng, 1 + (round_no // 11) % 4)
        elif round_no % 5 == 0:
            kick_state(work, rng, 2 + (round_no // 5) % 6)
        elif round_no % 3 == 0:
            perturb_extras(work, rng)
        else:
            if very_low_case and not scarce_case:
                temperature = (1.5, 4.0, 9.0, 18.0)[round_no % 4]
                pair_bias = (-30.0, -15.0, 0.0, 12.0)[(round_no // 2) % 4]
                w_bias = (20.0, 35.0, 50.0, 65.0)[round_no % 4]
            elif hard_case:
                temperature = (1.5, 4.0, 9.0, 18.0)[round_no % 4]
                pair_bias = (25.0, 50.0, 90.0, 140.0)[(round_no // 2) % 4]
                w_bias = (0.0, 10.0, 20.0, 30.0)[round_no % 4]
            else:
                temperature = (2.5, 7.5, 15.0, 30.0)[round_no % 4]
                pair_bias = (0.0, 8.0, 18.0, 32.0)[(round_no // 2) % 4]
                w_bias = (0.0, 8.0, 18.0)[(round_no // 5) % 3]
            work = init_shuffled_greedy(data, rng, temperature, pair_bias, w_bias)

        polish_courier_assignment(work, passes=2)

        if time_left() > 0.4:
            tabu_confchange(
                work, rng,
                deadline - SAFETY_MARGIN,
                max_steps=18 if time_left() > 1.5 else 6,
            )
            allocate_remaining_couriers_by_gain(work)
            polish_courier_assignment(work, passes=1)

        if better_state(work, best):
            consider(work)
            no_improve = 0
            last_improve_time = time.time()
        else:
            no_improve += 1
            if no_improve >= 4:
                # diversify by jumping back to best with stronger kick
                work = clone_state(best)
                kick_state(work, rng, 3 + (round_no % 5))
                no_improve = 0

    return format_solution(best)


def format_solution(state):
    out = []
    for g in sorted(state.active, key=lambda gid: state.data.group_names[gid]):
        couriers = list(state.assigned[g])
        cand = state.data.cand_by_group[g]
        couriers.sort(key=lambda c: (-cand[c][1], cand[c][0],
                                     state.data.courier_names[c]))
        names = [state.data.courier_names[c] for c in couriers]
        if names:
            out.append((state.data.group_names[g], names))
    return out
