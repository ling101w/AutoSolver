
from collections import defaultdict
import math
import random
import time

CONFIG = {'time_limit': 8.75, 'seed': 20260524, 'local_rounds': 3, 'loop_local_rounds': 1, 'extra_limit': 80, 'max_local_keys': 80, 'mutate_coverage': 15.0, 'mutate_pair': 20.0, 'mutate_willingness': 12.0, 'loop_random_weight': 8.0, 'beam_width': 160, 'beam_keep_per_group': 4, 'beam_task_limit': 42, 'use_flow': True, 'use_beam': True, 'use_sa': False, 'sa_temp': 30.0, 'sa_cooling': 0.93, 'sa_iters_per_temp': 30, 'sa_min_temp': 0.5, 'profiles': [{'coverage_weight': 45.0, 'pair_weight': 20.0, 'willingness_weight': 15.0, 'score_weight': 0.0, 'random_weight': 0.0}, {'coverage_weight': 30.0, 'pair_weight': 0.0, 'willingness_weight': 5.0, 'score_weight': 0.0, 'random_weight': 0.0}, {'coverage_weight': 6.0, 'pair_weight': 8.0, 'willingness_weight': 35.0, 'score_weight': 0.0, 'random_weight': 0.0}]}


def solve(input_text: str) -> list:
    start_time = time.time()
    deadline = start_time + CONFIG.get("time_limit", 8.75)

    rows = []
    by_key = defaultdict(dict)
    key_tasks = {}

    for line in input_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("task_id_list"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        raw_key, courier, score, willingness = parts[:4]
        try:
            tasks = tuple(t.strip() for t in raw_key.split(",") if t.strip())
            if not tasks:
                continue
            task_key = ",".join(tasks)
            courier = courier.strip()
            score = float(score)
            willingness = float(willingness)
        except Exception:
            continue
        rows.append((tasks, task_key, courier, score, willingness))
        by_key[task_key][courier] = (score, willingness)
        key_tasks[task_key] = tasks

    if not rows:
        return []

    all_tasks = sorted(set(t for tasks, _, _, _, _ in rows for t in tasks))
    all_couriers = sorted(set(c for _, _, c, _, _ in rows))

    def expected_one(tasks, score, willingness):
        return willingness * score + (1.0 - willingness) * 100.0 * len(tasks)

    def penalty(task_key, courier_list):
        fallback = 100.0 * len(key_tasks[task_key])
        reject_prob = 1.0
        weighted_score = 0.0
        weight = 0.0
        data = by_key[task_key]
        for courier in courier_list:
            if courier not in data:
                continue
            score, willingness = data[courier]
            reject_prob *= 1.0 - willingness
            weighted_score += willingness * score
            weight += willingness
        if weight <= 0.0:
            return fallback
        return reject_prob * fallback + (1.0 - reject_prob) * weighted_score / weight

    def clone_groups(groups):
        return dict((key, list(value)) for key, value in groups.items())

    def covered_count(groups):
        seen = set()
        for task_key in groups:
            if task_key in key_tasks:
                seen.update(key_tasks[task_key])
        return len(seen)

    def total_penalty(groups):
        miss = len(all_tasks) - covered_count(groups)
        return sum(penalty(k, v) for k, v in groups.items()) + 100.0 * miss

    def valid_groups(groups):
        used_tasks = set()
        used_couriers = set()
        for task_key, couriers in groups.items():
            if task_key not in key_tasks or not couriers:
                return False
            for task in key_tasks[task_key]:
                if task in used_tasks:
                    return False
                used_tasks.add(task)
            local = set()
            for courier in couriers:
                if courier in local or courier in used_couriers:
                    return False
                if courier not in by_key[task_key]:
                    return False
                local.add(courier)
                used_couriers.add(courier)
        return True

    def normalize(groups):
        result = {}
        used_tasks = set()
        used_couriers = set()
        items = []
        for task_key, couriers in groups.items():
            if task_key not in key_tasks:
                continue
            value = penalty(task_key, couriers) if couriers else 10 ** 18
            items.append((-len(key_tasks[task_key]), value, task_key, couriers))
        for _, _, task_key, couriers in sorted(items):
            if any(task in used_tasks for task in key_tasks[task_key]):
                continue
            clean = []
            local = set()
            for courier in couriers:
                if courier in local or courier in used_couriers:
                    continue
                if courier not in by_key[task_key]:
                    continue
                clean.append(courier)
                local.add(courier)
            if clean:
                result[task_key] = clean
                used_tasks.update(key_tasks[task_key])
                used_couriers.update(clean)
        return result

    def cover_unassigned(groups):
        groups = clone_groups(groups)
        used_tasks = set()
        used_couriers = set()
        for task_key, couriers in groups.items():
            used_tasks.update(key_tasks[task_key])
            used_couriers.update(couriers)
        while len(used_tasks) < len(all_tasks) and time.time() < deadline - 0.02:
            best = None
            for tasks, task_key, courier, score, willingness in rows:
                if courier in used_couriers:
                    continue
                if any(task in used_tasks for task in tasks):
                    continue
                expected = expected_one(tasks, score, willingness)
                item = (-len(tasks), expected / len(tasks), expected, -willingness, task_key, courier, tasks)
                if best is None or item < best:
                    best = item
            if best is None:
                break
            _, _, _, _, task_key, courier, tasks = best
            groups[task_key] = [courier]
            used_couriers.add(courier)
            used_tasks.update(tasks)
        return groups

    def weighted_greedy(profile, rng):
        groups = {}
        used_tasks = set()
        used_couriers = set()
        ranked = []
        coverage_weight = profile.get("coverage_weight", 0.0)
        pair_weight = profile.get("pair_weight", 0.0)
        willingness_weight = profile.get("willingness_weight", 0.0)
        score_weight = profile.get("score_weight", 0.0)
        random_weight = profile.get("random_weight", 0.0)
        for index, (tasks, task_key, courier, score, willingness) in enumerate(rows):
            expected = expected_one(tasks, score, willingness)
            value = expected / len(tasks)
            value -= coverage_weight * len(tasks)
            value -= pair_weight * (len(tasks) - 1)
            value -= willingness_weight * willingness
            value += score_weight * score / len(tasks)
            value += random_weight * rng.random()
            ranked.append((value, rng.random(), index))
        for _, _, index in sorted(ranked):
            tasks, task_key, courier, _, _ = rows[index]
            if courier in used_couriers:
                continue
            if any(task in used_tasks for task in tasks):
                continue
            groups[task_key] = [courier]
            used_couriers.add(courier)
            used_tasks.update(tasks)
        return cover_unassigned(groups)

    def fill_extra_couriers(groups):
        groups = clone_groups(groups)
        used = set(c for couriers in groups.values() for c in couriers)
        limit = CONFIG.get("extra_limit", 80)
        added = 0
        while added < limit and time.time() < deadline - 0.02:
            best = None
            for courier in all_couriers:
                if courier in used:
                    continue
                for task_key, couriers in groups.items():
                    if courier not in by_key[task_key]:
                        continue
                    delta = penalty(task_key, couriers + [courier]) - penalty(task_key, couriers)
                    if best is None or delta < best[0]:
                        best = (delta, task_key, courier)
            if best is None or best[0] >= -1e-12:
                break
            _, task_key, courier = best
            groups[task_key].append(courier)
            used.add(courier)
            added += 1
        return groups

    def local_improve(groups, rounds):
        groups = clone_groups(groups)
        max_keys = CONFIG.get("max_local_keys", 80)
        for _ in range(rounds):
            if time.time() >= deadline - 0.02:
                break
            keys = sorted(list(groups), key=lambda key: penalty(key, groups[key]), reverse=True)[:max_keys]
            costs = dict((key, penalty(key, groups[key])) for key in keys)
            best = (0.0, None)
            for a in keys:
                if time.time() >= deadline - 0.02:
                    break
                if len(groups[a]) <= 1:
                    continue
                old_a = costs[a]
                for courier in list(groups[a]):
                    without = [c for c in groups[a] if c != courier]
                    if not without:
                        continue
                    new_a = penalty(a, without)
                    delta = new_a - old_a
                    if delta < best[0]:
                        best = (delta, ("drop", a, courier, without))
                    for b in keys:
                        if a == b or courier not in by_key[b]:
                            continue
                        old_b = costs[b]
                        delta = new_a + penalty(b, groups[b] + [courier]) - old_a - old_b
                        if delta < best[0]:
                            best = (delta, ("move", a, b, courier, without))
            for i, a in enumerate(keys):
                if time.time() >= deadline - 0.02:
                    break
                for b in keys[i + 1:]:
                    old_a = costs[a]
                    old_b = costs[b]
                    for ca in list(groups[a]):
                        if ca not in by_key[b]:
                            continue
                        for cb in list(groups[b]):
                            if cb not in by_key[a]:
                                continue
                            next_a = [c for c in groups[a] if c != ca] + [cb]
                            next_b = [c for c in groups[b] if c != cb] + [ca]
                            delta = penalty(a, next_a) + penalty(b, next_b) - old_a - old_b
                            if delta < best[0]:
                                best = (delta, ("swap", a, b, next_a, next_b))
            if best[1] is None:
                break
            op = best[1]
            if op[0] == "drop":
                _, a, _, without = op
                groups[a] = without
            elif op[0] == "move":
                _, a, b, courier, without = op
                groups[a] = without
                groups[b].append(courier)
            else:
                _, a, b, next_a, next_b = op
                groups[a] = next_a
                groups[b] = next_b
        return groups

    def destroy_repair(groups, rng, drop_count):
        groups = clone_groups(groups)
        if not groups:
            return groups
        keys = list(groups)
        rng.shuffle(keys)
        drop = min(max(1, drop_count), len(keys))
        for key in keys[:drop]:
            del groups[key]
        recovered = cover_unassigned(groups)
        return recovered if valid_groups(recovered) else groups

    def kick_groups(groups, rng, strength):
        groups = clone_groups(groups)
        keys = list(groups)
        if len(keys) < 2:
            return groups
        for _ in range(max(1, strength)):
            a, b = rng.sample(keys, 2)
            if not groups.get(a) or not groups.get(b):
                continue
            ca = rng.choice(groups[a])
            cb = rng.choice(groups[b])
            if cb not in by_key.get(a, {}) or ca not in by_key.get(b, {}):
                continue
            groups[a] = [c for c in groups[a] if c != ca] + [cb]
            groups[b] = [c for c in groups[b] if c != cb] + [ca]
        if not valid_groups(groups):
            return None
        return groups

    def simulated_annealing(groups, rng):
        if not CONFIG.get("use_sa", False):
            return groups
        groups = clone_groups(groups)
        best_groups = clone_groups(groups)
        cur_cost = total_penalty(groups)
        best_cost = cur_cost
        temp = CONFIG.get("sa_temp", 30.0)
        cooling = CONFIG.get("sa_cooling", 0.93)
        iters_per_temp = CONFIG.get("sa_iters_per_temp", 30)
        min_temp = CONFIG.get("sa_min_temp", 0.5)
        while temp > min_temp and time.time() < deadline - 0.04:
            for _ in range(iters_per_temp):
                if time.time() >= deadline - 0.04:
                    break
                keys = list(groups)
                if not keys:
                    break
                op = rng.random()
                trial = clone_groups(groups)
                if op < 0.5 and len(keys) >= 2:
                    a, b = rng.sample(keys, 2)
                    if not trial[a] or not trial[b]:
                        continue
                    ca = rng.choice(trial[a])
                    cb = rng.choice(trial[b])
                    if ca not in by_key[b] or cb not in by_key[a]:
                        continue
                    trial[a] = [c for c in trial[a] if c != ca] + [cb]
                    trial[b] = [c for c in trial[b] if c != cb] + [ca]
                elif op < 0.8:
                    a = rng.choice(keys)
                    candidates = [c for c in by_key[a] if c not in trial[a]]
                    used = set(c for couriers in trial.values() for c in couriers)
                    candidates = [c for c in candidates if c not in used]
                    if not candidates or not trial[a]:
                        continue
                    new_c = rng.choice(candidates)
                    old_c = rng.choice(trial[a])
                    trial[a] = [c for c in trial[a] if c != old_c] + [new_c]
                else:
                    a = rng.choice(keys)
                    if len(trial[a]) <= 1:
                        continue
                    drop = rng.choice(trial[a])
                    trial[a] = [c for c in trial[a] if c != drop]
                if not valid_groups(trial):
                    continue
                new_cost = total_penalty(trial)
                delta = new_cost - cur_cost
                if delta < 0 or rng.random() < math.exp(-delta / max(1e-6, temp)):
                    groups = trial
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_groups = clone_groups(trial)
            temp *= cooling
        return best_groups

    def single_flow_initial():
        if not CONFIG.get("use_flow", False):
            return {}
        if len(all_couriers) < len(all_tasks):
            return {}
        single_rows = [row for row in rows if len(row[0]) == 1]
        if not single_rows:
            return {}
        task_pos = dict((task, i) for i, task in enumerate(all_tasks))
        courier_pos = dict((courier, i) for i, courier in enumerate(all_couriers))
        task_count = len(all_tasks)
        courier_count = len(all_couriers)
        source = task_count + courier_count
        sink = source + 1
        graph = [[] for _ in range(sink + 1)]

        def add_edge(src, dst, cap, cost):
            graph[src].append([dst, cap, cost, len(graph[dst])])
            graph[dst].append([src, 0, -cost, len(graph[src]) - 1])

        for task in all_tasks:
            add_edge(source, task_pos[task], 1, 0.0)
        for courier in all_couriers:
            add_edge(task_count + courier_pos[courier], sink, 1, 0.0)
        for tasks, task_key, courier, score, willingness in single_rows:
            add_edge(task_pos[task_key], task_count + courier_pos[courier], 1, expected_one(tasks, score, willingness))
        potential = [0.0] * len(graph)
        flow = 0
        while flow < task_count and time.time() < deadline - 0.03:
            dist = [10 ** 100] * len(graph)
            prev_node = [-1] * len(graph)
            prev_edge = [-1] * len(graph)
            used = [False] * len(graph)
            dist[source] = 0.0
            for _ in range(len(graph)):
                node = -1
                best_dist = 10 ** 100
                for i, value in enumerate(dist):
                    if not used[i] and value < best_dist:
                        best_dist = value
                        node = i
                if node < 0 or node == sink:
                    break
                used[node] = True
                for edge_index, edge in enumerate(graph[node]):
                    to_node, cap, cost, _ = edge
                    if cap <= 0:
                        continue
                    nd = dist[node] + cost + potential[node] - potential[to_node]
                    if nd < dist[to_node] - 1e-12:
                        dist[to_node] = nd
                        prev_node[to_node] = node
                        prev_edge[to_node] = edge_index
            if prev_node[sink] < 0:
                return {}
            for i, value in enumerate(dist):
                if value < 10 ** 90:
                    potential[i] += value
            node = sink
            while node != source:
                parent = prev_node[node]
                edge_index = prev_edge[node]
                edge = graph[parent][edge_index]
                edge[1] -= 1
                graph[node][edge[3]][1] += 1
                node = parent
            flow += 1
        if flow < task_count:
            return {}
        groups = {}
        for task in all_tasks:
            node = task_pos[task]
            for to_node, cap, _, _ in graph[node]:
                if task_count <= to_node < task_count + courier_count and cap == 0:
                    groups[task] = [all_couriers[to_node - task_count]]
                    break
        return groups

    def bit_count(value):
        count = 0
        while value:
            value &= value - 1
            count += 1
        return count

    def beam_initial():
        if not CONFIG.get("use_beam", False):
            return {}
        if len(all_tasks) > CONFIG.get("beam_task_limit", 42):
            return {}
        task_pos = dict((task, i) for i, task in enumerate(all_tasks))
        courier_pos = dict((courier, i) for i, courier in enumerate(all_couriers))
        full_mask = (1 << len(all_tasks)) - 1
        grouped = defaultdict(list)
        for tasks, task_key, courier, score, willingness in rows:
            grouped[task_key].append((expected_one(tasks, score, willingness), -willingness, score, tasks, task_key, courier))
        compact = []
        keep = CONFIG.get("beam_keep_per_group", 4)
        for _, items in grouped.items():
            seen = set()
            for item in sorted(items)[:keep]:
                courier = item[5]
                if courier in seen:
                    continue
                seen.add(courier)
                expected, _, _, tasks, task_key, courier = item
                mask = 0
                for task in tasks:
                    mask |= 1 << task_pos[task]
                compact.append((mask, 1 << courier_pos[courier], expected, task_key, courier))
        if not compact:
            return {}
        by_task = [[] for _ in all_tasks]
        for index, (task_mask, _, _, _, _) in enumerate(compact):
            mm = task_mask
            while mm:
                bit = mm & -mm
                by_task[bit.bit_length() - 1].append(index)
                mm -= bit
        states = [(0, 0, 0.0, ())]
        width = CONFIG.get("beam_width", 160)
        for _ in range(len(all_tasks)):
            if time.time() >= deadline - 0.03:
                break
            next_states = []
            for task_mask, courier_mask, cost, chosen in states:
                if task_mask == full_mask:
                    next_states.append((task_mask, courier_mask, cost, chosen))
                    continue
                remaining = full_mask ^ task_mask
                pick = None
                best_count = 10 ** 9
                mm = remaining
                while mm:
                    bit = mm & -mm
                    task_index = bit.bit_length() - 1
                    count = 0
                    for cand_index in by_task[task_index]:
                        cmask, cbit, _, _, _ = compact[cand_index]
                        if cmask & task_mask or cbit & courier_mask:
                            continue
                        count += 1
                    if count and count < best_count:
                        best_count = count
                        pick = task_index
                    mm -= bit
                if pick is None:
                    next_states.append((task_mask, courier_mask, cost, chosen))
                    continue
                for cand_index in by_task[pick]:
                    cmask, cbit, expected, task_key, courier = compact[cand_index]
                    if cmask & task_mask or cbit & courier_mask:
                        continue
                    next_states.append((task_mask | cmask, courier_mask | cbit, cost + expected, chosen + ((task_key, courier),)))
            if not next_states:
                break
            dedup = {}
            for state in next_states:
                key = (state[0], state[1])
                if key not in dedup or state[2] < dedup[key][2]:
                    dedup[key] = state
            states = sorted(dedup.values(), key=lambda state: (-bit_count(state[0]), state[2]))[:width]
        best_state = max(states, key=lambda state: (bit_count(state[0]), -state[2]))
        groups = {}
        for task_key, courier in best_state[3]:
            groups[task_key] = [courier]
        return groups

    seed = len(rows) * 1000003 + len(all_tasks) * 1009 + len(all_couriers) * 917 + CONFIG.get("seed", 0)
    rng = random.Random(seed)
    nonlocal_best = [None, None]
    profiles = CONFIG.get("profiles", [])

    def consider(groups):
        if not groups:
            return
        groups = normalize(groups)
        if not groups or not valid_groups(groups):
            return
        rank = (-covered_count(groups), total_penalty(groups))
        if nonlocal_best[1] is None or rank < nonlocal_best[1]:
            nonlocal_best[0] = clone_groups(groups)
            nonlocal_best[1] = rank

    for groups in (single_flow_initial(), beam_initial()):
        if time.time() >= deadline - 0.03:
            break
        if groups:
            groups = fill_extra_couriers(groups)
            groups = local_improve(groups, CONFIG.get("local_rounds", 3))
            consider(groups)

    for profile in profiles:
        if time.time() >= deadline - 0.03:
            break
        groups = weighted_greedy(profile, rng)
        groups = fill_extra_couriers(groups)
        groups = local_improve(groups, CONFIG.get("local_rounds", 3))
        consider(groups)

    if CONFIG.get("use_sa", False) and nonlocal_best[0] is not None:
        sa_groups = simulated_annealing(clone_groups(nonlocal_best[0]), rng)
        consider(sa_groups)

    round_no = 0
    no_improve = 0
    while time.time() < deadline - 0.03:
        round_no += 1
        groups = None
        # 80% weighted_greedy 重启 (主探索), 10% destroy-repair, 10% kick.
        roll = rng.random()
        if nonlocal_best[0] is not None and roll < 0.10:
            base = clone_groups(nonlocal_best[0])
            groups = destroy_repair(base, rng, 1 + (round_no % 3))
        elif nonlocal_best[0] is not None and roll < 0.20:
            base = clone_groups(nonlocal_best[0])
            kicked = kick_groups(base, rng, 1 + (no_improve % 3))
            groups = kicked if kicked else None
        elif nonlocal_best[0] is not None and no_improve >= 6 and CONFIG.get("use_sa", False):
            # \u9577\u671f\u4e0d\u63d0\u5347 + \u5141\u8bb8 SA \u65f6, \u7528 SA \u6270\u52a8 best.
            base = clone_groups(nonlocal_best[0])
            sa_groups = simulated_annealing(base, rng)
            groups = sa_groups if sa_groups and valid_groups(sa_groups) else None
            if groups:
                no_improve = 0
        else:
            base = profiles[round_no % len(profiles)] if profiles else {}
            profile = dict(base)
            profile["coverage_weight"] = profile.get("coverage_weight", 0.0) + rng.random() * CONFIG.get("mutate_coverage", 15.0)
            profile["pair_weight"] = profile.get("pair_weight", 0.0) + rng.random() * CONFIG.get("mutate_pair", 20.0)
            profile["willingness_weight"] = profile.get("willingness_weight", 0.0) + rng.random() * CONFIG.get("mutate_willingness", 12.0)
            profile["random_weight"] = max(profile.get("random_weight", 0.0), CONFIG.get("loop_random_weight", 8.0))
            groups = weighted_greedy(profile, rng)
        if not groups:
            no_improve += 1
            continue
        groups = fill_extra_couriers(groups)
        groups = local_improve(groups, CONFIG.get("loop_local_rounds", 1))
        prev_rank = nonlocal_best[1]
        consider(groups)
        if nonlocal_best[1] == prev_rank:
            no_improve += 1
        else:
            no_improve = 0

    if nonlocal_best[0] is None:
        return []
    answer = []
    for task_key in sorted(nonlocal_best[0]):
        couriers = list(dict.fromkeys(nonlocal_best[0][task_key]))
        couriers.sort(key=lambda courier: (-by_key[task_key][courier][1], by_key[task_key][courier][0]))
        answer.append((task_key, couriers))
    return answer
