from collections import defaultdict


def solve(input_text: str) -> list:
    rows = []
    for line in input_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("task_id_list"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        task_key, courier, score, willingness = parts[:4]
        try:
            tasks = tuple(t.strip() for t in task_key.split(",") if t.strip())
            rows.append((tasks, ",".join(tasks), courier.strip(), float(score), float(willingness)))
        except ValueError:
            pass

    if not rows:
        return []

    all_tasks = sorted({t for tasks, _, _, _, _ in rows for t in tasks})
    all_couriers = sorted({c for _, _, c, _, _ in rows})

    by_key = defaultdict(dict)
    key_tasks = {}
    for tasks, task_key, courier, score, willingness in rows:
        by_key[task_key][courier] = (score, willingness)
        key_tasks[task_key] = tasks

    def penalty(task_key, courier_list):
        """Expected penalty for one task group.

        If all assigned couriers reject, the judge charges 100 per task.
        Otherwise the accepted score is the willingness-weighted average score
        of assigned couriers.
        """

        fallback = 100.0 * len(key_tasks[task_key])
        reject_prob = 1.0
        weighted_score = 0.0
        weight = 0.0
        data = by_key[task_key]
        for courier in courier_list:
            if courier in data:
                score, willingness = data[courier]
                reject_prob *= 1.0 - willingness
                weighted_score += willingness * score
                weight += willingness
        if weight <= 0.0:
            return fallback
        return reject_prob * fallback + (1.0 - reject_prob) * weighted_score / weight

    def covered_count(groups):
        seen = set()
        for task_key in groups:
            seen.update(key_tasks[task_key])
        return len(seen)

    def total_penalty(groups):
        return sum(penalty(task_key, couriers) for task_key, couriers in groups.items())

    def clone_groups(groups):
        return {task_key: list(couriers) for task_key, couriers in groups.items()}

    def fill_extra_couriers(groups):
        groups = clone_groups(groups)
        used = {c for couriers in groups.values() for c in couriers}

        while True:
            best = None
            for courier in all_couriers:
                if courier in used:
                    continue
                for task_key, couriers in groups.items():
                    if courier not in by_key[task_key]:
                        continue
                    old = penalty(task_key, couriers)
                    new = penalty(task_key, couriers + [courier])
                    delta = new - old
                    if best is None or delta < best[0]:
                        best = (delta, task_key, courier)

            if best is None or best[0] >= -1e-12:
                break
            _, task_key, courier = best
            groups[task_key].append(courier)
            used.add(courier)

        return groups

    def improve(groups, max_rounds=8):
        groups = clone_groups(groups)
        keys = list(groups)

        for _ in range(max_rounds):
            best = (0.0, None)

            for a in keys:
                if len(groups[a]) <= 1:
                    continue
                old_a = penalty(a, groups[a])
                for courier in list(groups[a]):
                    without = [c for c in groups[a] if c != courier]
                    new_a = penalty(a, without)
                    delta = new_a - old_a
                    if delta < best[0]:
                        best = (delta, ("drop", a, courier))
                    for b in keys:
                        if a == b or courier not in by_key[b]:
                            continue
                        old_b = penalty(b, groups[b])
                        new_b = penalty(b, groups[b] + [courier])
                        delta = new_a + new_b - old_a - old_b
                        if delta < best[0]:
                            best = (delta, ("move", a, b, courier))

            for i, a in enumerate(keys):
                old_a = penalty(a, groups[a])
                for b in keys[i + 1 :]:
                    old_b = penalty(b, groups[b])
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
                                best = (delta, ("swap", a, b, ca, cb))

            if best[1] is None:
                break

            op = best[1]
            if op[0] == "drop":
                _, a, courier = op
                groups[a].remove(courier)
                groups = fill_extra_couriers(groups)
                keys = list(groups)
            elif op[0] == "move":
                _, a, b, courier = op
                groups[a].remove(courier)
                groups[b].append(courier)
            else:
                _, a, b, ca, cb = op
                groups[a].remove(ca)
                groups[b].remove(cb)
                groups[a].append(cb)
                groups[b].append(ca)

        return groups

    def greedy_initial(rank):
        groups = {}
        used_tasks = set()
        used_couriers = set()

        for tasks, task_key, courier, score, willingness in sorted(rows, key=rank):
            if courier in used_couriers:
                continue
            if any(t in used_tasks for t in tasks):
                continue
            groups[task_key] = [courier]
            used_couriers.add(courier)
            used_tasks.update(tasks)

        # Try to cover any missed task without breaking disjoint task coverage.
        changed = True
        while changed:
            changed = False
            best = None
            for tasks, task_key, courier, score, willingness in rows:
                if courier in used_couriers:
                    continue
                if any(t in used_tasks for t in tasks):
                    continue
                gain = sum(1 for t in tasks if t not in used_tasks)
                if gain <= 0:
                    continue
                one_shot = willingness * score + (1.0 - willingness) * 100.0 * len(tasks)
                item = (-gain, one_shot / len(tasks), one_shot, task_key, courier, tasks)
                if best is None or item < best:
                    best = item
            if best is not None:
                _, _, _, task_key, courier, tasks = best
                groups[task_key] = [courier]
                used_couriers.add(courier)
                used_tasks.update(tasks)
                changed = True

        return groups

    def beam_initial(width=160, per_key=3):
        if len(all_tasks) > 45:
            return {}

        task_pos = {task: i for i, task in enumerate(all_tasks)}
        courier_pos = {courier: i for i, courier in enumerate(all_couriers)}
        full_mask = (1 << len(all_tasks)) - 1

        compact = []
        grouped_rows = defaultdict(list)
        for tasks, task_key, courier, score, willingness in rows:
            one_shot = expected_one(tasks, score, willingness)
            grouped_rows[task_key].append((one_shot, -willingness, score, tasks, task_key, courier))

        for task_key, items in grouped_rows.items():
            chosen = []
            seen = set()
            for item in sorted(items)[:per_key]:
                if item[5] not in seen:
                    chosen.append(item)
                    seen.add(item[5])
            for item in sorted(items, key=lambda x: (x[2], x[0]))[:1]:
                if item[5] not in seen:
                    chosen.append(item)
                    seen.add(item[5])
            for one_shot, _, score, tasks, task_key, courier in chosen:
                task_mask = 0
                for task in tasks:
                    task_mask |= 1 << task_pos[task]
                compact.append(
                    (
                        task_mask,
                        1 << courier_pos[courier],
                        one_shot,
                        task_key,
                        courier,
                    )
                )

        by_task = [[] for _ in all_tasks]
        for index, (task_mask, _, _, _, _) in enumerate(compact):
            mm = task_mask
            while mm:
                bit = mm & -mm
                by_task[bit.bit_length() - 1].append(index)
                mm -= bit
        for indexes in by_task:
            indexes.sort(key=lambda i: (compact[i][2] / compact[i][0].bit_count(), compact[i][2]))

        # state: (covered_task_mask, used_courier_mask, base_cost, ((key, courier), ...))
        states = [(0, 0, 0.0, ())]
        best_full = None

        for _ in range(len(all_tasks)):
            next_states = []
            for task_mask, courier_mask, cost, chosen in states:
                if task_mask == full_mask:
                    if best_full is None or cost < best_full[2]:
                        best_full = (task_mask, courier_mask, cost, chosen)
                    next_states.append((task_mask, courier_mask, cost, chosen))
                    continue

                remaining = full_mask ^ task_mask
                pick = None
                best_count = 10**9
                mm = remaining
                while mm:
                    bit = mm & -mm
                    task_index = bit.bit_length() - 1
                    count = 0
                    for cand_index in by_task[task_index]:
                        cmask, cbit, _, _, _ = compact[cand_index]
                        if cmask & task_mask:
                            continue
                        if cbit & courier_mask:
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
                    cmask, cbit, one_shot, task_key, courier = compact[cand_index]
                    if cmask & task_mask:
                        continue
                    if cbit & courier_mask:
                        continue
                    next_states.append(
                        (
                            task_mask | cmask,
                            courier_mask | cbit,
                            cost + one_shot,
                            chosen + ((task_key, courier),),
                        )
                    )

            if not next_states:
                break

            dedup = {}
            for state in next_states:
                key = (state[0], state[1])
                if key not in dedup or state[2] < dedup[key][2]:
                    dedup[key] = state
            states = sorted(
                dedup.values(),
                key=lambda st: (-(st[0].bit_count()), st[2] / max(1, st[0].bit_count()), st[2]),
            )[:width]

        for state in states:
            if state[0] == full_mask and (best_full is None or state[2] < best_full[2]):
                best_full = state

        if best_full is None:
            best_state = max(states, key=lambda st: (st[0].bit_count(), -st[2]))
        else:
            best_state = best_full

        groups = {}
        for task_key, courier in best_state[3]:
            groups[task_key] = [courier]
        return groups

    def repartition(groups, max_rounds=3):
        groups = clone_groups(groups)

        def partitions(task_list):
            task_list = tuple(sorted(task_list))
            result = []

            def rec(rest, current):
                if not rest:
                    result.append(current[:])
                    return
                first = rest[0]
                single = first
                if single in by_key:
                    rec(rest[1:], current + [single])
                for i in range(1, len(rest)):
                    pair = first + "," + rest[i]
                    if pair in by_key:
                        nxt = list(rest[1:i] + rest[i + 1 :])
                        rec(tuple(nxt), current + [pair])

            rec(task_list, [])
            return result

        def best_assignment(part_list, courier_list):
            best = None
            part_count = len(part_list)
            assigned = [[] for _ in part_list]

            def rec(index):
                nonlocal best
                if index == len(courier_list):
                    if any(not bucket for bucket in assigned):
                        return
                    value = sum(penalty(part_list[i], assigned[i]) for i in range(part_count))
                    if best is None or value < best[0]:
                        best = (value, [bucket[:] for bucket in assigned])
                    return

                courier = courier_list[index]
                rec(index + 1)
                for i, part in enumerate(part_list):
                    if courier in by_key[part]:
                        assigned[i].append(courier)
                        rec(index + 1)
                        assigned[i].pop()

            rec(0)
            return best

        for _ in range(max_rounds):
            keys = list(groups)
            best = (0.0, None)

            for i, a in enumerate(keys):
                for b in keys[i + 1 :]:
                    union_tasks = tuple(sorted(key_tasks[a] + key_tasks[b]))
                    if len(set(union_tasks)) != len(union_tasks) or len(union_tasks) > 4:
                        continue
                    couriers = list(dict.fromkeys(groups[a] + groups[b]))
                    if len(couriers) > 4:
                        continue
                    old_value = penalty(a, groups[a]) + penalty(b, groups[b])
                    for part_list in partitions(union_tasks):
                        if len(part_list) > len(couriers):
                            continue
                        if set(part_list) == {a, b}:
                            continue
                        candidate = best_assignment(part_list, couriers)
                        if candidate is None:
                            continue
                        new_value, buckets = candidate
                        delta = new_value - old_value
                        if delta < best[0]:
                            best = (delta, (a, b, part_list, buckets))

            if best[1] is None:
                break

            a, b, part_list, buckets = best[1]
            del groups[a]
            del groups[b]
            for part, bucket in zip(part_list, buckets):
                groups[part] = bucket

        return groups

    def single_initial():
        single_rows = [r for r in rows if len(r[0]) == 1]
        if len(all_couriers) < len(all_tasks):
            return {}

        task_pos = {task: i for i, task in enumerate(all_tasks)}
        courier_pos = {courier: i for i, courier in enumerate(all_couriers)}
        task_count = len(all_tasks)
        courier_count = len(all_couriers)

        graph = [[] for _ in range(task_count + courier_count + 2)]
        source = task_count + courier_count
        sink = source + 1

        def add_edge(src, dst, cap, cost):
            graph[src].append([dst, cap, cost, len(graph[dst])])
            graph[dst].append([src, 0, -cost, len(graph[src]) - 1])

        for task in all_tasks:
            add_edge(source, task_pos[task], 1, 0.0)
        for courier in all_couriers:
            add_edge(task_count + courier_pos[courier], sink, 1, 0.0)
        for tasks, task_key, courier, score, willingness in single_rows:
            one_shot = willingness * score + (1.0 - willingness) * 100.0
            add_edge(task_pos[task_key], task_count + courier_pos[courier], 1, one_shot)

        potential = [0.0] * len(graph)
        flow = 0
        while flow < task_count:
            dist = [10**100] * len(graph)
            prev_node = [-1] * len(graph)
            prev_edge = [-1] * len(graph)
            dist[source] = 0.0
            used = [False] * len(graph)

            for _ in range(len(graph)):
                node = -1
                best_dist = 10**100
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
                if value < 10**90:
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

        groups = {}
        for task in all_tasks:
            node = task_pos[task]
            for to_node, cap, _, _ in graph[node]:
                if task_count <= to_node < task_count + courier_count and cap == 0:
                    groups[task] = [all_couriers[to_node - task_count]]
                    break
        return groups if covered_count(groups) == len(all_tasks) else {}

    def expected_one(tasks, score, willingness):
        return willingness * score + (1.0 - willingness) * 100.0 * len(tasks)

    starts = []
    beam = beam_initial()
    if beam:
        starts.append(beam)

    one = single_initial()
    if one:
        starts.append(one)

    starts.append(
        greedy_initial(
            lambda r: (
                expected_one(r[0], r[3], r[4]) / len(r[0]),
                expected_one(r[0], r[3], r[4]),
                -len(r[0]),
            )
        )
    )
    starts.append(
        greedy_initial(
            lambda r: (
                -len(r[0]),
                expected_one(r[0], r[3], r[4]) / len(r[0]),
                expected_one(r[0], r[3], r[4]),
            )
        )
    )
    starts.append(greedy_initial(lambda r: (r[3] / len(r[0]), r[3], -len(r[0]))))
    starts.append(greedy_initial(lambda r: (-r[4], r[3] / len(r[0]), r[3])))

    best_groups = None
    best_rank = None
    for start_groups in starts:
        if not start_groups:
            continue
        variants = [clone_groups(start_groups), repartition(start_groups)]
        for groups in variants:
            groups = fill_extra_couriers(groups)
            groups = improve(groups)
            groups = improve(groups)
            rank = (-covered_count(groups), total_penalty(groups))
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_groups = groups

    if best_groups is None:
        return []

    answer = []
    for task_key in sorted(best_groups):
        couriers = list(dict.fromkeys(best_groups[task_key]))
        couriers.sort(key=lambda c: (-by_key[task_key][c][1], by_key[task_key][c][0]))
        answer.append((task_key, couriers))
    return answer
