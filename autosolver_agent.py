from __future__ import print_function

from collections import defaultdict
import argparse
import copy
import json
import multiprocessing
import os
import random
import time
import traceback


SOLVER_TEMPLATE = r'''
from collections import defaultdict
import random
import time

CONFIG = __CONFIG__


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
        return sum(penalty(task_key, couriers) for task_key, couriers in groups.items())

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

    def consider(groups):
        nonlocal_best[0] = nonlocal_best[0]
        if not groups:
            return
        groups = normalize(groups)
        if not groups or not valid_groups(groups):
            return
        rank = (-covered_count(groups), total_penalty(groups))
        if nonlocal_best[1] is None or rank < nonlocal_best[1]:
            nonlocal_best[0] = clone_groups(groups)
            nonlocal_best[1] = rank

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

    round_no = 0
    while time.time() < deadline - 0.03:
        round_no += 1
        if nonlocal_best[0] is not None and round_no % 5 == 0:
            groups = clone_groups(nonlocal_best[0])
            keys = list(groups)
            rng.shuffle(keys)
            for key in keys[:1 + round_no % 3]:
                if key in groups:
                    del groups[key]
            groups = cover_unassigned(groups)
        else:
            base = profiles[round_no % len(profiles)] if profiles else {}
            profile = dict(base)
            profile["coverage_weight"] = profile.get("coverage_weight", 0.0) + rng.random() * CONFIG.get("mutate_coverage", 15.0)
            profile["pair_weight"] = profile.get("pair_weight", 0.0) + rng.random() * CONFIG.get("mutate_pair", 20.0)
            profile["willingness_weight"] = profile.get("willingness_weight", 0.0) + rng.random() * CONFIG.get("mutate_willingness", 12.0)
            profile["random_weight"] = max(profile.get("random_weight", 0.0), CONFIG.get("loop_random_weight", 8.0))
            groups = weighted_greedy(profile, rng)
        if not groups:
            continue
        groups = fill_extra_couriers(groups)
        groups = local_improve(groups, CONFIG.get("loop_local_rounds", 1))
        consider(groups)

    if nonlocal_best[0] is None:
        return []
    answer = []
    for task_key in sorted(nonlocal_best[0]):
        couriers = list(dict.fromkeys(nonlocal_best[0][task_key]))
        couriers.sort(key=lambda courier: (-by_key[task_key][courier][1], by_key[task_key][courier][0]))
        answer.append((task_key, couriers))
    return answer
'''


def _candidate_worker(code, case_text, queue):
    try:
        namespace = {}
        compiled = compile(code, "<candidate_solver>", "exec")
        exec(compiled, namespace)
        if "solve" not in namespace:
            queue.put(("error", None, 0.0, "missing solve"))
            return
        start = time.time()
        answer = namespace["solve"](case_text)
        elapsed = time.time() - start
        queue.put(("ok", answer, elapsed, None))
    except Exception:
        queue.put(("error", None, 0.0, traceback.format_exc()))


class AutoSolverAgent(object):
    def __init__(self, case_paths=None, output_path="generated_submit_solution.py", reference_solver_path="submit_solution.py", budget_seconds=90.0, per_case_timeout=10.0, seed=20260524, max_cases=3):
        self.case_paths = case_paths or []
        self.output_path = output_path
        self.reference_solver_path = reference_solver_path
        self.budget_seconds = budget_seconds
        self.per_case_timeout = per_case_timeout
        self.seed = seed
        self.max_cases = max_cases
        self.rng = random.Random(seed)
        self.cases = []
        self.history = []
        self.best_result = None
        self.reference_result = None

    def run(self):
        self.cases = self.load_cases()
        if not self.cases:
            self.cases = self.synthetic_cases()
        deadline = time.time() + self.budget_seconds
        frontier = self.initial_specs()
        self.evaluate_reference()
        iteration = 0
        while frontier and time.time() < deadline:
            iteration += 1
            next_results = []
            for spec in frontier:
                if time.time() >= deadline:
                    break
                result = self.evaluate_spec(spec)
                self.history.append(result)
                next_results.append(result)
                if result["rank"] is not None and (self.best_result is None or result["rank"] < self.best_result["rank"]):
                    self.best_result = result
            frontier = self.propose_next(next_results, iteration, deadline)
        if self.best_result is None:
            spec = self.initial_specs()[0]
            self.best_result = self.evaluate_spec(spec)
        self.write_solver(self.best_result)
        self.write_report()
        return self.summary()

    def load_cases(self):
        paths = list(self.case_paths)
        if not paths:
            for name in sorted(os.listdir(os.getcwd())):
                if not name.endswith(".txt"):
                    continue
                if name in ("describe.txt", "example_solution.txt"):
                    continue
                paths.append(os.path.abspath(name))
        result = []
        for path in paths:
            if len(result) >= self.max_cases:
                break
            try:
                with open(path, "r") as handle:
                    text = handle.read()
                if "task_id_list" not in text.splitlines()[0]:
                    continue
                result.append({"name": os.path.basename(path), "text": text})
            except Exception:
                continue
        return result

    def synthetic_cases(self):
        cases = []
        for case_id in range(3):
            rng = random.Random(self.seed + case_id)
            task_count = 8 + case_id * 4
            courier_count = 9 + case_id * 5
            lines = ["task_id_list\tcourier_id\ttotal_score\twillingness"]
            for task in range(task_count):
                couriers = list(range(courier_count))
                rng.shuffle(couriers)
                for courier in couriers[:min(courier_count, 6)]:
                    score = 5.0 + rng.random() * 80.0
                    willingness = 0.08 + rng.random() * 0.85
                    lines.append("t%d\tc%d\t%.6f\t%.6f" % (task, courier, score, willingness))
            for task in range(task_count - 1):
                if rng.random() < 0.65:
                    pair = "t%d,t%d" % (task, task + 1)
                    couriers = list(range(courier_count))
                    rng.shuffle(couriers)
                    for courier in couriers[:min(courier_count, 4)]:
                        score = 10.0 + rng.random() * 110.0
                        willingness = 0.05 + rng.random() * 0.75
                        lines.append("%s\tc%d\t%.6f\t%.6f" % (pair, courier, score, willingness))
            cases.append({"name": "synthetic_%d" % case_id, "text": "\n".join(lines)})
        return cases

    def initial_specs(self):
        solver_time_limit = min(8.75, max(0.3, self.per_case_timeout - 0.25))
        base = {
            "time_limit": solver_time_limit,
            "seed": self.seed,
            "local_rounds": 3,
            "loop_local_rounds": 1,
            "extra_limit": 80,
            "max_local_keys": 80,
            "mutate_coverage": 15.0,
            "mutate_pair": 20.0,
            "mutate_willingness": 12.0,
            "loop_random_weight": 8.0,
            "beam_width": 150,
            "beam_keep_per_group": 4,
            "beam_task_limit": 42,
            "use_flow": False,
            "use_beam": False,
            "profiles": [],
        }
        profiles = {
            "expected": {"coverage_weight": 0.0, "pair_weight": 0.0, "willingness_weight": 0.0, "score_weight": 0.0, "random_weight": 0.0},
            "coverage": {"coverage_weight": 30.0, "pair_weight": 0.0, "willingness_weight": 5.0, "score_weight": 0.0, "random_weight": 0.0},
            "bundle": {"coverage_weight": 12.0, "pair_weight": 45.0, "willingness_weight": 5.0, "score_weight": 0.0, "random_weight": 0.0},
            "willingness": {"coverage_weight": 6.0, "pair_weight": 8.0, "willingness_weight": 35.0, "score_weight": 0.0, "random_weight": 0.0},
            "score": {"coverage_weight": 0.0, "pair_weight": 0.0, "willingness_weight": 0.0, "score_weight": 0.15, "random_weight": 0.0},
        }
        specs = []
        specs.append(self.make_spec("expected_greedy", base, [profiles["expected"], profiles["coverage"], profiles["willingness"]], False, False))
        specs.append(self.make_spec("coverage_first", base, [profiles["coverage"], profiles["expected"], profiles["bundle"]], False, False))
        specs.append(self.make_spec("bundle_first", base, [profiles["bundle"], profiles["coverage"], profiles["expected"]], False, True))
        specs.append(self.make_spec("flow_expected", base, [profiles["expected"], profiles["willingness"], profiles["score"]], True, False))
        specs.append(self.make_spec("hybrid_full", base, [profiles["expected"], profiles["coverage"], profiles["bundle"], profiles["willingness"], profiles["score"]], True, True))
        specs.append(self.make_spec("low_willingness", base, [{"coverage_weight": 15.0, "pair_weight": 80.0, "willingness_weight": 45.0, "score_weight": 0.0, "random_weight": 0.0}, profiles["bundle"], profiles["coverage"]], False, True))
        return specs

    def make_spec(self, name, base, profiles, use_flow, use_beam):
        config = copy.deepcopy(base)
        config["profiles"] = copy.deepcopy(profiles)
        config["use_flow"] = bool(use_flow)
        config["use_beam"] = bool(use_beam)
        return {"name": name, "config": config, "code": self.render_solver(config)}

    def render_solver(self, config):
        return SOLVER_TEMPLATE.replace("__CONFIG__", repr(config))

    def evaluate_reference(self):
        if not self.reference_solver_path:
            return
        path = self.reference_solver_path
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as handle:
                code = handle.read()
            self.reference_result = self.evaluate_code("reference", code, {"reference": True})
        except Exception:
            self.reference_result = None

    def evaluate_spec(self, spec):
        return self.evaluate_code(spec["name"], spec["code"], spec["config"])

    def evaluate_code(self, name, code, config):
        case_results = []
        total_covered = 0
        total_tasks = 0
        total_penalty = 0.0
        total_runtime = 0.0
        failures = 0
        for case in self.cases:
            run = self.run_candidate(code, case["text"])
            parsed = self.parse_case(case["text"])
            total_tasks += len(parsed["all_tasks"])
            if run["status"] != "ok":
                failures += 1
                penalty = 1000000.0 + 100.0 * len(parsed["all_tasks"])
                case_results.append({"case": case["name"], "status": run["status"], "covered": 0, "tasks": len(parsed["all_tasks"]), "penalty": penalty, "runtime": run.get("runtime", 0.0), "error": run.get("error")})
                total_penalty += penalty
                total_runtime += run.get("runtime", 0.0)
                continue
            scored = self.score_answer(parsed, run["answer"])
            if not scored["valid"]:
                failures += 1
            total_covered += scored["covered"]
            total_penalty += scored["penalty"]
            total_runtime += run["runtime"]
            case_results.append({"case": case["name"], "status": "ok" if scored["valid"] else "invalid", "covered": scored["covered"], "tasks": len(parsed["all_tasks"]), "penalty": scored["penalty"], "runtime": run["runtime"], "error": scored.get("error")})
        rank = (failures, -total_covered, total_penalty, total_runtime)
        return {"name": name, "config": copy.deepcopy(config), "code": code, "rank": rank, "cases": case_results, "total_covered": total_covered, "total_tasks": total_tasks, "total_penalty": total_penalty, "total_runtime": total_runtime, "failures": failures}

    def run_candidate(self, code, case_text):
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=_candidate_worker, args=(code, case_text, queue))
        process.daemon = True
        start = time.time()
        process.start()
        process.join(self.per_case_timeout)
        runtime = time.time() - start
        if process.is_alive():
            process.terminate()
            process.join(0.2)
            return {"status": "timeout", "answer": None, "runtime": runtime, "error": "timeout"}
        if queue.empty():
            return {"status": "error", "answer": None, "runtime": runtime, "error": "empty result"}
        status, answer, child_runtime, error = queue.get()
        return {"status": status, "answer": answer, "runtime": child_runtime or runtime, "error": error}

    def parse_case(self, text):
        rows = []
        by_key = defaultdict(dict)
        key_tasks = {}
        for line in text.strip().splitlines():
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
        all_tasks = sorted(set(t for tasks, _, _, _, _ in rows for t in tasks))
        return {"rows": rows, "by_key": by_key, "key_tasks": key_tasks, "all_tasks": all_tasks}

    def score_answer(self, parsed, answer):
        if not isinstance(answer, list):
            return {"valid": False, "covered": 0, "penalty": 1000000.0, "error": "answer is not list"}
        by_key = parsed["by_key"]
        key_tasks = parsed["key_tasks"]
        used_tasks = set()
        used_couriers = set()
        total = 0.0
        for item in answer:
            if not isinstance(item, tuple) or len(item) != 2:
                return {"valid": False, "covered": len(used_tasks), "penalty": 1000000.0 + total, "error": "bad tuple"}
            raw_key, couriers = item
            if not isinstance(raw_key, str) or not isinstance(couriers, list) or not couriers:
                return {"valid": False, "covered": len(used_tasks), "penalty": 1000000.0 + total, "error": "bad fields"}
            tasks = tuple(t.strip() for t in raw_key.split(",") if t.strip())
            task_key = ",".join(tasks)
            if task_key not in key_tasks:
                return {"valid": False, "covered": len(used_tasks), "penalty": 1000000.0 + total, "error": "unknown task group"}
            for task in key_tasks[task_key]:
                if task in used_tasks:
                    return {"valid": False, "covered": len(used_tasks), "penalty": 1000000.0 + total, "error": "duplicate task"}
            local = set()
            for courier in couriers:
                if courier in local or courier in used_couriers:
                    return {"valid": False, "covered": len(used_tasks), "penalty": 1000000.0 + total, "error": "duplicate courier"}
                if courier not in by_key[task_key]:
                    return {"valid": False, "covered": len(used_tasks), "penalty": 1000000.0 + total, "error": "invalid courier"}
                local.add(courier)
            total += self.penalty(parsed, task_key, couriers)
            used_tasks.update(key_tasks[task_key])
            used_couriers.update(couriers)
        missing = len(parsed["all_tasks"]) - len(used_tasks)
        total += 100.0 * missing
        return {"valid": True, "covered": len(used_tasks), "penalty": total, "error": None}

    def penalty(self, parsed, task_key, couriers):
        fallback = 100.0 * len(parsed["key_tasks"][task_key])
        reject_prob = 1.0
        weighted_score = 0.0
        weight = 0.0
        data = parsed["by_key"][task_key]
        for courier in couriers:
            score, willingness = data[courier]
            reject_prob *= 1.0 - willingness
            weighted_score += willingness * score
            weight += willingness
        if weight <= 0.0:
            return fallback
        return reject_prob * fallback + (1.0 - reject_prob) * weighted_score / weight

    def propose_next(self, results, iteration, deadline):
        usable = [result for result in self.history if result["rank"] is not None]
        usable.sort(key=lambda result: result["rank"])
        parents = usable[:min(4, len(usable))]
        if not parents:
            return []
        proposals = []
        for parent in parents:
            if time.time() >= deadline:
                break
            proposals.append(self.mutate(parent, iteration, "adaptive_%d_%s" % (iteration, parent["name"])))
        return proposals[:6]

    def mutate(self, parent, iteration, name):
        config = copy.deepcopy(parent["config"])
        if config.get("reference"):
            config = self.initial_specs()[0]["config"]
        missing = max(0, parent.get("total_tasks", 0) - parent.get("total_covered", 0))
        if missing > 0:
            config["use_beam"] = True
            config["extra_limit"] = min(160, config.get("extra_limit", 80) + 15)
            config["max_local_keys"] = min(140, config.get("max_local_keys", 80) + 10)
            for profile in config.get("profiles", []):
                profile["coverage_weight"] = profile.get("coverage_weight", 0.0) + 10.0 + 12.0 * self.rng.random()
                profile["pair_weight"] = profile.get("pair_weight", 0.0) + 8.0 * self.rng.random()
        else:
            config["local_rounds"] = min(8, config.get("local_rounds", 3) + 1)
            config["loop_local_rounds"] = min(3, config.get("loop_local_rounds", 1) + (iteration % 2))
            for profile in config.get("profiles", []):
                profile["willingness_weight"] = max(0.0, profile.get("willingness_weight", 0.0) + self.rng.uniform(-8.0, 16.0))
                profile["pair_weight"] = max(0.0, profile.get("pair_weight", 0.0) + self.rng.uniform(-12.0, 18.0))
                profile["score_weight"] = max(0.0, profile.get("score_weight", 0.0) + self.rng.uniform(-0.05, 0.08))
        if self.rng.random() < 0.35:
            config["use_flow"] = not config.get("use_flow", False)
        if self.rng.random() < 0.45:
            config["use_beam"] = not config.get("use_beam", False)
        config["seed"] = self.seed + iteration * 1009 + self.rng.randrange(100000)
        config["loop_random_weight"] = max(1.0, config.get("loop_random_weight", 8.0) + self.rng.uniform(-3.0, 7.0))
        config["mutate_pair"] = max(2.0, config.get("mutate_pair", 20.0) + self.rng.uniform(-5.0, 10.0))
        config["mutate_coverage"] = max(2.0, config.get("mutate_coverage", 15.0) + self.rng.uniform(-4.0, 8.0))
        return {"name": name, "config": config, "code": self.render_solver(config)}

    def write_solver(self, result):
        path = self.output_path
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        with open(path, "w") as handle:
            handle.write(result["code"])

    def write_report(self):
        report_path = self.output_path + ".report.json"
        report = self.summary()
        try:
            with open(report_path, "w") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
        except Exception:
            pass

    def summary(self):
        best = None
        if self.best_result is not None:
            best = {
                "name": self.best_result["name"],
                "rank": self.best_result["rank"],
                "total_covered": self.best_result["total_covered"],
                "total_tasks": self.best_result["total_tasks"],
                "total_penalty": self.best_result["total_penalty"],
                "total_runtime": self.best_result["total_runtime"],
                "failures": self.best_result["failures"],
                "cases": self.best_result["cases"],
            }
        reference = None
        if self.reference_result is not None:
            reference = {
                "name": self.reference_result["name"],
                "rank": self.reference_result["rank"],
                "total_covered": self.reference_result["total_covered"],
                "total_tasks": self.reference_result["total_tasks"],
                "total_penalty": self.reference_result["total_penalty"],
                "total_runtime": self.reference_result["total_runtime"],
                "failures": self.reference_result["failures"],
            }
        return {"output_path": os.path.abspath(self.output_path), "cases": [case["name"] for case in self.cases], "best": best, "reference": reference, "evaluated_candidates": len(self.history)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--out", default="generated_submit_solution.py")
    parser.add_argument("--reference", default="submit_solution.py")
    parser.add_argument("--budget", type=float, default=90.0)
    parser.add_argument("--per-case-timeout", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--max-cases", type=int, default=3)
    args = parser.parse_args()
    agent = AutoSolverAgent(
        case_paths=args.cases,
        output_path=args.out,
        reference_solver_path=args.reference,
        budget_seconds=args.budget,
        per_case_timeout=args.per_case_timeout,
        seed=args.seed,
        max_cases=args.max_cases,
    )
    result = agent.run()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
