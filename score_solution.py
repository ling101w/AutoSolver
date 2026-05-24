from pathlib import Path
import sys
import time

import solver


def score(input_path: Path) -> int:
    input_text = input_path.read_text()

    start = time.perf_counter()
    answer = solver.solve(input_text)
    elapsed = time.perf_counter() - start

    data = solver.parse_input(input_text)
    group_by_name = {name: i for i, name in enumerate(data.group_names)}
    courier_by_name = {name: i for i, name in enumerate(data.courier_names)}

    errors = []
    assignments = []
    used_task_mask = 0
    used_couriers = set()

    for row_idx, item in enumerate(answer, 1):
        if not isinstance(item, tuple) or len(item) != 2:
            errors.append(f"row {row_idx}: invalid answer item {item!r}")
            continue

        group_name, courier_names = item
        group = group_by_name.get(group_name)
        if group is None:
            errors.append(f"row {row_idx}: unknown group {group_name!r}")
            continue
        if not courier_names:
            errors.append(f"row {row_idx}: empty courier list for {group_name!r}")
            continue
        if used_task_mask & data.group_masks[group]:
            errors.append(f"row {row_idx}: task overlap at {group_name!r}")
            continue

        local_seen = set()
        courier_ids = []
        for courier_name in courier_names:
            courier = courier_by_name.get(courier_name)
            if courier is None:
                errors.append(
                    f"row {row_idx}: unknown courier {courier_name!r} "
                    f"for {group_name!r}"
                )
                continue
            if courier in local_seen:
                errors.append(
                    f"row {row_idx}: duplicate courier {courier_name!r} "
                    f"in {group_name!r}"
                )
                continue
            if courier in used_couriers:
                errors.append(
                    f"row {row_idx}: courier {courier_name!r} reused by "
                    f"{group_name!r}"
                )
                continue
            if courier not in data.cand_by_group[group]:
                errors.append(
                    f"row {row_idx}: courier {courier_name!r} cannot serve "
                    f"{group_name!r}"
                )
                continue
            local_seen.add(courier)
            courier_ids.append(courier)

        if len(courier_ids) != len(courier_names):
            continue
        assignments.append((group, courier_ids))
        used_task_mask |= data.group_masks[group]
        used_couriers.update(courier_ids)

    state = solver.State(data)
    for group, courier_ids in assignments:
        for courier in courier_ids:
            solver.add_courier(state, group, courier)

    valid = not errors
    print(f"input: {input_path}")
    print(f"valid: {valid}")
    print(f"covered: {state.covered_count}/{data.total_tasks}")
    print(f"groups: {len(assignments)}")
    print(f"couriers: {len(used_couriers)}")
    print(f"penalty: {state.total_penalty:.6f}")
    print(f"energy: {state.energy:.6f}")
    print(f"elapsed: {elapsed:.3f}s")

    if errors:
        print("errors:")
        for error in errors[:20]:
            print(f"- {error}")
        if len(errors) > 20:
            print(f"- ... {len(errors) - 20} more")
    return 0 if valid else 1


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python score_solution.py INPUT", file=sys.stderr)
        return 2
    return score(Path(sys.argv[1]))


if __name__ == "__main__":
    raise SystemExit(main())
