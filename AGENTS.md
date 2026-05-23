# Project Notes

## Problem Terminology and Constraints

- A `task` is a single element inside `task_id_list`.
  - Example: in `T0037,T0039`, the tasks are `T0037` and `T0039`.
- A `task_id_list` is an order group. It may contain one task or multiple tasks.
  - Example: `T0037` is a single-task group; `T0037,T0039` is a combined-task group.
- A candidate is one input row: `(task_id_list, courier_id, total_score, willingness)`.
  - It represents assigning that whole `task_id_list` to that one courier.
- One courier can accept at most one `task_id_list`.
  - After a courier is selected for any candidate, no other candidate using that courier may be selected.
- The same `task_id_list` may be assigned to multiple couriers.
  - Example: both `T0037,T0039 -> C028` and `T0037,T0039 -> C034` may be selected if the couriers are different.
  - This is the problem statement's multi-courier assignment mechanism: multiple couriers may be offered the same order group, and the first one to accept gets it.
- Conflict checks involving orders must happen at the individual `task` level, not the raw `task_id_list` string level.
  - For example, `T0037,T0039` overlaps with `T0037`.
  - Whether that overlap is allowed depends on the solver's explicit multi-courier assignment model.
- The baseline solver currently enforces that each individual `task` is assigned at most once, so it does not use the multi-courier mechanism.
- Lower `total_score` is better after satisfying the primary objective of maximizing accepted or covered tasks.
- `willingness` is the predicted willingness/acceptance signal for the candidate. The baseline parses it but does not use it.

## Objective Function

- The judge-style objective for one `task_id_list` is an expected penalty over the couriers assigned to that same `task_id_list`.
- If all assigned couriers reject the `task_id_list`, the fallback penalty is `100.0 * number_of_tasks_in_task_id_list`.
- The probability that all assigned couriers reject is:

  ```text
  reject_prob = product(1.0 - willingness_i)
  ```

- If at least one assigned courier accepts, the accepted score is approximated as the willingness-weighted average of the assigned couriers' `total_score` values:

  ```text
  accepted_score = sum(willingness_i * total_score_i) / sum(willingness_i)
  ```

- The expected penalty for one `task_id_list` is:

  ```text
  penalty = reject_prob * fallback + (1.0 - reject_prob) * accepted_score
  ```

- If the total willingness weight is `0.0`, use the fallback penalty directly.
- Only couriers that have an input candidate row for that exact `task_id_list` may be assigned to it.
- A solution's objective value is computed by summing this expected penalty over the selected `task_id_list` groups.
