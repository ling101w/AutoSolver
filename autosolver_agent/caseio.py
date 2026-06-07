"""Case loading and shared scoring primitives."""

from __future__ import annotations

import os
import statistics
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

from autosolver_agent.models import Case, ParsedCase, ParseDiagnostic


class CaseParseError(RuntimeError):
    """Raised when case parsing fails."""

    def __init__(self, diagnostics: List[ParseDiagnostic]) -> None:
        self.diagnostics = diagnostics
        summary = "; ".join(item.message for item in diagnostics[:3])
        if len(diagnostics) > 3:
            summary += f"; and {len(diagnostics) - 3} more"
        super().__init__(summary or "case parse failed")


def parse_case(text: str, *, case_name: Optional[str] = None, path: Optional[str] = None) -> ParsedCase:
    parsed, diagnostics = _parse_case_with_diagnostics(text, case_name=case_name, path=path)
    if diagnostics:
        raise CaseParseError(diagnostics)
    return parsed


def _parse_case_with_diagnostics(
    text: str,
    *,
    case_name: Optional[str] = None,
    path: Optional[str] = None,
) -> Tuple[ParsedCase, List[ParseDiagnostic]]:
    rows: List[Tuple[Tuple[str, ...], str, str, float, float]] = []
    by_key: DefaultDict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)
    key_tasks: Dict[str, Tuple[str, ...]] = {}
    diagnostics: List[ParseDiagnostic] = []
    content_line = 0
    for line_number, raw_line in enumerate(text.strip().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        content_line += 1
        parts = raw_line.rstrip("\r\n").split("\t")
        if content_line == 1 and _is_header_row(parts):
            continue
        if len(parts) < 4:
            diagnostics.append(
                _diagnostic(
                    "malformed_row",
                    "row has fewer than 4 tab-separated fields",
                    path=path,
                    case_name=case_name,
                    line_number=line_number,
                    raw_line=raw_line,
                )
            )
            continue
        raw_key, courier, score, willingness = parts[:4]
        tasks = tuple(t.strip() for t in raw_key.split(",") if t.strip())
        if not tasks:
            diagnostics.append(
                _diagnostic(
                    "empty_task_key",
                    "task_id_list is empty",
                    path=path,
                    case_name=case_name,
                    line_number=line_number,
                    raw_line=raw_line,
                )
            )
            continue
        task_key = ",".join(tasks)
        courier = courier.strip()
        if not courier:
            diagnostics.append(
                _diagnostic(
                    "empty_courier",
                    "courier_id is empty",
                    path=path,
                    case_name=case_name,
                    line_number=line_number,
                    raw_line=raw_line,
                )
            )
            continue
        try:
            score_value = float(score)
            willingness_value = float(willingness)
        except ValueError:
            diagnostics.append(
                _diagnostic(
                    "bad_numeric_value",
                    "total_score or willingness is not numeric",
                    path=path,
                    case_name=case_name,
                    line_number=line_number,
                    raw_line=raw_line,
                )
            )
            continue
        rows.append((tasks, task_key, courier, score_value, willingness_value))
        by_key[task_key][courier] = (score_value, willingness_value)
        key_tasks[task_key] = tasks
    all_tasks = sorted({t for tasks, _, _, _, _ in rows for t in tasks})
    all_couriers = sorted({c for _, _, c, _, _ in rows})
    parsed = ParsedCase(
        rows=rows,
        by_key=dict(by_key),
        key_tasks=key_tasks,
        all_tasks=all_tasks,
        all_couriers=all_couriers,
    )
    return parsed, diagnostics


def _is_header_row(parts: List[str]) -> bool:
    return [part.strip() for part in parts[:4]] == ["task_id_list", "courier_id", "total_score", "willingness"]


def load_cases(paths: Iterable[str], max_cases: int) -> List[Case]:
    cases: List[Case] = []
    for path in paths:
        if len(cases) >= max_cases:
            break
        abs_path = os.path.abspath(path)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            raise CaseParseError(
                [
                    _diagnostic(
                        "case_read_error",
                        f"could not read case file: {exc}",
                        path=abs_path,
                        case_name=os.path.basename(path),
                    )
                ]
            ) from exc
        lines = text.splitlines()
        if not lines or "task_id_list" not in lines[0]:
            raise CaseParseError(
                [
                    _diagnostic(
                        "missing_header",
                        "case file is missing task_id_list header",
                        path=abs_path,
                        case_name=os.path.basename(path),
                        line_number=1 if lines else None,
                        raw_line=lines[0] if lines else None,
                    )
                ]
            )
        case = Case(name=os.path.basename(path), path=abs_path, text=text)
        parse_case(
            text,
            case_name=case.name,
            path=case.path,
        )
        cases.append(case)
    return cases


def discover_case_paths(root: str) -> List[str]:
    paths = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".txt"):
            continue
        if name in ("describe.txt", "example_solution.txt"):
            continue
        paths.append(os.path.abspath(os.path.join(root, name)))
    return paths


def dataset_features(parsed: ParsedCase) -> Dict[str, Any]:
    rows = parsed.rows
    if not rows:
        return {
            "row_count": 0,
            "task_count": 0,
            "courier_count": 0,
            "group_count": 0,
            "pair_ratio": 0.0,
            "bundle_ratio": 0.0,
            "singleton_group_ratio": 0.0,
            "max_group_size": 0,
            "avg_group_size": 0.0,
            "avg_willingness": 0.0,
            "avg_score": 0.0,
            "score_spread": 0.0,
            "score_cv": 0.0,
            "high_willingness_ratio": 0.0,
            "low_willingness_ratio": 0.0,
            "very_low_willingness_ratio": 0.0,
            "avg_candidates_per_group": 0.0,
            "max_candidates_per_group": 0,
            "avg_groups_per_courier": 0.0,
            "max_groups_per_courier": 0,
            "single_task_group_coverage": 0.0,
            "bundle_task_coverage": 0.0,
            "low_capacity": False,
            "capacity_ratio": 0.0,
        }
    pair_rows = sum(1 for tasks, *_ in rows if len(tasks) > 1)
    group_sizes = [len(tasks) for tasks in parsed.key_tasks.values()]
    bundle_groups = sum(1 for tasks in parsed.key_tasks.values() if len(tasks) > 1)
    singleton_groups = sum(1 for tasks in parsed.key_tasks.values() if len(tasks) == 1)
    willingness_values = [r[4] for r in rows]
    score_values = [r[3] for r in rows]
    task_count = len(parsed.all_tasks)
    courier_count = len(parsed.all_couriers)
    group_count = len(parsed.by_key)
    candidates_per_group = [len(couriers) for couriers in parsed.by_key.values()]
    groups_by_courier: Dict[str, set] = {}
    bundle_tasks: Set[str] = set()
    single_tasks: Set[str] = set()
    for task_key, tasks in parsed.key_tasks.items():
        target = bundle_tasks if len(tasks) > 1 else single_tasks
        target.update(tasks)
        for courier in parsed.by_key.get(task_key, {}):
            groups_by_courier.setdefault(courier, set()).add(task_key)
    groups_per_courier = [len(groups) for groups in groups_by_courier.values()]
    avg_score = statistics.fmean(score_values)
    score_stdev = statistics.pstdev(score_values) if len(score_values) > 1 else 0.0
    return {
        "row_count": len(rows),
        "task_count": task_count,
        "courier_count": courier_count,
        "group_count": group_count,
        "pair_ratio": pair_rows / max(1, len(rows)),
        "bundle_ratio": bundle_groups / max(1, group_count),
        "singleton_group_ratio": singleton_groups / max(1, group_count),
        "max_group_size": max(group_sizes) if group_sizes else 0,
        "avg_group_size": statistics.fmean(group_sizes) if group_sizes else 0.0,
        "avg_willingness": statistics.fmean(willingness_values),
        "avg_score": avg_score,
        "score_spread": max(score_values) - min(score_values),
        "score_cv": score_stdev / max(1.0, abs(avg_score)),
        "high_willingness_ratio": sum(1 for v in willingness_values if v >= 0.5)
        / max(1, len(willingness_values)),
        "low_willingness_ratio": sum(1 for v in willingness_values if v < 0.3)
        / max(1, len(willingness_values)),
        "very_low_willingness_ratio": sum(1 for v in willingness_values if v < 0.16)
        / max(1, len(willingness_values)),
        "min_score": min(score_values),
        "max_score": max(score_values),
        "avg_candidates_per_group": statistics.fmean(candidates_per_group) if candidates_per_group else 0.0,
        "max_candidates_per_group": max(candidates_per_group) if candidates_per_group else 0,
        "avg_groups_per_courier": statistics.fmean(groups_per_courier) if groups_per_courier else 0.0,
        "max_groups_per_courier": max(groups_per_courier) if groups_per_courier else 0,
        "single_task_group_coverage": len(single_tasks) / max(1, task_count),
        "bundle_task_coverage": len(bundle_tasks) / max(1, task_count),
        "low_capacity": courier_count < task_count,
        "capacity_ratio": courier_count / max(1, task_count),
    }


def aggregate_features(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not features:
        return {"case_count": 0}
    keys = sorted({key for item in features for key in item})
    aggregate: Dict[str, Any] = {"case_count": len(features)}
    for key in keys:
        values = [item[key] for item in features if key in item]
        if values and all(isinstance(v, bool) for v in values):
            aggregate[key] = any(values)
        elif values and all(isinstance(v, (int, float)) for v in values):
            aggregate[key] = round(statistics.fmean(values), 6)
        else:
            aggregate[key] = values
    return aggregate


def penalty_for_group(parsed: ParsedCase, task_key: str, couriers: List[str]) -> float:
    fallback = 100.0 * len(parsed.key_tasks[task_key])
    reject_prob = 1.0
    weighted_score = 0.0
    weight = 0.0
    data = parsed.by_key[task_key]
    for courier in couriers:
        score, willingness = data[courier]
        reject_prob *= 1.0 - willingness
        weighted_score += willingness * score
        weight += willingness
    if weight <= 0.0:
        return fallback
    return reject_prob * fallback + (1.0 - reject_prob) * weighted_score / weight


def score_answer(parsed: ParsedCase, answer: Any) -> Dict[str, Any]:
    if not isinstance(answer, list):
        return {"valid": False, "covered": 0, "penalty": 1_000_000.0, "error": "answer is not list"}
    used_tasks: Set[str] = set()
    used_couriers: Set[str] = set()
    total = 0.0
    for item in answer:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            return {
                "valid": False,
                "covered": len(used_tasks),
                "penalty": 1_000_000.0 + total,
                "error": "bad tuple: each answer item must be (task_id_list, courier_ids)",
            }
        raw_key, couriers = item
        if not isinstance(raw_key, str) or not isinstance(couriers, list) or not couriers:
            return {
                "valid": False,
                "covered": len(used_tasks),
                "penalty": 1_000_000.0 + total,
                "error": "bad fields: task_id_list must be str and courier_ids must be a non-empty list[str]",
            }
        tasks = tuple(t.strip() for t in raw_key.split(",") if t.strip())
        task_key = ",".join(tasks)
        if task_key not in parsed.key_tasks:
            return {
                "valid": False,
                "covered": len(used_tasks),
                "penalty": 1_000_000.0 + total,
                "error": "unknown task group",
            }
        for task in parsed.key_tasks[task_key]:
            if task in used_tasks:
                return {
                    "valid": False,
                    "covered": len(used_tasks),
                    "penalty": 1_000_000.0 + total,
                    "error": "duplicate task",
                }
        local = set()
        for courier in couriers:
            if courier in local or courier in used_couriers:
                return {
                    "valid": False,
                    "covered": len(used_tasks),
                    "penalty": 1_000_000.0 + total,
                    "error": "duplicate courier",
                }
            if courier not in parsed.by_key[task_key]:
                return {
                    "valid": False,
                    "covered": len(used_tasks),
                    "penalty": 1_000_000.0 + total,
                    "error": "invalid courier",
                }
            local.add(courier)
        total += penalty_for_group(parsed, task_key, couriers)
        used_tasks.update(parsed.key_tasks[task_key])
        used_couriers.update(couriers)
    total += 100.0 * (len(parsed.all_tasks) - len(used_tasks))
    return {"valid": True, "covered": len(used_tasks), "penalty": total, "error": None}


def _diagnostic(
    code: str,
    message: str,
    *,
    path: Optional[str] = None,
    case_name: Optional[str] = None,
    line_number: Optional[int] = None,
    raw_line: Optional[str] = None,
) -> ParseDiagnostic:
    return ParseDiagnostic(
        code=code,
        message=message,
        path=path,
        case_name=case_name,
        line_number=line_number,
        raw_line=raw_line,
    )
