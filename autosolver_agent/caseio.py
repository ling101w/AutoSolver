"""Case loading and shared scoring primitives."""

from __future__ import annotations

import os
import statistics
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

from autosolver_agent.models import Case, ParsedCase, ParseDiagnostic


class CaseParseError(RuntimeError):
    """Raised when strict case parsing encounters diagnostics."""

    def __init__(self, diagnostics: List[ParseDiagnostic]) -> None:
        self.diagnostics = diagnostics
        summary = "; ".join(item.message for item in diagnostics[:3])
        if len(diagnostics) > 3:
            summary += f"; and {len(diagnostics) - 3} more"
        super().__init__(summary or "case parse failed")


def parse_case(text: str) -> ParsedCase:
    parsed, _ = parse_case_with_diagnostics(text)
    return parsed


def parse_case_with_diagnostics(
    text: str,
    *,
    case_name: Optional[str] = None,
    path: Optional[str] = None,
    strict: bool = False,
) -> Tuple[ParsedCase, List[ParseDiagnostic]]:
    rows: List[Tuple[Tuple[str, ...], str, str, float, float]] = []
    by_key: DefaultDict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)
    key_tasks: Dict[str, Tuple[str, ...]] = {}
    diagnostics: List[ParseDiagnostic] = []
    for line_number, raw_line in enumerate(text.strip().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("task_id_list"):
            continue
        parts = raw_line.rstrip("\r\n").split("\t")
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
    if strict and diagnostics:
        raise CaseParseError(diagnostics)
    return parsed, diagnostics


def load_cases(paths: Iterable[str], max_cases: int) -> List[Case]:
    cases, _ = load_cases_with_diagnostics(paths, max_cases)
    return cases


def load_cases_with_diagnostics(
    paths: Iterable[str],
    max_cases: int,
    *,
    strict: bool = False,
) -> Tuple[List[Case], List[ParseDiagnostic]]:
    cases: List[Case] = []
    diagnostics: List[ParseDiagnostic] = []
    for path in paths:
        if len(cases) >= max_cases:
            break
        abs_path = os.path.abspath(path)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except UnicodeDecodeError:
            with open(path, "r") as handle:
                text = handle.read()
        except OSError as exc:
            diagnostics.append(
                _diagnostic(
                    "case_read_error",
                    f"could not read case file: {exc}",
                    path=abs_path,
                    case_name=os.path.basename(path),
                )
            )
            continue
        lines = text.splitlines()
        if not lines or "task_id_list" not in lines[0]:
            diagnostics.append(
                _diagnostic(
                    "missing_header",
                    "case file is missing task_id_list header",
                    path=abs_path,
                    case_name=os.path.basename(path),
                    line_number=1 if lines else None,
                    raw_line=lines[0] if lines else None,
                )
            )
            continue
        case = Case(name=os.path.basename(path), path=abs_path, text=text)
        _, parse_diagnostics = parse_case_with_diagnostics(
            text,
            case_name=case.name,
            path=case.path,
            strict=False,
        )
        diagnostics.extend(parse_diagnostics)
        cases.append(case)
    if strict and diagnostics:
        raise CaseParseError(diagnostics)
    return cases, diagnostics


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
            "pair_ratio": 0.0,
            "avg_willingness": 0.0,
            "avg_score": 0.0,
            "high_willingness_ratio": 0.0,
            "low_capacity": False,
        }
    pair_rows = sum(1 for tasks, *_ in rows if len(tasks) > 1)
    willingness_values = [r[4] for r in rows]
    score_values = [r[3] for r in rows]
    task_count = len(parsed.all_tasks)
    courier_count = len(parsed.all_couriers)
    return {
        "row_count": len(rows),
        "task_count": task_count,
        "courier_count": courier_count,
        "pair_ratio": pair_rows / max(1, len(rows)),
        "avg_willingness": statistics.fmean(willingness_values),
        "avg_score": statistics.fmean(score_values),
        "high_willingness_ratio": sum(1 for v in willingness_values if v >= 0.5)
        / max(1, len(willingness_values)),
        "min_score": min(score_values),
        "max_score": max(score_values),
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
                "error": "bad tuple",
            }
        raw_key, couriers = item
        if not isinstance(raw_key, str) or not isinstance(couriers, list) or not couriers:
            return {
                "valid": False,
                "covered": len(used_tasks),
                "penalty": 1_000_000.0 + total,
                "error": "bad fields",
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


def diagnostics_to_dicts(diagnostics: List[ParseDiagnostic]) -> List[Dict[str, Any]]:
    return [
        {
            "code": item.code,
            "message": item.message,
            "path": item.path,
            "case_name": item.case_name,
            "line_number": item.line_number,
            "raw_line": item.raw_line,
        }
        for item in diagnostics
    ]


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
