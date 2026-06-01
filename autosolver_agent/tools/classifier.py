"""Instance classifier tool."""

from __future__ import annotations

from typing import Any, Dict, List

from autosolver_agent.caseio import aggregate_features, dataset_features
from autosolver_agent.models import Case, ParsedCase


class InstanceClassifier:
    def classify(self, cases: List[Case], parsed_cases: List[ParsedCase]) -> Dict[str, Any]:
        per_case = []
        features = []
        for case, parsed in zip(cases, parsed_cases):
            item = dataset_features(parsed)
            item["name"] = case.name
            item["tags"] = self._tags(item)
            features.append({k: v for k, v in item.items() if k not in {"name", "tags"}})
            per_case.append(item)
        aggregate = aggregate_features(features)
        aggregate["tags"] = self._tags(aggregate)
        aggregate["recommended_focus"] = self._recommended_focus(aggregate)
        return {"aggregate": aggregate, "cases": per_case}

    def _tags(self, features: Dict[str, Any]) -> List[str]:
        tags = []
        task_count = int(features.get("task_count", 0) or 0)
        courier_count = int(features.get("courier_count", 0) or 0)
        pair_ratio = float(features.get("pair_ratio", 0.0) or 0.0)
        avg_willingness = float(features.get("avg_willingness", 0.0) or 0.0)
        capacity_ratio = float(features.get("capacity_ratio", 0.0) or 0.0)
        if task_count <= 45:
            tags.append("small_task_count")
        elif task_count >= 120:
            tags.append("large_task_count")
        if courier_count < task_count:
            tags.append("scarce_couriers")
        elif capacity_ratio >= 1.8:
            tags.append("many_couriers")
        if pair_ratio >= 0.2:
            tags.append("high_pair_ratio")
        else:
            tags.append("low_pair_ratio")
        if avg_willingness < 0.3:
            tags.append("low_willingness")
        elif avg_willingness >= 0.55:
            tags.append("high_willingness")
        return tags or ["general"]

    def _recommended_focus(self, features: Dict[str, Any]) -> List[str]:
        tags = set(features.get("tags", []))
        focus = ["local_search_repair", "expected_greedy"]
        if "high_pair_ratio" in tags or "scarce_couriers" in tags:
            focus.insert(0, "bundle_first")
        if "low_willingness" in tags:
            focus.insert(0, "willingness_weighted")
        if "small_task_count" in tags:
            focus.append("beam_cover")
        if "many_couriers" in tags:
            focus.append("flow_single_initial")
        return list(dict.fromkeys(focus))
