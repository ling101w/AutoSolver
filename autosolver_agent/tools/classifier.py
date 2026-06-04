"""Instance classifier tool."""

from __future__ import annotations

from typing import Any, Dict, List

from autosolver_agent.caseio import aggregate_features, dataset_features
from autosolver_agent.models import Case, ParsedCase

FEATURE_LIBRARY = [
    {
        "name": "capacity_shape",
        "signals": ["capacity_ratio", "low_capacity", "courier_count", "task_count"],
        "uses": ["choose between scarce-courier matching, bundle-first cover, and many-courier flow seeds"],
    },
    {
        "name": "bundle_structure",
        "signals": ["pair_ratio", "bundle_ratio", "bundle_task_coverage", "max_group_size"],
        "uses": ["detect when pair replacement, repartition, or bundle-first construction can beat singleton cover"],
    },
    {
        "name": "candidate_density",
        "signals": ["avg_candidates_per_group", "max_candidates_per_group", "avg_groups_per_courier"],
        "uses": ["size local move neighborhoods and decide whether swaps, extras, or tabu search are affordable"],
    },
    {
        "name": "acceptance_risk",
        "signals": ["avg_willingness", "low_willingness_ratio", "very_low_willingness_ratio", "high_willingness_ratio"],
        "uses": ["decide whether to add secondary couriers, bias randomized starts, or conserve couriers"],
    },
    {
        "name": "objective_ruggedness",
        "signals": ["score_spread", "score_cv", "row_count", "group_count"],
        "uses": ["detect when greedy tie-breaking is fragile and multi-start or local search is likely useful"],
    },
]


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
        aggregate["feature_profile"] = self._feature_profile(aggregate)
        return {"aggregate": aggregate, "cases": per_case, "feature_library": self.feature_library()}

    def feature_library(self) -> List[Dict[str, Any]]:
        return list(FEATURE_LIBRARY)

    def _tags(self, features: Dict[str, Any]) -> List[str]:
        tags = []
        task_count = int(features.get("task_count", 0) or 0)
        courier_count = int(features.get("courier_count", 0) or 0)
        pair_ratio = float(features.get("pair_ratio", 0.0) or 0.0)
        bundle_ratio = float(features.get("bundle_ratio", 0.0) or 0.0)
        bundle_task_coverage = float(features.get("bundle_task_coverage", 0.0) or 0.0)
        single_task_group_coverage = float(features.get("single_task_group_coverage", 0.0) or 0.0)
        avg_willingness = float(features.get("avg_willingness", 0.0) or 0.0)
        low_willingness_ratio = float(features.get("low_willingness_ratio", 0.0) or 0.0)
        very_low_willingness_ratio = float(features.get("very_low_willingness_ratio", 0.0) or 0.0)
        capacity_ratio = float(features.get("capacity_ratio", 0.0) or 0.0)
        avg_candidates_per_group = float(features.get("avg_candidates_per_group", 0.0) or 0.0)
        max_candidates_per_group = int(features.get("max_candidates_per_group", 0) or 0)
        avg_groups_per_courier = float(features.get("avg_groups_per_courier", 0.0) or 0.0)
        score_cv = float(features.get("score_cv", 0.0) or 0.0)
        if task_count <= 45:
            tags.append("small_task_count")
        elif task_count <= 90:
            tags.append("medium_task_count")
        elif task_count >= 120:
            tags.append("large_task_count")
        if courier_count < task_count:
            tags.append("scarce_couriers")
        elif capacity_ratio <= 1.35:
            tags.append("tight_capacity")
        elif capacity_ratio >= 1.8:
            tags.append("many_couriers")
        if pair_ratio >= 0.2:
            tags.append("high_pair_ratio")
        else:
            tags.append("low_pair_ratio")
        if bundle_ratio >= 0.25 or bundle_task_coverage >= 0.45:
            tags.append("bundle_rich")
        if single_task_group_coverage >= 0.9:
            tags.append("single_cover_available")
        if avg_willingness < 0.3:
            tags.append("low_willingness")
        if very_low_willingness_ratio >= 0.25:
            tags.append("very_low_willingness_tail")
        if low_willingness_ratio >= 0.45:
            tags.append("high_reject_risk")
        elif avg_willingness >= 0.55:
            tags.append("high_willingness")
        if avg_candidates_per_group >= 4.0 or max_candidates_per_group >= 8:
            tags.append("dense_candidate_groups")
        elif avg_candidates_per_group <= 1.5:
            tags.append("sparse_candidate_groups")
        if avg_groups_per_courier >= 3.0:
            tags.append("mobile_couriers")
        if score_cv >= 0.8:
            tags.append("high_score_variance")
        if task_count <= 50 and (bundle_ratio >= 0.15 or avg_candidates_per_group <= 4.0):
            tags.append("compact_mask_search_candidate")
        return tags or ["general"]

    def _recommended_focus(self, features: Dict[str, Any]) -> List[str]:
        tags = set(features.get("tags", []))
        focus = ["local_search_repair", "expected_greedy"]
        if "high_pair_ratio" in tags or "bundle_rich" in tags or "scarce_couriers" in tags:
            focus.insert(0, "bundle_first")
        if "low_willingness" in tags:
            focus.insert(0, "willingness_weighted")
        if "compact_mask_search_candidate" in tags:
            focus.append("beam_cover")
        if "single_cover_available" in tags and ("many_couriers" in tags or "low_pair_ratio" in tags):
            focus.append("flow_single_initial")
        if "tight_capacity" in tags or "scarce_couriers" in tags:
            focus.append("min_weight_matching_seed")
        if "bundle_rich" in tags:
            focus.append("pair_replacement_polish")
            focus.append("repartition_small_union")
        if "dense_candidate_groups" in tags or "mobile_couriers" in tags:
            focus.append("courier_relocation_swap")
            focus.append("three_cycle_polish")
        if "high_score_variance" in tags or "large_task_count" in tags or "very_low_willingness_tail" in tags:
            focus.append("hash_multi_start")
            focus.append("randomized_shuffled_greedy")
        if "large_task_count" in tags or "high_reject_risk" in tags:
            focus.append("destroy_repair_ils")
            focus.append("tabu_confchange")
        return list(dict.fromkeys(focus))

    def _feature_profile(self, features: Dict[str, Any]) -> Dict[str, Any]:
        tags = set(features.get("tags", []))
        return {
            "scale": _first_tag(tags, ["small_task_count", "medium_task_count", "large_task_count"], "general_scale"),
            "capacity": _first_tag(tags, ["scarce_couriers", "tight_capacity", "many_couriers"], "balanced_capacity"),
            "bundle": _first_tag(tags, ["bundle_rich", "high_pair_ratio", "low_pair_ratio"], "mixed_bundle_structure"),
            "acceptance": _first_tag(
                tags,
                ["very_low_willingness_tail", "high_reject_risk", "low_willingness", "high_willingness"],
                "medium_willingness",
            ),
            "neighborhood": _first_tag(
                tags,
                ["dense_candidate_groups", "mobile_couriers", "sparse_candidate_groups"],
                "moderate_candidate_density",
            ),
        }


def _first_tag(tags: set, candidates: List[str], fallback: str) -> str:
    for candidate in candidates:
        if candidate in tags:
            return candidate
    return fallback
