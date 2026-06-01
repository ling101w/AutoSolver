"""Artifact persistence helpers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from autosolver_agent.models import Candidate, IterationArtifact, ScoreResult, ValidationResult


class ArtifactStore:
    def __init__(self, artifact_dir: str) -> None:
        self.artifact_dir = os.path.abspath(artifact_dir)
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.artifacts: List[IterationArtifact] = []

    def iteration_dir(self, iteration: int) -> str:
        path = os.path.join(self.artifact_dir, f"iteration_{iteration:03d}")
        os.makedirs(path, exist_ok=True)
        return path

    def save_candidate(self, candidate: Candidate) -> IterationArtifact:
        path = self.iteration_dir(candidate.iteration)
        code_path = os.path.join(path, f"{candidate.name}.py")
        rationale_path = os.path.join(path, f"{candidate.name}.rationale.json")
        with open(code_path, "w", encoding="utf-8") as handle:
            handle.write(candidate.code)
        write_json(rationale_path, candidate.rationale)
        artifact = IterationArtifact(
            iteration=candidate.iteration,
            candidate_name=candidate.name,
            code_path=code_path,
            rationale_path=rationale_path,
            validation_path=os.path.join(path, f"{candidate.name}.validation.json"),
        )
        self.artifacts.append(artifact)
        return artifact

    def save_validation(self, artifact: IterationArtifact, validation: ValidationResult) -> None:
        write_json(artifact.validation_path, serialize(validation))

    def save_score(self, artifact: IterationArtifact, score: ScoreResult) -> None:
        artifact.score_path = os.path.join(
            self.iteration_dir(artifact.iteration),
            f"{artifact.candidate_name}.score.json",
        )
        write_json(artifact.score_path, serialize(score))

    def save_impact(self, artifact: IterationArtifact, impact: Dict[str, Any]) -> None:
        artifact.impact_path = os.path.join(
            self.iteration_dir(artifact.iteration),
            f"{artifact.candidate_name}.impact.json",
        )
        write_json(artifact.impact_path, impact)

    def disk_results(self, limit: int = 20) -> List[Dict[str, Any]]:
        results = []
        if not os.path.isdir(self.artifact_dir):
            return results
        for root, _, files in os.walk(self.artifact_dir):
            for name in files:
                if not name.endswith(".score.json"):
                    continue
                path = os.path.join(root, name)
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        value = json.load(handle)
                    value["_path"] = path
                    results.append(value)
                except Exception:
                    continue
        results.sort(key=lambda item: item.get("rank", [999, 0, 1e18, 1e18]))
        return results[:limit]

    def summary(self) -> List[Dict[str, Any]]:
        return [serialize(item) for item in self.artifacts]


def serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    return value


def write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(serialize(value), handle, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)
