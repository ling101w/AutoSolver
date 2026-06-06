"""LangChain planner tools for the AutoSolver Agent."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional


class PlannerToolbox:
    """Read-only toolbox exposed to the planning LLM."""

    def __init__(
        self,
        instance_features: Dict[str, Any],
        solver_framework: Dict[str, Any],
        memory: Any,
        artifacts: Any,
        feature_query: Dict[str, Any],
        memory_top_k: int,
        bandit_exploration: float,
        best_summary: Dict[str, Any],
        candidate_arms: Optional[List[str]] = None,
    ) -> None:
        self.instance_features = instance_features
        self.solver_framework = solver_framework
        self.memory = memory
        self.artifacts = artifacts
        self.feature_query = feature_query
        self.memory_top_k = memory_top_k
        self.bandit_exploration = bandit_exploration
        self.best_summary = best_summary
        self.candidate_arms = list(candidate_arms or [])
        self.trace: List[Dict[str, Any]] = []

    def get_instance_features(self) -> str:
        return self._record("get_instance_features", {}, self.instance_features)

    def get_solver_framework(self) -> str:
        return self._record("get_solver_framework", {}, self.solver_framework)

    def retrieve_similar_experiments(self, top_k: Optional[int] = None) -> str:
        limit = int(top_k or self.memory_top_k)
        result = self.memory.retrieve_similar(self.feature_query, top_k=limit)
        return self._record("retrieve_similar_experiments", {"top_k": limit}, result)

    def get_bandit_recommendations(self, limit: int = 5) -> str:
        candidates = self.candidate_arms or self.feature_query.get("recommended_focus", []) or self.feature_query.get("tags", [])
        result = self.memory.bandit_recommendations(
            candidate_arms=candidates,
            exploration=self.bandit_exploration,
            limit=limit,
        )
        return self._record("get_bandit_recommendations", {"limit": limit}, result)

    def get_best_artifact_summary(self) -> str:
        return self._record("get_best_artifact_summary", {}, self.best_summary)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "instance_features": self.instance_features,
            "solver_framework": self.solver_framework,
            "similar_experiments": self.memory.retrieve_similar(self.feature_query, top_k=self.memory_top_k),
            "bandit_recommendations": self.memory.bandit_recommendations(
                candidate_arms=self.candidate_arms or self.feature_query.get("recommended_focus", []) or self.feature_query.get("tags", []),
                exploration=self.bandit_exploration,
                limit=5,
            ),
            "best_artifact_summary": self.best_summary,
        }

    def _record(self, name: str, args: Dict[str, Any], result: Any) -> str:
        item = {"tool": name, "args": args, "result": result}
        self.trace.append(item)
        return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)


def build_langchain_tools(toolbox: PlannerToolbox) -> List[Any]:
    """Build StructuredTool objects for planner tool calls."""

    from langchain_core.tools import StructuredTool

    def make_tool(name: str, description: str, func: Callable[..., str]) -> Any:
        return StructuredTool.from_function(func=func, name=name, description=description)

    return [
        make_tool(
            "get_instance_features",
            "Return aggregate and per-case delivery assignment features for this run.",
            lambda: toolbox.get_instance_features(),
        ),
        make_tool(
            "get_solver_framework",
            "Return the LLM-maintained feature, strategy, and skill framework.",
            lambda: toolbox.get_solver_framework(),
        ),
        make_tool(
            "retrieve_similar_experiments",
            "Return historical experiments nearest to the current instance features.",
            lambda top_k=toolbox.memory_top_k: toolbox.retrieve_similar_experiments(top_k=top_k),
        ),
        make_tool(
            "get_bandit_recommendations",
            "Return UCB bandit recommendations for explore/exploit strategy selection.",
            lambda limit=5: toolbox.get_bandit_recommendations(limit=limit),
        ),
        make_tool(
            "get_best_artifact_summary",
            "Return the current best candidate score and a compact code summary.",
            lambda: toolbox.get_best_artifact_summary(),
        ),
    ]
