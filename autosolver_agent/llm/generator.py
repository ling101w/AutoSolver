"""LLM-driven planning, structured solver generation, and repair."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from autosolver_agent.llm.schema import (
    SolverPlan,
    candidate_schema_text,
    model_dump,
    parse_candidate_envelope,
    parse_solver_plan,
    plan_schema_text,
)
from autosolver_agent.models import Candidate
from autosolver_agent.tools.langchain_tools import PlannerToolbox, build_langchain_tools


class LLMCodeGenerator:
    """Plan, generate, and repair complete candidate solver code."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        llm: Any = None,
    ) -> None:
        self.model = model or os.environ.get("AUTOSOLVER_LLM_MODEL") or "gpt-4o-mini"
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        self.temperature = temperature
        self.wire_api = os.environ.get("AUTOSOLVER_WIRE_API") or os.environ.get("OPENAI_WIRE_API")
        self.reasoning_effort = os.environ.get("AUTOSOLVER_REASONING_EFFORT") or os.environ.get("OPENAI_REASONING_EFFORT")
        self.disable_response_storage = _env_bool("AUTOSOLVER_DISABLE_RESPONSE_STORAGE") or _env_bool(
            "OPENAI_DISABLE_RESPONSE_STORAGE"
        )
        self.llm = llm if llm is not None else self._build_langchain_llm()
        self.last_planner_trace: List[Dict[str, Any]] = []
        self.last_tool_calls: List[Dict[str, Any]] = []

    @staticmethod
    def validate_environment() -> None:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")):
            raise RuntimeError("LLM code generation requires OPENAI_API_KEY or OPENAI_KEY.")
        try:
            from langchain_openai import ChatOpenAI  # noqa: F401
        except Exception as exc:
            raise RuntimeError("LLM code generation requires langchain-openai. Install requirements.txt.") from exc

    def _build_langchain_llm(self) -> Any:
        self.validate_environment()
        from langchain_openai import ChatOpenAI

        kwargs: Dict[str, Any] = {"model": self.model, "temperature": self.temperature}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if str(self.wire_api or "").lower() == "responses":
            kwargs["use_responses_api"] = True
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.disable_response_storage:
            kwargs["store"] = False
        return ChatOpenAI(**kwargs)

    def plan(
        self,
        iteration: int,
        instance_features: Dict[str, Any],
        strategy_context: str,
        memory_digest: Dict[str, Any],
        previous_impact: List[Dict[str, Any]],
        toolbox: Optional[PlannerToolbox] = None,
    ) -> SolverPlan:
        """Produce a SolverPlan, using LangChain tools when the LLM supports them."""

        if not hasattr(self.llm, "bind_tools") or toolbox is None:
            raise RuntimeError("Planning requires an LLM with bind_tools support and a PlannerToolbox.")

        tools = build_langchain_tools(toolbox)
        if not tools:
            raise RuntimeError("Planning requires LangChain planner tools; build_langchain_tools returned no tools.")

        system = (
            "You are the planning controller for AutoSolver Agent. Use tools to inspect instance features, "
            "strategy guidance, memory, bandit recommendations, and the current best artifact. "
            "Return only JSON matching this SolverPlan schema:\n"
            + plan_schema_text()
        )
        user = (
            f"Plan solver generation for iteration {iteration}. "
            "Prefer strategies that fit current features and historical results. "
            "Use a balanced explore/exploit policy. Previous impact:\n"
            + _json(previous_impact[-5:])
        )
        messages: List[Any] = [("system", system), ("human", user)]
        trace: List[Dict[str, Any]] = []
        content = ""
        tool_llm = self.llm.bind_tools(tools)
        tool_by_name = {getattr(tool, "name", ""): tool for tool in tools}
        for _ in range(4):
            response = tool_llm.invoke(messages)
            tool_calls = list(getattr(response, "tool_calls", []) or [])
            if not tool_calls:
                content = self._content(response)
                break
            messages.append(response)
            for call in tool_calls:
                name = call.get("name")
                args = call.get("args") or {}
                tool = tool_by_name.get(name)
                if tool is None:
                    raise RuntimeError(f"Planning LLM requested unknown tool: {name}")
                result = tool.invoke(args)
                trace.append({"tool": name, "args": args, "result": result})
                messages.append(_tool_message(str(result), call.get("id"), name))
        if not content:
            raise RuntimeError("Planning LLM exhausted tool-call budget without returning SolverPlan JSON.")
        plan = parse_solver_plan(content)
        self.last_planner_trace = [{"mode": "tool_calling", "raw_response": content, "plan": model_dump(plan)}]
        self.last_tool_calls = trace or toolbox.trace
        return plan

    def generate_from_plan(
        self,
        iteration: int,
        plan: SolverPlan,
        instance_features: Dict[str, Any],
        solver_context: str,
        memory_digest: Dict[str, Any],
        disk_results: List[Dict[str, Any]],
        previous_impact: List[Dict[str, Any]],
        case_samples: List[str],
        per_case_timeout: float,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> Candidate:
        system = (
            "You are AutoSolver Agent's code generator. Generate one complete Python solver. "
            "The solver must use only Python standard library and define solve(input_text: str) -> list. "
            "Do not use file IO, network IO, subprocess, eval, exec, compile, or dynamic imports. "
            "When solver_examples are provided, use them as reference architectures and adapt their patterns "
            "to the current SolverPlan instead of copying every routine. "
            "Return exactly one JSON object matching this CandidateEnvelope schema:\n"
            + candidate_schema_text()
        )
        user = (
            "SolverPlan:\n{plan}\n\n"
            "Instance features:\n{features}\n\n"
            "Solver skill context:\n{solvers}\n\n"
            "Memory digest:\n{memory}\n\n"
            "Tool context:\n{tool_context}\n\n"
            "Disk historical results:\n{disk}\n\n"
            "Previous impact analysis:\n{impact}\n\n"
            "Input case samples:\n{samples}\n\n"
            "Constraint: per-case judge timeout is {timeout:.2f}s; keep an internal safety margin in solve(). "
            "The JSON must include rationale fields name, idea, strategy_combination, parameter_changes, "
            "expected_effect, risk_control, and code containing the complete Python implementation."
        ).format(
            plan=_json(model_dump(plan)),
            features=_json(instance_features),
            solvers=solver_context,
            memory=_json(memory_digest),
            tool_context=_json(tool_context or {}),
            disk=_json(disk_results[-8:]),
            impact=_json(previous_impact[-5:]),
            samples="\n\n---\n\n".join(case_samples),
            timeout=per_case_timeout,
        )
        text = self._invoke(system, user)
        envelope = parse_candidate_envelope(text)
        rationale = model_dump(envelope.rationale)
        rationale.setdefault("plan", model_dump(plan))
        return Candidate(
            name=sanitize_name(rationale.get("name"), f"llm_iter_{iteration:03d}"),
            code=envelope.code,
            rationale=rationale,
            iteration=iteration,
        )

    def repair(
        self,
        iteration: int,
        plan: SolverPlan,
        errors: List[Dict[str, Any]],
        instance_features: Dict[str, Any],
        solver_context: str,
        memory_digest: Dict[str, Any],
        case_samples: List[str],
        per_case_timeout: float,
        failed_code: str = "",
        failed_rationale: Optional[Dict[str, Any]] = None,
        raw_response: str = "",
        best_summary: Optional[Dict[str, Any]] = None,
        score_delta: Optional[Dict[str, Any]] = None,
        attempt: int = 1,
    ) -> Candidate:
        system = (
            "You repair AutoSolver candidate solvers. Return exactly one JSON object matching this "
            "CandidateEnvelope schema. Use solver_examples as reference architectures when fixing construction, "
            "coverage repair, or local-search issues, while keeping the solver self-contained:\n"
            + candidate_schema_text()
        )
        user = (
            "Repair attempt {attempt} for iteration {iteration}.\n\n"
            "Plan:\n{plan}\n\n"
            "Errors:\n{errors}\n\n"
            "Failed rationale:\n{rationale}\n\n"
            "Failed code:\n{code}\n\n"
            "Raw malformed response, if any:\n{raw}\n\n"
            "Instance features:\n{features}\n\n"
            "Memory digest:\n{memory}\n\n"
            "Best code summary:\n{best}\n\n"
            "Score delta/context:\n{score_delta}\n\n"
            "Solver skill context:\n{solvers}\n\n"
            "Input case samples:\n{samples}\n\n"
            "Keep solve() standard-library only and below {timeout:.2f}s per case."
        ).format(
            attempt=attempt,
            iteration=iteration,
            plan=_json(model_dump(plan)),
            errors=_json(errors),
            rationale=_json(failed_rationale or {}),
            code=failed_code[-8000:],
            raw=raw_response[-4000:],
            features=_json(instance_features),
            memory=_json(memory_digest),
            best=_json(best_summary or {}),
            score_delta=_json(score_delta or {}),
            solvers=solver_context,
            samples="\n\n---\n\n".join(case_samples),
            timeout=per_case_timeout,
        )
        text = self._invoke(system, user)
        envelope = parse_candidate_envelope(text)
        rationale = model_dump(envelope.rationale)
        rationale.setdefault("plan", model_dump(plan))
        rationale["repair_attempt"] = attempt
        return Candidate(
            name=sanitize_name(rationale.get("name"), f"llm_iter_{iteration:03d}_repair_{attempt}"),
            code=envelope.code,
            rationale=rationale,
            iteration=iteration,
            source="llm_repair",
        )

    def _invoke(self, system: str, user: str) -> str:
        response = self.llm.invoke([("system", system), ("human", user)])
        return self._content(response)

    def _content(self, response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)

def sanitize_name(value: Any, default_name: str) -> str:
    import re

    raw = str(value or default_name)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    return (name or default_name)[:80]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _tool_message(content: str, tool_call_id: Optional[str], name: Optional[str]) -> Any:
    from langchain_core.messages import ToolMessage

    return ToolMessage(content=content, tool_call_id=tool_call_id or name or "tool")


def _env_bool(name: str) -> bool:
    value = os.environ.get(name)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
