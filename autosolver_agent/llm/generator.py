"""LLM-driven planning, structured solver generation, and repair."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from autosolver_agent.framework import (
    FrameworkUpdate,
    InstanceInterpretation,
    SolverFramework,
    framework_update_schema_text,
    instance_interpretation_schema_text,
    parse_framework_update,
    parse_instance_interpretation,
    parse_solver_framework,
    solver_framework_schema_text,
)
from autosolver_agent.llm.schema import (
    SolverPlan,
    candidate_schema_text,
    model_dump,
    parse_candidate_envelope,
    parse_solver_plan,
    plan_schema_text,
)
from autosolver_agent.models import Candidate
from autosolver_agent.runtime import SAFE_IMPORT_ROOTS
from autosolver_agent.tools.langchain_tools import PlannerToolbox, build_langchain_tools

SOLVER_OUTPUT_CONTRACT = (
    "solve(input_text: str) must return one flat solution, not a portfolio of alternatives. "
    "The return value is list[tuple[str, list[str]]]. Each item is "
    "(task_id_list, courier_ids), where task_id_list is exactly one input task_id_list group string "
    "such as 't0' or 't1,t2', and courier_ids is a non-empty list of courier_id strings valid for that group. "
    "Do not return (task_id_list, courier_id) pairs with a bare string courier. "
    "Do not return dictionaries, row objects, nested lists of multiple solutions, scores, or metadata."
)
ALLOWED_IMPORT_ROOTS_TEXT = ", ".join(name for name in sorted(SAFE_IMPORT_ROOTS) if name != "__future__")


class LLMCodeGenerator:
    """Plan, generate, and repair complete candidate solver code."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        llm: Any = None,
    ) -> None:
        self.model = model or os.environ.get("AUTOSOLVER_LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        self.temperature = temperature
        self.wire_api = os.environ.get("AUTOSOLVER_WIRE_API") or os.environ.get("OPENAI_WIRE_API")
        self.reasoning_effort = os.environ.get("AUTOSOLVER_REASONING_EFFORT") or os.environ.get("OPENAI_REASONING_EFFORT")
        self.extra_body = _env_json("AUTOSOLVER_LLM_EXTRA_BODY", "OPENAI_EXTRA_BODY")
        self.request_timeout = _env_float("AUTOSOLVER_LLM_TIMEOUT", "OPENAI_TIMEOUT", "OPENAI_REQUEST_TIMEOUT", default=300.0)
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
        if self.request_timeout is not None:
            kwargs["timeout"] = self.request_timeout
        if str(self.wire_api or "").lower() == "responses":
            kwargs["use_responses_api"] = True
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.extra_body is not None:
            kwargs["extra_body"] = self.extra_body
        if self.disable_response_storage:
            kwargs["store"] = False
        return ChatOpenAI(**kwargs)

    def plan(
        self,
        iteration: int,
        instance_features: Dict[str, Any],
        solver_framework_context: str,
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
            "You are the planning controller for AutoSolver Agent. Use tools to inspect objective instance features, "
            "the LLM-maintained solver framework, memory, bandit recommendations, and the current best artifact. "
            "You may create strategy names that are not yet in the framework if the instance evidence supports them. "
            "Return only JSON matching this SolverPlan schema:\n"
            + plan_schema_text()
        )
        user = (
            f"Plan solver generation for iteration {iteration}. "
            "Prefer strategies that fit current interpreted features and historical results. "
            "Use a balanced explore/exploit policy and preserve the solver safety contract. "
            "Current solver framework context:\n"
            + solver_framework_context
            + "\n\nPrevious impact:\n"
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

    def bootstrap_framework(
        self,
        objective_features: Dict[str, Any],
        memory_digest: Dict[str, Any],
        case_samples: List[str],
    ) -> SolverFramework:
        system = (
            "You maintain AutoSolver Agent's feature, strategy, and implementation-skill framework. "
            "There is no hardcoded seed catalog. Create an initial solver framework from the objective case statistics, "
            "historical memory, and the fixed solver safety contract. Keep generated solve() implementations bounded and "
            "compatible with the runtime sandbox. Return only JSON matching this SolverFramework schema:\n"
            + solver_framework_schema_text()
        )
        user = (
            "Objective case features:\n{features}\n\n"
            "Memory digest:\n{memory}\n\n"
            "Input case samples:\n{samples}\n\n"
            "Create feature_dimensions, strategies, and skills that are useful for generating safe, self-contained "
            "solve(input_text: str) implementations that obey this output contract:\n{contract}\n\n"
            "Candidate code may only import from these validator-approved roots: {allowed_imports}. "
            "Optional non-standard-library imports must be guarded with deterministic fallbacks.\n\n"
            "Do not propose changes to validator, scorer, runtime, parser, or the output contract."
        ).format(
            features=_json(objective_features),
            memory=_json(memory_digest),
            samples="\n\n---\n\n".join(case_samples),
            contract=SOLVER_OUTPUT_CONTRACT,
            allowed_imports=ALLOWED_IMPORT_ROOTS_TEXT,
        )
        return parse_solver_framework(self._invoke(system, user))

    def interpret_instances(
        self,
        iteration: int,
        objective_features: Dict[str, Any],
        solver_framework_context: str,
        memory_digest: Dict[str, Any],
        case_samples: List[str],
    ) -> InstanceInterpretation:
        system = (
            "You interpret delivery-assignment instances using the LLM-maintained solver framework. "
            "Do not rely on hardcoded thresholds; derive tags, opportunities, risks, and recommended strategy focus "
            "from the supplied objective statistics and framework. Return only JSON matching this InstanceInterpretation schema:\n"
            + instance_interpretation_schema_text()
        )
        user = (
            "Iteration: {iteration}\n\n"
            "Objective case features:\n{features}\n\n"
            "Solver framework context:\n{framework}\n\n"
            "Memory digest:\n{memory}\n\n"
            "Input case samples:\n{samples}"
        ).format(
            iteration=iteration,
            features=_json(objective_features),
            framework=solver_framework_context,
            memory=_json(memory_digest),
            samples="\n\n---\n\n".join(case_samples),
        )
        return parse_instance_interpretation(self._invoke(system, user))

    def reflect_framework(
        self,
        iteration: int,
        solver_framework_context: str,
        instance_features: Dict[str, Any],
        plans: List[Dict[str, Any]],
        evaluations: List[Dict[str, Any]],
        experiments: List[Dict[str, Any]],
        previous_impact: List[Dict[str, Any]],
    ) -> FrameworkUpdate:
        system = (
            "You maintain AutoSolver Agent's persistent feature/strategy/skill framework after candidate evaluation. "
            "Return a partial update: add or replace only useful framework entries, or retire entries that the evidence "
            "shows are misleading. Keep framework skills compatible with the runtime sandbox. "
            "The validator, scorer, runtime sandbox, parser, and solve() contract are immutable. "
            "Return only JSON matching this FrameworkUpdate schema:\n"
            + framework_update_schema_text()
        )
        user = (
            "Iteration: {iteration}\n\n"
            "Current solver framework context:\n{framework}\n\n"
            "Interpreted instance features:\n{features}\n\n"
            "Plans:\n{plans}\n\n"
            "Candidate evaluations:\n{evaluations}\n\n"
            "Recent experiments:\n{experiments}\n\n"
            "Previous impact:\n{impact}\n\n"
            "Use both successes and failures as evidence. Keep updates concise and safe. Prefer input-size-bounded logic; "
            "time-aware guards may be used only within the external runtime limits. Preserve this immutable output contract:\n"
            "{contract}"
        ).format(
            iteration=iteration,
            framework=solver_framework_context,
            features=_json(instance_features),
            plans=_json(plans),
            evaluations=_json(evaluations),
            experiments=_json(experiments[-8:]),
            impact=_json(previous_impact[-8:]),
            contract=SOLVER_OUTPUT_CONTRACT,
        )
        return parse_framework_update(self._invoke(system, user))

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
            "The solver must define solve(input_text: str) -> list, "
            "and obey this output contract:\n"
            + SOLVER_OUTPUT_CONTRACT
            + "\n"
            f"Candidate code may only import from these validator-approved roots: {ALLOWED_IMPORT_ROOTS_TEXT}. "
            "Optional non-standard-library imports must be guarded with deterministic fallbacks. "
            "Do not use file IO, network IO, subprocess, eval, exec, compile, or dynamic imports. "
            "The standard-library time module is allowed for lightweight runtime guards, but candidate code must still "
            "remain bounded by the external runtime limits. "
            "Use the LLM-maintained solver framework as guidance, but feel free to create a new strategy variation "
            "when the current SolverPlan calls for it. "
            "Return exactly one JSON object matching this CandidateEnvelope schema:\n"
            + candidate_schema_text()
        )
        user = (
            "SolverPlan:\n{plan}\n\n"
            "Instance features:\n{features}\n\n"
            "LLM-maintained solver framework:\n{solvers}\n\n"
            "Memory digest:\n{memory}\n\n"
            "Tool context:\n{tool_context}\n\n"
            "Disk historical results:\n{disk}\n\n"
            "Previous impact analysis:\n{impact}\n\n"
            "Input case samples:\n{samples}\n\n"
            "Output contract:\n{contract}\n\n"
            "Constraint: the external judge timeout is {timeout:.2f}s per case. Keep the algorithm bounded by input size "
            "and fixed iteration limits; optional time checks should be secondary guards below that limit. "
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
            contract=SOLVER_OUTPUT_CONTRACT,
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
            "CandidateEnvelope schema. Use the LLM-maintained solver framework when fixing construction, "
            "coverage repair, or search issues, while keeping the solver self-contained. "
            f"Allowed import roots are: {ALLOWED_IMPORT_ROOTS_TEXT}. Optional non-standard-library imports must have "
            "deterministic fallbacks. The standard-library time module is allowed, but do not add unsafe imports "
            "or change the solve() contract. "
            "The repaired code must obey this output contract:\n"
            + SOLVER_OUTPUT_CONTRACT
            + "\n"
            "CandidateEnvelope schema:\n"
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
            "LLM-maintained solver framework:\n{solvers}\n\n"
            "Input case samples:\n{samples}\n\n"
            "Output contract:\n{contract}\n\n"
            "Keep solve() within the validator-approved import policy and efficient for an external "
            "{timeout:.2f}s per-case judge. "
            "Prefer deterministic input-size bounds; time-based cutoffs may be used as secondary guards."
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
            contract=SOLVER_OUTPUT_CONTRACT,
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


def _env_float(*names: str, default: Optional[float] = None) -> Optional[float]:
    for name in names:
        value = os.environ.get(name)
        if value is None or not str(value).strip():
            continue
        try:
            parsed = float(value)
        except ValueError as exc:
            raise RuntimeError(f"{name} must be a number.") from exc
        if parsed <= 0:
            raise RuntimeError(f"{name} must be greater than 0.")
        return parsed
    return default


def _env_json(*names: str) -> Optional[Dict[str, Any]]:
    for name in names:
        value = os.environ.get(name)
        if value is None or not str(value).strip():
            continue
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{name} must be valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{name} must be a JSON object.")
        return parsed
    return None
