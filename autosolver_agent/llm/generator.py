"""LLM-driven full solver code generation."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from autosolver_agent.models import Candidate


class LLMCodeGenerator:
    """Generate complete candidate solver code from prompt context.

    This class intentionally has no heuristic/template fallback. If no LLM is
    configured, construction fails and the workflow cannot proceed.
    """

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
        self.llm = llm if llm is not None else self._build_langchain_llm()

    def _build_langchain_llm(self) -> Any:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
        if not api_key:
            raise RuntimeError("LLM code generation requires OPENAI_API_KEY or OPENAI_KEY.")
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            raise RuntimeError("LLM code generation requires langchain-openai. Install requirements.txt.") from exc
        kwargs: Dict[str, Any] = {"model": self.model, "temperature": self.temperature}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return ChatOpenAI(**kwargs)

    def generate(
        self,
        iteration: int,
        instance_features: Dict[str, Any],
        strategy_context: str,
        solver_context: str,
        memory_digest: Dict[str, Any],
        disk_results: List[Dict[str, Any]],
        previous_impact: List[Dict[str, Any]],
        case_samples: List[str],
        per_case_timeout: float,
    ) -> Candidate:
        system = (
            "你是 AutoSolver Agent 的代码生成器。你必须生成一个完整 Python 求解器。"
            "求解器只能依赖 Python 标准库，必须定义 solve(input_text: str) -> list。"
            "禁止文件 IO、网络 IO、subprocess、eval、exec、compile、动态导入。"
            "输出必须包含一个 JSON 元数据代码块和一个 Python 代码块。"
        )
        user = (
            "实例特征:\n{features}\n\n"
            "策略库:\n{strategies}\n\n"
            "求解器库:\n{solvers}\n\n"
            "仓库记忆摘要:\n{memory}\n\n"
            "磁盘历史结果:\n{disk}\n\n"
            "上一轮影响分析:\n{impact}\n\n"
            "输入样例片段:\n{samples}\n\n"
            "约束: 单 case judge 超时 {timeout:.2f}s，请在 solve 内部自留安全余量。"
            "返回 JSON 元数据字段: name, idea, strategy_combination, parameter_changes, expected_effect, risk_control。"
        ).format(
            features=json.dumps(instance_features, ensure_ascii=False, indent=2, sort_keys=True),
            strategies=strategy_context,
            solvers=solver_context,
            memory=json.dumps(memory_digest, ensure_ascii=False, indent=2, sort_keys=True),
            disk=json.dumps(disk_results[-8:], ensure_ascii=False, indent=2, sort_keys=True),
            impact=json.dumps(previous_impact[-5:], ensure_ascii=False, indent=2, sort_keys=True),
            samples="\n\n---\n\n".join(case_samples),
            timeout=per_case_timeout,
        )
        text = self._invoke(system, user)
        metadata = extract_json_block(text)
        code = extract_python_code(text)
        if not code:
            raise RuntimeError("LLM response did not include a Python code block.")
        name = sanitize_name(metadata.get("name") if isinstance(metadata, dict) else None, f"llm_iter_{iteration:03d}")
        rationale = metadata if isinstance(metadata, dict) else {"raw_response": text[:2000]}
        return Candidate(name=name, code=code, rationale=rationale, iteration=iteration)

    def _invoke(self, system: str, user: str) -> str:
        messages = [("system", system), ("human", user)]
        response = self.llm.invoke(messages)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)


def extract_python_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    if "def solve" in text:
        return text[text.index("def solve") :].strip()
    return ""


def extract_json_block(text: str) -> Dict[str, Any]:
    blocks = re.findall(r"```json\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = blocks or re.findall(r"(\{.*?\})", text, flags=re.DOTALL)
    for raw in candidates:
        try:
            value = json.loads(raw)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return {}


def sanitize_name(value: Any, fallback: str) -> str:
    raw = str(value or fallback)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    return (name or fallback)[:80]
