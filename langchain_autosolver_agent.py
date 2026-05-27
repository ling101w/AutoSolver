"""
langchain_autosolver_agent.py
==============================

LangChain + LangGraph 重写版 AutoSolver Agent.

设计要点
--------
1. 使用 :class:`langgraph.graph.StateGraph` 串起 ``describe.txt`` 中的 6 阶段
   流程: 接收输入 → 分析 → 策略生成 → 策略执行 → 评估筛选 → 迭代改进 → 输出.
2. 求解器原语 (贪心 / 加权贪心 / 最小费用流 / 波束搜索 / 局部改进 / 模拟退火)
   被封装成 LangChain :class:`StructuredTool`, 由 ``控制器`` 通过工具调用驱动.
3. 当检测到 ``OPENAI_API_KEY`` (或兼容的 ``OPENAI_BASE_URL``) 时启用真实 LLM
   控制器 (``langchain_openai.ChatOpenAI`` + bind_tools), 由 LLM 自主决定每一
   轮要尝试什么策略; 否则退回启发式控制器, 完全离线可跑.
4. 最终落盘的 ``generated_submit_solution.py`` 仅依赖 Python 标准库, 可直接被
   judge 服务器加载, 与 Agent 框架本身解耦.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import os
import random
import statistics
import time
import traceback
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

# ---------------------------------------------------------------------------
# LangChain / LangGraph 依赖
# ---------------------------------------------------------------------------
try:
    from langgraph.graph import StateGraph, END
except ImportError as exc:  # pragma: no cover - 友好提示
    raise SystemExit(
        "缺少依赖 langgraph, 请先执行: pip install -r requirements.txt\n"
        f"原始错误: {exc}"
    )

from langchain_core.tools import StructuredTool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

try:  # 可选 LLM 依赖
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

from _solver_template import SOLVER_TEMPLATE


# ===========================================================================
# 1. 数据解析 / 评分 / 子进程执行
# ===========================================================================

def _candidate_worker(code: str, case_text: str, queue: "multiprocessing.Queue") -> None:
    """子进程入口: 编译并运行候选求解器, 把结果回传给主进程."""
    try:
        namespace: Dict[str, Any] = {}
        exec(compile(code, "<candidate_solver>", "exec"), namespace)
        if "solve" not in namespace:
            queue.put(("error", None, 0.0, "missing solve function"))
            return
        start = time.time()
        answer = namespace["solve"](case_text)
        elapsed = time.time() - start
        queue.put(("ok", answer, elapsed, None))
    except Exception:
        queue.put(("error", None, 0.0, traceback.format_exc()))


def parse_case(text: str) -> Dict[str, Any]:
    rows: List[Tuple[Tuple[str, ...], str, str, float, float]] = []
    by_key: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)
    key_tasks: Dict[str, Tuple[str, ...]] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("task_id_list"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        raw_key, courier, score, willingness = parts[:4]
        try:
            tasks = tuple(t.strip() for t in raw_key.split(",") if t.strip())
            if not tasks:
                continue
            task_key = ",".join(tasks)
            courier = courier.strip()
            score = float(score)
            willingness = float(willingness)
        except Exception:
            continue
        rows.append((tasks, task_key, courier, score, willingness))
        by_key[task_key][courier] = (score, willingness)
        key_tasks[task_key] = tasks
    all_tasks = sorted({t for tasks, _, _, _, _ in rows for t in tasks})
    all_couriers = sorted({c for _, _, c, _, _ in rows})
    return {
        "rows": rows,
        "by_key": by_key,
        "key_tasks": key_tasks,
        "all_tasks": all_tasks,
        "all_couriers": all_couriers,
    }


def dataset_features(parsed: Dict[str, Any]) -> Dict[str, Any]:
    rows = parsed["rows"]
    if not rows:
        return {
            "row_count": 0, "task_count": 0, "courier_count": 0,
            "pair_ratio": 0.0, "avg_willingness": 0.0, "avg_score": 0.0,
            "high_willingness_ratio": 0.0, "low_capacity": False,
        }
    pair_rows = sum(1 for tasks, *_ in rows if len(tasks) > 1)
    willingness_values = [r[4] for r in rows]
    score_values = [r[3] for r in rows]
    return {
        "row_count": len(rows),
        "task_count": len(parsed["all_tasks"]),
        "courier_count": len(parsed["all_couriers"]),
        "pair_ratio": pair_rows / max(1, len(rows)),
        "avg_willingness": statistics.fmean(willingness_values),
        "avg_score": statistics.fmean(score_values),
        "high_willingness_ratio": sum(1 for v in willingness_values if v >= 0.5) / max(1, len(willingness_values)),
        "min_score": min(score_values),
        "max_score": max(score_values),
        "low_capacity": len(parsed["all_couriers"]) < len(parsed["all_tasks"]),
    }


def render_solver(config: Dict[str, Any]) -> str:
    return SOLVER_TEMPLATE.replace("__CONFIG__", repr(config))


def penalty_for_group(parsed: Dict[str, Any], task_key: str, couriers: List[str]) -> float:
    fallback = 100.0 * len(parsed["key_tasks"][task_key])
    reject_prob = 1.0
    weighted_score = 0.0
    weight = 0.0
    data = parsed["by_key"][task_key]
    for courier in couriers:
        score, willingness = data[courier]
        reject_prob *= 1.0 - willingness
        weighted_score += willingness * score
        weight += willingness
    if weight <= 0.0:
        return fallback
    return reject_prob * fallback + (1.0 - reject_prob) * weighted_score / weight


def score_answer(parsed: Dict[str, Any], answer: Any) -> Dict[str, Any]:
    if not isinstance(answer, list):
        return {"valid": False, "covered": 0, "penalty": 1_000_000.0, "error": "answer is not list"}
    by_key = parsed["by_key"]
    key_tasks = parsed["key_tasks"]
    used_tasks: set = set()
    used_couriers: set = set()
    total = 0.0
    for item in answer:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            return {"valid": False, "covered": len(used_tasks), "penalty": 1_000_000.0 + total, "error": "bad tuple"}
        raw_key, couriers = item
        if not isinstance(raw_key, str) or not isinstance(couriers, list) or not couriers:
            return {"valid": False, "covered": len(used_tasks), "penalty": 1_000_000.0 + total, "error": "bad fields"}
        tasks = tuple(t.strip() for t in raw_key.split(",") if t.strip())
        task_key = ",".join(tasks)
        if task_key not in key_tasks:
            return {"valid": False, "covered": len(used_tasks), "penalty": 1_000_000.0 + total, "error": "unknown task group"}
        for task in key_tasks[task_key]:
            if task in used_tasks:
                return {"valid": False, "covered": len(used_tasks), "penalty": 1_000_000.0 + total, "error": "duplicate task"}
        local: set = set()
        for courier in couriers:
            if courier in local or courier in used_couriers:
                return {"valid": False, "covered": len(used_tasks), "penalty": 1_000_000.0 + total, "error": "duplicate courier"}
            if courier not in by_key[task_key]:
                return {"valid": False, "covered": len(used_tasks), "penalty": 1_000_000.0 + total, "error": "invalid courier"}
            local.add(courier)
        total += penalty_for_group(parsed, task_key, couriers)
        used_tasks.update(key_tasks[task_key])
        used_couriers.update(couriers)
    total += 100.0 * (len(parsed["all_tasks"]) - len(used_tasks))
    return {"valid": True, "covered": len(used_tasks), "penalty": total, "error": None}


def run_candidate(code: str, case_text: str, per_case_timeout: float) -> Dict[str, Any]:
    queue: "multiprocessing.Queue" = multiprocessing.Queue()
    process = multiprocessing.Process(target=_candidate_worker, args=(code, case_text, queue))
    process.daemon = True
    start = time.time()
    process.start()
    process.join(per_case_timeout)
    runtime = time.time() - start
    if process.is_alive():
        process.terminate()
        process.join(0.2)
        return {"status": "timeout", "answer": None, "runtime": runtime, "error": "timeout"}
    if queue.empty():
        return {"status": "error", "answer": None, "runtime": runtime, "error": "empty result"}
    status, answer, child_runtime, error = queue.get()
    return {"status": status, "answer": answer, "runtime": child_runtime or runtime, "error": error}


# ===========================================================================
# 2. 策略库
# ===========================================================================

PROFILE_LIBRARY: Dict[str, Dict[str, float]] = {
    "expected":     {"coverage_weight":  0.0, "pair_weight":  0.0, "willingness_weight":  0.0, "score_weight": 0.00, "random_weight": 0.0},
    "coverage":     {"coverage_weight": 30.0, "pair_weight":  0.0, "willingness_weight":  5.0, "score_weight": 0.00, "random_weight": 0.0},
    "bundle":       {"coverage_weight": 12.0, "pair_weight": 45.0, "willingness_weight":  5.0, "score_weight": 0.00, "random_weight": 0.0},
    "willingness":  {"coverage_weight":  6.0, "pair_weight":  8.0, "willingness_weight": 35.0, "score_weight": 0.00, "random_weight": 0.0},
    "score":        {"coverage_weight":  0.0, "pair_weight":  0.0, "willingness_weight":  0.0, "score_weight": 0.15, "random_weight": 0.0},
    "explore":      {"coverage_weight":  6.0, "pair_weight": 10.0, "willingness_weight": 10.0, "score_weight": 0.05, "random_weight": 6.0},
    "safe_cover":   {"coverage_weight": 45.0, "pair_weight": 20.0, "willingness_weight": 15.0, "score_weight": 0.00, "random_weight": 0.0},
    "aggressive":   {"coverage_weight": 15.0, "pair_weight": 80.0, "willingness_weight": 45.0, "score_weight": 0.05, "random_weight": 4.0},
}


def base_config(seed: int, per_case_timeout: float) -> Dict[str, Any]:
    solver_time_limit = min(8.75, max(0.3, per_case_timeout - 0.25))
    return {
        "time_limit": solver_time_limit,
        "seed": seed,
        "local_rounds": 3,
        "loop_local_rounds": 1,
        "extra_limit": 80,
        "max_local_keys": 80,
        "mutate_coverage": 15.0,
        "mutate_pair": 20.0,
        "mutate_willingness": 12.0,
        "loop_random_weight": 8.0,
        "beam_width": 160,
        "beam_keep_per_group": 4,
        "beam_task_limit": 42,
        "use_flow": False,
        "use_beam": False,
        "use_sa": False,
        "sa_temp": 30.0,
        "sa_cooling": 0.93,
        "sa_iters_per_temp": 30,
        "sa_min_temp": 0.5,
        "profiles": [],
    }


def make_strategy(
    name: str,
    seed: int,
    per_case_timeout: float,
    profile_names: List[str],
    use_flow: bool = False,
    use_beam: bool = False,
    use_sa: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = base_config(seed, per_case_timeout)
    cfg["profiles"] = [copy.deepcopy(PROFILE_LIBRARY[p]) for p in profile_names if p in PROFILE_LIBRARY]
    cfg["use_flow"] = bool(use_flow)
    cfg["use_beam"] = bool(use_beam)
    cfg["use_sa"] = bool(use_sa)
    if overrides:
        cfg.update(overrides)
    return {"name": name, "config": cfg, "code": render_solver(cfg)}


INITIAL_STRATEGY_RECIPES: List[Dict[str, Any]] = [
    {"name": "expected_greedy",   "profiles": ["expected", "coverage", "willingness"],          "use_flow": False, "use_beam": False, "use_sa": False},
    {"name": "coverage_first",    "profiles": ["coverage", "expected", "bundle"],                "use_flow": False, "use_beam": False, "use_sa": False},
    {"name": "bundle_first",      "profiles": ["bundle", "coverage", "expected"],                "use_flow": False, "use_beam": True,  "use_sa": False},
    {"name": "flow_expected",     "profiles": ["expected", "willingness", "score"],              "use_flow": True,  "use_beam": False, "use_sa": False},
    {"name": "hybrid_full",       "profiles": ["expected", "coverage", "bundle", "willingness", "score"], "use_flow": True,  "use_beam": True,  "use_sa": False},
    {"name": "explorer_sa",       "profiles": ["explore", "coverage", "bundle"],                 "use_flow": False, "use_beam": False, "use_sa": True},
    {"name": "safe_cover",        "profiles": ["safe_cover", "coverage", "willingness"],         "use_flow": True,  "use_beam": True,  "use_sa": False},
    {"name": "aggressive_bundle", "profiles": ["aggressive", "bundle", "coverage"],              "use_flow": False, "use_beam": True,  "use_sa": True},
]


# ===========================================================================
# 3. AgentContext - 工具共享的运行时上下文
# ===========================================================================

class AgentContext:
    """工具调用通过该对象读写共享状态; 单次 Agent 运行内全局唯一."""

    def __init__(
        self,
        cases: List[Dict[str, Any]],
        seed: int,
        per_case_timeout: float,
        deadline: float,
        rng: random.Random,
        verbose: bool = True,
        search_per_case_timeout: Optional[float] = None,
        final_solver_time_limit: float = 8.75,
        finalize_top_k: int = 3,
    ) -> None:
        self.cases = cases
        self.seed = seed
        # 判题机等价超时 (默认 10s); 仅用于 Top-K 复选 / final 渲染.
        self.per_case_timeout = per_case_timeout
        # 搜索期超时 (默认与判题一致); 调小则可跳更多迭代.
        self.search_per_case_timeout = search_per_case_timeout or per_case_timeout
        self.final_solver_time_limit = final_solver_time_limit
        self.finalize_top_k = max(1, finalize_top_k)
        self.deadline = deadline
        # 为 finalize 复选预留预算: top_k × case × 判题超时 + reference 评估 + 冗余.
        finalize_reserve = (
            self.finalize_top_k * len(cases) * per_case_timeout
            + len(cases) * per_case_timeout
            + 2.0
        )
        total_budget = max(0.0, deadline - time.time())
        if finalize_reserve > total_budget * 0.6:
            finalize_reserve = max(per_case_timeout * len(cases) + 2.0, total_budget * 0.4)
        self.search_deadline = max(time.time() + 1.0, deadline - finalize_reserve)
        self.rng = rng
        self.verbose = verbose
        self.parsed_cases: List[Dict[str, Any]] = [parse_case(c["text"]) for c in cases]
        self.features: List[Dict[str, Any]] = [dataset_features(p) for p in self.parsed_cases]
        self.history: List[Dict[str, Any]] = []
        self.best: Optional[Dict[str, Any]] = None
        self.reference: Optional[Dict[str, Any]] = None
        self.pending: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        self.tried_signatures: set = set()
        self.iteration: int = 0

    def time_left(self) -> float:
        return max(0.0, self.deadline - time.time())

    def search_time_left(self) -> float:
        return max(0.0, self.search_deadline - time.time())

    def out_of_time(self, margin: float = 0.5) -> bool:
        """默认检查搜索期截止时间 (不含 finalize 复选)."""
        return time.time() + margin >= self.search_deadline

    def out_of_total_time(self, margin: float = 0.5) -> bool:
        return time.time() + margin >= self.deadline

    def log(self, msg: str) -> None:
        self.notes.append(msg)
        if self.verbose:
            print(f"[agent] {msg}", flush=True)

    def aggregate_features(self) -> Dict[str, Any]:
        if not self.features:
            return {"case_count": 0}
        keys = list(self.features[0].keys())
        agg: Dict[str, Any] = {"case_count": len(self.features)}
        for k in keys:
            values = [f[k] for f in self.features]
            if all(isinstance(v, bool) for v in values):
                agg[k] = any(values)
            elif all(isinstance(v, (int, float)) for v in values):
                agg[k] = round(statistics.fmean(values), 4)
            else:
                agg[k] = values
        return agg

    @staticmethod
    def signature(strategy: Dict[str, Any]) -> str:
        cfg = strategy.get("config", {})
        sub = {
            "use_flow": cfg.get("use_flow"),
            "use_beam": cfg.get("use_beam"),
            "use_sa": cfg.get("use_sa"),
            "profiles": [
                tuple(sorted(p.items())) for p in cfg.get("profiles", [])
            ],
            "extra_limit": cfg.get("extra_limit"),
            "local_rounds": cfg.get("local_rounds"),
        }
        return json.dumps(sub, sort_keys=True, default=str)

    def enqueue(self, strategy: Dict[str, Any]) -> bool:
        sig = self.signature(strategy)
        if sig in self.tried_signatures:
            return False
        self.tried_signatures.add(sig)
        self.pending.append(strategy)
        return True

    def evaluate_strategy(
        self,
        strategy: Dict[str, Any],
        timeout: Optional[float] = None,
        track: bool = True,
    ) -> Dict[str, Any]:
        code = strategy["code"]
        per_case = timeout if timeout is not None else self.search_per_case_timeout
        case_results: List[Dict[str, Any]] = []
        total_covered = 0
        total_tasks = 0
        total_penalty = 0.0
        total_runtime = 0.0
        failures = 0
        for case, parsed in zip(self.cases, self.parsed_cases):
            run = run_candidate(code, case["text"], per_case)
            total_tasks += len(parsed["all_tasks"])
            if run["status"] != "ok":
                failures += 1
                pen = 1_000_000.0 + 100.0 * len(parsed["all_tasks"])
                case_results.append({
                    "case": case["name"], "status": run["status"], "covered": 0,
                    "tasks": len(parsed["all_tasks"]), "penalty": pen,
                    "runtime": run.get("runtime", 0.0), "error": run.get("error"),
                })
                total_penalty += pen
                total_runtime += run.get("runtime", 0.0)
                continue
            scored = score_answer(parsed, run["answer"])
            if not scored["valid"]:
                failures += 1
            total_covered += scored["covered"]
            total_penalty += scored["penalty"]
            total_runtime += run["runtime"]
            case_results.append({
                "case": case["name"],
                "status": "ok" if scored["valid"] else "invalid",
                "covered": scored["covered"], "tasks": len(parsed["all_tasks"]),
                "penalty": scored["penalty"], "runtime": run["runtime"],
                "error": scored.get("error"),
            })
        rank = (failures, -total_covered, total_penalty, total_runtime)
        result = {
            "name": strategy["name"],
            "config": copy.deepcopy(strategy["config"]),
            "code": code, "rank": rank, "cases": case_results,
            "total_covered": total_covered, "total_tasks": total_tasks,
            "total_penalty": total_penalty, "total_runtime": total_runtime,
            "failures": failures,
            "timeout_used": per_case,
        }
        if track:
            self.history.append(result)
            if self.best is None or rank < self.best["rank"]:
                self.best = result
                self.log(
                    f"new best: {strategy['name']} covered={total_covered}/{total_tasks} "
                    f"penalty={total_penalty:.2f} (timeout={per_case:.1f}s)"
                )
        return result

    def evaluate_reference(self, path: Optional[str]) -> None:
        if not path:
            return
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as handle:
                code = handle.read()
            strategy = {"name": "reference", "config": {"reference": True}, "code": code}
            # 参考求解器仅用作基线对比: 不进入 best 竞争也不进 history.
            result = self.evaluate_strategy(
                strategy, timeout=self.per_case_timeout, track=False
            )
            self.reference = result
            self.log(f"reference baseline penalty={result['total_penalty']:.2f}")
        except Exception as exc:
            self.log(f"reference evaluation failed: {exc}")
            self.reference = None

    def finalize_best(self) -> Optional[Dict[str, Any]]:
        """在 Top-3 中以 *判题等价超时* 复选, 并返回调高了 time_limit 的运行代码.

        返回 dict 含键 ``code`` (可直接写入 generated_submit_solution.py) 与
        ``config`` (为调高后的 final config).
        """
        if self.best is None and not self.history:
            return None
        candidates = sorted(self.history, key=lambda r: r["rank"])[:self.finalize_top_k]
        if not candidates:
            candidates = [self.best] if self.best else []
        # 如果搜索期使用了与判题不同的超时, 需要在判题等价超时下复选.
        results: List[Dict[str, Any]] = []
        need_recheck = abs(self.search_per_case_timeout - self.per_case_timeout) > 1e-6
        # 钳制 final time_limit: 必须给子进程留余量, 否则求解器内部 deadline 还没到
        # 就被外部 kill, 直接拿不到结果.
        safe_final_limit = min(self.final_solver_time_limit, self.per_case_timeout - 0.3)
        if safe_final_limit < self.final_solver_time_limit - 1e-6:
            self.log(
                f"clamped final time_limit {self.final_solver_time_limit:.2f}s -> "
                f"{safe_final_limit:.2f}s (per_case_timeout={self.per_case_timeout:.2f}s)"
            )
        for cand in candidates:
            cfg = copy.deepcopy(cand["config"])
            if cfg.get("reference"):
                continue
            cfg["time_limit"] = safe_final_limit
            code = render_solver(cfg)
            strategy = {"name": cand["name"] + "_final", "config": cfg, "code": code}
            margin = self.per_case_timeout * len(self.cases) + 0.5
            if need_recheck and not self.out_of_total_time(margin=margin):
                self.log(f"finalize recheck: {strategy['name']} at {self.per_case_timeout:.1f}s")
                res = self.evaluate_strategy(strategy, timeout=self.per_case_timeout, track=False)
                results.append({"strategy": strategy, "result": res, "rechecked": True})
            else:
                # 复选预算不足: 退化为搜索期 rank, 但仍写出调高 time_limit 的代码.
                results.append({"strategy": strategy, "result": cand, "rechecked": False})
        if not results:
            return None
        # finalize 比较忽略 runtime, 仅比 (failures, -covered, penalty); 长 runtime
        # 在搜索期是劣势, 但在落盘后由判题机统一给到固定时限, 不应再作为 tiebreaker.
        def finalize_key(item: Dict[str, Any]) -> Tuple[int, int, float]:
            r = item["result"]["rank"]
            return (r[0], r[1], r[2])

        results.sort(key=finalize_key)
        chosen = results[0]
        chosen_result = chosen["result"]
        chosen_result_copy = dict(chosen_result)
        chosen_result_copy["name"] = chosen["strategy"]["name"]
        chosen_result_copy["config"] = copy.deepcopy(chosen["strategy"]["config"])
        chosen_result_copy["code"] = chosen["strategy"]["code"]
        self.history.append(chosen_result_copy)
        # finalize 的 chosen 一定是要写出去的求解器, 直接覆盖 best 以保证报告与落盘一致.
        self.best = chosen_result_copy
        self.log(
            f"finalize: chose {chosen_result_copy['name']} "
            f"covered={chosen_result_copy['total_covered']}/{chosen_result_copy['total_tasks']} "
            f"penalty={chosen_result_copy['total_penalty']:.2f} "
            f"(rechecked={chosen['rechecked']})"
        )
        return chosen["strategy"]

# ===========================================================================
# 4. LangChain 工具定义
# ===========================================================================

class ProposeStrategyArgs(BaseModel):
    name: str = Field(..., description="策略名字, 用于日志/对比")
    profiles: List[str] = Field(
        ...,
        description="按顺序应用的 profile 名字; 可选: " + ", ".join(PROFILE_LIBRARY.keys()),
    )
    use_flow: bool = Field(False, description="启用最小费用流初始化 (单任务覆盖足够时受益)")
    use_beam: bool = Field(False, description="启用波束搜索初始化 (任务数 <= 42 时有效)")
    use_sa: bool = Field(False, description="启用模拟退火细化")
    extra_limit: Optional[int] = Field(None, description="多骑手填充上限")
    local_rounds: Optional[int] = Field(None, description="初始局部搜索轮数")
    seed_delta: Optional[int] = Field(None, description="种子偏移, 用于多样性")


class MutateBestArgs(BaseModel):
    name: str = Field(..., description="变异后策略名字")
    focus: str = Field(
        "balanced",
        description="变异方向: balanced/coverage/bundle/willingness/diversify",
    )
    seed_delta: Optional[int] = Field(None, description="种子偏移, 用于多样性")


class ViewHistoryArgs(BaseModel):
    top_k: int = Field(5, description="返回历史前 k 条")


class FinalizeArgs(BaseModel):
    reason: str = Field("", description="选择当前 best 落盘的简短理由")


def _profile_names(profiles: List[Dict[str, Any]]) -> List[str]:
    """根据值反查 profile 名字 (尽力匹配)."""
    out: List[str] = []
    for p in profiles:
        match = "custom"
        for name, ref in PROFILE_LIBRARY.items():
            if all(abs(p.get(k, 0.0) - ref.get(k, 0.0)) < 1e-6 for k in ref):
                match = name
                break
        out.append(match)
    return out


def build_tools(ctx: AgentContext) -> Tuple[List[StructuredTool], Dict[str, StructuredTool]]:
    """构造工具集合, 工具内部直接读写 ctx."""

    def _tool_inspect_dataset() -> str:
        agg = ctx.aggregate_features()
        per_case = []
        for case, feat in zip(ctx.cases, ctx.features):
            per_case.append({
                "name": case["name"],
                "tasks": feat.get("task_count"),
                "couriers": feat.get("courier_count"),
                "rows": feat.get("row_count"),
                "pair_ratio": round(feat.get("pair_ratio", 0.0), 3),
                "avg_willingness": round(feat.get("avg_willingness", 0.0), 3),
            })
        return json.dumps({"summary": agg, "cases": per_case}, ensure_ascii=False)

    def _tool_propose_strategy(
        name: str,
        profiles: List[str],
        use_flow: bool = False,
        use_beam: bool = False,
        use_sa: bool = False,
        extra_limit: Optional[int] = None,
        local_rounds: Optional[int] = None,
        seed_delta: Optional[int] = None,
    ) -> str:
        valid_profiles = [p for p in profiles if p in PROFILE_LIBRARY]
        if not valid_profiles:
            return json.dumps({"ok": False, "error": "至少需要一个有效 profile"})
        overrides: Dict[str, Any] = {}
        if extra_limit is not None:
            overrides["extra_limit"] = max(0, min(400, int(extra_limit)))
        if local_rounds is not None:
            overrides["local_rounds"] = max(0, min(10, int(local_rounds)))
        seed = ctx.seed + (int(seed_delta) if seed_delta else 0)
        strategy = make_strategy(
            name=name,
            seed=seed,
            per_case_timeout=ctx.search_per_case_timeout,
            profile_names=valid_profiles,
            use_flow=use_flow,
            use_beam=use_beam,
            use_sa=use_sa,
            overrides=overrides,
        )
        accepted = ctx.enqueue(strategy)
        return json.dumps({
            "ok": True, "queued": accepted,
            "duplicate": not accepted,
            "name": strategy["name"], "config_signature": ctx.signature(strategy),
        })

    def _tool_mutate_best(
        name: str,
        focus: str = "balanced",
        seed_delta: Optional[int] = None,
    ) -> str:
        if ctx.best is None:
            return json.dumps({"ok": False, "error": "尚无 best 可变异"})
        cfg = copy.deepcopy(ctx.best["config"])
        if cfg.get("reference"):
            cfg = base_config(ctx.seed, ctx.search_per_case_timeout)
            cfg["profiles"] = [copy.deepcopy(PROFILE_LIBRARY["expected"]),
                               copy.deepcopy(PROFILE_LIBRARY["coverage"])]
        rng = ctx.rng
        focus = (focus or "balanced").lower()
        if focus == "coverage":
            cfg["use_beam"] = True
            cfg["extra_limit"] = min(180, cfg.get("extra_limit", 80) + 20)
            for p in cfg.get("profiles", []):
                p["coverage_weight"] = p.get("coverage_weight", 0.0) + 12.0 + 8.0 * rng.random()
        elif focus == "bundle":
            for p in cfg.get("profiles", []):
                p["pair_weight"] = p.get("pair_weight", 0.0) + 12.0 + 12.0 * rng.random()
            cfg["use_beam"] = True
        elif focus == "willingness":
            for p in cfg.get("profiles", []):
                p["willingness_weight"] = p.get("willingness_weight", 0.0) + 8.0 + 10.0 * rng.random()
            cfg["use_flow"] = True
        elif focus == "diversify":
            cfg["use_sa"] = True
            cfg["sa_temp"] = max(15.0, cfg.get("sa_temp", 30.0) + rng.uniform(-5.0, 15.0))
            for p in cfg.get("profiles", []):
                p["random_weight"] = max(p.get("random_weight", 0.0), 6.0) + rng.uniform(0.0, 6.0)
        else:  # balanced
            cfg["local_rounds"] = min(8, cfg.get("local_rounds", 3) + 1)
            for p in cfg.get("profiles", []):
                p["willingness_weight"] = max(0.0, p.get("willingness_weight", 0.0) + rng.uniform(-6.0, 12.0))
                p["pair_weight"] = max(0.0, p.get("pair_weight", 0.0) + rng.uniform(-8.0, 14.0))
                p["score_weight"] = max(0.0, p.get("score_weight", 0.0) + rng.uniform(-0.04, 0.06))
        cfg["seed"] = ctx.seed + (int(seed_delta) if seed_delta else rng.randrange(1, 99991))
        strategy = {"name": name, "config": cfg, "code": render_solver(cfg)}
        accepted = ctx.enqueue(strategy)
        return json.dumps({
            "ok": True, "queued": accepted, "duplicate": not accepted,
            "name": name, "focus": focus,
            "config_signature": ctx.signature(strategy),
        })

    def _tool_run_pending() -> str:
        if not ctx.pending:
            return json.dumps({"ok": False, "error": "队列为空"})
        results = []
        while ctx.pending and not ctx.out_of_time(margin=ctx.search_per_case_timeout * len(ctx.cases) + 0.5):
            strategy = ctx.pending.pop(0)
            ctx.iteration += 1
            res = ctx.evaluate_strategy(strategy)
            results.append({
                "name": res["name"],
                "rank": list(res["rank"]),
                "covered": res["total_covered"],
                "total_tasks": res["total_tasks"],
                "penalty": round(res["total_penalty"], 4),
                "failures": res["failures"],
                "runtime": round(res["total_runtime"], 3),
            })
        if not results:
            return json.dumps({"ok": False, "error": "时间预算耗尽, 未执行"})
        best = ctx.best
        return json.dumps({
            "ok": True, "executed": results,
            "best_name": best["name"] if best else None,
            "best_penalty": round(best["total_penalty"], 4) if best else None,
            "best_covered": best["total_covered"] if best else None,
            "time_left": round(ctx.time_left(), 2),
        })

    def _tool_view_history(top_k: int = 5) -> str:
        usable = sorted(ctx.history, key=lambda r: r["rank"])[:max(1, top_k)]
        out = []
        for res in usable:
            out.append({
                "name": res["name"],
                "rank": list(res["rank"]),
                "covered": res["total_covered"],
                "penalty": round(res["total_penalty"], 4),
                "use_flow": res["config"].get("use_flow"),
                "use_beam": res["config"].get("use_beam"),
                "use_sa": res["config"].get("use_sa"),
                "profiles": _profile_names(res["config"].get("profiles", [])),
            })
        return json.dumps({"ok": True, "history": out, "total": len(ctx.history)})

    def _tool_finalize(reason: str = "") -> str:
        if reason:
            ctx.log(f"finalize requested: {reason}")
        return json.dumps({
            "ok": True,
            "best_name": ctx.best["name"] if ctx.best else None,
            "best_penalty": round(ctx.best["total_penalty"], 4) if ctx.best else None,
            "iterations": ctx.iteration,
        })

    tools = [
        StructuredTool.from_function(
            func=_tool_inspect_dataset, name="inspect_dataset",
            description="返回数据集统计特征 (任务/骑手数量、合单比例、平均意愿等), 帮助决定策略方向.",
        ),
        StructuredTool.from_function(
            func=_tool_propose_strategy, name="propose_strategy",
            description="把一个策略加入待评估队列. 通过 profile 列表 + use_flow/use_beam/use_sa 描述策略.",
            args_schema=ProposeStrategyArgs,
        ),
        StructuredTool.from_function(
            func=_tool_mutate_best, name="mutate_best",
            description="基于当前 best 在 focus 方向上变异, 加入队列. focus: balanced/coverage/bundle/willingness/diversify.",
            args_schema=MutateBestArgs,
        ),
        StructuredTool.from_function(
            func=_tool_run_pending, name="run_pending",
            description="执行 pending 队列中的所有策略 (受时间预算约束), 返回评估结果汇总.",
        ),
        StructuredTool.from_function(
            func=_tool_view_history, name="view_history",
            description="按排名返回历史评估结果前 k 条.",
            args_schema=ViewHistoryArgs,
        ),
        StructuredTool.from_function(
            func=_tool_finalize, name="finalize",
            description="终止迭代, 把当前 best 写出. 返回 best 摘要.",
            args_schema=FinalizeArgs,
        ),
    ]
    return tools, {tool.name: tool for tool in tools}


# ===========================================================================
# 5. LangGraph 状态机
# ===========================================================================

class AgentState(TypedDict, total=False):
    phase: str            # setup / analyze / loop / finalize
    iteration: int
    finalized: bool
    last_action: str
    note: str


def make_setup_node(ctx: AgentContext, reference_path: Optional[str]):
    def node(state: AgentState) -> AgentState:
        ctx.log(f"loaded {len(ctx.cases)} case(s); per-case timeout={ctx.per_case_timeout:.1f}s")
        ctx.evaluate_reference(reference_path)
        return {"phase": "analyze", "iteration": 0, "finalized": False}

    return node


def make_analyze_node(ctx: AgentContext):
    def node(state: AgentState) -> AgentState:
        agg = ctx.aggregate_features()
        ctx.log(
            "dataset summary: tasks={tasks} couriers={couriers} pair_ratio={pair:.2f} "
            "avg_willingness={will:.2f}".format(
                tasks=agg.get("task_count"),
                couriers=agg.get("courier_count"),
                pair=float(agg.get("pair_ratio") or 0.0),
                will=float(agg.get("avg_willingness") or 0.0),
            )
        )
        # 注入 8 个初始策略, 覆盖主流方向, 给后续控制器留迭代空间.
        for recipe in INITIAL_STRATEGY_RECIPES:
            strategy = make_strategy(
                name=recipe["name"],
                seed=ctx.seed,
                per_case_timeout=ctx.search_per_case_timeout,
                profile_names=recipe["profiles"],
                use_flow=recipe.get("use_flow", False),
                use_beam=recipe.get("use_beam", False),
                use_sa=recipe.get("use_sa", False),
            )
            ctx.enqueue(strategy)
        return {"phase": "loop", "note": "seeded initial strategies"}

    return node


def make_finalize_node(ctx: AgentContext, output_path: str):
    def node(state: AgentState) -> AgentState:
        if ctx.best is None and not ctx.history:
            ctx.log("no candidate yet -> falling back to expected_greedy")
            fallback = make_strategy(
                "expected_greedy_fallback", ctx.seed, ctx.search_per_case_timeout,
                ["expected", "coverage", "willingness"],
            )
            ctx.evaluate_strategy(fallback, timeout=ctx.search_per_case_timeout, track=True)
        chosen = ctx.finalize_best()
        if chosen is None:
            raise RuntimeError("Agent failed to produce any valid solution")
        path = output_path if os.path.isabs(output_path) else os.path.abspath(output_path)
        with open(path, "w") as handle:
            handle.write(chosen["code"])
        ctx.log(
            f"wrote solver to {path} "
            f"(time_limit={chosen['config'].get('time_limit')})"
        )
        return {"phase": "done", "finalized": True}

    return node


# ----- 启发式控制器 ---------------------------------------------------------

def heuristic_loop_node(ctx: AgentContext) -> Callable[[AgentState], AgentState]:
    """启发式控制器, 模拟 LLM 在工具间的反复调用."""

    def node(state: AgentState) -> AgentState:
        rng = ctx.rng
        # 第一步: 跑光初始候选.
        if ctx.pending:
            while ctx.pending and not ctx.out_of_time(margin=ctx.search_per_case_timeout * len(ctx.cases) + 0.5):
                ctx.iteration += 1
                ctx.evaluate_strategy(ctx.pending.pop(0))

        # 之后基于历史不断变异 + 评估, 直到时间耗尽.
        focus_cycle = ["balanced", "coverage", "bundle", "willingness", "diversify"]
        cycle_index = 0
        no_improve_streak = 0
        last_best_rank = ctx.best["rank"] if ctx.best else None
        while not ctx.out_of_time(margin=ctx.search_per_case_timeout * len(ctx.cases) + 0.5):
            focus = focus_cycle[cycle_index % len(focus_cycle)]
            cycle_index += 1
            mutated = _mutate_from_history(ctx, focus, rng)
            if mutated is None:
                continue
            if not ctx.enqueue(mutated):
                continue
            ctx.iteration += 1
            ctx.evaluate_strategy(mutated)
            current_rank = ctx.best["rank"] if ctx.best else None
            if current_rank == last_best_rank:
                no_improve_streak += 1
            else:
                no_improve_streak = 0
                last_best_rank = current_rank
            # 长时间没改进, 增加扰动强度.
            if no_improve_streak >= 4:
                shake = _shake_best(ctx, rng)
                if shake and ctx.enqueue(shake):
                    ctx.iteration += 1
                    ctx.evaluate_strategy(shake)
                no_improve_streak = 0

        return {"phase": "finalize", "iteration": ctx.iteration, "last_action": "heuristic_loop"}

    return node


def _mutate_from_history(ctx: AgentContext, focus: str, rng: random.Random) -> Optional[Dict[str, Any]]:
    candidates = sorted(ctx.history, key=lambda r: r["rank"])[:max(1, min(4, len(ctx.history)))]
    if not candidates:
        return None
    parent = rng.choice(candidates)
    cfg = copy.deepcopy(parent["config"])
    if cfg.get("reference"):
        cfg = base_config(ctx.seed, ctx.search_per_case_timeout)
        cfg["profiles"] = [copy.deepcopy(PROFILE_LIBRARY["expected"]),
                           copy.deepcopy(PROFILE_LIBRARY["coverage"])]
    if focus == "coverage":
        cfg["use_beam"] = True
        cfg["extra_limit"] = min(200, cfg.get("extra_limit", 80) + 15)
        for p in cfg.get("profiles", []):
            p["coverage_weight"] = p.get("coverage_weight", 0.0) + 8.0 + 8.0 * rng.random()
    elif focus == "bundle":
        for p in cfg.get("profiles", []):
            p["pair_weight"] = p.get("pair_weight", 0.0) + 8.0 + 12.0 * rng.random()
        cfg["use_beam"] = True
    elif focus == "willingness":
        cfg["use_flow"] = True
        for p in cfg.get("profiles", []):
            p["willingness_weight"] = p.get("willingness_weight", 0.0) + 6.0 + 10.0 * rng.random()
    elif focus == "diversify":
        cfg["use_sa"] = True
        cfg["sa_temp"] = cfg.get("sa_temp", 30.0) + rng.uniform(-5.0, 12.0)
        for p in cfg.get("profiles", []):
            p["random_weight"] = max(p.get("random_weight", 0.0), 4.0) + rng.uniform(0.0, 5.0)
    else:  # balanced
        cfg["local_rounds"] = min(8, cfg.get("local_rounds", 3) + (1 if rng.random() < 0.5 else 0))
        for p in cfg.get("profiles", []):
            p["willingness_weight"] = max(0.0, p.get("willingness_weight", 0.0) + rng.uniform(-6.0, 10.0))
            p["pair_weight"] = max(0.0, p.get("pair_weight", 0.0) + rng.uniform(-8.0, 12.0))
            p["score_weight"] = max(0.0, p.get("score_weight", 0.0) + rng.uniform(-0.04, 0.06))
    cfg["seed"] = ctx.seed + rng.randrange(1, 99991)
    name = f"mut_{focus}_{ctx.iteration + 1:03d}"
    return {"name": name, "config": cfg, "code": render_solver(cfg)}


def _shake_best(ctx: AgentContext, rng: random.Random) -> Optional[Dict[str, Any]]:
    """大幅扰动 best 配置, 用于跳出局部最优."""
    if ctx.best is None:
        return None
    cfg = copy.deepcopy(ctx.best["config"])
    if cfg.get("reference"):
        return None
    cfg["use_flow"] = not cfg.get("use_flow", False)
    cfg["use_beam"] = not cfg.get("use_beam", False)
    cfg["use_sa"] = True
    cfg["sa_temp"] = 40.0 + 20.0 * rng.random()
    cfg["loop_random_weight"] = max(cfg.get("loop_random_weight", 8.0), 12.0)
    for p in cfg.get("profiles", []):
        p["random_weight"] = max(p.get("random_weight", 0.0), 6.0) + rng.random() * 4.0
    cfg["seed"] = ctx.seed + rng.randrange(1, 99991)
    name = f"shake_{ctx.iteration + 1:03d}"
    return {"name": name, "config": cfg, "code": render_solver(cfg)}


# ----- LLM 控制器 -----------------------------------------------------------

LLM_SYSTEM_PROMPT = """\
你是 AutoSolver Agent 的策略大脑, 负责决策外卖配送任务-骑手指派问题的搜索方向.

你拥有以下工具:
1. inspect_dataset() — 查看数据集特征.
2. propose_strategy(name, profiles, use_flow, use_beam, use_sa, ...) — 加入新策略.
3. mutate_best(name, focus, seed_delta) — 基于当前 best 在指定方向变异.
4. run_pending() — 执行 pending 队列, 返回所有评估结果与新 best.
5. view_history(top_k) — 查看历史排名前 k 条.
6. finalize(reason) — 终止迭代.

工作流程:
1) 先调用 inspect_dataset 获取数据特征;
2) 通过 propose_strategy / mutate_best 提出 1~3 个策略;
3) 调用 run_pending 评估并获得反馈;
4) 根据反馈继续迭代; 时间紧或收益边际下降时调用 finalize.

重要原则:
- 已经在初始化时入队了 8 个 baseline 策略, 你的第一次 run_pending 会把它们全部评估;
- 评估开销大 (单条最多 ~30 秒), 因此每次 run_pending 之前最好已排好 1-3 个新策略;
- 优先优化排名 (rank 越小越好): rank=(failures, -covered, total_penalty, total_runtime);
- 时间预算耗尽时若还没 finalize, 系统会自动写出当前 best.
"""


def llm_loop_node(ctx: AgentContext, llm: "ChatOpenAI", tool_objs: List[StructuredTool]) -> Callable[[AgentState], AgentState]:
    tool_map = {t.name: t for t in tool_objs}
    llm_with_tools = llm.bind_tools(tool_objs)

    def node(state: AgentState) -> AgentState:
        messages: List[BaseMessage] = [
            SystemMessage(content=LLM_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "请开始. 你有总预算约 "
                    f"{ctx.time_left():.0f} 秒可用; 已经预先入队 8 个 baseline 策略, "
                    "建议第一步就 run_pending."
                )
            ),
        ]
        max_turns = 24
        finalized = False
        for turn in range(max_turns):
            if ctx.out_of_time(margin=ctx.per_case_timeout * len(ctx.cases) + 1.0):
                ctx.log("LLM loop: budget about to be exhausted -> stop calling LLM")
                break
            try:
                response: AIMessage = llm_with_tools.invoke(messages)  # type: ignore
            except Exception as exc:
                ctx.log(f"LLM invocation failed: {exc}; falling back to heuristic")
                heuristic_loop_node(ctx)({})
                break
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                ctx.log("LLM produced no tool calls -> stop")
                break
            for tc in tool_calls:
                tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                tool_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                tool_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                if tool_name not in tool_map:
                    payload = json.dumps({"ok": False, "error": f"unknown tool {tool_name}"})
                else:
                    try:
                        payload = tool_map[tool_name].invoke(tool_args or {})
                    except Exception as exc:
                        payload = json.dumps({"ok": False, "error": str(exc)})
                if not isinstance(payload, str):
                    payload = json.dumps(payload, default=str)
                messages.append(ToolMessage(content=payload, tool_call_id=tool_id))
                if tool_name == "finalize":
                    finalized = True
            if finalized:
                break
        # 残余: 如果 LLM 终止前还有 pending 未跑完, 自己跑掉.
        while ctx.pending and not ctx.out_of_time(margin=ctx.search_per_case_timeout * len(ctx.cases) + 0.5):
            ctx.iteration += 1
            ctx.evaluate_strategy(ctx.pending.pop(0))
        # 仍有时间就退化到启发式继续榨取.
        if not ctx.out_of_time(margin=ctx.search_per_case_timeout * len(ctx.cases) + 0.5):
            heuristic_loop_node(ctx)({})
        return {"phase": "finalize", "iteration": ctx.iteration, "last_action": "llm_loop"}

    return node


# ===========================================================================
# 6. Agent 顶层
# ===========================================================================

def maybe_build_llm() -> Optional["ChatOpenAI"]:
    if ChatOpenAI is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if not api_key:
        return None
    model = os.environ.get("AUTOSOLVER_LLM_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    try:
        kwargs: Dict[str, Any] = {"model": model, "temperature": 0.2}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)  # type: ignore
    except Exception as exc:
        print(f"[agent] LLM init failed: {exc}; falling back to heuristic", flush=True)
        return None


def build_graph(ctx: AgentContext, output_path: str, reference_path: Optional[str], use_llm: bool):
    tools, _ = build_tools(ctx)
    builder = StateGraph(AgentState)
    builder.add_node("setup", make_setup_node(ctx, reference_path))
    builder.add_node("analyze", make_analyze_node(ctx))
    if use_llm:
        llm = maybe_build_llm()
        if llm is None:
            use_llm = False
    if use_llm:
        builder.add_node("loop", llm_loop_node(ctx, llm, tools))  # type: ignore
        ctx.log("controller: LLM (langchain_openai)")
    else:
        builder.add_node("loop", heuristic_loop_node(ctx))
        ctx.log("controller: heuristic (offline)")
    builder.add_node("finalize", make_finalize_node(ctx, output_path))
    builder.set_entry_point("setup")
    builder.add_edge("setup", "analyze")
    builder.add_edge("analyze", "loop")
    builder.add_edge("loop", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


class AutoSolverLangChainAgent:
    def __init__(
        self,
        case_paths: Optional[List[str]] = None,
        output_path: str = "generated_submit_solution.py",
        reference_solver_path: Optional[str] = "submit_solution.py",
        budget_seconds: float = 90.0,
        per_case_timeout: float = 10.0,
        seed: int = 20260524,
        max_cases: int = 3,
        use_llm: Optional[bool] = None,
        verbose: bool = True,
        search_per_case_timeout: Optional[float] = None,
        final_solver_time_limit: float = 8.75,
    ) -> None:
        self.case_paths = case_paths or []
        self.output_path = output_path
        self.reference_solver_path = reference_solver_path
        self.budget_seconds = budget_seconds
        self.per_case_timeout = per_case_timeout
        self.search_per_case_timeout = search_per_case_timeout or per_case_timeout
        self.final_solver_time_limit = final_solver_time_limit
        self.seed = seed
        self.max_cases = max_cases
        self.use_llm = use_llm if use_llm is not None else True
        self.verbose = verbose

    def load_cases(self) -> List[Dict[str, Any]]:
        paths = list(self.case_paths)
        if not paths:
            for name in sorted(os.listdir(os.getcwd())):
                if not name.endswith(".txt"):
                    continue
                if name in ("describe.txt", "example_solution.txt"):
                    continue
                paths.append(os.path.abspath(name))
        result: List[Dict[str, Any]] = []
        for path in paths:
            if len(result) >= self.max_cases:
                break
            try:
                with open(path, "r") as handle:
                    text = handle.read()
                lines = text.splitlines()
                if not lines or "task_id_list" not in lines[0]:
                    continue
                result.append({"name": os.path.basename(path), "text": text})
            except Exception:
                continue
        return result

    def synthetic_cases(self) -> List[Dict[str, Any]]:
        cases = []
        rng = random.Random(self.seed)
        for case_id in range(3):
            sub = random.Random(self.seed + case_id)
            task_count = 8 + case_id * 4
            courier_count = 9 + case_id * 5
            lines = ["task_id_list\tcourier_id\ttotal_score\twillingness"]
            for task in range(task_count):
                couriers = list(range(courier_count))
                sub.shuffle(couriers)
                for courier in couriers[:min(courier_count, 6)]:
                    score = 5.0 + sub.random() * 80.0
                    willingness = 0.08 + sub.random() * 0.85
                    lines.append("t%d\tc%d\t%.6f\t%.6f" % (task, courier, score, willingness))
            for task in range(task_count - 1):
                if sub.random() < 0.65:
                    pair = "t%d,t%d" % (task, task + 1)
                    couriers = list(range(courier_count))
                    sub.shuffle(couriers)
                    for courier in couriers[:min(courier_count, 4)]:
                        score = 10.0 + sub.random() * 110.0
                        willingness = 0.05 + sub.random() * 0.75
                        lines.append("%s\tc%d\t%.6f\t%.6f" % (pair, courier, score, willingness))
            cases.append({"name": "synthetic_%d" % case_id, "text": "\n".join(lines)})
        rng.random()  # avoid unused warning
        return cases

    def run(self) -> Dict[str, Any]:
        cases = self.load_cases() or self.synthetic_cases()
        deadline = time.time() + self.budget_seconds
        rng = random.Random(self.seed)
        ctx = AgentContext(
            cases=cases,
            seed=self.seed,
            per_case_timeout=self.per_case_timeout,
            search_per_case_timeout=self.search_per_case_timeout,
            final_solver_time_limit=self.final_solver_time_limit,
            deadline=deadline,
            rng=rng,
            verbose=self.verbose,
        )
        graph = build_graph(
            ctx=ctx,
            output_path=self.output_path,
            reference_path=self.reference_solver_path,
            use_llm=self.use_llm,
        )
        graph.invoke({"phase": "setup"})
        report = self._build_report(ctx)
        try:
            with open(self.output_path + ".report.json", "w") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
        except Exception:
            pass
        return report

    def _build_report(self, ctx: AgentContext) -> Dict[str, Any]:
        def trim(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if result is None:
                return None
            return {
                "name": result["name"],
                "rank": list(result["rank"]),
                "total_covered": result["total_covered"],
                "total_tasks": result["total_tasks"],
                "total_penalty": round(result["total_penalty"], 4),
                "total_runtime": round(result["total_runtime"], 3),
                "failures": result["failures"],
                "cases": result["cases"],
            }

        return {
            "output_path": os.path.abspath(self.output_path),
            "cases": [c["name"] for c in ctx.cases],
            "best": trim(ctx.best),
            "reference": trim(ctx.reference) if ctx.reference else None,
            "evaluated_candidates": len(ctx.history),
            "iterations": ctx.iteration,
            "controller": "llm" if (self.use_llm and maybe_build_llm() is not None) else "heuristic",
        }


# ===========================================================================
# 7. CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain/LangGraph 重写版 AutoSolver Agent")
    parser.add_argument("--cases", nargs="*", default=None, help="测试用例文件路径")
    parser.add_argument("--out", default="generated_submit_solution.py", help="生成的求解器文件路径")
    parser.add_argument("--reference", default="submit_solution.py", help="参考求解器文件 (用于基线对比)")
    parser.add_argument("--budget", type=float, default=90.0, help="Agent 总时间预算 (秒)")
    parser.add_argument("--per-case-timeout", type=float, default=10.0, help="判题机等价超时 (秒), 用于 final 复选")
    parser.add_argument("--search-per-case-timeout", type=float, default=None, help="搜索期超时 (秒), 默认=per-case-timeout; 调小以多迭代")
    parser.add_argument("--final-time-limit", type=float, default=8.75, help="最终求解器内部 time_limit (秒)")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--no-llm", action="store_true", help="强制使用启发式控制器")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    agent = AutoSolverLangChainAgent(
        case_paths=args.cases,
        output_path=args.out,
        reference_solver_path=args.reference,
        budget_seconds=args.budget,
        per_case_timeout=args.per_case_timeout,
        search_per_case_timeout=args.search_per_case_timeout,
        final_solver_time_limit=args.final_time_limit,
        seed=args.seed,
        max_cases=args.max_cases,
        use_llm=not args.no_llm,
        verbose=not args.quiet,
    )
    report = agent.run()
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
