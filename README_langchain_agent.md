# LangChain AutoSolver Agent

`langchain_autosolver_agent.py` 是基于 **LangChain + LangGraph** 重写的
AutoSolver Agent，对应 `describe.txt` 中 6 阶段任务流程：

接收输入 → 分析 → 策略生成 → 策略执行 → 评估筛选 → 迭代改进 → 输出结果。

## 文件

- `langchain_autosolver_agent.py`: Agent 主入口（LangGraph 状态机 + 工具
  + 双模式控制器）。
- `_solver_template.py`: 求解器代码模板，渲染后落到 `generated_submit_solution.py`。
- `requirements.txt`: Python 依赖。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

### 启发式控制器（默认无需任何 API Key）

```bash
python3 langchain_autosolver_agent.py \
    --cases large_seed301.txt \
    --budget 90 --per-case-timeout 10
```

会在当前目录写出 `generated_submit_solution.py` 与 `.report.json`。

### LLM 控制器（OpenAI 兼容）

设置以下环境变量后运行：

```bash
export OPENAI_API_KEY=sk-xxxx
# 可选：使用兼容服务（DeepSeek/Kimi/Qwen 等）
export OPENAI_BASE_URL=https://api.deepseek.com/v1
export AUTOSOLVER_LLM_MODEL=deepseek-chat   # 默认 gpt-4o-mini
python3 langchain_autosolver_agent.py --cases large_seed301.txt
```

LLM 通过 `bind_tools` 调用以下工具来自主决策：

| 工具 | 作用 |
|------|------|
| `inspect_dataset` | 数据集统计（任务/骑手数、合单比例、平均意愿）|
| `propose_strategy` | 加入新策略到队列 |
| `mutate_best` | 基于当前 best 在指定方向变异 |
| `run_pending` | 执行 pending 队列并返回排名 |
| `view_history` | 历史排名前 k |
| `finalize` | 终止迭代 |

强制使用启发式：`--no-llm`。

## 与原版差异

| 维度 | 旧 `autosolver_agent.py` | 新 `langchain_autosolver_agent.py` |
|------|--------------------------|------------------------------------|
| 框架 | 自实现循环 | LangGraph `StateGraph` 节点化 |
| 控制器 | 固定启发式 | LLM 工具调用 + 启发式双模式 |
| 求解原语 | greedy/flow/beam/local | 同左 + **simulated annealing** |
| 策略库 | 5 个 profile | 8 个 profile + diversify focus |
| 去重 | 无 | `tried_signatures` 避免重复评估 |
| 失败回退 | 不会兜底 | LLM 失败自动回退启发式 |

## 输出说明

- `generated_submit_solution.py`: 仅依赖标准库，可直接被 judge 加载，对应
  `solve(input_text: str) -> list` 接口。
- `generated_submit_solution.py.report.json`: 评估报告，含 best/reference
  对比、覆盖任务数、总罚分、运行时间等。
