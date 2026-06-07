# AutoSolver Agent

当前发布版本：`v1.5.0`

AutoSolver Agent 是一个面向配送分配问题的自动求解器生成系统。它基于 LangChain、LangGraph 和 OpenAI 兼容 LLM 接口，将实例分析、策略规划、候选代码生成、安全验证、评分、长期记忆和最终复核组织成一条可追踪的迭代工作流。

项目目标不是固定运行某一个手写启发式 solver，而是根据输入 case 的结构和历史实验结果，让 LLM 维护一套可演进的求解框架，并持续生成满足 `solve(input_text: str) -> list` 契约的 Python 求解器。

## 主要能力

- LLM 维护的求解框架：运行时自动创建并更新 feature dimension、strategy 和 skill 记忆，不再依赖硬编码策略目录。
- LangGraph 工作流：核心链路为 `classify -> generate -> validate_and_score -> finalize`。
- LangChain 工具化规划：规划阶段可读取实例特征、求解框架、相似历史实验、UCB bandit 推荐和当前最佳 artifact 摘要。
- 多策略并行：`--strategy-workers` 为 1 时在当前进程运行，2 个以上时可启动独立 worker 进程并全局复评最优候选。
- 基线 solver 导入：可通过 `--baseline-solver` 或 `--base-solver` 将已有 `.py` solver 纳入同一验证、评分和最终候选池。
- 验证与修复闭环：结构化输出失败或候选验证失败后，可由 LLM 在受控次数内修复。
- 子进程沙箱：候选代码在独立进程中执行，受 import 白名单、危险调用检查、CPU 时间和内存限制保护。
- 可审计产物：每个候选的代码、rationale、validation、score、impact、事件日志、最终报告都会落盘。

## 适用问题

输入是 TSV 格式的配送分配候选集。每一行表示一个任务组 `task_id_list` 可以分配给某个 `courier_id`，并带有该组合的 `total_score` 和 `willingness`。

Agent 生成的 solver 需要返回一组不冲突的任务组和骑手列表。评分目标优先级为：

1. 更少失败 case。
2. 覆盖更多任务。
3. 更低总 penalty。
4. 更短运行时间。

内部排序 rank 为：

```text
(failures, -total_covered, total_penalty, total_runtime)
```

## 快速开始

准备 Python 3.10 及以上环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

配置 OpenAI 兼容 LLM：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export AUTOSOLVER_LLM_MODEL="gpt-4o-mini"
```

运行示例 case：

```bash
autosolver-agent \
  --cases examples/demo_case.txt \
  --out runs/manual/generated_submit_solution.py \
  --budget 90 \
  --iterations 3 \
  --strategy-workers 1 \
  --summary-out runs/manual/summary.json
```

也可以使用仓库中的脚本：

```bash
./run.sh examples/demo_case.txt
```

推荐通过环境变量提供 API key、模型、输出目录和超时配置，不要把真实密钥写入源码或提交到仓库。

## CLI 参数

命令入口来自 `pyproject.toml`：

```bash
autosolver-agent --help
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--cases` | 必填 | 一个或多个 case TSV 文件。 |
| `--out` | `generated_submit_solution.py` | 最终 solver 输出路径。 |
| `--budget` | `90.0` | 整体运行预算，单位秒。 |
| `--per-case-timeout` | `10.0` | 最终复核和正式评分的单 case 超时。 |
| `--search-per-case-timeout` | 同 `--per-case-timeout` | 生成搜索阶段的单 case 超时。 |
| `--iterations` | `3` | LLM 改进迭代次数。 |
| `--strategy-workers` | `5` | 策略 worker 数。`1` 为单工作流，`2+` 为多进程并行。 |
| `--baseline-solver` / `--base-solver` | 无 | 导入已有 solver 文件，可重复传入。 |
| `--finalize-top-k` | `3` | 最终复核排名靠前的候选数量。 |
| `--max-repair-attempts` | `2` | schema 或验证失败后的修复尝试次数。 |
| `--memory-top-k` | `5` | 相似历史实验检索数量。 |
| `--bandit-exploration` | `1.4` | UCB bandit 探索系数。 |
| `--memory-dir` | `runs/autosolver_memory` | 长期、短期和框架记忆目录。 |
| `--artifact-dir` | `runs/autosolver_artifacts` | 候选代码和中间产物目录。 |
| `--event-log` | `artifact-dir/events.jsonl` | JSONL 事件日志路径。 |
| `--summary-out` | 无 | 仅写出简要 summary JSON。 |
| `--llm-model` | 环境变量或 `gpt-4o-mini` | LLM 模型名。 |
| `--llm-base-url` | 环境变量 | OpenAI 兼容 base URL。 |
| `--quiet` | false | 关闭运行日志输出。 |

查看版本：

```bash
autosolver-agent --version
```

## Python API

```python
from autosolver_agent import AutoSolverLangChainAgent

agent = AutoSolverLangChainAgent(
    case_paths=["examples/demo_case.txt"],
    output_path="runs/manual/generated_submit_solution.py",
    budget_seconds=90,
    iterations=3,
    strategy_workers=1,
)

report = agent.run()
print(report["summary"])
```

测试中也支持传入 fake LLM：

```python
agent = AutoSolverLangChainAgent(
    case_paths=["case.txt"],
    output_path="generated_submit_solution.py",
    llm=fake_llm,
    strategy_workers=1,
)
```

传入 `llm` 时会在当前进程运行，方便单元测试和本地调试。

## 输入格式

case 文件必须是 UTF-8 TSV 文本，首行包含：

```text
task_id_list	courier_id	total_score	willingness
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `task_id_list` | 任务 ID 或合单任务 ID，多个任务用英文逗号连接，例如 `t0,t1`。 |
| `courier_id` | 可承接该任务组的骑手 ID。 |
| `total_score` | 该任务组分配给该骑手后的预计算成本或分数。 |
| `willingness` | 骑手接起该任务组的概率。 |

示例：

```text
task_id_list	courier_id	total_score	willingness
t0	c0	10	0.8
t0	c1	30	0.3
t1	c1	12	0.7
t0,t1	c2	40	0.6
```

解析器会拒绝缺少表头、字段不足、空任务组、空骑手或非数值分数的 case，并通过 `CaseParseError` 携带诊断信息。

## Solver 输出契约

最终生成文件必须定义顶层函数：

```python
def solve(input_text: str) -> list:
    ...
```

返回值必须是：

```python
list[tuple[str, list[str]]]
```

每个元素表示：

```python
("task_id_list", ["courier_id", "..."])
```

约束：

- `task_id_list` 必须是输入中存在的任务组字符串，例如 `t0` 或 `t0,t1`。
- `courier_ids` 必须是非空 `list[str]`，不能是单个字符串。
- 同一个任务不能在多个返回项中重复出现。
- 同一个骑手不能在同一返回项或不同返回项中重复使用。
- 每个骑手都必须对该任务组有效。
- 允许不覆盖全部任务，未覆盖任务会按每任务 `100.0` fallback 进入 penalty。

候选代码安全限制：

- 必须是自包含 Python 代码。
- 不允许文件 IO、网络 IO、subprocess、动态 import、`eval`、`exec`、`compile`。
- 允许的 import 根包括 `bisect`、`collections`、`copy`、`dataclasses`、`functools`、`heapq`、`itertools`、`math`、`operator`、`random`、`statistics`、`time`、`typing`，以及运行环境可用的 `numpy`、`scipy`、`networkx`。

## 评分模型

单个任务组的 penalty 由候选骑手集合共同决定：

```text
fallback = 100.0 * task_count
reject_prob = product(1.0 - willingness_i)
weighted_score = sum(willingness_i * score_i) / sum(willingness_i)
penalty = reject_prob * fallback + (1.0 - reject_prob) * weighted_score
```

如果没有有效 willingness 权重，则使用 fallback。

整份答案的 penalty 为所有返回任务组 penalty 之和，再加上未覆盖任务 fallback。无效答案、运行错误或超时会被赋予高额失败 penalty，并在 rank 中优先落后。

## 架构

```mermaid
flowchart TB
  CLI["CLI / Python API"] --> Agent["AutoSolverLangChainAgent"]
  Agent --> CaseIO["caseio: case loading, diagnostics, features, scoring primitives"]
  Agent --> Runner["AutoSolverRunner"]
  Runner --> Workflow["AutoSolverWorkflow"]
  Workflow --> Graph["LangGraph classify -> generate -> validate_and_score -> finalize"]
  Workflow --> Framework["FrameworkStore"]
  Workflow --> Memory["MemoryStore"]
  Workflow --> Artifacts["ArtifactStore"]
  Workflow --> LLM["LLMCodeGenerator"]
  Workflow --> Tools["PlannerToolbox"]
  Tools --> LLM
  LLM --> Candidate["Candidate solver code"]
  Candidate --> Validator["Validator"]
  Validator --> Runtime["runtime.run_candidate subprocess"]
  Runtime --> Scorer["Scorer"]
  Scorer --> Memory
  Scorer --> Artifacts
  Workflow --> Final["Final recheck and output solver"]
```

运行流程：

1. `AutoSolverLangChainAgent` 加载 case、解析数据、计算 deadline，并创建 `AutoSolverRunConfig`。
2. `AutoSolverRunner` 根据 `strategy_workers` 决定当前进程单工作流或多进程 worker 模式。
3. `AutoSolverWorkflow` 在 `classify` 阶段提取 objective features，并由 LLM 维护求解框架和实例解释。
4. `generate` 阶段通过 LangChain tools 规划 `SolverPlan`，再生成一个或多个 `CandidateEnvelope`。
5. `validate_and_score` 阶段先做 AST 静态检查，再运行 smoke case，最后对有效候选评分。
6. 验证失败或结构化输出失败时，`RepairService` 调用 LLM 修复候选。
7. 每轮评估后，LLM 根据实验结果向 `FrameworkStore` 提交 partial update。
8. `finalize` 阶段按正式超时复核 top-k 候选，输出最终 solver、summary 和完整 report。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `autosolver_agent.cli` | 命令行入口，解析参数并打印 JSON report。 |
| `autosolver_agent.agent` | Python API 主入口，装配 case、runner 和配置。 |
| `autosolver_agent.caseio` | case 解析、诊断、特征提取、penalty 与答案评分。 |
| `autosolver_agent.framework` | LLM 维护的 feature、strategy、skill 框架记忆及安全校验。 |
| `autosolver_agent.llm.generator` | LLM 调用、规划、代码生成、修复、框架 bootstrap 和反思更新。 |
| `autosolver_agent.llm.schema` | `SolverPlan`、`CandidateEnvelope` 等结构化输出协议。 |
| `autosolver_agent.workflow.runner` | 单进程或多 worker 运行编排，负责全局最终复评。 |
| `autosolver_agent.workflow.graph` | LangGraph 节点实现和候选生成、验证、评分、收尾逻辑。 |
| `autosolver_agent.workflow.services` | 生成、评估、修复、最终化和报告构建服务包装。 |
| `autosolver_agent.tools.langchain_tools` | 暴露给 planning LLM 的只读工具。 |
| `autosolver_agent.tools.validator` | 静态 AST 安全检查和 smoke runtime 校验。 |
| `autosolver_agent.tools.scorer` | 多 case 候选评分与收敛比较。 |
| `autosolver_agent.runtime` | 子进程执行候选 solver，并限制资源和 import。 |
| `autosolver_agent.memory` | 长期实验记忆、短期运行记忆、相似检索和 UCB bandit。 |
| `autosolver_agent.artifacts` | 候选代码、rationale、validation、score、impact 和 JSON 原子写入。 |
| `autosolver_agent.events` | JSONL 事件日志、阶段计时和候选代码 hash。 |
| `solvers` | 参考 solver、seed solver 和历史最佳 solver。 |
| `examples` | 示例 case、示例提交和 solver 模板。 |
| `tests` | 使用 fake LLM 覆盖 parser、validator、memory、runner、repair 和 CLI。 |

## 目录结构

```text
.
├── autosolver_agent/
│   ├── agent.py
│   ├── artifacts.py
│   ├── caseio.py
│   ├── cli.py
│   ├── events.py
│   ├── framework.py
│   ├── runtime.py
│   ├── llm/
│   ├── memory/
│   ├── skills/
│   ├── tools/
│   └── workflow/
├── examples/
├── solvers/
├── tests/
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── run.sh
├── README.md
└── RELEASE.md
```

运行产物默认写入 `runs/`，该目录不属于发布源码。

## 产物说明

默认产物包括：

| 路径 | 内容 |
| --- | --- |
| `--out` | 最终 solver Python 文件。 |
| `--out.report.json` | 完整运行报告。 |
| `--summary-out` | 可选简要 summary。 |
| `artifact-dir/events.jsonl` | 当前运行的结构化事件日志。 |
| `artifact-dir/iteration_XXX/*.py` | 每轮候选 solver。 |
| `artifact-dir/iteration_XXX/*.rationale.json` | 候选生成理由和策略信息。 |
| `artifact-dir/iteration_XXX/*.validation.json` | 静态和运行时验证结果。 |
| `artifact-dir/iteration_XXX/*.score.json` | 候选评分结果。 |
| `artifact-dir/iteration_XXX/*.impact.json` | 候选对当前最优解的影响分析。 |
| `memory-dir/long_term_memory.json` | 长期实验记忆和 bandit 统计。 |
| `memory-dir/framework_memory.json` | LLM 维护的求解框架。 |
| `memory-dir/short_term_last_run.json` | 最近一次短期运行记忆。 |

多 worker 模式下，每个 worker 会写入独立 artifact 子目录，例如：

```text
runs/autosolver_artifacts/worker_00/
runs/autosolver_artifacts/worker_01/
```

## Docker

构建镜像：

```bash
docker build \
  --build-arg VERSION=1.5.0 \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  -t autosolver-agent:1.5.0 .
```

查看版本：

```bash
docker run --rm autosolver-agent:1.5.0 --version
```

运行 case：

```bash
docker run --rm \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e OPENAI_BASE_URL="$OPENAI_BASE_URL" \
  -e AUTOSOLVER_LLM_MODEL="$AUTOSOLVER_LLM_MODEL" \
  -v "$PWD/examples:/app/examples:ro" \
  -v "$PWD/runs:/app/runs" \
  autosolver-agent:1.5.0 \
  --cases examples/demo_case.txt \
  --out runs/docker/generated_submit_solution.py \
  --budget 90 \
  --iterations 3
```

## 开发与验证

运行单元测试：

```bash
python -m unittest discover -s tests -v
```

运行静态检查：

```bash
ruff check .
mypy autosolver_agent
```

打包入口检查：

```bash
python -m pip install -e .
autosolver-agent --version
```

CI 当前执行：

- 安装 `.[dev]`。
- `ruff check .`。
- `mypy autosolver_agent`。
- `python -m unittest discover -s tests -v`。
- Docker build smoke test。

## 环境变量

| 环境变量 | 说明 |
| --- | --- |
| `OPENAI_API_KEY` / `OPENAI_KEY` | LLM API key。 |
| `OPENAI_BASE_URL` / `OPENAI_API_BASE` | OpenAI 兼容服务地址。 |
| `AUTOSOLVER_LLM_MODEL` | 默认模型名。 |
| `AUTOSOLVER_WIRE_API` / `OPENAI_WIRE_API` | 设为 `responses` 时启用 responses API。 |
| `AUTOSOLVER_REASONING_EFFORT` / `OPENAI_REASONING_EFFORT` | 传递 reasoning effort。 |
| `AUTOSOLVER_DISABLE_RESPONSE_STORAGE` / `OPENAI_DISABLE_RESPONSE_STORAGE` | 为真时向 ChatOpenAI 传入 `store=False`。 |
| `AUTOSOLVER_MEMORY_MAX_ITEMS` | 长期记忆每类列表保留上限。 |

`run.sh` 还会读取：

| 环境变量 | 默认值 |
| --- | --- |
| `PYTHON_BIN` | 自动检测。 |
| `AUTOSOLVER_OUT` | `runs/manual/generated_submit_solution.py` |
| `AUTOSOLVER_BUDGET` | `3600` |
| `AUTOSOLVER_ITERATIONS` | `100` |
| `AUTOSOLVER_STRATEGY_WORKERS` | `5` |
| `AUTOSOLVER_PER_CASE_TIMEOUT` | `10` |
| `AUTOSOLVER_SEARCH_PER_CASE_TIMEOUT` | `10` |
| `AUTOSOLVER_MEMORY_DIR` | `runs/autosolver_memory` |
| `AUTOSOLVER_ARTIFACT_DIR` | `runs/autosolver_artifacts` |
| `AUTOSOLVER_SUMMARY_OUT` | `runs/manual/summary.json` |
| `AUTOSOLVER_BASELINE_SOLVER` | 空，多个 solver 用 `:` 分隔。 |

## 发布说明

`v1.5.0` 将 README 作为唯一主说明文档，并同步 Python 包版本、CLI 版本和 Docker 镜像默认版本。详细发布记录见 `RELEASE.md`。
