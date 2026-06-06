# AutoSolver Agent

当前发布版本：`v1.0.0`

AutoSolver Agent 是一个基于 LangGraph / LangChain 的配送分配求解器生成系统。它不会只固定运行某个手写启发式算法，而是把“实例分析、策略规划、代码生成、验证、评分、记忆沉淀和迭代改进”组织成一个可审计的 Agent 工作流。

项目当前面向外卖配送分配问题：输入若干个 TSV case，LLM 生成一个完整的 Python `solve(input_text: str) -> list` 求解器，系统自动验证该求解器是否满足约束，按目标函数评分，将候选代码与实验结果写入磁盘，并在多轮迭代中保留当前最优解。

## 当前实现

- **LangGraph 工作流编排**：已实现 `classify -> generate -> validate_and_score -> finalize` 节点；`langgraph` 是硬依赖，缺失时启动即失败。
- **LangChain 工具化规划**：规划阶段向 LLM 暴露实例特征、策略库、相似历史实验、UCB bandit 推荐和当前最优 artifact 摘要。
- **结构化生成协议**：规划输出使用 `SolverPlan`，代码生成输出使用严格 JSON `CandidateEnvelope`，由 Pydantic 做 schema 校验；旧式 markdown 代码块不会被接受。
- **验证驱动修复**：支持 schema 失败修复和候选代码验证失败修复，修复次数由 `--max-repair-attempts` 控制。
- **可选并行策略探索**：`--strategy-workers 1` 运行单 workflow；`--strategy-workers 2+` 启动多个独立 worker 进程，每个 worker 维护自己的规划、生成、验证、评分循环，并在每轮结束后同步共享记忆。
- **标准库求解器约束**：生成的最终 solver 只允许依赖 Python 标准库，并必须提供 `solve(input_text: str) -> list`。
- **实验记忆**：保存短期运行记忆、长期实验记录、相似实验检索结果和策略组合的 UCB 统计。
- **可追溯产物**：每轮候选代码、生成理由、验证结果、评分结果、影响分析、最终报告都会落盘。

## v1.0.0 架构总览

系统按“入口层、工作流编排层、领域服务层、工具与知识层、持久化层、候选执行沙箱”拆分。主链路由 `AutoSolverLangChainAgent` 负责装配运行配置和依赖。`--strategy-workers 1` 时直接运行单个 `AutoSolverWorkflow`；`--strategy-workers 2+` 时交给 `ParallelAutoSolverRunner` 启动多个独立 worker 进程。

```mermaid
flowchart TB
  CLI["CLI / Python API"] --> Agent["AutoSolverLangChainAgent"]
  Agent --> CaseIO["caseio: TSV 读取、诊断、特征与评分基础函数"]
  Agent --> Single["AutoSolverWorkflow single worker"]
  Agent --> Parallel["ParallelAutoSolverRunner for 2+ workers"]
  Parallel --> Workflow["AutoSolverWorkflow per worker"]
  Single --> Framework["FrameworkStore: LLM 维护特征/策略/技能框架"]
  Workflow --> Framework["FrameworkStore: LLM 维护特征/策略/技能框架"]
  Workflow --> Planner["PlannerToolbox + SolverFramework"]
  Workflow --> LLM["LLMCodeGenerator"]
  Workflow --> Validator["Validator"]
  Workflow --> Scorer["Scorer"]
  Workflow --> Memory["MemoryStore"]
  Workflow --> Artifacts["ArtifactStore + EventRecorder"]
  Validator --> Runtime["runtime.run_candidate 子进程沙箱"]
  Scorer --> Runtime
  LLM --> Candidate["Candidate solve(input_text)"]
  Candidate --> Validator
  Validator --> Scorer
  Scorer --> Memory
  Scorer --> Artifacts
```

核心数据对象集中在 `autosolver_agent.models`：

- `Case` / `ParsedCase`：原始 case 与解析后的任务、骑手、任务组索引。
- `Candidate`：LLM 生成或修复后的完整 solver 代码及 rationale。
- `ValidationResult`：静态 AST 检查与 smoke run 的结果。
- `ScoreResult`：候选在 case 集上的 rank、覆盖数、penalty、运行时间与收敛信息。
- `IterationArtifact`：每轮候选代码、rationale、validation、score、impact 的文件路径。

运行时数据流：

1. CLI 或 Python API 创建 `AutoSolverLangChainAgent`。
2. `caseio` 加载 TSV case；坏行、坏 header 或读取失败会直接抛错，运行不会使用部分解析结果继续执行。
3. `caseio.dataset_features` / `aggregate_features` 汇总客观实例规模、任务组比例、意愿分布、容量等特征。
4. Agent 校验 `strategy_workers >= 1`。如果值为 1，则运行单 workflow；如果值大于等于 2，则为每个 worker 建立独立 artifact/event 目录和 LLM 客户端，共享同一个 memory 目录。
5. 每个 worker 每轮读取共享 `MemoryStore` 和 `FrameworkStore`，`PlannerToolbox` 向 LLM 暴露客观特征、LLM 维护的策略/技能框架、相似历史、UCB 推荐和当前最佳 artifact 摘要。
6. `LLMCodeGenerator` 为该 worker 生成 `SolverPlan` 和 `CandidateEnvelope`，并用 Pydantic 校验结构化输出。
7. `Validator` 先做 AST 安全检查，再用 `runtime.run_candidate` 在子进程中执行 smoke case；`Scorer` 对有效候选评分。
8. worker 每轮结束后将 candidate、validation、score、experiment 写入短期/长期记忆，并让 LLM 反思生成 `FrameworkUpdate` 写回共享 framework。
9. 下一轮 worker 重新读取共享 memory，让 LLM 基于最新记忆规划该 worker 的下一轮 plan。
10. 达到总迭代数或时间预算后，主进程汇总各 worker top-k 候选，按正式超时复评并输出全局最优 solver 与 JSON report。

## 目录结构

```text
.
├── README.md                            # 简短入口文档
├── README_langchain_agent.md            # 完整说明文档
├── RELEASE.md                           # v1.0.0 发布说明
├── Dockerfile                           # Docker 运行时镜像
├── pyproject.toml                       # 包元数据、依赖、工具配置
├── requirements.txt                     # 运行依赖清单
├── autosolver_agent/
│   ├── cli.py                           # autosolver-agent 命令行入口
│   ├── agent.py                         # 顶层 AutoSolverLangChainAgent
│   ├── artifacts.py                     # artifact 和 JSON 写盘
│   ├── caseio.py                        # case 解析、特征提取、评分基础函数
│   ├── events.py                        # JSONL 事件日志与阶段计时
│   ├── framework.py                     # LLM 维护框架 schema、校验与持久化
│   ├── models.py                        # 领域 dataclass
│   ├── runtime.py                       # 子进程执行候选 solver
│   ├── workflow/graph.py                # LangGraph 工作流
│   ├── workflow/services.py             # 工作流服务和运行状态封装
│   ├── llm/                             # LLM 规划、生成、修复和结构化 schema
│   ├── tools/                           # 验证器、评分器、LangChain planner tools
│   ├── memory/                          # 短期/长期记忆、相似检索、UCB bandit
│   └── skills/                          # 不再内置策略/技能目录，保留为空包
├── solvers/                             # 内置参考 solver 和 seed solver
├── examples/demo_case.txt               # 示例 case
├── tests/test_modular_agent.py          # 单元测试，使用 fake LLM
└── describe.txt                         # 原始需求与问题描述
```

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `autosolver_agent.cli` | 提供 `autosolver-agent` 可安装命令，解析 CLI 参数并输出完整 JSON report。 |
| `autosolver_agent.agent` | 装配 case、memory、artifact、LLM、workflow，是 Python API 主入口。 |
| `autosolver_agent.workflow.graph` | 定义实例解释、候选生成、验证评分、framework 反思更新和 finalize 流程。 |
| `autosolver_agent.workflow.services` | 封装生成、评估、修复、收尾、报告构建等服务边界。 |
| `autosolver_agent.llm` | 负责 `SolverFramework`、`InstanceInterpretation`、`SolverPlan`、`CandidateEnvelope`、`FrameworkUpdate` 的结构化生成与修复。 |
| `autosolver_agent.framework` | 持久化 LLM 维护的特征维度、策略知识、技能知识，并执行 schema/安全校验。 |
| `autosolver_agent.tools` | 提供 LangChain planner tools、安全验证和候选评分。 |
| `autosolver_agent.memory` | 持久化长期实验记忆，支持相似实验检索和 UCB explore/exploit 推荐。 |
| `autosolver_agent.runtime` | 在隔离子进程中运行候选 solver，限制内存、CPU、文件写入和 import 范围。 |
| `autosolver_agent.artifacts` | 将候选代码、rationale、validation、score、impact 和报告写入磁盘。 |
| `autosolver_agent.skills` | 不再提供内置策略/技能公共接口；策略和技能由 `FrameworkStore` 管理。 |
| `solvers` / `examples` | 提供可直接调用的 seed solver、历史最佳 solver、模板和示例 case。 |

## 输入格式

case 文件是制表符分隔的 TSV 文本，首行必须包含：

```text
task_id_list	courier_id	total_score	willingness
```

字段含义：

- `task_id_list`：任务 ID 或合单任务 ID，多个任务用逗号连接，例如 `t0,t1`。
- `courier_id`：可承接该任务组的骑手 ID。
- `total_score`：该任务组分配给该骑手后的预计算成本/分数。
- `willingness`：骑手接起该任务组的概率。

示例：

```text
task_id_list	courier_id	total_score	willingness
t0	c0	10	0.8
t1	c1	12	0.7
t0,t1	c2	40	0.6
```

## Solver 输出契约

Agent 最终生成的文件必须定义：

```python
def solve(input_text: str) -> list:
    ...
```

返回值必须是 Python list，形如：

```python
[("t0,t1", ["c2"]), ("t2", ["c0", "c3"])]
```

合法性约束：

- 每个任务最多出现一次。
- 每个骑手在全局最多出现一次。
- 输出中的 `task_key` 必须存在于输入。
- 输出中的每个骑手必须对该 `task_key` 有合法输入行。
- 每个任务组至少分配一个骑手。
- 返回 Python 对象，不是 JSON 字符串。

验证器还会拒绝候选代码中的危险能力，例如文件 IO、网络 IO、`subprocess`、动态 import、`eval`、`exec`、`compile` 等。

## 评分规则

系统对候选 solver 的排序目标是：

1. 失败 case 数更少。
2. 覆盖任务数更多。
3. 总 penalty 更低。
4. 总运行时间更短。

单个任务组分配给多个骑手时，系统按接单概率估算 penalty：

```text
reject_prob = Π(1 - willingness_i)
weighted_score = Σ(willingness_i * score_i) / Σ(willingness_i)
group_penalty = reject_prob * 100 * task_count + (1 - reject_prob) * weighted_score
```

未覆盖任务会额外按每个任务 `100` 加罚。候选排序 rank 在代码中表示为：

```text
(failures, -total_covered, total_penalty, total_runtime)
```

## Agent 工作流

```mermaid
flowchart LR
  A["读取 TSV cases"] --> B["实例分类器"]
  B --> E["启动 N 个 worker 进程"]
  E --> F1["worker 0: 读共享记忆 -> LLM plan -> 生成/验证/评分"]
  E --> F2["worker 1: 读共享记忆 -> LLM plan -> 生成/验证/评分"]
  E --> F3["worker N: 读共享记忆 -> LLM plan -> 生成/验证/评分"]
  F1 --> G["每轮结束加锁合并共享 memory"]
  F2 --> G
  F3 --> G
  G --> H["下一轮重新读取共享 memory"]
  H --> F1
  H --> F2
  H --> F3
  E --> J["主进程汇总 worker top-k"]
  J --> I["finalize top-k 复评"]
  I --> K["写出全局最优 solver 和报告"]
```

每轮迭代会记录：

- 规划 trace 和工具调用记录。
- 候选代码与 rationale。
- 验证结果与修复历史。
- 评分结果与是否改进当前最优。
- 对策略组合和参数变化的影响分析。
- 写入长期记忆的实验记录。

## 安装

建议在项目虚拟环境中安装依赖：

```bash
.venv/bin/python -m pip install -r requirements.txt
```

开发模式安装会同时注册 `autosolver-agent` 命令：

```bash
.venv/bin/python -m pip install -e ".[dev]"
autosolver-agent --version
```

真实 LLM 运行需要 OpenAI 兼容接口密钥：

```bash
export OPENAI_API_KEY=sk-...
export AUTOSOLVER_LLM_MODEL=gpt-4o-mini
```

可选配置：

```bash
export OPENAI_BASE_URL=https://api.example.com/v1
export AUTOSOLVER_WIRE_API=responses
export AUTOSOLVER_REASONING_EFFORT=medium
export AUTOSOLVER_DISABLE_RESPONSE_STORAGE=true
```

## 快速运行

```bash
autosolver-agent \
  --cases examples/demo_case.txt \
  --iterations 3 \
  --budget 60 \
  --per-case-timeout 5 \
  --search-per-case-timeout 2 \
  --strategy-workers 1 \
  --memory-dir runs/autosolver_memory \
  --artifact-dir runs/autosolver_artifacts \
  --summary-out runs/autosolver_summary.json \
  --out runs/generated_submit_solution.py
```

运行完成后会写出：

- `runs/generated_submit_solution.py`：最终 solver。
- `runs/generated_submit_solution.py.report.json`：完整运行报告。
- `runs/autosolver_summary.json`：可选摘要报告。
- `runs/autosolver_memory/long_term_memory.json`：长期实验记忆。
- `runs/autosolver_memory/short_term_last_run.json`：最近一次短期记忆。
- `runs/autosolver_artifacts/events.jsonl`：结构化运行事件日志。
- `runs/autosolver_artifacts/iteration_*/`：每轮候选代码、rationale、validation、score 和 impact。

## CLI 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--version` | - | 输出当前版本号并退出。 |
| `--cases` | 必填 | 一个或多个 case TSV 文件。 |
| `--out` | `generated_submit_solution.py` | 最终 solver 输出路径。 |
| `--budget` | `90.0` | 整体 Agent 运行预算，单位秒。 |
| `--per-case-timeout` | `10.0` | finalize 复评时的单 case 超时。 |
| `--search-per-case-timeout` | 同 `--per-case-timeout` | 迭代搜索阶段的单 case 超时。 |
| `--iterations` | `3` | LLM 改进迭代轮数。 |
| `--memory-dir` | `runs/autosolver_memory` | 短期/长期记忆目录。 |
| `--artifact-dir` | `runs/autosolver_artifacts` | 每轮 artifact 目录。 |
| `--llm-model` | 环境变量或 `gpt-4o-mini` | LLM 模型名。 |
| `--llm-base-url` | 环境变量 | OpenAI 兼容 API 地址。 |
| `--max-cases` | `3` | 本次最多加载的 case 数。 |
| `--finalize-top-k` | `3` | finalize 阶段复评排名前 K 的候选。 |
| `--max-repair-attempts` | `2` | schema 或验证失败后的最大修复次数。 |
| `--memory-top-k` | `5` | 规划时检索的相似历史实验数量。 |
| `--bandit-exploration` | `1.4` | UCB 探索系数。 |
| `--strategy-workers` | `5` | worker 数；`1` 表示单 workflow 不并行，`2+` 表示多进程并行并共享记忆。 |
| `--summary-out` | `None` | 可选 JSON 摘要输出路径。 |
| `--event-log` | `artifact-dir/events.jsonl` | 可选 JSONL 结构化事件日志路径。 |
| `--quiet` | `False` | 关闭运行日志。 |

## Python API

```python
from autosolver_agent import AutoSolverLangChainAgent

agent = AutoSolverLangChainAgent(
    case_paths=["examples/demo_case.txt"],
    output_path="runs/generated_submit_solution.py",
    budget_seconds=60,
    per_case_timeout=5,
    search_per_case_timeout=2,
    iterations=3,
    strategy_workers=1,
    event_log_path="runs/autosolver_artifacts/events.jsonl",
)

report = agent.run()
```

测试中也可以向 `AutoSolverLangChainAgent(llm=...)` 注入 fake LLM，用于离线验证工作流。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy autosolver_agent
.venv/bin/python -m coverage run -m unittest discover -s tests -v
.venv/bin/python -m coverage report
```

单元测试使用 fake LLM 响应，不需要网络访问。覆盖范围包括：

- case 解析、实例分类和评分。
- 静态验证、非法输出验证。
- 结构化生成 schema。
- 短期/长期记忆、相似实验检索和 UCB 推荐。
- Agent 端到端运行、schema 修复、验证修复和 CLI 参数解析。

## Docker

构建 v1.0.0 镜像：

```bash
docker build \
  --build-arg VERSION=1.0.0 \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -t autosolver-agent:1.0.0 \
  -t autosolver-agent:latest \
  .
```

验证镜像入口：

```bash
docker run --rm autosolver-agent:1.0.0 --version
docker run --rm autosolver-agent:1.0.0 --help
```

使用本地目录挂载 case 和产物：

```bash
docker run --rm \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -v "$PWD/examples:/data/examples:ro" \
  -v "$PWD/runs:/app/runs" \
  autosolver-agent:1.0.0 \
  --cases /data/examples/demo_case.txt \
  --iterations 3 \
  --budget 60 \
  --per-case-timeout 5 \
  --search-per-case-timeout 2 \
  --memory-dir /app/runs/autosolver_memory \
  --artifact-dir /app/runs/autosolver_artifacts \
  --summary-out /app/runs/autosolver_summary.json \
  --out /app/runs/generated_submit_solution.py
```

离线检查镜像内测试：

```bash
docker run --rm --entrypoint python autosolver-agent:1.0.0 -m unittest discover -s tests -v
```

Docker 镜像默认以非 root 用户 `autosolver` 运行，工作目录是 `/app`，入口命令是 `autosolver-agent`。

## v1.0.0 发布清单

发布前检查：

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m ruff check .
python -m mypy autosolver_agent
autosolver-agent --version
docker build --build-arg VERSION=1.0.0 --build-arg VCS_REF=$(git rev-parse --short HEAD) -t autosolver-agent:1.0.0 -t autosolver-agent:latest .
docker run --rm autosolver-agent:1.0.0 --version
```

建议发布标签：

```bash
git tag -a v1.0.0 -m "Release v1.0.0"
```

发布产物：

- Python 包版本：`autosolver-agent==1.0.0`
- Docker 镜像：`autosolver-agent:1.0.0`
- Docker latest 标签：`autosolver-agent:latest`
- 发布说明：[RELEASE.md](RELEASE.md)

## 当前边界

- 真实运行需要可用的 OpenAI 兼容 LLM 接口；未提供 API key 时不会自动退化成内置启发式求解器。
- 生成 solver 的运行环境被刻意限制，主要用于比赛/评测式纯函数求解，不适合依赖外部文件、网络或第三方库。
- 当前策略库是提示级知识库，实际最终算法由 LLM 生成，并通过验证、评分和记忆机制筛选。
- `runs/` 下的 artifact 和 memory 是实验状态，适合保留用于后续迭代，但不应当视作源码的一部分。
