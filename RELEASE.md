# Release Notes

## v1.5.3

发布日期：2026-06-07

### 发布定位

`v1.5.3` 是发布整理版本。它清理本地构建和运行产物，对齐开发环境要求。

### 主要更新

- Python 包、CLI 和 Docker 默认版本更新为 `1.5.3`。
- 开发依赖加入 `build`，用于生成 wheel 和 sdist 发布包。
- `.gitignore` 补充 `dist/`、`build/`、`.mypy_cache/`、`.ruff_cache/` 和 `.pytest_cache/`。
- CI Docker smoke test 改用当前版本号。
- GHCR 发布工作流改为 `main` 分支发布 `main` / `latest` 镜像，并继续支持 `v*` 标签发布。
- 清理旧版本构建包、egg-info、缓存和运行产物，避免生成文件混入发布源码。

### 发布产物

- Python 包：`autosolver-agent==1.5.3`
- CLI：`autosolver-agent --version` 输出 `autosolver-agent 1.5.3`
- Docker 镜像建议标签：`autosolver-agent:1.5.3`

### 验证清单

```bash
.venv/bin/ruff check .
.venv/bin/mypy autosolver_agent
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m autosolver_agent.cli --version
.venv/bin/python -m build
```

## v1.5.2

发布日期：2026-06-07

### 发布定位

`v1.5.2` 是 OpenAI 兼容端点稳定性修复版本。它修复真实 LLM 长测中暴露的框架元数据安全校验过严、单次 LLM 请求无超时兜底，以及多 worker 在后续 LLM 失败时丢失已有候选结果的问题。

### 主要更新

- 新增 `AUTOSOLVER_LLM_TIMEOUT` / `OPENAI_TIMEOUT` / `OPENAI_REQUEST_TIMEOUT`，默认单次 LLM 请求超时为 `300` 秒。
- 对 LLM 返回的 framework、instance interpretation 和 framework update 元数据执行递归文本清洗。
- 保留存储层严格安全校验：手动构造或磁盘持久化的危险 framework payload 仍会被拒绝。
- 多 worker 模式下，worker 在已经产生有效评分后遇到后续 LLM/API 错误时，会记录 `worker_stop_reason` 并返回已有 report，主进程可继续使用现有候选做全局最终复核。
- README 同步更新 OpenAI 兼容环境变量和 v1.5.2 Docker 示例。

### 发布产物

- Python 包：`autosolver-agent==1.5.2`
- CLI：`autosolver-agent --version` 输出 `autosolver-agent 1.5.2`
- Docker 镜像建议标签：`autosolver-agent:1.5.2`

### 验证清单

```bash
.venv/bin/ruff check .
.venv/bin/mypy autosolver_agent
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m autosolver_agent.cli --version
```

## v1.5.0

发布日期：2026-06-07

### 发布定位

`v1.5.0` 是 AutoSolver Agent 的一次文档和发布元数据对齐版本。说明文档已按当前代码从头重写，并把项目描述从旧的 v1.0.0 入口文档更新为当前的 LLM 框架记忆、多策略并行、基线导入、候选修复和事件审计工作流。

### 主要更新

- 将主说明文档统一到 `README.md`，删除对旧 `README_langchain_agent.md` 的依赖。
- 发布版本更新为 `v1.5.0`，同步 Python 包元数据、CLI `--version` 和 Docker 默认构建版本。
- 文档覆盖当前运行链路：`classify -> generate -> validate_and_score -> finalize`。
- 补充 LLM 维护的 `FrameworkStore` 说明，包括 feature dimensions、strategies、skills、bootstrap 和反思更新。
- 补充 `MemoryStore` 说明，包括长期实验记忆、相似实验检索和 UCB bandit 推荐。
- 补充多 worker 运行方式：`strategy_workers == 1` 为单工作流，`2+` 为多进程独立 worker 并全局最终复评。
- 补充 baseline solver 导入方式和最终候选池行为。
- 补充候选验证、schema 修复、validation 修复、最终 top-k 复核和事件日志说明。
- 补充 solver 输入输出契约、penalty 公式、安全沙箱限制、产物目录和 Docker 运行示例。

### 发布产物

- Python 包：`autosolver-agent==1.5.0`
- CLI：`autosolver-agent --version` 输出 `autosolver-agent 1.5.0`
- Docker 镜像建议标签：`autosolver-agent:1.5.0`

### 验证清单

```bash
python -m pip install -e ".[dev]"
ruff check .
mypy autosolver_agent
python -m unittest discover -s tests -v
autosolver-agent --version
docker build --build-arg VERSION=1.5.0 -t autosolver-agent:1.5.0 .
docker run --rm autosolver-agent:1.5.0 --version
docker run --rm autosolver-agent:1.5.0 --help
```

### 运行约束

- 真实 LLM 运行需要 `OPENAI_API_KEY` 或 `OPENAI_KEY`。
- 候选 solver 必须定义 `solve(input_text: str) -> list`。
- 候选 solver 只能使用受允许的 import 和内置函数。
- `runs/` 是运行态产物目录，不属于发布源码。
- 已有 memory 目录必须满足当前 schema：`MemoryStore` schema 为 `2`，`FrameworkStore` schema 为 `1`。

## v1.0.0

发布日期：2026-06-04

### 发布范围

- 梳理并固化模块化 AutoSolver Agent 架构。
- 将 Python 包版本提升到 `1.0.0`。
- 新增可安装命令 `autosolver-agent`，作为 CLI 入口。
- 补充 Docker 镜像元数据，默认入口切换为 `autosolver-agent`。
- 更新说明文档，覆盖架构、输入输出契约、运行方式、Docker 打包和发布检查项。

### 运行约束

- 真实 LLM 运行需要 `OPENAI_API_KEY`。
- 候选 solver 被限制为 Python 标准库纯函数实现。
- `runs/` 是运行态产物目录，不属于发布源码。
