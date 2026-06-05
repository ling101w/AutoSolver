# Release Notes

## v1.0.0

发布日期：2026-06-04

### 发布范围

- 梳理并固化当前模块化 AutoSolver Agent 架构。
- 将 Python 包版本提升到 `1.0.0`。
- 新增可安装命令 `autosolver-agent`，作为唯一 CLI 入口。
- 补充 Docker 镜像元数据，默认入口切换为 `autosolver-agent`。
- 更新说明文档，覆盖架构、输入输出契约、运行方式、Docker 打包和发布检查项。

### 发布产物

- Python 包：`autosolver-agent==1.0.0`
- Docker 镜像标签：`autosolver-agent:1.0.0`
- Docker latest 标签：`autosolver-agent:latest`

### 验证清单

```bash
python -m unittest discover -s tests -v
python -m ruff check .
python -m mypy autosolver_agent
python -m pip install -e .
autosolver-agent --version
docker build --build-arg VERSION=1.0.0 -t autosolver-agent:1.0.0 -t autosolver-agent:latest .
docker run --rm autosolver-agent:1.0.0 --version
docker run --rm autosolver-agent:1.0.0 --help
```

### 运行约束

- 真实 LLM 运行需要 `OPENAI_API_KEY`。
- 候选 solver 被限制为 Python 标准库纯函数实现。
- `runs/` 是运行态产物目录，不属于发布源码。
