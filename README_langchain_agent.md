# Modular LangChain AutoSolver Agent

This repository now uses a modular Agent architecture for the delivery assignment AutoSolver.

## Architecture

- `autosolver_agent/skills`: strategy and solver skill libraries. These describe how a strategy works, which instance features it fits, and what implementation constraints the LLM-generated solver must follow.
- `autosolver_agent/tools`: Agent tools for instance classification, static/runtime validation, and scoring.
- `autosolver_agent/memory`: JSON-backed short-term and long-term repository memory.
- `autosolver_agent/llm`: full-solver LLM code generation. There is no heuristic or template fallback.
- `autosolver_agent/workflow`: LangGraph workflow orchestration. If `langgraph` is installed, the graph runner is used.
- `langchain_autosolver_agent.py`: the new CLI entrypoint.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set an OpenAI-compatible key before running:

```bash
export OPENAI_API_KEY=sk-...
export AUTOSOLVER_LLM_MODEL=gpt-4o-mini
```

Optional compatible endpoint:

```bash
export OPENAI_BASE_URL=https://api.example.com/v1
```

## Run

```bash
python3 langchain_autosolver_agent.py \
  --cases large_seed301.txt \
  --iterations 3 \
  --budget 90 \
  --per-case-timeout 10 \
  --search-per-case-timeout 3 \
  --memory-dir runs/autosolver_memory \
  --artifact-dir runs/autosolver_artifacts \
  --out generated_submit_solution.py
```

The Agent writes:

- final solver: `generated_submit_solution.py`
- report: `generated_submit_solution.py.report.json`
- long-term memory: `runs/autosolver_memory/long_term_memory.json`
- last short-term memory: `runs/autosolver_memory/short_term_last_run.json`
- per-iteration code, rationale, validation, score, and impact files under `runs/autosolver_artifacts/`

If no LLM key/client is configured, the CLI fails immediately. This is intentional.

## Workflow

1. Load case files and classify instance features such as task count, courier count, pair ratio, willingness, and capacity pressure.
2. Select strategy guidance from the strategy library and combine it with solver-skill constraints.
3. Read short-term and long-term memory plus previous disk results.
4. Ask the LLM to generate a complete standard-library `solve(input_text: str) -> list` solver and JSON rationale.
5. Run AST/compile validation, then smoke-run validation in a subprocess.
6. Score legal candidates with `rank=(failures, -covered, total_penalty, total_runtime)`.
7. Persist code, rationale, validation, score, impact analysis, and memory updates.
8. After the configured iteration count or budget exhaustion, recheck Top-K candidates with the final timeout and write the global best solver.

## Test

```bash
python -m unittest discover -s tests -v
```

The tests use a fake LLM so they do not require network access.
