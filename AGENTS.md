# Repository Guidelines

## Project Structure & Module Organization

`autosolver_agent/` contains the installable Python package. Key entry points are `cli.py` for the `autosolver-agent` command, `agent.py` for the top-level API, `workflow/` for LangGraph orchestration, `llm/` for structured generation, `tools/` for validation/scoring/classification, and `memory/` for experiment persistence. `solvers/` stores seed and reference solvers. `examples/` contains demo TSV cases and solver templates. `tests/test_modular_agent.py` holds the unit tests. Runtime artifacts and memory belong under `runs/`.

## Build, Test, and Development Commands

- `python -m pip install -e ".[dev]"`: install the package in editable mode with Ruff, Mypy, and coverage.
- `autosolver-agent --version`: verify the CLI entry point.
- `python -m unittest discover -s tests -v`: run the offline unit tests with fake LLM responses.
- `ruff check .` and `mypy autosolver_agent`: run the same lint and type checks as CI.
- `coverage run -m unittest discover -s tests -v && coverage report`: measure package coverage.
- `docker build --build-arg VERSION=1.0.0 --build-arg VCS_REF=$(git rev-parse --short HEAD) -t autosolver-agent:ci .`: build the CI smoke-test image.

## Coding Style & Naming Conventions

Target Python 3.10. Use 4-space indentation, type hints for public interfaces, snake_case for functions, variables, and modules, and PascalCase for classes. Ruff enforces `E`, `F`, and `I` rules with a 160-character line length; `examples/`, `solvers/`, `.venv/`, and `runs/` are excluded. Generated solver code must stay standard-library only and preserve `solve(input_text: str) -> list`.

## Testing Guidelines

Tests use `unittest`, not pytest. Add tests under `tests/` with `test_` methods or files so discovery finds them. Prefer fake LLM responses; tests must not require network access or real API keys. For validation, scoring, memory, runtime sandboxing, or CLI changes, include focused regression tests and run Ruff, Mypy, and unittest before a PR.

## Commit & Pull Request Guidelines

Git history uses short, direct summaries, sometimes with Chinese notes and version labels; no strict Conventional Commits format is enforced. Keep commits focused and mention affected behavior, for example `Improve memory merge locking`. PRs should include a clear description, test results, linked issues when applicable, and screenshots or logs only when they clarify CLI, artifact, or Docker behavior. Do not commit `runs/`, secrets, or editor settings.

## Security & Configuration Tips

Real LLM runs require `OPENAI_API_KEY`; optional settings include `OPENAI_BASE_URL`, `AUTOSOLVER_LLM_MODEL`, and `AUTOSOLVER_WIRE_API`. Never store credentials in the repository. Candidate solver validation intentionally rejects file IO, network IO, subprocesses, dynamic imports, `eval`, and `exec`; keep that sandbox boundary intact.
