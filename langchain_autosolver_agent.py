"""CLI entrypoint for the modular LangChain AutoSolver Agent."""

from __future__ import annotations

import argparse
import json

from autosolver_agent import AutoSolverLangChainAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular LangChain AutoSolver Agent")
    parser.add_argument("--cases", nargs="+", required=True, help="Case TSV files to optimize against.")
    parser.add_argument("--out", default="generated_submit_solution.py", help="Final solver output path.")
    parser.add_argument("--budget", type=float, default=90.0, help="Total agent budget in seconds.")
    parser.add_argument("--per-case-timeout", type=float, default=10.0, help="Final judge-equivalent timeout.")
    parser.add_argument(
        "--search-per-case-timeout",
        type=float,
        default=None,
        help="Per-case timeout during candidate search; defaults to --per-case-timeout.",
    )
    parser.add_argument("--iterations", type=int, default=3, help="Number of LLM improvement iterations.")
    parser.add_argument("--memory-dir", default="runs/autosolver_memory", help="JSON long/short-term memory dir.")
    parser.add_argument("--artifact-dir", default="runs/autosolver_artifacts", help="Per-iteration artifact dir.")
    parser.add_argument("--llm-model", default=None, help="LLM model; defaults to AUTOSOLVER_LLM_MODEL or gpt-4o-mini.")
    parser.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--finalize-top-k", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    agent = AutoSolverLangChainAgent(
        case_paths=args.cases,
        output_path=args.out,
        budget_seconds=args.budget,
        per_case_timeout=args.per_case_timeout,
        search_per_case_timeout=args.search_per_case_timeout,
        iterations=args.iterations,
        memory_dir=args.memory_dir,
        artifact_dir=args.artifact_dir,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        max_cases=args.max_cases,
        verbose=not args.quiet,
        finalize_top_k=args.finalize_top_k,
    )
    report = agent.run()
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
