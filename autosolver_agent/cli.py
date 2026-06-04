"""Command-line interface for AutoSolver Agent."""

from __future__ import annotations

import argparse
import json

from autosolver_agent import AutoSolverLangChainAgent, __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular LangChain AutoSolver Agent")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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
    parser.add_argument("--max-repair-attempts", type=int, default=2)
    parser.add_argument("--memory-top-k", type=int, default=5)
    parser.add_argument("--bandit-exploration", type=float, default=1.4)
    parser.add_argument("--summary-out", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--strict-cases", action="store_true", help="Fail on malformed case rows instead of reporting diagnostics.")
    parser.add_argument("--event-log", default=None, help="Optional JSONL event log path; defaults to artifact-dir/events.jsonl.")
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
        max_repair_attempts=args.max_repair_attempts,
        memory_top_k=args.memory_top_k,
        bandit_exploration=args.bandit_exploration,
        summary_output_path=args.summary_out,
        strict_cases=args.strict_cases,
        event_log_path=args.event_log,
    )
    report = agent.run()
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
