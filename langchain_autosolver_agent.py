"""Compatibility wrapper for the AutoSolver Agent CLI."""

from __future__ import annotations

from autosolver_agent.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
