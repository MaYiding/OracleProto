"""CLI for offline fixed-points scoring; no provider credentials are required."""
from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from ..errors import AnalysisContractError
from . import run_analysis
from .scoring import CONTRACT


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute fixed Score and forecasting diagnostics.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--profiles", nargs="+", help="Explicit manifest model IDs or catalog profile IDs to compare.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Write coverage diagnostics; unavailable scores stay empty.")
    parser.add_argument("--bootstrap-iterations", type=int, default=CONTRACT["bootstrap_iterations"])
    parser.add_argument("--bootstrap-seed", type=int, default=CONTRACT["bootstrap_seed"])
    args = parser.parse_args(argv)
    try:
        paths = run_analysis(args.run_dir, allow_incomplete=args.allow_incomplete,
                             bootstrap_iterations=args.bootstrap_iterations, bootstrap_seed=args.bootstrap_seed,
                             profiles=args.profiles)
    except (AnalysisContractError, FileNotFoundError) as exc:
        logger.error("Scoring stopped: {}", exc)
        return 2
    logger.info("Scoring report: {}", paths[-1].parent / "score_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
