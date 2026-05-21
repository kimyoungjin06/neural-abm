#!/usr/bin/env python
"""Diagnose early time-to-ceiling bottlenecks in evidence-matrix runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_abm.time_to_ceiling_diagnostics import (
    write_time_to_ceiling_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=Path(
            "experiments/results/nabm_effect_matrix/"
            "toy24_basin_credit_objective_blend_quick_runs.csv"
        ),
        help="Evidence matrix run-level CSV containing run_dir values.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "experiments/results/nabm_effect_matrix/"
            "toy24_basin_credit_objective_blend_quick_ttc_bottlenecks"
        ),
        help="Prefix for epoch CSV, run CSV, and Markdown diagnostic outputs.",
    )
    parser.add_argument(
        "--early-epochs",
        type=int,
        default=10,
        help="Pre-ceiling epoch window used for run-level bottleneck summaries.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Small positive threshold for policy/action/credit movement.",
    )
    parser.add_argument(
        "--decision-gap",
        type=float,
        default=0.05,
        help="Flag when post-social policy probability exceeds action rate by this.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_time_to_ceiling_diagnostics(
        runs_csv=args.runs_csv,
        output_prefix=args.output_prefix,
        early_epochs=args.early_epochs,
        min_delta=args.min_delta,
        decision_gap=args.decision_gap,
    )
    print(f"Wrote epoch diagnostics: {result.epoch_path}")
    print(f"Wrote run diagnostics: {result.run_path}")
    print(f"Wrote diagnostic report: {result.markdown_path}")


if __name__ == "__main__":
    main()
