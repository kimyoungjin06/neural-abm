#!/usr/bin/env python
"""Diagnose whether revision pressure explains binary evidence-run bottlenecks."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_abm.revision_pressure_diagnostics import (
    write_revision_pressure_diagnostics,
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
            "toy24_basin_credit_objective_blend_quick_revision_pressure"
        ),
        help="Prefix for epoch CSV, run CSV, and Markdown diagnostic outputs.",
    )
    parser.add_argument(
        "--early-epochs",
        type=int,
        default=10,
        help="Pre-ceiling epoch window used for run-level summaries.",
    )
    parser.add_argument(
        "--pressure-threshold",
        type=float,
        default=0.05,
        help="Minimum proxy pressure treated as an active revision signal.",
    )
    parser.add_argument(
        "--policy-threshold",
        type=float,
        default=0.7,
        help="Mean post-social policy probability threshold for readiness.",
    )
    parser.add_argument(
        "--policy-ready-rate-threshold",
        type=float,
        default=0.5,
        help="Population share above 0.7 required for policy-readiness epoch.",
    )
    parser.add_argument(
        "--action-threshold",
        type=float,
        default=0.9,
        help="Action-rate threshold for aggregate action response.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_revision_pressure_diagnostics(
        runs_csv=args.runs_csv,
        output_prefix=args.output_prefix,
        early_epochs=args.early_epochs,
        pressure_threshold=args.pressure_threshold,
        policy_threshold=args.policy_threshold,
        policy_ready_rate_threshold=args.policy_ready_rate_threshold,
        action_threshold=args.action_threshold,
    )
    print(f"Wrote epoch diagnostics: {result.epoch_path}")
    print(f"Wrote run diagnostics: {result.run_path}")
    print(f"Wrote diagnostic report: {result.markdown_path}")


if __name__ == "__main__":
    main()
