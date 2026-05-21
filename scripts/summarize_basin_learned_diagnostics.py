#!/usr/bin/env python
"""Summarize read-only learned basin diagnostics from evidence runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_abm.basin_learned_diagnostic_summary import (
    summarize_basin_learned_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=Path(
            "experiments/results/nabm_effect_matrix/"
            "toy24_basin_learned_diagnostic_quick_runs.csv"
        ),
        help="Evidence matrix run-level CSV containing run_dir values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/basin_critic"),
        help="Directory for learned diagnostic summary outputs.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional output label override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize_basin_learned_diagnostics(
        args.runs_csv,
        output_dir=args.output_dir,
        label=args.label,
    )
    print(f"Wrote run diagnostic summary: {result.run_summary_path}")
    print(f"Wrote grouped diagnostic summary: {result.group_summary_path}")
    print(f"Wrote diagnostic report: {result.markdown_path}")


if __name__ == "__main__":
    main()
