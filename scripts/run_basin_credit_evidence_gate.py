#!/usr/bin/env python
"""Evaluate the Toy2/Toy4 basin-credit evidence success gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neural_abm.evidence_gate import run_evidence_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/evidence/toy24_basin_credit_quick.yaml"),
        help="Basin-credit evidence manifest with success_criteria.",
    )
    parser.add_argument(
        "--runs-path",
        type=Path,
        default=None,
        help="Optional run-level CSV. Defaults to <matrix-results-dir>/<label>_runs.csv.",
    )
    parser.add_argument(
        "--matrix-results-dir",
        type=Path,
        default=Path("experiments/results/nabm_effect_matrix"),
        help="Directory containing evidence-matrix run CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/evidence/results"),
        help="Directory for gate JSON and Markdown summaries.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional seed override matching the evaluated run-level CSV.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit non-zero when the evidence gate does not pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_evidence_gate(
        args.manifest,
        runs_path=args.runs_path,
        matrix_results_dir=args.matrix_results_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
    )
    print(f"Evidence gate status: {result.summary['status']}")
    print(f"Wrote JSON summary: {result.json_path}")
    print(f"Wrote Markdown summary: {result.markdown_path}")
    if args.require_pass and not result.summary["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
