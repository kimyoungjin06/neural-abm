#!/usr/bin/env python
"""Run the small Toy1-5 NABM effect evidence matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_abm.evidence_matrix import run_evidence_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/evidence/nabm_effect_matrix_quick.yaml"),
        help="Evidence matrix YAML manifest.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results/nabm_effect_matrix"),
        help="Directory for run-level CSV, effect CSV, and Markdown outputs.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional output label override.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional seed override for every case without case-level seeds.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for every case without case-level epochs.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Optional generated-config directory override.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Optional run artifact directory override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_evidence_matrix(
        args.manifest,
        results_dir=args.results_dir,
        label=args.label,
        seeds=args.seeds,
        epochs=args.epochs,
        config_dir=args.config_dir,
        runs_dir=args.runs_dir,
    )
    print(f"Wrote run rows: {result.runs_path}")
    print(f"Wrote effect rows: {result.effects_path}")
    print(f"Wrote pairwise effect rows: {result.pairwise_effects_path}")
    print(f"Wrote report: {result.markdown_path}")


if __name__ == "__main__":
    main()
