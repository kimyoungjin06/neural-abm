#!/usr/bin/env python
"""Run and evaluate the Toy2/Toy4 basin-credit evidence workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neural_abm.diagnostics.evidence_profile import profile_evidence_artifacts
from neural_abm.evidence_gate import run_evidence_gate
from neural_abm.evidence_matrix import run_evidence_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/evidence/toy24_basin_credit_quick.yaml"),
        help="Basin-credit evidence manifest.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results/nabm_effect_matrix"),
        help="Directory for evidence-matrix CSV and Markdown outputs.",
    )
    parser.add_argument(
        "--gate-output-dir",
        type=Path,
        default=Path("experiments/evidence/results"),
        help="Directory for gate JSON and Markdown summaries.",
    )
    parser.add_argument(
        "--runs-path",
        type=Path,
        default=None,
        help="Optional run-level CSV for --skip-matrix mode.",
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
    parser.add_argument(
        "--skip-matrix",
        action="store_true",
        help="Skip simulation runs and evaluate an existing run-level CSV.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit non-zero when the evidence gate does not pass.",
    )
    parser.add_argument(
        "--profile-output-dir",
        type=Path,
        default=None,
        help=(
            "Optional evidence-profile output directory. Defaults to "
            "--results-dir."
        ),
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip the read-only evidence profile artifact generation step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_path = args.runs_path
    if not args.skip_matrix:
        matrix_result = run_evidence_matrix(
            args.manifest,
            results_dir=args.results_dir,
            seeds=args.seeds,
            epochs=args.epochs,
            config_dir=args.config_dir,
            runs_dir=args.runs_dir,
        )
        runs_path = matrix_result.runs_path
        print(f"Wrote run rows: {matrix_result.runs_path}")
        print(f"Wrote effect rows: {matrix_result.effects_path}")
        print(f"Wrote pairwise effect rows: {matrix_result.pairwise_effects_path}")
        print(f"Wrote effect report: {matrix_result.markdown_path}")

    gate_result = run_evidence_gate(
        args.manifest,
        runs_path=runs_path,
        matrix_results_dir=args.results_dir,
        output_dir=args.gate_output_dir,
        seeds=args.seeds,
    )
    print(f"Evidence gate status: {gate_result.summary['status']}")
    print(f"Wrote gate JSON summary: {gate_result.json_path}")
    print(f"Wrote gate Markdown summary: {gate_result.markdown_path}")
    if not args.skip_profile:
        profile_output = profile_evidence_artifacts(
            args.manifest,
            runs_path=runs_path,
            results_dir=args.results_dir,
            gate_output_dir=args.gate_output_dir,
            output_dir=args.profile_output_dir,
            seeds=args.seeds,
        )
        if profile_output.json_path is not None:
            print(f"Wrote evidence profile JSON: {profile_output.json_path}")
        if profile_output.markdown_path is not None:
            print(f"Wrote evidence profile Markdown: {profile_output.markdown_path}")
        if profile_output.cases_path is not None:
            print(f"Wrote evidence profile cases CSV: {profile_output.cases_path}")
    if args.require_pass and not gate_result.summary["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
