#!/usr/bin/env python
"""Train and evaluate offline learned basin phase critics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neural_abm.basin_phase_critic import run_basin_phase_critic_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/evidence/toy24_basin_phase_critic_quality_quick.yaml"),
        help="Basin phase critic training manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit non-zero unless every critic case passes the quality gate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_basin_phase_critic_workflow(
        args.manifest,
        output_dir=args.output_dir,
    )
    print(f"Wrote critic summary CSV: {result.summary_csv_path}")
    print(f"Wrote critic summary JSON: {result.summary_json_path}")
    print(f"Wrote critic report: {result.markdown_path}")
    for row in result.rows:
        print(
            "{case}: status={status}, auc={auc}, pairwise_rank={rank}".format(
                case=row["case"],
                status=row["status"],
                auc=row["eval_auc"],
                rank=row["eval_pairwise_rank_accuracy"],
            )
        )
    if args.require_pass and any(row["status"] != "pass" for row in result.rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
