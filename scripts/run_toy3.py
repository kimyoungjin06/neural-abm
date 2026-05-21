#!/usr/bin/env python
"""Run Toy 3: opinion dynamics with endogenous rewiring."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_abm.config import load_toy3_config
from neural_abm.toy_opinion import run_toy3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a Toy 3 YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_toy3_config(args.config)
    result = run_toy3(config=config, config_path=args.config)
    print(f"run_dir={result.run_dir}")
    print(f"final_fragmentation_components={result.final_fragmentation_components}")
    for key in [
        "domain_final_opinion_mean",
        "domain_final_polarization_index",
        "domain_final_opinion_cluster_count",
        "domain_final_mean_edge_disagreement",
        "domain_cumulative_rewired_edge_count",
    ]:
        print(f"{key}={result.domain_metrics[key]}")


if __name__ == "__main__":
    main()
