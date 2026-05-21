#!/usr/bin/env python
"""Run Toy 1: Neural HK Classification."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_abm.config import load_toy1_config
from neural_abm.toy_classification import run_toy1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a Toy 1 YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_toy1_config(args.config)
    result = run_toy1(config=config, config_path=args.config)
    print(f"run_dir={result.run_dir}")
    print(
        "domain_final_mean_global_accuracy="
        f"{result.domain_metrics['domain_final_mean_global_accuracy']:.6f}"
    )
    print(
        "domain_final_mean_consensus="
        f"{result.domain_metrics['domain_final_mean_consensus']:.6f}"
    )
    print(f"final_fragmentation_components={result.final_fragmentation_components}")


if __name__ == "__main__":
    main()
