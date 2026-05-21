#!/usr/bin/env python
"""Run Toy 6: multi-action categorical spatial game."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_abm.config import load_toy6_config
from neural_abm.toy_categorical import run_toy6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a Toy 6 YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_toy6_config(args.config)
    result = run_toy6(config=config, config_path=args.config)
    print(f"run_dir={result.run_dir}")
    print(f"final_fragmentation_components={result.final_fragmentation_components}")
    for key in [
        "domain_final_mean_payoff",
        "domain_final_strategy_entropy",
        "domain_final_dominant_strategy",
        "domain_final_dominant_strategy_fraction",
    ]:
        print(f"{key}={result.domain_metrics[key]}")


if __name__ == "__main__":
    main()
