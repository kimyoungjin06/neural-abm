#!/usr/bin/env python
"""Run Toy 7: continuous extraction-intensity resource ABM."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_abm.config import load_toy7_config
from neural_abm.toy_resource import run_toy7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a Toy 7 YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_toy7_config(args.config)
    result = run_toy7(config=config, config_path=args.config)
    print(f"run_dir={result.run_dir}")
    print(f"final_fragmentation_components={result.final_fragmentation_components}")
    for key in [
        "domain_final_resource_level",
        "domain_final_resource_fraction",
        "domain_final_mean_intensity",
        "domain_final_intensity_variance",
        "domain_final_mean_payoff",
    ]:
        print(f"{key}={result.domain_metrics[key]}")


if __name__ == "__main__":
    main()
