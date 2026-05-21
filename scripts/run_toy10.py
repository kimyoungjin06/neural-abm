#!/usr/bin/env python
"""Run Toy 10 from a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_abm.config import load_toy10_config
from neural_abm.toy_market import run_toy10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Toy 10: dynamic-network market/ecology ABM."
    )
    parser.add_argument("--config", required=True, type=Path, help="Toy 10 YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_toy10_config(args.config)
    result = run_toy10(config=config, config_path=args.config)
    print(f"run_dir={result.run_dir}")
    print(f"final_fragmentation_components={result.final_fragmentation_components}")
    for key, value in result.domain_metrics.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
