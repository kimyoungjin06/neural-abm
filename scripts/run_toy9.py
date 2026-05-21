#!/usr/bin/env python
"""Run Toy 9 from a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_abm.config import load_toy9_config
from neural_abm.toy_heterogeneous import run_toy9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Toy 9: heterogeneous-agent binary adoption ABM."
    )
    parser.add_argument("--config", required=True, type=Path, help="Toy 9 YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_toy9_config(args.config)
    result = run_toy9(config=config, config_path=args.config)
    print(f"run_dir={result.run_dir}")
    print(f"final_fragmentation_components={result.final_fragmentation_components}")
    for key, value in result.domain_metrics.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
