#!/usr/bin/env python
"""Run Toy 8: asynchronous event-driven adoption/failure ABM."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_abm.config import load_toy8_config
from neural_abm.toy_async import run_toy8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a Toy 8 YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_toy8_config(args.config)
    result = run_toy8(config=config, config_path=args.config)
    print(f"run_dir={result.run_dir}")
    print(f"final_fragmentation_components={result.final_fragmentation_components}")
    for key in [
        "domain_final_time",
        "domain_final_active_fraction",
        "domain_final_failed_fraction",
        "domain_total_events",
        "domain_activation_events",
        "domain_failure_events",
        "domain_recovery_events",
    ]:
        print(f"{key}={result.domain_metrics[key]}")


if __name__ == "__main__":
    main()
