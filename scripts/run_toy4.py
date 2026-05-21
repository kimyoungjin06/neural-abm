#!/usr/bin/env python
"""Run Toy 4: neural public goods and commons."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_abm.config import load_toy4_config
from neural_abm.toy_public_goods import run_toy4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a Toy 4 YAML config.",
    )
    parser.add_argument(
        "--neural-update-backend",
        choices=["config", "loop", "batched", "tensor_batched", "auto"],
        default="config",
        help="Override model.policy.neural_update_backend for neural_policy runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_toy4_config(args.config)
    backend = (
        None
        if args.neural_update_backend == "config"
        else args.neural_update_backend
    )
    result = run_toy4(
        config=config,
        config_path=args.config,
        neural_update_backend=backend,
    )
    print(f"run_dir={result.run_dir}")
    print(f"toy={result.toy}")
    print(f"final_action_rate={result.final_action_rate:.6f}")
    print(f"final_mean_payoff={result.final_mean_payoff:.6f}")
    print(f"final_fragmentation_components={result.final_fragmentation_components}")
    print(f"domain_metrics={result.domain_metrics}")


if __name__ == "__main__":
    main()
