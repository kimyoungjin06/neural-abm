#!/usr/bin/env python
"""Plot initial Toy 2 no-social versus policy-output mixing runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_run_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Run arguments must use label=run_dir")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("Run label cannot be empty")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_arg,
        required=True,
        help="Run to plot as label=path/to/run_dir. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/toy2_initial_policy_mixing.png"),
        help="Output figure path.",
    )
    return parser.parse_args()


def load_aggregate(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "aggregate_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    fig.suptitle("Toy 2: initial spatial PD policy-mixing comparison", fontsize=11)

    for label, run_dir in args.run:
        df = load_aggregate(run_dir)
        axes[0].plot(
            df["epoch"],
            df["action_rate"],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=label,
        )
        axes[1].plot(
            df["epoch"],
            df["mean_payoff"],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=label,
        )
        axes[2].plot(
            df["epoch"],
            df["mean_policy_action_probability"],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=label,
        )

    axes[0].set_title("Realized action", fontsize=10)
    axes[0].set_ylabel("action rate")
    axes[0].set_ylim(0, 1)
    axes[1].set_title("Mean payoff", fontsize=10)
    axes[1].set_ylabel("mean payoff")
    axes[2].set_title("Policy action probability", fontsize=10)
    axes[2].set_ylabel("mean p(action)")
    axes[2].set_ylim(0, 1)

    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
    axes[2].legend(loc="best", fontsize=8, frameon=False)

    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
