#!/usr/bin/env python
"""Plot Toy 1 output-average alpha sweep results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "experiments/results/"
            "toy1_output_alpha_sweep_seeds01_05_grouped_summary.csv"
        ),
        help="Grouped summary CSV from run_toy1_sweep.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/toy1_output_alpha_accuracy_consensus.png"),
        help="Output figure path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input).sort_values("alpha")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharex=True)
    fig.suptitle("Toy 1: output social mixing alpha response", fontsize=11)

    axes[0].errorbar(
        df["alpha"],
        df["accuracy_mean"],
        yerr=df["accuracy_std"],
        marker="o",
        linewidth=1.8,
        capsize=3,
        color="#1f77b4",
    )
    axes[0].set_title("Task accuracy", fontsize=10)
    axes[0].set_xlabel("social influence alpha")
    axes[0].set_ylabel("mean global accuracy")
    axes[0].grid(True, alpha=0.25)

    axes[1].errorbar(
        df["alpha"],
        df["consensus_mean"],
        yerr=df["consensus_std"],
        marker="o",
        linewidth=1.8,
        capsize=3,
        color="#2ca02c",
    )
    axes[1].set_title("Prediction consensus", fontsize=10)
    axes[1].set_xlabel("social influence alpha")
    axes[1].set_ylabel("mean pairwise agreement")
    axes[1].grid(True, alpha=0.25)

    for ax in axes:
        ax.set_xticks(df["alpha"])

    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
