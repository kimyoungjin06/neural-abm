#!/usr/bin/env python
"""Plot the Toy 1 parameter-path threshold phase diagram."""

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
            "toy1_param_independent_low_threshold_seed01_grouped_summary.csv"
        ),
        help="Grouped summary CSV from the parameter low-threshold sweep.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/toy1_parameter_independent_phase_diagram.png"),
        help="Output figure path.",
    )
    return parser.parse_args()


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot(index="alpha", columns="threshold", values=metric).sort_index(
        ascending=True
    )


def plot_heatmap(
    ax: plt.Axes,
    table: pd.DataFrame,
    title: str,
    colorbar_label: str,
    cmap: str,
) -> None:
    image = ax.imshow(table.values, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("threshold")
    ax.set_ylabel("alpha")
    ax.set_xticks(range(len(table.columns)), [f"{value:g}" for value in table.columns])
    ax.set_yticks(range(len(table.index)), [f"{value:g}" for value in table.index])
    for row_idx, alpha in enumerate(table.index):
        for col_idx, threshold in enumerate(table.columns):
            value = table.loc[alpha, threshold]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.3f}" if abs(value) < 10 else f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > table.values.mean() else "black",
            )
    plt.colorbar(image, ax=ax, shrink=0.82, label=colorbar_label)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    seed_counts = sorted(df["seeds"].unique()) if "seeds" in df.columns else []
    if seed_counts == [1]:
        seed_note = "single-seed pilot"
    elif len(seed_counts) == 1:
        seed_note = f"mean over {seed_counts[0]} seeds"
    else:
        seed_note = "grouped sweep means"

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    fig.suptitle(
        f"Toy 1: independent-init parameter averaging phase behavior ({seed_note})",
        fontsize=11,
    )

    plot_heatmap(
        axes[0],
        pivot_metric(df, "accuracy_mean"),
        "Task accuracy",
        "mean accuracy",
        "viridis",
    )
    plot_heatmap(
        axes[1],
        pivot_metric(df, "consensus_mean"),
        "Prediction consensus",
        "mean agreement",
        "Greens",
    )
    plot_heatmap(
        axes[2],
        pivot_metric(df, "fragmentation_mean"),
        "Peer fragmentation",
        "components",
        "magma_r",
    )

    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
