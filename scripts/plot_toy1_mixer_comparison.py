#!/usr/bin/env python
"""Plot Toy 1 mixer-path comparison from the first ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CASE_LABELS = {
    "none_none_same_init": "No social\nsame",
    "output_average_output_similarity_same_init": "Output\nsame",
    "latent_average_state_similarity_same_init": "Latent\nsame",
    "parameter_average_state_similarity_same_init": "Param\nsame",
    "parameter_average_state_similarity_independent_init": "Param\nindep.",
}

CASE_ORDER = list(CASE_LABELS)

CASE_COLORS = {
    "none_none_same_init": "#6c757d",
    "output_average_output_similarity_same_init": "#1f77b4",
    "latent_average_state_similarity_same_init": "#9467bd",
    "parameter_average_state_similarity_same_init": "#2ca02c",
    "parameter_average_state_similarity_independent_init": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "experiments/results/"
            "toy1_first_ablation_seeds01_05_grouped_summary.csv"
        ),
        help="Grouped summary CSV from scripts/run_toy1_ablation.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/toy1_mixer_comparison.png"),
        help="Output figure path.",
    )
    return parser.parse_args()


def load_ordered_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_cases = sorted(set(CASE_ORDER) - set(df["case"]))
    if missing_cases:
        raise ValueError(f"Missing expected cases: {', '.join(missing_cases)}")
    return df.set_index("case").loc[CASE_ORDER].reset_index()


def plot_metric_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    std_metric: str | None,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None,
) -> None:
    x_values = range(len(df))
    colors = [CASE_COLORS[case] for case in df["case"]]
    yerr = df[std_metric] if std_metric else None
    ax.bar(
        x_values,
        df[metric],
        yerr=yerr,
        color=colors,
        capsize=3 if std_metric else 0,
        edgecolor="#202020",
        linewidth=0.5,
    )
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticks(x_values, [CASE_LABELS[case] for case in df["case"]])
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.25)


def annotate_fragmentation(ax: plt.Axes, df: pd.DataFrame) -> None:
    for index, value in enumerate(df["fragmentation_mean"]):
        ax.text(
            index,
            value + 1.2,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    args = parse_args()
    df = load_ordered_results(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    seed_counts = sorted(df["seeds"].unique())
    seed_note = (
        f"mean over {seed_counts[0]} seeds"
        if len(seed_counts) == 1
        else "grouped means"
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    fig.suptitle(
        f"Toy 1: mixer path comparison ({seed_note}, error bars = 1 SD)",
        fontsize=11,
    )

    plot_metric_bars(
        axes[0],
        df,
        "accuracy_mean",
        "accuracy_std",
        "Task accuracy",
        "mean global accuracy",
        (0.878, 0.902),
    )
    plot_metric_bars(
        axes[1],
        df,
        "consensus_mean",
        "consensus_std",
        "Prediction consensus",
        "mean pairwise agreement",
        (0.93, 0.99),
    )
    plot_metric_bars(
        axes[2],
        df,
        "fragmentation_mean",
        None,
        "Peer fragmentation",
        "final components",
        (0, 55),
    )
    annotate_fragmentation(axes[2], df)

    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
