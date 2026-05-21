#!/usr/bin/env python
"""Plot Toy 1 parameter alignment diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RAW_CASE = "parameter_average_state_similarity_independent_init"
ALIGNED_AVG_RAW_PEERS_CASE = (
    "parameter_aligned_average_state_similarity_independent_init"
)
ALIGNED_AVG_ALIGNED_PEERS_CASE = (
    "parameter_aligned_average_aligned_state_similarity_independent_init"
)

METHOD_LABELS = {
    RAW_CASE: "Raw avg\nraw peers",
    ALIGNED_AVG_RAW_PEERS_CASE: "Aligned avg\nraw peers",
    ALIGNED_AVG_ALIGNED_PEERS_CASE: "Aligned avg\naligned peers",
}

METHOD_COLORS = {
    RAW_CASE: "#6c757d",
    ALIGNED_AVG_RAW_PEERS_CASE: "#1f77b4",
    ALIGNED_AVG_ALIGNED_PEERS_CASE: "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-input",
        type=Path,
        default=Path(
            "experiments/results/"
            "toy1_param_independent_low_threshold_seeds01_05_grouped_summary.csv"
        ),
        help="Grouped summary CSV for raw independent-init parameter averaging.",
    )
    parser.add_argument(
        "--first-ablation-input",
        type=Path,
        default=Path(
            "experiments/results/"
            "toy1_first_ablation_seeds01_05_grouped_summary.csv"
        ),
        help="Grouped summary CSV containing the original threshold 0.8 raw case.",
    )
    parser.add_argument(
        "--aligned-input",
        type=Path,
        default=Path(
            "experiments/results/"
            "toy1_param_alignment_diagnostic_seeds01_05_grouped_summary.csv"
        ),
        help="Grouped summary CSV for aligned parameter diagnostics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/toy1_parameter_alignment_diagnostic.png"),
        help="Output figure path.",
    )
    return parser.parse_args()


def load_raw_reference(raw_path: Path, first_ablation_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    raw = raw[(raw["case"] == RAW_CASE) & (raw["alpha"] == 0.25)].copy()

    first_ablation = pd.read_csv(first_ablation_path)
    original = first_ablation[first_ablation["case"] == RAW_CASE].copy()
    if not original.empty:
        original["alpha"] = 0.25
        original["threshold"] = 0.8
        raw = pd.concat([raw, original], ignore_index=True)

    return raw.drop_duplicates(subset=["case", "alpha", "threshold"])


def load_plot_frame(
    raw_path: Path,
    first_ablation_path: Path,
    aligned_path: Path,
) -> pd.DataFrame:
    raw = load_raw_reference(raw_path, first_ablation_path)
    aligned = pd.read_csv(aligned_path)
    aligned = aligned[aligned["alpha"] == 0.25].copy()

    df = pd.concat([raw, aligned], ignore_index=True)
    df = df[df["case"].isin(METHOD_LABELS)].copy()
    df["method"] = df["case"].map(METHOD_LABELS)
    return df.sort_values(["case", "threshold"])


def plot_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    std_metric: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    for case, group in df.groupby("case", sort=False):
        group = group.sort_values("threshold")
        yerr = group[std_metric] if std_metric in group else None
        ax.errorbar(
            group["threshold"],
            group[metric],
            yerr=yerr,
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=METHOD_LABELS[case],
            color=METHOD_COLORS[case],
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("similarity threshold")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)


def main() -> None:
    args = parse_args()
    df = load_plot_frame(
        raw_path=args.raw_input,
        first_ablation_path=args.first_ablation_input,
        aligned_path=args.aligned_input,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    fig.suptitle(
        "Toy 1: parameter alignment diagnostic "
        "(independent init, alpha=0.25, mean over 5 seeds)",
        fontsize=11,
    )

    plot_metric(
        axes[0],
        df,
        "accuracy_mean",
        "accuracy_std",
        "Task accuracy",
        "mean global accuracy",
        (0.884, 0.898),
    )
    plot_metric(
        axes[1],
        df,
        "consensus_mean",
        "consensus_std",
        "Prediction consensus",
        "mean pairwise agreement",
        (0.94, 0.99),
    )
    plot_metric(
        axes[2],
        df,
        "fragmentation_mean",
        "fragmentation_std",
        "Peer fragmentation",
        "final components",
        (0, 55),
    )
    axes[2].legend(loc="upper left", fontsize=8, frameon=False)

    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
