#!/usr/bin/env python
"""Plot Toy 2 regime-sweep validation figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REGIME_ORDER = ["harsh_pd", "mild_pd", "soft_pd", "snowdrift", "stag_hunt"]
CONDITION_ORDER = [
    ("neural_policy", "none"),
    ("neural_policy", "output_average"),
    ("fermi_imitation", "none"),
    ("fermi_imitation", "output_average"),
    ("reputation_imitation", "none"),
    ("reputation_imitation", "output_average"),
    ("rd_well_mixed", "none"),
]
CONDITION_LABELS = {
    ("neural_policy", "none"): "Neural / none",
    ("neural_policy", "output_average"): "Neural / output",
    ("fermi_imitation", "none"): "Fermi / none",
    ("fermi_imitation", "output_average"): "Fermi / output",
    ("reputation_imitation", "none"): "Reputation / none",
    ("reputation_imitation", "output_average"): "Reputation / output",
    ("rd_well_mixed", "none"): "RD well-mixed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("experiments/results/toy2_regime_sweep_seeds01_05_summary.csv"),
        help="Toy 2 sweep summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Figure output directory.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=["all", "regime", "dynamics", "alpha", "basin", "reputation"],
        default=["all"],
        help="Subset of figures to generate.",
    )
    return parser.parse_args()


def condition_label(update_rule: str, mixer: str) -> str:
    return CONDITION_LABELS.get((update_rule, mixer), f"{update_rule} / {mixer}")


def ordered_regimes(df: pd.DataFrame) -> list[str]:
    present = set(df["regime"])
    return [regime for regime in REGIME_ORDER if regime in present]


def load_trajectories(summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in summary.itertuples(index=False):
        aggregate_path = Path(row.run_dir) / "aggregate_metrics.csv"
        if not aggregate_path.exists():
            continue
        frame = pd.read_csv(aggregate_path)
        frame["regime"] = row.regime
        frame["policy_rule"] = row.policy_rule
        frame["coordination_mixer"] = row.coordination_mixer
        frame["seed"] = row.seed
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No aggregate_metrics.csv files found")
    return pd.concat(frames, ignore_index=True)


def plot_regime_action_payoff(summary: pd.DataFrame, output_dir: Path) -> Path:
    df = summary[summary["policy_rule"] != "rd_well_mixed"].copy()
    grouped = (
        df.groupby(["regime", "policy_rule", "coordination_mixer"])
        .agg(
            action=("final_action_rate", "mean"),
            action_std=("final_action_rate", "std"),
            payoff=("final_mean_payoff", "mean"),
            payoff_std=("final_mean_payoff", "std"),
        )
        .reset_index()
    )
    regimes = ordered_regimes(grouped)
    x = np.arange(len(regimes))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    fig.suptitle("Toy 2: payoff-regime outcomes", fontsize=11)

    for update_rule, mixer in CONDITION_ORDER:
        if update_rule == "rd_well_mixed":
            continue
        condition = grouped[
            (grouped["policy_rule"] == update_rule) & (grouped["coordination_mixer"] == mixer)
        ].set_index("regime")
        if condition.empty:
            continue
        y_action = [condition.loc[regime, "action"] for regime in regimes]
        y_payoff = [condition.loc[regime, "payoff"] for regime in regimes]
        axes[0].plot(
            x,
            y_action,
            marker="o",
            linewidth=1.5,
            label=condition_label(update_rule, mixer),
        )
        axes[1].plot(
            x,
            y_payoff,
            marker="o",
            linewidth=1.5,
            label=condition_label(update_rule, mixer),
        )

    axes[0].set_title("Final action", fontsize=10)
    axes[0].set_ylabel("action rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_title("Final payoff", fontsize=10)
    axes[1].set_ylabel("mean payoff")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(regimes, rotation=25, ha="right")
        ax.grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8, frameon=False)

    path = output_dir / "toy2_regime_action_payoff.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_neural_vs_fermi_vs_rd(summary: pd.DataFrame, output_dir: Path) -> Path:
    trajectories = load_trajectories(summary)
    grouped = (
        trajectories.groupby(["regime", "policy_rule", "coordination_mixer", "epoch"])
        .agg(action=("action_rate", "mean"))
        .reset_index()
    )
    regimes = ordered_regimes(grouped)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), constrained_layout=True)
    fig.suptitle("Toy 2: neural policy vs Fermi imitation vs RD", fontsize=11)
    flat_axes = axes.ravel()

    for index, regime in enumerate(regimes):
        ax = flat_axes[index]
        ax.set_title(regime, fontsize=10)
        for update_rule, mixer in CONDITION_ORDER:
            condition = grouped[
                (grouped["regime"] == regime)
                & (grouped["policy_rule"] == update_rule)
                & (grouped["coordination_mixer"] == mixer)
            ]
            if condition.empty:
                continue
            linestyle = "--" if update_rule == "rd_well_mixed" else "-"
            ax.plot(
                condition["epoch"],
                condition["action"],
                linewidth=1.4,
                linestyle=linestyle,
                label=condition_label(update_rule, mixer),
            )
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("epoch")
        ax.set_ylabel("action")
        ax.grid(True, alpha=0.25)

    for ax in flat_axes[len(regimes) :]:
        ax.axis("off")
    flat_axes[0].legend(loc="best", fontsize=7, frameon=False)

    path = output_dir / "toy2_neural_vs_fermi_vs_rd.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_alpha_sensitivity(summary: pd.DataFrame, output_dir: Path) -> Path:
    df = summary[summary["policy_rule"] == "neural_policy"].copy()
    df["effective_alpha"] = np.where(df["coordination_mixer"] == "none", 0.0, df["alpha"])
    grouped = (
        df.groupby(["regime", "effective_alpha"])
        .agg(
            action=("final_action_rate", "mean"),
            cluster_fraction=(
                "domain_largest_action_cluster_fraction",
                "mean",
            ),
        )
        .reset_index()
    )
    regimes = ordered_regimes(grouped)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    fig.suptitle("Toy 2: output-mixing alpha response", fontsize=11)

    for regime in regimes:
        condition = grouped[grouped["regime"] == regime].sort_values(
            "effective_alpha"
        )
        axes[0].plot(
            condition["effective_alpha"],
            condition["action"],
            marker="o",
            linewidth=1.5,
            label=regime,
        )
        axes[1].plot(
            condition["effective_alpha"],
            condition["cluster_fraction"],
            marker="o",
            linewidth=1.5,
            label=regime,
        )
    axes[0].set_title("Final action", fontsize=10)
    axes[0].set_ylabel("action rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_title("Largest action cluster", fontsize=10)
    axes[1].set_ylabel("fraction of agents")
    axes[1].set_ylim(-0.02, 1.02)
    for ax in axes:
        ax.set_xlabel("effective output alpha")
        ax.grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8, frameon=False)

    path = output_dir / "toy2_alpha_sensitivity.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_basin_sensitivity(summary: pd.DataFrame, output_dir: Path) -> Path:
    if "initial_action_probability" not in summary.columns:
        raise ValueError("Basin figure requires initial_action_probability")
    grouped = (
        summary.groupby(
            [
                "regime",
                "initial_action_probability",
                "policy_rule",
                "coordination_mixer",
            ]
        )
        .agg(
            action=("final_action_rate", "mean"),
            cluster_fraction=(
                "domain_largest_action_cluster_fraction",
                "mean",
            ),
        )
        .reset_index()
    )
    regimes = ordered_regimes(grouped)
    fig, axes = plt.subplots(
        2,
        len(regimes),
        figsize=(4.2 * len(regimes), 6.2),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle("Toy 2: initial-condition basin sensitivity", fontsize=11)

    for col, regime in enumerate(regimes):
        regime_frame = grouped[grouped["regime"] == regime]
        axes[0, col].set_title(regime, fontsize=10)
        for update_rule, mixer in CONDITION_ORDER:
            condition = regime_frame[
                (regime_frame["policy_rule"] == update_rule)
                & (regime_frame["coordination_mixer"] == mixer)
            ].sort_values("initial_action_probability")
            if condition.empty:
                continue
            linestyle = "--" if update_rule == "rd_well_mixed" else "-"
            label = condition_label(update_rule, mixer)
            axes[0, col].plot(
                condition["initial_action_probability"],
                condition["action"],
                marker="o",
                linewidth=1.5,
                linestyle=linestyle,
                label=label,
            )
            axes[1, col].plot(
                condition["initial_action_probability"],
                condition["cluster_fraction"],
                marker="o",
                linewidth=1.5,
                linestyle=linestyle,
                label=label,
            )
        axes[0, col].set_ylim(-0.02, 1.02)
        axes[1, col].set_ylim(-0.02, 1.02)
        axes[0, col].set_ylabel("final action")
        axes[1, col].set_ylabel("largest action cluster")
        axes[1, col].set_xlabel("initial action")
        for row in range(2):
            axes[row, col].grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=7, frameon=False)

    path = output_dir / "toy2_basin_sensitivity.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_reputation_mobility(summary: pd.DataFrame, output_dir: Path) -> Path:
    required = {"final_mean_reputation", "final_mobility_rate"}
    if not required <= set(summary.columns):
        raise ValueError("Reputation figure requires final reputation/mobility fields")
    df = summary[summary["policy_rule"] != "rd_well_mixed"].copy()
    grouped = (
        df.groupby(["regime", "policy_rule", "coordination_mixer"])
        .agg(
            reputation=("final_mean_reputation", "mean"),
            mobility_rate=("final_mobility_rate", "mean"),
        )
        .reset_index()
    )
    regimes = ordered_regimes(grouped)
    x = np.arange(len(regimes))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    fig.suptitle("Toy 2: reputation and mobility diagnostics", fontsize=11)

    for update_rule, mixer in CONDITION_ORDER:
        if update_rule == "rd_well_mixed":
            continue
        condition = grouped[
            (grouped["policy_rule"] == update_rule) & (grouped["coordination_mixer"] == mixer)
        ].set_index("regime")
        if condition.empty:
            continue
        axes[0].plot(
            x,
            [condition.loc[regime, "reputation"] for regime in regimes],
            marker="o",
            linewidth=1.5,
            label=condition_label(update_rule, mixer),
        )
        axes[1].plot(
            x,
            [condition.loc[regime, "mobility_rate"] for regime in regimes],
            marker="o",
            linewidth=1.5,
            label=condition_label(update_rule, mixer),
        )

    axes[0].set_title("Final mean reputation", fontsize=10)
    axes[0].set_ylabel("mean reputation")
    axes[1].set_title("Final mobility rate", fontsize=10)
    axes[1].set_ylabel("mobility rate")
    for ax in axes:
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes, rotation=25, ha="right")
        ax.grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8, frameon=False)

    path = output_dir / "toy2_reputation_mobility_diagnostics.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary)
    if "all" in args.figures:
        selected = {"regime", "dynamics", "alpha"}
        if "initial_action_probability" in summary.columns:
            selected.add("basin")
        if {"final_mean_reputation", "final_mobility_rate"} <= set(summary.columns):
            selected.add("reputation")
    else:
        selected = set(args.figures)
    paths = []
    if "regime" in selected:
        paths.append(plot_regime_action_payoff(summary, args.output_dir))
    if "dynamics" in selected:
        paths.append(plot_neural_vs_fermi_vs_rd(summary, args.output_dir))
    if "alpha" in selected:
        paths.append(plot_alpha_sensitivity(summary, args.output_dir))
    if "basin" in selected:
        paths.append(plot_basin_sensitivity(summary, args.output_dir))
    if "reputation" in selected:
        paths.append(plot_reputation_mobility(summary, args.output_dir))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
