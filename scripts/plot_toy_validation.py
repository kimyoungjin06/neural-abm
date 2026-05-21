#!/usr/bin/env python
"""Plot the Toy 1-5 validation summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_LABELS = {
    "toy1_no_social": "No social",
    "toy1_output_average": "Output avg.",
    "toy2_harsh_pd_neural_none": "Neural\nnone",
    "toy2_harsh_pd_fermi_none": "Fermi\nnone",
    "toy2_harsh_pd_neural_output_average": "Neural\noutput",
    "toy2_harsh_pd_neural_reputation_observation_output_average": "Neural+rep\noutput",
    "toy2_harsh_pd_reputation_output_average": "Reputation\noutput",
    "toy3_hk_no_rewire": "HK\nno rewire",
    "toy3_hk_rewire": "HK\nrewire",
    "toy3_neural_output_average": "Neural\noutput",
    "toy4_static_imitation_none": "Static\nimitation",
    "toy4_static_neural_output_average": "Static\nneural",
    "toy4_static_neural_reputation_observation_output_average": "Static\nneural+rep",
    "toy4_static_reputation_output_average": "Static\nreputation",
    "toy4_commons_collapse": "Commons\ncollapse",
    "toy5_low_threshold_cascade": "Low\nthreshold",
    "toy5_high_threshold_block": "High\nthreshold",
    "toy5_heterogeneous_partial": "Hetero.",
    "toy5_neural_output_average": "Neural\noutput",
    "toy5_neural_reputation_observation_output_average": "Neural+rep\noutput",
    "toy5_reputation_output_average": "Reputation\noutput",
}

COLORS = {
    "baseline": "#6c757d",
    "social": "#1f77b4",
    "reference": "#2ca02c",
    "stress": "#d62728",
    "diagnostic": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "experiments/results/toy_validation_representative_seeds01_03_metrics.csv"
        ),
        help="Validation long-format metrics CSV.",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path(
            "experiments/results/toy_validation_representative_seeds01_03_runs.csv"
        ),
        help="Validation run-level CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/toy_validation_representative_summary.png"),
        help="Output figure path.",
    )
    return parser.parse_args()


def grouped_metrics(metrics_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        raise ValueError(f"No metric rows found in {metrics_path}")
    return (
        metrics.groupby(["toy", "scenario", "metric"], as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"), seeds=("seed", "nunique"))
        .fillna({"std": 0.0})
    )


def metric_value(
    grouped: pd.DataFrame,
    scenario: str,
    metric: str,
    field: str = "mean",
) -> float:
    rows = grouped[(grouped["scenario"] == scenario) & (grouped["metric"] == metric)]
    if rows.empty:
        raise ValueError(f"Missing metric {metric!r} for scenario {scenario!r}")
    return float(rows.iloc[0][field])


def optional_metric_value(
    grouped: pd.DataFrame,
    scenario: str,
    metric: str,
    field: str = "mean",
) -> float | None:
    try:
        return metric_value(grouped, scenario, metric, field)
    except ValueError:
        return None


def metric_values(
    grouped: pd.DataFrame,
    scenarios: list[str],
    metric: str,
) -> tuple[list[float], list[float]]:
    means = [metric_value(grouped, scenario, metric, "mean") for scenario in scenarios]
    stds = [metric_value(grouped, scenario, metric, "std") for scenario in scenarios]
    return means, stds


def present_scenarios(grouped: pd.DataFrame, scenarios: list[str]) -> list[str]:
    present = set(grouped["scenario"])
    selected = [scenario for scenario in scenarios if scenario in present]
    if not selected:
        raise ValueError("None of the requested scenarios are present")
    return selected


def scenario_labels(scenarios: list[str]) -> list[str]:
    return [SCENARIO_LABELS.get(scenario, scenario) for scenario in scenarios]


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)


def plot_toy1(ax: plt.Axes, grouped: pd.DataFrame) -> None:
    scenarios = present_scenarios(
        grouped,
        ["toy1_no_social", "toy1_output_average"],
    )
    x = np.arange(len(scenarios))
    width = 0.36
    accuracy, accuracy_std = metric_values(
        grouped,
        scenarios,
        "domain_final_mean_global_accuracy",
    )
    consensus, consensus_std = metric_values(
        grouped,
        scenarios,
        "domain_final_mean_consensus",
    )
    ax.bar(
        x - width / 2,
        accuracy,
        width,
        yerr=accuracy_std,
        capsize=3,
        color=COLORS["baseline"],
        label="accuracy",
    )
    ax.bar(
        x + width / 2,
        consensus,
        width,
        yerr=consensus_std,
        capsize=3,
        color=COLORS["social"],
        label="consensus",
    )
    ax.set_title("Toy 1: classification", fontsize=10)
    ax.set_ylabel("mean final metric")
    all_values = accuracy + consensus
    ax.set_ylim(
        max(0.0, min(all_values) - 0.05),
        min(1.0, max(0.05, max(all_values) * 1.05)),
    )
    ax.set_xticks(x, scenario_labels(scenarios))
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    style_axis(ax)


def plot_toy2(ax: plt.Axes, grouped: pd.DataFrame) -> None:
    scenarios = [
        "toy2_harsh_pd_neural_none",
        "toy2_harsh_pd_fermi_none",
        "toy2_harsh_pd_neural_output_average",
        "toy2_harsh_pd_neural_reputation_observation_output_average",
        "toy2_harsh_pd_reputation_output_average",
    ]
    scenarios = present_scenarios(grouped, scenarios)
    means, stds = metric_values(grouped, scenarios, "final_action_rate")
    x = np.arange(len(scenarios))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=3,
        color=[
            COLORS["diagnostic"]
            if "reputation_observation" in scenario
            else COLORS["reference"]
            if "fermi" in scenario
            else COLORS["social"]
            if "neural_output" in scenario
            else COLORS["baseline"]
            for scenario in scenarios
        ],
    )
    ax.axhline(0.20, color="#202020", linestyle="--", linewidth=1.0)
    ax.text(
        max(0.0, len(scenarios) - 1.0),
        0.205,
        "gate <= 0.20",
        fontsize=7,
        va="bottom",
    )
    ax.set_title("Toy 2: harsh PD", fontsize=10)
    ax.set_ylabel("action rate")
    ax.set_ylim(0.0, max(0.25, max(means) * 1.2))
    ax.set_xticks(x, scenario_labels(scenarios))
    style_axis(ax)


def plot_toy3(ax: plt.Axes, grouped: pd.DataFrame) -> None:
    scenarios = [
        "toy3_hk_no_rewire",
        "toy3_hk_rewire",
        "toy3_neural_output_average",
    ]
    scenarios = present_scenarios(grouped, scenarios)
    means, stds = metric_values(
        grouped, scenarios, "domain_final_mean_edge_disagreement"
    )
    x = np.arange(len(scenarios))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=3,
        color=[
            COLORS["stress"]
            if "rewire" in scenario and "no_rewire" not in scenario
            else COLORS["social"]
            if "neural" in scenario
            else COLORS["baseline"]
            for scenario in scenarios
        ],
    )
    rewired = optional_metric_value(
        grouped,
        "toy3_hk_rewire",
        "domain_cumulative_rewired_edge_count",
    )
    if rewired is not None and "toy3_hk_rewire" in scenarios:
        ax.text(
            scenarios.index("toy3_hk_rewire"),
            max(0.02, max(means) * 0.12),
            f"rewired\n{rewired:.0f}",
            ha="center",
            va="center",
            fontsize=8,
            color="#202020",
        )
    ax.set_title("Toy 3: opinion rewiring", fontsize=10)
    ax.set_ylabel("edge disagreement")
    ax.set_ylim(0.0, max(0.05, max(means) * 1.2))
    ax.set_xticks(x, scenario_labels(scenarios))
    style_axis(ax)


def plot_toy4(ax: plt.Axes, grouped: pd.DataFrame) -> None:
    scenarios = [
        "toy4_static_imitation_none",
        "toy4_static_neural_output_average",
        "toy4_static_neural_reputation_observation_output_average",
        "toy4_static_reputation_output_average",
        "toy4_commons_collapse",
    ]
    scenarios = present_scenarios(grouped, scenarios)
    means, stds = metric_values(grouped, scenarios, "final_action_rate")
    x = np.arange(len(scenarios))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=3,
        color=[
            COLORS["stress"]
            if "collapse" in scenario
            else COLORS["diagnostic"]
            if "reputation_observation" in scenario
            else COLORS["social"]
            if "neural" in scenario
            else COLORS["baseline"]
            for scenario in scenarios
        ],
    )
    collapse_time = optional_metric_value(
        grouped,
        "toy4_commons_collapse",
        "domain_collapse_time",
    )
    resource_level = optional_metric_value(
        grouped,
        "toy4_commons_collapse",
        "domain_resource_level",
    )
    if (
        "toy4_commons_collapse" in scenarios
        and collapse_time is not None
        and resource_level is not None
    ):
        ax.text(
            scenarios.index("toy4_commons_collapse"),
            0.09,
            f"collapse t={collapse_time:.0f}\nresource={resource_level:.0f}",
            ha="center",
            va="center",
            fontsize=8,
        )
    ax.set_title("Toy 4: public goods", fontsize=10)
    ax.set_ylabel("action rate")
    ax.set_ylim(0.0, max(0.12, max(means) * 1.2))
    ax.set_xticks(x, scenario_labels(scenarios))
    style_axis(ax)


def plot_toy5(ax: plt.Axes, grouped: pd.DataFrame) -> None:
    scenarios = [
        "toy5_low_threshold_cascade",
        "toy5_high_threshold_block",
        "toy5_heterogeneous_partial",
        "toy5_neural_output_average",
        "toy5_neural_reputation_observation_output_average",
        "toy5_reputation_output_average",
    ]
    scenarios = present_scenarios(grouped, scenarios)
    means, stds = metric_values(grouped, scenarios, "final_action_rate")
    x = np.arange(len(scenarios))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=3,
        color=[
            COLORS["stress"]
            if "high_threshold" in scenario
            else COLORS["diagnostic"]
            if "heterogeneous" in scenario or "reputation_observation" in scenario
            else COLORS["social"]
            if "neural_output" in scenario
            else COLORS["baseline"]
            if "reputation_output" in scenario
            else COLORS["reference"]
            for scenario in scenarios
        ],
    )
    ax.axhline(0.90, color="#202020", linestyle="--", linewidth=1.0)
    ax.axhline(0.20, color="#202020", linestyle=":", linewidth=1.0)
    gate_x = min(1.0, max(0.0, len(scenarios) - 1.0))
    ax.text(gate_x, 0.915, "cascade gate", fontsize=7, va="bottom")
    ax.text(gate_x, 0.215, "block gate", fontsize=7, va="bottom")
    ax.set_title("Toy 5: adoption", fontsize=10)
    ax.set_ylabel("action rate")
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks(x, scenario_labels(scenarios))
    style_axis(ax)


def plot_pass_fail(ax: plt.Axes, runs_path: Path) -> None:
    runs = pd.read_csv(runs_path)
    grouped = runs.groupby("toy", as_index=False).agg(
        runs=("status", "size"),
        passed=("status", lambda values: int((values == "pass").sum())),
    )
    grouped["failed"] = grouped["runs"] - grouped["passed"]
    x = np.arange(len(grouped))
    ax.bar(x, grouped["passed"], color=COLORS["reference"], label="pass")
    ax.bar(
        x,
        grouped["failed"],
        bottom=grouped["passed"],
        color=COLORS["stress"],
        label="fail",
    )
    for index, row in grouped.iterrows():
        ax.text(
            index,
            float(row["runs"]) + 0.15,
            f"{int(row['passed'])}/{int(row['runs'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("Validation status", fontsize=10)
    ax.set_ylabel("scenario-seed runs")
    ax.set_ylim(0, max(grouped["runs"]) + 2.0)
    ax.set_xticks(x, grouped["toy"])
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    style_axis(ax)


def main() -> None:
    args = parse_args()
    grouped = grouped_metrics(args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.1), constrained_layout=True)
    fig.suptitle("Toy 1-5 validation suite", fontsize=12)
    flat_axes = axes.ravel()

    plot_toy1(flat_axes[0], grouped)
    plot_toy2(flat_axes[1], grouped)
    plot_toy3(flat_axes[2], grouped)
    plot_toy4(flat_axes[3], grouped)
    plot_toy5(flat_axes[4], grouped)
    plot_pass_fail(flat_axes[5], args.runs)

    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
