#!/usr/bin/env python
"""Plot Toy 4 reputation and mobility sweep diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = [
    ("imitation", "none"),
    ("imitation", "output_average"),
    ("neural_policy", "none"),
    ("neural_policy", "output_average"),
    ("reputation_imitation", "none"),
    ("reputation_imitation", "output_average"),
]
CONDITION_LABELS = {
    ("imitation", "none"): "Imitation / none",
    ("imitation", "output_average"): "Imitation / output",
    ("neural_policy", "none"): "Neural / none",
    ("neural_policy", "output_average"): "Neural / output",
    ("reputation_imitation", "none"): "Reputation / none",
    ("reputation_imitation", "output_average"): "Reputation / output",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "experiments/results/toy4_reputation_sweep_seeds01_03_summary.csv"
        ),
        help="Toy 4 sweep summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Figure output directory.",
    )
    return parser.parse_args()


def condition_label(update_rule: str, mixer: str) -> str:
    return CONDITION_LABELS.get((update_rule, mixer), f"{update_rule} / {mixer}")


def bool_label(value: object) -> str:
    text = str(value).lower()
    if text in {"true", "1"}:
        return "mobility on"
    return "mobility off"


def require_columns(summary: pd.DataFrame) -> None:
    required = {
        "policy_rule",
        "coordination_mixer",
        "mobility_enabled",
        "final_action_rate",
        "final_mean_reputation",
        "final_mobility_rate",
        "final_mean_mobility_gain",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Toy 4 sweep summary missing columns: {', '.join(missing)}")


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> Path:
    require_columns(summary)
    numeric_columns = [
        "final_action_rate",
        "final_mean_reputation",
        "final_mobility_rate",
        "final_mean_mobility_gain",
    ]
    df = summary.copy()
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    grouped = (
        df.groupby(["policy_rule", "coordination_mixer", "mobility_enabled"], dropna=False)
        .agg(
            action=("final_action_rate", "mean"),
            reputation=("final_mean_reputation", "mean"),
            mobility_rate=("final_mobility_rate", "mean"),
            mobility_gain=("final_mean_mobility_gain", "mean"),
        )
        .reset_index()
    )
    grouped["condition"] = [
        condition_label(row.policy_rule, row.coordination_mixer)
        for row in grouped.itertuples(index=False)
    ]
    mobility_values = list(dict.fromkeys(grouped["mobility_enabled"].tolist()))
    colors = {"mobility off": "#6c757d", "mobility on": "#1f77b4"}

    x = np.arange(len(CONDITION_ORDER))
    width = 0.34 if len(mobility_values) > 1 else 0.52
    offsets = np.linspace(
        -width * (len(mobility_values) - 1) / 2,
        width * (len(mobility_values) - 1) / 2,
        len(mobility_values),
    )
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.8), constrained_layout=True)
    fig.suptitle("Toy 4: reputation and mobility diagnostics", fontsize=11)
    metrics = [
        ("action", "Final action", "action rate", (0.0, 1.02)),
        ("reputation", "Final reputation", "mean reputation", (0.0, 1.02)),
        ("mobility_rate", "Final mobility", "mobility rate", (0.0, 1.02)),
        ("mobility_gain", "Mobility gain", "mean gain", None),
    ]

    for offset, mobility_enabled in zip(offsets, mobility_values, strict=True):
        subset = grouped[grouped["mobility_enabled"] == mobility_enabled]
        subset = subset.set_index(["policy_rule", "coordination_mixer"])
        label = bool_label(mobility_enabled)
        for ax, (field, title, ylabel, ylim) in zip(axes, metrics, strict=True):
            values = [
                (
                    float(subset.loc[condition, field])
                    if condition in subset.index
                    else np.nan
                )
                for condition in CONDITION_ORDER
            ]
            ax.bar(
                x + offset,
                values,
                width,
                color=colors[label],
                alpha=0.82,
                label=label,
            )
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_xticks(
                x,
                [condition_label(*condition) for condition in CONDITION_ORDER],
                rotation=35,
                ha="right",
            )
            ax.grid(axis="y", alpha=0.25)
    axes[-1].legend(loc="best", fontsize=8, frameon=False)

    path = output_dir / "toy4_reputation_mobility_sweep.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary)
    path = plot_summary(summary, args.output_dir)
    print(path)


if __name__ == "__main__":
    main()
