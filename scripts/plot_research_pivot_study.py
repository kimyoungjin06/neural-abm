"""Render case-study figures for the replicated researcher PIVOT study.

Reads the JSON artifacts written by the researcher-pivot studies, writes the
case-study figures, and refreshes the selected manuscript copies in
``paper/figures``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"

SCENARIO_ORDER = (
    "baseline",
    "interdisciplinary_seed_grants",
    "hot_field_hype",
    "hype_with_support",
)
SCENARIO_LABELS = {
    "baseline": "Baseline",
    "interdisciplinary_seed_grants": "Seed grants",
    "hot_field_hype": "Hot-field hype",
    "hype_with_support": "Hype + support",
}
# Fixed entity-to-color mapping, validated for CVD separation and contrast.
SCENARIO_COLORS = {
    "baseline": MUTED,
    "interdisciplinary_seed_grants": "#2a78d6",
    "hot_field_hype": "#eb6834",
    "hype_with_support": "#4a3aa7",
}
STATUS_GOOD = "#0ca30c"
STATUS_SERIOUS = "#ec835a"

# Study 2 learning-arm colors (validated pair + gray reference).
ARM_COLORS = {
    "frozen": MUTED,
    "imitative": "#e87ba4",
    "cautionary": "#2a78d6",
}
ARM_LABELS = {
    "frozen": "Frozen rule",
    "imitative": "Imitative learning",
    "cautionary": "Cautionary learning",
}
LEARNING_SCENARIOS = (
    "baseline",
    "interdisciplinary_seed_grants",
    "hot_field_hype",
)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": BASELINE_AXIS,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "font.family": "sans-serif",
        }
    )


def plot_outcome_distributions(payload: dict[str, Any], target: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
    rng = np.random.default_rng(7)
    positions = np.arange(len(SCENARIO_ORDER))[::-1]
    for position, name in zip(positions, SCENARIO_ORDER, strict=True):
        values = np.asarray(payload["scenarios"][name]["outcome_values"])
        color = SCENARIO_COLORS[name]
        jitter = rng.uniform(-0.16, 0.16, size=values.size)
        ax.scatter(
            values,
            position + jitter,
            s=14,
            color=color,
            alpha=0.35,
            linewidths=0,
            zorder=2,
        )
        box = ax.boxplot(
            values,
            positions=[position],
            vert=False,
            widths=0.52,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": INK, "linewidth": 1.6},
            boxprops={"facecolor": "none", "edgecolor": color, "linewidth": 1.4},
            whiskerprops={"color": color, "linewidth": 1.2},
            capprops={"color": color, "linewidth": 1.2},
        )
        del box
        mean = float(values.mean())
        ax.annotate(
            f"{mean:.2f}",
            xy=(mean, position + 0.38),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels([SCENARIO_LABELS[name] for name in SCENARIO_ORDER])
    ax.set_xlabel("Productive pivot rate (share of all researchers, final step)")
    ax.set_xlim(left=-0.02)
    ax.grid(axis="y", visible=False)
    replicates = payload["config"]["replicates"]
    ax.set_title(f"Productive pivots by environment ({replicates} paired replicates)")
    fig.savefig(target, dpi=200)
    plt.close(fig)


def plot_pivot_composition(payload: dict[str, Any], target: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    x = np.arange(len(SCENARIO_ORDER))
    productive = []
    failed = []
    for name in SCENARIO_ORDER:
        summaries = payload["scenarios"][name]["aggregate_summaries"]
        productive.append(summaries["productive_pivot_rate"]["mean"])
        failed.append(summaries["failed_pivot_rate"]["mean"])
    bar_kwargs = {"width": 0.56, "edgecolor": SURFACE, "linewidth": 2.0}
    ax.bar(x, productive, color=STATUS_GOOD, label="Productive pivot", **bar_kwargs)
    ax.bar(
        x,
        failed,
        bottom=productive,
        color=STATUS_SERIOUS,
        label="Failed pivot",
        **bar_kwargs,
    )
    for index, name in enumerate(SCENARIO_ORDER):
        total = productive[index] + failed[index]
        share = productive[index] / total if total else 0.0
        ax.annotate(
            f"{total:.2f} pivot",
            xy=(index, total + 0.015),
            ha="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
        if total >= 0.08:
            ax.annotate(
                f"{share:.0%}\nproductive",
                xy=(index, productive[index] / 2),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=SURFACE,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[name] for name in SCENARIO_ORDER])
    ax.set_ylabel("Share of all researchers (mean over replicates)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("Hype buys pivots, support buys productive pivots")
    fig.savefig(target, dpi=200)
    plt.close(fig)


def plot_sensitivity(payload: dict[str, Any], target: Path) -> None:
    sensitivity = payload["sensitivity"]
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(9.6, 3.6),
        layout="constrained",
    )

    scales = [row["grant_scale"] for row in sensitivity["grant_scale_sweep"]]
    for key in ("interdisciplinary_seed_grants", "hype_with_support"):
        short = "seed_grants" if key == "interdisciplinary_seed_grants" else key
        rows = [row[short] for row in sensitivity["grant_scale_sweep"]]
        means = [row["mean_delta"] for row in rows]
        low = [row["delta_ci95"][0] for row in rows]
        high = [row["delta_ci95"][1] for row in rows]
        color = SCENARIO_COLORS[key]
        left.plot(
            scales,
            means,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=SCENARIO_LABELS[key],
        )
        left.fill_between(scales, low, high, color=color, alpha=0.14, linewidth=0)
    left.axhline(0.0, color=BASELINE_AXIS, linewidth=1.0)
    left.axhline(
        payload["config"]["success_min_delta"],
        color=MUTED,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
    )
    left.annotate(
        "success threshold",
        xy=(scales[0], payload["config"]["success_min_delta"] + 0.012),
        fontsize=8,
        color=MUTED,
    )
    left.set_xlabel("Grant intensity (x default seed-grant signals)")
    left.set_ylabel("Productive pivot rate delta vs baseline")
    left.set_title("Support dose-response")
    left.legend(loc="upper left")

    alphas = [row["social_alpha"] for row in sensitivity["social_alpha_sweep"]]
    for name in SCENARIO_ORDER:
        means = [
            row["scenarios"][name]["productive_pivot_rate"]["mean"]
            for row in sensitivity["social_alpha_sweep"]
        ]
        right.plot(
            alphas,
            means,
            color=SCENARIO_COLORS[name],
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=SCENARIO_LABELS[name],
        )
    right.set_xlabel("Social mixing weight (alpha)")
    right.set_ylabel("Productive pivot rate")
    right.set_title("Peer influence sensitivity")
    right.legend(loc="upper left")

    replicates = sensitivity["replicates_per_point"]
    fig.suptitle(
        f"Sensitivity sweeps ({replicates} replicates per point)",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(target, dpi=200)
    plt.close(fig)


def plot_readiness_trajectories(payload: dict[str, Any], target: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    steps = None
    for name in SCENARIO_ORDER:
        summary = payload["scenarios"][name]
        mean = np.asarray(summary["mean_state_trajectory"])
        std = np.asarray(summary["std_state_trajectory"])
        steps = np.arange(1, mean.size + 1)
        color = SCENARIO_COLORS[name]
        ax.plot(steps, mean, color=color, linewidth=2.0)
        ax.fill_between(
            steps,
            mean - std,
            mean + std,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax.annotate(
            SCENARIO_LABELS[name],
            xy=(steps[-1], mean[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )
    threshold = payload["config"]["pivot_threshold"]
    ax.axhline(threshold, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    assert steps is not None
    ax.annotate(
        "pivot threshold",
        xy=(steps[0], threshold + 0.008),
        fontsize=8,
        color=MUTED,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean pivot readiness (± s.d. across replicates)")
    ax.set_xlim(steps[0], steps[-1] + 1.6)
    ax.set_xticks(steps)
    ax.set_title("Population readiness trajectories by environment")
    fig.savefig(target, dpi=200)
    plt.close(fig)


def _spread_label_positions(
    values: list[float],
    min_gap: float,
) -> list[float]:
    """Nudge overlapping end-of-line label positions apart, preserving order."""

    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [values[index] for index in order]
    for position in range(1, len(adjusted)):
        if adjusted[position] - adjusted[position - 1] < min_gap:
            adjusted[position] = adjusted[position - 1] + min_gap
    spread = [0.0] * len(values)
    for rank, index in enumerate(order):
        spread[index] = adjusted[rank]
    return spread


def plot_learning_failed_trajectories(
    payload: dict[str, Any],
    target: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(LEARNING_SCENARIOS),
        figsize=(10.8, 3.4),
        layout="constrained",
        sharey=True,
    )
    steps = None
    for ax, scenario in zip(axes, LEARNING_SCENARIOS, strict=True):
        end_values: list[float] = []
        for arm, color in ARM_COLORS.items():
            summary = payload["scenarios"][scenario][arm]
            mean = np.asarray(summary["cumulative_failed_rate_mean"])
            std = np.asarray(summary["cumulative_failed_rate_std"])
            steps = np.arange(1, mean.size + 1)
            # The control arm is dashed and drawn on top so it stays visible
            # when a learning arm's curve coincides with it exactly.
            ax.plot(
                steps,
                mean,
                color=color,
                linewidth=2.0,
                linestyle=(0, (4, 2.5)) if arm == "frozen" else "-",
                zorder=3 if arm == "frozen" else 2,
            )
            ax.fill_between(
                steps,
                mean - std,
                mean + std,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            end_values.append(float(mean[-1]))
        label_positions = _spread_label_positions(end_values, min_gap=0.038)
        assert steps is not None
        for (arm, color), label_y in zip(
            ARM_COLORS.items(),
            label_positions,
            strict=True,
        ):
            ax.annotate(
                ARM_LABELS[arm].split()[0],
                xy=(steps[-1], label_y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=color,
            )
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_xlabel("Step")
        ax.set_xlim(steps[0], steps[-1] + 3.0)
    axes[0].set_ylabel("Cumulative failed pivot rate")
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=color,
            linewidth=2.0,
            linestyle=(0, (4, 2.5)) if arm == "frozen" else "-",
            label=ARM_LABELS[arm],
        )
        for arm, color in ARM_COLORS.items()
    ]
    axes[0].legend(handles=handles, loc="upper left")
    replicates = payload["config"]["replicates"]
    fig.suptitle(
        "Learning direction decides the failure curve "
        f"({replicates} paired replicates)",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(target, dpi=200)
    plt.close(fig)


def plot_learning_attention_weights(
    payload: dict[str, Any],
    target: Path,
) -> None:
    fig, (trajectory_ax, audit_ax) = plt.subplots(
        1,
        2,
        figsize=(11.2, 3.8),
        layout="constrained",
        gridspec_kw={"width_ratios": (0.9, 1.5)},
    )
    steps = None
    end_values: list[float] = []
    for scenario in LEARNING_SCENARIOS:
        summary = payload["scenarios"][scenario]["cautionary"]
        weights = np.asarray(summary["mean_attention_weight"])
        steps = np.arange(1, weights.size + 1)
        trajectory_ax.plot(
            steps,
            weights,
            color=SCENARIO_COLORS[scenario],
            linewidth=2.0,
            linestyle=(0, (4, 2.5)) if scenario == "baseline" else "-",
            zorder=3 if scenario == "baseline" else 2,
        )
        end_values.append(float(weights[-1]))
    assert steps is not None
    label_positions = _spread_label_positions(end_values, min_gap=0.016)
    trajectory_ax.set_ylim(
        min(end_values) - 0.02,
        max(max(label_positions), max(end_values)) + 0.012,
    )
    for scenario, label_y in zip(LEARNING_SCENARIOS, label_positions, strict=True):
        trajectory_ax.annotate(
            SCENARIO_LABELS[scenario],
            xy=(steps[-1], label_y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=SCENARIO_COLORS[scenario],
        )
    trajectory_ax.set_xlim(steps[0], steps[-1] + 3.4)
    trajectory_ax.set_xlabel("Step")
    trajectory_ax.set_ylabel("Mean attention coefficient")
    trajectory_ax.set_title("Attention coefficient trajectory")

    features = list(payload["features"])
    labels = [feature.replace("_", "\n") for feature in features] + ["bias"]
    initial = payload["policy_initialization"]
    scale = float(initial["logit_scale"])
    initial_values = np.asarray(initial["study1_linear_weights"], dtype=float) * scale
    initial_bias = scale * (float(initial["study1_linear_intercept"]) - 0.5)
    changes: list[list[float]] = []
    for scenario in LEARNING_SCENARIOS:
        summary = payload["scenarios"][scenario]["cautionary"]
        final_weights = np.asarray(
            [summary["mean_weight_trajectories"][feature][-1] for feature in features]
        )
        final_bias = float(summary["mean_bias_trajectory"][-1])
        changes.append([*(final_weights - initial_values), final_bias - initial_bias])
    change_array = np.asarray(changes)
    bound = max(0.01, float(np.max(np.abs(change_array))))
    image = audit_ax.imshow(
        change_array,
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
        aspect="auto",
    )
    audit_ax.set_xticks(np.arange(len(labels)))
    audit_ax.set_xticklabels(labels, fontsize=7.2)
    audit_ax.set_yticks(np.arange(len(LEARNING_SCENARIOS)))
    audit_ax.set_yticklabels(
        [SCENARIO_LABELS[scenario] for scenario in LEARNING_SCENARIOS]
    )
    audit_ax.grid(False)
    audit_ax.set_title("Final mean coefficient change from the prior")
    for row in range(change_array.shape[0]):
        for column in range(change_array.shape[1]):
            value = float(change_array[row, column])
            audit_ax.text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=6.7,
                color=SURFACE if abs(value) > 0.55 * bound else INK,
            )
    colorbar = fig.colorbar(image, ax=audit_ax, shrink=0.82, pad=0.02)
    colorbar.set_label("Change from initialization", fontsize=8)
    fig.suptitle(
        "Failure-only learning updates a parameter vector, not attention alone",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(target, dpi=200)
    plt.close(fig)


def render_case_study_figures(
    *,
    input_path: Path,
    learning_input: Path,
    figures: Path,
    paper_figures: Path,
) -> tuple[list[str], list[str]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    figures.mkdir(parents=True, exist_ok=True)
    _style()

    plot_outcome_distributions(
        payload,
        figures / "fig1_productive_pivot_distributions.png",
    )
    generated = ["fig1_productive_pivot_distributions.png"]
    composition = figures / "fig2_pivot_composition.png"
    plot_pivot_composition(payload, composition)
    generated.append(composition.name)
    manuscript_copies = [(composition, "pivot_composition.png")]
    if "sensitivity" in payload:
        sensitivity = figures / "fig3_sensitivity.png"
        plot_sensitivity(payload, sensitivity)
        generated.append(sensitivity.name)
    readiness = figures / "fig4_readiness_trajectories.png"
    plot_readiness_trajectories(
        payload,
        readiness,
    )
    generated.append(readiness.name)
    if learning_input.exists():
        learning_payload = json.loads(learning_input.read_text(encoding="utf-8"))
        failed_trajectories = figures / "fig5_learning_failed_trajectories.png"
        attention_weights = figures / "fig6_learning_attention_weights.png"
        plot_learning_failed_trajectories(
            learning_payload,
            failed_trajectories,
        )
        plot_learning_attention_weights(
            learning_payload,
            attention_weights,
        )
        generated.extend((failed_trajectories.name, attention_weights.name))
        manuscript_copies.extend(
            (
                (failed_trajectories, "pivot_learning_failed_trajectories.png"),
                (attention_weights, "pivot_learning_attention_weights.png"),
            )
        )

    paper_figures.mkdir(parents=True, exist_ok=True)
    for source, filename in manuscript_copies:
        shutil.copy2(source, paper_figures / filename)
    return generated, [filename for _source, filename in manuscript_copies]


def _assert_current(generated: Path, tracked: Path, names: list[str]) -> None:
    stale = [
        name
        for name in names
        if not (tracked / name).exists()
        or (generated / name).read_bytes() != (tracked / name).read_bytes()
    ]
    if stale:
        raise SystemExit(f"stale generated assets: {', '.join(stale)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/case-studies/researcher-pivot/data/study_results.json"),
    )
    parser.add_argument(
        "--learning-input",
        type=Path,
        default=Path(
            "docs/case-studies/researcher-pivot/data/learning_study_results.json"
        ),
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=Path("docs/case-studies/researcher-pivot/figures"),
    )
    parser.add_argument(
        "--paper-figures",
        type=Path,
        default=Path("paper/figures"),
        help="Directory receiving the manuscript copies of selected figures.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if tracked figures differ from a fresh render.",
    )
    args = parser.parse_args()

    if args.check:
        with tempfile.TemporaryDirectory(prefix="research-pivot-figures.") as raw:
            root = Path(raw)
            figures = root / "figures"
            paper_figures = root / "paper"
            generated, paper_generated = render_case_study_figures(
                input_path=args.input,
                learning_input=args.learning_input,
                figures=figures,
                paper_figures=paper_figures,
            )
            _assert_current(figures, args.figures, generated)
            _assert_current(paper_figures, args.paper_figures, paper_generated)
        print("research-pivot figures are current")
        return

    render_case_study_figures(
        input_path=args.input,
        learning_input=args.learning_input,
        figures=args.figures,
        paper_figures=args.paper_figures,
    )
    print(f"figures written to {args.figures}")
    print(f"manuscript copies written to {args.paper_figures}")


if __name__ == "__main__":
    main()
