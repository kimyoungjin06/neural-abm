#!/usr/bin/env python
"""Plot Toy 1 peer and output differentiation across snapshot runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
        help="Sweep summary CSV containing run_dir and threshold columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/toy1_parameter_cluster_comparison.png"),
        help="Output figure path.",
    )
    return parser.parse_args()


def load_snapshots(run_dir: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    paths = sorted((run_dir / "probe_predictions").glob("epoch_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No probe prediction snapshots found in {run_dir}")

    epochs = []
    shard_groups = None
    component_ids = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        epochs.append(int(data["epoch"]))
        if shard_groups is None:
            shard_groups = [str(value) for value in data["shard_groups"]]
        component_ids.append(data["component_ids"])
    return (
        np.array(epochs, dtype=np.int64),
        shard_groups or [],
        np.stack(component_ids, axis=0),
    )


def load_output_js_matrix(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    micro_state = pd.read_csv(run_dir / "micro_state.csv")
    pivot = micro_state.pivot(
        index="epoch",
        columns="agent_id",
        values="domain_output_js_to_population_mean",
    ).sort_index()
    return pivot.index.to_numpy(dtype=np.int64), pivot.to_numpy(dtype=np.float64)


def load_run_dynamics(run_dir: Path) -> dict[str, np.ndarray | list[str]]:
    epochs, shard_groups, peer_components = load_snapshots(run_dir)
    output_epochs, output_js_matrix = load_output_js_matrix(run_dir)
    if not np.array_equal(epochs, output_epochs):
        raise ValueError(f"Snapshot and micro-state epochs do not match for {run_dir}")
    return {
        "epochs": epochs,
        "peer_component_counts": np.array(
            [len(set(row.tolist())) for row in peer_components],
            dtype=np.int64,
        ),
        "peer_components": peer_components,
        "output_js_matrix": output_js_matrix,
        "shard_groups": shard_groups,
    }


def shard_boundaries(shard_groups: list[str]) -> list[float]:
    if not shard_groups:
        return []
    boundaries = []
    last = shard_groups[0]
    for index, group in enumerate(shard_groups[1:], start=1):
        if group != last:
            boundaries.append(index - 0.5)
            last = group
    return boundaries


def plot_condition(
    peer_ax: plt.Axes,
    output_ax: plt.Axes,
    line_ax: plt.Axes,
    row: pd.Series,
    dynamics: dict[str, np.ndarray | list[str]],
    output_js_vmax: float,
) -> None:
    epochs = dynamics["epochs"]
    peer_counts = dynamics["peer_component_counts"]
    peer_components = dynamics["peer_components"]
    output_js_matrix = dynamics["output_js_matrix"]
    shard_groups = dynamics["shard_groups"]

    peer_image = peer_ax.imshow(
        peer_components,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
    )
    title = (
        f"threshold={row.threshold:g}\n"
        f"final acc={row.domain_final_mean_global_accuracy:.3f}, "
        f"frag={row.final_fragmentation_components:.0f}"
    )
    peer_ax.set_title(title, fontsize=9)
    peer_ax.set_xlabel("agent id")
    peer_ax.set_ylabel("epoch")
    peer_ax.set_yticks(
        np.linspace(0, len(epochs) - 1, min(len(epochs), 5), dtype=int),
        epochs[np.linspace(0, len(epochs) - 1, min(len(epochs), 5), dtype=int)],
    )
    for boundary in shard_boundaries(shard_groups):
        peer_ax.axvline(boundary, color="white", linewidth=0.8, alpha=0.7)

    output_image = output_ax.imshow(
        output_js_matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=output_js_vmax,
        cmap="magma",
    )
    output_ax.set_xlabel("agent id")
    output_ax.set_ylabel("epoch")
    output_ax.set_title("output divergence to population mean", fontsize=9)
    output_ax.set_yticks(
        np.linspace(0, len(epochs) - 1, min(len(epochs), 5), dtype=int),
        epochs[np.linspace(0, len(epochs) - 1, min(len(epochs), 5), dtype=int)],
    )
    for boundary in shard_boundaries(shard_groups):
        output_ax.axvline(boundary, color="white", linewidth=0.8, alpha=0.55)

    line_ax.plot(
        epochs,
        peer_counts,
        marker="s",
        linewidth=1.5,
        markersize=3,
        label="peer components",
        color="#ff7f0e",
    )
    line_ax.set_xlabel("epoch")
    line_ax.set_ylabel("peer components")
    line_ax.set_ylim(0, 55)
    line_ax.grid(True, alpha=0.25)
    line_ax.set_title("component and divergence trajectory", fontsize=9)

    twin_ax = line_ax.twinx()
    twin_ax.plot(
        epochs,
        output_js_matrix.mean(axis=1),
        marker="o",
        linewidth=1.4,
        markersize=3,
        label="mean output JS",
        color="#1f77b4",
    )
    twin_ax.set_ylabel("mean output JS")
    twin_ax.set_ylim(0, output_js_vmax)
    return peer_image, output_image, twin_ax


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary_csv).sort_values("threshold")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    run_dynamics = [
        load_run_dynamics(run_dir=Path(row.run_dir))
        for row in df.itertuples(index=False)
    ]
    output_js_vmax = max(
        float(np.max(dynamics["output_js_matrix"])) for dynamics in run_dynamics
    )
    output_js_vmax = max(output_js_vmax, 1e-3)

    column_count = len(df)
    fig, axes = plt.subplots(
        3,
        column_count,
        figsize=(4.2 * column_count, 8.0),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        "Toy 1: parameter-path differentiation across peer thresholds",
        fontsize=11,
    )

    peer_image = None
    output_image = None
    twin_ax = None
    for column, (row, dynamics) in enumerate(
        zip(df.itertuples(index=False), run_dynamics, strict=True)
    ):
        peer_image, output_image, twin_ax = plot_condition(
            peer_ax=axes[0, column],
            output_ax=axes[1, column],
            line_ax=axes[2, column],
            row=row,
            dynamics=dynamics,
            output_js_vmax=output_js_vmax,
        )

    handles, labels = axes[2, -1].get_legend_handles_labels()
    if twin_ax is not None:
        twin_handles, twin_labels = twin_ax.get_legend_handles_labels()
        handles += twin_handles
        labels += twin_labels
    axes[2, -1].legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
    if peer_image is not None:
        fig.colorbar(
            peer_image,
            ax=axes[0, :],
            shrink=0.72,
            label="peer component id",
        )
    if output_image is not None:
        fig.colorbar(
            output_image,
            ax=axes[1, :],
            shrink=0.72,
            label="JS to population mean",
        )

    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
