#!/usr/bin/env python
"""Plot per-agent prediction cluster dynamics from Toy 1 snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing probe_predictions/*.npz.",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for prediction-cluster connected components.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Defaults to run_dir/cluster_dynamics.png.",
    )
    return parser.parse_args()


def js_divergence_matrix(probs: np.ndarray) -> np.ndarray:
    eps = 1e-8
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    count = probs.shape[0]
    matrix = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            p = probs[i]
            q = probs[j]
            m = 0.5 * (p + q)
            kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=-1)
            kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=-1)
            js = np.mean(0.5 * (kl_pm + kl_qm) / np.log(2.0))
            matrix[i, j] = js
            matrix[j, i] = js
    return matrix


def cluster_labels_from_probs(
    probs: np.ndarray, similarity_threshold: float
) -> tuple[np.ndarray, int]:
    js_matrix = js_divergence_matrix(probs)
    graph = nx.Graph()
    count = probs.shape[0]
    graph.add_nodes_from(range(count))
    for i in range(count):
        for j in range(i + 1, count):
            if 1.0 - js_matrix[i, j] >= similarity_threshold:
                graph.add_edge(i, j)

    labels = np.full(count, fill_value=-1, dtype=np.int64)
    components = sorted(nx.connected_components(graph), key=lambda nodes: min(nodes))
    for component_id, nodes in enumerate(components):
        for node in nodes:
            labels[node] = component_id
    return labels, len(components)


def load_snapshots(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    paths = sorted((run_dir / "probe_predictions").glob("epoch_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No probe prediction snapshots found in {run_dir}")

    epochs = []
    all_probs = []
    shard_groups = None
    component_ids = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        epochs.append(int(data["epoch"]))
        all_probs.append(data["probe_probs"])
        if shard_groups is None:
            shard_groups = [str(value) for value in data["shard_groups"]]
        component_ids.append(data["component_ids"])
    return (
        np.array(epochs, dtype=np.int64),
        np.stack(all_probs, axis=0),
        shard_groups or [],
        np.stack(component_ids, axis=0),
    )


def main() -> None:
    args = parse_args()
    output = args.output or args.run_dir / "cluster_dynamics.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    epochs, probs_by_epoch, shard_groups, peer_components = load_snapshots(args.run_dir)
    cluster_rows = []
    cluster_counts = []
    for probs in probs_by_epoch:
        labels, count = cluster_labels_from_probs(probs, args.cluster_threshold)
        cluster_rows.append(labels)
        cluster_counts.append(count)
    cluster_matrix = np.stack(cluster_rows, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.4), sharex=False)
    fig.suptitle(
        f"Toy 1 agent prediction differentiation\n{args.run_dir.name}",
        fontsize=11,
    )

    image = axes[0].imshow(cluster_matrix, aspect="auto", interpolation="nearest")
    axes[0].set_title(
        f"Prediction clusters over epochs (similarity >= {args.cluster_threshold:g})",
        fontsize=10,
    )
    axes[0].set_ylabel("epoch")
    axes[0].set_xlabel("agent id")
    axes[0].set_yticks(
        np.linspace(0, len(epochs) - 1, min(len(epochs), 6), dtype=int),
        epochs[np.linspace(0, len(epochs) - 1, min(len(epochs), 6), dtype=int)],
    )
    plt.colorbar(image, ax=axes[0], shrink=0.8, label="prediction cluster id")

    axes[1].plot(epochs, cluster_counts, marker="o", linewidth=1.6, label="prediction")
    peer_counts = [len(set(row.tolist())) for row in peer_components]
    axes[1].plot(
        epochs,
        peer_counts,
        marker="s",
        linewidth=1.2,
        label="peer graph",
        alpha=0.8,
    )
    axes[1].set_title("Cluster count trajectory", fontsize=10)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("component count")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    if shard_groups:
        boundaries = []
        last = shard_groups[0]
        for index, group in enumerate(shard_groups[1:], start=1):
            if group != last:
                boundaries.append(index - 0.5)
                last = group
        for boundary in boundaries:
            axes[0].axvline(boundary, color="white", linewidth=0.8, alpha=0.7)

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
