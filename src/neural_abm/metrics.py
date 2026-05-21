"""Metrics for task performance and social dynamics."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import torch


def accuracy_from_probs(probs: torch.Tensor, labels: torch.Tensor) -> float:
    pred = torch.argmax(probs, dim=-1)
    return float((pred == labels).float().mean().cpu())


def cross_entropy_from_probs(probs: torch.Tensor, labels: torch.Tensor) -> float:
    eps = 1e-8
    selected = probs[torch.arange(len(labels)), labels]
    return float((-torch.log(selected + eps)).mean().cpu())


def entropy_mean(probs: torch.Tensor) -> float:
    eps = 1e-8
    entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
    return float(entropy.mean().cpu())


def js_divergence_np(p: np.ndarray, q: np.ndarray) -> float:
    """Mean normalized Jensen-Shannon divergence across probe rows."""

    eps = 1e-8
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=-1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=-1)
    js = 0.5 * (kl_pm + kl_qm)
    return float(np.mean(js) / math.log(2.0))


def pairwise_output_js(probe_probs: np.ndarray) -> np.ndarray:
    count = probe_probs.shape[0]
    matrix = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            value = js_divergence_np(probe_probs[i], probe_probs[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def consensus(probe_probs: np.ndarray) -> float:
    labels = np.argmax(probe_probs, axis=-1)
    count = labels.shape[0]
    if count <= 1:
        return 1.0
    values = []
    for i in range(count):
        for j in range(i + 1, count):
            values.append(float(np.mean(labels[i] == labels[j])))
    return float(np.mean(values)) if values else 1.0


def polarization_clusters(js_matrix: np.ndarray, similarity_threshold: float) -> int:
    graph = nx.Graph()
    count = js_matrix.shape[0]
    graph.add_nodes_from(range(count))
    for i in range(count):
        for j in range(i + 1, count):
            similarity = 1.0 - js_matrix[i, j]
            if similarity >= similarity_threshold:
                graph.add_edge(i, j)
    return nx.number_connected_components(graph)


def edge_entropy(peer_ids: list[list[int]], agent_count: int) -> float:
    values = []
    normalizer = math.log(max(agent_count - 1, 2))
    for peers in peer_ids:
        if len(peers) <= 1:
            values.append(0.0)
            continue
        weight = 1.0 / len(peers)
        entropy = -len(peers) * weight * math.log(weight)
        values.append(entropy / normalizer)
    return float(np.mean(values)) if values else 0.0
