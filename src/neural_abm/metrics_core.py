"""Torch-free numerical metrics shared by lightweight package surfaces."""

from __future__ import annotations

import math

import numpy as np


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
