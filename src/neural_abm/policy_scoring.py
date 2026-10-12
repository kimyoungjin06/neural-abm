"""Torch-free out-of-sample scoring of agent policies.

Every agent policy is scored on held-out examples it never trained on. The
module computes scores only; comparing policies, choosing margins, and judging
claims stay with the caller (Decision 0016).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from neural_abm.social_core import validate_probability_matrix, validate_probability_vector


POLICY_SCORING_METRICS = ("brier", "log_loss", "accuracy", "auc")
DEFAULT_POLICY_SCORING_METRICS = ("brier", "log_loss", "accuracy")
LOG_LOSS_EPSILON = 1e-8


class PolicyScoringLeakError(ValueError):
    """A policy was trained on a unit it is being scored on."""


@dataclass(frozen=True)
class PolicyScore:
    """One policy's scores on the held-out examples."""

    policy_id: str
    scores: Mapping[str, float | None]

    def to_payload(self) -> dict[str, Any]:
        return {"policy_id": self.policy_id, "scores": dict(self.scores)}


@dataclass(frozen=True)
class PolicyScoreReport:
    """Per-policy, mean, and ensemble scores for one held-out split."""

    split: str
    examples: int
    classes: int
    metrics: tuple[str, ...]
    policies: tuple[PolicyScore, ...]
    mean_scores: Mapping[str, float | None]
    ensemble_scores: Mapping[str, float | None]
    leak_checked: bool

    def scores_of(self, policy_id: str) -> Mapping[str, float | None]:
        for policy in self.policies:
            if policy.policy_id == policy_id:
                return policy.scores
        raise KeyError(policy_id)

    def to_payload(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "examples": self.examples,
            "classes": self.classes,
            "metrics": list(self.metrics),
            "policies": [policy.to_payload() for policy in self.policies],
            "mean_scores": dict(self.mean_scores),
            "ensemble_scores": dict(self.ensemble_scores),
            "leak_checked": self.leak_checked,
        }


def _distribution(values: Any, examples: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        validate_probability_vector(array, name=name)
        array = np.column_stack([1.0 - array, array])
    else:
        validate_probability_matrix(array, name=name)
    if array.shape[0] != examples:
        raise ValueError(f"{name} must hold {examples} examples")
    return array


def _labels(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("labels must be a non-empty 1D array")
    if not np.all(np.isfinite(array.astype(np.float64))) or np.any(array.astype(np.float64) % 1 != 0):
        raise ValueError("labels must be integer class indices")
    return array.astype(np.int64)


def _auc(probability: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-based ROC AUC with average ranks for ties; ``None`` for one class."""

    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(probability, kind="mergesort")
    ordered = probability[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    boundaries = np.flatnonzero(np.diff(ordered)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [labels.size]])
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = (start + end + 1) / 2.0
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def _score(distribution: np.ndarray, labels: np.ndarray, metrics: Sequence[str]) -> dict[str, float | None]:
    examples, classes = distribution.shape
    truth = distribution[np.arange(examples), labels]
    scores: dict[str, float | None] = {}
    for metric in metrics:
        if metric == "brier":
            if classes == 2:
                scores[metric] = float(np.mean((distribution[:, 1] - labels) ** 2))
            else:
                onehot = np.zeros_like(distribution)
                onehot[np.arange(examples), labels] = 1.0
                scores[metric] = float(np.mean(((distribution - onehot) ** 2).sum(axis=1)))
        elif metric == "log_loss":
            scores[metric] = float(np.mean(-np.log(truth + LOG_LOSS_EPSILON)))
        elif metric == "accuracy":
            scores[metric] = float(np.mean(np.argmax(distribution, axis=1) == labels))
        elif metric == "auc":
            scores[metric] = _auc(distribution[:, 1], labels)
    return scores


def _check_leaks(held_out_ids: Sequence[Any] | None, training_ids: Mapping[str, Iterable[Any]] | None,
                 examples: int, policy_ids: Sequence[str]) -> bool:
    if training_ids is None:
        return False
    if held_out_ids is None:
        raise ValueError("training_ids requires held_out_ids")
    held_out = list(held_out_ids)
    if len(held_out) != examples or len(set(held_out)) != examples:
        raise ValueError(f"held_out_ids must name {examples} distinct held-out units")
    missing = sorted(set(policy_ids) - set(training_ids))
    if missing:
        raise ValueError(f"training_ids is missing policies: {missing}")
    held_out_set = set(held_out)
    leaks = {policy: len(held_out_set.intersection(training_ids[policy])) for policy in policy_ids}
    leaking = {policy: count for policy, count in leaks.items() if count}
    if leaking:
        raise PolicyScoringLeakError(f"policies were trained on held-out units: {leaking}")
    return True


def score_policies(
    predictions: Mapping[str, Any],
    labels: Any,
    *,
    metrics: Sequence[str] = DEFAULT_POLICY_SCORING_METRICS,
    split: str = "held_out",
    held_out_ids: Sequence[Any] | None = None,
    training_ids: Mapping[str, Iterable[Any]] | None = None,
) -> PolicyScoreReport:
    """Score each policy's predictions on the same held-out examples.

    ``predictions`` maps a policy id to a probability vector of shape ``(n,)``
    (the probability of class 1 in a binary task) or a row-stochastic matrix of
    shape ``(n, k)``. ``labels`` are integer class indices. When
    ``training_ids`` maps each policy to the units it trained on,
    ``held_out_ids`` must name the held-out units, and any overlap raises
    :class:`PolicyScoringLeakError` before a score is computed.
    """

    if not predictions:
        raise ValueError("predictions must name at least one policy")
    metrics = tuple(metrics)
    unknown = sorted(set(metrics) - set(POLICY_SCORING_METRICS))
    if unknown or not metrics or len(set(metrics)) != len(metrics):
        raise ValueError(f"metrics must be distinct names from {POLICY_SCORING_METRICS}; got {list(metrics)}")
    truth = _labels(labels)
    policy_ids = [str(policy) for policy in predictions]
    if len(set(policy_ids)) != len(policy_ids):
        raise ValueError("policy ids must be distinct")
    leak_checked = _check_leaks(held_out_ids, training_ids, truth.size, policy_ids)
    distributions = {str(policy): _distribution(values, truth.size, f"predictions[{policy!r}]") for policy, values in predictions.items()}
    classes = {d.shape[1] for d in distributions.values()}
    if len(classes) != 1:
        raise ValueError("every policy must predict the same number of classes")
    (count,) = classes
    if truth.min() < 0 or truth.max() >= count:
        raise ValueError(f"labels must lie in [0, {count})")
    if "auc" in metrics and count != 2:
        raise ValueError("auc is defined for binary tasks only")
    policies = tuple(PolicyScore(policy, _score(distributions[policy], truth, metrics)) for policy in policy_ids)
    mean_scores: dict[str, float | None] = {}
    for metric in metrics:
        values = [policy.scores[metric] for policy in policies]
        mean_scores[metric] = None if any(v is None for v in values) else float(np.mean(values))
    ensemble = np.mean(np.stack([distributions[policy] for policy in policy_ids]), axis=0)
    return PolicyScoreReport(
        split=str(split),
        examples=int(truth.size),
        classes=int(count),
        metrics=metrics,
        policies=policies,
        mean_scores=mean_scores,
        ensemble_scores=_score(ensemble, truth, metrics),
        leak_checked=leak_checked,
    )
