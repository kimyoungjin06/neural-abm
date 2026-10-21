"""Torch-free policy-ladder primitives (Decision 0017).

Two model families compare fixed rules, learned policies, and external priors
in one simulation lifecycle with these semantics: the repository-only
researcher-pivot ladder and the information-spread ladder. Both reproduce
their recorded runs byte for byte through them.

- ``keyed_seed`` and ``keyed_uniform``: randomness that depends only on a
  tuple of integers, so arms that share a key share their draws.
- ``ExplorationFloor``: take a declined action with probability epsilon on a
  keyed draw, independent of the prior.
- ``LinearPrior``: a rule written as ``logit = features @ weight + bias``.
  Learners start from it; evaluating a domain's own fixed rule stays with the
  domain, because a rule computed elsewhere can differ in the last bits.
- ``OutcomeRecord``: outcomes that become visible from a step, with pooled or
  per-agent views. Rows come back ordered by the step from which they are
  visible, then by insertion.
- ``fit_prior_boosted_trees``: Newton-boosted regression trees from the prior
  logit. scikit-learn is imported on first use (the ``research`` extra).

The torch-backed residual learners are in ``neural_abm.policy_learners``.
Worlds, measures, decision rules, and claims stay with the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


LOGIT_CLIP = 40.0


# -- keyed randomness ------------------------------------------------------


def keyed_seed(*components: int) -> int:
    """One integer seed for a tuple of non-negative integer components."""

    return int(np.random.SeedSequence(tuple(int(c) for c in components)).generate_state(1)[0])


def keyed_uniform(components: Sequence[int], count: int) -> np.ndarray:
    """``count`` uniforms in [0, 1) that depend only on ``components``."""

    sequence = np.random.SeedSequence(tuple(int(c) for c in components))
    return np.random.default_rng(sequence).random(count)


@dataclass(frozen=True)
class ExplorationFloor:
    """Take an action the policy declined, with probability ``epsilon``.

    Pass draws from their own keyed stream, so that changing epsilon perturbs
    no other randomness. At epsilon 0 the floor never acts.
    """

    epsilon: float

    def explores(self, uniforms: np.ndarray) -> np.ndarray:
        uniforms = np.asarray(uniforms)
        if self.epsilon <= 0.0:
            return np.zeros(uniforms.shape, dtype=bool)
        return uniforms < self.epsilon


# -- the prior -------------------------------------------------------------


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """The logistic function, with logits clipped to +/- 40."""

    clipped = np.clip(logits, -LOGIT_CLIP, LOGIT_CLIP)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class LinearPrior:
    """A rule written as ``logit = features @ weight + bias``."""

    weight: np.ndarray
    bias: float

    def logit(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weight + float(self.bias)

    def probability(self, features: np.ndarray) -> np.ndarray:
        return sigmoid(self.logit(features))


class Predictor(Protocol):
    """Anything that maps feature rows to probabilities."""

    def predict(self, features: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class PriorPredictor:
    """The prior itself, as a learner with no rows predicts."""

    prior: LinearPrior

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.prior.probability(features)


# -- the outcome record ----------------------------------------------------


@dataclass
class OutcomeRecord:
    """Outcomes, each visible from a step, with pooled and per-agent views.

    Each row carries an integer ``kind``. The default view returns kind 1,
    the outcomes of actions taken. A caller that also keeps other rows (for
    example, outcomes of declined actions) tags them with another kind and
    asks for them explicitly.

    **Ordering rule.** ``visible`` returns rows ordered by the step from
    which they are visible, then by insertion. A caller that adds outcomes
    when they are released gets them back in release order. So does one that
    adds them earlier with a later ``visible_from``.
    """

    width: int
    features: list[np.ndarray] = field(default_factory=list)
    labels: list[float] = field(default_factory=list)
    agents: list[int] = field(default_factory=list)
    visible_from: list[int] = field(default_factory=list)
    kinds: list[int] = field(default_factory=list)

    def add(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        agents: np.ndarray,
        visible_from: int,
        kinds: np.ndarray | None = None,
    ) -> None:
        if kinds is None:
            kinds = np.ones(len(labels), dtype=np.int64)
        for row, label, agent, kind in zip(features, labels, agents, kinds, strict=True):
            row = np.asarray(row, dtype=np.float64)
            if row.shape != (self.width,):
                raise ValueError(f"an outcome row must have {self.width} features")
            self.features.append(row)
            self.labels.append(float(label))
            self.agents.append(int(agent))
            self.visible_from.append(int(visible_from))
            self.kinds.append(int(kind))

    def visible(
        self, step: int, since: int | None = None, kinds: tuple[int, ...] | None = (1,)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Features, labels, and agents of the rows visible at ``step``.

        With ``since``, only rows that became visible at ``since`` or later (a
        forgetting window). ``kinds=None`` returns every kind.
        """

        visible_from = np.asarray(self.visible_from, dtype=np.int64)
        mask = visible_from <= step
        if since is not None:
            mask &= visible_from >= since
        if kinds is not None:
            mask &= np.isin(np.asarray(self.kinds, dtype=np.int64), kinds)
        if not mask.any():
            return np.zeros((0, self.width)), np.zeros(0), np.zeros(0, dtype=np.int64)
        rows = np.flatnonzero(mask)
        rows = rows[np.argsort(visible_from[rows], kind="stable")]
        return (
            np.stack([self.features[i] for i in rows]),
            np.asarray(self.labels, dtype=np.float64)[rows],
            np.asarray(self.agents, dtype=np.int64)[rows],
        )

    def visible_counts_by_agent(self, step: int, agents: int) -> np.ndarray:
        """How many kind-1 outcomes each agent has visible at ``step``."""

        _features, _labels, owners = self.visible(step)
        return np.bincount(owners, minlength=agents)


# -- prior-boosted trees ---------------------------------------------------


@dataclass
class BoostedTreesPredictor:
    """The prior logit plus the boosted trees' leaf values."""

    prior: LinearPrior
    learning_rate: float
    rounds: list[tuple[Any, np.ndarray]] = field(default_factory=list)

    def predict(self, features: np.ndarray) -> np.ndarray:
        margin = self.prior.logit(features)
        for tree, values in self.rounds:
            margin = margin + self.learning_rate * values[tree.apply(features)]
        return sigmoid(margin)


def fit_prior_boosted_trees(
    prior: LinearPrior,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    settings: Mapping[str, Any],
    seed: int,
) -> BoostedTreesPredictor:
    """Newton-boosted regression trees starting from the prior logit.

    ``settings`` holds ``rounds``, ``learning_rate``, ``leaf_l2``,
    ``max_depth``, and ``min_samples_leaf``. With no rows the result predicts
    the prior.
    """

    from sklearn.tree import DecisionTreeRegressor  # noqa: PLC0415

    learning_rate = float(settings["learning_rate"])
    predictor = BoostedTreesPredictor(prior, learning_rate)
    if len(train_labels) == 0:
        return predictor
    margin = prior.logit(train_features)
    leaf_l2 = float(settings["leaf_l2"])
    for _round in range(int(settings["rounds"])):
        probability = sigmoid(margin)
        gradient = train_labels - probability
        hessian = np.maximum(probability * (1.0 - probability), 1e-6)
        tree = DecisionTreeRegressor(
            max_depth=int(settings["max_depth"]),
            min_samples_leaf=int(settings["min_samples_leaf"]),
            random_state=seed,
        )
        tree.fit(train_features, gradient / hessian, sample_weight=hessian)
        train_leaves = tree.apply(train_features)
        node_count = tree.tree_.node_count
        values = np.bincount(train_leaves, weights=gradient, minlength=node_count) / (
            np.bincount(train_leaves, weights=hessian, minlength=node_count) + leaf_l2
        )
        margin = margin + learning_rate * values[train_leaves]
        predictor.rounds.append((tree, values))
    return predictor
