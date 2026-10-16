"""Domain-free policy-ladder semantics, repository-only (N14 candidate v1).

The Pivot policy ladder hard-codes its rule as the prior every learner starts
from. This module lifts the same semantics so a second domain can use them,
with the prior passed in as a linear rule:

- prior-anchored learners: a residual linear model, a residual MLP, and
  prior-boosted trees. Each predicts the prior exactly until it sees data.
- per-agent residual learners trained together as one batch, with no
  coupling between agents;
- keyed randomness and a prior-independent exploration floor;
- an outcome record with a visibility delay and pooled or per-agent views;
- policy snapshots, so a learned trajectory can be replayed as a table.

The single-learner fitters reproduce the Pivot implementations operation for
operation, so a Pivot prior gives byte-identical predictions. The parity test
in tests/test_policy_ladder_core_v1.py checks this.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor


# -- keyed randomness ------------------------------------------------------


def keyed_seed(*components: int) -> int:
    """One integer seed for a tuple of non-negative integer components."""

    return int(np.random.SeedSequence(tuple(int(c) for c in components)).generate_state(1)[0])


def keyed_uniform(components: Sequence[int], count: int) -> np.ndarray:
    """`count` uniforms that depend only on `components`."""

    sequence = np.random.SeedSequence(tuple(int(c) for c in components))
    return np.random.default_rng(sequence).random(count)


@dataclass(frozen=True)
class ExplorationFloor:
    """Take an action the policy declined, with probability epsilon.

    The draw comes from its own keyed stream, so changing epsilon perturbs no
    other randomness. At epsilon 0 the floor never acts.
    """

    epsilon: float

    def explores(self, uniforms: np.ndarray) -> np.ndarray:
        if self.epsilon <= 0.0:
            return np.zeros(uniforms.shape, dtype=bool)
        return uniforms < self.epsilon


# -- priors and predictors -------------------------------------------------


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class LinearPrior:
    """A rule written as logit = features @ weight + bias."""

    weight: np.ndarray
    bias: float

    def logit(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weight + float(self.bias)

    def probability(self, features: np.ndarray) -> np.ndarray:
        return sigmoid(self.logit(features))

    def torch_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(np.asarray(self.weight, dtype=np.float64)).clone(),
            torch.tensor(float(self.bias), dtype=torch.float64),
        )


class Predictor(Protocol):
    def predict(self, features: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class PriorPredictor:
    prior: LinearPrior

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.prior.probability(features)


# -- single learners (Pivot-equivalent) ------------------------------------


class ResidualLinear(torch.nn.Module):
    """Prior logit plus a zero-started linear residual."""

    def __init__(self, prior: LinearPrior) -> None:
        super().__init__()
        weight, bias = prior.torch_buffers()
        self.register_buffer("base_weight", weight)
        self.register_buffer("base_bias", bias)
        self.linear_delta = torch.nn.Linear(weight.shape[0], 1)
        torch.nn.init.zeros_(self.linear_delta.weight)
        torch.nn.init.zeros_(self.linear_delta.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base = features @ self.base_weight + self.base_bias
        return base + self.linear_delta(features).reshape(-1)


class ResidualMLP(torch.nn.Module):
    """Prior logit plus zero-started linear and nonlinear residuals."""

    def __init__(self, prior: LinearPrior, hidden_width: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        weight, bias = prior.torch_buffers()
        self.register_buffer("base_weight", weight)
        self.register_buffer("base_bias", bias)
        width = weight.shape[0]
        self.linear_delta = torch.nn.Linear(width, 1)
        self.hidden = torch.nn.Linear(width, hidden_width)
        self.output = torch.nn.Linear(hidden_width, 1)
        torch.nn.init.zeros_(self.linear_delta.weight)
        torch.nn.init.zeros_(self.linear_delta.bias)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base = features @ self.base_weight + self.base_bias
        linear = self.linear_delta(features).reshape(-1)
        nonlinear = self.output(torch.tanh(self.hidden(features))).reshape(-1)
        return base + linear + nonlinear


@dataclass
class TorchPredictor:
    policy: torch.nn.Module
    device: torch.device | str = "cpu"

    def predict(self, features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            evaluation = torch.as_tensor(features, dtype=torch.float64, device=self.device)
            return torch.sigmoid(self.policy(evaluation)).detach().cpu().numpy()


def train_residual_policy(
    policy: torch.nn.Module,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    weight_decay: float,
    device: torch.device | str = "cpu",
) -> TorchPredictor:
    """Full-batch Adam on binary cross-entropy, as in the Pivot ladder.

    On a GPU, set CUBLAS_WORKSPACE_CONFIG before CUDA starts, because
    deterministic algorithms are enforced.
    """

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    inputs = torch.as_tensor(train_features, dtype=torch.float64, device=device)
    targets = torch.as_tensor(train_labels, dtype=torch.float64, device=device)
    policy = policy.to(dtype=torch.float64, device=device)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            policy(inputs), targets
        )
        loss.backward()
        optimizer.step()
    return TorchPredictor(policy, device)


@dataclass
class BoostedTreesPredictor:
    prior: LinearPrior
    learning_rate: float
    rounds: list[tuple[DecisionTreeRegressor, np.ndarray]] = field(default_factory=list)

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
    """Newton-boosted regression trees starting from the prior logit."""

    margin = prior.logit(train_features)
    learning_rate = float(settings["learning_rate"])
    leaf_l2 = float(settings["leaf_l2"])
    predictor = BoostedTreesPredictor(prior, learning_rate)
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


FAMILIES = ("linear", "mlp", "tree", "linear_exact")


@dataclass
class LinearResidualPredictor:
    """Prior logit plus a fitted linear residual."""

    prior: LinearPrior
    weight: np.ndarray
    bias: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        return sigmoid(self.prior.logit(features) + features @ self.weight + self.bias)


def _ridge_logistic_newton(
    offset: np.ndarray, features: np.ndarray, labels: np.ndarray, l2: float,
    max_iter: int = 100, tol: float = 1e-10,
) -> np.ndarray:
    """Minimize mean cross-entropy of offset + [x, 1] @ theta, plus (l2/2)|theta|^2.

    Newton steps with step halving. The objective is strictly convex, so the
    solution is unique, and it has no step cap.
    """

    design = np.hstack([features, np.ones((len(features), 1))])
    theta = np.zeros(design.shape[1])
    n = float(len(labels))

    def objective(t: np.ndarray) -> float:
        z = offset + design @ t
        return float(np.mean(np.logaddexp(0.0, z) - labels * z) + 0.5 * l2 * t @ t)

    current = objective(theta)
    for _ in range(max_iter):
        p = sigmoid(offset + design @ theta)
        gradient = design.T @ (p - labels) / n + l2 * theta
        hessian = (design * (p * (1.0 - p))[:, None]).T @ design / n + l2 * np.eye(len(theta))
        step = np.linalg.solve(hessian, gradient)
        scale = 1.0
        while scale > 1e-8:
            candidate = theta - scale * step
            value = objective(candidate)
            if value <= current + 1e-15:
                break
            scale *= 0.5
        theta, current = candidate, value
        if np.max(np.abs(scale * step)) < tol:
            break
    return theta


def fit_exact_linear(
    prior: LinearPrior, train_features: np.ndarray, train_labels: np.ndarray, l2: float,
) -> Predictor:
    """A linear residual on the prior, fitted to convergence (no step cap)."""

    if len(train_labels) == 0:
        return PriorPredictor(prior)
    theta = _ridge_logistic_newton(prior.logit(train_features), train_features, train_labels, l2)
    return LinearResidualPredictor(prior, theta[:-1], float(theta[-1]))


def fit_batched_exact_linear(
    base_logits: np.ndarray,
    agent_ids: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    agents: int,
    l2: float,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> "BatchedResidualState":
    """Each agent's linear residual on `base_logits`, fitted to convergence.

    Per agent: mean cross-entropy over its own rows plus (l2/2)|theta|^2,
    solved by batched Newton steps with a common step-halving factor. Agents
    are independent; an agent with no rows keeps a zero residual.
    """

    width = train_features.shape[1] + 1
    theta = np.zeros((agents, width))
    if len(train_labels) == 0:
        return _batched_state_from_theta(theta)
    design = np.hstack([train_features, np.ones((len(train_features), 1))])
    counts = np.bincount(agent_ids, minlength=agents).astype(np.float64)
    active = counts > 0
    weight_per_row = 1.0 / counts[agent_ids]
    eye = np.eye(width)

    def objective(t: np.ndarray) -> np.ndarray:
        z = base_logits + (design * t[agent_ids]).sum(axis=1)
        per_row = (np.logaddexp(0.0, z) - train_labels * z) * weight_per_row
        return np.bincount(agent_ids, weights=per_row, minlength=agents) + 0.5 * l2 * (t * t).sum(axis=1)

    current = objective(theta)
    for _ in range(max_iter):
        z = base_logits + (design * theta[agent_ids]).sum(axis=1)
        p = sigmoid(z)
        gradient = np.zeros((agents, width))
        np.add.at(gradient, agent_ids, design * ((p - train_labels) * weight_per_row)[:, None])
        gradient += l2 * theta
        hessian = np.zeros((agents, width, width))
        outer = design[:, :, None] * design[:, None, :] * (p * (1.0 - p) * weight_per_row)[:, None, None]
        np.add.at(hessian, agent_ids, outer)
        hessian += l2 * eye
        step = np.linalg.solve(hessian, gradient[:, :, None])[:, :, 0]
        step[~active] = 0.0
        scale = np.ones(agents)
        candidate = theta - step
        value = objective(candidate)
        worse = value > current + 1e-15
        while worse.any() and scale.min() > 1e-8:
            scale[worse] *= 0.5
            candidate = theta - scale[:, None] * step
            value = objective(candidate)
            worse = value > current + 1e-15
        theta, current = candidate, value
        if np.max(np.abs(scale[:, None] * step)) < tol:
            break
    return _batched_state_from_theta(theta)


def _batched_state_from_theta(theta: np.ndarray) -> "BatchedResidualState":
    return BatchedResidualState(
        delta_weight=torch.as_tensor(np.ascontiguousarray(theta[:, :-1])),
        delta_bias=torch.as_tensor(np.ascontiguousarray(theta[:, -1])),
    )


def fit_residual(
    family: str,
    prior: LinearPrior,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    settings: Mapping[str, Any],
    seed: int,
    device: torch.device | str = "cpu",
) -> Predictor:
    """Fit one prior-anchored learner. With no rows it is the prior."""

    if len(train_labels) == 0:
        return PriorPredictor(prior)
    if family == "tree":
        return fit_prior_boosted_trees(prior, train_features, train_labels, settings, seed)
    if family == "linear_exact":
        return fit_exact_linear(prior, train_features, train_labels, float(settings["l2"]))
    if family == "linear":
        policy: torch.nn.Module = ResidualLinear(prior)
    elif family == "mlp":
        policy = ResidualMLP(prior, int(settings["hidden_width"]), seed)
    else:
        raise ValueError(f"unknown learner family: {family}")
    return train_residual_policy(
        policy,
        train_features,
        train_labels,
        learning_rate=float(settings["learning_rate"]),
        epochs=int(settings["epochs"]),
        weight_decay=float(settings["weight_decay"]),
        device=device,
    )


# -- per-agent learners, trained as one batch ------------------------------


@dataclass
class BatchedResidualState:
    """Per-agent residual parameters; row a belongs to agent a."""

    delta_weight: torch.Tensor  # [agents, features]
    delta_bias: torch.Tensor  # [agents]
    hidden_weight: torch.Tensor | None = None  # [agents, features, width]
    hidden_bias: torch.Tensor | None = None  # [agents, width]
    output_weight: torch.Tensor | None = None  # [agents, width]
    output_bias: torch.Tensor | None = None  # [agents]

    def tensors(self) -> list[torch.Tensor]:
        return [t for t in (
            self.delta_weight, self.delta_bias, self.hidden_weight,
            self.hidden_bias, self.output_weight, self.output_bias,
        ) if t is not None]

    def detached(self) -> "BatchedResidualState":
        return BatchedResidualState(*(
            None if t is None else t.detach().clone()
            for t in (self.delta_weight, self.delta_bias, self.hidden_weight,
                      self.hidden_bias, self.output_weight, self.output_bias)
        ))


def initial_batched_state(
    family: str, agents: int, features: int, hidden_width: int, seed: int,
    device: torch.device | str = "cpu",
) -> BatchedResidualState:
    """Zero residuals; the hidden layer of each agent starts from its own key."""

    zeros = dict(dtype=torch.float64, device=device)
    state = BatchedResidualState(
        delta_weight=torch.zeros((agents, features), **zeros),
        delta_bias=torch.zeros(agents, **zeros),
    )
    if family == "linear":
        return state
    if family != "mlp":
        raise ValueError(f"unknown batched family: {family}")
    bound = 1.0 / np.sqrt(features)
    rng = np.random.default_rng(np.random.SeedSequence((int(seed), agents, features, hidden_width)))
    state.hidden_weight = torch.as_tensor(
        rng.uniform(-bound, bound, size=(agents, features, hidden_width)), device=device
    )
    state.hidden_bias = torch.as_tensor(
        rng.uniform(-bound, bound, size=(agents, hidden_width)), device=device
    )
    state.output_weight = torch.zeros((agents, hidden_width), **zeros)
    state.output_bias = torch.zeros(agents, **zeros)
    return state


def batched_residual_logits(
    state: BatchedResidualState, agent_ids: torch.Tensor, features: torch.Tensor
) -> torch.Tensor:
    """Residual logit of each row for the agent that owns it."""

    linear = (features * state.delta_weight[agent_ids]).sum(-1) + state.delta_bias[agent_ids]
    if state.hidden_weight is None:
        return linear
    hidden = torch.tanh(
        torch.einsum("rf,rfh->rh", features, state.hidden_weight[agent_ids])
        + state.hidden_bias[agent_ids]
    )
    return linear + (hidden * state.output_weight[agent_ids]).sum(-1) + state.output_bias[agent_ids]


def fit_batched_residual(
    family: str,
    prior: LinearPrior,
    agent_ids: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    agents: int,
    settings: Mapping[str, Any],
    seed: int,
    device: torch.device | str = "cpu",
    base_logits: np.ndarray | None = None,
) -> BatchedResidualState:
    """Fit every agent's residual on its own rows, all in one Adam run.

    `base_logits` replaces the prior logit of each training row, so the
    residuals sit on top of another model, e.g. a pooled learner (partial
    pooling). By default the base is the prior.

    The loss is the sum over agents of each agent's mean cross-entropy. Adam
    and weight decay act elementwise, so each agent's parameters move only
    with its own rows. An agent with no rows keeps the prior exactly.
    """

    features = int(train_features.shape[1]) if train_features.ndim == 2 else len(prior.weight)
    state = initial_batched_state(
        family, agents, features, int(settings.get("hidden_width", 0)), seed, device
    )
    if len(train_labels) == 0:
        return state
    torch.use_deterministic_algorithms(True)
    ids = torch.as_tensor(agent_ids, dtype=torch.long, device=device)
    inputs = torch.as_tensor(train_features, dtype=torch.float64, device=device)
    targets = torch.as_tensor(train_labels, dtype=torch.float64, device=device)
    base_values = prior.logit(train_features) if base_logits is None else base_logits
    base = torch.as_tensor(base_values, dtype=torch.float64, device=device)
    counts = torch.bincount(ids, minlength=agents).to(torch.float64)
    row_weight = 1.0 / counts[ids]
    parameters = state.tensors()
    for tensor in parameters:
        tensor.requires_grad_(True)
    optimizer = torch.optim.Adam(
        parameters, lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    for _epoch in range(int(settings["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        logits = base + batched_residual_logits(state, ids, inputs)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        (losses * row_weight).sum().backward()
        optimizer.step()
    return state.detached()


def probability_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return np.log(clipped) - np.log1p(-clipped)


@dataclass
class BatchedPredictor:
    """Per-agent residuals on the prior, or on a shared `base` predictor."""

    prior: LinearPrior
    state: BatchedResidualState
    base: Predictor | None = None

    def base_logit(self, features: np.ndarray) -> np.ndarray:
        if self.base is None:
            return self.prior.logit(features)
        return probability_logit(self.base.predict(features))

    def predict_for(self, agent_ids: np.ndarray, features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            device = self.state.delta_weight.device
            ids = torch.as_tensor(agent_ids, dtype=torch.long, device=device)
            inputs = torch.as_tensor(features, dtype=torch.float64, device=device)
            residual = batched_residual_logits(self.state, ids, inputs).cpu().numpy()
        return sigmoid(self.base_logit(features) + residual)


# -- outcome record and policy snapshots -----------------------------------


@dataclass
class OutcomeRecord:
    """Outcomes with the step from which each becomes visible."""

    width: int
    features: list[np.ndarray] = field(default_factory=list)
    labels: list[float] = field(default_factory=list)
    agents: list[int] = field(default_factory=list)
    visible_from: list[int] = field(default_factory=list)
    # 1 for an outcome of an action taken (the usual case); 0 for an outcome
    # of an action declined, kept only by label-selection diagnostics.
    kinds: list[int] = field(default_factory=list)

    def add(
        self, features: np.ndarray, labels: np.ndarray, agents: np.ndarray,
        visible_from: int, kinds: np.ndarray | None = None,
    ) -> None:
        if kinds is None:
            kinds = np.ones(len(labels), dtype=np.int64)
        for row, label, agent, kind in zip(features, labels, agents, kinds, strict=True):
            self.features.append(np.asarray(row, dtype=np.float64))
            self.labels.append(float(label))
            self.agents.append(int(agent))
            self.visible_from.append(int(visible_from))
            self.kinds.append(int(kind))

    def visible(
        self, step: int, since: int | None = None, kinds: tuple[int, ...] | None = (1,)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows visible at `step`; with `since`, only those that became
        visible at `since` or later (a forgetting window). By default only
        outcomes of actions taken; `kinds=None` returns every kind."""

        visible_from = np.asarray(self.visible_from, dtype=np.int64)
        mask = visible_from <= step
        if since is not None:
            mask &= visible_from >= since
        if kinds is not None:
            mask &= np.isin(np.asarray(self.kinds, dtype=np.int64), kinds)
        if not mask.any():
            return np.zeros((0, self.width)), np.zeros(0), np.zeros(0, dtype=np.int64)
        rows = np.flatnonzero(mask)
        return (
            np.stack([self.features[i] for i in rows]),
            np.asarray(self.labels, dtype=np.float64)[rows],
            np.asarray(self.agents, dtype=np.int64)[rows],
        )

    def visible_counts_by_agent(self, step: int, agents: int) -> np.ndarray:
        _features, _labels, owners = self.visible(step)
        return np.bincount(owners, minlength=agents)


@dataclass
class PolicySnapshots:
    """The policy in force from each step on, for replay as a table."""

    steps: list[int] = field(default_factory=list)
    policies: list[Any] = field(default_factory=list)

    def record(self, step: int, policy: Any) -> None:
        if self.steps and step <= self.steps[-1]:
            raise ValueError("snapshots must be recorded in increasing step order")
        self.steps.append(int(step))
        self.policies.append(policy)

    def active(self, step: int) -> Any:
        position = int(np.searchsorted(np.asarray(self.steps), step, side="right")) - 1
        if position < 0:
            raise ValueError(f"no policy recorded at or before step {step}")
        return self.policies[position]
