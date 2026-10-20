"""Prior-anchored residual learners (Decision 0017; needs the ``torch`` extra).

Each learner predicts the prior exactly until it sees data:

- ``ResidualLinear``: the prior logit plus a zero-started linear residual;
- ``ResidualMLP``: the prior logit plus zero-started linear and one-hidden-
  layer tanh residuals.

``train_residual_policy`` fits one by full-batch Adam on binary
cross-entropy. ``fit_residual_policy`` builds and fits either family.

Unlike the repository-only scripts they came from, these functions change no
global state:

- the MLP's initial weights come from a private ``torch.Generator``, with the
  draws of ``torch.nn.Linear``'s default initialization;
- thread counts and deterministic-algorithm settings are left to the caller.
  Bit-for-bit reproduction of a recorded run needs the settings it was
  recorded with, typically one thread on CPU, and deterministic algorithms
  with ``CUBLAS_WORKSPACE_CONFIG`` set on a GPU.

The torch-free primitives (the prior, keyed randomness, the exploration
floor, the outcome record, and boosted trees) are in
``neural_abm.policy_ladder``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from neural_abm.policy_ladder import LinearPrior, Predictor, PriorPredictor


RESIDUAL_FAMILIES = ("linear", "mlp")


def _prior_buffers(prior: LinearPrior) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(np.asarray(prior.weight, dtype=np.float64)).clone(),
        torch.tensor(float(prior.bias), dtype=torch.float64),
    )


def _uninitialized_linear(in_features: int, out_features: int) -> torch.nn.Linear:
    return torch.nn.utils.skip_init(torch.nn.Linear, in_features, out_features)


def _default_linear_init(layer: torch.nn.Linear, generator: torch.Generator) -> None:
    """``torch.nn.Linear.reset_parameters``, drawing from ``generator``."""

    with torch.no_grad():
        torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=generator)
        fan_in = layer.weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        torch.nn.init.uniform_(layer.bias, -bound, bound, generator=generator)


def _zero(layer: torch.nn.Linear) -> None:
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()


class ResidualLinear(torch.nn.Module):
    """The prior logit plus a zero-started linear residual."""

    def __init__(self, prior: LinearPrior) -> None:
        super().__init__()
        weight, bias = _prior_buffers(prior)
        self.register_buffer("base_weight", weight)
        self.register_buffer("base_bias", bias)
        self.linear_delta = _uninitialized_linear(weight.shape[0], 1)
        _zero(self.linear_delta)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base = features @ self.base_weight + self.base_bias
        return base + self.linear_delta(features).reshape(-1)


class ResidualMLP(torch.nn.Module):
    """The prior logit plus zero-started linear and nonlinear residuals.

    The hidden layer starts from ``torch.nn.Linear``'s default draws under
    ``seed``. The residual output layers start at zero, so the model predicts
    the prior until it is trained.
    """

    def __init__(self, prior: LinearPrior, hidden_width: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed))
        weight, bias = _prior_buffers(prior)
        self.register_buffer("base_weight", weight)
        self.register_buffer("base_bias", bias)
        width = weight.shape[0]
        self.linear_delta = _uninitialized_linear(width, 1)
        self.hidden = _uninitialized_linear(width, int(hidden_width))
        self.output = _uninitialized_linear(int(hidden_width), 1)
        for layer in (self.linear_delta, self.hidden, self.output):
            _default_linear_init(layer, generator)
        _zero(self.linear_delta)
        _zero(self.output)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base = features @ self.base_weight + self.base_bias
        linear = self.linear_delta(features).reshape(-1)
        nonlinear = self.output(torch.tanh(self.hidden(features))).reshape(-1)
        return base + linear + nonlinear


@dataclass
class TorchPredictor:
    """A trained residual policy, predicting probabilities."""

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
    """Full-batch Adam on binary cross-entropy, in float64."""

    inputs = torch.as_tensor(train_features, dtype=torch.float64, device=device)
    targets = torch.as_tensor(train_labels, dtype=torch.float64, device=device)
    policy = policy.to(dtype=torch.float64, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(policy(inputs), targets)
        loss.backward()
        optimizer.step()
    return TorchPredictor(policy, device)


def fit_residual_policy(
    family: str,
    prior: LinearPrior,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    weight_decay: float,
    hidden_width: int | None = None,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> Predictor:
    """Fit one prior-anchored residual learner. With no rows it is the prior.

    ``family`` is ``"linear"`` or ``"mlp"``; the MLP needs ``hidden_width``
    and ``seed``.
    """

    if family not in RESIDUAL_FAMILIES:
        raise ValueError(f"unknown residual family: {family}")
    if len(train_labels) == 0:
        return PriorPredictor(prior)
    if family == "linear":
        policy: torch.nn.Module = ResidualLinear(prior)
    else:
        if hidden_width is None or seed is None:
            raise ValueError("the MLP needs hidden_width and seed")
        policy = ResidualMLP(prior, hidden_width, seed)
    return train_residual_policy(
        policy, train_features, train_labels,
        learning_rate=learning_rate, epochs=epochs, weight_decay=weight_decay, device=device,
    )
