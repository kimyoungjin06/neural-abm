"""Prototype binary revision-operator lifecycle for Neural ABM units."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from neural_abm.losses import LossVector, loss_values_at, mean_loss_value


REVISION_STAY = 0
REVISION_SWITCH_TO_ONE = 1
REVISION_SWITCH_TO_ZERO = 2
BINARY_REVISION_CHOICE_NAMES = (
    "stay",
    "switch_to_1",
    "switch_to_0",
)

StateArray = np.ndarray | torch.Tensor


class BinaryRevisionSignalBuilder(Protocol):
    """Build domain-owned signals consumed by a binary revision operator."""

    def __call__(
        self,
        agents: list[Any],
        observations: Any,
        current_actions: np.ndarray,
    ) -> Mapping[str, Any]: ...


class BinaryRevisionReadout(Protocol):
    """Collect per-agent stay/switch probabilities from revision signals."""

    def __call__(
        self,
        agents: list[Any],
        observations: Any,
        current_actions: np.ndarray,
        revision_signals: Mapping[str, Any],
        *,
        temperature: float,
    ) -> Any: ...


class BinaryRevisionSampler(Protocol):
    """Sample stay/switch revision choices from operator probabilities."""

    def __call__(
        self,
        revision_probs: Any,
        current_actions: np.ndarray,
    ) -> StateArray: ...


class BinaryRevisionLocalUpdateCommit(Protocol):
    """Commit local learning after revision choices have selected actions."""

    def __call__(self, actions: np.ndarray) -> LossVector: ...


class BinaryRevisionCacheRefresh(Protocol):
    """Refresh agent/cache state after local revision learning commits."""

    def __call__(self, agents: list[Any]) -> None: ...


@dataclass(frozen=True)
class BinaryRevisionLearningCallbacks:
    """Domain callbacks for the binary revision-operator prototype lifecycle."""

    collect_revision_probs: BinaryRevisionReadout
    sample_revision_choices: BinaryRevisionSampler
    local_update: BinaryRevisionLocalUpdateCommit
    collect_revision_signals: BinaryRevisionSignalBuilder | None = None
    refresh_revision_cache: BinaryRevisionCacheRefresh | None = None
    post_collect_revision_probs: BinaryRevisionReadout | None = None
    post_collect_revision_signals: BinaryRevisionSignalBuilder | None = None


@dataclass
class BinaryRevisionLearningResult:
    """Result of one binary revision readout, action revision, and local update."""

    current_actions: np.ndarray
    pre_revision_probs: Any
    revision_choices: np.ndarray
    actions_after_revision: np.ndarray
    revision_mask: np.ndarray
    local_losses: LossVector
    post_local_revision_probs: Any
    revision_signals: Mapping[str, Any] = field(default_factory=dict)
    post_local_revision_signals: Mapping[str, Any] = field(default_factory=dict)
    extras: MutableMapping[str, Any] = field(default_factory=dict)

    @property
    def agent_count(self) -> int:
        return int(len(self.current_actions))

    @property
    def realized_revision_rate(self) -> float:
        if self.agent_count == 0:
            return 0.0
        return float(np.mean(self.revision_mask))

    def aggregate_row(self) -> dict[str, Any]:
        probs = _revision_probs_to_numpy(
            normalize_binary_revision_probabilities(self.pre_revision_probs)
        )
        switch_probs = binary_revision_switch_probabilities(
            probs,
            self.current_actions,
        )
        effective_stay_probs = binary_revision_effective_stay_probabilities(
            probs,
            self.current_actions,
        )
        return {
            "mean_revision_stay_probability": _mean_or_zero(probs[:, 0]),
            "mean_revision_effective_stay_probability": _mean_or_zero(
                effective_stay_probs
            ),
            "mean_revision_switch_probability": _mean_or_zero(switch_probs),
            "mean_revision_switch_to_one_probability": _mean_or_zero(probs[:, 1]),
            "mean_revision_switch_to_zero_probability": _mean_or_zero(probs[:, 2]),
            "revision_choice_stay_rate": _choice_rate(
                self.revision_choices,
                REVISION_STAY,
            ),
            "revision_choice_switch_to_one_rate": _choice_rate(
                self.revision_choices,
                REVISION_SWITCH_TO_ONE,
            ),
            "revision_choice_switch_to_zero_rate": _choice_rate(
                self.revision_choices,
                REVISION_SWITCH_TO_ZERO,
            ),
            "realized_revision_rate": self.realized_revision_rate,
            "action_rate_after_revision": _mean_or_zero(self.actions_after_revision),
            "mean_revision_local_loss": mean_loss_value(self.local_losses),
        }

    def micro_rows(self) -> list[dict[str, Any]]:
        probs = _revision_probs_to_numpy(
            normalize_binary_revision_probabilities(self.pre_revision_probs)
        )
        switch_probs = binary_revision_switch_probabilities(
            probs,
            self.current_actions,
        )
        effective_stay_probs = binary_revision_effective_stay_probabilities(
            probs,
            self.current_actions,
        )
        loss_values = loss_values_at(self.local_losses, range(self.agent_count))
        rows: list[dict[str, Any]] = []
        for agent_id in range(self.agent_count):
            rows.append(
                {
                    "agent_id": agent_id,
                    "previous_action": int(self.current_actions[agent_id]),
                    "action": int(self.actions_after_revision[agent_id]),
                    "revision_choice": BINARY_REVISION_CHOICE_NAMES[
                        int(self.revision_choices[agent_id])
                    ],
                    "revised": bool(self.revision_mask[agent_id]),
                    "revision_stay_probability": float(probs[agent_id, 0]),
                    "revision_effective_stay_probability": float(
                        effective_stay_probs[agent_id]
                    ),
                    "revision_switch_probability": float(switch_probs[agent_id]),
                    "revision_switch_to_one_probability": float(probs[agent_id, 1]),
                    "revision_switch_to_zero_probability": float(probs[agent_id, 2]),
                    "local_loss": loss_values[agent_id],
                }
            )
        return rows


@dataclass
class BinaryRevisionLearningUnit:
    """Reusable lifecycle for a binary neural revision operator.

    Domain adapters own the meaning of revision signals. This unit only owns the
    sequence: signal build -> revision readout -> stay/switch selection -> local
    update -> cache refresh -> post-local readout.
    """

    agents: Sequence[Any]
    observations: Any
    current_actions: StateArray
    temperature: float
    callbacks: BinaryRevisionLearningCallbacks
    extras: MutableMapping[str, Any] = field(default_factory=dict)

    def run(self) -> BinaryRevisionLearningResult:
        agents = list(self.agents)
        current_actions = validate_binary_actions(self.current_actions)
        revision_signals = self._collect_revision_signals(
            agents=agents,
            observations=self.observations,
            current_actions=current_actions,
            post_local=False,
        )
        pre_revision_probs = normalize_binary_revision_probabilities(
            self.callbacks.collect_revision_probs(
                agents,
                self.observations,
                current_actions,
                revision_signals,
                temperature=self.temperature,
            )
        )
        choices = validate_binary_revision_choices(
            self.callbacks.sample_revision_choices(
                pre_revision_probs,
                current_actions,
            ),
            expected_size=len(current_actions),
        )
        actions_after_revision = apply_binary_revision_choices(
            current_actions,
            choices,
        )
        revision_mask = actions_after_revision != current_actions
        local_losses = self.callbacks.local_update(actions_after_revision)
        if self.callbacks.refresh_revision_cache is not None:
            self.callbacks.refresh_revision_cache(agents)
        post_signals = self._collect_revision_signals(
            agents=agents,
            observations=self.observations,
            current_actions=actions_after_revision,
            post_local=True,
        )
        post_readout = (
            self.callbacks.post_collect_revision_probs
            or self.callbacks.collect_revision_probs
        )
        post_local_revision_probs = normalize_binary_revision_probabilities(
            post_readout(
                agents,
                self.observations,
                actions_after_revision,
                post_signals,
                temperature=self.temperature,
            )
        )
        return BinaryRevisionLearningResult(
            current_actions=current_actions,
            pre_revision_probs=pre_revision_probs,
            revision_choices=choices,
            actions_after_revision=actions_after_revision,
            revision_mask=revision_mask,
            local_losses=local_losses,
            post_local_revision_probs=post_local_revision_probs,
            revision_signals=dict(revision_signals),
            post_local_revision_signals=dict(post_signals),
            extras=dict(self.extras),
        )

    def _collect_revision_signals(
        self,
        *,
        agents: list[Any],
        observations: Any,
        current_actions: np.ndarray,
        post_local: bool,
    ) -> Mapping[str, Any]:
        builder = (
            self.callbacks.post_collect_revision_signals
            if post_local and self.callbacks.post_collect_revision_signals is not None
            else self.callbacks.collect_revision_signals
        )
        if builder is None:
            return {}
        signals = builder(agents, observations, current_actions)
        return dict(signals)


def validate_binary_actions(actions: StateArray) -> np.ndarray:
    """Return validated binary actions as an integer NumPy vector."""

    array = _state_array_to_numpy(actions).astype(np.int64, copy=False)
    if array.ndim != 1:
        raise ValueError("binary revision actions must be a 1D vector")
    if not np.isin(array, [0, 1]).all():
        raise ValueError("binary revision actions must contain only 0 or 1")
    return array.copy()


def validate_binary_revision_choices(
    choices: StateArray,
    *,
    expected_size: int,
) -> np.ndarray:
    """Return validated stay/switch choices as an integer NumPy vector."""

    array = _state_array_to_numpy(choices).astype(np.int64, copy=False)
    if array.ndim != 1:
        raise ValueError("binary revision choices must be a 1D vector")
    if len(array) != expected_size:
        raise ValueError("binary revision choices must match action count")
    if not np.isin(
        array,
        [REVISION_STAY, REVISION_SWITCH_TO_ONE, REVISION_SWITCH_TO_ZERO],
    ).all():
        raise ValueError("binary revision choices must be stay/switch_to_1/switch_to_0")
    return array.copy()


def normalize_binary_revision_probabilities(values: Any) -> Any:
    """Validate and row-normalize stay/switch_to_1/switch_to_0 probabilities."""

    if isinstance(values, torch.Tensor):
        _validate_torch_revision_probs(values)
        row_sums = values.sum(dim=1, keepdim=True)
        return values / row_sums
    array = np.asarray(values, dtype=np.float64)
    _validate_numpy_revision_probs(array)
    return array / array.sum(axis=1, keepdims=True)


def apply_binary_revision_choices(
    current_actions: StateArray,
    choices: StateArray,
) -> np.ndarray:
    """Apply stay/switch choices and return revised binary actions."""

    current = validate_binary_actions(current_actions)
    choice_values = validate_binary_revision_choices(
        choices,
        expected_size=len(current),
    )
    revised = current.copy()
    revised[choice_values == REVISION_SWITCH_TO_ONE] = 1
    revised[choice_values == REVISION_SWITCH_TO_ZERO] = 0
    return revised


def binary_revision_probabilities_from_action_probs(
    action_probs: Any,
    current_actions: StateArray,
) -> Any:
    """Map binary action probabilities onto stay/switch revision probabilities.

    For action 0 agents, action-1 probability becomes switch-to-1 mass. For
    action 1 agents, action-0 probability becomes switch-to-0 mass.
    """

    if isinstance(action_probs, torch.Tensor):
        current = torch.as_tensor(
            validate_binary_actions(current_actions),
            dtype=torch.long,
            device=action_probs.device,
        )
        action1 = _torch_action1_probability(action_probs)
        stay = torch.where(current == 0, 1.0 - action1, action1)
        switch_to_one = torch.where(current == 0, action1, torch.zeros_like(action1))
        switch_to_zero = torch.where(
            current == 1,
            1.0 - action1,
            torch.zeros_like(action1),
        )
        return torch.stack((stay, switch_to_one, switch_to_zero), dim=1)

    current_np = validate_binary_actions(current_actions)
    action1_np = _numpy_action1_probability(action_probs)
    if len(action1_np) != len(current_np):
        raise ValueError("action probabilities must match action count")
    stay_np = np.where(current_np == 0, 1.0 - action1_np, action1_np)
    switch_to_one_np = np.where(current_np == 0, action1_np, 0.0)
    switch_to_zero_np = np.where(current_np == 1, 1.0 - action1_np, 0.0)
    return np.column_stack((stay_np, switch_to_one_np, switch_to_zero_np))


def binary_revision_switch_probabilities(
    revision_probs: Any,
    current_actions: StateArray,
) -> np.ndarray:
    """Return each agent's probability of switching away from its current action."""

    probs = _revision_probs_to_numpy(normalize_binary_revision_probabilities(revision_probs))
    current = validate_binary_actions(current_actions)
    if len(probs) != len(current):
        raise ValueError("revision probabilities must match action count")
    return np.where(current == 0, probs[:, 1], probs[:, 2])


def binary_revision_effective_stay_probabilities(
    revision_probs: Any,
    current_actions: StateArray,
) -> np.ndarray:
    """Return probability mass that leaves each agent's action unchanged."""

    probs = _revision_probs_to_numpy(normalize_binary_revision_probabilities(revision_probs))
    current = validate_binary_actions(current_actions)
    if len(probs) != len(current):
        raise ValueError("revision probabilities must match action count")
    return np.where(current == 0, probs[:, 0] + probs[:, 2], probs[:, 0] + probs[:, 1])


def _validate_torch_revision_probs(values: torch.Tensor) -> None:
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("binary revision probabilities must have shape (n, 3)")
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("binary revision probabilities must contain finite values")
    if bool(torch.any(values < 0.0)):
        raise ValueError("binary revision probabilities must be non-negative")
    if bool(torch.any(values.sum(dim=1) <= 0.0)):
        raise ValueError("binary revision probability rows must have positive mass")


def _validate_numpy_revision_probs(values: np.ndarray) -> None:
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("binary revision probabilities must have shape (n, 3)")
    if not np.isfinite(values).all():
        raise ValueError("binary revision probabilities must contain finite values")
    if (values < 0.0).any():
        raise ValueError("binary revision probabilities must be non-negative")
    if (values.sum(axis=1) <= 0.0).any():
        raise ValueError("binary revision probability rows must have positive mass")


def _state_array_to_numpy(values: StateArray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _revision_probs_to_numpy(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def _torch_action1_probability(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values.to(dtype=torch.float32)
    if values.ndim == 2 and values.shape[1] == 2:
        return values[:, 1]
    raise ValueError("action probabilities must have shape (n,) or (n, 2)")


def _numpy_action1_probability(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        action1 = array
    elif array.ndim == 2 and array.shape[1] == 2:
        action1 = array[:, 1]
    else:
        raise ValueError("action probabilities must have shape (n,) or (n, 2)")
    if not np.all(np.isfinite(action1)):
        raise ValueError("action probabilities must contain finite values")
    if np.any((action1 < 0.0) | (action1 > 1.0)):
        raise ValueError("action probabilities must lie in [0, 1]")
    return action1


def _mean_or_zero(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(array.mean())


def _choice_rate(choices: np.ndarray, choice: int) -> float:
    if len(choices) == 0:
        return 0.0
    return float(np.mean(choices == choice))
