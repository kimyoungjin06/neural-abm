"""Torch-free social exchange primitives for lightweight package profiles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from neural_abm.metrics_core import js_divergence_np

StateAlignment = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
SCALAR_PROBABILITY_CHANNEL = "scalar_probability"
BOUNDED_SCALAR_CHANNEL = "bounded_scalar"
PROBABILITY_DISTRIBUTION_CHANNEL = "probability_distribution"
TENSOR_CHANNEL = "tensor"
STATE_DICT_CHANNEL = "state_dict"
SUPPORTED_SOCIAL_CHANNEL_KINDS = (
    SCALAR_PROBABILITY_CHANNEL,
    BOUNDED_SCALAR_CHANNEL,
    PROBABILITY_DISTRIBUTION_CHANNEL,
    TENSOR_CHANNEL,
    STATE_DICT_CHANNEL,
)


@dataclass(frozen=True)
class SocialChannel:
    """Typed social channel contract for reusable mix dispatch."""

    name: str
    kind: str
    commit_mode: str
    align_state: StateAlignment | None = None
    lower_bound: float = 0.0
    upper_bound: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SocialChannel name must be non-empty")
        if self.kind not in SUPPORTED_SOCIAL_CHANNEL_KINDS:
            raise ValueError(f"Unsupported social channel kind: {self.kind}")
        if not self.commit_mode:
            raise ValueError("SocialChannel commit_mode must be non-empty")
        if self.align_state is not None and self.kind != STATE_DICT_CHANNEL:
            raise ValueError("align_state is only supported for state_dict channels")
        if self.kind == BOUNDED_SCALAR_CHANNEL:
            if not np.isfinite(self.lower_bound) or not np.isfinite(self.upper_bound):
                raise ValueError("bounded scalar channel bounds must be finite")
            if self.lower_bound > self.upper_bound:
                raise ValueError("bounded scalar lower_bound must be <= upper_bound")


@dataclass(frozen=True)
class PeerSelectionResult:
    """Peer ids plus optional compatibility scores from a social selection step."""

    peer_ids: list[list[int]]
    similarity: np.ndarray | None = None

    @property
    def peer_counts(self) -> list[int]:
        return [len(peers) for peers in self.peer_ids]


@dataclass(frozen=True)
class SocialMixResult:
    """Result of a typed social mix operation before toy-specific commit."""

    mixed_values: Any
    losses: list[float]
    update_norms: list[float]
    peer_ids: list[list[int]]
    channel: str
    commit_mode: str
    active_agent_ids: list[int] | None = None


def empty_peers(agent_count: int) -> list[list[int]]:
    """Return an empty peer list for each agent."""

    if agent_count < 0:
        raise ValueError("agent_count must be non-negative")
    return [[] for _ in range(agent_count)]


def copy_peer_ids(peer_ids: list[list[int]]) -> list[list[int]]:
    return [[int(peer_id) for peer_id in peers] for peers in peer_ids]


def uniform_peer_count(peer_ids: list[list[int]]) -> int | None:
    """Return the shared peer count when every peer list has the same length."""

    if not peer_ids:
        return 0
    first_count = len(peer_ids[0])
    if all(len(peers) == first_count for peers in peer_ids[1:]):
        return first_count
    return None


def validate_peer_ids(peer_ids: list[list[int]], agent_count: int) -> None:
    """Validate peer list length and peer id bounds."""

    if agent_count < 0:
        raise ValueError("agent_count must be non-negative")
    if len(peer_ids) != agent_count:
        raise ValueError(
            f"peer_ids length ({len(peer_ids)}) must equal agent_count ({agent_count})"
        )
    for agent_id, peers in enumerate(peer_ids):
        for peer_id in peers:
            if not 0 <= int(peer_id) < agent_count:
                raise ValueError(
                    f"Invalid peer id {peer_id} for agent {agent_id}; "
                    f"expected 0 <= peer_id < {agent_count}"
                )


def peer_ids_for_mixer(
    peer_ids: list[list[int]],
    mixer: str,
    agent_count: int,
    *,
    active_mixers: tuple[str, ...] = ("output_average",),
    copy_peers: bool = True,
    validate_peers: bool = True,
) -> list[list[int]]:
    """Return active peers for a mixer, enforcing the no-social empty-peer rule."""

    if mixer == "none":
        return empty_peers(agent_count)
    if mixer in active_mixers:
        if validate_peers:
            validate_peer_ids(peer_ids, agent_count)
        return copy_peer_ids(peer_ids) if copy_peers else peer_ids
    raise ValueError(f"Unsupported mixer: {mixer}")


def validate_probability_vector(values: np.ndarray, *, name: str = "values") -> None:
    """Require a one-dimensional finite vector in [0, 1]."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D probability vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} values must lie in [0, 1]")


def validate_bounded_scalar_vector(
    values: np.ndarray,
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    name: str = "values",
) -> None:
    """Require a one-dimensional finite scalar vector inside declared bounds."""

    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        raise ValueError(f"{name} bounds must be finite")
    if lower_bound > upper_bound:
        raise ValueError(f"{name} lower_bound must be <= upper_bound")

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D bounded scalar vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((array < lower_bound) | (array > upper_bound)):
        raise ValueError(
            f"{name} values must lie in [{lower_bound:g}, {upper_bound:g}]"
        )


def validate_probability_matrix(
    values: np.ndarray,
    *,
    name: str = "values",
    atol: float = 1e-6,
) -> None:
    """Require a finite row-stochastic probability matrix."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D probability matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} values must lie in [0, 1]")
    row_sums = array.sum(axis=1)
    if not np.allclose(row_sums, np.ones(len(row_sums)), atol=atol):
        raise ValueError(f"{name} rows must sum to 1")


def validate_probability_distributions(
    values: np.ndarray,
    *,
    name: str = "values",
    atol: float = 1e-6,
) -> None:
    """Require finite probability distributions over the final axis."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} values must lie in [0, 1]")
    row_sums = array.sum(axis=-1)
    if not np.allclose(row_sums, np.ones_like(row_sums), atol=atol):
        raise ValueError(f"{name} distributions must sum to 1 on the final axis")


def scalar_output_similarity_matrix(values: np.ndarray) -> np.ndarray:
    """Compatibility matrix for scalar Bernoulli-style output channels."""

    validate_probability_vector(values)
    array = np.asarray(values, dtype=np.float64)
    return 1.0 - np.abs(array[:, None] - array[None, :])


def bounded_scalar_similarity_matrix(
    values: np.ndarray,
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> np.ndarray:
    """Compatibility matrix for finite bounded scalar channels."""

    validate_bounded_scalar_vector(
        values,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    array = np.asarray(values, dtype=np.float64)
    span = upper_bound - lower_bound
    if span == 0.0:
        return np.ones((len(array), len(array)), dtype=np.float64)
    return 1.0 - np.abs(array[:, None] - array[None, :]) / span


def distribution_output_similarity_matrix(probe_probs: np.ndarray) -> np.ndarray:
    """Compatibility matrix for distribution-valued output channels."""

    validate_probability_distributions(probe_probs, name="probe_probs")
    count = probe_probs.shape[0]
    matrix = np.eye(count, dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            similarity = 1.0 - js_divergence_np(probe_probs[i], probe_probs[j])
            matrix[i, j] = similarity
            matrix[j, i] = similarity
    return matrix


def select_scalar_output_peers(
    neighbors: list[list[int]],
    values: np.ndarray,
    peer_rule: str,
    threshold: float,
    *,
    copy_peers: bool = True,
    validate_peers: bool = True,
) -> PeerSelectionResult:
    """Select graph-neighbor peers for scalar output or policy probabilities."""

    agent_count = len(neighbors)
    validate_probability_vector(values)
    if validate_peers:
        validate_peer_ids(neighbors, agent_count)
    if len(values) != agent_count:
        raise ValueError(
            f"values length ({len(values)}) must equal agent_count ({agent_count})"
        )
    if peer_rule == "none":
        return PeerSelectionResult(
            peer_ids=copy_peer_ids(neighbors) if copy_peers else neighbors,
            similarity=None,
        )
    if peer_rule != "output_similarity":
        raise ValueError(f"Unsupported peer rule: {peer_rule}")

    similarity = scalar_output_similarity_matrix(values)
    selected = [
        [
            int(peer_id)
            for peer_id in peers
            if similarity[agent_id, int(peer_id)] >= threshold
        ]
        for agent_id, peers in enumerate(neighbors)
    ]
    return PeerSelectionResult(peer_ids=selected, similarity=similarity)


def select_bounded_scalar_output_peers(
    neighbors: list[list[int]],
    values: np.ndarray,
    peer_rule: str,
    threshold: float,
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    copy_peers: bool = True,
    validate_peers: bool = True,
) -> PeerSelectionResult:
    """Select graph-neighbor peers for bounded scalar outputs."""

    agent_count = len(neighbors)
    validate_bounded_scalar_vector(
        values,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    if validate_peers:
        validate_peer_ids(neighbors, agent_count)
    if len(values) != agent_count:
        raise ValueError(
            f"values length ({len(values)}) must equal agent_count ({agent_count})"
        )
    if peer_rule == "none":
        return PeerSelectionResult(
            peer_ids=copy_peer_ids(neighbors) if copy_peers else neighbors,
            similarity=None,
        )
    if peer_rule != "output_similarity":
        raise ValueError(f"Unsupported peer rule: {peer_rule}")

    similarity = bounded_scalar_similarity_matrix(
        values,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    selected = [
        [
            int(peer_id)
            for peer_id in peers
            if similarity[agent_id, int(peer_id)] >= threshold
        ]
        for agent_id, peers in enumerate(neighbors)
    ]
    return PeerSelectionResult(peer_ids=selected, similarity=similarity)


def select_distribution_output_peers(
    neighbors: list[list[int]],
    probe_probs: np.ndarray,
    peer_rule: str,
    threshold: float,
) -> PeerSelectionResult:
    """Select graph-neighbor peers for distribution-valued output channels."""

    agent_count = len(neighbors)
    validate_probability_distributions(probe_probs, name="probe_probs")
    validate_peer_ids(neighbors, agent_count)
    if len(probe_probs) != agent_count:
        raise ValueError(
            "probe_probs first dimension "
            f"({len(probe_probs)}) must equal agent_count ({agent_count})"
        )
    if peer_rule == "none":
        return PeerSelectionResult(peer_ids=copy_peer_ids(neighbors), similarity=None)
    if peer_rule != "output_similarity":
        raise ValueError(f"Unsupported peer rule: {peer_rule}")

    similarity = distribution_output_similarity_matrix(probe_probs)
    selected = [
        [
            int(peer_id)
            for peer_id in peers
            if similarity[agent_id, int(peer_id)] >= threshold
        ]
        for agent_id, peers in enumerate(neighbors)
    ]
    return PeerSelectionResult(peer_ids=selected, similarity=similarity)


def mix_scalar_probabilities(
    values: np.ndarray,
    peer_ids: list[list[int]],
    alpha: float,
    *,
    channel: str = "scalar_probability",
    commit_mode: str = "scalar_probability_sample",
) -> SocialMixResult:
    """Mix a scalar probability channel toward selected peer means."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    validate_probability_vector(values)
    original = np.asarray(values, dtype=np.float64)
    validate_peer_ids(peer_ids, len(original))

    mixed = original.copy()
    losses: list[float] = []
    update_norms: list[float] = []
    for agent_id, peers in enumerate(peer_ids):
        if not peers or alpha == 0.0:
            losses.append(0.0)
            update_norms.append(0.0)
            continue
        peer_mean = float(np.mean(original[peers]))
        mixed_value = (1.0 - alpha) * float(original[agent_id]) + alpha * peer_mean
        mixed[agent_id] = float(np.clip(mixed_value, 0.0, 1.0))
        update_norm = abs(float(mixed[agent_id] - original[agent_id]))
        losses.append(update_norm)
        update_norms.append(update_norm)

    return SocialMixResult(
        mixed_values=mixed,
        losses=losses,
        update_norms=update_norms,
        peer_ids=copy_peer_ids(peer_ids),
        channel=channel,
        commit_mode=commit_mode,
    )


def mix_bounded_scalars(
    values: np.ndarray,
    peer_ids: list[list[int]],
    alpha: float,
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    channel: str = "bounded_scalar",
    commit_mode: str = "bounded_scalar_commit",
) -> SocialMixResult:
    """Mix bounded scalar values toward selected peer means."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    validate_bounded_scalar_vector(
        values,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    original = np.asarray(values, dtype=np.float64)
    validate_peer_ids(peer_ids, len(original))

    mixed = original.copy()
    losses: list[float] = []
    update_norms: list[float] = []
    for agent_id, peers in enumerate(peer_ids):
        if not peers or alpha == 0.0:
            losses.append(0.0)
            update_norms.append(0.0)
            continue
        peer_mean = float(np.mean(original[peers]))
        mixed_value = (1.0 - alpha) * float(original[agent_id]) + alpha * peer_mean
        mixed[agent_id] = float(np.clip(mixed_value, lower_bound, upper_bound))
        update_norm = abs(float(mixed[agent_id] - original[agent_id]))
        losses.append(update_norm)
        update_norms.append(update_norm)

    return SocialMixResult(
        mixed_values=mixed,
        losses=losses,
        update_norms=update_norms,
        peer_ids=copy_peer_ids(peer_ids),
        channel=channel,
        commit_mode=commit_mode,
    )


__all__ = [
    "BOUNDED_SCALAR_CHANNEL",
    "PROBABILITY_DISTRIBUTION_CHANNEL",
    "SCALAR_PROBABILITY_CHANNEL",
    "STATE_DICT_CHANNEL",
    "SUPPORTED_SOCIAL_CHANNEL_KINDS",
    "TENSOR_CHANNEL",
    "PeerSelectionResult",
    "SocialChannel",
    "SocialMixResult",
    "StateAlignment",
    "bounded_scalar_similarity_matrix",
    "copy_peer_ids",
    "distribution_output_similarity_matrix",
    "empty_peers",
    "mix_bounded_scalars",
    "mix_scalar_probabilities",
    "peer_ids_for_mixer",
    "scalar_output_similarity_matrix",
    "select_bounded_scalar_output_peers",
    "select_distribution_output_peers",
    "select_scalar_output_peers",
    "uniform_peer_count",
    "validate_bounded_scalar_vector",
    "validate_peer_ids",
    "validate_probability_distributions",
    "validate_probability_matrix",
    "validate_probability_vector",
]
