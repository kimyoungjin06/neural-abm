"""Reusable social-block helpers for Neural ABM toy adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from neural_abm.metrics import js_divergence_np

TensorState = dict[str, torch.Tensor]
StateAlignment = Callable[[TensorState, TensorState], TensorState]
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


@dataclass(frozen=True)
class PeerIndexCache:
    """Cached flattened peer indexes for variable-degree social mixing."""

    agent_count: int
    target_index: torch.Tensor
    peer_index: torch.Tensor
    counts: torch.Tensor
    active_mask: torch.Tensor
    active_agent_ids: tuple[int, ...]

    @classmethod
    def from_peer_ids(
        cls,
        peer_ids: list[list[int]],
        *,
        device: torch.device | str | None = None,
    ) -> PeerIndexCache:
        agent_count = len(peer_ids)
        validate_peer_ids(peer_ids, agent_count)

        active_agent_ids: list[int] = []
        target_ids: list[int] = []
        flat_peer_ids: list[int] = []
        count_values = [0 for _ in range(agent_count)]
        active_mask_values = [False for _ in range(agent_count)]
        for agent_id, peers in enumerate(peer_ids):
            peer_count = len(peers)
            if peer_count == 0:
                continue
            active_agent_ids.append(agent_id)
            target_ids.extend([agent_id] * peer_count)
            flat_peer_ids.extend(int(peer_id) for peer_id in peers)
            count_values[agent_id] = peer_count
            active_mask_values[agent_id] = True

        return cls(
            agent_count=agent_count,
            target_index=torch.as_tensor(target_ids, dtype=torch.long, device=device),
            peer_index=torch.as_tensor(flat_peer_ids, dtype=torch.long, device=device),
            counts=torch.as_tensor(count_values, dtype=torch.long, device=device),
            active_mask=torch.as_tensor(
                active_mask_values,
                dtype=torch.bool,
                device=device,
            ),
            active_agent_ids=tuple(active_agent_ids),
        )


def _validate_peer_index_cache(
    cache: PeerIndexCache,
    values: torch.Tensor,
    peer_ids: list[list[int]],
    *,
    validate_values: bool,
) -> None:
    agent_count = int(values.shape[0])
    if cache.agent_count != agent_count:
        raise ValueError("peer_index_cache agent_count must match values")
    tensors = (
        ("target_index", cache.target_index, torch.long),
        ("peer_index", cache.peer_index, torch.long),
        ("counts", cache.counts, torch.long),
        ("active_mask", cache.active_mask, torch.bool),
    )
    for name, tensor, dtype in tensors:
        if tensor.device != values.device:
            raise ValueError(f"peer_index_cache {name} must be on the values device")
        if tensor.dtype != dtype:
            raise ValueError(f"peer_index_cache {name} has invalid dtype")
        if tensor.ndim != 1:
            raise ValueError(f"peer_index_cache {name} must be 1D")
    if cache.target_index.shape != cache.peer_index.shape:
        raise ValueError("peer_index_cache target_index and peer_index must align")
    if int(cache.counts.shape[0]) != agent_count:
        raise ValueError("peer_index_cache counts length must match values")
    if int(cache.active_mask.shape[0]) != agent_count:
        raise ValueError("peer_index_cache active_mask length must match values")

    if not validate_values:
        return

    if bool(torch.any(cache.counts < 0)):
        raise ValueError("peer_index_cache counts must be non-negative")
    if cache.target_index.numel() > 0:
        if bool(
            torch.any((cache.target_index < 0) | (cache.target_index >= agent_count))
        ):
            raise ValueError("peer_index_cache target_index contains out-of-range ids")
        if bool(torch.any((cache.peer_index < 0) | (cache.peer_index >= agent_count))):
            raise ValueError("peer_index_cache peer_index contains out-of-range ids")
    expected_counts = torch.bincount(cache.target_index, minlength=agent_count)
    if not bool(torch.equal(expected_counts, cache.counts)):
        raise ValueError("peer_index_cache counts do not match target_index")
    if not bool(torch.equal(cache.counts > 0, cache.active_mask)):
        raise ValueError("peer_index_cache active_mask does not match counts")
    expected_active_ids = tuple(
        int(agent_id) for agent_id in torch.nonzero(cache.active_mask).flatten().tolist()
    )
    if cache.active_agent_ids != expected_active_ids:
        raise ValueError("peer_index_cache active_agent_ids do not match active_mask")
    expected_cache = PeerIndexCache.from_peer_ids(peer_ids, device=values.device)
    if (
        not bool(torch.equal(cache.target_index, expected_cache.target_index))
        or not bool(torch.equal(cache.peer_index, expected_cache.peer_index))
        or not bool(torch.equal(cache.counts, expected_cache.counts))
        or not bool(torch.equal(cache.active_mask, expected_cache.active_mask))
        or cache.active_agent_ids != expected_cache.active_agent_ids
    ):
        raise ValueError("peer_index_cache does not match peer_ids")


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


def validate_probability_tensor(
    values: torch.Tensor,
    *,
    name: str = "values",
    atol: float = 1e-6,
) -> None:
    """Require a finite torch tensor of distributions over the final axis."""

    if values.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dimensions")
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError(f"{name} must contain only finite values")
    if bool(torch.any((values < 0.0) | (values > 1.0))):
        raise ValueError(f"{name} values must lie in [0, 1]")
    row_sums = values.sum(dim=-1)
    if not bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=atol)):
        raise ValueError(f"{name} distributions must sum to 1 on the final axis")


def validate_finite_tensor(values: torch.Tensor, *, name: str = "values") -> None:
    """Require a finite tensor with an agent axis."""

    if values.ndim < 1:
        raise ValueError(f"{name} must have at least 1 dimension")
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError(f"{name} must contain only finite values")


def validate_state_dicts(
    states: list[TensorState],
    *,
    name: str = "states",
) -> None:
    """Require compatible finite floating-point state dictionaries."""

    if not states:
        return

    reference_keys = list(states[0])
    reference_key_set = set(reference_keys)
    reference_shapes = {key: states[0][key].shape for key in reference_keys}
    reference_dtypes = {key: states[0][key].dtype for key in reference_keys}
    for state_id, state in enumerate(states):
        if set(state) != reference_key_set:
            raise ValueError(f"{name}[{state_id}] must have the same keys")
        for key in reference_keys:
            value = state[key]
            if value.shape != reference_shapes[key]:
                raise ValueError(f"{name}[{state_id}][{key!r}] has incompatible shape")
            if value.dtype != reference_dtypes[key]:
                raise ValueError(f"{name}[{state_id}][{key!r}] has incompatible dtype")
            if not torch.is_floating_point(value):
                raise ValueError(f"{name}[{state_id}][{key!r}] must be floating point")
            validate_finite_tensor(value, name=f"{name}[{state_id}][{key!r}]")


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


def mix_probability_distributions(
    values: torch.Tensor,
    peer_ids: list[list[int]],
    alpha: float,
    *,
    channel: str = "output_distribution",
    commit_mode: str = "distillation_step",
    copy_peers: bool = True,
    uniform_peer_count: int | None = None,
    uniform_peer_index: torch.Tensor | None = None,
    peer_index_cache: PeerIndexCache | None = None,
    validate_peers: bool = True,
    collect_update_norms: bool = True,
) -> SocialMixResult:
    """Mix distribution-valued output channels toward selected peer means."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    validate_probability_tensor(values)
    if validate_peers:
        validate_peer_ids(peer_ids, int(values.shape[0]))

    original = values.detach()
    mixed = original.clone()
    update_norms_tensor = (
        torch.zeros(
            int(original.shape[0]),
            dtype=original.dtype,
            device=original.device,
        )
        if collect_update_norms
        else None
    )

    agent_count = int(original.shape[0])
    if peer_index_cache is not None and (
        uniform_peer_count is not None or uniform_peer_index is not None
    ):
        raise ValueError("peer_index_cache cannot be combined with uniform peer indexes")
    if uniform_peer_index is not None:
        if uniform_peer_index.ndim != 2:
            raise ValueError("uniform_peer_index must be a 2D tensor")
        if int(uniform_peer_index.shape[0]) != agent_count:
            raise ValueError(
                "uniform_peer_index first dimension must equal values first dimension"
            )
        if uniform_peer_index.device != original.device:
            raise ValueError("uniform_peer_index must be on the same device as values")
        if uniform_peer_index.dtype != torch.long:
            raise ValueError("uniform_peer_index must have dtype torch.long")
        inferred_peer_count = int(uniform_peer_index.shape[1])
        if uniform_peer_count is None:
            uniform_peer_count = inferred_peer_count
        elif uniform_peer_count != inferred_peer_count:
            raise ValueError("uniform_peer_count must match uniform_peer_index shape")
        if validate_peers and bool(
            torch.any((uniform_peer_index < 0) | (uniform_peer_index >= agent_count))
        ):
            raise ValueError("uniform_peer_index contains out-of-range peer ids")
    if peer_index_cache is not None and validate_peers:
        _validate_peer_index_cache(
            peer_index_cache,
            original,
            peer_ids,
            validate_values=validate_peers,
        )

    has_uniform_nonzero_peers = (
        uniform_peer_count is not None and uniform_peer_count > 0
    )
    if has_uniform_nonzero_peers:
        if validate_peers and any(
            len(peers) != uniform_peer_count for peers in peer_ids
        ):
            raise ValueError("uniform_peer_count must match every peer list length")
        active_agent_ids = list(range(agent_count))
    elif uniform_peer_count == 0:
        if validate_peers and any(peer_ids):
            raise ValueError("uniform_peer_count must match every peer list length")
        active_agent_ids = []
    elif peer_index_cache is not None:
        active_agent_ids = list(peer_index_cache.active_agent_ids)
    else:
        active_agent_ids = []
        target_ids: list[int] = []
        flat_peer_ids: list[int] = []
        for agent_id, peers in enumerate(peer_ids):
            peer_count = len(peers)
            if peer_count == 0:
                continue
            active_agent_ids.append(agent_id)
            target_ids.extend([agent_id] * peer_count)
            flat_peer_ids.extend(int(peer_id) for peer_id in peers)

    if has_uniform_nonzero_peers and alpha != 0.0:
        neighbor_index = (
            uniform_peer_index
            if uniform_peer_index is not None
            else torch.as_tensor(peer_ids, dtype=torch.long, device=original.device)
        )
        peer_values = original.index_select(0, neighbor_index.reshape(-1))
        peer_values = peer_values.reshape(
            (agent_count, uniform_peer_count) + tuple(original.shape[1:])
        )
        peer_mean = peer_values.mean(dim=1)
        mixed_active = (1.0 - alpha) * original + alpha * peer_mean
        mixed_active = mixed_active / mixed_active.sum(dim=-1, keepdim=True)
        mixed = mixed_active
        if update_norms_tensor is not None:
            norm_dims = tuple(range(1, mixed.ndim))
            update_norms_tensor = torch.linalg.vector_norm(
                mixed_active - original,
                dim=norm_dims,
            )
    elif not has_uniform_nonzero_peers and active_agent_ids and alpha != 0.0:
        if peer_index_cache is not None:
            target_index = peer_index_cache.target_index
            peer_index = peer_index_cache.peer_index
            counts = peer_index_cache.counts
            active_mask = peer_index_cache.active_mask
        else:
            target_index = torch.as_tensor(
                target_ids,
                dtype=torch.long,
                device=original.device,
            )
            peer_index = torch.as_tensor(
                flat_peer_ids,
                dtype=torch.long,
                device=original.device,
            )
            counts = torch.bincount(
                target_index,
                minlength=agent_count,
            )
            active_mask = counts > 0
        peer_sums = torch.zeros_like(original)
        peer_sums.index_add_(0, target_index, original.index_select(0, peer_index))
        count_shape = (agent_count,) + (1,) * (original.ndim - 1)
        peer_mean = peer_sums / counts.clamp_min(1).reshape(count_shape)
        mixed_active = (
            (1.0 - alpha) * original[active_mask] + alpha * peer_mean[active_mask]
        )
        mixed_active = mixed_active / mixed_active.sum(dim=-1, keepdim=True)
        mixed[active_mask] = mixed_active
        if update_norms_tensor is not None:
            norm_dims = tuple(range(1, mixed.ndim))
            update_norms_tensor[active_mask] = torch.linalg.vector_norm(
                mixed_active - original[active_mask],
                dim=norm_dims,
            )
    update_norms = (
        update_norms_tensor.detach().cpu().tolist()
        if update_norms_tensor is not None
        else []
    )

    return SocialMixResult(
        mixed_values=mixed,
        losses=update_norms,
        update_norms=update_norms,
        peer_ids=copy_peer_ids(peer_ids) if copy_peers else peer_ids,
        channel=channel,
        commit_mode=commit_mode,
        active_agent_ids=active_agent_ids,
    )


def mix_tensor_channel(
    values: torch.Tensor,
    peer_ids: list[list[int]],
    alpha: float,
    *,
    channel: str = "tensor",
    commit_mode: str = "tensor_target",
) -> SocialMixResult:
    """Mix a tensor-valued channel along the leading agent axis."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    validate_finite_tensor(values)
    validate_peer_ids(peer_ids, int(values.shape[0]))

    original = values.detach()
    mixed_values: list[torch.Tensor] = []
    losses: list[float] = []
    update_norms: list[float] = []
    for agent_id, peers in enumerate(peer_ids):
        if not peers or alpha == 0.0:
            mixed = original[agent_id].clone()
            update_norm = 0.0
        else:
            peer_mean = original[peers].mean(dim=0)
            mixed = (1.0 - alpha) * original[agent_id] + alpha * peer_mean
            update_norm = float(torch.linalg.vector_norm(mixed - original[agent_id]))
        mixed_values.append(mixed)
        losses.append(update_norm)
        update_norms.append(update_norm)

    return SocialMixResult(
        mixed_values=torch.stack(mixed_values, dim=0),
        losses=losses,
        update_norms=update_norms,
        peer_ids=copy_peer_ids(peer_ids),
        channel=channel,
        commit_mode=commit_mode,
    )


def _clone_state_dict(state: TensorState) -> TensorState:
    return {key: value.detach().clone() for key, value in state.items()}


def _state_delta_norm(
    before: TensorState,
    after: TensorState,
) -> float:
    total = 0.0
    for key, before_value in before.items():
        delta = after[key].detach() - before_value.detach()
        total += float(torch.sum(delta * delta).cpu())
    return total**0.5


def mix_state_dict_channel(
    states: list[TensorState],
    peer_ids: list[list[int]],
    alpha: float,
    *,
    align_state: StateAlignment | None = None,
    channel: str = "parameter_state",
    commit_mode: str = "state_dict_load",
) -> SocialMixResult:
    """Mix parameter-state dictionaries toward selected peer-state means."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    validate_peer_ids(peer_ids, len(states))
    validate_state_dicts(states)

    original = [_clone_state_dict(state) for state in states]
    mixed_states: list[TensorState] = []
    losses: list[float] = []
    update_norms: list[float] = []
    for agent_id, peers in enumerate(peer_ids):
        reference_state = original[agent_id]
        if not peers or alpha == 0.0:
            mixed_state = _clone_state_dict(reference_state)
            update_norm = 0.0
        else:
            peer_states = []
            for peer_id in peers:
                peer_state = original[peer_id]
                if align_state is not None:
                    peer_state = align_state(peer_state, reference_state)
                    validate_state_dicts(
                        [reference_state, peer_state],
                        name="aligned_states",
                    )
                peer_states.append(peer_state)

            mixed_state = {}
            for key, self_value in reference_state.items():
                peer_mean = torch.stack(
                    [peer_state[key].detach() for peer_state in peer_states],
                    dim=0,
                ).mean(dim=0)
                mixed_state[key] = (1.0 - alpha) * self_value + alpha * peer_mean
            update_norm = _state_delta_norm(reference_state, mixed_state)

        mixed_states.append(mixed_state)
        losses.append(update_norm)
        update_norms.append(update_norm)

    return SocialMixResult(
        mixed_values=mixed_states,
        losses=losses,
        update_norms=update_norms,
        peer_ids=copy_peer_ids(peer_ids),
        channel=channel,
        commit_mode=commit_mode,
    )


@dataclass(frozen=True)
class SocialBlock:
    """Reusable social block for peer normalization, selection, and channel mixing."""

    alpha: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")

    def peer_ids_for_mixer(
        self,
        peer_ids: list[list[int]],
        mixer: str,
        agent_count: int,
        *,
        active_mixers: tuple[str, ...] = ("output_average",),
    ) -> list[list[int]]:
        del self
        return peer_ids_for_mixer(
            peer_ids=peer_ids,
            mixer=mixer,
            agent_count=agent_count,
            active_mixers=active_mixers,
        )

    def select_scalar_output_peers(
        self,
        neighbors: list[list[int]],
        values: np.ndarray,
        peer_rule: str,
        threshold: float,
    ) -> PeerSelectionResult:
        del self
        return select_scalar_output_peers(
            neighbors=neighbors,
            values=values,
            peer_rule=peer_rule,
            threshold=threshold,
        )

    def select_distribution_output_peers(
        self,
        neighbors: list[list[int]],
        probe_probs: np.ndarray,
        peer_rule: str,
        threshold: float,
    ) -> PeerSelectionResult:
        del self
        return select_distribution_output_peers(
            neighbors=neighbors,
            probe_probs=probe_probs,
            peer_rule=peer_rule,
            threshold=threshold,
        )

    def mix(
        self,
        channel: SocialChannel,
        values: Any,
        peer_ids: list[list[int]],
    ) -> SocialMixResult:
        if channel.kind == SCALAR_PROBABILITY_CHANNEL:
            return mix_scalar_probabilities(
                values=values,
                peer_ids=peer_ids,
                alpha=self.alpha,
                channel=channel.name,
                commit_mode=channel.commit_mode,
            )
        if channel.kind == BOUNDED_SCALAR_CHANNEL:
            return mix_bounded_scalars(
                values=values,
                peer_ids=peer_ids,
                alpha=self.alpha,
                lower_bound=channel.lower_bound,
                upper_bound=channel.upper_bound,
                channel=channel.name,
                commit_mode=channel.commit_mode,
            )
        if channel.kind == PROBABILITY_DISTRIBUTION_CHANNEL:
            return mix_probability_distributions(
                values=values,
                peer_ids=peer_ids,
                alpha=self.alpha,
                channel=channel.name,
                commit_mode=channel.commit_mode,
            )
        if channel.kind == TENSOR_CHANNEL:
            return mix_tensor_channel(
                values=values,
                peer_ids=peer_ids,
                alpha=self.alpha,
                channel=channel.name,
                commit_mode=channel.commit_mode,
            )
        if channel.kind == STATE_DICT_CHANNEL:
            return mix_state_dict_channel(
                states=values,
                peer_ids=peer_ids,
                alpha=self.alpha,
                align_state=channel.align_state,
                channel=channel.name,
                commit_mode=channel.commit_mode,
            )
        raise ValueError(f"Unsupported social channel kind: {channel.kind}")
