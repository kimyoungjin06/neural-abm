"""Shared helpers for binary neural toy domains."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from neural_abm.accelerator import (
    AcceleratorTimingRecorder,
    AcceleratorTimingSynchronizer,
    BatchedAdamStateCache,
    BatchedMLPParameters,
    BatchedMLPUpdateResult,
    apply_batched_mlp_loss_gradients_with_result,
    batched_binary_policy_gradient_losses,
    batched_distribution_cross_entropy_losses,
    trainable_batched_mlp_parameters,
)
from neural_abm.social import PeerIndexCache, mix_probability_distributions


@runtime_checkable
class TensorPolicyRuntime(Protocol):
    """Capability contract for tensor-owned binary policy accelerators."""

    @property
    def agent_count(self) -> int:
        """Number of agents represented by the runtime."""

    @property
    def parameters(self) -> BatchedMLPParameters:
        """Current detached policy parameters."""

    def probabilities(
        self,
        observations: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Evaluate action probabilities for all represented agents."""

    def logits(self, observations: torch.Tensor) -> torch.Tensor:
        """Evaluate action logits for all represented agents."""

    def trainable_parameters(self) -> BatchedMLPParameters:
        """Return grad-enabled parameters for one batched update."""

    def apply_loss_gradients(
        self,
        parameters: BatchedMLPParameters,
        losses: torch.Tensor,
        *,
        active_agent_ids: Sequence[int] | None = None,
        timing_prefix: str | None = None,
        timing_recorder: AcceleratorTimingRecorder | None = None,
        timing_synchronizer: AcceleratorTimingSynchronizer | None = None,
    ) -> BatchedMLPUpdateResult:
        """Apply per-agent losses to runtime-owned parameters."""

    def flush_to_agents(self, agents: Sequence[object]) -> None:
        """Copy runtime-owned state back to concrete agents."""


def accelerator_timing_kwargs(
    context: Any | None,
    prefix: str,
) -> dict[str, object]:
    """Return optional accelerator timing hooks from a binary step context."""

    if context is None:
        return {}
    recorder = context.extras.get("_record_timing")
    synchronizer = context.extras.get("_synchronize_timing_device")
    return {
        "timing_prefix": prefix,
        "timing_recorder": recorder if callable(recorder) else None,
        "timing_synchronizer": synchronizer if callable(synchronizer) else None,
    }


def can_defer_static_output_average_agent_sync(
    *,
    peer_rule: str,
    mixer: str,
    alpha: float,
    uniform_neighbor_peer_count: int | None,
    peer_index_cache: Any | None,
    agent_count: int,
) -> bool:
    """Return whether local agent sync can wait for static all-agent averaging."""

    if peer_rule != "none" or mixer != "output_average":
        return False
    if alpha == 0.0:
        return False
    if uniform_neighbor_peer_count is not None:
        return uniform_neighbor_peer_count > 0
    return (
        peer_index_cache is not None
        and len(peer_index_cache.active_agent_ids) == agent_count
    )


def apply_tensor_binary_policy_gradient_update(
    *,
    runtime: TensorPolicyRuntime,
    observations: torch.Tensor,
    actions: Sequence[int] | torch.Tensor,
    advantages: Sequence[float] | torch.Tensor,
    revision_mask: np.ndarray | None = None,
    active_agent_ids: Sequence[int] | None = None,
    entropy_beta: float,
    timing_context: Any | None = None,
    timing_prefix: str = "local",
) -> BatchedMLPUpdateResult:
    """Run a tensor-runtime binary policy-gradient update."""

    if active_agent_ids is None and revision_mask is not None:
        active_agent_ids = _active_agent_ids_from_revision_mask(
            revision_mask,
            runtime.agent_count,
        )
    elif active_agent_ids is not None:
        active_agent_ids = _canonical_active_agent_ids(
            active_agent_ids,
            runtime.agent_count,
        )
    with _timed_context_stage(timing_context, f"{timing_prefix}_trainable_parameters"):
        update_parameters = runtime.trainable_parameters()
    with _timed_context_stage(timing_context, f"{timing_prefix}_loss_update"):
        with _timed_context_stage(timing_context, f"{timing_prefix}_loss_forward"):
            losses = batched_binary_policy_gradient_losses(
                update_parameters,
                observations,
                actions=actions,
                advantages=advantages,
                entropy_beta=entropy_beta,
            )
        with _timed_context_stage(timing_context, f"{timing_prefix}_optimizer_update"):
            return runtime.apply_loss_gradients(
                update_parameters,
                losses,
                active_agent_ids=active_agent_ids,
                **accelerator_timing_kwargs(timing_context, timing_prefix),
            )


def apply_tensor_output_average_distillation_update(
    *,
    runtime: TensorPolicyRuntime,
    observations: torch.Tensor,
    peer_ids: list[list[int]],
    alpha: float,
    previous_probs: torch.Tensor,
    uniform_peer_count: int | None = None,
    uniform_peer_index: torch.Tensor | None = None,
    peer_index_cache: PeerIndexCache | None = None,
    validate_peers: bool = True,
    timing_context: Any | None = None,
    timing_prefix: str = "social",
    loss_mode: str = "cross_entropy",
) -> BatchedMLPUpdateResult:
    """Run tensor-runtime output-average policy distillation."""

    if alpha == 0.0:
        return BatchedMLPUpdateResult(
            losses=[0.0 for _ in range(runtime.agent_count)],
            updated_parameters=runtime.parameters,
            used_batched_optimizer=True,
        )

    with _timed_context_stage(timing_context, f"{timing_prefix}_mix"):
        mix_result = mix_probability_distributions(
            values=previous_probs.detach(),
            peer_ids=peer_ids,
            alpha=alpha,
            channel="policy_distribution",
            commit_mode="distillation_step",
            copy_peers=False,
            uniform_peer_count=uniform_peer_count,
            uniform_peer_index=uniform_peer_index,
            peer_index_cache=peer_index_cache,
            validate_peers=validate_peers,
            collect_update_norms=False,
        )

    active_agent_ids = _active_agent_ids_from_mix_result(
        mix_result.active_agent_ids,
        runtime.agent_count,
    )
    with _timed_context_stage(timing_context, f"{timing_prefix}_trainable_parameters"):
        update_parameters = runtime.trainable_parameters()
    with _timed_context_stage(timing_context, f"{timing_prefix}_loss_update"):
        with _timed_context_stage(timing_context, f"{timing_prefix}_loss_forward"):
            losses = batched_distribution_cross_entropy_losses(
                update_parameters,
                observations,
                mix_result.mixed_values,
                loss_mode=loss_mode,
            )
        with _timed_context_stage(timing_context, f"{timing_prefix}_optimizer_update"):
            return runtime.apply_loss_gradients(
                update_parameters,
                losses,
                active_agent_ids=active_agent_ids,
                **accelerator_timing_kwargs(timing_context, timing_prefix),
            )


def apply_batched_output_average_distillation_update(
    *,
    agents: Sequence[object],
    observations: torch.Tensor,
    peer_ids: list[list[int]],
    alpha: float,
    previous_probs: torch.Tensor,
    parameters: BatchedMLPParameters | None = None,
    adam_state_cache: BatchedAdamStateCache | None = None,
    uniform_peer_count: int | None = None,
    uniform_peer_index: torch.Tensor | None = None,
    peer_index_cache: PeerIndexCache | None = None,
    validate_peers: bool = True,
    timing_context: Any | None = None,
    timing_prefix: str = "social",
    loss_mode: str = "cross_entropy",
    synchronize_model_parameters: bool = True,
    synchronize_optimizer_states: bool = True,
) -> BatchedMLPUpdateResult:
    """Run batched-agent output-average policy distillation."""

    if alpha == 0.0:
        return BatchedMLPUpdateResult(
            losses=[0.0 for _ in agents],
            updated_parameters=(
                parameters.detached()
                if parameters is not None
                else None
            ),
            used_batched_optimizer=parameters is not None,
        )

    with _timed_context_stage(timing_context, f"{timing_prefix}_mix"):
        mix_result = mix_probability_distributions(
            values=previous_probs.detach(),
            peer_ids=peer_ids,
            alpha=alpha,
            channel="policy_distribution",
            commit_mode="distillation_step",
            copy_peers=False,
            uniform_peer_count=uniform_peer_count,
            uniform_peer_index=uniform_peer_index,
            peer_index_cache=peer_index_cache,
            validate_peers=validate_peers,
            collect_update_norms=False,
        )

    active_agent_ids = mix_result.active_agent_ids or []
    update_parameters = (
        trainable_batched_mlp_parameters(agents, device=observations.device)
        if parameters is None
        else parameters
    )
    with _timed_context_stage(timing_context, f"{timing_prefix}_loss_forward"):
        losses = batched_distribution_cross_entropy_losses(
            update_parameters,
            observations,
            mix_result.mixed_values,
            loss_mode=loss_mode,
        )
    with _timed_context_stage(timing_context, f"{timing_prefix}_optimizer_update"):
        return apply_batched_mlp_loss_gradients_with_result(
            agents=agents,
            parameters=update_parameters,
            losses=losses,
            active_agent_ids=active_agent_ids,
            adam_state_cache=adam_state_cache,
            synchronize_model_parameters=synchronize_model_parameters,
            synchronize_optimizer_states=synchronize_optimizer_states,
            **accelerator_timing_kwargs(timing_context, timing_prefix),
        )


def _active_agent_ids_from_revision_mask(
    revision_mask: np.ndarray,
    agent_count: int,
) -> list[int] | None:
    mask = np.asarray(revision_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("revision_mask must be 1D")
    if int(mask.shape[0]) != agent_count:
        raise ValueError("revision_mask length must match runtime agent count")
    if bool(mask.all()):
        return None
    return [int(agent_id) for agent_id in np.flatnonzero(mask)]


def _active_agent_ids_from_mix_result(
    active_agent_ids: Sequence[int] | None,
    agent_count: int,
) -> list[int] | None:
    if active_agent_ids is None:
        return []
    return _canonical_active_agent_ids(active_agent_ids, agent_count)


def _canonical_active_agent_ids(
    active_agent_ids: Sequence[int],
    agent_count: int,
) -> list[int] | None:
    if len(active_agent_ids) != agent_count:
        return [int(agent_id) for agent_id in active_agent_ids]
    if all(int(agent_id) == index for index, agent_id in enumerate(active_agent_ids)):
        return None
    return [int(agent_id) for agent_id in active_agent_ids]


def _timed_context_stage(
    context: Any | None,
    stage: str,
) -> "_BinaryNeuralContextTimer":
    return _BinaryNeuralContextTimer(context=context, stage=stage)


@dataclass
class _BinaryNeuralContextTimer:
    context: Any | None
    stage: str
    start: float = 0.0

    def __enter__(self) -> None:
        if not self._enabled:
            return
        self._synchronize()
        self.start = time.perf_counter()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if not self._enabled:
            return
        self._synchronize()
        record = self.context.extras.get("_record_timing")
        if callable(record):
            record(self.stage, time.perf_counter() - self.start)

    @property
    def _enabled(self) -> bool:
        return bool(
            self.context is not None
            and callable(self.context.extras.get("_record_timing"))
        )

    def _synchronize(self) -> None:
        if self.context is None:
            return
        synchronize = self.context.extras.get("_synchronize_timing_device")
        if callable(synchronize):
            synchronize()
