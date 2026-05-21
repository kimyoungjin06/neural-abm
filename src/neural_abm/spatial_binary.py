"""Shared binary-action spatial runner contracts and primitives."""

from __future__ import annotations

import time
from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import networkx as nx
import torch

from neural_abm.accelerator import (
    BatchedAdamStateCache,
    BatchedMLPParameters,
    BatchedMLPUpdateResult,
    apply_batched_mlp_loss_gradients_with_result,
    batched_binary_policy_gradient_losses,
    batched_distribution_cross_entropy_losses,
    trainable_batched_mlp_parameters,
)
from neural_abm.graphs import component_map, graph_from_peer_ids
from neural_abm.logging import CsvLogWriter
from neural_abm.losses import (
    LossVector,
    TensorBackedLossVector,
    loss_values_at,
    mean_loss_value,
)
from neural_abm.metrics import edge_entropy
from neural_abm.mobility import MobilityParams, MobilityStepResult, mobility_summary
from neural_abm.mobility import apply_local_quality_mobility
from neural_abm.readiness import BinaryReadinessPropagationUnit
from neural_abm.reputation import reputation_summary, update_action_reputation
from neural_abm.results import write_binary_summary_artifact
from neural_abm.social import (
    PROBABILITY_DISTRIBUTION_CHANNEL,
    SocialBlock,
    SocialChannel,
    SocialMixResult,
    mix_scalar_probabilities,
    peer_ids_for_mixer as social_peer_ids_for_mixer,
    select_scalar_output_peers,
    validate_peer_ids,
)
from neural_abm.unit import (
    CommitReport,
    DistributionDistillationAdapter,
    LocalUpdateReport,
    NABMLocalStep,
    NABMStep,
    NABMUnit,
    NABMUnitReport,
    ObservationSpec,
    SocialMessageSpec,
    social_diagnostics,
)


BinarySocialMode = Literal["probability_mix", "policy_distill"]
BinarySocialConfidenceWeighting = Literal["none", "peer", "peer_direction"]
StateArray = np.ndarray | torch.Tensor


def is_torch_tensor(values: Any) -> bool:
    """Return whether values are backed by a torch tensor."""

    return isinstance(values, torch.Tensor)


def to_numpy_view(values: Any, dtype: Any | None = None) -> np.ndarray:
    """Return a NumPy representation for scalar logging and validation."""

    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if dtype is not None:
        return array.astype(dtype, copy=False)
    return array


def scalar_at(values: Any, index: int) -> Any:
    """Return one scalar value from a NumPy or torch-backed vector."""

    value = values[index]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def mean_value(values: Any) -> float:
    """Return a Python float mean for NumPy or torch-backed vectors."""

    if len(values) == 0:
        return 0.0
    if isinstance(values, torch.Tensor):
        return float(values.detach().to(dtype=torch.float64).mean().cpu())
    return float(np.mean(values))


def _copy_back_array(target: StateArray, values: np.ndarray) -> None:
    if isinstance(target, torch.Tensor):
        target.copy_(
            torch.as_tensor(values, dtype=target.dtype, device=target.device)
        )
        return
    target[:] = values


@dataclass
class BinarySpatialState:
    """Mutable state shared by binary spatial toy runners."""

    actions: StateArray
    payoffs: StateArray
    payoff_ema: StateArray
    previous_payoff_ema: StateArray
    reputation: StateArray
    agents: MutableSequence[Any] | None = None
    extras: MutableMapping[str, Any] = field(default_factory=dict)

    @property
    def agent_count(self) -> int:
        return int(len(self.actions))

    def state_arrays(self) -> dict[str, StateArray]:
        return {
            "actions": self.actions,
            "payoffs": self.payoffs,
            "payoff_ema": self.payoff_ema,
            "previous_payoff_ema": self.previous_payoff_ema,
            "reputation": self.reputation,
        }


@dataclass
class BinaryPolicyStepResult:
    """Shared result shape for one binary policy/revision step."""

    pre_revision_probs: Any
    post_local_probs: Any
    post_social_probs: Any
    local_losses: LossVector
    social_losses: LossVector
    peer_ids: list[list[int]]
    revision_mask: np.ndarray
    mobility_result: MobilityStepResult
    realized_revision_rate: float | None = None
    extras: MutableMapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.realized_revision_rate is None and len(self.revision_mask) > 0:
            self.realized_revision_rate = float(np.mean(self.revision_mask))
        elif self.realized_revision_rate is None:
            self.realized_revision_rate = 0.0


@dataclass
class BinaryStepContext:
    """Shared per-step context passed through hook-based binary domains."""

    epoch: int
    revision_mask: np.ndarray
    extras: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass
class BinaryLocalStepResult:
    """Domain-local policy result before runner-managed social coordination."""

    pre_revision_probs: Any
    candidate_action_probs: np.ndarray
    post_local_probs: Any
    local_losses: LossVector
    social_mode: BinarySocialMode
    actions_after_revision: np.ndarray | None = None
    extras: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass
class BinarySocialStepResult:
    """Runner/domain result after social coordination."""

    peer_ids: list[list[int]]
    post_social_probs: Any
    final_action_probs: np.ndarray
    social_losses: LossVector
    extras: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass
class BinaryPolicyLearningResult:
    """Result of one binary neural policy readout and local-learning phase."""

    pre_revision_probs: Any
    decision_action_probs: Any
    actions_after_revision: StateArray
    local_losses: LossVector
    post_local_probs: Any
    extras: MutableMapping[str, Any] = field(default_factory=dict)


class BinaryPolicyReadout(Protocol):
    """Collect per-agent binary policy probabilities for a shared observation batch."""

    def __call__(
        self,
        agents: list[Any],
        observations: torch.Tensor,
        *,
        temperature: float,
    ) -> Any: ...


class BinaryDecisionActionBuilder(Protocol):
    """Build decision-time binary action probabilities from policy readout."""

    def __call__(self, pre_revision_probs: Any) -> Any: ...


class BinaryActionSampler(Protocol):
    """Sample or select binary actions from decision-time action probabilities."""

    def __call__(self, decision_action_probs: Any) -> StateArray: ...


class BinaryLocalUpdateCommit(Protocol):
    """Commit local learning after domain-specific actions have been selected."""

    def __call__(self, actions: StateArray) -> LossVector: ...


class BinaryPolicyCacheRefresh(Protocol):
    """Refresh policy/cache state after local learning has committed."""

    def __call__(self, agents: list[Any]) -> None: ...


@dataclass(frozen=True)
class BinaryPolicyLearningCallbacks:
    """Domain callbacks required by the reusable binary policy-learning lifecycle."""

    collect_policy_probs: BinaryPolicyReadout
    decision_action_probs: BinaryDecisionActionBuilder
    sample_actions: BinaryActionSampler
    local_update: BinaryLocalUpdateCommit
    refresh_policy_cache: BinaryPolicyCacheRefresh | None = None
    post_collect_policy_probs: BinaryPolicyReadout | None = None


@dataclass
class BinaryPolicyLearningUnit:
    """Reusable binary policy readout, action sampling, and local commit flow."""

    agents: Sequence[Any]
    observations: torch.Tensor
    temperature: float
    callbacks: BinaryPolicyLearningCallbacks
    context: BinaryStepContext | None = None
    extras: MutableMapping[str, Any] = field(default_factory=dict)

    def run(self) -> BinaryPolicyLearningResult:
        agents = list(self.agents)
        with timed_context_stage(self.context, "policy_readout"):
            pre_revision_probs = self.callbacks.collect_policy_probs(
                agents,
                self.observations,
                temperature=self.temperature,
            )
            decision_probs = self.callbacks.decision_action_probs(pre_revision_probs)
        with timed_context_stage(self.context, "decision_selection"):
            actions = self.callbacks.sample_actions(decision_probs)
        with timed_context_stage(self.context, "local_training"):
            local_losses = self.callbacks.local_update(actions)
        with timed_context_stage(self.context, "cache_refresh"):
            if self.callbacks.refresh_policy_cache is not None:
                self.callbacks.refresh_policy_cache(agents)
        with timed_context_stage(self.context, "post_local_readout"):
            post_collect_policy_probs = (
                self.callbacks.post_collect_policy_probs
                or self.callbacks.collect_policy_probs
            )
            post_local_probs = post_collect_policy_probs(
                agents,
                self.observations,
                temperature=self.temperature,
            )
        return BinaryPolicyLearningResult(
            pre_revision_probs=pre_revision_probs,
            decision_action_probs=decision_probs,
            actions_after_revision=actions,
            local_losses=local_losses,
            post_local_probs=post_local_probs,
            extras=dict(self.extras),
        )


def run_binary_policy_learning_step(
    *,
    agents: Sequence[Any],
    observations: torch.Tensor,
    temperature: float,
    collect_policy_probs: BinaryPolicyReadout,
    decision_action_probs: BinaryDecisionActionBuilder,
    sample_actions: BinaryActionSampler,
    local_update: BinaryLocalUpdateCommit,
    refresh_policy_cache: BinaryPolicyCacheRefresh | None = None,
    post_collect_policy_probs: BinaryPolicyReadout | None = None,
    context: BinaryStepContext | None = None,
    extras: MutableMapping[str, Any] | None = None,
    unit_type: type[BinaryPolicyLearningUnit] = BinaryPolicyLearningUnit,
) -> BinaryPolicyLearningResult:
    """Wire domain callbacks into the reusable binary policy-learning unit."""

    return unit_type(
        agents=agents,
        observations=observations,
        temperature=temperature,
        callbacks=BinaryPolicyLearningCallbacks(
            collect_policy_probs=collect_policy_probs,
            decision_action_probs=decision_action_probs,
            sample_actions=sample_actions,
            local_update=local_update,
            refresh_policy_cache=refresh_policy_cache,
            post_collect_policy_probs=post_collect_policy_probs,
        ),
        context=context,
        extras=dict(extras or {}),
    ).run()


@dataclass(frozen=True)
class BinaryOutputDistillationReport:
    """NABMUnit-backed report for binary policy-distribution distillation."""

    social_losses: LossVector
    aggregate_diagnostics: Mapping[str, object] = field(default_factory=dict)
    micro_diagnostics: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    unit_report: NABMUnitReport | None = None

    @classmethod
    def from_unit_report(
        cls,
        unit_report: NABMUnitReport,
    ) -> "BinaryOutputDistillationReport":
        return cls(
            social_losses=unit_report.social_step.commit.losses,
            aggregate_diagnostics=unit_report.social_step.diagnostics.aggregate_row(),
            micro_diagnostics=[
                unit_report.social_step.diagnostics.micro_row(agent_id)
                for agent_id in range(len(unit_report.peer_ids))
            ],
            unit_report=unit_report,
        )

    @classmethod
    def from_accelerated_update_result(
        cls,
        update_result: Any,
        *,
        peer_ids: list[list[int]],
        agent_count: int | None = None,
        channel: str = "policy_distribution",
        commit_mode: str = "distillation_step",
        update_norms: Sequence[float] | None = None,
    ) -> "BinaryOutputDistillationReport":
        """Wrap an accelerated backend update in the common social report shape."""

        losses = update_result.losses
        resolved_agent_count = (
            int(agent_count)
            if agent_count is not None
            else max(len(peer_ids), len(losses))
        )
        loss_values = loss_values_at(losses, range(resolved_agent_count))
        update_norm_values = (
            [0.0 for _ in range(resolved_agent_count)]
            if update_norms is None
            else [float(value) for value in update_norms]
        )
        peer_counts = [len(peers) for peers in peer_ids]
        active_social_agent_count = sum(1 for count in peer_counts if count > 0)
        mean_update_norm = (
            float(np.mean(update_norm_values)) if update_norm_values else 0.0
        )
        max_update_norm = max(update_norm_values, default=0.0)
        return cls(
            social_losses=losses,
            aggregate_diagnostics={
                "social_channel": channel,
                "commit_mode": commit_mode,
                "mean_social_loss": mean_loss_value(losses),
                "mean_social_update_norm": mean_update_norm,
                "max_social_update_norm": max_update_norm,
                "active_social_agent_count": active_social_agent_count,
            },
            micro_diagnostics=[
                {
                    "social_channel": channel,
                    "commit_mode": commit_mode,
                    "social_loss": loss_values[agent_id],
                    "social_update_norm": update_norm_values[agent_id],
                }
                for agent_id in range(resolved_agent_count)
            ],
        )

    @classmethod
    def from_batched_update_result(
        cls,
        update_result: Any,
        *,
        peer_ids: list[list[int]],
        agent_count: int | None = None,
        channel: str = "policy_distribution",
        commit_mode: str = "distillation_step",
        update_norms: Sequence[float] | None = None,
    ) -> "BinaryOutputDistillationReport":
        """Compatibility alias for accelerated backend report wrapping."""

        return cls.from_accelerated_update_result(
            update_result,
            peer_ids=peer_ids,
            agent_count=agent_count,
            channel=channel,
            commit_mode=commit_mode,
            update_norms=update_norms,
        )

    def social_result_extras(self) -> dict[str, object]:
        extras: dict[str, object] = {}
        if self.aggregate_diagnostics:
            extras["social_unit_aggregate"] = dict(self.aggregate_diagnostics)
        if self.micro_diagnostics:
            extras["social_unit_micro"] = [
                dict(row) for row in self.micro_diagnostics
            ]
        return extras


@dataclass(frozen=True)
class BinarySocialConfidenceDiagnostics:
    """Diagnostics for confidence-weighted binary social propagation."""

    weighting: str
    peer_confidences: tuple[float, ...]
    effective_alphas: tuple[float, ...]
    peer_readiness: tuple[float, ...] = ()
    readiness_weight: float = 0.0

    def aggregate_row(self) -> dict[str, object]:
        peer_confidences = list(self.peer_confidences)
        effective_alphas = list(self.effective_alphas)
        peer_readiness = list(self.peer_readiness)
        return {
            "social_confidence_weighting": self.weighting,
            "mean_social_peer_confidence": (
                float(np.mean(peer_confidences)) if peer_confidences else 0.0
            ),
            "mean_social_effective_alpha": (
                float(np.mean(effective_alphas)) if effective_alphas else 0.0
            ),
            "max_social_effective_alpha": max(effective_alphas, default=0.0),
            "precommitment_social_feedback_enabled": self.readiness_weight > 0.0,
            "precommitment_social_feedback_weight": self.readiness_weight,
            "mean_social_peer_precommitment_readiness": (
                float(np.mean(peer_readiness)) if peer_readiness else 0.0
            ),
            "social_peer_precommitment_readiness_active_rate": (
                float(np.mean(np.asarray(peer_readiness) > 0.0))
                if peer_readiness
                else 0.0
            ),
        }

    def micro_row(self, agent_id: int) -> dict[str, object]:
        peer_readiness = (
            self.peer_readiness[agent_id]
            if len(self.peer_readiness) > agent_id
            else 0.0
        )
        return {
            "social_confidence_weighting": self.weighting,
            "social_peer_confidence": self.peer_confidences[agent_id],
            "social_effective_alpha": self.effective_alphas[agent_id],
            "social_peer_precommitment_readiness": peer_readiness,
        }


@dataclass(frozen=True)
class BinarySocialTailFloorDecision:
    """Effective confidence floor selected for near-ceiling social completion."""

    floor: float
    active: bool
    policy_rate: float
    action_rate: float

    def aggregate_row(self) -> dict[str, object]:
        return {
            "social_tail_floor_active": self.active,
            "social_tail_confidence_floor": self.floor,
            "social_tail_policy_rate": self.policy_rate,
            "social_tail_action_rate": self.action_rate,
        }


@dataclass(frozen=True)
class BinaryActionCommitmentResult:
    """Actions and diagnostics after optional binary commitment hysteresis."""

    actions: StateArray
    diagnostics: Mapping[str, object]


@dataclass
class BatchedDistributionDistillationAdapter:
    """Commit mixed distribution targets through the batched MLP backend."""

    agents: Sequence[Any]
    observations: torch.Tensor
    parameters: BatchedMLPParameters | None = None
    adam_state_cache: BatchedAdamStateCache | None = None
    loss_mode: str = "cross_entropy"
    timing_context: Any | None = None
    timing_prefix: str = "social"
    synchronize_model_parameters: bool = True
    synchronize_optimizer_states: bool = True
    update_result: BatchedMLPUpdateResult | None = field(default=None, init=False)

    def commit(self, mix_result: Any) -> CommitReport:
        update_parameters = (
            trainable_batched_mlp_parameters(
                self.agents,
                device=self.observations.device,
            )
            if self.parameters is None
            else self.parameters
        )
        active_agent_ids = (
            list(range(len(self.agents)))
            if mix_result.active_agent_ids is None
            else list(mix_result.active_agent_ids)
        )
        with timed_context_stage(self.timing_context, f"{self.timing_prefix}_loss_forward"):
            losses = batched_distribution_cross_entropy_losses(
                update_parameters,
                self.observations,
                mix_result.mixed_values,
                loss_mode=self.loss_mode,
            )
        with timed_context_stage(
            self.timing_context,
            f"{self.timing_prefix}_optimizer_update",
        ):
            timing_kwargs: dict[str, Any] = {"timing_prefix": self.timing_prefix}
            if self.timing_context is not None:
                timing_recorder = self.timing_context.extras.get("_record_timing")
                timing_synchronizer = self.timing_context.extras.get(
                    "_synchronize_timing_device",
                )
                if callable(timing_recorder):
                    timing_kwargs["timing_recorder"] = timing_recorder
                if callable(timing_synchronizer):
                    timing_kwargs["timing_synchronizer"] = timing_synchronizer
            self.update_result = apply_batched_mlp_loss_gradients_with_result(
                agents=self.agents,
                parameters=update_parameters,
                losses=losses,
                active_agent_ids=active_agent_ids,
                adam_state_cache=self.adam_state_cache,
                synchronize_model_parameters=self.synchronize_model_parameters,
                synchronize_optimizer_states=self.synchronize_optimizer_states,
                **timing_kwargs,
            )
        return CommitReport.from_mix_result(
            mix_result=mix_result,
            committed_agent_ids=[
                int(getattr(self.agents[agent_id], "agent_id", agent_id))
                for agent_id in active_agent_ids
            ],
            losses=list(self.update_result.losses),
        )


@dataclass
class TensorRuntimeDistributionDistillationAdapter:
    """Commit mixed distribution targets through a tensor-owned policy runtime."""

    runtime: Any
    observations: torch.Tensor
    loss_mode: str = "cross_entropy"
    timing_context: Any | None = None
    timing_prefix: str = "social"
    update_result: BatchedMLPUpdateResult | None = field(default=None, init=False)

    def commit(self, mix_result: Any) -> CommitReport:
        active_agent_ids = self._active_agent_ids(mix_result)
        with timed_context_stage(
            self.timing_context,
            f"{self.timing_prefix}_trainable_parameters",
        ):
            update_parameters = self.runtime.trainable_parameters()
        with timed_context_stage(self.timing_context, f"{self.timing_prefix}_loss_forward"):
            losses = batched_distribution_cross_entropy_losses(
                update_parameters,
                self.observations,
                mix_result.mixed_values,
                loss_mode=self.loss_mode,
            )
        with timed_context_stage(
            self.timing_context,
            f"{self.timing_prefix}_optimizer_update",
        ):
            timing_kwargs: dict[str, Any] = {"timing_prefix": self.timing_prefix}
            if self.timing_context is not None:
                timing_recorder = self.timing_context.extras.get("_record_timing")
                timing_synchronizer = self.timing_context.extras.get(
                    "_synchronize_timing_device",
                )
                if callable(timing_recorder):
                    timing_kwargs["timing_recorder"] = timing_recorder
                if callable(timing_synchronizer):
                    timing_kwargs["timing_synchronizer"] = timing_synchronizer
            self.update_result = self.runtime.apply_loss_gradients(
                update_parameters,
                losses,
                active_agent_ids=active_agent_ids,
                **timing_kwargs,
            )
        committed_ids = (
            list(range(int(self.runtime.agent_count)))
            if active_agent_ids is None
            else list(active_agent_ids)
        )
        return CommitReport.from_mix_result(
            mix_result=mix_result,
            committed_agent_ids=committed_ids,
            losses=list(self.update_result.losses),
        )

    def _active_agent_ids(self, mix_result: Any) -> list[int] | None:
        if mix_result.active_agent_ids is None:
            return []
        active_agent_ids = [int(agent_id) for agent_id in mix_result.active_agent_ids]
        if len(active_agent_ids) == int(self.runtime.agent_count) and all(
            agent_id == index for index, agent_id in enumerate(active_agent_ids)
        ):
            return None
        return active_agent_ids


@dataclass
class BatchedPolicyGradientLocalUpdateAdapter:
    """Commit binary policy-gradient losses through the batched MLP backend."""

    agents: Sequence[Any]
    observations: torch.Tensor
    actions: Any
    advantages: Any
    active_agent_ids: Sequence[int] | None
    entropy_beta: float
    parameters: BatchedMLPParameters | None = None
    adam_state_cache: BatchedAdamStateCache | None = None
    timing_context: Any | None = None
    timing_prefix: str = "local"
    synchronize_model_parameters: bool = True
    synchronize_optimizer_states: bool = True

    def update(self) -> LocalUpdateReport:
        update_parameters = (
            trainable_batched_mlp_parameters(
                self.agents,
                device=self.observations.device,
            )
            if self.parameters is None
            else self.parameters
        )
        with timed_context_stage(self.timing_context, f"{self.timing_prefix}_loss_forward"):
            losses = batched_binary_policy_gradient_losses(
                update_parameters,
                self.observations,
                actions=self.actions,
                advantages=self.advantages,
                entropy_beta=self.entropy_beta,
            )
        with timed_context_stage(
            self.timing_context,
            f"{self.timing_prefix}_optimizer_update",
        ):
            timing_kwargs: dict[str, Any] = {"timing_prefix": self.timing_prefix}
            if self.timing_context is not None:
                timing_recorder = self.timing_context.extras.get("_record_timing")
                timing_synchronizer = self.timing_context.extras.get(
                    "_synchronize_timing_device",
                )
                if callable(timing_recorder):
                    timing_kwargs["timing_recorder"] = timing_recorder
                if callable(timing_synchronizer):
                    timing_kwargs["timing_synchronizer"] = timing_synchronizer
            update_result = apply_batched_mlp_loss_gradients_with_result(
                agents=self.agents,
                parameters=update_parameters,
                losses=losses,
                active_agent_ids=self.active_agent_ids,
                adam_state_cache=self.adam_state_cache,
                synchronize_model_parameters=self.synchronize_model_parameters,
                synchronize_optimizer_states=self.synchronize_optimizer_states,
                **timing_kwargs,
            )
        return LocalUpdateReport(
            losses=update_result.losses,
            active_agent_ids=(
                None
                if self.active_agent_ids is None
                else [int(agent_id) for agent_id in self.active_agent_ids]
            ),
            update_result=update_result,
        )


@dataclass
class TensorRuntimePolicyGradientLocalUpdateAdapter:
    """Commit binary policy-gradient losses through a tensor-owned runtime."""

    runtime: Any
    observations: torch.Tensor
    actions: Any
    advantages: Any
    active_agent_ids: Sequence[int] | None
    entropy_beta: float
    timing_context: Any | None = None
    timing_prefix: str = "local"

    def update(self) -> LocalUpdateReport:
        with timed_context_stage(
            self.timing_context,
            f"{self.timing_prefix}_trainable_parameters",
        ):
            update_parameters = self.runtime.trainable_parameters()
        with timed_context_stage(self.timing_context, f"{self.timing_prefix}_loss_forward"):
            losses = batched_binary_policy_gradient_losses(
                update_parameters,
                self.observations,
                actions=self.actions,
                advantages=self.advantages,
                entropy_beta=self.entropy_beta,
            )
        with timed_context_stage(
            self.timing_context,
            f"{self.timing_prefix}_optimizer_update",
        ):
            timing_kwargs: dict[str, Any] = {"timing_prefix": self.timing_prefix}
            if self.timing_context is not None:
                timing_recorder = self.timing_context.extras.get("_record_timing")
                timing_synchronizer = self.timing_context.extras.get(
                    "_synchronize_timing_device",
                )
                if callable(timing_recorder):
                    timing_kwargs["timing_recorder"] = timing_recorder
                if callable(timing_synchronizer):
                    timing_kwargs["timing_synchronizer"] = timing_synchronizer
            update_result = self.runtime.apply_loss_gradients(
                update_parameters,
                losses,
                active_agent_ids=self.active_agent_ids,
                **timing_kwargs,
            )
        return LocalUpdateReport(
            losses=update_result.losses,
            active_agent_ids=(
                None
                if self.active_agent_ids is None
                else [int(agent_id) for agent_id in self.active_agent_ids]
            ),
            update_result=update_result,
        )


def run_batched_policy_gradient_local_update(
    *,
    agents: Sequence[Any],
    observations: torch.Tensor,
    actions: Any,
    advantages: Any,
    active_agent_ids: Sequence[int] | None,
    entropy_beta: float,
    parameters: BatchedMLPParameters | None = None,
    adam_state_cache: BatchedAdamStateCache | None = None,
    timing_context: Any | None = None,
    synchronize_model_parameters: bool = True,
    synchronize_optimizer_states: bool = True,
) -> LocalUpdateReport:
    """Run one batched binary policy-gradient local update via NABMLocalStep."""

    return NABMLocalStep(
        BatchedPolicyGradientLocalUpdateAdapter(
            agents=agents,
            observations=observations,
            actions=actions,
            advantages=advantages,
            active_agent_ids=active_agent_ids,
            entropy_beta=entropy_beta,
            parameters=parameters,
            adam_state_cache=adam_state_cache,
            timing_context=timing_context,
            synchronize_model_parameters=synchronize_model_parameters,
            synchronize_optimizer_states=synchronize_optimizer_states,
        ),
    ).run()


def run_tensor_runtime_policy_gradient_local_update(
    *,
    runtime: Any,
    observations: torch.Tensor,
    actions: Any,
    advantages: Any,
    active_agent_ids: Sequence[int] | None,
    entropy_beta: float,
    timing_context: Any | None = None,
) -> LocalUpdateReport:
    """Run one tensor-runtime binary policy-gradient update via NABMLocalStep."""

    return NABMLocalStep(
        TensorRuntimePolicyGradientLocalUpdateAdapter(
            runtime=runtime,
            observations=observations,
            actions=actions,
            advantages=advantages,
            active_agent_ids=active_agent_ids,
            entropy_beta=entropy_beta,
            timing_context=timing_context,
        ),
    ).run()


@dataclass(frozen=True)
class _SyntheticUnitAgent:
    """Minimal unit agent for tensor runtimes that keep state outside agents."""

    agent_id: int

    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec("synthetic", tensor_shape=None, dtype=None)

    def social_message_spec(self) -> SocialMessageSpec:
        return SocialMessageSpec()

    def observe(self, x: Any) -> Any:
        return x

    def act_or_predict(self, observation: Any) -> None:
        del observation
        return None

    def local_update(self, *args: Any, **kwargs: Any) -> float:
        del args, kwargs
        return 0.0

    def social_message(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "agent_id": self.agent_id,
            "latent_summary": torch.zeros(1),
            "confidence": 1.0,
            "param_norm": 0.0,
        }

    def log_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {"agent_id": self.agent_id}


def _unit_agents_for_distillation(
    agents: Sequence[Any],
    previous_probs: torch.Tensor,
    peer_ids: list[list[int]],
) -> Sequence[Any]:
    if agents:
        return agents
    return [
        _SyntheticUnitAgent(agent_id)
        for agent_id in range(_distillation_agent_count(agents, previous_probs, peer_ids))
    ]


def _distillation_agent_count(
    agents: Sequence[Any],
    previous_probs: torch.Tensor,
    peer_ids: list[list[int]],
) -> int:
    if agents:
        return len(agents)
    if hasattr(previous_probs, "shape") and len(previous_probs.shape) > 0:
        return int(previous_probs.shape[0])
    return len(peer_ids)


@dataclass
class BinaryPostStepStatePolicy:
    """Common post-step state updates requested by a hook-based domain."""

    payoff_ema_decay: float | None = None
    reputation_decay: float | None = None
    mobility_params: MobilityParams | None = None
    mobility_neighbors: list[list[int]] | None = None
    mobility_rng: np.random.Generator | None = None
    mobility_quality_signal: np.ndarray | None = None
    mobility_extra_state_arrays: MutableMapping[str, StateArray] | None = None
    mobility_extra_state_lists: MutableMapping[str, MutableSequence[Any]] | None = None


@dataclass
class BinaryToyResult:
    """Public result returned by Toy2/4/5 binary simulations."""

    run_dir: Path
    toy: str
    final_action_rate: float
    final_mean_payoff: float
    final_fragmentation_components: int
    final_mean_policy_action_probability: float
    final_mean_reputation: float
    final_reputation_dispersion: float
    domain_metrics: dict[str, object] = field(default_factory=dict)


BINARY_POLICY_PROBABILITY_THRESHOLDS = (
    ("0p5", 0.5),
    ("0p7", 0.7),
    ("0p9", 0.9),
)
BINARY_POLICY_PROBABILITY_QUANTILES = (
    ("p10", 0.1),
    ("p50", 0.5),
    ("p90", 0.9),
)


def binary_policy_probability_profile_fields(prefix: str) -> list[str]:
    """Return common aggregate fields for a binary action-probability profile."""

    return [
        *[
            f"{prefix}_gt_{label}_rate"
            for label, _threshold in BINARY_POLICY_PROBABILITY_THRESHOLDS
        ],
        f"{prefix}_dwell_0p4_0p6_rate",
        *[
            f"{prefix}_{label}"
            for label, _quantile in BINARY_POLICY_PROBABILITY_QUANTILES
        ],
    ]


BINARY_POLICY_TEMPORAL_FIELDS = [
    *[
        f"policy_probability_threshold_crossings_{label}_count"
        for label, _threshold in BINARY_POLICY_PROBABILITY_THRESHOLDS
    ],
    *[
        f"policy_probability_threshold_crossings_{label}_rate"
        for label, _threshold in BINARY_POLICY_PROBABILITY_THRESHOLDS
    ],
    "action_flip_count",
    "action_flip_rate",
]


REVISION_OPERATOR_AGGREGATE_FIELDS = [
    "revision_operator_enabled",
    "revision_operator_source",
    "mean_revision_stay_probability",
    "mean_revision_effective_stay_probability",
    "mean_revision_switch_probability",
    "mean_revision_switch_to_one_probability",
    "mean_revision_switch_to_zero_probability",
    "revision_choice_stay_rate",
    "revision_choice_switch_to_one_rate",
    "revision_choice_switch_to_zero_rate",
    "revision_operator_realized_switch_rate",
    "revision_operator_action_rate_after_revision",
    "mean_revision_operator_local_loss",
]


REVISION_OPERATOR_MICRO_FIELDS = [
    "revision_operator_enabled",
    "revision_operator_source",
    "revision_operator_revised",
    "revision_choice",
    "revision_stay_probability",
    "revision_effective_stay_probability",
    "revision_switch_probability",
    "revision_switch_to_one_probability",
    "revision_switch_to_zero_probability",
]


PRECOMMITMENT_TRAJECTORY_AGGREGATE_FIELDS = [
    "precommitment_first_ready_epoch",
    "precommitment_all_ready_epoch",
    "precommitment_first_forced_epoch",
    "precommitment_ready_to_forced_delay_mean",
    "precommitment_premature_exit_count",
    "precommitment_high_policy_rate",
    "precommitment_direction_score_mean",
    "precommitment_direction_score_positive_rate",
    "precommitment_direction_ok_rate",
    "precommitment_ready_largest_component_fraction",
    "precommitment_peer_evidence_enabled",
    "precommitment_peer_evidence_weight",
    "precommitment_peer_readiness_aggregation",
    "precommitment_peer_readiness_mean",
    "precommitment_peer_readiness_active_rate",
    "precommitment_peer_evidence_increment_mean",
]


PRECOMMITMENT_TRAJECTORY_MICRO_FIELDS = [
    "precommitment_evidence",
    "precommitment_ready",
    "precommitment_signal",
    "precommitment_high_policy",
    "precommitment_direction_ok",
    "precommitment_direction_score",
    "precommitment_forced_action",
    "precommitment_peer_readiness",
    "precommitment_peer_evidence_increment",
    "precommitment_first_ready_epoch",
    "precommitment_first_forced_epoch",
]


BINARY_AGGREGATE_COMMON_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "toy",
    "policy_rule",
    "coordination_mixer",
    "coordination_peer_rule",
    "policy_revision_rate",
    "realized_revision_rate",
    "action_rate",
    "mean_payoff",
    "mean_policy_action_probability",
    "mean_policy_action_probability_pre_revision",
    "mean_policy_action_probability_post_local",
    "mean_policy_action_probability_post_social",
    *binary_policy_probability_profile_fields(
        "policy_action_probability_pre_revision"
    ),
    *binary_policy_probability_profile_fields("policy_action_probability_post_local"),
    *binary_policy_probability_profile_fields("policy_action_probability_post_social"),
    *BINARY_POLICY_TEMPORAL_FIELDS,
    *REVISION_OPERATOR_AGGREGATE_FIELDS,
    "mean_local_loss",
    "mean_revised_local_loss",
    "mean_social_loss",
    "social_channel",
    "commit_mode",
    "mean_social_update_norm",
    "max_social_update_norm",
    "active_social_agent_count",
    "social_confidence_weighting",
    "social_tail_floor_active",
    "social_tail_confidence_floor",
    "social_tail_policy_rate",
    "social_tail_action_rate",
    "mean_social_peer_confidence",
    "mean_social_effective_alpha",
    "max_social_effective_alpha",
    "precommitment_social_feedback_enabled",
    "precommitment_social_feedback_weight",
    "mean_social_peer_precommitment_readiness",
    "social_peer_precommitment_readiness_active_rate",
    "commitment_enabled",
    "commitment_rate",
    "commitment_entry_count",
    "commitment_exit_count",
    "committed_action_rate",
    "uncommitted_high_policy_rate",
    "final_uncommitted_near_ceiling_count",
    "commitment_forced_action_count",
    "precommitment_enabled",
    "precommitment_rate",
    "precommitment_mean_evidence",
    "precommitment_ready_count",
    "precommitment_forced_action_count",
    "precommitment_signal_rate",
    *PRECOMMITMENT_TRAJECTORY_AGGREGATE_FIELDS,
    "precommitment_decision_feedback_enabled",
    "precommitment_decision_feedback_weight",
    "precommitment_decision_feedback_mean",
    "precommitment_decision_feedback_active_rate",
    "precommitment_decision_feedback_delta_mean",
    "mean_reputation",
    "reputation_dispersion",
    "mobility_rate",
    "mean_mobility_gain",
    "fragmentation_components",
    "mean_peer_count",
]


BINARY_MICRO_COMMON_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "agent_id",
    "toy",
    "policy_rule",
    "coordination_mixer",
    "coordination_peer_rule",
    "action",
    "action_probability",
    "policy_action_probability_pre_revision",
    "policy_action_probability_post_local",
    "policy_action_probability_post_social",
    "candidate_decision_action_probability_pre_revision",
    "realized_decision_action_probability",
    "payoff",
    "payoff_ema",
    "reputation",
    "peer_ids",
    "peer_count",
    "component_id",
    "revised",
    "local_loss",
    "social_loss",
    "social_channel",
    "commit_mode",
    "social_update_norm",
    "social_confidence_weighting",
    "social_peer_confidence",
    "social_effective_alpha",
    "social_peer_precommitment_readiness",
    *PRECOMMITMENT_TRAJECTORY_MICRO_FIELDS,
    *REVISION_OPERATOR_MICRO_FIELDS,
    "mobility_moved",
    "mobility_target",
    "mobility_gain",
]


TimingRows = MutableSequence[dict[str, object]]
TimingRecorder = Callable[[str, float], None]
TimingSynchronizer = Callable[[], None]


def timed_context_stage(
    context: BinaryStepContext | None,
    stage: str,
) -> "_ContextTimer":
    """Return a timer for optional domain-internal profiling."""

    return _ContextTimer(context=context, stage=stage)


class BinarySpatialDomain(Protocol):
    """Domain callbacks required by :class:`BinarySpatialRunner`.

    Toy adapters own domain-specific payoff/resource/exposure logic and row
    shapes. The runner owns the lifecycle, social coordination, action commit
    orchestration, common post-step state updates, and writer management.
    """

    micro_state_fields: list[str]
    aggregate_fields: list[str]

    def make_run_dir(self) -> Path:
        ...

    def write_metadata(self, run_dir: Path) -> None:
        ...

    def initial_state(self) -> BinarySpatialState:
        ...

    def initial_step_result(
        self,
        state: BinarySpatialState,
    ) -> BinaryPolicyStepResult:
        ...

    def build_step_context(
        self,
        epoch: int,
        state: BinarySpatialState,
        revision_mask: np.ndarray,
    ) -> BinaryStepContext:
        ...

    def local_step(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
    ) -> BinaryLocalStepResult:
        ...

    def select_peers(
        self,
        action_probs: np.ndarray,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
    ) -> list[list[int]]:
        ...

    def coordination_mixer(self) -> str:
        ...

    def coordination_alpha(self) -> float:
        ...

    def policy_tensor_from_action_probs(
        self,
        action_probs: np.ndarray,
        device_like: Any,
    ) -> Any:
        ...

    def sample_actions(
        self,
        state: BinarySpatialState,
        action_probs: np.ndarray,
        revision_mask: np.ndarray,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
    ) -> np.ndarray:
        ...

    def apply_social_distillation(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        peer_ids: list[list[int]],
    ) -> BinarySocialStepResult:
        ...

    def commit_actions(
        self,
        state: BinarySpatialState,
        actions: np.ndarray,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
    ) -> Mapping[str, Any]:
        ...

    def post_step_state_update(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
    ) -> BinaryPostStepStatePolicy:
        ...

    def finalize_hook_step(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
        mobility_result: MobilityStepResult,
    ) -> Mapping[str, Any]:
        ...

    def post_social_policy_update(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
        mobility_result: MobilityStepResult,
        post_social_probs: Any,
    ) -> Mapping[str, Any]:
        ...

    def aggregate_row(
        self,
        epoch: int,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> Mapping[str, Any]:
        ...

    def micro_rows(
        self,
        epoch: int,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> Iterable[Mapping[str, Any]]:
        ...

    def write_summary(
        self,
        run_dir: Path,
        final_row: Mapping[str, Any],
        state: BinarySpatialState,
    ) -> Any:
        ...


def sample_revision_mask(
    agent_count: int,
    revision_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a boolean per-agent revision mask."""

    if revision_rate <= 0.0:
        return np.zeros(agent_count, dtype=bool)
    if revision_rate >= 1.0:
        return np.ones(agent_count, dtype=bool)
    return rng.random(agent_count) < revision_rate


def empty_losses(agent_count: int) -> list[float]:
    """Return a zero loss vector for all agents."""

    if agent_count < 0:
        raise ValueError("agent_count must be non-negative")
    return [0.0 for _ in range(agent_count)]


def _contract_array(values: Any, *, name: str, dtype: Any | None = None) -> np.ndarray:
    try:
        return to_numpy_view(values, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be array-like") from exc


def validate_revision_mask(
    revision_mask: Any,
    agent_count: int,
    *,
    name: str = "revision_mask",
) -> None:
    """Require a one-dimensional boolean revision mask matching agent_count."""

    array = _contract_array(revision_mask, name=name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D bool array")
    if len(array) != agent_count:
        raise ValueError(
            f"{name} length ({len(array)}) must equal agent_count ({agent_count})"
        )
    if not np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must be a 1D bool array")


def validate_binary_action_probs(
    values: Any,
    agent_count: int,
    name: str,
) -> None:
    """Require a one-dimensional finite binary action probability vector."""

    array = _contract_array(values, name=name, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D probability vector")
    if len(array) != agent_count:
        raise ValueError(
            f"{name} length ({len(array)}) must equal agent_count ({agent_count})"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} values must lie in [0, 1]")


def validate_binary_actions(
    actions: Any,
    agent_count: int,
    *,
    name: str = "actions",
) -> None:
    """Require one binary integer/bool action per agent."""

    array = _contract_array(actions, name=name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D binary action vector")
    if len(array) != agent_count:
        raise ValueError(
            f"{name} length ({len(array)}) must equal agent_count ({agent_count})"
        )
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must contain integer/bool binary values")
    if not np.all(np.isin(array, [0, 1])):
        raise ValueError(f"{name} values must be binary (0 or 1)")


def validate_loss_vector(losses: Any, agent_count: int, name: str) -> None:
    """Require one finite numeric loss value per agent."""

    if isinstance(losses, TensorBackedLossVector):
        if len(losses) != agent_count:
            raise ValueError(
                f"{name} length ({len(losses)}) must equal agent_count ({agent_count})"
            )
        if not losses.isfinite_all():
            raise ValueError(f"{name} must contain only finite values")
        return

    array = _contract_array(losses, name=name, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D loss vector")
    if len(array) != agent_count:
        raise ValueError(
            f"{name} length ({len(array)}) must equal agent_count ({agent_count})"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")


def validate_step_extras(extras: Any, name: str) -> None:
    """Require hook extras/update payloads to be mappings."""

    if not isinstance(extras, Mapping):
        raise ValueError(f"{name} must be a mapping")


def validate_post_step_state_policy(policy: Any) -> None:
    """Require a domain-owned post-step update policy object."""

    if not isinstance(policy, BinaryPostStepStatePolicy):
        raise ValueError(
            "post_step_state_update result must be a BinaryPostStepStatePolicy"
        )


def realized_revision_rate(revision_mask: np.ndarray) -> float:
    """Return the fraction of agents marked for revision."""

    if len(revision_mask) == 0:
        return 0.0
    return float(np.mean(revision_mask))


def binary_policy_matrix(action_probs: np.ndarray) -> np.ndarray:
    """Return a two-column binary policy matrix from scalar action-1 probabilities."""

    clipped = np.clip(np.asarray(action_probs, dtype=np.float64), 0.0, 1.0)
    return np.column_stack([1.0 - clipped, clipped])


def mix_binary_output_average(
    action_probs: np.ndarray,
    peer_ids: list[list[int]],
    alpha: float,
) -> tuple[np.ndarray, list[float]]:
    """Mix a scalar binary action probability channel toward selected peers."""

    result = mix_scalar_probabilities(
        values=action_probs,
        peer_ids=peer_ids,
        alpha=alpha,
        channel="binary_action_probability",
        commit_mode="scalar_probability_sample",
    )
    return result.mixed_values, result.losses


def binary_policy_confidence(action_probs: np.ndarray) -> np.ndarray:
    """Return binary-policy confidence as distance from indifference."""

    values = np.asarray(action_probs, dtype=np.float64)
    validate_binary_action_probs(values, len(values), "action_probs")
    return np.clip(2.0 * np.abs(values - 0.5), 0.0, 1.0)


def binary_policy_confidence_weights(
    action_probs: np.ndarray,
    *,
    floor: float = 0.0,
    power: float = 1.0,
) -> np.ndarray:
    """Return bounded confidence weights for binary social messages."""

    if not 0.0 <= floor <= 1.0:
        raise ValueError("confidence_weight_floor must lie in [0, 1]")
    if power <= 0.0:
        raise ValueError("confidence_weight_power must be positive")
    confidence = binary_policy_confidence(action_probs)
    return floor + (1.0 - floor) * np.power(confidence, power)


def binary_policy_direction_alignment(
    action_probs: np.ndarray,
    direction_scores: np.ndarray,
) -> np.ndarray:
    """Return peer credibility from policy probability and signed objective direction."""

    values = np.asarray(action_probs, dtype=np.float64)
    validate_binary_action_probs(values, len(values), "action_probs")
    directions = np.asarray(direction_scores, dtype=np.float64)
    if directions.ndim != 1:
        raise ValueError("direction_scores must be a 1D vector")
    if len(directions) != len(values):
        raise ValueError("direction_scores must match action_probs length")
    if not np.all(np.isfinite(directions)):
        raise ValueError("direction_scores must contain only finite values")
    return np.where(
        directions > 0.0,
        values,
        np.where(directions < 0.0, 1.0 - values, 0.5),
    )


def mix_binary_output_confidence_weighted(
    action_probs: np.ndarray,
    peer_ids: list[list[int]],
    alpha: float,
    *,
    floor: float = 0.0,
    power: float = 1.0,
    direction_scores: np.ndarray | None = None,
    precommitment_readiness: np.ndarray | None = None,
    precommitment_readiness_weight: float = 0.0,
    weighting: str = "peer",
    channel: str = "binary_action_probability",
    commit_mode: str = "confidence_weighted_scalar_probability_sample",
) -> tuple[SocialMixResult, BinarySocialConfidenceDiagnostics]:
    """Mix binary action probabilities with peer confidence-scaled influence."""

    if weighting not in ("peer", "peer_direction"):
        raise ValueError("weighting must be one of: 'peer', 'peer_direction'")
    if weighting == "peer_direction" and direction_scores is None:
        raise ValueError("direction_scores are required for peer_direction weighting")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    original = np.asarray(action_probs, dtype=np.float64)
    validate_binary_action_probs(original, len(original), "action_probs")
    validate_peer_ids(peer_ids, len(original))
    weights = binary_policy_confidence_weights(original, floor=floor, power=power)
    if direction_scores is not None:
        weights = weights * binary_policy_direction_alignment(
            original,
            direction_scores,
        )
    readiness_values: np.ndarray | None = None
    if precommitment_readiness is not None and precommitment_readiness_weight > 0.0:
        if not 0.0 <= precommitment_readiness_weight <= 1.0:
            raise ValueError("precommitment_readiness_weight must lie in [0, 1]")
        readiness_values = np.asarray(precommitment_readiness, dtype=np.float64)
        if readiness_values.ndim != 1:
            raise ValueError("precommitment_readiness must be a 1D vector")
        if len(readiness_values) != len(original):
            raise ValueError("precommitment_readiness must match action_probs length")
        if not np.all(np.isfinite(readiness_values)):
            raise ValueError("precommitment_readiness must contain only finite values")
        readiness_values = np.clip(readiness_values, 0.0, 1.0)
        weights = weights + precommitment_readiness_weight * readiness_values * (
            1.0 - weights
        )

    mixed = original.copy()
    losses: list[float] = []
    update_norms: list[float] = []
    peer_confidences: list[float] = []
    peer_readiness_values: list[float] = []
    effective_alphas: list[float] = []
    active_agent_ids: list[int] = []
    for agent_id, peers in enumerate(peer_ids):
        if not peers or alpha == 0.0:
            peer_confidences.append(0.0)
            peer_readiness_values.append(0.0)
            effective_alphas.append(0.0)
            losses.append(0.0)
            update_norms.append(0.0)
            continue
        peer_weights = weights[peers]
        peer_confidence = float(np.mean(peer_weights)) if len(peer_weights) else 0.0
        peer_readiness = (
            float(np.mean(readiness_values[peers]))
            if readiness_values is not None and len(peer_weights)
            else 0.0
        )
        weight_sum = float(np.sum(peer_weights))
        effective_alpha = alpha * peer_confidence
        if weight_sum <= 0.0 or effective_alpha <= 0.0:
            effective_alpha = 0.0
        else:
            peer_mean = float(np.dot(peer_weights, original[peers]) / weight_sum)
            mixed_value = (
                (1.0 - effective_alpha) * float(original[agent_id])
                + effective_alpha * peer_mean
            )
            mixed[agent_id] = float(np.clip(mixed_value, 0.0, 1.0))
            active_agent_ids.append(agent_id)
        update_norm = abs(float(mixed[agent_id] - original[agent_id]))
        peer_confidences.append(peer_confidence)
        peer_readiness_values.append(peer_readiness)
        effective_alphas.append(effective_alpha)
        losses.append(update_norm)
        update_norms.append(update_norm)

    return (
        SocialMixResult(
            mixed_values=mixed,
            losses=losses,
            update_norms=update_norms,
            peer_ids=[list(peers) for peers in peer_ids],
            channel=channel,
            commit_mode=commit_mode,
            active_agent_ids=active_agent_ids,
        ),
        BinarySocialConfidenceDiagnostics(
            weighting=weighting,
            peer_confidences=tuple(peer_confidences),
            effective_alphas=tuple(effective_alphas),
            peer_readiness=tuple(peer_readiness_values),
            readiness_weight=precommitment_readiness_weight,
        ),
    )


def _policy_distribution_target_like(
    action_probs: np.ndarray,
    previous_probs: Any,
) -> Any:
    policy_matrix = binary_policy_matrix(action_probs)
    if isinstance(previous_probs, torch.Tensor):
        return torch.as_tensor(
            policy_matrix,
            dtype=previous_probs.dtype,
            device=previous_probs.device,
        )
    return policy_matrix


def _policy_distribution_update_norms(
    mixed_values: Any,
    previous_probs: Any,
) -> list[float]:
    if isinstance(mixed_values, torch.Tensor) and isinstance(previous_probs, torch.Tensor):
        norms = torch.linalg.vector_norm(
            mixed_values.detach() - previous_probs.detach(),
            dim=1,
        )
        return [float(value) for value in norms.detach().cpu().tolist()]
    mixed_array = np.asarray(mixed_values, dtype=np.float64)
    previous_array = np.asarray(previous_probs, dtype=np.float64)
    return [
        float(value)
        for value in np.linalg.norm(mixed_array - previous_array, axis=1).tolist()
    ]


def mix_binary_policy_distribution_confidence_weighted(
    previous_probs: Any,
    peer_ids: list[list[int]],
    alpha: float,
    *,
    floor: float = 0.0,
    power: float = 1.0,
    direction_scores: np.ndarray | None = None,
    precommitment_readiness: np.ndarray | None = None,
    precommitment_readiness_weight: float = 0.0,
    weighting: str = "peer",
    channel: str = "policy_distribution",
    commit_mode: str = "confidence_weighted_distillation_step",
) -> tuple[SocialMixResult, BinarySocialConfidenceDiagnostics]:
    """Build confidence-weighted binary policy-distribution distillation targets."""

    action_probs = binary_action_probability_values(previous_probs)
    scalar_result, confidence_diagnostics = mix_binary_output_confidence_weighted(
        action_probs=action_probs,
        peer_ids=peer_ids,
        alpha=alpha,
        floor=floor,
        power=power,
        direction_scores=direction_scores,
        precommitment_readiness=precommitment_readiness,
        precommitment_readiness_weight=precommitment_readiness_weight,
        weighting=weighting,
        channel="binary_action_probability",
        commit_mode="confidence_weighted_scalar_probability_sample",
    )
    mixed_values = _policy_distribution_target_like(
        scalar_result.mixed_values,
        previous_probs,
    )
    update_norms = _policy_distribution_update_norms(mixed_values, previous_probs)
    return (
        SocialMixResult(
            mixed_values=mixed_values,
            losses=update_norms,
            update_norms=update_norms,
            peer_ids=scalar_result.peer_ids,
            channel=channel,
            commit_mode=commit_mode,
            active_agent_ids=scalar_result.active_agent_ids,
        ),
        confidence_diagnostics,
    )


def apply_binary_output_distribution_distillation(
    *,
    agents: Sequence[Any],
    observations: torch.Tensor,
    peer_ids: list[list[int]],
    alpha: float,
    previous_probs: torch.Tensor,
    logits_fn: Callable[[Any, int, torch.Tensor], torch.Tensor] | None = None,
    loss_mode: str = "cross_entropy",
    optimizer_fn: Callable[[Any, int], torch.optim.Optimizer] | None = None,
    commit_adapter: Any | None = None,
    confidence_weighting: str = "none",
    confidence_weight_floor: float = 0.0,
    confidence_weight_power: float = 1.0,
    social_direction_scores: np.ndarray | None = None,
    precommitment_readiness: np.ndarray | None = None,
    precommitment_readiness_weight: float = 0.0,
) -> LossVector:
    """Apply a NABMUnit-backed binary policy-distribution distillation step."""

    return run_binary_output_distribution_distillation(
        agents=agents,
        observations=observations,
        peer_ids=peer_ids,
        alpha=alpha,
        previous_probs=previous_probs,
        logits_fn=logits_fn,
        loss_mode=loss_mode,
        optimizer_fn=optimizer_fn,
        commit_adapter=commit_adapter,
        confidence_weighting=confidence_weighting,
        confidence_weight_floor=confidence_weight_floor,
        confidence_weight_power=confidence_weight_power,
        social_direction_scores=social_direction_scores,
        precommitment_readiness=precommitment_readiness,
        precommitment_readiness_weight=precommitment_readiness_weight,
    ).social_losses


def run_binary_output_distribution_distillation(
    *,
    agents: Sequence[Any],
    observations: torch.Tensor,
    peer_ids: list[list[int]],
    alpha: float,
    previous_probs: torch.Tensor,
    logits_fn: Callable[[Any, int, torch.Tensor], torch.Tensor] | None = None,
    loss_mode: str = "cross_entropy",
    optimizer_fn: Callable[[Any, int], torch.optim.Optimizer] | None = None,
    commit_adapter: Any | None = None,
    confidence_weighting: str = "none",
    confidence_weight_floor: float = 0.0,
    confidence_weight_power: float = 1.0,
    social_direction_scores: np.ndarray | None = None,
    precommitment_readiness: np.ndarray | None = None,
    precommitment_readiness_weight: float = 0.0,
) -> BinaryOutputDistillationReport:
    """Run binary policy distillation and return reusable unit diagnostics."""

    if confidence_weighting not in ("none", "peer", "peer_direction"):
        raise ValueError(
            "confidence_weighting must be one of: 'none', 'peer', 'peer_direction'"
        )
    unit_agents = _unit_agents_for_distillation(agents, previous_probs, peer_ids)
    if alpha == 0.0:
        return BinaryOutputDistillationReport(
            social_losses=empty_losses(len(unit_agents)),
        )

    resolved_commit_adapter = commit_adapter
    if resolved_commit_adapter is None:
        if logits_fn is None:
            raise ValueError("logits_fn is required when commit_adapter is not provided")
        optimizer_for_agent = optimizer_fn or (lambda agent, _agent_id: agent.optimizer)
        resolved_commit_adapter = DistributionDistillationAdapter(
            agents=agents,
            logits_fn=lambda agent, agent_id: logits_fn(
                agent,
                agent_id,
                observations,
            ),
            optimizer_fn=optimizer_for_agent,
            loss_mode=loss_mode,
        )
    if confidence_weighting in ("peer", "peer_direction"):
        mix_result, confidence_diagnostics = (
            mix_binary_policy_distribution_confidence_weighted(
                previous_probs=previous_probs,
                peer_ids=peer_ids,
                alpha=alpha,
                floor=confidence_weight_floor,
                power=confidence_weight_power,
                direction_scores=social_direction_scores,
                precommitment_readiness=precommitment_readiness,
                precommitment_readiness_weight=precommitment_readiness_weight,
                weighting=confidence_weighting,
            )
        )
        commit_report = resolved_commit_adapter.commit(mix_result)
        diagnostics_result = SocialMixResult(
            mixed_values=mix_result.mixed_values,
            losses=list(commit_report.losses),
            update_norms=list(mix_result.update_norms),
            peer_ids=mix_result.peer_ids,
            channel=mix_result.channel,
            commit_mode=mix_result.commit_mode,
            active_agent_ids=mix_result.active_agent_ids,
        )
        diagnostics = social_diagnostics(diagnostics_result)
        aggregate_diagnostics = diagnostics.aggregate_row()
        aggregate_diagnostics.update(confidence_diagnostics.aggregate_row())
        micro_diagnostics = []
        for agent_id in range(len(unit_agents)):
            row = diagnostics.micro_row(agent_id)
            row.update(confidence_diagnostics.micro_row(agent_id))
            micro_diagnostics.append(row)
        return BinaryOutputDistillationReport(
            social_losses=commit_report.losses,
            aggregate_diagnostics=aggregate_diagnostics,
            micro_diagnostics=micro_diagnostics,
        )

    step = NABMStep(
        social_block=SocialBlock(alpha=alpha),
        channel=SocialChannel(
            name="policy_distribution",
            kind=PROBABILITY_DISTRIBUTION_CHANNEL,
            commit_mode="distillation_step",
        ),
        commit_adapter=resolved_commit_adapter,
    )
    unit = NABMUnit(
        agents=unit_agents,
        step=step,
        peer_selector=lambda _messages: peer_ids,
        social_value_builder=lambda _agents, _messages: previous_probs,
    )
    unit_report = unit.run(
        message_args=(observations,),
        run_local_update=False,
        collect_logs=False,
    )
    return BinaryOutputDistillationReport.from_unit_report(unit_report)


def select_binary_output_similarity_peers(
    *,
    neighbors: list[list[int]],
    action_probs: np.ndarray,
    peer_rule: str,
    threshold: float,
    error_label: str,
    copy_peers: bool = True,
    validate_peers: bool = True,
) -> list[list[int]]:
    """Select peers from scalar binary action probabilities."""

    try:
        return select_scalar_output_peers(
            neighbors=neighbors,
            values=action_probs,
            peer_rule=peer_rule,
            threshold=threshold,
            copy_peers=copy_peers,
            validate_peers=validate_peers,
        ).peer_ids
    except ValueError as exc:
        if "Unsupported peer rule" in str(exc):
            raise ValueError(f"Unsupported {error_label} peer rule: {peer_rule}") from exc
        raise


def peer_ids_for_binary_mixer(
    *,
    peer_ids: list[list[int]],
    mixer: str,
    agent_count: int,
    error_label: str,
    copy_peers: bool = True,
    validate_peers: bool = True,
) -> list[list[int]]:
    """Return active peer ids for a binary spatial mixer."""

    try:
        return social_peer_ids_for_mixer(
            peer_ids=peer_ids,
            mixer=mixer,
            agent_count=agent_count,
            copy_peers=copy_peers,
            validate_peers=validate_peers,
        )
    except ValueError as exc:
        if "Unsupported mixer" in str(exc):
            raise ValueError(f"Unsupported {error_label} mixer: {mixer}") from exc
        raise


def policy_tensor_from_action_probs(
    action_probs: np.ndarray,
    device_like: Any,
    converter: Callable[[np.ndarray, Any], Any],
) -> Any:
    """Convert scalar action probabilities with a domain-owned tensor hook."""

    return converter(action_probs, device_like)


def binary_action_probs_from_policy(policy_probs: Any) -> np.ndarray:
    """Extract the action-1 probability from a binary policy tensor-like value."""

    if hasattr(policy_probs, "detach"):
        return policy_probs[:, 1].detach().cpu().numpy()
    return np.asarray(policy_probs, dtype=np.float64)[:, 1]


def binary_action_probability_values(policy_probs: Any) -> np.ndarray:
    """Return action-1 probabilities from a vector or binary policy matrix."""

    if hasattr(policy_probs, "detach"):
        array = policy_probs.detach().cpu().numpy()
    else:
        array = np.asarray(policy_probs, dtype=np.float64)
    if array.ndim == 1:
        return np.asarray(array, dtype=np.float64)
    return np.asarray(array[:, 1], dtype=np.float64)


def binary_policy_prob(policy_probs: Any, agent_id: int) -> float:
    """Return one agent's action-1 probability from a binary policy-like value."""

    value = policy_probs[agent_id, 1]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return float(value)


def binary_action_probability(policy_probs: Any, agent_id: int) -> float:
    """Return an action-1 probability from either a vector or binary policy matrix."""

    if hasattr(policy_probs, "detach"):
        if policy_probs.ndim == 1:
            value = policy_probs[agent_id].detach().cpu()
            return float(value)
        return binary_policy_prob(policy_probs, agent_id)
    array = np.asarray(policy_probs, dtype=np.float64)
    if array.ndim == 1:
        return float(array[agent_id])
    return binary_policy_prob(array, agent_id)


def mean_binary_policy_prob(policy_probs: Any) -> float:
    """Return the mean action-1 probability from a binary policy-like value."""

    values = binary_action_probability_values(policy_probs)
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def binary_policy_probability_profile(
    policy_probs: Any,
    *,
    prefix: str,
) -> dict[str, object]:
    """Return threshold, dwell-band, and quantile diagnostics for policy probs."""

    values = binary_action_probability_values(policy_probs)
    if len(values) == 0:
        return {
            field: 0.0
            for field in binary_policy_probability_profile_fields(prefix)
        }
    return {
        **{
            f"{prefix}_gt_{label}_rate": float(np.mean(values > threshold))
            for label, threshold in BINARY_POLICY_PROBABILITY_THRESHOLDS
        },
        f"{prefix}_dwell_0p4_0p6_rate": float(
            np.mean((values >= 0.4) & (values <= 0.6))
        ),
        **{
            f"{prefix}_{label}": float(np.quantile(values, quantile))
            for label, quantile in BINARY_POLICY_PROBABILITY_QUANTILES
        },
    }


def binary_policy_temporal_metrics(
    current_policy_probs: Any,
    previous_policy_probs: Any | None,
    *,
    current_actions: Any,
    previous_actions: Any | None,
) -> dict[str, object]:
    """Return temporal threshold-crossing and action-flip diagnostics."""

    metrics: dict[str, object] = {
        **{
            f"policy_probability_threshold_crossings_{label}_count": ""
            for label, _threshold in BINARY_POLICY_PROBABILITY_THRESHOLDS
        },
        **{
            f"policy_probability_threshold_crossings_{label}_rate": ""
            for label, _threshold in BINARY_POLICY_PROBABILITY_THRESHOLDS
        },
        "action_flip_count": "",
        "action_flip_rate": "",
    }
    current_values = binary_action_probability_values(current_policy_probs)
    if previous_policy_probs is not None:
        previous_values = binary_action_probability_values(previous_policy_probs)
        if len(previous_values) != len(current_values):
            raise ValueError(
                "previous_policy_probs must have the same agent count as "
                "current_policy_probs"
            )
        denominator = max(len(current_values), 1)
        for label, threshold in BINARY_POLICY_PROBABILITY_THRESHOLDS:
            crossing_count = int(
                np.sum((previous_values > threshold) != (current_values > threshold))
            )
            metrics[f"policy_probability_threshold_crossings_{label}_count"] = (
                crossing_count
            )
            metrics[f"policy_probability_threshold_crossings_{label}_rate"] = float(
                crossing_count / denominator
            )
    if previous_actions is not None:
        current_action_values = to_numpy_view(current_actions, dtype=np.int64)
        previous_action_values = np.asarray(previous_actions, dtype=np.int64)
        if len(previous_action_values) != len(current_action_values):
            raise ValueError(
                "previous_actions must have the same agent count as current actions"
            )
        denominator = max(len(current_action_values), 1)
        flip_count = int(np.sum(previous_action_values != current_action_values))
        metrics["action_flip_count"] = flip_count
        metrics["action_flip_rate"] = float(flip_count / denominator)
    return metrics


def mean_binary_policy_prob_triplet(
    policy_probs: Any,
    policy_probs_pre_revision: Any | None,
    policy_probs_post_local: Any | None,
) -> tuple[float, float, float]:
    """Return post-social, pre-revision, and post-local action-1 means."""

    pre_revision = (
        policy_probs
        if policy_probs_pre_revision is None
        else policy_probs_pre_revision
    )
    post_local = (
        policy_probs
        if policy_probs_post_local is None
        else policy_probs_post_local
    )
    tensors = (policy_probs, pre_revision, post_local)
    if all(hasattr(values, "detach") for values in tensors):
        try:
            means = torch.stack(
                [values[:, 1].mean() for values in tensors],
            ).detach()
            mean_values = means.cpu().tolist()
            return (
                float(mean_values[0]),
                float(mean_values[1]),
                float(mean_values[2]),
            )
        except (IndexError, RuntimeError, TypeError):
            pass
    mean_policy, mean_policy_pre, mean_policy_post_local = (
        mean_binary_policy_prob_triplet(
            policy_probs,
            policy_probs_pre_revision,
            policy_probs_post_local,
        )
    )
    return mean_policy, mean_policy_pre, mean_policy_post_local


def detached_policy_probs(policy_probs: Any) -> Any:
    """Detach tensor policy probabilities when supported."""

    if hasattr(policy_probs, "detach"):
        return policy_probs.detach()
    return policy_probs


def binary_policy_social_result(
    *,
    peer_ids: list[list[int]],
    post_social_probs: Any,
    social_losses: LossVector,
    extras: Mapping[str, Any] | None = None,
) -> BinarySocialStepResult:
    """Build a social step result from a binary policy distribution."""

    return BinarySocialStepResult(
        peer_ids=peer_ids,
        post_social_probs=post_social_probs,
        final_action_probs=binary_action_probs_from_policy(post_social_probs),
        social_losses=social_losses,
        extras=dict(extras or {}),
    )


def distill_binary_policy_output_average(
    *,
    agents: list[Any],
    observations: Any,
    peer_ids: list[list[int]],
    alpha: float,
    previous_probs: Any,
    temperature: float,
    collect_policy_probs: Callable[..., Any],
    distill_policy: Callable[..., LossVector | BinaryOutputDistillationReport],
    refresh_policy_cache: Callable[[list[Any]], None] | None = None,
    agent_count: int | None = None,
    skip_when_alpha_zero: bool = False,
    context: BinaryStepContext | None = None,
    confidence_weighting: str = "none",
    confidence_weight_floor: float = 0.0,
    confidence_weight_power: float = 1.0,
    social_direction_scores: np.ndarray | None = None,
    precommitment_readiness: np.ndarray | None = None,
    precommitment_readiness_weight: float = 0.0,
) -> BinarySocialStepResult:
    """Apply neural output-average distillation and collect binary policy readout."""

    social_extras: dict[str, object] = {}
    if skip_when_alpha_zero and alpha == 0.0:
        social_losses = empty_losses(len(agents) if agent_count is None else agent_count)
    else:
        distillation_result = distill_policy(
            agents=agents,
            observations=observations,
            peer_ids=peer_ids,
            alpha=alpha,
            previous_probs=detached_policy_probs(previous_probs),
            context=context,
            confidence_weighting=confidence_weighting,
            confidence_weight_floor=confidence_weight_floor,
            confidence_weight_power=confidence_weight_power,
            social_direction_scores=social_direction_scores,
            precommitment_readiness=precommitment_readiness,
            precommitment_readiness_weight=precommitment_readiness_weight,
        )
        if isinstance(distillation_result, BinaryOutputDistillationReport):
            social_losses = distillation_result.social_losses
            social_extras = distillation_result.social_result_extras()
        else:
            social_losses = distillation_result
        if refresh_policy_cache is not None:
            refresh_policy_cache(agents)
    post_social_probs = collect_policy_probs(
        agents,
        observations,
        temperature=temperature,
    )
    return binary_policy_social_result(
        peer_ids=peer_ids,
        post_social_probs=post_social_probs,
        social_losses=social_losses,
        extras=social_extras,
    )


def public_step_extras(extras: Mapping[str, Any]) -> dict[str, Any]:
    """Return extras intended for downstream row/summary code."""

    return {
        str(key): value
        for key, value in extras.items()
        if not str(key).startswith("_")
    }


def mean_loss(losses: Sequence[float] | None) -> float:
    """Return the mean loss, treating missing or empty loss vectors as zero."""

    return mean_loss_value(losses)


def binary_loss_metrics(
    *,
    local_losses: Sequence[float] | None = None,
    social_losses: Sequence[float] | None = None,
    revised_local_losses: Sequence[float] | None = None,
) -> dict[str, float]:
    """Return common aggregate loss metrics for binary spatial domains."""

    metrics = {
        "mean_local_loss": mean_loss(local_losses),
        "mean_social_loss": mean_loss(social_losses),
    }
    if revised_local_losses is not None:
        metrics["mean_revised_local_loss"] = mean_loss(revised_local_losses)
    return metrics


def binary_peer_metrics(
    *,
    peer_ids: list[list[int]],
    agent_count: int,
    include_edge_entropy: bool = False,
) -> dict[str, float | int]:
    """Return common aggregate metrics from active binary social peers."""

    peer_graph = graph_from_peer_ids(agent_count, peer_ids)
    peer_counts = [len(peers) for peers in peer_ids]
    metrics: dict[str, float | int] = {
        "fragmentation_components": nx.number_connected_components(peer_graph),
        "mean_peer_count": float(np.mean(peer_counts)) if peer_counts else 0.0,
    }
    if include_edge_entropy:
        metrics["edge_entropy"] = edge_entropy(peer_ids, agent_count)
    return metrics


def binary_peer_component_map(
    *,
    peer_ids: list[list[int]],
    agent_count: int,
) -> dict[int, int]:
    """Return connected component ids from active binary social peers."""

    return component_map(graph_from_peer_ids(agent_count, peer_ids))


def binary_reputation_metrics(
    reputation: StateArray | None,
) -> dict[str, float | str]:
    """Return common reputation metrics with CSV-friendly empty values."""

    if reputation is None:
        return {"mean_reputation": "", "reputation_dispersion": ""}
    return reputation_summary(to_numpy_view(reputation, dtype=np.float64))


def binary_mobility_metrics(
    mobility_result: MobilityStepResult | None,
) -> dict[str, float]:
    """Return common aggregate mobility metrics with no-mobility defaults."""

    if mobility_result is None:
        return {"mobility_rate": 0.0, "mean_mobility_gain": 0.0}
    return mobility_summary(mobility_result)


def _epoch_value_or_empty(value: object) -> int | float | str:
    """Return a CSV-friendly epoch value, preserving empty unset markers."""

    numeric = float(value)
    if not np.isfinite(numeric):
        return ""
    integer_value = int(numeric)
    if numeric == float(integer_value):
        return integer_value
    return numeric


def binary_ready_largest_component_fraction(
    *,
    peer_ids: list[list[int]],
    ready: np.ndarray,
) -> float:
    """Return the population share in the largest ready peer component."""

    ready_values = np.asarray(ready, dtype=bool)
    agent_count = len(ready_values)
    if agent_count == 0 or not bool(np.any(ready_values)):
        return 0.0
    graph = graph_from_peer_ids(agent_count, peer_ids)
    ready_nodes = [int(index) for index in np.flatnonzero(ready_values)]
    ready_graph = graph.subgraph(ready_nodes)
    largest = max(
        (len(component) for component in nx.connected_components(ready_graph)),
        default=0,
    )
    return float(largest / agent_count)


def binary_precommitment_micro_fields(
    state: BinarySpatialState,
    agent_id: int,
) -> dict[str, object]:
    """Return per-agent precommitment trajectory diagnostics."""

    def array_value(
        key: str,
        *,
        dtype: Any,
        default: object,
    ) -> object:
        values = state.extras.get(key)
        if values is None:
            return default
        array = np.asarray(values, dtype=dtype)
        if len(array) != state.agent_count:
            raise ValueError(f"{key} length mismatch")
        return array[agent_id]

    first_ready = array_value(
        "_binary_action_precommitment_first_ready_epoch",
        dtype=np.float64,
        default=np.nan,
    )
    first_forced = array_value(
        "_binary_action_precommitment_first_forced_epoch",
        dtype=np.float64,
        default=np.nan,
    )
    return {
        "precommitment_evidence": float(
            array_value(
                "_binary_action_precommitment_evidence",
                dtype=np.float64,
                default=0.0,
            )
        ),
        "precommitment_ready": bool(
            array_value(
                "_binary_action_precommitment_ready",
                dtype=bool,
                default=False,
            )
        ),
        "precommitment_signal": bool(
            array_value(
                "_binary_action_precommitment_signal",
                dtype=bool,
                default=False,
            )
        ),
        "precommitment_high_policy": bool(
            array_value(
                "_binary_action_precommitment_high_policy",
                dtype=bool,
                default=False,
            )
        ),
        "precommitment_direction_ok": bool(
            array_value(
                "_binary_action_precommitment_direction_ok",
                dtype=bool,
                default=False,
            )
        ),
        "precommitment_direction_score": float(
            array_value(
                "_binary_action_precommitment_direction_score",
                dtype=np.float64,
                default=0.0,
            )
        ),
        "precommitment_forced_action": bool(
            array_value(
                "_binary_action_precommitment_forced_action",
                dtype=bool,
                default=False,
            )
        ),
        "precommitment_peer_readiness": float(
            array_value(
                "_binary_action_precommitment_peer_readiness",
                dtype=np.float64,
                default=0.0,
            )
        ),
        "precommitment_peer_evidence_increment": float(
            array_value(
                "_binary_action_precommitment_peer_evidence_increment",
                dtype=np.float64,
                default=0.0,
            )
        ),
        "precommitment_first_ready_epoch": _epoch_value_or_empty(first_ready),
        "precommitment_first_forced_epoch": _epoch_value_or_empty(first_forced),
    }


def binary_micro_common_fields(
    *,
    run_id: str,
    seed: int,
    epoch: int,
    agent_id: int,
    coordination_mixer: str,
    coordination_peer_rule: str,
    peer_ids: list[list[int]],
    components: Mapping[int, int],
    revision_mask: np.ndarray,
    local_losses: Sequence[float],
    social_losses: Sequence[float],
    social_unit_micro: Mapping[str, object] | None = None,
    precommitment_micro: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return common per-agent CSV fields for binary spatial domains."""

    social_diagnostics = dict(social_unit_micro or {})
    precommitment_diagnostics = dict(precommitment_micro or {})
    return {
        "run_id": run_id,
        "seed": seed,
        "epoch": epoch,
        "agent_id": agent_id,
        "coordination_mixer": coordination_mixer,
        "coordination_peer_rule": coordination_peer_rule,
        "peer_ids": peer_ids[agent_id],
        "peer_count": len(peer_ids[agent_id]),
        "component_id": components.get(agent_id, -1),
        "revised": bool(revision_mask[agent_id]),
        "local_loss": local_losses[agent_id],
        "social_loss": social_losses[agent_id],
        "social_channel": social_diagnostics.get("social_channel", ""),
        "commit_mode": social_diagnostics.get("commit_mode", ""),
        "social_update_norm": social_diagnostics.get("social_update_norm", 0.0),
        "social_confidence_weighting": social_diagnostics.get(
            "social_confidence_weighting",
            "none",
        ),
        "social_peer_confidence": social_diagnostics.get(
            "social_peer_confidence",
            0.0,
        ),
        "social_effective_alpha": social_diagnostics.get(
            "social_effective_alpha",
            0.0,
        ),
        "social_peer_precommitment_readiness": social_diagnostics.get(
            "social_peer_precommitment_readiness",
            0.0,
        ),
        "precommitment_evidence": precommitment_diagnostics.get(
            "precommitment_evidence",
            0.0,
        ),
        "precommitment_ready": precommitment_diagnostics.get(
            "precommitment_ready",
            False,
        ),
        "precommitment_signal": precommitment_diagnostics.get(
            "precommitment_signal",
            False,
        ),
        "precommitment_high_policy": precommitment_diagnostics.get(
            "precommitment_high_policy",
            False,
        ),
        "precommitment_direction_ok": precommitment_diagnostics.get(
            "precommitment_direction_ok",
            False,
        ),
        "precommitment_forced_action": precommitment_diagnostics.get(
            "precommitment_forced_action",
            False,
        ),
        "precommitment_peer_readiness": precommitment_diagnostics.get(
            "precommitment_peer_readiness",
            0.0,
        ),
        "precommitment_peer_evidence_increment": precommitment_diagnostics.get(
            "precommitment_peer_evidence_increment",
            0.0,
        ),
        "precommitment_first_ready_epoch": precommitment_diagnostics.get(
            "precommitment_first_ready_epoch",
            "",
        ),
        "precommitment_first_forced_epoch": precommitment_diagnostics.get(
            "precommitment_first_forced_epoch",
            "",
        ),
    }


def revision_operator_aggregate_fields(
    diagnostics: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return common aggregate fields for the optional revision operator."""

    values = dict(diagnostics or {})
    return {
        "revision_operator_enabled": values.get("revision_operator_enabled", False),
        "revision_operator_source": values.get("revision_operator_source", ""),
        "mean_revision_stay_probability": values.get(
            "mean_revision_stay_probability",
            0.0,
        ),
        "mean_revision_effective_stay_probability": values.get(
            "mean_revision_effective_stay_probability",
            0.0,
        ),
        "mean_revision_switch_probability": values.get(
            "mean_revision_switch_probability",
            0.0,
        ),
        "mean_revision_switch_to_one_probability": values.get(
            "mean_revision_switch_to_one_probability",
            0.0,
        ),
        "mean_revision_switch_to_zero_probability": values.get(
            "mean_revision_switch_to_zero_probability",
            0.0,
        ),
        "revision_choice_stay_rate": values.get("revision_choice_stay_rate", 0.0),
        "revision_choice_switch_to_one_rate": values.get(
            "revision_choice_switch_to_one_rate",
            0.0,
        ),
        "revision_choice_switch_to_zero_rate": values.get(
            "revision_choice_switch_to_zero_rate",
            0.0,
        ),
        "revision_operator_realized_switch_rate": values.get(
            "revision_operator_realized_switch_rate",
            values.get("realized_revision_rate", 0.0),
        ),
        "revision_operator_action_rate_after_revision": values.get(
            "revision_operator_action_rate_after_revision",
            values.get("action_rate_after_revision", 0.0),
        ),
        "mean_revision_operator_local_loss": values.get(
            "mean_revision_operator_local_loss",
            values.get("mean_revision_local_loss", 0.0),
        ),
    }


def revision_operator_micro_fields(
    diagnostics: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return common per-agent fields for the optional revision operator."""

    values = dict(diagnostics or {})
    return {
        "revision_operator_enabled": values.get("revision_operator_enabled", False),
        "revision_operator_source": values.get("revision_operator_source", ""),
        "revision_operator_revised": values.get("revised", ""),
        "revision_choice": values.get("revision_choice", ""),
        "revision_stay_probability": values.get("revision_stay_probability", 0.0),
        "revision_effective_stay_probability": values.get(
            "revision_effective_stay_probability",
            0.0,
        ),
        "revision_switch_probability": values.get("revision_switch_probability", 0.0),
        "revision_switch_to_one_probability": values.get(
            "revision_switch_to_one_probability",
            0.0,
        ),
        "revision_switch_to_zero_probability": values.get(
            "revision_switch_to_zero_probability",
            0.0,
        ),
    }


def binary_micro_mobility_fields(
    mobility_result: MobilityStepResult,
    agent_id: int,
) -> dict[str, object]:
    """Return common per-agent mobility CSV fields."""

    moved = bool(mobility_result.moved[agent_id])
    return {
        "mobility_moved": moved,
        "mobility_target": int(mobility_result.targets[agent_id]) if moved else "",
        "mobility_gain": float(mobility_result.gains[agent_id]),
    }


def binary_aggregate_common_fields(
    *,
    config: Any,
    toy: str,
    epoch: int,
    actions: StateArray,
    payoffs: StateArray,
    policy_probs: Any,
    peer_ids: list[list[int]],
    realized_revision_rate: float,
    reputation: StateArray | None,
    mobility_result: MobilityStepResult | None,
    policy_probs_pre_revision: Any | None = None,
    policy_probs_post_local: Any | None = None,
    policy_probs_previous_post_social: Any | None = None,
    previous_actions: Any | None = None,
    local_losses: Sequence[float] | None = None,
    revised_local_losses: Sequence[float] | None = None,
    social_losses: Sequence[float] | None = None,
    include_edge_entropy: bool = False,
    peer_metrics: Mapping[str, float | int] | None = None,
    social_unit_aggregate: Mapping[str, object] | None = None,
    revision_operator_aggregate: Mapping[str, object] | None = None,
    commitment_aggregate: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return public generic aggregate fields for binary toy simulations."""

    social_diagnostics = dict(social_unit_aggregate or {})
    commitment_diagnostics = dict(commitment_aggregate or {})
    mean_policy = mean_binary_policy_prob(policy_probs)
    mean_policy_pre = (
        mean_binary_policy_prob(policy_probs_pre_revision)
        if policy_probs_pre_revision is not None
        else mean_policy
    )
    mean_policy_post_local = (
        mean_binary_policy_prob(policy_probs_post_local)
        if policy_probs_post_local is not None
        else mean_policy
    )
    resolved_pre_revision_probs = (
        policy_probs_pre_revision
        if policy_probs_pre_revision is not None
        else policy_probs
    )
    resolved_post_local_probs = (
        policy_probs_post_local
        if policy_probs_post_local is not None
        else policy_probs
    )
    resolved_peer_metrics = (
        binary_peer_metrics(
            peer_ids=peer_ids,
            agent_count=len(actions),
            include_edge_entropy=include_edge_entropy,
        )
        if peer_metrics is None
        else peer_metrics
    )
    return {
        "run_id": config.run.name,
        "seed": config.run.seed,
        "epoch": epoch,
        "toy": toy,
        "policy_rule": config.policy.rule,
        "coordination_mixer": config.coordination.mixer,
        "coordination_peer_rule": config.coordination.peer_rule,
        "policy_revision_rate": config.policy.revision_rate,
        "realized_revision_rate": realized_revision_rate,
        "action_rate": mean_value(actions),
        "mean_payoff": mean_value(payoffs),
        "mean_policy_action_probability": mean_policy,
        "mean_policy_action_probability_pre_revision": mean_policy_pre,
        "mean_policy_action_probability_post_local": mean_policy_post_local,
        "mean_policy_action_probability_post_social": mean_policy,
        **binary_policy_probability_profile(
            resolved_pre_revision_probs,
            prefix="policy_action_probability_pre_revision",
        ),
        **binary_policy_probability_profile(
            resolved_post_local_probs,
            prefix="policy_action_probability_post_local",
        ),
        **binary_policy_probability_profile(
            policy_probs,
            prefix="policy_action_probability_post_social",
        ),
        **binary_policy_temporal_metrics(
            policy_probs,
            policy_probs_previous_post_social,
            current_actions=actions,
            previous_actions=previous_actions,
        ),
        **revision_operator_aggregate_fields(revision_operator_aggregate),
        **binary_loss_metrics(
            local_losses=local_losses,
            revised_local_losses=(
                [] if revised_local_losses is None else revised_local_losses
            ),
            social_losses=social_losses,
        ),
        "social_channel": social_diagnostics.get("social_channel", ""),
        "commit_mode": social_diagnostics.get("commit_mode", ""),
        "mean_social_update_norm": social_diagnostics.get(
            "mean_social_update_norm",
            0.0,
        ),
        "max_social_update_norm": social_diagnostics.get(
            "max_social_update_norm",
            0.0,
        ),
        "active_social_agent_count": social_diagnostics.get(
            "active_social_agent_count",
            0,
        ),
        "social_confidence_weighting": social_diagnostics.get(
            "social_confidence_weighting",
            "none",
        ),
        "social_tail_floor_active": social_diagnostics.get(
            "social_tail_floor_active",
            False,
        ),
        "social_tail_confidence_floor": social_diagnostics.get(
            "social_tail_confidence_floor",
            0.0,
        ),
        "social_tail_policy_rate": social_diagnostics.get(
            "social_tail_policy_rate",
            0.0,
        ),
        "social_tail_action_rate": social_diagnostics.get(
            "social_tail_action_rate",
            0.0,
        ),
        "mean_social_peer_confidence": social_diagnostics.get(
            "mean_social_peer_confidence",
            0.0,
        ),
        "mean_social_effective_alpha": social_diagnostics.get(
            "mean_social_effective_alpha",
            0.0,
        ),
        "max_social_effective_alpha": social_diagnostics.get(
            "max_social_effective_alpha",
            0.0,
        ),
        "precommitment_social_feedback_enabled": social_diagnostics.get(
            "precommitment_social_feedback_enabled",
            False,
        ),
        "precommitment_social_feedback_weight": social_diagnostics.get(
            "precommitment_social_feedback_weight",
            0.0,
        ),
        "mean_social_peer_precommitment_readiness": social_diagnostics.get(
            "mean_social_peer_precommitment_readiness",
            0.0,
        ),
        "social_peer_precommitment_readiness_active_rate": social_diagnostics.get(
            "social_peer_precommitment_readiness_active_rate",
            0.0,
        ),
        "commitment_enabled": commitment_diagnostics.get(
            "commitment_enabled",
            False,
        ),
        "commitment_rate": commitment_diagnostics.get("commitment_rate", 0.0),
        "commitment_entry_count": commitment_diagnostics.get(
            "commitment_entry_count",
            0,
        ),
        "commitment_exit_count": commitment_diagnostics.get(
            "commitment_exit_count",
            0,
        ),
        "committed_action_rate": commitment_diagnostics.get(
            "committed_action_rate",
            0.0,
        ),
        "uncommitted_high_policy_rate": commitment_diagnostics.get(
            "uncommitted_high_policy_rate",
            0.0,
        ),
        "final_uncommitted_near_ceiling_count": commitment_diagnostics.get(
            "final_uncommitted_near_ceiling_count",
            0,
        ),
        "commitment_forced_action_count": commitment_diagnostics.get(
            "commitment_forced_action_count",
            0,
        ),
        "precommitment_enabled": commitment_diagnostics.get(
            "precommitment_enabled",
            False,
        ),
        "precommitment_rate": commitment_diagnostics.get(
            "precommitment_rate",
            0.0,
        ),
        "precommitment_mean_evidence": commitment_diagnostics.get(
            "precommitment_mean_evidence",
            0.0,
        ),
        "precommitment_ready_count": commitment_diagnostics.get(
            "precommitment_ready_count",
            0,
        ),
        "precommitment_forced_action_count": commitment_diagnostics.get(
            "precommitment_forced_action_count",
            0,
        ),
        "precommitment_signal_rate": commitment_diagnostics.get(
            "precommitment_signal_rate",
            0.0,
        ),
        "precommitment_first_ready_epoch": commitment_diagnostics.get(
            "precommitment_first_ready_epoch",
            "",
        ),
        "precommitment_all_ready_epoch": commitment_diagnostics.get(
            "precommitment_all_ready_epoch",
            "",
        ),
        "precommitment_first_forced_epoch": commitment_diagnostics.get(
            "precommitment_first_forced_epoch",
            "",
        ),
        "precommitment_ready_to_forced_delay_mean": commitment_diagnostics.get(
            "precommitment_ready_to_forced_delay_mean",
            "",
        ),
        "precommitment_premature_exit_count": commitment_diagnostics.get(
            "precommitment_premature_exit_count",
            0,
        ),
        "precommitment_high_policy_rate": commitment_diagnostics.get(
            "precommitment_high_policy_rate",
            0.0,
        ),
        "precommitment_direction_score_mean": commitment_diagnostics.get(
            "precommitment_direction_score_mean",
            0.0,
        ),
        "precommitment_direction_score_positive_rate": commitment_diagnostics.get(
            "precommitment_direction_score_positive_rate",
            0.0,
        ),
        "precommitment_direction_ok_rate": commitment_diagnostics.get(
            "precommitment_direction_ok_rate",
            0.0,
        ),
        "precommitment_ready_largest_component_fraction": (
            commitment_diagnostics.get(
                "precommitment_ready_largest_component_fraction",
                0.0,
            )
        ),
        "precommitment_peer_evidence_enabled": commitment_diagnostics.get(
            "precommitment_peer_evidence_enabled",
            False,
        ),
        "precommitment_peer_evidence_weight": commitment_diagnostics.get(
            "precommitment_peer_evidence_weight",
            0.0,
        ),
        "precommitment_peer_readiness_aggregation": commitment_diagnostics.get(
            "precommitment_peer_readiness_aggregation",
            "mean",
        ),
        "precommitment_peer_readiness_mean": commitment_diagnostics.get(
            "precommitment_peer_readiness_mean",
            0.0,
        ),
        "precommitment_peer_readiness_active_rate": commitment_diagnostics.get(
            "precommitment_peer_readiness_active_rate",
            0.0,
        ),
        "precommitment_peer_evidence_increment_mean": commitment_diagnostics.get(
            "precommitment_peer_evidence_increment_mean",
            0.0,
        ),
        "precommitment_decision_feedback_enabled": commitment_diagnostics.get(
            "precommitment_decision_feedback_enabled",
            False,
        ),
        "precommitment_decision_feedback_weight": commitment_diagnostics.get(
            "precommitment_decision_feedback_weight",
            0.0,
        ),
        "precommitment_decision_feedback_mean": commitment_diagnostics.get(
            "precommitment_decision_feedback_mean",
            0.0,
        ),
        "precommitment_decision_feedback_active_rate": commitment_diagnostics.get(
            "precommitment_decision_feedback_active_rate",
            0.0,
        ),
        "precommitment_decision_feedback_delta_mean": commitment_diagnostics.get(
            "precommitment_decision_feedback_delta_mean",
            0.0,
        ),
        **binary_reputation_metrics(reputation),
        **binary_mobility_metrics(mobility_result),
        "fragmentation_components": resolved_peer_metrics["fragmentation_components"],
        "mean_peer_count": resolved_peer_metrics["mean_peer_count"],
        **(
            {"edge_entropy": resolved_peer_metrics["edge_entropy"]}
            if include_edge_entropy
            else {}
        ),
    }


def binary_micro_base_fields(
    *,
    config: Any,
    toy: str,
    epoch: int,
    agent_id: int,
    state: BinarySpatialState,
    step_result: BinaryPolicyStepResult,
    components: Mapping[int, int],
    action_probability_source: Any | None = None,
    decision_probability_source: Any | None = None,
) -> dict[str, object]:
    """Return public generic micro fields for one binary toy agent."""

    social_unit_micro: Mapping[str, object] | None = None
    social_unit_micro_rows = step_result.extras.get("social_unit_micro")
    if (
        isinstance(social_unit_micro_rows, Sequence)
        and not isinstance(social_unit_micro_rows, (str, bytes))
        and agent_id < len(social_unit_micro_rows)
    ):
        candidate_social_unit_micro = social_unit_micro_rows[agent_id]
        if isinstance(candidate_social_unit_micro, Mapping):
            social_unit_micro = candidate_social_unit_micro
    revision_operator_micro: Mapping[str, object] | None = None
    revision_operator_micro_rows = step_result.extras.get("revision_operator_micro")
    if (
        isinstance(revision_operator_micro_rows, Sequence)
        and not isinstance(revision_operator_micro_rows, (str, bytes))
        and agent_id < len(revision_operator_micro_rows)
    ):
        candidate_revision_operator_micro = revision_operator_micro_rows[agent_id]
        if isinstance(candidate_revision_operator_micro, Mapping):
            revision_operator_micro = candidate_revision_operator_micro
    action_probs = (
        step_result.post_social_probs
        if action_probability_source is None
        else action_probability_source
    )
    decision_probs = (
        step_result.post_local_probs
        if decision_probability_source is None
        else decision_probability_source
    )
    realized_decision_probability: float | str = ""
    if bool(step_result.revision_mask[agent_id]):
        realized_decision_probability = binary_action_probability(
            decision_probs,
            agent_id,
        )
    return {
        **binary_micro_common_fields(
            run_id=config.run.name,
            seed=config.run.seed,
            epoch=epoch,
            agent_id=agent_id,
            coordination_mixer=config.coordination.mixer,
            coordination_peer_rule=config.coordination.peer_rule,
            peer_ids=step_result.peer_ids,
            components=components,
            revision_mask=step_result.revision_mask,
            local_losses=step_result.local_losses,
            social_losses=step_result.social_losses,
            social_unit_micro=social_unit_micro,
            precommitment_micro=binary_precommitment_micro_fields(state, agent_id),
        ),
        "toy": toy,
        "policy_rule": config.policy.rule,
        "action": int(scalar_at(state.actions, agent_id)),
        "action_probability": binary_action_probability(action_probs, agent_id),
        "policy_action_probability_pre_revision": binary_action_probability(
            step_result.pre_revision_probs,
            agent_id,
        ),
        "policy_action_probability_post_local": binary_action_probability(
            step_result.post_local_probs,
            agent_id,
        ),
        "policy_action_probability_post_social": binary_action_probability(
            step_result.post_social_probs,
            agent_id,
        ),
        "candidate_decision_action_probability_pre_revision": (
            binary_action_probability(decision_probs, agent_id)
        ),
        "realized_decision_action_probability": realized_decision_probability,
        "payoff": float(scalar_at(state.payoffs, agent_id)),
        "payoff_ema": float(scalar_at(state.payoff_ema, agent_id)),
        "reputation": float(scalar_at(state.reputation, agent_id)),
        **revision_operator_micro_fields(revision_operator_micro),
        **binary_micro_mobility_fields(step_result.mobility_result, agent_id),
    }


def _summary_float(value: object, default: float = 0.0) -> float:
    if value == "" or value is None:
        return default
    return float(value)


def _summary_value(value: object) -> object:
    if value == "":
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class BinaryToyDomainBase:
    """Shared adapter methods for Toy2/4/5 binary-action domains."""

    toy: str = "binary"
    include_edge_entropy: bool = False

    def coordination_mixer(self) -> str:
        return self.config.coordination.mixer

    def coordination_alpha(self) -> float:
        return self.config.coordination.alpha

    def coordination_confidence_weighting(self) -> str:
        return getattr(self.config.coordination, "confidence_weighting", "none")

    def coordination_confidence_weight_floor(self) -> float:
        return float(
            getattr(self.config.coordination, "confidence_weight_floor", 0.0)
        )

    def coordination_confidence_weight_power(self) -> float:
        return float(
            getattr(self.config.coordination, "confidence_weight_power", 1.0)
        )

    def coordination_confidence_tail_floor(self) -> float:
        return float(getattr(self.config.coordination, "confidence_tail_floor", 0.0))

    def coordination_confidence_tail_min_policy_rate(self) -> float:
        return float(
            getattr(
                self.config.coordination,
                "confidence_tail_min_policy_rate",
                1.0,
            )
        )

    def coordination_confidence_tail_min_action_rate(self) -> float:
        return float(
            getattr(
                self.config.coordination,
                "confidence_tail_min_action_rate",
                1.0,
            )
        )

    def coordination_commitment_enabled(self) -> bool:
        return bool(getattr(self.config.coordination, "commitment_enabled", False))

    def coordination_commitment_min_policy_probability(self) -> float:
        return float(
            getattr(
                self.config.coordination,
                "commitment_min_policy_probability",
                1.0,
            )
        )

    def coordination_commitment_min_action_streak(self) -> int:
        return int(
            getattr(
                self.config.coordination,
                "commitment_min_action_streak",
                1,
            )
        )

    def coordination_commitment_requires_direction(self) -> bool:
        return bool(
            getattr(
                self.config.coordination,
                "commitment_requires_direction",
                True,
            )
        )

    def coordination_commitment_min_direction(self) -> float:
        return float(
            getattr(
                self.config.coordination,
                "commitment_min_direction",
                0.0,
            )
        )

    def coordination_commitment_exit_policy_probability(self) -> float:
        return float(
            getattr(
                self.config.coordination,
                "commitment_exit_policy_probability",
                0.0,
            )
        )

    def coordination_commitment_exit_on_negative_direction(self) -> bool:
        return bool(
            getattr(
                self.config.coordination,
                "commitment_exit_on_negative_direction",
                True,
            )
        )

    def coordination_precommitment_enabled(self) -> bool:
        return bool(getattr(self.config.coordination, "precommitment_enabled", False))

    def coordination_precommitment_min_policy_probability(self) -> float:
        return float(
            getattr(
                self.config.coordination,
                "precommitment_min_policy_probability",
                1.0,
            )
        )

    def coordination_precommitment_min_evidence(self) -> float:
        return float(
            getattr(
                self.config.coordination,
                "precommitment_min_evidence",
                1.0,
            )
        )

    def coordination_precommitment_evidence_increment(self) -> float:
        return float(
            getattr(
                self.config.coordination,
                "precommitment_evidence_increment",
                1.0,
            )
        )

    def coordination_precommitment_evidence_decay(self) -> float:
        return float(
            getattr(
                self.config.coordination,
                "precommitment_evidence_decay",
                0.0,
            )
        )

    def coordination_precommitment_requires_direction(self) -> bool:
        return bool(
            getattr(
                self.config.coordination,
                "precommitment_requires_direction",
                True,
            )
        )

    def coordination_precommitment_min_direction(self) -> float:
        return float(
            getattr(
                self.config.coordination,
                "precommitment_min_direction",
                0.0,
            )
        )

    def coordination_precommitment_decision_feedback_enabled(self) -> bool:
        return bool(
            getattr(
                self.config.coordination,
                "precommitment_decision_feedback_enabled",
                False,
            )
        )

    def coordination_precommitment_decision_feedback_weight(self) -> float:
        return float(
            getattr(
                self.config.coordination,
                "precommitment_decision_feedback_weight",
                0.0,
            )
        )

    def coordination_precommitment_social_feedback_enabled(self) -> bool:
        return bool(
            getattr(
                self.config.coordination,
                "precommitment_social_feedback_enabled",
                False,
            )
        )

    def coordination_precommitment_social_feedback_weight(self) -> float:
        if not self.coordination_precommitment_social_feedback_enabled():
            return 0.0
        return float(
            getattr(
                self.config.coordination,
                "precommitment_social_feedback_weight",
                0.0,
            )
        )

    def coordination_precommitment_peer_evidence_enabled(self) -> bool:
        return bool(
            getattr(
                self.config.coordination,
                "precommitment_peer_evidence_enabled",
                False,
            )
        )

    def coordination_precommitment_peer_evidence_weight(self) -> float:
        if not self.coordination_precommitment_peer_evidence_enabled():
            return 0.0
        return float(
            getattr(
                self.config.coordination,
                "precommitment_peer_evidence_weight",
                0.0,
            )
        )

    def coordination_precommitment_peer_readiness_aggregation(self) -> str:
        return str(
            getattr(
                self.config.coordination,
                "precommitment_peer_readiness_aggregation",
                "mean",
            )
        )

    def precommitment_readiness_propagation_unit(
        self,
    ) -> BinaryReadinessPropagationUnit:
        return BinaryReadinessPropagationUnit(
            enabled=self.coordination_precommitment_peer_evidence_enabled(),
            weight=self.coordination_precommitment_peer_evidence_weight(),
            aggregation=self.coordination_precommitment_peer_readiness_aggregation(),
        )

    def social_direction_scores(
        self,
        local_result: BinaryLocalStepResult,
    ) -> np.ndarray | None:
        components = local_result.extras.get("state_continuation_components")
        if components is None or not hasattr(components, "effective"):
            return None
        return np.asarray(components.effective, dtype=np.float64)

    def precommitment_direction_scores(
        self,
        state: BinarySpatialState,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
        action_probs: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray | None:
        del state, social_result, action_probs, active
        return self.social_direction_scores(local_result)

    def social_tail_floor_decision(
        self,
        local_result: BinaryLocalStepResult,
    ) -> BinarySocialTailFloorDecision:
        base_floor = self.coordination_confidence_weight_floor()
        policy_rate = mean_binary_policy_prob(local_result.post_local_probs)
        if local_result.actions_after_revision is None:
            action_rate = float(np.mean(local_result.candidate_action_probs))
        else:
            action_rate = mean_value(local_result.actions_after_revision)
        tail_floor = self.coordination_confidence_tail_floor()
        if (
            self.coordination_confidence_weighting() not in ("peer", "peer_direction")
            or tail_floor <= base_floor
        ):
            return BinarySocialTailFloorDecision(
                floor=base_floor,
                active=False,
                policy_rate=policy_rate,
                action_rate=action_rate,
            )
        active = (
            policy_rate >= self.coordination_confidence_tail_min_policy_rate()
            or action_rate >= self.coordination_confidence_tail_min_action_rate()
        )
        return BinarySocialTailFloorDecision(
            floor=max(base_floor, tail_floor) if active else base_floor,
            active=active,
            policy_rate=policy_rate,
            action_rate=action_rate,
        )

    def _commitment_arrays(
        self,
        state: BinarySpatialState,
    ) -> tuple[np.ndarray, np.ndarray]:
        agent_count = state.agent_count
        active = state.extras.get("_binary_action_commitment_active")
        streaks = state.extras.get("_binary_action_commitment_streaks")
        if active is None:
            active = np.zeros(agent_count, dtype=bool)
            state.extras["_binary_action_commitment_active"] = active
        if streaks is None:
            streaks = np.zeros(agent_count, dtype=np.int64)
            state.extras["_binary_action_commitment_streaks"] = streaks
        active_array = np.asarray(active, dtype=bool)
        streak_array = np.asarray(streaks, dtype=np.int64)
        if len(active_array) != agent_count:
            raise ValueError("binary action commitment active state length mismatch")
        if len(streak_array) != agent_count:
            raise ValueError("binary action commitment streak state length mismatch")
        state.extras["_binary_action_commitment_active"] = active_array
        state.extras["_binary_action_commitment_streaks"] = streak_array
        return active_array, streak_array

    def _precommitment_evidence_array(
        self,
        state: BinarySpatialState,
    ) -> np.ndarray:
        agent_count = state.agent_count
        evidence = state.extras.get("_binary_action_precommitment_evidence")
        if evidence is None:
            evidence = np.zeros(agent_count, dtype=np.float64)
            state.extras["_binary_action_precommitment_evidence"] = evidence
        evidence_array = np.asarray(evidence, dtype=np.float64)
        if len(evidence_array) != agent_count:
            raise ValueError("binary action precommitment evidence length mismatch")
        state.extras["_binary_action_precommitment_evidence"] = evidence_array
        return evidence_array

    def _precommitment_tracking_arrays(
        self,
        state: BinarySpatialState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        agent_count = state.agent_count
        first_ready = state.extras.get(
            "_binary_action_precommitment_first_ready_epoch"
        )
        first_forced = state.extras.get(
            "_binary_action_precommitment_first_forced_epoch"
        )
        previous_ready = state.extras.get("_binary_action_precommitment_previous_ready")
        if first_ready is None:
            first_ready = np.full(agent_count, np.nan, dtype=np.float64)
            state.extras[
                "_binary_action_precommitment_first_ready_epoch"
            ] = first_ready
        if first_forced is None:
            first_forced = np.full(agent_count, np.nan, dtype=np.float64)
            state.extras[
                "_binary_action_precommitment_first_forced_epoch"
            ] = first_forced
        if previous_ready is None:
            previous_ready = np.zeros(agent_count, dtype=bool)
            state.extras["_binary_action_precommitment_previous_ready"] = (
                previous_ready
            )
        first_ready_array = np.asarray(first_ready, dtype=np.float64)
        first_forced_array = np.asarray(first_forced, dtype=np.float64)
        previous_ready_array = np.asarray(previous_ready, dtype=bool)
        if len(first_ready_array) != agent_count:
            raise ValueError("binary action precommitment first-ready length mismatch")
        if len(first_forced_array) != agent_count:
            raise ValueError("binary action precommitment first-forced length mismatch")
        if len(previous_ready_array) != agent_count:
            raise ValueError(
                "binary action precommitment previous-ready length mismatch"
            )
        state.extras["_binary_action_precommitment_first_ready_epoch"] = (
            first_ready_array
        )
        state.extras["_binary_action_precommitment_first_forced_epoch"] = (
            first_forced_array
        )
        state.extras["_binary_action_precommitment_previous_ready"] = (
            previous_ready_array
        )
        return first_ready_array, first_forced_array, previous_ready_array

    def _precommitment_decision_feedback_scores(
        self,
        state: BinarySpatialState,
    ) -> np.ndarray:
        evidence = state.extras.get("_binary_action_precommitment_evidence")
        if evidence is None:
            scores = np.zeros(state.agent_count, dtype=np.float64)
        else:
            evidence_values = np.asarray(evidence, dtype=np.float64)
            if len(evidence_values) != state.agent_count:
                raise ValueError("binary action precommitment evidence length mismatch")
            min_evidence = self.coordination_precommitment_min_evidence()
            if min_evidence <= 0.0:
                scores = (evidence_values > 0.0).astype(np.float64)
            else:
                scores = np.clip(evidence_values / min_evidence, 0.0, 1.0)
        active = state.extras.get("_binary_action_commitment_active")
        if active is not None:
            active_values = np.asarray(active, dtype=bool)
            if len(active_values) != state.agent_count:
                raise ValueError("binary action commitment active state length mismatch")
            scores = np.maximum(scores, active_values.astype(np.float64))
        return scores

    def precommitment_social_feedback_scores(
        self,
        state: BinarySpatialState,
    ) -> np.ndarray | None:
        if self.coordination_precommitment_social_feedback_weight() <= 0.0:
            return None
        return self._precommitment_decision_feedback_scores(state)

    def apply_precommitment_decision_feedback(
        self,
        state: BinarySpatialState,
        action_probs: Any,
    ) -> Any:
        """Blend prior precommitment readiness into decision-time probabilities."""

        enabled = self.coordination_precommitment_decision_feedback_enabled()
        weight = self.coordination_precommitment_decision_feedback_weight()
        if not enabled or weight <= 0.0:
            state.extras["_binary_precommitment_decision_feedback_diagnostics"] = {
                "precommitment_decision_feedback_enabled": enabled,
                "precommitment_decision_feedback_weight": weight,
                "precommitment_decision_feedback_mean": 0.0,
                "precommitment_decision_feedback_active_rate": 0.0,
                "precommitment_decision_feedback_delta_mean": 0.0,
            }
            return action_probs
        scores = self._precommitment_decision_feedback_scores(state)
        feedback = np.clip(scores * weight, 0.0, 1.0)
        if isinstance(action_probs, torch.Tensor):
            if action_probs.ndim != 2 or action_probs.shape[1] != 2:
                raise ValueError(
                    "precommitment decision feedback expects binary probabilities"
                )
            feedback_tensor = torch.as_tensor(
                feedback,
                dtype=action_probs.dtype,
                device=action_probs.device,
            )
            action1 = action_probs[:, 1]
            adjusted_action1 = action1 + feedback_tensor * (1.0 - action1)
            adjusted = torch.stack((1.0 - adjusted_action1, adjusted_action1), dim=1)
            delta_values = (
                adjusted_action1.detach().cpu().numpy()
                - action1.detach().cpu().numpy()
            )
        else:
            values = np.asarray(action_probs, dtype=np.float64).copy()
            if values.ndim == 1:
                action1 = values
                adjusted_action1 = action1 + feedback * (1.0 - action1)
                adjusted = adjusted_action1
            elif values.ndim == 2 and values.shape[1] == 2:
                action1 = values[:, 1]
                adjusted_action1 = action1 + feedback * (1.0 - action1)
                values[:, 0] = 1.0 - adjusted_action1
                values[:, 1] = adjusted_action1
                adjusted = values
            else:
                raise ValueError(
                    "precommitment decision feedback expects binary probabilities"
                )
            delta_values = adjusted_action1 - action1
        state.extras["_binary_precommitment_decision_feedback_diagnostics"] = {
            "precommitment_decision_feedback_enabled": enabled,
            "precommitment_decision_feedback_weight": weight,
            "precommitment_decision_feedback_mean": (
                float(np.mean(feedback)) if len(feedback) else 0.0
            ),
            "precommitment_decision_feedback_active_rate": (
                float(np.mean(feedback > 0.0)) if len(feedback) else 0.0
            ),
            "precommitment_decision_feedback_delta_mean": (
                float(np.mean(delta_values)) if len(delta_values) else 0.0
            ),
        }
        return adjusted

    def apply_action_commitment(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
        actions: StateArray,
    ) -> BinaryActionCommitmentResult:
        action_values = to_numpy_view(actions, dtype=np.int64).copy()
        commitment_enabled = self.coordination_commitment_enabled()
        precommitment_enabled = self.coordination_precommitment_enabled()
        if not commitment_enabled and not precommitment_enabled:
            return BinaryActionCommitmentResult(
                actions=actions,
                diagnostics={
                    "commitment_enabled": False,
                    "commitment_rate": 0.0,
                    "commitment_entry_count": 0,
                    "commitment_exit_count": 0,
                    "committed_action_rate": 0.0,
                    "uncommitted_high_policy_rate": 0.0,
                    "final_uncommitted_near_ceiling_count": 0,
                    "commitment_forced_action_count": 0,
                    "precommitment_enabled": False,
                    "precommitment_rate": 0.0,
                    "precommitment_mean_evidence": 0.0,
                    "precommitment_ready_count": 0,
                    "precommitment_forced_action_count": 0,
                    "precommitment_signal_rate": 0.0,
                    "precommitment_first_ready_epoch": "",
                    "precommitment_all_ready_epoch": "",
                    "precommitment_first_forced_epoch": "",
                    "precommitment_ready_to_forced_delay_mean": "",
                    "precommitment_premature_exit_count": 0,
                    "precommitment_high_policy_rate": 0.0,
                    "precommitment_direction_score_mean": 0.0,
                    "precommitment_direction_score_positive_rate": 0.0,
                    "precommitment_direction_ok_rate": 0.0,
                    "precommitment_ready_largest_component_fraction": 0.0,
                    "precommitment_peer_evidence_enabled": (
                        self.coordination_precommitment_peer_evidence_enabled()
                    ),
                    "precommitment_peer_evidence_weight": (
                        self.coordination_precommitment_peer_evidence_weight()
                    ),
                    "precommitment_peer_readiness_aggregation": (
                        self.coordination_precommitment_peer_readiness_aggregation()
                    ),
                    "precommitment_peer_readiness_mean": 0.0,
                    "precommitment_peer_readiness_active_rate": 0.0,
                    "precommitment_peer_evidence_increment_mean": 0.0,
                    **state.extras.get(
                        "_binary_precommitment_decision_feedback_diagnostics",
                        {
                            "precommitment_decision_feedback_enabled": False,
                            "precommitment_decision_feedback_weight": 0.0,
                            "precommitment_decision_feedback_mean": 0.0,
                            "precommitment_decision_feedback_active_rate": 0.0,
                            "precommitment_decision_feedback_delta_mean": 0.0,
                        },
                    ),
                },
            )
        if commitment_enabled:
            active, streaks = self._commitment_arrays(state)
        else:
            active = np.zeros(state.agent_count, dtype=bool)
            streaks = np.zeros(state.agent_count, dtype=np.int64)
        previous_active = active.copy()
        action_probs = np.asarray(
            social_result.final_action_probs,
            dtype=np.float64,
        )
        validate_binary_action_probs(action_probs, state.agent_count, "action_probs")
        commitment_direction_scores = self.social_direction_scores(local_result)
        if commitment_direction_scores is None:
            commitment_direction_ok = np.full(
                state.agent_count,
                not self.coordination_commitment_requires_direction(),
                dtype=bool,
            )
            direction_exit = np.zeros(state.agent_count, dtype=bool)
        else:
            directions = np.asarray(commitment_direction_scores, dtype=np.float64)
            if len(directions) != state.agent_count:
                raise ValueError("commitment direction scores length mismatch")
            commitment_direction_ok = (
                directions >= self.coordination_commitment_min_direction()
            )
            direction_exit = (
                directions < self.coordination_commitment_min_direction()
                if self.coordination_commitment_exit_on_negative_direction()
                else np.zeros(state.agent_count, dtype=bool)
            )
        if commitment_enabled:
            exit_mask = previous_active & (
                (action_probs < self.coordination_commitment_exit_policy_probability())
                | direction_exit
            )
            active[exit_mask] = False
        else:
            exit_mask = np.zeros(state.agent_count, dtype=bool)

        if precommitment_enabled:
            precommitment_direction_scores = self.precommitment_direction_scores(
                state=state,
                local_result=local_result,
                social_result=social_result,
                action_probs=action_probs,
                active=active,
            )
            if precommitment_direction_scores is None:
                precommitment_direction_ok = np.full(
                    state.agent_count,
                    not self.coordination_precommitment_requires_direction(),
                    dtype=bool,
                )
                precommitment_direction_values = np.zeros(
                    state.agent_count,
                    dtype=np.float64,
                )
            else:
                precommitment_direction_values = np.asarray(
                    precommitment_direction_scores,
                    dtype=np.float64,
                )
                if len(precommitment_direction_values) != state.agent_count:
                    raise ValueError(
                        "precommitment direction scores length mismatch"
                    )
                precommitment_direction_ok = (
                    precommitment_direction_values
                    >= self.coordination_precommitment_min_direction()
                )
            precommitment_evidence = self._precommitment_evidence_array(state)
            (
                precommitment_first_ready,
                precommitment_first_forced,
                precommitment_previous_ready,
            ) = self._precommitment_tracking_arrays(state)
            readiness_propagation = (
                self.precommitment_readiness_propagation_unit().propagate(
                    peer_ids=social_result.peer_ids,
                    previous_readiness=self._precommitment_decision_feedback_scores(
                        state
                    ),
                    active=active,
                    direction_ok=precommitment_direction_ok,
                )
            )
            precommitment_evidence *= self.coordination_precommitment_evidence_decay()
            precommitment_high_policy = (
                action_probs
                >= self.coordination_precommitment_min_policy_probability()
            )
            precommitment_signal = (
                (~active) & precommitment_high_policy & precommitment_direction_ok
            )
            precommitment_evidence[precommitment_signal] += (
                self.coordination_precommitment_evidence_increment()
            )
            precommitment_peer_readiness = readiness_propagation.peer_readiness
            precommitment_peer_evidence_increment = (
                readiness_propagation.peer_evidence_increment
            )
            precommitment_peer_weight = readiness_propagation.weight
            precommitment_evidence += precommitment_peer_evidence_increment
            precommitment_ready = (~active) & (
                precommitment_evidence
                >= self.coordination_precommitment_min_evidence()
            )
            precommitment_forced_mask = precommitment_ready & (action_values != 1)
            action_values[precommitment_ready] = 1
            newly_ready = precommitment_ready & ~np.isfinite(
                precommitment_first_ready
            )
            precommitment_first_ready[newly_ready] = float(context.epoch)
            newly_forced = precommitment_forced_mask & ~np.isfinite(
                precommitment_first_forced
            )
            precommitment_first_forced[newly_forced] = float(context.epoch)
            precommitment_premature_exit_mask = (
                precommitment_previous_ready & ~precommitment_ready & ~active
            )
            precommitment_previous_ready[:] = precommitment_ready
            state.extras["_binary_action_precommitment_ready"] = (
                precommitment_ready.copy()
            )
            state.extras["_binary_action_precommitment_signal"] = (
                precommitment_signal.copy()
            )
            state.extras["_binary_action_precommitment_high_policy"] = (
                precommitment_high_policy.copy()
            )
            state.extras["_binary_action_precommitment_direction_ok"] = (
                precommitment_direction_ok.copy()
            )
            state.extras["_binary_action_precommitment_direction_score"] = (
                precommitment_direction_values.copy()
            )
            state.extras["_binary_action_precommitment_forced_action"] = (
                precommitment_forced_mask.copy()
            )
            state.extras["_binary_action_precommitment_peer_readiness"] = (
                precommitment_peer_readiness.copy()
            )
            state.extras[
                "_binary_action_precommitment_peer_evidence_increment"
            ] = precommitment_peer_evidence_increment.copy()
            if (
                state.extras.get("_binary_action_precommitment_all_ready_epoch")
                is None
                and len(precommitment_first_ready)
                and bool(np.all(np.isfinite(precommitment_first_ready)))
            ):
                state.extras["_binary_action_precommitment_all_ready_epoch"] = float(
                    context.epoch
                )
        else:
            precommitment_evidence = np.zeros(state.agent_count, dtype=np.float64)
            precommitment_signal = np.zeros(state.agent_count, dtype=bool)
            precommitment_ready = np.zeros(state.agent_count, dtype=bool)
            precommitment_forced_mask = np.zeros(state.agent_count, dtype=bool)
            precommitment_high_policy = np.zeros(state.agent_count, dtype=bool)
            precommitment_direction_ok = np.zeros(state.agent_count, dtype=bool)
            precommitment_direction_values = np.zeros(
                state.agent_count,
                dtype=np.float64,
            )
            precommitment_premature_exit_mask = np.zeros(
                state.agent_count,
                dtype=bool,
            )
            precommitment_peer_weight = 0.0
            precommitment_peer_readiness = np.zeros(
                state.agent_count,
                dtype=np.float64,
            )
            precommitment_peer_evidence_increment = np.zeros(
                state.agent_count,
                dtype=np.float64,
            )
            precommitment_first_ready = np.full(
                state.agent_count,
                np.nan,
                dtype=np.float64,
            )
            precommitment_first_forced = np.full(
                state.agent_count,
                np.nan,
                dtype=np.float64,
            )

        if commitment_enabled:
            min_policy = self.coordination_commitment_min_policy_probability()
            high_policy = action_probs >= min_policy
            entry_eligible = (
                (~active)
                & (action_values == 1)
                & high_policy
                & commitment_direction_ok
            )
            streaks[entry_eligible] += 1
            streaks[~entry_eligible] = 0
            entry_mask = entry_eligible & (
                streaks >= self.coordination_commitment_min_action_streak()
            )
            active[entry_mask] = True
            forced_mask = active & (action_values != 1)
            action_values[active] = 1
            if precommitment_enabled:
                precommitment_evidence[active] = 0.0
            state.extras["_binary_action_commitment_active"] = active
            state.extras["_binary_action_commitment_streaks"] = streaks
        else:
            high_policy = np.zeros(state.agent_count, dtype=bool)
            entry_mask = np.zeros(state.agent_count, dtype=bool)
            forced_mask = np.zeros(state.agent_count, dtype=bool)
        uncommitted_high_policy = (~active) & high_policy
        committed_actions = action_values[active]
        precommitment_ready_epochs = precommitment_first_ready[
            np.isfinite(precommitment_first_ready)
        ]
        precommitment_forced_epochs = precommitment_first_forced[
            np.isfinite(precommitment_first_forced)
        ]
        precommitment_forced_with_ready = np.isfinite(
            precommitment_first_ready
        ) & np.isfinite(precommitment_first_forced)
        precommitment_all_ready_epoch = state.extras.get(
            "_binary_action_precommitment_all_ready_epoch"
        )
        diagnostics = {
            "commitment_enabled": commitment_enabled,
            "commitment_rate": float(np.mean(active)) if len(active) else 0.0,
            "commitment_entry_count": int(np.sum(entry_mask)),
            "commitment_exit_count": int(np.sum(exit_mask)),
            "committed_action_rate": (
                float(np.mean(committed_actions)) if len(committed_actions) else 0.0
            ),
            "uncommitted_high_policy_rate": (
                float(np.mean(uncommitted_high_policy))
                if len(uncommitted_high_policy)
                else 0.0
            ),
            "final_uncommitted_near_ceiling_count": int(
                np.sum(uncommitted_high_policy)
            ),
            "commitment_forced_action_count": int(np.sum(forced_mask)),
            "precommitment_enabled": precommitment_enabled,
            "precommitment_rate": (
                float(np.mean(precommitment_ready))
                if len(precommitment_ready)
                else 0.0
            ),
            "precommitment_mean_evidence": (
                float(np.mean(precommitment_evidence))
                if len(precommitment_evidence)
                else 0.0
            ),
            "precommitment_ready_count": int(np.sum(precommitment_ready)),
            "precommitment_forced_action_count": int(
                np.sum(precommitment_forced_mask)
            ),
            "precommitment_signal_rate": (
                float(np.mean(precommitment_signal))
                if len(precommitment_signal)
                else 0.0
            ),
            "precommitment_first_ready_epoch": (
                _epoch_value_or_empty(np.min(precommitment_ready_epochs))
                if len(precommitment_ready_epochs)
                else ""
            ),
            "precommitment_all_ready_epoch": (
                _epoch_value_or_empty(precommitment_all_ready_epoch)
                if precommitment_all_ready_epoch is not None
                else ""
            ),
            "precommitment_first_forced_epoch": (
                _epoch_value_or_empty(np.min(precommitment_forced_epochs))
                if len(precommitment_forced_epochs)
                else ""
            ),
            "precommitment_ready_to_forced_delay_mean": (
                float(
                    np.mean(
                        precommitment_first_forced[precommitment_forced_with_ready]
                        - precommitment_first_ready[precommitment_forced_with_ready]
                    )
                )
                if bool(np.any(precommitment_forced_with_ready))
                else ""
            ),
            "precommitment_premature_exit_count": int(
                np.sum(precommitment_premature_exit_mask)
            ),
            "precommitment_high_policy_rate": (
                float(np.mean(precommitment_high_policy))
                if len(precommitment_high_policy)
                else 0.0
            ),
            "precommitment_direction_score_mean": (
                float(np.mean(precommitment_direction_values))
                if len(precommitment_direction_values)
                else 0.0
            ),
            "precommitment_direction_score_positive_rate": (
                float(np.mean(precommitment_direction_values > 0.0))
                if len(precommitment_direction_values)
                else 0.0
            ),
            "precommitment_direction_ok_rate": (
                float(np.mean(precommitment_direction_ok))
                if len(precommitment_direction_ok)
                else 0.0
            ),
            "precommitment_ready_largest_component_fraction": (
                binary_ready_largest_component_fraction(
                    peer_ids=social_result.peer_ids,
                    ready=precommitment_ready,
                )
            ),
            "precommitment_peer_evidence_enabled": (
                self.coordination_precommitment_peer_evidence_enabled()
            ),
            "precommitment_peer_evidence_weight": precommitment_peer_weight,
            "precommitment_peer_readiness_aggregation": (
                self.coordination_precommitment_peer_readiness_aggregation()
            ),
            "precommitment_peer_readiness_mean": (
                float(np.mean(precommitment_peer_readiness))
                if len(precommitment_peer_readiness)
                else 0.0
            ),
            "precommitment_peer_readiness_active_rate": (
                float(np.mean(precommitment_peer_readiness > 0.0))
                if len(precommitment_peer_readiness)
                else 0.0
            ),
            "precommitment_peer_evidence_increment_mean": (
                float(np.mean(precommitment_peer_evidence_increment))
                if len(precommitment_peer_evidence_increment)
                else 0.0
            ),
            **state.extras.get(
                "_binary_precommitment_decision_feedback_diagnostics",
                {
                    "precommitment_decision_feedback_enabled": False,
                    "precommitment_decision_feedback_weight": 0.0,
                    "precommitment_decision_feedback_mean": 0.0,
                    "precommitment_decision_feedback_active_rate": 0.0,
                    "precommitment_decision_feedback_delta_mean": 0.0,
                },
            ),
        }
        committed_actions_output: StateArray
        if isinstance(actions, torch.Tensor):
            committed_actions_output = torch.as_tensor(
                action_values,
                dtype=actions.dtype,
                device=actions.device,
            )
        else:
            committed_actions_output = action_values
        return BinaryActionCommitmentResult(
            actions=committed_actions_output,
            diagnostics=diagnostics,
        )

    def peer_selection_neighbors(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext | None,
        local_result: BinaryLocalStepResult | None,
    ) -> list[list[int]]:
        del state, context, local_result
        return self.neighbors

    def peer_selection_action_probs(
        self,
        action_probs: np.ndarray,
        local_result: BinaryLocalStepResult | None,
    ) -> np.ndarray:
        if local_result is not None and local_result.social_mode == "policy_distill":
            return binary_action_probs_from_policy(local_result.post_local_probs)
        return action_probs

    def can_reuse_static_peer_ids(self) -> bool:
        return False

    def selected_peer_ids_are_validated(self) -> bool:
        return self.can_reuse_static_peer_ids()

    def select_initial_peers(
        self,
        state: BinarySpatialState,
        policy_probs: Any,
    ) -> list[list[int]]:
        action_probs = binary_action_probs_from_policy(policy_probs)
        reuse_static_peer_ids = self.can_reuse_static_peer_ids()
        copy_peers = not reuse_static_peer_ids
        validate_peers = not reuse_static_peer_ids
        peer_ids = select_binary_output_similarity_peers(
            neighbors=self.peer_selection_neighbors(state, None, None),
            action_probs=action_probs,
            peer_rule=self.config.coordination.peer_rule,
            threshold=self.config.coordination.threshold,
            error_label=self.toy,
            copy_peers=copy_peers,
            validate_peers=validate_peers,
        )
        return peer_ids_for_binary_mixer(
            peer_ids=peer_ids,
            mixer=self.config.coordination.mixer,
            agent_count=state.agent_count,
            error_label=self.toy,
            copy_peers=copy_peers,
            validate_peers=validate_peers,
        )

    def select_peers(
        self,
        action_probs: np.ndarray,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
    ) -> list[list[int]]:
        selected_action_probs = self.peer_selection_action_probs(
            action_probs,
            local_result,
        )
        reuse_static_peer_ids = self.can_reuse_static_peer_ids()
        copy_peers = not reuse_static_peer_ids
        validate_peers = not reuse_static_peer_ids
        peer_ids = select_binary_output_similarity_peers(
            neighbors=self.peer_selection_neighbors(state, context, local_result),
            action_probs=selected_action_probs,
            peer_rule=self.config.coordination.peer_rule,
            threshold=self.config.coordination.threshold,
            error_label=self.toy,
            copy_peers=copy_peers,
            validate_peers=validate_peers,
        )
        return peer_ids_for_binary_mixer(
            peer_ids=peer_ids,
            mixer=self.config.coordination.mixer,
            agent_count=state.agent_count,
            error_label=self.toy,
            copy_peers=copy_peers,
            validate_peers=validate_peers,
        )

    def collect_policy_probs(
        self,
        agents: list[Any],
        observations: Any,
        temperature: float,
    ) -> Any:
        raise NotImplementedError

    def distill_policy(
        self,
        agents: list[Any],
        observations: Any,
        peer_ids: list[list[int]],
        alpha: float,
        previous_probs: Any,
        context: BinaryStepContext | None = None,
        confidence_weighting: str = "none",
        confidence_weight_floor: float = 0.0,
        confidence_weight_power: float = 1.0,
        social_direction_scores: np.ndarray | None = None,
        precommitment_readiness: np.ndarray | None = None,
        precommitment_readiness_weight: float = 0.0,
    ) -> LossVector | BinaryOutputDistillationReport:
        del confidence_weighting, confidence_weight_floor, confidence_weight_power
        del social_direction_scores, precommitment_readiness
        del precommitment_readiness_weight
        raise NotImplementedError

    def refresh_policy_cache(self, agents: list[Any]) -> None:
        del agents

    def apply_social_distillation(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        peer_ids: list[list[int]],
    ) -> BinarySocialStepResult:
        tail_decision = self.social_tail_floor_decision(local_result)
        with timed_context_stage(context, "social_distillation"):
            result = distill_binary_policy_output_average(
                agents=list(state.agents or []),
                observations=local_result.extras["_observations"],
                peer_ids=peer_ids,
                alpha=self.config.coordination.alpha,
                previous_probs=local_result.post_local_probs,
                temperature=self.config.policy.temperature,
                collect_policy_probs=self.collect_policy_probs,
                distill_policy=self.distill_policy,
                refresh_policy_cache=self.refresh_policy_cache,
                agent_count=state.agent_count,
                skip_when_alpha_zero=True,
                context=context,
                confidence_weighting=self.coordination_confidence_weighting(),
                confidence_weight_floor=tail_decision.floor,
                confidence_weight_power=self.coordination_confidence_weight_power(),
                social_direction_scores=self.social_direction_scores(local_result),
                precommitment_readiness=(
                    self.precommitment_social_feedback_scores(state)
                ),
                precommitment_readiness_weight=(
                    self.coordination_precommitment_social_feedback_weight()
                ),
            )
        aggregate_diagnostics = dict(result.extras.get("social_unit_aggregate", {}))
        aggregate_diagnostics.update(tail_decision.aggregate_row())
        result.extras["social_unit_aggregate"] = aggregate_diagnostics
        return result

    def payoff_ema_decay(self) -> float | None:
        return None

    def reputation_decay(self) -> float | None:
        return (
            self.config.state.reputation.decay
            if self.config.state.reputation.enabled
            else None
        )

    def mobility_params(self) -> MobilityParams | None:
        return None

    def mobility_neighbors(self) -> list[list[int]] | None:
        return None

    def mobility_random_generator(self) -> np.random.Generator | None:
        return None

    def post_step_state_update(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
    ) -> BinaryPostStepStatePolicy:
        del state, context, local_result, social_result
        return BinaryPostStepStatePolicy(
            payoff_ema_decay=self.payoff_ema_decay(),
            reputation_decay=self.reputation_decay(),
            mobility_params=self.mobility_params(),
            mobility_neighbors=self.mobility_neighbors(),
            mobility_rng=self.mobility_random_generator(),
        )

    def post_social_policy_update(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
        mobility_result: MobilityStepResult,
        post_social_probs: Any,
    ) -> Mapping[str, Any]:
        del state, context, local_result, social_result, mobility_result
        del post_social_probs
        return {}

    def aggregate_payoffs(
        self,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> np.ndarray:
        del step_result
        return state.payoffs

    def aggregate_peer_metrics(
        self,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> Mapping[str, float | int] | None:
        del state, step_result
        return None

    def domain_aggregate_fields(
        self,
        epoch: int,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> Mapping[str, object]:
        del epoch, state, step_result
        return {}

    def aggregate_row(
        self,
        epoch: int,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> dict[str, object]:
        return {
            **binary_aggregate_common_fields(
                config=self.config,
                toy=self.toy,
                epoch=epoch,
                actions=state.actions,
                payoffs=self.aggregate_payoffs(state, step_result),
                policy_probs=step_result.post_social_probs,
                peer_ids=step_result.peer_ids,
                realized_revision_rate=step_result.realized_revision_rate or 0.0,
                reputation=state.reputation,
                mobility_result=step_result.mobility_result,
                policy_probs_pre_revision=step_result.pre_revision_probs,
                policy_probs_post_local=step_result.post_local_probs,
                policy_probs_previous_post_social=step_result.extras.get(
                    "_previous_post_social_probs"
                ),
                previous_actions=step_result.extras.get("_previous_actions"),
                local_losses=step_result.local_losses,
                revised_local_losses=step_result.extras.get("revised_local_losses"),
                social_losses=step_result.social_losses,
                include_edge_entropy=self.include_edge_entropy,
                peer_metrics=self.aggregate_peer_metrics(state, step_result),
                social_unit_aggregate=step_result.extras.get("social_unit_aggregate"),
                revision_operator_aggregate=step_result.extras.get(
                    "revision_operator_aggregate"
                ),
                commitment_aggregate=step_result.extras.get(
                    "commitment_diagnostics"
                ),
            ),
            **self.domain_aggregate_fields(epoch, state, step_result),
        }

    def micro_action_probability_source(
        self,
        step_result: BinaryPolicyStepResult,
    ) -> Any:
        return step_result.post_social_probs

    def micro_decision_probability_source(
        self,
        step_result: BinaryPolicyStepResult,
    ) -> Any:
        return step_result.extras.get("decision_action_probs", step_result.post_local_probs)

    def domain_micro_fields(
        self,
        agent_id: int,
        epoch: int,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> Mapping[str, object]:
        del agent_id, epoch, state, step_result
        return {}

    def micro_rows(
        self,
        epoch: int,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> list[dict[str, object]]:
        components = binary_peer_component_map(
            peer_ids=step_result.peer_ids,
            agent_count=state.agent_count,
        )
        action_probs = self.micro_action_probability_source(step_result)
        decision_probs = self.micro_decision_probability_source(step_result)
        rows: list[dict[str, object]] = []
        for agent_id in range(state.agent_count):
            rows.append(
                {
                    **binary_micro_base_fields(
                        config=self.config,
                        toy=self.toy,
                        epoch=epoch,
                        agent_id=agent_id,
                        state=state,
                        step_result=step_result,
                        components=components,
                        action_probability_source=action_probs,
                        decision_probability_source=decision_probs,
                    ),
                    **self.domain_micro_fields(
                        agent_id,
                        epoch,
                        state,
                        step_result,
                    ),
                }
            )
        return rows

    def write_summary(
        self,
        run_dir: Path,
        final_row: Mapping[str, object],
        state: BinarySpatialState,
    ) -> BinaryToyResult:
        del state
        domain_metrics = {
            key: _summary_value(value)
            for key, value in final_row.items()
            if str(key).startswith("domain_")
        }
        write_binary_summary_artifact(
            run_dir=run_dir,
            toy=self.toy,
            final_action_rate=final_row["action_rate"],
            final_mean_payoff=final_row["mean_payoff"],
            final_fragmentation_components=final_row["fragmentation_components"],
            final_mean_policy_action_probability=final_row[
                "mean_policy_action_probability"
            ],
            final_mean_reputation=final_row["mean_reputation"],
            final_reputation_dispersion=final_row["reputation_dispersion"],
            domain_metrics=domain_metrics,
            strict_capability=False,
        )
        return BinaryToyResult(
            run_dir=run_dir,
            toy=self.toy,
            final_action_rate=float(final_row["action_rate"]),
            final_mean_payoff=float(final_row["mean_payoff"]),
            final_fragmentation_components=int(final_row["fragmentation_components"]),
            final_mean_policy_action_probability=float(
                final_row["mean_policy_action_probability"]
            ),
            final_mean_reputation=_summary_float(final_row["mean_reputation"]),
            final_reputation_dispersion=_summary_float(
                final_row["reputation_dispersion"]
            ),
            domain_metrics=domain_metrics,
        )


def update_payoff_ema(
    payoff_ema: StateArray,
    previous_payoff_ema: StateArray,
    payoffs: StateArray,
    decay: float,
) -> None:
    """Update payoff EMA arrays in-place."""

    if isinstance(payoff_ema, torch.Tensor):
        if not isinstance(previous_payoff_ema, torch.Tensor) or not isinstance(
            payoffs,
            torch.Tensor,
        ):
            raise ValueError("torch payoff EMA updates require all tensors")
        previous_payoff_ema.copy_(payoff_ema)
        payoff_ema.mul_(decay).add_(payoffs, alpha=1.0 - decay)
        return
    previous_payoff_ema[:] = payoff_ema
    payoff_ema[:] = decay * payoff_ema + (1.0 - decay) * payoffs


def update_reputation_ema(
    reputation: StateArray,
    actions: StateArray,
    decay: float,
) -> None:
    """Update binary action reputation in-place."""

    if isinstance(reputation, torch.Tensor):
        if not 0.0 <= decay < 1.0:
            raise ValueError("reputation decay must lie in [0, 1)")
        action_values = actions.to(dtype=reputation.dtype, device=reputation.device)
        reputation.mul_(decay).add_(action_values, alpha=1.0 - decay)
        return
    update_action_reputation(reputation=reputation, actions=actions, decay=decay)


def apply_mobility_swaps(
    state: BinarySpatialState,
    neighbors: list[list[int]],
    rng: np.random.Generator,
    params: MobilityParams,
    *,
    quality_signal: StateArray | None = None,
    extra_state_arrays: MutableMapping[str, StateArray] | None = None,
    extra_state_lists: MutableMapping[str, MutableSequence[Any]] | None = None,
) -> MobilityStepResult:
    """Apply fixed-cell mobility to common binary state arrays."""

    arrays = state.state_arrays()
    if extra_state_arrays:
        arrays.update(extra_state_arrays)
    state_lists = dict(extra_state_lists or {})
    if state.agents:
        state_lists["agents"] = state.agents
    if any(isinstance(values, torch.Tensor) for values in arrays.values()):
        original_arrays = dict(arrays)
        mobility_arrays = {
            name: to_numpy_view(values).copy()
            for name, values in original_arrays.items()
        }
        result = apply_local_quality_mobility(
            state_arrays=mobility_arrays,
            quality_signal=to_numpy_view(
                state.payoff_ema if quality_signal is None else quality_signal,
                dtype=np.float64,
            ).copy(),
            neighbors=neighbors,
            rng=rng,
            params=params,
            state_lists=state_lists or None,
        )
        for name, values in mobility_arrays.items():
            _copy_back_array(original_arrays[name], values)
        return result
    return apply_local_quality_mobility(
        state_arrays=arrays,
        quality_signal=state.payoff_ema if quality_signal is None else quality_signal,
        neighbors=neighbors,
        rng=rng,
        params=params,
        state_lists=state_lists or None,
    )


@dataclass
class BinarySpatialRunner:
    """Generic lifecycle runner for Toy2/4/5-style binary spatial domains."""

    domain: BinarySpatialDomain
    epochs: int
    revision_rate: float
    revision_rng: np.random.Generator
    logging_interval: int = 1
    log_micro_state: bool = True
    log_aggregate_metrics: bool = True
    timing_rows: TimingRows | None = None

    def run(self) -> Any:
        with self._timed(0, "make_run_dir"):
            run_dir = self.domain.make_run_dir()
        with self._timed(0, "write_metadata"):
            self.domain.write_metadata(run_dir)
        with self._timed(0, "initial_state"):
            state = self.domain.initial_state()
        with self._timed(0, "writer_setup"):
            micro_writer = CsvLogWriter(
                run_dir / "micro_state.csv",
                self.domain.micro_state_fields,
            )
            aggregate_writer = CsvLogWriter(
                run_dir / "aggregate_metrics.csv",
                self.domain.aggregate_fields,
            )
        final_row: Mapping[str, Any] | None = None
        try:
            with self._timed(0, "initial_step_result"):
                initial_step = self.domain.initial_step_result(state)
            with self._timed(0, "initial_aggregate_row"):
                initial_row = self.domain.aggregate_row(
                    epoch=0,
                    state=state,
                    step_result=initial_step,
                )
            final_row = initial_row
            if self.log_aggregate_metrics:
                with self._timed(0, "write_initial_aggregate"):
                    aggregate_writer.write(dict(initial_row))
            previous_post_social_probs = initial_step.post_social_probs

            for epoch in range(1, self.epochs + 1):
                with self._timed(epoch, "sample_revision_mask"):
                    revision_mask = sample_revision_mask(
                        agent_count=state.agent_count,
                        revision_rate=self.revision_rate,
                        rng=self.revision_rng,
                    )
                previous_actions = to_numpy_view(state.actions, dtype=np.int64).copy()
                with self._timed(epoch, "hooked_step_total"):
                    step_result = self._hooked_step(
                        epoch=epoch,
                        state=state,
                        revision_mask=revision_mask,
                    )
                step_result.extras["_previous_post_social_probs"] = (
                    previous_post_social_probs
                )
                step_result.extras["_previous_actions"] = previous_actions
                with self._timed(epoch, "aggregate_row"):
                    row = self.domain.aggregate_row(
                        epoch=epoch,
                        state=state,
                        step_result=step_result,
                    )
                final_row = row
                if self.log_aggregate_metrics:
                    with self._timed(epoch, "write_aggregate"):
                        aggregate_writer.write(dict(row))
                if self.log_micro_state and epoch % self.logging_interval == 0:
                    with self._timed(epoch, "write_micro"):
                        for micro_row in self.domain.micro_rows(
                            epoch=epoch,
                            state=state,
                            step_result=step_result,
                        ):
                            micro_writer.write(dict(micro_row))
                previous_post_social_probs = step_result.post_social_probs
        finally:
            micro_writer.close()
            aggregate_writer.close()

        if final_row is None:
            raise RuntimeError("binary spatial runner produced no aggregate rows")
        return self.domain.write_summary(
            run_dir=run_dir,
            final_row=final_row,
            state=state,
        )

    def _hooked_step(
        self,
        epoch: int,
        state: BinarySpatialState,
        revision_mask: np.ndarray,
    ) -> BinaryPolicyStepResult:
        domain = self.domain
        agent_count = state.agent_count
        validate_revision_mask(revision_mask, agent_count)
        with self._timed(epoch, "build_step_context"):
            context = domain.build_step_context(
                epoch=epoch,
                state=state,
                revision_mask=revision_mask,
            )
        if self.timing_rows is not None:
            context.extras["_record_timing"] = self._timing_recorder(epoch)
            context.extras["_synchronize_timing_device"] = (
                self._synchronize_timing_device
            )
        validate_revision_mask(
            context.revision_mask,
            agent_count,
            name="context.revision_mask",
        )
        validate_step_extras(context.extras, "context.extras")
        with self._timed(epoch, "local_step"):
            local_result = domain.local_step(state=state, context=context)
        validate_binary_action_probs(
            local_result.candidate_action_probs,
            agent_count,
            "candidate_action_probs",
        )
        validate_loss_vector(local_result.local_losses, agent_count, "local_losses")
        if local_result.actions_after_revision is not None:
            validate_binary_actions(
                local_result.actions_after_revision,
                agent_count,
                name="actions_after_revision",
            )
        validate_step_extras(local_result.extras, "local_result.extras")
        with self._timed(epoch, "select_peers"):
            peer_ids = domain.select_peers(
                action_probs=local_result.candidate_action_probs,
                state=state,
                context=context,
                local_result=local_result,
            )
        selected_peer_ids_are_validated = getattr(
            domain,
            "selected_peer_ids_are_validated",
            lambda: False,
        )
        if not selected_peer_ids_are_validated():
            try:
                validate_peer_ids(peer_ids, agent_count)
            except TypeError as exc:
                raise ValueError("peer_ids must be a list of peer-id lists") from exc
        with self._timed(epoch, "social_step"):
            social_result = self._social_step(
                state=state,
                context=context,
                local_result=local_result,
                peer_ids=peer_ids,
            )
        validate_binary_action_probs(
            social_result.final_action_probs,
            agent_count,
            "final_action_probs",
        )
        validate_loss_vector(social_result.social_losses, agent_count, "social_losses")
        if not selected_peer_ids_are_validated():
            try:
                validate_peer_ids(social_result.peer_ids, agent_count)
            except TypeError as exc:
                raise ValueError(
                    "social_result.peer_ids must be a list of peer-id lists"
                ) from exc
        validate_step_extras(social_result.extras, "social_result.extras")
        with self._timed(epoch, "sample_actions"):
            if local_result.actions_after_revision is None:
                actions = domain.sample_actions(
                    state=state,
                    action_probs=social_result.final_action_probs,
                    revision_mask=revision_mask,
                    context=context,
                    local_result=local_result,
                )
            else:
                actions = local_result.actions_after_revision
        validate_binary_actions(actions, agent_count)
        commitment_diagnostics: Mapping[str, object] = {}
        apply_action_commitment = getattr(domain, "apply_action_commitment", None)
        if apply_action_commitment is not None:
            with self._timed(epoch, "action_commitment"):
                commitment_result = apply_action_commitment(
                    state=state,
                    context=context,
                    local_result=local_result,
                    social_result=social_result,
                    actions=actions,
                )
            actions = commitment_result.actions
            validate_binary_actions(actions, agent_count)
            commitment_diagnostics = dict(commitment_result.diagnostics)
            validate_step_extras(
                commitment_diagnostics,
                "action_commitment diagnostics",
            )

        extras: dict[str, Any] = {}
        extras.update(public_step_extras(local_result.extras))
        extras.update(public_step_extras(social_result.extras))
        if commitment_diagnostics:
            extras["commitment_diagnostics"] = dict(commitment_diagnostics)
        with self._timed(epoch, "commit_actions"):
            commit_updates = domain.commit_actions(
                state=state,
                actions=actions,
                context=context,
                local_result=local_result,
                social_result=social_result,
            )
        validate_step_extras(commit_updates, "commit_actions result")
        extras.update(public_step_extras(commit_updates))

        with self._timed(epoch, "post_step_state_policy"):
            post_step_policy = domain.post_step_state_update(
                state=state,
                context=context,
                local_result=local_result,
                social_result=social_result,
            )
        validate_post_step_state_policy(post_step_policy)
        if post_step_policy.payoff_ema_decay is not None:
            with self._timed(epoch, "update_payoff_ema"):
                update_payoff_ema(
                    payoff_ema=state.payoff_ema,
                    previous_payoff_ema=state.previous_payoff_ema,
                    payoffs=state.payoffs,
                    decay=post_step_policy.payoff_ema_decay,
                )
        if post_step_policy.reputation_decay is not None:
            with self._timed(epoch, "update_reputation_ema"):
                update_reputation_ema(
                    reputation=state.reputation,
                    actions=state.actions,
                    decay=post_step_policy.reputation_decay,
                )

        mobility_result = MobilityStepResult.none(state.agent_count)
        if post_step_policy.mobility_params is not None:
            if post_step_policy.mobility_neighbors is None:
                raise ValueError("mobility_neighbors are required for mobility updates")
            if post_step_policy.mobility_rng is None:
                raise ValueError("mobility_rng is required for mobility updates")
            mobility_extra_arrays = dict(
                post_step_policy.mobility_extra_state_arrays or {}
            )
            commitment_active = state.extras.get("_binary_action_commitment_active")
            commitment_streaks = state.extras.get("_binary_action_commitment_streaks")
            precommitment_evidence = state.extras.get(
                "_binary_action_precommitment_evidence"
            )
            if commitment_active is not None:
                mobility_extra_arrays["_binary_action_commitment_active"] = (
                    commitment_active
                )
            if commitment_streaks is not None:
                mobility_extra_arrays["_binary_action_commitment_streaks"] = (
                    commitment_streaks
                )
            if precommitment_evidence is not None:
                mobility_extra_arrays["_binary_action_precommitment_evidence"] = (
                    precommitment_evidence
                )
            with self._timed(epoch, "mobility_swaps"):
                mobility_result = apply_mobility_swaps(
                    state=state,
                    neighbors=post_step_policy.mobility_neighbors,
                    rng=post_step_policy.mobility_rng,
                    params=post_step_policy.mobility_params,
                    quality_signal=post_step_policy.mobility_quality_signal,
                    extra_state_arrays=mobility_extra_arrays,
                    extra_state_lists=post_step_policy.mobility_extra_state_lists,
                )

        post_social_probs = social_result.post_social_probs
        with self._timed(epoch, "finalize_hook_step"):
            finalize_updates = domain.finalize_hook_step(
                state=state,
                context=context,
                local_result=local_result,
                social_result=social_result,
                mobility_result=mobility_result,
            )
        validate_step_extras(finalize_updates, "finalize_hook_step result")
        maybe_post_social = finalize_updates.get("post_social_probs")
        if maybe_post_social is not None:
            try:
                maybe_action_probs = binary_action_probs_from_policy(maybe_post_social)
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(
                    "finalize_hook_step post_social_probs must expose binary action "
                    "probabilities"
                ) from exc
            validate_binary_action_probs(
                maybe_action_probs,
                agent_count,
                "finalize_hook_step post_social_probs",
            )
            post_social_probs = maybe_post_social
        finalize_extras = finalize_updates.get("extras", {})
        validate_step_extras(finalize_extras, "finalize_hook_step extras")
        extras.update(public_step_extras(finalize_extras))

        with self._timed(epoch, "post_social_policy_update"):
            post_social_update_hook = getattr(
                domain,
                "post_social_policy_update",
                None,
            )
            if post_social_update_hook is None:
                post_social_updates = {}
            else:
                post_social_updates = post_social_update_hook(
                    state=state,
                    context=context,
                    local_result=local_result,
                    social_result=social_result,
                    mobility_result=mobility_result,
                    post_social_probs=post_social_probs,
                )
        validate_step_extras(
            post_social_updates,
            "post_social_policy_update result",
        )
        maybe_post_social = post_social_updates.get("post_social_probs")
        if maybe_post_social is not None:
            try:
                maybe_action_probs = binary_action_probs_from_policy(maybe_post_social)
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(
                    "post_social_policy_update post_social_probs must expose "
                    "binary action probabilities"
                ) from exc
            validate_binary_action_probs(
                maybe_action_probs,
                agent_count,
                "post_social_policy_update post_social_probs",
            )
            post_social_probs = maybe_post_social
        post_social_extras = post_social_updates.get("extras", {})
        validate_step_extras(
            post_social_extras,
            "post_social_policy_update extras",
        )
        extras.update(public_step_extras(post_social_extras))

        return BinaryPolicyStepResult(
            pre_revision_probs=local_result.pre_revision_probs,
            post_local_probs=local_result.post_local_probs,
            post_social_probs=post_social_probs,
            local_losses=local_result.local_losses,
            social_losses=social_result.social_losses,
            peer_ids=social_result.peer_ids,
            revision_mask=revision_mask,
            mobility_result=mobility_result,
            realized_revision_rate=realized_revision_rate(revision_mask),
            extras=extras,
        )

    def _social_step(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        peer_ids: list[list[int]],
    ) -> BinarySocialStepResult:
        domain = self.domain
        mixer = domain.coordination_mixer()
        agent_count = state.agent_count
        if mixer == "none":
            post_social_probs = local_result.post_local_probs
            return BinarySocialStepResult(
                peer_ids=peer_ids,
                post_social_probs=post_social_probs,
                final_action_probs=binary_action_probs_from_policy(post_social_probs),
                social_losses=empty_losses(agent_count),
            )
        if mixer != "output_average":
            raise ValueError(f"Unsupported binary coordination mixer: {mixer}")

        if local_result.social_mode == "probability_mix":
            confidence_weighting = getattr(
                domain,
                "coordination_confidence_weighting",
                lambda: "none",
            )()
            social_extras: dict[str, object] = {}
            if confidence_weighting in ("peer", "peer_direction"):
                tail_decision = domain.social_tail_floor_decision(local_result)
                mix_result, confidence_diagnostics = (
                    mix_binary_output_confidence_weighted(
                        action_probs=local_result.candidate_action_probs,
                        peer_ids=peer_ids,
                        alpha=domain.coordination_alpha(),
                        floor=tail_decision.floor,
                        power=getattr(
                            domain,
                            "coordination_confidence_weight_power",
                            lambda: 1.0,
                        )(),
                        direction_scores=(
                            getattr(
                                domain,
                                "social_direction_scores",
                                lambda _local_result: None,
                            )(local_result)
                        ),
                        precommitment_readiness=(
                            getattr(
                                domain,
                                "precommitment_social_feedback_scores",
                                lambda _state: None,
                            )(state)
                        ),
                        precommitment_readiness_weight=(
                            getattr(
                                domain,
                                "coordination_precommitment_social_feedback_weight",
                                lambda: 0.0,
                            )()
                        ),
                        weighting=confidence_weighting,
                    )
                )
                diagnostics = social_diagnostics(mix_result)
                aggregate_diagnostics = diagnostics.aggregate_row()
                aggregate_diagnostics.update(confidence_diagnostics.aggregate_row())
                aggregate_diagnostics.update(tail_decision.aggregate_row())
                micro_diagnostics = []
                for agent_id in range(agent_count):
                    row = diagnostics.micro_row(agent_id)
                    row.update(confidence_diagnostics.micro_row(agent_id))
                    micro_diagnostics.append(row)
                final_action_probs = mix_result.mixed_values
                social_losses = mix_result.losses
                social_extras = {
                    "social_unit_aggregate": aggregate_diagnostics,
                    "social_unit_micro": micro_diagnostics,
                }
            elif confidence_weighting == "none":
                final_action_probs, social_losses = mix_binary_output_average(
                    action_probs=local_result.candidate_action_probs,
                    peer_ids=peer_ids,
                    alpha=domain.coordination_alpha(),
                )
            else:
                raise ValueError(
                    "confidence_weighting must be one of: "
                    "'none', 'peer', 'peer_direction'"
                )
            return BinarySocialStepResult(
                peer_ids=peer_ids,
                post_social_probs=policy_tensor_from_action_probs(
                    final_action_probs,
                    local_result.post_local_probs,
                    domain.policy_tensor_from_action_probs,
                ),
                final_action_probs=final_action_probs,
                social_losses=social_losses,
                extras=social_extras,
            )
        if local_result.social_mode == "policy_distill":
            return domain.apply_social_distillation(
                state=state,
                context=context,
                local_result=local_result,
                peer_ids=peer_ids,
            )
        raise ValueError(f"Unsupported binary social mode: {local_result.social_mode}")

    def _timing_recorder(self, epoch: int) -> TimingRecorder:
        def record(stage: str, seconds: float) -> None:
            self._record_timing(epoch, stage, seconds)

        return record

    def _record_timing(self, epoch: int, stage: str, seconds: float) -> None:
        if self.timing_rows is None:
            return
        config = getattr(self.domain, "config", None)
        self.timing_rows.append(
            {
                "run_id": getattr(getattr(config, "run", None), "name", ""),
                "seed": getattr(getattr(config, "run", None), "seed", ""),
                "epoch": epoch,
                "toy": getattr(self.domain, "toy", ""),
                "policy_rule": getattr(getattr(config, "policy", None), "rule", ""),
                "coordination_mixer": getattr(
                    getattr(config, "coordination", None),
                    "mixer",
                    "",
                ),
                "device": str(getattr(self.domain, "device", "")),
                "agent_count": self._timing_agent_count(config),
                "stage": stage,
                "seconds": seconds,
            }
        )

    def _timing_agent_count(self, config: object) -> object:
        if hasattr(self.domain, "agent_count"):
            return getattr(self.domain, "agent_count")
        try:
            return getattr(config, "agent_count")
        except AttributeError:
            pass
        agents = getattr(config, "agents", None)
        if hasattr(agents, "count"):
            return getattr(agents, "count")
        environment = getattr(config, "environment", None)
        if hasattr(environment, "grid_width") and hasattr(environment, "grid_height"):
            return getattr(environment, "grid_width") * getattr(
                environment,
                "grid_height",
            )
        return ""

    def _timed(self, epoch: int, stage: str) -> "_RunnerTimer":
        return _RunnerTimer(self, epoch, stage)

    def _synchronize_timing_device(self) -> None:
        device = getattr(self.domain, "device", None)
        if device is not None and getattr(device, "type", None) == "cuda":
            try:
                import torch

                torch.cuda.synchronize(device)
            except RuntimeError:
                pass


@dataclass
class _RunnerTimer:
    runner: BinarySpatialRunner
    epoch: int
    stage: str
    start: float = 0.0

    def __enter__(self) -> None:
        if self.runner.timing_rows is None:
            return
        self.runner._synchronize_timing_device()
        self.start = time.perf_counter()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if self.runner.timing_rows is None:
            return
        self.runner._synchronize_timing_device()
        self.runner._record_timing(
            self.epoch,
            self.stage,
            time.perf_counter() - self.start,
        )


@dataclass
class _ContextTimer:
    context: BinaryStepContext | None
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
        if self.context is None:
            return False
        return callable(self.context.extras.get("_record_timing"))

    def _synchronize(self) -> None:
        if self.context is None:
            return
        synchronize = self.context.extras.get("_synchronize_timing_device")
        if callable(synchronize):
            synchronize()
