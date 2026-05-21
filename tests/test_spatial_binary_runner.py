from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from neural_abm.accelerator import (
    TensorBatchedMLPRuntime,
    apply_batched_mlp_loss_gradients_with_result,
    batched_binary_policy_gradient_losses,
    trainable_batched_mlp_parameters,
)
from neural_abm.binary_neural import (
    apply_batched_output_average_distillation_update,
    apply_tensor_binary_policy_gradient_update,
    apply_tensor_output_average_distillation_update,
)
from neural_abm.config import (
    CommunicationBudgetConfig,
    Toy2CoordinationConfig,
    Toy4CoordinationConfig,
    Toy5CoordinationConfig,
)
from neural_abm.losses import TensorBackedLossVector
from neural_abm.mobility import MobilityParams, MobilityStepResult
from neural_abm.social import (
    PROBABILITY_DISTRIBUTION_CHANNEL,
    SocialBlock,
    SocialChannel,
)
from neural_abm.spatial_binary import (
    BatchedDistributionDistillationAdapter,
    BinaryLocalStepResult,
    BinaryOutputDistillationReport,
    BinaryPolicyLearningCallbacks,
    BinaryPolicyLearningUnit,
    BinaryPolicyStepResult,
    BinaryPostStepStatePolicy,
    BinarySocialStepResult,
    BinarySpatialRunner,
    BinarySpatialState,
    BinaryStepContext,
    BinaryToyDomainBase,
    BinaryToyResult,
    BatchedPolicyGradientLocalUpdateAdapter,
    TensorRuntimeDistributionDistillationAdapter,
    TensorRuntimePolicyGradientLocalUpdateAdapter,
    binary_loss_metrics,
    binary_aggregate_common_fields,
    binary_policy_confidence,
    binary_policy_direction_alignment,
    binary_policy_confidence_weights,
    binary_micro_common_fields,
    binary_micro_base_fields,
    binary_micro_mobility_fields,
    binary_peer_component_map,
    binary_peer_metrics,
    binary_policy_prob,
    binary_policy_matrix,
    distill_binary_policy_output_average,
    mean_binary_policy_prob,
    mix_binary_output_confidence_weighted,
    mix_binary_policy_distribution_confidence_weighted,
    peer_ids_for_binary_mixer,
    run_binary_policy_learning_step,
    select_binary_output_similarity_peers,
)
from neural_abm.unit import NABMLocalStep, NABMStep


def binary_policy(probs: np.ndarray) -> np.ndarray:
    return binary_policy_matrix(probs)


class _TinyBatchedPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 4)
        self.activation = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class _TinyBatchedAgent:
    def __init__(self, agent_id: int, model: _TinyBatchedPolicy) -> None:
        self.agent_id = agent_id
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.03)


def _make_tiny_batched_agents(
    agent_count: int = 3,
) -> tuple[list[_TinyBatchedAgent], list[_TinyBatchedAgent]]:
    torch.manual_seed(20260518)
    source_models = [_TinyBatchedPolicy() for _ in range(agent_count)]
    agents = [
        _TinyBatchedAgent(agent_id, model)
        for agent_id, model in enumerate(source_models)
    ]
    clones: list[_TinyBatchedAgent] = []
    for agent_id, source_model in enumerate(source_models):
        model = _TinyBatchedPolicy()
        model.load_state_dict(source_model.state_dict())
        clones.append(_TinyBatchedAgent(agent_id, model))
    return agents, clones


def _clone_agent_parameters(
    agents: list[_TinyBatchedAgent],
) -> list[dict[str, torch.Tensor]]:
    return [
        {
            name: parameter.detach().clone()
            for name, parameter in agent.model.state_dict().items()
        }
        for agent in agents
    ]


def _assert_agent_parameters_match(
    left_agents: list[_TinyBatchedAgent],
    right_agents: list[_TinyBatchedAgent],
) -> None:
    for left_agent, right_agent in zip(left_agents, right_agents, strict=True):
        for left_parameter, right_parameter in zip(
            left_agent.model.parameters(),
            right_agent.model.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(
                left_parameter,
                right_parameter,
                atol=1e-6,
                rtol=1e-6,
            )


def _assert_agent_parameters_equal_snapshot(
    agents: list[_TinyBatchedAgent],
    snapshot: list[dict[str, torch.Tensor]],
) -> None:
    for agent, model_snapshot in zip(agents, snapshot, strict=True):
        current = agent.model.state_dict()
        for name, value in model_snapshot.items():
            torch.testing.assert_close(current[name], value, atol=0.0, rtol=0.0)


def _clone_runtime_parameters(
    runtime: TensorBatchedMLPRuntime,
) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.detach().clone() for tensor in runtime.parameters.tensors())


def _assert_runtime_parameters_match(
    left_runtime: TensorBatchedMLPRuntime,
    right_runtime: TensorBatchedMLPRuntime,
) -> None:
    for left_tensor, right_tensor in zip(
        left_runtime.parameters.tensors(),
        right_runtime.parameters.tensors(),
        strict=True,
    ):
        torch.testing.assert_close(left_tensor, right_tensor, atol=1e-6, rtol=1e-6)


def _assert_runtime_parameters_equal_snapshot(
    runtime: TensorBatchedMLPRuntime,
    snapshot: tuple[torch.Tensor, ...],
) -> None:
    for current, previous in zip(runtime.parameters.tensors(), snapshot, strict=True):
        torch.testing.assert_close(current, previous, atol=0.0, rtol=0.0)


def test_binary_coordination_configs_expose_confidence_defaults() -> None:
    communication_budget = CommunicationBudgetConfig(
        probe_predictions=1,
        latent_dim=8,
        scalar_summary=8,
    )

    toy2 = Toy2CoordinationConfig(
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        threshold=0.0,
        communication_budget=communication_budget,
    )
    toy4 = Toy4CoordinationConfig()
    toy5 = Toy5CoordinationConfig()

    for config in (toy2, toy4, toy5):
        assert config.confidence_weighting == "none"
        assert config.confidence_weight_floor == pytest.approx(0.0)
        assert config.confidence_weight_power == pytest.approx(1.0)
        assert config.confidence_tail_floor == pytest.approx(0.0)
        assert config.confidence_tail_min_policy_rate == pytest.approx(1.0)
        assert config.confidence_tail_min_action_rate == pytest.approx(1.0)
        assert config.commitment_enabled is False
        assert config.commitment_min_policy_probability == pytest.approx(1.0)
        assert config.commitment_min_action_streak == 1
        assert config.commitment_requires_direction is True
        assert config.commitment_min_direction == pytest.approx(0.0)
        assert config.commitment_exit_policy_probability == pytest.approx(0.0)
        assert config.commitment_exit_on_negative_direction is True
        assert config.precommitment_enabled is False
        assert config.precommitment_min_policy_probability == pytest.approx(1.0)
        assert config.precommitment_min_evidence == pytest.approx(1.0)
        assert config.precommitment_evidence_increment == pytest.approx(1.0)
        assert config.precommitment_evidence_decay == pytest.approx(0.0)
        assert config.precommitment_requires_direction is True
        assert config.precommitment_min_direction == pytest.approx(0.0)
        assert config.precommitment_direction_source == "social"
        assert config.precommitment_readiness_direction_weight == pytest.approx(1.0)
        assert config.precommitment_decision_feedback_enabled is False
        assert config.precommitment_decision_feedback_weight == pytest.approx(0.0)
        assert config.precommitment_social_feedback_enabled is False
        assert config.precommitment_social_feedback_weight == pytest.approx(0.0)
        assert config.precommitment_peer_evidence_enabled is False
        assert config.precommitment_peer_evidence_weight == pytest.approx(0.0)
        assert config.precommitment_peer_readiness_aggregation == "mean"

    for config in (toy2, toy4):
        assert config.revision_operator_enabled is False
        assert config.revision_operator_source == "policy_probability"

    active = Toy4CoordinationConfig(
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        threshold=0.0,
        confidence_weighting="peer",
        confidence_weight_floor=0.2,
        confidence_weight_power=2.0,
        confidence_tail_floor=0.6,
        confidence_tail_min_policy_rate=0.9,
        confidence_tail_min_action_rate=0.95,
        commitment_enabled=True,
        commitment_min_policy_probability=0.9,
        commitment_min_action_streak=2,
        commitment_requires_direction=True,
        commitment_min_direction=0.01,
        commitment_exit_policy_probability=0.75,
        commitment_exit_on_negative_direction=True,
        precommitment_enabled=True,
        precommitment_min_policy_probability=0.75,
        precommitment_min_evidence=1.5,
        precommitment_evidence_increment=1.0,
        precommitment_evidence_decay=0.8,
        precommitment_requires_direction=True,
        precommitment_min_direction=0.02,
        precommitment_direction_source="social",
        precommitment_readiness_direction_weight=0.75,
        precommitment_decision_feedback_enabled=True,
        precommitment_decision_feedback_weight=0.5,
        precommitment_social_feedback_enabled=True,
        precommitment_social_feedback_weight=0.75,
        precommitment_peer_evidence_enabled=True,
        precommitment_peer_evidence_weight=0.5,
        precommitment_peer_readiness_aggregation="max",
        revision_operator_enabled=True,
        revision_operator_source="policy_probability",
    )
    assert active.confidence_weighting == "peer"
    assert active.confidence_weight_floor == pytest.approx(0.2)
    assert active.confidence_weight_power == pytest.approx(2.0)
    assert active.confidence_tail_floor == pytest.approx(0.6)
    assert active.confidence_tail_min_policy_rate == pytest.approx(0.9)
    assert active.confidence_tail_min_action_rate == pytest.approx(0.95)
    assert active.commitment_enabled is True
    assert active.commitment_min_policy_probability == pytest.approx(0.9)
    assert active.commitment_min_action_streak == 2
    assert active.commitment_min_direction == pytest.approx(0.01)
    assert active.commitment_exit_policy_probability == pytest.approx(0.75)
    assert active.precommitment_enabled is True
    assert active.precommitment_min_policy_probability == pytest.approx(0.75)
    assert active.precommitment_min_evidence == pytest.approx(1.5)
    assert active.precommitment_evidence_increment == pytest.approx(1.0)
    assert active.precommitment_evidence_decay == pytest.approx(0.8)
    assert active.precommitment_min_direction == pytest.approx(0.02)
    assert active.precommitment_direction_source == "social"
    assert active.precommitment_readiness_direction_weight == pytest.approx(0.75)
    assert active.precommitment_decision_feedback_enabled is True
    assert active.precommitment_decision_feedback_weight == pytest.approx(0.5)
    assert active.precommitment_social_feedback_enabled is True
    assert active.precommitment_social_feedback_weight == pytest.approx(0.75)
    assert active.precommitment_peer_evidence_enabled is True
    assert active.precommitment_peer_evidence_weight == pytest.approx(0.5)
    assert active.precommitment_peer_readiness_aggregation == "max"
    assert active.revision_operator_enabled is True
    assert active.revision_operator_source == "policy_probability"

    directional = Toy5CoordinationConfig(
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        threshold=0.0,
        confidence_weighting="peer_direction",
        precommitment_direction_source="readiness_augmented_threshold",
        precommitment_readiness_direction_weight=0.5,
    )
    assert directional.confidence_weighting == "peer_direction"
    assert directional.precommitment_direction_source == (
        "readiness_augmented_threshold"
    )
    assert directional.precommitment_readiness_direction_weight == pytest.approx(0.5)


def test_social_tail_floor_decision_only_activates_near_ceiling() -> None:
    domain = FakeBinaryToyDomain()
    domain.config.coordination.confidence_weighting = "peer"
    domain.config.coordination.confidence_weight_floor = 0.0
    domain.config.coordination.confidence_tail_floor = 0.5
    domain.config.coordination.confidence_tail_min_policy_rate = 0.9
    domain.config.coordination.confidence_tail_min_action_rate = 0.9
    high_local = BinaryLocalStepResult(
        pre_revision_probs=binary_policy(np.asarray([0.3, 0.4, 0.5])),
        candidate_action_probs=np.asarray([0.92, 0.95, 0.98]),
        post_local_probs=binary_policy(np.asarray([0.92, 0.95, 0.98])),
        local_losses=[0.0, 0.0, 0.0],
        social_mode="probability_mix",
        actions_after_revision=np.asarray([1, 1, 1]),
    )

    decision = domain.social_tail_floor_decision(high_local)

    assert decision.active is True
    assert decision.floor == pytest.approx(0.5)
    assert decision.aggregate_row() == {
        "social_tail_floor_active": True,
        "social_tail_confidence_floor": pytest.approx(0.5),
        "social_tail_policy_rate": pytest.approx(0.95),
        "social_tail_action_rate": pytest.approx(1.0),
    }

    low_local = BinaryLocalStepResult(
        pre_revision_probs=binary_policy(np.asarray([0.3, 0.4, 0.5])),
        candidate_action_probs=np.asarray([0.65, 0.7, 0.75]),
        post_local_probs=binary_policy(np.asarray([0.65, 0.7, 0.75])),
        local_losses=[0.0, 0.0, 0.0],
        social_mode="probability_mix",
        actions_after_revision=np.asarray([1, 0, 1]),
    )

    inactive = domain.social_tail_floor_decision(low_local)

    assert inactive.active is False
    assert inactive.floor == pytest.approx(0.0)
    assert inactive.policy_rate == pytest.approx(0.7)
    assert inactive.action_rate == pytest.approx(2.0 / 3.0)

    domain.config.coordination.confidence_weighting = "none"
    disabled = domain.social_tail_floor_decision(high_local)
    assert disabled.active is False
    assert disabled.floor == pytest.approx(0.0)


def test_action_commitment_uses_hysteresis_and_forces_committed_actions() -> None:
    domain = FakeBinaryToyDomain()
    domain.config.coordination.commitment_enabled = True
    domain.config.coordination.commitment_min_policy_probability = 0.9
    domain.config.coordination.commitment_min_action_streak = 2
    domain.config.coordination.commitment_requires_direction = True
    domain.config.coordination.commitment_min_direction = 0.0
    domain.config.coordination.commitment_exit_policy_probability = 0.75
    domain.config.coordination.commitment_exit_on_negative_direction = True
    state = make_state([1, 1, 1])
    context = BinaryStepContext(epoch=1, revision_mask=np.ones(3, dtype=bool))
    local_result = BinaryLocalStepResult(
        pre_revision_probs=binary_policy(np.asarray([0.5, 0.5, 0.5])),
        candidate_action_probs=np.asarray([0.95, 0.95, 0.95]),
        post_local_probs=binary_policy(np.asarray([0.95, 0.95, 0.95])),
        local_losses=[0.0, 0.0, 0.0],
        social_mode="policy_distill",
        actions_after_revision=np.asarray([1, 1, 1]),
        extras={
            "state_continuation_components": SimpleNamespace(
                effective=np.asarray([0.1, 0.1, 0.1]),
            ),
        },
    )
    social_result = BinarySocialStepResult(
        peer_ids=[[1], [0], [1]],
        post_social_probs=binary_policy(np.asarray([0.95, 0.95, 0.95])),
        final_action_probs=np.asarray([0.95, 0.95, 0.95]),
        social_losses=[0.0, 0.0, 0.0],
    )

    first = domain.apply_action_commitment(
        state=state,
        context=context,
        local_result=local_result,
        social_result=social_result,
        actions=np.asarray([1, 1, 1]),
    )
    second = domain.apply_action_commitment(
        state=state,
        context=context,
        local_result=local_result,
        social_result=social_result,
        actions=np.asarray([1, 1, 1]),
    )

    assert first.diagnostics["commitment_entry_count"] == 0
    assert first.diagnostics["commitment_rate"] == pytest.approx(0.0)
    assert second.diagnostics["commitment_entry_count"] == 3
    assert second.diagnostics["commitment_rate"] == pytest.approx(1.0)

    forced = domain.apply_action_commitment(
        state=state,
        context=context,
        local_result=local_result,
        social_result=social_result,
        actions=np.asarray([0, 0, 0]),
    )

    np.testing.assert_array_equal(forced.actions, [1, 1, 1])
    assert forced.diagnostics["commitment_forced_action_count"] == 3
    assert forced.diagnostics["committed_action_rate"] == pytest.approx(1.0)

    low_social = BinarySocialStepResult(
        peer_ids=[[1], [0], [1]],
        post_social_probs=binary_policy(np.asarray([0.6, 0.6, 0.6])),
        final_action_probs=np.asarray([0.6, 0.6, 0.6]),
        social_losses=[0.0, 0.0, 0.0],
    )
    exited = domain.apply_action_commitment(
        state=state,
        context=context,
        local_result=local_result,
        social_result=low_social,
        actions=np.asarray([0, 0, 0]),
    )

    np.testing.assert_array_equal(exited.actions, [0, 0, 0])
    assert exited.diagnostics["commitment_exit_count"] == 3
    assert exited.diagnostics["commitment_rate"] == pytest.approx(0.0)


def test_action_precommitment_accumulates_evidence_before_hard_commitment() -> None:
    domain = FakeBinaryToyDomain()
    domain.config.coordination.commitment_enabled = True
    domain.config.coordination.commitment_min_policy_probability = 0.9
    domain.config.coordination.commitment_min_action_streak = 2
    domain.config.coordination.commitment_requires_direction = True
    domain.config.coordination.commitment_min_direction = 0.0
    domain.config.coordination.commitment_exit_policy_probability = 0.75
    domain.config.coordination.commitment_exit_on_negative_direction = True
    domain.config.coordination.precommitment_enabled = True
    domain.config.coordination.precommitment_min_policy_probability = 0.75
    domain.config.coordination.precommitment_min_evidence = 2.0
    domain.config.coordination.precommitment_evidence_increment = 1.0
    domain.config.coordination.precommitment_evidence_decay = 1.0
    domain.config.coordination.precommitment_requires_direction = True
    domain.config.coordination.precommitment_min_direction = 0.0
    state = make_state([0, 0, 0])
    first_context = BinaryStepContext(epoch=1, revision_mask=np.ones(3, dtype=bool))
    second_context = BinaryStepContext(epoch=2, revision_mask=np.ones(3, dtype=bool))
    third_context = BinaryStepContext(epoch=3, revision_mask=np.ones(3, dtype=bool))
    local_result = BinaryLocalStepResult(
        pre_revision_probs=binary_policy(np.asarray([0.5, 0.5, 0.5])),
        candidate_action_probs=np.asarray([0.92, 0.92, 0.92]),
        post_local_probs=binary_policy(np.asarray([0.92, 0.92, 0.92])),
        local_losses=[0.0, 0.0, 0.0],
        social_mode="policy_distill",
        actions_after_revision=np.asarray([0, 0, 0]),
        extras={
            "state_continuation_components": SimpleNamespace(
                effective=np.asarray([0.1, 0.1, 0.1]),
            ),
        },
    )
    social_result = BinarySocialStepResult(
        peer_ids=[[1], [0], [1]],
        post_social_probs=binary_policy(np.asarray([0.92, 0.92, 0.92])),
        final_action_probs=np.asarray([0.92, 0.92, 0.92]),
        social_losses=[0.0, 0.0, 0.0],
    )

    first = domain.apply_action_commitment(
        state=state,
        context=first_context,
        local_result=local_result,
        social_result=social_result,
        actions=np.asarray([0, 0, 0]),
    )
    second = domain.apply_action_commitment(
        state=state,
        context=second_context,
        local_result=local_result,
        social_result=social_result,
        actions=np.asarray([0, 0, 0]),
    )
    third = domain.apply_action_commitment(
        state=state,
        context=third_context,
        local_result=local_result,
        social_result=social_result,
        actions=np.asarray([0, 0, 0]),
    )

    np.testing.assert_array_equal(first.actions, [0, 0, 0])
    assert first.diagnostics["precommitment_ready_count"] == 0
    assert first.diagnostics["precommitment_mean_evidence"] == pytest.approx(1.0)
    assert first.diagnostics["precommitment_first_ready_epoch"] == ""
    assert first.diagnostics["precommitment_all_ready_epoch"] == ""
    assert first.diagnostics["precommitment_high_policy_rate"] == pytest.approx(1.0)
    assert first.diagnostics["precommitment_direction_ok_rate"] == pytest.approx(1.0)
    np.testing.assert_array_equal(second.actions, [1, 1, 1])
    assert second.diagnostics["precommitment_ready_count"] == 3
    assert second.diagnostics["precommitment_forced_action_count"] == 3
    assert second.diagnostics["precommitment_first_ready_epoch"] == 2
    assert second.diagnostics["precommitment_all_ready_epoch"] == 2
    assert second.diagnostics["precommitment_first_forced_epoch"] == 2
    assert second.diagnostics["precommitment_ready_to_forced_delay_mean"] == (
        pytest.approx(0.0)
    )
    assert second.diagnostics["precommitment_premature_exit_count"] == 0
    assert second.diagnostics["commitment_entry_count"] == 0
    np.testing.assert_array_equal(third.actions, [1, 1, 1])
    assert third.diagnostics["precommitment_ready_count"] == 3
    assert third.diagnostics["commitment_entry_count"] == 3
    assert third.diagnostics["commitment_rate"] == pytest.approx(1.0)
    assert third.diagnostics["precommitment_first_ready_epoch"] == 2
    assert third.diagnostics["precommitment_all_ready_epoch"] == 2
    assert third.diagnostics["precommitment_first_forced_epoch"] == 2
    np.testing.assert_array_equal(
        state.extras["_binary_action_precommitment_first_ready_epoch"],
        [2.0, 2.0, 2.0],
    )
    np.testing.assert_array_equal(
        state.extras["_binary_action_precommitment_first_forced_epoch"],
        [2.0, 2.0, 2.0],
    )


def test_action_precommitment_accumulates_peer_readiness_evidence() -> None:
    domain = FakeBinaryToyDomain()
    domain.config.coordination.precommitment_enabled = True
    domain.config.coordination.precommitment_min_policy_probability = 1.0
    domain.config.coordination.precommitment_min_evidence = 1.0
    domain.config.coordination.precommitment_evidence_increment = 0.0
    domain.config.coordination.precommitment_evidence_decay = 1.0
    domain.config.coordination.precommitment_requires_direction = False
    domain.config.coordination.precommitment_peer_evidence_enabled = True
    domain.config.coordination.precommitment_peer_evidence_weight = 1.0
    state = make_state([0, 0, 0])
    state.extras["_binary_action_precommitment_evidence"] = np.asarray(
        [1.0, 0.0, 0.0],
        dtype=np.float64,
    )
    context = BinaryStepContext(epoch=1, revision_mask=np.ones(3, dtype=bool))
    local_result = BinaryLocalStepResult(
        pre_revision_probs=binary_policy(np.asarray([0.5, 0.5, 0.5])),
        candidate_action_probs=np.asarray([0.5, 0.5, 0.5]),
        post_local_probs=binary_policy(np.asarray([0.5, 0.5, 0.5])),
        local_losses=[0.0, 0.0, 0.0],
        social_mode="policy_distill",
        actions_after_revision=np.asarray([0, 0, 0]),
    )
    social_result = BinarySocialStepResult(
        peer_ids=[[1], [0], [0]],
        post_social_probs=binary_policy(np.asarray([0.5, 0.5, 0.5])),
        final_action_probs=np.asarray([0.5, 0.5, 0.5]),
        social_losses=[0.0, 0.0, 0.0],
    )

    result = domain.apply_action_commitment(
        state=state,
        context=context,
        local_result=local_result,
        social_result=social_result,
        actions=np.asarray([0, 0, 0]),
    )

    np.testing.assert_array_equal(result.actions, [1, 1, 1])
    assert result.diagnostics["precommitment_ready_count"] == 3
    assert result.diagnostics["precommitment_signal_rate"] == pytest.approx(0.0)
    assert result.diagnostics["precommitment_peer_evidence_enabled"] is True
    assert result.diagnostics["precommitment_peer_evidence_weight"] == pytest.approx(
        1.0
    )
    assert result.diagnostics["precommitment_peer_readiness_aggregation"] == "mean"
    assert result.diagnostics["precommitment_peer_readiness_mean"] == pytest.approx(
        2.0 / 3.0
    )
    assert result.diagnostics["precommitment_peer_readiness_active_rate"] == (
        pytest.approx(2.0 / 3.0)
    )
    assert result.diagnostics["precommitment_peer_evidence_increment_mean"] == (
        pytest.approx(2.0 / 3.0)
    )
    assert result.diagnostics["precommitment_ready_largest_component_fraction"] == (
        pytest.approx(1.0)
    )
    np.testing.assert_allclose(
        state.extras["_binary_action_precommitment_peer_readiness"],
        [0.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(
        state.extras["_binary_action_precommitment_peer_evidence_increment"],
        [0.0, 1.0, 1.0],
    )


def test_action_precommitment_diagnostics_count_ready_drop_without_commitment() -> None:
    domain = FakeBinaryToyDomain()
    domain.config.coordination.precommitment_enabled = True
    domain.config.coordination.precommitment_min_policy_probability = 0.75
    domain.config.coordination.precommitment_min_evidence = 1.0
    domain.config.coordination.precommitment_evidence_increment = 1.0
    domain.config.coordination.precommitment_evidence_decay = 0.0
    domain.config.coordination.precommitment_requires_direction = False
    state = make_state([0, 0, 0])
    first_context = BinaryStepContext(epoch=1, revision_mask=np.ones(3, dtype=bool))
    second_context = BinaryStepContext(epoch=2, revision_mask=np.ones(3, dtype=bool))
    high_local = BinaryLocalStepResult(
        pre_revision_probs=binary_policy(np.asarray([0.5, 0.5, 0.5])),
        candidate_action_probs=np.asarray([0.92, 0.92, 0.92]),
        post_local_probs=binary_policy(np.asarray([0.92, 0.92, 0.92])),
        local_losses=[0.0, 0.0, 0.0],
        social_mode="policy_distill",
        actions_after_revision=np.asarray([0, 0, 0]),
    )
    high_social = BinarySocialStepResult(
        peer_ids=[[1], [0], [1]],
        post_social_probs=binary_policy(np.asarray([0.92, 0.92, 0.92])),
        final_action_probs=np.asarray([0.92, 0.92, 0.92]),
        social_losses=[0.0, 0.0, 0.0],
    )
    low_social = BinarySocialStepResult(
        peer_ids=[[1], [0], [1]],
        post_social_probs=binary_policy(np.asarray([0.4, 0.4, 0.4])),
        final_action_probs=np.asarray([0.4, 0.4, 0.4]),
        social_losses=[0.0, 0.0, 0.0],
    )

    first = domain.apply_action_commitment(
        state=state,
        context=first_context,
        local_result=high_local,
        social_result=high_social,
        actions=np.asarray([0, 0, 0]),
    )
    second = domain.apply_action_commitment(
        state=state,
        context=second_context,
        local_result=high_local,
        social_result=low_social,
        actions=np.asarray([1, 1, 1]),
    )

    assert first.diagnostics["precommitment_ready_count"] == 3
    assert first.diagnostics["precommitment_first_ready_epoch"] == 1
    assert second.diagnostics["precommitment_ready_count"] == 0
    assert second.diagnostics["precommitment_premature_exit_count"] == 3
    assert second.diagnostics["precommitment_first_ready_epoch"] == 1


def test_precommitment_decision_feedback_blends_prior_readiness() -> None:
    domain = FakeBinaryToyDomain()
    domain.config.coordination.precommitment_enabled = True
    domain.config.coordination.precommitment_min_evidence = 2.0
    domain.config.coordination.precommitment_decision_feedback_enabled = True
    domain.config.coordination.precommitment_decision_feedback_weight = 0.5
    state = make_state([0, 0, 0])
    state.extras["_binary_action_precommitment_evidence"] = np.asarray(
        [0.0, 1.0, 2.0],
        dtype=np.float64,
    )

    adjusted = domain.apply_precommitment_decision_feedback(
        state,
        torch.tensor(
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
            dtype=torch.float32,
        ),
    )

    torch.testing.assert_close(
        adjusted,
        torch.tensor(
            [[0.5, 0.5], [0.375, 0.625], [0.25, 0.75]],
            dtype=torch.float32,
        ),
    )
    diagnostics = state.extras[
        "_binary_precommitment_decision_feedback_diagnostics"
    ]
    assert diagnostics["precommitment_decision_feedback_enabled"] is True
    assert diagnostics["precommitment_decision_feedback_weight"] == pytest.approx(0.5)
    assert diagnostics["precommitment_decision_feedback_mean"] == pytest.approx(0.25)
    assert diagnostics["precommitment_decision_feedback_active_rate"] == pytest.approx(
        2.0 / 3.0
    )
    assert diagnostics["precommitment_decision_feedback_delta_mean"] == pytest.approx(
        0.125
    )


def test_binary_policy_readout_helpers_accept_numpy_policy_matrix() -> None:
    policy = binary_policy(np.asarray([0.1, 0.9, 0.4]))

    assert binary_policy_prob(policy, 1) == pytest.approx(0.9)
    assert mean_binary_policy_prob(policy) == pytest.approx((0.1 + 0.9 + 0.4) / 3.0)


def test_binary_peer_metrics_include_components_counts_and_entropy() -> None:
    peer_ids = [[1, 2], [0], [0]]

    metrics = binary_peer_metrics(
        peer_ids=peer_ids,
        agent_count=3,
        include_edge_entropy=True,
    )
    components = binary_peer_component_map(peer_ids=peer_ids, agent_count=3)

    assert metrics["fragmentation_components"] == 1
    assert metrics["mean_peer_count"] == pytest.approx(4.0 / 3.0)
    assert metrics["edge_entropy"] == pytest.approx(1.0 / 3.0)
    assert components == {0: 0, 1: 0, 2: 0}


def test_binary_loss_metrics_handle_empty_and_missing_losses() -> None:
    metrics = binary_loss_metrics(
        local_losses=[1.0, 3.0],
        social_losses=None,
        revised_local_losses=[],
    )

    assert metrics == {
        "mean_local_loss": 2.0,
        "mean_social_loss": 0.0,
        "mean_revised_local_loss": 0.0,
    }


def test_binary_loss_metrics_accept_tensor_backed_losses() -> None:
    losses = TensorBackedLossVector.from_tensor(torch.tensor([1.0, 2.0, 3.0]))

    metrics = binary_loss_metrics(
        local_losses=losses,
        social_losses=losses,
        revised_local_losses=losses,
    )

    assert metrics == {
        "mean_local_loss": 2.0,
        "mean_social_loss": 2.0,
        "mean_revised_local_loss": 2.0,
    }


def test_binary_policy_confidence_measures_distance_from_indifference() -> None:
    confidence = binary_policy_confidence(np.asarray([0.5, 0.75, 0.0, 1.0]))
    weights = binary_policy_confidence_weights(
        np.asarray([0.5, 0.75, 0.0, 1.0]),
        floor=0.2,
        power=1.0,
    )

    np.testing.assert_allclose(confidence, [0.0, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(weights, [0.2, 0.6, 1.0, 1.0])


def test_binary_policy_direction_alignment_scores_policy_objective_agreement() -> None:
    alignment = binary_policy_direction_alignment(
        np.asarray([0.8, 0.8, 0.2, 0.2, 0.9]),
        np.asarray([1.0, -1.0, 1.0, -1.0, 0.0]),
    )

    np.testing.assert_allclose(alignment, [0.8, 0.2, 0.2, 0.8, 0.5])


def test_confidence_weighted_binary_mix_preserves_against_ambivalent_peer() -> None:
    mix_result, diagnostics = mix_binary_output_confidence_weighted(
        action_probs=np.asarray([0.8, 0.5, 0.2]),
        peer_ids=[[1], [0], [0]],
        alpha=0.5,
    )

    np.testing.assert_allclose(mix_result.mixed_values, [0.8, 0.59, 0.38])
    assert mix_result.update_norms[0] == pytest.approx(0.0)
    assert diagnostics.micro_row(0)["social_peer_confidence"] == pytest.approx(0.0)
    assert diagnostics.micro_row(0)["social_effective_alpha"] == pytest.approx(0.0)
    assert diagnostics.micro_row(1)["social_peer_confidence"] == pytest.approx(0.6)
    assert diagnostics.micro_row(1)["social_effective_alpha"] == pytest.approx(0.3)
    assert diagnostics.aggregate_row()["social_confidence_weighting"] == "peer"
    assert diagnostics.aggregate_row()["max_social_effective_alpha"] == pytest.approx(
        0.3
    )


def test_precommitment_readiness_boosts_peer_confidence_weight() -> None:
    mix_result, diagnostics = mix_binary_output_confidence_weighted(
        action_probs=np.asarray([0.8, 0.5, 0.2]),
        peer_ids=[[1], [0], [0]],
        alpha=0.5,
        precommitment_readiness=np.asarray([0.0, 1.0, 0.0]),
        precommitment_readiness_weight=1.0,
    )

    np.testing.assert_allclose(mix_result.mixed_values, [0.65, 0.59, 0.38])
    assert diagnostics.micro_row(0)["social_peer_confidence"] == pytest.approx(1.0)
    assert diagnostics.micro_row(0)[
        "social_peer_precommitment_readiness"
    ] == pytest.approx(1.0)
    assert diagnostics.micro_row(0)["social_effective_alpha"] == pytest.approx(0.5)
    assert diagnostics.aggregate_row()[
        "precommitment_social_feedback_enabled"
    ] is True
    assert diagnostics.aggregate_row()[
        "mean_social_peer_precommitment_readiness"
    ] == pytest.approx(1.0 / 3.0)


def test_directional_confidence_mix_downweights_objective_misaligned_peers() -> None:
    action_probs = np.asarray([0.2, 0.8])
    peer_ids = [[1], [0]]

    peer_mix, _peer_diagnostics = mix_binary_output_confidence_weighted(
        action_probs=action_probs,
        peer_ids=peer_ids,
        alpha=0.5,
    )
    directional_mix, directional_diagnostics = mix_binary_output_confidence_weighted(
        action_probs=action_probs,
        peer_ids=peer_ids,
        alpha=0.5,
        direction_scores=np.asarray([1.0, -1.0]),
        weighting="peer_direction",
    )

    np.testing.assert_allclose(peer_mix.mixed_values, [0.38, 0.62])
    np.testing.assert_allclose(directional_mix.mixed_values, [0.236, 0.764])
    assert directional_diagnostics.aggregate_row()[
        "social_confidence_weighting"
    ] == "peer_direction"
    assert directional_diagnostics.micro_row(0)[
        "social_effective_alpha"
    ] == pytest.approx(0.06)


def test_confidence_weighted_distribution_mix_uses_binary_policy_target() -> None:
    previous = torch.tensor(
        [[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]],
        dtype=torch.float32,
    )

    mix_result, diagnostics = mix_binary_policy_distribution_confidence_weighted(
        previous_probs=previous,
        peer_ids=[[1], [0], [0]],
        alpha=0.5,
    )

    expected = torch.tensor(
        [[0.2, 0.8], [0.41, 0.59], [0.62, 0.38]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(mix_result.mixed_values, expected)
    assert mix_result.commit_mode == "confidence_weighted_distillation_step"
    assert mix_result.active_agent_ids == [1, 2]
    assert diagnostics.aggregate_row()["mean_social_effective_alpha"] == pytest.approx(
        0.2
    )


def test_directional_distribution_mix_requires_direction_scores() -> None:
    previous = torch.tensor(
        [[0.8, 0.2], [0.2, 0.8]],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="direction_scores"):
        mix_binary_policy_distribution_confidence_weighted(
            previous_probs=previous,
            peer_ids=[[1], [0]],
            alpha=0.5,
            weighting="peer_direction",
        )


def test_binary_common_rows_accept_torch_state_arrays() -> None:
    config = SimpleNamespace(
        run=SimpleNamespace(name="run", seed=7),
        policy=SimpleNamespace(rule="neural_policy", revision_rate=0.5),
        coordination=SimpleNamespace(mixer="none", peer_rule="none"),
    )
    state = BinarySpatialState(
        actions=torch.tensor([1, 0], dtype=torch.long),
        payoffs=torch.tensor([3.0, 1.0], dtype=torch.float64),
        payoff_ema=torch.tensor([2.0, 0.5], dtype=torch.float64),
        previous_payoff_ema=torch.tensor([1.5, 0.25], dtype=torch.float64),
        reputation=torch.tensor([0.8, 0.2], dtype=torch.float64),
    )
    policy = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
    step_result = BinaryPolicyStepResult(
        pre_revision_probs=policy,
        post_local_probs=policy,
        post_social_probs=policy,
        local_losses=[0.1, 0.2],
        social_losses=[0.3, 0.4],
        peer_ids=[[1], [0]],
        revision_mask=np.asarray([True, False]),
        mobility_result=MobilityStepResult.none(2),
        realized_revision_rate=0.5,
        extras={
            "social_unit_aggregate": {
                "social_channel": "policy_distribution",
                "commit_mode": "distillation_step",
                "mean_social_update_norm": 0.25,
                "max_social_update_norm": 0.4,
                "active_social_agent_count": 2,
            },
            "social_unit_micro": [
                {
                    "social_channel": "policy_distribution",
                    "commit_mode": "distillation_step",
                    "social_update_norm": 0.1,
                },
                {
                    "social_channel": "policy_distribution",
                    "commit_mode": "distillation_step",
                    "social_update_norm": 0.4,
                },
            ],
        },
    )

    aggregate = binary_aggregate_common_fields(
        config=config,
        toy="toy",
        epoch=1,
        actions=state.actions,
        payoffs=state.payoffs,
        policy_probs=policy,
        peer_ids=step_result.peer_ids,
        realized_revision_rate=0.5,
        reputation=state.reputation,
        mobility_result=step_result.mobility_result,
        local_losses=step_result.local_losses,
        social_losses=step_result.social_losses,
        social_unit_aggregate=step_result.extras["social_unit_aggregate"],
    )
    micro = binary_micro_base_fields(
        config=config,
        toy="toy",
        epoch=1,
        agent_id=0,
        state=state,
        step_result=step_result,
        components={0: 0, 1: 0},
    )

    assert aggregate["action_rate"] == pytest.approx(0.5)
    assert aggregate["mean_payoff"] == pytest.approx(2.0)
    assert aggregate["mean_reputation"] == pytest.approx(0.5)
    assert aggregate["social_channel"] == "policy_distribution"
    assert aggregate["commit_mode"] == "distillation_step"
    assert aggregate["mean_social_update_norm"] == pytest.approx(0.25)
    assert aggregate["max_social_update_norm"] == pytest.approx(0.4)
    assert aggregate["active_social_agent_count"] == 2
    assert aggregate["social_confidence_weighting"] == "none"
    assert aggregate["social_tail_floor_active"] is False
    assert aggregate["social_tail_confidence_floor"] == pytest.approx(0.0)
    assert aggregate["social_tail_policy_rate"] == pytest.approx(0.0)
    assert aggregate["social_tail_action_rate"] == pytest.approx(0.0)
    assert aggregate["mean_social_effective_alpha"] == pytest.approx(0.0)
    assert aggregate["precommitment_social_feedback_enabled"] is False
    assert aggregate["precommitment_social_feedback_weight"] == pytest.approx(0.0)
    assert aggregate["mean_social_peer_precommitment_readiness"] == pytest.approx(0.0)
    assert aggregate["social_peer_precommitment_readiness_active_rate"] == pytest.approx(
        0.0
    )
    assert aggregate["commitment_enabled"] is False
    assert aggregate["commitment_rate"] == pytest.approx(0.0)
    assert aggregate["commitment_entry_count"] == 0
    assert aggregate["commitment_exit_count"] == 0
    assert aggregate["committed_action_rate"] == pytest.approx(0.0)
    assert aggregate["uncommitted_high_policy_rate"] == pytest.approx(0.0)
    assert aggregate["final_uncommitted_near_ceiling_count"] == 0
    assert aggregate["commitment_forced_action_count"] == 0
    assert aggregate["precommitment_enabled"] is False
    assert aggregate["precommitment_rate"] == pytest.approx(0.0)
    assert aggregate["precommitment_mean_evidence"] == pytest.approx(0.0)
    assert aggregate["precommitment_ready_count"] == 0
    assert aggregate["precommitment_forced_action_count"] == 0
    assert aggregate["precommitment_signal_rate"] == pytest.approx(0.0)
    assert aggregate["precommitment_first_ready_epoch"] == ""
    assert aggregate["precommitment_all_ready_epoch"] == ""
    assert aggregate["precommitment_first_forced_epoch"] == ""
    assert aggregate["precommitment_ready_to_forced_delay_mean"] == ""
    assert aggregate["precommitment_premature_exit_count"] == 0
    assert aggregate["precommitment_high_policy_rate"] == pytest.approx(0.0)
    assert aggregate["precommitment_direction_ok_rate"] == pytest.approx(0.0)
    assert aggregate["precommitment_ready_largest_component_fraction"] == (
        pytest.approx(0.0)
    )
    assert aggregate["precommitment_peer_evidence_enabled"] is False
    assert aggregate["precommitment_peer_evidence_weight"] == pytest.approx(0.0)
    assert aggregate["precommitment_peer_readiness_mean"] == pytest.approx(0.0)
    assert aggregate["precommitment_peer_readiness_active_rate"] == pytest.approx(0.0)
    assert aggregate["precommitment_peer_evidence_increment_mean"] == pytest.approx(
        0.0
    )
    assert aggregate["precommitment_decision_feedback_enabled"] is False
    assert aggregate["precommitment_decision_feedback_weight"] == pytest.approx(0.0)
    assert aggregate["precommitment_decision_feedback_mean"] == pytest.approx(0.0)
    assert aggregate["precommitment_decision_feedback_active_rate"] == pytest.approx(
        0.0
    )
    assert aggregate["precommitment_decision_feedback_delta_mean"] == pytest.approx(
        0.0
    )
    assert micro["action"] == 1
    assert micro["payoff"] == pytest.approx(3.0)
    assert micro["payoff_ema"] == pytest.approx(2.0)
    assert micro["reputation"] == pytest.approx(0.8)
    assert micro["social_channel"] == "policy_distribution"
    assert micro["commit_mode"] == "distillation_step"
    assert micro["social_update_norm"] == pytest.approx(0.1)
    assert micro["social_confidence_weighting"] == "none"
    assert micro["social_effective_alpha"] == pytest.approx(0.0)
    assert micro["social_peer_precommitment_readiness"] == pytest.approx(0.0)
    assert micro["precommitment_evidence"] == pytest.approx(0.0)
    assert micro["precommitment_ready"] is False
    assert micro["precommitment_signal"] is False
    assert micro["precommitment_high_policy"] is False
    assert micro["precommitment_direction_ok"] is False
    assert micro["precommitment_forced_action"] is False
    assert micro["precommitment_peer_readiness"] == pytest.approx(0.0)
    assert micro["precommitment_peer_evidence_increment"] == pytest.approx(0.0)
    assert micro["precommitment_first_ready_epoch"] == ""
    assert micro["precommitment_first_forced_epoch"] == ""


def test_binary_micro_common_fields_preserve_peer_revision_and_loss_values() -> None:
    fields = binary_micro_common_fields(
        run_id="run",
        seed=7,
        epoch=2,
        agent_id=1,
        coordination_mixer="output_average",
        coordination_peer_rule="output_similarity",
        peer_ids=[[1], [0, 2], []],
        components={0: 0, 1: 0, 2: 1},
        revision_mask=np.asarray([False, True, False]),
        local_losses=[0.1, 0.2, 0.3],
        social_losses=[0.4, 0.5, 0.6],
    )

    assert fields == {
        "run_id": "run",
        "seed": 7,
        "epoch": 2,
        "agent_id": 1,
        "coordination_mixer": "output_average",
        "coordination_peer_rule": "output_similarity",
        "peer_ids": [0, 2],
        "peer_count": 2,
        "component_id": 0,
        "revised": True,
        "local_loss": 0.2,
        "social_loss": 0.5,
        "social_channel": "",
        "commit_mode": "",
        "social_update_norm": 0.0,
        "social_confidence_weighting": "none",
        "social_peer_confidence": 0.0,
        "social_effective_alpha": 0.0,
        "social_peer_precommitment_readiness": 0.0,
        "precommitment_evidence": 0.0,
        "precommitment_ready": False,
        "precommitment_signal": False,
        "precommitment_high_policy": False,
        "precommitment_direction_ok": False,
        "precommitment_forced_action": False,
        "precommitment_peer_readiness": 0.0,
        "precommitment_peer_evidence_increment": 0.0,
        "precommitment_first_ready_epoch": "",
        "precommitment_first_forced_epoch": "",
    }


def test_binary_micro_mobility_fields_preserve_empty_target_convention() -> None:
    mobility_result = MobilityStepResult(
        moved=np.asarray([False, True]),
        targets=np.asarray([-1, 0]),
        gains=np.asarray([0.0, 1.5]),
    )

    assert binary_micro_mobility_fields(mobility_result, 0) == {
        "mobility_moved": False,
        "mobility_target": "",
        "mobility_gain": 0.0,
    }
    assert binary_micro_mobility_fields(mobility_result, 1) == {
        "mobility_moved": True,
        "mobility_target": 0,
        "mobility_gain": 1.5,
    }


def test_binary_policy_learning_unit_runs_readout_update_and_refresh() -> None:
    agents = ["a0", "a1"]
    observations = torch.eye(2, dtype=torch.float32)
    calls: list[str] = []

    def collect_policy_probs(
        agents_arg: list[str],
        observations_arg: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        assert agents_arg == agents
        torch.testing.assert_close(observations_arg, observations)
        assert temperature == pytest.approx(0.75)
        calls.append("readout")
        return torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float32)

    def collect_post_policy_probs(
        agents_arg: list[str],
        observations_arg: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        assert agents_arg == agents
        torch.testing.assert_close(observations_arg, observations)
        assert temperature == pytest.approx(0.75)
        calls.append("post_readout")
        return torch.tensor([[0.1, 0.9], [0.7, 0.3]], dtype=torch.float32)

    def decision_action_probs(policy_probs: torch.Tensor) -> torch.Tensor:
        calls.append("decision")
        return policy_probs

    def sample_actions(policy_probs: torch.Tensor) -> np.ndarray:
        calls.append("sample")
        return (policy_probs[:, 1].detach().cpu().numpy() > 0.5).astype(np.int64)

    def local_update(actions: np.ndarray) -> list[float]:
        calls.append("local")
        np.testing.assert_array_equal(actions, [1, 0])
        return [0.1, 0.0]

    def refresh_policy_cache(agents_arg: list[str]) -> None:
        assert agents_arg == agents
        calls.append("refresh")

    result = BinaryPolicyLearningUnit(
        agents=agents,
        observations=observations,
        temperature=0.75,
        callbacks=BinaryPolicyLearningCallbacks(
            collect_policy_probs=collect_policy_probs,
            decision_action_probs=decision_action_probs,
            sample_actions=sample_actions,
            local_update=local_update,
            refresh_policy_cache=refresh_policy_cache,
            post_collect_policy_probs=collect_post_policy_probs,
        ),
    ).run()

    assert calls == [
        "readout",
        "decision",
        "sample",
        "local",
        "refresh",
        "post_readout",
    ]
    np.testing.assert_array_equal(result.actions_after_revision, [1, 0])
    assert result.local_losses == [0.1, 0.0]
    torch.testing.assert_close(
        result.post_local_probs,
        torch.tensor([[0.1, 0.9], [0.7, 0.3]], dtype=torch.float32),
    )


def test_run_binary_policy_learning_step_wires_callbacks_through_unit() -> None:
    agents = ["a0", "a1"]
    observations = torch.eye(2, dtype=torch.float32)
    calls: list[str] = []

    def collect_policy_probs(
        agents_arg: list[str],
        observations_arg: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        assert agents_arg == agents
        torch.testing.assert_close(observations_arg, observations)
        assert temperature == pytest.approx(0.5)
        calls.append("readout")
        return torch.tensor([[0.4, 0.6], [0.7, 0.3]], dtype=torch.float32)

    def build_decision_probs(policy_probs: torch.Tensor) -> torch.Tensor:
        calls.append("decision")
        return policy_probs

    def sample_actions(policy_probs: torch.Tensor) -> np.ndarray:
        calls.append("sample")
        return (policy_probs[:, 1].detach().cpu().numpy() >= 0.5).astype(np.int64)

    def local_update(actions: np.ndarray) -> list[float]:
        calls.append("local")
        np.testing.assert_array_equal(actions, [1, 0])
        return [0.2, 0.3]

    def refresh_policy_cache(agents_arg: list[str]) -> None:
        assert agents_arg == agents
        calls.append("refresh")

    result = run_binary_policy_learning_step(
        agents=agents,
        observations=observations,
        temperature=0.5,
        collect_policy_probs=collect_policy_probs,
        decision_action_probs=build_decision_probs,
        sample_actions=sample_actions,
        local_update=local_update,
        refresh_policy_cache=refresh_policy_cache,
        extras={"source": "helper-contract"},
    )

    assert calls == [
        "readout",
        "decision",
        "sample",
        "local",
        "refresh",
        "readout",
    ]
    np.testing.assert_array_equal(result.actions_after_revision, [1, 0])
    assert result.local_losses == [0.2, 0.3]
    assert result.extras == {"source": "helper-contract"}


def test_binary_policy_learning_unit_contract_preserves_context_and_extras() -> None:
    agents = ["holdout-a0", "holdout-a1", "holdout-a2"]
    observations = torch.eye(3, dtype=torch.float32)
    calls: list[str] = []
    timing_stages: list[str] = []
    sync_count = 0
    extras = {"fixture": "toy_independent_binary_policy_contract"}

    def record_timing(stage: str, seconds: float) -> None:
        assert seconds >= 0.0
        timing_stages.append(stage)

    def synchronize_timing_device() -> None:
        nonlocal sync_count
        sync_count += 1

    context = BinaryStepContext(
        epoch=7,
        revision_mask=np.asarray([True, False, True], dtype=bool),
        extras={
            "_record_timing": record_timing,
            "_synchronize_timing_device": synchronize_timing_device,
        },
    )
    pre_probs = torch.tensor(
        [[0.30, 0.70], [0.55, 0.45], [0.10, 0.90]],
        dtype=torch.float32,
    )
    decision_probs = torch.tensor(
        [[0.20, 0.80], [0.80, 0.20], [0.40, 0.60]],
        dtype=torch.float32,
    )
    post_probs = torch.tensor(
        [[0.10, 0.90], [0.70, 0.30], [0.35, 0.65]],
        dtype=torch.float32,
    )
    refreshed = False

    def collect_policy_probs(
        agents_arg: list[str],
        observations_arg: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        assert agents_arg == agents
        torch.testing.assert_close(observations_arg, observations)
        assert temperature == pytest.approx(0.5)
        calls.append("readout")
        return pre_probs

    def build_decision_probs(policy_probs: torch.Tensor) -> torch.Tensor:
        calls.append("decision")
        torch.testing.assert_close(policy_probs, pre_probs)
        return decision_probs

    def sample_actions(action_probs: torch.Tensor) -> np.ndarray:
        calls.append("sample")
        torch.testing.assert_close(action_probs, decision_probs)
        return np.asarray([1, 0, 1], dtype=np.int64)

    def local_update(actions: np.ndarray) -> list[float]:
        calls.append("local")
        np.testing.assert_array_equal(actions, [1, 0, 1])
        return [0.10, 0.20, 0.30]

    def refresh_policy_cache(agents_arg: list[str]) -> None:
        nonlocal refreshed
        calls.append("refresh")
        assert agents_arg == agents
        refreshed = True

    def post_collect_policy_probs(
        agents_arg: list[str],
        observations_arg: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        assert refreshed
        assert agents_arg == agents
        torch.testing.assert_close(observations_arg, observations)
        assert temperature == pytest.approx(0.5)
        calls.append("post_readout")
        return post_probs

    result = BinaryPolicyLearningUnit(
        agents=agents,
        observations=observations,
        temperature=0.5,
        callbacks=BinaryPolicyLearningCallbacks(
            collect_policy_probs=collect_policy_probs,
            decision_action_probs=build_decision_probs,
            sample_actions=sample_actions,
            local_update=local_update,
            refresh_policy_cache=refresh_policy_cache,
            post_collect_policy_probs=post_collect_policy_probs,
        ),
        context=context,
        extras=extras,
    ).run()

    assert calls == [
        "readout",
        "decision",
        "sample",
        "local",
        "refresh",
        "post_readout",
    ]
    assert timing_stages == [
        "policy_readout",
        "decision_selection",
        "local_training",
        "cache_refresh",
        "post_local_readout",
    ]
    assert sync_count == 10
    torch.testing.assert_close(result.pre_revision_probs, pre_probs)
    torch.testing.assert_close(result.decision_action_probs, decision_probs)
    np.testing.assert_array_equal(result.actions_after_revision, [1, 0, 1])
    assert result.local_losses == [0.10, 0.20, 0.30]
    torch.testing.assert_close(result.post_local_probs, post_probs)
    assert result.extras == extras
    assert result.extras is not extras


def test_binary_policy_learning_callbacks_reuse_readout_for_post_when_unspecified() -> None:
    observations = torch.eye(2, dtype=torch.float32)
    calls: list[str] = []

    def collect_policy_probs(
        agents_arg: list[str],
        observations_arg: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        assert agents_arg == ["a0", "a1"]
        torch.testing.assert_close(observations_arg, observations)
        assert temperature == pytest.approx(1.0)
        calls.append("readout")
        return torch.tensor([[0.25, 0.75], [0.8, 0.2]], dtype=torch.float32)

    result = BinaryPolicyLearningUnit(
        agents=["a0", "a1"],
        observations=observations,
        temperature=1.0,
        callbacks=BinaryPolicyLearningCallbacks(
            collect_policy_probs=collect_policy_probs,
            decision_action_probs=lambda probs: probs,
            sample_actions=lambda probs: np.asarray([1, 0], dtype=np.int64),
            local_update=lambda actions: [0.0 for _ in actions],
        ),
    ).run()

    assert calls == ["readout", "readout"]
    torch.testing.assert_close(result.pre_revision_probs, result.post_local_probs)


def test_distill_binary_policy_output_average_builds_social_result() -> None:
    agents = ["agent-a", "agent-b"]
    observations = np.asarray([[1.0], [2.0]])
    previous_probs = binary_policy(np.asarray([0.1, 0.9]))
    captured: dict[str, Any] = {}

    def distill_policy(**kwargs: Any) -> list[float]:
        captured.update(kwargs)
        return [0.4, 0.5]

    def collect_policy_probs(
        agents_arg: list[str],
        observations_arg: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        assert agents_arg == agents
        np.testing.assert_allclose(observations_arg, observations)
        assert temperature == 0.75
        return binary_policy(np.asarray([0.7, 0.2]))

    result = distill_binary_policy_output_average(
        agents=agents,
        observations=observations,
        peer_ids=[[1], [0]],
        alpha=0.25,
        previous_probs=previous_probs,
        temperature=0.75,
        collect_policy_probs=collect_policy_probs,
        distill_policy=distill_policy,
    )

    assert captured["agents"] == agents
    assert captured["peer_ids"] == [[1], [0]]
    assert captured["alpha"] == 0.25
    np.testing.assert_allclose(captured["observations"], observations)
    np.testing.assert_allclose(captured["previous_probs"], previous_probs)
    np.testing.assert_allclose(result.final_action_probs, [0.7, 0.2])
    np.testing.assert_allclose(result.post_social_probs[:, 1], [0.7, 0.2])
    assert result.social_losses == [0.4, 0.5]


def test_distill_binary_policy_output_average_preserves_unit_diagnostics() -> None:
    previous_probs = binary_policy(np.asarray([0.1, 0.9]))

    def distill_policy(**kwargs: Any) -> BinaryOutputDistillationReport:
        del kwargs
        return BinaryOutputDistillationReport(
            social_losses=[0.4, 0.5],
            aggregate_diagnostics={
                "social_channel": "policy_distribution",
                "commit_mode": "distillation_step",
                "mean_social_loss": 0.45,
                "mean_social_update_norm": 0.2,
                "max_social_update_norm": 0.3,
                "active_social_agent_count": 2,
            },
            micro_diagnostics=[
                {
                    "social_channel": "policy_distribution",
                    "commit_mode": "distillation_step",
                    "social_loss": 0.4,
                    "social_update_norm": 0.1,
                },
                {
                    "social_channel": "policy_distribution",
                    "commit_mode": "distillation_step",
                    "social_loss": 0.5,
                    "social_update_norm": 0.3,
                },
            ],
        )

    result = distill_binary_policy_output_average(
        agents=["agent-a", "agent-b"],
        observations=np.asarray([[1.0], [2.0]]),
        peer_ids=[[1], [0]],
        alpha=0.25,
        previous_probs=previous_probs,
        temperature=1.0,
        collect_policy_probs=lambda *_args, **_kwargs: previous_probs,
        distill_policy=distill_policy,
    )

    assert result.extras["social_unit_aggregate"]["social_channel"] == (
        "policy_distribution"
    )
    assert result.extras["social_unit_aggregate"]["active_social_agent_count"] == 2
    assert result.extras["social_unit_micro"][1]["social_update_norm"] == pytest.approx(
        0.3,
    )


def test_accelerated_distillation_report_exposes_common_diagnostics() -> None:
    update_result = SimpleNamespace(losses=TensorBackedLossVector.from_tensor(
        torch.tensor([0.2, 0.0, 0.4]),
    ))

    report = BinaryOutputDistillationReport.from_accelerated_update_result(
        update_result,
        peer_ids=[[1], [], [0]],
        agent_count=3,
    )

    assert report.social_losses is update_result.losses
    assert report.aggregate_diagnostics["social_channel"] == "policy_distribution"
    assert report.aggregate_diagnostics["commit_mode"] == "distillation_step"
    assert report.aggregate_diagnostics["mean_social_loss"] == pytest.approx(0.2)
    assert report.aggregate_diagnostics["mean_social_update_norm"] == pytest.approx(0.0)
    assert report.aggregate_diagnostics["max_social_update_norm"] == pytest.approx(0.0)
    assert report.aggregate_diagnostics["active_social_agent_count"] == 2
    assert report.micro_diagnostics[2]["social_loss"] == pytest.approx(0.4)
    assert report.micro_diagnostics[2]["social_update_norm"] == pytest.approx(0.0)


def test_batched_distillation_report_alias_remains_compatible() -> None:
    update_result = SimpleNamespace(losses=[0.2, 0.0])

    report = BinaryOutputDistillationReport.from_batched_update_result(
        update_result,
        peer_ids=[[1], []],
        agent_count=2,
    )

    assert report.aggregate_diagnostics["active_social_agent_count"] == 1
    assert report.micro_diagnostics[0]["social_loss"] == pytest.approx(0.2)


def test_batched_distribution_distillation_adapter_matches_legacy_update() -> None:
    adapter_agents, legacy_agents = _make_tiny_batched_agents()
    observations = torch.tensor(
        [
            [0.2, -0.4, 0.8],
            [1.0, 0.3, -0.1],
            [-0.7, 0.6, 0.5],
        ],
        dtype=torch.float32,
    )
    previous_probs = torch.softmax(
        torch.tensor(
            [
                [0.3, -0.1],
                [-0.5, 0.7],
                [0.2, 0.4],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    peer_ids = [[1, 2], [0], []]
    alpha = 0.35
    adapter = BatchedDistributionDistillationAdapter(
        agents=adapter_agents,
        observations=observations,
        loss_mode="cross_entropy",
    )
    step = NABMStep(
        social_block=SocialBlock(alpha=alpha),
        channel=SocialChannel(
            name="policy_distribution",
            kind=PROBABILITY_DISTRIBUTION_CHANNEL,
            commit_mode="distillation_step",
        ),
        commit_adapter=adapter,
    )

    result = step.run(values=previous_probs, peer_ids=peer_ids)
    legacy_result = apply_batched_output_average_distillation_update(
        agents=legacy_agents,
        observations=observations,
        peer_ids=peer_ids,
        alpha=alpha,
        previous_probs=previous_probs,
        loss_mode="cross_entropy",
    )

    assert adapter.update_result is not None
    assert result.commit.committed_agent_ids == [0, 1]
    assert result.diagnostics.active_agent_count == 2
    assert result.diagnostics.max_update_norm > 0.0
    np.testing.assert_allclose(
        result.commit.losses,
        list(legacy_result.losses),
        atol=1e-6,
    )
    _assert_agent_parameters_match(adapter_agents, legacy_agents)


def test_batched_distribution_distillation_adapter_skips_empty_peers() -> None:
    agents, _legacy_agents = _make_tiny_batched_agents()
    before = _clone_agent_parameters(agents)
    observations = torch.tensor(
        [
            [0.2, -0.4, 0.8],
            [1.0, 0.3, -0.1],
            [-0.7, 0.6, 0.5],
        ],
        dtype=torch.float32,
    )
    previous_probs = torch.softmax(
        torch.tensor(
            [
                [0.3, -0.1],
                [-0.5, 0.7],
                [0.2, 0.4],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    adapter = BatchedDistributionDistillationAdapter(
        agents=agents,
        observations=observations,
        loss_mode="cross_entropy",
    )
    step = NABMStep(
        social_block=SocialBlock(alpha=0.5),
        channel=SocialChannel(
            name="policy_distribution",
            kind=PROBABILITY_DISTRIBUTION_CHANNEL,
            commit_mode="distillation_step",
        ),
        commit_adapter=adapter,
    )

    result = step.run(values=previous_probs, peer_ids=[[], [], []])

    assert adapter.update_result is not None
    assert result.mix.active_agent_ids == []
    assert result.commit.committed_agent_ids == []
    assert result.commit.losses == [0.0, 0.0, 0.0]
    assert result.diagnostics.active_agent_count == 0
    assert result.diagnostics.mean_update_norm == 0.0
    _assert_agent_parameters_equal_snapshot(agents, before)


def test_batched_policy_gradient_local_update_adapter_matches_direct_update() -> None:
    adapter_agents, direct_agents = _make_tiny_batched_agents()
    observations = torch.tensor(
        [
            [0.2, -0.4, 0.8],
            [1.0, 0.3, -0.1],
            [-0.7, 0.6, 0.5],
        ],
        dtype=torch.float32,
    )
    actions = torch.tensor([0, 1, 1], dtype=torch.long)
    advantages = torch.tensor([0.5, -0.25, 0.8], dtype=torch.float32)
    active_agent_ids = [0, 2]
    adapter = BatchedPolicyGradientLocalUpdateAdapter(
        agents=adapter_agents,
        observations=observations,
        actions=actions,
        advantages=advantages,
        active_agent_ids=active_agent_ids,
        entropy_beta=0.02,
    )

    report = NABMLocalStep(adapter).run()
    direct_parameters = trainable_batched_mlp_parameters(direct_agents)
    direct_losses = batched_binary_policy_gradient_losses(
        direct_parameters,
        observations,
        actions=actions,
        advantages=advantages,
        entropy_beta=0.02,
    )
    direct_result = apply_batched_mlp_loss_gradients_with_result(
        agents=direct_agents,
        parameters=direct_parameters,
        losses=direct_losses,
        active_agent_ids=active_agent_ids,
    )

    assert report.update_result is not None
    assert report.active_agent_ids == active_agent_ids
    np.testing.assert_allclose(
        list(report.losses),
        list(direct_result.losses),
        atol=1e-6,
    )
    _assert_agent_parameters_match(adapter_agents, direct_agents)


def test_tensor_runtime_distribution_distillation_adapter_matches_legacy_update() -> None:
    adapter_agents, legacy_agents = _make_tiny_batched_agents()
    adapter_runtime = TensorBatchedMLPRuntime.from_agents(adapter_agents)
    legacy_runtime = TensorBatchedMLPRuntime.from_agents(legacy_agents)
    observations = torch.tensor(
        [
            [0.2, -0.4, 0.8],
            [1.0, 0.3, -0.1],
            [-0.7, 0.6, 0.5],
        ],
        dtype=torch.float32,
    )
    previous_probs = torch.softmax(
        torch.tensor(
            [
                [0.3, -0.1],
                [-0.5, 0.7],
                [0.2, 0.4],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    peer_ids = [[1, 2], [0], []]
    alpha = 0.35
    adapter = TensorRuntimeDistributionDistillationAdapter(
        runtime=adapter_runtime,
        observations=observations,
        loss_mode="cross_entropy",
    )
    step = NABMStep(
        social_block=SocialBlock(alpha=alpha),
        channel=SocialChannel(
            name="policy_distribution",
            kind=PROBABILITY_DISTRIBUTION_CHANNEL,
            commit_mode="distillation_step",
        ),
        commit_adapter=adapter,
    )

    result = step.run(values=previous_probs, peer_ids=peer_ids)
    legacy_result = apply_tensor_output_average_distillation_update(
        runtime=legacy_runtime,
        observations=observations,
        peer_ids=peer_ids,
        alpha=alpha,
        previous_probs=previous_probs,
        loss_mode="cross_entropy",
    )

    assert adapter.update_result is not None
    assert result.commit.committed_agent_ids == [0, 1]
    assert result.diagnostics.active_agent_count == 2
    assert result.diagnostics.max_update_norm > 0.0
    np.testing.assert_allclose(
        result.commit.losses,
        list(legacy_result.losses),
        atol=1e-6,
    )
    _assert_runtime_parameters_match(adapter_runtime, legacy_runtime)


def test_tensor_runtime_distribution_distillation_adapter_skips_empty_peers() -> None:
    agents, _legacy_agents = _make_tiny_batched_agents()
    runtime = TensorBatchedMLPRuntime.from_agents(agents)
    before = _clone_runtime_parameters(runtime)
    observations = torch.tensor(
        [
            [0.2, -0.4, 0.8],
            [1.0, 0.3, -0.1],
            [-0.7, 0.6, 0.5],
        ],
        dtype=torch.float32,
    )
    previous_probs = torch.softmax(
        torch.tensor(
            [
                [0.3, -0.1],
                [-0.5, 0.7],
                [0.2, 0.4],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    adapter = TensorRuntimeDistributionDistillationAdapter(
        runtime=runtime,
        observations=observations,
        loss_mode="cross_entropy",
    )
    step = NABMStep(
        social_block=SocialBlock(alpha=0.5),
        channel=SocialChannel(
            name="policy_distribution",
            kind=PROBABILITY_DISTRIBUTION_CHANNEL,
            commit_mode="distillation_step",
        ),
        commit_adapter=adapter,
    )

    result = step.run(values=previous_probs, peer_ids=[[], [], []])

    assert adapter.update_result is not None
    assert result.mix.active_agent_ids == []
    assert result.commit.committed_agent_ids == []
    assert result.commit.losses == [0.0, 0.0, 0.0]
    assert result.diagnostics.active_agent_count == 0
    assert result.diagnostics.mean_update_norm == 0.0
    _assert_runtime_parameters_equal_snapshot(runtime, before)


def test_tensor_runtime_policy_gradient_local_update_adapter_matches_legacy_update() -> None:
    adapter_agents, legacy_agents = _make_tiny_batched_agents()
    adapter_runtime = TensorBatchedMLPRuntime.from_agents(adapter_agents)
    legacy_runtime = TensorBatchedMLPRuntime.from_agents(legacy_agents)
    observations = torch.tensor(
        [
            [0.2, -0.4, 0.8],
            [1.0, 0.3, -0.1],
            [-0.7, 0.6, 0.5],
        ],
        dtype=torch.float32,
    )
    actions = torch.tensor([0, 1, 1], dtype=torch.long)
    advantages = torch.tensor([0.5, -0.25, 0.8], dtype=torch.float32)
    active_agent_ids = [0, 2]
    adapter = TensorRuntimePolicyGradientLocalUpdateAdapter(
        runtime=adapter_runtime,
        observations=observations,
        actions=actions,
        advantages=advantages,
        active_agent_ids=active_agent_ids,
        entropy_beta=0.02,
    )

    report = NABMLocalStep(adapter).run()
    legacy_result = apply_tensor_binary_policy_gradient_update(
        runtime=legacy_runtime,
        observations=observations,
        actions=actions,
        advantages=advantages,
        active_agent_ids=active_agent_ids,
        entropy_beta=0.02,
    )

    assert report.update_result is not None
    assert report.active_agent_ids == active_agent_ids
    np.testing.assert_allclose(
        list(report.losses),
        list(legacy_result.losses),
        atol=1e-6,
    )
    _assert_runtime_parameters_match(adapter_runtime, legacy_runtime)


def test_distill_binary_policy_output_average_can_skip_zero_alpha() -> None:
    def fail_distill_policy(**kwargs: Any) -> list[float]:
        del kwargs
        raise AssertionError("zero-alpha fast path must not distill")

    def collect_policy_probs(
        agents_arg: list[str],
        observations_arg: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        del agents_arg, observations_arg, temperature
        return binary_policy(np.asarray([0.1, 0.9, 0.4]))

    result = distill_binary_policy_output_average(
        agents=["a", "b", "c"],
        observations=np.zeros((3, 1)),
        peer_ids=[[], [], []],
        alpha=0.0,
        previous_probs=binary_policy(np.asarray([0.1, 0.9, 0.4])),
        temperature=1.0,
        collect_policy_probs=collect_policy_probs,
        distill_policy=fail_distill_policy,
        skip_when_alpha_zero=True,
    )

    assert result.social_losses == [0.0, 0.0, 0.0]
    np.testing.assert_allclose(result.final_action_probs, [0.1, 0.9, 0.4])


def test_binary_peer_helpers_preserve_domain_error_labels() -> None:
    with pytest.raises(ValueError, match="Unsupported Toy X peer rule: invalid"):
        select_binary_output_similarity_peers(
            neighbors=[[1], [0]],
            action_probs=np.asarray([0.2, 0.8]),
            peer_rule="invalid",
            threshold=0.5,
            error_label="Toy X",
        )

    with pytest.raises(ValueError, match="Unsupported Toy X mixer: invalid"):
        peer_ids_for_binary_mixer(
            peer_ids=[[1], [0]],
            mixer="invalid",
            agent_count=2,
            error_label="Toy X",
        )


def test_binary_peer_helpers_can_reuse_static_none_rule_peers() -> None:
    neighbors = [[1], [0]]

    selected = select_binary_output_similarity_peers(
        neighbors=neighbors,
        action_probs=np.asarray([0.2, 0.8]),
        peer_rule="none",
        threshold=0.0,
        error_label="Toy X",
        copy_peers=False,
    )
    mixed = peer_ids_for_binary_mixer(
        peer_ids=selected,
        mixer="output_average",
        agent_count=2,
        error_label="Toy X",
        copy_peers=False,
    )

    assert selected is neighbors
    assert mixed is neighbors


class FakeBinaryToyDomain(BinaryToyDomainBase):
    toy = "fake"
    include_edge_entropy = True

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            run=SimpleNamespace(name="fake_run", seed=11),
            policy=SimpleNamespace(rule="neural_policy", revision_rate=0.5, temperature=1.0),
            coordination=SimpleNamespace(
                mixer="output_average",
                peer_rule="output_similarity",
                alpha=0.25,
                threshold=0.3,
            ),
            state=SimpleNamespace(
                reputation=SimpleNamespace(enabled=True, decay=0.8),
            ),
        )
        self.neighbors = [[1], [0, 2], [1]]

    def domain_aggregate_fields(
        self,
        epoch: int,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> dict[str, object]:
        del state, step_result
        return {"domain_epoch_seen": epoch, "domain_optional": ""}

    def domain_micro_fields(
        self,
        agent_id: int,
        epoch: int,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> dict[str, object]:
        del epoch, state, step_result
        return {"domain_agent_marker": agent_id + 10}


def test_binary_toy_domain_base_assembles_rows_and_summary(tmp_path: Path) -> None:
    domain = FakeBinaryToyDomain()
    state = make_state([1, 0, 1], reputation=[0.2, 0.6, 1.0])
    state.payoffs[:] = [1.0, 2.0, 3.0]
    policy = binary_policy(np.asarray([0.8, 0.25, 0.6]))
    step_result = BinaryPolicyStepResult(
        pre_revision_probs=policy,
        post_local_probs=policy,
        post_social_probs=policy,
        local_losses=[0.1, 0.2, 0.3],
        social_losses=[0.4, 0.5, 0.6],
        peer_ids=[[1], [0, 2], [1]],
        revision_mask=np.asarray([True, False, True]),
        mobility_result=MobilityStepResult.none(3),
        realized_revision_rate=2.0 / 3.0,
        extras={
            "_previous_post_social_probs": binary_policy(
                np.asarray([0.45, 0.75, 0.65])
            ),
            "_previous_actions": np.asarray([0, 0, 1], dtype=np.int64),
        },
    )

    aggregate = domain.aggregate_row(2, state, step_result)
    micro_rows = domain.micro_rows(2, state, step_result)
    result = domain.write_summary(tmp_path, aggregate, state)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert aggregate["toy"] == "fake"
    assert aggregate["action_rate"] == pytest.approx(2.0 / 3.0)
    assert aggregate["mean_policy_action_probability"] == pytest.approx(0.55)
    assert aggregate[
        "policy_action_probability_post_social_gt_0p5_rate"
    ] == pytest.approx(2.0 / 3.0)
    assert aggregate[
        "policy_action_probability_post_social_gt_0p7_rate"
    ] == pytest.approx(1.0 / 3.0)
    assert aggregate[
        "policy_action_probability_post_social_dwell_0p4_0p6_rate"
    ] == pytest.approx(1.0 / 3.0)
    assert aggregate["policy_action_probability_post_social_p50"] == pytest.approx(
        0.6
    )
    assert aggregate[
        "policy_probability_threshold_crossings_0p5_count"
    ] == pytest.approx(2)
    assert aggregate[
        "policy_probability_threshold_crossings_0p5_rate"
    ] == pytest.approx(2.0 / 3.0)
    assert aggregate["action_flip_count"] == pytest.approx(1)
    assert aggregate["action_flip_rate"] == pytest.approx(1.0 / 3.0)
    assert aggregate["domain_optional"] == ""
    assert micro_rows[1]["action"] == 0
    assert micro_rows[1]["action_probability"] == pytest.approx(0.25)
    assert micro_rows[1]["domain_agent_marker"] == 11
    assert isinstance(result, BinaryToyResult)
    assert result.final_action_rate == pytest.approx(2.0 / 3.0)
    assert result.domain_metrics["domain_optional"] is None
    assert summary["domain_metrics"]["domain_optional"] is None


class FakeBinaryDomain:
    micro_state_fields: list[str] = []
    aggregate_fields: list[str] = []

    def __init__(
        self,
        *,
        candidate_probs: np.ndarray,
        mixer: str = "output_average",
        social_mode: str = "probability_mix",
        peer_ids: list[list[int]] | None = None,
        alpha: float = 0.5,
        actions_after_revision: np.ndarray | None = None,
        distilled_probs: np.ndarray | None = None,
        fail_on_sample: bool = False,
        payoff_ema_decay: float | None = None,
        reputation_decay: float | None = None,
        mobility_params: MobilityParams | None = None,
        mobility_rng: np.random.Generator | None = None,
    ) -> None:
        self.candidate_probs = np.asarray(candidate_probs, dtype=np.float64)
        self._mixer = mixer
        self.social_mode = social_mode
        self.selected_peer_ids = (
            peer_ids
            if peer_ids is not None
            else [[] for _ in range(len(self.candidate_probs))]
        )
        self.alpha = alpha
        self.actions_after_revision = actions_after_revision
        self.distilled_probs = (
            np.asarray(distilled_probs, dtype=np.float64)
            if distilled_probs is not None
            else self.candidate_probs
        )
        self.fail_on_sample = fail_on_sample
        self.payoff_ema_decay = payoff_ema_decay
        self.reputation_decay = reputation_decay
        self.mobility_params = mobility_params
        self.mobility_rng = mobility_rng
        self.calls: list[str] = []
        self.sampled_action_probs: np.ndarray | None = None
        self.commit_seen_actions: np.ndarray | None = None

    def make_run_dir(self) -> Path:
        raise NotImplementedError

    def write_metadata(self, run_dir: Path) -> None:
        raise NotImplementedError

    def initial_state(self) -> BinarySpatialState:
        raise NotImplementedError

    def initial_step_result(self, state: BinarySpatialState) -> BinaryPolicyStepResult:
        raise NotImplementedError

    def build_step_context(
        self,
        epoch: int,
        state: BinarySpatialState,
        revision_mask: np.ndarray,
    ) -> BinaryStepContext:
        del state
        self.calls.append("context")
        return BinaryStepContext(
            epoch=epoch,
            revision_mask=revision_mask,
            extras={"context_extra": epoch},
        )

    def local_step(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
    ) -> BinaryLocalStepResult:
        del context
        self.calls.append("local")
        return BinaryLocalStepResult(
            pre_revision_probs=binary_policy(np.zeros(state.agent_count)),
            candidate_action_probs=self.candidate_probs.copy(),
            post_local_probs=binary_policy(self.candidate_probs),
            local_losses=[0.1 for _ in range(state.agent_count)],
            social_mode=self.social_mode,  # type: ignore[arg-type]
            actions_after_revision=None
            if self.actions_after_revision is None
            else self.actions_after_revision.copy(),
            extras={"local_extra": "local", "_private_local": "hidden"},
        )

    def select_peers(
        self,
        action_probs: np.ndarray,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
    ) -> list[list[int]]:
        del state, context, local_result
        self.calls.append("select_peers")
        np.testing.assert_allclose(action_probs, self.candidate_probs)
        return [list(peers) for peers in self.selected_peer_ids]

    def coordination_mixer(self) -> str:
        return self._mixer

    def coordination_alpha(self) -> float:
        return self.alpha

    def policy_tensor_from_action_probs(
        self,
        action_probs: np.ndarray,
        device_like: Any,
    ) -> np.ndarray:
        del device_like
        return binary_policy(action_probs)

    def sample_actions(
        self,
        state: BinarySpatialState,
        action_probs: np.ndarray,
        revision_mask: np.ndarray,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
    ) -> np.ndarray:
        del context, local_result
        self.calls.append("sample")
        if self.fail_on_sample:
            raise AssertionError("policy_distill must not sample current actions")
        self.sampled_action_probs = action_probs.copy()
        actions = state.actions.copy()
        actions[revision_mask] = (action_probs[revision_mask] >= 0.5).astype(np.int64)
        return actions

    def apply_social_distillation(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        peer_ids: list[list[int]],
    ) -> BinarySocialStepResult:
        del context, local_result
        self.calls.append("distill")
        return BinarySocialStepResult(
            peer_ids=peer_ids,
            post_social_probs=binary_policy(self.distilled_probs),
            final_action_probs=self.distilled_probs.copy(),
            social_losses=[0.2 for _ in range(state.agent_count)],
            extras={"social_extra": "distilled", "_private_social": "hidden"},
        )

    def commit_actions(
        self,
        state: BinarySpatialState,
        actions: np.ndarray,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
    ) -> dict[str, Any]:
        del context, local_result, social_result
        self.calls.append("commit")
        self.commit_seen_actions = actions.copy()
        state.actions[:] = actions
        state.payoffs[:] = np.arange(1, state.agent_count + 1, dtype=np.float64) * 10.0
        return {"commit_total_actions": int(np.sum(actions)), "_private_commit": "hidden"}

    def post_step_state_update(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
    ) -> BinaryPostStepStatePolicy:
        del context, local_result, social_result
        self.calls.append("post_policy")
        return BinaryPostStepStatePolicy(
            payoff_ema_decay=self.payoff_ema_decay,
            reputation_decay=self.reputation_decay,
            mobility_params=self.mobility_params,
            mobility_neighbors=[[] for _ in range(state.agent_count)]
            if self.mobility_params is not None
            else None,
            mobility_rng=self.mobility_rng,
        )

    def finalize_hook_step(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
        mobility_result: MobilityStepResult,
    ) -> dict[str, Any]:
        del context, local_result, social_result
        self.calls.append("finalize")
        return {
            "extras": {
                "final_actions": state.actions.copy(),
                "final_payoff_ema": state.payoff_ema.copy(),
                "final_previous_payoff_ema": state.previous_payoff_ema.copy(),
                "final_reputation": state.reputation.copy(),
                "mobility_moved": mobility_result.moved.copy(),
            }
        }

    def aggregate_row(
        self,
        epoch: int,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def micro_rows(
        self,
        epoch: int,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def write_summary(
        self,
        run_dir: Path,
        final_row: dict[str, Any],
        state: BinarySpatialState,
    ) -> Any:
        raise NotImplementedError


def make_state(
    actions: list[int],
    *,
    payoff_ema: list[float] | None = None,
    reputation: list[float] | None = None,
) -> BinarySpatialState:
    agent_count = len(actions)
    return BinarySpatialState(
        actions=np.asarray(actions, dtype=np.int64),
        payoffs=np.zeros(agent_count, dtype=np.float64),
        payoff_ema=np.asarray(
            payoff_ema if payoff_ema is not None else [0.0] * agent_count,
            dtype=np.float64,
        ),
        previous_payoff_ema=np.full(agent_count, -1.0, dtype=np.float64),
        reputation=np.asarray(
            reputation if reputation is not None else [0.0] * agent_count,
            dtype=np.float64,
        ),
    )


def make_runner(domain: FakeBinaryDomain) -> BinarySpatialRunner:
    return BinarySpatialRunner(
        domain=domain,
        epochs=1,
        revision_rate=1.0,
        revision_rng=np.random.default_rng(1),
    )


def run_hook_contract_step(
    domain: FakeBinaryDomain,
    *,
    revision_mask: np.ndarray | None = None,
) -> BinaryPolicyStepResult:
    if revision_mask is None:
        revision_mask = np.ones(3, dtype=bool)
    return make_runner(domain)._hooked_step(
        epoch=1,
        state=make_state([0, 0, 1]),
        revision_mask=revision_mask,
    )


class BadContextRevisionMaskDomain(FakeBinaryDomain):
    def __init__(self, context_revision_mask: np.ndarray) -> None:
        super().__init__(candidate_probs=np.asarray([0.2, 0.8, 0.4]))
        self.context_revision_mask = context_revision_mask

    def build_step_context(
        self,
        epoch: int,
        state: BinarySpatialState,
        revision_mask: np.ndarray,
    ) -> BinaryStepContext:
        base_context = super().build_step_context(epoch, state, revision_mask)
        return BinaryStepContext(
            epoch=base_context.epoch,
            revision_mask=self.context_revision_mask,
            extras=base_context.extras,
        )


class BadLocalLossDomain(FakeBinaryDomain):
    def local_step(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
    ) -> BinaryLocalStepResult:
        result = super().local_step(state, context)
        result.local_losses = [0.1]
        return result


class BadSocialLossDomain(FakeBinaryDomain):
    def apply_social_distillation(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        peer_ids: list[list[int]],
    ) -> BinarySocialStepResult:
        result = super().apply_social_distillation(
            state,
            context,
            local_result,
            peer_ids,
        )
        result.social_losses = [0.2]
        return result


class BadSampleActionsDomain(FakeBinaryDomain):
    def sample_actions(
        self,
        state: BinarySpatialState,
        action_probs: np.ndarray,
        revision_mask: np.ndarray,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
    ) -> np.ndarray:
        del state, action_probs, revision_mask, context, local_result
        return np.asarray([0, 2, 1], dtype=np.int64)


class BadCommitResultDomain(FakeBinaryDomain):
    def commit_actions(
        self,
        state: BinarySpatialState,
        actions: np.ndarray,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
    ) -> Any:
        del state, actions, context, local_result, social_result
        return []


class BadPostStepPolicyDomain(FakeBinaryDomain):
    def post_step_state_update(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
    ) -> Any:
        del state, context, local_result, social_result
        return {}


class BadFinalizeResultDomain(FakeBinaryDomain):
    def finalize_hook_step(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
        mobility_result: MobilityStepResult,
    ) -> Any:
        del state, context, local_result, social_result, mobility_result
        return []


class BadFinalizeExtrasDomain(FakeBinaryDomain):
    def finalize_hook_step(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
        mobility_result: MobilityStepResult,
    ) -> dict[str, Any]:
        del state, context, local_result, social_result, mobility_result
        return {"extras": []}


class BadFinalizePostSocialDomain(FakeBinaryDomain):
    def __init__(self, post_social_probs: np.ndarray) -> None:
        super().__init__(candidate_probs=np.asarray([0.2, 0.8, 0.4]))
        self.post_social_probs = post_social_probs

    def finalize_hook_step(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
        mobility_result: MobilityStepResult,
    ) -> dict[str, Any]:
        del state, context, local_result, social_result, mobility_result
        return {"post_social_probs": self.post_social_probs, "extras": {}}


@pytest.mark.parametrize(
    ("context_revision_mask", "match"),
    [
        (np.asarray([True, False]), "context.revision_mask length"),
        (
            np.asarray([1, 0, 1], dtype=np.int64),
            "context.revision_mask must be a 1D bool array",
        ),
    ],
)
def test_hooked_step_rejects_bad_context_revision_mask(
    context_revision_mask: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_hook_contract_step(BadContextRevisionMaskDomain(context_revision_mask))


@pytest.mark.parametrize(
    ("candidate_probs", "match"),
    [
        (
            np.asarray([[0.2, 0.8, 0.4]]),
            "candidate_action_probs must be a 1D probability vector",
        ),
        (
            np.asarray([0.2, 1.2, 0.4]),
            r"candidate_action_probs values must lie in \[0, 1\]",
        ),
    ],
)
def test_hooked_step_rejects_bad_local_candidate_probs(
    candidate_probs: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_hook_contract_step(FakeBinaryDomain(candidate_probs=candidate_probs))


@pytest.mark.parametrize(
    ("actions_after_revision", "match"),
    [
        (
            np.asarray([0, 1], dtype=np.int64),
            "actions_after_revision length",
        ),
        (
            np.asarray([0, 2, 1], dtype=np.int64),
            r"actions_after_revision values must be binary",
        ),
    ],
)
def test_hooked_step_rejects_bad_actions_after_revision(
    actions_after_revision: np.ndarray,
    match: str,
) -> None:
    domain = FakeBinaryDomain(
        candidate_probs=np.asarray([0.2, 0.8, 0.4]),
        social_mode="policy_distill",
        actions_after_revision=actions_after_revision,
    )

    with pytest.raises(ValueError, match=match):
        run_hook_contract_step(domain)


def test_hooked_step_rejects_bad_sampled_actions() -> None:
    domain = BadSampleActionsDomain(candidate_probs=np.asarray([0.2, 0.8, 0.4]))

    with pytest.raises(ValueError, match=r"actions values must be binary"):
        run_hook_contract_step(domain)


def test_hooked_step_rejects_out_of_range_peer_ids() -> None:
    domain = FakeBinaryDomain(
        candidate_probs=np.asarray([0.2, 0.8, 0.4]),
        peer_ids=[[3], [], []],
    )

    with pytest.raises(ValueError, match="Invalid peer id"):
        run_hook_contract_step(domain)


@pytest.mark.parametrize(
    ("domain", "match"),
    [
        (
            BadLocalLossDomain(candidate_probs=np.asarray([0.2, 0.8, 0.4])),
            "local_losses length",
        ),
        (
            BadSocialLossDomain(
                candidate_probs=np.asarray([0.2, 0.8, 0.4]),
                social_mode="policy_distill",
                peer_ids=[[1], [0], []],
                actions_after_revision=np.asarray([0, 1, 1], dtype=np.int64),
            ),
            "social_losses length",
        ),
    ],
)
def test_hooked_step_rejects_bad_loss_vectors(
    domain: FakeBinaryDomain,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_hook_contract_step(domain)


@pytest.mark.parametrize(
    ("domain", "match"),
    [
        (
            BadCommitResultDomain(candidate_probs=np.asarray([0.2, 0.8, 0.4])),
            "commit_actions result must be a mapping",
        ),
        (
            BadFinalizeResultDomain(candidate_probs=np.asarray([0.2, 0.8, 0.4])),
            "finalize_hook_step result must be a mapping",
        ),
        (
            BadFinalizeExtrasDomain(candidate_probs=np.asarray([0.2, 0.8, 0.4])),
            "finalize_hook_step extras must be a mapping",
        ),
    ],
)
def test_hooked_step_rejects_bad_commit_or_finalize_mappings(
    domain: FakeBinaryDomain,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_hook_contract_step(domain)


def test_hooked_step_rejects_bad_post_step_policy_result() -> None:
    domain = BadPostStepPolicyDomain(candidate_probs=np.asarray([0.2, 0.8, 0.4]))

    with pytest.raises(
        ValueError,
        match="post_step_state_update result must be a BinaryPostStepStatePolicy",
    ):
        run_hook_contract_step(domain)


@pytest.mark.parametrize(
    ("post_social_probs", "match"),
    [
        (
            np.asarray([0.2, 0.8, 0.4]),
            "finalize_hook_step post_social_probs must expose binary action",
        ),
        (
            np.asarray(
                [
                    [0.8, 0.2],
                    [-0.2, 1.2],
                    [0.6, 0.4],
                ],
            ),
            r"finalize_hook_step post_social_probs values must lie in \[0, 1\]",
        ),
    ],
)
def test_hooked_step_rejects_bad_finalize_post_social_probs(
    post_social_probs: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_hook_contract_step(BadFinalizePostSocialDomain(post_social_probs))


def test_hooked_step_probability_mix_output_average_samples_mixed_probs() -> None:
    state = make_state([0, 0, 1])
    domain = FakeBinaryDomain(
        candidate_probs=np.asarray([0.2, 0.8, 0.4]),
        peer_ids=[[1], [0], []],
        alpha=0.5,
    )

    result = make_runner(domain)._hooked_step(
        epoch=1,
        state=state,
        revision_mask=np.asarray([True, True, False]),
    )

    expected_mixed = np.asarray([0.5, 0.5, 0.4])
    np.testing.assert_allclose(domain.sampled_action_probs, expected_mixed)
    np.testing.assert_allclose(result.post_social_probs[:, 1], expected_mixed)
    np.testing.assert_allclose(result.social_losses, [0.3, 0.3, 0.0])
    assert state.actions.tolist() == [1, 1, 1]
    assert result.peer_ids == [[1], [0], []]
    assert result.realized_revision_rate == pytest.approx(2.0 / 3.0)
    assert domain.calls == [
        "context",
        "local",
        "select_peers",
        "sample",
        "commit",
        "post_policy",
        "finalize",
    ]
    assert result.extras["local_extra"] == "local"
    assert result.extras["commit_total_actions"] == 3
    assert "_private_local" not in result.extras
    assert "_private_commit" not in result.extras


def test_hooked_step_none_mixer_bypasses_social_mixing() -> None:
    candidate_probs = np.asarray([0.2, 0.8, 0.4])
    state = make_state([1, 0, 1])
    domain = FakeBinaryDomain(
        candidate_probs=candidate_probs,
        mixer="none",
        peer_ids=[[], [], []],
    )

    result = make_runner(domain)._hooked_step(
        epoch=1,
        state=state,
        revision_mask=np.asarray([True, True, False]),
    )

    np.testing.assert_allclose(domain.sampled_action_probs, candidate_probs)
    np.testing.assert_allclose(result.post_social_probs[:, 1], candidate_probs)
    assert result.social_losses == [0.0, 0.0, 0.0]
    assert result.peer_ids == [[], [], []]
    assert "distill" not in domain.calls


def test_hooked_step_policy_distill_commits_local_actions_without_resampling() -> None:
    local_actions = np.asarray([0, 1, 1], dtype=np.int64)
    distilled_probs = np.asarray([0.9, 0.1, 0.8])
    state = make_state([1, 0, 0])
    domain = FakeBinaryDomain(
        candidate_probs=np.asarray([0.2, 0.8, 0.4]),
        social_mode="policy_distill",
        peer_ids=[[1], [0], []],
        actions_after_revision=local_actions,
        distilled_probs=distilled_probs,
        fail_on_sample=True,
    )

    result = make_runner(domain)._hooked_step(
        epoch=1,
        state=state,
        revision_mask=np.ones(3, dtype=bool),
    )

    assert domain.commit_seen_actions is not None
    assert domain.commit_seen_actions.tolist() == local_actions.tolist()
    assert state.actions.tolist() == local_actions.tolist()
    np.testing.assert_allclose(result.post_social_probs[:, 1], distilled_probs)
    assert result.social_losses == [0.2, 0.2, 0.2]
    assert "sample" not in domain.calls
    assert domain.calls == [
        "context",
        "local",
        "select_peers",
        "distill",
        "commit",
        "post_policy",
        "finalize",
    ]
    assert result.extras["social_extra"] == "distilled"
    assert "_private_social" not in result.extras


def test_hooked_step_common_updates_run_after_commit_before_finalize() -> None:
    state = make_state(
        [0, 0],
        payoff_ema=[2.0, 6.0],
        reputation=[0.2, 0.8],
    )
    domain = FakeBinaryDomain(
        candidate_probs=np.asarray([1.0, 0.0]),
        mixer="none",
        payoff_ema_decay=0.25,
        reputation_decay=0.5,
    )

    result = make_runner(domain)._hooked_step(
        epoch=1,
        state=state,
        revision_mask=np.ones(2, dtype=bool),
    )

    assert domain.calls == [
        "context",
        "local",
        "select_peers",
        "sample",
        "commit",
        "post_policy",
        "finalize",
    ]
    np.testing.assert_allclose(result.extras["final_previous_payoff_ema"], [2.0, 6.0])
    np.testing.assert_allclose(result.extras["final_payoff_ema"], [8.0, 16.5])
    np.testing.assert_allclose(result.extras["final_reputation"], [0.6, 0.4])


def test_hooked_step_mobility_runs_before_finalize() -> None:
    state = make_state([0, 0], payoff_ema=[0.0, 0.0])
    domain = FakeBinaryDomain(
        candidate_probs=np.asarray([0.0, 1.0]),
        mixer="none",
        payoff_ema_decay=0.0,
        mobility_params=MobilityParams(
            enabled=True,
            rate=1.0,
            candidate_pool_size=1,
        ),
        mobility_rng=np.random.default_rng(13),
    )

    result = make_runner(domain)._hooked_step(
        epoch=1,
        state=state,
        revision_mask=np.ones(2, dtype=bool),
    )

    assert result.mobility_result.moved.tolist() == [True, False]
    assert result.mobility_result.targets.tolist() == [1, -1]
    assert result.extras["final_actions"].tolist() == [1, 0]
    assert result.extras["mobility_moved"].tolist() == [True, False]
