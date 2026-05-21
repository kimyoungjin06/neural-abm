from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neural_abm.accelerator import (
    TensorBatchedMLPRuntime,
    apply_batched_mlp_loss_gradients_with_result,
    batched_binary_policy_gradient_losses,
    batched_distribution_cross_entropy_losses,
    trainable_batched_mlp_parameters,
)
from neural_abm.binary_neural import (
    TensorPolicyRuntime,
    accelerator_timing_kwargs,
    apply_batched_output_average_distillation_update,
    apply_tensor_binary_policy_gradient_update,
    apply_tensor_output_average_distillation_update,
    can_defer_static_output_average_agent_sync,
)
from neural_abm.social import PeerIndexCache, mix_probability_distributions
from neural_abm.toy_pd import PolicyMLP


def test_accelerator_timing_kwargs_uses_callable_context_hooks() -> None:
    calls: list[tuple[str, float]] = []
    recorder = calls.append
    context = SimpleNamespace(
        extras={
            "_record_timing": recorder,
            "_synchronize_timing_device": lambda: None,
        },
    )

    kwargs = accelerator_timing_kwargs(context, "local")

    assert kwargs["timing_prefix"] == "local"
    assert kwargs["timing_recorder"] is recorder
    assert callable(kwargs["timing_synchronizer"])


def test_accelerator_timing_kwargs_ignores_non_callable_hooks() -> None:
    context = SimpleNamespace(
        extras={
            "_record_timing": "not-callable",
            "_synchronize_timing_device": object(),
        },
    )

    kwargs = accelerator_timing_kwargs(context, "social")

    assert kwargs == {
        "timing_prefix": "social",
        "timing_recorder": None,
        "timing_synchronizer": None,
    }


def test_static_output_average_agent_sync_deferral_requires_all_active_agents() -> None:
    all_active_cache = PeerIndexCache.from_peer_ids(
        [[1], [0], [1]],
        device=torch.device("cpu"),
    )
    partial_cache = PeerIndexCache.from_peer_ids(
        [[1], [], [1]],
        device=torch.device("cpu"),
    )

    assert can_defer_static_output_average_agent_sync(
        peer_rule="none",
        mixer="output_average",
        alpha=0.5,
        uniform_neighbor_peer_count=None,
        peer_index_cache=all_active_cache,
        agent_count=3,
    )
    assert not can_defer_static_output_average_agent_sync(
        peer_rule="none",
        mixer="output_average",
        alpha=0.5,
        uniform_neighbor_peer_count=None,
        peer_index_cache=partial_cache,
        agent_count=3,
    )


def test_static_output_average_agent_sync_deferral_rejects_inactive_modes() -> None:
    assert not can_defer_static_output_average_agent_sync(
        peer_rule="output_similarity",
        mixer="output_average",
        alpha=0.5,
        uniform_neighbor_peer_count=1,
        peer_index_cache=None,
        agent_count=3,
    )
    assert not can_defer_static_output_average_agent_sync(
        peer_rule="none",
        mixer="none",
        alpha=0.5,
        uniform_neighbor_peer_count=1,
        peer_index_cache=None,
        agent_count=3,
    )
    assert not can_defer_static_output_average_agent_sync(
        peer_rule="none",
        mixer="output_average",
        alpha=0.0,
        uniform_neighbor_peer_count=1,
        peer_index_cache=None,
        agent_count=3,
    )
    assert can_defer_static_output_average_agent_sync(
        peer_rule="none",
        mixer="output_average",
        alpha=0.5,
        uniform_neighbor_peer_count=1,
        peer_index_cache=None,
        agent_count=3,
    )


def test_tensor_policy_runtime_protocol_accepts_mlp_runtime() -> None:
    runtime = TensorBatchedMLPRuntime.from_agents(_adam_agents(agent_count=2))

    assert isinstance(runtime, TensorPolicyRuntime)


def test_tensor_binary_policy_gradient_update_matches_direct_runtime() -> None:
    torch.manual_seed(810)
    direct_agents = _adam_agents(agent_count=4)
    helper_agents = _adam_agents(agent_count=4)
    active_helper_agents = _adam_agents(agent_count=4)
    _copy_agent_models(source=direct_agents, target=helper_agents)
    _copy_agent_models(source=direct_agents, target=active_helper_agents)
    observations = torch.randn(4, 4)
    actions = torch.tensor([0, 1, 1, 0])
    advantages = torch.tensor([0.5, -0.25, 1.0, 0.75])
    revision_mask = np.asarray([True, False, True, True])

    direct_runtime = TensorBatchedMLPRuntime.from_agents(direct_agents)
    direct_parameters = direct_runtime.trainable_parameters()
    direct_losses = batched_binary_policy_gradient_losses(
        direct_parameters,
        observations,
        actions=actions,
        advantages=advantages,
        entropy_beta=0.03,
    )
    direct_result = direct_runtime.apply_loss_gradients(
        direct_parameters,
        direct_losses,
        active_agent_ids=[0, 2, 3],
    )

    helper_runtime = TensorBatchedMLPRuntime.from_agents(helper_agents)
    helper_result = apply_tensor_binary_policy_gradient_update(
        runtime=helper_runtime,
        observations=observations,
        actions=actions,
        advantages=advantages,
        revision_mask=revision_mask,
        entropy_beta=0.03,
    )

    active_helper_runtime = TensorBatchedMLPRuntime.from_agents(active_helper_agents)
    active_helper_result = apply_tensor_binary_policy_gradient_update(
        runtime=active_helper_runtime,
        observations=observations,
        actions=actions,
        advantages=advantages,
        active_agent_ids=[0, 2, 3],
        entropy_beta=0.03,
    )

    assert list(helper_result.losses) == pytest.approx(list(direct_result.losses))
    assert list(active_helper_result.losses) == pytest.approx(
        list(direct_result.losses),
    )
    _assert_runtimes_match(helper_runtime, direct_runtime)
    _assert_runtimes_match(active_helper_runtime, direct_runtime)


def test_tensor_output_average_distillation_update_matches_direct_runtime() -> None:
    torch.manual_seed(811)
    direct_agents = _adam_agents(agent_count=4)
    helper_agents = _adam_agents(agent_count=4)
    _copy_agent_models(source=direct_agents, target=helper_agents)
    observations = torch.randn(4, 4)
    previous_probs = torch.softmax(torch.randn(4, 2), dim=-1)
    peer_ids = [[1, 2], [0, 2], [1, 3], [0, 2]]
    alpha = 0.25

    direct_runtime = TensorBatchedMLPRuntime.from_agents(direct_agents)
    mix_result = mix_probability_distributions(
        values=previous_probs.detach(),
        peer_ids=peer_ids,
        alpha=alpha,
        channel="policy_distribution",
        commit_mode="distillation_step",
        copy_peers=False,
        collect_update_norms=False,
    )
    direct_parameters = direct_runtime.trainable_parameters()
    direct_losses = batched_distribution_cross_entropy_losses(
        direct_parameters,
        observations,
        mix_result.mixed_values,
    )
    direct_result = direct_runtime.apply_loss_gradients(
        direct_parameters,
        direct_losses,
        active_agent_ids=mix_result.active_agent_ids or [],
    )

    helper_runtime = TensorBatchedMLPRuntime.from_agents(helper_agents)
    helper_result = apply_tensor_output_average_distillation_update(
        runtime=helper_runtime,
        observations=observations,
        peer_ids=peer_ids,
        alpha=alpha,
        previous_probs=previous_probs,
    )

    assert list(helper_result.losses) == pytest.approx(list(direct_result.losses))
    _assert_runtimes_match(helper_runtime, direct_runtime)


def test_batched_output_average_distillation_update_matches_direct_update() -> None:
    torch.manual_seed(814)
    direct_agents = _adam_agents(agent_count=4)
    helper_agents = _adam_agents(agent_count=4)
    _copy_agent_models(source=direct_agents, target=helper_agents)
    observations = torch.randn(4, 4)
    previous_probs = torch.softmax(torch.randn(4, 2), dim=-1)
    peer_ids = [[1, 2], [0, 2], [1, 3], [0, 2]]
    alpha = 0.25

    mix_result = mix_probability_distributions(
        values=previous_probs.detach(),
        peer_ids=peer_ids,
        alpha=alpha,
        channel="policy_distribution",
        commit_mode="distillation_step",
        copy_peers=False,
        collect_update_norms=False,
    )
    direct_parameters = trainable_batched_mlp_parameters(
        direct_agents,
        device=observations.device,
    )
    direct_losses = batched_distribution_cross_entropy_losses(
        direct_parameters,
        observations,
        mix_result.mixed_values,
    )
    direct_result = apply_batched_mlp_loss_gradients_with_result(
        agents=direct_agents,
        parameters=direct_parameters,
        losses=direct_losses,
        active_agent_ids=mix_result.active_agent_ids or [],
    )

    helper_result = apply_batched_output_average_distillation_update(
        agents=helper_agents,
        observations=observations,
        peer_ids=peer_ids,
        alpha=alpha,
        previous_probs=previous_probs,
    )

    assert list(helper_result.losses) == pytest.approx(list(direct_result.losses))
    _assert_agent_models_match(helper_agents, direct_agents)


def test_tensor_output_average_distillation_kl_loss_matches_direct_runtime() -> None:
    torch.manual_seed(813)
    direct_agents = _adam_agents(agent_count=4)
    helper_agents = _adam_agents(agent_count=4)
    _copy_agent_models(source=direct_agents, target=helper_agents)
    observations = torch.randn(4, 4)
    previous_probs = torch.softmax(torch.randn(4, 2), dim=-1)
    peer_ids = [[1], [0, 2], [1, 3], [2]]
    alpha = 0.4

    direct_runtime = TensorBatchedMLPRuntime.from_agents(direct_agents)
    mix_result = mix_probability_distributions(
        values=previous_probs.detach(),
        peer_ids=peer_ids,
        alpha=alpha,
        channel="policy_distribution",
        commit_mode="distillation_step",
        copy_peers=False,
        collect_update_norms=False,
    )
    direct_parameters = direct_runtime.trainable_parameters()
    direct_losses = batched_distribution_cross_entropy_losses(
        direct_parameters,
        observations,
        mix_result.mixed_values,
        loss_mode="kl",
    )
    direct_result = direct_runtime.apply_loss_gradients(
        direct_parameters,
        direct_losses,
        active_agent_ids=mix_result.active_agent_ids or [],
    )

    helper_runtime = TensorBatchedMLPRuntime.from_agents(helper_agents)
    helper_result = apply_tensor_output_average_distillation_update(
        runtime=helper_runtime,
        observations=observations,
        peer_ids=peer_ids,
        alpha=alpha,
        previous_probs=previous_probs,
        loss_mode="kl",
    )

    assert list(helper_result.losses) == pytest.approx(list(direct_result.losses))
    _assert_runtimes_match(helper_runtime, direct_runtime)


def test_tensor_output_average_distillation_uniform_index_matches_generic() -> None:
    torch.manual_seed(814)
    generic_agents = _adam_agents(agent_count=4)
    indexed_agents = _adam_agents(agent_count=4)
    _copy_agent_models(source=generic_agents, target=indexed_agents)
    observations = torch.randn(4, 4)
    previous_probs = torch.softmax(torch.randn(4, 2), dim=-1)
    peer_ids = [[1, 2], [0, 2], [1, 3], [0, 2]]
    peer_index = torch.as_tensor(peer_ids, dtype=torch.long)
    alpha = 0.4

    generic_runtime = TensorBatchedMLPRuntime.from_agents(generic_agents)
    generic_result = apply_tensor_output_average_distillation_update(
        runtime=generic_runtime,
        observations=observations,
        peer_ids=peer_ids,
        alpha=alpha,
        previous_probs=previous_probs,
        loss_mode="kl",
    )

    indexed_runtime = TensorBatchedMLPRuntime.from_agents(indexed_agents)
    indexed_result = apply_tensor_output_average_distillation_update(
        runtime=indexed_runtime,
        observations=observations,
        peer_ids=peer_ids,
        alpha=alpha,
        previous_probs=previous_probs,
        uniform_peer_count=2,
        uniform_peer_index=peer_index,
        validate_peers=False,
        loss_mode="kl",
    )

    assert list(indexed_result.losses) == pytest.approx(list(generic_result.losses))
    _assert_runtimes_match(indexed_runtime, generic_runtime)


def test_tensor_output_average_distillation_alpha_zero_is_noop() -> None:
    torch.manual_seed(812)
    runtime = TensorBatchedMLPRuntime.from_agents(_adam_agents(agent_count=3))
    before_parameters = [tensor.detach().clone() for tensor in runtime.parameters.tensors()]
    before_steps = [tensor.detach().clone() for tensor in runtime.steps]

    result = apply_tensor_output_average_distillation_update(
        runtime=runtime,
        observations=torch.randn(3, 4),
        peer_ids=[[1], [0], [1]],
        alpha=0.0,
        previous_probs=torch.softmax(torch.randn(3, 2), dim=-1),
    )

    assert result.losses == [0.0, 0.0, 0.0]
    for before, after in zip(before_parameters, runtime.parameters.tensors(), strict=True):
        assert torch.equal(before, after)
    for before, after in zip(before_steps, runtime.steps, strict=True):
        assert torch.equal(before, after)


def _adam_agents(agent_count: int) -> list[SimpleNamespace]:
    agents = []
    for agent_id in range(agent_count):
        torch.manual_seed(900 + agent_id)
        model = PolicyMLP(input_dim=4, hidden_dim=7, output_dim=2)
        agents.append(
            SimpleNamespace(
                model=model,
                optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            )
        )
    return agents


def _copy_agent_models(
    *,
    source: list[SimpleNamespace],
    target: list[SimpleNamespace],
) -> None:
    for source_agent, target_agent in zip(source, target, strict=True):
        target_agent.model.load_state_dict(source_agent.model.state_dict())


def _assert_runtimes_match(
    left: TensorBatchedMLPRuntime,
    right: TensorBatchedMLPRuntime,
) -> None:
    for left_tensor, right_tensor in zip(
        left.parameters.tensors(),
        right.parameters.tensors(),
        strict=True,
    ):
        assert torch.allclose(left_tensor, right_tensor, atol=1e-6)
    for left_tensor, right_tensor in zip(
        left.exp_avg.tensors(),
        right.exp_avg.tensors(),
        strict=True,
    ):
        assert torch.allclose(left_tensor, right_tensor, atol=1e-6)
    for left_tensor, right_tensor in zip(
        left.exp_avg_sq.tensors(),
        right.exp_avg_sq.tensors(),
        strict=True,
    ):
        assert torch.allclose(left_tensor, right_tensor, atol=1e-6)
    for left_tensor, right_tensor in zip(left.steps, right.steps, strict=True):
        assert torch.allclose(left_tensor, right_tensor, atol=1e-6)


def _assert_agent_models_match(
    left: list[SimpleNamespace],
    right: list[SimpleNamespace],
) -> None:
    for left_agent, right_agent in zip(left, right, strict=True):
        for left_tensor, right_tensor in zip(
            left_agent.model.state_dict().values(),
            right_agent.model.state_dict().values(),
            strict=True,
        ):
            assert torch.allclose(left_tensor, right_tensor, atol=1e-6)
