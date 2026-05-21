from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from neural_abm.accelerator import (
    BatchedAdamStateCache,
    BatchedMLPParameters,
    BatchedMLPPolicyCache,
    TensorBatchedMLPRuntime,
    apply_batched_mlp_loss_gradients_with_result,
    batched_binary_policy_gradient_losses,
    batched_distribution_cross_entropy_losses,
    batched_mlp_policy_probs,
    is_accelerator_device,
    resolve_neural_update_backend,
    resolve_torch_device,
    trainable_batched_mlp_parameters,
)
from neural_abm.losses import TensorBackedLossVector
from neural_abm.social import mix_probability_distributions
from neural_abm.toy_contagion import AdoptionMLP
from neural_abm.toy_pd import PolicyMLP
from neural_abm.toy_public_goods import PublicGoodsMLP


@pytest.mark.parametrize("model_cls", [PolicyMLP, PublicGoodsMLP, AdoptionMLP])
def test_batched_mlp_policy_probs_match_per_agent_loop(model_cls: type[nn.Module]) -> None:
    models = []
    for agent_id in range(5):
        torch.manual_seed(100 + agent_id)
        models.append(model_cls(input_dim=4, hidden_dim=7, output_dim=2))
    observations = torch.randn(5, 4)
    temperature = 0.7

    loop_probs = torch.stack(
        [
            torch.softmax(model(observations[index].unsqueeze(0)) / temperature, dim=-1)[
                0
            ]
            for index, model in enumerate(models)
        ],
        dim=0,
    )
    batched_probs = batched_mlp_policy_probs(
        models,
        observations,
        temperature=temperature,
    )

    assert torch.allclose(batched_probs, loop_probs, atol=1e-6)


@pytest.mark.parametrize("model_cls", [PolicyMLP, PublicGoodsMLP, AdoptionMLP])
def test_batched_mlp_policy_cache_matches_per_agent_loop(
    model_cls: type[nn.Module],
) -> None:
    agents = []
    for agent_id in range(5):
        torch.manual_seed(300 + agent_id)
        agents.append(
            SimpleNamespace(
                model=model_cls(input_dim=4, hidden_dim=7, output_dim=2),
            )
        )
    observations = torch.randn(5, 4)
    temperature = 0.6

    loop_logits = torch.stack(
        [
            agent.model(observations[index].unsqueeze(0))[0]
            for index, agent in enumerate(agents)
        ],
        dim=0,
    )
    loop_probs = torch.softmax(loop_logits / temperature, dim=-1)
    cache = BatchedMLPPolicyCache.from_agents(agents)

    assert torch.allclose(cache.logits(observations), loop_logits, atol=1e-6)
    assert torch.allclose(
        cache.probabilities(observations, temperature=temperature),
        loop_probs,
        atol=1e-6,
    )


def test_batched_mlp_policy_cache_refresh_reflects_model_mutation() -> None:
    torch.manual_seed(11)
    agents = [
        SimpleNamespace(model=PolicyMLP(input_dim=4, hidden_dim=7, output_dim=2))
    ]
    observations = torch.randn(1, 4)
    cache = BatchedMLPPolicyCache.from_agents(agents)
    before = cache.probabilities(observations)

    with torch.no_grad():
        agents[0].model.fc2.bias.add_(torch.tensor([1.0, -1.0]))

    stale = cache.probabilities(observations)
    cache.refresh(agents)
    refreshed = cache.probabilities(observations)
    loop = torch.softmax(agents[0].model(observations), dim=-1)

    assert torch.allclose(stale, before)
    assert not torch.allclose(refreshed, before)
    assert torch.allclose(refreshed, loop, atol=1e-6)


@pytest.mark.parametrize("model_cls", [PolicyMLP, PublicGoodsMLP, AdoptionMLP])
def test_tensor_batched_runtime_readout_matches_policy_cache(
    model_cls: type[nn.Module],
) -> None:
    agents = []
    for agent_id in range(5):
        torch.manual_seed(350 + agent_id)
        model = model_cls(input_dim=4, hidden_dim=7, output_dim=2)
        agents.append(
            SimpleNamespace(
                model=model,
                optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            )
        )
    observations = torch.randn(5, 4)
    temperature = 0.6
    cache = BatchedMLPPolicyCache.from_agents(agents)
    runtime = TensorBatchedMLPRuntime.from_agents(agents)

    assert torch.allclose(runtime.logits(observations), cache.logits(observations))
    assert torch.allclose(
        runtime.probabilities(observations, temperature=temperature),
        cache.probabilities(observations, temperature=temperature),
        atol=1e-6,
    )


def test_tensor_batched_runtime_trainable_parameters_share_storage() -> None:
    torch.manual_seed(755)
    agents = _adam_agents(agent_count=3)
    runtime = TensorBatchedMLPRuntime.from_agents(agents)
    before = runtime.parameters.detached_clone()

    parameters = runtime.trainable_parameters()

    for trainable_tensor, runtime_tensor in zip(
        parameters.tensors(),
        runtime.parameters.tensors(),
        strict=True,
    ):
        assert trainable_tensor.requires_grad
        assert not runtime_tensor.requires_grad
        assert trainable_tensor.data_ptr() == runtime_tensor.data_ptr()

    observations = torch.randn(3, 4)
    losses = parameters.logits(observations).square().sum(dim=-1)

    for before_tensor, runtime_tensor in zip(
        before.tensors(),
        runtime.parameters.tensors(),
        strict=True,
    ):
        assert torch.allclose(before_tensor, runtime_tensor)

    result = runtime.apply_loss_gradients(parameters, losses)

    assert result.updated_parameters is not None
    for updated_tensor, trainable_tensor in zip(
        result.updated_parameters.tensors(),
        parameters.tensors(),
        strict=True,
    ):
        assert not updated_tensor.requires_grad
        assert updated_tensor.data_ptr() == trainable_tensor.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_batched_mlp_policy_probs_runs_on_cuda() -> None:
    models = []
    for agent_id in range(3):
        torch.manual_seed(200 + agent_id)
        models.append(PolicyMLP(input_dim=4, hidden_dim=7, output_dim=2).to("cuda"))
    observations = torch.randn(3, 4, device="cuda")

    probs = batched_mlp_policy_probs(models, observations)

    assert probs.device.type == "cuda"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_batched_mlp_policy_cache_runs_on_cuda() -> None:
    agents = []
    for agent_id in range(3):
        torch.manual_seed(400 + agent_id)
        agents.append(
            SimpleNamespace(
                model=PolicyMLP(input_dim=4, hidden_dim=7, output_dim=2).to("cuda"),
            )
        )
    observations = torch.randn(3, 4, device="cuda")
    cache = BatchedMLPPolicyCache.from_agents(agents, device="cuda")

    probs = cache.probabilities(observations)

    assert probs.device.type == "cuda"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3, device="cuda"))


def test_batched_mlp_requires_models_and_matching_shapes() -> None:
    with pytest.raises(ValueError, match="At least one model"):
        BatchedMLPParameters.from_models([])

    first = PolicyMLP(input_dim=4, hidden_dim=7, output_dim=2)
    second = PolicyMLP(input_dim=5, hidden_dim=7, output_dim=2)

    with pytest.raises(ValueError, match="must share fc1/fc2 dimensions"):
        BatchedMLPParameters.from_models([first, second])


def test_batched_mlp_requires_relu_two_layer_shape() -> None:
    class BadMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 7)
            self.activation = nn.Tanh()
            self.fc2 = nn.Linear(7, 2)

    with pytest.raises(ValueError, match="require ReLU"):
        BatchedMLPParameters.from_models([BadMLP()])


def test_batched_mlp_checks_observation_shape() -> None:
    model = PolicyMLP(input_dim=4, hidden_dim=7, output_dim=2)
    params = BatchedMLPParameters.from_models([model])

    with pytest.raises(ValueError, match=r"shape \[agents, input\]"):
        params.logits(torch.randn(1, 3, 4))

    with pytest.raises(ValueError, match="Observation count"):
        params.logits(torch.randn(2, 4))


def test_batched_binary_policy_gradient_losses_match_manual() -> None:
    torch.manual_seed(61)
    models = [PolicyMLP(input_dim=4, hidden_dim=7, output_dim=2) for _ in range(3)]
    params = BatchedMLPParameters.from_models(models, requires_grad=True)
    observations = torch.randn(3, 4)
    actions = torch.tensor([0, 1, 0])
    advantages = torch.tensor([0.5, -0.25, 1.25])
    entropy_beta = 0.03

    logits = params.logits(observations)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    expected = (
        -advantages * log_probs.gather(1, actions[:, None]).squeeze(1)
        - entropy_beta * entropy
    )

    actual = batched_binary_policy_gradient_losses(
        params,
        observations,
        actions=actions,
        advantages=advantages,
        entropy_beta=entropy_beta,
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_batched_distribution_cross_entropy_losses_match_manual() -> None:
    torch.manual_seed(62)
    models = [PolicyMLP(input_dim=4, hidden_dim=7, output_dim=2) for _ in range(3)]
    params = BatchedMLPParameters.from_models(models, requires_grad=True)
    observations = torch.randn(3, 4)
    targets = torch.softmax(torch.randn(3, 2), dim=-1)

    logits = params.logits(observations)
    expected = -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    actual = batched_distribution_cross_entropy_losses(
        params,
        observations,
        targets,
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_batched_distribution_kl_losses_match_manual() -> None:
    torch.manual_seed(63)
    models = [PolicyMLP(input_dim=4, hidden_dim=7, output_dim=2) for _ in range(3)]
    params = BatchedMLPParameters.from_models(models, requires_grad=True)
    observations = torch.randn(3, 4)
    targets = torch.softmax(torch.randn(3, 2), dim=-1)

    logits = params.logits(observations)
    expected = torch.nn.functional.kl_div(
        torch.log_softmax(logits, dim=-1),
        targets,
        reduction="none",
    ).sum(dim=-1)
    actual = batched_distribution_cross_entropy_losses(
        params,
        observations,
        targets,
        loss_mode="kl",
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_resolve_torch_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_torch_device("auto") == torch.device("cpu")
    assert resolve_torch_device("gpu") == torch.device("cpu")
    assert resolve_torch_device(None) == torch.device("cpu")


def test_resolve_torch_device_auto_prefers_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_torch_device("auto") == torch.device("cuda")


def test_resolve_neural_update_backend_auto_policy() -> None:
    assert (
        resolve_neural_update_backend("loop", device="cuda", agent_count=1024)
        == "loop"
    )
    assert (
        resolve_neural_update_backend("batched", device="cpu", agent_count=4)
        == "batched"
    )
    assert (
        resolve_neural_update_backend("tensor_batched", device="cpu", agent_count=4)
        == "tensor_batched"
    )
    assert (
        resolve_neural_update_backend("auto", device="cpu", agent_count=255)
        == "loop"
    )
    assert (
        resolve_neural_update_backend("auto", device="cpu", agent_count=256)
        == "batched"
    )
    assert (
        resolve_neural_update_backend("auto", device="cuda", agent_count=4)
        == "batched"
    )

    with pytest.raises(ValueError, match="neural_update_backend"):
        resolve_neural_update_backend("unknown", device="cpu", agent_count=4)


def test_resolve_torch_device_explicit_cuda_requires_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_torch_device("cuda")


def test_accelerator_device_predicate() -> None:
    assert is_accelerator_device(torch.device("cuda"))
    assert not is_accelerator_device(torch.device("cpu"))


@pytest.mark.parametrize("use_state_cache", [False, True])
def test_batched_adam_update_matches_loop_optimizer_state(
    use_state_cache: bool,
) -> None:
    torch.manual_seed(711)
    loop_agents = _adam_agents(agent_count=4)
    batched_agents = _adam_agents(agent_count=4)
    for loop_agent, batched_agent in zip(loop_agents, batched_agents, strict=True):
        batched_agent.model.load_state_dict(loop_agent.model.state_dict())
    observations = torch.randn(4, 4)
    state_cache = (
        BatchedAdamStateCache.from_agents(batched_agents)
        if use_state_cache
        else None
    )

    for _ in range(2):
        loop_losses = []
        for agent_id, agent in enumerate(loop_agents):
            logits = agent.model(observations[agent_id])
            loss = logits.square().sum()
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            loop_losses.append(float(loss.detach()))

        parameters = trainable_batched_mlp_parameters(batched_agents)
        batched_logits = parameters.logits(observations)
        batched_losses = batched_logits.square().sum(dim=-1)
        result = apply_batched_mlp_loss_gradients_with_result(
            agents=batched_agents,
            parameters=parameters,
            losses=batched_losses,
            adam_state_cache=state_cache,
        )

        assert result.used_batched_optimizer is True
        assert isinstance(result.losses, TensorBackedLossVector)
        assert result.losses == pytest.approx(loop_losses, abs=1e-6)
        assert result.updated_parameters is not None

    _assert_agents_and_adam_states_match(batched_agents, loop_agents)
    if state_cache is not None:
        expected_cache = BatchedAdamStateCache.from_agents(loop_agents)
        _assert_adam_state_caches_match(state_cache, expected_cache)


def test_batched_adam_state_cache_falls_back_when_agent_refs_change() -> None:
    torch.manual_seed(731)
    agents = _adam_agents(agent_count=3)
    state_cache = BatchedAdamStateCache.from_agents(agents)
    agents[0].optimizer = torch.optim.SGD(agents[0].model.parameters(), lr=0.01)
    observations = torch.randn(3, 4)
    parameters = trainable_batched_mlp_parameters(agents)
    losses = parameters.logits(observations).square().sum(dim=-1)

    result = apply_batched_mlp_loss_gradients_with_result(
        agents=agents,
        parameters=parameters,
        losses=losses,
        adam_state_cache=state_cache,
    )

    assert result.used_batched_optimizer is False
    assert result.updated_parameters is None
    assert isinstance(result.losses, TensorBackedLossVector)


def test_batched_adam_deferred_optimizer_state_sync_matches_loop() -> None:
    torch.manual_seed(741)
    loop_agents = _adam_agents(agent_count=4)
    batched_agents = _adam_agents(agent_count=4)
    for loop_agent, batched_agent in zip(loop_agents, batched_agents, strict=True):
        batched_agent.model.load_state_dict(loop_agent.model.state_dict())
    observations = torch.randn(4, 4)
    state_cache = BatchedAdamStateCache.from_agents(batched_agents)

    for step_index in range(2):
        loop_losses = []
        for agent_id, agent in enumerate(loop_agents):
            logits = agent.model(observations[agent_id])
            loss = (step_index + 1) * logits.square().sum()
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            loop_losses.append(float(loss.detach()))

        parameters = trainable_batched_mlp_parameters(batched_agents)
        batched_logits = parameters.logits(observations)
        batched_losses = (step_index + 1) * batched_logits.square().sum(dim=-1)
        result = apply_batched_mlp_loss_gradients_with_result(
            agents=batched_agents,
            parameters=parameters,
            losses=batched_losses,
            adam_state_cache=state_cache,
            synchronize_optimizer_states=step_index == 1,
        )

        assert result.used_batched_optimizer is True
        assert result.losses == pytest.approx(loop_losses, abs=2e-6)

    _assert_agents_and_adam_states_match(batched_agents, loop_agents)
    expected_cache = BatchedAdamStateCache.from_agents(loop_agents)
    _assert_adam_state_caches_match(state_cache, expected_cache)


def test_batched_adam_deferred_model_parameter_sync_matches_loop() -> None:
    torch.manual_seed(743)
    loop_agents = _adam_agents(agent_count=4)
    batched_agents = _adam_agents(agent_count=4)
    for loop_agent, batched_agent in zip(loop_agents, batched_agents, strict=True):
        batched_agent.model.load_state_dict(loop_agent.model.state_dict())
    observations = torch.randn(4, 4)
    state_cache = BatchedAdamStateCache.from_agents(batched_agents)
    pending_parameters: BatchedMLPParameters = trainable_batched_mlp_parameters(
        batched_agents,
    )

    for step_index in range(2):
        loop_losses = []
        for agent_id, agent in enumerate(loop_agents):
            logits = agent.model(observations[agent_id])
            loss = (step_index + 1) * logits.square().sum()
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            loop_losses.append(float(loss.detach()))

        batched_logits = pending_parameters.logits(observations)
        batched_losses = (step_index + 1) * batched_logits.square().sum(dim=-1)
        result = apply_batched_mlp_loss_gradients_with_result(
            agents=batched_agents,
            parameters=pending_parameters,
            losses=batched_losses,
            adam_state_cache=state_cache,
            synchronize_model_parameters=step_index == 1,
            synchronize_optimizer_states=step_index == 1,
        )

        assert result.used_batched_optimizer is True
        assert result.updated_parameters is not None
        assert result.losses == pytest.approx(loop_losses, abs=2e-6)
        if step_index == 0:
            stale_parameters = trainable_batched_mlp_parameters(batched_agents)
            assert not torch.allclose(
                stale_parameters.fc1_weight,
                result.updated_parameters.fc1_weight,
            )
        pending_parameters = result.updated_parameters.trainable_clone()

    _assert_agents_and_adam_states_match(batched_agents, loop_agents)
    expected_cache = BatchedAdamStateCache.from_agents(loop_agents)
    _assert_adam_state_caches_match(state_cache, expected_cache)


def test_batched_adam_cache_can_flush_deferred_agent_state() -> None:
    torch.manual_seed(745)
    loop_agents = _adam_agents(agent_count=4)
    batched_agents = _adam_agents(agent_count=4)
    for loop_agent, batched_agent in zip(loop_agents, batched_agents, strict=True):
        batched_agent.model.load_state_dict(loop_agent.model.state_dict())
    observations = torch.randn(4, 4)
    state_cache = BatchedAdamStateCache.from_agents(batched_agents)
    pending_parameters: BatchedMLPParameters = trainable_batched_mlp_parameters(
        batched_agents,
    )

    for step_index in range(2):
        for agent_id, agent in enumerate(loop_agents):
            logits = agent.model(observations[agent_id])
            loss = (step_index + 1) * logits.square().sum()
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        batched_logits = pending_parameters.logits(observations)
        batched_losses = (step_index + 1) * batched_logits.square().sum(dim=-1)
        result = apply_batched_mlp_loss_gradients_with_result(
            agents=batched_agents,
            parameters=pending_parameters,
            losses=batched_losses,
            adam_state_cache=state_cache,
            synchronize_model_parameters=False,
            synchronize_optimizer_states=False,
        )

        assert result.used_batched_optimizer is True
        assert result.updated_parameters is not None
        pending_parameters = result.updated_parameters.trainable_clone()

    state_cache.synchronize_agent_state(pending_parameters)

    _assert_agents_and_adam_states_match(batched_agents, loop_agents)
    expected_cache = BatchedAdamStateCache.from_agents(loop_agents)
    _assert_adam_state_caches_match(state_cache, expected_cache)


def test_tensor_batched_runtime_local_update_flush_matches_batched() -> None:
    torch.manual_seed(751)
    batched_agents = _adam_agents(agent_count=5)
    runtime_agents = _adam_agents(agent_count=5)
    for batched_agent, runtime_agent in zip(
        batched_agents,
        runtime_agents,
        strict=True,
    ):
        runtime_agent.model.load_state_dict(batched_agent.model.state_dict())
    observations = torch.randn(5, 4)
    actions = torch.tensor([0, 1, 1, 0, 1])
    advantages = torch.tensor([0.5, -0.25, 1.0, 0.0, 0.75])
    active_agent_ids = [0, 2, 4]

    batched_cache = BatchedAdamStateCache.from_agents(batched_agents)
    batched_parameters = trainable_batched_mlp_parameters(batched_agents)
    batched_losses = batched_binary_policy_gradient_losses(
        batched_parameters,
        observations,
        actions=actions,
        advantages=advantages,
        entropy_beta=0.01,
    )
    batched_result = apply_batched_mlp_loss_gradients_with_result(
        agents=batched_agents,
        parameters=batched_parameters,
        losses=batched_losses,
        active_agent_ids=active_agent_ids,
        adam_state_cache=batched_cache,
    )

    runtime = TensorBatchedMLPRuntime.from_agents(runtime_agents)
    runtime_parameters = runtime.trainable_parameters()
    runtime_losses = batched_binary_policy_gradient_losses(
        runtime_parameters,
        observations,
        actions=actions,
        advantages=advantages,
        entropy_beta=0.01,
    )
    runtime_result = runtime.apply_loss_gradients(
        runtime_parameters,
        runtime_losses,
        active_agent_ids=active_agent_ids,
    )
    runtime.flush_to_agents(runtime_agents)

    assert runtime_result.losses == pytest.approx(batched_result.losses, abs=1e-6)
    _assert_agents_and_adam_states_match(runtime_agents, batched_agents)


def test_tensor_batched_runtime_full_active_list_matches_batched() -> None:
    torch.manual_seed(752)
    batched_agents = _adam_agents(agent_count=5)
    runtime_agents = _adam_agents(agent_count=5)
    for batched_agent, runtime_agent in zip(
        batched_agents,
        runtime_agents,
        strict=True,
    ):
        runtime_agent.model.load_state_dict(batched_agent.model.state_dict())
    observations = torch.randn(5, 4)
    actions = torch.tensor([0, 1, 1, 0, 1])
    advantages = torch.tensor([0.5, -0.25, 1.0, 0.0, 0.75])
    active_agent_ids = list(range(5))

    batched_cache = BatchedAdamStateCache.from_agents(batched_agents)
    batched_parameters = trainable_batched_mlp_parameters(batched_agents)
    batched_losses = batched_binary_policy_gradient_losses(
        batched_parameters,
        observations,
        actions=actions,
        advantages=advantages,
        entropy_beta=0.01,
    )
    batched_result = apply_batched_mlp_loss_gradients_with_result(
        agents=batched_agents,
        parameters=batched_parameters,
        losses=batched_losses,
        active_agent_ids=None,
        adam_state_cache=batched_cache,
    )

    runtime = TensorBatchedMLPRuntime.from_agents(runtime_agents)
    runtime_parameters = runtime.trainable_parameters()
    runtime_losses = batched_binary_policy_gradient_losses(
        runtime_parameters,
        observations,
        actions=actions,
        advantages=advantages,
        entropy_beta=0.01,
    )
    runtime_result = runtime.apply_loss_gradients(
        runtime_parameters,
        runtime_losses,
        active_agent_ids=active_agent_ids,
    )
    runtime.flush_to_agents(runtime_agents)

    assert runtime_result.losses == pytest.approx(batched_result.losses, abs=1e-6)
    _assert_agents_and_adam_states_match(runtime_agents, batched_agents)


def test_tensor_batched_runtime_social_update_flush_matches_batched() -> None:
    torch.manual_seed(753)
    batched_agents = _adam_agents(agent_count=4)
    runtime_agents = _adam_agents(agent_count=4)
    for batched_agent, runtime_agent in zip(
        batched_agents,
        runtime_agents,
        strict=True,
    ):
        runtime_agent.model.load_state_dict(batched_agent.model.state_dict())
    observations = torch.randn(4, 4)
    previous_probs = torch.softmax(torch.randn(4, 2), dim=-1)
    peer_ids = [[1, 2], [0, 2], [1, 3], [0, 2]]
    mix_result = mix_probability_distributions(
        values=previous_probs,
        peer_ids=peer_ids,
        alpha=0.25,
        copy_peers=False,
        uniform_peer_count=2,
        validate_peers=False,
        collect_update_norms=False,
    )

    batched_cache = BatchedAdamStateCache.from_agents(batched_agents)
    batched_parameters = trainable_batched_mlp_parameters(batched_agents)
    batched_losses = batched_distribution_cross_entropy_losses(
        batched_parameters,
        observations,
        mix_result.mixed_values,
    )
    batched_result = apply_batched_mlp_loss_gradients_with_result(
        agents=batched_agents,
        parameters=batched_parameters,
        losses=batched_losses,
        active_agent_ids=mix_result.active_agent_ids,
        adam_state_cache=batched_cache,
    )

    runtime = TensorBatchedMLPRuntime.from_agents(runtime_agents)
    runtime_parameters = runtime.trainable_parameters()
    runtime_losses = batched_distribution_cross_entropy_losses(
        runtime_parameters,
        observations,
        mix_result.mixed_values,
    )
    runtime_result = runtime.apply_loss_gradients(
        runtime_parameters,
        runtime_losses,
        active_agent_ids=mix_result.active_agent_ids,
    )
    runtime.flush_to_agents(runtime_agents)

    assert runtime_result.losses == pytest.approx(batched_result.losses, abs=1e-6)
    _assert_agents_and_adam_states_match(runtime_agents, batched_agents)


def test_tensor_batched_runtime_divergent_adam_steps_match_batched() -> None:
    torch.manual_seed(754)
    batched_agents = _adam_agents(agent_count=4)
    runtime_agents = _adam_agents(agent_count=4)
    for batched_agent, runtime_agent in zip(
        batched_agents,
        runtime_agents,
        strict=True,
    ):
        runtime_agent.model.load_state_dict(batched_agent.model.state_dict())
    _seed_divergent_adam_state(batched_agents)
    _seed_divergent_adam_state(runtime_agents)
    observations = torch.randn(4, 4)

    batched_cache = BatchedAdamStateCache.from_agents(batched_agents)
    batched_parameters = trainable_batched_mlp_parameters(batched_agents)
    batched_losses = batched_parameters.logits(observations).square().sum(dim=-1)
    apply_batched_mlp_loss_gradients_with_result(
        agents=batched_agents,
        parameters=batched_parameters,
        losses=batched_losses,
        active_agent_ids=None,
        adam_state_cache=batched_cache,
    )

    runtime = TensorBatchedMLPRuntime.from_agents(runtime_agents)
    assert runtime.shared_step_groups is False
    runtime_parameters = runtime.trainable_parameters()
    runtime_losses = runtime_parameters.logits(observations).square().sum(dim=-1)
    runtime.apply_loss_gradients(
        runtime_parameters,
        runtime_losses,
        active_agent_ids=None,
    )
    runtime.flush_to_agents(runtime_agents)

    _assert_agents_and_adam_states_match(runtime_agents, batched_agents)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tensor_batched_runtime_cuda_local_update_smoke() -> None:
    device = torch.device("cuda")
    agents = _adam_agents(agent_count=3)
    for agent in agents:
        agent.model.to(device)
    observations = torch.randn(3, 4, device=device)
    actions = torch.tensor([0, 1, 0], device=device)
    advantages = torch.tensor([0.25, 0.5, -0.25], device=device)
    runtime = TensorBatchedMLPRuntime.from_agents(agents, device=device)
    parameters = runtime.trainable_parameters()
    losses = batched_binary_policy_gradient_losses(
        parameters,
        observations,
        actions=actions,
        advantages=advantages,
        entropy_beta=0.01,
    )

    result = runtime.apply_loss_gradients(parameters, losses)

    assert result.updated_parameters is not None
    assert result.updated_parameters.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_batched_adam_flat_cuda_update_matches_loop_for_active_subset() -> None:
    device = torch.device("cuda")
    torch.manual_seed(747)
    loop_agents = _adam_agents(agent_count=5)
    batched_agents = _adam_agents(agent_count=5)
    for loop_agent, batched_agent in zip(loop_agents, batched_agents, strict=True):
        batched_agent.model.load_state_dict(loop_agent.model.state_dict())
        loop_agent.model.to(device)
        batched_agent.model.to(device)
    observations = torch.randn(5, 4, device=device)
    active_agent_ids = [0, 2, 4]
    BatchedAdamStateCache.from_agents(loop_agents, device=device)
    state_cache = BatchedAdamStateCache.from_agents(batched_agents, device=device)

    loop_losses = [0.0 for _ in loop_agents]
    for agent_id in active_agent_ids:
        agent = loop_agents[agent_id]
        logits = agent.model(observations[agent_id])
        loss = logits.square().sum()
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
        loop_losses[agent_id] = float(loss.detach().cpu())

    parameters = trainable_batched_mlp_parameters(batched_agents, device=device)
    batched_logits = parameters.logits(observations)
    batched_losses = batched_logits.square().sum(dim=-1)
    result = apply_batched_mlp_loss_gradients_with_result(
        agents=batched_agents,
        parameters=parameters,
        losses=batched_losses,
        active_agent_ids=active_agent_ids,
        adam_state_cache=state_cache,
    )

    assert result.used_batched_optimizer is True
    assert result.losses == pytest.approx(loop_losses, abs=1e-6)
    _assert_agents_and_adam_states_match(batched_agents, loop_agents)
    expected_cache = BatchedAdamStateCache.from_agents(loop_agents, device=device)
    _assert_adam_state_caches_match(state_cache, expected_cache)


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


def _seed_divergent_adam_state(agents: list[SimpleNamespace]) -> None:
    for agent in agents:
        for parameter_index, parameter in enumerate(agent.model.parameters()):
            state = agent.optimizer.state[parameter]
            state["step"] = torch.tensor(float(parameter_index + 1), dtype=torch.float32)
            state["exp_avg"] = torch.full_like(parameter, 0.001 * (parameter_index + 1))
            state["exp_avg_sq"] = torch.full_like(
                parameter,
                0.0001 * (parameter_index + 1),
            )


def _assert_agents_and_adam_states_match(
    left_agents: list[SimpleNamespace],
    right_agents: list[SimpleNamespace],
) -> None:
    for left_agent, right_agent in zip(left_agents, right_agents, strict=True):
        for left_param, right_param in zip(
            left_agent.model.parameters(),
            right_agent.model.parameters(),
            strict=True,
        ):
            assert torch.allclose(left_param, right_param, atol=1e-6)
            left_state = left_agent.optimizer.state[left_param]
            right_state = right_agent.optimizer.state[right_param]
            assert left_state.keys() == right_state.keys()
            for key in left_state:
                assert torch.allclose(left_state[key], right_state[key], atol=1e-6)


def _assert_adam_state_caches_match(
    left: BatchedAdamStateCache,
    right: BatchedAdamStateCache,
) -> None:
    for left_group, right_group in zip(
        left.state_tensors(),
        right.state_tensors(),
        strict=True,
    ):
        for left_tensor, right_tensor in zip(left_group, right_group, strict=True):
            assert torch.allclose(left_tensor, right_tensor, atol=1e-6)
