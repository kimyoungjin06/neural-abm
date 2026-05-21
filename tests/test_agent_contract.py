from __future__ import annotations

import torch

from neural_abm.core import (
    ClassificationMLP,
    NeuralClassificationAgent,
    flatten_parameters,
)
from neural_abm.toy_contagion import AdoptionMLP, NeuralAdoptionAgent
from neural_abm.toy_opinion import NeuralOpinionAgent, OpinionMLP
from neural_abm.toy_pd import NeuralPDAgent, PolicyMLP
from neural_abm.toy_public_goods import NeuralPublicGoodsAgent, PublicGoodsMLP
from neural_abm.unit import NABMAgent


def make_agent() -> NeuralClassificationAgent:
    torch.manual_seed(11)
    model = ClassificationMLP(input_dim=2, hidden_dim=8, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_x = torch.randn(64, 2)
    train_y = (train_x[:, 1] > 0).long()
    return NeuralClassificationAgent(
        agent_id=3,
        shard_group="contract_test",
        model=model,
        optimizer=optimizer,
        train_x=train_x,
        train_y=train_y,
    )


def test_agent_observe_and_predict_contract() -> None:
    agent = make_agent()
    x = torch.randn(5, 2)

    observation = agent.observe(x)
    probs = agent.act_or_predict(observation)

    assert observation.shape == (5, 2)
    assert probs.shape == (5, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.all(probs >= 0)


def test_agent_local_update_changes_parameters() -> None:
    agent = make_agent()
    before = flatten_parameters(agent.model).clone()

    loss = agent.local_update(batch_size=16, steps=3)
    after = flatten_parameters(agent.model)

    assert loss > 0
    assert not torch.allclose(before, after)


def test_agent_social_message_contract() -> None:
    agent = make_agent()
    probe_x = torch.randn(7, 2)

    message = agent.social_message(probe_x)

    assert message["agent_id"] == 3
    assert message["shard_group"] == "contract_test"
    assert message["probe_probs"].shape == (7, 2)
    assert message["latent_summary"].shape == (8,)
    assert 0.0 <= message["confidence"] <= 1.0
    assert message["param_norm"] > 0


def test_agent_log_state_is_flat_summary() -> None:
    agent = make_agent()
    probe_x = torch.randn(7, 2)

    state = agent.log_state(probe_x)

    assert state["agent_id"] == 3
    assert state["shard_group"] == "contract_test"
    assert 0.0 <= state["confidence"] <= 1.0
    assert state["param_norm"] > 0
    assert state["latent_norm"] >= 0


def make_policy_agent(agent_cls, model_cls, input_dim: int, output_dim: int):
    model = model_cls(input_dim=input_dim, hidden_dim=8, output_dim=output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return agent_cls(agent_id=1, model=model, optimizer=optimizer)


def assert_nabm_agent_contract(agent: NABMAgent, observation: torch.Tensor) -> None:
    assert isinstance(agent, NABMAgent)

    observed = agent.observe(observation)
    agent.observation_spec().validate(observed)
    prediction = agent.act_or_predict(observed)
    message = agent.social_message(observed)
    agent.social_message_spec().validate(message)
    state = agent.log_state(observed)

    assert torch.is_tensor(prediction)
    assert message["agent_id"] == agent.agent_id
    assert state["agent_id"] == agent.agent_id
    assert 0.0 <= state["confidence"] <= 1.0
    assert state["param_norm"] >= 0.0
    assert state["latent_norm"] >= 0.0


def test_toy2_neural_agent_contract() -> None:
    agent = make_policy_agent(NeuralPDAgent, PolicyMLP, input_dim=4, output_dim=2)

    assert_nabm_agent_contract(agent, torch.randn(5, 4))


def test_toy3_neural_agent_contract() -> None:
    agent = make_policy_agent(NeuralOpinionAgent, OpinionMLP, input_dim=6, output_dim=1)

    assert_nabm_agent_contract(agent, torch.randn(5, 6))


def test_toy4_neural_agent_contract() -> None:
    agent = make_policy_agent(
        NeuralPublicGoodsAgent,
        PublicGoodsMLP,
        input_dim=5,
        output_dim=2,
    )

    assert_nabm_agent_contract(agent, torch.randn(5, 5))


def test_toy5_neural_agent_contract() -> None:
    agent = make_policy_agent(
        NeuralAdoptionAgent,
        AdoptionMLP,
        input_dim=7,
        output_dim=2,
    )

    assert_nabm_agent_contract(agent, torch.randn(5, 7))
