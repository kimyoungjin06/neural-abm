from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from binary_config_helpers import binary_toy_config
import neural_abm.toy_contagion as toy_contagion
from neural_abm.config import load_toy5_config
from neural_abm.social import PeerIndexCache
from neural_abm.spatial_binary import (
    BinaryLocalStepResult,
    BinarySocialStepResult,
    BinarySpatialRunner,
)
from neural_abm.toy_contagion import (
    Toy5SpatialDomain,
    apply_output_average_distillation,
    apply_output_average_distillation_batched,
    build_observations,
    complex_threshold_adoption_probabilities,
    create_agents,
    initialize_adoptions,
    initialize_thresholds,
    make_model,
    run_toy5,
    simple_contagion_adoption_probabilities,
    train_neural_local_policies_batched,
    train_neural_local_policy,
    toy5_local_policy_advantages,
)


def result_value(result: Any, field: str) -> Any:
    if hasattr(result, field):
        return getattr(result, field)
    return result.domain_metrics[field]


def final_aggregate_row(run_dir: Path) -> dict[str, str]:
    with (run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    return rows[-1]


PRECOMMITMENT_PEER_AGGREGATE_FIELDS = [
    "precommitment_peer_evidence_enabled",
    "precommitment_peer_evidence_weight",
    "precommitment_peer_readiness_aggregation",
    "precommitment_peer_readiness_mean",
    "precommitment_peer_readiness_active_rate",
    "precommitment_peer_evidence_increment_mean",
]

PRECOMMITMENT_PEER_MICRO_FIELDS = [
    "precommitment_peer_readiness",
    "precommitment_peer_evidence_increment",
]


def tiny_config_dict(
    tmp_path: Path,
    update_rule: str = "complex_threshold",
    mixer: str = "none",
    domain_threshold: float = 0.5,
    threshold_mode: str = "homogeneous",
) -> dict:
    config = {
        "run": {
            "name": f"tiny_toy5_{update_rule}_{mixer}",
            "seed": 7,
            "output_dir": str(tmp_path / "runs"),
        },
        "simulation": {
            "epochs": 3,
            "sync_mode": "synchronous",
            "device": "cpu",
        },
        "environment": {
            "initial_action_fraction": 1.0 / 6.0,
            "seed_selection": "first_agent",
            "threshold_mode": threshold_mode,
            "homogeneous_threshold": domain_threshold,
            "heterogeneous_threshold_low": 0.25,
            "heterogeneous_threshold_high": 0.75,
            "simple_contagion_probability": 1.0,
        },
        "policy": {
            "rule": update_rule,
            "learning_enabled": True,
            "revision_rate": 1.0,
            "temperature": 1.0,
            "decision": {
                "mode": "sampled",
                "action_temperature": 1.0,
                "exploration_epsilon": 0.0,
            },
            "domain": {
                "repeated_exposure_decay": 0.0,
                "adoption_is_absorbing": True,
            },
        },
        "agents": {
            "count": 6,
            "init_mode": "same_init",
            "model": {
                "input_dim": 6,
                "hidden_dim": 8,
                "output_dim": 2,
                "activation": "relu",
            },
            "optimizer": {
                "name": "adam",
                "learning_rate": 0.01,
            },
        },
        "graph": {
            "type": "watts_strogatz",
            "k": 2,
            "rewire_probability": 0.0,
        },
        "coordination": {
            "mixer": mixer,
            "peer_rule": "output_similarity" if mixer == "output_average" else "none",
            "alpha": 0.25 if mixer == "output_average" else 0.0,
            "threshold": 0.0,
        },
        "state": {
            "reputation": {
                "enabled": True,
                "decay": 0.9,
                "peer_rule": "spatial",
                "temperature": 1.0,
                "noise": 0.0,
                "observation_mode": "none",
            },
            "mobility": {
                "enabled": False,
                "rate": 0.0,
                "candidate_pool_size": 8,
                "selection_rule": "local_quality",
                "move_cost": 0.0,
            },
        },
        "logging": {
            "micro_state": True,
            "interval": 1,
            "aggregate_metrics": True,
            "probe_predictions": False,
            "probe_prediction_interval": 1,
        },
    }
    return binary_toy_config(config, "toy5")


def write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "toy5.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def assert_agent_parameters_match(
    left_agents: list[Any],
    right_agents: list[Any],
    *,
    atol: float = 1e-6,
) -> None:
    for left_agent, right_agent in zip(left_agents, right_agents, strict=True):
        for left, right in zip(
            left_agent.model.parameters(),
            right_agent.model.parameters(),
            strict=True,
        ):
            assert torch.allclose(left, right, atol=atol)


def test_toy5_baseline_config_loads() -> None:
    config = load_toy5_config(
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "configs"
        / "toy5_contagion_adoption_baseline.yaml"
    )

    assert config.agents.count == 100
    assert config.policy.rule == "complex_threshold"
    assert config.environment.homogeneous_threshold == pytest.approx(0.25)
    assert config.policy.domain.adoption_is_absorbing is True
    assert config.policy.neural_update_backend == "loop"


def test_toy5_policy_prior_defaults_to_none(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy")
    config_path = write_config(tmp_path, raw)

    assert load_toy5_config(config_path).agents.policy_prior_action_probability is None


@pytest.mark.parametrize("policy_prior", [0.0, 0.5, 1.0])
def test_toy5_policy_prior_accepts_unit_interval(
    tmp_path: Path,
    policy_prior: float,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy")
    raw["agents"]["policy_prior_action_probability"] = policy_prior
    config_path = write_config(tmp_path, raw)

    assert (
        load_toy5_config(config_path).agents.policy_prior_action_probability
        == pytest.approx(policy_prior)
    )


@pytest.mark.parametrize("policy_prior", [-0.01, 1.01])
def test_toy5_policy_prior_rejects_outside_unit_interval(
    tmp_path: Path,
    policy_prior: float,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy")
    raw["agents"]["policy_prior_action_probability"] = policy_prior
    config_path = write_config(tmp_path, raw)

    with pytest.raises(ValueError):
        load_toy5_config(config_path)


def test_toy5_policy_prior_initializes_output_head(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy")
    raw["agents"]["policy_prior_action_probability"] = 0.2
    config = load_toy5_config(write_config(tmp_path, raw))

    model = make_model(config)
    observation = torch.zeros(config.agents.model.input_dim)
    probs = torch.softmax(model(observation), dim=-1)

    assert torch.allclose(model.fc2.weight, torch.zeros_like(model.fc2.weight))
    assert float(probs[1].detach()) == pytest.approx(0.2)


def test_toy5_neural_policy_prior_no_seed_argmax_control(tmp_path: Path) -> None:
    raw = tiny_config_dict(
        tmp_path,
        update_rule="neural_policy",
        mixer="output_average",
        domain_threshold=0.25,
    )
    raw["simulation"]["epochs"] = 2
    raw["environment"]["initial_action_fraction"] = 0.0
    raw["policy"]["learning_enabled"] = False
    raw["policy"]["neural_update_backend"] = "tensor_batched"
    raw["policy"]["decision"]["mode"] = "argmax"
    raw["agents"]["policy_prior_action_probability"] = 0.0
    raw["coordination"]["threshold"] = -1.0
    config_path = write_config(tmp_path, raw)

    result = run_toy5(config=load_toy5_config(config_path), config_path=config_path)

    assert result.final_action_rate == pytest.approx(0.0)
    assert result.domain_metrics["domain_non_adoption_rate"] == pytest.approx(1.0)


def test_toy5_threshold_target_local_advantage_matches_action_target() -> None:
    advantages = toy5_local_policy_advantages(
        actions=np.asarray([0, 1, 0, 1]),
        utility_proxy=np.asarray([-0.25, -0.25, 0.75, 0.75]),
        current_actions=np.asarray([0, 0, 0, 0]),
        local_update_rule="threshold_target",
        adoption_is_absorbing=True,
    )

    assert advantages.tolist() == pytest.approx([0.25, -0.25, -0.75, 0.75])


def test_toy5_threshold_target_local_advantage_preserves_absorbing_adoption() -> None:
    advantages = toy5_local_policy_advantages(
        actions=np.asarray([1, 0]),
        utility_proxy=np.asarray([-0.25, -0.25]),
        current_actions=np.asarray([1, 1]),
        local_update_rule="threshold_target",
        adoption_is_absorbing=True,
    )

    assert advantages.tolist() == pytest.approx([0.25, -0.25])


def test_toy5_local_update_rule_defaults_to_adoption_utility(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy")
    config_path = write_config(tmp_path, raw)

    assert load_toy5_config(config_path).policy.domain.local_update_rule == (
        "adoption_utility"
    )


def test_toy5_rejects_unknown_local_update_rule(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy")
    raw["policy"]["domain"]["local_update_rule"] = "unknown"
    config_path = write_config(tmp_path, raw)

    with pytest.raises(ValueError):
        load_toy5_config(config_path)


def test_toy5_threshold_target_learning_preserves_neural_no_seed_control(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        update_rule="neural_policy",
        mixer="output_average",
        domain_threshold=0.25,
    )
    raw["simulation"]["epochs"] = 3
    raw["environment"]["initial_action_fraction"] = 0.0
    raw["policy"]["learning_enabled"] = True
    raw["policy"]["neural_update_backend"] = "tensor_batched"
    raw["policy"]["decision"]["mode"] = "argmax"
    raw["policy"]["domain"]["local_update_rule"] = "threshold_target"
    raw["agents"]["policy_prior_action_probability"] = 0.0
    raw["coordination"]["threshold"] = -1.0
    config_path = write_config(tmp_path, raw)

    result = run_toy5(config=load_toy5_config(config_path), config_path=config_path)

    assert result.final_action_rate == pytest.approx(0.0)
    assert result.domain_metrics["domain_non_adoption_rate"] == pytest.approx(1.0)


def test_toy5_accepts_neural_update_backend_config(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy")
    raw["policy"]["neural_update_backend"] = "tensor_batched"
    path = write_config(tmp_path, raw)

    config = load_toy5_config(path)

    assert config.policy.neural_update_backend == "tensor_batched"


def test_toy5_non_neural_policy_canonicalizes_backend(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="complex_threshold")
    raw["policy"]["neural_update_backend"] = "tensor_batched"
    path = write_config(tmp_path, raw)

    config = load_toy5_config(path)

    assert config.policy.neural_update_backend == "loop"


def test_toy5_tensor_batched_output_similarity_runner_smoke(tmp_path: Path) -> None:
    raw = tiny_config_dict(
        tmp_path,
        update_rule="neural_policy",
        mixer="output_average",
    )
    raw["simulation"]["epochs"] = 1
    raw["policy"]["neural_update_backend"] = "tensor_batched"
    path = write_config(tmp_path, raw)

    result = run_toy5(config=load_toy5_config(path), config_path=path)

    assert 0.0 <= result.final_action_rate <= 1.0
    assert (result.run_dir / "summary.json").exists()
    final_row = final_aggregate_row(result.run_dir)
    assert final_row["social_channel"] == "policy_distribution"
    assert final_row["commit_mode"] == "distillation_step"
    assert float(final_row["mean_social_update_norm"]) >= 0.0
    assert float(final_row["max_social_update_norm"]) >= 0.0
    assert int(final_row["active_social_agent_count"]) > 0


def test_toy5_tensor_batched_neural_backend_runner_smoke(tmp_path: Path) -> None:
    raw = tiny_config_dict(
        tmp_path,
        update_rule="neural_policy",
        mixer="output_average",
    )
    raw["simulation"]["epochs"] = 1
    raw["policy"]["neural_update_backend"] = "tensor_batched"
    raw["coordination"]["peer_rule"] = "none"
    config_path = write_config(tmp_path, raw)

    result = run_toy5(
        config=load_toy5_config(config_path),
        config_path=config_path,
    )

    assert 0.0 <= result.final_action_rate <= 1.0
    assert (result.run_dir / "summary.json").exists()
    final_row = final_aggregate_row(result.run_dir)
    assert final_row["social_channel"] == "policy_distribution"
    assert final_row["commit_mode"] == "distillation_step"
    assert float(final_row["mean_social_update_norm"]) >= 0.0
    assert float(final_row["max_social_update_norm"]) >= 0.0
    assert int(final_row["active_social_agent_count"]) > 0


@pytest.mark.parametrize("peer_rule", ["none", "output_similarity"])
def test_toy5_tensor_batched_backend_matches_batched_runner(
    tmp_path: Path,
    peer_rule: str,
) -> None:
    base = tiny_config_dict(
        tmp_path,
        update_rule="neural_policy",
        mixer="output_average",
    )
    base["simulation"]["epochs"] = 2
    base["logging"]["micro_state"] = False
    base["coordination"]["peer_rule"] = peer_rule

    batched = copy.deepcopy(base)
    batched["run"]["name"] = f"toy5_batched_reference_{peer_rule}"
    batched["policy"]["neural_update_backend"] = "batched"
    batched_path = tmp_path / "toy5_batched.yaml"
    batched_path.write_text(yaml.safe_dump(batched, sort_keys=False), encoding="utf-8")

    tensor_batched = copy.deepcopy(base)
    tensor_batched["run"]["name"] = f"toy5_tensor_batched_{peer_rule}"
    tensor_batched["policy"]["neural_update_backend"] = "tensor_batched"
    tensor_path = tmp_path / "toy5_tensor_batched.yaml"
    tensor_path.write_text(
        yaml.safe_dump(tensor_batched, sort_keys=False),
        encoding="utf-8",
    )

    batched_result = run_toy5(
        config=load_toy5_config(batched_path),
        config_path=batched_path,
    )
    tensor_result = run_toy5(
        config=load_toy5_config(tensor_path),
        config_path=tensor_path,
    )

    assert tensor_result.final_action_rate == pytest.approx(
        batched_result.final_action_rate,
        abs=1e-6,
    )
    assert tensor_result.final_mean_policy_action_probability == pytest.approx(
        batched_result.final_mean_policy_action_probability,
        abs=1e-6,
    )
    assert tensor_result.final_mean_reputation == pytest.approx(
        batched_result.final_mean_reputation,
        abs=1e-6,
    )


def test_toy5_tensor_batched_no_social_backend_matches_batched_runner(
    tmp_path: Path,
) -> None:
    base = tiny_config_dict(
        tmp_path,
        update_rule="neural_policy",
        mixer="none",
    )
    base["simulation"]["epochs"] = 2
    base["logging"]["micro_state"] = False
    base["coordination"]["peer_rule"] = "none"

    batched = copy.deepcopy(base)
    batched["run"]["name"] = "toy5_batched_reference_no_social"
    batched["policy"]["neural_update_backend"] = "batched"
    batched_path = tmp_path / "toy5_batched_no_social.yaml"
    batched_path.write_text(yaml.safe_dump(batched, sort_keys=False), encoding="utf-8")

    tensor_batched = copy.deepcopy(base)
    tensor_batched["run"]["name"] = "toy5_tensor_batched_no_social"
    tensor_batched["policy"]["neural_update_backend"] = "tensor_batched"
    tensor_path = tmp_path / "toy5_tensor_batched_no_social.yaml"
    tensor_path.write_text(
        yaml.safe_dump(tensor_batched, sort_keys=False),
        encoding="utf-8",
    )

    batched_result = run_toy5(
        config=load_toy5_config(batched_path),
        config_path=batched_path,
    )
    tensor_result = run_toy5(
        config=load_toy5_config(tensor_path),
        config_path=tensor_path,
    )

    assert tensor_result.final_action_rate == pytest.approx(
        batched_result.final_action_rate,
        abs=1e-6,
    )
    assert tensor_result.final_mean_policy_action_probability == pytest.approx(
        batched_result.final_mean_policy_action_probability,
        abs=1e-6,
    )
    assert tensor_result.final_mean_reputation == pytest.approx(
        batched_result.final_mean_reputation,
        abs=1e-6,
    )


def test_toy5_batched_local_training_matches_loop(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        tiny_config_dict(tmp_path, update_rule="neural_policy"),
    )
    config = load_toy5_config(config_path)
    device = torch.device("cpu")
    loop_agents = create_agents(config, device)
    batched_agents = create_agents(config, device)
    torch.manual_seed(601)
    observations = torch.randn(config.agents.count, config.agents.model.input_dim)
    actions = np.asarray([agent_id % 2 for agent_id in range(config.agents.count)])
    utility_proxy = np.linspace(-1.4, 1.3, config.agents.count)
    revision_mask = np.asarray(
        [agent_id % 3 != 2 for agent_id in range(config.agents.count)]
    )

    for _ in range(3):
        loop_losses = [0.0 for _ in range(config.agents.count)]
        for agent_id in np.flatnonzero(revision_mask):
            loop_losses[int(agent_id)] = train_neural_local_policy(
                agent=loop_agents[int(agent_id)],
                observation=observations[int(agent_id)],
                action=int(actions[int(agent_id)]),
                utility_proxy=float(utility_proxy[int(agent_id)]),
            )
        batched_losses = train_neural_local_policies_batched(
            agents=batched_agents,
            observations=observations,
            actions=actions,
            utility_proxy=utility_proxy,
            revision_mask=revision_mask,
        )

        assert np.allclose(batched_losses, loop_losses, atol=1e-6)
    assert_agent_parameters_match(batched_agents, loop_agents)


def test_toy5_batched_distillation_matches_loop(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        tiny_config_dict(
            tmp_path,
            update_rule="neural_policy",
            mixer="output_average",
        ),
    )
    config = load_toy5_config(config_path)
    device = torch.device("cpu")
    loop_agents = create_agents(config, device)
    batched_agents = create_agents(config, device)
    torch.manual_seed(602)
    observations = torch.randn(config.agents.count, config.agents.model.input_dim)
    previous_probs = torch.softmax(torch.randn(config.agents.count, 2), dim=-1)
    peer_ids = [[1], [], [0, 3], [2, 4], [5], [4]]
    peer_index_cache = PeerIndexCache.from_peer_ids(peer_ids, device=device)

    for _ in range(3):
        loop_losses = apply_output_average_distillation(
            agents=loop_agents,
            observations=observations,
            peer_ids=peer_ids,
            alpha=0.25,
            previous_probs=previous_probs,
        )
        batched_losses = apply_output_average_distillation_batched(
            agents=batched_agents,
            observations=observations,
            peer_ids=peer_ids,
            alpha=0.25,
            previous_probs=previous_probs,
            peer_index_cache=peer_index_cache,
            validate_peers=False,
        )

        assert np.allclose(batched_losses, loop_losses, atol=1e-6)
    assert_agent_parameters_match(batched_agents, loop_agents)


def test_toy5_batched_neural_backend_runner_smoke(tmp_path: Path) -> None:
    raw = tiny_config_dict(
        tmp_path,
        update_rule="neural_policy",
        mixer="output_average",
    )
    raw["simulation"]["epochs"] = 1
    raw["policy"]["neural_update_backend"] = "batched"
    config_path = write_config(tmp_path, raw)

    result = run_toy5(
        config=load_toy5_config(config_path),
        config_path=config_path,
    )

    assert 0.0 <= result.final_action_rate <= 1.0
    assert (result.run_dir / "summary.json").exists()
    final_row = final_aggregate_row(result.run_dir)
    assert final_row["social_channel"] == "policy_distribution"
    assert final_row["commit_mode"] == "distillation_step"
    assert float(final_row["mean_social_update_norm"]) >= 0.0
    assert float(final_row["max_social_update_norm"]) >= 0.0
    assert int(final_row["active_social_agent_count"]) > 0


def test_toy5_rejects_invalid_heterogeneous_domain_threshold_order(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, threshold_mode="heterogeneous")
    raw["environment"]["heterogeneous_threshold_low"] = 0.8
    raw["environment"]["heterogeneous_threshold_high"] = 0.2
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="heterogeneous_threshold_low"):
        load_toy5_config(path)


def test_toy5_rejects_invalid_graph_domain_degree(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["graph"]["k"] = 6
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="graph.k"):
        load_toy5_config(path)


def test_toy5_rejects_invalid_neural_model_shape(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy")
    raw["agents"]["model"]["input_dim"] = 5
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="model.input_dim"):
        load_toy5_config(path)


def test_toy5_legacy_shared_dsl_is_rejected(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="complex_threshold")
    raw["dynamics"] = {"update_rule": "complex_threshold"}
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError):
        load_toy5_config(path)


def test_toy5_reputation_config_defaults(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="reputation_imitation")
    path = write_config(tmp_path, raw)

    config = load_toy5_config(path)

    assert config.policy.rule == "reputation_imitation"
    assert config.state.reputation.enabled is True
    assert config.state.reputation.decay == pytest.approx(0.9)


def test_toy5_neural_reputation_observation_requires_8d_model(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy")
    raw["state"]["reputation"] = {
        "enabled": True,
        "observation_mode": "self_neighbor_mean",
    }
    raw["agents"]["model"]["input_dim"] = 8
    path = write_config(tmp_path, raw)

    config = load_toy5_config(path)

    assert config.state.reputation.observation_mode == "self_neighbor_mean"
    assert config.agents.model.input_dim == 8


def test_toy5_observation_appends_reputation_features() -> None:
    observations, rates, domain_utility_proxy = build_observations(
        adopted=np.asarray([1, 0, 1], dtype=np.int64),
        neighbors=[[1, 2], [0], []],
        thresholds=np.asarray([0.25, 0.5, 0.75], dtype=np.float64),
        exposure_counts=np.asarray([1.0, 2.0, 0.0], dtype=np.float64),
        device=torch.device("cpu"),
        reputation=np.asarray([0.3, 0.7, 0.5], dtype=np.float64),
        reputation_observation_mode="self_neighbor_mean",
    )

    assert tuple(observations.shape) == (3, 8)
    np.testing.assert_allclose(
        observations[:, 6:].numpy(),
        np.asarray(
            [
                [0.3, 0.6],
                [0.7, 0.3],
                [0.5, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert rates.tolist() == pytest.approx([0.5, 1.0, 0.0])
    assert domain_utility_proxy.tolist() == pytest.approx([0.25, 0.5, -0.75])


def test_toy5_tensor_observations_match_numpy() -> None:
    adopted = np.asarray([1, 0, 1], dtype=np.int64)
    neighbors = [[1, 2], [0, 2], [0, 1]]
    thresholds = np.asarray([0.25, 0.5, 0.75], dtype=np.float64)
    exposure_counts = np.asarray([1.0, 2.0, 0.0], dtype=np.float64)
    reputation = np.asarray([0.3, 0.7, 0.5], dtype=np.float64)
    peer_index = np.asarray(neighbors, dtype=np.int64)

    numpy_observations, numpy_rates, numpy_utility_proxy = build_observations(
        adopted=adopted,
        neighbors=neighbors,
        thresholds=thresholds,
        exposure_counts=exposure_counts,
        device=torch.device("cpu"),
        reputation=reputation,
        reputation_observation_mode="self_neighbor_mean",
    )
    tensor_observations, tensor_rates, tensor_utility_proxy = build_observations(
        adopted=torch.as_tensor(adopted, dtype=torch.long),
        neighbors=neighbors,
        thresholds=torch.as_tensor(thresholds, dtype=torch.float64),
        exposure_counts=torch.as_tensor(exposure_counts, dtype=torch.float64),
        device=torch.device("cpu"),
        reputation=torch.as_tensor(reputation, dtype=torch.float64),
        reputation_observation_mode="self_neighbor_mean",
        peer_index=torch.as_tensor(peer_index, dtype=torch.long),
    )

    assert tensor_observations.dtype == torch.float32
    assert torch.allclose(tensor_observations, numpy_observations)
    assert isinstance(tensor_rates, torch.Tensor)
    assert tensor_rates.dtype == torch.float64
    np.testing.assert_allclose(tensor_rates.numpy(), numpy_rates)
    np.testing.assert_allclose(tensor_utility_proxy.numpy(), numpy_utility_proxy)


def test_toy5_tensor_batched_initial_state_uses_torch_arrays(tmp_path: Path) -> None:
    raw = tiny_config_dict(
        tmp_path,
        update_rule="neural_policy",
        mixer="output_average",
    )
    raw["policy"]["neural_update_backend"] = "tensor_batched"
    raw["coordination"]["peer_rule"] = "none"
    config_path = write_config(tmp_path, raw)
    config = load_toy5_config(config_path)
    domain = Toy5SpatialDomain(
        config=config,
        config_path=config_path,
        rng=np.random.default_rng(config.run.seed),
        device=torch.device("cpu"),
        neural_update_backend="tensor_batched",
    )

    state = domain.initial_state()

    assert isinstance(state.actions, torch.Tensor)
    assert state.actions.dtype == torch.long
    for values in (
        state.payoffs,
        state.payoff_ema,
        state.previous_payoff_ema,
        state.reputation,
        state.extras["exposure_counts"],
    ):
        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float64


def test_toy5_neural_local_step_routes_through_binary_policy_learning_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        update_rule="neural_policy",
        mixer="none",
    )
    raw["simulation"]["epochs"] = 1
    raw["policy"]["decision"]["mode"] = "argmax"
    config_path = write_config(tmp_path, raw)
    config = load_toy5_config(config_path)
    domain = Toy5SpatialDomain(
        config=config,
        config_path=config_path,
        rng=np.random.default_rng(config.run.seed),
        device=torch.device("cpu"),
        neural_update_backend=config.policy.neural_update_backend,
    )
    state = domain.initial_state()
    context = domain.build_step_context(
        epoch=1,
        state=state,
        revision_mask=np.ones(config.agents.count, dtype=bool),
    )
    original_unit = toy_contagion.BinaryPolicyLearningUnit
    seen: dict[str, Any] = {}

    class SpyBinaryPolicyLearningUnit:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._inner = original_unit(*args, **kwargs)
            callbacks = kwargs["callbacks"]
            seen["agent_count"] = len(kwargs["agents"])
            seen["context"] = kwargs["context"]
            seen["callbacks_type"] = type(callbacks).__name__
            seen["has_collect_policy_probs"] = callable(callbacks.collect_policy_probs)
            seen["has_decision_action_probs"] = callable(
                callbacks.decision_action_probs
            )
            seen["has_sample_actions"] = callable(callbacks.sample_actions)
            seen["has_local_update"] = callable(callbacks.local_update)
            seen["has_refresh_policy_cache"] = callable(callbacks.refresh_policy_cache)

        def run(self) -> Any:
            seen["run_called"] = True
            result = self._inner.run()
            seen["decision_shape"] = tuple(result.decision_action_probs.shape)
            seen["post_shape"] = tuple(result.post_local_probs.shape)
            return result

    monkeypatch.setattr(
        toy_contagion,
        "BinaryPolicyLearningUnit",
        SpyBinaryPolicyLearningUnit,
    )

    result = domain.local_step(state, context)

    assert seen == {
        "agent_count": config.agents.count,
        "context": context,
        "callbacks_type": "BinaryPolicyLearningCallbacks",
        "has_collect_policy_probs": True,
        "has_decision_action_probs": True,
        "has_sample_actions": True,
        "has_local_update": True,
        "has_refresh_policy_cache": True,
        "run_called": True,
        "decision_shape": (config.agents.count, 2),
        "post_shape": (config.agents.count, 2),
    }
    assert isinstance(result, BinaryLocalStepResult)
    assert result.social_mode == "policy_distill"
    assert result.actions_after_revision is not None
    assert result.extras["decision_action_probs"].shape == (config.agents.count, 2)
    assert result.extras["_observations"].shape[0] == config.agents.count


def test_toy5_readiness_augmented_direction_counts_ready_neighbors(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="complex_threshold")
    raw["coordination"].update(
        {
            "precommitment_direction_source": "readiness_augmented_threshold",
            "precommitment_readiness_direction_weight": 1.0,
        }
    )
    config_path = write_config(tmp_path, raw)
    config = load_toy5_config(config_path)
    domain = Toy5SpatialDomain(
        config=config,
        config_path=config_path,
        rng=np.random.default_rng(config.run.seed),
        device=torch.device("cpu"),
        neural_update_backend="loop",
    )
    state = domain.initial_state()
    agent_count = config.agents.count
    policy = torch.full((agent_count, 2), 0.5, dtype=torch.float32)
    local_result = BinaryLocalStepResult(
        pre_revision_probs=policy,
        candidate_action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        post_local_probs=policy,
        local_losses=[0.0 for _ in range(agent_count)],
        social_mode="probability_mix",
    )
    social_result = BinarySocialStepResult(
        peer_ids=domain.neighbors,
        post_social_probs=policy,
        final_action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        social_losses=[0.0 for _ in range(agent_count)],
    )
    state.extras["_binary_action_precommitment_evidence"] = np.asarray(
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    augmented = domain.precommitment_direction_scores(
        state=state,
        local_result=local_result,
        social_result=social_result,
        action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        active=np.zeros(agent_count, dtype=bool),
    )

    assert augmented is not None
    assert augmented[2] == pytest.approx(0.0)
    raw["coordination"]["precommitment_direction_source"] = "local_threshold"
    local_config_path = write_config(tmp_path, raw)
    local_domain = Toy5SpatialDomain(
        config=load_toy5_config(local_config_path),
        config_path=local_config_path,
        rng=np.random.default_rng(config.run.seed),
        device=torch.device("cpu"),
        neural_update_backend="loop",
    )
    local_state = local_domain.initial_state()
    local_scores = local_domain.precommitment_direction_scores(
        state=local_state,
        local_result=local_result,
        social_result=social_result,
        action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        active=np.zeros(agent_count, dtype=bool),
    )

    assert local_scores is not None
    assert local_scores[2] == pytest.approx(-0.5)


def test_toy5_readiness_augmented_direction_action_anchor_marks_seed(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="complex_threshold")
    raw["environment"]["initial_action_fraction"] = 1.0 / 6.0
    raw["coordination"].update(
        {
            "precommitment_direction_source": (
                "readiness_augmented_threshold_with_action_anchor"
            ),
            "precommitment_readiness_direction_weight": 1.0,
        }
    )
    config_path = write_config(tmp_path, raw)
    config = load_toy5_config(config_path)
    domain = Toy5SpatialDomain(
        config=config,
        config_path=config_path,
        rng=np.random.default_rng(config.run.seed),
        device=torch.device("cpu"),
        neural_update_backend="loop",
    )
    state = domain.initial_state()
    agent_count = config.agents.count
    policy = torch.full((agent_count, 2), 0.5, dtype=torch.float32)
    local_result = BinaryLocalStepResult(
        pre_revision_probs=policy,
        candidate_action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        post_local_probs=policy,
        local_losses=[0.0 for _ in range(agent_count)],
        social_mode="probability_mix",
    )
    social_result = BinarySocialStepResult(
        peer_ids=domain.neighbors,
        post_social_probs=policy,
        final_action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        social_losses=[0.0 for _ in range(agent_count)],
    )

    anchored = domain.precommitment_direction_scores(
        state=state,
        local_result=local_result,
        social_result=social_result,
        action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        active=np.zeros(agent_count, dtype=bool),
    )

    assert anchored is not None
    assert anchored[0] == pytest.approx(1.0)
    assert anchored[1] > 0.0
    assert anchored[3] == pytest.approx(-0.5)


def test_toy5_action_anchor_does_not_create_no_seed_direction(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="complex_threshold")
    raw["environment"]["initial_action_fraction"] = 0.0
    raw["coordination"].update(
        {
            "precommitment_direction_source": (
                "readiness_augmented_threshold_with_action_anchor"
            ),
            "precommitment_readiness_direction_weight": 2.0,
        }
    )
    config_path = write_config(tmp_path, raw)
    config = load_toy5_config(config_path)
    domain = Toy5SpatialDomain(
        config=config,
        config_path=config_path,
        rng=np.random.default_rng(config.run.seed),
        device=torch.device("cpu"),
        neural_update_backend="loop",
    )
    state = domain.initial_state()
    agent_count = config.agents.count
    policy = torch.full((agent_count, 2), 0.5, dtype=torch.float32)
    local_result = BinaryLocalStepResult(
        pre_revision_probs=policy,
        candidate_action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        post_local_probs=policy,
        local_losses=[0.0 for _ in range(agent_count)],
        social_mode="probability_mix",
    )
    social_result = BinarySocialStepResult(
        peer_ids=domain.neighbors,
        post_social_probs=policy,
        final_action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        social_losses=[0.0 for _ in range(agent_count)],
    )

    anchored = domain.precommitment_direction_scores(
        state=state,
        local_result=local_result,
        social_result=social_result,
        action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        active=np.zeros(agent_count, dtype=bool),
    )

    assert anchored is not None
    assert np.max(anchored) < 0.0


def test_toy5_threshold_aware_direction_uses_readiness_aggregation(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        update_rule="complex_threshold",
        domain_threshold=0.85,
    )
    raw["environment"]["initial_action_fraction"] = 1.0 / 6.0
    raw["coordination"].update(
        {
            "precommitment_direction_source": (
                "readiness_augmented_threshold_with_action_anchor"
            ),
            "precommitment_readiness_direction_weight": 0.5,
        }
    )
    config_path = write_config(tmp_path, raw)
    config = load_toy5_config(config_path)
    domain = Toy5SpatialDomain(
        config=config,
        config_path=config_path,
        rng=np.random.default_rng(config.run.seed),
        device=torch.device("cpu"),
        neural_update_backend="loop",
    )
    state = domain.initial_state()
    agent_count = config.agents.count
    policy = torch.full((agent_count, 2), 0.5, dtype=torch.float32)
    local_result = BinaryLocalStepResult(
        pre_revision_probs=policy,
        candidate_action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        post_local_probs=policy,
        local_losses=[0.0 for _ in range(agent_count)],
        social_mode="probability_mix",
    )
    social_result = BinarySocialStepResult(
        peer_ids=domain.neighbors,
        post_social_probs=policy,
        final_action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        social_losses=[0.0 for _ in range(agent_count)],
    )

    mean_scores = domain.precommitment_direction_scores(
        state=state,
        local_result=local_result,
        social_result=social_result,
        action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        active=np.zeros(agent_count, dtype=bool),
    )

    assert mean_scores is not None
    assert mean_scores[1] < 0.0

    raw["coordination"]["precommitment_peer_readiness_aggregation"] = "max"
    max_config_path = write_config(tmp_path, raw)
    max_config = load_toy5_config(max_config_path)
    max_domain = Toy5SpatialDomain(
        config=max_config,
        config_path=max_config_path,
        rng=np.random.default_rng(max_config.run.seed),
        device=torch.device("cpu"),
        neural_update_backend="loop",
    )
    max_state = max_domain.initial_state()
    max_scores = max_domain.precommitment_direction_scores(
        state=max_state,
        local_result=local_result,
        social_result=BinarySocialStepResult(
            peer_ids=max_domain.neighbors,
            post_social_probs=policy,
            final_action_probs=np.full(agent_count, 0.5, dtype=np.float64),
            social_losses=[0.0 for _ in range(agent_count)],
        ),
        action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        active=np.zeros(agent_count, dtype=bool),
    )

    assert max_scores is not None
    assert max_scores[1] > 0.0


def test_toy5_exposure_anchor_opens_seeded_frontier_only(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="complex_threshold")
    raw["environment"]["initial_action_fraction"] = 1.0 / 6.0
    raw["coordination"].update(
        {
            "precommitment_direction_source": (
                "readiness_exposure_with_action_anchor"
            ),
            "precommitment_readiness_direction_weight": 2.0,
        }
    )
    config_path = write_config(tmp_path, raw)
    config = load_toy5_config(config_path)
    domain = Toy5SpatialDomain(
        config=config,
        config_path=config_path,
        rng=np.random.default_rng(config.run.seed),
        device=torch.device("cpu"),
        neural_update_backend="loop",
    )
    state = domain.initial_state()
    agent_count = config.agents.count
    policy = torch.full((agent_count, 2), 0.5, dtype=torch.float32)
    local_result = BinaryLocalStepResult(
        pre_revision_probs=policy,
        candidate_action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        post_local_probs=policy,
        local_losses=[0.0 for _ in range(agent_count)],
        social_mode="probability_mix",
    )
    social_result = BinarySocialStepResult(
        peer_ids=domain.neighbors,
        post_social_probs=policy,
        final_action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        social_losses=[0.0 for _ in range(agent_count)],
    )

    seeded = domain.precommitment_direction_scores(
        state=state,
        local_result=local_result,
        social_result=social_result,
        action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        active=np.zeros(agent_count, dtype=bool),
    )

    assert seeded is not None
    assert seeded[0] == pytest.approx(1.0)
    assert seeded[1] > 0.0
    assert seeded[3] == pytest.approx(-1.0)

    raw["environment"]["initial_action_fraction"] = 0.0
    no_seed_path = write_config(tmp_path, raw)
    no_seed_config = load_toy5_config(no_seed_path)
    no_seed_domain = Toy5SpatialDomain(
        config=no_seed_config,
        config_path=no_seed_path,
        rng=np.random.default_rng(no_seed_config.run.seed),
        device=torch.device("cpu"),
        neural_update_backend="loop",
    )
    no_seed_state = no_seed_domain.initial_state()
    no_seed = no_seed_domain.precommitment_direction_scores(
        state=no_seed_state,
        local_result=local_result,
        social_result=BinarySocialStepResult(
            peer_ids=no_seed_domain.neighbors,
            post_social_probs=policy,
            final_action_probs=np.full(agent_count, 0.5, dtype=np.float64),
            social_losses=[0.0 for _ in range(agent_count)],
        ),
        action_probs=np.full(agent_count, 0.5, dtype=np.float64),
        active=np.zeros(agent_count, dtype=bool),
    )

    assert no_seed is not None
    assert np.max(no_seed) < 0.0


def test_toy5_reputation_imitation_requires_enabled_reputation(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="reputation_imitation")
    raw["state"]["reputation"] = {"enabled": False}
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="reputation.enabled"):
        load_toy5_config(path)


def test_toy5_rejects_unknown_reputation_option(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["state"]["reputation"] = {"enabled": True, "unknown": 1}
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError):
        load_toy5_config(path)


def test_toy5_nonzero_initial_fraction_seeds_at_least_one(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["agents"]["count"] = 10
    raw["environment"]["initial_action_fraction"] = 0.05
    config = load_toy5_config(write_config(tmp_path, raw))

    adopted = initialize_adoptions(config, np.random.default_rng(1))

    assert int(np.sum(adopted)) == 1
    assert adopted[0] == 1


def test_simple_contagion_probability_uses_independent_exposures() -> None:
    adopted = np.asarray([1, 1, 0], dtype=np.int64)
    neighbors = [[2], [2], [0, 1]]

    probabilities = simple_contagion_adoption_probabilities(
        adopted=adopted,
        neighbors=neighbors,
        exposure_probability=0.5,
        adoption_is_absorbing=True,
    )

    assert probabilities.tolist() == pytest.approx([1.0, 1.0, 0.75])


def test_complex_threshold_low_domain_threshold_adopts_and_high_domain_threshold_blocks() -> None:
    adopted = np.asarray([1, 0, 0], dtype=np.int64)
    neighbors = [[1], [0, 2], [1]]

    low = complex_threshold_adoption_probabilities(
        adopted=adopted,
        neighbors=neighbors,
        thresholds=np.asarray([0.5, 0.5, 0.5]),
        adoption_is_absorbing=True,
    )
    high = complex_threshold_adoption_probabilities(
        adopted=adopted,
        neighbors=neighbors,
        thresholds=np.asarray([0.75, 0.75, 0.75]),
        adoption_is_absorbing=True,
    )

    assert low.tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert high.tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_heterogeneous_thresholds_create_low_and_high_groups(tmp_path: Path) -> None:
    config = load_toy5_config(
        write_config(
            tmp_path,
            tiny_config_dict(tmp_path, threshold_mode="heterogeneous"),
        )
    )

    thresholds, groups = initialize_thresholds(config, np.random.default_rng(3))

    assert set(groups) == {"low", "high"}
    assert set(np.unique(thresholds)) == {0.25, 0.75}


def test_low_domain_threshold_runner_reaches_cascade(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, domain_threshold=0.5)
    config_path = write_config(tmp_path, raw)

    result = run_toy5(config=load_toy5_config(config_path), config_path=config_path)

    assert result.final_action_rate == pytest.approx(1.0)
    assert result_value(result, "domain_time_to_50_action") is not None
    assert result_value(result, "domain_failed_cascade") is False


def test_high_domain_threshold_runner_blocks_cascade(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, domain_threshold=0.75)
    config_path = write_config(tmp_path, raw)

    result = run_toy5(config=load_toy5_config(config_path), config_path=config_path)

    assert result.final_action_rate == pytest.approx(1.0 / 6.0)
    assert result_value(result, "domain_time_to_50_action") is None
    assert result_value(result, "domain_failed_cascade") is True


def test_toy5_no_social_initial_peer_metrics_are_empty(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="simple_contagion", mixer="none")
    raw["simulation"]["epochs"] = 1
    config_path = write_config(tmp_path, raw)

    result = run_toy5(config=load_toy5_config(config_path), config_path=config_path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_rows = list(csv.DictReader(handle))

    assert float(aggregate_rows[0]["mean_peer_count"]) == pytest.approx(0.0)
    assert int(aggregate_rows[0]["fragmentation_components"]) == raw["agents"]["count"]


def test_toy5_hook_policy_distill_keeps_current_actions_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy", mixer="output_average")
    config_path = write_config(tmp_path, raw)
    config = load_toy5_config(config_path)
    rng = np.random.default_rng(config.run.seed)
    domain = Toy5SpatialDomain(
        config=config,
        config_path=config_path,
        rng=rng,
        device=torch.device(config.simulation.device),
    )
    state = domain.initial_state()
    runner = BinarySpatialRunner(
        domain=domain,
        epochs=1,
        revision_rate=config.policy.revision_rate,
        revision_rng=rng,
    )

    def fail_sample_actions(*args: Any, **kwargs: Any) -> np.ndarray:
        del args, kwargs
        raise AssertionError("policy_distill must not resample after social distillation")

    monkeypatch.setattr(domain, "sample_actions", fail_sample_actions)

    step_result = runner._hooked_step(
        epoch=1,
        state=state,
        revision_mask=np.ones(config.agents.count, dtype=bool),
    )

    assert step_result.extras["final_action_probs"] == pytest.approx(
        step_result.post_social_probs[:, 1].detach().cpu().numpy()
    )


@pytest.mark.parametrize(
    ("update_rule", "mixer"),
    [
        ("simple_contagion", "none"),
        ("complex_threshold", "none"),
        ("complex_threshold", "output_average"),
        ("reputation_imitation", "none"),
        ("reputation_imitation", "output_average"),
        ("neural_policy", "output_average"),
    ],
)
def test_toy5_runner_smoke_writes_expected_outputs(
    tmp_path: Path,
    update_rule: str,
    mixer: str,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule=update_rule, mixer=mixer)
    raw["simulation"]["epochs"] = 1
    config_path = write_config(tmp_path, raw)

    result = run_toy5(config=load_toy5_config(config_path), config_path=config_path)

    assert result.run_dir.exists()
    for filename in [
        "config.yaml",
        "resolved_config.yaml",
        "metadata.json",
        "aggregate_metrics.csv",
        "micro_state.csv",
        "summary.json",
    ]:
        assert (result.run_dir / filename).exists()

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_rows = list(csv.DictReader(handle))
    assert aggregate_rows
    final = aggregate_rows[-1]
    assert 0.0 <= float(final["action_rate"]) <= 1.0
    assert float(final["domain_non_adoption_rate"]) == pytest.approx(
        1.0 - float(final["action_rate"])
    )
    assert int(final["domain_cascade_size"]) >= 1
    assert "domain_time_to_50_action" in final
    assert "domain_failed_cascade" in final
    assert "mean_reputation" in final
    assert "reputation_dispersion" in final
    for field in PRECOMMITMENT_PEER_AGGREGATE_FIELDS:
        assert field in final
    assert final["precommitment_peer_evidence_enabled"] == "False"
    assert float(final["precommitment_peer_evidence_weight"]) == pytest.approx(0.0)
    assert final["precommitment_peer_readiness_aggregation"] == "mean"
    assert float(final["precommitment_peer_readiness_mean"]) == pytest.approx(0.0)
    assert float(final["precommitment_peer_readiness_active_rate"]) == pytest.approx(
        0.0
    )
    assert float(final["precommitment_peer_evidence_increment_mean"]) == pytest.approx(
        0.0
    )

    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_reader = csv.DictReader(handle)
        micro_rows = list(micro_reader)
    assert micro_rows
    for field in [
        "action",
        "action_probability",
        "reputation",
        "domain_threshold",
        "domain_neighbor_action_rate",
        "domain_repeated_exposure_count",
        "domain_degree",
        "peer_count",
        "revised",
        "domain_threshold_group",
        *PRECOMMITMENT_PEER_MICRO_FIELDS,
    ]:
        assert field in micro_reader.fieldnames


def test_toy5_precommitment_peer_evidence_logs_active_contract(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        update_rule="complex_threshold",
        mixer="output_average",
    )
    raw["simulation"]["epochs"] = 2
    raw["coordination"].update(
        {
            "threshold": -1.0,
            "precommitment_enabled": True,
            "precommitment_min_policy_probability": 0.0,
            "precommitment_min_evidence": 1.0,
            "precommitment_evidence_increment": 1.0,
            "precommitment_evidence_decay": 1.0,
            "precommitment_requires_direction": False,
            "precommitment_peer_evidence_enabled": True,
            "precommitment_peer_evidence_weight": 1.0,
        }
    )
    config_path = write_config(tmp_path, raw)

    result = run_toy5(config=load_toy5_config(config_path), config_path=config_path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_rows = list(csv.DictReader(handle))
    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_reader = csv.DictReader(handle)
        micro_rows = list(micro_reader)

    assert aggregate_rows
    assert micro_rows
    final = aggregate_rows[-1]
    assert final["precommitment_enabled"] == "True"
    assert final["precommitment_peer_evidence_enabled"] == "True"
    assert float(final["precommitment_peer_evidence_weight"]) == pytest.approx(1.0)
    assert final["precommitment_peer_readiness_aggregation"] == "mean"
    assert float(final["precommitment_peer_readiness_mean"]) > 0.0
    assert float(final["precommitment_peer_readiness_active_rate"]) > 0.0
    assert float(final["precommitment_peer_evidence_increment_mean"]) > 0.0
    for field in PRECOMMITMENT_PEER_MICRO_FIELDS:
        assert field in micro_reader.fieldnames
    assert any(
        float(row["precommitment_peer_readiness"]) > 0.0 for row in micro_rows
    )
    assert any(
        float(row["precommitment_peer_evidence_increment"]) > 0.0
        for row in micro_rows
    )


@pytest.mark.parametrize(
    ("update_rule", "expected"),
    [
        (
            "simple_contagion",
            {
                "final_action_rate": 1.0,
                "domain_cascade_size": 6,
                "domain_time_to_50_action": 1,
                "domain_failed_cascade": False,
                "domain_action_cluster_count": 1,
                "domain_largest_action_cluster_fraction": 1.0,
                "domain_low_threshold_action_rate": None,
                "domain_high_threshold_action_rate": None,
                "final_mean_reputation": 0.33699999999999997,
                "final_reputation_dispersion": 0.3021224917148672,
            },
        ),
        (
            "complex_threshold",
            {
                "final_action_rate": 1.0,
                "domain_cascade_size": 6,
                "domain_time_to_50_action": 1,
                "domain_failed_cascade": False,
                "domain_action_cluster_count": 1,
                "domain_largest_action_cluster_fraction": 1.0,
                "domain_low_threshold_action_rate": None,
                "domain_high_threshold_action_rate": None,
                "final_mean_reputation": 0.33699999999999997,
                "final_reputation_dispersion": 0.3021224917148672,
            },
        ),
        (
            "neural_policy",
            {
                "final_action_rate": 0.8333333333333334,
                "domain_cascade_size": 5,
                "domain_time_to_50_action": 1,
                "domain_failed_cascade": False,
                "domain_action_cluster_count": 1,
                "domain_largest_action_cluster_fraction": 0.8333333333333334,
                "domain_low_threshold_action_rate": None,
                "domain_high_threshold_action_rate": None,
                "final_mean_reputation": 0.3053333333333333,
                "final_reputation_dispersion": 0.32496444249931233,
            },
        ),
        (
            "reputation_imitation",
            {
                "final_action_rate": 1.0,
                "domain_cascade_size": 6,
                "domain_time_to_50_action": 1,
                "domain_failed_cascade": False,
                "domain_action_cluster_count": 1,
                "domain_largest_action_cluster_fraction": 1.0,
                "domain_low_threshold_action_rate": None,
                "domain_high_threshold_action_rate": None,
                "final_mean_reputation": 0.33699999999999997,
                "final_reputation_dispersion": 0.3021224917148672,
            },
        ),
    ],
)
def test_toy5_tiny_runner_golden_metrics(
    tmp_path: Path,
    update_rule: str,
    expected: dict[str, float | int | bool | None],
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule=update_rule, mixer="none")
    config_path = write_config(tmp_path, raw)

    result = run_toy5(config=load_toy5_config(config_path), config_path=config_path)

    for field, value in expected.items():
        actual = result_value(result, field)
        if value is None or isinstance(value, bool | int):
            assert actual == value
        else:
            assert actual == pytest.approx(value)


def test_toy5_heterogeneous_runner_logs_domain_threshold_group_rates(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, threshold_mode="heterogeneous")
    raw["simulation"]["epochs"] = 1
    config_path = write_config(tmp_path, raw)

    result = run_toy5(config=load_toy5_config(config_path), config_path=config_path)

    assert result_value(result, "domain_low_threshold_action_rate") is not None
    assert result_value(result, "domain_high_threshold_action_rate") is not None


def test_toy5_reputation_imitation_logs_reputation_metrics(tmp_path: Path) -> None:
    raw = tiny_config_dict(
        tmp_path,
        update_rule="reputation_imitation",
        mixer="output_average",
    )
    raw["simulation"]["epochs"] = 2
    config_path = write_config(tmp_path, raw)

    result = run_toy5(config=load_toy5_config(config_path), config_path=config_path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_rows = list(csv.DictReader(handle))
    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_rows = list(csv.DictReader(handle))
    summary = yaml.safe_load((result.run_dir / "summary.json").read_text())

    assert aggregate_rows
    assert micro_rows
    assert all(0.0 <= float(row["mean_reputation"]) <= 1.0 for row in aggregate_rows)
    assert all(0.0 <= float(row["reputation"]) <= 1.0 for row in micro_rows)
    assert 0.0 <= float(summary["final_mean_reputation"]) <= 1.0
