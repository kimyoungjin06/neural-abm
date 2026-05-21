from __future__ import annotations

import csv
import copy
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from pydantic import ValidationError

from binary_config_helpers import binary_toy_config
import neural_abm.toy_pd as toy_pd_module
import neural_abm.spatial_binary as spatial_binary_module
from neural_abm.accelerator import TensorBatchedMLPRuntime
from neural_abm.basin_phase_critic import (
    DEFAULT_BASIN_PHASE_CRITIC_FEATURES,
    LearnedBasinPhaseCritic,
    save_critic_bundle_npz,
)
from neural_abm.config import load_toy2_config
from neural_abm.mobility import MobilityStepResult
from neural_abm.spatial_binary import (
    BinaryLocalStepResult,
    BinarySocialStepResult,
    BinarySpatialRunner,
    BinarySpatialState,
    BinaryStepContext,
    apply_mobility_swaps,
    mean_binary_policy_prob_triplet,
    update_payoff_ema,
    update_reputation_ema,
)
from neural_abm.toy_pd import (
    DecisionKernel,
    Toy2SpatialDomain,
    apply_action_temperature,
    apply_output_average,
    apply_output_average_distillation_batched,
    apply_payoff_threshold_calibration,
    apply_output_average_to_cooperation_probs,
    build_observations,
    counterfactual_action_payoffs,
    counterfactual_policy_advantage_components,
    counterfactual_policy_targets_and_advantages,
    counterfactual_policy_targets_and_advantages_tensor,
    compute_payoffs_from_peer_index,
    compute_payoffs_from_peer_ids,
    cooperation_probs_to_policy_tensor,
    create_agents,
    create_tensor_batched_runtime,
    interaction_peer_ids,
    neural_context_peer_ids,
    policy_consensus,
    reputation_imitation_cooperation_probs,
    run_toy2,
    sample_actions,
    transformed_advantage,
    train_counterfactual_policy,
    train_neural_local_policies_batched,
    train_neural_local_policy,
    mobility_params_from_config,
    uniform_peer_index,
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


def assert_agent_parameters_match(
    left: list[Any],
    right: list[Any],
    *,
    atol: float = 1e-6,
) -> None:
    for left_agent, right_agent in zip(left, right, strict=True):
        for left_tensor, right_tensor in zip(
            left_agent.model.state_dict().values(),
            right_agent.model.state_dict().values(),
            strict=True,
        ):
            assert torch.allclose(left_tensor, right_tensor, atol=atol)


def tiny_config_dict(
    tmp_path: Path,
    mixer: str,
    peer_rule: str,
    update_rule: str = "neural_policy",
    alpha: float = 0.25,
    learning_enabled: bool | None = None,
    revision_rate: float | None = None,
    local_update_rule: str | None = None,
    neural_peer_mode: str | None = None,
    interaction_mode: str | None = None,
    decision_mode: str | None = None,
    action_temperature: float | None = None,
    exploration_epsilon: float | None = None,
    calibration_mode: str | None = None,
    calibration_strength: float | None = None,
    epochs: int = 2,
    grid_width: int = 4,
    grid_height: int = 4,
    initial_action_probability: float = 0.5,
    policy_prior_action_probability: float | None = None,
    seed: int = 13,
) -> dict:
    config = {
        "run": {
            "name": f"tiny_toy2_{update_rule}_{mixer}",
            "seed": seed,
            "output_dir": str(tmp_path / "runs"),
        },
        "simulation": {
            "epochs": epochs,
            "sync_mode": "synchronous",
            "device": "cpu",
        },
        "game": {
            "family": "prisoner_dilemma",
            "payoff": {
                "T": 5.0,
                "R": 3.0,
                "P": 1.0,
                "S": 0.0,
            },
        },
        "policy": {
            "rule": update_rule,
            "learning_enabled": True,
            "revision_rate": 1.0,
            "selection_strength": 1.0,
            "temperature": 1.0,
            "decision": {
                "mode": "sampled",
                "action_temperature": 1.0,
                "exploration_epsilon": 0.0,
            },
            "domain": {
                "local_update_rule": "sampled_policy_gradient",
                "neural_peer_mode": "spatial",
                "interaction_mode": "spatial",
                "payoff_transform": "linear",
            },
        },
        "environment": {
            "grid_width": grid_width,
            "grid_height": grid_height,
            "neighborhood": "von_neumann",
            "periodic": True,
            "initial_action_probability": initial_action_probability,
            "reward_ema_decay": 0.9,
            "entropy_beta": 0.01,
            "payoff_R": 3.0,
            "payoff_S": 0.0,
            "payoff_T": 5.0,
            "payoff_P": 1.0,
        },
        "agents": {
            "init_mode": "independent_init",
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
        "coordination": {
            "mixer": mixer,
            "peer_rule": peer_rule,
            "alpha": alpha,
            "threshold": 0.0,
            "communication_budget": {
                "probe_predictions": 1,
                "latent_dim": 8,
                "scalar_summary": 8,
            },
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
    if policy_prior_action_probability is not None:
        config["agents"]["policy_prior_action_probability"] = (
            policy_prior_action_probability
        )
    if learning_enabled is not None:
        config["policy"]["learning_enabled"] = learning_enabled
    if revision_rate is not None:
        config["policy"]["revision_rate"] = revision_rate
    if local_update_rule is not None:
        config["policy"]["domain"]["local_update_rule"] = local_update_rule
    if neural_peer_mode is not None:
        config["policy"]["domain"]["neural_peer_mode"] = neural_peer_mode
    if interaction_mode is not None:
        config["policy"]["domain"]["interaction_mode"] = interaction_mode
    if decision_mode is not None:
        config["policy"]["decision"]["mode"] = decision_mode
    if action_temperature is not None:
        config["policy"]["decision"]["action_temperature"] = action_temperature
    if exploration_epsilon is not None:
        config["policy"]["decision"]["exploration_epsilon"] = exploration_epsilon
    if calibration_mode is not None or calibration_strength is not None:
        calibration = config["policy"]["decision"].setdefault("calibration", {})
        if calibration_mode is not None:
            calibration["mode"] = calibration_mode
        if calibration_strength is not None:
            calibration["strength"] = calibration_strength
    return binary_toy_config(config, "toy2")


def write_learned_basin_test_model(tmp_path: Path) -> Path:
    feature_count = len(DEFAULT_BASIN_PHASE_CRITIC_FEATURES)
    weights = np.zeros(feature_count, dtype=np.float64)
    weights[0] = 4.0
    critic = LearnedBasinPhaseCritic(
        feature_columns=DEFAULT_BASIN_PHASE_CRITIC_FEATURES,
        feature_mean=np.zeros(feature_count, dtype=np.float64),
        feature_scale=np.ones(feature_count, dtype=np.float64),
        weights=weights,
        bias=-2.0,
    )
    path = tmp_path / "learned_basin_test_model.npz"
    save_critic_bundle_npz(path, critic, (critic, critic))
    return path


def write_tiny_config(
    tmp_path: Path,
    mixer: str,
    peer_rule: str,
    update_rule: str = "neural_policy",
    alpha: float = 0.25,
    learning_enabled: bool | None = None,
    revision_rate: float | None = None,
    local_update_rule: str | None = None,
    neural_peer_mode: str | None = None,
    interaction_mode: str | None = None,
    decision_mode: str | None = None,
    action_temperature: float | None = None,
    exploration_epsilon: float | None = None,
    calibration_mode: str | None = None,
    calibration_strength: float | None = None,
    epochs: int = 2,
    grid_width: int = 4,
    grid_height: int = 4,
    initial_action_probability: float = 0.5,
    policy_prior_action_probability: float | None = None,
    seed: int = 13,
) -> Path:
    config = tiny_config_dict(
        tmp_path=tmp_path,
        mixer=mixer,
        peer_rule=peer_rule,
        update_rule=update_rule,
        alpha=alpha,
        learning_enabled=learning_enabled,
        revision_rate=revision_rate,
        local_update_rule=local_update_rule,
        neural_peer_mode=neural_peer_mode,
        interaction_mode=interaction_mode,
        decision_mode=decision_mode,
        action_temperature=action_temperature,
        exploration_epsilon=exploration_epsilon,
        calibration_mode=calibration_mode,
        calibration_strength=calibration_strength,
        epochs=epochs,
        grid_width=grid_width,
        grid_height=grid_height,
        initial_action_probability=initial_action_probability,
        policy_prior_action_probability=policy_prior_action_probability,
        seed=seed,
    )
    path = tmp_path / f"{update_rule}_{mixer}_{peer_rule}_{seed}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("update_rule", "mixer", "peer_rule"),
    [
        ("neural_policy", "none", "none"),
        ("neural_policy", "output_average", "none"),
        ("neural_policy", "output_average", "output_similarity"),
        ("fermi_imitation", "none", "none"),
        ("fermi_imitation", "output_average", "none"),
        ("reputation_imitation", "none", "none"),
        ("reputation_imitation", "output_average", "none"),
        ("rd_well_mixed", "none", "none"),
    ],
)
def test_toy2_runner_smoke(
    tmp_path: Path,
    update_rule: str,
    mixer: str,
    peer_rule: str,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer=mixer,
        peer_rule=peer_rule,
        update_rule=update_rule,
    )
    config = load_toy2_config(config_path)
    result = run_toy2(config=config, config_path=config_path)

    assert result.run_dir.exists()
    assert (result.run_dir / "micro_state.csv").exists()
    assert (result.run_dir / "aggregate_metrics.csv").exists()
    assert (result.run_dir / "summary.json").exists()
    assert 0.0 <= result.final_action_rate <= 1.0
    assert 0.0 <= result.final_mean_payoff <= config.environment.payoff_T
    assert result.final_fragmentation_components >= 0


def test_toy2_precommitment_trajectory_fields_are_logged(tmp_path: Path) -> None:
    raw = tiny_config_dict(
        tmp_path=tmp_path,
        mixer="none",
        peer_rule="none",
        learning_enabled=False,
        decision_mode="argmax",
        epochs=1,
        initial_action_probability=0.0,
        policy_prior_action_probability=0.0,
    )
    raw["coordination"].update(
        {
            "precommitment_enabled": True,
            "precommitment_min_policy_probability": 0.0,
            "precommitment_min_evidence": 1.0,
            "precommitment_evidence_increment": 1.0,
            "precommitment_evidence_decay": 1.0,
            "precommitment_requires_direction": False,
        }
    )
    config_path = tmp_path / "toy2_precommitment_trajectory.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(config_path), config_path=config_path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_reader = csv.DictReader(handle)
        aggregate_rows = list(aggregate_reader)
    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_reader = csv.DictReader(handle)
        micro_rows = list(micro_reader)

    final = aggregate_rows[-1]
    for field in [
        "precommitment_first_ready_epoch",
        "precommitment_all_ready_epoch",
        "precommitment_first_forced_epoch",
        "precommitment_ready_to_forced_delay_mean",
        "precommitment_premature_exit_count",
        "precommitment_high_policy_rate",
        "precommitment_direction_ok_rate",
        "precommitment_ready_largest_component_fraction",
        "precommitment_peer_evidence_enabled",
        "precommitment_peer_evidence_weight",
        "precommitment_peer_readiness_mean",
        "precommitment_peer_readiness_active_rate",
        "precommitment_peer_evidence_increment_mean",
    ]:
        assert field in aggregate_reader.fieldnames
    for field in [
        "precommitment_evidence",
        "precommitment_ready",
        "precommitment_signal",
        "precommitment_high_policy",
        "precommitment_direction_ok",
        "precommitment_forced_action",
        "precommitment_peer_readiness",
        "precommitment_peer_evidence_increment",
        "precommitment_first_ready_epoch",
        "precommitment_first_forced_epoch",
    ]:
        assert field in micro_reader.fieldnames
    assert final["precommitment_first_ready_epoch"] == "1"
    assert final["precommitment_all_ready_epoch"] == "1"
    assert final["precommitment_first_forced_epoch"] == "1"
    assert float(final["precommitment_high_policy_rate"]) == pytest.approx(1.0)
    assert float(final["precommitment_direction_ok_rate"]) == pytest.approx(1.0)
    assert final["precommitment_peer_evidence_enabled"] == "False"
    assert float(final["precommitment_peer_evidence_weight"]) == pytest.approx(0.0)
    assert micro_rows[-1]["precommitment_ready"] == "True"
    assert micro_rows[-1]["precommitment_first_ready_epoch"] == "1"


@pytest.mark.parametrize(
    "local_update_rule",
    ["sampled_policy_gradient", "counterfactual_advantage"],
)
def test_toy2_neural_update_backends_match_loop_runner(
    tmp_path: Path,
    local_update_rule: str,
) -> None:
    results = {}
    for backend in ["loop", "batched", "tensor_batched"]:
        raw = tiny_config_dict(
            tmp_path,
            mixer="output_average",
            peer_rule="none",
            local_update_rule=local_update_rule,
            epochs=3,
            seed=37,
        )
        raw["run"]["name"] = f"toy2_{local_update_rule}_{backend}"
        raw["logging"]["micro_state"] = False
        raw["policy"]["neural_update_backend"] = backend
        config_path = tmp_path / f"toy2_{local_update_rule}_{backend}.yaml"
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        results[backend] = run_toy2(
            config=load_toy2_config(config_path),
            config_path=config_path,
        )

    loop_result = results["loop"]
    for backend in ["batched", "tensor_batched"]:
        result = results[backend]
        assert result.final_action_rate == pytest.approx(
            loop_result.final_action_rate,
            abs=1e-9,
        )
        assert result.final_mean_policy_action_probability == pytest.approx(
            loop_result.final_mean_policy_action_probability,
            abs=1e-5,
        )
        assert result.final_mean_reputation == pytest.approx(
            loop_result.final_mean_reputation,
            abs=1e-9,
        )
    for result in results.values():
        final_row = final_aggregate_row(result.run_dir)
        assert final_row["social_channel"] == "policy_distribution"
        assert final_row["commit_mode"] == "distillation_step"
        assert float(final_row["mean_social_update_norm"]) >= 0.0
        assert float(final_row["max_social_update_norm"]) >= 0.0
        assert int(final_row["active_social_agent_count"]) > 0


def test_toy2_tensor_batched_output_similarity_runner_smoke(tmp_path: Path) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="output_average",
        peer_rule="output_similarity",
        epochs=1,
    )
    raw["policy"]["neural_update_backend"] = "tensor_batched"
    raw["logging"]["micro_state"] = False
    config_path = tmp_path / "toy2_tensor_output_similarity.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy2(
        config=load_toy2_config(config_path),
        config_path=config_path,
    )

    assert 0.0 <= result.final_action_rate <= 1.0
    assert (result.run_dir / "summary.json").exists()


def test_toy2_tensor_batched_reputation_observation_mobility_smoke(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        epochs=2,
    )
    raw["policy"]["neural_update_backend"] = "tensor_batched"
    raw["agents"]["model"]["input_dim"] = 8
    raw["state"]["reputation"]["observation_mode"] = "self_neighbor_mean"
    raw["state"]["mobility"] = {
        "enabled": True,
        "rate": 1.0,
        "candidate_pool_size": 4,
        "selection_rule": "local_quality",
        "move_cost": 0.0,
    }
    raw["logging"]["micro_state"] = False
    config_path = tmp_path / "toy2_tensor_reputation_mobility.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy2(
        config=load_toy2_config(config_path),
        config_path=config_path,
    )

    assert 0.0 <= result.final_action_rate <= 1.0
    assert 0.0 <= result.final_mean_reputation <= 1.0
    assert (result.run_dir / "summary.json").exists()


def test_toy2_tensor_batched_initial_state_uses_torch_state(tmp_path: Path) -> None:
    config_path = write_tiny_config(tmp_path, mixer="none", peer_rule="none")
    config = load_toy2_config(config_path)
    domain = Toy2SpatialDomain(
        config=config,
        config_path=config_path,
        rng=np.random.default_rng(config.run.seed),
        neural_peer_rng=np.random.default_rng(config.run.seed + 1_000_003),
        interaction_rng=np.random.default_rng(config.run.seed + 2_000_003),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
        device=torch.device("cpu"),
        neural_update_backend="tensor_batched",
    )

    state = domain.initial_state()

    assert isinstance(state.actions, torch.Tensor)
    assert state.actions.dtype == torch.long
    assert state.payoffs.dtype == torch.float64
    assert state.payoff_ema.dtype == torch.float64
    assert state.previous_payoff_ema.dtype == torch.float64
    assert state.reputation.dtype == torch.float64
    assert state.agents == []


def test_toy2_static_policy_distill_peer_selection_reuses_neighbors(
    tmp_path: Path,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="output_average",
        peer_rule="none",
    )
    config = load_toy2_config(config_path)
    domain = Toy2SpatialDomain(
        config=config,
        config_path=config_path,
        rng=np.random.default_rng(config.run.seed),
        neural_peer_rng=np.random.default_rng(config.run.seed + 1_000_003),
        interaction_rng=np.random.default_rng(config.run.seed + 2_000_003),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
        device=torch.device("cpu"),
        neural_update_backend="tensor_batched",
    )

    peer_ids = domain.select_peers(
        action_probs=np.zeros(domain.agent_count, dtype=np.float64),
        state=domain.initial_state(),
        context=BinaryStepContext(
            epoch=1,
            revision_mask=np.ones(domain.agent_count, dtype=bool),
        ),
        local_result=BinaryLocalStepResult(
            pre_revision_probs=torch.zeros(domain.agent_count, 2),
            candidate_action_probs=np.zeros(domain.agent_count, dtype=np.float64),
            post_local_probs=object(),
            local_losses=[0.0 for _ in range(domain.agent_count)],
            social_mode="policy_distill",
        ),
    )

    assert peer_ids is domain.neighbors
    assert domain.selected_peer_ids_are_validated()


def test_toy2_static_peer_aggregate_metrics_are_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="output_average",
        peer_rule="none",
    )
    config = load_toy2_config(config_path)
    domain = Toy2SpatialDomain(
        config=config,
        config_path=config_path,
        rng=np.random.default_rng(config.run.seed),
        neural_peer_rng=np.random.default_rng(config.run.seed + 1_000_003),
        interaction_rng=np.random.default_rng(config.run.seed + 2_000_003),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
        device=torch.device("cpu"),
        neural_update_backend="tensor_batched",
    )
    state = domain.initial_state()
    step_result = domain.initial_step_result(state)

    def fail_peer_metrics(*_args: Any, **_kwargs: Any) -> dict[str, float | int]:
        raise AssertionError("static peer metrics should be reused")

    monkeypatch.setattr(
        spatial_binary_module,
        "binary_peer_metrics",
        fail_peer_metrics,
    )

    row = domain.aggregate_row(epoch=0, state=state, step_result=step_result)

    assert row["fragmentation_components"] == 1
    assert row["mean_peer_count"] == pytest.approx(4.0)
    assert row["edge_entropy"] == pytest.approx(domain._static_peer_metrics["edge_entropy"])


def test_mean_binary_policy_prob_triplet_matches_individual_means() -> None:
    post_social = torch.tensor(
        [[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]],
        dtype=torch.float32,
    )
    pre_revision = torch.tensor(
        [[0.3, 0.7], [0.2, 0.8], [0.5, 0.5]],
        dtype=torch.float32,
    )
    post_local = torch.tensor(
        [[0.6, 0.4], [0.9, 0.1], [0.2, 0.8]],
        dtype=torch.float32,
    )

    means = mean_binary_policy_prob_triplet(
        post_social,
        pre_revision,
        post_local,
    )

    assert means == pytest.approx(
        (
            float(post_social[:, 1].mean()),
            float(pre_revision[:, 1].mean()),
            float(post_local[:, 1].mean()),
        ),
    )


@pytest.mark.parametrize("init_mode", ["independent_init", "same_init"])
@pytest.mark.parametrize("policy_prior", [None, 0.8])
def test_toy2_tensor_batched_runtime_initialization_matches_agents(
    tmp_path: Path,
    init_mode: str,
    policy_prior: float | None,
) -> None:
    config_dict = tiny_config_dict(
        tmp_path=tmp_path,
        mixer="none",
        peer_rule="none",
        policy_prior_action_probability=policy_prior,
    )
    config_dict["agents"]["init_mode"] = init_mode
    prior_name = "none" if policy_prior is None else str(policy_prior).replace(".", "_")
    config_path = tmp_path / f"toy2_tensor_runtime_{init_mode}_{prior_name}.yaml"
    config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    config = load_toy2_config(config_path)
    device = torch.device("cpu")

    torch.manual_seed(999)
    agent_runtime = TensorBatchedMLPRuntime.from_agents(
        create_agents(config, device),
        device=device,
    )
    agent_rng_state = torch.random.get_rng_state()

    torch.manual_seed(999)
    direct_runtime = create_tensor_batched_runtime(config, device)
    direct_rng_state = torch.random.get_rng_state()

    for direct, from_agents in zip(
        direct_runtime.parameters.tensors(),
        agent_runtime.parameters.tensors(),
        strict=True,
    ):
        assert torch.allclose(direct, from_agents)
    for direct, from_agents in zip(
        direct_runtime.exp_avg.tensors(),
        agent_runtime.exp_avg.tensors(),
        strict=True,
    ):
        assert torch.equal(direct, from_agents)
    for direct, from_agents in zip(
        direct_runtime.exp_avg_sq.tensors(),
        agent_runtime.exp_avg_sq.tensors(),
        strict=True,
    ):
        assert torch.equal(direct, from_agents)
    for direct, from_agents in zip(
        direct_runtime.steps,
        agent_runtime.steps,
        strict=True,
    ):
        assert torch.equal(direct, from_agents)
    assert direct_runtime.lr == agent_runtime.lr
    assert direct_runtime.betas == agent_runtime.betas
    assert direct_runtime.eps == agent_runtime.eps
    assert direct_runtime.weight_decay == agent_runtime.weight_decay
    assert direct_runtime.shared_step_groups
    assert torch.equal(direct_rng_state, agent_rng_state)


def test_toy2_tensor_batched_finalize_defers_agent_payoff_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_tiny_config(tmp_path, mixer="none", peer_rule="none")
    config = load_toy2_config(config_path)
    domain = Toy2SpatialDomain(
        config=config,
        config_path=config_path,
        rng=np.random.default_rng(config.run.seed),
        neural_peer_rng=np.random.default_rng(config.run.seed + 1_000_003),
        interaction_rng=np.random.default_rng(config.run.seed + 2_000_003),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
        device=torch.device("cpu"),
        neural_update_backend="tensor_batched",
    )
    state = domain.initial_state()

    def fail_sync(_state: BinarySpatialState) -> None:
        raise AssertionError("tensor_batched should defer agent payoff EMA sync")

    monkeypatch.setattr(domain, "_sync_agent_payoff_ema", fail_sync)

    updates = domain.finalize_hook_step(
        state=state,
        context=BinaryStepContext(
            epoch=1,
            revision_mask=np.ones(domain.agent_count, dtype=bool),
        ),
        local_result=BinaryLocalStepResult(
            pre_revision_probs=torch.zeros(domain.agent_count, 2),
            candidate_action_probs=np.zeros(domain.agent_count, dtype=np.float64),
            post_local_probs=torch.zeros(domain.agent_count, 2),
            local_losses=[0.0 for _ in range(domain.agent_count)],
            social_mode="policy_distill",
        ),
        social_result=BinarySocialStepResult(
            peer_ids=[[] for _ in range(domain.agent_count)],
            post_social_probs=torch.zeros(domain.agent_count, 2),
            final_action_probs=np.zeros(domain.agent_count, dtype=np.float64),
            social_losses=[0.0 for _ in range(domain.agent_count)],
        ),
        mobility_result=MobilityStepResult.none(domain.agent_count),
    )

    assert updates == {"extras": {}}


def test_toy2_hook_probability_mix_samples_social_probabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="output_average",
        peer_rule="none",
        update_rule="fermi_imitation",
    )
    config = load_toy2_config(config_path)
    rng = np.random.default_rng(config.run.seed)
    domain = Toy2SpatialDomain(
        config=config,
        config_path=config_path,
        rng=rng,
        neural_peer_rng=np.random.default_rng(config.run.seed + 1_000_003),
        interaction_rng=np.random.default_rng(config.run.seed + 2_000_003),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
        device=torch.device(config.simulation.device),
    )
    state = domain.initial_state()
    runner = BinarySpatialRunner(
        domain=domain,
        epochs=1,
        revision_rate=config.policy.revision_rate,
        revision_rng=rng,
    )
    seen: dict[str, np.ndarray] = {}
    original_sample_actions = domain.sample_actions

    def spy_sample_actions(*args: Any, **kwargs: Any) -> np.ndarray:
        seen["action_probs"] = kwargs["action_probs"].copy()
        return original_sample_actions(*args, **kwargs)

    monkeypatch.setattr(domain, "sample_actions", spy_sample_actions)

    step_result = runner._hooked_step(
        epoch=1,
        state=state,
        revision_mask=np.ones(domain.agent_count, dtype=bool),
    )

    assert seen["action_probs"] == pytest.approx(
        step_result.post_social_probs[:, 1].detach().cpu().numpy()
    )


def test_toy2_hook_policy_distill_keeps_current_actions_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="output_average",
        peer_rule="none",
        update_rule="neural_policy",
    )
    config = load_toy2_config(config_path)
    rng = np.random.default_rng(config.run.seed)
    domain = Toy2SpatialDomain(
        config=config,
        config_path=config_path,
        rng=rng,
        neural_peer_rng=np.random.default_rng(config.run.seed + 1_000_003),
        interaction_rng=np.random.default_rng(config.run.seed + 2_000_003),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
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
        revision_mask=np.ones(domain.agent_count, dtype=bool),
    )

    assert step_result.extras["decision_action_probs"].shape == (
        domain.agent_count,
        2,
    )


def test_toy2_baseline_config_loads() -> None:
    config = load_toy2_config(
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "configs"
        / "toy2_spatial_pd_baseline.yaml"
    )

    assert config.environment.grid_width * config.environment.grid_height == 100
    assert config.policy.rule == "neural_policy"
    assert config.policy.domain.local_update_rule == "counterfactual_advantage"
    assert config.coordination.mixer == "none"


@pytest.mark.parametrize(
    ("update_rule", "expected"),
    [
        (
            "neural_policy",
            {
                "final_action_rate": 0.3125,
                "final_mean_payoff": 1.8125,
                "final_mean_policy_action_probability": 0.5008547306060791,
                "domain_action_components": 1,
                "domain_largest_action_cluster_fraction": 0.3125,
                "final_fragmentation_components": 16,
                "final_mean_reputation": 0.42500000000000004,
                "final_reputation_dispersion": 0.40489196089821294,
            },
        ),
        (
            "fermi_imitation",
            {
                "final_action_rate": 0.25,
                "final_mean_payoff": 1.6875,
                "final_mean_policy_action_probability": 0.2785697877407074,
                "domain_action_components": 2,
                "domain_largest_action_cluster_fraction": 0.1875,
                "final_fragmentation_components": 16,
                "final_mean_reputation": 0.413125,
                "final_reputation_dispersion": 0.4020217461468969,
            },
        ),
        (
            "reputation_imitation",
            {
                "final_action_rate": 1.0,
                "final_mean_payoff": 3.0,
                "final_mean_policy_action_probability": 1.0,
                "domain_action_components": 1,
                "domain_largest_action_cluster_fraction": 1.0,
                "final_fragmentation_components": 16,
                "final_mean_reputation": 0.5387500000000001,
                "final_reputation_dispersion": 0.39599360790295596,
            },
        ),
        (
            "rd_well_mixed",
            {
                "final_action_rate": 0.355353125,
                "final_mean_payoff": 2.094375,
                "final_mean_policy_action_probability": 0.355353125,
                "domain_action_components": 0,
                "domain_largest_action_cluster_fraction": 0.0,
                "final_fragmentation_components": 0,
                "final_mean_reputation": 0.0,
                "final_reputation_dispersion": 0.0,
            },
        ),
    ],
)
def test_toy2_tiny_runner_golden_metrics(
    tmp_path: Path,
    update_rule: str,
    expected: dict[str, float | int],
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule=update_rule,
        epochs=2,
    )

    result = run_toy2(config=load_toy2_config(config_path), config_path=config_path)

    for field, value in expected.items():
        actual = result_value(result, field)
        if isinstance(value, int):
            assert actual == value
        else:
            assert actual == pytest.approx(value)


def test_toy2_rd_well_mixed_artifacts_use_shared_dsl_fields(
    tmp_path: Path,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule="rd_well_mixed",
        epochs=2,
    )

    result = run_toy2(config=load_toy2_config(config_path), config_path=config_path)

    metadata = yaml.safe_load((result.run_dir / "metadata.json").read_text())
    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_reader = csv.DictReader(handle)
        aggregate_rows = list(aggregate_reader)
    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_reader = csv.DictReader(handle)
        micro_rows = list(micro_reader)

    assert aggregate_rows
    assert micro_rows == []
    for field in [
        "policy_rule",
        "coordination_mixer",
        "coordination_peer_rule",
    ]:
        assert field in metadata
        assert field in aggregate_reader.fieldnames
        assert field in micro_reader.fieldnames
    assert "policy_revision_rate" in aggregate_reader.fieldnames
    assert "realized_revision_rate" in aggregate_reader.fieldnames
    for legacy_field in ["update_rule", "mixer", "peer_rule", "revision_rate"]:
        assert legacy_field not in metadata
        assert legacy_field not in aggregate_reader.fieldnames
        assert legacy_field not in micro_reader.fieldnames


def test_toy2_legacy_shared_dsl_is_rejected(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["dynamics"] = {"update_rule": "neural_policy"}
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_toy2_learning_enabled_defaults_and_explicit_false(tmp_path: Path) -> None:
    default_config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    default_path = tmp_path / "default_learning.yaml"
    default_path.write_text(yaml.safe_dump(default_config), encoding="utf-8")

    disabled_config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        learning_enabled=False,
    )
    disabled_path = tmp_path / "disabled_learning.yaml"
    disabled_path.write_text(yaml.safe_dump(disabled_config), encoding="utf-8")

    assert load_toy2_config(default_path).policy.learning_enabled is True
    assert load_toy2_config(disabled_path).policy.learning_enabled is False
    assert load_toy2_config(default_path).policy.revision_rate == 1.0


def test_toy2_local_update_rule_defaults_to_sampled_policy_gradient(
    tmp_path: Path,
) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    path = tmp_path / "default_local_update.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert (
        load_toy2_config(path).policy.domain.local_update_rule == "sampled_policy_gradient"
    )


@pytest.mark.parametrize(
    "local_update_rule",
    ["sampled_policy_gradient", "counterfactual_advantage"],
)
def test_toy2_local_update_rule_accepts_supported_values(
    tmp_path: Path,
    local_update_rule: str,
) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule=local_update_rule,
    )
    path = tmp_path / f"{local_update_rule}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert load_toy2_config(path).policy.domain.local_update_rule == local_update_rule


def test_toy2_accepts_neural_update_backend_config(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["policy"]["neural_update_backend"] = "tensor_batched"
    path = tmp_path / "tensor_backend.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert load_toy2_config(path).policy.neural_update_backend == "tensor_batched"


def test_toy2_state_continuation_objective_requires_loop_backend(
    tmp_path: Path,
) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    config["policy"]["neural_update_backend"] = "batched"
    config["policy"]["domain"]["objective"] = {
        "mode": "state_continuation",
        "welfare_weight": 1.0,
    }
    path = tmp_path / "state_continuation_batched.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError, match="state_continuation"):
        load_toy2_config(path)


def test_toy2_profile_objective_requires_loop_backend(tmp_path: Path) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    config["policy"]["neural_update_backend"] = "batched"
    config["policy"]["domain"]["objective"] = {"profile": "linear_balanced"}
    path = tmp_path / "profile_batched.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError, match="state_continuation"):
        load_toy2_config(path)


def test_toy2_domain_bootstrap_defaults_disabled(tmp_path: Path) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    path = tmp_path / "bootstrap_default.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_toy2_config(path)

    assert loaded.policy.domain.bootstrap.enabled is False
    assert loaded.policy.domain.bootstrap.decision_enabled is False
    assert loaded.policy.domain.bootstrap.replay_enabled is False
    assert loaded.policy.domain.bootstrap.distill_stable_teacher_only is False
    assert loaded.policy.domain.bootstrap.distill_gradient_gate_enabled is False


def test_toy2_domain_bootstrap_rejects_unknown_teacher(tmp_path: Path) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    config["policy"]["domain"]["objective"] = {"profile": "linear_welfare_heavy"}
    config["policy"]["domain"]["bootstrap"] = {
        "enabled": True,
        "teacher": "imitation",
    }
    path = tmp_path / "bootstrap_bad_teacher.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)

    config["policy"]["domain"]["bootstrap"] = {
        "replay_enabled": True,
        "replay_teacher": "imitation",
    }
    path = tmp_path / "bootstrap_bad_replay_teacher.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)

    config["policy"]["domain"]["bootstrap"] = {
        "distill_enabled": True,
        "distill_teacher": "imitation",
    }
    path = tmp_path / "bootstrap_bad_distill_teacher.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_toy2_domain_bootstrap_requires_neural_loop_state_continuation(
    tmp_path: Path,
) -> None:
    non_neural = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule="reputation_imitation",
    )
    non_neural["policy"]["domain"]["objective"] = {"profile": "linear_welfare_heavy"}
    non_neural["policy"]["domain"]["bootstrap"] = {"enabled": True}
    non_neural_path = tmp_path / "bootstrap_non_neural.yaml"
    non_neural_path.write_text(yaml.safe_dump(non_neural), encoding="utf-8")
    with pytest.raises(ValidationError, match="domain bootstrap"):
        load_toy2_config(non_neural_path)

    non_loop = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    non_loop["policy"]["neural_update_backend"] = "batched"
    non_loop["policy"]["domain"]["objective"] = {"profile": "linear_welfare_heavy"}
    non_loop["policy"]["domain"]["bootstrap"] = {"enabled": True}
    non_loop_path = tmp_path / "bootstrap_non_loop.yaml"
    non_loop_path.write_text(yaml.safe_dump(non_loop), encoding="utf-8")
    with pytest.raises(ValidationError, match="state_continuation|domain bootstrap"):
        load_toy2_config(non_loop_path)

    material = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    material["policy"]["domain"]["bootstrap"] = {"enabled": True}
    material_path = tmp_path / "bootstrap_material.yaml"
    material_path.write_text(yaml.safe_dump(material), encoding="utf-8")
    with pytest.raises(ValidationError, match="state_continuation"):
        load_toy2_config(material_path)


def test_toy2_state_continuation_can_flip_counterfactual_target(
    tmp_path: Path,
) -> None:
    material_raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    material_path = tmp_path / "material_counterfactual.yaml"
    material_path.write_text(yaml.safe_dump(material_raw), encoding="utf-8")
    material_config = load_toy2_config(material_path)
    actions = np.asarray([0, 0, 0], dtype=np.int64)
    peers = [[1, 2], [0, 2], [0, 1]]

    material_targets, _ = counterfactual_policy_targets_and_advantages(
        actions=actions,
        peer_ids=peers,
        config=material_config,
    )

    continuation_raw = copy.deepcopy(material_raw)
    continuation_raw["policy"]["domain"]["objective"] = {
        "mode": "state_continuation",
        "social_weight": 0.0,
        "welfare_weight": 1.0,
        "clip_abs": None,
    }
    continuation_path = tmp_path / "continuation_counterfactual.yaml"
    continuation_path.write_text(yaml.safe_dump(continuation_raw), encoding="utf-8")
    continuation_config = load_toy2_config(continuation_path)
    components = counterfactual_policy_advantage_components(
        actions=actions,
        peer_ids=peers,
        config=continuation_config,
    )
    continuation_targets, _ = counterfactual_policy_targets_and_advantages(
        actions=actions,
        peer_ids=peers,
        config=continuation_config,
    )

    assert material_targets[0] == 0
    assert components.material[0] == pytest.approx(-0.2)
    assert components.welfare[0] == pytest.approx(7.0 / 15.0)
    assert components.effective[0] > 0.0
    assert continuation_targets[0] == 1


def test_toy2_objective_profiles_resolve_counterfactual_advantage(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    path = tmp_path / "profile_counterfactual.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_toy2_config(path)

    components = counterfactual_policy_advantage_components(
        actions=np.asarray([0, 0, 0], dtype=np.int64),
        peer_ids=[[1, 2], [0, 2], [0, 1]],
        config=config,
    )

    assert config.policy.domain.objective.mode == "state_continuation"
    assert config.policy.domain.objective.profile == "linear_welfare_heavy"
    assert components.material[0] == pytest.approx(-0.2)
    assert components.social[0] == pytest.approx(0.1)
    assert components.welfare[0] == pytest.approx(7.0 / 15.0)
    assert components.linear[0] == pytest.approx(-0.2 + 0.5 * 0.1 + 2.0 * 7.0 / 15.0)
    assert components.effective[0] == pytest.approx(components.linear[0])


def test_toy2_domain_bootstrap_weight_zero_matches_objective_behavior(
    tmp_path: Path,
) -> None:
    base = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
        epochs=3,
        seed=19,
    )
    base["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    base["run"]["name"] = "toy2_bootstrap_weight_zero_base"
    base_path = tmp_path / "bootstrap_weight_zero_base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    bootstrapped = copy.deepcopy(base)
    bootstrapped["run"]["name"] = "toy2_bootstrap_weight_zero"
    bootstrapped["policy"]["domain"]["bootstrap"] = {
        "enabled": True,
        "weight": 0.0,
    }
    bootstrapped_path = tmp_path / "bootstrap_weight_zero.yaml"
    bootstrapped_path.write_text(
        yaml.safe_dump(bootstrapped, sort_keys=False),
        encoding="utf-8",
    )

    base_result = run_toy2(
        config=load_toy2_config(base_path),
        config_path=base_path,
    )
    bootstrapped_result = run_toy2(
        config=load_toy2_config(bootstrapped_path),
        config_path=bootstrapped_path,
    )

    assert bootstrapped_result.final_action_rate == pytest.approx(
        base_result.final_action_rate
    )
    assert bootstrapped_result.final_mean_payoff == pytest.approx(
        base_result.final_mean_payoff
    )
    assert bootstrapped_result.final_mean_policy_action_probability == pytest.approx(
        base_result.final_mean_policy_action_probability
    )


def test_toy2_domain_bootstrap_weight_one_uses_reputation_teacher_direction(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
        epochs=1,
        seed=23,
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["bootstrap"] = {
        "enabled": True,
        "weight": 1.0,
        "epochs": 3,
        "decay": "linear",
    }
    path = tmp_path / "bootstrap_weight_one.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_toy2_config(path)
    domain = Toy2SpatialDomain(
        config=config,
        config_path=path,
        rng=np.random.default_rng(config.run.seed),
        neural_peer_rng=np.random.default_rng(config.run.seed + 1_000_003),
        interaction_rng=np.random.default_rng(config.run.seed + 2_000_003),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
        device=torch.device("cpu"),
    )
    state = domain.initial_state()
    runner = BinarySpatialRunner(
        domain=domain,
        epochs=1,
        revision_rate=1.0,
        revision_rng=np.random.default_rng(config.run.seed + 5_000_003),
    )

    step_result = runner._hooked_step(
        epoch=1,
        state=state,
        revision_mask=np.ones(domain.agent_count, dtype=bool),
    )
    diagnostics = step_result.extras["domain_bootstrap_diagnostics"]

    np.testing.assert_allclose(
        diagnostics.bootstrapped_effective,
        diagnostics.teacher_signed,
    )


def test_toy2_domain_bootstrap_diagnostics_columns_are_logged(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
        epochs=1,
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["bootstrap"] = {"enabled": True, "weight": 1.0}
    path = tmp_path / "bootstrap_logging.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(path), config_path=path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_reader = csv.DictReader(handle)
        aggregate_rows = list(aggregate_reader)
    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_reader = csv.DictReader(handle)
        micro_rows = list(micro_reader)

    for field in [
        "domain_bootstrap_weight",
        "domain_teacher_signed_advantage_mean",
        "domain_bootstrapped_effective_advantage_mean",
        "domain_bootstrap_teacher",
    ]:
        assert field in aggregate_reader.fieldnames
        assert field in micro_reader.fieldnames
    assert aggregate_rows[-1]["domain_bootstrap_teacher"] == "reputation_imitation"
    assert micro_rows[0]["domain_bootstrap_teacher"] == "reputation_imitation"


def test_toy2_basin_credit_requires_neural_loop_state_continuation(
    tmp_path: Path,
) -> None:
    material = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    material["policy"]["domain"]["basin_credit"] = {"enabled": True}
    material_path = tmp_path / "basin_credit_material.yaml"
    material_path.write_text(yaml.safe_dump(material, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValidationError, match="state_continuation"):
        load_toy2_config(material_path)

    non_loop = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    non_loop["policy"]["neural_update_backend"] = "batched"
    non_loop["policy"]["domain"]["objective"] = {"profile": "linear_welfare_heavy"}
    non_loop["policy"]["domain"]["basin_credit"] = {"enabled": True}
    non_loop_path = tmp_path / "basin_credit_non_loop.yaml"
    non_loop_path.write_text(
        yaml.safe_dump(non_loop, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="state_continuation|basin credit"):
        load_toy2_config(non_loop_path)


def test_toy2_basin_credit_diagnostics_columns_are_logged(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="output_average",
        peer_rule="output_similarity",
        local_update_rule="counterfactual_advantage",
        epochs=1,
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["basin_credit"] = {
        "enabled": True,
        "critic": "prototype_phase",
        "credit_method": "one_step_ablation",
        "objective_weight": 0.5,
        "individual_weight": 0.25,
        "local_social_weight": 0.125,
        "basin_weight": 0.5,
        "training_scope": "all",
        "training_passes": 2,
        "target_basin": "ceiling",
    }
    path = tmp_path / "basin_credit_logging.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(path), config_path=path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_reader = csv.DictReader(handle)
        aggregate_rows = list(aggregate_reader)
    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_reader = csv.DictReader(handle)
        micro_rows = list(micro_reader)

    for field in [
        "domain_basin_training_scope",
        "domain_basin_training_pass_schedule",
        "domain_basin_training_passes",
        "domain_basin_training_passes_configured",
        "domain_basin_min_training_passes",
        "domain_basin_training_pass_score_threshold",
        "domain_basin_training_pass_credit_positive_threshold",
        "domain_basin_training_pass_credit_delta_threshold",
        "domain_basin_training_candidate_rate",
        "domain_basin_objective_weight",
        "domain_basin_individual_weight",
        "domain_basin_local_social_weight",
        "domain_basin_credit_weight",
        "domain_basin_score_mean",
        "domain_basin_score_delta_mean",
        "domain_basin_credit_positive_rate",
        "domain_basin_phase_confidence_mean",
        "domain_basin_target",
        "domain_basin_training_credit_source",
        "domain_basin_training_replay_selection",
        "domain_basin_training_replay_min_selected_rate",
        "domain_basin_training_replay_selected_rate",
        "domain_basin_training_replay_weight_mean",
        "domain_basin_training_replay_weight_positive_rate",
        "domain_basin_training_learned_credit_rate",
        "domain_basin_action1_advantage_mean",
        "domain_basin_action1_advantage_positive_rate",
        "domain_basin_training_effective_advantage_mean",
        "domain_basin_training_effective_advantage_positive_rate",
        "domain_basin_training_effective_advantage_abs_mean",
    ]:
        assert field in aggregate_reader.fieldnames
    for field in [
        "domain_basin_credit",
        "domain_basin_score_observed",
        "domain_basin_score_counterfactual",
        "domain_basin_credit_applied",
        "domain_basin_training_credit_source",
        "domain_basin_training_replay_selection",
        "domain_basin_training_replay_selected",
        "domain_basin_training_replay_weight",
        "domain_basin_training_learned_credit_used",
        "domain_basin_action1_advantage",
        "domain_basin_training_effective_advantage",
    ]:
        assert field in micro_reader.fieldnames
    assert aggregate_rows[-1]["domain_basin_training_scope"] == "all"
    assert aggregate_rows[-1]["domain_basin_training_pass_schedule"] == "fixed"
    assert int(aggregate_rows[-1]["domain_basin_training_passes"]) == 2
    assert int(aggregate_rows[-1]["domain_basin_training_passes_configured"]) == 2
    assert int(aggregate_rows[-1]["domain_basin_min_training_passes"]) == 1
    assert float(
        aggregate_rows[-1]["domain_basin_training_pass_score_threshold"]
    ) == pytest.approx(0.995)
    assert float(
        aggregate_rows[-1]["domain_basin_training_pass_credit_positive_threshold"]
    ) == pytest.approx(0.6)
    assert float(
        aggregate_rows[-1]["domain_basin_training_pass_credit_delta_threshold"]
    ) == pytest.approx(0.0)
    assert float(
        aggregate_rows[-1]["domain_basin_training_candidate_rate"]
    ) == pytest.approx(1.0)
    assert float(aggregate_rows[-1]["domain_basin_objective_weight"]) == pytest.approx(
        0.5
    )
    assert float(aggregate_rows[-1]["domain_basin_individual_weight"]) == pytest.approx(
        0.25
    )
    assert float(
        aggregate_rows[-1]["domain_basin_local_social_weight"]
    ) == pytest.approx(0.125)
    assert float(aggregate_rows[-1]["domain_basin_credit_weight"]) == pytest.approx(0.5)
    assert aggregate_rows[-1]["domain_basin_target"] == "ceiling"
    assert aggregate_rows[-1]["domain_basin_training_credit_source"] == "prototype"
    assert aggregate_rows[-1]["domain_basin_training_replay_selection"] == "all"
    assert float(
        aggregate_rows[-1]["domain_basin_training_replay_selected_rate"]
    ) == pytest.approx(1.0)
    assert float(
        aggregate_rows[-1]["domain_basin_training_replay_weight_mean"]
    ) == pytest.approx(1.0)
    assert float(
        aggregate_rows[-1]["domain_basin_training_replay_weight_positive_rate"]
    ) == pytest.approx(1.0)
    assert aggregate_rows[-1]["domain_basin_training_learned_credit_rate"] == "0.0"
    assert aggregate_rows[-1]["domain_basin_action1_advantage_mean"] != ""
    assert aggregate_rows[-1]["domain_basin_training_effective_advantage_mean"] != ""
    assert micro_rows[0]["domain_basin_credit_applied"] == "True"
    assert micro_rows[0]["domain_basin_credit"] != ""
    assert micro_rows[0]["domain_basin_training_credit_source"] == "prototype"
    assert micro_rows[0]["domain_basin_training_replay_selection"] == "all"
    assert micro_rows[0]["domain_basin_training_replay_selected"] == "True"
    assert float(micro_rows[0]["domain_basin_training_replay_weight"]) == pytest.approx(
        1.0
    )
    assert micro_rows[0]["domain_basin_training_learned_credit_used"] == "False"
    assert micro_rows[0]["domain_basin_action1_advantage"] != ""
    assert micro_rows[0]["domain_basin_training_effective_advantage"] != ""


def test_toy2_basin_transition_samples_artifact_is_written(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="output_average",
        peer_rule="output_similarity",
        local_update_rule="counterfactual_advantage",
        epochs=1,
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["basin_credit"] = {
        "enabled": True,
        "critic": "prototype_phase",
        "credit_method": "one_step_ablation",
        "objective_weight": 0.5,
        "basin_weight": 0.5,
        "training_scope": "all",
        "training_passes": 2,
        "target_basin": "ceiling",
    }
    path = tmp_path / "basin_transition_samples.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(path), config_path=path)

    artifact = result.run_dir / "basin_transition_samples.parquet"
    assert artifact.exists()
    frame = pd.read_parquet(artifact)
    agent_count = raw["environment"]["grid_width"] * raw["environment"]["grid_height"]
    assert len(frame) == agent_count
    for field in [
        "sample_schema_version",
        "toy",
        "run_id",
        "seed",
        "epoch",
        "agent_id",
        "agent_count",
        "target_basin",
        "critic",
        "credit_method",
        "objective_weight",
        "basin_weight",
        "training_scope",
        "training_passes",
        "action_observed",
        "action_counterfactual",
        "policy_action_probability",
        "phase_payoff_alignment",
        "phase_action_rate",
        "phase_policy_rate",
        "phase_consensus",
        "phase_payoff_stability",
        "score_observed",
        "score_counterfactual",
        "selected_action_credit",
        "basin_action1_advantage",
        "training_basin_action1_advantage",
        "training_credit_source",
        "training_replay_selection",
        "training_replay_min_selected_rate",
        "training_replay_selected",
        "training_replay_weight",
        "learned_credit_used",
        "objective_effective_advantage",
        "training_effective_advantage",
        "future_basin_horizon",
        "future_mean_payoff",
        "future_basin_score_delta",
        "future_ceiling_reached",
        "future_epochs_to_ceiling",
        "future_basin_motion_positive",
        "final_mean_payoff",
        "final_ceiling_reached",
        "time_to_ceiling",
        "domain_game_family",
    ]:
        assert field in frame.columns
    assert set(frame["sample_schema_version"]) == {1}
    assert set(frame["toy"]) == {"toy2"}
    assert set(frame["target_basin"]) == {"ceiling"}
    assert set(frame["critic"]) == {"prototype_phase"}
    assert set(frame["training_scope"]) == {"all"}
    for field in [
        "phase_payoff_alignment",
        "phase_action_rate",
        "phase_policy_rate",
        "phase_consensus",
        "phase_payoff_stability",
        "score_observed",
        "score_counterfactual",
    ]:
        assert np.isfinite(frame[field].to_numpy(dtype=float)).all()


def test_toy2_basin_credit_logs_learned_read_only_diagnostics(
    tmp_path: Path,
) -> None:
    model_path = write_learned_basin_test_model(tmp_path)
    raw = tiny_config_dict(
        tmp_path,
        mixer="output_average",
        peer_rule="output_similarity",
        local_update_rule="counterfactual_advantage",
        epochs=1,
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["basin_credit"] = {
        "enabled": True,
        "critic": "prototype_phase",
        "credit_method": "one_step_ablation",
        "objective_weight": 0.5,
        "basin_weight": 0.5,
        "training_scope": "all",
        "target_basin": "ceiling",
        "learned_diagnostic_enabled": True,
        "learned_diagnostic_model_path": str(model_path),
        "learned_diagnostic_abstention_margin_threshold": 0.005,
        "learned_diagnostic_uncertainty_threshold": 0.05,
    }
    path = tmp_path / "basin_learned_diagnostics.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(path), config_path=path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_reader = csv.DictReader(handle)
        aggregate_rows = list(aggregate_reader)
    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_reader = csv.DictReader(handle)
        micro_rows = list(micro_reader)

    for field in [
        "domain_basin_learned_model_path",
        "domain_basin_learned_ensemble_size",
        "domain_basin_learned_score_action0_mean",
        "domain_basin_learned_score_action1_mean",
        "domain_basin_learned_action1_advantage_mean",
        "domain_basin_learned_uncertainty_mean",
        "domain_basin_learned_abstention_rate",
    ]:
        assert field in aggregate_reader.fieldnames
        assert aggregate_rows[-1][field] != ""
    assert (
        "domain_basin_learned_prototype_advantage_correlation"
        in aggregate_reader.fieldnames
    )
    for field in [
        "domain_basin_learned_score_action0",
        "domain_basin_learned_score_action1",
        "domain_basin_learned_action1_advantage",
        "domain_basin_learned_uncertainty",
        "domain_basin_learned_abstain",
    ]:
        assert field in micro_reader.fieldnames
        assert micro_rows[0][field] != ""
    assert int(aggregate_rows[-1]["domain_basin_learned_ensemble_size"]) == 2


def test_toy2_basin_credit_can_train_from_gated_learned_credit(
    tmp_path: Path,
) -> None:
    model_path = write_learned_basin_test_model(tmp_path)
    raw = tiny_config_dict(
        tmp_path,
        mixer="output_average",
        peer_rule="output_similarity",
        local_update_rule="counterfactual_advantage",
        epochs=1,
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["basin_credit"] = {
        "enabled": True,
        "critic": "prototype_phase",
        "credit_method": "one_step_ablation",
        "objective_weight": 0.5,
        "basin_weight": 0.5,
        "training_scope": "all",
        "target_basin": "ceiling",
        "learned_credit_enabled": True,
        "learned_credit_model_path": str(model_path),
        "learned_credit_abstention_margin_threshold": 0.005,
        "learned_credit_uncertainty_threshold": 0.05,
        "learned_credit_fallback": "zero",
    }
    path = tmp_path / "basin_learned_credit.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(path), config_path=path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_reader = csv.DictReader(handle)
        aggregate_rows = list(aggregate_reader)
    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_reader = csv.DictReader(handle)
        micro_rows = list(micro_reader)

    assert "domain_basin_training_credit_source" in aggregate_reader.fieldnames
    assert "domain_basin_training_replay_selection" in aggregate_reader.fieldnames
    assert (
        "domain_basin_training_replay_min_selected_rate"
        in aggregate_reader.fieldnames
    )
    assert "domain_basin_training_replay_selected_rate" in aggregate_reader.fieldnames
    assert (
        "domain_basin_training_replay_weight_mean" in aggregate_reader.fieldnames
    )
    assert (
        "domain_basin_training_replay_weight_positive_rate"
        in aggregate_reader.fieldnames
    )
    assert "domain_basin_training_learned_credit_rate" in aggregate_reader.fieldnames
    assert "domain_basin_training_credit_source" in micro_reader.fieldnames
    assert "domain_basin_training_replay_selection" in micro_reader.fieldnames
    assert "domain_basin_training_replay_selected" in micro_reader.fieldnames
    assert "domain_basin_training_replay_weight" in micro_reader.fieldnames
    assert "domain_basin_training_learned_credit_used" in micro_reader.fieldnames
    assert (
        aggregate_rows[-1]["domain_basin_training_credit_source"]
        == "learned_gated_zero"
    )
    assert float(
        aggregate_rows[-1]["domain_basin_training_learned_credit_rate"]
    ) == pytest.approx(1.0)
    assert aggregate_rows[-1]["domain_basin_training_replay_selection"] == "all"
    assert float(
        aggregate_rows[-1]["domain_basin_training_replay_selected_rate"]
    ) == pytest.approx(1.0)
    assert float(
        aggregate_rows[-1]["domain_basin_training_replay_weight_mean"]
    ) == pytest.approx(1.0)
    assert float(
        aggregate_rows[-1]["domain_basin_training_replay_weight_positive_rate"]
    ) == pytest.approx(1.0)
    assert micro_rows[0]["domain_basin_training_credit_source"] == "learned_gated_zero"
    assert micro_rows[0]["domain_basin_training_replay_selection"] == "all"
    assert micro_rows[0]["domain_basin_training_replay_selected"] == "True"
    assert float(micro_rows[0]["domain_basin_training_replay_weight"]) == pytest.approx(
        1.0
    )
    assert micro_rows[0]["domain_basin_training_learned_credit_used"] == "True"
    assert aggregate_rows[-1]["domain_basin_learned_action1_advantage_mean"] != ""


def test_toy2_basin_credit_replay_all_scope_trains_all_candidates_per_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="output_average",
        peer_rule="output_similarity",
        local_update_rule="counterfactual_advantage",
        epochs=1,
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["basin_credit"] = {
        "enabled": True,
        "critic": "prototype_phase",
        "credit_method": "one_step_ablation",
        "objective_weight": 0.5,
        "basin_weight": 0.5,
        "training_scope": "all",
        "training_passes": 2,
        "target_basin": "ceiling",
    }
    path = tmp_path / "basin_credit_replay_all.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_toy2_config(path)
    domain = Toy2SpatialDomain(
        config=config,
        config_path=path,
        rng=np.random.default_rng(config.run.seed),
        neural_peer_rng=np.random.default_rng(config.run.seed + 1_000_003),
        interaction_rng=np.random.default_rng(config.run.seed + 2_000_003),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
        device=torch.device("cpu"),
        neural_update_backend="loop",
    )
    state = domain.initial_state()
    revision_mask = np.zeros(domain.agent_count, dtype=bool)
    revision_mask[[0, 2]] = True
    calls: list[int] = []

    def fake_train_neural_local_policy(**kwargs: object) -> float:
        calls.append(int(agent_index[id(kwargs["agent"])]))
        return 0.0

    agent_index = {id(agent): idx for idx, agent in enumerate(state.agents or [])}
    monkeypatch.setattr(
        toy_pd_module,
        "train_neural_local_policy",
        fake_train_neural_local_policy,
    )

    updates = domain.post_social_policy_update(
        state,
        BinaryStepContext(epoch=1, revision_mask=revision_mask),
        SimpleNamespace(extras={"logged_neighbors": domain.neighbors}),
        SimpleNamespace(),
        MobilityStepResult.none(domain.agent_count),
        torch.full((domain.agent_count, 2), 0.5, dtype=torch.float64),
    )

    assert len(calls) == domain.agent_count * 2
    assert sorted(set(calls)) == list(range(domain.agent_count))
    diagnostics = updates["extras"]["basin_credit_diagnostics"]
    assert diagnostics.applied_mask.tolist() == [True] * domain.agent_count
    assert diagnostics.training_scope == "all"
    assert diagnostics.training_pass_schedule == "fixed"
    assert diagnostics.training_passes == 2
    assert diagnostics.configured_training_passes == 2


def test_toy2_basin_credit_adaptive_replay_uses_effective_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="output_average",
        peer_rule="output_similarity",
        local_update_rule="counterfactual_advantage",
        epochs=1,
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["basin_credit"] = {
        "enabled": True,
        "critic": "prototype_phase",
        "credit_method": "one_step_ablation",
        "objective_weight": 0.5,
        "basin_weight": 0.5,
        "training_scope": "all",
        "training_passes": 3,
        "training_pass_schedule": "target_score_decay",
        "min_training_passes": 2,
        "training_pass_score_threshold": -1.0,
        "target_basin": "ceiling",
    }
    path = tmp_path / "basin_credit_adaptive_replay.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_toy2_config(path)
    domain = Toy2SpatialDomain(
        config=config,
        config_path=path,
        rng=np.random.default_rng(config.run.seed),
        neural_peer_rng=np.random.default_rng(config.run.seed + 1_000_003),
        interaction_rng=np.random.default_rng(config.run.seed + 2_000_003),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
        device=torch.device("cpu"),
        neural_update_backend="loop",
    )
    state = domain.initial_state()
    calls: list[int] = []
    agent_index = {id(agent): idx for idx, agent in enumerate(state.agents or [])}

    def fake_train_neural_local_policy(**kwargs: object) -> float:
        calls.append(int(agent_index[id(kwargs["agent"])]))
        return 0.0

    monkeypatch.setattr(
        toy_pd_module,
        "train_neural_local_policy",
        fake_train_neural_local_policy,
    )

    updates = domain.post_social_policy_update(
        state,
        BinaryStepContext(
            epoch=1,
            revision_mask=np.ones(domain.agent_count, dtype=bool),
        ),
        SimpleNamespace(extras={"logged_neighbors": domain.neighbors}),
        SimpleNamespace(),
        MobilityStepResult.none(domain.agent_count),
        torch.full((domain.agent_count, 2), 0.5, dtype=torch.float64),
    )

    assert len(calls) == domain.agent_count * 2
    diagnostics = updates["extras"]["basin_credit_diagnostics"]
    assert diagnostics.training_pass_schedule == "target_score_decay"
    assert diagnostics.training_passes == 2
    assert diagnostics.configured_training_passes == 3


def test_toy2_domain_decision_bootstrap_requires_neural_loop_state_continuation(
    tmp_path: Path,
) -> None:
    material = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    material["policy"]["domain"]["bootstrap"] = {"decision_enabled": True}
    material_path = tmp_path / "decision_bootstrap_material.yaml"
    material_path.write_text(yaml.safe_dump(material), encoding="utf-8")
    with pytest.raises(ValidationError, match="state_continuation"):
        load_toy2_config(material_path)

    non_loop = copy.deepcopy(material)
    non_loop["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    non_loop["policy"]["neural_update_backend"] = "batched"
    non_loop_path = tmp_path / "decision_bootstrap_non_loop.yaml"
    non_loop_path.write_text(yaml.safe_dump(non_loop), encoding="utf-8")
    with pytest.raises(ValidationError, match="state_continuation|domain bootstrap"):
        load_toy2_config(non_loop_path)


def test_toy2_domain_decision_replay_requires_neural_loop_state_continuation(
    tmp_path: Path,
) -> None:
    material = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    material["policy"]["domain"]["bootstrap"] = {"replay_enabled": True}
    material_path = tmp_path / "decision_replay_material.yaml"
    material_path.write_text(yaml.safe_dump(material), encoding="utf-8")
    with pytest.raises(ValidationError, match="state_continuation"):
        load_toy2_config(material_path)

    non_loop = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    non_loop["policy"]["neural_update_backend"] = "batched"
    non_loop["policy"]["domain"]["objective"] = {"profile": "linear_welfare_heavy"}
    non_loop["policy"]["domain"]["bootstrap"] = {"replay_enabled": True}
    non_loop_path = tmp_path / "decision_replay_non_loop.yaml"
    non_loop_path.write_text(yaml.safe_dump(non_loop), encoding="utf-8")
    with pytest.raises(ValidationError, match="state_continuation|domain bootstrap"):
        load_toy2_config(non_loop_path)


def test_toy2_domain_decision_bootstrap_weight_zero_matches_objective_behavior(
    tmp_path: Path,
) -> None:
    base = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
        epochs=3,
        seed=29,
    )
    base["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    base["run"]["name"] = "toy2_decision_bootstrap_weight_zero_base"
    base_path = tmp_path / "decision_bootstrap_weight_zero_base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    bootstrapped = copy.deepcopy(base)
    bootstrapped["run"]["name"] = "toy2_decision_bootstrap_weight_zero"
    bootstrapped["policy"]["domain"]["bootstrap"] = {
        "decision_enabled": True,
        "decision_weight": 0.0,
    }
    bootstrapped_path = tmp_path / "decision_bootstrap_weight_zero.yaml"
    bootstrapped_path.write_text(
        yaml.safe_dump(bootstrapped, sort_keys=False),
        encoding="utf-8",
    )

    base_result = run_toy2(
        config=load_toy2_config(base_path),
        config_path=base_path,
    )
    bootstrapped_result = run_toy2(
        config=load_toy2_config(bootstrapped_path),
        config_path=bootstrapped_path,
    )

    assert bootstrapped_result.final_action_rate == pytest.approx(
        base_result.final_action_rate
    )
    assert bootstrapped_result.final_mean_payoff == pytest.approx(
        base_result.final_mean_payoff
    )
    assert bootstrapped_result.final_mean_policy_action_probability == pytest.approx(
        base_result.final_mean_policy_action_probability
    )


def test_toy2_domain_decision_bootstrap_weight_one_uses_teacher_probability(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
        epochs=1,
        seed=31,
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["bootstrap"] = {
        "decision_enabled": True,
        "decision_weight": 1.0,
        "decision_epochs": 3,
        "decision_decay": "linear",
    }
    path = tmp_path / "decision_bootstrap_weight_one.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_toy2_config(path)
    domain = Toy2SpatialDomain(
        config=config,
        config_path=path,
        rng=np.random.default_rng(config.run.seed),
        neural_peer_rng=np.random.default_rng(config.run.seed + 1_000_003),
        interaction_rng=np.random.default_rng(config.run.seed + 2_000_003),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
        device=torch.device("cpu"),
    )
    state = domain.initial_state()
    runner = BinarySpatialRunner(
        domain=domain,
        epochs=1,
        revision_rate=1.0,
        revision_rng=np.random.default_rng(config.run.seed + 5_000_003),
    )

    step_result = runner._hooked_step(
        epoch=1,
        state=state,
        revision_mask=np.ones(domain.agent_count, dtype=bool),
    )
    diagnostics = step_result.extras["domain_decision_bootstrap_diagnostics"]
    candidate_probs = step_result.extras["decision_action_probs"][:, 1].detach().numpy()

    np.testing.assert_allclose(
        diagnostics.bootstrapped_probabilities,
        diagnostics.teacher_probabilities,
    )
    np.testing.assert_allclose(candidate_probs, diagnostics.teacher_probabilities)


def test_toy2_domain_decision_bootstrap_diagnostics_columns_and_expiry_are_logged(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
        epochs=2,
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["bootstrap"] = {
        "decision_enabled": True,
        "decision_weight": 1.0,
        "decision_epochs": 1,
    }
    path = tmp_path / "decision_bootstrap_logging.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(path), config_path=path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_reader = csv.DictReader(handle)
        aggregate_rows = list(aggregate_reader)
    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_reader = csv.DictReader(handle)
        micro_rows = list(micro_reader)

    for field in [
        "domain_decision_bootstrap_weight",
        "domain_teacher_decision_probability_mean",
        "domain_neural_decision_probability_mean",
        "domain_bootstrapped_decision_probability_mean",
        "domain_decision_bootstrap_teacher",
    ]:
        assert field in aggregate_reader.fieldnames
        assert field in micro_reader.fieldnames
    active_aggregate_row = next(row for row in aggregate_rows if row["epoch"] == "1")
    active_micro_row = next(row for row in micro_rows if row["epoch"] == "1")
    assert active_aggregate_row["domain_decision_bootstrap_teacher"] == (
        "reputation_imitation"
    )
    assert active_micro_row["domain_decision_bootstrap_teacher"] == (
        "reputation_imitation"
    )
    assert float(aggregate_rows[-1]["domain_decision_bootstrap_weight"]) == 0.0
    assert aggregate_rows[-1]["domain_decision_bootstrap_teacher"] == ""
    assert aggregate_rows[-1]["domain_teacher_decision_probability_mean"] == ""


def test_toy2_domain_decision_replay_diagnostics_columns_and_expiry_are_logged(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
        epochs=2,
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["bootstrap"] = {
        "decision_enabled": True,
        "decision_weight": 1.0,
        "decision_epochs": 1,
        "replay_enabled": True,
        "replay_weight": 1.0,
        "replay_epochs": 1,
        "replay_stable_teacher_only": False,
        "replay_require_objective_agreement": False,
        "replay_require_postsocial_alignment_improvement": False,
    }
    path = tmp_path / "decision_replay_logging.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(path), config_path=path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_reader = csv.DictReader(handle)
        aggregate_rows = list(aggregate_reader)
    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_reader = csv.DictReader(handle)
        micro_rows = list(micro_reader)

    for field in [
        "domain_decision_replay_weight",
        "domain_decision_replay_candidate_rate",
        "domain_decision_replay_applied_rate",
        "domain_decision_replay_rejected_unstable_teacher_rate",
        "domain_decision_replay_rejected_objective_rate",
        "domain_decision_replay_rejected_postsocial_rate",
        "domain_decision_replay_teacher_probability_mean",
        "domain_decision_replay_teacher",
    ]:
        assert field in aggregate_reader.fieldnames
        assert field in micro_reader.fieldnames
    active_aggregate_row = next(row for row in aggregate_rows if row["epoch"] == "1")
    active_micro_row = next(row for row in micro_rows if row["epoch"] == "1")
    assert active_aggregate_row["domain_decision_replay_teacher"] == (
        "reputation_imitation"
    )
    assert active_micro_row["domain_decision_replay_teacher"] == (
        "reputation_imitation"
    )
    assert float(active_aggregate_row["domain_decision_replay_candidate_rate"]) > 0.0
    assert float(active_aggregate_row["domain_decision_replay_applied_rate"]) > 0.0
    assert float(aggregate_rows[-1]["domain_decision_replay_weight"]) == 0.0
    assert aggregate_rows[-1]["domain_decision_replay_teacher"] == ""
    assert aggregate_rows[-1]["domain_decision_replay_teacher_probability_mean"] == ""


def test_toy2_domain_distill_bootstrap_requires_neural_loop_state_continuation(
    tmp_path: Path,
) -> None:
    material = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    material["policy"]["domain"]["bootstrap"] = {"distill_enabled": True}
    material_path = tmp_path / "distill_bootstrap_material.yaml"
    material_path.write_text(yaml.safe_dump(material), encoding="utf-8")
    with pytest.raises(ValidationError, match="state_continuation"):
        load_toy2_config(material_path)

    non_loop = copy.deepcopy(material)
    non_loop["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    non_loop["policy"]["neural_update_backend"] = "batched"
    non_loop_path = tmp_path / "distill_bootstrap_non_loop.yaml"
    non_loop_path.write_text(yaml.safe_dump(non_loop), encoding="utf-8")
    with pytest.raises(ValidationError, match="state_continuation|domain bootstrap"):
        load_toy2_config(non_loop_path)


def test_toy2_domain_distill_bootstrap_weight_zero_matches_objective_behavior(
    tmp_path: Path,
) -> None:
    base = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
        epochs=3,
        seed=37,
    )
    base["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    base["run"]["name"] = "toy2_distill_bootstrap_weight_zero_base"
    base_path = tmp_path / "distill_bootstrap_weight_zero_base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    bootstrapped = copy.deepcopy(base)
    bootstrapped["run"]["name"] = "toy2_distill_bootstrap_weight_zero"
    bootstrapped["policy"]["domain"]["bootstrap"] = {
        "distill_enabled": True,
        "distill_weight": 0.0,
    }
    bootstrapped_path = tmp_path / "distill_bootstrap_weight_zero.yaml"
    bootstrapped_path.write_text(
        yaml.safe_dump(bootstrapped, sort_keys=False),
        encoding="utf-8",
    )

    base_result = run_toy2(
        config=load_toy2_config(base_path),
        config_path=base_path,
    )
    bootstrapped_result = run_toy2(
        config=load_toy2_config(bootstrapped_path),
        config_path=bootstrapped_path,
    )

    assert bootstrapped_result.final_action_rate == pytest.approx(
        base_result.final_action_rate
    )
    assert bootstrapped_result.final_mean_payoff == pytest.approx(
        base_result.final_mean_payoff
    )
    assert bootstrapped_result.final_mean_policy_action_probability == pytest.approx(
        base_result.final_mean_policy_action_probability
    )


def test_toy2_teacher_distillation_loss_moves_probability_toward_teacher(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
        policy_prior_action_probability=0.2,
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["bootstrap"] = {
        "distill_enabled": True,
        "distill_weight": 1.0,
    }
    path = tmp_path / "distill_single_agent.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_toy2_config(path)
    agent = create_agents(config, torch.device("cpu"))[0]
    observation = torch.zeros(config.agents.model.input_dim)

    before = float(
        torch.softmax(agent.model(observation.unsqueeze(0)), dim=-1)[0, 1].detach()
    )
    train_neural_local_policy(
        agent=agent,
        observation=observation,
        action=0,
        payoff=0.0,
        peer_actions=np.asarray([0, 0, 0, 0], dtype=np.int64),
        config=config,
        teacher_distill_probability=1.0,
        teacher_distill_weight=1.0,
        base_loss_weight=0.0,
    )
    after = float(
        torch.softmax(agent.model(observation.unsqueeze(0)), dim=-1)[0, 1].detach()
    )

    assert after > before


def test_toy2_domain_distill_bootstrap_diagnostics_columns_and_expiry_are_logged(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
        epochs=2,
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["bootstrap"] = {
        "distill_enabled": True,
        "distill_weight": 1.0,
        "distill_epochs": 1,
    }
    path = tmp_path / "distill_bootstrap_logging.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(path), config_path=path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_reader = csv.DictReader(handle)
        aggregate_rows = list(aggregate_reader)
    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_reader = csv.DictReader(handle)
        micro_rows = list(micro_reader)

    for field in [
        "domain_distill_bootstrap_weight",
        "domain_distill_teacher",
        "domain_teacher_policy_bce_mean",
        "domain_teacher_policy_kl_mean",
        "domain_teacher_neural_probability_mean",
        "domain_teacher_probability_mean",
        "domain_teacher_neural_argmax_agreement",
        "domain_teacher_realized_action_agreement",
    ]:
        assert field in aggregate_reader.fieldnames
        assert field in micro_reader.fieldnames
    active_aggregate_row = next(row for row in aggregate_rows if row["epoch"] == "1")
    active_micro_row = next(row for row in micro_rows if row["epoch"] == "1")
    assert active_aggregate_row["domain_distill_teacher"] == "reputation_imitation"
    assert active_micro_row["domain_distill_teacher"] == "reputation_imitation"
    assert float(aggregate_rows[-1]["domain_distill_bootstrap_weight"]) == 0.0
    assert aggregate_rows[-1]["domain_distill_teacher"] == ""
    assert aggregate_rows[-1]["domain_teacher_policy_bce_mean"] == ""


def test_toy2_teacher_alignment_diagnostics_columns_are_logged(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
        epochs=2,
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["bootstrap"] = {
        "distill_enabled": True,
        "distill_weight": 1.0,
        "distill_epochs": 1,
    }
    path = tmp_path / "teacher_alignment_logging.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(path), config_path=path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_reader = csv.DictReader(handle)
        aggregate_rows = list(aggregate_reader)
    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_reader = csv.DictReader(handle)
        micro_rows = list(micro_reader)

    for field in [
        "domain_teacher_policy_bce_pre_local_mean",
        "domain_teacher_policy_bce_post_local_mean",
        "domain_teacher_policy_bce_post_social_mean",
        "domain_teacher_bce_delta_local",
        "domain_teacher_bce_delta_social",
        "domain_teacher_neural_argmax_agreement_post_social",
        "domain_teacher_target_shift_mean",
        "domain_effective_advantage_teacher_sign_agreement",
        "domain_base_grad_norm_mean",
        "domain_base_distill_grad_cosine_mean",
        "domain_distill_candidate_rate",
        "domain_distill_applied_rate",
        "domain_distill_rejected_unstable_teacher_rate",
        "domain_distill_rejected_gradient_rate",
        "domain_teacher_alignment_teacher",
    ]:
        assert field in aggregate_reader.fieldnames
        assert field in micro_reader.fieldnames
    active_aggregate_row = next(row for row in aggregate_rows if row["epoch"] == "1")
    active_micro_row = next(row for row in micro_rows if row["epoch"] == "1")
    assert active_aggregate_row["domain_teacher_alignment_teacher"] == (
        "reputation_imitation"
    )
    assert active_micro_row["domain_teacher_alignment_teacher"] == (
        "reputation_imitation"
    )
    assert active_aggregate_row["domain_teacher_policy_bce_pre_local_mean"] != ""
    assert active_aggregate_row["domain_effective_advantage_teacher_sign_agreement"] != ""
    assert active_aggregate_row["domain_base_grad_norm_mean"] != ""
    assert active_aggregate_row["domain_distill_candidate_rate"] != ""
    assert active_aggregate_row["domain_distill_applied_rate"] != ""
    assert aggregate_rows[-1]["domain_teacher_alignment_teacher"] == (
        "reputation_imitation"
    )
    assert aggregate_rows[-1]["domain_base_grad_norm_mean"] == ""


def test_toy2_distill_gradient_gate_rejects_conflicting_teacher_updates(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
        epochs=1,
    )
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["bootstrap"] = {
        "distill_enabled": True,
        "distill_weight": 1.0,
        "distill_gradient_gate_enabled": True,
        "distill_gradient_min_cosine": 1.0,
    }
    path = tmp_path / "distill_gradient_gate.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(path), config_path=path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        [_, active_row] = list(csv.DictReader(handle))

    assert float(active_row["domain_distill_candidate_rate"]) == pytest.approx(1.0)
    assert float(active_row["domain_distill_applied_rate"]) < 1.0
    assert float(active_row["domain_distill_rejected_gradient_rate"]) > 0.0


def test_toy2_non_neural_policy_canonicalizes_backend(tmp_path: Path) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule="fermi_imitation",
    )
    config["policy"]["neural_update_backend"] = "tensor_batched"
    path = tmp_path / "classical_backend.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert load_toy2_config(path).policy.neural_update_backend == "loop"


def test_toy2_counterfactual_vector_targets_match_loop_formula(
    tmp_path: Path,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    config = load_toy2_config(config_path)
    actions = np.asarray([0, 1, 1, 0, 1], dtype=np.int64)
    peer_ids = [[1, 2], [0, 3], [], [2, 4], [0]]

    targets, advantages = counterfactual_policy_targets_and_advantages(
        actions=actions,
        peer_ids=peer_ids,
        config=config,
    )

    payoff = config.game.payoff
    safe_scale = max(abs(payoff.T), abs(payoff.R), abs(payoff.P), abs(payoff.S), 1.0)
    expected_targets = []
    expected_advantages = []
    for peers in peer_ids:
        defect_payoff, cooperate_payoff = counterfactual_action_payoffs(
            peer_actions=actions[peers],
            config=config,
        )
        signed_advantage = transformed_advantage(
            (cooperate_payoff - defect_payoff) / safe_scale,
            config,
        )
        expected_targets.append(1 if signed_advantage >= 0.0 else 0)
        expected_advantages.append(abs(signed_advantage))

    np.testing.assert_array_equal(targets, np.asarray(expected_targets))
    np.testing.assert_allclose(advantages, np.asarray(expected_advantages))


def test_toy2_uniform_peer_index_payoffs_match_peer_ids(tmp_path: Path) -> None:
    config_path = write_tiny_config(tmp_path, mixer="none", peer_rule="none")
    config = load_toy2_config(config_path)
    actions = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int64)
    peer_ids = [[1, 2], [0, 3], [1, 4], [0, 5], [2, 5], [3, 4]]
    peer_index = uniform_peer_index(peer_ids)
    assert peer_index is not None

    loop_payoffs = compute_payoffs_from_peer_ids(actions, peer_ids, config)
    indexed_payoffs = compute_payoffs_from_peer_index(actions, peer_index, config)

    np.testing.assert_allclose(indexed_payoffs, loop_payoffs)


def test_toy2_policy_consensus_matches_pairwise_formula() -> None:
    coop = np.asarray([0.1, 0.9, 0.4, 0.7], dtype=np.float64)
    policy_probs = torch.as_tensor(
        np.column_stack([1.0 - coop, coop]),
        dtype=torch.float32,
    )
    diffs = [
        abs(float(coop[i] - coop[j]))
        for i in range(len(coop))
        for j in range(i + 1, len(coop))
    ]
    expected = 1.0 - float(np.mean(diffs))

    assert policy_consensus(policy_probs) == pytest.approx(expected)


def test_toy2_vectorized_observations_match_loop(tmp_path: Path) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        grid_width=3,
        grid_height=2,
    )
    config = load_toy2_config(config_path)
    device = torch.device("cpu")
    agents = create_agents(config, device)
    actions = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int64)
    payoffs = np.linspace(-1.0, 1.0, len(actions))
    peer_ids = [[1, 2], [0, 3], [1, 4], [0, 5], [2, 5], [3, 4]]
    peer_index = uniform_peer_index(peer_ids)
    assert peer_index is not None
    payoff_ema = np.linspace(0.1, 0.6, len(actions))
    previous_payoff_ema = payoff_ema - 0.05
    for agent, current, previous in zip(
        agents,
        payoff_ema,
        previous_payoff_ema,
        strict=True,
    ):
        agent.payoff_ema = float(current)
        agent.previous_payoff_ema = float(previous)

    loop_observations = build_observations(
        actions=actions,
        payoffs=payoffs,
        agents=agents,
        neighbors=peer_ids,
        payoff_normalizer=5.0,
        device=device,
    )
    indexed_observations = build_observations(
        actions=actions,
        payoffs=payoffs,
        agents=agents,
        neighbors=peer_ids,
        payoff_normalizer=5.0,
        device=device,
        peer_index=peer_index,
        payoff_ema=payoff_ema,
        previous_payoff_ema=previous_payoff_ema,
    )

    assert torch.allclose(indexed_observations, loop_observations)


def test_toy2_torch_observation_payoff_counterfactual_helpers_match_numpy(
    tmp_path: Path,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
        grid_width=3,
        grid_height=2,
    )
    config = load_toy2_config(config_path)
    device = torch.device("cpu")
    actions = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int64)
    payoffs = np.linspace(-1.0, 1.0, len(actions))
    reputation = np.linspace(0.1, 0.6, len(actions))
    payoff_ema = np.linspace(0.2, 0.7, len(actions))
    previous_payoff_ema = payoff_ema - 0.1
    peer_ids = [[1, 2], [0, 3], [1, 4], [0, 5], [2, 5], [3, 4]]
    peer_index = uniform_peer_index(peer_ids)
    assert peer_index is not None

    numpy_observations = build_observations(
        actions=actions,
        payoffs=payoffs,
        agents=[],
        neighbors=peer_ids,
        payoff_normalizer=5.0,
        device=device,
        reputation=reputation,
        reputation_observation_mode="self_neighbor_mean",
        peer_index=peer_index,
        payoff_ema=payoff_ema,
        previous_payoff_ema=previous_payoff_ema,
    )
    torch_observations = build_observations(
        actions=torch.as_tensor(actions, dtype=torch.long),
        payoffs=torch.as_tensor(payoffs, dtype=torch.float64),
        agents=[],
        neighbors=peer_ids,
        payoff_normalizer=5.0,
        device=device,
        reputation=torch.as_tensor(reputation, dtype=torch.float64),
        reputation_observation_mode="self_neighbor_mean",
        peer_index=torch.as_tensor(peer_index, dtype=torch.long),
        payoff_ema=torch.as_tensor(payoff_ema, dtype=torch.float64),
        previous_payoff_ema=torch.as_tensor(
            previous_payoff_ema,
            dtype=torch.float64,
        ),
    )
    assert torch.allclose(torch_observations, numpy_observations)

    numpy_payoffs = compute_payoffs_from_peer_index(actions, peer_index, config)
    torch_payoffs = compute_payoffs_from_peer_index(
        torch.as_tensor(actions, dtype=torch.long),
        torch.as_tensor(peer_index, dtype=torch.long),
        config,
    )
    np.testing.assert_allclose(torch_payoffs.numpy(), numpy_payoffs)

    numpy_targets, numpy_advantages = counterfactual_policy_targets_and_advantages(
        actions=actions,
        peer_ids=peer_ids,
        config=config,
        peer_index=peer_index,
    )
    torch_targets, torch_advantages = counterfactual_policy_targets_and_advantages_tensor(
        actions=torch.as_tensor(actions, dtype=torch.long),
        peer_ids=peer_ids,
        config=config,
        peer_index=torch.as_tensor(peer_index, dtype=torch.long),
        device=device,
    )
    np.testing.assert_array_equal(torch_targets.numpy(), numpy_targets)
    np.testing.assert_allclose(torch_advantages.numpy(), numpy_advantages)


def test_toy2_counterfactual_uniform_peer_index_matches_peer_ids(
    tmp_path: Path,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule="counterfactual_advantage",
    )
    config = load_toy2_config(config_path)
    actions = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int64)
    peer_ids = [[1, 2], [0, 3], [1, 4], [0, 5], [2, 5], [3, 4]]
    peer_index = uniform_peer_index(peer_ids)
    assert peer_index is not None

    loop_targets, loop_advantages = counterfactual_policy_targets_and_advantages(
        actions=actions,
        peer_ids=peer_ids,
        config=config,
    )
    indexed_targets, indexed_advantages = counterfactual_policy_targets_and_advantages(
        actions=actions,
        peer_ids=peer_ids,
        config=config,
        peer_index=peer_index,
    )

    np.testing.assert_array_equal(indexed_targets, loop_targets)
    np.testing.assert_allclose(indexed_advantages, loop_advantages)


def test_toy2_well_mixed_context_skips_static_peer_index(tmp_path: Path) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        neural_peer_mode="well_mixed",
        interaction_mode="well_mixed_resampled",
    )
    config = load_toy2_config(config_path)
    rng = np.random.default_rng(config.run.seed)
    domain = Toy2SpatialDomain(
        config=config,
        config_path=config_path,
        rng=rng,
        neural_peer_rng=np.random.default_rng(config.run.seed + 1_000_003),
        interaction_rng=np.random.default_rng(config.run.seed + 2_000_003),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
        device=torch.device(config.simulation.device),
    )
    state = domain.initial_state()

    context = domain.build_step_context(
        epoch=1,
        state=state,
        revision_mask=np.ones(domain.agent_count, dtype=bool),
    )

    assert "current_interaction_peer_index" not in context.extras
    assert domain._spatial_neural_peer_index() is None


def test_toy2_neural_local_step_routes_through_binary_policy_learning_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        decision_mode="argmax",
        epochs=1,
    )
    config = load_toy2_config(config_path)
    domain = Toy2SpatialDomain(
        config=config,
        config_path=config_path,
        rng=np.random.default_rng(config.run.seed),
        neural_peer_rng=np.random.default_rng(config.run.seed + 1_000_003),
        interaction_rng=np.random.default_rng(config.run.seed + 2_000_003),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
        device=torch.device("cpu"),
        neural_update_backend=config.policy.neural_update_backend,
    )
    state = domain.initial_state()
    context = domain.build_step_context(
        epoch=1,
        state=state,
        revision_mask=np.ones(domain.agent_count, dtype=bool),
    )
    original_unit = toy_pd_module.BinaryPolicyLearningUnit
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
            seen["has_post_collect_policy_probs"] = callable(
                callbacks.post_collect_policy_probs
            )

        def run(self) -> Any:
            seen["run_called"] = True
            result = self._inner.run()
            seen["decision_shape"] = tuple(result.decision_action_probs.shape)
            seen["post_shape"] = tuple(result.post_local_probs.shape)
            return result

    monkeypatch.setattr(
        toy_pd_module,
        "BinaryPolicyLearningUnit",
        SpyBinaryPolicyLearningUnit,
    )

    result = domain.local_step(state, context)

    assert seen == {
        "agent_count": domain.agent_count,
        "context": context,
        "callbacks_type": "BinaryPolicyLearningCallbacks",
        "has_collect_policy_probs": True,
        "has_decision_action_probs": True,
        "has_sample_actions": True,
        "has_local_update": True,
        "has_refresh_policy_cache": True,
        "has_post_collect_policy_probs": True,
        "run_called": True,
        "decision_shape": (domain.agent_count, 2),
        "post_shape": (domain.agent_count, 2),
    }
    assert isinstance(result, BinaryLocalStepResult)
    assert result.social_mode == "policy_distill"
    assert result.actions_after_revision is not None
    assert result.extras["decision_action_probs"].shape == (domain.agent_count, 2)
    assert "state_continuation_components" in result.extras


@pytest.mark.parametrize(
    "local_update_rule",
    ["sampled_policy_gradient", "counterfactual_advantage"],
)
def test_toy2_batched_local_training_matches_loop(
    tmp_path: Path,
    local_update_rule: str,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        local_update_rule=local_update_rule,
    )
    config = load_toy2_config(config_path)
    device = torch.device("cpu")
    loop_agents = create_agents(config, device)
    batched_agents = create_agents(config, device)
    agent_count = config.environment.grid_width * config.environment.grid_height
    torch.manual_seed(611)
    observations = torch.randn(agent_count, config.agents.model.input_dim)
    actions = np.asarray([agent_id % 2 for agent_id in range(agent_count)])
    payoffs = np.linspace(-1.2, 1.4, agent_count)
    peer_ids = [
        [int((agent_id - 1) % agent_count), int((agent_id + 1) % agent_count)]
        for agent_id in range(agent_count)
    ]
    peer_index = uniform_peer_index(peer_ids)
    assert peer_index is not None
    revision_mask = np.asarray(
        [agent_id % 3 != 1 for agent_id in range(agent_count)]
    )

    for _ in range(3):
        loop_losses = [0.0 for _ in range(agent_count)]
        update_agent_ids = (
            range(agent_count)
            if local_update_rule == "counterfactual_advantage"
            else np.flatnonzero(revision_mask)
        )
        for agent_id in update_agent_ids:
            loop_losses[int(agent_id)] = train_neural_local_policy(
                agent=loop_agents[int(agent_id)],
                observation=observations[int(agent_id)],
                action=int(actions[int(agent_id)]),
                payoff=float(payoffs[int(agent_id)]),
                peer_actions=actions[peer_ids[int(agent_id)]],
                config=config,
            )
        batched_losses = train_neural_local_policies_batched(
            agents=batched_agents,
            observations=observations,
            actions=actions,
            payoffs=payoffs,
            peer_ids=peer_ids,
            revision_mask=revision_mask,
            config=config,
            peer_index=peer_index,
        )

        assert np.allclose(batched_losses, loop_losses, atol=1e-6)
    assert_agent_parameters_match(batched_agents, loop_agents)


def test_toy2_batched_kl_distillation_matches_loop(tmp_path: Path) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="output_average",
        peer_rule="none",
    )
    config = load_toy2_config(config_path)
    device = torch.device("cpu")
    loop_agents = create_agents(config, device)
    batched_agents = create_agents(config, device)
    agent_count = config.environment.grid_width * config.environment.grid_height
    torch.manual_seed(612)
    observations = torch.randn(agent_count, config.agents.model.input_dim)
    previous_probs = torch.softmax(torch.randn(agent_count, 2), dim=-1)
    peer_ids = [
        [int((agent_id - 1) % agent_count), int((agent_id + 1) % agent_count)]
        for agent_id in range(agent_count)
    ]

    for _ in range(3):
        loop_losses = apply_output_average(
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
        )

        assert np.allclose(batched_losses, loop_losses, atol=1e-6)
    assert_agent_parameters_match(batched_agents, loop_agents)


def test_toy2_local_update_rule_rejects_unknown_value(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["policy"]["domain"]["local_update_rule"] = "unknown"
    path = tmp_path / "invalid_local_update.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_toy2_neural_peer_mode_defaults_to_spatial(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    path = tmp_path / "default_neural_peer_mode.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert load_toy2_config(path).policy.domain.neural_peer_mode == "spatial"


@pytest.mark.parametrize("neural_peer_mode", ["spatial", "well_mixed"])
def test_toy2_neural_peer_mode_accepts_supported_values(
    tmp_path: Path,
    neural_peer_mode: str,
) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        neural_peer_mode=neural_peer_mode,
    )
    path = tmp_path / f"{neural_peer_mode}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert load_toy2_config(path).policy.domain.neural_peer_mode == neural_peer_mode


def test_toy2_neural_peer_mode_rejects_unknown_value(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["policy"]["domain"]["neural_peer_mode"] = "unknown"
    path = tmp_path / "invalid_neural_peer_mode.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_toy2_interaction_mode_defaults_to_spatial(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    path = tmp_path / "default_interaction_mode.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert load_toy2_config(path).policy.domain.interaction_mode == "spatial"


@pytest.mark.parametrize("interaction_mode", ["spatial", "well_mixed_resampled"])
def test_toy2_interaction_mode_accepts_supported_values(
    tmp_path: Path,
    interaction_mode: str,
) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        interaction_mode=interaction_mode,
    )
    path = tmp_path / f"{interaction_mode}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert load_toy2_config(path).policy.domain.interaction_mode == interaction_mode


def test_toy2_interaction_mode_rejects_unknown_value(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["policy"]["domain"]["interaction_mode"] = "unknown"
    path = tmp_path / "invalid_interaction_mode.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_toy2_reputation_imitation_config_defaults_enabled(
    tmp_path: Path,
) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule="reputation_imitation",
    )
    path = tmp_path / "reputation_imitation_default.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_toy2_config(path)

    assert loaded.policy.rule == "reputation_imitation"
    assert loaded.state.reputation.enabled is True
    assert loaded.state.reputation.decay == pytest.approx(0.9)
    assert loaded.state.mobility.enabled is False


def test_toy2_neural_reputation_observation_requires_8d_model(
    tmp_path: Path,
) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["state"]["reputation"] = {
        "enabled": True,
        "observation_mode": "self_neighbor_mean",
    }
    config["agents"]["model"]["input_dim"] = 8
    path = tmp_path / "neural_reputation_observation.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_toy2_config(path)

    assert loaded.state.reputation.observation_mode == "self_neighbor_mean"
    assert loaded.agents.model.input_dim == 8


def test_toy2_neural_reputation_observation_rejects_legacy_6d_model(
    tmp_path: Path,
) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["state"]["reputation"] = {
        "enabled": True,
        "observation_mode": "self_neighbor_mean",
    }
    path = tmp_path / "neural_reputation_observation_invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError, match="model.input_dim=8"):
        load_toy2_config(path)


def test_toy2_observation_appends_reputation_features() -> None:
    agents = [
        SimpleNamespace(payoff_ema=0.2, previous_payoff_ema=0.1),
        SimpleNamespace(payoff_ema=0.4, previous_payoff_ema=0.3),
    ]
    observations = build_observations(
        actions=np.asarray([1, 0], dtype=np.int64),
        payoffs=np.asarray([3.0, 1.0], dtype=np.float64),
        agents=agents,
        neighbors=[[1], [0]],
        payoff_normalizer=5.0,
        device=torch.device("cpu"),
        reputation=np.asarray([0.7, 0.2], dtype=np.float64),
        reputation_observation_mode="self_neighbor_mean",
    )

    assert tuple(observations.shape) == (2, 8)
    np.testing.assert_allclose(
        observations[:, 6:].numpy(),
        np.asarray([[0.7, 0.2], [0.2, 0.7]], dtype=np.float32),
    )


def test_toy2_reputation_config_rejects_unknown_option(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["state"]["reputation"] = {"enabled": True, "unknown": 1}
    path = tmp_path / "invalid_reputation_option.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_toy2_mobility_config_rejects_unknown_option(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["state"]["mobility"] = {"enabled": False, "unknown": 1}
    path = tmp_path / "invalid_mobility_option.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_toy2_reputation_imitation_requires_enabled_reputation(
    tmp_path: Path,
) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule="reputation_imitation",
    )
    config["state"]["reputation"] = {"enabled": False}
    path = tmp_path / "disabled_reputation_imitation.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_toy2_decision_mode_defaults_to_sampled(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    path = tmp_path / "default_decision_mode.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert load_toy2_config(path).policy.decision.mode == "sampled"
    assert load_toy2_config(path).policy.decision.terminal_argmax_epochs == 0


@pytest.mark.parametrize("decision_mode", ["sampled", "argmax"])
def test_toy2_decision_mode_accepts_supported_values(
    tmp_path: Path,
    decision_mode: str,
) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        decision_mode=decision_mode,
    )
    path = tmp_path / f"{decision_mode}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert load_toy2_config(path).policy.decision.mode == decision_mode


def test_toy2_terminal_argmax_epochs_accepts_nonnegative_value(
    tmp_path: Path,
) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["policy"]["decision"]["terminal_argmax_epochs"] = 3
    path = tmp_path / "terminal_argmax_epochs.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert load_toy2_config(path).policy.decision.terminal_argmax_epochs == 3


def test_toy2_terminal_argmax_epochs_rejects_negative_value(
    tmp_path: Path,
) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["policy"]["decision"]["terminal_argmax_epochs"] = -1
    path = tmp_path / "invalid_terminal_argmax_epochs.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_toy2_decision_mode_rejects_unknown_value(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["policy"]["decision"]["mode"] = "unknown"
    path = tmp_path / "invalid_decision_mode.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_toy2_action_temperature_rejects_nonpositive_value(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["policy"]["decision"]["action_temperature"] = 0.0
    path = tmp_path / "invalid_action_temperature.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_toy2_decision_calibration_defaults_to_none(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["policy"]["decision"].pop("calibration", None)
    path = tmp_path / "default_decision_calibration.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    calibration = load_toy2_config(path).policy.decision.calibration

    assert calibration.mode == "none"
    assert calibration.strength == pytest.approx(4.0)


def test_toy2_payoff_threshold_calibration_accepts_stag_hunt(
    tmp_path: Path,
) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        calibration_mode="payoff_threshold",
        calibration_strength=2.0,
    )
    config["game"] = {
        "family": "stag_hunt",
        "payoff": {
            "T": 3.0,
            "R": 4.0,
            "P": 2.0,
            "S": 0.0,
        },
    }
    path = tmp_path / "stag_hunt_payoff_threshold.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_toy2_config(path)
    kernel = DecisionKernel.from_config(loaded)

    assert loaded.policy.decision.calibration.mode == "payoff_threshold"
    assert loaded.policy.decision.calibration.strength == pytest.approx(2.0)
    assert kernel.decision_threshold == pytest.approx(2.0 / 3.0)


def test_toy2_decision_calibration_rejects_unknown_mode(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["policy"]["decision"]["calibration"] = {
        "mode": "unknown",
        "strength": 4.0,
    }
    path = tmp_path / "invalid_calibration_mode.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


@pytest.mark.parametrize("strength", [0.0, -1.0])
def test_toy2_decision_calibration_rejects_nonpositive_strength(
    tmp_path: Path,
    strength: float,
) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        calibration_mode="none",
        calibration_strength=strength,
    )
    path = tmp_path / f"invalid_calibration_strength_{strength}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_toy2_payoff_threshold_calibration_requires_interior_threshold(
    tmp_path: Path,
) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        calibration_mode="payoff_threshold",
    )
    path = tmp_path / "pd_no_interior_threshold.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_toy2_argmax_decision_canonicalizes_inactive_fields(tmp_path: Path) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        decision_mode="argmax",
        action_temperature=0.25,
        exploration_epsilon=1.0,
        calibration_mode="payoff_threshold",
        calibration_strength=2.0,
    )
    path = tmp_path / "argmax_canonical_decision.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    decision = load_toy2_config(path).policy.decision

    assert decision.mode == "argmax"
    assert decision.action_temperature == pytest.approx(1.0)
    assert decision.exploration_epsilon == pytest.approx(0.0)
    assert decision.terminal_argmax_epochs == 0
    assert decision.calibration.mode == "none"
    assert decision.calibration.strength == pytest.approx(4.0)


def test_toy2_argmax_metadata_records_inactive_calibration(tmp_path: Path) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        decision_mode="argmax",
        action_temperature=0.25,
        exploration_epsilon=1.0,
        calibration_mode="payoff_threshold",
        calibration_strength=2.0,
        learning_enabled=False,
        epochs=1,
    )

    result = run_toy2(config=load_toy2_config(config_path), config_path=config_path)

    run_metadata = yaml.safe_load((result.run_dir / "metadata.json").read_text())
    assert run_metadata["decision_mode"] == "argmax"
    assert run_metadata["action_temperature"] == pytest.approx(1.0)
    assert run_metadata["exploration_epsilon"] == pytest.approx(0.0)
    assert run_metadata["terminal_argmax_epochs"] == 0
    assert run_metadata["decision_calibration_mode"] == "none"
    assert run_metadata["decision_calibration_strength"] == pytest.approx(4.0)
    assert run_metadata["decision_threshold"] is None


def test_toy2_rejects_removed_flat_decision_fields(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["policy"]["action_selection_mode"] = "argmax"
    config["policy"]["exploration_epsilon"] = 0.1
    path = tmp_path / "removed_flat_decision_fields.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_toy2_revision_rate_explicit_value(tmp_path: Path) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        revision_rate=0.25,
    )
    path = tmp_path / "revision_rate.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert load_toy2_config(path).policy.revision_rate == 0.25


def test_well_mixed_neural_context_samples_same_degree_without_self(
    tmp_path: Path,
) -> None:
    config = load_toy2_config(
        write_tiny_config(
            tmp_path,
            mixer="none",
            peer_rule="none",
            neural_peer_mode="well_mixed",
        )
    )
    neighbors = [
        [1, 3],
        [0, 2],
        [1, 3],
        [0, 2],
    ]
    rng = np.random.default_rng(42)

    peer_ids = neural_context_peer_ids(
        config=config,
        neighbors=neighbors,
        agent_count=4,
        rng=rng,
    )

    assert [len(peers) for peers in peer_ids] == [2, 2, 2, 2]
    assert all(agent_id not in peers for agent_id, peers in enumerate(peer_ids))
    assert all(len(set(peers)) == len(peers) for peers in peer_ids)


def test_well_mixed_interaction_samples_same_degree_without_self(
    tmp_path: Path,
) -> None:
    config = load_toy2_config(
        write_tiny_config(
            tmp_path,
            mixer="none",
            peer_rule="none",
            interaction_mode="well_mixed_resampled",
        )
    )
    neighbors = [
        [1, 3],
        [0, 2],
        [1, 3],
        [0, 2],
    ]
    rng = np.random.default_rng(42)

    peer_ids = interaction_peer_ids(
        config=config,
        neighbors=neighbors,
        agent_count=4,
        rng=rng,
    )

    assert [len(peers) for peers in peer_ids] == [2, 2, 2, 2]
    assert all(agent_id not in peers for agent_id, peers in enumerate(peer_ids))
    assert all(len(set(peers)) == len(peers) for peers in peer_ids)


def test_compute_payoffs_from_peer_ids_matches_stag_hunt_pair_values(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    raw["game"] = {
        "family": "stag_hunt",
        "payoff": {
            "T": 3.0,
            "R": 4.0,
            "P": 2.0,
            "S": 0.0,
        },
    }
    path = tmp_path / "stag_hunt_payoff_peer_ids.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_toy2_config(path)
    actions = np.array([1, 0], dtype=np.int64)

    payoffs = compute_payoffs_from_peer_ids(
        actions=actions,
        peer_ids=[[1], [0]],
        config=config,
    )

    assert payoffs.tolist() == [0.0, 3.0]


def test_update_reputation_ema_tracks_cooperation_actions() -> None:
    reputation = np.array([0.2, 0.8], dtype=np.float64)
    actions = np.array([1, 0], dtype=np.int64)

    update_reputation_ema(reputation=reputation, actions=actions, decay=0.75)

    assert reputation.tolist() == pytest.approx([0.4, 0.6])


def test_torch_ema_updates_match_numpy() -> None:
    payoff_ema = np.array([2.0, 6.0], dtype=np.float64)
    previous_payoff_ema = np.array([1.0, 5.0], dtype=np.float64)
    payoffs = np.array([10.0, 20.0], dtype=np.float64)
    torch_payoff_ema = torch.as_tensor(payoff_ema.copy(), dtype=torch.float64)
    torch_previous_payoff_ema = torch.as_tensor(
        previous_payoff_ema.copy(),
        dtype=torch.float64,
    )
    torch_payoffs = torch.as_tensor(payoffs, dtype=torch.float64)

    update_payoff_ema(
        payoff_ema=payoff_ema,
        previous_payoff_ema=previous_payoff_ema,
        payoffs=payoffs,
        decay=0.25,
    )
    update_payoff_ema(
        payoff_ema=torch_payoff_ema,
        previous_payoff_ema=torch_previous_payoff_ema,
        payoffs=torch_payoffs,
        decay=0.25,
    )

    np.testing.assert_allclose(torch_payoff_ema.numpy(), payoff_ema)
    np.testing.assert_allclose(torch_previous_payoff_ema.numpy(), previous_payoff_ema)

    reputation = np.array([0.2, 0.8], dtype=np.float64)
    actions = np.array([1, 0], dtype=np.int64)
    torch_reputation = torch.as_tensor(reputation.copy(), dtype=torch.float64)
    torch_actions = torch.as_tensor(actions, dtype=torch.long)

    update_reputation_ema(reputation=reputation, actions=actions, decay=0.75)
    update_reputation_ema(
        reputation=torch_reputation,
        actions=torch_actions,
        decay=0.75,
    )

    np.testing.assert_allclose(torch_reputation.numpy(), reputation)


def test_deterministic_reputation_imitation_follows_highest_reputation_peer(
    tmp_path: Path,
) -> None:
    config = load_toy2_config(
        write_tiny_config(
            tmp_path,
            mixer="none",
            peer_rule="none",
            update_rule="reputation_imitation",
        )
    )
    actions = np.array([0, 1, 0], dtype=np.int64)
    reputation = np.array([0.2, 0.9, 0.1], dtype=np.float64)
    neighbors = [[1, 2], [0, 2], [1]]

    cooperation_probs = reputation_imitation_cooperation_probs(
        actions=actions,
        reputation=reputation,
        neighbors=neighbors,
        revision_mask=np.ones(3, dtype=bool),
        rng=np.random.default_rng(3),
        config=config,
    )

    assert cooperation_probs.tolist() == [1.0, 0.0, 1.0]


def test_noisy_reputation_imitation_is_seed_reproducible(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule="reputation_imitation",
    )
    raw["state"]["reputation"] = {
        "enabled": True,
        "decay": 0.9,
        "peer_rule": "spatial",
        "temperature": 0.7,
        "noise": 0.4,
    }
    path = tmp_path / "noisy_reputation.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_toy2_config(path)
    actions = np.array([0, 1, 0, 1], dtype=np.int64)
    reputation = np.array([0.2, 0.9, 0.5, 0.1], dtype=np.float64)
    neighbors = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]

    first = reputation_imitation_cooperation_probs(
        actions=actions,
        reputation=reputation,
        neighbors=neighbors,
        revision_mask=np.ones(4, dtype=bool),
        rng=np.random.default_rng(11),
        config=config,
    )
    second = reputation_imitation_cooperation_probs(
        actions=actions,
        reputation=reputation,
        neighbors=neighbors,
        revision_mask=np.ones(4, dtype=bool),
        rng=np.random.default_rng(11),
        config=config,
    )

    assert first.tolist() == pytest.approx(second.tolist())
    assert np.all((0.0 <= first) & (first <= 1.0))


def test_mobility_swaps_cell_state_together(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    raw["state"]["mobility"] = {
        "enabled": True,
        "rate": 1.0,
        "candidate_pool_size": 1,
        "selection_rule": "local_quality",
        "move_cost": 0.0,
    }
    path = tmp_path / "mobility_swap.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_toy2_config(path)
    actions = np.array([0, 1], dtype=np.int64)
    payoff_ema = np.array([0.0, 10.0], dtype=np.float64)
    previous_payoff_ema = np.array([-1.0, 9.0], dtype=np.float64)
    reputation = np.array([0.2, 0.8], dtype=np.float64)

    state = BinarySpatialState(
        actions=actions,
        payoffs=np.zeros(2, dtype=np.float64),
        payoff_ema=payoff_ema,
        previous_payoff_ema=previous_payoff_ema,
        reputation=reputation,
    )

    result = apply_mobility_swaps(
        state=state,
        neighbors=[[], []],
        rng=np.random.default_rng(19),
        params=mobility_params_from_config(config),
    )

    assert result.moved.tolist() == [True, False]
    assert result.targets.tolist() == [1, -1]
    assert result.gains[0] == pytest.approx(10.0)
    assert actions.tolist() == [1, 0]
    assert payoff_ema.tolist() == [10.0, 0.0]
    assert previous_payoff_ema.tolist() == [9.0, -1.0]
    assert reputation.tolist() == [0.8, 0.2]


def test_torch_mobility_swaps_copy_back_state_and_agents(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    raw["state"]["mobility"] = {
        "enabled": True,
        "rate": 1.0,
        "candidate_pool_size": 1,
        "selection_rule": "local_quality",
        "move_cost": 0.0,
    }
    path = tmp_path / "torch_mobility_swap.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_toy2_config(path)
    agents = ["left", "right"]
    state = BinarySpatialState(
        actions=torch.tensor([0, 1], dtype=torch.long),
        payoffs=torch.zeros(2, dtype=torch.float64),
        payoff_ema=torch.tensor([0.0, 10.0], dtype=torch.float64),
        previous_payoff_ema=torch.tensor([-1.0, 9.0], dtype=torch.float64),
        reputation=torch.tensor([0.2, 0.8], dtype=torch.float64),
        agents=agents,
    )

    result = apply_mobility_swaps(
        state=state,
        neighbors=[[], []],
        rng=np.random.default_rng(19),
        params=mobility_params_from_config(config),
    )

    assert result.moved.tolist() == [True, False]
    assert state.actions.tolist() == [1, 0]
    assert state.payoff_ema.tolist() == [10.0, 0.0]
    assert state.previous_payoff_ema.tolist() == [9.0, -1.0]
    assert state.reputation.tolist() == pytest.approx([0.8, 0.2])
    assert agents == ["right", "left"]


def test_toy2_policy_prior_defaults_to_none(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    path = tmp_path / "policy_prior_default.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert load_toy2_config(path).agents.policy_prior_action_probability is None


@pytest.mark.parametrize("policy_prior", [0.0, 0.5, 1.0])
def test_toy2_policy_prior_accepts_unit_interval(
    tmp_path: Path,
    policy_prior: float,
) -> None:
    config = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        policy_prior_action_probability=policy_prior,
    )
    path = tmp_path / f"policy_prior_{policy_prior}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_toy2_config(path)

    assert loaded.agents.policy_prior_action_probability == policy_prior


@pytest.mark.parametrize("policy_prior", [-0.01, 1.01])
def test_toy2_policy_prior_rejects_outside_unit_interval(
    tmp_path: Path,
    policy_prior: float,
) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["agents"]["policy_prior_action_probability"] = policy_prior
    path = tmp_path / f"invalid_policy_prior_{policy_prior}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_toy2_rejects_invalid_pd_payoffs(tmp_path: Path) -> None:
    config = tiny_config_dict(tmp_path, mixer="none", peer_rule="none")
    config["game"]["payoff"] = {
        "T": 2.0,
        "R": 3.0,
        "P": 1.0,
        "S": 0.0,
    }
    path = tmp_path / "invalid_pd.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_toy2_config(path)


def test_output_average_probability_channel_preserves_shape_and_sum() -> None:
    cooperation_probs = np.array([0.1, 0.9, 0.4], dtype=np.float64)
    peer_ids = [[1, 2], [0, 2], [0, 1]]

    mixed, losses = apply_output_average_to_cooperation_probs(
        cooperation_probs=cooperation_probs,
        peer_ids=peer_ids,
        alpha=0.25,
    )
    policy_probs = cooperation_probs_to_policy_tensor(mixed, device=torch.device("cpu"))

    assert policy_probs.shape == (3, 2)
    assert torch.allclose(policy_probs.sum(dim=1), torch.ones(3))
    assert len(losses) == 3
    assert np.all((0.0 <= mixed) & (mixed <= 1.0))


def test_output_average_probability_channel_alpha_zero_is_noop() -> None:
    cooperation_probs = np.array([0.1, 0.9, 0.4], dtype=np.float64)
    peer_ids = [[1, 2], [0, 2], [0, 1]]

    mixed, losses = apply_output_average_to_cooperation_probs(
        cooperation_probs=cooperation_probs,
        peer_ids=peer_ids,
        alpha=0.0,
    )

    assert mixed is not cooperation_probs
    assert np.array_equal(mixed, cooperation_probs)
    assert losses == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "update_rule",
    ["neural_policy", "fermi_imitation", "reputation_imitation"],
)
def test_output_average_alpha_zero_matches_no_mixer(
    tmp_path: Path,
    update_rule: str,
) -> None:
    none_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule=update_rule,
    )
    output_path = write_tiny_config(
        tmp_path,
        mixer="output_average",
        peer_rule="none",
        update_rule=update_rule,
        alpha=0.0,
    )

    none_result = run_toy2(config=load_toy2_config(none_path), config_path=none_path)
    output_result = run_toy2(
        config=load_toy2_config(output_path),
        config_path=output_path,
    )

    assert output_result.final_action_rate == pytest.approx(
        none_result.final_action_rate
    )
    assert output_result.final_mean_payoff == pytest.approx(
        none_result.final_mean_payoff
    )
    assert output_result.final_mean_policy_action_probability == pytest.approx(
        none_result.final_mean_policy_action_probability
    )


def test_toy2_revision_rate_zero_keeps_actions_and_local_losses_zero(
    tmp_path: Path,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        revision_rate=0.0,
        epochs=4,
        learning_enabled=True,
    )
    result = run_toy2(config=load_toy2_config(config_path), config_path=config_path)

    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_rows = list(csv.DictReader(handle))
    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_rows = list(csv.DictReader(handle))

    actions_by_agent: dict[int, list[int]] = {}
    for row in micro_rows:
        actions_by_agent.setdefault(int(row["agent_id"]), []).append(int(row["action"]))
    assert actions_by_agent
    assert all(len(set(actions)) == 1 for actions in actions_by_agent.values())
    assert all(row["revised"] == "False" for row in micro_rows)
    assert all(float(row["local_loss"]) == 0.0 for row in micro_rows)
    assert all(float(row["realized_revision_rate"]) == 0.0 for row in aggregate_rows)


def test_toy2_revision_rate_one_marks_all_agents_revised(tmp_path: Path) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        revision_rate=1.0,
        epochs=2,
        learning_enabled=False,
    )
    result = run_toy2(config=load_toy2_config(config_path), config_path=config_path)

    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_rows = list(csv.DictReader(handle))
    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_rows = list(csv.DictReader(handle))

    assert micro_rows
    assert all(row["revised"] == "True" for row in micro_rows)
    realized_after_initial = [
        float(row["realized_revision_rate"])
        for row in aggregate_rows
        if int(row["epoch"]) > 0
    ]
    assert realized_after_initial == [1.0, 1.0]


def test_decision_kernel_argmax_chooses_highest_probability_actions() -> None:
    current_actions = np.array([0, 1, 0, 1], dtype=np.int64)
    policy_probs = torch.tensor(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.4, 0.6],
            [0.7, 0.3],
        ],
        dtype=torch.float32,
    )
    revision_mask = np.array([True, False, True, False])

    actions = DecisionKernel(mode="argmax").select(
        current_actions=current_actions,
        policy_probs=policy_probs,
        revision_mask=revision_mask,
    )

    assert actions.tolist() == [1, 1, 1, 1]


def test_decision_kernel_sampled_temperature_one_uses_existing_sampling_path() -> None:
    policy_probs = torch.tensor(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.4, 0.6],
        ],
        dtype=torch.float32,
    )
    revision_mask = np.ones(3, dtype=bool)

    torch.manual_seed(123)
    expected = sample_actions(policy_probs, exploration_epsilon=0.0)
    torch.manual_seed(123)
    actions = DecisionKernel(
        mode="sampled",
        action_temperature=1.0,
        exploration_epsilon=0.0,
    ).select(
        current_actions=np.zeros(3, dtype=np.int64),
        policy_probs=policy_probs,
        revision_mask=revision_mask,
    )

    assert actions.tolist() == expected.tolist()


def test_decision_kernel_uses_argmax_only_in_terminal_window() -> None:
    kernel = DecisionKernel(mode="sampled", terminal_argmax_epochs=2)

    assert kernel.for_epoch(epoch=8, total_epochs=10).mode == "sampled"
    assert kernel.for_epoch(epoch=9, total_epochs=10).mode == "argmax"
    assert kernel.for_epoch(epoch=10, total_epochs=10).mode == "argmax"


def test_decision_kernel_select_tensor_matches_existing_selection_path() -> None:
    current_actions = np.array([0, 1, 0, 1], dtype=np.int64)
    policy_probs = torch.tensor(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.4, 0.6],
            [0.7, 0.3],
        ],
        dtype=torch.float32,
    )
    revision_mask = np.array([True, False, True, False])

    numpy_actions = DecisionKernel(mode="argmax").select(
        current_actions=current_actions,
        policy_probs=policy_probs,
        revision_mask=revision_mask,
    )
    tensor_actions = DecisionKernel(mode="argmax").select_tensor(
        current_actions=torch.as_tensor(current_actions, dtype=torch.long),
        policy_probs=policy_probs,
        revision_mask=revision_mask,
    )

    assert tensor_actions.dtype == torch.long
    assert tensor_actions.tolist() == numpy_actions.tolist()

    torch.manual_seed(1234)
    sampled_numpy_actions = DecisionKernel(mode="sampled").select(
        current_actions=current_actions,
        policy_probs=policy_probs,
        revision_mask=revision_mask,
    )
    torch.manual_seed(1234)
    sampled_tensor_actions = DecisionKernel(mode="sampled").select_tensor(
        current_actions=torch.as_tensor(current_actions, dtype=torch.long),
        policy_probs=policy_probs,
        revision_mask=revision_mask,
    )

    assert sampled_tensor_actions.tolist() == sampled_numpy_actions.tolist()


def test_decision_kernel_sampled_action_temperature_sharpens_probs() -> None:
    policy_probs = torch.tensor(
        [
            [0.2, 0.8],
            [0.7, 0.3],
        ],
        dtype=torch.float32,
    )

    sharpened = apply_action_temperature(policy_probs, action_temperature=0.5)

    assert sharpened[0, 1] > policy_probs[0, 1]
    assert sharpened[1, 0] > policy_probs[1, 0]
    assert torch.allclose(sharpened.sum(dim=1), torch.ones(2))


def test_payoff_threshold_calibration_centers_stag_hunt_threshold() -> None:
    threshold = 2.0 / 3.0
    policy_probs = torch.tensor(
        [
            [1.0 - threshold, threshold],
            [0.4, 0.6],
            [0.3, 0.7],
        ],
        dtype=torch.float32,
    )

    calibrated = apply_payoff_threshold_calibration(
        policy_probs=policy_probs,
        action_temperature=1.0,
        calibration_strength=4.0,
        decision_threshold=threshold,
    )

    assert calibrated[0, 1] == pytest.approx(0.5)
    assert calibrated[1, 1] < 0.5
    assert calibrated[2, 1] > 0.5
    assert torch.allclose(calibrated.sum(dim=1), torch.ones(3))


def test_payoff_threshold_calibration_strength_sharpens_threshold_gap() -> None:
    threshold = 2.0 / 3.0
    policy_probs = torch.tensor(
        [
            [0.4, 0.6],
            [0.3, 0.7],
        ],
        dtype=torch.float32,
    )

    weak = apply_payoff_threshold_calibration(
        policy_probs=policy_probs,
        action_temperature=1.0,
        calibration_strength=1.0,
        decision_threshold=threshold,
    )
    strong = apply_payoff_threshold_calibration(
        policy_probs=policy_probs,
        action_temperature=1.0,
        calibration_strength=6.0,
        decision_threshold=threshold,
    )

    assert strong[0, 1] < weak[0, 1]
    assert strong[1, 1] > weak[1, 1]


def test_decision_kernel_no_revision_mask_keeps_actions() -> None:
    current_actions = np.array([0, 1, 0], dtype=np.int64)
    policy_probs = torch.tensor(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=torch.float32,
    )

    actions = DecisionKernel(mode="argmax").select(
        current_actions=current_actions,
        policy_probs=policy_probs,
        revision_mask=np.zeros(3, dtype=bool),
    )

    assert actions.tolist() == current_actions.tolist()


def test_decision_kernel_payoff_threshold_respects_revision_mask() -> None:
    current_actions = np.array([0, 1, 0], dtype=np.int64)
    policy_probs = torch.tensor(
        [
            [0.3, 0.7],
            [0.3, 0.7],
            [0.9, 0.1],
        ],
        dtype=torch.float32,
    )

    torch.manual_seed(42)
    actions = DecisionKernel(
        mode="sampled",
        action_temperature=1.0,
        exploration_epsilon=0.0,
        calibration_mode="payoff_threshold",
        calibration_strength=6.0,
        decision_threshold=2.0 / 3.0,
    ).select(
        current_actions=current_actions,
        policy_probs=policy_probs,
        revision_mask=np.array([True, False, True]),
    )

    assert actions[1] == current_actions[1]


@pytest.mark.parametrize(
    "update_rule",
    ["fermi_imitation", "reputation_imitation", "rd_well_mixed"],
)
def test_non_neural_update_rules_ignore_decision_config(
    tmp_path: Path,
    update_rule: str,
) -> None:
    sampled = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule=update_rule,
        revision_rate=0.5,
        epochs=4,
        seed=17,
    )
    argmax = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule=update_rule,
        decision_mode="argmax",
        action_temperature=0.25,
        exploration_epsilon=1.0,
        calibration_mode="payoff_threshold",
        calibration_strength=2.0,
        revision_rate=0.5,
        epochs=4,
        seed=17,
    )
    sampled_path = tmp_path / f"{update_rule}_sampled.yaml"
    argmax_path = tmp_path / f"{update_rule}_argmax.yaml"
    sampled_path.write_text(yaml.safe_dump(sampled), encoding="utf-8")
    argmax_path.write_text(yaml.safe_dump(argmax), encoding="utf-8")

    sampled_result = run_toy2(
        config=load_toy2_config(sampled_path),
        config_path=sampled_path,
    )
    argmax_result = run_toy2(
        config=load_toy2_config(argmax_path),
        config_path=argmax_path,
    )

    assert argmax_result.final_action_rate == pytest.approx(
        sampled_result.final_action_rate
    )
    assert argmax_result.final_mean_payoff == pytest.approx(
        sampled_result.final_mean_payoff
    )
    assert argmax_result.final_mean_policy_action_probability == pytest.approx(
        sampled_result.final_mean_policy_action_probability
    )

    metadata = yaml.safe_load((argmax_result.run_dir / "config.yaml").read_text())
    assert metadata["model"]["policy"]["decision"]["mode"] == "argmax"

    resolved_config = yaml.safe_load(
        (argmax_result.run_dir / "resolved_config.yaml").read_text()
    )
    assert resolved_config["model"]["policy"]["decision"]["mode"] == "sampled"
    assert resolved_config["model"]["policy"]["decision"]["action_temperature"] == (
        pytest.approx(1.0)
    )
    assert resolved_config["model"]["policy"]["decision"]["exploration_epsilon"] == (
        pytest.approx(0.0)
    )
    assert resolved_config["model"]["policy"]["decision"]["calibration"]["mode"] == "none"
    assert resolved_config["model"]["policy"]["decision"]["calibration"]["strength"] == (
        pytest.approx(4.0)
    )

    run_metadata = yaml.safe_load((argmax_result.run_dir / "metadata.json").read_text())
    assert run_metadata["decision_mode"] == "sampled"
    assert run_metadata["action_temperature"] == pytest.approx(1.0)
    assert run_metadata["exploration_epsilon"] == pytest.approx(0.0)
    assert run_metadata["decision_calibration_mode"] == "none"
    assert run_metadata["decision_calibration_strength"] == pytest.approx(4.0)
    assert run_metadata["decision_threshold"] is None


def test_toy2_partial_revision_rate_realizes_configured_average(
    tmp_path: Path,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule="fermi_imitation",
        revision_rate=0.1,
        epochs=50,
        grid_width=10,
        grid_height=10,
    )
    result = run_toy2(config=load_toy2_config(config_path), config_path=config_path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["epoch"]) > 0]

    realized = [float(row["realized_revision_rate"]) for row in rows]
    assert abs(float(np.mean(realized)) - 0.1) <= 0.05


@pytest.mark.parametrize("p0", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_toy2_epoch_zero_preserves_sampled_initial_condition(
    tmp_path: Path,
    p0: float,
    seed: int,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule="fermi_imitation",
        revision_rate=0.1,
        epochs=1,
        grid_width=10,
        grid_height=10,
        initial_action_probability=p0,
        seed=seed,
    )
    result = run_toy2(config=load_toy2_config(config_path), config_path=config_path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        initial_row = next(csv.DictReader(handle))

    assert int(initial_row["epoch"]) == 0
    assert abs(float(initial_row["action_rate"]) - p0) <= 0.12


def test_toy2_policy_prior_initializes_epoch_zero_policy(
    tmp_path: Path,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        learning_enabled=False,
        epochs=1,
        policy_prior_action_probability=0.8,
    )
    result = run_toy2(config=load_toy2_config(config_path), config_path=config_path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        initial_row = next(csv.DictReader(handle))

    assert int(initial_row["epoch"]) == 0
    assert float(initial_row["mean_policy_action_probability"]) == pytest.approx(0.8)
    assert float(initial_row["mean_policy_action_probability_pre_revision"]) == (
        pytest.approx(0.8)
    )
    assert float(initial_row["mean_policy_action_probability_post_local"]) == (
        pytest.approx(0.8)
    )
    assert float(initial_row["mean_policy_action_probability_post_social"]) == (
        pytest.approx(0.8)
    )
    assert float(initial_row["mean_local_loss"]) == 0.0
    assert float(initial_row["mean_revised_local_loss"]) == 0.0
    assert float(initial_row["mean_social_loss"]) == 0.0


def test_policy_temperature_changes_logged_policy_readout(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        learning_enabled=False,
        epochs=1,
        policy_prior_action_probability=0.8,
    )
    raw["policy"]["temperature"] = 0.5
    path = tmp_path / "policy_temperature_readout.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(path), config_path=path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        initial_row = next(csv.DictReader(handle))
    expected = (0.8**2) / ((0.2**2) + (0.8**2))
    assert float(initial_row["mean_policy_action_probability"]) == pytest.approx(expected)


def test_policy_temperature_does_not_change_realized_actions(
    tmp_path: Path,
) -> None:
    base = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        learning_enabled=False,
        revision_rate=1.0,
        epochs=4,
        grid_width=10,
        grid_height=10,
        policy_prior_action_probability=0.8,
        seed=31,
    )
    cool = base.clone()
    cool["policy"]["temperature"] = 0.5
    warm = base.clone()
    warm["policy"]["temperature"] = 2.0
    cool_path = tmp_path / "cool_policy_readout.yaml"
    warm_path = tmp_path / "warm_policy_readout.yaml"
    cool_path.write_text(yaml.safe_dump(cool), encoding="utf-8")
    warm_path.write_text(yaml.safe_dump(warm), encoding="utf-8")

    cool_result = run_toy2(config=load_toy2_config(cool_path), config_path=cool_path)
    warm_result = run_toy2(config=load_toy2_config(warm_path), config_path=warm_path)

    assert warm_result.final_action_rate == pytest.approx(
        cool_result.final_action_rate
    )
    assert warm_result.final_mean_payoff == pytest.approx(cool_result.final_mean_payoff)
    assert warm_result.final_mean_policy_action_probability != pytest.approx(
        cool_result.final_mean_policy_action_probability
    )


def test_micro_log_separates_policy_and_decision_probabilities(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        learning_enabled=False,
        revision_rate=1.0,
        epochs=1,
        policy_prior_action_probability=0.8,
    )
    raw["policy"]["temperature"] = 0.5
    raw["policy"]["decision"]["action_temperature"] = 1.0
    path = tmp_path / "policy_decision_micro_log.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(path), config_path=path)

    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        row = next(csv.DictReader(handle))

    expected_policy_readout = (0.8**2) / ((0.2**2) + (0.8**2))
    assert "policy_action_probability_pre_revision" in row
    assert "policy_action_probability_post_local" in row
    assert "policy_action_probability_post_social" in row
    assert "candidate_decision_action_probability_pre_revision" in row
    assert "realized_decision_action_probability" in row
    assert float(row["action_probability"]) == pytest.approx(
        expected_policy_readout
    )
    assert float(row["policy_action_probability_pre_revision"]) == pytest.approx(
        expected_policy_readout
    )
    assert float(row["policy_action_probability_post_local"]) == pytest.approx(
        expected_policy_readout
    )
    assert float(row["policy_action_probability_post_social"]) == pytest.approx(
        expected_policy_readout
    )
    assert float(
        row["candidate_decision_action_probability_pre_revision"]
    ) == pytest.approx(0.8)
    assert float(row["realized_decision_action_probability"]) == pytest.approx(0.8)
    assert row["revision_operator_enabled"] == "False"
    assert row["revision_operator_source"] == ""
    assert row["revision_choice"] == ""


def test_revision_operator_adapter_logs_toy2_diagnostics(tmp_path: Path) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        learning_enabled=False,
        revision_rate=1.0,
        epochs=1,
        decision_mode="argmax",
        initial_action_probability=0.0,
        policy_prior_action_probability=0.8,
    )
    raw["coordination"]["revision_operator_enabled"] = True
    raw["coordination"]["revision_operator_source"] = "policy_probability"
    path = tmp_path / "toy2_revision_operator.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(path), config_path=path)
    aggregate = final_aggregate_row(result.run_dir)
    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_reader = csv.DictReader(handle)
        micro_row = next(micro_reader)

    assert aggregate["revision_operator_enabled"] == "True"
    assert aggregate["revision_operator_source"] == "policy_probability"
    assert float(aggregate["mean_revision_switch_probability"]) == pytest.approx(1.0)
    assert float(aggregate["revision_choice_switch_to_one_rate"]) == pytest.approx(1.0)
    for field in [
        "revision_operator_enabled",
        "revision_operator_source",
        "revision_choice",
        "revision_switch_probability",
    ]:
        assert field in micro_reader.fieldnames
    assert micro_row["revision_operator_enabled"] == "True"
    assert micro_row["revision_operator_source"] == "policy_probability"
    assert micro_row["revision_choice"] == "switch_to_1"
    assert float(micro_row["revision_switch_probability"]) == pytest.approx(1.0)


def test_micro_log_records_payoff_threshold_calibrated_candidate_probability(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        learning_enabled=False,
        revision_rate=1.0,
        epochs=1,
        policy_prior_action_probability=2.0 / 3.0,
        calibration_mode="payoff_threshold",
    )
    raw["game"] = {
        "family": "stag_hunt",
        "payoff": {
            "T": 3.0,
            "R": 4.0,
            "P": 2.0,
            "S": 0.0,
        },
    }
    path = tmp_path / "payoff_threshold_micro_log.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(path), config_path=path)

    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        row = next(csv.DictReader(handle))

    assert float(
        row["policy_action_probability_pre_revision"]
    ) == pytest.approx(2.0 / 3.0)
    assert float(
        row["candidate_decision_action_probability_pre_revision"]
    ) == pytest.approx(0.5)
    assert float(row["realized_decision_action_probability"]) == pytest.approx(0.5)


def test_micro_log_blanks_realized_decision_probability_for_unrevised_agents(
    tmp_path: Path,
) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        learning_enabled=False,
        revision_rate=0.0,
        epochs=1,
        policy_prior_action_probability=0.8,
    )

    result = run_toy2(config=load_toy2_config(config_path), config_path=config_path)

    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        row = next(csv.DictReader(handle))

    assert row["revised"] == "False"
    assert float(
        row["candidate_decision_action_probability_pre_revision"]
    ) == pytest.approx(0.8)
    assert row["realized_decision_action_probability"] == ""


@pytest.mark.parametrize(
    ("peer_actions", "expected_direction"),
    [
        (np.array([0, 0, 0, 0], dtype=np.int64), "down"),
        (np.array([1, 1, 1, 1], dtype=np.int64), "up"),
    ],
)
def test_counterfactual_policy_update_follows_stag_hunt_advantage(
    tmp_path: Path,
    peer_actions: np.ndarray,
    expected_direction: str,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        policy_prior_action_probability=0.5,
    )
    raw["game"] = {
        "family": "stag_hunt",
        "payoff": {
            "T": 3.0,
            "R": 4.0,
            "P": 2.0,
            "S": 0.0,
        },
    }
    raw["environment"]["payoff_T"] = 3.0
    raw["environment"]["payoff_R"] = 4.0
    raw["environment"]["payoff_P"] = 2.0
    raw["environment"]["payoff_S"] = 0.0
    raw["environment"]["entropy_beta"] = 0.0
    raw["agents"]["optimizer"]["learning_rate"] = 0.1
    path = tmp_path / f"counterfactual_{expected_direction}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_toy2_config(path)
    observation = torch.zeros(config.agents.model.input_dim)
    agent = create_agents(config=config, device=torch.device("cpu"))[0]
    before = float(agent.act_or_predict(observation)[1])

    for _ in range(10):
        train_counterfactual_policy(
            agent=agent,
            observation=observation,
            peer_actions=peer_actions,
            config=config,
        )

    after = float(agent.act_or_predict(observation)[1])
    if expected_direction == "down":
        assert after < before
    else:
        assert after > before


def test_toy2_aggregate_log_includes_generic_peer_fields(tmp_path: Path) -> None:
    config_path = write_tiny_config(tmp_path, mixer="output_average", peer_rule="none")
    result = run_toy2(config=load_toy2_config(config_path), config_path=config_path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        row = next(reader)

    assert "fragmentation_components" in reader.fieldnames
    assert "mean_peer_count" in reader.fieldnames
    assert "edge_entropy" in reader.fieldnames
    assert "mean_policy_action_probability_pre_revision" in reader.fieldnames
    assert "mean_policy_action_probability_post_local" in reader.fieldnames
    assert "mean_policy_action_probability_post_social" in reader.fieldnames
    assert "policy_action_probability_post_social_gt_0p7_rate" in reader.fieldnames
    assert (
        "policy_action_probability_post_social_dwell_0p4_0p6_rate"
        in reader.fieldnames
    )
    assert "policy_probability_threshold_crossings_0p5_count" in reader.fieldnames
    assert "action_flip_rate" in reader.fieldnames
    assert "mean_local_loss" in reader.fieldnames
    assert "mean_revised_local_loss" in reader.fieldnames
    assert "mean_social_loss" in reader.fieldnames
    assert int(row["fragmentation_components"]) >= 0
    assert float(row["mean_peer_count"]) >= 0.0


def test_toy2_reputation_mobility_runner_logs_extended_fields(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule="reputation_imitation",
        epochs=2,
    )
    raw["state"]["mobility"] = {
        "enabled": True,
        "rate": 1.0,
        "candidate_pool_size": 4,
        "selection_rule": "local_quality",
        "move_cost": 0.0,
    }
    path = tmp_path / "reputation_mobility_smoke.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = run_toy2(config=load_toy2_config(path), config_path=path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_reader = csv.DictReader(handle)
        aggregate_rows = list(aggregate_reader)
    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_reader = csv.DictReader(handle)
        micro_row = next(micro_reader)

    assert "mean_reputation" in aggregate_reader.fieldnames
    assert "reputation_dispersion" in aggregate_reader.fieldnames
    assert "mobility_rate" in aggregate_reader.fieldnames
    assert "mean_mobility_gain" in aggregate_reader.fieldnames
    assert "reputation" in micro_reader.fieldnames
    assert "mobility_moved" in micro_reader.fieldnames
    assert "mobility_target" in micro_reader.fieldnames
    assert "mobility_gain" in micro_reader.fieldnames
    assert aggregate_rows
    assert all(0.0 <= float(row["mean_reputation"]) <= 1.0 for row in aggregate_rows)
    assert 0.0 <= float(micro_row["reputation"]) <= 1.0


@pytest.mark.parametrize("update_rule", ["fermi_imitation", "rd_well_mixed"])
def test_toy2_policy_prior_does_not_affect_non_neural_paths(
    tmp_path: Path,
    update_rule: str,
) -> None:
    base_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule=update_rule,
        epochs=4,
        grid_width=6,
        grid_height=6,
        initial_action_probability=0.7,
        seed=5,
    )
    base_result = run_toy2(config=load_toy2_config(base_path), config_path=base_path)
    prior_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule=update_rule,
        epochs=4,
        grid_width=6,
        grid_height=6,
        initial_action_probability=0.7,
        policy_prior_action_probability=0.2,
        seed=5,
    )
    prior_result = run_toy2(
        config=load_toy2_config(prior_path),
        config_path=prior_path,
    )

    assert prior_result.final_action_rate == pytest.approx(
        base_result.final_action_rate
    )
    assert prior_result.final_mean_payoff == pytest.approx(
        base_result.final_mean_payoff
    )
    assert prior_result.final_mean_policy_action_probability == pytest.approx(
        base_result.final_mean_policy_action_probability
    )


@pytest.mark.parametrize("update_rule", ["fermi_imitation", "rd_well_mixed"])
def test_toy2_neural_peer_mode_does_not_affect_non_neural_paths(
    tmp_path: Path,
    update_rule: str,
) -> None:
    base_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule=update_rule,
        epochs=4,
        grid_width=6,
        grid_height=6,
        initial_action_probability=0.7,
        seed=5,
    )
    base_result = run_toy2(config=load_toy2_config(base_path), config_path=base_path)
    well_mixed_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule=update_rule,
        epochs=4,
        grid_width=6,
        grid_height=6,
        initial_action_probability=0.7,
        neural_peer_mode="well_mixed",
        seed=5,
    )
    well_mixed_result = run_toy2(
        config=load_toy2_config(well_mixed_path),
        config_path=well_mixed_path,
    )

    assert well_mixed_result.final_action_rate == pytest.approx(
        base_result.final_action_rate
    )
    assert well_mixed_result.final_mean_payoff == pytest.approx(
        base_result.final_mean_payoff
    )
    assert well_mixed_result.final_mean_policy_action_probability == pytest.approx(
        base_result.final_mean_policy_action_probability
    )


def test_toy2_learning_disabled_keeps_local_losses_zero(tmp_path: Path) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        learning_enabled=False,
    )
    result = run_toy2(config=load_toy2_config(config_path), config_path=config_path)

    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert all(float(row["local_loss"]) == 0.0 for row in rows)
    cooperation_probabilities = [float(row["action_probability"]) for row in rows]
    assert all(math.isfinite(value) for value in cooperation_probabilities)
    assert all(0.0 <= value <= 1.0 for value in cooperation_probabilities)


def test_toy2_fixed_seed_summary_is_reproducible(tmp_path: Path) -> None:
    config_path = write_tiny_config(tmp_path, mixer="output_average", peer_rule="none")
    first = run_toy2(config=load_toy2_config(config_path), config_path=config_path)
    second = run_toy2(config=load_toy2_config(config_path), config_path=config_path)

    assert second.final_action_rate == pytest.approx(first.final_action_rate)
    assert second.final_mean_payoff == pytest.approx(first.final_mean_payoff)
    assert second.final_mean_policy_action_probability == pytest.approx(
        first.final_mean_policy_action_probability
    )
