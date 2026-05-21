from __future__ import annotations

import copy
import csv
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
import neural_abm.toy_public_goods as toy_public_goods_module
from neural_abm.basin_phase_critic import (
    DEFAULT_BASIN_PHASE_CRITIC_FEATURES,
    LearnedBasinPhaseCritic,
    save_critic_bundle_npz,
)
from neural_abm.config import load_toy4_config
from neural_abm.spatial_binary import BinarySpatialRunner, BinaryStepContext
from neural_abm.toy_public_goods import (
    Toy4SpatialDomain,
    apply_output_average_distillation,
    apply_output_average_distillation_batched,
    build_observations,
    contribution_advantage_components,
    compute_public_goods_payoffs,
    create_agents,
    decision_action_probs,
    imitation_candidate_probabilities,
    local_groups,
    resource_action_lookahead_advantages,
    resource_break_even_fraction,
    resource_extraction_rates,
    resource_sustain_action_rate,
    resource_threshold_continuation_advantages,
    run_toy4,
    toy4_decision_mode_for_epoch,
    train_neural_local_policies_batched,
    train_neural_local_policy,
    uniform_group_member_index,
    update_resource_level,
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


def tiny_config_dict(
    tmp_path: Path,
    update_rule: str = "neural_policy",
    mixer: str = "none",
    resource_enabled: bool = False,
) -> dict:
    config = {
        "run": {
            "name": f"tiny_toy4_{update_rule}_{mixer}",
            "seed": 7,
            "output_dir": str(tmp_path / "runs"),
        },
        "simulation": {
            "epochs": 2,
            "sync_mode": "synchronous",
            "device": "cpu",
        },
        "environment": {
            "grid_width": 3,
            "grid_height": 3,
            "initial_action_probability": 0.5,
            "reward_ema_decay": 0.90,
            "entropy_beta": 0.01,
            "resource_enabled": resource_enabled,
            "resource_initial": 10.0,
            "resource_carrying_capacity": 10.0,
            "resource_recovery_rate": 0.0,
            "resource_extraction_per_defector": 2.0,
            "resource_collapse_threshold": 0.0,
        },
        "game": {
            "multiplier": 1.6,
            "contribution_cost": 1.0,
            "group_mode": "local_neighborhood",
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
            "domain": {},
        },
        "agents": {
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
            "type": "grid",
            "neighborhood": "von_neumann",
            "periodic": True,
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
    return binary_toy_config(config, "toy4")


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


def write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "toy4.yaml"
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


def test_toy4_baseline_config_loads() -> None:
    config = load_toy4_config(
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "configs"
        / "toy4_public_goods_baseline.yaml"
    )

    assert config.agent_count == 100
    assert config.policy.rule == "neural_policy"
    assert config.game.multiplier == pytest.approx(1.6)
    assert config.environment.resource_enabled is False
    assert config.policy.neural_update_backend == "loop"


def test_toy4_accepts_neural_update_backend_config(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["policy"]["neural_update_backend"] = "tensor_batched"
    path = write_config(tmp_path, raw)

    config = load_toy4_config(path)

    assert config.policy.neural_update_backend == "tensor_batched"


def test_toy4_empty_domain_defaults_to_material_objective(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    path = write_config(tmp_path, raw)

    config = load_toy4_config(path)

    assert config.policy.domain.objective.mode == "material"


def test_toy4_state_continuation_objective_requires_loop_backend(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["policy"]["neural_update_backend"] = "batched"
    raw["policy"]["domain"]["objective"] = {
        "mode": "state_continuation",
        "welfare_weight": 1.0,
    }
    path = write_config(tmp_path, raw)

    with pytest.raises(ValidationError, match="state_continuation"):
        load_toy4_config(path)


def test_toy4_profile_objective_requires_loop_backend(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["policy"]["neural_update_backend"] = "batched"
    raw["policy"]["domain"]["objective"] = {"profile": "linear_balanced"}
    path = write_config(tmp_path, raw)

    with pytest.raises(ValidationError, match="state_continuation"):
        load_toy4_config(path)


def test_toy4_domain_bootstrap_defaults_disabled(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    path = write_config(tmp_path, raw)

    config = load_toy4_config(path)

    assert config.policy.domain.bootstrap.enabled is False
    assert config.policy.domain.bootstrap.decision_enabled is False
    assert config.policy.domain.bootstrap.replay_enabled is False
    assert config.policy.domain.bootstrap.distill_stable_teacher_only is False
    assert config.policy.domain.bootstrap.distill_gradient_gate_enabled is False


def test_toy4_domain_bootstrap_rejects_unknown_teacher(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["policy"]["domain"]["objective"] = {"profile": "linear_welfare_heavy"}
    raw["policy"]["domain"]["bootstrap"] = {
        "enabled": True,
        "teacher": "imitation",
    }
    path = write_config(tmp_path, raw)

    with pytest.raises(ValidationError):
        load_toy4_config(path)

    raw["policy"]["domain"]["bootstrap"] = {
        "replay_enabled": True,
        "replay_teacher": "imitation",
    }
    path = write_config(tmp_path, raw)

    with pytest.raises(ValidationError):
        load_toy4_config(path)

    raw["policy"]["domain"]["bootstrap"] = {
        "distill_enabled": True,
        "distill_teacher": "imitation",
    }
    path = write_config(tmp_path, raw)

    with pytest.raises(ValidationError):
        load_toy4_config(path)


def test_toy4_domain_bootstrap_requires_neural_loop_state_continuation(
    tmp_path: Path,
) -> None:
    non_neural = tiny_config_dict(tmp_path, update_rule="reputation_imitation")
    non_neural["policy"]["domain"]["objective"] = {"profile": "linear_welfare_heavy"}
    non_neural["policy"]["domain"]["bootstrap"] = {"enabled": True}
    non_neural_path = tmp_path / "toy4_bootstrap_non_neural.yaml"
    non_neural_path.write_text(
        yaml.safe_dump(non_neural, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="domain bootstrap"):
        load_toy4_config(non_neural_path)

    non_loop = tiny_config_dict(tmp_path)
    non_loop["policy"]["neural_update_backend"] = "batched"
    non_loop["policy"]["domain"]["objective"] = {"profile": "linear_welfare_heavy"}
    non_loop["policy"]["domain"]["bootstrap"] = {"enabled": True}
    non_loop_path = tmp_path / "toy4_bootstrap_non_loop.yaml"
    non_loop_path.write_text(
        yaml.safe_dump(non_loop, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="state_continuation|domain bootstrap"):
        load_toy4_config(non_loop_path)

    material = tiny_config_dict(tmp_path)
    material["policy"]["domain"]["bootstrap"] = {"enabled": True}
    material_path = tmp_path / "toy4_bootstrap_material.yaml"
    material_path.write_text(
        yaml.safe_dump(material, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="state_continuation"):
        load_toy4_config(material_path)


def test_toy4_state_continuation_components_include_group_welfare(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["policy"]["domain"]["objective"] = {
        "mode": "state_continuation",
        "social_weight": 0.0,
        "welfare_weight": 1.0,
        "clip_abs": None,
    }
    path = write_config(tmp_path, raw)
    config = load_toy4_config(path)

    components = contribution_advantage_components(
        actions=np.asarray([0, 0], dtype=np.int64),
        groups=[[0, 1], [0, 1]],
        config=config,
        resource_level=config.environment.resource_carrying_capacity,
    )

    assert components.material[0] == pytest.approx(-0.125)
    assert components.welfare[0] == pytest.approx(0.375)
    assert components.effective[0] == pytest.approx(0.25)


def test_toy4_resource_environment_component_is_zero_without_resource(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    path = write_config(tmp_path, raw)
    config = load_toy4_config(path)

    components = contribution_advantage_components(
        actions=np.asarray([0, 0], dtype=np.int64),
        groups=[[0, 1], [0, 1]],
        config=config,
        resource_level=config.environment.resource_carrying_capacity,
    )

    assert components.environment.tolist() == pytest.approx([0.0, 0.0])


def test_toy4_resource_environment_component_tracks_collapse_pressure(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, resource_enabled=True)
    raw["environment"]["resource_initial"] = 60.0
    raw["environment"]["resource_carrying_capacity"] = 100.0
    raw["environment"]["resource_recovery_rate"] = 0.03
    raw["environment"]["resource_extraction_per_defector"] = 0.05
    raw["policy"]["domain"]["objective"] = {
        "mode": "state_continuation",
        "material_weight": 0.0,
        "social_weight": 0.0,
        "welfare_weight": 0.0,
        "environment_weight": 1.0,
        "clip_abs": None,
    }
    path = write_config(tmp_path, raw)
    config = load_toy4_config(path)

    actions = np.asarray([0, 0, 1, 1], dtype=np.int64)
    components = contribution_advantage_components(
        actions=actions,
        groups=[[0, 1, 2, 3] for _ in range(4)],
        config=config,
        resource_level=60.0,
    )
    sustain_gap = (
        resource_sustain_action_rate(config) - float(np.mean(actions))
    ) / resource_sustain_action_rate(config)
    resource_gap = (
        resource_break_even_fraction(config) - 0.6
    ) / resource_break_even_fraction(config)
    expected_pressure = sustain_gap + resource_gap

    assert resource_sustain_action_rate(config) == pytest.approx(0.625)
    assert resource_break_even_fraction(config) == pytest.approx(0.625)
    assert components.environment.tolist() == pytest.approx(
        [expected_pressure] * len(actions)
    )
    assert components.effective.tolist() == pytest.approx(
        [expected_pressure] * len(actions)
    )


def test_toy4_resource_environment_lookahead_is_opt_in(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, resource_enabled=True)
    raw["environment"]["resource_initial"] = 60.0
    raw["environment"]["resource_carrying_capacity"] = 100.0
    raw["environment"]["resource_recovery_rate"] = 0.03
    raw["environment"]["resource_extraction_per_defector"] = 0.05
    raw["policy"]["domain"]["objective"] = {
        "mode": "state_continuation",
        "material_weight": 0.0,
        "social_weight": 0.0,
        "welfare_weight": 0.0,
        "environment_weight": 1.0,
        "clip_abs": None,
    }
    path = write_config(tmp_path, raw)
    config = load_toy4_config(path)
    actions = np.asarray([0, 0, 1, 1], dtype=np.int64)

    components = contribution_advantage_components(
        actions=actions,
        groups=[[0, 1, 2, 3] for _ in range(4)],
        config=config,
        resource_level=60.0,
    )

    assert config.policy.domain.resource_environment_pressure_weight == pytest.approx(
        1.0
    )
    assert config.policy.domain.resource_environment_lookahead_weight == pytest.approx(
        0.0
    )
    assert config.policy.domain.resource_environment_threshold_weight == pytest.approx(
        0.0
    )
    sustain_gap = (
        resource_sustain_action_rate(config) - float(np.mean(actions))
    ) / resource_sustain_action_rate(config)
    resource_gap = (
        resource_break_even_fraction(config) - 0.6
    ) / resource_break_even_fraction(config)
    assert components.environment.tolist() == pytest.approx(
        [sustain_gap + resource_gap] * len(actions)
    )


def test_toy4_resource_lookahead_tracks_action_conditioned_stock_value(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, resource_enabled=True)
    raw["environment"]["resource_initial"] = 60.0
    raw["environment"]["resource_carrying_capacity"] = 100.0
    raw["environment"]["resource_recovery_rate"] = 0.03
    raw["environment"]["resource_extraction_per_defector"] = 0.05
    raw["policy"]["domain"] = {
        "resource_environment_pressure_weight": 0.0,
        "resource_environment_lookahead_weight": 1.0,
        "objective": {
            "mode": "state_continuation",
            "material_weight": 0.0,
            "social_weight": 0.0,
            "welfare_weight": 0.0,
            "environment_weight": 1.0,
            "clip_abs": None,
        },
    }
    path = write_config(tmp_path, raw)
    config = load_toy4_config(path)
    actions = np.asarray([0, 0, 1, 1], dtype=np.int64)

    components = contribution_advantage_components(
        actions=actions,
        groups=[[0, 1, 2, 3] for _ in range(4)],
        config=config,
        resource_level=60.0,
    )
    lookahead = resource_action_lookahead_advantages(
        actions=actions,
        config=config,
        resource_level=60.0,
    )
    expected_delta = (
        config.environment.resource_recovery_rate
        + config.environment.resource_extraction_per_defector
    )
    expected_value = (
        expected_delta
        / config.environment.resource_carrying_capacity
        * len(actions)
        * config.game.multiplier
        * resource_sustain_action_rate(config)
        / config.game.multiplier
    )

    assert lookahead.tolist() == pytest.approx([expected_value] * len(actions))
    assert components.environment.tolist() == pytest.approx(lookahead.tolist())
    assert components.effective.tolist() == pytest.approx(lookahead.tolist())


def test_toy4_resource_lookahead_respects_capacity_clip(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, resource_enabled=True)
    raw["environment"]["resource_recovery_rate"] = 0.03
    raw["environment"]["resource_extraction_per_defector"] = 0.05
    path = write_config(tmp_path, raw)
    config = load_toy4_config(path)

    lookahead = resource_action_lookahead_advantages(
        actions=np.asarray([1, 1, 1, 1], dtype=np.int64),
        config=config,
        resource_level=config.environment.resource_carrying_capacity,
    )

    assert lookahead.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_toy4_resource_threshold_tracks_population_sustain_gap(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, resource_enabled=True)
    raw["environment"]["resource_recovery_rate"] = 0.03
    raw["environment"]["resource_extraction_per_defector"] = 0.05
    raw["policy"]["domain"] = {
        "resource_environment_pressure_weight": 0.0,
        "resource_environment_lookahead_weight": 0.0,
        "resource_environment_threshold_weight": 1.0,
        "resource_environment_threshold_scope": "population",
        "objective": {
            "mode": "state_continuation",
            "material_weight": 0.0,
            "social_weight": 0.0,
            "welfare_weight": 0.0,
            "environment_weight": 1.0,
            "clip_abs": None,
        },
    }
    path = write_config(tmp_path, raw)
    config = load_toy4_config(path)
    actions = np.asarray([0, 0, 1, 1], dtype=np.int64)
    groups = [[0, 1, 2, 3] for _ in range(4)]

    components = contribution_advantage_components(
        actions=actions,
        groups=groups,
        config=config,
        resource_level=0.0,
    )
    threshold = resource_threshold_continuation_advantages(
        actions=actions,
        groups=groups,
        config=config,
    )

    assert resource_sustain_action_rate(config) == pytest.approx(0.625)
    assert threshold.tolist() == pytest.approx([0.16, 0.16, 0.96, 0.96])
    assert components.environment.tolist() == pytest.approx(threshold.tolist())
    assert components.effective.tolist() == pytest.approx(threshold.tolist())


def test_toy4_resource_threshold_local_scope_uses_neighborhoods(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, resource_enabled=True)
    raw["environment"]["resource_recovery_rate"] = 0.03
    raw["environment"]["resource_extraction_per_defector"] = 0.05
    raw["policy"]["domain"] = {
        "resource_environment_pressure_weight": 0.0,
        "resource_environment_lookahead_weight": 0.0,
        "resource_environment_threshold_weight": 1.0,
        "resource_environment_threshold_scope": "local",
        "objective": {
            "mode": "state_continuation",
            "material_weight": 0.0,
            "social_weight": 0.0,
            "welfare_weight": 0.0,
            "environment_weight": 1.0,
            "clip_abs": None,
        },
    }
    path = write_config(tmp_path, raw)
    config = load_toy4_config(path)
    actions = np.asarray([0, 0, 1, 1], dtype=np.int64)
    groups = [[0, 1], [2, 3]]

    threshold = resource_threshold_continuation_advantages(
        actions=actions,
        groups=groups,
        config=config,
    )

    assert threshold.tolist() == pytest.approx([1.6, 1.6, 0.08, 0.08])


def test_toy4_resource_threshold_is_inert_without_resource(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, resource_enabled=False)
    raw["policy"]["domain"] = {
        "resource_environment_pressure_weight": 0.0,
        "resource_environment_lookahead_weight": 0.0,
        "resource_environment_threshold_weight": 1.0,
        "resource_environment_threshold_scope": "local",
        "objective": {
            "mode": "state_continuation",
            "material_weight": 0.0,
            "social_weight": 0.0,
            "welfare_weight": 0.0,
            "environment_weight": 1.0,
            "clip_abs": None,
        },
    }
    path = write_config(tmp_path, raw)
    config = load_toy4_config(path)
    actions = np.asarray([0, 0, 1, 1], dtype=np.int64)
    groups = [[0, 1], [2, 3]]

    components = contribution_advantage_components(
        actions=actions,
        groups=groups,
        config=config,
        resource_level=config.environment.resource_carrying_capacity,
    )
    threshold = resource_threshold_continuation_advantages(
        actions=actions,
        groups=groups,
        config=config,
    )

    assert threshold.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert components.environment.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert components.effective.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_toy4_resource_extraction_heterogeneity_updates_resource_level(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, resource_enabled=True)
    raw["environment"]["grid_width"] = 2
    raw["environment"]["grid_height"] = 2
    raw["environment"]["resource_initial"] = 60.0
    raw["environment"]["resource_carrying_capacity"] = 100.0
    raw["environment"]["resource_recovery_rate"] = 0.03
    raw["environment"]["resource_extraction_per_defector"] = 0.05
    raw["environment"]["resource_extraction_heterogeneity"] = 0.5
    raw["environment"]["resource_extraction_heterogeneity_mode"] = "checkerboard"
    path = write_config(tmp_path, raw)
    config = load_toy4_config(path)

    rates = resource_extraction_rates(config, 4)
    high_defector_resource = update_resource_level(
        60.0,
        np.asarray([0, 1, 1, 1], dtype=np.int64),
        config,
    )
    low_defector_resource = update_resource_level(
        60.0,
        np.asarray([1, 0, 1, 1], dtype=np.int64),
        config,
    )

    assert rates.tolist() == pytest.approx([0.075, 0.025, 0.025, 0.075])
    assert resource_sustain_action_rate(config) == pytest.approx(0.625)
    assert high_defector_resource == pytest.approx(60.015)
    assert low_defector_resource == pytest.approx(60.065)


def test_toy4_resource_threshold_heterogeneity_uses_local_extraction_rates(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, resource_enabled=True)
    raw["environment"]["grid_width"] = 2
    raw["environment"]["grid_height"] = 2
    raw["environment"]["resource_recovery_rate"] = 0.03
    raw["environment"]["resource_extraction_per_defector"] = 0.05
    raw["environment"]["resource_extraction_heterogeneity"] = 0.5
    raw["environment"]["resource_extraction_heterogeneity_mode"] = "checkerboard"
    raw["policy"]["domain"] = {
        "resource_environment_pressure_weight": 0.0,
        "resource_environment_lookahead_weight": 0.0,
        "resource_environment_threshold_weight": 1.0,
        "resource_environment_threshold_scope": "local",
        "objective": {
            "mode": "state_continuation",
            "material_weight": 0.0,
            "social_weight": 0.0,
            "welfare_weight": 0.0,
            "environment_weight": 1.0,
            "clip_abs": None,
        },
    }
    path = write_config(tmp_path, raw)
    config = load_toy4_config(path)
    actions = np.asarray([0, 1, 1, 0], dtype=np.int64)
    groups = [[0, 3], [1, 2]]

    threshold = resource_threshold_continuation_advantages(
        actions=actions,
        groups=groups,
        config=config,
    )
    components = contribution_advantage_components(
        actions=actions,
        groups=groups,
        config=config,
        resource_level=0.0,
    )

    assert threshold.tolist() == pytest.approx([1.4, 0.0, 0.0, 1.4])
    assert components.environment.tolist() == pytest.approx(threshold.tolist())
    assert components.effective.tolist() == pytest.approx(threshold.tolist())


def test_toy4_objective_profiles_resolve_contribution_advantage(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["policy"]["domain"]["objective"] = {
        "profile": "nonlinear_interaction",
        "clip_abs": None,
    }
    path = write_config(tmp_path, raw)
    config = load_toy4_config(path)

    components = contribution_advantage_components(
        actions=np.asarray([0, 0], dtype=np.int64),
        groups=[[0, 1], [0, 1]],
        config=config,
        resource_level=config.environment.resource_carrying_capacity,
    )

    assert config.policy.domain.objective.mode == "state_continuation"
    assert config.policy.domain.objective.profile == "nonlinear_interaction"
    assert components.material[0] == pytest.approx(-0.125)
    assert components.social[0] == pytest.approx(0.1)
    assert components.welfare[0] == pytest.approx(0.375)
    expected_linear = -0.125 + 0.1 + 0.375
    expected_interaction = 0.1 * 0.375
    assert components.linear[0] == pytest.approx(expected_linear)
    assert components.interaction[0] == pytest.approx(expected_interaction)
    assert components.effective[0] == pytest.approx(
        np.tanh(expected_linear + expected_interaction)
    )


def test_toy4_domain_bootstrap_weight_zero_matches_objective_behavior(
    tmp_path: Path,
) -> None:
    base = tiny_config_dict(tmp_path)
    base["run"]["name"] = "toy4_bootstrap_weight_zero_base"
    base["simulation"]["epochs"] = 3
    base["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    base_path = tmp_path / "toy4_bootstrap_weight_zero_base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    bootstrapped = copy.deepcopy(base)
    bootstrapped["run"]["name"] = "toy4_bootstrap_weight_zero"
    bootstrapped["policy"]["domain"]["bootstrap"] = {
        "enabled": True,
        "weight": 0.0,
    }
    bootstrapped_path = tmp_path / "toy4_bootstrap_weight_zero.yaml"
    bootstrapped_path.write_text(
        yaml.safe_dump(bootstrapped, sort_keys=False),
        encoding="utf-8",
    )

    base_result = run_toy4(config=load_toy4_config(base_path), config_path=base_path)
    bootstrapped_result = run_toy4(
        config=load_toy4_config(bootstrapped_path),
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


def test_toy4_domain_bootstrap_weight_one_uses_reputation_teacher_direction(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["simulation"]["epochs"] = 1
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
    path = tmp_path / "toy4_bootstrap_weight_one.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_toy4_config(path)
    domain = Toy4SpatialDomain(
        config=config,
        config_path=path,
        rng=np.random.default_rng(config.run.seed),
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
        revision_mask=np.ones(config.agent_count, dtype=bool),
    )
    diagnostics = step_result.extras["domain_bootstrap_diagnostics"]

    np.testing.assert_allclose(
        diagnostics.bootstrapped_effective,
        diagnostics.teacher_signed,
    )


def test_toy4_domain_bootstrap_diagnostics_columns_are_logged(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["simulation"]["epochs"] = 1
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["bootstrap"] = {"enabled": True, "weight": 1.0}
    path = tmp_path / "toy4_bootstrap_logging.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy4(config=load_toy4_config(path), config_path=path)

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


def test_revision_operator_adapter_logs_toy4_diagnostics(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["simulation"]["epochs"] = 1
    raw["environment"]["initial_action_probability"] = 0.0
    raw["policy"]["learning_enabled"] = False
    raw["policy"]["revision_rate"] = 1.0
    raw["policy"]["decision"]["mode"] = "argmax"
    raw["coordination"]["revision_operator_enabled"] = True
    raw["coordination"]["revision_operator_source"] = "policy_probability"
    path = tmp_path / "toy4_revision_operator.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy4(config=load_toy4_config(path), config_path=path)
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
    assert 0.0 <= float(aggregate["mean_revision_switch_probability"]) <= 1.0
    for field in [
        "revision_operator_enabled",
        "revision_operator_source",
        "revision_choice",
        "revision_effective_stay_probability",
        "revision_switch_probability",
    ]:
        assert field in micro_reader.fieldnames
    assert micro_row["revision_operator_enabled"] == "True"
    assert micro_row["revision_operator_source"] == "policy_probability"
    assert micro_row["revision_choice"] in {
        "stay",
        "switch_to_1",
        "switch_to_0",
    }
    assert 0.0 <= float(micro_row["revision_switch_probability"]) <= 1.0


def test_toy4_terminal_argmax_epochs_switches_only_terminal_window(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["simulation"]["epochs"] = 10
    raw["policy"]["decision"]["terminal_argmax_epochs"] = 2
    path = tmp_path / "toy4_terminal_argmax.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_toy4_config(path)

    assert toy4_decision_mode_for_epoch(config, 8) == "sampled"
    assert toy4_decision_mode_for_epoch(config, 9) == "argmax"
    assert toy4_decision_mode_for_epoch(config, 10) == "argmax"

    policy_probs = torch.tensor([[0.45, 0.55], [0.8, 0.2]], dtype=torch.float32)
    decision_probs = decision_action_probs(policy_probs, config, mode="argmax")

    assert decision_probs.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_toy4_basin_credit_requires_neural_loop_state_continuation(
    tmp_path: Path,
) -> None:
    material = tiny_config_dict(tmp_path)
    material["policy"]["domain"]["basin_credit"] = {"enabled": True}
    material_path = tmp_path / "toy4_basin_credit_material.yaml"
    material_path.write_text(
        yaml.safe_dump(material, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="state_continuation"):
        load_toy4_config(material_path)

    non_loop = tiny_config_dict(tmp_path)
    non_loop["policy"]["neural_update_backend"] = "batched"
    non_loop["policy"]["domain"]["objective"] = {"profile": "linear_welfare_heavy"}
    non_loop["policy"]["domain"]["basin_credit"] = {"enabled": True}
    non_loop_path = tmp_path / "toy4_basin_credit_non_loop.yaml"
    non_loop_path.write_text(
        yaml.safe_dump(non_loop, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="state_continuation|basin credit"):
        load_toy4_config(non_loop_path)


def test_toy4_basin_credit_diagnostics_columns_are_logged(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, mixer="output_average")
    raw["simulation"]["epochs"] = 1
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
    path = tmp_path / "toy4_basin_credit_logging.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy4(config=load_toy4_config(path), config_path=path)

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


def test_toy4_basin_transition_samples_artifact_is_written(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, mixer="output_average")
    raw["simulation"]["epochs"] = 1
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
    path = tmp_path / "toy4_basin_transition_samples.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy4(config=load_toy4_config(path), config_path=path)

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
        "domain_resource_enabled",
        "domain_resource_level",
        "domain_resource_fraction",
    ]:
        assert field in frame.columns
    assert set(frame["sample_schema_version"]) == {1}
    assert set(frame["toy"]) == {"toy4"}
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


def test_toy4_basin_credit_logs_learned_read_only_diagnostics(
    tmp_path: Path,
) -> None:
    model_path = write_learned_basin_test_model(tmp_path)
    raw = tiny_config_dict(tmp_path, mixer="output_average")
    raw["simulation"]["epochs"] = 1
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
    path = tmp_path / "toy4_basin_learned_diagnostics.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy4(config=load_toy4_config(path), config_path=path)

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


def test_toy4_basin_credit_can_train_from_gated_learned_credit(
    tmp_path: Path,
) -> None:
    model_path = write_learned_basin_test_model(tmp_path)
    raw = tiny_config_dict(tmp_path, mixer="output_average")
    raw["simulation"]["epochs"] = 1
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
    path = tmp_path / "toy4_basin_learned_credit.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy4(config=load_toy4_config(path), config_path=path)

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


def test_toy4_basin_credit_replay_all_scope_trains_all_candidates_per_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tiny_config_dict(tmp_path, mixer="output_average")
    raw["simulation"]["epochs"] = 1
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
    path = tmp_path / "toy4_basin_credit_replay_all.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_toy4_config(path)
    domain = Toy4SpatialDomain(
        config=config,
        config_path=path,
        rng=np.random.default_rng(config.run.seed),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
        device=torch.device("cpu"),
    )
    state = domain.initial_state()
    revision_mask = np.zeros(config.agent_count, dtype=bool)
    revision_mask[[0, 2]] = True
    calls: list[int] = []

    def fake_train_neural_local_policy(**kwargs: object) -> float:
        calls.append(int(agent_index[id(kwargs["agent"])]))
        return 0.0

    agent_index = {id(agent): idx for idx, agent in enumerate(state.agents or [])}
    monkeypatch.setattr(
        toy_public_goods_module,
        "train_neural_local_policy",
        fake_train_neural_local_policy,
    )

    updates = domain.post_social_policy_update(
        state,
        BinaryStepContext(
            epoch=1,
            revision_mask=revision_mask,
            extras={"resource_level": float(state.extras["resource_level"])},
        ),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        torch.full((config.agent_count, 2), 0.5, dtype=torch.float64),
    )

    assert len(calls) == config.agent_count * 2
    assert sorted(set(calls)) == list(range(config.agent_count))
    diagnostics = updates["extras"]["basin_credit_diagnostics"]
    assert diagnostics.applied_mask.tolist() == [True] * config.agent_count
    assert diagnostics.training_scope == "all"
    assert diagnostics.training_pass_schedule == "fixed"
    assert diagnostics.training_passes == 2
    assert diagnostics.configured_training_passes == 2


def test_toy4_basin_credit_adaptive_replay_uses_effective_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tiny_config_dict(tmp_path, mixer="output_average")
    raw["simulation"]["epochs"] = 1
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
    path = tmp_path / "toy4_basin_credit_adaptive_replay.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_toy4_config(path)
    domain = Toy4SpatialDomain(
        config=config,
        config_path=path,
        rng=np.random.default_rng(config.run.seed),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
        device=torch.device("cpu"),
    )
    state = domain.initial_state()
    calls: list[int] = []
    agent_index = {id(agent): idx for idx, agent in enumerate(state.agents or [])}

    def fake_train_neural_local_policy(**kwargs: object) -> float:
        calls.append(int(agent_index[id(kwargs["agent"])]))
        return 0.0

    monkeypatch.setattr(
        toy_public_goods_module,
        "train_neural_local_policy",
        fake_train_neural_local_policy,
    )

    updates = domain.post_social_policy_update(
        state,
        BinaryStepContext(
            epoch=1,
            revision_mask=np.ones(config.agent_count, dtype=bool),
            extras={"resource_level": float(state.extras["resource_level"])},
        ),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        torch.full((config.agent_count, 2), 0.5, dtype=torch.float64),
    )

    assert len(calls) == config.agent_count * 2
    diagnostics = updates["extras"]["basin_credit_diagnostics"]
    assert diagnostics.training_pass_schedule == "target_score_decay"
    assert diagnostics.training_passes == 2
    assert diagnostics.configured_training_passes == 3


def test_toy4_domain_decision_bootstrap_requires_neural_loop_state_continuation(
    tmp_path: Path,
) -> None:
    material = tiny_config_dict(tmp_path)
    material["policy"]["domain"]["bootstrap"] = {"decision_enabled": True}
    material_path = tmp_path / "toy4_decision_bootstrap_material.yaml"
    material_path.write_text(
        yaml.safe_dump(material, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="state_continuation"):
        load_toy4_config(material_path)

    non_loop = copy.deepcopy(material)
    non_loop["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    non_loop["policy"]["neural_update_backend"] = "batched"
    non_loop_path = tmp_path / "toy4_decision_bootstrap_non_loop.yaml"
    non_loop_path.write_text(
        yaml.safe_dump(non_loop, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="state_continuation|domain bootstrap"):
        load_toy4_config(non_loop_path)


def test_toy4_domain_decision_replay_requires_neural_loop_state_continuation(
    tmp_path: Path,
) -> None:
    material = tiny_config_dict(tmp_path)
    material["policy"]["domain"]["bootstrap"] = {"replay_enabled": True}
    material_path = tmp_path / "toy4_decision_replay_material.yaml"
    material_path.write_text(
        yaml.safe_dump(material, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="state_continuation"):
        load_toy4_config(material_path)

    non_loop = tiny_config_dict(tmp_path)
    non_loop["policy"]["neural_update_backend"] = "batched"
    non_loop["policy"]["domain"]["objective"] = {"profile": "linear_welfare_heavy"}
    non_loop["policy"]["domain"]["bootstrap"] = {"replay_enabled": True}
    non_loop_path = tmp_path / "toy4_decision_replay_non_loop.yaml"
    non_loop_path.write_text(
        yaml.safe_dump(non_loop, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="state_continuation|domain bootstrap"):
        load_toy4_config(non_loop_path)


def test_toy4_domain_decision_bootstrap_weight_zero_matches_objective_behavior(
    tmp_path: Path,
) -> None:
    base = tiny_config_dict(tmp_path)
    base["run"]["name"] = "toy4_decision_bootstrap_weight_zero_base"
    base["simulation"]["epochs"] = 3
    base["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    base_path = tmp_path / "toy4_decision_bootstrap_weight_zero_base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    bootstrapped = copy.deepcopy(base)
    bootstrapped["run"]["name"] = "toy4_decision_bootstrap_weight_zero"
    bootstrapped["policy"]["domain"]["bootstrap"] = {
        "decision_enabled": True,
        "decision_weight": 0.0,
    }
    bootstrapped_path = tmp_path / "toy4_decision_bootstrap_weight_zero.yaml"
    bootstrapped_path.write_text(
        yaml.safe_dump(bootstrapped, sort_keys=False),
        encoding="utf-8",
    )

    base_result = run_toy4(config=load_toy4_config(base_path), config_path=base_path)
    bootstrapped_result = run_toy4(
        config=load_toy4_config(bootstrapped_path),
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


def test_toy4_domain_decision_bootstrap_weight_one_uses_teacher_probability(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["simulation"]["epochs"] = 1
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
    path = tmp_path / "toy4_decision_bootstrap_weight_one.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_toy4_config(path)
    domain = Toy4SpatialDomain(
        config=config,
        config_path=path,
        rng=np.random.default_rng(config.run.seed),
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
        revision_mask=np.ones(config.agent_count, dtype=bool),
    )
    diagnostics = step_result.extras["domain_decision_bootstrap_diagnostics"]
    candidate_probs = step_result.extras["decision_action_probs"][:, 1].detach().numpy()

    np.testing.assert_allclose(
        diagnostics.bootstrapped_probabilities,
        diagnostics.teacher_probabilities,
    )
    np.testing.assert_allclose(candidate_probs, diagnostics.teacher_probabilities)


def test_toy4_domain_decision_bootstrap_diagnostics_columns_and_expiry_are_logged(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["simulation"]["epochs"] = 2
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["bootstrap"] = {
        "decision_enabled": True,
        "decision_weight": 1.0,
        "decision_epochs": 1,
    }
    path = tmp_path / "toy4_decision_bootstrap_logging.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy4(config=load_toy4_config(path), config_path=path)

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


def test_toy4_domain_decision_replay_diagnostics_columns_and_expiry_are_logged(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["simulation"]["epochs"] = 2
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
    path = tmp_path / "toy4_decision_replay_logging.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy4(config=load_toy4_config(path), config_path=path)

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


def test_toy4_domain_distill_bootstrap_requires_neural_loop_state_continuation(
    tmp_path: Path,
) -> None:
    material = tiny_config_dict(tmp_path)
    material["policy"]["domain"]["bootstrap"] = {"distill_enabled": True}
    material_path = tmp_path / "toy4_distill_bootstrap_material.yaml"
    material_path.write_text(
        yaml.safe_dump(material, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="state_continuation"):
        load_toy4_config(material_path)

    non_loop = copy.deepcopy(material)
    non_loop["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    non_loop["policy"]["neural_update_backend"] = "batched"
    non_loop_path = tmp_path / "toy4_distill_bootstrap_non_loop.yaml"
    non_loop_path.write_text(
        yaml.safe_dump(non_loop, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="state_continuation|domain bootstrap"):
        load_toy4_config(non_loop_path)


def test_toy4_domain_distill_bootstrap_weight_zero_matches_objective_behavior(
    tmp_path: Path,
) -> None:
    base = tiny_config_dict(tmp_path)
    base["run"]["name"] = "toy4_distill_bootstrap_weight_zero_base"
    base["simulation"]["epochs"] = 3
    base["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    base_path = tmp_path / "toy4_distill_bootstrap_weight_zero_base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    bootstrapped = copy.deepcopy(base)
    bootstrapped["run"]["name"] = "toy4_distill_bootstrap_weight_zero"
    bootstrapped["policy"]["domain"]["bootstrap"] = {
        "distill_enabled": True,
        "distill_weight": 0.0,
    }
    bootstrapped_path = tmp_path / "toy4_distill_bootstrap_weight_zero.yaml"
    bootstrapped_path.write_text(
        yaml.safe_dump(bootstrapped, sort_keys=False),
        encoding="utf-8",
    )

    base_result = run_toy4(config=load_toy4_config(base_path), config_path=base_path)
    bootstrapped_result = run_toy4(
        config=load_toy4_config(bootstrapped_path),
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


def test_toy4_teacher_distillation_loss_moves_probability_toward_teacher(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["bootstrap"] = {
        "distill_enabled": True,
        "distill_weight": 1.0,
    }
    path = write_config(tmp_path, raw)
    config = load_toy4_config(path)
    agent = create_agents(config, torch.device("cpu"))[0]
    observation = torch.zeros(config.agents.model.input_dim)

    before = float(torch.softmax(agent.model(observation), dim=-1)[1].detach())
    target = 1.0 if before < 0.9 else 0.0
    train_neural_local_policy(
        agent=agent,
        observation=observation,
        action=0,
        payoff=0.0,
        payoff_baseline=0.0,
        config=config,
        teacher_distill_probability=target,
        teacher_distill_weight=1.0,
        base_loss_weight=0.0,
    )
    after = float(torch.softmax(agent.model(observation), dim=-1)[1].detach())

    if target == 1.0:
        assert after > before
    else:
        assert after < before


def test_toy4_domain_distill_bootstrap_diagnostics_columns_and_expiry_are_logged(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["simulation"]["epochs"] = 2
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["bootstrap"] = {
        "distill_enabled": True,
        "distill_weight": 1.0,
        "distill_epochs": 1,
    }
    path = tmp_path / "toy4_distill_bootstrap_logging.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy4(config=load_toy4_config(path), config_path=path)

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


def test_toy4_teacher_alignment_diagnostics_columns_are_logged(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["simulation"]["epochs"] = 2
    raw["policy"]["domain"]["objective"] = {
        "profile": "linear_welfare_heavy",
        "clip_abs": None,
    }
    raw["policy"]["domain"]["bootstrap"] = {
        "distill_enabled": True,
        "distill_weight": 1.0,
        "distill_epochs": 1,
    }
    path = tmp_path / "toy4_teacher_alignment_logging.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy4(config=load_toy4_config(path), config_path=path)

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


def test_toy4_distill_gradient_gate_rejects_conflicting_teacher_updates(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["simulation"]["epochs"] = 1
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
    path = tmp_path / "toy4_distill_gradient_gate.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = run_toy4(config=load_toy4_config(path), config_path=path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        [_, active_row] = list(csv.DictReader(handle))

    assert float(active_row["domain_distill_candidate_rate"]) == pytest.approx(1.0)
    assert float(active_row["domain_distill_applied_rate"]) < 1.0
    assert float(active_row["domain_distill_rejected_gradient_rate"]) > 0.0


def test_toy4_non_neural_policy_canonicalizes_backend(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="imitation")
    raw["policy"]["neural_update_backend"] = "tensor_batched"
    path = write_config(tmp_path, raw)

    config = load_toy4_config(path)

    assert config.policy.neural_update_backend == "loop"


def test_toy4_tensor_batched_output_similarity_runner_smoke(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, mixer="output_average")
    raw["simulation"]["epochs"] = 1
    raw["policy"]["neural_update_backend"] = "tensor_batched"
    path = write_config(tmp_path, raw)

    result = run_toy4(config=load_toy4_config(path), config_path=path)

    assert 0.0 <= result.final_action_rate <= 1.0
    assert (result.run_dir / "summary.json").exists()
    final_row = final_aggregate_row(result.run_dir)
    assert final_row["social_channel"] == "policy_distribution"
    assert final_row["commit_mode"] == "distillation_step"
    assert float(final_row["mean_social_update_norm"]) >= 0.0
    assert float(final_row["max_social_update_norm"]) >= 0.0
    assert int(final_row["active_social_agent_count"]) > 0


def test_toy4_tensor_batched_neural_backend_runner_smoke(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, mixer="output_average")
    raw["simulation"]["epochs"] = 1
    raw["policy"]["neural_update_backend"] = "tensor_batched"
    raw["coordination"]["peer_rule"] = "none"
    config_path = write_config(tmp_path, raw)

    result = run_toy4(
        config=load_toy4_config(config_path),
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
def test_toy4_tensor_batched_backend_matches_batched_runner(
    tmp_path: Path,
    peer_rule: str,
) -> None:
    base = tiny_config_dict(tmp_path, mixer="output_average")
    base["simulation"]["epochs"] = 2
    base["logging"]["micro_state"] = False
    base["coordination"]["peer_rule"] = peer_rule

    batched = copy.deepcopy(base)
    batched["run"]["name"] = f"toy4_batched_reference_{peer_rule}"
    batched["policy"]["neural_update_backend"] = "batched"
    batched_path = tmp_path / "toy4_batched.yaml"
    batched_path.write_text(yaml.safe_dump(batched, sort_keys=False), encoding="utf-8")

    tensor_batched = copy.deepcopy(base)
    tensor_batched["run"]["name"] = f"toy4_tensor_batched_{peer_rule}"
    tensor_batched["policy"]["neural_update_backend"] = "tensor_batched"
    tensor_path = tmp_path / "toy4_tensor_batched.yaml"
    tensor_path.write_text(
        yaml.safe_dump(tensor_batched, sort_keys=False),
        encoding="utf-8",
    )

    batched_result = run_toy4(
        config=load_toy4_config(batched_path),
        config_path=batched_path,
    )
    tensor_result = run_toy4(
        config=load_toy4_config(tensor_path),
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


def test_toy4_tensor_batched_no_social_backend_matches_batched_runner(
    tmp_path: Path,
) -> None:
    base = tiny_config_dict(tmp_path, mixer="none")
    base["simulation"]["epochs"] = 2
    base["logging"]["micro_state"] = False
    base["coordination"]["peer_rule"] = "none"

    batched = copy.deepcopy(base)
    batched["run"]["name"] = "toy4_batched_reference_no_social"
    batched["policy"]["neural_update_backend"] = "batched"
    batched_path = tmp_path / "toy4_batched_no_social.yaml"
    batched_path.write_text(yaml.safe_dump(batched, sort_keys=False), encoding="utf-8")

    tensor_batched = copy.deepcopy(base)
    tensor_batched["run"]["name"] = "toy4_tensor_batched_no_social"
    tensor_batched["policy"]["neural_update_backend"] = "tensor_batched"
    tensor_path = tmp_path / "toy4_tensor_batched_no_social.yaml"
    tensor_path.write_text(
        yaml.safe_dump(tensor_batched, sort_keys=False),
        encoding="utf-8",
    )

    batched_result = run_toy4(
        config=load_toy4_config(batched_path),
        config_path=batched_path,
    )
    tensor_result = run_toy4(
        config=load_toy4_config(tensor_path),
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


def test_toy4_batched_local_training_matches_loop(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, tiny_config_dict(tmp_path))
    config = load_toy4_config(config_path)
    device = torch.device("cpu")
    loop_agents = create_agents(config, device)
    batched_agents = create_agents(config, device)
    torch.manual_seed(501)
    observations = torch.randn(config.agent_count, config.agents.model.input_dim)
    actions = np.asarray([agent_id % 2 for agent_id in range(config.agent_count)])
    payoffs = np.linspace(-0.4, 1.8, config.agent_count)
    payoff_baseline = np.linspace(0.2, 0.8, config.agent_count)
    revision_mask = np.asarray(
        [agent_id % 3 != 1 for agent_id in range(config.agent_count)]
    )

    for _ in range(3):
        loop_losses = [0.0 for _ in range(config.agent_count)]
        for agent_id in np.flatnonzero(revision_mask):
            loop_losses[int(agent_id)] = train_neural_local_policy(
                agent=loop_agents[int(agent_id)],
                observation=observations[int(agent_id)],
                action=int(actions[int(agent_id)]),
                payoff=float(payoffs[int(agent_id)]),
                payoff_baseline=float(payoff_baseline[int(agent_id)]),
                config=config,
            )
        batched_losses = train_neural_local_policies_batched(
            agents=batched_agents,
            observations=observations,
            actions=actions,
            payoffs=payoffs,
            payoff_baseline=payoff_baseline,
            revision_mask=revision_mask,
            config=config,
        )

        assert np.allclose(batched_losses, loop_losses, atol=1e-6)
    assert_agent_parameters_match(batched_agents, loop_agents)


def test_toy4_batched_distillation_matches_loop(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        tiny_config_dict(tmp_path, mixer="output_average"),
    )
    config = load_toy4_config(config_path)
    device = torch.device("cpu")
    loop_agents = create_agents(config, device)
    batched_agents = create_agents(config, device)
    torch.manual_seed(502)
    observations = torch.randn(config.agent_count, config.agents.model.input_dim)
    previous_probs = torch.softmax(torch.randn(config.agent_count, 2), dim=-1)
    peer_ids = [
        [1, 3],
        [],
        [0, 5],
        [2],
        [3, 5],
        [4],
        [7],
        [6, 8],
        [7],
    ]

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
        )

        assert np.allclose(batched_losses, loop_losses, atol=1e-6)
    assert_agent_parameters_match(batched_agents, loop_agents)


def test_toy4_batched_neural_backend_runner_smoke(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, mixer="output_average")
    raw["simulation"]["epochs"] = 1
    raw["policy"]["neural_update_backend"] = "batched"
    config_path = write_config(tmp_path, raw)

    result = run_toy4(
        config=load_toy4_config(config_path),
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


def test_toy4_rejects_invalid_resource_bounds(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["environment"]["resource_initial"] = 12.0
    raw["environment"]["resource_carrying_capacity"] = 10.0
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="resource_initial"):
        load_toy4_config(path)


def test_toy4_rejects_invalid_neural_model_shape(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy")
    raw["agents"]["model"]["output_dim"] = 1
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="model.output_dim"):
        load_toy4_config(path)


def test_toy4_legacy_shared_dsl_is_rejected(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy")
    raw["dynamics"] = {"update_rule": "neural_policy"}
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError):
        load_toy4_config(path)


def test_toy4_reputation_and_mobility_config_defaults(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="reputation_imitation")
    path = write_config(tmp_path, raw)

    config = load_toy4_config(path)

    assert config.policy.rule == "reputation_imitation"
    assert config.state.reputation.enabled is True
    assert config.state.reputation.decay == pytest.approx(0.9)
    assert config.state.mobility.enabled is False


def test_toy4_neural_reputation_observation_requires_8d_model(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy")
    raw["state"]["reputation"] = {
        "enabled": True,
        "observation_mode": "self_neighbor_mean",
    }
    raw["agents"]["model"]["input_dim"] = 8
    path = write_config(tmp_path, raw)

    config = load_toy4_config(path)

    assert config.state.reputation.observation_mode == "self_neighbor_mean"
    assert config.agents.model.input_dim == 8


def test_toy4_observation_appends_reputation_features(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy")
    raw["state"]["reputation"] = {
        "enabled": True,
        "observation_mode": "self_neighbor_mean",
    }
    raw["agents"]["model"]["input_dim"] = 8
    config = load_toy4_config(write_config(tmp_path, raw))

    observations = build_observations(
        actions=np.asarray([1, 0, 1], dtype=np.int64),
        payoffs=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        payoff_ema=np.asarray([0.5, 0.25, 0.0], dtype=np.float64),
        groups=[[0, 1], [0, 1, 2], [1, 2]],
        resource_level=10.0,
        config=config,
        device=torch.device("cpu"),
        reputation=np.asarray([0.2, 0.8, 0.5], dtype=np.float64),
    )

    assert tuple(observations.shape) == (3, 8)
    np.testing.assert_allclose(
        observations[:, 6:].numpy(),
        np.asarray(
            [
                [0.2, 0.5],
                [0.8, 0.5],
                [0.5, 0.65],
            ],
            dtype=np.float32,
        ),
    )


def test_toy4_resource_observation_hidden_masks_global_fraction(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy", resource_enabled=True)
    raw["environment"]["resource_carrying_capacity"] = 100.0
    raw["environment"]["resource_observation_mode"] = "hidden"
    config = load_toy4_config(write_config(tmp_path, raw))

    observations = build_observations(
        actions=np.asarray([1, 0, 1], dtype=np.int64),
        payoffs=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        payoff_ema=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        groups=[[0, 1], [0, 1, 2], [1, 2]],
        resource_level=20.0,
        config=config,
        device=torch.device("cpu"),
    )

    assert observations[:, 4].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_toy4_resource_observation_local_sustain_uses_group_sustain_rates(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy", resource_enabled=True)
    raw["environment"]["grid_width"] = 2
    raw["environment"]["grid_height"] = 2
    raw["environment"]["resource_recovery_rate"] = 0.03
    raw["environment"]["resource_extraction_per_defector"] = 0.05
    raw["environment"]["resource_extraction_heterogeneity"] = 0.5
    raw["environment"]["resource_extraction_heterogeneity_mode"] = "checkerboard"
    raw["environment"]["resource_observation_mode"] = "local_sustain"
    config = load_toy4_config(write_config(tmp_path, raw))

    observations = build_observations(
        actions=np.asarray([0, 1, 1, 0], dtype=np.int64),
        payoffs=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        payoff_ema=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        groups=[[0, 3], [1, 2], [1, 2], [0, 3]],
        resource_level=20.0,
        config=config,
        device=torch.device("cpu"),
    )

    assert observations[:, 4].tolist() == pytest.approx([0.0, 1.0, 1.0, 0.0])


def test_toy4_indexed_observations_match_group_loop(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy")
    raw["state"]["reputation"] = {
        "enabled": True,
        "observation_mode": "self_neighbor_mean",
    }
    raw["agents"]["model"]["input_dim"] = 8
    config = load_toy4_config(write_config(tmp_path, raw))
    groups = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    group_member_index = uniform_group_member_index(groups)
    assert group_member_index is not None

    kwargs = {
        "actions": np.asarray([1, 0, 1], dtype=np.int64),
        "payoffs": np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        "payoff_ema": np.asarray([0.5, 0.25, 0.0], dtype=np.float64),
        "groups": groups,
        "resource_level": 10.0,
        "config": config,
        "device": torch.device("cpu"),
        "reputation": np.asarray([0.2, 0.8, 0.5], dtype=np.float64),
    }

    loop_observations = build_observations(**kwargs)
    indexed_observations = build_observations(
        **kwargs,
        group_member_index=group_member_index,
    )

    assert torch.allclose(indexed_observations, loop_observations)


def test_toy4_tensor_observations_match_numpy(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy")
    raw["state"]["reputation"] = {
        "enabled": True,
        "observation_mode": "self_neighbor_mean",
    }
    raw["agents"]["model"]["input_dim"] = 8
    config = load_toy4_config(write_config(tmp_path, raw))
    groups = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    group_member_index = uniform_group_member_index(groups)
    assert group_member_index is not None

    actions = np.asarray([1, 0, 1], dtype=np.int64)
    payoffs = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    payoff_ema = np.asarray([0.5, 0.25, 0.0], dtype=np.float64)
    reputation = np.asarray([0.2, 0.8, 0.5], dtype=np.float64)
    device = torch.device("cpu")

    numpy_observations = build_observations(
        actions=actions,
        payoffs=payoffs,
        payoff_ema=payoff_ema,
        groups=groups,
        resource_level=10.0,
        config=config,
        device=device,
        reputation=reputation,
        group_member_index=group_member_index,
    )
    tensor_observations = build_observations(
        actions=torch.as_tensor(actions, dtype=torch.long),
        payoffs=torch.as_tensor(payoffs, dtype=torch.float64),
        payoff_ema=torch.as_tensor(payoff_ema, dtype=torch.float64),
        groups=groups,
        resource_level=10.0,
        config=config,
        device=device,
        reputation=torch.as_tensor(reputation, dtype=torch.float64),
        group_member_index=torch.as_tensor(group_member_index, dtype=torch.long),
    )

    assert tensor_observations.dtype == torch.float32
    assert torch.allclose(tensor_observations, numpy_observations)


def test_toy4_uniform_group_member_index_rejects_ragged_groups() -> None:
    assert uniform_group_member_index([[0, 1], [0, 1, 2]]) is None


def test_toy4_reputation_imitation_requires_enabled_reputation(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="reputation_imitation")
    raw["state"]["reputation"] = {"enabled": False}
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="reputation.enabled"):
        load_toy4_config(path)


def test_toy4_rejects_unknown_reputation_option(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["state"]["reputation"] = {"enabled": True, "unknown": 1}
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError):
        load_toy4_config(path)


def test_toy4_rejects_unknown_mobility_option(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["state"]["mobility"] = {"enabled": False, "unknown": 1}
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError):
        load_toy4_config(path)


def test_public_goods_payoff_exposes_free_rider_advantage() -> None:
    actions = np.asarray([1, 0], dtype=np.int64)
    payoffs = compute_public_goods_payoffs(
        actions=actions,
        groups=[[0, 1]],
        multiplier=1.6,
        contribution_cost=1.0,
    )

    assert payoffs[0] == pytest.approx(-0.2)
    assert payoffs[1] == pytest.approx(0.8)
    assert payoffs[1] > payoffs[0]


def test_public_goods_indexed_payoffs_match_group_loop() -> None:
    actions = np.asarray([1, 0, 1, 0], dtype=np.int64)
    groups = [[0, 1, 2], [0, 1, 2], [1, 2, 3], [1, 2, 3]]
    group_member_index = uniform_group_member_index(groups)
    assert group_member_index is not None

    loop_payoffs = compute_public_goods_payoffs(
        actions=actions,
        groups=groups,
        multiplier=1.6,
        contribution_cost=1.0,
        resource_multiplier=0.75,
    )
    indexed_payoffs = compute_public_goods_payoffs(
        actions=actions,
        groups=groups,
        multiplier=1.6,
        contribution_cost=1.0,
        resource_multiplier=0.75,
        group_member_index=group_member_index,
    )

    np.testing.assert_allclose(indexed_payoffs, loop_payoffs)


def test_public_goods_tensor_indexed_payoffs_match_numpy() -> None:
    actions = np.asarray([1, 0, 1, 0], dtype=np.int64)
    groups = [[0, 1, 2], [0, 1, 2], [1, 2, 3], [1, 2, 3]]
    group_member_index = uniform_group_member_index(groups)
    assert group_member_index is not None

    numpy_payoffs = compute_public_goods_payoffs(
        actions=actions,
        groups=groups,
        multiplier=1.6,
        contribution_cost=1.0,
        resource_multiplier=0.75,
        group_member_index=group_member_index,
    )
    tensor_payoffs = compute_public_goods_payoffs(
        actions=torch.as_tensor(actions, dtype=torch.long),
        groups=groups,
        multiplier=1.6,
        contribution_cost=1.0,
        resource_multiplier=0.75,
        group_member_index=torch.as_tensor(group_member_index, dtype=torch.long),
    )

    assert isinstance(tensor_payoffs, torch.Tensor)
    assert tensor_payoffs.dtype == torch.float64
    np.testing.assert_allclose(tensor_payoffs.numpy(), numpy_payoffs)


def test_local_groups_include_self_and_neighbors() -> None:
    groups = local_groups([[1], [0, 2], [1]])

    assert groups == [[0, 1], [0, 1, 2], [1, 2]]


def test_resource_variant_can_collapse_under_low_contribution(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, resource_enabled=True)
    config = load_toy4_config(write_config(tmp_path, raw))
    actions = np.zeros(config.agent_count, dtype=np.int64)

    resource = update_resource_level(10.0, actions, config)

    assert resource == pytest.approx(0.0)


def test_toy4_tensor_batched_initial_state_uses_torch_arrays(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, mixer="output_average")
    raw["policy"]["neural_update_backend"] = "tensor_batched"
    raw["coordination"]["peer_rule"] = "none"
    config_path = write_config(tmp_path, raw)
    config = load_toy4_config(config_path)
    domain = Toy4SpatialDomain(
        config=config,
        config_path=config_path,
        rng=np.random.default_rng(config.run.seed),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
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
    ):
        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float64


def test_toy4_neural_local_step_routes_through_binary_policy_learning_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy", mixer="none")
    raw["simulation"]["epochs"] = 1
    raw["policy"]["decision"]["mode"] = "argmax"
    config_path = write_config(tmp_path, raw)
    config = load_toy4_config(config_path)
    domain = Toy4SpatialDomain(
        config=config,
        config_path=config_path,
        rng=np.random.default_rng(config.run.seed),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
        device=torch.device("cpu"),
        neural_update_backend=config.policy.neural_update_backend,
    )
    state = domain.initial_state()
    context = domain.build_step_context(
        epoch=1,
        state=state,
        revision_mask=np.ones(config.agent_count, dtype=bool),
    )
    original_unit = toy_public_goods_module.BinaryPolicyLearningUnit
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
        toy_public_goods_module,
        "BinaryPolicyLearningUnit",
        SpyBinaryPolicyLearningUnit,
    )

    result = domain.local_step(state, context)

    assert seen == {
        "agent_count": config.agent_count,
        "context": context,
        "callbacks_type": "BinaryPolicyLearningCallbacks",
        "has_collect_policy_probs": True,
        "has_decision_action_probs": True,
        "has_sample_actions": True,
        "has_local_update": True,
        "has_refresh_policy_cache": True,
        "has_post_collect_policy_probs": True,
        "run_called": True,
        "decision_shape": (config.agent_count, 2),
        "post_shape": (config.agent_count, 2),
    }
    assert result.social_mode == "policy_distill"
    assert result.actions_after_revision is not None
    assert result.extras["decision_action_probs"].shape == (config.agent_count, 2)
    assert "state_continuation_components" in result.extras


def test_imitation_candidate_copies_higher_payoff_neighbor() -> None:
    actions = np.asarray([1, 0, 1], dtype=np.int64)
    payoffs = np.asarray([-0.2, 0.8, -0.2], dtype=np.float64)
    neighbors = [[1], [0, 2], [1]]
    revision_mask = np.ones(3, dtype=bool)

    probabilities = imitation_candidate_probabilities(
        actions=actions,
        payoffs=payoffs,
        neighbors=neighbors,
        revision_mask=revision_mask,
        selection_strength=100.0,
    )

    assert probabilities.tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_imitation_selection_strength_scales_copy_probability() -> None:
    actions = np.asarray([1, 0], dtype=np.int64)
    payoffs = np.asarray([0.0, 1.0], dtype=np.float64)
    neighbors = [[1], [0]]
    revision_mask = np.ones(2, dtype=bool)

    off = imitation_candidate_probabilities(
        actions=actions,
        payoffs=payoffs,
        neighbors=neighbors,
        revision_mask=revision_mask,
        selection_strength=0.0,
    )
    weak = imitation_candidate_probabilities(
        actions=actions,
        payoffs=payoffs,
        neighbors=neighbors,
        revision_mask=revision_mask,
        selection_strength=0.25,
    )
    strong = imitation_candidate_probabilities(
        actions=actions,
        payoffs=payoffs,
        neighbors=neighbors,
        revision_mask=revision_mask,
        selection_strength=2.0,
    )

    assert off.tolist() == pytest.approx([1.0, 0.0])
    assert 0.0 < strong[0] < weak[0] < off[0]
    assert weak[1] == pytest.approx(0.0)
    assert strong[1] == pytest.approx(0.0)


def test_toy4_no_social_initial_peer_metrics_are_empty(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="imitation", mixer="none")
    raw["simulation"]["epochs"] = 1
    config_path = write_config(tmp_path, raw)

    result = run_toy4(config=load_toy4_config(config_path), config_path=config_path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_rows = list(csv.DictReader(handle))

    assert float(aggregate_rows[0]["mean_peer_count"]) == pytest.approx(0.0)
    assert int(aggregate_rows[0]["fragmentation_components"]) == raw[
        "environment"
    ]["grid_width"] * raw["environment"]["grid_height"]


def test_toy4_hook_probability_mix_samples_social_probabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="imitation", mixer="output_average")
    config_path = write_config(tmp_path, raw)
    config = load_toy4_config(config_path)
    rng = np.random.default_rng(config.run.seed)
    domain = Toy4SpatialDomain(
        config=config,
        config_path=config_path,
        rng=rng,
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
        revision_mask=np.ones(config.agent_count, dtype=bool),
    )

    assert seen["action_probs"] == pytest.approx(
        step_result.post_social_probs[:, 1].detach().cpu().numpy()
    )


@pytest.mark.parametrize(
    ("update_rule", "mixer"),
    [
        ("neural_policy", "none"),
        ("neural_policy", "output_average"),
        ("imitation", "none"),
        ("imitation", "output_average"),
        ("reputation_imitation", "none"),
        ("reputation_imitation", "output_average"),
    ],
)
def test_toy4_runner_smoke_writes_expected_outputs(
    tmp_path: Path,
    update_rule: str,
    mixer: str,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule=update_rule, mixer=mixer)
    config_path = write_config(tmp_path, raw)

    result = run_toy4(config=load_toy4_config(config_path), config_path=config_path)

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
    assert float(final["domain_payoff_gini"]) >= 0.0
    assert "domain_exploitation_index" in final
    assert "mean_reputation" in final
    assert "reputation_dispersion" in final
    assert "mobility_rate" in final
    assert "mean_mobility_gain" in final
    assert "policy_action_probability_post_social_gt_0p7_rate" in final
    assert "policy_action_probability_post_social_dwell_0p4_0p6_rate" in final
    assert "policy_probability_threshold_crossings_0p5_count" in final
    assert "action_flip_rate" in final

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
        "payoff",
        "payoff_ema",
        "reputation",
        "domain_local_action_rate",
        "domain_group_payoff_mean",
        "domain_resource_level",
        "peer_count",
        "revised",
        "mobility_moved",
        "mobility_target",
        "mobility_gain",
    ]:
        assert field in micro_reader.fieldnames


def test_toy4_precommitment_trajectory_fields_are_logged(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="neural_policy", mixer="none")
    raw["simulation"]["epochs"] = 1
    raw["environment"]["initial_action_probability"] = 0.0
    raw["policy"]["learning_enabled"] = False
    raw["policy"]["decision"]["mode"] = "argmax"
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
    config_path = write_config(tmp_path, raw)

    result = run_toy4(config=load_toy4_config(config_path), config_path=config_path)

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
    ("update_rule", "expected"),
    [
        (
            "neural_policy",
            {
                "final_action_rate": 0.4444444444444444,
                "final_mean_payoff": 0.26666666666666683,
                "domain_payoff_gini": 0.4412755000990295,
                "domain_resource_level": 10.0,
                "domain_collapse_time": None,
                "domain_action_components": 1,
                "domain_largest_action_cluster_fraction": 0.4444444444444444,
                "domain_exploitation_index": 0.3733333333333333,
                "final_mean_reputation": 0.34444444444444444,
                "final_reputation_dispersion": 0.4042307127567158,
            },
        ),
        (
            "imitation",
            {
                "final_action_rate": 0.1111111111111111,
                "final_mean_payoff": 0.06666666666666668,
                "domain_payoff_gini": 0.12804232804232804,
                "domain_resource_level": 10.0,
                "domain_collapse_time": None,
                "domain_action_components": 1,
                "domain_largest_action_cluster_fraction": 0.1111111111111111,
                "domain_exploitation_index": 0.09333333333333332,
                "final_mean_reputation": 0.2911111111111111,
                "final_reputation_dispersion": 0.4149282252375288,
            },
        ),
        (
            "reputation_imitation",
            {
                "final_action_rate": 1.0,
                "final_mean_payoff": 0.6000000000000001,
                "domain_payoff_gini": 0.0,
                "domain_resource_level": 10.0,
                "domain_collapse_time": None,
                "domain_action_components": 1,
                "domain_largest_action_cluster_fraction": 1.0,
                "domain_exploitation_index": 0.0,
                "final_mean_reputation": 0.44999999999999996,
                "final_reputation_dispersion": 0.3898717737923586,
            },
        ),
    ],
)
def test_toy4_tiny_runner_golden_metrics(
    tmp_path: Path,
    update_rule: str,
    expected: dict[str, float | int | None],
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule=update_rule, mixer="none")
    config_path = write_config(tmp_path, raw)

    result = run_toy4(config=load_toy4_config(config_path), config_path=config_path)

    for field, value in expected.items():
        actual = result_value(result, field)
        if value is None or isinstance(value, int):
            assert actual == value
        else:
            assert actual == pytest.approx(value)


@pytest.mark.parametrize(
    ("mobility_enabled", "expected_mobility_rate", "expected_mean_mobility_gain"),
    [
        (False, 0.0, 0.0),
        (True, 0.4444444444444444, 0.016855999999999996),
    ],
)
def test_toy4_tiny_mobility_golden_metrics(
    tmp_path: Path,
    mobility_enabled: bool,
    expected_mobility_rate: float,
    expected_mean_mobility_gain: float,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="imitation", mixer="none")
    if mobility_enabled:
        raw["state"]["mobility"] = {
            "enabled": True,
            "rate": 1.0,
            "candidate_pool_size": 4,
            "selection_rule": "local_quality",
            "move_cost": 0.0,
        }
    config_path = write_config(tmp_path, raw)

    result = run_toy4(config=load_toy4_config(config_path), config_path=config_path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        final_row = list(csv.DictReader(handle))[-1]
    assert float(final_row["mobility_rate"]) == pytest.approx(expected_mobility_rate)
    assert float(final_row["mean_mobility_gain"]) == pytest.approx(
        expected_mean_mobility_gain
    )


def test_toy4_resource_runner_records_domain_collapse_time(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="imitation", resource_enabled=True)
    raw["environment"]["initial_action_probability"] = 0.0
    raw["simulation"]["epochs"] = 2
    config_path = write_config(tmp_path, raw)

    result = run_toy4(config=load_toy4_config(config_path), config_path=config_path)

    assert result_value(result, "domain_resource_level") == pytest.approx(0.0)
    assert result_value(result, "domain_collapse_time") == 1


def test_toy4_reputation_mobility_runner_records_extended_metrics(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path, update_rule="reputation_imitation")
    raw["simulation"]["epochs"] = 2
    raw["state"]["mobility"] = {
        "enabled": True,
        "rate": 1.0,
        "candidate_pool_size": 4,
        "selection_rule": "local_quality",
        "move_cost": 0.0,
    }
    config_path = write_config(tmp_path, raw)

    result = run_toy4(config=load_toy4_config(config_path), config_path=config_path)

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

    assert aggregate_rows
    assert micro_rows
    assert all(0.0 <= float(row["mean_reputation"]) <= 1.0 for row in aggregate_rows)
    assert all(0.0 <= float(row["mobility_rate"]) <= 1.0 for row in aggregate_rows)
    assert all(0.0 <= float(row["reputation"]) <= 1.0 for row in micro_rows)
