from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from binary_config_helpers import toy9_config
from neural_abm.config import load_toy9_config
from neural_abm.toy_heterogeneous import (
    assign_agent_groups,
    group_counts,
    run_toy9,
    select_peer_ids,
)


def tiny_config_dict(
    tmp_path: Path,
    *,
    mixer: str = "output_average",
    peer_rule: str | None = None,
) -> dict:
    resolved_peer_rule = "none" if peer_rule is None and mixer == "none" else peer_rule
    if resolved_peer_rule is None:
        resolved_peer_rule = "output_similarity"
    return toy9_config(
        {
            "run": {
                "name": f"tiny_toy9_{mixer}_{resolved_peer_rule}",
                "seed": 1,
                "output_dir": str(tmp_path / "runs"),
            },
            "simulation": {
                "epochs": 3,
                "sync_mode": "synchronous",
                "device": "cpu",
            },
            "policy": {
                "rule": "heterogeneous_rules",
                "decision_mode": "sampled",
                "temperature": 1.0,
                "exploration_epsilon": 0.0,
            },
            "agents": {
                "count": 8,
                "init_mode": "group_priors",
                "groups": [
                    {
                        "name": "threshold_social",
                        "fraction": 0.5,
                        "local_rule": "threshold",
                        "initial_action_probability": 0.35,
                        "threshold": 0.4,
                        "revision_rate": 1.0,
                        "learning_rate": 0.2,
                        "coordination_enabled": True,
                    },
                    {
                        "name": "payoff_individual",
                        "fraction": 0.5,
                        "local_rule": "payoff_learning",
                        "initial_action_probability": 0.35,
                        "threshold": None,
                        "revision_rate": 1.0,
                        "learning_rate": 0.18,
                        "coordination_enabled": False,
                    },
                ],
            },
            "coordination": {
                "mixer": mixer,
                "peer_rule": resolved_peer_rule,
                "alpha": 0.25 if mixer != "none" else 0.0,
                "threshold": 0.0,
            },
            "environment": {
                "initial_action_probability": 0.35,
                "threshold": 0.45,
                "benefit": 1.2,
                "action_cost": 0.35,
                "payoff_ema_decay": 0.9,
            },
            "graph": {
                "type": "watts_strogatz",
                "k": 2,
                "rewire_probability": 0.0,
            },
            "logging": {
                "micro_state": True,
                "interval": 1,
                "aggregate_metrics": True,
                "probe_predictions": False,
                "probe_prediction_interval": 1,
            },
        }
    )


def write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "toy9.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_toy9_group_assignment_respects_fraction_counts(tmp_path: Path) -> None:
    config = load_toy9_config(write_config(tmp_path, tiny_config_dict(tmp_path)))
    counts = group_counts(config)
    group_ids = assign_agent_groups(config, np.random.default_rng(1))

    assert counts.tolist() == [4, 4]
    assert np.bincount(group_ids, minlength=2).tolist() == [4, 4]


def test_toy9_coordination_gate_disables_individual_group_peers(
    tmp_path: Path,
) -> None:
    config = load_toy9_config(write_config(tmp_path, tiny_config_dict(tmp_path)))
    probabilities = np.full(config.agents.count, 0.5)
    neighbors = [[1], [0], [3], [2], [5], [4], [7], [6]]
    group_ids = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    enabled_by_group = np.asarray([True, False], dtype=bool)

    peer_ids = select_peer_ids(
        probabilities,
        neighbors,
        group_ids,
        enabled_by_group,
        config,
    )

    assert peer_ids[0] == [1]
    assert peer_ids[1] == []
    assert peer_ids[2] == [3]
    assert peer_ids[3] == []


@pytest.mark.parametrize(
    ("mixer", "peer_rule"),
    [("none", "none"), ("output_average", "none"), ("output_average", "output_similarity")],
)
def test_toy9_runner_smoke_writes_expected_outputs(
    tmp_path: Path,
    mixer: str,
    peer_rule: str,
) -> None:
    config_path = write_config(
        tmp_path,
        tiny_config_dict(tmp_path, mixer=mixer, peer_rule=peer_rule),
    )

    result = run_toy9(config=load_toy9_config(config_path), config_path=config_path)

    assert result.toy == "toy9"
    assert result.run_dir.exists()
    assert (result.run_dir / "aggregate_metrics.csv").exists()
    assert (result.run_dir / "micro_state.csv").exists()
    assert 0.0 <= result.domain_metrics["domain_final_action_rate"] <= 1.0
    assert 0.0 <= result.domain_metrics["domain_final_group_action_rate_gap"] <= 1.0
    summary = json.loads((result.run_dir / "summary.json").read_text("utf-8"))
    assert summary["toy"] == "toy9"


def test_toy9_rejects_duplicate_group_names(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["model"]["agents"]["groups"][1]["name"] = "threshold_social"
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="group names must be unique"):
        load_toy9_config(path)
