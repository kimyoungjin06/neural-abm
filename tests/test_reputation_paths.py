from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from binary_config_helpers import binary_toy_config
from neural_abm.config import load_toy2_config, load_toy4_config, load_toy5_config
from neural_abm.toy_contagion import run_toy5
from neural_abm.toy_pd import run_toy2
from neural_abm.toy_public_goods import run_toy4


def toy2_reputation_config(tmp_path: Path) -> dict[str, Any]:
    return binary_toy_config(
        {
            "run": {
                "name": "toy2_reputation_smoke",
                "seed": 3,
                "output_dir": str(tmp_path),
            },
            "simulation": {"epochs": 1, "sync_mode": "synchronous", "device": "cpu"},
            "game": {
                "family": "prisoner_dilemma",
                "payoff": {"T": 5.0, "R": 3.0, "P": 1.0, "S": 0.0},
            },
            "policy": {"rule": "reputation_imitation", "revision_rate": 1.0},
            "environment": {
                "grid_width": 3,
                "grid_height": 3,
                "neighborhood": "von_neumann",
                "periodic": True,
                "initial_action_probability": 0.5,
                "reward_ema_decay": 0.9,
                "entropy_beta": 0.01,
            },
            "agents": {
                "init_mode": "same_init",
                "model": {"input_dim": 6, "hidden_dim": 8, "output_dim": 2},
                "optimizer": {"name": "adam", "learning_rate": 0.01},
            },
            "coordination": {
                "mixer": "output_average",
                "peer_rule": "none",
                "alpha": 0.25,
                "threshold": 0.0,
                "communication_budget": {
                    "probe_predictions": 1,
                    "latent_dim": 8,
                    "scalar_summary": 8,
                },
            },
            "logging": {
                "micro_state": True,
                "interval": 1,
                "aggregate_metrics": True,
            },
            "state": {},
        },
        "toy2",
    )


def toy4_reputation_config(tmp_path: Path) -> dict[str, Any]:
    return binary_toy_config(
        {
            "run": {
                "name": "toy4_reputation_smoke",
                "seed": 3,
                "output_dir": str(tmp_path),
            },
            "simulation": {"epochs": 1, "sync_mode": "synchronous", "device": "cpu"},
            "environment": {
                "grid_width": 3,
                "grid_height": 3,
                "initial_action_probability": 0.5,
                "reward_ema_decay": 0.9,
                "entropy_beta": 0.01,
            },
            "game": {"multiplier": 1.6, "contribution_cost": 1.0},
            "policy": {"rule": "reputation_imitation", "revision_rate": 1.0},
            "agents": {
                "init_mode": "same_init",
                "model": {"input_dim": 6, "hidden_dim": 8, "output_dim": 2},
                "optimizer": {"name": "adam", "learning_rate": 0.01},
            },
            "graph": {"type": "grid", "neighborhood": "von_neumann", "periodic": True},
            "coordination": {
                "mixer": "output_average",
                "peer_rule": "none",
                "alpha": 0.25,
                "threshold": 0.0,
            },
            "logging": {
                "micro_state": True,
                "interval": 1,
                "aggregate_metrics": True,
            },
            "state": {},
        },
        "toy4",
    )


def toy5_reputation_config(tmp_path: Path) -> dict[str, Any]:
    return binary_toy_config(
        {
            "run": {
                "name": "toy5_reputation_smoke",
                "seed": 3,
                "output_dir": str(tmp_path),
            },
            "simulation": {"epochs": 1, "sync_mode": "synchronous", "device": "cpu"},
            "environment": {
                "initial_action_fraction": 1.0 / 6.0,
                "seed_selection": "first_agent",
                "threshold_mode": "homogeneous",
                "homogeneous_threshold": 0.5,
                "simple_contagion_probability": 1.0,
            },
            "policy": {
                "rule": "reputation_imitation",
                "revision_rate": 1.0,
                "domain": {"adoption_is_absorbing": True},
            },
            "agents": {
                "count": 6,
                "init_mode": "same_init",
                "model": {"input_dim": 6, "hidden_dim": 8, "output_dim": 2},
                "optimizer": {"name": "adam", "learning_rate": 0.01},
            },
            "graph": {"type": "watts_strogatz", "k": 2, "rewire_probability": 0.0},
            "coordination": {
                "mixer": "output_average",
                "peer_rule": "none",
                "alpha": 0.25,
                "threshold": 0.0,
            },
            "logging": {
                "micro_state": True,
                "interval": 1,
                "aggregate_metrics": True,
            },
            "state": {},
        },
        "toy5",
    )


@pytest.mark.parametrize(
    ("toy", "raw_factory", "loader", "runner"),
    [
        ("toy2", toy2_reputation_config, load_toy2_config, run_toy2),
        ("toy4", toy4_reputation_config, load_toy4_config, run_toy4),
        ("toy5", toy5_reputation_config, load_toy5_config, run_toy5),
    ],
)
def test_reputation_imitation_smoke_matrix(
    tmp_path: Path,
    toy: str,
    raw_factory: Callable[[Path], dict[str, Any]],
    loader: Callable[[Path], Any],
    runner: Callable[[Any, Path], Any],
) -> None:
    path = tmp_path / f"{toy}_reputation.yaml"
    path.write_text(yaml.safe_dump(raw_factory(tmp_path / "runs")), encoding="utf-8")

    result = runner(config=loader(path), config_path=path)

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
    metadata = json.loads((result.run_dir / "metadata.json").read_text())
    for field in [
        "policy_rule",
        "coordination_mixer",
        "coordination_peer_rule",
    ]:
        assert field in metadata
        assert field in aggregate_rows[0]
        assert field in micro_rows[0]
    assert "policy_revision_rate" in aggregate_rows[0]
    assert "realized_revision_rate" in aggregate_rows[0]
    for legacy_field in ["update_rule", "mixer", "peer_rule", "revision_rate"]:
        assert legacy_field not in metadata
        assert legacy_field not in aggregate_rows[0]
        assert legacy_field not in micro_rows[0]
    assert all(0.0 <= float(row["mean_reputation"]) <= 1.0 for row in aggregate_rows)
    assert all(0.0 <= float(row["reputation"]) <= 1.0 for row in micro_rows)
