from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from binary_config_helpers import toy10_config
from neural_abm.config import load_toy10_config
from neural_abm.toy_market import (
    harvest_from_channels,
    initialize_channel,
    market_price,
    run_toy10,
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
    return toy10_config(
        {
            "run": {
                "name": f"tiny_toy10_{mixer}_{resolved_peer_rule}",
                "seed": 1,
                "output_dir": str(tmp_path / "runs"),
            },
            "simulation": {
                "epochs": 3,
                "sync_mode": "synchronous",
                "device": "cpu",
            },
            "policy": {
                "rule": "market_ecology",
                "learning_rate": 0.18,
                "revision_rate": 1.0,
                "exploration_std": 0.02,
                "conservation_harvest_weight": 0.75,
                "social_harvest_gain": 1.0,
                "social_disagreement_penalty": 0.0,
                "reward_ema_decay": 0.9,
            },
            "agents": {
                "count": 8,
                "init_mode": "independent_init",
            },
            "coordination": {
                "mixer": mixer,
                "peer_rule": resolved_peer_rule,
                "alpha": 0.25 if mixer != "none" else 0.0,
                "threshold": 0.0,
            },
            "environment": {
                "resource_initial": 80.0,
                "resource_carrying_capacity": 100.0,
                "resource_recovery_rate": 0.05,
                "extraction_scale": 5.0,
                "extraction_cost": 0.3,
                "base_price": 0.45,
                "demand_sensitivity": 0.5,
                "supply_sensitivity": 0.25,
                "initial_price_expectation_mean": 0.5,
                "initial_price_expectation_std": 0.05,
                "initial_conservation_norm_mean": 0.35,
                "initial_conservation_norm_std": 0.05,
            },
            "network": {
                "type": "watts_strogatz",
                "k": 2,
                "rewire_probability": 0.0,
                "dynamic_rewire_rate": 0.5,
                "candidate_pool_size": 3,
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
    path = tmp_path / "toy10.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_toy10_initialize_channel_same_init_is_constant() -> None:
    values = initialize_channel(
        mean=0.4,
        std=0.1,
        count=5,
        same_init=True,
        rng=np.random.default_rng(1),
    )

    assert values.tolist() == [0.4] * 5


def test_toy10_market_price_tracks_demand_supply(tmp_path: Path) -> None:
    config = load_toy10_config(write_config(tmp_path, tiny_config_dict(tmp_path)))
    price, imbalance = market_price(
        harvest_intensities=np.asarray([0.2, 0.4]),
        price_expectations=np.asarray([0.6, 0.8]),
        config=config,
    )

    assert imbalance == pytest.approx(0.4)
    assert price == pytest.approx(0.725)


def test_toy10_social_harvest_gain_amplifies_mixed_channel_shift(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["model"]["policy"]["exploration_std"] = 0.0
    raw["model"]["policy"]["social_harvest_gain"] = 2.0
    config = load_toy10_config(write_config(tmp_path, raw))

    harvest = harvest_from_channels(
        price_expectations=np.asarray([0.5, 0.5]),
        conservation_norms=np.asarray([0.1, 0.1]),
        rng=np.random.default_rng(1),
        config=config,
        base_price_expectations=np.asarray([0.2, 0.8]),
        base_conservation_norms=np.asarray([0.1, 0.1]),
    )

    assert harvest == pytest.approx([0.74, 0.185])


def test_toy10_social_disagreement_penalty_reduces_harvest(
    tmp_path: Path,
) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["model"]["policy"]["exploration_std"] = 0.0
    raw["model"]["policy"]["social_disagreement_penalty"] = 0.5
    config = load_toy10_config(write_config(tmp_path, raw))

    harvest = harvest_from_channels(
        price_expectations=np.asarray([0.5, 0.5]),
        conservation_norms=np.asarray([0.1, 0.1]),
        rng=np.random.default_rng(1),
        config=config,
        base_price_expectations=np.asarray([0.2, 0.8]),
        base_conservation_norms=np.asarray([0.1, 0.1]),
    )

    assert harvest == pytest.approx([0.3875, 0.3875])


@pytest.mark.parametrize(
    ("mixer", "peer_rule"),
    [("none", "none"), ("output_average", "none"), ("output_average", "output_similarity")],
)
def test_toy10_runner_smoke_writes_expected_outputs(
    tmp_path: Path,
    mixer: str,
    peer_rule: str,
) -> None:
    config_path = write_config(
        tmp_path,
        tiny_config_dict(tmp_path, mixer=mixer, peer_rule=peer_rule),
    )

    result = run_toy10(config=load_toy10_config(config_path), config_path=config_path)

    assert result.toy == "toy10"
    assert result.run_dir.exists()
    assert (result.run_dir / "aggregate_metrics.csv").exists()
    assert (result.run_dir / "micro_state.csv").exists()
    assert 0.0 <= result.domain_metrics["domain_final_resource_fraction"] <= 1.0
    assert 0.0 <= result.domain_metrics["domain_final_market_price"] <= 1.0
    assert result.domain_metrics["domain_cumulative_rewired_edge_count"] >= 0
    summary = json.loads((result.run_dir / "summary.json").read_text("utf-8"))
    assert summary["toy"] == "toy10"


def test_toy10_rejects_invalid_resource_initial(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["domain"]["environment"]["resource_initial"] = 101.0
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="resource_initial"):
        load_toy10_config(path)
