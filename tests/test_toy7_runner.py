from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from binary_config_helpers import toy7_config
from neural_abm.config import load_toy7_config
from neural_abm.toy_resource import (
    adaptive_target,
    compute_payoffs,
    run_toy7,
    select_peer_ids,
)


def tiny_config_dict(
    tmp_path: Path,
    mixer: str = "none",
    peer_rule: str | None = None,
) -> dict:
    resolved_peer_rule = (
        peer_rule
        if peer_rule is not None
        else "output_similarity"
        if mixer == "output_average"
        else "none"
    )
    return toy7_config(
        {
            "run": {
                "name": f"tiny_toy7_{mixer}_{resolved_peer_rule}",
                "seed": 7,
                "output_dir": str(tmp_path / "runs"),
            },
            "simulation": {
                "epochs": 2,
                "sync_mode": "synchronous",
                "device": "cpu",
            },
            "policy": {
                "rule": "adaptive_intensity",
                "learning_rate": 0.2,
                "revision_rate": 1.0,
                "exploration_std": 0.02,
                "reward_ema_decay": 0.9,
            },
            "agents": {
                "count": 8,
                "init_mode": "independent_init",
            },
            "coordination": {
                "mixer": mixer,
                "peer_rule": resolved_peer_rule,
                "alpha": 0.25 if mixer == "output_average" else 0.0,
                "threshold": 0.8 if resolved_peer_rule == "output_similarity" else 0.0,
            },
            "environment": {
                "resource_initial": 80.0,
                "resource_carrying_capacity": 100.0,
                "resource_recovery_rate": 0.05,
                "extraction_scale": 8.0,
                "extraction_cost": 0.35,
                "initial_intensity_mean": 0.35,
                "initial_intensity_std": 0.05,
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
    path = tmp_path / "toy7.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_toy7_payoff_penalizes_costly_extraction(tmp_path: Path) -> None:
    config = load_toy7_config(write_config(tmp_path, tiny_config_dict(tmp_path)))

    payoffs = compute_payoffs(np_array([0.2, 0.8]), 80.0, config)

    assert payoffs[0] > 0.0
    assert payoffs[1] < 0.8 * 0.8


def test_toy7_adaptive_target_is_bounded(tmp_path: Path) -> None:
    config = load_toy7_config(write_config(tmp_path, tiny_config_dict(tmp_path)))

    assert 0.0 <= adaptive_target(80.0, config) <= 1.0


def test_toy7_output_similarity_selects_scalar_peers(tmp_path: Path) -> None:
    config = load_toy7_config(
        write_config(tmp_path, tiny_config_dict(tmp_path, mixer="output_average"))
    )

    peer_ids = select_peer_ids(
        np_array([0.1, 0.2, 0.9]),
        neighbors=[[1, 2], [0, 2], [0, 1]],
        config=config,
    )

    assert peer_ids == [[1], [0], []]


@pytest.mark.parametrize(
    ("mixer", "peer_rule"),
    [
        ("none", "none"),
        ("output_average", "none"),
        ("output_average", "output_similarity"),
    ],
)
def test_toy7_runner_smoke_writes_expected_outputs(
    tmp_path: Path,
    mixer: str,
    peer_rule: str,
) -> None:
    config_path = write_config(
        tmp_path,
        tiny_config_dict(tmp_path, mixer=mixer, peer_rule=peer_rule),
    )

    result = run_toy7(config=load_toy7_config(config_path), config_path=config_path)

    assert result.toy == "toy7"
    assert result.run_dir.exists()
    assert 0.0 <= result.domain_metrics["domain_final_resource_fraction"] <= 1.0
    assert 0.0 <= result.domain_metrics["domain_final_mean_intensity"] <= 1.0
    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_rows = list(csv.DictReader(handle))
    assert aggregate_rows
    assert "domain_resource_level" in aggregate_rows[-1]
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["toy"] == "toy7"


def test_toy7_rejects_invalid_graph_degree(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["domain"]["graph"]["k"] = raw["model"]["agents"]["count"]
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="graph.k"):
        load_toy7_config(path)


def np_array(values: object):
    import numpy as np

    return np.asarray(values, dtype=np.float64)
