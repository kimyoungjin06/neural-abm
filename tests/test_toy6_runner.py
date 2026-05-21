from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from binary_config_helpers import toy6_config
from neural_abm.config import load_toy6_config
from neural_abm.toy_categorical import (
    compute_cyclic_payoffs,
    run_toy6,
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
    return toy6_config(
        {
            "run": {
                "name": f"tiny_toy6_{mixer}_{resolved_peer_rule}",
                "seed": 7,
                "output_dir": str(tmp_path / "runs"),
            },
            "simulation": {
                "epochs": 2,
                "sync_mode": "synchronous",
                "device": "cpu",
            },
            "policy": {
                "rule": "categorical_learning",
                "learning_rate": 0.15,
                "revision_rate": 1.0,
                "temperature": 1.0,
                "decision_mode": "sampled",
            },
            "agents": {
                "init_mode": "independent_init",
                "logit_noise": 0.1,
            },
            "coordination": {
                "mixer": mixer,
                "peer_rule": resolved_peer_rule,
                "alpha": 0.25 if mixer == "output_average" else 0.0,
                "threshold": 0.8 if resolved_peer_rule == "output_similarity" else 0.0,
            },
            "environment": {
                "grid_width": 3,
                "grid_height": 3,
                "initial_strategy_probabilities": [0.34, 0.33, 0.33],
                "reward_ema_decay": 0.9,
            },
            "game": {
                "strategy_count": 3,
                "win_payoff": 1.0,
                "loss_payoff": -1.0,
                "draw_payoff": 0.0,
            },
            "graph": {
                "type": "grid",
                "neighborhood": "von_neumann",
                "periodic": True,
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
    path = tmp_path / "toy6.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_toy6_cyclic_payoff_matrix() -> None:
    actions = [0, 1, 2]
    neighbors = [[1, 2], [0, 2], [0, 1]]

    payoffs = compute_cyclic_payoffs(
        actions=np_array(actions),
        neighbors=neighbors,
        strategy_count=3,
        win_payoff=1.0,
        loss_payoff=-1.0,
        draw_payoff=0.0,
    )

    assert payoffs.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_toy6_output_similarity_selects_distribution_peers(tmp_path: Path) -> None:
    config = load_toy6_config(
        write_config(tmp_path, tiny_config_dict(tmp_path, mixer="output_average"))
    )
    probabilities = np_array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )

    peer_ids = select_peer_ids(
        probabilities,
        neighbors=[[1, 2], [0, 2], [0, 1]],
        config=config,
    )

    assert peer_ids[0] == [1]
    assert peer_ids[1] == [0]


@pytest.mark.parametrize(
    ("mixer", "peer_rule"),
    [
        ("none", "none"),
        ("output_average", "none"),
        ("output_average", "output_similarity"),
    ],
)
def test_toy6_runner_smoke_writes_expected_outputs(
    tmp_path: Path,
    mixer: str,
    peer_rule: str,
) -> None:
    config_path = write_config(
        tmp_path,
        tiny_config_dict(tmp_path, mixer=mixer, peer_rule=peer_rule),
    )

    result = run_toy6(config=load_toy6_config(config_path), config_path=config_path)

    assert result.toy == "toy6"
    assert result.run_dir.exists()
    assert 0.0 <= result.domain_metrics["domain_final_strategy_entropy"] <= 1.0
    assert 0.0 <= result.domain_metrics["domain_final_dominant_strategy_fraction"] <= 1.0
    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_rows = list(csv.DictReader(handle))
    assert aggregate_rows
    assert "domain_strategy_entropy" in aggregate_rows[-1]
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["toy"] == "toy6"


def test_toy6_rejects_bad_initial_strategy_probabilities(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["domain"]["environment"]["initial_strategy_probabilities"] = [1.0, 0.0]
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="initial_strategy_probabilities"):
        load_toy6_config(path)


def np_array(values: object):
    import numpy as np

    return np.asarray(values, dtype=np.float64)
