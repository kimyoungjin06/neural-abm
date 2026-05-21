from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from binary_config_helpers import toy8_config
from neural_abm.config import load_toy8_config
from neural_abm.toy_async import (
    STATE_ACTIVE,
    STATE_FAILED,
    STATE_INACTIVE,
    compute_rate_snapshot,
    graph_neighbors,
    initialize_states,
    run_toy8,
)
from neural_abm.graphs import build_graph


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
    return toy8_config(
        {
            "run": {
                "name": f"tiny_toy8_{mixer}_{resolved_peer_rule}",
                "seed": 7,
                "output_dir": str(tmp_path / "runs"),
            },
            "simulation": {
                "epochs": 8,
                "sync_mode": "synchronous",
                "device": "cpu",
            },
            "policy": {
                "rule": "event_hazard",
            },
            "agents": {
                "count": 8,
            },
            "coordination": {
                "mixer": mixer,
                "peer_rule": resolved_peer_rule,
                "alpha": 0.25 if mixer == "output_average" else 0.0,
                "threshold": 0.8 if resolved_peer_rule == "output_similarity" else 0.0,
            },
            "environment": {
                "initial_active_fraction": 0.25,
                "initial_failed_fraction": 0.125,
                "base_activation_rate": 0.05,
                "peer_activation_rate": 0.4,
                "failure_rate": 0.03,
                "overload_failure_rate": 0.1,
                "recovery_rate": 0.02,
                "max_time": 30.0,
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
    path = tmp_path / "toy8.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_toy8_initialize_states_respects_initial_fractions(tmp_path: Path) -> None:
    import numpy as np

    config = load_toy8_config(write_config(tmp_path, tiny_config_dict(tmp_path)))
    states = initialize_states(config, np.random.default_rng(7))

    assert int((states == STATE_ACTIVE).sum()) == 2
    assert int((states == STATE_FAILED).sum()) == 1
    assert int((states == STATE_INACTIVE).sum()) == 5


def test_toy8_rate_snapshot_uses_neighbor_activation(tmp_path: Path) -> None:
    config = load_toy8_config(write_config(tmp_path, tiny_config_dict(tmp_path)))
    graph = build_graph(config.graph, config.agents.count, config.run.seed)
    neighbors = graph_neighbors(graph, config.agents.count)
    states = np_array([1, 0, 0, 0, 0, 0, 0, 0], dtype=int)

    snapshot = compute_rate_snapshot(states, neighbors, config)

    assert snapshot.activation_rates[1] > config.environment.base_activation_rate
    assert snapshot.failure_rates[0] >= config.environment.failure_rate
    assert snapshot.recovery_rates.sum() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("mixer", "peer_rule"),
    [
        ("none", "none"),
        ("output_average", "none"),
        ("output_average", "output_similarity"),
    ],
)
def test_toy8_runner_smoke_writes_expected_outputs(
    tmp_path: Path,
    mixer: str,
    peer_rule: str,
) -> None:
    config_path = write_config(
        tmp_path,
        tiny_config_dict(tmp_path, mixer=mixer, peer_rule=peer_rule),
    )

    result = run_toy8(config=load_toy8_config(config_path), config_path=config_path)

    assert result.toy == "toy8"
    assert result.run_dir.exists()
    assert 0.0 <= result.domain_metrics["domain_final_active_fraction"] <= 1.0
    assert 0.0 <= result.domain_metrics["domain_final_failed_fraction"] <= 1.0
    assert result.domain_metrics["domain_total_events"] <= 8
    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_rows = list(csv.DictReader(handle))
    assert aggregate_rows
    assert "domain_event_type" in aggregate_rows[-1]
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["toy"] == "toy8"


def test_toy8_rejects_invalid_initial_fractions(tmp_path: Path) -> None:
    raw = tiny_config_dict(tmp_path)
    raw["domain"]["environment"]["initial_active_fraction"] = 0.8
    raw["domain"]["environment"]["initial_failed_fraction"] = 0.3
    path = write_config(tmp_path, raw)

    with pytest.raises(ValueError, match="initial_active_fraction"):
        load_toy8_config(path)


def np_array(values: object, dtype: type = float):
    import numpy as np

    return np.asarray(values, dtype=dtype)
