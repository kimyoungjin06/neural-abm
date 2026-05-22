from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from binary_config_helpers import toy8_config
from neural_abm.config import load_toy8_config
from neural_abm.toy_async import (
    STATE_ACTIVE,
    STATE_FAILED,
    STATE_INACTIVE,
    Toy8RateSnapshot,
    Toy8StepResult,
    apply_output_average as apply_toy8_output_average,
    aggregate_row as aggregate_toy8_row,
    compute_rate_snapshot,
    graph_neighbors,
    initialize_states,
    micro_rows as toy8_micro_rows,
    run_toy8,
    select_peer_ids,
)
from neural_abm.graphs import build_graph
from neural_abm.social import mix_scalar_probabilities


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


def test_toy8_output_average_matches_unit_scalar_parity(tmp_path: Path) -> None:
    config = load_toy8_config(
        write_config(tmp_path, tiny_config_dict(tmp_path, mixer="output_average"))
    )
    graph = build_graph(config.graph, config.agents.count, config.run.seed)
    neighbors = graph_neighbors(graph, config.agents.count)
    activation_propensities = np_array(
        [0.05, 0.3, 0.35, 0.55, 0.6, 0.65, 0.8, 0.95],
    )
    peer_ids = select_peer_ids(activation_propensities, neighbors, config)

    expected = mix_scalar_probabilities(
        activation_propensities,
        peer_ids,
        alpha=config.coordination.alpha,
        channel="activation_propensity",
        commit_mode="event_hazard_commit",
    )
    mixed, losses, update_norms = apply_toy8_output_average(
        activation_propensities,
        peer_ids,
        config,
    )

    assert mixed.tolist() == pytest.approx(expected.mixed_values.tolist())
    assert losses == pytest.approx(expected.losses)
    assert update_norms == pytest.approx(expected.update_norms)


def test_toy8_output_average_routes_through_unit_scalar_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_toy8_config(
        write_config(tmp_path, tiny_config_dict(tmp_path, mixer="output_average"))
    )
    values = np_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    peer_ids = [[1], [0, 2], [], [4], [3], [6], [5, 7], [6]]
    calls: list[dict[str, object]] = []

    def fake_apply_scalar_output_average(**kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        return SimpleNamespace(
            mix=SimpleNamespace(
                mixed_values=values + 0.01,
                update_norms=[0.01 for _ in values],
            ),
            commit=SimpleNamespace(losses=[0.02 for _ in values]),
        )

    monkeypatch.setattr(
        "neural_abm.toy_async.apply_scalar_output_average",
        fake_apply_scalar_output_average,
    )

    mixed, losses, update_norms = apply_toy8_output_average(values, peer_ids, config)

    assert len(calls) == 1
    assert calls[0]["channel"] == "activation_propensity"
    assert calls[0]["commit_mode"] == "event_hazard_commit"
    assert calls[0]["alpha"] == config.coordination.alpha
    assert calls[0]["peer_ids"] == peer_ids
    assert mixed.tolist() == pytest.approx((values + 0.01).tolist())
    assert losses == pytest.approx([0.02 for _ in values])
    assert update_norms == pytest.approx([0.01 for _ in values])


def test_toy8_rows_route_social_diagnostics_through_mapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_toy8_config(write_config(tmp_path, tiny_config_dict(tmp_path)))
    agent_count = config.agents.count
    snapshot = Toy8RateSnapshot(
        activation_rates=np_array([0.1 for _ in range(agent_count)]),
        failure_rates=np_array([0.2 for _ in range(agent_count)]),
        recovery_rates=np_array([0.3 for _ in range(agent_count)]),
        neighbor_active_fraction=np_array([0.4 for _ in range(agent_count)]),
        activation_propensities=np_array([0.5 for _ in range(agent_count)]),
        peer_ids=[[] for _ in range(agent_count)],
        social_losses=[0.1 for _ in range(agent_count)],
        social_update_norms=[0.2 for _ in range(agent_count)],
    )
    step = Toy8StepResult(
        states=np_array([0, 1, 2, 0, 1, 2, 0, 1], dtype=int),
        event_time=1.5,
        event_type="activation",
        event_agent_id=1,
        activation_events=2,
        failure_events=1,
        recovery_events=0,
        snapshot=snapshot,
    )
    aggregate_calls: list[dict[str, object]] = []
    micro_calls: list[dict[str, object]] = []

    def fake_aggregate_mapper(**kwargs: object) -> dict[str, float]:
        aggregate_calls.append(dict(kwargs))
        return {
            "mean_peer_count": 5.0,
            "mean_social_loss": 0.5,
            "mean_social_update_norm": 0.05,
        }

    def fake_micro_mapper(**kwargs: object) -> dict[str, object]:
        micro_calls.append(dict(kwargs))
        return {
            "peer_ids": [55],
            "peer_count": 1,
            "component_id": 3,
            "social_loss": 0.4,
            "social_update_norm": 0.04,
        }

    monkeypatch.setattr(
        "neural_abm.toy_async.aggregate_social_diagnostic_fields",
        fake_aggregate_mapper,
    )
    monkeypatch.setattr(
        "neural_abm.toy_async.micro_social_diagnostic_fields",
        fake_micro_mapper,
    )

    aggregate = aggregate_toy8_row(config, 3, step)
    rows = toy8_micro_rows(config, 3, step)

    assert aggregate["mean_peer_count"] == pytest.approx(5.0)
    assert aggregate["mean_social_loss"] == pytest.approx(0.5)
    assert aggregate["mean_social_update_norm"] == pytest.approx(0.05)
    assert aggregate_calls == [
        {
            "peer_ids": snapshot.peer_ids,
            "social_losses": snapshot.social_losses,
            "social_update_norms": snapshot.social_update_norms,
        }
    ]
    assert rows[0]["peer_ids"] == [55]
    assert rows[0]["peer_count"] == 1
    assert rows[0]["component_id"] == 3
    assert rows[0]["social_loss"] == pytest.approx(0.4)
    assert rows[0]["social_update_norm"] == pytest.approx(0.04)
    assert micro_calls[0]["agent_id"] == 0
    assert micro_calls[0]["peer_ids"] == snapshot.peer_ids
    assert "component_id" in micro_calls[0]


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
