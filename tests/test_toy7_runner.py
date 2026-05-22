from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from binary_config_helpers import toy7_config
from neural_abm.config import load_toy7_config
from neural_abm.social import mix_bounded_scalars
from neural_abm.toy_resource import (
    Toy7StepResult,
    adaptive_target,
    apply_output_average as apply_toy7_output_average,
    aggregate_row as aggregate_toy7_row,
    compute_payoffs,
    micro_rows as toy7_micro_rows,
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


def test_toy7_output_average_matches_unit_bounded_scalar_parity(
    tmp_path: Path,
) -> None:
    config = load_toy7_config(
        write_config(tmp_path, tiny_config_dict(tmp_path, mixer="output_average"))
    )
    propensities = np_array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
    peer_ids = [[1], [], [3, 4], [], [2], [], [7], []]

    expected = mix_bounded_scalars(
        propensities,
        peer_ids,
        alpha=config.coordination.alpha,
        lower_bound=0.0,
        upper_bound=1.0,
        channel="extraction_intensity",
        commit_mode="continuous_intensity_commit",
    )
    mixed, losses, update_norms = apply_toy7_output_average(
        propensities,
        peer_ids,
        config,
    )

    assert mixed.tolist() == pytest.approx(expected.mixed_values.tolist())
    assert losses == pytest.approx(expected.losses)
    assert update_norms == pytest.approx(expected.update_norms)


def test_toy7_output_average_routes_through_unit_bounded_scalar_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_toy7_config(
        write_config(tmp_path, tiny_config_dict(tmp_path, mixer="output_average"))
    )
    propensities = np_array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
    peer_ids = [[1], [], [3, 4], [], [2], [], [7], []]
    calls: list[dict[str, object]] = []

    def fake_apply_bounded_scalar_output_average(**kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        return SimpleNamespace(
            mix=SimpleNamespace(
                mixed_values=propensities + 0.01,
                update_norms=[0.01 for _ in propensities],
            ),
            commit=SimpleNamespace(losses=[0.02 for _ in propensities]),
        )

    monkeypatch.setattr(
        "neural_abm.toy_resource.apply_bounded_scalar_output_average",
        fake_apply_bounded_scalar_output_average,
    )

    mixed, losses, update_norms = apply_toy7_output_average(
        propensities,
        peer_ids,
        config,
    )

    assert len(calls) == 1
    assert calls[0]["channel"] == "extraction_intensity"
    assert calls[0]["commit_mode"] == "continuous_intensity_commit"
    assert calls[0]["lower_bound"] == 0.0
    assert calls[0]["upper_bound"] == 1.0
    assert calls[0]["alpha"] == config.coordination.alpha
    assert calls[0]["peer_ids"] == peer_ids
    assert mixed.tolist() == pytest.approx((propensities + 0.01).tolist())
    assert losses == pytest.approx([0.02 for _ in propensities])
    assert update_norms == pytest.approx([0.01 for _ in propensities])


def test_toy7_rows_route_social_diagnostics_through_mapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_toy7_config(write_config(tmp_path, tiny_config_dict(tmp_path)))
    agent_count = config.agents.count
    step = Toy7StepResult(
        intensities=np_array([0.1 * index for index in range(agent_count)]),
        propensities=np_array([0.2 for _ in range(agent_count)]),
        payoffs=np_array([1.0 for _ in range(agent_count)]),
        payoff_ema=np_array([0.5 for _ in range(agent_count)]),
        resource_level=50.0,
        peer_ids=[[] for _ in range(agent_count)],
        social_losses=[0.1 for _ in range(agent_count)],
        social_update_norms=[0.2 for _ in range(agent_count)],
    )
    aggregate_calls: list[dict[str, object]] = []
    micro_calls: list[dict[str, object]] = []

    def fake_aggregate_mapper(**kwargs: object) -> dict[str, float]:
        aggregate_calls.append(dict(kwargs))
        return {
            "mean_peer_count": 6.0,
            "mean_social_loss": 0.6,
            "mean_social_update_norm": 0.06,
        }

    def fake_micro_mapper(**kwargs: object) -> dict[str, object]:
        micro_calls.append(dict(kwargs))
        return {
            "peer_ids": [77],
            "peer_count": 1,
            "component_id": 4,
            "social_loss": 0.8,
            "social_update_norm": 0.08,
        }

    monkeypatch.setattr(
        "neural_abm.toy_resource.aggregate_social_diagnostic_fields",
        fake_aggregate_mapper,
    )
    monkeypatch.setattr(
        "neural_abm.toy_resource.micro_social_diagnostic_fields",
        fake_micro_mapper,
    )

    aggregate = aggregate_toy7_row(config, 3, step)
    rows = toy7_micro_rows(config, 3, step)

    assert aggregate["mean_peer_count"] == pytest.approx(6.0)
    assert aggregate["mean_social_loss"] == pytest.approx(0.6)
    assert aggregate["mean_social_update_norm"] == pytest.approx(0.06)
    assert aggregate_calls == [
        {
            "peer_ids": step.peer_ids,
            "social_losses": step.social_losses,
            "social_update_norms": step.social_update_norms,
        }
    ]
    assert rows[0]["peer_ids"] == [77]
    assert rows[0]["peer_count"] == 1
    assert rows[0]["component_id"] == 4
    assert rows[0]["social_loss"] == pytest.approx(0.8)
    assert rows[0]["social_update_norm"] == pytest.approx(0.08)
    assert micro_calls[0]["agent_id"] == 0
    assert micro_calls[0]["peer_ids"] == step.peer_ids
    assert micro_calls[0]["component_id"] == 0


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
