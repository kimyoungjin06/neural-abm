from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from binary_config_helpers import toy9_config
from neural_abm.config import load_toy9_config
from neural_abm.toy_heterogeneous import (
    Toy9StepResult,
    apply_output_average as apply_toy9_output_average,
    aggregate_row as aggregate_toy9_row,
    assign_agent_groups,
    group_counts,
    micro_rows as toy9_micro_rows,
    run_toy9,
    select_peer_ids,
)
from neural_abm.social import mix_scalar_probabilities


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


def test_toy9_output_average_matches_unit_scalar_parity(tmp_path: Path) -> None:
    config = load_toy9_config(write_config(tmp_path, tiny_config_dict(tmp_path)))
    probabilities = np.asarray([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
    peer_ids = [[1], [], [3, 4], [], [2], [], [7], []]

    expected = mix_scalar_probabilities(
        probabilities,
        peer_ids,
        alpha=config.coordination.alpha,
        channel="heterogeneous_action_probability",
        commit_mode="group_gated_probability_commit",
    )
    mixed, losses, update_norms = apply_toy9_output_average(
        probabilities,
        peer_ids,
        config,
    )

    assert mixed.tolist() == pytest.approx(expected.mixed_values.tolist())
    assert losses == pytest.approx(expected.losses)
    assert update_norms == pytest.approx(expected.update_norms)


def test_toy9_output_average_routes_through_unit_scalar_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_toy9_config(write_config(tmp_path, tiny_config_dict(tmp_path)))
    probabilities = np.asarray([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
    peer_ids = [[1], [], [3, 4], [], [2], [], [7], []]
    calls: list[dict[str, object]] = []

    def fake_apply_scalar_output_average(**kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        return SimpleNamespace(
            mix=SimpleNamespace(
                mixed_values=probabilities + 0.01,
                update_norms=[0.01 for _ in probabilities],
            ),
            commit=SimpleNamespace(losses=[0.02 for _ in probabilities]),
        )

    monkeypatch.setattr(
        "neural_abm.toy_heterogeneous.apply_scalar_output_average",
        fake_apply_scalar_output_average,
    )

    mixed, losses, update_norms = apply_toy9_output_average(
        probabilities,
        peer_ids,
        config,
    )

    assert len(calls) == 1
    assert calls[0]["channel"] == "heterogeneous_action_probability"
    assert calls[0]["commit_mode"] == "group_gated_probability_commit"
    assert calls[0]["alpha"] == config.coordination.alpha
    assert calls[0]["peer_ids"] == peer_ids
    assert mixed.tolist() == pytest.approx((probabilities + 0.01).tolist())
    assert losses == pytest.approx([0.02 for _ in probabilities])
    assert update_norms == pytest.approx([0.01 for _ in probabilities])


def test_toy9_rows_route_social_diagnostics_through_mapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_toy9_config(write_config(tmp_path, tiny_config_dict(tmp_path)))
    agent_count = config.agents.count
    group_ids = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    step = Toy9StepResult(
        actions=np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        action_probabilities=np.asarray([0.4 for _ in range(agent_count)]),
        propensities=np.asarray([0.5 for _ in range(agent_count)]),
        payoffs=np.asarray([1.0 for _ in range(agent_count)]),
        payoff_ema=np.asarray([0.5 for _ in range(agent_count)]),
        neighbor_action_rates=np.asarray([0.25 for _ in range(agent_count)]),
        peer_ids=[[] for _ in range(agent_count)],
        social_losses=[0.1 for _ in range(agent_count)],
        social_update_norms=[0.2 for _ in range(agent_count)],
        group_ids=group_ids,
        group_names=["threshold_social", "payoff_individual"],
        local_rules=["threshold", "payoff_learning"],
        coordination_enabled=np.asarray([True, False], dtype=bool),
    )
    aggregate_calls: list[dict[str, object]] = []
    micro_calls: list[dict[str, object]] = []

    def fake_aggregate_mapper(**kwargs: object) -> dict[str, float]:
        aggregate_calls.append(dict(kwargs))
        return {
            "mean_peer_count": 4.0,
            "mean_social_loss": 0.4,
            "mean_social_update_norm": 0.04,
        }

    def fake_micro_mapper(**kwargs: object) -> dict[str, object]:
        micro_calls.append(dict(kwargs))
        return {
            "peer_ids": [44],
            "peer_count": 1,
            "component_id": 2,
            "social_loss": 0.3,
            "social_update_norm": 0.03,
        }

    monkeypatch.setattr(
        "neural_abm.toy_heterogeneous.aggregate_social_diagnostic_fields",
        fake_aggregate_mapper,
    )
    monkeypatch.setattr(
        "neural_abm.toy_heterogeneous.micro_social_diagnostic_fields",
        fake_micro_mapper,
    )

    aggregate = aggregate_toy9_row(config, 3, step)
    rows = toy9_micro_rows(config, 3, step)

    assert aggregate["mean_peer_count"] == pytest.approx(4.0)
    assert aggregate["mean_social_loss"] == pytest.approx(0.4)
    assert aggregate["mean_social_update_norm"] == pytest.approx(0.04)
    assert aggregate_calls == [
        {
            "peer_ids": step.peer_ids,
            "social_losses": step.social_losses,
            "social_update_norms": step.social_update_norms,
        }
    ]
    assert rows[0]["peer_ids"] == [44]
    assert rows[0]["peer_count"] == 1
    assert rows[0]["component_id"] == 2
    assert rows[0]["social_loss"] == pytest.approx(0.3)
    assert rows[0]["social_update_norm"] == pytest.approx(0.03)
    assert micro_calls[0]["agent_id"] == 0
    assert micro_calls[0]["peer_ids"] == step.peer_ids
    assert "component_id" in micro_calls[0]


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
