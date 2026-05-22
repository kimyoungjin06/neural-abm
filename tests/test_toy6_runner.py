from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from binary_config_helpers import toy6_config
from neural_abm.config import load_toy6_config
from neural_abm.social import mix_probability_distributions
from neural_abm.toy_categorical import (
    Toy6StepResult,
    apply_output_average as apply_toy6_output_average,
    aggregate_row as aggregate_toy6_row,
    compute_cyclic_payoffs,
    micro_rows as toy6_micro_rows,
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


def test_toy6_output_average_matches_unit_distribution_parity(
    tmp_path: Path,
) -> None:
    config = load_toy6_config(
        write_config(tmp_path, tiny_config_dict(tmp_path, mixer="output_average"))
    )
    probabilities = np_array(
        [
            [0.80, 0.10, 0.10],
            [0.60, 0.30, 0.10],
            [0.20, 0.30, 0.50],
            [0.20, 0.70, 0.10],
            [0.10, 0.20, 0.70],
            [0.50, 0.25, 0.25],
            [0.33, 0.34, 0.33],
            [0.10, 0.80, 0.10],
            [0.70, 0.10, 0.20],
        ]
    )
    peer_ids = [[1, 2], [], [3, 4], [], [2], [], [7], [], [0]]
    values = torch.as_tensor(probabilities, dtype=torch.float32)

    expected = mix_probability_distributions(
        values,
        peer_ids,
        alpha=config.coordination.alpha,
        channel="strategy_distribution",
        commit_mode="categorical_probability_commit",
    )
    mixed, losses, update_norms = apply_toy6_output_average(
        probabilities,
        peer_ids,
        config,
        torch.device("cpu"),
    )

    np.testing.assert_allclose(
        mixed,
        expected.mixed_values.detach().cpu().numpy(),
        atol=1e-6,
    )
    assert losses == pytest.approx(expected.losses)
    assert update_norms == pytest.approx(expected.update_norms)


def test_toy6_output_average_routes_through_unit_distribution_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_toy6_config(
        write_config(tmp_path, tiny_config_dict(tmp_path, mixer="output_average"))
    )
    probabilities = np_array(
        [
            [0.80, 0.10, 0.10],
            [0.60, 0.30, 0.10],
            [0.20, 0.30, 0.50],
            [0.20, 0.70, 0.10],
            [0.10, 0.20, 0.70],
            [0.50, 0.25, 0.25],
            [0.33, 0.34, 0.33],
            [0.10, 0.80, 0.10],
            [0.70, 0.10, 0.20],
        ]
    )
    peer_ids = [[1, 2], [], [3, 4], [], [2], [], [7], [], [0]]
    calls: list[dict[str, object]] = []

    def fake_apply_distribution_output_average(**kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        values = kwargs["values"]
        assert torch.is_tensor(values)
        mixed_values = values + torch.as_tensor([0.01, 0.0, -0.01])
        return SimpleNamespace(
            mix=SimpleNamespace(
                mixed_values=mixed_values,
                update_norms=[0.01 for _ in range(len(probabilities))],
            ),
            commit=SimpleNamespace(losses=[0.02 for _ in range(len(probabilities))]),
        )

    monkeypatch.setattr(
        "neural_abm.toy_categorical.apply_distribution_output_average",
        fake_apply_distribution_output_average,
    )

    mixed, losses, update_norms = apply_toy6_output_average(
        probabilities,
        peer_ids,
        config,
        torch.device("cpu"),
    )

    assert len(calls) == 1
    assert calls[0]["channel"] == "strategy_distribution"
    assert calls[0]["commit_mode"] == "categorical_probability_commit"
    assert calls[0]["alpha"] == config.coordination.alpha
    assert calls[0]["peer_ids"] == peer_ids
    assert mixed[0].tolist() == pytest.approx([0.81, 0.10, 0.09])
    assert losses == pytest.approx([0.02 for _ in range(len(probabilities))])
    assert update_norms == pytest.approx([0.01 for _ in range(len(probabilities))])


def test_toy6_rows_route_social_diagnostics_through_mapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_toy6_config(write_config(tmp_path, tiny_config_dict(tmp_path)))
    agent_count = config.agent_count
    step = Toy6StepResult(
        actions=np_array([0, 1, 2] * 3).astype(int),
        payoffs=np_array([0.1 * index for index in range(agent_count)]),
        probabilities=np_array([[0.6, 0.3, 0.1] for _ in range(agent_count)]),
        peer_ids=[[] for _ in range(agent_count)],
        social_losses=[0.1 for _ in range(agent_count)],
        social_update_norms=[0.2 for _ in range(agent_count)],
    )
    aggregate_calls: list[dict[str, object]] = []
    micro_calls: list[dict[str, object]] = []

    def fake_aggregate_mapper(**kwargs: object) -> dict[str, float]:
        aggregate_calls.append(dict(kwargs))
        return {
            "mean_peer_count": 7.0,
            "mean_social_loss": 0.7,
            "mean_social_update_norm": 0.07,
        }

    def fake_micro_mapper(**kwargs: object) -> dict[str, object]:
        micro_calls.append(dict(kwargs))
        return {
            "peer_ids": [99],
            "peer_count": 1,
            "component_id": 5,
            "social_loss": 0.9,
            "social_update_norm": 0.09,
        }

    monkeypatch.setattr(
        "neural_abm.toy_categorical.aggregate_social_diagnostic_fields",
        fake_aggregate_mapper,
    )
    monkeypatch.setattr(
        "neural_abm.toy_categorical.micro_social_diagnostic_fields",
        fake_micro_mapper,
    )

    aggregate = aggregate_toy6_row(config, 3, step)
    rows = toy6_micro_rows(
        config,
        3,
        step,
        payoff_ema=np_array([0.0 for _ in range(agent_count)]),
    )

    assert aggregate["mean_peer_count"] == pytest.approx(7.0)
    assert aggregate["mean_social_loss"] == pytest.approx(0.7)
    assert aggregate["mean_social_update_norm"] == pytest.approx(0.07)
    assert aggregate_calls == [
        {
            "peer_ids": step.peer_ids,
            "social_losses": step.social_losses,
            "social_update_norms": step.social_update_norms,
        }
    ]
    assert rows[0]["peer_ids"] == [99]
    assert rows[0]["peer_count"] == 1
    assert rows[0]["component_id"] == 5
    assert rows[0]["social_loss"] == pytest.approx(0.9)
    assert rows[0]["social_update_norm"] == pytest.approx(0.09)
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
