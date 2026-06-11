from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from neural_abm.config import (
    load_toy1_config,
    load_toy10_config,
    load_toy2_config,
    load_toy3_config,
    load_toy4_config,
    load_toy5_config,
    load_toy6_config,
    load_toy7_config,
    load_toy8_config,
    load_toy9_config,
)
from neural_abm.toy_async import run_toy8
from neural_abm.toy_classification import run_toy1
from neural_abm.toy_contagion import run_toy5
from neural_abm.toy_categorical import run_toy6
from neural_abm.toy_heterogeneous import run_toy9
from neural_abm.toy_market import run_toy10
from neural_abm.toy_opinion import run_toy3
from neural_abm.toy_pd import run_toy2
from neural_abm.toy_public_goods import run_toy4
from neural_abm.toy_resource import run_toy7
from test_toy_public_api_contract import (
    _toy1_path,
    _toy2_path,
    _toy3_path,
    _toy4_path,
    _toy5_path,
    _toy6_path,
    _toy7_path,
    _toy8_path,
    _toy9_path,
    _toy10_path,
)


GOLDEN_METRICS: dict[str, dict[str, object]] = {
    "toy1": {
        "final_fragmentation_components": 5,
        "domain_final_mean_global_accuracy": 0.5,
        "domain_final_mean_consensus": 1.0,
    },
    "toy2": {
        "final_action_rate": 0.25,
        "final_mean_payoff": 1.6875,
        "final_fragmentation_components": 16,
        "final_mean_policy_action_probability": 0.2785697802901268,
        "final_mean_reputation": 0.413125,
        "final_reputation_dispersion": 0.4020217461468969,
        "domain_action_components": 2,
        "domain_largest_action_cluster_fraction": 0.1875,
        "domain_policy_consensus": 0.7921708270907402,
    },
    "toy3": {
        "final_fragmentation_components": 1,
        "domain_final_opinion_mean": -0.0003480854662061411,
        "domain_final_opinion_variance": 0.16218925452092303,
        "domain_final_polarization_index": 0.16218925452092303,
        "domain_final_opinion_cluster_count": 2,
        "domain_final_mean_edge_disagreement": 0.40304344600093567,
        "domain_final_largest_connected_component_fraction": 1.0,
        "domain_cumulative_rewired_edge_count": 0,
    },
    "toy4": {
        "final_action_rate": 0.1111111111111111,
        "final_mean_payoff": 0.06666666666666668,
        "final_fragmentation_components": 9,
        "final_mean_policy_action_probability": 0.03275981214311388,
        "final_mean_reputation": 0.2911111111111111,
        "final_reputation_dispersion": 0.4149282252375288,
        "domain_payoff_gini": 0.12804232804232804,
        "domain_resource_level": 10.0,
        "domain_collapse_time": None,
        "domain_exploitation_index": 0.09333333333333332,
    },
    "toy5": {
        "final_action_rate": 1.0,
        "final_mean_payoff": 0.0,
        "final_fragmentation_components": 6,
        "final_mean_policy_action_probability": 1.0,
        "final_mean_reputation": 0.33699999999999997,
        "final_reputation_dispersion": 0.3021224917148672,
        "domain_cascade_size": 6,
        "domain_time_to_50_action": 1,
        "domain_failed_cascade": False,
        "domain_largest_action_cluster_fraction": 1.0,
        "domain_mean_neighbor_action_rate": 1.0,
    },
    "toy6": {
        "final_fragmentation_components": 9,
        "domain_final_mean_payoff": 0.0,
        "domain_final_strategy_entropy": 0.965633607142825,
        "domain_final_dominant_strategy": 2,
        "domain_final_dominant_strategy_fraction": 0.4444444444444444,
    },
    "toy7": {
        "final_fragmentation_components": 8,
        "domain_final_resource_level": 75.63837399879161,
        "domain_final_resource_fraction": 0.7563837399879161,
        "domain_final_mean_intensity": 0.4628698732030991,
        "domain_final_intensity_variance": 0.0010184390464807918,
        "domain_final_mean_payoff": 0.28687091578886514,
    },
    "toy8": {
        "final_fragmentation_components": 8,
        "domain_final_time": 9.58844340304998,
        "domain_final_inactive_fraction": 0.125,
        "domain_final_active_fraction": 0.25,
        "domain_final_failed_fraction": 0.625,
        "domain_total_events": 8,
        "domain_activation_events": 4,
        "domain_failure_events": 4,
        "domain_recovery_events": 0,
        "domain_absorbed": False,
    },
    "toy9": {
        "final_fragmentation_components": 8,
        "domain_final_action_rate": 0.25,
        "domain_final_mean_action_probability": 0.36236468624999996,
        "domain_final_mean_payoff": 0.06249999999999999,
        "domain_final_payoff_variance": 0.060468749999999995,
        "domain_final_group_action_rate_gap": 0.5,
        "domain_final_coordination_enabled_action_rate": 0.5,
        "domain_final_coordination_disabled_action_rate": 0.0,
    },
    "toy10": {
        "final_fragmentation_components": 8,
        "domain_final_resource_level": 77.11749328288285,
        "domain_final_resource_fraction": 0.7711749328288284,
        "domain_final_market_price": 0.6222051262806275,
        "domain_final_market_imbalance": 0.13386612756544614,
        "domain_final_mean_harvest_intensity": 0.42108824999161776,
        "domain_final_mean_price_expectation": 0.5724763789255698,
        "domain_final_mean_conservation_norm": 0.2787600588720236,
        "domain_final_mean_payoff": 0.1328130185380532,
        "domain_cumulative_rewired_edge_count": 11,
    },
}


@pytest.mark.parametrize(
    ("toy", "path_factory", "loader", "runner"),
    [
        ("toy1", _toy1_path, load_toy1_config, run_toy1),
        ("toy2", _toy2_path, load_toy2_config, run_toy2),
        ("toy3", _toy3_path, load_toy3_config, run_toy3),
        ("toy4", _toy4_path, load_toy4_config, run_toy4),
        ("toy5", _toy5_path, load_toy5_config, run_toy5),
        ("toy6", _toy6_path, load_toy6_config, run_toy6),
        ("toy7", _toy7_path, load_toy7_config, run_toy7),
        ("toy8", _toy8_path, load_toy8_config, run_toy8),
        ("toy9", _toy9_path, load_toy9_config, run_toy9),
        ("toy10", _toy10_path, load_toy10_config, run_toy10),
    ],
)
def test_toy_tiny_golden_regression(
    tmp_path: Path,
    toy: str,
    path_factory: Callable[[Path], Path],
    loader: Callable[[Path], Any],
    runner: Callable[[Any, Path], Any],
) -> None:
    config_path = path_factory(tmp_path)
    result = runner(loader(config_path), config_path)
    summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
    flattened = {
        key: value
        for key, value in summary.items()
        if key not in {"run_dir", "domain_metrics"}
    }
    flattened.update(summary["domain_metrics"])

    for metric, expected in GOLDEN_METRICS[toy].items():
        actual = flattened[metric]
        if isinstance(expected, float):
            assert actual == pytest.approx(expected, rel=1e-8, abs=1e-10)
        else:
            assert actual == expected
