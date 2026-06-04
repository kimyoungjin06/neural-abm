from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_abm.config import (
    load_toy10_config,
    load_toy6_config,
    load_toy7_config,
    load_toy8_config,
    load_toy9_config,
)
from neural_abm.toy_async import TOY8_AGGREGATE_FIELDS, TOY8_MICRO_STATE_FIELDS, run_toy8
from neural_abm.toy_categorical import (
    TOY6_AGGREGATE_FIELDS,
    TOY6_MICRO_STATE_FIELDS,
    run_toy6,
)
from neural_abm.toy_heterogeneous import (
    TOY9_AGGREGATE_FIELDS,
    TOY9_MICRO_STATE_FIELDS,
    run_toy9,
)
from neural_abm.toy_market import (
    TOY10_AGGREGATE_FIELDS,
    TOY10_MICRO_STATE_FIELDS,
    run_toy10,
)
from neural_abm.toy_resource import TOY7_AGGREGATE_FIELDS, TOY7_MICRO_STATE_FIELDS, run_toy7
from test_toy10_runner import tiny_config_dict as toy10_config_dict
from test_toy10_runner import write_config as write_toy10_config
from test_toy6_runner import tiny_config_dict as toy6_config_dict
from test_toy6_runner import write_config as write_toy6_config
from test_toy7_runner import tiny_config_dict as toy7_config_dict
from test_toy7_runner import write_config as write_toy7_config
from test_toy8_runner import tiny_config_dict as toy8_config_dict
from test_toy8_runner import write_config as write_toy8_config
from test_toy9_runner import tiny_config_dict as toy9_config_dict
from test_toy9_runner import write_config as write_toy9_config


EXPECTED_TOY6_MICRO_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "agent_id",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_strategy",
    "domain_payoff",
    "domain_payoff_ema",
    "domain_strategy_probability",
    "domain_dominant_strategy",
    "peer_ids",
    "peer_count",
    "component_id",
    "social_loss",
    "social_update_norm",
]

EXPECTED_TOY6_AGGREGATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_mean_payoff",
    "domain_strategy_entropy",
    "domain_dominant_strategy",
    "domain_dominant_strategy_fraction",
    "fragmentation_components",
    "mean_peer_count",
    "mean_social_loss",
    "mean_social_update_norm",
]

EXPECTED_TOY7_MICRO_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "agent_id",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_intensity",
    "domain_propensity",
    "domain_payoff",
    "domain_payoff_ema",
    "domain_resource_level",
    "peer_ids",
    "peer_count",
    "component_id",
    "social_loss",
    "social_update_norm",
]

EXPECTED_TOY7_AGGREGATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_resource_level",
    "domain_resource_fraction",
    "domain_mean_intensity",
    "domain_intensity_variance",
    "domain_mean_payoff",
    "fragmentation_components",
    "mean_peer_count",
    "mean_social_loss",
    "mean_social_update_norm",
]

EXPECTED_TOY8_MICRO_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "domain_time",
    "agent_id",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_state",
    "domain_activation_rate",
    "domain_failure_rate",
    "domain_recovery_rate",
    "domain_neighbor_active_fraction",
    "domain_activation_propensity",
    "event_type",
    "event_agent_id",
    "peer_ids",
    "peer_count",
    "component_id",
    "social_loss",
    "social_update_norm",
]

EXPECTED_TOY8_AGGREGATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "domain_time",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_inactive_fraction",
    "domain_active_fraction",
    "domain_failed_fraction",
    "domain_event_type",
    "domain_event_agent_id",
    "domain_cumulative_activation_events",
    "domain_cumulative_failure_events",
    "domain_cumulative_recovery_events",
    "fragmentation_components",
    "mean_peer_count",
    "mean_activation_rate",
    "mean_failure_rate",
    "mean_recovery_rate",
    "mean_social_loss",
    "mean_social_update_norm",
]

EXPECTED_TOY9_MICRO_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "agent_id",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_agent_group",
    "domain_local_rule",
    "domain_coordination_enabled",
    "domain_action",
    "domain_action_probability",
    "domain_propensity",
    "domain_payoff",
    "domain_payoff_ema",
    "domain_neighbor_action_rate",
    "peer_ids",
    "peer_count",
    "component_id",
    "social_loss",
    "social_update_norm",
]

EXPECTED_TOY9_AGGREGATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_action_rate",
    "domain_mean_action_probability",
    "domain_mean_payoff",
    "domain_payoff_variance",
    "domain_threshold_group_action_rate",
    "domain_payoff_learning_group_action_rate",
    "domain_coordination_enabled_action_rate",
    "domain_coordination_disabled_action_rate",
    "domain_group_action_rate_gap",
    "fragmentation_components",
    "mean_peer_count",
    "mean_social_loss",
    "mean_social_update_norm",
]

EXPECTED_TOY10_MICRO_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "agent_id",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_harvest_intensity",
    "domain_price_expectation",
    "domain_conservation_norm",
    "domain_payoff",
    "domain_payoff_ema",
    "domain_market_price",
    "domain_resource_level",
    "domain_resource_fraction",
    "domain_local_price_expectation",
    "domain_local_conservation_norm",
    "peer_ids",
    "peer_count",
    "component_id",
    "social_loss",
    "social_update_norm",
]

EXPECTED_TOY10_AGGREGATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_resource_level",
    "domain_resource_fraction",
    "domain_market_price",
    "domain_market_imbalance",
    "domain_mean_harvest_intensity",
    "domain_harvest_variance",
    "domain_mean_price_expectation",
    "domain_mean_conservation_norm",
    "domain_mean_payoff",
    "domain_payoff_variance",
    "domain_cumulative_rewired_edge_count",
    "fragmentation_components",
    "mean_peer_count",
    "mean_social_loss",
    "mean_social_update_norm",
]


@dataclass(frozen=True)
class ToyArtifactContract:
    toy: str
    declared_micro_fields: Sequence[str]
    declared_aggregate_fields: Sequence[str]
    expected_micro_fields: Sequence[str]
    expected_aggregate_fields: Sequence[str]
    config_dict: Callable[[Path], dict[str, Any]]
    write_config: Callable[[Path, dict[str, Any]], Path]
    load_config: Callable[[Path], Any]
    run: Callable[[Any, Path], Any]


TOY_ARTIFACT_CONTRACTS = [
    ToyArtifactContract(
        toy="toy6",
        declared_micro_fields=TOY6_MICRO_STATE_FIELDS,
        declared_aggregate_fields=TOY6_AGGREGATE_FIELDS,
        expected_micro_fields=EXPECTED_TOY6_MICRO_FIELDS,
        expected_aggregate_fields=EXPECTED_TOY6_AGGREGATE_FIELDS,
        config_dict=toy6_config_dict,
        write_config=write_toy6_config,
        load_config=load_toy6_config,
        run=run_toy6,
    ),
    ToyArtifactContract(
        toy="toy7",
        declared_micro_fields=TOY7_MICRO_STATE_FIELDS,
        declared_aggregate_fields=TOY7_AGGREGATE_FIELDS,
        expected_micro_fields=EXPECTED_TOY7_MICRO_FIELDS,
        expected_aggregate_fields=EXPECTED_TOY7_AGGREGATE_FIELDS,
        config_dict=toy7_config_dict,
        write_config=write_toy7_config,
        load_config=load_toy7_config,
        run=run_toy7,
    ),
    ToyArtifactContract(
        toy="toy8",
        declared_micro_fields=TOY8_MICRO_STATE_FIELDS,
        declared_aggregate_fields=TOY8_AGGREGATE_FIELDS,
        expected_micro_fields=EXPECTED_TOY8_MICRO_FIELDS,
        expected_aggregate_fields=EXPECTED_TOY8_AGGREGATE_FIELDS,
        config_dict=toy8_config_dict,
        write_config=write_toy8_config,
        load_config=load_toy8_config,
        run=run_toy8,
    ),
    ToyArtifactContract(
        toy="toy9",
        declared_micro_fields=TOY9_MICRO_STATE_FIELDS,
        declared_aggregate_fields=TOY9_AGGREGATE_FIELDS,
        expected_micro_fields=EXPECTED_TOY9_MICRO_FIELDS,
        expected_aggregate_fields=EXPECTED_TOY9_AGGREGATE_FIELDS,
        config_dict=toy9_config_dict,
        write_config=write_toy9_config,
        load_config=load_toy9_config,
        run=run_toy9,
    ),
    ToyArtifactContract(
        toy="toy10",
        declared_micro_fields=TOY10_MICRO_STATE_FIELDS,
        declared_aggregate_fields=TOY10_AGGREGATE_FIELDS,
        expected_micro_fields=EXPECTED_TOY10_MICRO_FIELDS,
        expected_aggregate_fields=EXPECTED_TOY10_AGGREGATE_FIELDS,
        config_dict=toy10_config_dict,
        write_config=write_toy10_config,
        load_config=load_toy10_config,
        run=run_toy10,
    ),
]


def test_compatible_toy_declared_artifact_fields_are_stable() -> None:
    for contract in TOY_ARTIFACT_CONTRACTS:
        assert list(contract.declared_micro_fields) == list(
            contract.expected_micro_fields
        ), contract.toy
        assert list(contract.declared_aggregate_fields) == list(
            contract.expected_aggregate_fields
        ), contract.toy


def test_compatible_toy_csv_headers_match_declared_artifact_fields(
    tmp_path: Path,
) -> None:
    for contract in TOY_ARTIFACT_CONTRACTS:
        config_path = contract.write_config(
            tmp_path,
            contract.config_dict(tmp_path, mixer="output_average", peer_rule="none"),
        )
        result = contract.run(contract.load_config(config_path), config_path)

        assert _csv_fields(result.run_dir / "micro_state.csv") == list(
            contract.expected_micro_fields
        ), contract.toy
        assert _csv_fields(result.run_dir / "aggregate_metrics.csv") == list(
            contract.expected_aggregate_fields
        ), contract.toy


def _csv_fields(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])
