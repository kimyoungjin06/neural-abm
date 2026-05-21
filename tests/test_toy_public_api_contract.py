from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from neural_abm.capabilities import toy_capability
from neural_abm.results import DomainToyResult
from neural_abm.spatial_binary import BinaryToyResult
from neural_abm.toy_classification import run_toy1
from neural_abm.toy_contagion import run_toy5
from neural_abm.toy_categorical import run_toy6
from neural_abm.toy_opinion import run_toy3
from neural_abm.toy_pd import run_toy2
from neural_abm.toy_public_goods import run_toy4
from neural_abm.toy_resource import run_toy7
from neural_abm.toy_async import run_toy8
from neural_abm.toy_heterogeneous import run_toy9
from neural_abm.toy_market import run_toy10
from test_toy1_runner import write_tiny_config as write_toy1_config
from test_toy2_runner import write_tiny_config as write_toy2_config
from test_toy3_runner import tiny_config_dict as toy3_config_dict
from test_toy3_runner import write_config as write_toy3_config
from test_toy4_runner import tiny_config_dict as toy4_config_dict
from test_toy4_runner import write_config as write_toy4_config
from test_toy5_runner import tiny_config_dict as toy5_config_dict
from test_toy5_runner import write_config as write_toy5_config
from test_toy6_runner import tiny_config_dict as toy6_config_dict
from test_toy6_runner import write_config as write_toy6_config
from test_toy7_runner import tiny_config_dict as toy7_config_dict
from test_toy7_runner import write_config as write_toy7_config
from test_toy8_runner import tiny_config_dict as toy8_config_dict
from test_toy8_runner import write_config as write_toy8_config
from test_toy9_runner import tiny_config_dict as toy9_config_dict
from test_toy9_runner import write_config as write_toy9_config
from test_toy10_runner import tiny_config_dict as toy10_config_dict
from test_toy10_runner import write_config as write_toy10_config

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


TOP_LEVEL_FIELDS = {"run", "simulation", "model", "domain", "logging"}
BINARY_SUMMARY_FIELDS = {
    "run_dir",
    "toy",
    "nabm_status",
    "neural_role",
    "social_channels",
    "reference_policies",
    "final_action_rate",
    "final_mean_payoff",
    "final_fragmentation_components",
    "final_mean_policy_action_probability",
    "final_mean_reputation",
    "final_reputation_dispersion",
    "domain_metrics",
}
DOMAIN_SUMMARY_FIELDS = {
    "run_dir",
    "toy",
    "nabm_status",
    "neural_role",
    "social_channels",
    "reference_policies",
    "final_fragmentation_components",
    "domain_metrics",
}
LEGACY_PUBLIC_FIELDS = {
    "agents",
    "data",
    "dynamics",
    "environment",
    "graph",
    "policy",
    "coordination",
    "state",
    "social",
    "rewiring",
    "mixer",
    "peer_rule",
    "init_mode",
    "shard_group",
    "update_rule",
    "cooperation_rate",
    "contribution_rate",
    "adoption_rate",
    "final_cooperation_rate",
    "final_contribution_rate",
    "final_adoption_rate",
    "final_mean_global_accuracy",
    "final_mean_consensus",
    "final_opinion_mean",
    "final_polarization_index",
    "final_opinion_cluster_count",
    "final_mean_edge_disagreement",
    "final_connected_components",
    "final_largest_connected_component_fraction",
    "mean_global_accuracy",
    "mean_probe_accuracy",
    "mean_consensus",
    "mean_output_js",
    "output_js_to_population_mean",
    "neural_acceptance_probability_pre_social",
    "neural_acceptance_probability_post_social",
}


def _read_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        rows = list(reader)
    assert rows
    return reader.fieldnames


def _toy1_path(tmp_path: Path) -> Path:
    return write_toy1_config(tmp_path, mixer="none", peer_rule="none")


def _toy2_path(tmp_path: Path) -> Path:
    return write_toy2_config(
        tmp_path,
        mixer="none",
        peer_rule="none",
        update_rule="fermi_imitation",
        epochs=2,
    )


def _toy3_path(tmp_path: Path) -> Path:
    return write_toy3_config(
        tmp_path,
        toy3_config_dict(
            tmp_path,
            update_rule="hk",
            mixer="none",
            rewiring_enabled=False,
            rewiring_rate=0.0,
        ),
    )


def _toy4_path(tmp_path: Path) -> Path:
    return write_toy4_config(
        tmp_path,
        toy4_config_dict(tmp_path, update_rule="imitation", mixer="none"),
    )


def _toy5_path(tmp_path: Path) -> Path:
    return write_toy5_config(
        tmp_path,
        toy5_config_dict(
            tmp_path,
            update_rule="complex_threshold",
            mixer="none",
            domain_threshold=0.5,
        ),
    )


def _toy6_path(tmp_path: Path) -> Path:
    return write_toy6_config(
        tmp_path,
        toy6_config_dict(tmp_path, mixer="none", peer_rule="none"),
    )


def _toy7_path(tmp_path: Path) -> Path:
    return write_toy7_config(
        tmp_path,
        toy7_config_dict(tmp_path, mixer="none", peer_rule="none"),
    )


def _toy8_path(tmp_path: Path) -> Path:
    return write_toy8_config(
        tmp_path,
        toy8_config_dict(tmp_path, mixer="none", peer_rule="none"),
    )


def _toy9_path(tmp_path: Path) -> Path:
    return write_toy9_config(
        tmp_path,
        toy9_config_dict(tmp_path, mixer="none", peer_rule="none"),
    )


def _toy10_path(tmp_path: Path) -> Path:
    return write_toy10_config(
        tmp_path,
        toy10_config_dict(tmp_path, mixer="none", peer_rule="none"),
    )


@pytest.mark.parametrize(
    ("toy", "path_factory", "loader", "runner", "result_type", "summary_fields"),
    [
        (
            "toy1",
            _toy1_path,
            load_toy1_config,
            run_toy1,
            DomainToyResult,
            DOMAIN_SUMMARY_FIELDS,
        ),
        (
            "toy2",
            _toy2_path,
            load_toy2_config,
            run_toy2,
            BinaryToyResult,
            BINARY_SUMMARY_FIELDS,
        ),
        (
            "toy3",
            _toy3_path,
            load_toy3_config,
            run_toy3,
            DomainToyResult,
            DOMAIN_SUMMARY_FIELDS,
        ),
        (
            "toy4",
            _toy4_path,
            load_toy4_config,
            run_toy4,
            BinaryToyResult,
            BINARY_SUMMARY_FIELDS,
        ),
        (
            "toy5",
            _toy5_path,
            load_toy5_config,
            run_toy5,
            BinaryToyResult,
            BINARY_SUMMARY_FIELDS,
        ),
        (
            "toy6",
            _toy6_path,
            load_toy6_config,
            run_toy6,
            DomainToyResult,
            DOMAIN_SUMMARY_FIELDS,
        ),
        (
            "toy7",
            _toy7_path,
            load_toy7_config,
            run_toy7,
            DomainToyResult,
            DOMAIN_SUMMARY_FIELDS,
        ),
        (
            "toy8",
            _toy8_path,
            load_toy8_config,
            run_toy8,
            DomainToyResult,
            DOMAIN_SUMMARY_FIELDS,
        ),
        (
            "toy9",
            _toy9_path,
            load_toy9_config,
            run_toy9,
            DomainToyResult,
            DOMAIN_SUMMARY_FIELDS,
        ),
        (
            "toy10",
            _toy10_path,
            load_toy10_config,
            run_toy10,
            DomainToyResult,
            DOMAIN_SUMMARY_FIELDS,
        ),
    ],
)
def test_toy_public_api_contract(
    tmp_path: Path,
    toy: str,
    path_factory: Callable[[Path], Path],
    loader: Callable[[Path], Any],
    runner: Callable[[Any, Path], DomainToyResult | BinaryToyResult],
    result_type: type[DomainToyResult] | type[BinaryToyResult],
    summary_fields: set[str],
) -> None:
    config_path = path_factory(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert set(raw) == TOP_LEVEL_FIELDS
    assert raw["domain"]["toy"] == toy
    assert not LEGACY_PUBLIC_FIELDS.intersection(set(raw))

    result = runner(loader(config_path), config_path)
    assert isinstance(result, result_type)
    assert result.toy == toy
    assert result.run_dir.exists()
    assert isinstance(result.final_fragmentation_components, int)
    assert isinstance(result.domain_metrics, dict)
    assert result.domain_metrics
    assert all(key.startswith("domain_") for key in result.domain_metrics)

    summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == summary_fields
    assert summary["toy"] == toy
    capability = toy_capability(toy)
    assert summary["nabm_status"] == capability.nabm_status
    assert summary["neural_role"] == capability.neural_role
    assert summary["social_channels"] == list(capability.social_channels)
    assert summary["reference_policies"] == list(capability.reference_policies)
    assert Path(summary["run_dir"]) == result.run_dir
    assert (
        summary["final_fragmentation_components"]
        == result.final_fragmentation_components
    )
    assert all(key.startswith("domain_") for key in summary["domain_metrics"])

    metadata = json.loads((result.run_dir / "metadata.json").read_text("utf-8"))
    assert metadata["nabm_status"] == capability.nabm_status
    assert metadata["neural_role"] == capability.neural_role
    assert metadata["social_channels"] == list(capability.social_channels)
    assert metadata["reference_policies"] == list(capability.reference_policies)

    aggregate_fields = set(_read_fieldnames(result.run_dir / "aggregate_metrics.csv"))
    micro_fields = set(_read_fieldnames(result.run_dir / "micro_state.csv"))
    for fields in (aggregate_fields, micro_fields):
        assert not LEGACY_PUBLIC_FIELDS.intersection(fields)
        assert {"run_id", "seed", "epoch"}.issubset(fields)

    if isinstance(result, BinaryToyResult):
        assert {
            "toy",
            "policy_rule",
            "coordination_mixer",
            "coordination_peer_rule",
        }.issubset(aggregate_fields)
        assert {"action_rate", "mean_policy_action_probability"}.issubset(
            aggregate_fields
        )
        assert {"action", "action_probability"}.issubset(micro_fields)
    else:
        assert {"coordination_mixer", "coordination_peer_rule"}.issubset(
            aggregate_fields
        )
