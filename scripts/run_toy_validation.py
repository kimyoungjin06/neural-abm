#!/usr/bin/env python
"""Run representative Toy 1-10 research validation scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from neural_abm.capabilities import supports_coordination
from neural_abm.config import (
    load_toy1_config,
    load_toy2_config,
    load_toy3_config,
    load_toy4_config,
    load_toy5_config,
    load_toy6_config,
    load_toy7_config,
    load_toy8_config,
    load_toy9_config,
    load_toy10_config,
)
from neural_abm.reputation import reputation_observation_extra_dim
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


RUN_FIELDS = [
    "label",
    "toy",
    "scenario",
    "seed",
    "status",
    "failed_checks",
    "config_path",
    "run_dir",
    "policy_rule",
    "coordination_mixer",
    "epochs",
    "summary_path",
    "aggregate_rows",
    "micro_rows",
]

METRIC_FIELDS = ["label", "toy", "scenario", "seed", "metric", "value"]


@dataclass(frozen=True)
class ScenarioSpec:
    toy: str
    name: str
    base_config: Path
    mutate: Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ValidationPreset:
    label: str
    seeds: list[int]
    epochs: int
    scenarios: list[str] | None


@dataclass
class ToyHandler:
    loader: Callable[[Path], Any]
    runner: Callable[[Any, Path], Any]
    key_metrics: tuple[str, ...]


@dataclass
class RunRecord:
    label: str
    toy: str
    scenario: str
    seed: int
    config_path: Path
    run_dir: Path | None
    policy_rule: str
    coordination_mixer: str
    epochs: int
    failed_checks: list[str]
    summary: dict[str, Any]
    aggregate_rows: int
    micro_rows: int

    @property
    def status(self) -> str:
        return "fail" if self.failed_checks else "pass"

    def to_row(self) -> dict[str, Any]:
        summary_path = self.run_dir / "summary.json" if self.run_dir else ""
        return {
            "label": self.label,
            "toy": self.toy,
            "scenario": self.scenario,
            "seed": self.seed,
            "status": self.status,
            "failed_checks": "; ".join(self.failed_checks),
            "config_path": str(self.config_path),
            "run_dir": "" if self.run_dir is None else str(self.run_dir),
            "policy_rule": self.policy_rule,
            "coordination_mixer": self.coordination_mixer,
            "epochs": self.epochs,
            "summary_path": str(summary_path),
            "aggregate_rows": self.aggregate_rows,
            "micro_rows": self.micro_rows,
        }


@dataclass
class ValidationResult:
    runs_path: Path
    metrics_path: Path
    report_path: Path
    records: list[RunRecord]
    metric_rows: list[dict[str, Any]]

    @property
    def failed(self) -> bool:
        return any(record.failed_checks for record in self.records)


TOY_HANDLERS: dict[str, ToyHandler] = {
    "toy1": ToyHandler(
        loader=load_toy1_config,
        runner=run_toy1,
        key_metrics=(
            "domain_final_mean_global_accuracy",
            "domain_final_mean_consensus",
            "final_fragmentation_components",
        ),
    ),
    "toy2": ToyHandler(
        loader=load_toy2_config,
        runner=run_toy2,
        key_metrics=(
            "final_action_rate",
            "final_mean_policy_action_probability",
            "final_mean_payoff",
            "final_mean_reputation",
        ),
    ),
    "toy3": ToyHandler(
        loader=load_toy3_config,
        runner=run_toy3,
        key_metrics=(
            "domain_final_polarization_index",
            "domain_final_opinion_cluster_count",
            "domain_final_mean_edge_disagreement",
            "domain_cumulative_rewired_edge_count",
        ),
    ),
    "toy4": ToyHandler(
        loader=load_toy4_config,
        runner=run_toy4,
        key_metrics=(
            "final_action_rate",
            "domain_payoff_gini",
            "domain_resource_level",
            "final_mean_reputation",
            "domain_collapse_time",
        ),
    ),
    "toy5": ToyHandler(
        loader=load_toy5_config,
        runner=run_toy5,
        key_metrics=(
            "final_action_rate",
            "domain_cascade_size",
            "final_mean_reputation",
            "domain_low_threshold_action_rate",
            "domain_high_threshold_action_rate",
        ),
    ),
    "toy6": ToyHandler(
        loader=load_toy6_config,
        runner=run_toy6,
        key_metrics=(
            "domain_final_mean_payoff",
            "domain_final_strategy_entropy",
            "domain_final_dominant_strategy_fraction",
            "final_fragmentation_components",
        ),
    ),
    "toy7": ToyHandler(
        loader=load_toy7_config,
        runner=run_toy7,
        key_metrics=(
            "domain_final_resource_fraction",
            "domain_final_mean_intensity",
            "domain_final_mean_payoff",
            "final_fragmentation_components",
        ),
    ),
    "toy8": ToyHandler(
        loader=load_toy8_config,
        runner=run_toy8,
        key_metrics=(
            "domain_final_active_fraction",
            "domain_final_failed_fraction",
            "domain_total_events",
            "domain_final_time",
        ),
    ),
    "toy9": ToyHandler(
        loader=load_toy9_config,
        runner=run_toy9,
        key_metrics=(
            "domain_final_action_rate",
            "domain_final_group_action_rate_gap",
            "domain_final_mean_payoff",
            "final_fragmentation_components",
        ),
    ),
    "toy10": ToyHandler(
        loader=load_toy10_config,
        runner=run_toy10,
        key_metrics=(
            "domain_final_resource_fraction",
            "domain_final_market_price",
            "domain_final_mean_harvest_intensity",
            "domain_cumulative_rewired_edge_count",
        ),
    ),
}


VALIDATION_PRESETS = {
    "quick": ValidationPreset(
        label="toy_validation_quick",
        seeds=[1],
        epochs=3,
        scenarios=[
            "toy1_no_social",
            "toy2_harsh_pd_neural_none",
            "toy3_hk_no_rewire",
            "toy4_static_imitation_none",
            "toy5_low_threshold_cascade",
            "toy6_categorical_output_average",
            "toy7_resource_output_average",
            "toy8_async_output_average",
            "toy9_heterogeneous_output_average",
            "toy10_market_output_average",
        ],
    ),
    "representative": ValidationPreset(
        label="toy_validation_representative_seeds01_03",
        seeds=[1, 2, 3],
        epochs=50,
        scenarios=None,
    ),
    "paper-candidate": ValidationPreset(
        label="toy_validation_paper_candidate_seeds01_05",
        seeds=[1, 2, 3, 4, 5],
        epochs=100,
        scenarios=None,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=sorted(VALIDATION_PRESETS),
        default=None,
        help=(
            "Optional validation preset. quick runs one short scenario per toy; "
            "representative matches the default full diagnostic suite; "
            "paper-candidate runs the full suite with more seeds and epochs."
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Validation label used for generated configs and result files.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to run for every scenario.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs for every generated scenario config.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("experiments/configs/generated"),
        help="Root directory for generated scenario configs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Directory for validation CSV and Markdown outputs.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Optional run output directory override for generated configs.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Optional subset of scenario names to run.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first run-level validation failure.",
    )
    return parser.parse_args()


def resolve_cli_selection(
    args: argparse.Namespace,
) -> tuple[str, list[int], int, list[str] | None]:
    preset = VALIDATION_PRESETS.get(args.preset or "representative")
    label = args.label or preset.label
    seeds = args.seeds if args.seeds is not None else preset.seeds
    epochs = args.epochs if args.epochs is not None else preset.epochs
    scenarios = args.scenarios if args.scenarios is not None else preset.scenarios
    return label, seeds, epochs, scenarios


def load_raw_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return raw


def safe_scenario_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def set_no_social(raw: dict[str, Any]) -> None:
    coordination_section(raw)["mixer"] = "none"
    coordination_section(raw)["peer_rule"] = "none"
    coordination_section(raw)["alpha"] = 0.0


def set_output_average(
    raw: dict[str, Any],
    *,
    peer_rule: str = "output_similarity",
    alpha: float = 0.25,
) -> None:
    coordination_section(raw)["mixer"] = "output_average"
    coordination_section(raw)["peer_rule"] = peer_rule
    coordination_section(raw)["alpha"] = alpha


def set_reputation_observation(raw: dict[str, Any]) -> None:
    state_section(raw)["reputation"] = {
        "enabled": True,
        "decay": 0.9,
        "peer_rule": "spatial",
        "temperature": 1.0,
        "noise": 0.0,
        "observation_mode": "self_neighbor_mean",
    }
    agents_section(raw)["model"]["input_dim"] = 6 + reputation_observation_extra_dim(
        "self_neighbor_mean"
    )


def policy_section(raw: dict[str, Any]) -> dict[str, Any]:
    if "model" in raw and "policy" in raw["model"]:
        return raw["model"]["policy"]
    if "policy" in raw:
        return raw["policy"]
    return raw["dynamics"]


def coordination_section(raw: dict[str, Any]) -> dict[str, Any]:
    if "model" in raw and "coordination" in raw["model"]:
        return raw["model"]["coordination"]
    if "coordination" in raw:
        return raw["coordination"]
    return raw["social"]


def state_section(raw: dict[str, Any]) -> dict[str, Any]:
    if "model" in raw:
        return raw["model"].setdefault("state", {})
    return raw.setdefault("state", {})


def agents_section(raw: dict[str, Any]) -> dict[str, Any]:
    if "model" in raw:
        return raw["model"]["agents"]
    return raw["agents"]


def domain_section(raw: dict[str, Any]) -> dict[str, Any]:
    return raw.setdefault("domain", {})


def environment_section(raw: dict[str, Any]) -> dict[str, Any]:
    if "domain" in raw and "environment" in raw["domain"]:
        return raw["domain"]["environment"]
    return raw["environment"]


def graph_section(raw: dict[str, Any]) -> dict[str, Any]:
    if "domain" in raw and "graph" in raw["domain"]:
        return raw["domain"]["graph"]
    return raw["graph"]


def game_section(raw: dict[str, Any]) -> dict[str, Any]:
    if "domain" in raw:
        return raw["domain"].setdefault("game", {})
    return raw["game"]


def rewiring_section(raw: dict[str, Any]) -> dict[str, Any]:
    if "domain" in raw:
        return raw["domain"]["rewiring"]
    return raw["rewiring"]


def set_update_rule(raw: dict[str, Any], rule: str) -> None:
    section = policy_section(raw)
    if "update_rule" in section:
        section["update_rule"] = rule
    else:
        section["rule"] = rule


def mutate_toy1_no_social(raw: dict[str, Any]) -> None:
    set_no_social(raw)


def mutate_toy1_output_average(raw: dict[str, Any]) -> None:
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)


def mutate_toy2_harsh_pd_neural_none(raw: dict[str, Any]) -> None:
    game_section(raw)["family"] = "prisoner_dilemma"
    game_section(raw)["payoff"] = {"T": 5.0, "R": 3.0, "P": 1.0, "S": 0.0}
    set_update_rule(raw, "neural_policy")
    set_no_social(raw)


def mutate_toy2_harsh_pd_fermi_none(raw: dict[str, Any]) -> None:
    game_section(raw)["family"] = "prisoner_dilemma"
    game_section(raw)["payoff"] = {"T": 5.0, "R": 3.0, "P": 1.0, "S": 0.0}
    set_update_rule(raw, "fermi_imitation")
    set_no_social(raw)


def mutate_toy2_harsh_pd_neural_output_average(raw: dict[str, Any]) -> None:
    game_section(raw)["family"] = "prisoner_dilemma"
    game_section(raw)["payoff"] = {"T": 5.0, "R": 3.0, "P": 1.0, "S": 0.0}
    set_update_rule(raw, "neural_policy")
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)


def mutate_toy2_harsh_pd_neural_reputation_observation_output_average(
    raw: dict[str, Any],
) -> None:
    game_section(raw)["family"] = "prisoner_dilemma"
    game_section(raw)["payoff"] = {"T": 5.0, "R": 3.0, "P": 1.0, "S": 0.0}
    set_update_rule(raw, "neural_policy")
    set_reputation_observation(raw)
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)


def mutate_toy2_harsh_pd_reputation_output_average(raw: dict[str, Any]) -> None:
    game_section(raw)["family"] = "prisoner_dilemma"
    game_section(raw)["payoff"] = {"T": 5.0, "R": 3.0, "P": 1.0, "S": 0.0}
    set_update_rule(raw, "reputation_imitation")
    state_section(raw)["reputation"] = {
        "enabled": True,
        "decay": 0.9,
        "peer_rule": "spatial",
        "temperature": 1.0,
        "noise": 0.0,
    }
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)


def mutate_toy3_hk_no_rewire(raw: dict[str, Any]) -> None:
    set_update_rule(raw, "hk")
    set_no_social(raw)
    rewiring_section(raw)["enabled"] = False
    rewiring_section(raw)["rate"] = 0.0


def mutate_toy3_hk_rewire(raw: dict[str, Any]) -> None:
    set_update_rule(raw, "hk")
    set_no_social(raw)
    rewiring_section(raw)["enabled"] = True
    rewiring_section(raw)["rate"] = 0.25
    rewiring_section(raw)["threshold"] = 0.7


def mutate_toy3_neural_output_average(raw: dict[str, Any]) -> None:
    set_update_rule(raw, "neural_policy")
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)
    rewiring_section(raw)["enabled"] = False
    rewiring_section(raw)["rate"] = 0.0


def mutate_toy4_static_imitation_none(raw: dict[str, Any]) -> None:
    set_update_rule(raw, "imitation")
    environment_section(raw)["resource_enabled"] = False
    set_no_social(raw)


def mutate_toy4_static_neural_output_average(raw: dict[str, Any]) -> None:
    set_update_rule(raw, "neural_policy")
    environment_section(raw)["resource_enabled"] = False
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)


def mutate_toy4_static_neural_reputation_observation_output_average(
    raw: dict[str, Any],
) -> None:
    set_update_rule(raw, "neural_policy")
    environment_section(raw)["resource_enabled"] = False
    set_reputation_observation(raw)
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)


def mutate_toy4_static_reputation_output_average(raw: dict[str, Any]) -> None:
    set_update_rule(raw, "reputation_imitation")
    environment_section(raw)["resource_enabled"] = False
    state_section(raw)["reputation"] = {
        "enabled": True,
        "decay": 0.9,
        "peer_rule": "spatial",
        "temperature": 1.0,
        "noise": 0.0,
    }
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)


def mutate_toy4_commons_collapse(raw: dict[str, Any]) -> None:
    set_update_rule(raw, "imitation")
    environment_section(raw)["resource_enabled"] = True
    environment_section(raw)["initial_action_probability"] = 0.0
    set_no_social(raw)


def mutate_toy5_low_threshold_cascade(raw: dict[str, Any]) -> None:
    set_update_rule(raw, "complex_threshold")
    environment_section(raw)["threshold_mode"] = "homogeneous"
    environment_section(raw)["homogeneous_threshold"] = 0.25
    environment_section(raw)["seed_selection"] = "first_agent"
    environment_section(raw)["initial_action_fraction"] = 0.01
    graph_section(raw)["k"] = 2
    graph_section(raw)["rewire_probability"] = 0.0
    set_no_social(raw)


def mutate_toy5_high_threshold_block(raw: dict[str, Any]) -> None:
    set_update_rule(raw, "complex_threshold")
    environment_section(raw)["threshold_mode"] = "homogeneous"
    environment_section(raw)["homogeneous_threshold"] = 0.75
    set_no_social(raw)


def mutate_toy5_heterogeneous_partial(raw: dict[str, Any]) -> None:
    set_update_rule(raw, "complex_threshold")
    environment_section(raw)["threshold_mode"] = "heterogeneous"
    environment_section(raw)["heterogeneous_threshold_low"] = 0.15
    environment_section(raw)["heterogeneous_threshold_high"] = 0.55
    set_no_social(raw)


def mutate_toy5_neural_output_average(raw: dict[str, Any]) -> None:
    set_update_rule(raw, "neural_policy")
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)


def mutate_toy5_neural_reputation_observation_output_average(
    raw: dict[str, Any],
) -> None:
    set_update_rule(raw, "neural_policy")
    set_reputation_observation(raw)
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)


def mutate_toy5_reputation_output_average(raw: dict[str, Any]) -> None:
    set_update_rule(raw, "reputation_imitation")
    environment_section(raw)["seed_selection"] = "first_agent"
    environment_section(raw)["initial_action_fraction"] = 0.01
    graph_section(raw)["k"] = 2
    graph_section(raw)["rewire_probability"] = 0.0
    state_section(raw)["reputation"] = {
        "enabled": True,
        "decay": 0.9,
        "peer_rule": "spatial",
        "temperature": 1.0,
        "noise": 0.0,
    }
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)


def mutate_toy6_categorical_output_average(raw: dict[str, Any]) -> None:
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)


def mutate_toy7_resource_output_average(raw: dict[str, Any]) -> None:
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)


def mutate_toy8_async_output_average(raw: dict[str, Any]) -> None:
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)


def mutate_toy9_heterogeneous_output_average(raw: dict[str, Any]) -> None:
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)


def mutate_toy10_market_output_average(raw: dict[str, Any]) -> None:
    set_output_average(raw, peer_rule="output_similarity", alpha=0.25)


def scenario_specs() -> list[ScenarioSpec]:
    config_root = Path("experiments/configs")
    toy1_base = config_root / "toy1_neural_hk_baseline.yaml"
    toy2_base = config_root / "toy2_spatial_pd_baseline.yaml"
    toy3_base = config_root / "toy3_opinion_rewiring_baseline.yaml"
    toy4_base = config_root / "toy4_public_goods_baseline.yaml"
    toy5_base = config_root / "toy5_contagion_adoption_baseline.yaml"
    toy6_base = config_root / "toy6_categorical_spatial_baseline.yaml"
    toy7_base = config_root / "toy7_resource_intensity_baseline.yaml"
    toy8_base = config_root / "toy8_async_event_baseline.yaml"
    toy9_base = config_root / "toy9_heterogeneous_agents_baseline.yaml"
    toy10_base = config_root / "toy10_market_ecology_baseline.yaml"
    return [
        ScenarioSpec("toy1", "toy1_no_social", toy1_base, mutate_toy1_no_social),
        ScenarioSpec(
            "toy1",
            "toy1_output_average",
            toy1_base,
            mutate_toy1_output_average,
        ),
        ScenarioSpec(
            "toy2",
            "toy2_harsh_pd_neural_none",
            toy2_base,
            mutate_toy2_harsh_pd_neural_none,
        ),
        ScenarioSpec(
            "toy2",
            "toy2_harsh_pd_fermi_none",
            toy2_base,
            mutate_toy2_harsh_pd_fermi_none,
        ),
        ScenarioSpec(
            "toy2",
            "toy2_harsh_pd_neural_output_average",
            toy2_base,
            mutate_toy2_harsh_pd_neural_output_average,
        ),
        ScenarioSpec(
            "toy2",
            "toy2_harsh_pd_neural_reputation_observation_output_average",
            toy2_base,
            mutate_toy2_harsh_pd_neural_reputation_observation_output_average,
        ),
        ScenarioSpec(
            "toy2",
            "toy2_harsh_pd_reputation_output_average",
            toy2_base,
            mutate_toy2_harsh_pd_reputation_output_average,
        ),
        ScenarioSpec("toy3", "toy3_hk_no_rewire", toy3_base, mutate_toy3_hk_no_rewire),
        ScenarioSpec("toy3", "toy3_hk_rewire", toy3_base, mutate_toy3_hk_rewire),
        ScenarioSpec(
            "toy3",
            "toy3_neural_output_average",
            toy3_base,
            mutate_toy3_neural_output_average,
        ),
        ScenarioSpec(
            "toy4",
            "toy4_static_imitation_none",
            toy4_base,
            mutate_toy4_static_imitation_none,
        ),
        ScenarioSpec(
            "toy4",
            "toy4_static_neural_output_average",
            toy4_base,
            mutate_toy4_static_neural_output_average,
        ),
        ScenarioSpec(
            "toy4",
            "toy4_static_neural_reputation_observation_output_average",
            toy4_base,
            mutate_toy4_static_neural_reputation_observation_output_average,
        ),
        ScenarioSpec(
            "toy4",
            "toy4_static_reputation_output_average",
            toy4_base,
            mutate_toy4_static_reputation_output_average,
        ),
        ScenarioSpec(
            "toy4",
            "toy4_commons_collapse",
            toy4_base,
            mutate_toy4_commons_collapse,
        ),
        ScenarioSpec(
            "toy5",
            "toy5_low_threshold_cascade",
            toy5_base,
            mutate_toy5_low_threshold_cascade,
        ),
        ScenarioSpec(
            "toy5",
            "toy5_high_threshold_block",
            toy5_base,
            mutate_toy5_high_threshold_block,
        ),
        ScenarioSpec(
            "toy5",
            "toy5_heterogeneous_partial",
            toy5_base,
            mutate_toy5_heterogeneous_partial,
        ),
        ScenarioSpec(
            "toy5",
            "toy5_neural_output_average",
            toy5_base,
            mutate_toy5_neural_output_average,
        ),
        ScenarioSpec(
            "toy5",
            "toy5_neural_reputation_observation_output_average",
            toy5_base,
            mutate_toy5_neural_reputation_observation_output_average,
        ),
        ScenarioSpec(
            "toy5",
            "toy5_reputation_output_average",
            toy5_base,
            mutate_toy5_reputation_output_average,
        ),
        ScenarioSpec(
            "toy6",
            "toy6_categorical_output_average",
            toy6_base,
            mutate_toy6_categorical_output_average,
        ),
        ScenarioSpec(
            "toy7",
            "toy7_resource_output_average",
            toy7_base,
            mutate_toy7_resource_output_average,
        ),
        ScenarioSpec(
            "toy8",
            "toy8_async_output_average",
            toy8_base,
            mutate_toy8_async_output_average,
        ),
        ScenarioSpec(
            "toy9",
            "toy9_heterogeneous_output_average",
            toy9_base,
            mutate_toy9_heterogeneous_output_average,
        ),
        ScenarioSpec(
            "toy10",
            "toy10_market_output_average",
            toy10_base,
            mutate_toy10_market_output_average,
        ),
    ]


def selected_specs(names: Iterable[str] | None) -> list[ScenarioSpec]:
    specs = scenario_specs()
    if names is None:
        return specs
    requested = set(names)
    selected = [spec for spec in specs if spec.name in requested]
    found = {spec.name for spec in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"Unknown validation scenario(s): {', '.join(missing)}")
    return selected


def build_scenario_raw(
    spec: ScenarioSpec,
    *,
    label: str,
    seed: int,
    epochs: int,
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    raw = deepcopy(load_raw_yaml(spec.base_config))
    spec.mutate(raw)
    raw["run"]["name"] = f"{label}_{spec.name}"
    raw["run"]["seed"] = seed
    if runs_dir is not None:
        raw["run"]["output_dir"] = str(runs_dir)
    raw["simulation"]["epochs"] = epochs
    validate_scenario_capability(spec, raw)
    return raw


def validate_scenario_capability(spec: ScenarioSpec, raw: dict[str, Any]) -> None:
    coordination = coordination_section(raw)
    mixer = str(coordination.get("mixer", ""))
    peer_rule = str(coordination.get("peer_rule", ""))
    if not supports_coordination(spec.toy, mixer, peer_rule):
        raise ValueError(
            f"{spec.name} uses unsupported coordination for {spec.toy}: "
            f"{mixer}/{peer_rule}"
        )


def write_scenario_config(
    spec: ScenarioSpec,
    *,
    label: str,
    seed: int,
    epochs: int,
    config_root: Path,
    runs_dir: Path | None = None,
) -> Path:
    raw = build_scenario_raw(
        spec,
        label=label,
        seed=seed,
        epochs=epochs,
        runs_dir=runs_dir,
    )
    config_dir = config_root / label
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{safe_scenario_name(spec.name)}_seed{seed:02d}.yaml"
    config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def policy_rule_from_raw(toy: str, raw: dict[str, Any]) -> str:
    if toy == "toy1":
        return "neural_hk"
    if "model" in raw and isinstance(raw["model"], dict):
        policy = raw["model"].get("policy", {})
        if isinstance(policy, dict):
            return str(policy.get("rule", policy.get("update_rule", "")))
    policy = raw.get("policy", {})
    if isinstance(policy, dict):
        return str(policy.get("rule", ""))
    dynamics = raw.get("dynamics", {})
    if isinstance(dynamics, dict):
        return str(dynamics.get("update_rule", ""))
    return ""


def coordination_mixer_from_raw(raw: dict[str, Any]) -> str:
    if "model" in raw and isinstance(raw["model"], dict):
        coordination = raw["model"].get("coordination", {})
        if isinstance(coordination, dict):
            return str(coordination.get("mixer", ""))
    coordination = raw.get("coordination", {})
    if isinstance(coordination, dict):
        return str(coordination.get("mixer", ""))
    social = raw.get("social", {})
    if isinstance(social, dict):
        return str(social.get("mixer", ""))
    return ""


def numeric_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def metric_is_probability_like(field: str) -> bool:
    return (
        "probability" in field
        or field.endswith("_accuracy")
        or field.endswith("_rate")
        or field.endswith("_fraction")
        or field.endswith("_consensus")
    )


def validate_numeric_field(field: str, value: Any) -> str | None:
    number = numeric_or_none(value)
    if number is None:
        return None
    if not math.isfinite(number):
        return f"{field} is not finite"
    if metric_is_probability_like(field) and not 0.0 <= number <= 1.0:
        return f"{field}={number:g} outside [0, 1]"
    return None


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_run_outputs(run_dir: Path) -> tuple[dict[str, Any], int, int, list[str]]:
    failures: list[str] = []
    summary: dict[str, Any] = {}

    summary_path = run_dir / "summary.json"
    aggregate_path = run_dir / "aggregate_metrics.csv"
    micro_path = run_dir / "micro_state.csv"

    if not summary_path.exists():
        failures.append("missing summary.json")
    else:
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                summary = loaded
                domain_metrics = summary.get("domain_metrics")
                if isinstance(domain_metrics, dict):
                    for key, value in domain_metrics.items():
                        summary.setdefault(key, value)
            else:
                failures.append("summary.json is not an object")
        except json.JSONDecodeError as exc:
            failures.append(f"summary.json is invalid JSON: {exc}")

    aggregate_rows: list[dict[str, str]] = []
    if not aggregate_path.exists():
        failures.append("missing aggregate_metrics.csv")
    else:
        aggregate_rows = read_csv_dicts(aggregate_path)
        if not aggregate_rows:
            failures.append("aggregate_metrics.csv has no data rows")

    micro_rows = 0
    if not micro_path.exists():
        failures.append("missing micro_state.csv")
    else:
        micro_rows = len(read_csv_dicts(micro_path))
        if micro_rows == 0:
            failures.append("micro_state.csv has no data rows")

    for key, value in summary.items():
        failure = validate_numeric_field(key, value)
        if failure is not None:
            failures.append(f"summary {failure}")

    for row_index, row in enumerate(aggregate_rows, start=1):
        for key, value in row.items():
            failure = validate_numeric_field(key, value)
            if failure is not None:
                failures.append(f"aggregate row {row_index}: {failure}")

    return summary, len(aggregate_rows), micro_rows, failures


def metric_rows_from_summary(record: RunRecord) -> list[dict[str, Any]]:
    rows = []
    for metric, value in sorted(record.summary.items()):
        number = numeric_or_none(value)
        if number is None:
            continue
        rows.append(
            {
                "label": record.label,
                "toy": record.toy,
                "scenario": record.scenario,
                "seed": record.seed,
                "metric": metric,
                "value": number,
            }
        )
    return rows


def scenario_records(records: list[RunRecord], scenario: str) -> list[RunRecord]:
    return [record for record in records if record.scenario == scenario]


def metric_values(
    records: list[RunRecord],
    scenario: str,
    metric: str,
) -> list[float]:
    values = []
    for record in scenario_records(records, scenario):
        if record.failed_checks:
            continue
        value = numeric_or_none(record.summary.get(metric))
        if value is not None and math.isfinite(value):
            values.append(value)
    return values


def metric_mean(
    records: list[RunRecord],
    scenario: str,
    metric: str,
) -> float | None:
    values = metric_values(records, scenario, metric)
    if not values:
        return None
    return float(sum(values) / len(values))


def add_scenario_failure(
    records: list[RunRecord],
    scenario: str,
    check: str,
) -> None:
    for record in scenario_records(records, scenario):
        if check not in record.failed_checks:
            record.failed_checks.append(check)


def require_mean_range(
    records: list[RunRecord],
    scenario: str,
    metric: str,
    lower: float,
    upper: float,
) -> None:
    mean = metric_mean(records, scenario, metric)
    if mean is None:
        add_scenario_failure(records, scenario, f"missing mean {metric}")
        return
    if not lower <= mean <= upper:
        add_scenario_failure(
            records,
            scenario,
            f"mean {metric}={mean:.6g} outside [{lower:g}, {upper:g}]",
        )


def require_mean_at_most(
    records: list[RunRecord],
    scenario: str,
    metric: str,
    upper: float,
) -> None:
    mean = metric_mean(records, scenario, metric)
    if mean is None:
        add_scenario_failure(records, scenario, f"missing mean {metric}")
        return
    if mean > upper:
        add_scenario_failure(
            records,
            scenario,
            f"mean {metric}={mean:.6g} exceeds {upper:g}",
        )


def require_mean_at_least(
    records: list[RunRecord],
    scenario: str,
    metric: str,
    lower: float,
) -> None:
    mean = metric_mean(records, scenario, metric)
    if mean is None:
        add_scenario_failure(records, scenario, f"missing mean {metric}")
        return
    if mean < lower:
        add_scenario_failure(
            records,
            scenario,
            f"mean {metric}={mean:.6g} below {lower:g}",
        )


def require_all_present(
    records: list[RunRecord],
    scenario: str,
    metric: str,
) -> None:
    present = [
        numeric_or_none(record.summary.get(metric)) is not None
        for record in scenario_records(records, scenario)
        if not record.failed_checks
    ]
    if not present or not all(present):
        add_scenario_failure(records, scenario, f"missing {metric}")


def require_resource_bounds(records: list[RunRecord]) -> None:
    for record in scenario_records(records, "toy4_commons_collapse"):
        value = numeric_or_none(record.summary.get("domain_resource_level"))
        if value is None:
            record.failed_checks.append("missing domain_resource_level")
            continue
        raw = load_raw_yaml(record.config_path)
        capacity = float(environment_section(raw)["resource_carrying_capacity"])
        if not 0.0 <= value <= capacity:
            record.failed_checks.append(
                f"domain_resource_level={value:.6g} outside [0, {capacity:g}]"
            )


def apply_directional_gates(records: list[RunRecord]) -> None:
    require_mean_range(
        records, "toy1_no_social", "domain_final_mean_global_accuracy", 0, 1
    )
    require_mean_range(records, "toy1_no_social", "domain_final_mean_consensus", 0, 1)
    require_mean_range(
        records,
        "toy1_output_average",
        "domain_final_mean_global_accuracy",
        0,
        1,
    )
    require_mean_range(
        records, "toy1_output_average", "domain_final_mean_consensus", 0, 1
    )
    toy1_no_consensus = metric_mean(
        records,
        "toy1_no_social",
        "domain_final_mean_consensus",
    )
    toy1_out_consensus = metric_mean(
        records,
        "toy1_output_average",
        "domain_final_mean_consensus",
    )
    if (
        toy1_no_consensus is not None
        and toy1_out_consensus is not None
        and toy1_out_consensus < toy1_no_consensus
    ):
        add_scenario_failure(
            records,
            "toy1_output_average",
            "mean consensus below no-social baseline",
        )
    toy1_no_acc = metric_mean(
        records, "toy1_no_social", "domain_final_mean_global_accuracy"
    )
    toy1_out_acc = metric_mean(
        records,
        "toy1_output_average",
        "domain_final_mean_global_accuracy",
    )
    if (
        toy1_no_acc is not None
        and toy1_out_acc is not None
        and toy1_out_acc < toy1_no_acc - 0.05
    ):
        add_scenario_failure(
            records,
            "toy1_output_average",
            "mean accuracy more than 0.05 below no-social baseline",
        )

    for scenario in (
        "toy2_harsh_pd_neural_none",
        "toy2_harsh_pd_fermi_none",
        "toy2_harsh_pd_neural_output_average",
        "toy2_harsh_pd_neural_reputation_observation_output_average",
        "toy2_harsh_pd_reputation_output_average",
    ):
        require_mean_range(records, scenario, "final_action_rate", 0, 1)
        require_mean_range(
            records,
            scenario,
            "final_mean_policy_action_probability",
            0,
            1,
        )
        require_mean_range(records, scenario, "final_mean_reputation", 0, 1)
    require_mean_at_most(
        records,
        "toy2_harsh_pd_neural_none",
        "final_action_rate",
        0.20,
    )

    for scenario in (
        "toy3_hk_no_rewire",
        "toy3_hk_rewire",
        "toy3_neural_output_average",
    ):
        require_mean_range(records, scenario, "domain_final_polarization_index", 0, 1)
        require_mean_at_least(
            records, scenario, "domain_final_opinion_cluster_count", 1
        )
    require_mean_at_least(
        records, "toy3_hk_no_rewire", "domain_final_opinion_cluster_count", 2
    )
    require_mean_at_least(
        records, "toy3_hk_rewire", "domain_cumulative_rewired_edge_count", 1
    )
    no_rewire_disagreement = metric_mean(
        records,
        "toy3_hk_no_rewire",
        "domain_final_mean_edge_disagreement",
    )
    rewire_disagreement = metric_mean(
        records,
        "toy3_hk_rewire",
        "domain_final_mean_edge_disagreement",
    )
    if (
        no_rewire_disagreement is not None
        and rewire_disagreement is not None
        and rewire_disagreement > no_rewire_disagreement + 0.05
    ):
        add_scenario_failure(
            records,
            "toy3_hk_rewire",
            "mean edge disagreement more than 0.05 above no-rewire baseline",
        )

    for scenario in (
        "toy4_static_imitation_none",
        "toy4_static_neural_output_average",
        "toy4_static_neural_reputation_observation_output_average",
        "toy4_static_reputation_output_average",
        "toy4_commons_collapse",
    ):
        require_mean_range(records, scenario, "final_action_rate", 0, 1)
        require_mean_at_least(records, scenario, "domain_payoff_gini", 0)
        require_mean_range(records, scenario, "final_mean_reputation", 0, 1)
    require_mean_at_most(
        records,
        "toy4_static_imitation_none",
        "final_action_rate",
        0.25,
    )
    require_all_present(records, "toy4_commons_collapse", "domain_collapse_time")
    require_resource_bounds(records)

    for scenario in (
        "toy5_low_threshold_cascade",
        "toy5_high_threshold_block",
        "toy5_heterogeneous_partial",
        "toy5_neural_output_average",
        "toy5_neural_reputation_observation_output_average",
        "toy5_reputation_output_average",
    ):
        require_mean_range(records, scenario, "final_action_rate", 0, 1)
        require_mean_at_least(records, scenario, "domain_cascade_size", 0)
        require_mean_range(records, scenario, "final_mean_reputation", 0, 1)
    require_mean_at_least(
        records,
        "toy5_low_threshold_cascade",
        "final_action_rate",
        0.90,
    )
    require_mean_at_most(
        records,
        "toy5_high_threshold_block",
        "final_action_rate",
        0.20,
    )
    require_all_present(
        records,
        "toy5_heterogeneous_partial",
        "domain_low_threshold_action_rate",
    )
    require_all_present(
        records,
        "toy5_heterogeneous_partial",
        "domain_high_threshold_action_rate",
    )

    require_mean_range(
        records,
        "toy6_categorical_output_average",
        "domain_final_strategy_entropy",
        0,
        1,
    )
    require_mean_range(
        records,
        "toy6_categorical_output_average",
        "domain_final_dominant_strategy_fraction",
        0,
        1,
    )
    require_mean_range(
        records,
        "toy7_resource_output_average",
        "domain_final_resource_fraction",
        0,
        1,
    )
    require_mean_range(
        records,
        "toy7_resource_output_average",
        "domain_final_mean_intensity",
        0,
        1,
    )
    require_mean_range(
        records,
        "toy8_async_output_average",
        "domain_final_active_fraction",
        0,
        1,
    )
    require_mean_range(
        records,
        "toy8_async_output_average",
        "domain_final_failed_fraction",
        0,
        1,
    )
    require_mean_at_least(
        records,
        "toy8_async_output_average",
        "domain_total_events",
        1,
    )
    require_mean_range(
        records,
        "toy9_heterogeneous_output_average",
        "domain_final_action_rate",
        0,
        1,
    )
    require_mean_range(
        records,
        "toy9_heterogeneous_output_average",
        "domain_final_group_action_rate_gap",
        0,
        1,
    )
    require_mean_range(
        records,
        "toy10_market_output_average",
        "domain_final_resource_fraction",
        0,
        1,
    )
    require_mean_range(
        records,
        "toy10_market_output_average",
        "domain_final_market_price",
        0,
        1,
    )
    require_mean_at_least(
        records,
        "toy10_market_output_average",
        "domain_cumulative_rewired_edge_count",
        0,
    )


def apply_gates_for_present_scenarios(records: list[RunRecord]) -> None:
    present = {record.scenario for record in records}
    all_names = {spec.name for spec in scenario_specs()}
    if present >= all_names:
        apply_directional_gates(records)
        return

    for record in records:
        if record.failed_checks:
            continue
        toy = record.toy
        if toy == "toy1":
            for metric in (
                "domain_final_mean_global_accuracy",
                "domain_final_mean_consensus",
            ):
                failure = validate_numeric_field(metric, record.summary.get(metric))
                if failure is not None:
                    record.failed_checks.append(failure)
        elif toy == "toy2":
            for metric in (
                "final_action_rate",
                "final_mean_policy_action_probability",
            ):
                failure = validate_numeric_field(metric, record.summary.get(metric))
                if failure is not None:
                    record.failed_checks.append(failure)
        elif toy == "toy3":
            for metric in ("domain_final_polarization_index",):
                failure = validate_numeric_field(metric, record.summary.get(metric))
                if failure is not None:
                    record.failed_checks.append(failure)
            cluster_count = numeric_or_none(
                record.summary.get("domain_final_opinion_cluster_count")
            )
            if cluster_count is None or cluster_count < 1:
                record.failed_checks.append(
                    "domain_final_opinion_cluster_count below 1"
                )
        elif toy == "toy4":
            for metric in ("final_action_rate",):
                failure = validate_numeric_field(metric, record.summary.get(metric))
                if failure is not None:
                    record.failed_checks.append(failure)
            gini = numeric_or_none(record.summary.get("domain_payoff_gini"))
            if gini is None or gini < 0.0:
                record.failed_checks.append("domain_payoff_gini below 0")
        elif toy == "toy5":
            failure = validate_numeric_field(
                "final_action_rate",
                record.summary.get("final_action_rate"),
            )
            if failure is not None:
                record.failed_checks.append(failure)
            cascade_size = numeric_or_none(record.summary.get("domain_cascade_size"))
            if cascade_size is None or cascade_size < 0:
                record.failed_checks.append("domain_cascade_size below 0")
        elif toy == "toy6":
            for metric in (
                "domain_final_strategy_entropy",
                "domain_final_dominant_strategy_fraction",
            ):
                failure = validate_numeric_field(metric, record.summary.get(metric))
                if failure is not None:
                    record.failed_checks.append(failure)
        elif toy == "toy7":
            for metric in (
                "domain_final_resource_fraction",
                "domain_final_mean_intensity",
            ):
                failure = validate_numeric_field(metric, record.summary.get(metric))
                if failure is not None:
                    record.failed_checks.append(failure)
        elif toy == "toy8":
            for metric in (
                "domain_final_active_fraction",
                "domain_final_failed_fraction",
            ):
                failure = validate_numeric_field(metric, record.summary.get(metric))
                if failure is not None:
                    record.failed_checks.append(failure)
            event_count = numeric_or_none(record.summary.get("domain_total_events"))
            if event_count is None or event_count < 0:
                record.failed_checks.append("domain_total_events below 0")
        elif toy == "toy9":
            for metric in (
                "domain_final_action_rate",
                "domain_final_group_action_rate_gap",
            ):
                failure = validate_numeric_field(metric, record.summary.get(metric))
                if failure is not None:
                    record.failed_checks.append(failure)
        elif toy == "toy10":
            for metric in (
                "domain_final_resource_fraction",
                "domain_final_market_price",
                "domain_final_mean_harvest_intensity",
            ):
                failure = validate_numeric_field(metric, record.summary.get(metric))
                if failure is not None:
                    record.failed_checks.append(failure)
            rewire_count = numeric_or_none(
                record.summary.get("domain_cumulative_rewired_edge_count")
            )
            if rewire_count is None or rewire_count < 0:
                record.failed_checks.append(
                    "domain_cumulative_rewired_edge_count below 0"
                )


def write_runs_csv(path: Path, records: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})


def means_by_scenario(records: list[RunRecord]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.failed_checks:
            continue
        for metric, value in record.summary.items():
            number = numeric_or_none(value)
            if number is not None and math.isfinite(number):
                grouped[record.scenario][metric].append(number)
    return {
        scenario: {
            metric: float(sum(values) / len(values))
            for metric, values in metrics.items()
            if values
        }
        for scenario, metrics in grouped.items()
    }


def fmt_mean(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4g}"


def comparison_lines(records: list[RunRecord]) -> list[str]:
    means = means_by_scenario(records)
    lines: list[str] = []

    toy1_no = means.get("toy1_no_social", {})
    toy1_out = means.get("toy1_output_average", {})
    if toy1_no or toy1_out:
        lines.append(
            "- Toy1 output_average vs no_social: consensus delta "
            f"{fmt_mean(toy1_out.get('domain_final_mean_consensus'))} - "
            f"{fmt_mean(toy1_no.get('domain_final_mean_consensus'))}; accuracy delta "
            f"{fmt_mean(toy1_out.get('domain_final_mean_global_accuracy'))} - "
            f"{fmt_mean(toy1_no.get('domain_final_mean_global_accuracy'))}."
        )

    toy2_no = means.get("toy2_harsh_pd_neural_none", {})
    toy2_out = means.get("toy2_harsh_pd_neural_output_average", {})
    if toy2_no or toy2_out:
        lines.append(
            "- Toy2 output_average diagnostic: action "
            f"{fmt_mean(toy2_out.get('final_action_rate'))} vs no-social "
            f"{fmt_mean(toy2_no.get('final_action_rate'))}."
        )

    toy3_no = means.get("toy3_hk_no_rewire", {})
    toy3_rewire = means.get("toy3_hk_rewire", {})
    if toy3_no or toy3_rewire:
        lines.append(
            "- Toy3 rewiring: edge disagreement "
            f"{fmt_mean(toy3_rewire.get('domain_final_mean_edge_disagreement'))} vs "
            f"{fmt_mean(toy3_no.get('domain_final_mean_edge_disagreement'))}; "
            "rewired edges mean "
            f"{fmt_mean(toy3_rewire.get('domain_cumulative_rewired_edge_count'))}."
        )

    toy4_static = means.get("toy4_static_imitation_none", {})
    toy4_collapse = means.get("toy4_commons_collapse", {})
    if toy4_static or toy4_collapse:
        lines.append(
            "- Toy4 commons: static imitation action "
            f"{fmt_mean(toy4_static.get('final_action_rate'))}; "
            "domain_collapse_time mean "
            f"{fmt_mean(toy4_collapse.get('domain_collapse_time'))}."
        )

    toy5_low = means.get("toy5_low_threshold_cascade", {})
    toy5_high = means.get("toy5_high_threshold_block", {})
    toy5_neural = means.get("toy5_neural_output_average", {})
    if toy5_low or toy5_high or toy5_neural:
        lines.append(
            "- Toy5 thresholds: low action "
            f"{fmt_mean(toy5_low.get('final_action_rate'))}, high action "
            f"{fmt_mean(toy5_high.get('final_action_rate'))}, neural/social "
            f"{fmt_mean(toy5_neural.get('final_action_rate'))}."
        )

    toy6 = means.get("toy6_categorical_output_average", {})
    if toy6:
        lines.append(
            "- Toy6 categorical: entropy "
            f"{fmt_mean(toy6.get('domain_final_strategy_entropy'))}, dominant "
            "strategy fraction "
            f"{fmt_mean(toy6.get('domain_final_dominant_strategy_fraction'))}."
        )

    toy7 = means.get("toy7_resource_output_average", {})
    if toy7:
        lines.append(
            "- Toy7 resource: resource fraction "
            f"{fmt_mean(toy7.get('domain_final_resource_fraction'))}, intensity "
            f"{fmt_mean(toy7.get('domain_final_mean_intensity'))}."
        )

    toy8 = means.get("toy8_async_output_average", {})
    if toy8:
        lines.append(
            "- Toy8 async events: active fraction "
            f"{fmt_mean(toy8.get('domain_final_active_fraction'))}, failed "
            f"{fmt_mean(toy8.get('domain_final_failed_fraction'))}, events "
            f"{fmt_mean(toy8.get('domain_total_events'))}."
        )

    toy9 = means.get("toy9_heterogeneous_output_average", {})
    if toy9:
        lines.append(
            "- Toy9 heterogeneous agents: action fraction "
            f"{fmt_mean(toy9.get('domain_final_action_rate'))}, group gap "
            f"{fmt_mean(toy9.get('domain_final_group_action_rate_gap'))}."
        )

    toy10 = means.get("toy10_market_output_average", {})
    if toy10:
        lines.append(
            "- Toy10 market/ecology: resource fraction "
            f"{fmt_mean(toy10.get('domain_final_resource_fraction'))}, price "
            f"{fmt_mean(toy10.get('domain_final_market_price'))}, rewired edges "
            f"{fmt_mean(toy10.get('domain_cumulative_rewired_edge_count'))}."
        )

    return lines


def write_report(path: Path, label: str, records: list[RunRecord]) -> None:
    means = means_by_scenario(records)
    specs_by_name = {spec.name: spec for spec in scenario_specs()}
    lines = [
        f"# Toy Validation Report: {label}",
        "",
        "## Pass/Fail",
        "",
        "| Toy | Scenario | Seeds | Status | Failed checks |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for scenario in sorted({record.scenario for record in records}):
        scenario_group = scenario_records(records, scenario)
        failed = [record for record in scenario_group if record.failed_checks]
        checks = sorted({check for record in failed for check in record.failed_checks})
        spec = specs_by_name.get(scenario)
        toy = spec.toy if spec else scenario_group[0].toy
        lines.append(
            "| {toy} | {scenario} | {seeds} | {status} | {checks} |".format(
                toy=toy,
                scenario=scenario,
                seeds=len({record.seed for record in scenario_group}),
                status="fail" if failed else "pass",
                checks="<br>".join(checks) if checks else "",
            )
        )

    lines.extend(["", "## Key Metric Means", ""])
    for scenario in sorted(means):
        spec = specs_by_name.get(scenario)
        handler = TOY_HANDLERS.get(spec.toy if spec else "")
        if handler is None:
            continue
        pieces = []
        for metric in handler.key_metrics:
            pieces.append(f"{metric}={fmt_mean(means[scenario].get(metric))}")
        lines.append(f"- {scenario}: " + ", ".join(pieces))

    lines.extend(["", "## Directional Comparisons", ""])
    comparison_summary = comparison_lines(records)
    lines.extend(comparison_summary if comparison_summary else ["- n/a"])
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_one(
    spec: ScenarioSpec,
    *,
    label: str,
    seed: int,
    epochs: int,
    config_dir: Path,
    runs_dir: Path | None,
) -> RunRecord:
    config_path = write_scenario_config(
        spec,
        label=label,
        seed=seed,
        epochs=epochs,
        config_root=config_dir,
        runs_dir=runs_dir,
    )
    raw = load_raw_yaml(config_path)
    handler = TOY_HANDLERS[spec.toy]
    failed_checks: list[str] = []
    summary: dict[str, Any] = {}
    aggregate_rows = 0
    micro_rows = 0
    run_dir: Path | None = None

    try:
        config = handler.loader(config_path)
        result = handler.runner(config, config_path)
        run_dir = Path(result.run_dir)
        summary, aggregate_rows, micro_rows, failed_checks = validate_run_outputs(
            run_dir
        )
    except Exception as exc:  # noqa: BLE001 - keep validation moving across runs.
        failed_checks.append(f"runner error: {type(exc).__name__}: {exc}")

    return RunRecord(
        label=label,
        toy=spec.toy,
        scenario=spec.name,
        seed=seed,
        config_path=config_path,
        run_dir=run_dir,
        policy_rule=policy_rule_from_raw(spec.toy, raw),
        coordination_mixer=coordination_mixer_from_raw(raw),
        epochs=epochs,
        failed_checks=failed_checks,
        summary=summary,
        aggregate_rows=aggregate_rows,
        micro_rows=micro_rows,
    )


def finalize_outputs(
    *,
    label: str,
    results_dir: Path,
    records: list[RunRecord],
) -> ValidationResult:
    apply_gates_for_present_scenarios(records)
    metric_rows = [
        row for record in records for row in metric_rows_from_summary(record)
    ]
    runs_path = results_dir / f"{label}_runs.csv"
    metrics_path = results_dir / f"{label}_metrics.csv"
    report_path = results_dir / f"{label}_report.md"
    write_runs_csv(runs_path, records)
    write_metrics_csv(metrics_path, metric_rows)
    write_report(report_path, label, records)
    return ValidationResult(
        runs_path=runs_path,
        metrics_path=metrics_path,
        report_path=report_path,
        records=records,
        metric_rows=metric_rows,
    )


def run_validation(
    *,
    label: str,
    seeds: list[int],
    epochs: int,
    config_dir: Path,
    results_dir: Path,
    runs_dir: Path | None = None,
    scenario_names: list[str] | None = None,
    stop_on_failure: bool = False,
) -> ValidationResult:
    records: list[RunRecord] = []
    for spec in selected_specs(scenario_names):
        for seed in seeds:
            record = run_one(
                spec,
                label=label,
                seed=seed,
                epochs=epochs,
                config_dir=config_dir,
                runs_dir=runs_dir,
            )
            records.append(record)
            print(
                f"{record.status} {record.scenario} seed={seed} "
                f"run_dir={record.run_dir or 'n/a'}"
            )
            if stop_on_failure and record.failed_checks:
                return finalize_outputs(
                    label=label,
                    results_dir=results_dir,
                    records=records,
                )
    return finalize_outputs(label=label, results_dir=results_dir, records=records)


def main() -> int:
    args = parse_args()
    label, seeds, epochs, scenarios = resolve_cli_selection(args)
    result = run_validation(
        label=label,
        seeds=seeds,
        epochs=epochs,
        config_dir=args.config_dir,
        results_dir=args.results_dir,
        runs_dir=args.runs_dir,
        scenario_names=scenarios,
        stop_on_failure=args.stop_on_failure,
    )
    print(f"runs_csv={result.runs_path}")
    print(f"metrics_csv={result.metrics_path}")
    print(f"report_md={result.report_path}")
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
