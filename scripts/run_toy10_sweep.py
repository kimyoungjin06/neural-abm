#!/usr/bin/env python
"""Run Toy 10 market/ecology sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from neural_abm.config import load_toy10_config
from neural_abm.sweep import (
    SweepOutputSpec,
    add_common_sweep_args,
    build_result_row,
    iter_parameter_grid,
    make_sweep_output_helpers,
    prepare_sweep_case,
    run_sweep_from_args,
    safe_float,
    write_prepared_sweep_case_config,
)
from neural_abm.toy_market import run_toy10


SUMMARY_FIELDS = [
    "label",
    "recovery_rate",
    "extraction_cost",
    "dynamic_rewire_rate",
    "initial_price_expectation_mean",
    "initial_conservation_norm_mean",
    "social_harvest_gain",
    "social_disagreement_penalty",
    "conservation_harvest_weight",
    "coordination_mixer",
    "coordination_peer_rule",
    "alpha",
    "coordination_threshold",
    "seed",
    "epochs",
    "run_dir",
    "domain_final_resource_fraction",
    "domain_final_resource_level",
    "domain_final_market_price",
    "domain_final_market_imbalance",
    "domain_final_mean_harvest_intensity",
    "domain_final_mean_price_expectation",
    "domain_final_mean_conservation_norm",
    "domain_final_mean_payoff",
    "domain_cumulative_rewired_edge_count",
    "final_fragmentation_components",
]

GROUP_FIELDS = [
    "label",
    "recovery_rate",
    "extraction_cost",
    "dynamic_rewire_rate",
    "initial_price_expectation_mean",
    "initial_conservation_norm_mean",
    "social_harvest_gain",
    "social_disagreement_penalty",
    "conservation_harvest_weight",
    "coordination_mixer",
    "coordination_peer_rule",
    "alpha",
    "coordination_threshold",
]

GROUP_AGGREGATIONS = {
    "seeds": ("seed", "nunique"),
    "final_resource_fraction_mean": ("domain_final_resource_fraction", "mean"),
    "final_resource_fraction_std": ("domain_final_resource_fraction", "std"),
    "final_market_price_mean": ("domain_final_market_price", "mean"),
    "final_mean_harvest_intensity_mean": (
        "domain_final_mean_harvest_intensity",
        "mean",
    ),
    "final_mean_payoff_mean": ("domain_final_mean_payoff", "mean"),
    "cumulative_rewired_edge_count_mean": (
        "domain_cumulative_rewired_edge_count",
        "mean",
    ),
    "final_fragmentation_components_mean": (
        "final_fragmentation_components",
        "mean",
    ),
}

DOMAIN_METRIC_KEYS = [
    "domain_final_resource_fraction",
    "domain_final_resource_level",
    "domain_final_market_price",
    "domain_final_market_imbalance",
    "domain_final_mean_harvest_intensity",
    "domain_final_mean_price_expectation",
    "domain_final_mean_conservation_norm",
    "domain_final_mean_payoff",
    "domain_cumulative_rewired_edge_count",
]

CONFIG_VALUE_PATHS = {
    "social_harvest_gain": "policy.social_harvest_gain",
    "social_disagreement_penalty": "policy.social_disagreement_penalty",
    "conservation_harvest_weight": "policy.conservation_harvest_weight",
}

OUTPUT_SPEC = SweepOutputSpec(
    summary_fields=SUMMARY_FIELDS,
    group_fields=GROUP_FIELDS,
    aggregations=GROUP_AGGREGATIONS,
    metric_keys=DOMAIN_METRIC_KEYS,
    config_value_paths=CONFIG_VALUE_PATHS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_sweep_args(
        parser,
        base_config=Path("experiments/configs/toy10_market_ecology_baseline.yaml"),
        default_label="toy10_market_sweep_seeds01_05",
        toy_name="Toy 10",
        threshold_argument="--coordination-thresholds",
    )
    parser.add_argument(
        "--recovery-rates",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.1],
        help="Resource recovery rates to sweep.",
    )
    parser.add_argument(
        "--extraction-costs",
        type=float,
        nargs="+",
        default=[0.25, 0.3, 0.45],
        help="Harvest extraction costs to sweep.",
    )
    parser.add_argument(
        "--dynamic-rewire-rates",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1],
        help="Network churn rates to sweep.",
    )
    parser.add_argument(
        "--initial-price-expectation-means",
        type=float,
        nargs="+",
        default=[0.5],
        help="Initial price expectation means to sweep.",
    )
    parser.add_argument(
        "--initial-conservation-norm-means",
        type=float,
        nargs="+",
        default=[0.35],
        help="Initial conservation norm means to sweep.",
    )
    parser.add_argument(
        "--social-harvest-gains",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional Toy 10 social harvest gains. Defaults to the base config "
            "value."
        ),
    )
    parser.add_argument(
        "--social-disagreement-penalties",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional Toy 10 social disagreement penalties. Defaults to the "
            "base config value."
        ),
    )
    parser.add_argument(
        "--conservation-harvest-weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional Toy 10 conservation harvest weights. Defaults to the base "
            "config value."
        ),
    )
    return parser.parse_args()


def case_run_name(
    *,
    label: str,
    recovery_rate: float,
    extraction_cost: float,
    dynamic_rewire_rate: float,
    initial_price_expectation_mean: float,
    initial_conservation_norm_mean: float,
    mixer: str,
    peer_rule: str,
    effective_alpha: float,
    effective_threshold: float,
    social_harvest_gain: float | None = None,
    social_disagreement_penalty: float | None = None,
    conservation_harvest_weight: float | None = None,
) -> str:
    social_gain_for_name = (
        social_harvest_gain if social_harvest_gain is not None else 1.0
    )
    social_penalty_for_name = (
        social_disagreement_penalty
        if social_disagreement_penalty is not None
        else 0.0
    )
    conservation_weight_for_name = (
        conservation_harvest_weight
        if conservation_harvest_weight is not None
        else 0.75
    )
    return (
        f"{label}_r{safe_float(recovery_rate)}_c{safe_float(extraction_cost)}_"
        f"rw{safe_float(dynamic_rewire_rate)}_"
        f"p{safe_float(initial_price_expectation_mean)}_"
        f"n{safe_float(initial_conservation_norm_mean)}_"
        f"sg{safe_float(social_gain_for_name)}_"
        f"sp{safe_float(social_penalty_for_name)}_"
        f"cw{safe_float(conservation_weight_for_name)}_"
        f"{mixer}_{peer_rule}_a{safe_float(effective_alpha)}_"
        f"t{safe_float(effective_threshold)}"
    )


def case_config_updates(
    *,
    recovery_rate: float,
    extraction_cost: float,
    dynamic_rewire_rate: float,
    initial_price_expectation_mean: float,
    initial_conservation_norm_mean: float,
    social_harvest_gain: float | None = None,
    social_disagreement_penalty: float | None = None,
    conservation_harvest_weight: float | None = None,
) -> dict[str, object]:
    updates: dict[str, object] = {
        "domain.environment.resource_recovery_rate": recovery_rate,
        "domain.environment.extraction_cost": extraction_cost,
        "domain.environment.initial_price_expectation_mean": (
            initial_price_expectation_mean
        ),
        "domain.environment.initial_conservation_norm_mean": (
            initial_conservation_norm_mean
        ),
        "domain.network.dynamic_rewire_rate": dynamic_rewire_rate,
    }
    if social_harvest_gain is not None:
        updates["model.policy.social_harvest_gain"] = social_harvest_gain
    if social_disagreement_penalty is not None:
        updates["model.policy.social_disagreement_penalty"] = social_disagreement_penalty
    if conservation_harvest_weight is not None:
        updates["model.policy.conservation_harvest_weight"] = (
            conservation_harvest_weight
        )
    return updates


def write_case_config(
    *,
    base: dict[str, Any],
    label: str,
    recovery_rate: float,
    extraction_cost: float,
    dynamic_rewire_rate: float,
    initial_price_expectation_mean: float,
    initial_conservation_norm_mean: float,
    mixer: str,
    peer_rule: str,
    alpha: float,
    seed: int,
    epochs: int | None,
    config_dir: Path,
    social_harvest_gain: float | None = None,
    social_disagreement_penalty: float | None = None,
    conservation_harvest_weight: float | None = None,
    coordination_threshold: float = 0.0,
) -> Path:
    prepared = prepare_sweep_case(
        base=base,
        toy="toy10",
        mixer=mixer,
        peer_rule=peer_rule,
        alpha=alpha,
        seed=seed,
        epochs=epochs,
        coordination_threshold=coordination_threshold,
    )
    return write_prepared_sweep_case_config(
        prepared,
        run_name=case_run_name(
            label=label,
            recovery_rate=recovery_rate,
            extraction_cost=extraction_cost,
            dynamic_rewire_rate=dynamic_rewire_rate,
            initial_price_expectation_mean=initial_price_expectation_mean,
            initial_conservation_norm_mean=initial_conservation_norm_mean,
            mixer=mixer,
            peer_rule=peer_rule,
            effective_alpha=prepared.effective_alpha,
            effective_threshold=prepared.effective_threshold,
            social_harvest_gain=social_harvest_gain,
            social_disagreement_penalty=social_disagreement_penalty,
            conservation_harvest_weight=conservation_harvest_weight,
        ),
        updates=case_config_updates(
            recovery_rate=recovery_rate,
            extraction_cost=extraction_cost,
            dynamic_rewire_rate=dynamic_rewire_rate,
            initial_price_expectation_mean=initial_price_expectation_mean,
            initial_conservation_norm_mean=initial_conservation_norm_mean,
            social_harvest_gain=social_harvest_gain,
            social_disagreement_penalty=social_disagreement_penalty,
            conservation_harvest_weight=conservation_harvest_weight,
        ),
        config_dir=config_dir,
    )


def result_row(
    *,
    label: str,
    recovery_rate: float,
    extraction_cost: float,
    dynamic_rewire_rate: float,
    initial_price_expectation_mean: float,
    initial_conservation_norm_mean: float,
    mixer: str,
    peer_rule: str,
    alpha: float,
    coordination_threshold: float,
    seed: int,
    epochs: int,
    run_dir: Path,
    domain_final_resource_fraction: float,
    domain_final_resource_level: float,
    domain_final_market_price: float,
    domain_final_market_imbalance: float,
    domain_final_mean_harvest_intensity: float,
    domain_final_mean_price_expectation: float,
    domain_final_mean_conservation_norm: float,
    domain_final_mean_payoff: float,
    domain_cumulative_rewired_edge_count: float,
    final_fragmentation_components: int,
    social_harvest_gain: float = 1.0,
    social_disagreement_penalty: float = 0.0,
    conservation_harvest_weight: float = 0.75,
) -> dict[str, object]:
    return build_result_row(OUTPUT_SPEC.summary_fields, locals())


write_summary_csv, build_grouped_summary = make_sweep_output_helpers(OUTPUT_SPEC)


def build_domain_points(
    base: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    base_policy = base["model"]["policy"]
    social_harvest_gains = args.social_harvest_gains or [
        float(base_policy.get("social_harvest_gain", 1.0))
    ]
    social_disagreement_penalties = args.social_disagreement_penalties or [
        float(base_policy.get("social_disagreement_penalty", 0.0))
    ]
    conservation_harvest_weights = args.conservation_harvest_weights or [
        float(base_policy.get("conservation_harvest_weight", 0.75))
    ]
    return iter_parameter_grid(
        {
            "recovery_rate": args.recovery_rates,
            "extraction_cost": args.extraction_costs,
            "dynamic_rewire_rate": args.dynamic_rewire_rates,
            "initial_price_expectation_mean": args.initial_price_expectation_means,
            "initial_conservation_norm_mean": (
                args.initial_conservation_norm_means
            ),
            "social_harvest_gain": social_harvest_gains,
            "social_disagreement_penalty": social_disagreement_penalties,
            "conservation_harvest_weight": conservation_harvest_weights,
        }
    )


def main() -> None:
    args = parse_args()
    run_sweep_from_args(
        args=args,
        toy="toy10",
        output_spec=OUTPUT_SPEC,
        domain_points_builder=build_domain_points,
        thresholds=args.coordination_thresholds,
        write_case_config=write_case_config,
        load_config=load_toy10_config,
        run_case=run_toy10,
        row_builder=result_row,
    )


if __name__ == "__main__":
    main()
