#!/usr/bin/env python
"""Run Toy 7 resource-intensity sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from neural_abm.config import load_toy7_config
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
from neural_abm.toy_resource import run_toy7


SUMMARY_FIELDS = [
    "label",
    "recovery_rate",
    "extraction_cost",
    "coordination_mixer",
    "coordination_peer_rule",
    "alpha",
    "coordination_threshold",
    "initial_intensity_mean",
    "initial_intensity_std",
    "exploration_std",
    "seed",
    "epochs",
    "run_dir",
    "domain_final_resource_fraction",
    "domain_final_resource_level",
    "domain_final_mean_intensity",
    "domain_final_intensity_variance",
    "domain_final_mean_payoff",
    "final_fragmentation_components",
]

GROUP_FIELDS = [
    "label",
    "recovery_rate",
    "extraction_cost",
    "coordination_mixer",
    "coordination_peer_rule",
    "alpha",
    "coordination_threshold",
    "initial_intensity_mean",
    "initial_intensity_std",
    "exploration_std",
]

GROUP_AGGREGATIONS = {
    "seeds": ("seed", "nunique"),
    "final_resource_fraction_mean": ("domain_final_resource_fraction", "mean"),
    "final_resource_fraction_std": ("domain_final_resource_fraction", "std"),
    "final_mean_intensity_mean": ("domain_final_mean_intensity", "mean"),
    "final_mean_intensity_std": ("domain_final_mean_intensity", "std"),
    "final_mean_payoff_mean": ("domain_final_mean_payoff", "mean"),
    "final_mean_payoff_std": ("domain_final_mean_payoff", "std"),
    "final_fragmentation_components_mean": (
        "final_fragmentation_components",
        "mean",
    ),
}

DOMAIN_METRIC_KEYS = [
    "domain_final_resource_fraction",
    "domain_final_resource_level",
    "domain_final_mean_intensity",
    "domain_final_intensity_variance",
    "domain_final_mean_payoff",
]

CONFIG_VALUE_PATHS = {
    "initial_intensity_mean": "environment.initial_intensity_mean",
    "initial_intensity_std": "environment.initial_intensity_std",
    "exploration_std": "policy.exploration_std",
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
        base_config=Path("experiments/configs/toy7_resource_intensity_baseline.yaml"),
        default_label="toy7_resource_sweep_seeds01_05",
        toy_name="Toy 7",
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
        default=[0.25, 0.35, 0.5],
        help="Extraction costs to sweep.",
    )
    parser.add_argument(
        "--initial-intensity-means",
        type=float,
        nargs="+",
        default=[0.35],
        help="Initial extraction-intensity means to sweep.",
    )
    parser.add_argument(
        "--initial-intensity-stds",
        type=float,
        nargs="+",
        default=[0.05],
        help="Initial extraction-intensity standard deviations to sweep.",
    )
    parser.add_argument(
        "--exploration-stds",
        type=float,
        nargs="+",
        default=[0.02],
        help="Adaptive-intensity exploration standard deviations to sweep.",
    )
    return parser.parse_args()


def resolve_case_values(
    *,
    case: dict[str, Any],
    initial_intensity_mean: float | None,
    initial_intensity_std: float | None,
    exploration_std: float | None,
) -> dict[str, float]:
    environment = case["domain"]["environment"]
    policy = case["model"]["policy"]
    return {
        "initial_intensity_mean": (
            float(environment["initial_intensity_mean"])
            if initial_intensity_mean is None
            else initial_intensity_mean
        ),
        "initial_intensity_std": (
            float(environment["initial_intensity_std"])
            if initial_intensity_std is None
            else initial_intensity_std
        ),
        "exploration_std": (
            float(policy["exploration_std"])
            if exploration_std is None
            else exploration_std
        ),
    }


def case_run_name(
    *,
    label: str,
    recovery_rate: float,
    extraction_cost: float,
    mixer: str,
    peer_rule: str,
    effective_alpha: float,
    effective_threshold: float,
    initial_intensity_std: float,
    exploration_std: float,
) -> str:
    return (
        f"{label}_r{safe_float(recovery_rate)}_c{safe_float(extraction_cost)}_"
        f"{mixer}_{peer_rule}_a{safe_float(effective_alpha)}_"
        f"t{safe_float(effective_threshold)}_"
        f"i{safe_float(initial_intensity_std)}_"
        f"e{safe_float(exploration_std)}"
    )


def case_config_updates(
    *,
    recovery_rate: float,
    extraction_cost: float,
    initial_intensity_mean: float,
    initial_intensity_std: float,
    exploration_std: float,
) -> dict[str, object]:
    return {
        "model.policy.exploration_std": exploration_std,
        "domain.environment.resource_recovery_rate": recovery_rate,
        "domain.environment.extraction_cost": extraction_cost,
        "domain.environment.initial_intensity_mean": initial_intensity_mean,
        "domain.environment.initial_intensity_std": initial_intensity_std,
    }


def write_case_config(
    *,
    base: dict[str, Any],
    label: str,
    recovery_rate: float,
    extraction_cost: float,
    mixer: str,
    peer_rule: str,
    alpha: float,
    seed: int,
    epochs: int | None,
    config_dir: Path,
    coordination_threshold: float = 0.0,
    initial_intensity_mean: float | None = None,
    initial_intensity_std: float | None = None,
    exploration_std: float | None = None,
) -> Path:
    prepared = prepare_sweep_case(
        base=base,
        toy="toy7",
        mixer=mixer,
        peer_rule=peer_rule,
        alpha=alpha,
        seed=seed,
        epochs=epochs,
        coordination_threshold=coordination_threshold,
    )
    case = prepared.case
    resolved = resolve_case_values(
        case=case,
        initial_intensity_mean=initial_intensity_mean,
        initial_intensity_std=initial_intensity_std,
        exploration_std=exploration_std,
    )

    return write_prepared_sweep_case_config(
        prepared,
        run_name=case_run_name(
            label=label,
            recovery_rate=recovery_rate,
            extraction_cost=extraction_cost,
            mixer=mixer,
            peer_rule=peer_rule,
            effective_alpha=prepared.effective_alpha,
            effective_threshold=prepared.effective_threshold,
            initial_intensity_std=resolved["initial_intensity_std"],
            exploration_std=resolved["exploration_std"],
        ),
        updates=case_config_updates(
            recovery_rate=recovery_rate,
            extraction_cost=extraction_cost,
            **resolved,
        ),
        config_dir=config_dir,
    )


def result_row(
    *,
    label: str,
    recovery_rate: float,
    extraction_cost: float,
    mixer: str,
    peer_rule: str,
    alpha: float,
    seed: int,
    epochs: int,
    run_dir: Path,
    domain_final_resource_fraction: float,
    domain_final_resource_level: float,
    domain_final_mean_intensity: float,
    domain_final_intensity_variance: float,
    domain_final_mean_payoff: float,
    final_fragmentation_components: int,
    coordination_threshold: float = 0.0,
    initial_intensity_mean: float = 0.35,
    initial_intensity_std: float = 0.05,
    exploration_std: float = 0.02,
) -> dict[str, object]:
    return build_result_row(OUTPUT_SPEC.summary_fields, locals())


write_summary_csv, build_grouped_summary = make_sweep_output_helpers(OUTPUT_SPEC)


def build_domain_points(
    _base: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    return iter_parameter_grid(
        {
            "recovery_rate": args.recovery_rates,
            "extraction_cost": args.extraction_costs,
            "initial_intensity_mean": args.initial_intensity_means,
            "initial_intensity_std": args.initial_intensity_stds,
            "exploration_std": args.exploration_stds,
        }
    )


def main() -> None:
    args = parse_args()
    run_sweep_from_args(
        args=args,
        toy="toy7",
        output_spec=OUTPUT_SPEC,
        domain_points_builder=build_domain_points,
        thresholds=args.thresholds,
        write_case_config=write_case_config,
        load_config=load_toy7_config,
        run_case=run_toy7,
        row_builder=result_row,
    )


if __name__ == "__main__":
    main()
