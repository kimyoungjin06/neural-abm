#!/usr/bin/env python
"""Run Toy 9 heterogeneous-agent sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from neural_abm.config import load_toy9_config
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
from neural_abm.toy_heterogeneous import run_toy9


SUMMARY_FIELDS = [
    "label",
    "threshold_group_fraction",
    "coordination_gate_mode",
    "environment_threshold",
    "benefit",
    "action_cost",
    "payoff_learning_rate",
    "coordination_mixer",
    "coordination_peer_rule",
    "alpha",
    "coordination_threshold",
    "seed",
    "epochs",
    "run_dir",
    "domain_final_action_rate",
    "domain_final_mean_action_probability",
    "domain_final_mean_payoff",
    "domain_final_payoff_variance",
    "domain_final_group_action_rate_gap",
    "domain_final_coordination_enabled_action_rate",
    "domain_final_coordination_disabled_action_rate",
    "final_fragmentation_components",
]

GROUP_FIELDS = [
    "label",
    "threshold_group_fraction",
    "coordination_gate_mode",
    "environment_threshold",
    "benefit",
    "action_cost",
    "payoff_learning_rate",
    "coordination_mixer",
    "coordination_peer_rule",
    "alpha",
    "coordination_threshold",
]

GROUP_AGGREGATIONS = {
    "seeds": ("seed", "nunique"),
    "final_action_rate_mean": ("domain_final_action_rate", "mean"),
    "final_action_rate_std": ("domain_final_action_rate", "std"),
    "final_mean_payoff_mean": ("domain_final_mean_payoff", "mean"),
    "final_mean_payoff_std": ("domain_final_mean_payoff", "std"),
    "final_group_action_rate_gap_mean": (
        "domain_final_group_action_rate_gap",
        "mean",
    ),
    "final_group_action_rate_gap_std": (
        "domain_final_group_action_rate_gap",
        "std",
    ),
    "final_fragmentation_components_mean": (
        "final_fragmentation_components",
        "mean",
    ),
}

DOMAIN_METRIC_KEYS = [
    "domain_final_action_rate",
    "domain_final_mean_action_probability",
    "domain_final_mean_payoff",
    "domain_final_payoff_variance",
    "domain_final_group_action_rate_gap",
    "domain_final_coordination_enabled_action_rate",
    "domain_final_coordination_disabled_action_rate",
]

OUTPUT_SPEC = SweepOutputSpec(
    summary_fields=SUMMARY_FIELDS,
    group_fields=GROUP_FIELDS,
    aggregations=GROUP_AGGREGATIONS,
    metric_keys=DOMAIN_METRIC_KEYS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_sweep_args(
        parser,
        base_config=Path("experiments/configs/toy9_heterogeneous_agents_baseline.yaml"),
        default_label="toy9_heterogeneous_sweep_seeds01_05",
        toy_name="Toy 9",
        threshold_argument="--coordination-thresholds",
    )
    parser.add_argument(
        "--threshold-group-fractions",
        type=float,
        nargs="+",
        default=[0.5],
        help="Fraction assigned to the threshold-rule group.",
    )
    parser.add_argument(
        "--coordination-gate-modes",
        nargs="+",
        default=["gated"],
        choices=["gated", "all_enabled", "all_disabled"],
        help="Whether coordination is group-gated, enabled for all, or disabled.",
    )
    parser.add_argument(
        "--environment-thresholds",
        type=float,
        nargs="+",
        default=[0.45],
        help="Threshold adoption targets to sweep.",
    )
    parser.add_argument(
        "--benefits",
        type=float,
        nargs="+",
        default=[1.2],
        help="Local action benefit coefficients to sweep.",
    )
    parser.add_argument(
        "--action-costs",
        type=float,
        nargs="+",
        default=[0.35],
        help="Action costs to sweep.",
    )
    parser.add_argument(
        "--payoff-learning-rates",
        type=float,
        nargs="+",
        default=[0.18],
        help="Learning rates for the payoff-learning group.",
    )
    return parser.parse_args()


def set_gate_mode(groups: list[dict[str, Any]], gate_mode: str) -> None:
    if gate_mode == "gated":
        groups[0]["coordination_enabled"] = True
        groups[1]["coordination_enabled"] = False
    elif gate_mode == "all_enabled":
        for group in groups:
            group["coordination_enabled"] = True
    elif gate_mode == "all_disabled":
        for group in groups:
            group["coordination_enabled"] = False
    else:
        raise ValueError(f"Unknown Toy 9 coordination gate mode: {gate_mode}")


def validate_threshold_group_fraction(threshold_group_fraction: float) -> None:
    if not 0.0 < threshold_group_fraction < 1.0:
        raise ValueError("threshold_group_fraction must lie in (0, 1)")


def case_run_name(
    *,
    label: str,
    threshold_group_fraction: float,
    coordination_gate_mode: str,
    environment_threshold: float,
    benefit: float,
    action_cost: float,
    payoff_learning_rate: float,
    mixer: str,
    peer_rule: str,
    effective_alpha: float,
    effective_threshold: float,
) -> str:
    return (
        f"{label}_f{safe_float(threshold_group_fraction)}_"
        f"{coordination_gate_mode}_th{safe_float(environment_threshold)}_"
        f"b{safe_float(benefit)}_c{safe_float(action_cost)}_"
        f"lr{safe_float(payoff_learning_rate)}_{mixer}_{peer_rule}_"
        f"a{safe_float(effective_alpha)}_t{safe_float(effective_threshold)}"
    )


def case_config_updates(
    *,
    threshold_group_fraction: float,
    environment_threshold: float,
    benefit: float,
    action_cost: float,
    payoff_learning_rate: float,
) -> dict[str, object]:
    return {
        "model.agents.groups.0.fraction": threshold_group_fraction,
        "model.agents.groups.1.fraction": 1.0 - threshold_group_fraction,
        "model.agents.groups.0.threshold": environment_threshold,
        "model.agents.groups.1.learning_rate": payoff_learning_rate,
        "domain.environment.threshold": environment_threshold,
        "domain.environment.benefit": benefit,
        "domain.environment.action_cost": action_cost,
    }


def write_case_config(
    *,
    base: dict[str, Any],
    label: str,
    threshold_group_fraction: float,
    coordination_gate_mode: str,
    environment_threshold: float,
    benefit: float,
    action_cost: float,
    payoff_learning_rate: float,
    mixer: str,
    peer_rule: str,
    alpha: float,
    seed: int,
    epochs: int | None,
    config_dir: Path,
    coordination_threshold: float = 0.0,
) -> Path:
    prepared = prepare_sweep_case(
        base=base,
        toy="toy9",
        mixer=mixer,
        peer_rule=peer_rule,
        alpha=alpha,
        seed=seed,
        epochs=epochs,
        coordination_threshold=coordination_threshold,
    )
    validate_threshold_group_fraction(threshold_group_fraction)

    def apply_coordination_gate(case: dict[str, Any]) -> None:
        set_gate_mode(case["model"]["agents"]["groups"], coordination_gate_mode)

    return write_prepared_sweep_case_config(
        prepared,
        run_name=case_run_name(
            label=label,
            threshold_group_fraction=threshold_group_fraction,
            coordination_gate_mode=coordination_gate_mode,
            environment_threshold=environment_threshold,
            benefit=benefit,
            action_cost=action_cost,
            payoff_learning_rate=payoff_learning_rate,
            mixer=mixer,
            peer_rule=peer_rule,
            effective_alpha=prepared.effective_alpha,
            effective_threshold=prepared.effective_threshold,
        ),
        updates=case_config_updates(
            threshold_group_fraction=threshold_group_fraction,
            environment_threshold=environment_threshold,
            benefit=benefit,
            action_cost=action_cost,
            payoff_learning_rate=payoff_learning_rate,
        ),
        mutate_case=apply_coordination_gate,
        config_dir=config_dir,
    )


def result_row(
    *,
    label: str,
    threshold_group_fraction: float,
    coordination_gate_mode: str,
    environment_threshold: float,
    benefit: float,
    action_cost: float,
    payoff_learning_rate: float,
    mixer: str,
    peer_rule: str,
    alpha: float,
    coordination_threshold: float,
    seed: int,
    epochs: int,
    run_dir: Path,
    domain_final_action_rate: float,
    domain_final_mean_action_probability: float,
    domain_final_mean_payoff: float,
    domain_final_payoff_variance: float,
    domain_final_group_action_rate_gap: float,
    domain_final_coordination_enabled_action_rate: float,
    domain_final_coordination_disabled_action_rate: float,
    final_fragmentation_components: int,
) -> dict[str, object]:
    return build_result_row(OUTPUT_SPEC.summary_fields, locals())


write_summary_csv, build_grouped_summary = make_sweep_output_helpers(OUTPUT_SPEC)


def build_domain_points(
    _base: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    return iter_parameter_grid(
        {
            "threshold_group_fraction": args.threshold_group_fractions,
            "coordination_gate_mode": args.coordination_gate_modes,
            "environment_threshold": args.environment_thresholds,
            "benefit": args.benefits,
            "action_cost": args.action_costs,
            "payoff_learning_rate": args.payoff_learning_rates,
        }
    )


def main() -> None:
    args = parse_args()
    run_sweep_from_args(
        args=args,
        toy="toy9",
        output_spec=OUTPUT_SPEC,
        domain_points_builder=build_domain_points,
        thresholds=args.coordination_thresholds,
        write_case_config=write_case_config,
        load_config=load_toy9_config,
        run_case=run_toy9,
        row_builder=result_row,
    )


if __name__ == "__main__":
    main()
