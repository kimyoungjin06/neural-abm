#!/usr/bin/env python
"""Run Toy 3 opinion-dynamics sweeps."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_abm.config import load_toy3_config
from neural_abm.sweep import (
    CoordinationSweepPoint,
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
from neural_abm.toy_opinion import run_toy3


SUMMARY_FIELDS = [
    "label",
    "policy_rule",
    "coordination_mixer",
    "coordination_peer_rule",
    "seed",
    "domain_confidence_threshold",
    "domain_rewiring_enabled",
    "domain_rewiring_threshold",
    "domain_rewiring_rate",
    "alpha",
    "coordination_threshold",
    "run_dir",
    "domain_final_opinion_mean",
    "domain_final_polarization_index",
    "domain_final_opinion_cluster_count",
    "domain_final_mean_edge_disagreement",
    "final_fragmentation_components",
    "domain_final_largest_connected_component_fraction",
    "domain_cumulative_rewired_edge_count",
]

GROUP_FIELDS = [
    "label",
    "policy_rule",
    "coordination_mixer",
    "coordination_peer_rule",
    "alpha",
    "coordination_threshold",
    "domain_confidence_threshold",
    "domain_rewiring_rate",
    "domain_rewiring_threshold",
]

GROUP_AGGREGATIONS = {
    "seeds": ("seed", "nunique"),
    "final_polarization_index_mean": ("domain_final_polarization_index", "mean"),
    "final_polarization_index_std": ("domain_final_polarization_index", "std"),
    "final_opinion_cluster_count_mean": (
        "domain_final_opinion_cluster_count",
        "mean",
    ),
    "final_mean_edge_disagreement_mean": (
        "domain_final_mean_edge_disagreement",
        "mean",
    ),
    "final_connected_components_mean": ("final_fragmentation_components", "mean"),
    "cumulative_rewired_edge_count_mean": (
        "domain_cumulative_rewired_edge_count",
        "mean",
    ),
}

DOMAIN_METRIC_KEYS = [
    "domain_final_opinion_mean",
    "domain_final_polarization_index",
    "domain_final_opinion_cluster_count",
    "domain_final_mean_edge_disagreement",
    "domain_final_largest_connected_component_fraction",
    "domain_cumulative_rewired_edge_count",
]

CONFIG_VALUE_PATHS = {
    "confidence_threshold": "policy.confidence_threshold",
    "rewiring_enabled": "rewiring.enabled",
    "rewiring_threshold": "rewiring.threshold",
    "rewiring_rate": "rewiring.rate",
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
        base_config=Path("experiments/configs/toy3_opinion_rewiring_baseline.yaml"),
        default_label="toy3_opinion_rewiring_sweep_seeds01_05",
        toy_name="Toy 3",
        default_alphas=None,
        peer_rule_choices=("none", "bounded_confidence", "output_similarity"),
        peer_rules_help=(
            "Optional peer rules to sweep. Defaults to none for mixer=none and "
            "bounded_confidence for output_average."
        ),
    )
    parser.add_argument(
        "--update-rules",
        nargs="+",
        default=["hk", "deffuant", "neural_policy"],
        choices=["hk", "deffuant", "neural_policy"],
        help="Opinion update rules to run.",
    )
    parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.35, 0.5],
        help="Bounded-confidence thresholds.",
    )
    parser.add_argument(
        "--rewiring-rates",
        type=float,
        nargs="+",
        default=[0.0, 0.25],
        help="Per-edge rewiring rates; 0 disables rewiring.",
    )
    parser.add_argument(
        "--rewiring-threshold",
        type=float,
        default=None,
        help="Opinion-disagreement threshold for edge rewiring.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Social influence strength for output_average.",
    )
    return parser.parse_args()


def resolve_peer_rule(mixer: str, peer_rule: str | None) -> str:
    if peer_rule is not None:
        return peer_rule
    if mixer == "output_average":
        return "bounded_confidence"
    return "none"


def case_run_name(
    *,
    label: str,
    update_rule: str,
    mixer: str,
    peer_rule: str,
    confidence_threshold: float,
    rewiring_rate: float,
    effective_alpha: float,
    effective_threshold: float,
) -> str:
    return (
        f"{label}_{update_rule}_{mixer}_{peer_rule}"
        f"_a{safe_float(effective_alpha)}_t{safe_float(effective_threshold)}"
        f"_c{safe_float(confidence_threshold)}_r{safe_float(rewiring_rate)}"
    )


def case_config_updates(
    *,
    update_rule: str,
    confidence_threshold: float,
    rewiring_rate: float,
    rewiring_threshold: float,
) -> dict[str, object]:
    return {
        "model.policy.update_rule": update_rule,
        "model.policy.confidence_threshold": confidence_threshold,
        "domain.rewiring.enabled": rewiring_rate > 0.0,
        "domain.rewiring.rate": rewiring_rate,
        "domain.rewiring.threshold": rewiring_threshold,
    }


def write_case_config(
    base: dict[str, Any],
    label: str,
    update_rule: str,
    mixer: str,
    seed: int,
    confidence_threshold: float,
    rewiring_rate: float,
    rewiring_threshold: float,
    alpha: float,
    epochs: int | None,
    config_dir: Path,
    peer_rule: str | None = None,
    coordination_threshold: float = 0.0,
) -> Path:
    resolved_peer_rule = resolve_peer_rule(mixer, peer_rule)
    prepared = prepare_sweep_case(
        base=base,
        toy="toy3",
        mixer=mixer,
        peer_rule=resolved_peer_rule,
        alpha=alpha,
        seed=seed,
        epochs=epochs,
        coordination_threshold=coordination_threshold,
    )
    case = prepared.case
    base_coordination = base.get("model", {}).get("coordination", {})
    if coordination_threshold == 0.0 and "threshold" in base_coordination:
        case["model"]["coordination"]["threshold"] = base_coordination["threshold"]
    effective_threshold = float(case["model"]["coordination"].get("threshold", 0.0))
    return write_prepared_sweep_case_config(
        prepared,
        run_name=case_run_name(
            label=label,
            update_rule=update_rule,
            mixer=mixer,
            peer_rule=resolved_peer_rule,
            confidence_threshold=confidence_threshold,
            rewiring_rate=rewiring_rate,
            effective_alpha=prepared.effective_alpha,
            effective_threshold=effective_threshold,
        ),
        updates=case_config_updates(
            update_rule=update_rule,
            confidence_threshold=confidence_threshold,
            rewiring_rate=rewiring_rate,
            rewiring_threshold=rewiring_threshold,
        ),
        config_dir=config_dir,
    )


def result_row(
    label: str,
    update_rule: str,
    mixer: str,
    peer_rule: str,
    seed: int,
    confidence_threshold: float,
    rewiring_enabled: bool,
    rewiring_threshold: float,
    rewiring_rate: float,
    alpha: float,
    run_dir: Path,
    domain_final_opinion_mean: float,
    domain_final_polarization_index: float,
    domain_final_opinion_cluster_count: int,
    domain_final_mean_edge_disagreement: float,
    final_fragmentation_components: int,
    domain_final_largest_connected_component_fraction: float,
    domain_cumulative_rewired_edge_count: int,
    epochs: int | None = None,
    coordination_threshold: float = 0.0,
) -> dict[str, object]:
    return build_result_row(
        OUTPUT_SPEC.summary_fields,
        locals(),
        aliases={
            "policy_rule": "update_rule",
            "domain_confidence_threshold": "confidence_threshold",
            "domain_rewiring_enabled": "rewiring_enabled",
            "domain_rewiring_threshold": "rewiring_threshold",
            "domain_rewiring_rate": "rewiring_rate",
        },
    )


write_summary_csv, build_grouped_summary = make_sweep_output_helpers(OUTPUT_SPEC)


def build_domain_points(
    base: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rewiring_threshold = (
        args.rewiring_threshold
        if args.rewiring_threshold is not None
        else float(base["domain"]["rewiring"]["threshold"])
    )
    return iter_parameter_grid(
        {
            "update_rule": args.update_rules,
            "confidence_threshold": args.confidence_thresholds,
            "rewiring_rate": args.rewiring_rates,
            "rewiring_threshold": [rewiring_threshold],
        }
    )


def should_run_case(
    domain_fields: Mapping[str, object],
    coordination: CoordinationSweepPoint,
) -> bool:
    return (
        domain_fields["update_rule"] == "neural_policy"
        or coordination.peer_rule != "output_similarity"
    )


def main() -> None:
    args = parse_args()
    alpha_values = args.alphas if args.alphas is not None else [args.alpha]
    run_sweep_from_args(
        args=args,
        toy="toy3",
        output_spec=OUTPUT_SPEC,
        domain_points_builder=build_domain_points,
        thresholds=args.thresholds,
        write_case_config=write_case_config,
        load_config=load_toy3_config,
        run_case=run_toy3,
        row_builder=result_row,
        alphas=alpha_values,
        coordination_social_default="bounded_confidence",
        should_run=should_run_case,
    )


if __name__ == "__main__":
    main()
