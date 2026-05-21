#!/usr/bin/env python
"""Run Toy 8 asynchronous event-hazard sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from neural_abm.config import load_toy8_config
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
from neural_abm.toy_async import run_toy8


SUMMARY_FIELDS = [
    "label",
    "initial_active_fraction",
    "initial_failed_fraction",
    "base_activation_rate",
    "peer_activation_rate",
    "failure_rate",
    "overload_failure_rate",
    "recovery_rate",
    "max_time",
    "graph_k",
    "graph_rewire_probability",
    "coordination_mixer",
    "coordination_peer_rule",
    "alpha",
    "coordination_threshold",
    "seed",
    "epochs",
    "run_dir",
    "domain_final_time",
    "domain_final_inactive_fraction",
    "domain_final_active_fraction",
    "domain_final_failed_fraction",
    "domain_total_events",
    "domain_activation_events",
    "domain_failure_events",
    "domain_recovery_events",
    "domain_absorbed",
    "final_fragmentation_components",
]

GROUP_FIELDS = [
    "label",
    "initial_active_fraction",
    "initial_failed_fraction",
    "base_activation_rate",
    "peer_activation_rate",
    "failure_rate",
    "overload_failure_rate",
    "recovery_rate",
    "max_time",
    "graph_k",
    "graph_rewire_probability",
    "coordination_mixer",
    "coordination_peer_rule",
    "alpha",
    "coordination_threshold",
]

GROUP_AGGREGATIONS = {
    "seeds": ("seed", "nunique"),
    "final_time_mean": ("domain_final_time", "mean"),
    "final_active_fraction_mean": ("domain_final_active_fraction", "mean"),
    "final_active_fraction_std": ("domain_final_active_fraction", "std"),
    "final_failed_fraction_mean": ("domain_final_failed_fraction", "mean"),
    "total_events_mean": ("domain_total_events", "mean"),
    "activation_events_mean": ("domain_activation_events", "mean"),
    "failure_events_mean": ("domain_failure_events", "mean"),
    "recovery_events_mean": ("domain_recovery_events", "mean"),
    "absorbed_rate": ("domain_absorbed", "mean"),
    "final_fragmentation_components_mean": (
        "final_fragmentation_components",
        "mean",
    ),
}

DOMAIN_METRIC_KEYS = [
    "domain_final_time",
    "domain_final_inactive_fraction",
    "domain_final_active_fraction",
    "domain_final_failed_fraction",
    "domain_total_events",
    "domain_activation_events",
    "domain_failure_events",
    "domain_recovery_events",
    "domain_absorbed",
]

CONFIG_VALUE_PATHS = {
    "initial_active_fraction": "environment.initial_active_fraction",
    "initial_failed_fraction": "environment.initial_failed_fraction",
    "base_activation_rate": "environment.base_activation_rate",
    "peer_activation_rate": "environment.peer_activation_rate",
    "failure_rate": "environment.failure_rate",
    "overload_failure_rate": "environment.overload_failure_rate",
    "recovery_rate": "environment.recovery_rate",
    "max_time": "environment.max_time",
    "graph_k": "graph.k",
    "graph_rewire_probability": "graph.rewire_probability",
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
        base_config=Path("experiments/configs/toy8_async_event_baseline.yaml"),
        default_label="toy8_async_sweep_seeds01_05",
        toy_name="Toy 8",
        epochs_help=(
            "Optional maximum event count; defaults to the base config value."
        ),
    )
    parser.add_argument(
        "--initial-active-fractions",
        type=float,
        nargs="+",
        default=[0.1],
        help="Initial active fractions to sweep.",
    )
    parser.add_argument(
        "--initial-failed-fractions",
        type=float,
        nargs="+",
        default=[0.0],
        help="Initial failed fractions to sweep.",
    )
    parser.add_argument(
        "--base-activation-rates",
        type=float,
        nargs="+",
        default=[0.02],
        help="Baseline spontaneous activation rates to sweep.",
    )
    parser.add_argument(
        "--peer-activation-rates",
        type=float,
        nargs="+",
        default=[0.15, 0.3],
        help="Neighbor-driven activation rates to sweep.",
    )
    parser.add_argument(
        "--failure-rates",
        type=float,
        nargs="+",
        default=[0.03],
        help="Baseline active-state failure rates to sweep.",
    )
    parser.add_argument(
        "--overload-failure-rates",
        type=float,
        nargs="+",
        default=[0.04, 0.08],
        help="Neighbor-active overload failure rates to sweep.",
    )
    parser.add_argument(
        "--recovery-rates",
        type=float,
        nargs="+",
        default=[0.01, 0.05],
        help="Failed-state recovery rates to sweep.",
    )
    parser.add_argument(
        "--max-times",
        type=float,
        nargs="+",
        default=None,
        help="Optional simulation time horizons; defaults to base config.",
    )
    parser.add_argument(
        "--graph-ks",
        type=int,
        nargs="+",
        default=None,
        help="Optional Watts-Strogatz k values; defaults to base config.",
    )
    parser.add_argument(
        "--graph-rewire-probabilities",
        type=float,
        nargs="+",
        default=None,
        help="Optional Watts-Strogatz rewire probabilities; defaults to base config.",
    )
    return parser.parse_args()


def case_run_name(
    *,
    label: str,
    initial_active_fraction: float,
    initial_failed_fraction: float,
    base_activation_rate: float,
    peer_activation_rate: float,
    failure_rate: float,
    overload_failure_rate: float,
    recovery_rate: float,
    graph_k: int,
    graph_rewire_probability: float,
    mixer: str,
    peer_rule: str,
    effective_alpha: float,
    effective_threshold: float,
) -> str:
    return (
        f"{label}_ia{safe_float(initial_active_fraction)}_"
        f"if{safe_float(initial_failed_fraction)}_"
        f"ba{safe_float(base_activation_rate)}_"
        f"pa{safe_float(peer_activation_rate)}_"
        f"fr{safe_float(failure_rate)}_"
        f"of{safe_float(overload_failure_rate)}_"
        f"rr{safe_float(recovery_rate)}_"
        f"k{graph_k}_rw{safe_float(graph_rewire_probability)}_"
        f"{mixer}_{peer_rule}_a{safe_float(effective_alpha)}_"
        f"t{safe_float(effective_threshold)}"
    )


def case_config_updates(
    *,
    initial_active_fraction: float,
    initial_failed_fraction: float,
    base_activation_rate: float,
    peer_activation_rate: float,
    failure_rate: float,
    overload_failure_rate: float,
    recovery_rate: float,
    max_time: float,
    graph_k: int,
    graph_rewire_probability: float,
) -> dict[str, object]:
    return {
        "domain.environment.initial_active_fraction": initial_active_fraction,
        "domain.environment.initial_failed_fraction": initial_failed_fraction,
        "domain.environment.base_activation_rate": base_activation_rate,
        "domain.environment.peer_activation_rate": peer_activation_rate,
        "domain.environment.failure_rate": failure_rate,
        "domain.environment.overload_failure_rate": overload_failure_rate,
        "domain.environment.recovery_rate": recovery_rate,
        "domain.environment.max_time": max_time,
        "domain.graph.k": graph_k,
        "domain.graph.rewire_probability": graph_rewire_probability,
    }


def write_case_config(
    *,
    base: dict[str, Any],
    label: str,
    initial_active_fraction: float,
    initial_failed_fraction: float,
    base_activation_rate: float,
    peer_activation_rate: float,
    failure_rate: float,
    overload_failure_rate: float,
    recovery_rate: float,
    max_time: float,
    graph_k: int,
    graph_rewire_probability: float,
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
        toy="toy8",
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
            initial_active_fraction=initial_active_fraction,
            initial_failed_fraction=initial_failed_fraction,
            base_activation_rate=base_activation_rate,
            peer_activation_rate=peer_activation_rate,
            failure_rate=failure_rate,
            overload_failure_rate=overload_failure_rate,
            recovery_rate=recovery_rate,
            graph_k=graph_k,
            graph_rewire_probability=graph_rewire_probability,
            mixer=mixer,
            peer_rule=peer_rule,
            effective_alpha=prepared.effective_alpha,
            effective_threshold=prepared.effective_threshold,
        ),
        updates=case_config_updates(
            initial_active_fraction=initial_active_fraction,
            initial_failed_fraction=initial_failed_fraction,
            base_activation_rate=base_activation_rate,
            peer_activation_rate=peer_activation_rate,
            failure_rate=failure_rate,
            overload_failure_rate=overload_failure_rate,
            recovery_rate=recovery_rate,
            max_time=max_time,
            graph_k=graph_k,
            graph_rewire_probability=graph_rewire_probability,
        ),
        config_dir=config_dir,
    )


def result_row(
    *,
    label: str,
    initial_active_fraction: float,
    initial_failed_fraction: float,
    base_activation_rate: float,
    peer_activation_rate: float,
    failure_rate: float,
    overload_failure_rate: float,
    recovery_rate: float,
    max_time: float,
    graph_k: int,
    graph_rewire_probability: float,
    mixer: str,
    peer_rule: str,
    alpha: float,
    coordination_threshold: float,
    seed: int,
    epochs: int,
    run_dir: Path,
    domain_final_time: float,
    domain_final_inactive_fraction: float,
    domain_final_active_fraction: float,
    domain_final_failed_fraction: float,
    domain_total_events: int,
    domain_activation_events: int,
    domain_failure_events: int,
    domain_recovery_events: int,
    domain_absorbed: bool,
    final_fragmentation_components: int,
) -> dict[str, object]:
    return build_result_row(OUTPUT_SPEC.summary_fields, locals())


write_summary_csv, build_grouped_summary = make_sweep_output_helpers(OUTPUT_SPEC)


def build_domain_points(
    base: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    base_environment = base["domain"]["environment"]
    base_graph = base["domain"]["graph"]
    max_times = args.max_times or [float(base_environment["max_time"])]
    graph_ks = args.graph_ks or [int(base_graph["k"])]
    graph_rewire_probabilities = args.graph_rewire_probabilities or [
        float(base_graph["rewire_probability"])
    ]
    return iter_parameter_grid(
        {
            "initial_active_fraction": args.initial_active_fractions,
            "initial_failed_fraction": args.initial_failed_fractions,
            "base_activation_rate": args.base_activation_rates,
            "peer_activation_rate": args.peer_activation_rates,
            "failure_rate": args.failure_rates,
            "overload_failure_rate": args.overload_failure_rates,
            "recovery_rate": args.recovery_rates,
            "max_time": max_times,
            "graph_k": graph_ks,
            "graph_rewire_probability": graph_rewire_probabilities,
        }
    )


def main() -> None:
    args = parse_args()
    run_sweep_from_args(
        args=args,
        toy="toy8",
        output_spec=OUTPUT_SPEC,
        domain_points_builder=build_domain_points,
        thresholds=args.thresholds,
        write_case_config=write_case_config,
        load_config=load_toy8_config,
        run_case=run_toy8,
        row_builder=result_row,
    )


if __name__ == "__main__":
    main()
