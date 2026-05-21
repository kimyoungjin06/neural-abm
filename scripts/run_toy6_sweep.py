#!/usr/bin/env python
"""Run Toy 6 categorical spatial-game sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from neural_abm.config import load_toy6_config
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
from neural_abm.toy_categorical import run_toy6


SUMMARY_FIELDS = [
    "label",
    "strategy_count",
    "initial_distribution",
    "coordination_mixer",
    "coordination_peer_rule",
    "alpha",
    "coordination_threshold",
    "payoff_profile",
    "win_payoff",
    "loss_payoff",
    "draw_payoff",
    "seed",
    "epochs",
    "run_dir",
    "domain_final_mean_payoff",
    "domain_final_strategy_entropy",
    "domain_final_dominant_strategy",
    "domain_final_dominant_strategy_fraction",
    "final_fragmentation_components",
]

GROUP_FIELDS = [
    "label",
    "strategy_count",
    "initial_distribution",
    "coordination_mixer",
    "coordination_peer_rule",
    "alpha",
    "coordination_threshold",
    "payoff_profile",
    "win_payoff",
    "loss_payoff",
    "draw_payoff",
]

GROUP_AGGREGATIONS = {
    "seeds": ("seed", "nunique"),
    "final_mean_payoff_mean": ("domain_final_mean_payoff", "mean"),
    "final_mean_payoff_std": ("domain_final_mean_payoff", "std"),
    "final_strategy_entropy_mean": ("domain_final_strategy_entropy", "mean"),
    "final_strategy_entropy_std": ("domain_final_strategy_entropy", "std"),
    "final_dominant_strategy_fraction_mean": (
        "domain_final_dominant_strategy_fraction",
        "mean",
    ),
    "final_fragmentation_components_mean": (
        "final_fragmentation_components",
        "mean",
    ),
}

DOMAIN_METRIC_KEYS = [
    "domain_final_mean_payoff",
    "domain_final_strategy_entropy",
    "domain_final_dominant_strategy",
    "domain_final_dominant_strategy_fraction",
]

CONFIG_VALUE_PATHS = {
    "win_payoff": "game.win_payoff",
    "loss_payoff": "game.loss_payoff",
    "draw_payoff": "game.draw_payoff",
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
        base_config=Path("experiments/configs/toy6_categorical_spatial_baseline.yaml"),
        default_label="toy6_categorical_sweep_seeds01_05",
        toy_name="Toy 6",
    )
    parser.add_argument(
        "--payoff-profiles",
        nargs="+",
        default=["baseline"],
        choices=["baseline", "win_bonus", "loss_heavy"],
        help="Built-in cyclic payoff profiles to sweep.",
    )
    parser.add_argument(
        "--strategy-counts",
        type=int,
        nargs="+",
        default=[3],
        help="Categorical strategy counts to sweep.",
    )
    parser.add_argument(
        "--initial-distribution-labels",
        nargs="+",
        default=["balanced", "biased"],
        choices=["balanced", "biased"],
        help="Built-in initial strategy distributions to sweep.",
    )
    return parser.parse_args()


def initial_distribution(label: str, strategy_count: int) -> list[float]:
    if strategy_count < 3:
        raise ValueError("Toy 6 strategy_count must be at least 3")
    if label == "balanced":
        return [1.0 / strategy_count] * strategy_count
    if label == "biased":
        remainder = 0.4 / (strategy_count - 1)
        return [0.6, *([remainder] * (strategy_count - 1))]
    raise ValueError(f"Unknown Toy 6 initial distribution: {label}")


def payoff_values(base: dict[str, Any], profile: str) -> tuple[float, float, float]:
    game = base["domain"]["game"]
    if profile == "baseline":
        return (
            float(game["win_payoff"]),
            float(game["loss_payoff"]),
            float(game["draw_payoff"]),
        )
    if profile == "win_bonus":
        return (1.5, -1.0, 0.0)
    if profile == "loss_heavy":
        return (1.0, -1.5, 0.0)
    raise ValueError(f"Unknown Toy 6 payoff profile: {profile}")


def case_run_name(
    *,
    label: str,
    strategy_count: int,
    initial_distribution_label: str,
    payoff_profile: str,
    mixer: str,
    peer_rule: str,
    effective_alpha: float,
    effective_threshold: float,
) -> str:
    return (
        f"{label}_k{strategy_count}_{initial_distribution_label}_"
        f"{payoff_profile}_{mixer}_{peer_rule}_"
        f"a{safe_float(effective_alpha)}_t{safe_float(effective_threshold)}"
    )


def case_config_updates(
    *,
    base: dict[str, Any],
    strategy_count: int,
    initial_distribution_label: str,
    payoff_profile: str,
) -> dict[str, object]:
    win_payoff, loss_payoff, draw_payoff = payoff_values(base, payoff_profile)
    return {
        "domain.game.strategy_count": strategy_count,
        "domain.game.win_payoff": win_payoff,
        "domain.game.loss_payoff": loss_payoff,
        "domain.game.draw_payoff": draw_payoff,
        "domain.environment.initial_strategy_probabilities": initial_distribution(
            initial_distribution_label,
            strategy_count,
        ),
    }


def write_case_config(
    *,
    base: dict[str, Any],
    label: str,
    strategy_count: int,
    initial_distribution_label: str,
    mixer: str,
    peer_rule: str,
    alpha: float,
    seed: int,
    epochs: int | None,
    config_dir: Path,
    coordination_threshold: float = 0.0,
    payoff_profile: str = "baseline",
) -> Path:
    prepared = prepare_sweep_case(
        base=base,
        toy="toy6",
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
            strategy_count=strategy_count,
            initial_distribution_label=initial_distribution_label,
            payoff_profile=payoff_profile,
            mixer=mixer,
            peer_rule=peer_rule,
            effective_alpha=prepared.effective_alpha,
            effective_threshold=prepared.effective_threshold,
        ),
        updates=case_config_updates(
            base=base,
            strategy_count=strategy_count,
            initial_distribution_label=initial_distribution_label,
            payoff_profile=payoff_profile,
        ),
        config_dir=config_dir,
    )


def result_row(
    *,
    label: str,
    strategy_count: int,
    initial_distribution_label: str,
    mixer: str,
    peer_rule: str,
    alpha: float,
    seed: int,
    epochs: int,
    run_dir: Path,
    domain_final_mean_payoff: float,
    domain_final_strategy_entropy: float,
    domain_final_dominant_strategy: int,
    domain_final_dominant_strategy_fraction: float,
    final_fragmentation_components: int,
    coordination_threshold: float = 0.0,
    payoff_profile: str = "baseline",
    win_payoff: float = 1.0,
    loss_payoff: float = -1.0,
    draw_payoff: float = 0.0,
) -> dict[str, object]:
    return build_result_row(
        OUTPUT_SPEC.summary_fields,
        locals(),
        aliases={"initial_distribution": "initial_distribution_label"},
    )


write_summary_csv, build_grouped_summary = make_sweep_output_helpers(OUTPUT_SPEC)


def build_domain_points(
    _base: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    return iter_parameter_grid(
        {
            "strategy_count": args.strategy_counts,
            "initial_distribution_label": args.initial_distribution_labels,
            "payoff_profile": args.payoff_profiles,
        }
    )


def main() -> None:
    args = parse_args()
    run_sweep_from_args(
        args=args,
        toy="toy6",
        output_spec=OUTPUT_SPEC,
        domain_points_builder=build_domain_points,
        thresholds=args.thresholds,
        write_case_config=write_case_config,
        load_config=load_toy6_config,
        run_case=run_toy6,
        row_builder=result_row,
    )


if __name__ == "__main__":
    main()
