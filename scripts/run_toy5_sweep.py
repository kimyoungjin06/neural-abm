#!/usr/bin/env python
"""Run Toy 5 contagion and threshold-adoption sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from neural_abm.config import load_toy5_config
from neural_abm.reputation import reputation_observation_extra_dim
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
from neural_abm.toy_contagion import run_toy5


SUMMARY_FIELDS = [
    "label",
    "policy_rule",
    "coordination_mixer",
    "coordination_peer_rule",
    "alpha",
    "coordination_threshold",
    "seed",
    "epochs",
    "initial_action_fraction",
    "threshold_mode",
    "homogeneous_threshold",
    "heterogeneous_threshold_low",
    "heterogeneous_threshold_high",
    "simple_contagion_probability",
    "policy_revision_rate",
    "repeated_exposure_decay",
    "adoption_is_absorbing",
    "neural_update_backend",
    "reputation_decay",
    "reputation_temperature",
    "reputation_noise",
    "reputation_observation_mode",
    "run_dir",
    "final_action_rate",
    "final_mean_payoff",
    "final_mean_policy_action_probability",
    "final_mean_reputation",
    "final_reputation_dispersion",
    "domain_cascade_size",
    "domain_time_to_50_action",
    "domain_failed_cascade",
    "domain_mean_neighbor_action_rate",
    "domain_mean_repeated_exposure_count",
    "domain_low_threshold_action_rate",
    "domain_high_threshold_action_rate",
    "final_fragmentation_components",
]

GROUP_FIELDS = [
    "label",
    "policy_rule",
    "coordination_mixer",
    "coordination_peer_rule",
    "alpha",
    "coordination_threshold",
    "threshold_mode",
    "homogeneous_threshold",
    "repeated_exposure_decay",
    "adoption_is_absorbing",
]

GROUP_AGGREGATIONS = {
    "seeds": ("seed", "nunique"),
    "final_action_rate_mean": ("final_action_rate", "mean"),
    "final_action_rate_std": ("final_action_rate", "std"),
    "final_mean_payoff_mean": ("final_mean_payoff", "mean"),
    "final_mean_policy_action_probability_mean": (
        "final_mean_policy_action_probability",
        "mean",
    ),
    "final_mean_reputation_mean": ("final_mean_reputation", "mean"),
    "cascade_size_mean": ("domain_cascade_size", "mean"),
    "final_fragmentation_components_mean": (
        "final_fragmentation_components",
        "mean",
    ),
}

DOMAIN_METRIC_KEYS = [
    "domain_cascade_size",
    "domain_time_to_50_action",
    "domain_failed_cascade",
    "domain_mean_neighbor_action_rate",
    "domain_mean_repeated_exposure_count",
    "domain_low_threshold_action_rate",
    "domain_high_threshold_action_rate",
]

CONFIG_VALUE_PATHS = {
    "initial_action_fraction": "environment.initial_action_fraction",
    "threshold_mode": "environment.threshold_mode",
    "homogeneous_threshold": "environment.homogeneous_threshold",
    "heterogeneous_threshold_low": "environment.heterogeneous_threshold_low",
    "heterogeneous_threshold_high": "environment.heterogeneous_threshold_high",
    "simple_contagion_probability": "environment.simple_contagion_probability",
    "policy_revision_rate": "policy.revision_rate",
    "repeated_exposure_decay": "policy.domain.repeated_exposure_decay",
    "adoption_is_absorbing": "policy.domain.adoption_is_absorbing",
    "neural_update_backend": "policy.neural_update_backend",
    "reputation_decay": "state.reputation.decay",
    "reputation_temperature": "state.reputation.temperature",
    "reputation_noise": "state.reputation.noise",
    "reputation_observation_mode": "state.reputation.observation_mode",
}

RESULT_VALUE_PATHS = {
    "final_action_rate": "final_action_rate",
    "final_mean_payoff": "final_mean_payoff",
    "final_mean_policy_action_probability": "final_mean_policy_action_probability",
    "final_mean_reputation": "final_mean_reputation",
    "final_reputation_dispersion": "final_reputation_dispersion",
}

OUTPUT_SPEC = SweepOutputSpec(
    summary_fields=SUMMARY_FIELDS,
    group_fields=GROUP_FIELDS,
    aggregations=GROUP_AGGREGATIONS,
    metric_keys=DOMAIN_METRIC_KEYS,
    config_value_paths=CONFIG_VALUE_PATHS,
    result_value_paths=RESULT_VALUE_PATHS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_sweep_args(
        parser,
        base_config=Path("experiments/configs/toy5_contagion_adoption_baseline.yaml"),
        default_label="toy5_contagion_sweep_seeds01_05",
        toy_name="Toy 5",
    )
    parser.add_argument(
        "--update-rules",
        nargs="+",
        choices=[
            "simple_contagion",
            "complex_threshold",
            "neural_policy",
            "reputation_imitation",
        ],
        default=["simple_contagion", "complex_threshold", "neural_policy"],
        help="Adoption policy rules to sweep.",
    )
    parser.add_argument(
        "--initial-action-fractions",
        type=float,
        nargs="+",
        default=[0.05],
        help="Initial adopter fractions.",
    )
    parser.add_argument(
        "--threshold-modes",
        nargs="+",
        choices=["homogeneous", "heterogeneous"],
        default=["homogeneous", "heterogeneous"],
        help="Threshold assignment modes.",
    )
    parser.add_argument(
        "--homogeneous-thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.75],
        help="Homogeneous complex-threshold values.",
    )
    parser.add_argument(
        "--heterogeneous-threshold-lows",
        type=float,
        nargs="+",
        default=[0.15],
        help="Low threshold value for heterogeneous mode.",
    )
    parser.add_argument(
        "--heterogeneous-threshold-highs",
        type=float,
        nargs="+",
        default=[0.55],
        help="High threshold value for heterogeneous mode.",
    )
    parser.add_argument(
        "--simple-contagion-probabilities",
        type=float,
        nargs="+",
        default=[0.08],
        help="Simple-contagion per-exposure adoption probabilities.",
    )
    parser.add_argument(
        "--revision-rate",
        type=float,
        default=1.0,
        help="Policy revision probability per agent per epoch.",
    )
    parser.add_argument(
        "--repeated-exposure-decays",
        type=float,
        nargs="+",
        default=[0.0],
        help="Repeated exposure decay values.",
    )
    parser.add_argument(
        "--adoption-absorbing-values",
        nargs="+",
        choices=["true", "false"],
        default=["true"],
        help="Whether adopted agents remain adopted.",
    )
    parser.add_argument(
        "--neural-update-backend",
        choices=["loop", "batched", "tensor_batched", "auto"],
        default="loop",
        help="Neural update backend for neural_policy runs.",
    )
    parser.add_argument("--reputation-decay", type=float, default=0.9)
    parser.add_argument("--reputation-temperature", type=float, default=1.0)
    parser.add_argument("--reputation-noise", type=float, default=0.0)
    parser.add_argument(
        "--reputation-observation-mode",
        choices=["none", "self_neighbor_mean"],
        default="none",
        help=(
            "Optional neural observation reputation features. "
            "self_neighbor_mean switches Toy 5 neural configs to model.input_dim=8."
        ),
    )
    return parser.parse_args()


def resolved_reputation_observation_mode(
    update_rule: str,
    reputation_observation_mode: str,
) -> str:
    return reputation_observation_mode if update_rule == "neural_policy" else "none"


def case_run_name(
    *,
    label: str,
    update_rule: str,
    threshold_mode: str,
    homogeneous_threshold: float,
    repeated_exposure_decay: float,
    adoption_is_absorbing: bool,
    mixer: str,
    peer_rule: str,
    effective_alpha: float,
    effective_threshold: float,
) -> str:
    return (
        f"{label}_{update_rule}_{threshold_mode}_"
        f"h{safe_float(homogeneous_threshold)}_"
        f"d{safe_float(repeated_exposure_decay)}_"
        f"abs{str(adoption_is_absorbing).lower()}_"
        f"{mixer}_{peer_rule}_"
        f"a{safe_float(effective_alpha)}_t{safe_float(effective_threshold)}"
    )


def case_config_updates(
    *,
    update_rule: str,
    initial_action_fraction: float,
    threshold_mode: str,
    homogeneous_threshold: float,
    heterogeneous_threshold_low: float,
    heterogeneous_threshold_high: float,
    simple_contagion_probability: float,
    revision_rate: float,
    repeated_exposure_decay: float,
    adoption_is_absorbing: bool,
    neural_update_backend: str,
    reputation_decay: float,
    reputation_temperature: float,
    reputation_noise: float,
    reputation_observation_mode: str,
) -> dict[str, object]:
    observation_mode = resolved_reputation_observation_mode(
        update_rule,
        reputation_observation_mode,
    )
    return {
        "domain.environment.initial_action_fraction": initial_action_fraction,
        "domain.environment.threshold_mode": threshold_mode,
        "domain.environment.homogeneous_threshold": homogeneous_threshold,
        "domain.environment.heterogeneous_threshold_low": heterogeneous_threshold_low,
        "domain.environment.heterogeneous_threshold_high": heterogeneous_threshold_high,
        "domain.environment.simple_contagion_probability": (
            simple_contagion_probability
        ),
        "model.policy.rule": update_rule,
        "model.policy.revision_rate": revision_rate,
        "model.policy.neural_update_backend": neural_update_backend,
        "model.policy.domain.repeated_exposure_decay": repeated_exposure_decay,
        "model.policy.domain.adoption_is_absorbing": adoption_is_absorbing,
        "model.state.reputation": {
            "enabled": True,
            "decay": reputation_decay,
            "peer_rule": "spatial",
            "temperature": reputation_temperature,
            "noise": reputation_noise,
            "observation_mode": observation_mode,
        },
    }


def write_case_config(
    *,
    base: dict[str, Any],
    label: str,
    update_rule: str,
    initial_action_fraction: float,
    threshold_mode: str,
    homogeneous_threshold: float,
    heterogeneous_threshold_low: float,
    heterogeneous_threshold_high: float,
    simple_contagion_probability: float,
    repeated_exposure_decay: float,
    adoption_is_absorbing: bool,
    mixer: str,
    peer_rule: str,
    alpha: float,
    seed: int,
    epochs: int | None,
    config_dir: Path,
    coordination_threshold: float = 0.0,
    revision_rate: float = 1.0,
    neural_update_backend: str = "loop",
    reputation_decay: float = 0.9,
    reputation_temperature: float = 1.0,
    reputation_noise: float = 0.0,
    reputation_observation_mode: str = "none",
) -> Path:
    prepared = prepare_sweep_case(
        base=base,
        toy="toy5",
        mixer=mixer,
        peer_rule=peer_rule,
        alpha=alpha,
        seed=seed,
        epochs=epochs,
        coordination_threshold=coordination_threshold,
    )
    def update_neural_input_dim(case: dict[str, Any]) -> None:
        if update_rule != "neural_policy":
            return
        observation_mode = case["model"]["state"]["reputation"]["observation_mode"]
        case["model"]["agents"]["model"]["input_dim"] = (
            6 + reputation_observation_extra_dim(observation_mode)
        )

    return write_prepared_sweep_case_config(
        prepared,
        run_name=case_run_name(
            label=label,
            update_rule=update_rule,
            threshold_mode=threshold_mode,
            homogeneous_threshold=homogeneous_threshold,
            repeated_exposure_decay=repeated_exposure_decay,
            adoption_is_absorbing=adoption_is_absorbing,
            mixer=mixer,
            peer_rule=peer_rule,
            effective_alpha=prepared.effective_alpha,
            effective_threshold=prepared.effective_threshold,
        ),
        updates=case_config_updates(
            update_rule=update_rule,
            initial_action_fraction=initial_action_fraction,
            threshold_mode=threshold_mode,
            homogeneous_threshold=homogeneous_threshold,
            heterogeneous_threshold_low=heterogeneous_threshold_low,
            heterogeneous_threshold_high=heterogeneous_threshold_high,
            simple_contagion_probability=simple_contagion_probability,
            revision_rate=revision_rate,
            repeated_exposure_decay=repeated_exposure_decay,
            adoption_is_absorbing=adoption_is_absorbing,
            neural_update_backend=neural_update_backend,
            reputation_decay=reputation_decay,
            reputation_temperature=reputation_temperature,
            reputation_noise=reputation_noise,
            reputation_observation_mode=reputation_observation_mode,
        ),
        mutate_case=update_neural_input_dim,
        config_dir=config_dir,
    )


def result_row(
    *,
    label: str,
    update_rule: str,
    mixer: str,
    peer_rule: str,
    alpha: float,
    coordination_threshold: float,
    seed: int,
    epochs: int,
    initial_action_fraction: float,
    threshold_mode: str,
    homogeneous_threshold: float,
    heterogeneous_threshold_low: float,
    heterogeneous_threshold_high: float,
    simple_contagion_probability: float,
    policy_revision_rate: float,
    repeated_exposure_decay: float,
    adoption_is_absorbing: bool,
    neural_update_backend: str,
    reputation_decay: float,
    reputation_temperature: float,
    reputation_noise: float,
    reputation_observation_mode: str,
    run_dir: Path,
    final_action_rate: float,
    final_mean_payoff: float,
    final_mean_policy_action_probability: float,
    final_mean_reputation: float,
    final_reputation_dispersion: float,
    domain_cascade_size: int,
    domain_time_to_50_action: int | str,
    domain_failed_cascade: bool,
    domain_mean_neighbor_action_rate: float,
    domain_mean_repeated_exposure_count: float,
    domain_low_threshold_action_rate: float | str,
    domain_high_threshold_action_rate: float | str,
    final_fragmentation_components: int,
    revision_rate: float | None = None,
) -> dict[str, object]:
    return build_result_row(
        OUTPUT_SPEC.summary_fields,
        locals(),
        aliases={"policy_rule": "update_rule"},
    )


write_summary_csv, build_grouped_summary = make_sweep_output_helpers(OUTPUT_SPEC)


def build_domain_points(
    _base: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    return iter_parameter_grid(
        {
            "update_rule": args.update_rules,
            "initial_action_fraction": args.initial_action_fractions,
            "threshold_mode": args.threshold_modes,
            "homogeneous_threshold": args.homogeneous_thresholds,
            "heterogeneous_threshold_low": args.heterogeneous_threshold_lows,
            "heterogeneous_threshold_high": args.heterogeneous_threshold_highs,
            "simple_contagion_probability": args.simple_contagion_probabilities,
            "repeated_exposure_decay": args.repeated_exposure_decays,
            "adoption_is_absorbing": [
                value == "true" for value in args.adoption_absorbing_values
            ],
            "revision_rate": [args.revision_rate],
            "neural_update_backend": [args.neural_update_backend],
            "reputation_decay": [args.reputation_decay],
            "reputation_temperature": [args.reputation_temperature],
            "reputation_noise": [args.reputation_noise],
            "reputation_observation_mode": [args.reputation_observation_mode],
        }
    )


def main() -> None:
    args = parse_args()
    run_sweep_from_args(
        args=args,
        toy="toy5",
        output_spec=OUTPUT_SPEC,
        domain_points_builder=build_domain_points,
        thresholds=args.thresholds,
        write_case_config=write_case_config,
        load_config=load_toy5_config,
        run_case=run_toy5,
        row_builder=result_row,
    )


if __name__ == "__main__":
    main()
