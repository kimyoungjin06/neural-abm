#!/usr/bin/env python
"""Run Toy 4 public-goods sweeps including reputation and mobility diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from neural_abm.config import load_toy4_config
from neural_abm.reputation import reputation_observation_extra_dim
from neural_abm.sweep import (
    SweepOutputSpec,
    add_common_sweep_args,
    build_result_row,
    iter_parameter_grid,
    make_sweep_output_helpers,
    prepare_sweep_case,
    read_final_aggregate_metrics,
    run_sweep_from_args,
    safe_float,
    write_prepared_sweep_case_config,
)
from neural_abm.toy_public_goods import run_toy4


SUMMARY_FIELDS = [
    "label",
    "policy_rule",
    "coordination_mixer",
    "coordination_peer_rule",
    "seed",
    "epochs",
    "alpha",
    "coordination_threshold",
    "policy_revision_rate",
    "resource_enabled",
    "reputation_decay",
    "reputation_temperature",
    "reputation_noise",
    "reputation_observation_mode",
    "mobility_enabled",
    "mobility_rate",
    "mobility_move_cost",
    "run_dir",
    "final_action_rate",
    "final_mean_payoff",
    "domain_payoff_gini",
    "domain_resource_level",
    "domain_collapse_time",
    "final_mean_reputation",
    "final_reputation_dispersion",
    "final_mobility_rate",
    "final_mean_mobility_gain",
]

GROUP_FIELDS = [
    "policy_rule",
    "coordination_mixer",
    "coordination_peer_rule",
    "alpha",
    "coordination_threshold",
    "reputation_observation_mode",
    "mobility_enabled",
]

GROUP_AGGREGATIONS = {
    "seeds": ("seed", "count"),
    "action_mean": ("final_action_rate", "mean"),
    "payoff_mean": ("final_mean_payoff", "mean"),
    "reputation_mean": ("final_mean_reputation", "mean"),
    "mobility_rate_mean": ("final_mobility_rate", "mean"),
    "mobility_gain_mean": ("final_mean_mobility_gain", "mean"),
}

DOMAIN_METRIC_KEYS = [
    "domain_payoff_gini",
    "domain_resource_level",
    "domain_collapse_time",
]

CONFIG_VALUE_PATHS = {
    "policy_revision_rate": "policy.revision_rate",
    "resource_enabled": "environment.resource_enabled",
    "reputation_decay": "state.reputation.decay",
    "reputation_temperature": "state.reputation.temperature",
    "reputation_noise": "state.reputation.noise",
    "reputation_observation_mode": "state.reputation.observation_mode",
    "mobility_enabled": "state.mobility.enabled",
    "mobility_rate": "state.mobility.rate",
    "mobility_move_cost": "state.mobility.move_cost",
}

RESULT_VALUE_PATHS = {
    "final_action_rate": "final_action_rate",
    "final_mean_payoff": "final_mean_payoff",
    "final_mean_reputation": "final_mean_reputation",
    "final_reputation_dispersion": "final_reputation_dispersion",
    "final_mobility_rate": "final_mobility_rate",
    "final_mean_mobility_gain": "final_mean_mobility_gain",
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
        base_config=Path("experiments/configs/toy4_public_goods_baseline.yaml"),
        default_label="toy4_reputation_sweep_seeds01_03",
        toy_name="Toy 4",
        default_seeds=(1, 2, 3),
        default_epochs=50,
        default_alphas=None,
        default_config_dir=Path("experiments/configs/generated"),
    )
    parser.add_argument(
        "--update-rules",
        nargs="+",
        choices=["imitation", "neural_policy", "reputation_imitation"],
        default=["imitation", "neural_policy", "reputation_imitation"],
    )
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--revision-rate", type=float, default=1.0)
    parser.add_argument("--resource-enabled", action="store_true")
    parser.add_argument("--reputation-decay", type=float, default=0.9)
    parser.add_argument("--reputation-temperature", type=float, default=1.0)
    parser.add_argument("--reputation-noise", type=float, default=0.0)
    parser.add_argument(
        "--reputation-observation-mode",
        choices=["none", "self_neighbor_mean"],
        default="none",
        help=(
            "Optional neural observation reputation features. "
            "self_neighbor_mean switches Toy 4 neural configs to model.input_dim=8."
        ),
    )
    parser.add_argument(
        "--mobility-enabled-values",
        nargs="+",
        choices=["true", "false"],
        default=["false"],
    )
    parser.add_argument("--mobility-rate", type=float, default=0.25)
    parser.add_argument("--mobility-move-cost", type=float, default=0.0)
    return parser.parse_args()


def resolve_peer_rule(mixer: str, peer_rule: str | None) -> str:
    if peer_rule is not None:
        return peer_rule
    return "none" if mixer == "none" else "output_similarity"


def resolved_reputation_observation_mode(
    update_rule: str,
    reputation_observation_mode: str,
) -> str:
    return reputation_observation_mode if update_rule == "neural_policy" else "none"


def case_run_name(
    *,
    label: str,
    update_rule: str,
    mixer: str,
    peer_rule: str,
    mobility_enabled: bool,
    effective_alpha: float,
    effective_threshold: float,
) -> str:
    return (
        f"{label}_{update_rule}_{mixer}_{peer_rule}_"
        f"a{safe_float(effective_alpha)}_t{safe_float(effective_threshold)}_"
        f"mobility_{str(mobility_enabled).lower()}"
    )


def case_config_updates(
    *,
    update_rule: str,
    revision_rate: float,
    resource_enabled: bool,
    reputation_decay: float,
    reputation_temperature: float,
    reputation_noise: float,
    reputation_observation_mode: str,
    mobility_enabled: bool,
    mobility_rate: float,
    mobility_move_cost: float,
) -> dict[str, object]:
    observation_mode = resolved_reputation_observation_mode(
        update_rule,
        reputation_observation_mode,
    )
    return {
        "domain.environment.resource_enabled": resource_enabled,
        "model.policy.rule": update_rule,
        "model.policy.revision_rate": revision_rate,
        "model.state.reputation": {
            "enabled": True,
            "decay": reputation_decay,
            "peer_rule": "spatial",
            "temperature": reputation_temperature,
            "noise": reputation_noise,
            "observation_mode": observation_mode,
        },
        "model.state.mobility": {
            "enabled": mobility_enabled,
            "rate": mobility_rate if mobility_enabled else 0.0,
            "candidate_pool_size": 8,
            "selection_rule": "local_quality",
            "move_cost": mobility_move_cost,
        },
    }


def write_toy4_case_config(
    *,
    base: dict[str, Any],
    label: str,
    update_rule: str,
    mixer: str,
    peer_rule: str | None,
    alpha: float,
    seed: int,
    epochs: int | None,
    config_dir: Path,
    mobility_enabled: bool,
    revision_rate: float,
    resource_enabled: bool,
    reputation_decay: float,
    reputation_temperature: float,
    reputation_noise: float,
    reputation_observation_mode: str,
    mobility_rate: float,
    mobility_move_cost: float,
    coordination_threshold: float = 0.0,
) -> Path:
    resolved_peer_rule = resolve_peer_rule(mixer, peer_rule)
    prepared = prepare_sweep_case(
        base=base,
        toy="toy4",
        mixer=mixer,
        peer_rule=resolved_peer_rule,
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
            mixer=mixer,
            peer_rule=resolved_peer_rule,
            mobility_enabled=mobility_enabled,
            effective_alpha=prepared.effective_alpha,
            effective_threshold=prepared.effective_threshold,
        ),
        updates=case_config_updates(
            update_rule=update_rule,
            revision_rate=revision_rate,
            resource_enabled=resource_enabled,
            reputation_decay=reputation_decay,
            reputation_temperature=reputation_temperature,
            reputation_noise=reputation_noise,
            reputation_observation_mode=reputation_observation_mode,
            mobility_enabled=mobility_enabled,
            mobility_rate=mobility_rate,
            mobility_move_cost=mobility_move_cost,
        ),
        mutate_case=update_neural_input_dim,
        config_dir=config_dir,
    )


def write_case_config(
    base: dict[str, Any],
    args: argparse.Namespace,
    update_rule: str,
    mixer: str,
    seed: int,
    mobility_enabled: bool,
) -> Path:
    return write_toy4_case_config(
        base=base,
        label=args.label,
        update_rule=update_rule,
        mixer=mixer,
        peer_rule=None,
        alpha=args.alpha,
        seed=seed,
        epochs=args.epochs,
        config_dir=args.config_dir / args.label,
        mobility_enabled=mobility_enabled,
        revision_rate=args.revision_rate,
        resource_enabled=args.resource_enabled,
        reputation_decay=args.reputation_decay,
        reputation_temperature=args.reputation_temperature,
        reputation_noise=args.reputation_noise,
        reputation_observation_mode=args.reputation_observation_mode,
        mobility_rate=args.mobility_rate,
        mobility_move_cost=args.mobility_move_cost,
    )


def result_row(
    *,
    label: str,
    update_rule: str,
    mixer: str,
    peer_rule: str,
    seed: int,
    epochs: int,
    alpha: float,
    policy_revision_rate: float,
    resource_enabled: bool,
    reputation_decay: float,
    reputation_temperature: float,
    reputation_noise: float,
    reputation_observation_mode: str,
    mobility_enabled: bool,
    mobility_rate: float,
    mobility_move_cost: float,
    run_dir: Path,
    final_action_rate: float,
    final_mean_payoff: float,
    domain_payoff_gini: float | str,
    domain_resource_level: float | str,
    domain_collapse_time: int | float | str,
    final_mean_reputation: float | str,
    final_reputation_dispersion: float | str,
    final_mobility_rate: float | str,
    final_mean_mobility_gain: float | str,
    final_fragmentation_components: int | None = None,
    coordination_threshold: float = 0.0,
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
    mobility_values = [value == "true" for value in args.mobility_enabled_values]
    return iter_parameter_grid(
        {
            "update_rule": args.update_rules,
            "mobility_enabled": mobility_values,
            "revision_rate": [args.revision_rate],
            "resource_enabled": [args.resource_enabled],
            "reputation_decay": [args.reputation_decay],
            "reputation_temperature": [args.reputation_temperature],
            "reputation_noise": [args.reputation_noise],
            "reputation_observation_mode": [args.reputation_observation_mode],
            "mobility_rate": [args.mobility_rate],
            "mobility_move_cost": [args.mobility_move_cost],
        }
    )


def run_toy4_case(*, config: Any, config_path: Path) -> SimpleNamespace:
    result = run_toy4(config=config, config_path=config_path)
    final_metrics = read_final_aggregate_metrics(result.run_dir)
    domain_metrics = {
        key: result.domain_metrics.get(key, "")
        for key in OUTPUT_SPEC.metric_keys
    }
    return SimpleNamespace(
        run_dir=result.run_dir,
        toy=result.toy,
        final_fragmentation_components=result.final_fragmentation_components,
        final_action_rate=result.final_action_rate,
        final_mean_payoff=result.final_mean_payoff,
        final_mean_reputation=final_metrics.get("mean_reputation", ""),
        final_reputation_dispersion=final_metrics.get(
            "reputation_dispersion",
            "",
        ),
        final_mobility_rate=final_metrics.get("mobility_rate", ""),
        final_mean_mobility_gain=final_metrics.get("mean_mobility_gain", ""),
        domain_metrics=domain_metrics,
    )


def main() -> None:
    args = parse_args()
    alpha_values = args.alphas if args.alphas is not None else [args.alpha]
    run_args = SimpleNamespace(**vars(args))
    run_args.config_dir = args.config_dir / args.label
    run_sweep_from_args(
        args=run_args,
        toy="toy4",
        output_spec=OUTPUT_SPEC,
        domain_points_builder=build_domain_points,
        thresholds=args.thresholds,
        write_case_config=write_toy4_case_config,
        load_config=load_toy4_config,
        run_case=run_toy4_case,
        row_builder=result_row,
        alphas=alpha_values,
    )


if __name__ == "__main__":
    main()
