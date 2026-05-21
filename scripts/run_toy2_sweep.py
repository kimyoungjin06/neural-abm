#!/usr/bin/env python
"""Run Toy 2 game-regime sweeps against classical policy references."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from neural_abm.config import (
    Toy2PayoffConfig,
    load_toy2_config,
    validate_toy2_payoff_threshold,
)
from neural_abm.reputation import reputation_observation_extra_dim
from neural_abm.sweep import (
    CoordinationSweepPoint,
    SweepOutputSpec,
    add_common_sweep_args,
    apply_config_updates,
    build_result_row,
    iter_coordination_sweep,
    make_sweep_output_helpers,
    non_overwriting_path,
    read_final_aggregate_metrics,
    run_case_points,
    run_point_sweep_from_args,
)
from neural_abm.toy_pd import run_toy2


@dataclass(frozen=True)
class RegimePreset:
    name: str
    family: str
    payoff: dict[str, float]
    purpose: str


@dataclass(frozen=True)
class PolicyPriorSpec:
    mode: str
    action_probability: float | None


@dataclass(frozen=True)
class Toy2SweepSettings:
    regimes: list[RegimePreset]
    coordination_points: list[CoordinationSweepPoint]
    initial_action_probabilities: list[float]
    revision_rates: list[float]
    selection_strengths: list[float]
    policy_temperatures: list[float]
    action_temperatures: list[float]
    learning_enabled_values: list[bool]
    policy_prior_specs: list[PolicyPriorSpec]
    local_update_rules: list[str]
    neural_peer_modes: list[str]
    interaction_modes: list[str]
    action_selection_modes: list[str]
    decision_calibrations: list[str]
    calibration_strengths: list[float]


@dataclass(frozen=True)
class Toy2SweepBlock:
    regime: RegimePreset
    initial_action_probability: float
    learning_enabled: bool
    revision_rate: float
    selection_strength: float


@dataclass(frozen=True)
class Toy2CasePoint:
    regime: RegimePreset
    initial_action_probability: float
    learning_enabled: bool
    revision_rate: float
    selection_strength: float
    update_rule: str
    policy_temperature: float
    action_selection_mode: str
    action_temperature: float
    decision_calibration_mode: str
    calibration_strength: float
    decision_threshold: float | None
    policy_prior_spec: PolicyPriorSpec
    policy_prior_action_probability: float | None
    local_update_rule: str
    neural_peer_mode: str
    interaction_mode: str
    mixer: str
    peer_rule: str
    alpha: float
    coordination_threshold: float
    seed: int


@dataclass(frozen=True)
class Toy2RdPoint:
    regime: RegimePreset
    initial_action_probability: float
    learning_enabled: bool
    revision_rate: float
    selection_strength: float
    seed: int


DEFAULT_POLICY_PRIOR_SPEC = PolicyPriorSpec(
    mode="default",
    action_probability=None,
)
DEFAULT_LOCAL_UPDATE_RULE = "sampled_policy_gradient"
DEFAULT_NEURAL_PEER_MODE = "spatial"
DEFAULT_INTERACTION_MODE = "spatial"
DEFAULT_ACTION_SELECTION_MODE = "sampled"
DEFAULT_ACTION_TEMPERATURE = 1.0
DEFAULT_DECISION_CALIBRATION_MODE = "none"
DEFAULT_CALIBRATION_STRENGTH = 4.0


REGIME_PRESETS = [
    RegimePreset(
        name="harsh_pd",
        family="prisoner_dilemma",
        payoff={"T": 5.0, "R": 3.0, "P": 1.0, "S": 0.0},
        purpose="defection-dominant baseline",
    ),
    RegimePreset(
        name="mild_pd",
        family="prisoner_dilemma",
        payoff={"T": 3.5, "R": 3.0, "P": 1.0, "S": 0.0},
        purpose="spatial cooperation survival",
    ),
    RegimePreset(
        name="soft_pd",
        family="prisoner_dilemma",
        payoff={"T": 3.1, "R": 3.0, "P": 1.0, "S": 0.5},
        purpose="weak dilemma",
    ),
    RegimePreset(
        name="snowdrift",
        family="snowdrift",
        payoff={"T": 5.0, "R": 3.0, "P": 0.0, "S": 1.5},
        purpose="mixed strategy regime",
    ),
    RegimePreset(
        name="stag_hunt",
        family="stag_hunt",
        payoff={"T": 3.0, "R": 4.0, "P": 2.0, "S": 0.0},
        purpose="basin sensitivity",
    ),
]


SUMMARY_FIELDS = [
    "label",
    "regime",
    "domain_game_family",
    "purpose",
    "domain_payoff_T",
    "domain_payoff_R",
    "domain_payoff_P",
    "domain_payoff_S",
    "policy_rule",
    "coordination_mixer",
    "coordination_peer_rule",
    "coordination_threshold",
    "seed",
    "initial_action_probability",
    "policy_prior_mode",
    "policy_prior_action_probability",
    "local_update_rule",
    "neural_peer_mode",
    "interaction_mode",
    "decision_mode",
    "action_temperature",
    "decision_calibration_mode",
    "decision_calibration_strength",
    "decision_threshold",
    "policy_revision_rate",
    "alpha",
    "selection_strength",
    "policy_temperature",
    "payoff_transform",
    "exploration_epsilon",
    "learning_enabled",
    "reputation_decay",
    "reputation_temperature",
    "reputation_noise",
    "reputation_observation_mode",
    "mobility_enabled",
    "run_dir",
    "final_action_rate",
    "final_mean_payoff",
    "final_mean_policy_action_probability",
    "final_mean_reputation",
    "final_reputation_dispersion",
    "final_mobility_rate",
    "final_mean_mobility_gain",
    "domain_action_components",
    "domain_largest_action_cluster_fraction",
    "final_fragmentation_components",
]

GROUP_FIELDS = [
    "regime",
    "domain_game_family",
    "policy_rule",
    "coordination_mixer",
    "coordination_peer_rule",
    "coordination_threshold",
    "initial_action_probability",
    "policy_prior_mode",
    "policy_prior_action_probability",
    "local_update_rule",
    "neural_peer_mode",
    "interaction_mode",
    "decision_mode",
    "action_temperature",
    "decision_calibration_mode",
    "decision_calibration_strength",
    "decision_threshold",
    "policy_revision_rate",
    "alpha",
    "selection_strength",
    "policy_temperature",
    "payoff_transform",
    "exploration_epsilon",
    "learning_enabled",
    "reputation_decay",
    "reputation_temperature",
    "reputation_noise",
    "reputation_observation_mode",
    "mobility_enabled",
]

GROUP_AGGREGATIONS = {
    "seeds": ("seed", "count"),
    "action_mean": ("final_action_rate", "mean"),
    "action_std": ("final_action_rate", "std"),
    "payoff_mean": ("final_mean_payoff", "mean"),
    "payoff_std": ("final_mean_payoff", "std"),
    "policy_action_mean": ("final_mean_policy_action_probability", "mean"),
    "reputation_mean": ("final_mean_reputation", "mean"),
    "mobility_rate_mean": ("final_mobility_rate", "mean"),
    "mobility_gain_mean": ("final_mean_mobility_gain", "mean"),
    "cluster_fraction_mean": (
        "domain_largest_action_cluster_fraction",
        "mean",
    ),
    "fragmentation_mean": ("final_fragmentation_components", "mean"),
}

OUTPUT_SPEC = SweepOutputSpec(
    summary_fields=SUMMARY_FIELDS,
    group_fields=GROUP_FIELDS,
    aggregations=GROUP_AGGREGATIONS,
    metric_keys=[],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_sweep_args(
        parser,
        base_config=Path("experiments/configs/toy2_spatial_pd_baseline.yaml"),
        default_label="toy2_regime_sweep_seeds01_05",
        toy_name="Toy 2",
        default_epochs=50,
        default_alphas=None,
        default_config_dir=Path("experiments/configs/generated"),
        epochs_help="Epochs per run.",
        peer_rules_help=(
            "Optional coordination peer rules. Defaults to none for Toy 2, "
            "including output_average, to preserve legacy sweep behavior."
        ),
        threshold_argument="--coordination-thresholds",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=[regime.name for regime in REGIME_PRESETS],
        choices=[regime.name for regime in REGIME_PRESETS],
        help="Payoff regimes to run.",
    )
    parser.add_argument(
        "--update-rules",
        nargs="+",
        default=["neural_policy", "fermi_imitation"],
        choices=["neural_policy", "fermi_imitation", "reputation_imitation"],
        help="Spatial ABM update rules to run.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Single coordination influence strength for output_average.",
    )
    parser.add_argument(
        "--initial-action-probabilities",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional initial cooperation probabilities; defaults to the base "
            "config value."
        ),
    )
    parser.add_argument(
        "--policy-priors",
        nargs="+",
        default=["default"],
        help=(
            "Policy prior modes for neural initialization: default, match_p0, "
            "or numeric probabilities such as 0.5."
        ),
    )
    parser.add_argument(
        "--revision-rates",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional per-agent action revision rates; defaults to the base "
            "config policy value or 1.0."
        ),
    )
    parser.add_argument(
        "--selection-strength",
        type=float,
        default=1.0,
        help="Single selection strength for neural, Fermi, and RD updates.",
    )
    parser.add_argument(
        "--selection-strengths",
        type=float,
        nargs="+",
        default=None,
        help="Optional selection strengths to sweep; overrides --selection-strength.",
    )
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=1.0,
        help="Single softmax temperature for neural policy readout.",
    )
    parser.add_argument(
        "--policy-temperatures",
        type=float,
        nargs="+",
        default=None,
        help="Optional policy temperatures to sweep; overrides --policy-temperature.",
    )
    parser.add_argument(
        "--payoff-transform",
        choices=["linear", "tanh"],
        default="linear",
        help="Payoff advantage transform.",
    )
    parser.add_argument(
        "--exploration-epsilon",
        type=float,
        default=0.0,
        help="Uniform action exploration rate.",
    )
    parser.add_argument(
        "--disable-learning",
        action="store_true",
        help="Disable neural local policy-gradient updates.",
    )
    parser.add_argument(
        "--learning-enabled-values",
        nargs="+",
        choices=["true", "false"],
        default=None,
        help=(
            "Optional learning flags to sweep in one invocation. Overrides "
            "--disable-learning when provided."
        ),
    )
    parser.add_argument(
        "--reputation-observation-mode",
        choices=["none", "self_neighbor_mean"],
        default="none",
        help=(
            "Optional neural observation reputation features. "
            "self_neighbor_mean switches Toy 2 neural configs to model.input_dim=8."
        ),
    )
    parser.add_argument(
        "--local-update-rules",
        nargs="+",
        choices=["sampled_policy_gradient", "counterfactual_advantage"],
        default=["sampled_policy_gradient"],
        help="Neural local policy update rules to sweep.",
    )
    parser.add_argument(
        "--neural-peer-modes",
        nargs="+",
        choices=["spatial", "well_mixed"],
        default=[DEFAULT_NEURAL_PEER_MODE],
        help=(
            "Neural observation/local-update peer source to sweep. Non-neural "
            "references always use spatial."
        ),
    )
    parser.add_argument(
        "--interaction-modes",
        nargs="+",
        choices=["spatial", "well_mixed_resampled"],
        default=[DEFAULT_INTERACTION_MODE],
        help=(
            "Payoff/action interaction source to sweep for spatial ABM updates. "
            "RD references are emitted once per p0 as the analytic well-mixed "
            "baseline."
        ),
    )
    parser.add_argument(
        "--action-selection-modes",
        nargs="+",
        choices=["sampled", "argmax"],
        default=[DEFAULT_ACTION_SELECTION_MODE],
        help=(
            "Neural decision modes to sweep. Non-neural references are recorded "
            "with sampled."
        ),
    )
    parser.add_argument(
        "--action-temperatures",
        type=float,
        nargs="+",
        default=[DEFAULT_ACTION_TEMPERATURE],
        help=(
            "Sampled-action temperature values for the neural decision kernel. "
            "Argmax ignores this value; non-neural references are recorded with 1.0."
        ),
    )
    parser.add_argument(
        "--decision-calibrations",
        nargs="+",
        choices=["none", "payoff_threshold"],
        default=[DEFAULT_DECISION_CALIBRATION_MODE],
        help=(
            "Sampled neural decision calibration modes. Non-neural references "
            "and argmax are recorded with none."
        ),
    )
    parser.add_argument(
        "--calibration-strengths",
        type=float,
        nargs="+",
        default=[DEFAULT_CALIBRATION_STRENGTH],
        help=(
            "Payoff-threshold calibration strengths. Only sampled neural "
            "payoff_threshold cases expand this sweep."
        ),
    )
    parser.add_argument(
        "--diagnostic-preset",
        choices=["payoff_threshold"],
        default=None,
        help="Apply the Stag-Hunt payoff-threshold calibration diagnostic grid.",
    )
    parser.add_argument(
        "--skip-rd",
        action="store_true",
        help="Skip regime-level rd_well_mixed references.",
    )
    return parser.parse_args()


def apply_diagnostic_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.diagnostic_preset != "payoff_threshold":
        return args
    args.regimes = ["stag_hunt"]
    args.update_rules = ["neural_policy"]
    args.mixers = ["none"]
    args.seeds = list(range(1, 11))
    args.policy_priors = ["match_p0"]
    args.initial_action_probabilities = [0.55, 0.60, 0.65, 0.70, 0.75]
    args.learning_enabled_values = ["true"]
    args.local_update_rules = ["counterfactual_advantage"]
    args.neural_peer_modes = ["spatial"]
    args.interaction_modes = ["spatial"]
    args.revision_rates = [0.25]
    args.policy_temperatures = [1.0]
    args.action_selection_modes = ["sampled"]
    args.decision_calibrations = ["payoff_threshold"]
    args.action_temperatures = [1.0, 0.75, 0.5, 0.4, 0.25]
    args.calibration_strengths = [1.0, 2.0, 4.0, 6.0]
    args.alphas = [0.0]
    args.skip_rd = True
    return args


def format_number_for_slug(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def parse_policy_prior_specs(tokens: list[str]) -> list[PolicyPriorSpec]:
    if not tokens:
        return [DEFAULT_POLICY_PRIOR_SPEC]
    specs: list[PolicyPriorSpec] = []
    for token in tokens:
        if token == "default":
            specs.append(DEFAULT_POLICY_PRIOR_SPEC)
            continue
        if token == "match_p0":
            specs.append(PolicyPriorSpec(mode="match_p0", action_probability=None))
            continue
        try:
            probability = float(token)
        except ValueError as exc:
            raise ValueError(
                "Policy prior tokens must be default, match_p0, or a numeric "
                f"probability in [0, 1]; got {token!r}"
            ) from exc
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Policy prior numeric probabilities must be in [0, 1]; got {token!r}"
            )
        specs.append(
            PolicyPriorSpec(
                mode=token,
                action_probability=probability,
            )
        )
    return specs


def effective_policy_prior_specs(
    update_rule: str,
    policy_prior_specs: list[PolicyPriorSpec],
) -> list[PolicyPriorSpec]:
    if update_rule == "neural_policy":
        return policy_prior_specs
    return [DEFAULT_POLICY_PRIOR_SPEC]


def effective_local_update_rules(
    update_rule: str,
    local_update_rules: list[str],
) -> list[str]:
    if update_rule == "neural_policy":
        return local_update_rules
    return [DEFAULT_LOCAL_UPDATE_RULE]


def effective_neural_peer_modes(
    update_rule: str,
    neural_peer_modes: list[str],
) -> list[str]:
    if update_rule == "neural_policy":
        return neural_peer_modes
    return [DEFAULT_NEURAL_PEER_MODE]


def effective_interaction_modes(
    update_rule: str,
    interaction_modes: list[str],
) -> list[str]:
    if update_rule in {"neural_policy", "fermi_imitation", "reputation_imitation"}:
        return interaction_modes
    return [DEFAULT_INTERACTION_MODE]


def effective_action_selection_modes(
    update_rule: str,
    action_selection_modes: list[str],
) -> list[str]:
    if update_rule == "neural_policy":
        return action_selection_modes
    return [DEFAULT_ACTION_SELECTION_MODE]


def effective_action_temperatures(
    update_rule: str,
    action_temperatures: list[float],
    decision_mode: str = DEFAULT_ACTION_SELECTION_MODE,
) -> list[float]:
    if update_rule == "neural_policy" and decision_mode == "sampled":
        return action_temperatures
    return [DEFAULT_ACTION_TEMPERATURE]


def effective_decision_calibrations(
    update_rule: str,
    decision_calibrations: list[str],
    decision_mode: str = DEFAULT_ACTION_SELECTION_MODE,
) -> list[str]:
    if update_rule == "neural_policy" and decision_mode == "sampled":
        return decision_calibrations
    return [DEFAULT_DECISION_CALIBRATION_MODE]


def effective_calibration_strengths(
    update_rule: str,
    calibration_strengths: list[float],
    decision_mode: str = DEFAULT_ACTION_SELECTION_MODE,
    decision_calibration_mode: str = DEFAULT_DECISION_CALIBRATION_MODE,
) -> list[float]:
    if (
        update_rule == "neural_policy"
        and decision_mode == "sampled"
        and decision_calibration_mode == "payoff_threshold"
    ):
        return calibration_strengths
    return [DEFAULT_CALIBRATION_STRENGTH]


def effective_policy_temperatures(
    update_rule: str,
    policy_temperatures: list[float],
) -> list[float]:
    if update_rule == "neural_policy":
        return policy_temperatures
    return [1.0]


def resolve_policy_prior_probability(
    spec: PolicyPriorSpec,
    initial_action_probability: float,
) -> float | None:
    if spec.mode == "default":
        return None
    if spec.mode == "match_p0":
        return initial_action_probability
    return spec.action_probability


def format_optional_float(value: float | None) -> float | str:
    return "" if value is None else value


def policy_prior_slug(spec: PolicyPriorSpec) -> str:
    if spec.mode == "default":
        return "prior_default"
    if spec.mode == "match_p0":
        return "prior_match_p0"
    if spec.action_probability is None:
        raise ValueError(f"Numeric policy prior missing value: {spec.mode}")
    return f"prior_{format_number_for_slug(spec.action_probability)}"


def local_update_slug(local_update_rule: str) -> str:
    if local_update_rule == "sampled_policy_gradient":
        return "sampled_pg"
    if local_update_rule == "counterfactual_advantage":
        return "counterfactual_adv"
    raise ValueError(f"Unsupported local update rule: {local_update_rule}")


def neural_peer_mode_slug(neural_peer_mode: str) -> str:
    if neural_peer_mode == "spatial":
        return "peer_spatial"
    if neural_peer_mode == "well_mixed":
        return "peer_well_mixed"
    raise ValueError(f"Unsupported neural peer mode: {neural_peer_mode}")


def interaction_mode_slug(interaction_mode: str) -> str:
    if interaction_mode == "spatial":
        return "interaction_spatial"
    if interaction_mode == "well_mixed_resampled":
        return "interaction_well_mixed"
    raise ValueError(f"Unsupported interaction mode: {interaction_mode}")


def action_selection_mode_slug(action_selection_mode: str) -> str:
    if action_selection_mode == "sampled":
        return "action_sampled"
    if action_selection_mode == "argmax":
        return "action_argmax"
    raise ValueError(f"Unsupported action selection mode: {action_selection_mode}")


def action_temperature_slug(action_temperature: float) -> str:
    return f"action_temp_{format_number_for_slug(action_temperature)}"


def coordination_name_suffix(peer_rule: str, coordination_threshold: float) -> str:
    if peer_rule == "none" and coordination_threshold == 0.0:
        return ""
    return (
        f"_{peer_rule}_"
        f"coord_th{format_number_for_slug(coordination_threshold)}"
    )


def decision_calibration_slug(
    decision_calibration_mode: str,
    calibration_strength: float,
) -> str:
    if decision_calibration_mode == "none":
        return "cal_none"
    if decision_calibration_mode == "payoff_threshold":
        return f"cal_payoff_threshold_s{format_number_for_slug(calibration_strength)}"
    raise ValueError(
        f"Unsupported decision calibration mode: {decision_calibration_mode}"
    )


def decision_threshold_for_payoff(
    payoff: dict[str, float],
    decision_calibration_mode: str,
) -> float | None:
    if decision_calibration_mode != "payoff_threshold":
        return None
    return validate_toy2_payoff_threshold(Toy2PayoffConfig.model_validate(payoff))


def apply_policy_prior_to_raw(
    raw: dict[str, Any],
    action_probability: float | None,
) -> None:
    agents = raw.setdefault("model", {}).setdefault("agents", {})
    if action_probability is None:
        agents.pop("policy_prior_action_probability", None)
    else:
        agents["policy_prior_action_probability"] = action_probability


def selected_regimes(names: list[str]) -> list[RegimePreset]:
    selected = set(names)
    return [regime for regime in REGIME_PRESETS if regime.name in selected]


def resolved_decision_settings(
    *,
    update_rule: str,
    action_selection_mode: str,
    action_temperature: float,
    decision_calibration_mode: str,
    calibration_strength: float,
    exploration_epsilon: float,
) -> dict[str, Any]:
    resolved_action_temperature = (
        action_temperature
        if update_rule == "neural_policy" and action_selection_mode == "sampled"
        else DEFAULT_ACTION_TEMPERATURE
    )
    resolved_calibration_mode = (
        decision_calibration_mode
        if update_rule == "neural_policy" and action_selection_mode == "sampled"
        else DEFAULT_DECISION_CALIBRATION_MODE
    )
    resolved_calibration_strength = (
        calibration_strength
        if resolved_calibration_mode == "payoff_threshold"
        else DEFAULT_CALIBRATION_STRENGTH
    )
    resolved_exploration_epsilon = (
        exploration_epsilon
        if update_rule == "neural_policy" and action_selection_mode == "sampled"
        else 0.0
    )
    return {
        "mode": action_selection_mode,
        "action_temperature": resolved_action_temperature,
        "exploration_epsilon": resolved_exploration_epsilon,
        "calibration_mode": resolved_calibration_mode,
        "calibration_strength": resolved_calibration_strength,
    }


def case_run_name(
    *,
    label: str,
    regime: RegimePreset,
    update_rule: str,
    mixer: str,
    peer_rule: str,
    coordination_threshold: float,
    initial_action_probability: float,
    policy_prior_spec: PolicyPriorSpec,
    local_update_rule: str,
    neural_peer_mode: str,
    interaction_mode: str,
    decision_mode: str,
    action_temperature: float,
    decision_calibration_mode: str,
    decision_calibration_strength: float,
    revision_rate: float,
    alpha: float,
) -> str:
    return (
        f"{label}_{regime.name}_{update_rule}_{mixer}"
        f"{coordination_name_suffix(peer_rule, coordination_threshold)}_"
        f"p{format_number_for_slug(initial_action_probability)}_"
        f"{policy_prior_slug(policy_prior_spec)}_"
        f"{local_update_slug(local_update_rule)}_"
        f"{neural_peer_mode_slug(neural_peer_mode)}_"
        f"{interaction_mode_slug(interaction_mode)}_"
        f"{action_selection_mode_slug(decision_mode)}_"
        f"{action_temperature_slug(action_temperature)}_"
        f"{decision_calibration_slug(decision_calibration_mode, decision_calibration_strength)}_"
        f"r{format_number_for_slug(revision_rate)}_"
        f"a{format_number_for_slug(alpha)}"
    )


def case_config_updates(
    *,
    label: str,
    regime: RegimePreset,
    update_rule: str,
    mixer: str,
    peer_rule: str,
    coordination_threshold: float,
    seed: int,
    epochs: int,
    initial_action_probability: float,
    policy_prior_spec: PolicyPriorSpec,
    local_update_rule: str,
    neural_peer_mode: str,
    interaction_mode: str,
    action_selection_mode: str,
    action_temperature: float,
    decision_calibration_mode: str,
    calibration_strength: float,
    alpha: float,
    revision_rate: float,
    selection_strength: float,
    policy_temperature: float,
    payoff_transform: str,
    exploration_epsilon: float,
    learning_enabled: bool,
    reputation_observation_mode: str,
) -> dict[str, Any]:
    decision = resolved_decision_settings(
        update_rule=update_rule,
        action_selection_mode=action_selection_mode,
        action_temperature=action_temperature,
        decision_calibration_mode=decision_calibration_mode,
        calibration_strength=calibration_strength,
        exploration_epsilon=exploration_epsilon,
    )
    return {
        "run.name": case_run_name(
            label=label,
            regime=regime,
            update_rule=update_rule,
            mixer=mixer,
            peer_rule=peer_rule,
            coordination_threshold=coordination_threshold,
            initial_action_probability=initial_action_probability,
            policy_prior_spec=policy_prior_spec,
            local_update_rule=local_update_rule,
            neural_peer_mode=neural_peer_mode,
            interaction_mode=interaction_mode,
            decision_mode=decision["mode"],
            action_temperature=decision["action_temperature"],
            decision_calibration_mode=decision["calibration_mode"],
            decision_calibration_strength=decision["calibration_strength"],
            revision_rate=revision_rate,
            alpha=alpha,
        ),
        "run.seed": seed,
        "simulation.epochs": epochs,
        "domain.game": {
            "family": regime.family,
            "payoff": regime.payoff,
        },
        "domain.environment.payoff_T": regime.payoff["T"],
        "domain.environment.payoff_R": regime.payoff["R"],
        "domain.environment.payoff_P": regime.payoff["P"],
        "domain.environment.payoff_S": regime.payoff["S"],
        "domain.environment.initial_action_probability": initial_action_probability,
        "model.policy": {
            "rule": update_rule,
            "learning_enabled": learning_enabled,
            "revision_rate": revision_rate,
            "selection_strength": selection_strength,
            "temperature": policy_temperature,
            "decision": {
                "mode": decision["mode"],
                "action_temperature": decision["action_temperature"],
                "exploration_epsilon": decision["exploration_epsilon"],
                "calibration": {
                    "mode": decision["calibration_mode"],
                    "strength": decision["calibration_strength"],
                },
            },
            "domain": {
                "local_update_rule": local_update_rule,
                "neural_peer_mode": neural_peer_mode,
                "interaction_mode": interaction_mode,
                "payoff_transform": payoff_transform,
            },
        },
        "model.coordination.mixer": mixer,
        "model.coordination.peer_rule": peer_rule,
        "model.coordination.alpha": alpha,
        "model.coordination.threshold": coordination_threshold,
        "model.state.reputation.observation_mode": (
            reputation_observation_mode if update_rule == "neural_policy" else "none"
        ),
    }


def case_config_filename(
    *,
    regime: RegimePreset,
    update_rule: str,
    mixer: str,
    peer_rule: str,
    coordination_threshold: float,
    seed: int,
    initial_action_probability: float,
    policy_prior_spec: PolicyPriorSpec,
    local_update_rule: str,
    neural_peer_mode: str,
    interaction_mode: str,
    action_selection_mode: str,
    action_temperature: float,
    decision_calibration_mode: str,
    calibration_strength: float,
    alpha: float,
    revision_rate: float,
) -> str:
    decision = resolved_decision_settings(
        update_rule=update_rule,
        action_selection_mode=action_selection_mode,
        action_temperature=action_temperature,
        decision_calibration_mode=decision_calibration_mode,
        calibration_strength=calibration_strength,
        exploration_epsilon=0.0,
    )
    return (
        f"{regime.name}_{update_rule}_{mixer}"
        f"{coordination_name_suffix(peer_rule, coordination_threshold)}_"
        f"p{format_number_for_slug(initial_action_probability)}_"
        f"{policy_prior_slug(policy_prior_spec)}_"
        f"{local_update_slug(local_update_rule)}_"
        f"{neural_peer_mode_slug(neural_peer_mode)}_"
        f"{interaction_mode_slug(interaction_mode)}_"
        f"{action_selection_mode_slug(decision['mode'])}_"
        f"{action_temperature_slug(decision['action_temperature'])}_"
        f"{decision_calibration_slug(decision['calibration_mode'], decision['calibration_strength'])}_"
        f"r{format_number_for_slug(revision_rate)}_"
        f"a{format_number_for_slug(alpha)}_seed{seed:02d}.yaml"
    )


def rd_case_run_name(
    *,
    label: str,
    regime: RegimePreset,
    initial_action_probability: float,
    revision_rate: float,
) -> str:
    return (
        f"{label}_{regime.name}_rd_well_mixed_"
        f"p{format_number_for_slug(initial_action_probability)}_"
        f"r{format_number_for_slug(revision_rate)}_reference"
    )


def rd_case_config_updates(
    *,
    label: str,
    regime: RegimePreset,
    seed: int,
    epochs: int,
    initial_action_probability: float,
    revision_rate: float,
    selection_strength: float,
    payoff_transform: str,
    learning_enabled: bool,
) -> dict[str, Any]:
    return {
        "run.name": rd_case_run_name(
            label=label,
            regime=regime,
            initial_action_probability=initial_action_probability,
            revision_rate=revision_rate,
        ),
        "run.seed": seed,
        "simulation.epochs": epochs,
        "domain.game": {
            "family": regime.family,
            "payoff": regime.payoff,
        },
        "domain.environment.payoff_T": regime.payoff["T"],
        "domain.environment.payoff_R": regime.payoff["R"],
        "domain.environment.payoff_P": regime.payoff["P"],
        "domain.environment.payoff_S": regime.payoff["S"],
        "domain.environment.initial_action_probability": initial_action_probability,
        "model.policy": {
            "rule": "rd_well_mixed",
            "learning_enabled": learning_enabled,
            "revision_rate": revision_rate,
            "selection_strength": selection_strength,
            "temperature": 1.0,
            "decision": {
                "mode": DEFAULT_ACTION_SELECTION_MODE,
                "action_temperature": DEFAULT_ACTION_TEMPERATURE,
                "exploration_epsilon": 0.0,
                "calibration": {
                    "mode": DEFAULT_DECISION_CALIBRATION_MODE,
                    "strength": DEFAULT_CALIBRATION_STRENGTH,
                },
            },
            "domain": {
                "local_update_rule": DEFAULT_LOCAL_UPDATE_RULE,
                "neural_peer_mode": DEFAULT_NEURAL_PEER_MODE,
                "interaction_mode": DEFAULT_INTERACTION_MODE,
                "payoff_transform": payoff_transform,
            },
        },
        "model.coordination.mixer": "none",
        "model.coordination.peer_rule": "none",
        "model.coordination.alpha": 0.0,
        "model.coordination.threshold": 0.0,
    }


def ensure_toy2_state_defaults(raw: dict[str, Any]) -> None:
    state = raw.setdefault("model", {}).setdefault("state", {})
    state.setdefault(
        "reputation",
        {
            "enabled": True,
            "decay": 0.9,
            "peer_rule": "spatial",
            "temperature": 1.0,
            "noise": 0.0,
        },
    )
    state.setdefault(
        "mobility",
        {
            "enabled": False,
            "rate": 0.0,
            "candidate_pool_size": 8,
            "selection_rule": "local_quality",
            "move_cost": 0.0,
        },
    )


def write_case_config(
    base: dict[str, Any],
    label: str,
    regime: RegimePreset,
    update_rule: str,
    mixer: str,
    seed: int,
    epochs: int,
    initial_action_probability: float,
    policy_prior_spec: PolicyPriorSpec,
    policy_prior_action_probability: float | None,
    local_update_rule: str,
    neural_peer_mode: str,
    interaction_mode: str,
    action_selection_mode: str,
    action_temperature: float,
    decision_calibration_mode: str,
    calibration_strength: float,
    alpha: float,
    revision_rate: float,
    selection_strength: float,
    policy_temperature: float,
    payoff_transform: str,
    exploration_epsilon: float,
    learning_enabled: bool,
    config_dir: Path,
    peer_rule: str = "none",
    coordination_threshold: float = 0.0,
    reputation_observation_mode: str = "none",
) -> Path:
    raw = deepcopy(base)
    ensure_toy2_state_defaults(raw)
    apply_config_updates(
        raw,
        case_config_updates(
            label=label,
            regime=regime,
            update_rule=update_rule,
            mixer=mixer,
            peer_rule=peer_rule,
            coordination_threshold=coordination_threshold,
            seed=seed,
            epochs=epochs,
            initial_action_probability=initial_action_probability,
            policy_prior_spec=policy_prior_spec,
            local_update_rule=local_update_rule,
            neural_peer_mode=neural_peer_mode,
            interaction_mode=interaction_mode,
            action_selection_mode=action_selection_mode,
            action_temperature=action_temperature,
            decision_calibration_mode=decision_calibration_mode,
            calibration_strength=calibration_strength,
            alpha=alpha,
            revision_rate=revision_rate,
            selection_strength=selection_strength,
            policy_temperature=policy_temperature,
            payoff_transform=payoff_transform,
            exploration_epsilon=exploration_epsilon,
            learning_enabled=learning_enabled,
            reputation_observation_mode=reputation_observation_mode,
        ),
    )
    apply_policy_prior_to_raw(raw, policy_prior_action_probability)
    if update_rule == "neural_policy":
        state = raw["model"]["state"]
        raw["model"]["agents"]["model"]["input_dim"] = (
            6
            + reputation_observation_extra_dim(
                state["reputation"]["observation_mode"]
            )
        )

    case_dir = config_dir / label
    case_dir.mkdir(parents=True, exist_ok=True)
    path = non_overwriting_path(
        case_dir
        / case_config_filename(
            regime=regime,
            update_rule=update_rule,
            mixer=mixer,
            peer_rule=peer_rule,
            coordination_threshold=coordination_threshold,
            seed=seed,
            initial_action_probability=initial_action_probability,
            policy_prior_spec=policy_prior_spec,
            local_update_rule=local_update_rule,
            neural_peer_mode=neural_peer_mode,
            interaction_mode=interaction_mode,
            action_selection_mode=action_selection_mode,
            action_temperature=action_temperature,
            decision_calibration_mode=decision_calibration_mode,
            calibration_strength=calibration_strength,
            alpha=alpha,
            revision_rate=revision_rate,
        )
    )
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def write_rd_config(
    base: dict[str, Any],
    label: str,
    regime: RegimePreset,
    seed: int,
    epochs: int,
    initial_action_probability: float,
    revision_rate: float,
    selection_strength: float,
    payoff_transform: str,
    exploration_epsilon: float,
    learning_enabled: bool,
    config_dir: Path,
) -> Path:
    raw = deepcopy(base)
    ensure_toy2_state_defaults(raw)
    apply_config_updates(
        raw,
        rd_case_config_updates(
            label=label,
            regime=regime,
            seed=seed,
            epochs=epochs,
            initial_action_probability=initial_action_probability,
            revision_rate=revision_rate,
            selection_strength=selection_strength,
            payoff_transform=payoff_transform,
            learning_enabled=learning_enabled,
        ),
    )
    apply_policy_prior_to_raw(raw, None)

    case_dir = config_dir / label
    case_dir.mkdir(parents=True, exist_ok=True)
    path = non_overwriting_path(
        case_dir
        / (
            f"{regime.name}_rd_well_mixed_"
            f"p{format_number_for_slug(initial_action_probability)}_"
            f"r{format_number_for_slug(revision_rate)}_"
            f"seed{seed:02d}.yaml"
        )
    )
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def result_row(
    label: str,
    regime: RegimePreset,
    update_rule: str,
    mixer: str,
    peer_rule: str,
    seed: int,
    initial_action_probability: float,
    policy_prior_mode: str,
    policy_prior_action_probability: float | None,
    local_update_rule: str,
    neural_peer_mode: str,
    interaction_mode: str,
    decision_mode: str,
    action_temperature: float,
    decision_calibration_mode: str,
    decision_calibration_strength: float,
    decision_threshold: float | None,
    revision_rate: float,
    alpha: float,
    selection_strength: float,
    policy_temperature: float,
    payoff_transform: str,
    exploration_epsilon: float,
    learning_enabled: bool,
    reputation_decay: float,
    reputation_temperature: float,
    reputation_noise: float,
    reputation_observation_mode: str,
    mobility_enabled: bool,
    run_dir: Path,
    final_action_rate: float,
    final_mean_payoff: float,
    final_mean_policy_action_probability: float,
    final_mean_reputation: float | str,
    final_reputation_dispersion: float | str,
    final_mobility_rate: float | str,
    final_mean_mobility_gain: float | str,
    domain_action_components: int,
    domain_largest_action_cluster_fraction: float,
    final_fragmentation_components: int,
    coordination_threshold: float = 0.0,
) -> dict[str, Any]:
    values = dict(locals())
    values.update(
        {
            "regime": regime.name,
            "domain_game_family": regime.family,
            "purpose": regime.purpose,
            "domain_payoff_T": regime.payoff["T"],
            "domain_payoff_R": regime.payoff["R"],
            "domain_payoff_P": regime.payoff["P"],
            "domain_payoff_S": regime.payoff["S"],
            "policy_prior_action_probability": format_optional_float(
                policy_prior_action_probability
            ),
            "decision_threshold": format_optional_float(decision_threshold),
            "policy_revision_rate": revision_rate,
        }
    )
    return build_result_row(
        OUTPUT_SPEC.summary_fields,
        values,
        aliases={
            "policy_rule": "update_rule",
            "coordination_mixer": "mixer",
            "coordination_peer_rule": "peer_rule",
        },
    )


write_summary_csv, build_grouped_summary = make_sweep_output_helpers(OUTPUT_SPEC)


def markdown_optional_number(value: object) -> str:
    if value == "" or pd.isna(value):
        return ""
    return f"{float(value):g}"


def write_grouped_markdown(path: Path, label: str, grouped: pd.DataFrame) -> None:
    lines = [
        f"# Toy 2 Game-Regime Sweep: {label}",
        "",
        "| Regime | Update Rule | Mixer | Peer Rule | Coord Threshold | Learning | Local Update | Neural Peer | Interaction | Decision | Action Temp | Calibration | Cal Strength | Decision Threshold | Init Action | Policy Prior | Rep Obs | Revision | Alpha | Selection | Policy Temp | Explore | Seeds | Action Mean | Payoff Mean | Policy Action Mean | Reputation Mean | Mobility Rate | Mobility Gain | Cluster Fraction Mean | Active Peer Components Mean |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in grouped.sort_values(
        ["regime", "policy_rule", "coordination_mixer"]
    ).itertuples(index=False):
        lines.append(
            f"| `{row.regime}` | `{row.policy_rule}` | "
            f"`{row.coordination_mixer}` | "
            f"`{row.coordination_peer_rule}` | "
            f"{row.coordination_threshold:g} | "
            f"`{row.learning_enabled}` | "
            f"`{row.local_update_rule}` | "
            f"`{row.neural_peer_mode}` | "
            f"`{row.interaction_mode}` | "
            f"`{row.decision_mode}` | "
            f"{row.action_temperature:g} | "
            f"`{row.decision_calibration_mode}` | "
            f"{row.decision_calibration_strength:g} | "
            f"{markdown_optional_number(row.decision_threshold)} | "
            f"{row.initial_action_probability:g} | "
            f"`{row.policy_prior_mode}` | "
            f"`{row.reputation_observation_mode}` | "
            f"{row.policy_revision_rate:g} | {row.alpha:g} | "
            f"{row.selection_strength:g} | {row.policy_temperature:g} | "
            f"{row.exploration_epsilon:g} | "
            f"{row.seeds} | {row.action_mean:.6f} | "
            f"{row.payoff_mean:.6f} | "
            f"{row.policy_action_mean:.6f} | "
            f"{row.reputation_mean:.6f} | "
            f"{row.mobility_rate_mean:.6f} | "
            f"{row.mobility_gain_mean:.6f} | "
            f"{row.cluster_fraction_mean:.6f} | {row.fragmentation_mean:.2f} |"
        )
    lines += [
        "",
        "## Readout",
        "",
        "This table is the Toy 2 validation gate summary across payoff regimes, "
        "neural policy updates, Fermi spatial imitation, and one RD well-mixed "
        "reference per regime.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def print_progress(message: str) -> None:
    print(message, flush=True)


def resolve_toy2_sweep_settings(
    base: dict[str, Any],
    args: argparse.Namespace,
) -> Toy2SweepSettings:
    regimes = selected_regimes(args.regimes)
    if not regimes:
        raise ValueError("No regimes selected")
    if not args.seeds:
        raise ValueError("At least one seed is required")

    base_initial_action = float(
        base["domain"]["environment"]["initial_action_probability"]
    )
    base_revision_rate = float(
        base.get("model", {}).get("policy", {}).get("revision_rate", 1.0)
    )
    alpha_values = list(args.alphas if args.alphas is not None else [args.alpha])
    learning_enabled_values = (
        [value == "true" for value in args.learning_enabled_values]
        if args.learning_enabled_values is not None
        else [not args.disable_learning]
    )
    return Toy2SweepSettings(
        regimes=regimes,
        coordination_points=iter_coordination_sweep(
            "toy2",
            args.mixers,
            args.peer_rules,
            alpha_values,
            args.coordination_thresholds,
            social_default="none",
        ),
        initial_action_probabilities=list(
            args.initial_action_probabilities
            if args.initial_action_probabilities is not None
            else [base_initial_action]
        ),
        revision_rates=list(
            args.revision_rates
            if args.revision_rates is not None
            else [base_revision_rate]
        ),
        selection_strengths=list(
            args.selection_strengths
            if args.selection_strengths is not None
            else [args.selection_strength]
        ),
        policy_temperatures=list(
            args.policy_temperatures
            if args.policy_temperatures is not None
            else [args.policy_temperature]
        ),
        action_temperatures=list(args.action_temperatures),
        learning_enabled_values=learning_enabled_values,
        policy_prior_specs=parse_policy_prior_specs(args.policy_priors),
        local_update_rules=list(args.local_update_rules),
        neural_peer_modes=list(args.neural_peer_modes),
        interaction_modes=list(args.interaction_modes),
        action_selection_modes=list(args.action_selection_modes),
        decision_calibrations=list(args.decision_calibrations),
        calibration_strengths=list(args.calibration_strengths),
    )


def iter_toy2_sweep_blocks(
    settings: Toy2SweepSettings,
) -> Iterator[Toy2SweepBlock]:
    for regime in settings.regimes:
        for initial_action_probability in settings.initial_action_probabilities:
            for learning_enabled in settings.learning_enabled_values:
                for revision_rate in settings.revision_rates:
                    for selection_strength in settings.selection_strengths:
                        yield Toy2SweepBlock(
                            regime=regime,
                            initial_action_probability=initial_action_probability,
                            learning_enabled=learning_enabled,
                            revision_rate=revision_rate,
                            selection_strength=selection_strength,
                        )


def iter_toy2_case_points(
    block: Toy2SweepBlock,
    *,
    settings: Toy2SweepSettings,
    update_rules: list[str],
    seeds: list[int],
) -> Iterator[Toy2CasePoint]:
    for update_rule in update_rules:
        update_policy_temperatures = effective_policy_temperatures(
            update_rule,
            settings.policy_temperatures,
        )
        update_policy_prior_specs = effective_policy_prior_specs(
            update_rule,
            settings.policy_prior_specs,
        )
        update_local_update_rules = effective_local_update_rules(
            update_rule,
            settings.local_update_rules,
        )
        update_neural_peer_modes = effective_neural_peer_modes(
            update_rule,
            settings.neural_peer_modes,
        )
        update_interaction_modes = effective_interaction_modes(
            update_rule,
            settings.interaction_modes,
        )
        update_action_selection_modes = effective_action_selection_modes(
            update_rule,
            settings.action_selection_modes,
        )
        for policy_temperature in update_policy_temperatures:
            for action_selection_mode in update_action_selection_modes:
                update_action_temperatures = effective_action_temperatures(
                    update_rule,
                    settings.action_temperatures,
                    decision_mode=action_selection_mode,
                )
                for action_temperature in update_action_temperatures:
                    update_decision_calibrations = effective_decision_calibrations(
                        update_rule,
                        settings.decision_calibrations,
                        decision_mode=action_selection_mode,
                    )
                    for decision_calibration_mode in update_decision_calibrations:
                        update_calibration_strengths = effective_calibration_strengths(
                            update_rule,
                            settings.calibration_strengths,
                            decision_mode=action_selection_mode,
                            decision_calibration_mode=decision_calibration_mode,
                        )
                        for calibration_strength in update_calibration_strengths:
                            decision_threshold = decision_threshold_for_payoff(
                                block.regime.payoff,
                                decision_calibration_mode,
                            )
                            for policy_prior_spec in update_policy_prior_specs:
                                policy_prior_action_probability = (
                                    resolve_policy_prior_probability(
                                        policy_prior_spec,
                                        block.initial_action_probability,
                                    )
                                )
                                for local_update_rule in update_local_update_rules:
                                    for neural_peer_mode in update_neural_peer_modes:
                                        for interaction_mode in (
                                            update_interaction_modes
                                        ):
                                            for coordination in (
                                                settings.coordination_points
                                            ):
                                                for seed in seeds:
                                                    yield Toy2CasePoint(
                                                        regime=block.regime,
                                                        initial_action_probability=(
                                                            block.initial_action_probability
                                                        ),
                                                        learning_enabled=(
                                                            block.learning_enabled
                                                        ),
                                                        revision_rate=(
                                                            block.revision_rate
                                                        ),
                                                        selection_strength=(
                                                            block.selection_strength
                                                        ),
                                                        update_rule=update_rule,
                                                        policy_temperature=(
                                                            policy_temperature
                                                        ),
                                                        action_selection_mode=(
                                                            action_selection_mode
                                                        ),
                                                        action_temperature=(
                                                            action_temperature
                                                        ),
                                                        decision_calibration_mode=(
                                                            decision_calibration_mode
                                                        ),
                                                        calibration_strength=(
                                                            calibration_strength
                                                        ),
                                                        decision_threshold=(
                                                            decision_threshold
                                                        ),
                                                        policy_prior_spec=(
                                                            policy_prior_spec
                                                        ),
                                                        policy_prior_action_probability=(
                                                            policy_prior_action_probability
                                                        ),
                                                        local_update_rule=(
                                                            local_update_rule
                                                        ),
                                                        neural_peer_mode=(
                                                            neural_peer_mode
                                                        ),
                                                        interaction_mode=(
                                                            interaction_mode
                                                        ),
                                                        mixer=coordination.mixer,
                                                        peer_rule=coordination.peer_rule,
                                                        alpha=coordination.alpha,
                                                        coordination_threshold=(
                                                            coordination.threshold
                                                        ),
                                                        seed=seed,
                                                    )


def toy2_case_progress_message(
    point: Toy2CasePoint,
    result: Any,
) -> str:
    coordination_label = point.mixer
    if point.peer_rule != "none":
        coordination_label = (
            f"{point.mixer}/{point.peer_rule}"
            f"/th={point.coordination_threshold:g}"
        )
    return (
        f"{point.regime.name} "
        f"{point.update_rule}/{coordination_label} "
        f"p0={point.initial_action_probability:g} "
        f"prior={point.policy_prior_spec.mode} "
        f"local={point.local_update_rule} "
        f"peer={point.neural_peer_mode} "
        f"interaction={point.interaction_mode} "
        f"decision={point.action_selection_mode} "
        f"action_temp={point.action_temperature:g} "
        f"cal={point.decision_calibration_mode} "
        f"cal_strength={point.calibration_strength:g} "
        f"revision={point.revision_rate:g} "
        f"selection={point.selection_strength:g} "
        f"temp={point.policy_temperature:g} "
        f"learning={point.learning_enabled} "
        f"seed={point.seed} alpha={point.alpha:g} "
        f"action={result.final_action_rate:.4f} "
        f"payoff={result.final_mean_payoff:.4f}"
    )


def run_toy2_case_point(
    *,
    base: dict[str, Any],
    label: str,
    point: Toy2CasePoint,
    epochs: int,
    payoff_transform: str,
    exploration_epsilon: float,
    reputation_observation_mode: str,
    config_dir: Path,
) -> tuple[dict[str, Any], str]:
    config_path = write_case_config(
        base=base,
        label=label,
        regime=point.regime,
        update_rule=point.update_rule,
        mixer=point.mixer,
        peer_rule=point.peer_rule,
        coordination_threshold=point.coordination_threshold,
        seed=point.seed,
        epochs=epochs,
        initial_action_probability=point.initial_action_probability,
        policy_prior_spec=point.policy_prior_spec,
        policy_prior_action_probability=point.policy_prior_action_probability,
        local_update_rule=point.local_update_rule,
        neural_peer_mode=point.neural_peer_mode,
        interaction_mode=point.interaction_mode,
        action_selection_mode=point.action_selection_mode,
        action_temperature=point.action_temperature,
        decision_calibration_mode=point.decision_calibration_mode,
        calibration_strength=point.calibration_strength,
        alpha=point.alpha,
        revision_rate=point.revision_rate,
        selection_strength=point.selection_strength,
        policy_temperature=point.policy_temperature,
        payoff_transform=payoff_transform,
        exploration_epsilon=exploration_epsilon,
        learning_enabled=point.learning_enabled,
        config_dir=config_dir,
        reputation_observation_mode=reputation_observation_mode,
    )
    config = load_toy2_config(config_path)
    result = run_toy2(config=config, config_path=config_path)
    final_metrics = read_final_aggregate_metrics(result.run_dir)
    row = result_row(
        label=label,
        regime=point.regime,
        update_rule=point.update_rule,
        mixer=point.mixer,
        peer_rule=config.coordination.peer_rule,
        coordination_threshold=config.coordination.threshold,
        seed=point.seed,
        initial_action_probability=point.initial_action_probability,
        policy_prior_mode=point.policy_prior_spec.mode,
        policy_prior_action_probability=point.policy_prior_action_probability,
        local_update_rule=point.local_update_rule,
        neural_peer_mode=point.neural_peer_mode,
        interaction_mode=point.interaction_mode,
        decision_mode=config.policy.decision.mode,
        action_temperature=config.policy.decision.action_temperature,
        decision_calibration_mode=config.policy.decision.calibration.mode,
        decision_calibration_strength=config.policy.decision.calibration.strength,
        decision_threshold=point.decision_threshold,
        revision_rate=point.revision_rate,
        alpha=point.alpha,
        selection_strength=point.selection_strength,
        policy_temperature=point.policy_temperature,
        payoff_transform=payoff_transform,
        exploration_epsilon=config.policy.decision.exploration_epsilon,
        learning_enabled=point.learning_enabled,
        reputation_decay=config.state.reputation.decay,
        reputation_temperature=config.state.reputation.temperature,
        reputation_noise=config.state.reputation.noise,
        reputation_observation_mode=config.state.reputation.observation_mode,
        mobility_enabled=config.state.mobility.enabled,
        run_dir=result.run_dir,
        final_action_rate=result.final_action_rate,
        final_mean_payoff=result.final_mean_payoff,
        final_mean_policy_action_probability=(
            result.final_mean_policy_action_probability
        ),
        final_mean_reputation=final_metrics.get("mean_reputation", ""),
        final_reputation_dispersion=final_metrics.get("reputation_dispersion", ""),
        final_mobility_rate=final_metrics.get("mobility_rate", ""),
        final_mean_mobility_gain=final_metrics.get("mean_mobility_gain", ""),
        domain_action_components=int(
            result.domain_metrics.get("domain_action_components", 0)
        ),
        domain_largest_action_cluster_fraction=float(
            result.domain_metrics.get("domain_largest_action_cluster_fraction", 0.0)
        ),
        final_fragmentation_components=result.final_fragmentation_components,
    )
    return row, toy2_case_progress_message(point, result)


def toy2_rd_progress_message(
    point: Toy2RdPoint,
    result: Any,
) -> str:
    return (
        f"{point.regime.name} rd_well_mixed "
        f"p0={point.initial_action_probability:g} "
        f"revision={point.revision_rate:g} "
        f"selection={point.selection_strength:g} "
        f"learning={point.learning_enabled} seed={point.seed} "
        f"action={result.final_action_rate:.4f} "
        f"payoff={result.final_mean_payoff:.4f}"
    )


def run_toy2_rd_point(
    *,
    base: dict[str, Any],
    label: str,
    point: Toy2RdPoint,
    epochs: int,
    payoff_transform: str,
    exploration_epsilon: float,
    config_dir: Path,
) -> tuple[dict[str, Any], str]:
    config_path = write_rd_config(
        base=base,
        label=label,
        regime=point.regime,
        seed=point.seed,
        epochs=epochs,
        initial_action_probability=point.initial_action_probability,
        revision_rate=point.revision_rate,
        selection_strength=point.selection_strength,
        payoff_transform=payoff_transform,
        exploration_epsilon=exploration_epsilon,
        learning_enabled=point.learning_enabled,
        config_dir=config_dir,
    )
    config = load_toy2_config(config_path)
    result = run_toy2(config=config, config_path=config_path)
    final_metrics = read_final_aggregate_metrics(result.run_dir)
    row = result_row(
        label=label,
        regime=point.regime,
        update_rule="rd_well_mixed",
        mixer="none",
        peer_rule="none",
        seed=point.seed,
        initial_action_probability=point.initial_action_probability,
        policy_prior_mode="default",
        policy_prior_action_probability=None,
        local_update_rule=DEFAULT_LOCAL_UPDATE_RULE,
        neural_peer_mode=DEFAULT_NEURAL_PEER_MODE,
        interaction_mode=DEFAULT_INTERACTION_MODE,
        decision_mode=DEFAULT_ACTION_SELECTION_MODE,
        action_temperature=DEFAULT_ACTION_TEMPERATURE,
        decision_calibration_mode=DEFAULT_DECISION_CALIBRATION_MODE,
        decision_calibration_strength=DEFAULT_CALIBRATION_STRENGTH,
        decision_threshold=None,
        revision_rate=point.revision_rate,
        alpha=0.0,
        selection_strength=point.selection_strength,
        policy_temperature=1.0,
        payoff_transform=payoff_transform,
        exploration_epsilon=0.0,
        learning_enabled=point.learning_enabled,
        reputation_decay=config.state.reputation.decay,
        reputation_temperature=config.state.reputation.temperature,
        reputation_noise=config.state.reputation.noise,
        reputation_observation_mode=config.state.reputation.observation_mode,
        mobility_enabled=config.state.mobility.enabled,
        run_dir=result.run_dir,
        final_action_rate=result.final_action_rate,
        final_mean_payoff=result.final_mean_payoff,
        final_mean_policy_action_probability=(
            result.final_mean_policy_action_probability
        ),
        final_mean_reputation=final_metrics.get("mean_reputation", ""),
        final_reputation_dispersion=final_metrics.get("reputation_dispersion", ""),
        final_mobility_rate=final_metrics.get("mobility_rate", ""),
        final_mean_mobility_gain=final_metrics.get("mean_mobility_gain", ""),
        domain_action_components=0,
        domain_largest_action_cluster_fraction=0.0,
        final_fragmentation_components=0,
    )
    return row, toy2_rd_progress_message(point, result)


def run_toy2_sweep_rows(
    *,
    base: dict[str, Any],
    label: str,
    settings: Toy2SweepSettings,
    update_rules: list[str],
    seeds: list[int],
    epochs: int,
    payoff_transform: str,
    exploration_epsilon: float,
    reputation_observation_mode: str,
    skip_rd: bool,
    config_dir: Path,
    progress: Callable[[str], None] | None = print_progress,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def run_case_point(point: Toy2CasePoint) -> tuple[dict[str, Any], str]:
        return run_toy2_case_point(
            base=base,
            label=label,
            point=point,
            epochs=epochs,
            payoff_transform=payoff_transform,
            exploration_epsilon=exploration_epsilon,
            reputation_observation_mode=reputation_observation_mode,
            config_dir=config_dir,
        )

    def run_rd_point(point: Toy2RdPoint) -> tuple[dict[str, Any], str]:
        return run_toy2_rd_point(
            base=base,
            label=label,
            point=point,
            epochs=epochs,
            payoff_transform=payoff_transform,
            exploration_epsilon=exploration_epsilon,
            config_dir=config_dir,
        )

    for block in iter_toy2_sweep_blocks(settings):
        rows.extend(
            run_case_points(
                points=iter_toy2_case_points(
                    block,
                    settings=settings,
                    update_rules=update_rules,
                    seeds=seeds,
                ),
                run_point=run_case_point,
                progress=progress,
            )
        )

        if skip_rd:
            continue
        rd_point = Toy2RdPoint(
            regime=block.regime,
            initial_action_probability=block.initial_action_probability,
            learning_enabled=block.learning_enabled,
            revision_rate=block.revision_rate,
            selection_strength=block.selection_strength,
            seed=seeds[0],
        )
        rows.extend(
            run_case_points(
                points=[rd_point],
                run_point=run_rd_point,
                progress=progress,
            )
        )
    return rows


def build_toy2_rows_from_args(
    base: dict[str, Any],
    args: argparse.Namespace,
    label: str,
    config_dir: Path,
    progress: Callable[[str], None] | None,
) -> list[dict[str, Any]]:
    settings = resolve_toy2_sweep_settings(base, args)
    return run_toy2_sweep_rows(
        base=base,
        label=label,
        settings=settings,
        update_rules=list(args.update_rules),
        seeds=list(args.seeds),
        epochs=args.epochs,
        payoff_transform=args.payoff_transform,
        exploration_epsilon=args.exploration_epsilon,
        reputation_observation_mode=args.reputation_observation_mode,
        skip_rd=args.skip_rd,
        config_dir=config_dir,
        progress=progress,
    )


def main() -> None:
    args = apply_diagnostic_preset(parse_args())
    run_point_sweep_from_args(
        args=args,
        output_spec=OUTPUT_SPEC,
        rows_builder=build_toy2_rows_from_args,
        grouped_markdown_writer=write_grouped_markdown,
        avoid_overwrite=True,
        toy="toy2",
        progress=print_progress,
    )


if __name__ == "__main__":
    main()
