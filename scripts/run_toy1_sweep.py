#!/usr/bin/env python
"""Run Toy 1 alpha/threshold sweeps from a base config."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from neural_abm.config import load_toy1_config
from neural_abm.sweep import (
    SweepOutputSpec,
    apply_config_updates,
    build_result_row,
    make_sweep_output_helpers,
    run_explicit_sweep_from_args,
    safe_float,
)
from neural_abm.toy_classification import run_toy1


@dataclass(frozen=True)
class SweepCase:
    mixer: str
    peer_rule: str
    init_mode: str

    @property
    def slug(self) -> str:
        return f"{self.mixer}_{self.peer_rule}_{self.init_mode}"


SWEEP_CASES = [
    SweepCase("output_average", "output_similarity", "same_init"),
    SweepCase("latent_average", "state_similarity", "same_init"),
    SweepCase("parameter_average", "state_similarity", "same_init"),
    SweepCase("parameter_average", "state_similarity", "independent_init"),
    SweepCase("parameter_aligned_average", "state_similarity", "independent_init"),
    SweepCase(
        "parameter_aligned_average",
        "aligned_state_similarity",
        "independent_init",
    ),
]


SUMMARY_FIELDS = [
    "label",
    "case",
    "seed",
    "coordination_mixer",
    "coordination_peer_rule",
    "model_init_mode",
    "alpha",
    "threshold",
    "run_dir",
    "domain_final_mean_global_accuracy",
    "domain_final_mean_consensus",
    "final_fragmentation_components",
]

GROUP_FIELDS = [
    "case",
    "coordination_mixer",
    "coordination_peer_rule",
    "model_init_mode",
    "alpha",
    "threshold",
]

GROUP_AGGREGATIONS = {
    "seeds": ("seed", "count"),
    "accuracy_mean": ("domain_final_mean_global_accuracy", "mean"),
    "accuracy_std": ("domain_final_mean_global_accuracy", "std"),
    "consensus_mean": ("domain_final_mean_consensus", "mean"),
    "consensus_std": ("domain_final_mean_consensus", "std"),
    "fragmentation_mean": ("final_fragmentation_components", "mean"),
    "fragmentation_std": ("final_fragmentation_components", "std"),
}

DOMAIN_METRIC_KEYS = [
    "domain_final_mean_global_accuracy",
    "domain_final_mean_consensus",
]

CONFIG_VALUE_PATHS = {
    "model_init_mode": "agents.init_mode",
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
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("experiments/configs/toy1_neural_hk_baseline.yaml"),
        help="Base Toy 1 YAML config.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Sweep label. Defaults to a timestamped alpha-threshold label.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1],
        help="Seeds to run for each sweep point.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5],
        help="Social influence strengths to sweep.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.6, 0.8, 0.95],
        help="Peer similarity thresholds to sweep.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epochs per run; defaults to the base config value.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        choices=[case.slug for case in SWEEP_CASES],
        help="Optional subset of sweep case slugs to run.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("experiments/configs/generated"),
        help="Directory for generated per-run configs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Directory for sweep summaries.",
    )
    return parser.parse_args()


def load_raw_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_sweep_case(
    case: SweepCase | str,
    *,
    mixer: str | None = None,
    peer_rule: str | None = None,
    init_mode: str | None = None,
) -> SweepCase:
    if isinstance(case, SweepCase):
        return case
    if mixer is None or peer_rule is None or init_mode is None:
        raise ValueError("mixer, peer_rule, and init_mode are required for case slug")
    return SweepCase(mixer=mixer, peer_rule=peer_rule, init_mode=init_mode)


def case_config_updates(
    *,
    run_name: str,
    seed: int,
    case: SweepCase,
    alpha: float,
    threshold: float,
    epochs: int | None,
) -> dict[str, object]:
    updates: dict[str, object] = {
        "run.name": run_name,
        "run.seed": seed,
        "model.agents.init_mode": case.init_mode,
        "model.coordination.mixer": case.mixer,
        "model.coordination.peer_rule": case.peer_rule,
        "model.coordination.alpha": alpha,
        "model.coordination.threshold": threshold,
    }
    if epochs is not None:
        updates["simulation.epochs"] = epochs
    return updates


def write_case_config(
    base: dict[str, Any],
    case: SweepCase | str,
    seed: int,
    alpha: float,
    threshold: float | None = None,
    label: str | None = None,
    config_dir: Path | None = None,
    epochs: int | None = None,
    *,
    mixer: str | None = None,
    peer_rule: str | None = None,
    init_mode: str | None = None,
    coordination_threshold: float | None = None,
) -> Path:
    if label is None:
        raise ValueError("label is required")
    if config_dir is None:
        raise ValueError("config_dir is required")
    resolved_case = resolve_sweep_case(
        case,
        mixer=mixer,
        peer_rule=peer_rule,
        init_mode=init_mode,
    )
    resolved_threshold = (
        coordination_threshold
        if coordination_threshold is not None
        else threshold
    )
    if resolved_threshold is None:
        raise ValueError("threshold or coordination_threshold is required")
    raw = deepcopy(base)
    run_name = (
        f"{label}_{resolved_case.slug}_a{safe_float(alpha)}_"
        f"t{safe_float(resolved_threshold)}"
    )
    apply_config_updates(
        raw,
        case_config_updates(
            run_name=run_name,
            seed=seed,
            case=resolved_case,
            alpha=alpha,
            threshold=resolved_threshold,
            epochs=epochs,
        ),
    )

    case_dir = config_dir / label
    case_dir.mkdir(parents=True, exist_ok=True)
    path = (
        case_dir / f"{resolved_case.slug}_a{safe_float(alpha)}_"
        f"t{safe_float(resolved_threshold)}_seed{seed:02d}.yaml"
    )
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def result_row(
    *,
    label: str,
    case: str,
    seed: int,
    mixer: str,
    peer_rule: str,
    model_init_mode: str,
    alpha: float,
    coordination_threshold: float,
    run_dir: Path,
    domain_final_mean_global_accuracy: float,
    domain_final_mean_consensus: float,
    final_fragmentation_components: int,
    init_mode: str | None = None,
    epochs: int | None = None,
) -> dict[str, object]:
    return build_result_row(
        OUTPUT_SPEC.summary_fields,
        locals(),
        aliases={
            "coordination_mixer": "mixer",
            "coordination_peer_rule": "peer_rule",
            "threshold": "coordination_threshold",
        },
    )


write_summary_csv, build_grouped_summary = make_sweep_output_helpers(OUTPUT_SPEC)


def write_grouped_markdown(path: Path, label: str, grouped: pd.DataFrame) -> None:
    seed_counts = sorted(grouped["seeds"].unique())
    if len(seed_counts) == 1 and seed_counts[0] == 1:
        readout = "This is a single-seed pilot sweep; treat it as diagnostic evidence."
    elif len(seed_counts) == 1:
        readout = (
            f"This sweep reports means over {int(seed_counts[0])} seeds; use the "
            "standard deviations in the CSV when judging stability."
        )
    else:
        readout = (
            "This sweep has mixed seed counts across rows; inspect the `Seeds` "
            "column before comparing conditions."
        )
    lines = [
        f"# Toy 1 Alpha/Threshold Sweep: {label}",
        "",
        "| Case | Alpha | Threshold | Seeds | Accuracy Mean | Consensus Mean | Fragmentation Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in grouped.itertuples(index=False):
        lines.append(
            f"| `{row.case}` | {row.alpha:g} | {row.threshold:g} | "
            f"{row.seeds} | {row.accuracy_mean:.6f} | "
            f"{row.consensus_mean:.6f} | {row.fragmentation_mean:.2f} |"
        )
    lines += [
        "",
        "## Initial Readout",
        "",
        "This sweep is intended to inspect phase behavior across social influence "
        f"strength and peer threshold. {readout}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def selected_sweep_cases(case_slugs: list[str] | None) -> list[SweepCase]:
    selected_cases = (
        [case for case in SWEEP_CASES if case.slug in set(case_slugs)]
        if case_slugs
        else SWEEP_CASES
    )
    if not selected_cases:
        raise ValueError("No sweep cases selected")
    return selected_cases


def build_sweep_points(
    cases: list[SweepCase],
    alphas: list[float],
    thresholds: list[float],
) -> list[dict[str, object]]:
    return [
        {
            "case": case.slug,
            "mixer": case.mixer,
            "peer_rule": case.peer_rule,
            "init_mode": case.init_mode,
            "alpha": alpha,
            "coordination_threshold": threshold,
        }
        for case in cases
        for alpha in alphas
        for threshold in thresholds
    ]


def build_sweep_points_from_args(
    _base: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    return build_sweep_points(
        selected_sweep_cases(args.cases),
        args.alphas,
        args.thresholds,
    )


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.label or f"{timestamp}_toy1_alpha_threshold_sweep"
    run_explicit_sweep_from_args(
        args=args,
        label=label,
        output_spec=OUTPUT_SPEC,
        sweep_points_builder=build_sweep_points_from_args,
        write_case_config=write_case_config,
        load_config=load_toy1_config,
        run_case=run_toy1,
        row_builder=result_row,
        grouped_markdown_writer=write_grouped_markdown,
        toy="toy1",
    )


if __name__ == "__main__":
    main()
