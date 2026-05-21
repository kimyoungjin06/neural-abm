"""Small NABM effect matrix runner and summarizer."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from neural_abm.capabilities import (
    NABM_ARTIFACT_FIELDS,
    supports_coordination,
    sweep_capability_metadata,
)
from neural_abm.config import (
    load_toy1_config,
    load_toy2_config,
    load_toy3_config,
    load_toy4_config,
    load_toy5_config,
)
from neural_abm.toy_classification import run_toy1
from neural_abm.toy_contagion import run_toy5
from neural_abm.toy_opinion import run_toy3
from neural_abm.toy_pd import run_toy2
from neural_abm.toy_public_goods import run_toy4


EffectDirection = Literal["maximize", "minimize"]


TERMINAL_CEILING_WINDOW = 5

CEILING_STABILITY_RUN_FIELDS = [
    "ever_ceiling_final_miss",
    "late_flip_count_after_first_ceiling",
    "late_flip_rate_after_first_ceiling",
    "terminal_window_size",
    "terminal_window_ceiling_rate",
    "terminal_window_mean_ceiling_metric",
]

CEILING_STABILITY_SUMMARY_FIELDS = [
    "ever_ceiling_final_miss_rate",
    "late_flip_count_after_first_ceiling_mean",
    "late_flip_rate_after_first_ceiling_mean",
    "terminal_window_ceiling_rate_mean",
    "terminal_window_mean_ceiling_metric_mean",
]

PRECOMMITMENT_TRAJECTORY_RUN_FIELDS = [
    "precommitment_first_ready_epoch",
    "precommitment_all_ready_epoch",
    "precommitment_first_forced_epoch",
    "precommitment_ready_to_forced_delay_mean",
    "precommitment_premature_exit_count",
    "precommitment_high_policy_rate",
    "precommitment_direction_score_mean",
    "precommitment_direction_score_positive_rate",
    "precommitment_direction_ok_rate",
    "precommitment_ready_largest_component_fraction",
    "precommitment_peer_evidence_enabled",
    "precommitment_peer_evidence_weight",
    "precommitment_peer_readiness_aggregation",
    "precommitment_peer_readiness_mean",
    "precommitment_peer_readiness_active_rate",
    "precommitment_peer_evidence_increment_mean",
]

RUN_FIELDS = [
    "label",
    "case",
    "toy",
    "variant",
    "group",
    "baseline_group",
    "nabm_group",
    "seed",
    "primary_metric",
    "metric_value",
    "direction",
    "ceiling_metric",
    "ceiling_value",
    "ceiling_tolerance",
    "ceiling_metric_value",
    "ceiling_gap",
    "final_within_ceiling",
    "ever_reached_ceiling",
    "time_to_ceiling",
    *CEILING_STABILITY_RUN_FIELDS,
    *PRECOMMITMENT_TRAJECTORY_RUN_FIELDS,
    "config_path",
    "run_dir",
    *NABM_ARTIFACT_FIELDS,
]

EFFECT_FIELDS = [
    "label",
    "case",
    "toy",
    "primary_metric",
    "direction",
    "baseline_group",
    "nabm_group",
    "baseline_n",
    "baseline_mean",
    "baseline_std",
    "baseline_ci95",
    "nabm_n",
    "nabm_mean",
    "nabm_std",
    "nabm_ci95",
    "effect_n",
    "effect_mean",
    "effect_std",
    "effect_ci95",
    "positive_effect_favors",
    "ceiling_metric",
    "ceiling_value",
    "ceiling_tolerance",
    "baseline_final_ceiling_rate",
    "nabm_final_ceiling_rate",
    "baseline_ever_ceiling_rate",
    "nabm_ever_ceiling_rate",
    "baseline_time_to_ceiling_mean",
    "nabm_time_to_ceiling_mean",
    "time_to_ceiling_effect_mean",
    *[
        f"{prefix}_{field}"
        for field in CEILING_STABILITY_SUMMARY_FIELDS
        for prefix in ("baseline", "nabm")
    ],
    "ceiling_outcome",
    *NABM_ARTIFACT_FIELDS,
]

PAIRWISE_EFFECT_FIELDS = [
    "label",
    "case",
    "toy",
    "primary_metric",
    "direction",
    "baseline_variant",
    "nabm_variant",
    "baseline_n",
    "baseline_mean",
    "baseline_std",
    "baseline_ci95",
    "nabm_n",
    "nabm_mean",
    "nabm_std",
    "nabm_ci95",
    "effect_n",
    "effect_mean",
    "effect_std",
    "effect_ci95",
    "positive_effect_favors",
    "ceiling_metric",
    "ceiling_value",
    "ceiling_tolerance",
    "baseline_final_ceiling_rate",
    "nabm_final_ceiling_rate",
    "baseline_ever_ceiling_rate",
    "nabm_ever_ceiling_rate",
    "baseline_time_to_ceiling_mean",
    "nabm_time_to_ceiling_mean",
    "time_to_ceiling_effect_mean",
    *[
        f"{prefix}_{field}"
        for field in CEILING_STABILITY_SUMMARY_FIELDS
        for prefix in ("baseline", "nabm")
    ],
    "ceiling_outcome",
    *NABM_ARTIFACT_FIELDS,
]


@dataclass(frozen=True)
class MatrixVariant:
    name: str
    group: str
    updates: Mapping[str, Any]


@dataclass(frozen=True)
class MatrixCase:
    toy: str
    name: str
    base_config: Path
    primary_metric: str
    direction: EffectDirection
    variants: tuple[MatrixVariant, ...]
    baseline_group: str = "baseline"
    nabm_group: str = "nabm"
    seeds: tuple[int, ...] | None = None
    epochs: int | None = None
    ceiling_metric: str | None = None
    ceiling_value: float | None = None
    ceiling_tolerance: float = 0.0


@dataclass(frozen=True)
class EvidenceManifest:
    label: str
    seeds: tuple[int, ...]
    epochs: int
    config_dir: Path
    runs_dir: Path
    cases: tuple[MatrixCase, ...]


@dataclass(frozen=True)
class ToyRunHandler:
    loader: Callable[[Path], Any]
    runner: Callable[[Any, Path], Any]


@dataclass(frozen=True)
class MatrixRunSpec:
    case: MatrixCase
    variant: MatrixVariant
    seed: int


@dataclass(frozen=True)
class SummaryStats:
    n: int
    mean: float
    std: float
    ci95: float


@dataclass
class EvidenceMatrixResult:
    runs_path: Path
    effects_path: Path
    pairwise_effects_path: Path
    markdown_path: Path
    run_rows: list[dict[str, Any]]
    effect_rows: list[dict[str, Any]]
    pairwise_effect_rows: list[dict[str, Any]]


DEFAULT_HANDLERS: dict[str, ToyRunHandler] = {
    "toy1": ToyRunHandler(loader=load_toy1_config, runner=run_toy1),
    "toy2": ToyRunHandler(loader=load_toy2_config, runner=run_toy2),
    "toy3": ToyRunHandler(loader=load_toy3_config, runner=run_toy3),
    "toy4": ToyRunHandler(loader=load_toy4_config, runner=run_toy4),
    "toy5": ToyRunHandler(loader=load_toy5_config, runner=run_toy5),
}


def load_manifest(path: Path) -> EvidenceManifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return manifest_from_mapping(raw, source_path=path)


def manifest_from_mapping(
    raw: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> EvidenceManifest:
    required = {"label", "seeds", "epochs", "cases"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Evidence manifest missing field(s): {', '.join(missing)}")
    cases_raw = raw["cases"]
    if not isinstance(cases_raw, Sequence) or isinstance(cases_raw, str | bytes):
        raise ValueError("Evidence manifest cases must be a list")
    cases = tuple(_case_from_mapping(case) for case in cases_raw)
    if not cases:
        raise ValueError("Evidence manifest must contain at least one case")
    seeds = tuple(int(seed) for seed in _required_sequence(raw, "seeds"))
    if not seeds:
        raise ValueError("Evidence manifest seeds must not be empty")
    return EvidenceManifest(
        label=str(raw["label"]),
        seeds=seeds,
        epochs=int(raw["epochs"]),
        config_dir=Path(
            raw.get("config_dir", "experiments/configs/generated/evidence_matrix")
        ),
        runs_dir=Path(raw.get("runs_dir", "experiments/runs")),
        cases=cases,
    )


def run_evidence_matrix(
    manifest_path: Path,
    *,
    results_dir: Path = Path("experiments/results/nabm_effect_matrix"),
    label: str | None = None,
    seeds: Sequence[int] | None = None,
    epochs: int | None = None,
    config_dir: Path | None = None,
    runs_dir: Path | None = None,
) -> EvidenceMatrixResult:
    manifest = load_manifest(manifest_path)
    if label is not None:
        manifest = replace(manifest, label=label)
    if seeds is not None:
        manifest = replace(manifest, seeds=tuple(int(seed) for seed in seeds))
    if epochs is not None:
        manifest = replace(manifest, epochs=int(epochs))
    if config_dir is not None:
        manifest = replace(manifest, config_dir=config_dir)
    if runs_dir is not None:
        manifest = replace(manifest, runs_dir=runs_dir)
    return run_matrix(manifest, results_dir=results_dir)


def run_matrix(
    manifest: EvidenceManifest,
    *,
    results_dir: Path,
    handlers: Mapping[str, ToyRunHandler] | None = None,
) -> EvidenceMatrixResult:
    resolved_handlers = DEFAULT_HANDLERS if handlers is None else handlers
    run_rows: list[dict[str, Any]] = []
    for spec in expand_cases(manifest):
        run_rows.append(
            run_case(
                manifest=manifest,
                spec=spec,
                handlers=resolved_handlers,
            )
        )
    effect_rows = build_effect_summary(run_rows)
    pairwise_effect_rows = build_pairwise_effect_summary(run_rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_path = results_dir / f"{manifest.label}_runs.csv"
    effects_path = results_dir / f"{manifest.label}_effects.csv"
    pairwise_effects_path = results_dir / f"{manifest.label}_pairwise_effects.csv"
    markdown_path = results_dir / f"{manifest.label}_effects.md"
    write_csv(runs_path, run_rows, RUN_FIELDS)
    write_csv(effects_path, effect_rows, EFFECT_FIELDS)
    write_csv(pairwise_effects_path, pairwise_effect_rows, PAIRWISE_EFFECT_FIELDS)
    markdown_path.write_text(
        render_effect_markdown(effect_rows, pairwise_effect_rows),
        encoding="utf-8",
    )
    return EvidenceMatrixResult(
        runs_path=runs_path,
        effects_path=effects_path,
        pairwise_effects_path=pairwise_effects_path,
        markdown_path=markdown_path,
        run_rows=run_rows,
        effect_rows=effect_rows,
        pairwise_effect_rows=pairwise_effect_rows,
    )


def expand_cases(manifest: EvidenceManifest) -> list[MatrixRunSpec]:
    specs: list[MatrixRunSpec] = []
    for case in manifest.cases:
        seeds = case.seeds or manifest.seeds
        for variant in case.variants:
            for seed in seeds:
                specs.append(MatrixRunSpec(case=case, variant=variant, seed=seed))
    return specs


def run_case(
    *,
    manifest: EvidenceManifest,
    spec: MatrixRunSpec,
    handlers: Mapping[str, ToyRunHandler],
) -> dict[str, Any]:
    handler = handlers.get(spec.case.toy)
    if handler is None:
        raise ValueError(f"Unknown evidence matrix toy: {spec.case.toy}")
    config_path = write_case_config(manifest, spec)
    config = handler.loader(config_path)
    result = handler.runner(config, config_path)
    metric_value = extract_metric(
        result,
        spec.case.primary_metric,
        toy=spec.case.toy,
        case=spec.case.name,
    )
    run_dir = Path(str(getattr(result, "run_dir", "")))
    ceiling_fields = case_ceiling_run_fields(
        case=spec.case,
        run_dir=run_dir,
    )
    return {
        "label": manifest.label,
        "case": spec.case.name,
        "toy": spec.case.toy,
        "variant": spec.variant.name,
        "group": spec.variant.group,
        "baseline_group": spec.case.baseline_group,
        "nabm_group": spec.case.nabm_group,
        "seed": spec.seed,
        "primary_metric": spec.case.primary_metric,
        "metric_value": metric_value,
        "direction": spec.case.direction,
        **ceiling_fields,
        **precommitment_trajectory_run_fields(run_dir),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        **sweep_capability_metadata(spec.case.toy),
    }


def write_case_config(manifest: EvidenceManifest, spec: MatrixRunSpec) -> Path:
    raw = deepcopy(load_raw_yaml(spec.case.base_config))
    apply_updates(raw, spec.variant.updates)
    raw.setdefault("run", {})
    raw["run"]["name"] = f"{manifest.label}_{spec.case.name}_{spec.variant.name}"
    raw["run"]["seed"] = spec.seed
    raw["run"]["output_dir"] = str(manifest.runs_dir)
    raw.setdefault("simulation", {})
    raw["simulation"]["epochs"] = (
        spec.case.epochs if spec.case.epochs is not None else manifest.epochs
    )
    validate_coordination(spec.case.toy, raw)

    config_dir = manifest.config_dir / manifest.label
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / (
        f"{safe_name(spec.case.name)}_{safe_name(spec.variant.name)}"
        f"_seed{spec.seed:02d}.yaml"
    )
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config_path


def build_effect_summary(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows_by_case: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        rows_by_case[(str(row["label"]), str(row["case"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (label, case_name), rows in rows_by_case.items():
        first = rows[0]
        baseline_group = str(first.get("baseline_group", "baseline"))
        nabm_group = str(first.get("nabm_group", "nabm"))
        by_group_seed = _group_seed_values(rows)
        baseline_by_seed = by_group_seed.get(baseline_group)
        nabm_by_seed = by_group_seed.get(nabm_group)
        if not baseline_by_seed or not nabm_by_seed:
            raise ValueError(
                f"Case {case_name} requires groups {baseline_group!r} "
                f"and {nabm_group!r}"
            )
        baseline_seed_means = _seed_means(baseline_by_seed)
        nabm_seed_means = _seed_means(nabm_by_seed)
        common_seeds = sorted(set(baseline_seed_means) & set(nabm_seed_means))
        if not common_seeds:
            raise ValueError(f"Case {case_name} has no matched baseline/NABM seeds")
        direction = _validate_direction(str(first["direction"]))
        if direction == "maximize":
            paired_effects = [
                nabm_seed_means[seed] - baseline_seed_means[seed]
                for seed in common_seeds
            ]
        else:
            paired_effects = [
                baseline_seed_means[seed] - nabm_seed_means[seed]
                for seed in common_seeds
            ]

        baseline_stats = summary_stats(
            [baseline_seed_means[seed] for seed in common_seeds]
        )
        nabm_stats = summary_stats([nabm_seed_means[seed] for seed in common_seeds])
        effect_stats = summary_stats(paired_effects)
        baseline_rows = [
            row
            for row in rows
            if str(row["group"]) == baseline_group and int(row["seed"]) in common_seeds
        ]
        nabm_rows = [
            row
            for row in rows
            if str(row["group"]) == nabm_group and int(row["seed"]) in common_seeds
        ]
        summaries.append(
            {
                "label": label,
                "case": case_name,
                "toy": first["toy"],
                "primary_metric": first["primary_metric"],
                "direction": direction,
                "baseline_group": baseline_group,
                "nabm_group": nabm_group,
                **_prefixed_stats("baseline", baseline_stats),
                **_prefixed_stats("nabm", nabm_stats),
                **_prefixed_stats("effect", effect_stats),
                "positive_effect_favors": "nabm",
                **_ceiling_comparison_fields(
                    baseline_rows=baseline_rows,
                    nabm_rows=nabm_rows,
                ),
                **{
                    field: first[field]
                    for field in NABM_ARTIFACT_FIELDS
                    if field in first
                },
            }
        )
    return summaries


def build_pairwise_effect_summary(
    run_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_case: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        rows_by_case[(str(row["label"]), str(row["case"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (label, case_name), rows in rows_by_case.items():
        first = rows[0]
        baseline_group = str(first.get("baseline_group", "baseline"))
        nabm_group = str(first.get("nabm_group", "nabm"))
        baseline_variants = _variant_seed_values(rows, group=baseline_group)
        nabm_variants = _variant_seed_values(rows, group=nabm_group)
        if not baseline_variants or not nabm_variants:
            raise ValueError(
                f"Case {case_name} requires groups {baseline_group!r} "
                f"and {nabm_group!r}"
            )
        direction = _validate_direction(str(first["direction"]))
        for baseline_variant, baseline_by_seed in sorted(baseline_variants.items()):
            baseline_seed_means = _seed_means(baseline_by_seed)
            for nabm_variant, nabm_by_seed in sorted(nabm_variants.items()):
                nabm_seed_means = _seed_means(nabm_by_seed)
                common_seeds = sorted(set(baseline_seed_means) & set(nabm_seed_means))
                if not common_seeds:
                    raise ValueError(
                        f"Case {case_name} has no matched seeds for "
                        f"{baseline_variant} vs {nabm_variant}"
                    )
                paired_effects = [
                    direction_effect(
                        baseline=baseline_seed_means[seed],
                        nabm=nabm_seed_means[seed],
                        direction=direction,
                    )
                    for seed in common_seeds
                ]
                baseline_stats = summary_stats(
                    [baseline_seed_means[seed] for seed in common_seeds]
                )
                nabm_stats = summary_stats(
                    [nabm_seed_means[seed] for seed in common_seeds]
                )
                effect_stats = summary_stats(paired_effects)
                baseline_rows = [
                    row
                    for row in rows
                    if str(row["group"]) == baseline_group
                    and str(row["variant"]) == baseline_variant
                    and int(row["seed"]) in common_seeds
                ]
                nabm_rows = [
                    row
                    for row in rows
                    if str(row["group"]) == nabm_group
                    and str(row["variant"]) == nabm_variant
                    and int(row["seed"]) in common_seeds
                ]
                summaries.append(
                    {
                        "label": label,
                        "case": case_name,
                        "toy": first["toy"],
                        "primary_metric": first["primary_metric"],
                        "direction": direction,
                        "baseline_variant": baseline_variant,
                        "nabm_variant": nabm_variant,
                        **_prefixed_stats("baseline", baseline_stats),
                        **_prefixed_stats("nabm", nabm_stats),
                        **_prefixed_stats("effect", effect_stats),
                        "positive_effect_favors": "nabm",
                        **_ceiling_comparison_fields(
                            baseline_rows=baseline_rows,
                            nabm_rows=nabm_rows,
                        ),
                        **{
                            field: first[field]
                            for field in NABM_ARTIFACT_FIELDS
                            if field in first
                        },
                    }
                )
    return summaries


def summary_stats(values: Sequence[float]) -> SummaryStats:
    if not values:
        raise ValueError("summary_stats requires at least one value")
    n = len(values)
    mean = math.fsum(values) / n
    std = statistics.stdev(values) if n > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return SummaryStats(n=n, mean=mean, std=std, ci95=ci95)


def direction_effect(
    *,
    baseline: float,
    nabm: float,
    direction: EffectDirection,
) -> float:
    if direction == "maximize":
        return nabm - baseline
    if direction == "minimize":
        return baseline - nabm
    raise ValueError(f"Unsupported effect direction: {direction}")


def _ceiling_comparison_fields(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    nabm_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first = next(
        (
            row
            for row in (*baseline_rows, *nabm_rows)
            if row.get("ceiling_metric", "") != ""
        ),
        None,
    )
    if first is None:
        blank_fields = {
            "ceiling_metric": "",
            "ceiling_value": "",
            "ceiling_tolerance": "",
            "baseline_final_ceiling_rate": "",
            "nabm_final_ceiling_rate": "",
            "baseline_ever_ceiling_rate": "",
            "nabm_ever_ceiling_rate": "",
            "baseline_time_to_ceiling_mean": "",
            "nabm_time_to_ceiling_mean": "",
            "time_to_ceiling_effect_mean": "",
            "ceiling_outcome": "",
        }
        for field in CEILING_STABILITY_SUMMARY_FIELDS:
            blank_fields[f"baseline_{field}"] = ""
            blank_fields[f"nabm_{field}"] = ""
        return blank_fields
    baseline_stats = _ceiling_stats(baseline_rows)
    nabm_stats = _ceiling_stats(nabm_rows)
    time_effect = _time_to_ceiling_effect(
        baseline_stats["time_to_ceiling_mean"],
        nabm_stats["time_to_ceiling_mean"],
    )
    return {
        "ceiling_metric": first.get("ceiling_metric", ""),
        "ceiling_value": first.get("ceiling_value", ""),
        "ceiling_tolerance": first.get("ceiling_tolerance", ""),
        "baseline_final_ceiling_rate": baseline_stats["final_ceiling_rate"],
        "nabm_final_ceiling_rate": nabm_stats["final_ceiling_rate"],
        "baseline_ever_ceiling_rate": baseline_stats["ever_ceiling_rate"],
        "nabm_ever_ceiling_rate": nabm_stats["ever_ceiling_rate"],
        "baseline_time_to_ceiling_mean": baseline_stats["time_to_ceiling_mean"],
        "nabm_time_to_ceiling_mean": nabm_stats["time_to_ceiling_mean"],
        "time_to_ceiling_effect_mean": time_effect,
        **{
            f"baseline_{field}": baseline_stats[field]
            for field in CEILING_STABILITY_SUMMARY_FIELDS
        },
        **{
            f"nabm_{field}": nabm_stats[field]
            for field in CEILING_STABILITY_SUMMARY_FIELDS
        },
        "ceiling_outcome": _ceiling_outcome(
            baseline_final_rate=baseline_stats["final_ceiling_rate"],
            nabm_final_rate=nabm_stats["final_ceiling_rate"],
            time_effect=time_effect,
        ),
    }


def _ceiling_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    final_hits = [
        value
        for value in (_optional_bool(row.get("final_within_ceiling")) for row in rows)
        if value is not None
    ]
    ever_hits = [
        value
        for value in (_optional_bool(row.get("ever_reached_ceiling")) for row in rows)
        if value is not None
    ]
    times = [
        value
        for value in (_optional_float(row.get("time_to_ceiling")) for row in rows)
        if value is not None
    ]
    ever_final_misses = [
        value
        for value in (
            _optional_bool(row.get("ever_ceiling_final_miss")) for row in rows
        )
        if value is not None
    ]
    late_flip_counts = [
        value
        for value in (
            _optional_float(row.get("late_flip_count_after_first_ceiling"))
            for row in rows
        )
        if value is not None
    ]
    late_flip_rates = [
        value
        for value in (
            _optional_float(row.get("late_flip_rate_after_first_ceiling"))
            for row in rows
        )
        if value is not None
    ]
    terminal_rates = [
        value
        for value in (
            _optional_float(row.get("terminal_window_ceiling_rate")) for row in rows
        )
        if value is not None
    ]
    terminal_means = [
        value
        for value in (
            _optional_float(row.get("terminal_window_mean_ceiling_metric"))
            for row in rows
        )
        if value is not None
    ]
    return {
        "final_ceiling_rate": _rate_or_blank(final_hits),
        "ever_ceiling_rate": _rate_or_blank(ever_hits),
        "time_to_ceiling_mean": _mean_or_blank(times),
        "ever_ceiling_final_miss_rate": _rate_or_blank(ever_final_misses),
        "late_flip_count_after_first_ceiling_mean": _mean_or_blank(late_flip_counts),
        "late_flip_rate_after_first_ceiling_mean": _mean_or_blank(late_flip_rates),
        "terminal_window_ceiling_rate_mean": _mean_or_blank(terminal_rates),
        "terminal_window_mean_ceiling_metric_mean": _mean_or_blank(terminal_means),
    }


def _rate_or_blank(values: Sequence[bool]) -> float | str:
    if not values:
        return ""
    return math.fsum(1.0 if value else 0.0 for value in values) / len(values)


def _mean_or_blank(values: Sequence[float]) -> float | str:
    if not values:
        return ""
    return math.fsum(values) / len(values)


def _time_to_ceiling_effect(
    baseline_time: Any,
    nabm_time: Any,
) -> float | str:
    baseline = _optional_float(baseline_time)
    nabm = _optional_float(nabm_time)
    if baseline is None or nabm is None:
        return ""
    return baseline - nabm


def _ceiling_outcome(
    *,
    baseline_final_rate: Any,
    nabm_final_rate: Any,
    time_effect: Any,
) -> str:
    baseline_rate = _optional_float(baseline_final_rate)
    nabm_rate = _optional_float(nabm_final_rate)
    if baseline_rate is None or nabm_rate is None:
        return ""
    if nabm_rate > baseline_rate:
        return "nabm_more_final_ceiling_hits"
    if nabm_rate < baseline_rate:
        return "baseline_more_final_ceiling_hits"
    resolved_time_effect = _optional_float(time_effect)
    if resolved_time_effect is None:
        return "ceiling_tie"
    if resolved_time_effect > 0.0:
        return "nabm_faster_to_ceiling"
    if resolved_time_effect < 0.0:
        return "baseline_faster_to_ceiling"
    return "ceiling_tie_equal_time"


def _optional_float(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value == "" or value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def extract_metric(result: Any, metric: str, *, toy: str, case: str) -> float:
    if hasattr(result, metric):
        return _as_finite_float(getattr(result, metric), metric=metric)
    domain_metrics = getattr(result, "domain_metrics", {})
    if isinstance(domain_metrics, Mapping) and metric in domain_metrics:
        return _as_finite_float(domain_metrics[metric], metric=metric)
    raise ValueError(f"Missing primary metric {metric!r} for {toy} case {case}")


def case_ceiling_run_fields(
    *,
    case: MatrixCase,
    run_dir: Path,
) -> dict[str, Any]:
    if case.ceiling_metric is None or case.ceiling_value is None:
        return {
            "ceiling_metric": "",
            "ceiling_value": "",
            "ceiling_tolerance": "",
            "ceiling_metric_value": "",
            "ceiling_gap": "",
            "final_within_ceiling": "",
            "ever_reached_ceiling": "",
            "time_to_ceiling": "",
            **{field: "" for field in CEILING_STABILITY_RUN_FIELDS},
        }
    ceiling_metric_value = final_aggregate_metric(run_dir, case.ceiling_metric)
    time = time_to_ceiling(
        run_dir=run_dir,
        metric=case.ceiling_metric,
        ceiling_value=case.ceiling_value,
        tolerance=case.ceiling_tolerance,
    )
    if ceiling_metric_value is None:
        gap: float | str = ""
        final_within: bool | str = ""
    else:
        gap = ceiling_gap(
            metric_value=ceiling_metric_value,
            ceiling_value=case.ceiling_value,
        )
        final_within = ceiling_metric_value >= (
            case.ceiling_value - case.ceiling_tolerance
        )
    return {
        "ceiling_metric": case.ceiling_metric,
        "ceiling_value": case.ceiling_value,
        "ceiling_tolerance": case.ceiling_tolerance,
        "ceiling_metric_value": (
            "" if ceiling_metric_value is None else ceiling_metric_value
        ),
        "ceiling_gap": gap,
        "final_within_ceiling": final_within,
        "ever_reached_ceiling": time != "",
        "time_to_ceiling": time,
        **ceiling_stability_run_fields(
            run_dir=run_dir,
            metric=case.ceiling_metric,
            ceiling_value=case.ceiling_value,
            tolerance=case.ceiling_tolerance,
            time_to_ceiling_value=time,
            final_within_ceiling=final_within,
        ),
    }


def ceiling_gap(*, metric_value: float, ceiling_value: float) -> float:
    return max(0.0, ceiling_value - metric_value)


def final_aggregate_metric(run_dir: Path, metric: str) -> float | None:
    rows = aggregate_metric_rows(run_dir)
    if not rows:
        return None
    raw_value = rows[-1].get(metric, "")
    if raw_value == "":
        return None
    return _as_finite_float(raw_value, metric=metric)


def ceiling_stability_run_fields(
    *,
    run_dir: Path,
    metric: str,
    ceiling_value: float,
    tolerance: float,
    time_to_ceiling_value: int | str,
    final_within_ceiling: bool | str,
    terminal_window_size: int = TERMINAL_CEILING_WINDOW,
) -> dict[str, Any]:
    rows = aggregate_metric_rows(run_dir)
    if not rows:
        return {field: "" for field in CEILING_STABILITY_RUN_FIELDS}

    threshold = ceiling_value - tolerance
    terminal_rows = rows[-terminal_window_size:]
    terminal_values = _metric_values(terminal_rows, metric=metric)
    first_ceiling_epoch = _optional_float(time_to_ceiling_value)

    if terminal_values:
        terminal_ceiling_rate: float | str = _rate_or_blank(
            [value >= threshold for value in terminal_values]
        )
        terminal_mean: float | str = _mean_or_blank(terminal_values)
        actual_window_size: int | str = len(terminal_values)
    else:
        terminal_ceiling_rate = ""
        terminal_mean = ""
        actual_window_size = ""

    if first_ceiling_epoch is None:
        late_flip_count: float | str = ""
        late_flip_rate: float | str = ""
    else:
        post_ceiling_rows = [
            row
            for row in rows
            if (
                epoch := _optional_float(row.get("epoch", ""))
            ) is not None
            and epoch > first_ceiling_epoch
        ]
        late_flip_count = _sum_optional_numeric(
            row.get("action_flip_count", "") for row in post_ceiling_rows
        )
        late_flip_rate = _mean_or_blank(
            [
                value
                for value in (
                    _optional_float(row.get("action_flip_rate", ""))
                    for row in post_ceiling_rows
                )
                if value is not None
            ]
        )

    final_within = _optional_bool(final_within_ceiling)
    return {
        "ever_ceiling_final_miss": (
            "" if final_within is None else first_ceiling_epoch is not None and not final_within
        ),
        "late_flip_count_after_first_ceiling": late_flip_count,
        "late_flip_rate_after_first_ceiling": late_flip_rate,
        "terminal_window_size": actual_window_size,
        "terminal_window_ceiling_rate": terminal_ceiling_rate,
        "terminal_window_mean_ceiling_metric": terminal_mean,
    }


def aggregate_metric_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "aggregate_metrics.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def precommitment_trajectory_run_fields(run_dir: Path) -> dict[str, Any]:
    rows = aggregate_metric_rows(run_dir)
    if not rows:
        return {field: "" for field in PRECOMMITMENT_TRAJECTORY_RUN_FIELDS}
    final_row = rows[-1]
    return {
        field: final_row.get(field, "")
        for field in PRECOMMITMENT_TRAJECTORY_RUN_FIELDS
    }


def _metric_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> list[float]:
    return [
        value
        for value in (_optional_float(row.get(metric, "")) for row in rows)
        if value is not None
    ]


def _sum_optional_numeric(values: Sequence[Any]) -> float | str:
    numbers = [
        number
        for number in (_optional_float(value) for value in values)
        if number is not None
    ]
    if not numbers:
        return ""
    return math.fsum(numbers)


def time_to_ceiling(
    *,
    run_dir: Path,
    metric: str,
    ceiling_value: float,
    tolerance: float,
) -> int | str:
    threshold = ceiling_value - tolerance
    for row in aggregate_metric_rows(run_dir):
        raw_value = row.get(metric, "")
        if raw_value == "":
            continue
        value = _as_finite_float(raw_value, metric=metric)
        if value < threshold:
            continue
        raw_epoch = row.get("epoch", "")
        if raw_epoch == "":
            return ""
        return int(float(raw_epoch))
    return ""


def apply_updates(raw: dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if "." in key:
            set_dotted(raw, key, value)
        elif isinstance(value, Mapping) and isinstance(raw.get(key), dict):
            apply_updates(raw[key], value)
        else:
            raw[key] = deepcopy(value)


def set_dotted(raw: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = raw
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set {dotted_key}: {part} is not a mapping")
        cursor = next_value
    cursor[parts[-1]] = deepcopy(value)


def validate_coordination(toy: str, raw: Mapping[str, Any]) -> None:
    coordination = coordination_section(raw)
    if not coordination:
        return
    mixer = str(coordination.get("mixer", ""))
    peer_rule = str(coordination.get("peer_rule", ""))
    if not supports_coordination(toy, mixer, peer_rule):
        raise ValueError(
            f"Unsupported evidence matrix coordination for {toy}: "
            f"{mixer}/{peer_rule}"
        )


def coordination_section(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    model = raw.get("model")
    if isinstance(model, Mapping):
        coordination = model.get("coordination")
        if isinstance(coordination, Mapping):
            return coordination
    coordination = raw.get("coordination")
    if isinstance(coordination, Mapping):
        return coordination
    social = raw.get("social")
    if isinstance(social, Mapping):
        return social
    return {}


def load_raw_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return raw


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_effect_markdown(
    rows: Sequence[Mapping[str, Any]],
    pairwise_rows: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    has_ceiling = _has_ceiling_metadata(rows) or _has_ceiling_metadata(
        pairwise_rows or []
    )
    lines = [
        "# NABM Effect Matrix",
        "",
        "## Grouped Effects",
        "",
    ]
    if has_ceiling:
        lines.extend(
            [
                "| Case | Toy | Metric | Direction | Baseline Mean | NABM Mean | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N | Ever-Final Miss B/N | Terminal Ceiling B/N |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| Case | Toy | Metric | Direction | Baseline Mean | NABM Mean | Effect | 95% CI |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
    for row in rows:
        if has_ceiling:
            lines.append(
                "| {case} | {toy} | {metric} | {direction} | {baseline:.6g} | "
                "{nabm:.6g} | {effect:.6g} | {ci95:.6g} | {outcome} | "
                "{baseline_rate}/{nabm_rate} | {baseline_time}/{nabm_time} | "
                "{baseline_final_miss}/{nabm_final_miss} | "
                "{baseline_terminal}/{nabm_terminal} |".format(
                    case=row["case"],
                    toy=row["toy"],
                    metric=row["primary_metric"],
                    direction=row["direction"],
                    baseline=float(row["baseline_mean"]),
                    nabm=float(row["nabm_mean"]),
                    effect=float(row["effect_mean"]),
                    ci95=float(row["effect_ci95"]),
                    outcome=row.get("ceiling_outcome", ""),
                    baseline_rate=_format_optional_float(
                        row.get("baseline_final_ceiling_rate", "")
                    ),
                    nabm_rate=_format_optional_float(
                        row.get("nabm_final_ceiling_rate", "")
                    ),
                    baseline_time=_format_optional_float(
                        row.get("baseline_time_to_ceiling_mean", "")
                    ),
                    nabm_time=_format_optional_float(
                        row.get("nabm_time_to_ceiling_mean", "")
                    ),
                    baseline_final_miss=_format_optional_float(
                        row.get("baseline_ever_ceiling_final_miss_rate", "")
                    ),
                    nabm_final_miss=_format_optional_float(
                        row.get("nabm_ever_ceiling_final_miss_rate", "")
                    ),
                    baseline_terminal=_format_optional_float(
                        row.get("baseline_terminal_window_ceiling_rate_mean", "")
                    ),
                    nabm_terminal=_format_optional_float(
                        row.get("nabm_terminal_window_ceiling_rate_mean", "")
                    ),
                )
            )
        else:
            lines.append(
                "| {case} | {toy} | {metric} | {direction} | {baseline:.6g} | "
                "{nabm:.6g} | {effect:.6g} | {ci95:.6g} |".format(
                    case=row["case"],
                    toy=row["toy"],
                    metric=row["primary_metric"],
                    direction=row["direction"],
                    baseline=float(row["baseline_mean"]),
                    nabm=float(row["nabm_mean"]),
                    effect=float(row["effect_mean"]),
                    ci95=float(row["effect_ci95"]),
                )
            )
    lines.append("")
    if pairwise_rows:
        lines.extend(
            [
                "## Pairwise Baseline Effects",
                "",
            ]
        )
        if has_ceiling:
            lines.extend(
                [
                    "| Case | Toy | Baseline Variant | NABM Variant | Effect | 95% CI | Ceiling Outcome | Final Ceiling Rate B/N | Time To Ceiling B/N | Ever-Final Miss B/N | Terminal Ceiling B/N |",
                    "| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
                ]
            )
        else:
            lines.extend(
                [
                    "| Case | Toy | Baseline Variant | NABM Variant | Effect | 95% CI |",
                    "| --- | --- | --- | --- | ---: | ---: |",
                ]
            )
        for row in pairwise_rows:
            if has_ceiling:
                lines.append(
                    "| {case} | {toy} | {baseline} | {nabm} | {effect:.6g} | "
                    "{ci95:.6g} | {outcome} | {baseline_rate}/{nabm_rate} | "
                    "{baseline_time}/{nabm_time} | "
                    "{baseline_final_miss}/{nabm_final_miss} | "
                    "{baseline_terminal}/{nabm_terminal} |".format(
                        case=row["case"],
                        toy=row["toy"],
                        baseline=row["baseline_variant"],
                        nabm=row["nabm_variant"],
                        effect=float(row["effect_mean"]),
                        ci95=float(row["effect_ci95"]),
                        outcome=row.get("ceiling_outcome", ""),
                        baseline_rate=_format_optional_float(
                            row.get("baseline_final_ceiling_rate", "")
                        ),
                        nabm_rate=_format_optional_float(
                            row.get("nabm_final_ceiling_rate", "")
                        ),
                        baseline_time=_format_optional_float(
                            row.get("baseline_time_to_ceiling_mean", "")
                        ),
                        nabm_time=_format_optional_float(
                            row.get("nabm_time_to_ceiling_mean", "")
                        ),
                        baseline_final_miss=_format_optional_float(
                            row.get("baseline_ever_ceiling_final_miss_rate", "")
                        ),
                        nabm_final_miss=_format_optional_float(
                            row.get("nabm_ever_ceiling_final_miss_rate", "")
                        ),
                        baseline_terminal=_format_optional_float(
                            row.get("baseline_terminal_window_ceiling_rate_mean", "")
                        ),
                        nabm_terminal=_format_optional_float(
                            row.get("nabm_terminal_window_ceiling_rate_mean", "")
                        ),
                    )
                )
            else:
                lines.append(
                    "| {case} | {toy} | {baseline} | {nabm} | {effect:.6g} | "
                    "{ci95:.6g} |".format(
                        case=row["case"],
                        toy=row["toy"],
                        baseline=row["baseline_variant"],
                        nabm=row["nabm_variant"],
                        effect=float(row["effect_mean"]),
                        ci95=float(row["effect_ci95"]),
                    )
                )
        lines.append("")
    lines.append("Positive effect values favor the NABM group.")
    return "\n".join(lines) + "\n"


def _has_ceiling_metadata(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(row.get("ceiling_metric", "") != "" for row in rows)


def _format_optional_float(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return ""
    return f"{number:.6g}"


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _case_from_mapping(raw: Any) -> MatrixCase:
    if not isinstance(raw, Mapping):
        raise ValueError("Evidence case must be a mapping")
    required = {"toy", "name", "base_config", "primary_metric", "direction", "variants"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Evidence case missing field(s): {', '.join(missing)}")
    direction = _validate_direction(str(raw["direction"]))
    variants_raw = raw["variants"]
    if not isinstance(variants_raw, Sequence) or isinstance(variants_raw, str | bytes):
        raise ValueError(f"Evidence case {raw['name']} variants must be a list")
    variants = tuple(_variant_from_mapping(item) for item in variants_raw)
    if not variants:
        raise ValueError(f"Evidence case {raw['name']} must contain variants")
    seeds = None
    if "seeds" in raw:
        seeds = tuple(int(seed) for seed in _required_sequence(raw, "seeds"))
    epochs = int(raw["epochs"]) if "epochs" in raw else None
    ceiling_metric = raw.get("ceiling_metric")
    ceiling_value = raw.get("ceiling_value")
    if (ceiling_metric is None) != (ceiling_value is None):
        raise ValueError(
            f"Evidence case {raw['name']} must set ceiling_metric "
            "and ceiling_value together"
        )
    ceiling_tolerance = float(raw.get("ceiling_tolerance", 0.0))
    return MatrixCase(
        toy=str(raw["toy"]),
        name=str(raw["name"]),
        base_config=Path(raw["base_config"]),
        primary_metric=str(raw["primary_metric"]),
        direction=direction,
        variants=variants,
        baseline_group=str(raw.get("baseline_group", "baseline")),
        nabm_group=str(raw.get("nabm_group", "nabm")),
        seeds=seeds,
        epochs=epochs,
        ceiling_metric=None if ceiling_metric is None else str(ceiling_metric),
        ceiling_value=None if ceiling_value is None else float(ceiling_value),
        ceiling_tolerance=ceiling_tolerance,
    )


def _variant_from_mapping(raw: Any) -> MatrixVariant:
    if not isinstance(raw, Mapping):
        raise ValueError("Evidence variant must be a mapping")
    required = {"name", "group", "updates"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Evidence variant missing field(s): {', '.join(missing)}")
    updates = raw["updates"]
    if not isinstance(updates, Mapping):
        raise ValueError(f"Evidence variant {raw['name']} updates must be a mapping")
    return MatrixVariant(
        name=str(raw["name"]),
        group=str(raw["group"]),
        updates=updates,
    )


def _required_sequence(raw: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = raw[key]
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"Evidence manifest field {key} must be a list")
    return value


def _validate_direction(value: str) -> EffectDirection:
    if value not in {"maximize", "minimize"}:
        raise ValueError(f"Unsupported effect direction: {value}")
    return value  # type: ignore[return-value]


def _as_finite_float(value: Any, *, metric: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Metric {metric!r} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Metric {metric!r} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"Metric {metric!r} is not finite")
    return number


def _group_seed_values(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, list[float]]]:
    values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        values[str(row["group"])][int(row["seed"])].append(
            _as_finite_float(row["metric_value"], metric=str(row["primary_metric"]))
        )
    return values


def _variant_seed_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    group: str,
) -> dict[str, dict[int, list[float]]]:
    values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row["group"]) != group:
            continue
        values[str(row["variant"])][int(row["seed"])].append(
            _as_finite_float(row["metric_value"], metric=str(row["primary_metric"]))
        )
    return values


def _seed_means(seed_values: Mapping[int, Sequence[float]]) -> dict[int, float]:
    return {seed: math.fsum(values) / len(values) for seed, values in seed_values.items()}


def _prefixed_stats(prefix: str, stats: SummaryStats) -> dict[str, Any]:
    return {
        f"{prefix}_n": stats.n,
        f"{prefix}_mean": stats.mean,
        f"{prefix}_std": stats.std,
        f"{prefix}_ci95": stats.ci95,
    }
