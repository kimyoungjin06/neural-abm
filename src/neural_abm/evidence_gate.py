"""Pass/fail evidence gates for audited NABM claims."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from neural_abm.evidence_matrix import (
    EvidenceManifest,
    MatrixCase,
    MatrixVariant,
    load_manifest,
)


GateStatus = Literal["pass", "fail", "inconclusive"]
REQUIRED_RUN_COLUMNS = (
    "label",
    "case",
    "toy",
    "variant",
    "group",
    "seed",
    "primary_metric",
    "metric_value",
    "direction",
    "final_within_ceiling",
    "time_to_ceiling",
)


class EvidenceGateInputError(ValueError):
    """Raised when run rows do not match the evidence manifest contract."""


@dataclass(frozen=True)
class EvidenceGateCaseCriterion:
    final_ceiling_min_hits: int
    mean_time_to_ceiling_lt: float


@dataclass(frozen=True)
class EvidenceGateCriteria:
    main_group: str
    require_without_teacher_bootstrap_replay: bool
    cases: Mapping[str, EvidenceGateCaseCriterion]


@dataclass
class EvidenceGateResult:
    json_path: Path
    markdown_path: Path
    summary: dict[str, Any]


def load_gate_manifest(path: Path) -> tuple[EvidenceManifest, EvidenceGateCriteria]:
    manifest = load_manifest(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected mapping YAML at {path}")
    criteria = gate_criteria_from_mapping(
        raw.get("success_criteria"),
        manifest=manifest,
    )
    return manifest, criteria


def gate_criteria_from_mapping(
    raw: Any,
    *,
    manifest: EvidenceManifest,
) -> EvidenceGateCriteria:
    if not isinstance(raw, Mapping):
        raise ValueError("Evidence gate requires a success_criteria mapping")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, Mapping):
        raise ValueError("Evidence gate success_criteria.cases must be a mapping")

    cases: dict[str, EvidenceGateCaseCriterion] = {}
    for case in manifest.cases:
        case_raw = cases_raw.get(case.name)
        if not isinstance(case_raw, Mapping):
            raise ValueError(
                f"Evidence gate missing success criteria for case {case.name!r}"
            )
        cases[case.name] = _case_criterion_from_mapping(case.name, case_raw)

    return EvidenceGateCriteria(
        main_group=str(raw.get("main_group", "nabm")),
        require_without_teacher_bootstrap_replay=bool(
            raw.get("require_without_teacher_bootstrap_replay", True)
        ),
        cases=cases,
    )


def run_evidence_gate(
    manifest_path: Path,
    *,
    runs_path: Path | None = None,
    matrix_results_dir: Path = Path("experiments/results/nabm_effect_matrix"),
    output_dir: Path = Path("experiments/evidence/results"),
    seeds: Sequence[int] | None = None,
) -> EvidenceGateResult:
    manifest, criteria = load_gate_manifest(manifest_path)
    if seeds is not None:
        manifest = replace(manifest, seeds=tuple(int(seed) for seed in seeds))
    resolved_runs_path = (
        runs_path
        if runs_path is not None
        else matrix_results_dir / f"{manifest.label}_runs.csv"
    )
    run_rows = read_csv_rows(resolved_runs_path)
    summary = evaluate_evidence_gate(
        manifest=manifest,
        criteria=criteria,
        run_rows=run_rows,
        runs_path=resolved_runs_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{manifest.label}.summary.json"
    markdown_path = output_dir / f"{manifest.label}.summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_gate_markdown(summary), encoding="utf-8")
    return EvidenceGateResult(
        json_path=json_path,
        markdown_path=markdown_path,
        summary=summary,
    )


def evaluate_evidence_gate(
    *,
    manifest: EvidenceManifest,
    criteria: EvidenceGateCriteria,
    run_rows: Sequence[Mapping[str, Any]],
    runs_path: Path | None = None,
) -> dict[str, Any]:
    rows_by_case_variant, input_validation = validate_run_rows(
        manifest=manifest,
        run_rows=run_rows,
    )

    case_summaries = [
        _evaluate_case(
            manifest=manifest,
            case=case,
            criteria=criteria,
            rows_by_case_variant=rows_by_case_variant,
        )
        for case in manifest.cases
    ]
    malformed_rows = _collect_malformed_rows(case_summaries)
    input_validation["malformed_rows"] = malformed_rows
    overall_status = _overall_status(case_summaries)
    return {
        "label": manifest.label,
        "status": overall_status,
        "passed": overall_status == "pass",
        "main_group": criteria.main_group,
        "require_without_teacher_bootstrap_replay": (
            criteria.require_without_teacher_bootstrap_replay
        ),
        "runs_path": "" if runs_path is None else str(runs_path),
        "input_validation": input_validation,
        "cases": case_summaries,
    }


def validate_run_rows(
    *,
    manifest: EvidenceManifest,
    run_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], list[Mapping[str, Any]]], dict[str, Any]]:
    cases = {case.name: case for case in manifest.cases}
    variants_by_case = {
        case.name: {variant.name: variant for variant in case.variants}
        for case in manifest.cases
    }
    expected_keys = {
        (case.name, variant.name, int(seed))
        for case in manifest.cases
        for variant in case.variants
        for seed in (case.seeds or manifest.seeds)
    }
    rows_by_case_variant: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    seen_keys: set[tuple[str, str, int]] = set()
    validation: dict[str, Any] = {
        "status": "valid",
        "required_columns": list(REQUIRED_RUN_COLUMNS),
        "missing_runs": [],
        "malformed_rows": [],
        "duplicate_rows": [],
        "unknown_rows": [],
        "missing_required_columns": [],
    }

    for index, row in enumerate(run_rows, start=2):
        missing_columns = [
            column for column in REQUIRED_RUN_COLUMNS if row.get(column) is None
        ]
        if missing_columns:
            validation["missing_required_columns"].append(
                {"row": index, "columns": missing_columns}
            )
            continue

        label = str(row["label"])
        case_name = str(row["case"])
        variant_name = str(row["variant"])
        seed = _parse_seed(row["seed"])
        if (
            label != manifest.label
            or case_name not in cases
            or variant_name not in variants_by_case.get(case_name, {})
            or seed is None
        ):
            validation["unknown_rows"].append(
                {
                    "row": index,
                    "label": label,
                    "case": case_name,
                    "variant": variant_name,
                    "seed": row["seed"],
                }
            )
            continue

        case = cases[case_name]
        variant = variants_by_case[case_name][variant_name]
        expected_seeds = set(int(value) for value in (case.seeds or manifest.seeds))
        mismatch = _manifest_row_mismatch(row=row, case=case, variant=variant)
        if seed not in expected_seeds or mismatch:
            validation["unknown_rows"].append(
                {
                    "row": index,
                    "label": label,
                    "case": case_name,
                    "variant": variant_name,
                    "seed": seed,
                    "mismatch": mismatch,
                }
            )
            continue

        key = (case_name, variant_name, seed)
        if key in seen_keys:
            validation["duplicate_rows"].append(
                {
                    "row": index,
                    "case": case_name,
                    "variant": variant_name,
                    "seed": seed,
                }
            )
            continue
        seen_keys.add(key)
        rows_by_case_variant[(case_name, variant_name)].append(row)

    missing_runs = [
        {"case": case, "variant": variant, "seed": seed}
        for case, variant, seed in sorted(expected_keys - seen_keys)
    ]
    validation["missing_runs"] = missing_runs
    fatal_sections = (
        "missing_required_columns",
        "unknown_rows",
        "duplicate_rows",
    )
    if any(validation[name] for name in fatal_sections):
        validation["status"] = "error"
        raise EvidenceGateInputError(_render_input_validation_error(validation))
    return rows_by_case_variant, validation


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Evidence gate run rows not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_gate_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# Evidence Gate: {summary['label']}",
        "",
        f"Overall status: **{summary['status']}**",
        "",
        "## Main Claim Cases",
        "",
        "| Case | Toy | Status | Best Main Variant | Final Ceiling Hits | Mean TtC | Trajectory | Failure Mode | Ever-Final Misses | Terminal Rate | Late Flip Rate | Baseline Improved |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for case in summary["cases"]:
        best_main = case.get("best_main_variant") or {}
        lines.append(
            "| {case} | {toy} | {status} | {variant} | {hits}/{expected} | "
            "{ttc} | {trajectory} | {failure_mode} | {final_miss} | "
            "{terminal_rate} | {late_flip_rate} | {improved} |".format(
                case=case["case"],
                toy=case["toy"],
                status=case["status"],
                variant=best_main.get("variant", ""),
                hits=_format_optional_number(best_main.get("final_ceiling_hits")),
                expected=_format_optional_number(best_main.get("expected_seed_count")),
                ttc=_format_optional_number(best_main.get("mean_time_to_ceiling")),
                trajectory=best_main.get("trajectory_status", ""),
                failure_mode=best_main.get("failure_mode", ""),
                final_miss=_format_optional_number(
                    best_main.get("ever_ceiling_final_miss_count")
                ),
                terminal_rate=_format_optional_number(
                    best_main.get("terminal_window_ceiling_rate_mean")
                ),
                late_flip_rate=_format_optional_number(
                    best_main.get("late_flip_rate_after_first_ceiling_mean")
                ),
                improved=_format_optional_bool(
                    case.get("baseline_improved_by_best_main")
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Next Diagnostics",
            "",
        ]
    )
    diagnostics = [
        _case_diagnostic_hint(case)
        for case in summary["cases"]
        if case["status"] != "pass"
    ]
    if diagnostics:
        lines.extend(f"- {hint}" for hint in diagnostics)
    else:
        lines.append("- Gate passed; preserve these artifacts before expanding claims.")
    lines.extend(
        [
            "",
            "## Variant Details",
            "",
            "| Case | Variant | Group | Eligible Main | Status | Final Hits | Mean TtC | Trajectory | Failure Mode | Ever-Final Misses | Terminal Rate | Late Flip Rate | Metric Mean | Reasons |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for case in summary["cases"]:
        for variant in case["variants"]:
            lines.append(
                "| {case} | {variant} | {group} | {eligible} | {status} | "
                "{hits} | {ttc} | {trajectory} | {failure_mode} | "
                "{final_miss} | {terminal_rate} | {late_flip_rate} | "
                "{metric} | {reasons} |".format(
                    case=case["case"],
                    variant=variant["variant"],
                    group=variant["group"],
                    eligible=_format_optional_bool(variant["eligible_main"]),
                    status=variant["status"],
                    hits=_format_optional_number(variant["final_ceiling_hits"]),
                    ttc=_format_optional_number(variant["mean_time_to_ceiling"]),
                    trajectory=variant.get("trajectory_status", ""),
                    failure_mode=variant.get("failure_mode", ""),
                    final_miss=_format_optional_number(
                        variant.get("ever_ceiling_final_miss_count")
                    ),
                    terminal_rate=_format_optional_number(
                        variant.get("terminal_window_ceiling_rate_mean")
                    ),
                    late_flip_rate=_format_optional_number(
                        variant.get("late_flip_rate_after_first_ceiling_mean")
                    ),
                    metric=_format_optional_number(variant["metric_mean"]),
                    reasons="; ".join(variant["reasons"]),
                )
            )
    return "\n".join(lines) + "\n"


def _evaluate_case(
    *,
    manifest: EvidenceManifest,
    case: MatrixCase,
    criteria: EvidenceGateCriteria,
    rows_by_case_variant: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    criterion = criteria.cases[case.name]
    expected_seeds = tuple(case.seeds or manifest.seeds)
    variants = [
        _evaluate_variant(
            variant=variant,
            rows=rows_by_case_variant.get((case.name, variant.name), ()),
            expected_seeds=expected_seeds,
            criterion=criterion,
            main_group=criteria.main_group,
            require_without_teacher_bootstrap_replay=(
                criteria.require_without_teacher_bootstrap_replay
            ),
        )
        for variant in case.variants
    ]
    eligible_main = [variant for variant in variants if variant["eligible_main"]]
    passing_main = [variant for variant in eligible_main if variant["status"] == "pass"]
    best_main = _best_variant(eligible_main)
    baseline_variants = [
        variant for variant in variants if variant["group"] == case.baseline_group
    ]
    best_baseline = _best_variant(baseline_variants)
    status = _case_status(eligible_main, passing_main)
    return {
        "case": case.name,
        "toy": case.toy,
        "status": status,
        "passed": status == "pass",
        "criteria": {
            "final_ceiling_min_hits": criterion.final_ceiling_min_hits,
            "mean_time_to_ceiling_lt": criterion.mean_time_to_ceiling_lt,
        },
        "best_main_variant": best_main,
        "best_baseline_variant": best_baseline,
        "baseline_comparison": _baseline_comparison(
            best_main=best_main,
            best_baseline=best_baseline,
        ),
        "baseline_improved_by_best_main": _baseline_improved(
            best_main=best_main,
            best_baseline=best_baseline,
        ),
        "variants": variants,
    }


def _evaluate_variant(
    *,
    variant: MatrixVariant,
    rows: Sequence[Mapping[str, Any]],
    expected_seeds: Sequence[int],
    criterion: EvidenceGateCaseCriterion,
    main_group: str,
    require_without_teacher_bootstrap_replay: bool,
) -> dict[str, Any]:
    stats = _variant_stats(rows=rows, expected_seeds=expected_seeds)
    uses_bootstrap = variant_uses_teacher_bootstrap_replay(variant)
    eligible_main = variant.group == main_group and (
        not require_without_teacher_bootstrap_replay or not uses_bootstrap
    )
    status, reasons = _variant_status_and_reasons(
        stats=stats,
        criterion=criterion,
        eligible_main=eligible_main,
        uses_teacher_bootstrap_replay=uses_bootstrap,
    )
    trajectory = _variant_trajectory_outcome(
        stats=stats,
        criterion=criterion,
        eligible_main=eligible_main,
        status=status,
        uses_teacher_bootstrap_replay=uses_bootstrap,
    )
    return {
        "variant": variant.name,
        "group": variant.group,
        "eligible_main": eligible_main,
        "uses_teacher_bootstrap_replay": uses_bootstrap,
        "status": status,
        "reasons": reasons,
        **trajectory,
        **stats,
    }


def _variant_stats(
    *,
    rows: Sequence[Mapping[str, Any]],
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    expected = tuple(int(seed) for seed in expected_seeds)
    expected_set = set(expected)
    selected_rows = [
        row
        for row in rows
        if _optional_int(row.get("seed")) is not None
        and _optional_int(row.get("seed")) in expected_set
    ]
    observed_seeds = sorted(
        {
            seed
            for row in selected_rows
            if (seed := _optional_int(row.get("seed"))) is not None
        }
    )
    missing_seeds = [seed for seed in expected if seed not in set(observed_seeds)]
    final_hits = 0
    ever_hits = 0
    ever_final_misses = 0
    times: list[float] = []
    metric_values: list[float] = []
    late_flip_counts: list[float] = []
    late_flip_rates: list[float] = []
    terminal_rates: list[float] = []
    terminal_means: list[float] = []
    malformed_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        seed = _optional_int(row.get("seed"))
        malformed_fields: list[str] = []
        final_hit = _optional_bool(row.get("final_within_ceiling"))
        if final_hit is None:
            malformed_fields.append("final_within_ceiling")
        elif final_hit:
            final_hits += 1

        ever_hit = _optional_bool(row.get("ever_reached_ceiling"))
        if row.get("ever_reached_ceiling") not in {None, ""} and ever_hit is None:
            malformed_fields.append("ever_reached_ceiling")
        elif ever_hit:
            ever_hits += 1

        ever_final_miss = _optional_bool(row.get("ever_ceiling_final_miss"))
        if (
            row.get("ever_ceiling_final_miss") not in {None, ""}
            and ever_final_miss is None
        ):
            malformed_fields.append("ever_ceiling_final_miss")
        elif ever_final_miss:
            ever_final_misses += 1

        raw_time = row.get("time_to_ceiling")
        time = _optional_float(raw_time)
        if raw_time not in {None, ""} and time is None:
            malformed_fields.append("time_to_ceiling")
        elif time is not None:
            times.append(time)
        if final_hit is True and time is None:
            malformed_fields.append("time_to_ceiling")

        metric = _optional_float(row.get("metric_value"))
        if metric is None:
            malformed_fields.append("metric_value")
        else:
            metric_values.append(metric)

        _append_optional_metric(
            row=row,
            field="late_flip_count_after_first_ceiling",
            values=late_flip_counts,
            malformed_fields=malformed_fields,
        )
        _append_optional_metric(
            row=row,
            field="late_flip_rate_after_first_ceiling",
            values=late_flip_rates,
            malformed_fields=malformed_fields,
        )
        _append_optional_metric(
            row=row,
            field="terminal_window_ceiling_rate",
            values=terminal_rates,
            malformed_fields=malformed_fields,
        )
        _append_optional_metric(
            row=row,
            field="terminal_window_mean_ceiling_metric",
            values=terminal_means,
            malformed_fields=malformed_fields,
        )

        if malformed_fields:
            malformed_rows.append(
                {
                    "seed": seed,
                    "fields": sorted(set(malformed_fields)),
                }
            )
    return {
        "expected_seed_count": len(expected),
        "observed_seed_count": len(observed_seeds),
        "observed_seeds": observed_seeds,
        "missing_seeds": missing_seeds,
        "complete": not missing_seeds,
        "malformed_rows": malformed_rows,
        "final_ceiling_hits": final_hits,
        "ever_ceiling_hits": ever_hits,
        "ever_ceiling_final_miss_count": ever_final_misses,
        "mean_time_to_ceiling": _mean_or_none(times),
        "std_time_to_ceiling": _std_or_none(times),
        "late_flip_count_after_first_ceiling_mean": _mean_or_none(late_flip_counts),
        "late_flip_rate_after_first_ceiling_mean": _mean_or_none(late_flip_rates),
        "terminal_window_ceiling_rate_mean": _mean_or_none(terminal_rates),
        "terminal_window_mean_ceiling_metric_mean": _mean_or_none(terminal_means),
        "metric_mean": _mean_or_none(metric_values),
        "metric_std": _std_or_none(metric_values),
    }


def variant_uses_teacher_bootstrap_replay(variant: MatrixVariant) -> bool:
    for key, value in _flatten_mapping(variant.updates):
        normalized_key = key.lower()
        if ".bootstrap." not in normalized_key:
            continue
        if isinstance(value, bool) and value:
            return True
    return False


def _append_optional_metric(
    *,
    row: Mapping[str, Any],
    field: str,
    values: list[float],
    malformed_fields: list[str],
) -> None:
    raw_value = row.get(field)
    value = _optional_float(raw_value)
    if raw_value in {None, ""}:
        return
    if value is None:
        malformed_fields.append(field)
        return
    values.append(value)


def _variant_status_and_reasons(
    *,
    stats: Mapping[str, Any],
    criterion: EvidenceGateCaseCriterion,
    eligible_main: bool,
    uses_teacher_bootstrap_replay: bool,
) -> tuple[str, list[str]]:
    if not eligible_main:
        if uses_teacher_bootstrap_replay:
            return "diagnostic_only", ["uses teacher/bootstrap/replay path"]
        return "diagnostic_only", ["not in main claim group"]
    if stats["missing_seeds"]:
        return "inconclusive", [
            "missing expected seeds: "
            + ", ".join(str(seed) for seed in stats["missing_seeds"])
        ]
    if stats["malformed_rows"]:
        return "inconclusive", ["malformed observed run rows"]
    if stats["final_ceiling_hits"] < criterion.final_ceiling_min_hits:
        return "fail", [
            "final ceiling hits {actual} < {required}".format(
                actual=stats["final_ceiling_hits"],
                required=criterion.final_ceiling_min_hits,
            )
        ]
    mean_time = stats["mean_time_to_ceiling"]
    if mean_time is None:
        return "inconclusive", ["missing time-to-ceiling values"]
    if not mean_time < criterion.mean_time_to_ceiling_lt:
        return "fail", [
            "mean time-to-ceiling {actual:.6g} >= {required:.6g}".format(
                actual=mean_time,
                required=criterion.mean_time_to_ceiling_lt,
            )
        ]
    return "pass", []


def _variant_trajectory_outcome(
    *,
    stats: Mapping[str, Any],
    criterion: EvidenceGateCaseCriterion,
    eligible_main: bool,
    status: str,
    uses_teacher_bootstrap_replay: bool,
) -> dict[str, str]:
    if not eligible_main:
        return {
            "trajectory_status": "diagnostic",
            "failure_mode": (
                "teacher_bootstrap_replay"
                if uses_teacher_bootstrap_replay
                else "not_main_group"
            ),
        }
    if status == "inconclusive":
        return {
            "trajectory_status": "inconclusive",
            "failure_mode": "missing_or_malformed_rows",
        }
    if status == "pass":
        return {"trajectory_status": "success", "failure_mode": ""}

    required = criterion.final_ceiling_min_hits
    final_hits = int(stats["final_ceiling_hits"])
    ever_hits = int(stats.get("ever_ceiling_hits", 0))
    mean_time = _optional_float(stats.get("mean_time_to_ceiling"))
    late_flip_rate = _optional_float(
        stats.get("late_flip_rate_after_first_ceiling_mean")
    )
    terminal_rate = _optional_float(stats.get("terminal_window_ceiling_rate_mean"))
    ever_final_misses = int(stats.get("ever_ceiling_final_miss_count", 0))

    if final_hits >= required and (
        mean_time is None or not mean_time < criterion.mean_time_to_ceiling_lt
    ):
        return {
            "trajectory_status": "trajectory_success_slow_ttc",
            "failure_mode": "slow_time_to_ceiling",
        }
    if ever_hits >= required and (
        ever_final_misses > 0
        or _mean_or_zero(late_flip_rate) > 0.0
        or _mean_or_zero(terminal_rate) >= 0.75
    ):
        return {
            "trajectory_status": "stochastic_gate_brittleness",
            "failure_mode": "final_epoch_hazard",
        }
    if ever_hits < required:
        return {
            "trajectory_status": "trajectory_ceiling_miss",
            "failure_mode": "mechanism_failure_candidate",
        }
    return {
        "trajectory_status": "final_ceiling_miss",
        "failure_mode": "unclassified_final_miss",
    }


def _case_status(
    eligible_main: Sequence[Mapping[str, Any]],
    passing_main: Sequence[Mapping[str, Any]],
) -> GateStatus:
    if passing_main:
        return "pass"
    if any(variant["status"] == "inconclusive" for variant in eligible_main):
        return "inconclusive"
    return "fail"


def _overall_status(case_summaries: Sequence[Mapping[str, Any]]) -> GateStatus:
    statuses = {str(case["status"]) for case in case_summaries}
    if statuses == {"pass"}:
        return "pass"
    if "inconclusive" in statuses:
        return "inconclusive"
    return "fail"


def _best_variant(variants: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not variants:
        return None
    return dict(
        max(
            variants,
            key=lambda variant: (
                int(variant["final_ceiling_hits"]),
                -_sort_time(variant.get("mean_time_to_ceiling")),
                _sort_metric(variant.get("metric_mean")),
            ),
        )
    )


def _baseline_improved(
    *,
    best_main: Mapping[str, Any] | None,
    best_baseline: Mapping[str, Any] | None,
) -> bool | None:
    if best_main is None or best_baseline is None:
        return None
    main_hits = int(best_main["final_ceiling_hits"])
    baseline_hits = int(best_baseline["final_ceiling_hits"])
    if main_hits != baseline_hits:
        return main_hits > baseline_hits
    main_time = _optional_float(best_main.get("mean_time_to_ceiling"))
    baseline_time = _optional_float(best_baseline.get("mean_time_to_ceiling"))
    if main_time is not None and baseline_time is not None and main_time != baseline_time:
        return main_time < baseline_time
    main_metric = _optional_float(best_main.get("metric_mean"))
    baseline_metric = _optional_float(best_baseline.get("metric_mean"))
    if main_metric is None or baseline_metric is None:
        return None
    return main_metric > baseline_metric


def _baseline_comparison(
    *,
    best_main: Mapping[str, Any] | None,
    best_baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if best_main is None or best_baseline is None:
        return {
            "final_ceiling_hit_delta": None,
            "time_to_ceiling_delta": None,
            "metric_mean_delta": None,
        }
    main_time = _optional_float(best_main.get("mean_time_to_ceiling"))
    baseline_time = _optional_float(best_baseline.get("mean_time_to_ceiling"))
    main_metric = _optional_float(best_main.get("metric_mean"))
    baseline_metric = _optional_float(best_baseline.get("metric_mean"))
    return {
        "final_ceiling_hit_delta": (
            int(best_main["final_ceiling_hits"])
            - int(best_baseline["final_ceiling_hits"])
        ),
        "time_to_ceiling_delta": (
            None
            if main_time is None or baseline_time is None
            else baseline_time - main_time
        ),
        "metric_mean_delta": (
            None
            if main_metric is None or baseline_metric is None
            else main_metric - baseline_metric
        ),
    }


def _case_criterion_from_mapping(
    case_name: str,
    raw: Mapping[str, Any],
) -> EvidenceGateCaseCriterion:
    required = {"final_ceiling_min_hits", "mean_time_to_ceiling_lt"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(
            f"Evidence gate case {case_name!r} missing field(s): "
            + ", ".join(missing)
        )
    return EvidenceGateCaseCriterion(
        final_ceiling_min_hits=int(raw["final_ceiling_min_hits"]),
        mean_time_to_ceiling_lt=float(raw["mean_time_to_ceiling_lt"]),
    )


def _manifest_row_mismatch(
    *,
    row: Mapping[str, Any],
    case: MatrixCase,
    variant: MatrixVariant,
) -> list[str]:
    mismatches: list[str] = []
    expected = {
        "toy": case.toy,
        "group": variant.group,
        "primary_metric": case.primary_metric,
        "direction": case.direction,
    }
    for field, value in expected.items():
        if str(row[field]) != str(value):
            mismatches.append(field)
    return mismatches


def _render_input_validation_error(validation: Mapping[str, Any]) -> str:
    parts = ["Evidence gate run rows do not match the manifest"]
    for key in ("missing_required_columns", "unknown_rows", "duplicate_rows"):
        rows = validation[key]
        if rows:
            parts.append(f"{key}={len(rows)}")
    return "; ".join(parts)


def _collect_malformed_rows(
    case_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    malformed: list[dict[str, Any]] = []
    for case in case_summaries:
        for variant in case["variants"]:
            for row in variant["malformed_rows"]:
                malformed.append(
                    {
                        "case": case["case"],
                        "variant": variant["variant"],
                        **row,
                    }
                )
    return malformed


def _case_diagnostic_hint(case: Mapping[str, Any]) -> str:
    case_name = str(case["case"])
    if case["status"] == "inconclusive":
        return f"{case_name}: re-run missing or malformed seeds before interpreting basin credit."
    best_main = case.get("best_main_variant") or {}
    reasons = "; ".join(best_main.get("reasons", []))
    if "final ceiling hits" in reasons:
        final_misses = _optional_float(best_main.get("ever_ceiling_final_miss_count"))
        if final_misses and final_misses > 0.0:
            return (
                f"{case_name}: separate final-epoch stochastic misses from "
                "true post-ceiling instability before adding new revision bias."
            )
        return f"{case_name}: inspect aggregate trajectories before adding a contrastive critic."
    if "time-to-ceiling" in reasons:
        return f"{case_name}: inspect time-to-ceiling trajectories and seed variance."
    return f"{case_name}: compare main variants against baseline trajectories first."


def _flatten_mapping(
    raw: Mapping[str, Any],
    *,
    prefix: str = "",
) -> list[tuple[str, Any]]:
    flattened: list[tuple[str, Any]] = []
    for key, value in raw.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.extend(_flatten_mapping(value, prefix=full_key))
        else:
            flattened.append((full_key, value))
    return flattened


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return math.fsum(values) / len(values)


def _std_or_none(values: Sequence[float]) -> float | None:
    if len(values) <= 1:
        return None
    mean = _mean_or_none(values)
    assert mean is not None
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _mean_or_zero(value: float | None) -> float:
    return 0.0 if value is None else value


def _sort_time(value: Any) -> float:
    number = _optional_float(value)
    if number is None:
        return math.inf
    return number


def _sort_metric(value: Any) -> float:
    number = _optional_float(value)
    if number is None:
        return -math.inf
    return number


def _optional_float(value: Any) -> float | None:
    if value == "" or value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    return int(number)


def _parse_seed(value: Any) -> int | None:
    number = _optional_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


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


def _format_optional_number(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return ""
    return f"{number:.6g}"


def _format_optional_bool(value: Any) -> str:
    if value is None:
        return ""
    return "true" if bool(value) else "false"
