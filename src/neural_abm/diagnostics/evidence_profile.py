"""Read-only evidence profiler for manifests and generated run rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from neural_abm.diagnostics.adapters import adapter_for_case
from neural_abm.diagnostics.schema import (
    CaseProfile,
    EvidenceProfile,
    VariantProfile,
    format_number,
    numeric_summary,
    optional_bool,
    optional_int,
)
from neural_abm.evidence_gate import (
    evaluate_evidence_gate,
    load_gate_manifest,
    read_csv_rows,
)
from neural_abm.evidence_matrix import MatrixCase, MatrixVariant


@dataclass(frozen=True)
class EvidenceProfileOutput:
    profile: EvidenceProfile
    json_path: Path | None
    markdown_path: Path | None
    cases_path: Path | None


def profile_evidence_artifacts(
    manifest_path: Path,
    *,
    runs_path: Path | None = None,
    results_dir: Path = Path("experiments/results/nabm_effect_matrix"),
    gate_output_dir: Path = Path("experiments/evidence/results"),
    output_dir: Path | None = None,
    seeds: Sequence[int] | None = None,
    write_outputs: bool = True,
) -> EvidenceProfileOutput:
    """Build and optionally write a read-only diagnostic profile."""

    manifest, criteria = load_gate_manifest(manifest_path)
    if seeds is not None:
        manifest = replace(manifest, seeds=tuple(int(seed) for seed in seeds))
    resolved_runs_path = (
        runs_path
        if runs_path is not None
        else results_dir / f"{manifest.label}_runs.csv"
    )
    gate_summary_path = gate_output_dir / f"{manifest.label}.summary.json"
    run_rows = read_csv_rows(resolved_runs_path)
    gate_summary = evaluate_evidence_gate(
        manifest=manifest,
        criteria=criteria,
        run_rows=run_rows,
        runs_path=resolved_runs_path,
    )
    rows_by_case_variant = _rows_by_case_variant(run_rows)
    gate_cases = {
        str(case_summary.get("case")): case_summary
        for case_summary in gate_summary.get("cases", [])
        if isinstance(case_summary, Mapping)
    }
    case_profiles = [
        _build_case_profile(
            case=case,
            default_seeds=manifest.seeds,
            main_group=criteria.main_group,
            rows_by_case_variant=rows_by_case_variant,
            gate_case=gate_cases.get(case.name),
        )
        for case in manifest.cases
    ]
    profile = EvidenceProfile(
        label=manifest.label,
        status=str(gate_summary.get("status", "unknown")),
        passed=bool(gate_summary.get("passed"))
        if gate_summary.get("passed") is not None
        else None,
        manifest_path=str(manifest_path),
        runs_path=str(resolved_runs_path),
        gate_summary_path=str(gate_summary_path) if gate_summary_path.exists() else "",
        cases=case_profiles,
        input_validation=dict(gate_summary.get("input_validation", {})),
        notes=_profile_notes(case_profiles),
    )
    if not write_outputs:
        return EvidenceProfileOutput(
            profile=profile,
            json_path=None,
            markdown_path=None,
            cases_path=None,
        )
    resolved_output_dir = output_dir if output_dir is not None else results_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output_dir / f"{manifest.label}_profile.json"
    markdown_path = resolved_output_dir / f"{manifest.label}_profile.md"
    cases_path = resolved_output_dir / f"{manifest.label}_profile_cases.csv"
    json_path.write_text(
        json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_profile_markdown(profile), encoding="utf-8")
    write_profile_cases_csv(profile, cases_path)
    return EvidenceProfileOutput(
        profile=profile,
        json_path=json_path,
        markdown_path=markdown_path,
        cases_path=cases_path,
    )


def render_profile_markdown(profile: EvidenceProfile) -> str:
    lines = [
        f"# Evidence Profile: {profile.label}",
        "",
        "## Inputs",
        "",
        f"- Manifest: `{profile.manifest_path}`",
        f"- Runs: `{profile.runs_path}`",
        f"- Gate summary: `{profile.gate_summary_path or 'not provided'}`",
        "",
        "## Overview",
        "",
        f"- Gate status: `{profile.status}`",
        f"- Passed: `{profile.passed}`",
    ]
    if profile.notes:
        lines.append(f"- Notes: {', '.join(profile.notes)}")
    lines.extend(
        [
            "",
            "## Case Summary",
            "",
            "| Case | Toy | Status | Best Main | Final Hits | Mean TtC | Metric Mean | Issues | Notes |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for case in profile.cases:
        best_main = _find_variant(case, case.best_main_variant)
        lines.append(
            "| {case} | {toy} | {status} | {variant} | {hits} | {ttc} | "
            "{metric} | {issues} | {notes} |".format(
                case=case.case,
                toy=case.toy,
                status=case.status,
                variant=case.best_main_variant or "",
                hits=(
                    f"{best_main.final_ceiling_hits}/"
                    f"{best_main.expected_seed_count}"
                    if best_main is not None
                    else ""
                ),
                ttc=format_number(
                    best_main.time_to_ceiling.mean if best_main else None
                ),
                metric=format_number(best_main.metric.mean if best_main else None),
                issues=", ".join(case.issue_codes),
                notes=", ".join(case.notes),
            )
        )
    lines.extend(
        [
            "",
            "## Variant Details",
            "",
            "| Case | Variant | Role | Status | Final Hits | Mean TtC | Metric Mean | Terminal Rate | Issues |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for case in profile.cases:
        for variant in case.variants:
            lines.append(
                "| {case} | {variant} | {role} | {status} | {hits}/{expected} | "
                "{ttc} | {metric} | {terminal} | {issues} |".format(
                    case=case.case,
                    variant=variant.variant,
                    role=variant.role,
                    status=variant.status,
                    hits=variant.final_ceiling_hits,
                    expected=variant.expected_seed_count,
                    ttc=format_number(variant.time_to_ceiling.mean),
                    metric=format_number(variant.metric.mean),
                    terminal=format_number(variant.terminal_ceiling_rate.mean),
                    issues=", ".join(variant.issue_codes),
                )
            )
    return "\n".join(lines) + "\n"


def write_profile_cases_csv(profile: EvidenceProfile, path: Path) -> None:
    fields = [
        "label",
        "case",
        "toy",
        "status",
        "best_main_variant",
        "best_baseline_variant",
        "final_ceiling_hits",
        "expected_seed_count",
        "mean_time_to_ceiling",
        "metric_mean",
        "terminal_ceiling_rate_mean",
        "issue_codes",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in profile.cases:
            best_main = _find_variant(case, case.best_main_variant)
            writer.writerow(
                {
                    "label": profile.label,
                    "case": case.case,
                    "toy": case.toy,
                    "status": case.status,
                    "best_main_variant": case.best_main_variant or "",
                    "best_baseline_variant": case.best_baseline_variant or "",
                    "final_ceiling_hits": (
                        best_main.final_ceiling_hits if best_main else ""
                    ),
                    "expected_seed_count": (
                        best_main.expected_seed_count if best_main else ""
                    ),
                    "mean_time_to_ceiling": (
                        best_main.time_to_ceiling.mean if best_main else ""
                    ),
                    "metric_mean": best_main.metric.mean if best_main else "",
                    "terminal_ceiling_rate_mean": (
                        best_main.terminal_ceiling_rate.mean if best_main else ""
                    ),
                    "issue_codes": ",".join(case.issue_codes),
                    "notes": ",".join(case.notes),
                }
            )


def main() -> None:
    args = _parse_args()
    output = profile_evidence_artifacts(
        args.manifest,
        runs_path=args.runs,
        results_dir=args.results_dir,
        gate_output_dir=args.gate_output_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        write_outputs=not args.no_write,
    )
    if output.json_path is not None:
        print(f"Wrote profile JSON: {output.json_path}")
    if output.markdown_path is not None:
        print(f"Wrote profile Markdown: {output.markdown_path}")
    if output.cases_path is not None:
        print(f"Wrote profile case CSV: {output.cases_path}")
    print(f"Evidence profile status: {output.profile.status}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=None)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results/nabm_effect_matrix"),
    )
    parser.add_argument(
        "--gate-output-dir",
        type=Path,
        default=Path("experiments/evidence/results"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional seed override matching the evaluated run-level CSV.",
    )
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def _build_case_profile(
    *,
    case: MatrixCase,
    default_seeds: Sequence[int],
    main_group: str,
    rows_by_case_variant: Mapping[tuple[str, str], list[Mapping[str, Any]]],
    gate_case: Mapping[str, Any] | None,
) -> CaseProfile:
    gate_variants = _gate_variants_by_name(gate_case)
    adapter = adapter_for_case(case)
    variants = [
        _build_variant_profile(
            case=case,
            variant=variant,
            main_group=main_group,
            expected_seeds=tuple(int(seed) for seed in (case.seeds or default_seeds)),
            rows=rows_by_case_variant.get((case.name, variant.name), []),
            gate_variant=gate_variants.get(variant.name),
            adapter=adapter,
        )
        for variant in case.variants
    ]
    profile = CaseProfile(
        case=case.name,
        toy=case.toy,
        status=str(gate_case.get("status", "unknown")) if gate_case else "unknown",
        primary_metric=case.primary_metric,
        baseline_group=case.baseline_group,
        nabm_group=case.nabm_group,
        main_group=main_group,
        best_main_variant=_gate_variant_name(gate_case, "best_main_variant")
        or _best_variant_name([variant for variant in variants if variant.role == "main"]),
        best_baseline_variant=_gate_variant_name(gate_case, "best_baseline_variant")
        or _best_variant_name(
            [variant for variant in variants if variant.role == "baseline"]
        ),
        variants=variants,
    )
    if profile.status == "fail":
        profile.issue_codes.append("gate_case_fail")
    if any(variant.missing_seeds for variant in variants):
        profile.issue_codes.append("missing_seed_rows")
    adapter.annotate_case(case=case, profile=profile)
    return profile


def _build_variant_profile(
    *,
    case: MatrixCase,
    variant: MatrixVariant,
    main_group: str,
    expected_seeds: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    gate_variant: Mapping[str, Any] | None,
    adapter: Any,
) -> VariantProfile:
    resolved_expected_seeds = tuple(int(seed) for seed in expected_seeds)
    observed_seeds = sorted(
        {
            seed
            for row in rows
            if (seed := optional_int(row.get("seed"))) is not None
        }
    )
    expected_seed_set = set(resolved_expected_seeds)
    missing_seeds = [
        seed for seed in resolved_expected_seeds if seed not in observed_seeds
    ]
    failed_seeds = [
        seed
        for row in rows
        if (seed := optional_int(row.get("seed"))) is not None
        and seed in expected_seed_set
        and optional_bool(row.get("final_within_ceiling")) is False
    ]
    final_hits = sum(
        1 for row in rows if optional_bool(row.get("final_within_ceiling")) is True
    )
    ever_hits = sum(
        1 for row in rows if optional_bool(row.get("ever_reached_ceiling")) is True
    )
    profile = VariantProfile(
        variant=variant.name,
        group=variant.group,
        role=_variant_role(variant=variant, case=case, main_group=main_group),
        status=str(gate_variant.get("status", "unknown"))
        if gate_variant
        else "unknown",
        expected_seed_count=len(resolved_expected_seeds),
        observed_seed_count=len(observed_seeds),
        final_ceiling_hits=final_hits,
        ever_ceiling_hits=ever_hits,
        missing_seeds=missing_seeds,
        failed_seeds=failed_seeds,
        metric=numeric_summary(row.get("metric_value") for row in rows),
        time_to_ceiling=numeric_summary(row.get("time_to_ceiling") for row in rows),
        terminal_ceiling_rate=numeric_summary(
            row.get("terminal_window_ceiling_rate") for row in rows
        ),
        late_flip_rate=numeric_summary(
            row.get("late_flip_rate_after_first_ceiling") for row in rows
        ),
        details=adapter.variant_details(case=case, variant=variant, rows=rows),
    )
    if missing_seeds:
        profile.issue_codes.append("missing_seed_rows")
    if profile.role == "main" and final_hits < len(resolved_expected_seeds):
        profile.issue_codes.append("main_final_ceiling_miss")
    if failed_seeds:
        profile.issue_codes.append("seed_level_final_miss")
    if _mean_or_zero(profile.late_flip_rate.mean) > 0.0:
        profile.notes.append("late_flip_observed")
    return profile


def _rows_by_case_variant(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("case", "")), str(row.get("variant", "")))
        grouped.setdefault(key, []).append(row)
    return grouped


def _gate_variants_by_name(
    gate_case: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if not gate_case:
        return {}
    variants = gate_case.get("variants", [])
    if not isinstance(variants, Sequence):
        return {}
    return {
        str(variant.get("variant")): variant
        for variant in variants
        if isinstance(variant, Mapping)
    }


def _gate_variant_name(
    gate_case: Mapping[str, Any] | None,
    field: str,
) -> str | None:
    if not gate_case:
        return None
    variant = gate_case.get(field)
    if not isinstance(variant, Mapping):
        return None
    name = variant.get("variant")
    return str(name) if name else None


def _best_variant_name(variants: Sequence[VariantProfile]) -> str | None:
    if not variants:
        return None

    def key(variant: VariantProfile) -> tuple[int, float, float]:
        time = variant.time_to_ceiling.mean
        metric = variant.metric.mean
        return (
            variant.final_ceiling_hits,
            -(time if time is not None else float("inf")),
            metric if metric is not None else float("-inf"),
        )

    return max(variants, key=key).variant


def _find_variant(case: CaseProfile, name: str | None) -> VariantProfile | None:
    if name is None:
        return None
    for variant in case.variants:
        if variant.variant == name:
            return variant
    return None


def _variant_role(
    *,
    variant: MatrixVariant,
    case: MatrixCase,
    main_group: str,
) -> str:
    if variant.group == case.baseline_group:
        return "baseline"
    if variant.group == main_group:
        return "main"
    return "diagnostic"


def _profile_notes(cases: Sequence[CaseProfile]) -> list[str]:
    notes: list[str] = []
    if any(case.issue_codes for case in cases):
        notes.append("profile_has_case_issues")
    if any("toy5_threshold_aware_direction" in case.notes for case in cases):
        notes.append("toy5_threshold_aware_evidence")
    if any("toy5_mean_readiness_frontier_stall" in case.notes for case in cases):
        notes.append("toy5_mean_vs_max_frontier_contrast")
    if any("toy24_objective_basin_blend" in case.notes for case in cases):
        notes.append("toy24_objective_basin_evidence")
    if any(
        "toy24_material_basin_collapse_diagnostic" in case.notes for case in cases
    ):
        notes.append("toy24_material_basin_collapse_contrast")
    if any("toy24_revision_operator_path" in case.notes for case in cases):
        notes.append("toy24_revision_operator_evidence")
    if any("toy24_final_vs_ever_gap" in case.notes for case in cases):
        notes.append("toy24_final_epoch_hazard_evidence")
    if any("toy24_triage_success" in case.notes for case in cases):
        notes.append("toy24_triage_success_evidence")
    if any(
        "toy24_triage_stochastic_gate_brittleness" in case.notes for case in cases
    ):
        notes.append("toy24_stochastic_gate_brittleness_evidence")
    if any(
        "toy24_triage_baseline_favored_environment" in case.notes for case in cases
    ):
        notes.append("toy24_baseline_favored_environment_evidence")
    if any(
        "toy24_triage_true_mechanism_failure_candidate" in case.notes
        for case in cases
    ):
        notes.append("toy24_true_mechanism_failure_candidate_evidence")
    if any("toy24_trajectory_success_slow_ttc" in case.notes for case in cases):
        notes.append("toy24_slow_ttc_gate_lag_evidence")
    return notes


def _mean_or_zero(value: float | None) -> float:
    return 0.0 if value is None else value


if __name__ == "__main__":
    main()
