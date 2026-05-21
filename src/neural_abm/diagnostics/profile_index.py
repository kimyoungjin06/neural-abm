"""Build a registry over generated evidence profiles."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from neural_abm.diagnostics.evidence_profile import (
    EvidenceProfileOutput,
    profile_evidence_artifacts,
)
from neural_abm.diagnostics.schema import CaseProfile, EvidenceProfile, format_number
from neural_abm.evidence_gate import load_gate_manifest


DEFAULT_RESULTS_DIR = Path("experiments/results/nabm_effect_matrix")
DEFAULT_GATE_OUTPUT_DIR = Path("experiments/evidence/results")
DEFAULT_MANIFEST_GLOB = "experiments/evidence/*.yaml"
DEFAULT_INDEX_LABEL = "evidence_profile_index"


@dataclass(frozen=True)
class EvidenceProfileIndexRow:
    label: str
    manifest_path: str
    runs_path: str
    profile_json_path: str
    profile_markdown_path: str
    profile_cases_path: str
    status: str
    passed: bool | None
    case: str
    toy: str
    case_status: str
    best_main_variant: str
    best_baseline_variant: str
    final_ceiling_hits: int | str
    expected_seed_count: int | str
    mean_time_to_ceiling: float | str
    metric_mean: float | str
    terminal_ceiling_rate_mean: float | str
    issue_codes: str
    notes: str
    profile_notes: str
    skipped_reason: str = ""


@dataclass(frozen=True)
class EvidenceProfileIndexOutput:
    rows: tuple[EvidenceProfileIndexRow, ...]
    json_path: Path | None
    markdown_path: Path | None
    csv_path: Path | None


def build_evidence_profile_index(
    *,
    manifest_paths: Sequence[Path] | None = None,
    manifest_globs: Sequence[str] | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    gate_output_dir: Path = DEFAULT_GATE_OUTPUT_DIR,
    output_dir: Path | None = None,
    profile_output_dir: Path | None = None,
    index_label: str = DEFAULT_INDEX_LABEL,
    seeds: Sequence[int] | None = None,
    write_profiles: bool = True,
    write_outputs: bool = True,
    skip_missing_runs: bool = True,
    skip_invalid_manifests: bool = True,
) -> EvidenceProfileIndexOutput:
    """Profile multiple evidence manifests and write a case-level registry."""

    resolved_manifests = resolve_manifest_paths(
        manifest_paths=manifest_paths,
        manifest_globs=manifest_globs,
    )
    rows: list[EvidenceProfileIndexRow] = []
    resolved_profile_output_dir = (
        profile_output_dir if profile_output_dir is not None else results_dir
    )
    for manifest_path in resolved_manifests:
        rows.extend(
            _profile_manifest_index_rows(
                manifest_path=manifest_path,
                results_dir=results_dir,
                gate_output_dir=gate_output_dir,
                profile_output_dir=resolved_profile_output_dir,
                seeds=seeds,
                write_profiles=write_profiles,
                skip_missing_runs=skip_missing_runs,
                skip_invalid_manifests=skip_invalid_manifests,
            )
        )

    output = EvidenceProfileIndexOutput(
        rows=tuple(rows),
        json_path=None,
        markdown_path=None,
        csv_path=None,
    )
    if not write_outputs:
        return output

    resolved_output_dir = output_dir if output_dir is not None else results_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output_dir / f"{index_label}.json"
    markdown_path = resolved_output_dir / f"{index_label}.md"
    csv_path = resolved_output_dir / f"{index_label}.csv"
    write_profile_index_csv(rows, csv_path)
    markdown_path.write_text(
        render_profile_index_markdown(rows, base_dir=resolved_output_dir),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return EvidenceProfileIndexOutput(
        rows=tuple(rows),
        json_path=json_path,
        markdown_path=markdown_path,
        csv_path=csv_path,
    )


def resolve_manifest_paths(
    *,
    manifest_paths: Sequence[Path] | None = None,
    manifest_globs: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    explicit_paths = list(manifest_paths or ())
    glob_patterns = list(manifest_globs or ())
    if not explicit_paths and not glob_patterns:
        glob_patterns = [DEFAULT_MANIFEST_GLOB]

    resolved: dict[str, Path] = {}
    for path in explicit_paths:
        resolved[str(path)] = path
    for pattern in glob_patterns:
        for path in sorted(Path(match) for match in glob.glob(pattern)):
            resolved[str(path)] = path
    return tuple(resolved[key] for key in sorted(resolved))


def render_profile_index_markdown(
    rows: Sequence[EvidenceProfileIndexRow],
    *,
    base_dir: Path | None = None,
) -> str:
    status_counts = Counter(row.status for row in rows)
    skipped_count = sum(1 for row in rows if row.skipped_reason)
    lines = [
        "# Evidence Profile Index",
        "",
        "## Overview",
        "",
        f"- Rows: {len(rows)}",
        f"- Status counts: {_format_counter(status_counts)}",
        f"- Skipped rows: {skipped_count}",
        "",
        "## Cases",
        "",
        "| Label | Status | Case | Toy | Best Main | Hits | Mean TtC | Issues | Notes | Profile |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        hits = (
            f"{row.final_ceiling_hits}/{row.expected_seed_count}"
            if row.final_ceiling_hits != ""
            else ""
        )
        profile_link = (
            f"[profile]({_markdown_link_target(row.profile_markdown_path, base_dir)})"
            if row.profile_markdown_path
            else ""
        )
        issue_text = row.issue_codes or row.skipped_reason
        lines.append(
            "| {label} | {status} | {case} | {toy} | {best_main} | {hits} | "
            "{ttc} | {issues} | {notes} | {profile} |".format(
                label=_escape_markdown_cell(row.label),
                status=_escape_markdown_cell(row.status),
                case=_escape_markdown_cell(row.case),
                toy=_escape_markdown_cell(row.toy),
                best_main=_escape_markdown_cell(row.best_main_variant),
                hits=_escape_markdown_cell(hits),
                ttc=_escape_markdown_cell(format_number(row.mean_time_to_ceiling)),
                issues=_escape_markdown_cell(issue_text),
                notes=_escape_markdown_cell(row.notes or row.profile_notes),
                profile=profile_link,
            )
        )
    return "\n".join(lines) + "\n"


def write_profile_index_csv(
    rows: Sequence[EvidenceProfileIndexRow],
    path: Path,
) -> None:
    fieldnames = list(EvidenceProfileIndexRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    args = _parse_args()
    output = build_evidence_profile_index(
        manifest_paths=args.manifest,
        manifest_globs=args.manifest_glob,
        results_dir=args.results_dir,
        gate_output_dir=args.gate_output_dir,
        output_dir=args.output_dir,
        profile_output_dir=args.profile_output_dir,
        index_label=args.index_label,
        seeds=args.seeds,
        write_profiles=not args.no_write_profiles,
        skip_missing_runs=not args.fail_on_missing_runs,
        skip_invalid_manifests=not args.fail_on_invalid_manifest,
    )
    if output.json_path is not None:
        print(f"Wrote profile index JSON: {output.json_path}")
    if output.markdown_path is not None:
        print(f"Wrote profile index Markdown: {output.markdown_path}")
    if output.csv_path is not None:
        print(f"Wrote profile index CSV: {output.csv_path}")
    print(f"Evidence profile index rows: {len(output.rows)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=[],
        help="Evidence manifest to profile. May be passed multiple times.",
    )
    parser.add_argument(
        "--manifest-glob",
        action="append",
        default=[],
        help=(
            "Glob of evidence manifests to profile. Defaults to "
            f"{DEFAULT_MANIFEST_GLOB!r} when no manifest or glob is provided."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--gate-output-dir",
        type=Path,
        default=DEFAULT_GATE_OUTPUT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--profile-output-dir", type=Path, default=None)
    parser.add_argument("--index-label", default=DEFAULT_INDEX_LABEL)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional seed override matching every evaluated run-level CSV.",
    )
    parser.add_argument(
        "--no-write-profiles",
        action="store_true",
        help="Build only the registry without rewriting per-manifest profiles.",
    )
    parser.add_argument(
        "--fail-on-missing-runs",
        action="store_true",
        help="Raise when a manifest does not have a matching run CSV.",
    )
    parser.add_argument(
        "--fail-on-invalid-manifest",
        action="store_true",
        help="Raise when a manifest cannot be interpreted as gated evidence.",
    )
    return parser.parse_args()


def _profile_manifest_index_rows(
    *,
    manifest_path: Path,
    results_dir: Path,
    gate_output_dir: Path,
    profile_output_dir: Path,
    seeds: Sequence[int] | None,
    write_profiles: bool,
    skip_missing_runs: bool,
    skip_invalid_manifests: bool,
) -> list[EvidenceProfileIndexRow]:
    try:
        manifest, _criteria = load_gate_manifest(manifest_path)
    except Exception:
        if not skip_invalid_manifests:
            raise
        return [
            _skipped_row(
                label=manifest_path.stem,
                manifest_path=manifest_path,
                skipped_reason="invalid_manifest",
            )
        ]

    runs_path = results_dir / f"{manifest.label}_runs.csv"
    if not runs_path.exists():
        if not skip_missing_runs:
            raise FileNotFoundError(runs_path)
        return [
            _skipped_row(
                label=manifest.label,
                manifest_path=manifest_path,
                runs_path=runs_path,
                skipped_reason="missing_runs",
            )
        ]

    profile_output = profile_evidence_artifacts(
        manifest_path,
        runs_path=runs_path,
        results_dir=results_dir,
        gate_output_dir=gate_output_dir,
        output_dir=profile_output_dir,
        seeds=seeds,
        write_outputs=write_profiles,
    )
    return _rows_from_profile_output(profile_output)


def _rows_from_profile_output(
    output: EvidenceProfileOutput,
) -> list[EvidenceProfileIndexRow]:
    profile = output.profile
    return [
        _row_from_case(
            profile=profile,
            case=case,
            json_path=output.json_path,
            markdown_path=output.markdown_path,
            cases_path=output.cases_path,
        )
        for case in profile.cases
    ]


def _row_from_case(
    *,
    profile: EvidenceProfile,
    case: CaseProfile,
    json_path: Path | None,
    markdown_path: Path | None,
    cases_path: Path | None,
) -> EvidenceProfileIndexRow:
    best_main = _find_case_variant(case, case.best_main_variant)
    return EvidenceProfileIndexRow(
        label=profile.label,
        manifest_path=profile.manifest_path,
        runs_path=profile.runs_path,
        profile_json_path=str(json_path or ""),
        profile_markdown_path=str(markdown_path or ""),
        profile_cases_path=str(cases_path or ""),
        status=profile.status,
        passed=profile.passed,
        case=case.case,
        toy=case.toy,
        case_status=case.status,
        best_main_variant=case.best_main_variant or "",
        best_baseline_variant=case.best_baseline_variant or "",
        final_ceiling_hits=best_main.final_ceiling_hits if best_main else "",
        expected_seed_count=best_main.expected_seed_count if best_main else "",
        mean_time_to_ceiling=best_main.time_to_ceiling.mean if best_main else "",
        metric_mean=best_main.metric.mean if best_main else "",
        terminal_ceiling_rate_mean=(
            best_main.terminal_ceiling_rate.mean if best_main else ""
        ),
        issue_codes=",".join(case.issue_codes),
        notes=",".join(case.notes),
        profile_notes=",".join(profile.notes),
    )


def _skipped_row(
    *,
    label: str,
    manifest_path: Path,
    skipped_reason: str,
    runs_path: Path | None = None,
) -> EvidenceProfileIndexRow:
    return EvidenceProfileIndexRow(
        label=label,
        manifest_path=str(manifest_path),
        runs_path=str(runs_path or ""),
        profile_json_path="",
        profile_markdown_path="",
        profile_cases_path="",
        status="skipped",
        passed=None,
        case="",
        toy="",
        case_status="skipped",
        best_main_variant="",
        best_baseline_variant="",
        final_ceiling_hits="",
        expected_seed_count="",
        mean_time_to_ceiling="",
        metric_mean="",
        terminal_ceiling_rate_mean="",
        issue_codes="",
        notes="",
        profile_notes="",
        skipped_reason=skipped_reason,
    )


def _find_case_variant(case: CaseProfile, name: str | None) -> Any:
    if not name:
        return None
    for variant in case.variants:
        if variant.variant == name:
            return variant
    return None


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter))


def _escape_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _markdown_link_target(path: str, base_dir: Path | None) -> str:
    if base_dir is None:
        return path
    return os.path.relpath(Path(path), base_dir)


if __name__ == "__main__":
    main()
