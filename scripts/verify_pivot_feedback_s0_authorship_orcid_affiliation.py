#!/usr/bin/env python3
"""Independently verify an S0 ORCID and affiliation projection output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "pivot_feedback_s0_authorship_orcid_affiliation_report.v1"
FIELD_YEAR_SCHEMA = (
    "pivot_feedback_s0_authorship_orcid_affiliation_field_year.v1"
)
OUTPUT_SCHEMA = "pivot_feedback_s0_authorship_orcid_affiliation_outputs.v1"
FULL_STATUS = (
    "PASS_S0_FULL_AUTHORSHIP_ORCID_AND_AFFILIATION_PROJECTION_"
    "WITH_TOKEN_AND_CURRENT_SNAPSHOT_LIMITATIONS"
)
PILOT_STATUS = "PILOT_S0_AUTHORSHIP_ORCID_AND_AFFILIATION_NONPROMOTABLE"
FULL_VERIFY_STATUS = (
    "PASS_INDEPENDENT_S0_AUTHORSHIP_ORCID_AFFILIATION_CONSERVATION_"
    "COAUTHOR_GRAPH_STILL_NOT_RUN"
)
PILOT_VERIFY_STATUS = (
    "PASS_INDEPENDENT_NONPROMOTABLE_S0_AUTHORSHIP_ORCID_AFFILIATION_PILOT"
)
EXPECTED_FILES = {
    "authorship_orcid_affiliation_report.json",
    "authorship_orcid_affiliation_report.md",
    "authorship_orcid_affiliation_report.outputs.json",
    "field_year_orcid_affiliation.parquet",
}
FIELD_YEAR_METRICS = (
    "raw_authorship_rows",
    "grammar_valid_author_id_rows",
    "nonnull_orcid_rows",
    "syntax_valid_orcid_rows",
    "checksum_valid_orcid_rows",
    "author_work_affiliation_rows",
    "author_work_institution_rows",
    "stable_orcid_author_work_pairs",
    "stable_orcid_author_tokens",
    "distinct_institutions",
)
FORBIDDEN_COLUMNS = {
    "eligible_author_count",
    "matured_entry_opportunity_count",
    "fixed_horizon_outcome_coverage",
    "independent_attention_shock_count",
    "entry_effect_size",
    "persistence_effect_size",
    "overshoot_timing_gap",
    "predictive_score",
    "selected_best_field",
    "selected_best_threshold",
    "g4_holdout_outcome",
    "cited_by_count",
}


class ProjectionVerificationError(RuntimeError):
    """Fail-closed projection verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProjectionVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_report(path: Path, result: Mapping[str, Any], *, root: Path) -> None:
    root = root.resolve(strict=True)
    parent = path.parent.resolve(strict=True)
    target = parent / path.name
    require(
        target != root and not target.is_relative_to(root),
        "verification report must be outside output root",
    )
    payload = (
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(target, 0o600, follow_symlinks=False)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProjectionVerificationError(f"invalid {label}: {error}") from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ProjectionVerificationError(f"invalid {label}: {error}") from error


def _schema_version(schema: pa.Schema) -> str | None:
    raw = (schema.metadata or {}).get(b"schema_version")
    return None if raw is None else raw.decode("utf-8")


def _validate_receipt(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = _read_json(
        root / "authorship_orcid_affiliation_report.outputs.json",
        "output receipt",
    )
    require(receipt.get("schema_version") == OUTPUT_SCHEMA, "receipt schema mismatch")
    status = receipt.get("status")
    require(status in {FULL_STATUS, PILOT_STATUS}, "receipt status mismatch")
    full_scope = status == FULL_STATUS
    require(receipt.get("full_scope") is full_scope, "receipt scope mismatch")
    require(receipt.get("component_pass") is full_scope, "receipt pass mismatch")
    require(receipt.get("s0_gate_pass") is False, "receipt improperly passes S0")
    require(receipt.get("d0_authorized") is False, "receipt improperly authorizes D0")
    require(
        receipt.get("completion_contract")
        == "SCRATCH_REMOVED_RECEIPT_WRITTEN_LAST_ATOMIC_DIRECTORY_RENAME",
        "completion contract mismatch",
    )
    declared = receipt.get("artifacts")
    require(isinstance(declared, list), "receipt artifacts must be a list")
    require(receipt.get("artifact_count") == len(declared) == 3, "artifact count mismatch")
    expected_artifacts = EXPECTED_FILES - {
        "authorship_orcid_affiliation_report.outputs.json"
    }
    observed: list[dict[str, Any]] = []
    for raw in declared:
        item = _mapping(raw, "receipt artifact")
        relative = item.get("relative_path")
        require(
            isinstance(relative, str)
            and "/" not in relative
            and relative in expected_artifacts,
            "unexpected artifact path",
        )
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"invalid artifact: {relative}")
        actual = {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        require(item == actual, f"artifact identity mismatch: {relative}")
        observed.append(actual)
    require(
        {item["relative_path"] for item in observed} == expected_artifacts,
        "artifact set mismatch",
    )
    require(
        receipt.get("artifact_tree_sha256")
        == hashlib.sha256(_canonical_bytes(observed)).hexdigest(),
        "artifact tree mismatch",
    )
    return receipt, observed


def _validate_provenance(
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = _mapping(report.get("provenance"), "provenance")
    require(
        provenance.get("config_relative_path")
        == "experiments/configs/pivot_feedback_s0_authorship_orcid_affiliation_v1.yaml",
        "unexpected config path",
    )
    require(
        provenance.get("runner_relative_path")
        == "scripts/build_pivot_feedback_s0_authorship_orcid_affiliation.py",
        "unexpected runner path",
    )
    config_path = REPO_ROOT / provenance["config_relative_path"]
    runner_path = REPO_ROOT / provenance["runner_relative_path"]
    require(_sha256_file(config_path) == provenance.get("config_sha256"), "config SHA mismatch")
    require(_sha256_file(runner_path) == provenance.get("runner_sha256"), "runner SHA mismatch")
    config = _read_yaml(config_path, "projection config")
    replay = json.loads(json.dumps(report))
    expected = replay["provenance"].pop("result_fingerprint_sha256")
    observed = hashlib.sha256(_canonical_bytes(replay)).hexdigest()
    require(
        observed == expected == receipt.get("result_fingerprint_sha256"),
        "result fingerprint mismatch",
    )
    return config


def _validate_source_accounting(
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    full_scope: bool,
) -> dict[str, int]:
    counts = _mapping(report.get("source_accounting"), "source accounting")
    raw = int(counts["raw_rows"])
    retained = int(counts["retained_rows"])
    w_minus_1 = int(counts["w_minus_1_rows"])
    nonnull_author = int(counts["nonnull_author_id_rows"])
    null_author = int(counts["null_author_id_rows"])
    grammar_author = int(counts["grammar_valid_author_id_rows"])
    nonnull_orcid = int(counts["nonnull_orcid_rows"])
    syntax_orcid = int(counts["syntax_valid_orcid_rows"])
    checksum_orcid = int(counts["checksum_valid_orcid_rows"])
    require(retained + w_minus_1 == raw, "work row partition does not conserve")
    require(nonnull_author + null_author == raw, "author null margin does not conserve")
    require(grammar_author <= nonnull_author <= raw, "author ID margin is not monotone")
    require(
        checksum_orcid <= syntax_orcid <= nonnull_orcid <= raw,
        "ORCID global margin is not monotone",
    )
    require(
        int(counts["author_work_institution_rows"])
        <= int(counts["author_work_affiliation_rows"])
        <= grammar_author,
        "affiliation global margin is not monotone",
    )
    require(int(counts["raw_affiliation_present_rows"]) <= raw, "raw affiliation margin exceeds rows")
    require(int(counts["institution_present_rows"]) <= raw, "institution margin exceeds rows")
    require(
        int(counts["author_orcid_spool_rows"]) <= checksum_orcid,
        "author ORCID spool exceeds checksum-valid rows",
    )
    require(
        int(counts["institution_spool_rows"])
        <= int(counts["grammar_valid_institution_occurrences"]),
        "institution spool exceeds valid occurrences",
    )
    if full_scope:
        source = _mapping(config.get("source"), "source config")
        for observed_key, expected_key in (
            ("raw_rows", "expected_raw_rows"),
            ("w_minus_1_rows", "expected_w_minus_1_rows"),
            ("retained_rows", "expected_retained_rows"),
            ("nonnull_author_id_rows", "expected_nonnull_author_id_rows"),
            ("null_author_id_rows", "expected_null_author_id_rows"),
            ("nonnull_orcid_rows", "expected_nonnull_orcid_rows"),
        ):
            require(
                counts[observed_key] == source[expected_key],
                f"full source closure mismatch: {observed_key}",
            )
        require(counts["unmatched_valid_work_rows"] == 0, "unmatched valid works present")
    return {key: int(value) for key, value in counts.items()}


def _validate_token_diagnostics(report: Mapping[str, Any]) -> dict[str, int]:
    values = _mapping(report.get("orcid_token_diagnostics"), "ORCID diagnostics")
    pairs = int(values["distinct_author_orcid_pairs"])
    tokens = int(values["author_tokens_with_valid_orcid"])
    stable = int(values["stable_single_valid_orcid_author_tokens"])
    multiple = int(values["multiple_valid_orcid_author_tokens"])
    distinct_orcids = int(values["distinct_valid_orcids_on_stable_tokens"])
    shared = int(values["shared_valid_orcids_across_stable_tokens"])
    shared_tokens = int(values["stable_author_tokens_on_shared_valid_orcids"])
    maximum = int(values["maximum_stable_author_tokens_per_valid_orcid"])
    require(stable + multiple == tokens <= pairs, "author-ORCID multiplicity does not conserve")
    require(distinct_orcids <= stable, "distinct ORCIDs exceed stable tokens")
    require(shared <= distinct_orcids, "shared ORCIDs exceed distinct ORCIDs")
    require(2 * shared <= shared_tokens <= stable, "shared ORCID token margin invalid")
    require(
        (distinct_orcids == 0 and maximum == 0)
        or (distinct_orcids > 0 and 1 <= maximum <= stable),
        "maximum token multiplicity invalid",
    )
    return {key: int(value) for key, value in values.items()}


def _validate_field_year(
    table: pa.Table,
    report: Mapping[str, Any],
    source_counts: Mapping[str, int],
) -> dict[str, Any]:
    inventory = _mapping(report.get("field_year_inventory"), "field-year inventory")
    require(_schema_version(table.schema) == FIELD_YEAR_SCHEMA, "field-year schema version mismatch")
    expected_schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("publication_year", pa.uint16(), nullable=True),
            *[
                pa.field(metric, pa.uint64(), nullable=False)
                for metric in FIELD_YEAR_METRICS
            ],
        ]
    )
    require(
        table.schema.equals(expected_schema, check_metadata=False),
        "field-year physical schema mismatch",
    )
    require(len(table) == inventory.get("rows"), "field-year row count mismatch")
    require(FORBIDDEN_COLUMNS.isdisjoint(table.column_names), "forbidden field-year column present")
    fields = table["field_id"].to_pylist()
    years = table["publication_year"].to_pylist()
    keys = list(
        zip(fields, (-1 if year is None else int(year) for year in years), strict=True)
    )
    require(keys == sorted(keys), "field-year rows are not sorted")
    require(len(keys) == len(set(keys)), "duplicate field-year row")
    require(all(11 <= int(value) <= 36 for value in fields), "field domain mismatch")
    require(set(fields) == set(range(11, 37)), "field coverage mismatch")
    require(
        all(year is None or 1000 <= int(year) <= 2050 for year in years),
        "publication year domain mismatch",
    )
    columns = {metric: table[metric].to_pylist() for metric in FIELD_YEAR_METRICS}
    for index in range(len(table)):
        require(
            columns["checksum_valid_orcid_rows"][index]
            <= columns["syntax_valid_orcid_rows"][index]
            <= columns["nonnull_orcid_rows"][index]
            <= columns["raw_authorship_rows"][index],
            "ORCID cell monotonicity failed",
        )
        require(
            columns["author_work_institution_rows"][index]
            <= columns["author_work_affiliation_rows"][index]
            <= columns["grammar_valid_author_id_rows"][index]
            <= columns["raw_authorship_rows"][index],
            "affiliation cell monotonicity failed",
        )
    sums = {metric: int(sum(values)) for metric, values in columns.items()}
    require(sums == inventory.get("metric_sums"), "field-year metric sums mismatch")
    for metric in FIELD_YEAR_METRICS[:7]:
        require(
            sums[metric] == source_counts[f"classified_field_year_{metric}"],
            f"source-to-field-year margin mismatch: {metric}",
        )
    events = _mapping(report.get("stable_event_diagnostics"), "event diagnostics")
    institutions = _mapping(
        report.get("institution_diagnostics"), "institution diagnostics"
    )
    require(
        sums["stable_orcid_author_work_pairs"]
        == events.get("stable_orcid_author_work_pairs"),
        "stable ORCID event margin mismatch",
    )
    require(
        sums["stable_orcid_author_tokens"]
        == events.get("stable_orcid_author_tokens_summed_within_field_year"),
        "stable ORCID token margin mismatch",
    )
    require(
        sums["distinct_institutions"]
        == institutions.get("distinct_institution_field_year_pairs"),
        "institution field-year margin mismatch",
    )
    return {
        "rows": len(table),
        "field_count": len(set(fields)),
        "null_year_cells": sum(year is None for year in years),
        "metric_sums": sums,
    }


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "output root is invalid")
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, "output root mode mismatch")
    require({path.name for path in root.iterdir()} == EXPECTED_FILES, "output file set mismatch")
    for path in root.iterdir():
        require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"output file mode mismatch: {path.name}")
    receipt, artifacts = _validate_receipt(root)
    report = _read_json(
        root / "authorship_orcid_affiliation_report.json", "projection report"
    )
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema mismatch")
    require(report.get("status") == receipt["status"], "report status mismatch")
    full_scope = receipt["status"] == FULL_STATUS
    scope = _mapping(report.get("scope"), "scope")
    require(scope.get("full_scope") is full_scope, "report full scope mismatch")
    require(scope.get("component_pass") is full_scope, "report component pass mismatch")
    require(scope.get("pilot_promotion_allowed") is False, "pilot promotion allowed")
    require(scope.get("s0_gate_pass") is False, "report improperly passes S0")
    require(scope.get("d0_authorized") is False, "report improperly authorizes D0")
    require(scope.get("model_fitting_authorized") is False, "report authorizes model fitting")
    require(scope.get("gpu_used") is False, "report claims GPU use")
    require(report.get("remaining_s0_component") == "DISTINCT_COAUTHOR_GRAPH_FIELD_YEAR", "remaining S0 component mismatch")
    gate = _mapping(report.get("gate_status"), "gate status")
    require(gate.get("d0_discovery_pilot") == "NOT_AUTHORIZED", "D0 gate mismatch")
    require(gate.get("g0_measurement_freeze") == "PENDING", "G0 gate mismatch")
    require(gate.get("empirical_g1_signature") == "NOT_RUN", "empirical G1 mismatch")
    require(gate.get("scientific_effect") == "NOT_ESTABLISHED", "effect claim overstated")
    require(
        _mapping(report.get("execution"), "execution").get(
            "scratch_removed_before_completion"
        )
        is True,
        "scratch completion boundary missing",
    )

    config = _validate_provenance(report, receipt)
    require(report.get("claim_boundary") == config.get("claim_boundary"), "claim boundary mismatch")
    source_counts = _validate_source_accounting(
        report, config, full_scope=full_scope
    )
    token_diagnostics = _validate_token_diagnostics(report)
    source_binding = _mapping(report.get("source_binding"), "source binding")
    require(source_binding.get("source_content_rehashed_in_this_run") is False, "source rehash claim mismatch")
    require(source_binding.get("current_exact_stat_verified_for_every_source_file") is True, "source stat verification missing")
    require(
        source_binding.get("selected_checkpoints_verified")
        == source_binding.get("selected_source_files")
        == scope.get("source_files"),
        "selected source binding does not conserve",
    )
    if full_scope:
        source_config = _mapping(config.get("source"), "source config")
        require(
            source_binding.get("selected_source_files")
            == source_config.get("expected_files"),
            "full source file count mismatch",
        )
        require(
            source_binding.get("selected_source_bytes")
            == source_config.get("expected_bytes"),
            "full source byte count mismatch",
        )
        author_config = _mapping(
            config.get("author_history_parent"), "author config"
        )
        events = _mapping(report.get("stable_event_diagnostics"), "event diagnostics")
        require(
            events.get("event_artifacts_rehashed")
            == author_config.get("expected_event_artifacts"),
            "full event artifact count mismatch",
        )
        require(
            events.get("event_rows_scanned")
            == author_config.get("expected_event_rows"),
            "full event row count mismatch",
        )
        cutoff_config = _mapping(config.get("cutoff_parent"), "cutoff config")
        work = _mapping(report.get("work_lookup_binding"), "work binding")
        require(
            work.get("artifacts_rehashed")
            == cutoff_config.get("expected_work_artifacts"),
            "work artifact count mismatch",
        )
        require(
            work.get("rows") == cutoff_config.get("expected_work_rows"),
            "work row count mismatch",
        )

    parquet_path = root / "field_year_orcid_affiliation.parquet"
    table = pq.read_table(parquet_path)
    field_result = _validate_field_year(table, report, source_counts)
    artifact = _mapping(
        _mapping(report["field_year_inventory"], "field-year inventory").get(
            "artifact"
        ),
        "field-year artifact",
    )
    require(
        artifact
        == {
            "relative_path": parquet_path.name,
            "bytes": parquet_path.stat().st_size,
            "sha256": _sha256_file(parquet_path),
            "rows": len(table),
        },
        "field-year artifact identity mismatch",
    )
    return {
        "schema_version": (
            "pivot_feedback_s0_authorship_orcid_affiliation_verification.v1"
        ),
        "status": FULL_VERIFY_STATUS if full_scope else PILOT_VERIFY_STATUS,
        "output_root_basename": root.name,
        "full_scope": full_scope,
        "component_pass": full_scope,
        "artifacts_rehashed": len(artifacts) + 1,
        "output_receipt_sha256": _sha256_file(
            root / "authorship_orcid_affiliation_report.outputs.json"
        ),
        "field_year_verification": field_result,
        "orcid_token_diagnostics": token_diagnostics,
        "coauthor_graph_still_not_run": True,
        "s0_gate_pass": False,
        "d0_authorized": False,
        "model_fitting_authorized": False,
        "scientific_effect": "NOT_ESTABLISHED",
        "verifier_sha256": _sha256_file(Path(__file__).resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        help="write a new private verification report outside the output root",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify(args.output_root)
        if args.report is not None:
            _write_report(args.report, result, root=args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (
        ProjectionVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        pa.ArrowException,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_S0_AUTHORSHIP_ORCID_AFFILIATION_VERIFICATION",
                    "error": str(error),
                }
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
