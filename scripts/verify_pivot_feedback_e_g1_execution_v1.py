#!/usr/bin/env python3
"""Independently verify the terminal BLOCKED E/G1 execution package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_e_g1_execution_v1.yaml"
DEFAULT_OUTPUT_ROOT = Path("/ssd/pivot-feedback-e-g1-execution-v1-20260805")
EXECUTOR = ROOT / "scripts/execute_pivot_feedback_e_g1_v1.py"
VERIFIER = Path(__file__).resolve()
CONFIG_SHA = "1881f70ff9f59c26c2c76f79fbb039bae0a256669935d91a93585845a546a6d2"
TERMINAL = "BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION"

EXPECTED_COLUMNS = {
    "g1_risk_set_and_labels.parquet": (
        "pivot_feedback_e_g1_risk_set_and_labels.v1",
        [
            "analysis_partition_id", "primary_field_id", "certified_episode_id",
            "topic_id", "decision_year", "opportunity_id",
            "current_snapshot_author_token", "entry_outcome", "persistence_outcome",
            "one_off_entry", "alternative_pivot_entry", "peer_active",
            "peer_visible_success", "capacity_bridge", "entry_mode", "sampling_role",
            "inclusion_probability", "raw_sampling_weight", "final_analytic_weight",
            "exclusion_reason",
        ],
    ),
    "g1_episode_estimates.parquet": (
        "pivot_feedback_e_g1_episode_estimates.v1",
        [
            "hypothesis_id", "primary_field_id", "certified_episode_id",
            "pivot_definition", "point_estimate", "interval_lower", "interval_upper",
            "effective_sample_size", "support_status", "estimator_status",
        ],
    ),
    "g1_field_replication_decisions.parquet": (
        "pivot_feedback_e_g1_field_replication_decisions.v1",
        [
            "primary_field_id", "independent_episode_count",
            "field_point_estimate_cohort_years",
            "field_minus_other_fields_one_sided_95_lower_bound_cohort_years",
            "leave_one_field_out_point_estimate_cohort_years",
            "alternative_pivot_field_point_estimate_cohort_years", "direction_pass",
            "replication_noninferiority_pass", "failure_reasons",
        ],
    ),
    "g1_negative_control_decisions.parquet": (
        "pivot_feedback_e_g1_negative_control_decisions.v1",
        [
            "control_id", "control_family", "point_estimate", "interval_lower",
            "interval_upper", "registered_margin_lower", "registered_margin_upper",
            "decision", "failure_reason",
        ],
    ),
    "g1_hypothesis_decisions.parquet": (
        "pivot_feedback_e_g1_hypothesis_decisions.v1",
        [
            "hypothesis_id", "stage", "point_estimate", "interval_lower",
            "interval_upper", "raw_p_value", "holm_adjusted_p_value",
            "prerequisites_pass", "registered_decision", "failure_reason",
        ],
    ),
    "g1_identity_affiliation_sensitivities.parquet": (
        "pivot_feedback_e_g1_identity_affiliation_sensitivities.v1",
        [
            "sensitivity_id", "population_filter", "primary_field_id",
            "hypothesis_id", "point_estimate", "interval_lower", "interval_upper",
            "effective_sample_size", "missing_proxy_category_included",
            "registered_decision_impact", "status", "limitation",
        ],
    ),
}
REPORT_FIELDS = {
    "schema_version", "status", "input_bindings", "access_accounting",
    "scope_accounting", "support_and_weight_diagnostics",
    "h3_primary_and_replication", "negative_control_decisions",
    "field_and_alternative_pivot_stability", "hierarchical_hypothesis_decisions",
    "claim_boundary",
}
RECEIPT_FIELDS = {
    "schema_version", "status", "artifact_count", "artifacts",
    "artifact_tree_sha256", "input_preflight_sha256", "execution_tool_sha256",
    "confirmatory_rows_read", "holdout_rows_read", "model_selection_scores_read",
    "g4_open", "receipt_written_last",
}


class G1ExecutionVerificationError(RuntimeError):
    """Raised when the terminal execution package does not fail closed exactly."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G1ExecutionVerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def verify(config_path: Path, output_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    output_root = output_root.resolve(strict=True)
    require(_sha256(config_path) == CONFIG_SHA, "execution config hash drift")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(config.get("schema_version") == "pivot_feedback_e_g1_execution.v1", "execution config schema drift")
    require(output_root.name == config["output_contract"]["root_basename"], "output root basename drift")
    require(output_root.is_dir() and not output_root.is_symlink(), "unsafe output root")
    require(output_root.stat().st_mode & 0o777 == 0o700, "unsafe output root mode")

    receipt_path = output_root / "g1_execution.outputs.json"
    require(receipt_path.is_file() and not receipt_path.is_symlink(), "missing receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(set(receipt) == RECEIPT_FIELDS, "receipt field drift")
    require(receipt["schema_version"] == "pivot_feedback_e_g1_execution_outputs.v1", "receipt schema drift")
    require(receipt["status"] == TERMINAL, "receipt terminal status drift")
    require(receipt["artifact_count"] == 7, "receipt artifact count drift")
    require(receipt["input_preflight_sha256"] == config["execution_authority"]["preflight_v2"]["sha256"], "effective preflight hash drift")
    require(receipt["execution_tool_sha256"] == _sha256(EXECUTOR), "executor hash drift")
    require(receipt["confirmatory_rows_read"] == 0, "confirmatory rows were read")
    require(receipt["holdout_rows_read"] == 0, "holdout rows were read")
    require(receipt["model_selection_scores_read"] == 0, "model-selection scores were read")
    require(receipt["g4_open"] is False, "G4 opened")
    require(receipt["receipt_written_last"] is True, "receipt not declared last")

    artifacts = receipt["artifacts"]
    require(isinstance(artifacts, list) and len(artifacts) == 7, "artifact array drift")
    require(artifacts == sorted(artifacts, key=lambda row: row["relative_path"]), "artifact order drift")
    require(len({row["relative_path"] for row in artifacts}) == 7, "duplicate artifact path")
    require(hashlib.sha256(_canonical(artifacts)).hexdigest() == receipt["artifact_tree_sha256"], "artifact tree drift")
    expected_names = {*EXPECTED_COLUMNS, "g1_execution_report.json"}
    require({row["relative_path"] for row in artifacts} == expected_names, "artifact name set drift")
    artifact_by_name = {row["relative_path"]: row for row in artifacts}
    for name, (version, columns) in EXPECTED_COLUMNS.items():
        path = output_root / name
        row = artifact_by_name[name]
        require(path.is_file() and not path.is_symlink(), f"missing artifact: {name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"unsafe file mode: {name}")
        require(path.stat().st_size == row["bytes"], f"artifact bytes drift: {name}")
        require(_sha256(path) == row["sha256"], f"artifact hash drift: {name}")
        parquet = pq.ParquetFile(path)
        metadata = parquet.schema_arrow.metadata or {}
        require(metadata.get(b"schema_version", b"").decode() == version, f"artifact schema version drift: {name}")
        require(parquet.schema_arrow.names == columns, f"artifact columns drift: {name}")
        require(parquet.metadata.num_rows == row["rows"] == 0, f"blocked artifact contains rows: {name}")

    report_path = output_root / "g1_execution_report.json"
    report_row = artifact_by_name[report_path.name]
    require(report_path.stat().st_mode & 0o777 == 0o600, "unsafe report mode")
    require(report_path.stat().st_size == report_row["bytes"], "report bytes drift")
    require(_sha256(report_path) == report_row["sha256"], "report hash drift")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(set(report) == REPORT_FIELDS, "report field drift")
    require(report["schema_version"] == "pivot_feedback_e_g1_execution_report.v1", "report schema drift")
    require(report["status"] == TERMINAL, "report status drift")
    block_id = config["blocking_condition"]["block_id"]
    require(report["input_bindings"]["contradiction_replayed"] is True, "conflict not replayed")
    require(report["input_bindings"]["author_cold_guard_rows"] == 8_174_698, "guard row metadata drift")
    require(report["input_bindings"]["config_sha256"] == CONFIG_SHA, "report config hash drift")
    require(report["input_bindings"]["execution_tool_sha256"] == _sha256(EXECUTOR), "report executor hash drift")
    expected_access = {
        "repository_metadata_documents_read": 7,
        "operator_private_g4_artifacts_read": 0,
        "confirmatory_rows_read": 0,
        "holdout_rows_read": 0,
        "labels_materialized": 0,
        "estimators_run": 0,
        "bootstrap_refits_run": 0,
        "model_selection_scores_read": 0,
        "g4_open": False,
        "seal_open_events": 0,
    }
    require(report["access_accounting"] == expected_access, "report access accounting drift")
    scope = report["scope_accounting"]
    require(scope["partition_id"] == "g1_empirical_replication", "partition drift")
    require(scope["entry_years"] == [2013, 2016], "year scope drift")
    require(scope["field_ids_in_registered_slot_order"] == [16, 32, 11], "field scope drift")
    require(scope["episode_count"] == 28 and scope["family_count"] == 13, "episode/family drift")
    require(scope["scope_rows_materialized"] == 0 and scope["scope_changed"] is False, "unsafe scope materialization")
    require(scope["g4_field_or_year_read"] is False, "G4 field/year read")
    for key in (
        "support_and_weight_diagnostics", "h3_primary_and_replication",
        "negative_control_decisions", "field_and_alternative_pivot_stability",
        "hierarchical_hypothesis_decisions",
    ):
        require(report[key]["block_id"] == block_id, f"block lineage drift: {key}")
    require(report["negative_control_decisions"]["evaluated_control_count"] == 0, "negative control evaluated")
    require(report["hierarchical_hypothesis_decisions"]["evaluated_hypothesis_count"] == 0, "hypothesis evaluated")

    require(receipt_path.stat().st_mtime_ns >= max((output_root / row["relative_path"]).stat().st_mtime_ns for row in artifacts), "receipt not written last by mtime")
    require({path.name for path in output_root.iterdir() if path.is_file()} == expected_names | {receipt_path.name}, "unregistered output file")

    return {
        "schema_version": "pivot_feedback_e_g1_execution_summary.v1",
        "status": "PASS_INDEPENDENT_VERIFICATION_OF_TERMINAL_BLOCKED_E_G1_NO_CONFIRMATORY_READ",
        "production_artifact": {
            "root": str(output_root),
            "root_basename": output_root.name,
            "receipt_sha256": _sha256(receipt_path),
            "artifact_count_excluding_receipt": 7,
            "artifact_tree_sha256": receipt["artifact_tree_sha256"],
            "receipt_written_last_by_mtime_ns": True,
        },
        "bound_implementation": {
            "config_relative_path": config_path.relative_to(ROOT).as_posix(),
            "config_sha256": CONFIG_SHA,
            "executor_relative_path": EXECUTOR.relative_to(ROOT).as_posix(),
            "executor_sha256": _sha256(EXECUTOR),
            "independent_verifier_relative_path": VERIFIER.relative_to(ROOT).as_posix(),
            "independent_verifier_sha256": _sha256(VERIFIER),
        },
        "blocking_condition_replay": {
            "block_id": block_id,
            "frozen_g4_author_cold_anti_join_required": True,
            "exact_guard_application_pending": True,
            "private_g4_artifact_access_authorized": False,
            "g4_seal_open_authorized": False,
            "g4_field_or_year_reconstruction_authorized": False,
            "safe_compliant_execution_exists": False,
            "resolution_requires_new_explicit_authority": True,
        },
        "registered_scope_replay": {
            "partition_id": "g1_empirical_replication",
            "entry_years": [2013, 2016],
            "field_ids_in_registered_slot_order": [16, 32, 11],
            "episode_count": 28,
            "family_count": 13,
            "scope_changed": False,
        },
        "access_accounting": expected_access,
        "terminal_state": {
            "empirical_g1_status": "BLOCKED",
            "pass_e": False,
            "pass_g1": False,
            "confirmatory_rows_read": 0,
            "holdout_rows_read": 0,
            "labels_materialized": 0,
            "estimators_run": 0,
            "model_selection_scores_read": 0,
            "g4_open": False,
            "holdout_seal_opened": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "authorization": config["authorization_after_terminal_block"],
        "claim_boundary": config["claim_boundary"],
    }


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise G1ExecutionVerificationError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = verify(args.config, args.output_root)
    if args.output is not None:
        _write_create_only(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
