#!/usr/bin/env python3
"""Execute G1 fail-closed and emit the registered terminal BLOCKED package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_e_g1_execution_v1.yaml"
DEFAULT_OUTPUT = Path("/ssd/pivot-feedback-e-g1-execution-v1-20260805")
CONFIG_SCHEMA = "pivot_feedback_e_g1_execution.v1"
TERMINAL = "BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION"


class G1ExecutionError(RuntimeError):
    """Raised when the fail-closed execution package cannot be produced safely."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G1ExecutionError(message)


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


def _safe_repo_path(relative: str) -> Path:
    pure = PurePosixPath(relative)
    require(relative and not pure.is_absolute() and ".." not in pure.parts, "unsafe repository path")
    path = (ROOT / pure).resolve(strict=True)
    path.relative_to(ROOT.resolve())
    require(path.is_file() and not path.is_symlink(), "invalid repository path")
    return path


def _load_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _safe_repo_path(str(binding["relative_path"]))
    require(_sha256(path) == binding["sha256"], f"binding hash drift: {path.name}")
    payload = (
        yaml.safe_load(path.read_text(encoding="utf-8"))
        if path.suffix == ".yaml"
        else json.loads(path.read_text(encoding="utf-8"))
    )
    require(payload.get("schema_version") == binding["required_schema_version"], f"binding schema drift: {path.name}")
    require(payload.get("status") == binding["required_status"], f"binding status drift: {path.name}")
    return payload


def _schema(fields: list[tuple[str, pa.DataType]]) -> pa.Schema:
    return pa.schema([pa.field(name, kind, nullable=True) for name, kind in fields])


RISK_SCHEMA = _schema(
    [
        ("analysis_partition_id", pa.string()),
        ("primary_field_id", pa.uint8()),
        ("certified_episode_id", pa.uint16()),
        ("topic_id", pa.uint32()),
        ("decision_year", pa.uint16()),
        ("opportunity_id", pa.string()),
        ("current_snapshot_author_token", pa.string()),
        ("entry_outcome", pa.bool_()),
        ("persistence_outcome", pa.bool_()),
        ("one_off_entry", pa.bool_()),
        ("alternative_pivot_entry", pa.bool_()),
        ("peer_active", pa.bool_()),
        ("peer_visible_success", pa.bool_()),
        ("capacity_bridge", pa.bool_()),
        ("entry_mode", pa.string()),
        ("sampling_role", pa.string()),
        ("inclusion_probability", pa.float64()),
        ("raw_sampling_weight", pa.float64()),
        ("final_analytic_weight", pa.float64()),
        ("exclusion_reason", pa.string()),
    ]
)
EPISODE_SCHEMA = _schema(
    [
        ("hypothesis_id", pa.string()),
        ("primary_field_id", pa.uint8()),
        ("certified_episode_id", pa.uint16()),
        ("pivot_definition", pa.string()),
        ("point_estimate", pa.float64()),
        ("interval_lower", pa.float64()),
        ("interval_upper", pa.float64()),
        ("effective_sample_size", pa.float64()),
        ("support_status", pa.string()),
        ("estimator_status", pa.string()),
    ]
)
FIELD_SCHEMA = _schema(
    [
        ("primary_field_id", pa.uint8()),
        ("independent_episode_count", pa.uint16()),
        ("field_point_estimate_cohort_years", pa.float64()),
        ("field_minus_other_fields_one_sided_95_lower_bound_cohort_years", pa.float64()),
        ("leave_one_field_out_point_estimate_cohort_years", pa.float64()),
        ("alternative_pivot_field_point_estimate_cohort_years", pa.float64()),
        ("direction_pass", pa.bool_()),
        ("replication_noninferiority_pass", pa.bool_()),
        ("failure_reasons", pa.string()),
    ]
)
CONTROL_SCHEMA = _schema(
    [
        ("control_id", pa.string()),
        ("control_family", pa.string()),
        ("point_estimate", pa.float64()),
        ("interval_lower", pa.float64()),
        ("interval_upper", pa.float64()),
        ("registered_margin_lower", pa.float64()),
        ("registered_margin_upper", pa.float64()),
        ("decision", pa.string()),
        ("failure_reason", pa.string()),
    ]
)
HYPOTHESIS_SCHEMA = _schema(
    [
        ("hypothesis_id", pa.string()),
        ("stage", pa.string()),
        ("point_estimate", pa.float64()),
        ("interval_lower", pa.float64()),
        ("interval_upper", pa.float64()),
        ("raw_p_value", pa.float64()),
        ("holm_adjusted_p_value", pa.float64()),
        ("prerequisites_pass", pa.bool_()),
        ("registered_decision", pa.string()),
        ("failure_reason", pa.string()),
    ]
)
SENSITIVITY_SCHEMA = _schema(
    [
        ("sensitivity_id", pa.string()),
        ("population_filter", pa.string()),
        ("primary_field_id", pa.uint8()),
        ("hypothesis_id", pa.string()),
        ("point_estimate", pa.float64()),
        ("interval_lower", pa.float64()),
        ("interval_upper", pa.float64()),
        ("effective_sample_size", pa.float64()),
        ("missing_proxy_category_included", pa.bool_()),
        ("registered_decision_impact", pa.string()),
        ("status", pa.string()),
        ("limitation", pa.string()),
    ]
)

SCHEMAS = {
    "g1_risk_set_and_labels.parquet": (RISK_SCHEMA, "pivot_feedback_e_g1_risk_set_and_labels.v1"),
    "g1_episode_estimates.parquet": (EPISODE_SCHEMA, "pivot_feedback_e_g1_episode_estimates.v1"),
    "g1_field_replication_decisions.parquet": (FIELD_SCHEMA, "pivot_feedback_e_g1_field_replication_decisions.v1"),
    "g1_negative_control_decisions.parquet": (CONTROL_SCHEMA, "pivot_feedback_e_g1_negative_control_decisions.v1"),
    "g1_hypothesis_decisions.parquet": (HYPOTHESIS_SCHEMA, "pivot_feedback_e_g1_hypothesis_decisions.v1"),
    "g1_identity_affiliation_sensitivities.parquet": (SENSITIVITY_SCHEMA, "pivot_feedback_e_g1_identity_affiliation_sensitivities.v1"),
}


def _write_parquet(path: Path, schema: pa.Schema, version: str) -> None:
    versioned = schema.with_metadata({b"schema_version": version.encode("utf-8")})
    pq.write_table(pa.Table.from_pylist([], schema=versioned), path, compression="zstd", version="2.6")
    os.chmod(path, 0o600)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
    os.chmod(path, 0o600)


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        row["rows"] = pq.ParquetFile(path).metadata.num_rows
    return row


def _validate_conflict(config: Mapping[str, Any]) -> dict[str, Any]:
    authority = config["execution_authority"]
    v1 = _load_binding(authority["preflight_v1"])
    v1_summary = _load_binding(authority["preflight_v1_verification"])
    v2 = _load_binding(authority["preflight_v2"])
    v2_summary = _load_binding(authority["preflight_v2_verification"])
    evidence = config["conflict_evidence"]
    sections = _load_binding(evidence["frozen_scientific_sections"])
    guard = _load_binding(evidence["author_cold_guard_summary"])
    active = _load_binding(evidence["active_state"])

    require(v1_summary["fail_closed_state"]["confirmatory_rows_read"] == 0, "predecessor confirmatory access")
    require(v2_summary["fail_closed_state"]["confirmatory_rows_read"] == 0, "v2 confirmatory access")
    require(active["lifecycle"]["empirical_g1_status"] == "NOT_RUN", "G1 no longer not-run")
    atomicity = sections["sections"]["partition_and_atomic_fold_resolution"]["atomicity"]
    require(atomicity["g4_authors_removed_from_all_earlier_fit_or_selection_rows"] is True, "author-cold rule not required")
    cold = guard["author_cold_state"]
    require(cold["exclude_from_g1"] is True, "guard not required for G1")
    require(cold["exact_projection_complete"] is True, "guard not exact")
    require(cold["earlier_partition_guard_application_complete"] is False, "guard already applied")
    require(guard["production_artifact"]["author_cold_guard_rows"] == evidence["author_cold_guard_summary"]["required_guard_rows"], "guard row count drift")
    scope = v1["registered_execution_scope"]
    authorization = v1["authorization"]
    require(scope["g4_private_artifact_root_or_manifest_is_execution_input"] is False, "private G4 unexpectedly included")
    require(scope["g4_field_allowed"] is False and scope["g4_entry_years_allowed"] is False, "G4 reconstruction unexpectedly allowed")
    require(authorization["g4_artifact_access"] == "NOT_AUTHORIZED", "G4 artifact access unexpectedly authorized")
    require(authorization["g4_seal_open"] == "NOT_AUTHORIZED", "G4 seal unexpectedly authorized")
    require(v2["effective_execution_input_contract"]["private_g4_holdout_included"] is False, "v2 private G4 included")
    require(config["blocking_condition"]["safe_compliant_execution_exists"] is False, "config does not fail closed")
    return {
        "preflight_v1_sha256": authority["preflight_v1"]["sha256"],
        "preflight_v1_verification_sha256": authority["preflight_v1_verification"]["sha256"],
        "preflight_v2_sha256": authority["preflight_v2"]["sha256"],
        "preflight_v2_verification_sha256": authority["preflight_v2_verification"]["sha256"],
        "scientific_sections_sha256": evidence["frozen_scientific_sections"]["sha256"],
        "author_cold_guard_summary_sha256": evidence["author_cold_guard_summary"]["sha256"],
        "active_state_sha256": evidence["active_state"]["sha256"],
        "author_cold_guard_rows": evidence["author_cold_guard_summary"]["required_guard_rows"],
        "contradiction_replayed": True,
    }


def execute(config_path: Path, output_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    require(output_root.name == config["output_contract"]["root_basename"], "output basename drift")
    require(not output_root.exists() and output_root.parent.is_dir(), "output directory must be new")
    bindings = _validate_conflict(config)

    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    os.chmod(staging, 0o700)
    for name, (schema, version) in SCHEMAS.items():
        _write_parquet(staging / name, schema, version)

    block_id = config["blocking_condition"]["block_id"]
    report = {
        "schema_version": config["output_contract"]["report_schema_version"],
        "status": TERMINAL,
        "input_bindings": {**bindings, "config_sha256": _sha256(config_path), "execution_tool_sha256": _sha256(Path(__file__).resolve())},
        "access_accounting": {
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
        },
        "scope_accounting": {
            **config["registered_scope"],
            "scope_rows_materialized": 0,
            "scope_changed": False,
            "g4_field_or_year_read": False,
        },
        "support_and_weight_diagnostics": {"status": "NOT_EVALUATED", "block_id": block_id},
        "h3_primary_and_replication": {"status": "NOT_RUN", "block_id": block_id},
        "negative_control_decisions": {"required_control_count": 7, "evaluated_control_count": 0, "status": "NOT_RUN", "block_id": block_id},
        "field_and_alternative_pivot_stability": {"evaluated_field_count": 0, "status": "NOT_RUN", "block_id": block_id},
        "hierarchical_hypothesis_decisions": {"evaluated_hypothesis_count": 0, "status": "NOT_RUN", "block_id": block_id},
        "claim_boundary": config["claim_boundary"],
    }
    report_path = staging / "g1_execution_report.json"
    _write_json(report_path, report)
    artifact_paths = sorted(path for path in staging.iterdir() if path.is_file())
    artifacts = [_artifact(path, staging) for path in artifact_paths]
    require(len(artifacts) == config["output_contract"]["artifact_count_excluding_receipt"], "artifact count before receipt")
    receipt = {
        "schema_version": config["output_contract"]["receipt_schema_version"],
        "status": TERMINAL,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "artifact_tree_sha256": hashlib.sha256(_canonical(artifacts)).hexdigest(),
        "input_preflight_sha256": config["execution_authority"]["preflight_v2"]["sha256"],
        "execution_tool_sha256": _sha256(Path(__file__).resolve()),
        "confirmatory_rows_read": 0,
        "holdout_rows_read": 0,
        "model_selection_scores_read": 0,
        "g4_open": False,
        "receipt_written_last": True,
    }
    _write_json(staging / "g1_execution.outputs.json", receipt)
    os.replace(staging, output_root)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = execute(args.config, args.output_root)
    print(json.dumps({
        "status": receipt["status"],
        "artifact_count": receipt["artifact_count"],
        "confirmatory_rows_read": receipt["confirmatory_rows_read"],
        "g4_open": receipt["g4_open"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
