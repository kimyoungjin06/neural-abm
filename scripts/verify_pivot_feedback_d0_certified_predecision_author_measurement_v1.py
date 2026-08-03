#!/usr/bin/env python3
"""Independently verify the consolidated predecision author measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/"
    "pivot_feedback_d0_certified_predecision_author_measurement_v1.yaml"
)
CONFIG_SCHEMA = (
    "pivot_feedback_d0_certified_predecision_author_measurement_config.v1"
)
REPORT_SCHEMA = (
    "pivot_feedback_d0_certified_predecision_author_measurement_report.v1"
)
RECEIPT_SCHEMA = (
    "pivot_feedback_d0_certified_predecision_author_measurement_outputs.v1"
)
MEASUREMENT_SCHEMA = (
    "pivot_feedback_d0_certified_predecision_author_measurement.v1"
)
REGISTRY_SCHEMA = "pivot_feedback_d0_column_availability_registry.v1"
VERIFY_SCHEMA = (
    "pivot_feedback_d0_certified_predecision_author_measurement_verification.v1"
)
PRODUCER_STATUS = (
    "PASS_EXACT_FOUR_OUTCOME_BLIND_AUTHOR_MEMBERSHIP_AND_"
    "CONSOLIDATED_PREDECISION_JOIN"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_EXACT_FOUR_AUTHOR_MEMBERSHIP_CONSOLIDATED_"
    "TEMPORAL_LEAKAGE_AUDIT"
)
KEY_COLUMNS = (
    "certified_episode_index",
    "field_id",
    "topic_id",
    "entry_year",
    "author_numeric_id",
)
ORCID_COLUMNS = frozenset(
    {
        "raw_authorship_row_count",
        "nonnull_orcid_row_count",
        "distinct_checksum_valid_orcid_count",
        "raw_authorship_observed",
        "orcid_stable_single_valid_token",
    }
)
RAW_DUPLICATE_PEER_COLUMNS = tuple(
    f"prior_target_active_peer_count_cap_{cap}" for cap in (10, 50, 100)
)
EXPECTED_ROOT_FILES = {
    "CERTIFIED_PREDECISION_AUTHOR_MEASUREMENT.md",
    "certified_predecision_author_measurement.json",
    "certified_predecision_author_measurement.outputs.json",
    "column_availability_registry.json",
    "predecision_author_measurement.parquet",
}
FORBIDDEN_COLUMN_FRAGMENTS = (
    "entry_outcome",
    "post_entry",
    "persistence",
    "realized_citation",
    "visible_success",
    "effect_size",
    "predictive_score",
    "pivot_class",
    "g4_",
    "selected_threshold",
    "selected_cap",
    "final_eligible",
)


class CertifiedPredecisionAuthorMeasurementVerificationError(RuntimeError):
    """Fail-closed independent consolidated-measurement verifier error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CertifiedPredecisionAuthorMeasurementVerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} must be a mapping")
    return value


def _yaml(path: Path, label: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} must be a mapping")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be a mapping")
    return value


def _lookup_dotted(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        require(isinstance(current, dict) and part in current, f"missing {dotted}")
        current = current[part]
    return current


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _artifact_tree_sha256(artifacts: Mapping[str, Mapping[str, Any]]) -> str:
    payload = "\n".join(
        f"{name}\t{artifact['sha256']}\t{artifact['bytes']}"
        for name, artifact in sorted(artifacts.items())
    )
    return hashlib.sha256((payload + "\n").encode()).hexdigest()


def _validate_config_and_parents(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    operator_base: Path,
) -> dict[str, dict[str, Any]]:
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    gate = _mapping(
        _mapping(config.get("repository_bindings"), "repository bindings").get(
            "gate_state"
        ),
        "gate state",
    )
    gate_path = REPO_ROOT / str(gate.get("relative_path"))
    require(gate_path.is_file() and _sha256(gate_path) == gate.get("sha256"), "gate binding")
    gate_value = _yaml(gate_path, "gate")
    require(gate_value.get("schema_version") == gate.get("required_schema_version"), "gate schema")
    require(gate_value.get("status") == gate.get("required_status"), "gate status")
    for dotted, expected in _mapping(gate.get("required_fields"), "gate fields").items():
        require(_lookup_dotted(gate_value, str(dotted)) == expected, f"gate field {dotted}")

    join = _mapping(config.get("join_contract"), "join")
    require(tuple(join.get("exact_key_columns", ())) == KEY_COLUMNS, "join keys")
    require(join.get("required_rows") == 3_256_271, "required rows")
    membership = _mapping(config.get("registered_author_membership"), "membership")
    require(
        membership.get("rule_id") == "EXACT_VERIFIED_AUTHOR_OPPORTUNITY_BASE_V1",
        "membership rule",
    )
    require(
        membership.get("upstream_structural_predicate")
        == {
            "column": "portfolio_window_focal_field_work_count_all",
            "operator": "GREATER_THAN_OR_EQUAL_TO",
            "value": 1,
        },
        "membership predicate",
    )
    require(membership.get("single_minimum_history_threshold_selected") is False, "history threshold")
    require(membership.get("threshold_specific_membership_flags_materialized") is False, "threshold flags")
    require(membership.get("graph_team_cap_selected") is False, "graph cap")
    forbidden = _mapping(config.get("forbidden_inputs_and_outputs"), "forbidden")
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden contract")

    resolved: dict[str, dict[str, Any]] = {}
    components = _mapping(config.get("verified_operator_components"), "components")
    for label, raw_spec in components.items():
        spec = _mapping(raw_spec, f"component {label}")
        root = operator_base / str(spec.get("root_basename"))
        receipt = root / str(spec.get("receipt_basename"))
        verification = operator_base / str(spec.get("verification_basename"))
        artifact = root / str(spec.get("artifact_basename"))
        require(root.is_dir() and (root.stat().st_mode & 0o777) == 0o700, f"{label} root")
        for path, expected, kind in (
            (receipt, spec.get("receipt_sha256"), "receipt"),
            (verification, spec.get("verification_sha256"), "verification"),
            (artifact, spec.get("artifact_sha256"), "artifact"),
        ):
            require(path.is_file() and not path.is_symlink(), f"{label} {kind}")
            require(_sha256(path) == expected, f"{label} {kind} hash")
        receipt_value = _json(receipt, f"{label} receipt")
        verification_value = _json(verification, f"{label} verification")
        require(receipt_value.get("schema_version") == spec.get("receipt_schema_version"), f"{label} receipt schema")
        require(receipt_value.get("status") == spec.get("required_status"), f"{label} receipt status")
        require(verification_value.get("status") == spec.get("verification_status"), f"{label} verification status")
        parquet = pq.ParquetFile(artifact)
        require(parquet.metadata.num_rows == spec.get("rows"), f"{label} rows")
        schema = parquet.schema_arrow
        require(
            schema.metadata is not None
            and schema.metadata.get(b"schema_version")
            == str(spec.get("artifact_schema_version")).encode(),
            f"{label} schema",
        )
        resolved[label] = {
            "artifact": artifact,
            "schema": schema,
            "receipt_sha256": spec["receipt_sha256"],
            "verification_sha256": spec["verification_sha256"],
        }
    require(
        tuple(resolved)
        == ("author_opportunity", "leaveout_prior_peer", "identity_affiliation"),
        "component order",
    )
    require(config_path.is_file(), "config missing")
    return resolved


def _expected_schema(parents: Mapping[str, Mapping[str, Any]]) -> pa.Schema:
    base: pa.Schema = parents["author_opportunity"]["schema"]
    leaveout: pa.Schema = parents["leaveout_prior_peer"]["schema"]
    identity: pa.Schema = parents["identity_affiliation"]["schema"]
    fields = list(base)
    fields.append(pa.field("registered_measurement_member", pa.bool_(), nullable=False))
    fields.extend(leaveout.field(name) for name in leaveout.names if name not in KEY_COLUMNS)
    fields.extend(
        identity.field(name)
        for name in identity.names
        if name not in KEY_COLUMNS and name not in RAW_DUPLICATE_PEER_COLUMNS
    )
    require(len(fields) == 74, "expected schema width")
    return pa.schema(fields, metadata={b"schema_version": MEASUREMENT_SCHEMA.encode()})


def _validate_output_root(
    root: Path,
    *,
    config_path: Path,
    expected_schema: pa.Schema,
) -> dict[str, Any]:
    require(root.is_absolute() and root.is_dir(), "output root")
    require((root.stat().st_mode & 0o777) == 0o700, "output root mode")
    paths = list(root.iterdir())
    require({path.name for path in paths} == EXPECTED_ROOT_FILES, "output file set")
    for path in paths:
        require(path.is_file() and not path.is_symlink(), f"unsafe output {path.name}")
        require((path.stat().st_mode & 0o777) == 0o600, f"mode {path.name}")

    receipt_path = root / "certified_predecision_author_measurement.outputs.json"
    report_path = root / "certified_predecision_author_measurement.json"
    registry_path = root / "column_availability_registry.json"
    measurement_path = root / "predecision_author_measurement.parquet"
    receipt = _json(receipt_path, "receipt")
    report = _json(report_path, "report")
    registry = _json(registry_path, "registry")
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema")
    require(receipt.get("status") == PRODUCER_STATUS, "receipt status")
    require(receipt.get("config_sha256") == _sha256(config_path), "receipt config")
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema")
    require(report.get("status") == PRODUCER_STATUS, "report status")
    require(report.get("config_sha256") == _sha256(config_path), "report config")
    require(registry.get("schema_version") == REGISTRY_SCHEMA, "registry schema")
    require(registry.get("measurement_schema_version") == MEASUREMENT_SCHEMA, "registry measurement schema")

    artifacts = _mapping(receipt.get("artifacts"), "receipt artifacts")
    require(set(artifacts) == EXPECTED_ROOT_FILES - {receipt_path.name}, "receipt artifact names")
    observed_artifacts: dict[str, dict[str, Any]] = {}
    for name in sorted(artifacts):
        path = root / name
        observed = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        require(observed == artifacts[name], f"artifact binding {name}")
        observed_artifacts[name] = observed
    require(receipt.get("artifact_count") == len(observed_artifacts), "artifact count")
    require(
        receipt.get("artifact_tree_sha256") == _artifact_tree_sha256(observed_artifacts),
        "artifact tree",
    )

    parquet = pq.ParquetFile(measurement_path)
    require(parquet.metadata.num_rows == 3_256_271, "measurement rows")
    require(parquet.schema_arrow.equals(expected_schema, check_metadata=True), "measurement schema")
    require(receipt.get("measurement_rows") == 3_256_271, "receipt rows")
    require(receipt.get("measurement_columns") == 74, "receipt columns")
    require(receipt.get("registered_measurement_members") == 3_256_271, "receipt members")
    for flag in (
        "single_minimum_history_threshold_selected",
        "graph_team_cap_selected",
        "outcomes_read",
        "realized_citation_rows_read",
        "peer_visible_success_computed",
        "api_requests_created",
        "independent_temporal_verification_complete",
        "measurement_completeness_decision_authorized",
        "production_pivot_classification_authorized",
        "model_fitting_authorized",
        "g4_open",
        "gpu_used",
    ):
        require(receipt.get(flag) in (False, 0), f"receipt flag {flag}")
    return {
        "receipt": receipt,
        "report": report,
        "registry": registry,
        "receipt_path": receipt_path,
        "measurement_path": measurement_path,
        "artifacts": observed_artifacts,
        "row_groups": parquet.metadata.num_row_groups,
    }


def _availability_for(column: str) -> tuple[str, str]:
    if column in KEY_COLUMNS:
        return "IDENTIFIER_OPERATOR_LOCAL", "JOIN_KEY"
    if column == "registered_measurement_member":
        return "REGISTERED_MEMBERSHIP", "OUTCOME_BLIND_STRUCTURAL_MEMBERSHIP"
    if column in ORCID_COLUMNS:
        return "CURRENT_SNAPSHOT_SENSITIVITY", "ORCID_SUPPORT_OR_STABILITY_SENSITIVITY"
    if "institution_comparable" in column or "same_affiliation" in column:
        return "CURRENT_SNAPSHOT_WORK_TIME_PROXY", "AFFILIATION_COVERAGE_OR_BRIDGE_PROXY"
    return "PREDECISION_T_MINUS_1", "PREDECISION_AUTHOR_MEASUREMENT"


def _verify_registry(registry: Mapping[str, Any], schema: pa.Schema) -> dict[str, Any]:
    columns = registry.get("columns")
    require(isinstance(columns, list) and len(columns) == len(schema), "registry columns")
    counts: dict[str, int] = {}
    for field, raw in zip(schema, columns, strict=True):
        entry = _mapping(raw, f"registry {field.name}")
        availability, role = _availability_for(field.name)
        expected = {
            "name": field.name,
            "arrow_type": str(field.type),
            "nullable": field.nullable,
            "availability": availability,
            "role": role,
            "primary_model_input_authorized_now": False,
            "raw_numeric_identifier": field.name in KEY_COLUMNS,
        }
        require(dict(entry) == expected, f"registry entry {field.name}")
        counts[availability] = counts.get(availability, 0) + 1
    require(registry.get("column_count") == len(schema), "registry count")
    require(registry.get("availability_counts") == counts, "registry availability counts")
    for flag in (
        "current_snapshot_topic_annotation_is_historical_observation",
        "current_snapshot_orcid_is_historical_observation",
        "work_time_institution_proxy_is_historical_affiliation_truth",
        "primary_model_input_authorized_now",
        "raw_numeric_ids_repository_export_authorized",
    ):
        require(registry.get(flag) is False, f"registry boundary {flag}")
    return {
        "column_count": len(schema),
        "availability_counts": counts,
        "unclassified_columns": 0,
        "current_snapshot_sensitivity_mislabeled_predecision": 0,
    }


def _strictly_sorted_unique(table: pa.Table) -> bool:
    arrays = [
        table[column].combine_chunks().to_numpy(zero_copy_only=False)
        for column in KEY_COLUMNS
    ]
    length = table.num_rows
    require(length > 0, "empty measurement")
    prefix_equal = np.ones(length - 1, dtype=np.bool_)
    strictly_greater = np.zeros(length - 1, dtype=np.bool_)
    for values in arrays:
        left = values[:-1]
        right = values[1:]
        strictly_greater |= prefix_equal & (right > left)
        prefix_equal &= right == left
    return bool(np.all(strictly_greater))


def _sorted_indices(table: pa.Table) -> pa.Array:
    return pc.sort_indices(
        table.select(KEY_COLUMNS),
        sort_keys=[(column, "ascending") for column in KEY_COLUMNS],
    )


def _arrays_equal(left: pa.ChunkedArray | pa.Array, right: pa.ChunkedArray | pa.Array) -> bool:
    left_array = left.combine_chunks() if isinstance(left, pa.ChunkedArray) else left
    right_array = right.combine_chunks() if isinstance(right, pa.ChunkedArray) else right
    return left_array.equals(right_array)


def _compare_component(
    *,
    parent_path: Path,
    measurement_path: Path,
    parent_columns: Sequence[str],
    output_columns: Sequence[str],
) -> dict[str, Any]:
    require(len(parent_columns) == len(output_columns), "comparison column pairing")
    requested_parent = list(dict.fromkeys((*KEY_COLUMNS, *parent_columns)))
    requested_output = list(dict.fromkeys((*KEY_COLUMNS, *output_columns)))
    parent = pq.read_table(parent_path, columns=requested_parent)
    observed = pq.read_table(measurement_path, columns=requested_output)
    require(parent.num_rows == observed.num_rows == 3_256_271, "comparison rows")
    indices = _sorted_indices(parent)
    mismatches: list[str] = []
    for key in KEY_COLUMNS:
        expected = pc.take(parent[key], indices)
        if not _arrays_equal(expected, observed[key]):
            mismatches.append(f"key:{key}")
    for parent_name, output_name in zip(parent_columns, output_columns, strict=True):
        expected = pc.take(parent[parent_name], indices)
        if not _arrays_equal(expected, observed[output_name]):
            mismatches.append(f"{parent_name}->{output_name}")
    return {
        "rows": parent.num_rows,
        "columns_compared": len(KEY_COLUMNS) + len(parent_columns),
        "bidirectional_key_difference": sum(
            item.startswith("key:") for item in mismatches
        ),
        "value_column_mismatches": sum(
            not item.startswith("key:") for item in mismatches
        ),
        "mismatch_labels": mismatches,
    }


def _boolean_all(array: pa.ChunkedArray, expected: bool) -> bool:
    values = array.combine_chunks().to_numpy(zero_copy_only=False)
    return bool(np.all(values == expected))


def _count_violations(
    table: pa.Table,
    columns: Sequence[str],
    *,
    direction: str = "nondecreasing",
) -> int:
    arrays = [
        table[column].combine_chunks().to_numpy(zero_copy_only=False)
        for column in columns
    ]
    violation = np.zeros(table.num_rows, dtype=np.bool_)
    for left, right in zip(arrays, arrays[1:]):
        if direction == "nondecreasing":
            violation |= left > right
        else:
            violation |= left < right
    return int(np.count_nonzero(violation))


def _temporal_and_sensitivity_audit(
    *,
    measurement_path: Path,
    schema: pa.Schema,
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    forbidden_columns = [
        name
        for name in schema.names
        if any(fragment in name.lower() for fragment in FORBIDDEN_COLUMN_FRAGMENTS)
    ]
    require(not forbidden_columns, f"forbidden measurement columns: {forbidden_columns}")
    require(not any("history_at_least" in name for name in schema.names), "threshold flags")
    require(not any("minimum_history" in name for name in schema.names), "minimum-history column")

    columns = [
        "entry_year",
        "first_target_topic_year",
        "portfolio_window_focal_field_work_count_all",
        "registered_measurement_member",
    ]
    for cap in (10, 50, 100):
        columns.extend(
            [
                f"prior_target_active_peer_count_cap_{cap}",
                f"has_prior_target_active_peer_cap_{cap}",
                f"institution_comparable_prior_peer_count_cap_{cap}",
                f"same_affiliation_prior_peer_count_cap_{cap}",
                f"has_institution_comparable_prior_peer_cap_{cap}",
                f"has_same_affiliation_bridge_cap_{cap}",
            ]
        )
    for prefix in (
        "pre_entry_all_field_work_count",
        "pre_entry_focal_field_work_count",
        "portfolio_window_all_field_work_count",
        "portfolio_window_focal_field_work_count",
    ):
        columns.extend(
            [
                f"{prefix}_cap_10",
                f"{prefix}_cap_50",
                f"{prefix}_cap_100",
                f"{prefix}_all",
            ]
        )
    table = pq.read_table(measurement_path, columns=list(dict.fromkeys(columns)))
    require(_boolean_all(table["registered_measurement_member"], True), "membership flag")
    history_all = table[
        "portfolio_window_focal_field_work_count_all"
    ].combine_chunks().to_numpy(zero_copy_only=False)
    structural_failures = int(np.count_nonzero(history_all < 1))
    require(structural_failures == 0, "structural membership failures")

    years = table["entry_year"].combine_chunks().to_numpy(zero_copy_only=False)
    first = table["first_target_topic_year"].combine_chunks()
    first_values = first.fill_null(0).to_numpy(zero_copy_only=False)
    first_valid = pc.is_valid(first).to_numpy(zero_copy_only=False)
    first_year_violations = int(
        np.count_nonzero(first_valid & (first_values > (years - 1)))
    )
    require(first_year_violations == 0, "first target year after t-1")

    monotonicity: dict[str, int] = {}
    for prefix in (
        "pre_entry_all_field_work_count",
        "pre_entry_focal_field_work_count",
        "portfolio_window_all_field_work_count",
        "portfolio_window_focal_field_work_count",
    ):
        value = _count_violations(
            table,
            [
                f"{prefix}_cap_10",
                f"{prefix}_cap_50",
                f"{prefix}_cap_100",
                f"{prefix}_all",
            ],
        )
        require(value == 0, f"{prefix} monotonicity")
        monotonicity[prefix] = value
    for prefix in (
        "prior_target_active_peer_count",
        "institution_comparable_prior_peer_count",
        "same_affiliation_prior_peer_count",
        "has_prior_target_active_peer",
        "has_institution_comparable_prior_peer",
        "has_same_affiliation_bridge",
    ):
        value = _count_violations(
            table,
            [f"{prefix}_cap_{cap}" for cap in (10, 50, 100)],
        )
        require(value == 0, f"{prefix} monotonicity")
        monotonicity[prefix] = value

    for source, label in (
        (config["forbidden_inputs_and_outputs"], "config"),
        (report["forbidden_inputs_and_outputs"], "report"),
    ):
        require(
            isinstance(source, dict) and source and all(value is False for value in source.values()),
            f"{label} forbidden flags",
        )
    for source, label in ((report["membership"], "report"), (receipt, "receipt")):
        require(source.get("single_minimum_history_threshold_selected") is False, f"{label} threshold")
        require(source.get("graph_team_cap_selected") is False, f"{label} cap")
    return {
        "forbidden_measurement_columns": forbidden_columns,
        "structural_membership_predicate_failures": structural_failures,
        "first_target_topic_year_after_t_minus_1_violations": first_year_violations,
        "nonranked_cap_monotonicity_violations": monotonicity,
        "single_minimum_history_threshold_selected": False,
        "threshold_specific_membership_flags_materialized": False,
        "graph_team_cap_selected": False,
        "entry_or_persistence_rows_read": False,
        "realized_citation_outcome_rows_read": False,
        "peer_visible_success_computed": False,
        "current_snapshot_orcid_role": "SENSITIVITY_ONLY",
        "affiliation_role": "SECONDARY_CURRENT_SNAPSHOT_WORK_TIME_PROXY",
        "raw_numeric_ids_role": "PRIVATE_OPERATOR_LOCAL_ONLY",
    }


def run(
    *,
    config_path: Path,
    operator_base: Path,
    producer_root: Path,
    verification_path: Path,
) -> dict[str, Any]:
    started = time.time()
    config_path = config_path.resolve()
    operator_base = operator_base.resolve()
    producer_root = producer_root.resolve()
    verification_path = verification_path.resolve()
    require(verification_path.is_absolute(), "verification path must be absolute")
    require(not verification_path.exists(), "verification path already exists")
    require(verification_path.parent == operator_base, "verification must be external sibling")
    config = _yaml(config_path, "config")
    parents = _validate_config_and_parents(
        config,
        config_path=config_path,
        operator_base=operator_base,
    )
    schema = _expected_schema(parents)
    output = _validate_output_root(
        producer_root,
        config_path=config_path,
        expected_schema=schema,
    )
    registry_audit = _verify_registry(output["registry"], schema)

    measurement_path: Path = output["measurement_path"]
    output_keys = pq.read_table(measurement_path, columns=list(KEY_COLUMNS))
    sorted_unique = _strictly_sorted_unique(output_keys)
    require(sorted_unique, "output keys are not globally strict sorted unique")

    base_names = [
        name
        for name in parents["author_opportunity"]["schema"].names
        if name not in KEY_COLUMNS
    ]
    leaveout_names = [
        name
        for name in parents["leaveout_prior_peer"]["schema"].names
        if name not in KEY_COLUMNS
    ]
    identity_names = [
        name
        for name in parents["identity_affiliation"]["schema"].names
        if name not in KEY_COLUMNS and name not in RAW_DUPLICATE_PEER_COLUMNS
    ]
    component_replays = {
        "author_opportunity": _compare_component(
            parent_path=parents["author_opportunity"]["artifact"],
            measurement_path=measurement_path,
            parent_columns=base_names,
            output_columns=base_names,
        ),
        "leaveout_prior_peer": _compare_component(
            parent_path=parents["leaveout_prior_peer"]["artifact"],
            measurement_path=measurement_path,
            parent_columns=leaveout_names,
            output_columns=leaveout_names,
        ),
        "identity_affiliation": _compare_component(
            parent_path=parents["identity_affiliation"]["artifact"],
            measurement_path=measurement_path,
            parent_columns=identity_names,
            output_columns=identity_names,
        ),
        "identity_peer_count_cross_component": _compare_component(
            parent_path=parents["identity_affiliation"]["artifact"],
            measurement_path=measurement_path,
            parent_columns=RAW_DUPLICATE_PEER_COLUMNS,
            output_columns=RAW_DUPLICATE_PEER_COLUMNS,
        ),
    }
    for label, replay in component_replays.items():
        require(replay["bidirectional_key_difference"] == 0, f"{label} key replay")
        require(replay["value_column_mismatches"] == 0, f"{label} value replay")

    temporal = _temporal_and_sensitivity_audit(
        measurement_path=measurement_path,
        schema=schema,
        config=config,
        report=output["report"],
        receipt=output["receipt"],
    )
    verification = {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "verifier": {
            "strategy": "PYARROW_INDEPENDENT_KEY_SORT_AND_COLUMN_EXACT_REPLAY",
            "imports_producer_module": False,
            "uses_producer_join_engine": False,
            "producer_join_engine": "DUCKDB_HASH_JOIN",
            "verification_engine": "PYARROW_SORT_INDICES_AND_EXACT_ARRAY_COMPARISON",
            "elapsed_seconds": round(time.time() - started, 3),
        },
        "output_root_basename": producer_root.name,
        "output_receipt_sha256": _sha256(output["receipt_path"]),
        "output_artifacts": output["artifacts"],
        "scope": {
            "exact_certified_episode_rows": 4,
            "author_episode_rows": 3_256_271,
            "measurement_columns": 74,
            "output_keys_globally_strict_sorted_unique": sorted_unique,
            "outcome_blind": True,
        },
        "registry_audit": registry_audit,
        "component_replays": component_replays,
        "temporal_leakage_audit": temporal,
        "authorization": {
            "membership_rule_independently_verified": True,
            "consolidated_predecision_join_independently_verified": True,
            "temporal_leakage_audit_complete": True,
            "successor_gate_may_decide_measurement_completeness": True,
            "measurement_completeness_established_by_verifier_alone": False,
            "production_pivot_classification_authorized": False,
            "discovery_outcome_opening_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
        },
        "claim_boundary": (
            "This independent pass establishes exact preservation of the three "
            "verified author components, the threshold-free registered "
            "membership population, and explicit temporal classification of "
            "every consolidated column. ORCID remains current-snapshot "
            "sensitivity only and affiliation remains a secondary work-time "
            "proxy. The verifier does not itself select a history threshold or "
            "graph cap, establish final eligibility, open outcomes, classify "
            "pivots, fit a model, open G4, or establish a scientific effect."
        ),
    }
    _write_json_new(verification_path, verification)
    os.chmod(verification_path, 0o600)
    return verification


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--operator-base", type=Path, default=Path("/ssd"))
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--verification-path", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    verification = run(
        config_path=args.config,
        operator_base=args.operator_base,
        producer_root=args.producer_root,
        verification_path=args.verification_path,
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
