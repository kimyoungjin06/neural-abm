#!/usr/bin/env python3
"""Register and consolidate the exact-four predecision author measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow as pa
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
STATUS = (
    "PASS_EXACT_FOUR_OUTCOME_BLIND_AUTHOR_MEMBERSHIP_AND_"
    "CONSOLIDATED_PREDECISION_JOIN"
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


class CertifiedPredecisionAuthorMeasurementError(RuntimeError):
    """Fail-closed consolidated author-measurement error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CertifiedPredecisionAuthorMeasurementError(message)


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


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)


def _write_text_new(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _sql_identifier(name: str) -> str:
    require(name.replace("_", "").isalnum(), f"unsafe column name: {name}")
    return f'"{name}"'


def _scalar(connection: duckdb.DuckDBPyConnection, query: str) -> Any:
    row = connection.execute(query).fetchone()
    require(row is not None and len(row) == 1, "scalar query returned no value")
    return row[0]


def _artifact_tree_sha256(artifacts: Mapping[str, Mapping[str, Any]]) -> str:
    payload = "\n".join(
        f"{name}\t{artifact['sha256']}\t{artifact['bytes']}"
        for name, artifact in sorted(artifacts.items())
    )
    return hashlib.sha256((payload + "\n").encode()).hexdigest()


def _lookup_dotted(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        require(isinstance(current, dict) and part in current, f"missing {dotted}")
        current = current[part]
    return current


def _validate_repository_binding(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> None:
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(config.get("status_before_execution") == "AUTHORIZED_BY_GATE_STATE_V40", "gate authorization")
    bindings = _mapping(config.get("repository_bindings"), "repository_bindings")
    gate = _mapping(bindings.get("gate_state"), "repository_bindings.gate_state")
    gate_path = REPO_ROOT / str(gate.get("relative_path"))
    require(gate_path.is_file(), "gate-state file missing")
    require(_sha256(gate_path) == gate.get("sha256"), "gate-state SHA-256")
    gate_value = _yaml(gate_path, "gate state")
    require(gate_value.get("schema_version") == gate.get("required_schema_version"), "gate schema")
    require(gate_value.get("status") == gate.get("required_status"), "gate status")
    for dotted, expected in _mapping(gate.get("required_fields"), "gate required fields").items():
        require(_lookup_dotted(gate_value, str(dotted)) == expected, f"gate field {dotted}")
    require(config_path.resolve().is_file(), "config file missing")


def _validate_parent(
    *,
    operator_base: Path,
    spec: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    root = operator_base / str(spec.get("root_basename"))
    require(root.is_dir(), f"{label} root missing")
    require((root.stat().st_mode & 0o777) == 0o700, f"{label} root mode")

    receipt = root / str(spec.get("receipt_basename"))
    verification = operator_base / str(spec.get("verification_basename"))
    artifact = root / str(spec.get("artifact_basename"))
    for path, expected_sha, name in (
        (receipt, spec.get("receipt_sha256"), "receipt"),
        (verification, spec.get("verification_sha256"), "verification"),
        (artifact, spec.get("artifact_sha256"), "artifact"),
    ):
        require(path.is_file() and not path.is_symlink(), f"{label} {name} missing")
        require(_sha256(path) == expected_sha, f"{label} {name} SHA-256")

    receipt_value = _json(receipt, f"{label} receipt")
    require(receipt_value.get("schema_version") == spec.get("receipt_schema_version"), f"{label} receipt schema")
    require(receipt_value.get("status") == spec.get("required_status"), f"{label} receipt status")
    verification_value = _json(verification, f"{label} verification")
    require(verification_value.get("status") == spec.get("verification_status"), f"{label} verification status")

    parquet = pq.ParquetFile(artifact)
    require(parquet.metadata.num_rows == spec.get("rows"), f"{label} artifact rows")
    schema = parquet.schema_arrow
    require(
        schema.metadata is not None
        and schema.metadata.get(b"schema_version")
        == str(spec.get("artifact_schema_version")).encode(),
        f"{label} artifact schema",
    )
    require(tuple(schema.names[: len(KEY_COLUMNS)]) == KEY_COLUMNS, f"{label} key prefix")
    return {
        "root": root,
        "receipt": receipt,
        "verification": verification,
        "artifact": artifact,
        "schema": schema,
        "rows": parquet.metadata.num_rows,
    }


def validate_config(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    operator_base: Path,
) -> dict[str, dict[str, Any]]:
    _validate_repository_binding(config, config_path=config_path)
    require(operator_base.is_absolute() and operator_base.is_dir(), "operator base")
    components = _mapping(config.get("verified_operator_components"), "components")
    require(
        tuple(components) == ("author_opportunity", "leaveout_prior_peer", "identity_affiliation"),
        "component order",
    )
    resolved = {
        label: _validate_parent(
            operator_base=operator_base,
            spec=_mapping(spec, f"component {label}"),
            label=label,
        )
        for label, spec in components.items()
    }
    join = _mapping(config.get("join_contract"), "join contract")
    require(tuple(join.get("exact_key_columns", ())) == KEY_COLUMNS, "join keys")
    require(join.get("required_rows") == 3_256_271, "join rows")
    require(join.get("require_unique_key_in_every_component") is True, "unique-key contract")
    require(join.get("require_exact_bidirectional_key_equality") is True, "key equality contract")
    require(
        join.get("require_identity_affiliation_peer_counts_equal_leaveout_peer_counts") is True,
        "peer-count equality contract",
    )
    membership = _mapping(config.get("registered_author_membership"), "membership")
    predicate = _mapping(membership.get("upstream_structural_predicate"), "membership predicate")
    require(membership.get("rule_id") == "EXACT_VERIFIED_AUTHOR_OPPORTUNITY_BASE_V1", "membership rule")
    require(predicate == {
        "column": "portfolio_window_focal_field_work_count_all",
        "operator": "GREATER_THAN_OR_EQUAL_TO",
        "value": 1,
    }, "membership predicate")
    require(membership.get("single_minimum_history_threshold_selected") is False, "history threshold")
    require(membership.get("threshold_specific_membership_flags_materialized") is False, "history flags")
    require(membership.get("graph_team_cap_selected") is False, "graph-cap selection")
    forbidden = _mapping(config.get("forbidden_inputs_and_outputs"), "forbidden")
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden flags")
    output = _mapping(config.get("output_contract"), "output")
    require(output.get("passing_status") == STATUS, "output status")
    require(output.get("measurement_schema_version") == MEASUREMENT_SCHEMA, "measurement schema")
    require(output.get("registry_schema_version") == REGISTRY_SCHEMA, "registry schema")
    return resolved


def _output_schema(parents: Mapping[str, Mapping[str, Any]]) -> pa.Schema:
    base = parents["author_opportunity"]["schema"]
    leaveout = parents["leaveout_prior_peer"]["schema"]
    identity = parents["identity_affiliation"]["schema"]
    fields = list(base)
    fields.append(pa.field("registered_measurement_member", pa.bool_(), nullable=False))
    fields.extend(leaveout.field(name) for name in leaveout.names if name not in KEY_COLUMNS)
    fields.extend(
        identity.field(name)
        for name in identity.names
        if name not in KEY_COLUMNS and name not in RAW_DUPLICATE_PEER_COLUMNS
    )
    names = [field.name for field in fields]
    require(len(names) == len(set(names)), "duplicate consolidated column")
    require(len(fields) == 74, "consolidated schema width")
    return pa.schema(fields, metadata={b"schema_version": MEASUREMENT_SCHEMA.encode()})


def _availability_for(column: str) -> tuple[str, str]:
    if column in KEY_COLUMNS:
        return "IDENTIFIER_OPERATOR_LOCAL", "JOIN_KEY"
    if column == "registered_measurement_member":
        return "REGISTERED_MEMBERSHIP", "OUTCOME_BLIND_STRUCTURAL_MEMBERSHIP"
    if column in ORCID_COLUMNS:
        return "CURRENT_SNAPSHOT_SENSITIVITY", "ORCID_SUPPORT_OR_STABILITY_SENSITIVITY"
    if (
        "institution_comparable" in column
        or "same_affiliation" in column
    ):
        return "CURRENT_SNAPSHOT_WORK_TIME_PROXY", "AFFILIATION_COVERAGE_OR_BRIDGE_PROXY"
    return "PREDECISION_T_MINUS_1", "PREDECISION_AUTHOR_MEASUREMENT"


def build_column_registry(schema: pa.Schema) -> dict[str, Any]:
    columns = []
    counts: dict[str, int] = {}
    for field in schema:
        availability, role = _availability_for(field.name)
        counts[availability] = counts.get(availability, 0) + 1
        columns.append(
            {
                "name": field.name,
                "arrow_type": str(field.type),
                "nullable": field.nullable,
                "availability": availability,
                "role": role,
                "primary_model_input_authorized_now": False,
                "raw_numeric_identifier": field.name in KEY_COLUMNS,
            }
        )
    return {
        "schema_version": REGISTRY_SCHEMA,
        "status": "PASS_EXACT_COLUMN_COVERAGE_WITH_EXPLICIT_TEMPORAL_AVAILABILITY",
        "measurement_schema_version": MEASUREMENT_SCHEMA,
        "column_count": len(columns),
        "availability_counts": counts,
        "columns": columns,
        "current_snapshot_topic_annotation_is_historical_observation": False,
        "current_snapshot_orcid_is_historical_observation": False,
        "work_time_institution_proxy_is_historical_affiliation_truth": False,
        "primary_model_input_authorized_now": False,
        "raw_numeric_ids_repository_export_authorized": False,
    }


def _key_sql(alias: str) -> str:
    return ", ".join(f"{alias}.{_sql_identifier(column)}" for column in KEY_COLUMNS)


def _key_row_sql(alias: str) -> str:
    return f"row({_key_sql(alias)})"


def _join_sql(
    parents: Mapping[str, Mapping[str, Any]],
    output_schema: pa.Schema,
) -> str:
    base_path = _sql_path(parents["author_opportunity"]["artifact"])
    leaveout_path = _sql_path(parents["leaveout_prior_peer"]["artifact"])
    identity_path = _sql_path(parents["identity_affiliation"]["artifact"])
    base_names = set(parents["author_opportunity"]["schema"].names)
    leaveout_names = set(parents["leaveout_prior_peer"]["schema"].names)
    columns: list[str] = []
    for name in output_schema.names:
        quoted = _sql_identifier(name)
        if name in base_names:
            columns.append(f"b.{quoted} AS {quoted}")
        elif name == "registered_measurement_member":
            columns.append(f"TRUE AS {quoted}")
        elif name in leaveout_names:
            columns.append(f"l.{quoted} AS {quoted}")
        else:
            columns.append(f"r.{quoted} AS {quoted}")
    using = ", ".join(_sql_identifier(column) for column in KEY_COLUMNS)
    order = ", ".join(_sql_identifier(column) for column in KEY_COLUMNS)
    return f"""
        SELECT {", ".join(columns)}
        FROM read_parquet('{base_path}') AS b
        INNER JOIN read_parquet('{leaveout_path}') AS l USING ({using})
        INNER JOIN read_parquet('{identity_path}') AS r USING ({using})
        ORDER BY {order}
    """


def _audit_parent_relations(
    connection: duckdb.DuckDBPyConnection,
    parents: Mapping[str, Mapping[str, Any]],
    required_rows: int,
) -> dict[str, Any]:
    aliases = {
        "author_opportunity": "b",
        "leaveout_prior_peer": "l",
        "identity_affiliation": "r",
    }
    views: dict[str, str] = {}
    for label, alias in aliases.items():
        path = _sql_path(parents[label]["artifact"])
        view = f"parent_{alias}"
        connection.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{path}')")
        views[label] = view
        rows = int(_scalar(connection, f"SELECT count(*) FROM {view}"))
        unique = int(
            _scalar(
                connection,
                f"SELECT count(DISTINCT {_key_row_sql(alias)}) FROM {view} AS {alias}",
            )
        )
        require(rows == required_rows, f"{label} rows")
        require(unique == required_rows, f"{label} duplicate key")

    key_columns = ", ".join(_sql_identifier(column) for column in KEY_COLUMNS)
    base_keys = f"SELECT {key_columns} FROM {views['author_opportunity']}"
    differences: dict[str, int] = {}
    for label in ("leaveout_prior_peer", "identity_affiliation"):
        other_keys = f"SELECT {key_columns} FROM {views[label]}"
        difference = int(
            _scalar(
                connection,
                f"""
                SELECT count(*) FROM (
                    ({base_keys} EXCEPT ALL {other_keys})
                    UNION ALL
                    ({other_keys} EXCEPT ALL {base_keys})
                )
                """,
            )
        )
        require(difference == 0, f"{label} bidirectional key difference")
        differences[label] = difference

    using = ", ".join(_sql_identifier(column) for column in KEY_COLUMNS)
    mismatch_terms = " OR ".join(
        f"l.{_sql_identifier(column)} <> r.{_sql_identifier(column)}"
        for column in RAW_DUPLICATE_PEER_COLUMNS
    )
    peer_mismatches = int(
        _scalar(
            connection,
            f"""
            SELECT count(*)
            FROM {views['leaveout_prior_peer']} AS l
            INNER JOIN {views['identity_affiliation']} AS r USING ({using})
            WHERE {mismatch_terms}
            """,
        )
    )
    require(peer_mismatches == 0, "duplicated prior-peer counts differ")
    structural_nonmembers = int(
        _scalar(
            connection,
            f"""
            SELECT count(*) FROM {views['author_opportunity']}
            WHERE portfolio_window_focal_field_work_count_all < 1
            """,
        )
    )
    require(structural_nonmembers == 0, "upstream structural membership invariant")
    return {
        "component_rows": {label: required_rows for label in aliases},
        "component_unique_keys": {label: required_rows for label in aliases},
        "bidirectional_key_differences": differences,
        "duplicated_prior_peer_count_mismatches": peer_mismatches,
        "structural_membership_predicate_failures": structural_nonmembers,
    }


def _write_measurement(
    connection: duckdb.DuckDBPyConnection,
    *,
    query: str,
    schema: pa.Schema,
    path: Path,
) -> None:
    partial = path.with_name(f".{path.name}.partial")
    require(not path.exists() and not partial.exists(), "measurement output exists")
    reader = connection.execute(query).fetch_record_batch(rows_per_batch=262_144)
    writer = pq.ParquetWriter(
        partial,
        schema,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
    )
    rows = 0
    try:
        for batch in reader:
            cast = batch.cast(schema)
            writer.write_batch(cast)
            rows += cast.num_rows
    finally:
        writer.close()
    require(rows == 3_256_271, "written measurement rows")
    os.chmod(partial, 0o600)
    partial.rename(path)


def _measurement_accounting(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> dict[str, Any]:
    escaped = _sql_path(path)
    connection.execute(
        f"CREATE VIEW consolidated AS SELECT * FROM read_parquet('{escaped}')"
    )
    rows = int(_scalar(connection, "SELECT count(*) FROM consolidated"))
    members = int(
        _scalar(
            connection,
            "SELECT count(*) FROM consolidated WHERE registered_measurement_member",
        )
    )
    nonmembers = rows - members
    require(rows == 3_256_271 and members == rows and nonmembers == 0, "membership closure")
    episode_rows = connection.execute(
        """
        SELECT certified_episode_index, field_id, topic_id, entry_year, count(*)
        FROM consolidated
        GROUP BY ALL
        ORDER BY certified_episode_index
        """
    ).fetchall()
    expected = [
        (1, 17, 14430, 2003, 549629),
        (2, 17, 10579, 2012, 1215043),
        (3, 27, 10151, 1993, 1330658),
        (4, 27, 12421, 1953, 160941),
    ]
    require(episode_rows == expected, "episode accounting")
    snapshot = connection.execute(
        """
        SELECT
          count(*) FILTER (WHERE orcid_stable_single_valid_token),
          count(*) FILTER (WHERE raw_authorship_observed),
          count(*) FILTER (WHERE has_institution_comparable_prior_peer_cap_100),
          count(*) FILTER (WHERE has_same_affiliation_bridge_cap_100)
        FROM consolidated
        """
    ).fetchone()
    require(snapshot is not None, "snapshot accounting")
    return {
        "rows": rows,
        "registered_measurement_members": members,
        "registered_measurement_nonmembers": nonmembers,
        "episode_rows": [
            {
                "certified_episode_index": row[0],
                "field_id": row[1],
                "topic_id": row[2],
                "entry_year": row[3],
                "author_rows": row[4],
            }
            for row in episode_rows
        ],
        "current_snapshot_sensitivity_counts": {
            "orcid_stable_single_valid_token": snapshot[0],
            "raw_authorship_observed": snapshot[1],
            "institution_comparable_prior_peer_cap_100": snapshot[2],
            "same_affiliation_bridge_cap_100": snapshot[3],
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    accounting = _mapping(report["accounting"], "report accounting")
    snapshot = _mapping(
        accounting["current_snapshot_sensitivity_counts"], "snapshot counts"
    )
    return f"""# Certified predecision author measurement

Status: `{report['status']}`

The outcome-blind membership registry retains all
{accounting['registered_measurement_members']:,} independently verified
author-episode rows. The rule adds no minimum-history threshold: membership is
the already frozen author-opportunity base, whose upstream structural ABI
requires at least one eligible focal-field work in `t-3..t-1`.

The consolidated operator-local Parquet has {report['column_count']} columns.
Every column is registered as predecision (`t-1`), current-snapshot
sensitivity, current-snapshot work-time affiliation proxy, membership, or
operator-local identifier. No model input is authorized by this build.

Current-snapshot sensitivity counts:

- single checksum-valid ORCID token: {snapshot['orcid_stable_single_valid_token']:,}
- raw authorship observed: {snapshot['raw_authorship_observed']:,}
- institution-comparable prior peer at cap 100: {snapshot['institution_comparable_prior_peer_cap_100']:,}
- same-affiliation bridge at cap 100: {snapshot['same_affiliation_bridge_cap_100']:,}

This build reads no entry, persistence, realized-citation, peer-success, model,
or G4 rows and selects no graph cap or history threshold. Independent temporal
verification is still required before a measurement-completeness decision.
"""


def run(
    *,
    config_path: Path,
    operator_base: Path,
    output_dir: Path,
    threads: int,
) -> dict[str, Any]:
    started = time.time()
    config_path = config_path.resolve()
    operator_base = operator_base.resolve()
    output_dir = output_dir.resolve()
    config = _yaml(config_path, "config")
    parents = validate_config(
        config,
        config_path=config_path,
        operator_base=operator_base,
    )
    output_contract = _mapping(config["output_contract"], "output contract")
    require(output_dir.is_absolute(), "output directory must be absolute")
    require(output_dir.name == output_contract.get("root_basename"), "output basename")
    require(not output_dir.exists(), "output directory already exists")
    output_dir.mkdir(mode=0o700, parents=False)
    os.chmod(output_dir, 0o700)

    connection = duckdb.connect(database=":memory:")
    connection.execute(f"PRAGMA threads={max(1, threads)}")
    connection.execute(f"PRAGMA temp_directory='{_sql_path(output_dir)}'")
    required_rows = int(_mapping(config["join_contract"], "join")["required_rows"])
    parent_audit = _audit_parent_relations(connection, parents, required_rows)
    schema = _output_schema(parents)
    registry = build_column_registry(schema)

    registry_path = output_dir / "column_availability_registry.json"
    _write_json_new(registry_path, registry)
    measurement_path = output_dir / "predecision_author_measurement.parquet"
    _write_measurement(
        connection,
        query=_join_sql(parents, schema),
        schema=schema,
        path=measurement_path,
    )
    accounting = _measurement_accounting(connection, measurement_path)
    parquet = pq.ParquetFile(measurement_path)
    require(
        parquet.schema_arrow.metadata is not None
        and parquet.schema_arrow.metadata.get(b"schema_version")
        == MEASUREMENT_SCHEMA.encode(),
        "written measurement schema version",
    )
    require(parquet.metadata.num_rows == required_rows, "written measurement metadata rows")

    report = {
        "schema_version": REPORT_SCHEMA,
        "status": STATUS,
        "mode": config["mode"],
        "config_sha256": _sha256(config_path),
        "membership": {
            "rule_id": config["registered_author_membership"]["rule_id"],
            "upstream_structural_predicate": config["registered_author_membership"][
                "upstream_structural_predicate"
            ],
            "single_minimum_history_threshold_selected": False,
            "threshold_specific_membership_flags_materialized": False,
            "continuous_history_counts_retained": True,
            "graph_team_cap_profiles_retained_nonranked": [10, 50, 100],
            "graph_team_cap_selected": False,
            "final_candidate_eligibility_established": False,
        },
        "parent_audit": parent_audit,
        "accounting": accounting,
        "column_count": len(schema),
        "availability_counts": registry["availability_counts"],
        "forbidden_inputs_and_outputs": config["forbidden_inputs_and_outputs"],
        "authorization": {
            "independent_temporal_verification_complete": False,
            "measurement_completeness_decision_authorized": False,
            "production_pivot_classification_authorized": False,
            "discovery_outcome_opening_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
        },
        "runtime": {
            "gpu_used": False,
            "api_requests_created": 0,
            "elapsed_seconds": round(time.time() - started, 3),
        },
        "claim_boundary": config["claim_boundary"],
    }
    report_path = output_dir / "certified_predecision_author_measurement.json"
    _write_json_new(report_path, report)
    markdown_path = output_dir / "CERTIFIED_PREDECISION_AUTHOR_MEASUREMENT.md"
    _write_text_new(markdown_path, _markdown(report))

    artifacts: dict[str, dict[str, Any]] = {}
    for path in (markdown_path, report_path, registry_path, measurement_path):
        os.chmod(path, 0o600)
        artifacts[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": STATUS,
        "root_basename": output_dir.name,
        "config_sha256": _sha256(config_path),
        "artifact_count": len(artifacts),
        "artifact_tree_sha256": _artifact_tree_sha256(artifacts),
        "artifacts": artifacts,
        "measurement_rows": accounting["rows"],
        "registered_measurement_members": accounting[
            "registered_measurement_members"
        ],
        "measurement_columns": len(schema),
        "membership_rule_registered": True,
        "single_minimum_history_threshold_selected": False,
        "graph_team_cap_selected": False,
        "outcomes_read": False,
        "realized_citation_rows_read": False,
        "peer_visible_success_computed": False,
        "api_requests_created": 0,
        "independent_temporal_verification_complete": False,
        "measurement_completeness_decision_authorized": False,
        "production_pivot_classification_authorized": False,
        "model_fitting_authorized": False,
        "g4_open": False,
        "gpu_used": False,
    }
    receipt_path = output_dir / "certified_predecision_author_measurement.outputs.json"
    _write_json_new(receipt_path, receipt)
    os.chmod(receipt_path, 0o600)
    require(
        {path.name for path in output_dir.iterdir()} == EXPECTED_ROOT_FILES,
        "unexpected output-root files",
    )
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--operator-base", type=Path, default=Path("/ssd"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = run(
        config_path=args.config,
        operator_base=args.operator_base,
        output_dir=args.output_dir,
        threads=args.threads,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
