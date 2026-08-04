#!/usr/bin/env python3
"""Independently verify the complete private G4 feature/source package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_f_g4_private_feature_source_v1.yaml"
CONFIG_SCHEMA = "pivot_feedback_f_g4_private_feature_source.v1"
VERIFY_SCHEMA = "pivot_feedback_f_g4_private_feature_source_verification.v1"
PASS_STATUS = (
    "PASS_INDEPENDENT_COMPLETE_PRIVATE_G4_FEATURE_AND_POSTDECISION_SOURCE_"
    "NO_LABEL_NO_ANALYTIC_OPEN_NO_F_PASS"
)
EPISODE_BASENAME = re.compile(r"episode-(\d{4})\.parquet")

FEATURE_FIELDS = [
    ("episode_id", "uint16", False),
    ("author_numeric_id", "uint64", False),
    ("field_id", "uint8", False),
    ("topic_id", "uint32", False),
    ("subfield_id", "uint16", False),
    ("decision_year", "uint16", False),
    ("base_reference_count", "uint64", False),
    ("base_recent_count", "uint64", False),
    ("excluded_reference_count", "uint64", False),
    ("excluded_recent_count", "uint64", False),
    ("network_leaveout_reference_count", "uint64", False),
    ("network_leaveout_recent_count", "uint64", False),
    ("network_leaveout_surge_present", "bool", False),
    ("pre_entry_eligible_work_count", "uint32", False),
    ("pre_entry_focal_field_work_count", "uint32", False),
    ("portfolio_3y_eligible_work_count", "uint32", False),
    ("portfolio_3y_focal_field_work_count", "uint32", False),
    ("trailing_3y_target_topic_work_count", "uint32", False),
    ("trailing_3y_target_subfield_work_count", "uint32", False),
    ("first_eligible_work_year", "uint16", False),
    ("career_age_years", "uint16", False),
    ("predecision_network_degree", "uint64", False),
    ("prior_target_active_peer_count", "uint64", False),
    ("capacity_bridge_peer_count", "uint64", False),
    ("peer_visible_success_count", "uint64", False),
    ("has_prior_target_active_peer", "bool", False),
    ("has_capacity_bridge", "bool", False),
    ("has_peer_visible_success", "bool", False),
    ("complete_predecision_feature_row", "bool", False),
]

RELATION_FIELDS = [
    ("episode_id", "uint16", False),
    ("focal_author_numeric_id", "uint64", False),
    ("peer_author_numeric_id", "uint64", False),
    ("first_shared_work_year", "uint16", False),
    ("last_shared_work_year", "uint16", False),
    ("distinct_shared_eligible_work_count", "uint32", False),
    ("capacity_bridge", "bool", False),
    ("peer_visible_success", "bool", False),
    ("earliest_success_available_year", "uint16", True),
    ("canonical_success_source_work_numeric_id", "uint64", True),
    ("contributing_source", "bool", False),
]

SUCCESS_FIELDS = [
    ("work_numeric_id", "uint64", False),
    ("field_id", "uint8", False),
    ("publication_year", "uint16", False),
    ("citation_count_age_3", "uint64", False),
    ("reference_cohort_count", "uint64", False),
    ("target_count", "uint64", False),
    ("citation_cutoff_age_3", "uint64", False),
    ("public_success_proxy", "bool", False),
]

POST_FIELDS = [
    ("episode_id", "uint16", False),
    ("focal_author_numeric_id", "uint64", False),
    ("work_numeric_id", "uint64", False),
    ("publication_year", "uint16", False),
    ("event_offset", "int8", False),
    ("declared_authors_count", "uint32", False),
]

BYLINE_FIELDS = [
    ("episode_id", "uint16", False),
    ("work_numeric_id", "uint64", False),
    ("publication_year", "uint16", False),
    ("byline_author_numeric_id", "uint64", False),
]

GUARD_FIELDS = [
    ("author_numeric_id", "uint64", False),
    ("first_g4_decision_year", "uint16", False),
    ("last_g4_decision_year", "uint16", False),
    ("g4_episode_count", "uint16", False),
    ("exclude_from_discovery", "bool", False),
    ("exclude_from_g1", "bool", False),
    ("exclude_from_model_selection", "bool", False),
]

EPISODE_FIELDS = [
    ("episode_id", "uint16", False),
    ("field_id", "uint8", False),
    ("topic_id", "uint32", False),
    ("subfield_id", "uint16", False),
    ("decision_year", "uint16", False),
    ("feature_rows", "uint64", False),
    ("contributing_peer_rows", "uint64", False),
    ("postdecision_target_event_rows", "uint64", False),
    ("postdecision_target_work_byline_rows", "uint64", False),
    ("membership_count_closure", "bool", False),
    ("peer_count_closure", "bool", False),
    ("labels_materialized", "bool", False),
    ("operator_only", "bool", False),
    ("sealed", "bool", False),
]


class G4PrivateFeatureSourceVerificationError(RuntimeError):
    """Fail-closed private G4 feature/source verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G4PrivateFeatureSourceVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _safe_repo_path(relative: Any) -> Path:
    require(isinstance(relative, str) and relative, "empty repository path")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts, "unsafe repository path")
    path = (ROOT / pure).resolve(strict=True)
    path.relative_to(ROOT.resolve())
    require(path.is_file() and not path.is_symlink(), "invalid repository path")
    return path


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _sql_files(paths: Sequence[Path]) -> str:
    return "[" + ",".join(f"'{_sql_path(path)}'" for path in paths) + "]"


def _assert_schema(path: Path, expected: Sequence[tuple[str, str, bool]], version: str) -> None:
    schema = pq.ParquetFile(path).schema_arrow
    observed = [(field.name, str(field.type), field.nullable) for field in schema]
    require(observed == list(expected), f"schema fields: {path}")
    metadata = schema.metadata or {}
    require(metadata.get(b"schema_version", b"").decode() == version, f"schema version: {path}")


def _numeric_suffix(value: str, marker: str) -> int:
    require(marker in value, f"taxonomy identifier marker: {value}")
    suffix = value.rsplit(marker, 1)[1]
    require(suffix.isdigit(), f"taxonomy numeric suffix: {value}")
    return int(suffix)


def _load_taxonomy(path: Path, binding: Mapping[str, Any]) -> pa.Table:
    path = path.resolve(strict=True)
    require(path.name == binding["expected_basename"], "taxonomy basename")
    require(path.is_file() and not path.is_symlink(), "taxonomy input")
    require(_sha256(path) == binding["sha256"], "taxonomy SHA")
    table = pq.read_table(path, columns=list(binding["required_columns"]))
    require(len(table) == binding["rows"], "taxonomy rows")
    rows = [
        {
            "topic_id": _numeric_suffix(str(row["id"]), "/T"),
            "field_id": _numeric_suffix(str(row["field__id"]), "/fields/"),
            "subfield_id": _numeric_suffix(str(row["subfield__id"]), "/subfields/"),
        }
        for row in table.to_pylist()
    ]
    require(len({row["topic_id"] for row in rows}) == len(rows), "taxonomy topic uniqueness")
    schema = pa.schema(
        [
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("subfield_id", pa.uint16(), nullable=False),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _verify_repository_bindings(config: Mapping[str, Any]) -> None:
    authorization = _mapping(config["authorization_parent"], "authorization parent")
    authorization_path = _safe_repo_path(authorization["relative_path"])
    require(_sha256(authorization_path) == authorization["sha256"], "authorization SHA")
    authorization_doc = _yaml(authorization_path)
    require(authorization_doc.get("schema_version") == authorization["required_schema_version"], "authorization schema")
    require(
        authorization_doc["authorization_after_pass"][authorization["required_authorization"]]
        == authorization["required_value"],
        "feature/source authorization",
    )
    for label, binding in config["bound_design_sources"].items():
        path = _safe_repo_path(binding["relative_path"])
        require(_sha256(path) == binding["sha256"], f"{label} SHA")
        doc = _json(path) if path.suffix == ".json" else _yaml(path)
        require(doc.get("schema_version") == binding["required_schema_version"], f"{label} schema")
        if "required_status" in binding:
            require(doc.get("status") == binding["required_status"], f"{label} status")
        if label == "measurement_proxy":
            require(
                doc["precommitment_choices"]["public_success_threshold"]["rule_id"]
                == binding["required_public_success_rule"],
                "success rule",
            )
        if label == "scientific_sections":
            for section_id in binding["required_sections"]:
                require(doc["sections"][section_id]["freeze_state"] == "FROZEN_OUTCOME_BLIND", f"scientific section: {section_id}")


def _verify_membership_parent(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    root = root.resolve(strict=True)
    require(root.name == binding["expected_root_basename"], "membership root basename")
    require(root.is_dir() and not root.is_symlink(), "membership root")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "membership receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "membership receipt schema")
    require(receipt.get("status") == binding["required_status"], "membership receipt status")
    require(receipt.get("artifact_tree_sha256") == binding["artifact_tree_sha256"], "membership tree SHA")
    artifact_rows = {
        str(row["relative_path"]): dict(row)
        for row in receipt.get("artifacts", [])
        if str(row["relative_path"]).startswith(str(binding["risk_prefix"]))
    }
    risk_paths = sorted((root / str(binding["risk_prefix"])).glob("*.parquet"))
    require(len(risk_paths) == binding["risk_file_count"], "membership risk file count")
    require(sum(pq.ParquetFile(path).metadata.num_rows for path in risk_paths) == binding["risk_rows"], "membership risk rows")
    require({path.relative_to(root).as_posix() for path in risk_paths} == set(artifact_rows), "membership risk artifact set")
    for path in risk_paths:
        row = artifact_rows[path.relative_to(root).as_posix()]
        require(path.stat().st_size == row["bytes"] and _sha256(path) == row["sha256"], f"membership risk artifact: {path.name}")
    guard_path = root / str(binding["author_cold_guard_basename"])
    episode_path = root / str(binding["episode_manifest_basename"])
    require(_sha256(guard_path) == binding["author_cold_guard_sha256"], "membership guard SHA")
    require(_sha256(episode_path) == binding["episode_manifest_sha256"], "membership episode SHA")
    verification_path = Path(str(binding["verification_path"])).resolve(strict=True)
    require(_sha256(verification_path) == binding["verification_sha256"], "membership verification SHA")
    require(_json(verification_path).get("status") == binding["verification_status"], "membership verification status")
    for key in ("pass_f", "pass_g0", "g4_open", "model_fitted"):
        require(receipt.get(key) is False, f"unsafe membership state: {key}")
    return {
        "risk_paths": risk_paths,
        "guard_path": guard_path,
        "episode_path": episode_path,
        "receipt_sha256": binding["receipt_sha256"],
        "verification_sha256": binding["verification_sha256"],
    }


def _verify_source(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    root = root.resolve(strict=True)
    require(root.name == binding["expected_root_basename"], f"{label} root basename")
    require(root.is_dir() and not root.is_symlink(), f"{label} root")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], f"{label} receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], f"{label} receipt schema")
    require(receipt.get("status") == binding["required_status"], f"{label} receipt status")
    artifact_root = root / str(binding["artifact_prefix"])
    files = sorted(artifact_root.glob("*.parquet"))
    require(len(files) == binding["artifact_count"], f"{label} artifact count")
    require(all(path.is_file() and not path.is_symlink() for path in files), f"{label} artifact type")
    receipt_artifacts = {
        str(row["relative_path"]): dict(row)
        for row in receipt.get("artifacts", [])
        if str(row["relative_path"]).startswith(str(binding["artifact_prefix"]))
    }
    require({path.relative_to(root).as_posix() for path in files} == set(receipt_artifacts), f"{label} receipt artifact set")
    for path in files:
        row = receipt_artifacts[path.relative_to(root).as_posix()]
        require(path.stat().st_size == row["bytes"] and _sha256(path) == row["sha256"], f"{label} artifact: {path.name}")
    return artifact_root / "*.parquet"


def _verify_output_tree(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require(root.is_dir() and not root.is_symlink(), "output root")
    require((root.stat().st_mode & 0o777) == 0o700, "output root mode")
    directories = [path for path in root.rglob("*") if path.is_dir()]
    require(all(not path.is_symlink() and (path.stat().st_mode & 0o777) == 0o700 for path in directories), "nested directory mode")
    files = [path for path in root.rglob("*") if path.is_file()]
    require(all(not path.is_symlink() and (path.stat().st_mode & 0o777) == 0o600 for path in files), "output file mode")
    receipt_path = root / "g4_private_feature_source_materialization.outputs.json"
    require(receipt_path in files, "missing output receipt")
    receipt = _json(receipt_path)
    output = config["output_contract"]
    require(receipt.get("schema_version") == output["receipt_schema_version"], "receipt schema")
    require(receipt.get("status") == output["passing_status"], "receipt status")
    require(receipt.get("receipt_written_last") is True, "receipt written last flag")
    artifacts = {str(row["relative_path"]): dict(row) for row in receipt.get("artifacts", [])}
    actual = {path.relative_to(root).as_posix() for path in files if path != receipt_path}
    require(set(artifacts) == actual, "receipt artifact set")
    require(receipt.get("artifact_count") == len(artifacts), "receipt artifact count")
    tree = []
    for relative in sorted(artifacts):
        path = root / relative
        row = artifacts[relative]
        require(path.stat().st_size == row["bytes"], f"artifact bytes: {relative}")
        require(_sha256(path) == row["sha256"], f"artifact SHA: {relative}")
        if path.suffix == ".parquet":
            require(pq.ParquetFile(path).metadata.num_rows == row["rows"], f"artifact rows: {relative}")
        tree.append((relative, row["bytes"], row["sha256"]))
    require(hashlib.sha256(_canonical_bytes(tree)).hexdigest() == receipt["artifact_tree_sha256"], "artifact tree SHA")
    require(receipt_path.stat().st_mtime_ns > max((root / relative).stat().st_mtime_ns for relative in artifacts), "receipt not last")
    for key in (
        "g4_private_predecision_membership_complete",
        "g4_author_cold_guard_complete",
        "g4_private_predecision_features_complete",
        "g4_postdecision_target_source_complete",
        "complete_private_g4_artifact_set",
    ):
        require(receipt.get(key) is True, f"incomplete receipt state: {key}")
    for key in (
        "earlier_partition_guard_application_complete",
        "entry_or_persistence_labels_materialized",
        "holdout_preflight_complete",
        "holdout_byte_seal_complete",
        "model_score_read",
        "model_fitted",
        "g4_open",
        "pass_f",
        "pass_g0",
    ):
        require(receipt.get(key) is False, f"unsafe receipt state: {key}")
    require(receipt.get("confirmatory_or_holdout_analytic_rows_read") == 0, "analytic rows read")
    return receipt, artifacts


def _episode_paths(root: Path, directory: str, expected_count: int) -> list[Path]:
    paths = sorted((root / directory).glob("*.parquet"))
    require(len(paths) == expected_count, f"file count: {directory}")
    ids = []
    for path in paths:
        match = EPISODE_BASENAME.fullmatch(path.name)
        require(match is not None, f"episode basename: {path}")
        ids.append(int(match.group(1)))
    require(len(ids) == len(set(ids)), f"episode file IDs: {directory}")
    return paths


def _verify_schemas_and_manifest(
    root: Path,
    membership: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    output = config["output_contract"]
    feature_paths = _episode_paths(root, "private/g4_predecision_feature", output["feature_file_count"])
    relation_paths = _episode_paths(root, "private/g4_contributing_peer_source", output["relation_file_count"])
    post_paths = _episode_paths(root, "private/g4_postdecision_target_event_source", output["postdecision_file_count"])
    byline_paths = _episode_paths(root, "private/g4_postdecision_target_work_byline_source", output["byline_file_count"])
    groups = (
        (feature_paths, FEATURE_FIELDS, output["feature_schema_version"]),
        (relation_paths, RELATION_FIELDS, output["relation_schema_version"]),
        (post_paths, POST_FIELDS, output["postdecision_schema_version"]),
        (byline_paths, BYLINE_FIELDS, output["byline_schema_version"]),
    )
    for paths, fields, version in groups:
        for path in paths:
            _assert_schema(path, fields, version)
    guard_path = root / "private/g4_author_cold_guard.parquet"
    success_path = root / "private/g4_public_success_work_source.parquet"
    manifest_path = root / "private/g4_feature_source_episode_manifest.parquet"
    _assert_schema(guard_path, GUARD_FIELDS, output["guard_schema_version"])
    _assert_schema(success_path, SUCCESS_FIELDS, output["success_work_schema_version"])
    _assert_schema(manifest_path, EPISODE_FIELDS, output["episode_schema_version"])
    require(_sha256(guard_path) == _sha256(membership["guard_path"]), "guard exact copy")
    parent_rows = {int(row["episode_id"]): row for row in pq.read_table(membership["episode_path"]).to_pylist()}
    manifest_rows = pq.read_table(manifest_path).to_pylist()
    require(len(manifest_rows) == config["feature_contract"]["episode_count"], "manifest row count")
    observed_ids = {int(row["episode_id"]) for row in manifest_rows}
    require(observed_ids == set(parent_rows), "manifest parent episode set")
    for paths in (feature_paths, relation_paths, post_paths, byline_paths):
        require({int(EPISODE_BASENAME.fullmatch(path.name).group(1)) for path in paths} == observed_ids, "artifact episode set")
    path_maps = [
        {int(EPISODE_BASENAME.fullmatch(path.name).group(1)): path for path in paths}
        for paths in (feature_paths, relation_paths, post_paths, byline_paths)
    ]
    totals = {"feature_rows": 0, "contributing_peer_rows": 0, "postdecision_target_event_rows": 0, "postdecision_target_work_byline_rows": 0}
    count_keys = tuple(totals)
    by_id = {int(row["episode_id"]): row for row in manifest_rows}
    for episode_id, row in by_id.items():
        parent = parent_rows[episode_id]
        for key in ("field_id", "topic_id", "subfield_id", "decision_year"):
            require(int(row[key]) == int(parent[key]), f"manifest parent {key}: {episode_id}")
        counts = [pq.ParquetFile(mapping[episode_id]).metadata.num_rows for mapping in path_maps]
        for key, count in zip(count_keys, counts):
            require(int(row[key]) == count, f"manifest artifact count {key}: {episode_id}")
            totals[key] += count
        require(counts[0] == int(parent["materialized_risk_opportunity_count"]), f"membership count closure: {episode_id}")
        require(row["membership_count_closure"] and row["peer_count_closure"], f"manifest closure flag: {episode_id}")
        require(not row["labels_materialized"] and row["operator_only"] and not row["sealed"], f"manifest boundary: {episode_id}")
    require(totals["feature_rows"] == config["feature_contract"]["feature_rows"], "total feature rows")
    require(totals["contributing_peer_rows"] == config["feature_contract"]["contributing_peer_relation_rows"], "total relation rows")
    require(pq.ParquetFile(success_path).metadata.num_rows == config["public_success_contract"]["expected_reference_work_rows"], "success rows")
    return {
        "feature_paths": feature_paths,
        "relation_paths": relation_paths,
        "post_paths": post_paths,
        "byline_paths": byline_paths,
        "guard_path": guard_path,
        "success_path": success_path,
        "manifest_path": manifest_path,
        "manifest_rows": manifest_rows,
        **totals,
    }


def _configure_connection(con: duckdb.DuckDBPyConnection, threads: int, memory_limit_gb: int, temp_directory: Path) -> None:
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB'")
    con.execute(f"PRAGMA temp_directory='{_sql_path(temp_directory)}'")
    con.execute("PRAGMA preserve_insertion_order=false")


def _create_output_views(con: duckdb.DuckDBPyConnection, artifacts: Mapping[str, Any], membership: Mapping[str, Any]) -> None:
    con.execute(f"CREATE VIEW observed_feature AS SELECT * FROM read_parquet({_sql_files(artifacts['feature_paths'])})")
    con.execute(f"CREATE VIEW observed_relation AS SELECT * FROM read_parquet({_sql_files(artifacts['relation_paths'])})")
    con.execute(f"CREATE VIEW observed_post AS SELECT * FROM read_parquet({_sql_files(artifacts['post_paths'])})")
    con.execute(f"CREATE VIEW observed_byline AS SELECT * FROM read_parquet({_sql_files(artifacts['byline_paths'])})")
    con.execute(f"CREATE VIEW observed_success AS SELECT * FROM read_parquet('{_sql_path(artifacts['success_path'])}')")
    con.execute(f"CREATE VIEW observed_guard AS SELECT * FROM read_parquet('{_sql_path(artifacts['guard_path'])}')")
    con.execute(f"CREATE VIEW parent_risk AS SELECT * FROM read_parquet({_sql_files(membership['risk_paths'])})")


def _verify_output_row_algebra(
    con: duckdb.DuckDBPyConnection,
    artifacts: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, int]:
    duplicate_feature = int(
        con.execute(
            """
            SELECT count(*) FROM (
              SELECT episode_id,author_numeric_id,count(*) n
              FROM observed_feature GROUP BY episode_id,author_numeric_id HAVING n<>1
            )
            """
        ).fetchone()[0]
    )
    require(duplicate_feature == 0, "duplicate feature key")
    parent_mismatch = int(
        con.execute(
            """
            SELECT count(*) FROM observed_feature f FULL OUTER JOIN parent_risk r
              ON f.episode_id=r.episode_id AND f.author_numeric_id=r.author_numeric_id
            WHERE f.episode_id IS NULL OR r.episode_id IS NULL
               OR f.field_id IS DISTINCT FROM r.field_id
               OR f.topic_id IS DISTINCT FROM r.topic_id
               OR f.subfield_id IS DISTINCT FROM r.subfield_id
               OR f.decision_year IS DISTINCT FROM r.decision_year
               OR f.base_reference_count IS DISTINCT FROM r.base_reference_count
               OR f.base_recent_count IS DISTINCT FROM r.base_recent_count
               OR f.excluded_reference_count IS DISTINCT FROM r.excluded_reference_count
               OR f.excluded_recent_count IS DISTINCT FROM r.excluded_recent_count
               OR f.network_leaveout_reference_count IS DISTINCT FROM r.network_leaveout_reference_count
               OR f.network_leaveout_recent_count IS DISTINCT FROM r.network_leaveout_recent_count
               OR f.network_leaveout_surge_present IS DISTINCT FROM r.network_leaveout_surge_present
               OR f.prior_target_active_peer_count IS DISTINCT FROM r.prior_target_active_peer_count
            """
        ).fetchone()[0]
    )
    require(parent_mismatch == 0, "feature versus parent risk mismatch")
    bad_feature = int(
        con.execute(
            """
            SELECT count(*) FROM observed_feature
            WHERE pre_entry_eligible_work_count<portfolio_3y_eligible_work_count
               OR pre_entry_focal_field_work_count<portfolio_3y_focal_field_work_count
               OR portfolio_3y_focal_field_work_count=0
               OR trailing_3y_target_topic_work_count<>0
               OR first_eligible_work_year>=decision_year
               OR career_age_years<>decision_year-first_eligible_work_year
               OR capacity_bridge_peer_count>prior_target_active_peer_count
               OR peer_visible_success_count>prior_target_active_peer_count
               OR has_prior_target_active_peer<>(prior_target_active_peer_count>0)
               OR has_capacity_bridge<>(capacity_bridge_peer_count>0)
               OR has_peer_visible_success<>(peer_visible_success_count>0)
               OR NOT complete_predecision_feature_row
            """
        ).fetchone()[0]
    )
    require(bad_feature == 0, "feature row algebra")
    duplicate_relation = int(
        con.execute(
            """
            SELECT count(*) FROM (
              SELECT episode_id,focal_author_numeric_id,peer_author_numeric_id,count(*) n
              FROM observed_relation
              GROUP BY episode_id,focal_author_numeric_id,peer_author_numeric_id HAVING n<>1
            )
            """
        ).fetchone()[0]
    )
    require(duplicate_relation == 0, "duplicate relation key")
    bad_relation = int(
        con.execute(
            """
            SELECT count(*) FROM observed_relation r
            LEFT JOIN observed_feature f
              ON f.episode_id=r.episode_id AND f.author_numeric_id=r.focal_author_numeric_id
            WHERE f.episode_id IS NULL OR r.focal_author_numeric_id=r.peer_author_numeric_id
               OR r.first_shared_work_year>r.last_shared_work_year
               OR r.first_shared_work_year<f.decision_year-5
               OR r.last_shared_work_year>f.decision_year-1
               OR r.distinct_shared_eligible_work_count<1
               OR r.capacity_bridge<>(r.last_shared_work_year>=f.decision_year-3)
               OR r.peer_visible_success<>(r.earliest_success_available_year IS NOT NULL)
               OR r.peer_visible_success<>(r.canonical_success_source_work_numeric_id IS NOT NULL)
               OR (r.peer_visible_success AND r.earliest_success_available_year>f.decision_year-1)
               OR NOT r.contributing_source
            """
        ).fetchone()[0]
    )
    require(bad_relation == 0, "relation row algebra")
    relation_count_mismatch = int(
        con.execute(
            """
            WITH counts AS (
              SELECT episode_id,focal_author_numeric_id,
                     count(*) n,
                     count(*) FILTER (WHERE capacity_bridge) capacity_n,
                     count(*) FILTER (WHERE peer_visible_success) success_n
              FROM observed_relation GROUP BY episode_id,focal_author_numeric_id
            )
            SELECT count(*) FROM observed_feature f LEFT JOIN counts c
              ON c.episode_id=f.episode_id AND c.focal_author_numeric_id=f.author_numeric_id
            WHERE f.prior_target_active_peer_count<>coalesce(c.n,0)
               OR f.capacity_bridge_peer_count<>coalesce(c.capacity_n,0)
               OR f.peer_visible_success_count<>coalesce(c.success_n,0)
            """
        ).fetchone()[0]
    )
    require(relation_count_mismatch == 0, "relation-to-feature count mismatch")
    duplicate_success = int(
        con.execute("SELECT count(*)-count(DISTINCT work_numeric_id) FROM observed_success").fetchone()[0]
    )
    require(duplicate_success == 0, "duplicate success work")
    bad_success = int(
        con.execute(
            """
            WITH grouped AS (
              SELECT publication_year,count(*) n,min(reference_cohort_count) min_n,
                     max(reference_cohort_count) max_n,min(target_count) min_target,
                     max(target_count) max_target,min(citation_cutoff_age_3) min_cutoff,
                     max(citation_cutoff_age_3) max_cutoff,
                     min(citation_count_age_3) FILTER (WHERE public_success_proxy) min_positive,
                     max(citation_count_age_3) FILTER (WHERE NOT public_success_proxy) max_negative
              FROM observed_success GROUP BY publication_year
            )
            SELECT count(*) FROM grouped
            WHERE n<10 OR min_n<>n OR max_n<>n OR min_n<>max_n
               OR min_target<>ceil(n*0.10) OR max_target<>ceil(n*0.10)
               OR min_target<>max_target OR min_cutoff<>max_cutoff
               OR min_positive<min_cutoff
               OR (max_negative IS NOT NULL AND max_negative>=min_cutoff)
            """
        ).fetchone()[0]
    )
    require(bad_success == 0, "success cohort algebra")
    cutoff_mismatch = int(
        con.execute(
            """
            WITH ranked AS (
              SELECT publication_year,citation_count_age_3,
                     row_number() OVER (
                       PARTITION BY publication_year
                       ORDER BY citation_count_age_3 DESC,work_numeric_id
                     ) rank,
                     max(target_count) OVER (PARTITION BY publication_year) target_count,
                     max(citation_cutoff_age_3) OVER (PARTITION BY publication_year) registered_cutoff
              FROM observed_success
            )
            SELECT count(*) FROM ranked
            WHERE rank=target_count AND citation_count_age_3<>registered_cutoff
            """
        ).fetchone()[0]
    )
    require(cutoff_mismatch == 0, "success target-rank cutoff")
    duplicate_post = int(
        con.execute(
            """
            SELECT count(*) FROM (
              SELECT episode_id,focal_author_numeric_id,work_numeric_id,count(*) n
              FROM observed_post GROUP BY episode_id,focal_author_numeric_id,work_numeric_id HAVING n<>1
            )
            """
        ).fetchone()[0]
    )
    require(duplicate_post == 0, "duplicate postdecision key")
    bad_post = int(
        con.execute(
            """
            SELECT count(*) FROM observed_post p LEFT JOIN observed_feature f
              ON f.episode_id=p.episode_id AND f.author_numeric_id=p.focal_author_numeric_id
            WHERE f.episode_id IS NULL OR p.publication_year-f.decision_year<>p.event_offset
               OR p.event_offset<0 OR p.event_offset>3 OR p.declared_authors_count<1
            """
        ).fetchone()[0]
    )
    require(bad_post == 0, "postdecision row algebra")
    duplicate_byline = int(
        con.execute(
            """
            SELECT count(*) FROM (
              SELECT episode_id,work_numeric_id,byline_author_numeric_id,count(*) n
              FROM observed_byline GROUP BY episode_id,work_numeric_id,byline_author_numeric_id HAVING n<>1
            )
            """
        ).fetchone()[0]
    )
    require(duplicate_byline == 0, "duplicate byline key")
    orphan_byline = int(
        con.execute(
            """
            SELECT count(*) FROM observed_byline b ANTI JOIN (
              SELECT DISTINCT episode_id,work_numeric_id,publication_year FROM observed_post
            ) p USING (episode_id,work_numeric_id,publication_year)
            """
        ).fetchone()[0]
    )
    require(orphan_byline == 0, "orphan byline row")
    missing_focal_byline = int(
        con.execute(
            """
            SELECT count(*) FROM observed_post p ANTI JOIN observed_byline b
              ON b.episode_id=p.episode_id AND b.work_numeric_id=p.work_numeric_id
             AND b.byline_author_numeric_id=p.focal_author_numeric_id
            """
        ).fetchone()[0]
    )
    require(missing_focal_byline == 0, "missing focal byline")
    forbidden = {
        "entry_label",
        "persistence_label",
        "outcome_state",
        "descriptive_rate",
        "model_score",
        "predicted_probability",
        "treatment_effect",
    }
    for paths in (
        artifacts["feature_paths"],
        artifacts["relation_paths"],
        artifacts["post_paths"],
        artifacts["byline_paths"],
        [artifacts["success_path"], artifacts["guard_path"], artifacts["manifest_path"]],
    ):
        for path in paths:
            require(not (forbidden & set(pq.ParquetFile(path).schema_arrow.names)), f"forbidden column: {path}")
    return {
        "feature_key_duplicates": duplicate_feature,
        "parent_risk_mismatch_rows": parent_mismatch,
        "feature_algebra_failures": bad_feature,
        "relation_key_duplicates": duplicate_relation,
        "relation_algebra_failures": bad_relation,
        "relation_count_mismatch_rows": relation_count_mismatch,
        "success_key_duplicates": duplicate_success,
        "success_cohort_failures": bad_success,
        "success_cutoff_failures": cutoff_mismatch,
        "post_key_duplicates": duplicate_post,
        "post_algebra_failures": bad_post,
        "byline_key_duplicates": duplicate_byline,
        "orphan_byline_rows": orphan_byline,
        "missing_focal_byline_rows": missing_focal_byline,
    }


def _full_outer_mismatch(con: duckdb.DuckDBPyConnection, query: str, label: str) -> int:
    value = int(con.execute(query).fetchone()[0])
    require(value == 0, label)
    return value


def _verify_source_replay(
    con: duckdb.DuckDBPyConnection,
    *,
    work_glob: Path,
    event_glob: Path,
    citation_glob: Path,
    taxonomy: pa.Table,
    config: Mapping[str, Any],
) -> dict[str, int]:
    con.register("taxonomy_input", taxonomy)
    con.execute("CREATE TABLE verify_taxonomy AS SELECT * FROM taxonomy_input")
    con.execute(
        """
        CREATE TABLE verify_episode AS
        SELECT DISTINCT episode_id,field_id,topic_id,subfield_id,decision_year
        FROM observed_feature
        """
    )
    work = _sql_path(work_glob)
    event = _sql_path(event_glob)
    citation = _sql_path(citation_glob)
    eligible = """
        e.temporal_disposition IN (2,3)
        AND w.publication_year=e.publication_year
        AND w.primary_field_id=e.primary_field_id
        AND w.temporal_disposition=e.temporal_disposition
        AND w.primary_topic_id IS NOT NULL
        AND w.work_type_code IN (1,2,3,4,7,16,18,21)
        AND NOT w.is_paratext
        AND coalesce(w.is_retracted,false)=false
    """
    con.execute(
        f"""
        CREATE TABLE verify_guard_event AS
        SELECT DISTINCT e.author_numeric_id AS author_id,e.work_numeric_id AS work_id,
               e.publication_year,e.declared_authors_count,
               e.primary_field_id AS field_id,w.primary_topic_id AS topic_id,
               tx.subfield_id
        FROM read_parquet('{event}') e
        JOIN observed_guard g ON g.author_numeric_id=e.author_numeric_id
        JOIN read_parquet('{work}') w ON w.work_id_numeric=e.work_numeric_id
        JOIN verify_taxonomy tx ON tx.topic_id=w.primary_topic_id AND tx.field_id=w.primary_field_id
        WHERE e.publication_year<=2025 AND {eligible}
        """
    )
    con.execute("CREATE INDEX verify_guard_event_author ON verify_guard_event(author_id,publication_year)")
    con.execute(
        f"""
        CREATE TABLE verify_recent_event AS
        SELECT DISTINCT e.author_numeric_id AS author_id,e.work_numeric_id AS work_id,e.publication_year
        FROM read_parquet('{event}') e
        JOIN read_parquet('{work}') w ON w.work_id_numeric=e.work_numeric_id
        WHERE e.publication_year BETWEEN 2015 AND 2021
          AND e.primary_field_id BETWEEN 11 AND 36 AND {eligible}
        """
    )
    con.execute("CREATE INDEX verify_recent_event_author ON verify_recent_event(author_id,publication_year)")
    con.execute("CREATE INDEX verify_recent_event_work ON verify_recent_event(work_id)")
    con.execute(
        f"""
        CREATE TABLE verify_target_event AS
        SELECT DISTINCT e.author_numeric_id AS author_id,e.work_numeric_id AS work_id,
               e.publication_year,e.declared_authors_count,
               e.primary_field_id AS field_id,w.primary_topic_id AS topic_id
        FROM read_parquet('{event}') e
        JOIN read_parquet('{work}') w ON w.work_id_numeric=e.work_numeric_id
        JOIN (SELECT DISTINCT field_id,topic_id FROM verify_episode) p
          ON p.field_id=e.primary_field_id AND p.topic_id=w.primary_topic_id
        WHERE e.publication_year<=2025 AND {eligible}
        """
    )
    con.execute("CREATE INDEX verify_target_event_topic_author ON verify_target_event(field_id,topic_id,author_id)")
    con.execute("CREATE INDEX verify_target_event_work ON verify_target_event(work_id)")
    con.execute(
        """
        CREATE TABLE verify_author_year AS
        SELECT DISTINCT author_numeric_id AS author_id,decision_year FROM observed_feature
        """
    )
    con.execute(
        """
        CREATE TABLE verify_author_feature AS
        SELECT ay.author_id,ay.decision_year,
               count(DISTINCT ge.work_id) FILTER (WHERE ge.publication_year<=ay.decision_year-1) pre_entry_count,
               count(DISTINCT ge.work_id) FILTER (
                 WHERE ge.publication_year<=ay.decision_year-1 AND ge.field_id=27
               ) pre_entry_focal_count,
               count(DISTINCT ge.work_id) FILTER (
                 WHERE ge.publication_year BETWEEN ay.decision_year-3 AND ay.decision_year-1
               ) portfolio_count,
               count(DISTINCT ge.work_id) FILTER (
                 WHERE ge.publication_year BETWEEN ay.decision_year-3 AND ay.decision_year-1 AND ge.field_id=27
               ) portfolio_focal_count,
               min(ge.publication_year) FILTER (WHERE ge.publication_year<=ay.decision_year-1) first_year
        FROM verify_author_year ay JOIN verify_guard_event ge ON ge.author_id=ay.author_id
        WHERE ge.publication_year<=ay.decision_year-1
        GROUP BY ay.author_id,ay.decision_year
        """
    )
    con.execute("CREATE INDEX verify_author_feature_key ON verify_author_feature(author_id,decision_year)")
    con.execute(
        """
        CREATE TABLE verify_author_subfield_year AS
        SELECT author_id,publication_year,subfield_id,count(DISTINCT work_id) work_count
        FROM verify_guard_event GROUP BY author_id,publication_year,subfield_id
        """
    )
    con.execute(
        """
        CREATE TABLE verify_risk_subfield AS
        SELECT f.episode_id,f.author_numeric_id,coalesce(sum(s.work_count),0) trailing_subfield_count
        FROM observed_feature f LEFT JOIN verify_author_subfield_year s
          ON s.author_id=f.author_numeric_id AND s.subfield_id=f.subfield_id
         AND s.publication_year BETWEEN f.decision_year-3 AND f.decision_year-1
        GROUP BY f.episode_id,f.author_numeric_id
        """
    )
    con.execute("CREATE INDEX verify_risk_subfield_key ON verify_risk_subfield(episode_id,author_numeric_id)")
    con.execute(
        """
        CREATE TABLE verify_shared_work AS
        SELECT DISTINCT ay.author_id AS focal_author_id,ay.decision_year,re.work_id,re.publication_year
        FROM verify_author_year ay JOIN verify_recent_event re ON re.author_id=ay.author_id
        WHERE re.publication_year BETWEEN ay.decision_year-5 AND ay.decision_year-1
        """
    )
    con.execute("CREATE INDEX verify_shared_work_id ON verify_shared_work(work_id)")
    con.execute(
        """
        CREATE TABLE verify_prior_peer AS
        SELECT f.focal_author_id,f.decision_year,co.author_id AS peer_author_id,
               min(f.publication_year) first_shared_year,max(f.publication_year) last_shared_year,
               count(DISTINCT f.work_id) shared_work_count
        FROM verify_shared_work f JOIN verify_recent_event co ON co.work_id=f.work_id
        WHERE co.author_id<>f.focal_author_id
        GROUP BY f.focal_author_id,f.decision_year,co.author_id
        """
    )
    con.execute("CREATE INDEX verify_prior_peer_focal ON verify_prior_peer(focal_author_id,decision_year)")
    con.execute(
        """
        CREATE TABLE verify_network_feature AS
        SELECT focal_author_id,decision_year,count(*) network_degree
        FROM verify_prior_peer GROUP BY focal_author_id,decision_year
        """
    )
    con.execute("CREATE INDEX verify_network_key ON verify_network_feature(focal_author_id,decision_year)")
    con.execute(
        """
        CREATE TABLE verify_target_first AS
        SELECT ep.episode_id,te.author_id,min(te.publication_year) first_target_year
        FROM verify_episode ep JOIN verify_target_event te
          ON te.field_id=ep.field_id AND te.topic_id=ep.topic_id
        WHERE te.publication_year<=ep.decision_year-1
        GROUP BY ep.episode_id,te.author_id
        """
    )
    con.execute("CREATE INDEX verify_target_first_key ON verify_target_first(episode_id,author_id)")
    con.execute(
        """
        CREATE TABLE verify_relation_base AS
        SELECT f.episode_id,f.author_numeric_id AS focal_author_id,p.peer_author_id,
               p.first_shared_year,p.last_shared_year,p.shared_work_count,
               p.last_shared_year>=f.decision_year-3 AS capacity_bridge
        FROM observed_feature f
        JOIN verify_prior_peer p
          ON p.focal_author_id=f.author_numeric_id AND p.decision_year=f.decision_year
        JOIN verify_target_first tf
          ON tf.episode_id=f.episode_id AND tf.author_id=p.peer_author_id
        """
    )
    relation_base_rows = int(con.execute("SELECT count(*) FROM verify_relation_base").fetchone()[0])
    require(relation_base_rows == config["feature_contract"]["contributing_peer_relation_rows"], "source replay relation rows")
    con.execute(
        f"""
        CREATE TABLE verify_field_work AS
        SELECT w.work_id_numeric AS work_id,w.primary_field_id AS field_id,w.publication_year
        FROM read_parquet('{work}') w
        WHERE w.primary_field_id=27 AND w.publication_year<=2018
          AND w.publication_year IS NOT NULL AND w.temporal_disposition IN (2,3)
          AND w.work_type_code IN (1,2,3,4,7,16,18,21)
          AND NOT w.is_paratext AND coalesce(w.is_retracted,false)=false
        """
    )
    field_work_rows = int(con.execute("SELECT count(*) FROM verify_field_work").fetchone()[0])
    require(field_work_rows == config["public_success_contract"]["expected_reference_work_rows"], "source replay field work rows")
    con.execute("CREATE INDEX verify_field_work_id ON verify_field_work(work_id)")
    con.execute(
        f"""
        CREATE TABLE verify_field_citation AS
        SELECT fw.work_id,fw.field_id,fw.publication_year,count(c.citing_work_id) citation_count_age_3
        FROM verify_field_work fw LEFT JOIN read_parquet('{citation}') c
          ON c.cited_work_id=fw.work_id
         AND c.available_at<make_date(fw.publication_year+4,1,1)
        GROUP BY fw.work_id,fw.field_id,fw.publication_year
        """
    )
    con.execute(
        """
        CREATE TABLE verify_success AS
        WITH ranked AS (
          SELECT *,row_number() OVER (
                   PARTITION BY field_id,publication_year
                   ORDER BY citation_count_age_3 DESC,work_id
                 ) citation_rank,
                 count(*) OVER (PARTITION BY field_id,publication_year) cohort_count
          FROM verify_field_citation
        ), cutoffs AS (
          SELECT field_id,publication_year,max(cohort_count) cohort_count,
                 ceil(max(cohort_count)*0.10)::UBIGINT target_count,
                 max(citation_count_age_3) FILTER (
                   WHERE citation_rank=ceil(cohort_count*0.10)
                 ) cutoff
          FROM ranked GROUP BY field_id,publication_year
        )
        SELECT r.work_id,r.field_id,r.publication_year,r.citation_count_age_3,
               c.cohort_count,c.target_count,c.cutoff,
               r.citation_count_age_3>=c.cutoff public_success_proxy
        FROM ranked r JOIN cutoffs c USING (field_id,publication_year)
        """
    )
    con.execute("CREATE INDEX verify_success_id ON verify_success(work_id)")
    success_mismatch = _full_outer_mismatch(
        con,
        """
        SELECT count(*) FROM verify_success e FULL OUTER JOIN observed_success o
          ON o.work_numeric_id=e.work_id
        WHERE e.work_id IS NULL OR o.work_numeric_id IS NULL
           OR o.field_id IS DISTINCT FROM e.field_id
           OR o.publication_year IS DISTINCT FROM e.publication_year
           OR o.citation_count_age_3 IS DISTINCT FROM e.citation_count_age_3
           OR o.reference_cohort_count IS DISTINCT FROM e.cohort_count
           OR o.target_count IS DISTINCT FROM e.target_count
           OR o.citation_cutoff_age_3 IS DISTINCT FROM e.cutoff
           OR o.public_success_proxy IS DISTINCT FROM e.public_success_proxy
        """,
        "source replay public-success mismatch",
    )
    con.execute(
        """
        CREATE TABLE verify_peer_success_ranked AS
        SELECT ep.episode_id,te.author_id,te.publication_year+3 AS available_year,
               te.work_id,row_number() OVER (
                 PARTITION BY ep.episode_id,te.author_id
                 ORDER BY te.publication_year+3,te.author_id,te.work_id
               ) source_order
        FROM verify_episode ep JOIN verify_target_event te
          ON te.field_id=ep.field_id AND te.topic_id=ep.topic_id
        JOIN verify_success ps ON ps.work_id=te.work_id AND ps.public_success_proxy
        WHERE te.publication_year+3<=ep.decision_year-1
        """
    )
    con.execute(
        """
        CREATE TABLE verify_peer_success AS
        SELECT episode_id,author_id,min(available_year) earliest_available_year,
               max(work_id) FILTER (WHERE source_order=1) canonical_source_work_id
        FROM verify_peer_success_ranked GROUP BY episode_id,author_id
        """
    )
    con.execute("CREATE INDEX verify_peer_success_key ON verify_peer_success(episode_id,author_id)")
    con.execute(
        """
        CREATE TABLE verify_relation AS
        SELECT b.episode_id,b.focal_author_id,b.peer_author_id,b.first_shared_year,
               b.last_shared_year,b.shared_work_count,b.capacity_bridge,
               ps.author_id IS NOT NULL peer_visible_success,
               ps.earliest_available_year,ps.canonical_source_work_id
        FROM verify_relation_base b LEFT JOIN verify_peer_success ps
          ON ps.episode_id=b.episode_id AND ps.author_id=b.peer_author_id
        """
    )
    relation_mismatch = _full_outer_mismatch(
        con,
        """
        SELECT count(*) FROM verify_relation e FULL OUTER JOIN observed_relation o
          ON o.episode_id=e.episode_id
         AND o.focal_author_numeric_id=e.focal_author_id
         AND o.peer_author_numeric_id=e.peer_author_id
        WHERE e.episode_id IS NULL OR o.episode_id IS NULL
           OR o.first_shared_work_year IS DISTINCT FROM e.first_shared_year
           OR o.last_shared_work_year IS DISTINCT FROM e.last_shared_year
           OR o.distinct_shared_eligible_work_count IS DISTINCT FROM e.shared_work_count
           OR o.capacity_bridge IS DISTINCT FROM e.capacity_bridge
           OR o.peer_visible_success IS DISTINCT FROM e.peer_visible_success
           OR o.earliest_success_available_year IS DISTINCT FROM e.earliest_available_year
           OR o.canonical_success_source_work_numeric_id IS DISTINCT FROM e.canonical_source_work_id
           OR NOT o.contributing_source
        """,
        "source replay relation mismatch",
    )
    con.execute(
        """
        CREATE TABLE verify_peer_feature AS
        SELECT episode_id,focal_author_id,count(*) active_count,
               count(*) FILTER (WHERE capacity_bridge) capacity_count,
               count(*) FILTER (WHERE peer_visible_success) visible_success_count
        FROM verify_relation GROUP BY episode_id,focal_author_id
        """
    )
    feature_mismatch = _full_outer_mismatch(
        con,
        """
        SELECT count(*) FROM observed_feature f
        LEFT JOIN verify_author_feature h
          ON h.author_id=f.author_numeric_id AND h.decision_year=f.decision_year
        LEFT JOIN verify_risk_subfield sf
          ON sf.episode_id=f.episode_id AND sf.author_numeric_id=f.author_numeric_id
        LEFT JOIN verify_network_feature n
          ON n.focal_author_id=f.author_numeric_id AND n.decision_year=f.decision_year
        LEFT JOIN verify_peer_feature p
          ON p.episode_id=f.episode_id AND p.focal_author_id=f.author_numeric_id
        WHERE h.author_id IS NULL OR sf.episode_id IS NULL
           OR f.pre_entry_eligible_work_count IS DISTINCT FROM h.pre_entry_count
           OR f.pre_entry_focal_field_work_count IS DISTINCT FROM h.pre_entry_focal_count
           OR f.portfolio_3y_eligible_work_count IS DISTINCT FROM h.portfolio_count
           OR f.portfolio_3y_focal_field_work_count IS DISTINCT FROM h.portfolio_focal_count
           OR f.trailing_3y_target_topic_work_count<>0
           OR f.trailing_3y_target_subfield_work_count IS DISTINCT FROM sf.trailing_subfield_count
           OR f.first_eligible_work_year IS DISTINCT FROM h.first_year
           OR f.career_age_years IS DISTINCT FROM f.decision_year-h.first_year
           OR f.predecision_network_degree IS DISTINCT FROM coalesce(n.network_degree,0)
           OR f.prior_target_active_peer_count IS DISTINCT FROM coalesce(p.active_count,0)
           OR f.capacity_bridge_peer_count IS DISTINCT FROM coalesce(p.capacity_count,0)
           OR f.peer_visible_success_count IS DISTINCT FROM coalesce(p.visible_success_count,0)
        """,
        "source replay feature mismatch",
    )
    con.execute(
        """
        CREATE TABLE verify_post AS
        SELECT f.episode_id,f.author_numeric_id AS focal_author_id,te.work_id,
               te.publication_year,(te.publication_year-f.decision_year)::TINYINT event_offset,
               te.declared_authors_count
        FROM observed_feature f JOIN verify_target_event te
          ON te.author_id=f.author_numeric_id AND te.field_id=f.field_id AND te.topic_id=f.topic_id
        WHERE te.publication_year BETWEEN f.decision_year AND f.decision_year+3
        """
    )
    con.execute("CREATE INDEX verify_post_work ON verify_post(episode_id,work_id)")
    post_mismatch = _full_outer_mismatch(
        con,
        """
        SELECT count(*) FROM verify_post e FULL OUTER JOIN observed_post o
          ON o.episode_id=e.episode_id
         AND o.focal_author_numeric_id=e.focal_author_id
         AND o.work_numeric_id=e.work_id
        WHERE e.episode_id IS NULL OR o.episode_id IS NULL
           OR o.publication_year IS DISTINCT FROM e.publication_year
           OR o.event_offset IS DISTINCT FROM e.event_offset
           OR o.declared_authors_count IS DISTINCT FROM e.declared_authors_count
        """,
        "source replay postdecision mismatch",
    )
    con.execute(
        """
        CREATE TABLE verify_byline AS
        SELECT DISTINCT p.episode_id,p.work_id,p.publication_year,te.author_id AS byline_author_id
        FROM (SELECT DISTINCT episode_id,work_id,publication_year FROM verify_post) p
        JOIN verify_target_event te ON te.work_id=p.work_id AND te.publication_year=p.publication_year
        """
    )
    byline_mismatch = _full_outer_mismatch(
        con,
        """
        SELECT count(*) FROM verify_byline e FULL OUTER JOIN observed_byline o
          ON o.episode_id=e.episode_id AND o.work_numeric_id=e.work_id
         AND o.byline_author_numeric_id=e.byline_author_id
        WHERE e.episode_id IS NULL OR o.episode_id IS NULL
           OR o.publication_year IS DISTINCT FROM e.publication_year
        """,
        "source replay byline mismatch",
    )
    return {
        "guard_event_rows_replayed": int(con.execute("SELECT count(*) FROM verify_guard_event").fetchone()[0]),
        "recent_event_rows_replayed": int(con.execute("SELECT count(*) FROM verify_recent_event").fetchone()[0]),
        "target_event_rows_replayed": int(con.execute("SELECT count(*) FROM verify_target_event").fetchone()[0]),
        "risk_author_year_rows_replayed": int(con.execute("SELECT count(*) FROM verify_author_year").fetchone()[0]),
        "prior_peer_year_rows_replayed": int(con.execute("SELECT count(*) FROM verify_prior_peer").fetchone()[0]),
        "relation_rows_replayed": relation_base_rows,
        "public_success_reference_work_rows_replayed": field_work_rows,
        "public_success_mismatch_rows": success_mismatch,
        "feature_mismatch_rows": feature_mismatch,
        "relation_mismatch_rows": relation_mismatch,
        "postdecision_mismatch_rows": post_mismatch,
        "byline_mismatch_rows": byline_mismatch,
    }


def _verify_report(
    *,
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    membership: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    source_replay: Mapping[str, int],
    con: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    report = _json(root / "g4_private_feature_source_materialization.json")
    output = config["output_contract"]
    require(report.get("schema_version") == output["report_schema_version"], "report schema")
    require(report.get("status") == receipt["status"], "report status")
    execution = report["execution"]
    require(execution["config_relative_path"] == config_path.relative_to(ROOT).as_posix(), "executed config path")
    require(execution["config_sha256"] == _sha256(config_path), "executed config SHA")
    builder = _safe_repo_path(execution["builder_relative_path"])
    require(_sha256(builder) == execution["builder_sha256"], "executed builder SHA")
    require(report["parent_bindings"]["membership_receipt_sha256"] == membership["receipt_sha256"], "report membership receipt")
    require(report["parent_bindings"]["membership_verification_sha256"] == membership["verification_sha256"], "report membership verification")
    for label, key in (
        ("cutoff_work_metadata", "cutoff_receipt_sha256"),
        ("author_history_events", "author_history_receipt_sha256"),
        ("citation_chronology", "citation_receipt_sha256"),
    ):
        require(report["normalized_input_bindings"][key] == config["normalized_inputs"][label]["receipt_sha256"], f"report source binding: {label}")
    taxonomy_binding = report["normalized_input_bindings"]["taxonomy"]
    require(taxonomy_binding["sha256"] == config["normalized_inputs"]["topic_taxonomy_projection"]["sha256"], "report taxonomy SHA")
    require(int(taxonomy_binding["rows"]) == config["normalized_inputs"]["topic_taxonomy_projection"]["rows"], "report taxonomy rows")
    accounting = report["accounting"]
    expected_accounting = {
        "risk_rows": artifacts["feature_rows"],
        "guard_event_rows": source_replay["guard_event_rows_replayed"],
        "recent_event_rows": source_replay["recent_event_rows_replayed"],
        "target_event_rows": source_replay["target_event_rows_replayed"],
        "risk_author_year_rows": source_replay["risk_author_year_rows_replayed"],
        "prior_peer_year_rows": source_replay["prior_peer_year_rows_replayed"],
        "contributing_peer_rows": artifacts["contributing_peer_rows"],
        "public_success_reference_work_rows": source_replay["public_success_reference_work_rows_replayed"],
        "public_success_work_rows": config["public_success_contract"]["expected_reference_work_rows"],
        "feature_rows": artifacts["feature_rows"],
        "postdecision_target_event_rows": artifacts["postdecision_target_event_rows"],
        "postdecision_target_work_byline_rows": artifacts["postdecision_target_work_byline_rows"],
    }
    for key, expected in expected_accounting.items():
        require(int(accounting[key]) == int(expected), f"report accounting: {key}")
    positive_success = int(con.execute("SELECT count(*) FROM observed_success WHERE public_success_proxy").fetchone()[0])
    require(int(accounting["public_success_positive_work_rows"]) == positive_success, "report success positives")
    require(int(accounting["peer_count_mismatch_rows"]) == 0, "report peer mismatch")
    require(int(accounting["bad_feature_rows"]) == 0, "report bad features")
    require(accounting["postdecision_offsets_read"] == [0, 1, 2, 3], "report postdecision offsets")
    require(accounting["labels_materialized"] is False, "report labels")
    require(int(accounting["analytic_outcome_rows_read"]) == 0, "report analytic rows")
    report_manifest = {int(row["episode_id"]): row for row in report["episode_manifest"]}
    parquet_manifest = {int(row["episode_id"]): row for row in artifacts["manifest_rows"]}
    require(report_manifest == parquet_manifest, "report manifest equality")
    completion = report["completion"]
    for key in (
        "g4_private_predecision_membership_complete",
        "g4_author_cold_guard_complete",
        "g4_private_predecision_features_complete",
        "g4_postdecision_target_source_complete",
        "complete_private_g4_artifact_set",
    ):
        require(completion.get(key) is True, f"incomplete report state: {key}")
    for key in (
        "earlier_partition_guard_application_complete",
        "entry_or_persistence_labels_materialized",
        "holdout_preflight_complete",
        "holdout_byte_seal_complete",
        "model_score_read",
        "model_fitted",
        "g4_open",
        "pass_f",
        "pass_g0",
    ):
        require(completion.get(key) is False, f"unsafe report state: {key}")
    require(completion.get("confirmatory_or_holdout_analytic_rows_read") == 0, "report confirmatory rows")
    authorization = report["authorization"]
    require(
        authorization["private_source_root_and_complete_explicit_artifact_list_binding"] == "AUTHORIZED_NEXT",
        "next authorization",
    )
    require(
        authorization["explicit_file_list_preflight"] == "NOT_AUTHORIZED_BEFORE_SEPARATE_ROOT_AND_LIST_BINDING",
        "preflight boundary",
    )
    require(authorization["byte_seal_creation"] == "NOT_AUTHORIZED_WITHOUT_PREFLIGHT", "seal boundary")
    require(authorization["label_derivation_or_analytic_read"] == "NOT_AUTHORIZED_BEFORE_F_PASS_SUCCESSOR", "label boundary")
    require(authorization["model_fitting"] == "NOT_AUTHORIZED", "model-fit boundary")
    require(authorization["g4_open"] == "NOT_AUTHORIZED", "G4 boundary")
    return report


def verify(
    *,
    config_path: Path,
    membership_root: Path,
    cutoff_root: Path,
    author_history_root: Path,
    citation_root: Path,
    taxonomy_topics: Path,
    output_root: Path,
    verification_output: Path,
    threads: int,
    memory_limit_gb: int,
) -> dict[str, Any]:
    require(1 <= threads <= 64, "threads out of range")
    require(64 <= memory_limit_gb <= 512, "memory limit out of range")
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    _verify_repository_bindings(config)
    membership = _verify_membership_parent(
        membership_root,
        _mapping(config["verified_private_membership_parent"], "membership parent"),
    )
    inputs = _mapping(config["normalized_inputs"], "normalized inputs")
    work_glob = _verify_source(cutoff_root, inputs["cutoff_work_metadata"], "cutoff source")
    event_glob = _verify_source(author_history_root, inputs["author_history_events"], "author-history source")
    citation_glob = _verify_source(citation_root, inputs["citation_chronology"], "citation source")
    taxonomy = _load_taxonomy(taxonomy_topics, inputs["topic_taxonomy_projection"])
    output_root = output_root.resolve(strict=True)
    require(output_root.name == config["output_contract"]["root_basename"], "output basename")
    verification_output = verification_output.resolve()
    require(not verification_output.exists() and verification_output.parent.is_dir(), "verification output must be new")
    receipt, receipt_artifacts = _verify_output_tree(output_root, config)
    package = _verify_schemas_and_manifest(output_root, membership, config)
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.verify.", dir=output_root.parent) as temporary:
        temp_path = Path(temporary)
        os.chmod(temp_path, 0o700)
        con = duckdb.connect()
        try:
            _configure_connection(con, threads, memory_limit_gb, temp_path)
            _create_output_views(con, package, membership)
            row_algebra = _verify_output_row_algebra(con, package, config)
            source_replay = _verify_source_replay(
                con,
                work_glob=work_glob,
                event_glob=event_glob,
                citation_glob=citation_glob,
                taxonomy=taxonomy,
                config=config,
            )
            report = _verify_report(
                root=output_root,
                config_path=config_path,
                config=config,
                receipt=receipt,
                membership=membership,
                artifacts=package,
                source_replay=source_replay,
                con=con,
            )
            public_success_positive_rows = int(
                con.execute("SELECT count(*) FROM observed_success WHERE public_success_proxy").fetchone()[0]
            )
        finally:
            con.close()
    verification = {
        "schema_version": VERIFY_SCHEMA,
        "status": PASS_STATUS,
        "output_root_basename": output_root.name,
        "output_receipt_sha256": _sha256(output_root / "g4_private_feature_source_materialization.outputs.json"),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "artifact_count": len(receipt_artifacts),
        "feature_episode_files_verified": len(package["feature_paths"]),
        "relation_episode_files_verified": len(package["relation_paths"]),
        "postdecision_episode_files_verified": len(package["post_paths"]),
        "byline_episode_files_verified": len(package["byline_paths"]),
        "feature_rows_verified": package["feature_rows"],
        "contributing_peer_rows_verified": package["contributing_peer_rows"],
        "public_success_work_rows_verified": config["public_success_contract"]["expected_reference_work_rows"],
        "public_success_positive_work_rows_verified": public_success_positive_rows,
        "postdecision_target_event_rows_verified": package["postdecision_target_event_rows"],
        "postdecision_target_work_byline_rows_verified": package["postdecision_target_work_byline_rows"],
        "output_row_algebra": row_algebra,
        "source_replay": source_replay,
        "parent_membership_and_risk_projection_verified": True,
        "source_replay_all_materialized_columns_verified": True,
        "inclusive_tie_public_success_cutoff_verified": True,
        "postdecision_focal_byline_closure_verified": True,
        "forbidden_label_rate_score_columns_absent": True,
        "g4_private_predecision_membership_complete": True,
        "g4_author_cold_guard_complete": True,
        "g4_private_predecision_features_complete": True,
        "g4_postdecision_target_source_complete": True,
        "complete_private_g4_artifact_set": True,
        "earlier_partition_guard_application_complete": False,
        "entry_or_persistence_labels_materialized": False,
        "holdout_preflight_complete": False,
        "holdout_byte_seal_complete": False,
        "confirmatory_or_holdout_analytic_rows_read": 0,
        "model_score_read": False,
        "model_fitted": False,
        "g4_open": False,
        "pass_f": False,
        "pass_g0": False,
        "authorization": report["authorization"],
    }
    with verification_output.open("x", encoding="utf-8") as handle:
        json.dump(verification, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(verification_output, 0o600)
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--membership-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--citation-root", type=Path, required=True)
    parser.add_argument("--taxonomy-topics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--memory-limit-gb", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(
        config_path=args.config,
        membership_root=args.membership_root,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        citation_root=args.citation_root,
        taxonomy_topics=args.taxonomy_topics,
        output_root=args.output_root,
        verification_output=args.verification_output,
        threads=args.threads,
        memory_limit_gb=args.memory_limit_gb,
    )
    safe = {
        key: result[key]
        for key in (
            "schema_version",
            "status",
            "output_root_basename",
            "feature_rows_verified",
            "contributing_peer_rows_verified",
            "public_success_work_rows_verified",
            "postdecision_target_event_rows_verified",
            "postdecision_target_work_byline_rows_verified",
            "source_replay_all_materialized_columns_verified",
            "complete_private_g4_artifact_set",
            "holdout_preflight_complete",
            "holdout_byte_seal_complete",
            "confirmatory_or_holdout_analytic_rows_read",
            "pass_f",
            "pass_g0",
            "g4_open",
        )
    }
    print(json.dumps(safe, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
