#!/usr/bin/env python3
"""Materialize the complete private G4 feature and postdecision source package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_f_g4_private_feature_source_v1.yaml"
CONFIG_SCHEMA = "pivot_feedback_f_g4_private_feature_source.v1"

FEATURE_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.uint16(), nullable=False),
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("subfield_id", pa.uint16(), nullable=False),
        pa.field("decision_year", pa.uint16(), nullable=False),
        pa.field("base_reference_count", pa.uint64(), nullable=False),
        pa.field("base_recent_count", pa.uint64(), nullable=False),
        pa.field("excluded_reference_count", pa.uint64(), nullable=False),
        pa.field("excluded_recent_count", pa.uint64(), nullable=False),
        pa.field("network_leaveout_reference_count", pa.uint64(), nullable=False),
        pa.field("network_leaveout_recent_count", pa.uint64(), nullable=False),
        pa.field("network_leaveout_surge_present", pa.bool_(), nullable=False),
        pa.field("pre_entry_eligible_work_count", pa.uint32(), nullable=False),
        pa.field("pre_entry_focal_field_work_count", pa.uint32(), nullable=False),
        pa.field("portfolio_3y_eligible_work_count", pa.uint32(), nullable=False),
        pa.field("portfolio_3y_focal_field_work_count", pa.uint32(), nullable=False),
        pa.field("trailing_3y_target_topic_work_count", pa.uint32(), nullable=False),
        pa.field("trailing_3y_target_subfield_work_count", pa.uint32(), nullable=False),
        pa.field("first_eligible_work_year", pa.uint16(), nullable=False),
        pa.field("career_age_years", pa.uint16(), nullable=False),
        pa.field("predecision_network_degree", pa.uint64(), nullable=False),
        pa.field("prior_target_active_peer_count", pa.uint64(), nullable=False),
        pa.field("capacity_bridge_peer_count", pa.uint64(), nullable=False),
        pa.field("peer_visible_success_count", pa.uint64(), nullable=False),
        pa.field("has_prior_target_active_peer", pa.bool_(), nullable=False),
        pa.field("has_capacity_bridge", pa.bool_(), nullable=False),
        pa.field("has_peer_visible_success", pa.bool_(), nullable=False),
        pa.field("complete_predecision_feature_row", pa.bool_(), nullable=False),
    ]
)

RELATION_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.uint16(), nullable=False),
        pa.field("focal_author_numeric_id", pa.uint64(), nullable=False),
        pa.field("peer_author_numeric_id", pa.uint64(), nullable=False),
        pa.field("first_shared_work_year", pa.uint16(), nullable=False),
        pa.field("last_shared_work_year", pa.uint16(), nullable=False),
        pa.field("distinct_shared_eligible_work_count", pa.uint32(), nullable=False),
        pa.field("capacity_bridge", pa.bool_(), nullable=False),
        pa.field("peer_visible_success", pa.bool_(), nullable=False),
        pa.field("earliest_success_available_year", pa.uint16()),
        pa.field("canonical_success_source_work_numeric_id", pa.uint64()),
        pa.field("contributing_source", pa.bool_(), nullable=False),
    ]
)

SUCCESS_SCHEMA = pa.schema(
    [
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("citation_count_age_3", pa.uint64(), nullable=False),
        pa.field("reference_cohort_count", pa.uint64(), nullable=False),
        pa.field("target_count", pa.uint64(), nullable=False),
        pa.field("citation_cutoff_age_3", pa.uint64(), nullable=False),
        pa.field("public_success_proxy", pa.bool_(), nullable=False),
    ]
)

POST_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.uint16(), nullable=False),
        pa.field("focal_author_numeric_id", pa.uint64(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("event_offset", pa.int8(), nullable=False),
        pa.field("declared_authors_count", pa.uint32(), nullable=False),
    ]
)

BYLINE_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.uint16(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("byline_author_numeric_id", pa.uint64(), nullable=False),
    ]
)

EPISODE_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.uint16(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("subfield_id", pa.uint16(), nullable=False),
        pa.field("decision_year", pa.uint16(), nullable=False),
        pa.field("feature_rows", pa.uint64(), nullable=False),
        pa.field("contributing_peer_rows", pa.uint64(), nullable=False),
        pa.field("postdecision_target_event_rows", pa.uint64(), nullable=False),
        pa.field("postdecision_target_work_byline_rows", pa.uint64(), nullable=False),
        pa.field("membership_count_closure", pa.bool_(), nullable=False),
        pa.field("peer_count_closure", pa.bool_(), nullable=False),
        pa.field("labels_materialized", pa.bool_(), nullable=False),
        pa.field("operator_only", pa.bool_(), nullable=False),
        pa.field("sealed", pa.bool_(), nullable=False),
    ]
)


class G4PrivateFeatureSourceBuildError(RuntimeError):
    """Fail-closed private G4 feature/source materialization error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G4PrivateFeatureSourceBuildError(message)


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


def _schema_version(path: Path) -> str:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    return metadata.get(b"schema_version", b"").decode()


def _numeric_suffix(value: str, marker: str) -> int:
    require(marker in value, f"taxonomy identifier missing {marker}")
    suffix = value.rsplit(marker, 1)[1]
    require(suffix.isdigit(), "taxonomy numeric suffix")
    return int(suffix)


def _validate_repository_bindings(config: Mapping[str, Any]) -> dict[str, Any]:
    authorization = _mapping(config["authorization_parent"], "authorization parent")
    authorization_path = _safe_repo_path(authorization["relative_path"])
    require(_sha256(authorization_path) == authorization["sha256"], "authorization SHA")
    authorization_doc = _yaml(authorization_path)
    require(
        authorization_doc.get("schema_version") == authorization["required_schema_version"],
        "authorization schema",
    )
    require(
        authorization_doc["authorization_after_pass"][authorization["required_authorization"]]
        == authorization["required_value"],
        "feature/source materialization not authorized",
    )
    result = {
        "authorization_relative_path": authorization["relative_path"],
        "authorization_sha256": authorization["sha256"],
    }
    sources = _mapping(config["bound_design_sources"], "bound design sources")
    for key in ("membership_summary", "scientific_sections", "measurement_proxy"):
        binding = _mapping(sources[key], key)
        path = _safe_repo_path(binding["relative_path"])
        require(_sha256(path) == binding["sha256"], f"{key} SHA")
        document = _json(path) if path.suffix == ".json" else _yaml(path)
        require(document.get("schema_version") == binding["required_schema_version"], f"{key} schema")
        if "required_status" in binding:
            require(document.get("status") == binding["required_status"], f"{key} status")
        result[f"{key}_relative_path"] = binding["relative_path"]
        result[f"{key}_sha256"] = binding["sha256"]
    scientific = _yaml(_safe_repo_path(sources["scientific_sections"]["relative_path"]))
    for section_id in sources["scientific_sections"]["required_sections"]:
        require(scientific["sections"][section_id]["freeze_state"] == "FROZEN_OUTCOME_BLIND", f"section {section_id}")
    proxy = _yaml(_safe_repo_path(sources["measurement_proxy"]["relative_path"]))
    require(
        proxy["precommitment_choices"]["public_success_threshold"]["rule_id"]
        == sources["measurement_proxy"]["required_public_success_rule"],
        "public-success rule",
    )
    return result


def _validate_membership_parent(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    root = root.resolve(strict=True)
    require(root.name == binding["expected_root_basename"], "membership root basename")
    require(root.is_dir() and not root.is_symlink(), "membership root")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "membership receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "membership receipt schema")
    require(receipt.get("status") == binding["required_status"], "membership receipt status")
    require(receipt.get("artifact_tree_sha256") == binding["artifact_tree_sha256"], "membership tree")
    verification_path = Path(str(binding["verification_path"])).resolve(strict=True)
    require(_sha256(verification_path) == binding["verification_sha256"], "membership verification SHA")
    require(_json(verification_path).get("status") == binding["verification_status"], "membership verification status")
    artifacts = {str(row["relative_path"]): dict(row) for row in receipt["artifacts"]}
    risk_rows = [
        row for relative, row in artifacts.items()
        if relative.startswith(str(binding["risk_prefix"]))
    ]
    require(len(risk_rows) == binding["risk_file_count"], "membership risk file count")
    require(sum(int(row["rows"]) for row in risk_rows) == binding["risk_rows"], "membership risk rows")
    risk_paths = []
    for row in sorted(risk_rows, key=lambda item: str(item["relative_path"])):
        path = root / str(row["relative_path"])
        require(path.stat().st_size == row["bytes"], "membership risk bytes")
        require(_sha256(path) == row["sha256"], "membership risk SHA")
        require(_schema_version(path) == binding["risk_schema_version"], "membership risk schema")
        risk_paths.append(path)
    guard_path = root / str(binding["author_cold_guard_basename"])
    episode_path = root / str(binding["episode_manifest_basename"])
    for path, prefix in ((guard_path, "author_cold_guard"), (episode_path, "episode_manifest")):
        require(_sha256(path) == binding[f"{prefix}_sha256"], f"membership {prefix} SHA")
        require(_schema_version(path) == binding[f"{prefix}_schema_version"], f"membership {prefix} schema")
        require(pq.ParquetFile(path).metadata.num_rows == binding[f"{prefix}_rows"], f"membership {prefix} rows")
    for key in ("complete_private_g4_artifact_set", "holdout_preflight_complete", "holdout_byte_seal_complete", "g4_open", "pass_f", "pass_g0"):
        require(receipt.get(key) is False, f"unsafe membership state: {key}")
    return {
        "root": root,
        "receipt_sha256": binding["receipt_sha256"],
        "verification_sha256": binding["verification_sha256"],
        "risk_paths": risk_paths,
        "guard_path": guard_path,
        "episode_path": episode_path,
    }


def _validate_partitioned_source(root: Path, binding: Mapping[str, Any]) -> Path:
    root = root.resolve(strict=True)
    require(root.name == binding["expected_root_basename"], "source root basename")
    require(root.is_dir() and not root.is_symlink(), "source root")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "source receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "source receipt schema")
    require(receipt.get("status") == binding["required_status"], "source receipt status")
    artifact_root = root / str(binding["artifact_prefix"])
    files = sorted(artifact_root.glob("*.parquet"))
    require(len(files) == binding["artifact_count"], "source artifact count")
    require(all(path.is_file() and not path.is_symlink() for path in files), "source artifact type")
    return artifact_root / "*.parquet"


def _load_taxonomy(path: Path, binding: Mapping[str, Any]) -> tuple[pa.Table, dict[str, Any]]:
    path = path.resolve(strict=True)
    require(path.name == binding["expected_basename"], "taxonomy basename")
    require(path.is_file() and not path.is_symlink(), "taxonomy input")
    require(_sha256(path) == binding["sha256"], "taxonomy SHA")
    columns = list(binding["required_columns"])
    table = pq.read_table(path, columns=columns)
    require(len(table) == binding["rows"], "taxonomy rows")
    rows = []
    for row in table.to_pylist():
        rows.append(
            {
                "topic_id": _numeric_suffix(str(row["id"]), "/T"),
                "field_id": _numeric_suffix(str(row["field__id"]), "/fields/"),
                "subfield_id": _numeric_suffix(str(row["subfield__id"]), "/subfields/"),
            }
        )
    require(len({row["topic_id"] for row in rows}) == len(rows), "taxonomy topic uniqueness")
    schema = pa.schema(
        [
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("subfield_id", pa.uint16(), nullable=False),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema), {
        "basename": path.name,
        "sha256": binding["sha256"],
        "rows": len(rows),
    }


def _available_memory_bytes() -> int:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    return values["MemAvailable"]


def _resource_preflight(output_parent: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    contract = _mapping(config["resource_preflight"], "resource preflight")
    disk = shutil.disk_usage(output_parent)
    memory = _available_memory_bytes()
    nofile_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    require(disk.free >= contract["minimum_available_disk_bytes"], "insufficient free disk")
    require(memory >= contract["minimum_available_memory_bytes"], "insufficient available memory")
    require(nofile_soft >= contract["minimum_nofile_soft_limit"], "insufficient nofile limit")
    return {
        "available_disk_bytes": disk.free,
        "minimum_available_disk_bytes": contract["minimum_available_disk_bytes"],
        "available_memory_bytes": memory,
        "minimum_available_memory_bytes": contract["minimum_available_memory_bytes"],
        "nofile_soft_limit": nofile_soft,
        "minimum_nofile_soft_limit": contract["minimum_nofile_soft_limit"],
        "passed_before_source_payload_open": True,
    }


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _sql_files(paths: Sequence[Path]) -> str:
    return "[" + ",".join(f"'{_sql_path(path)}'" for path in paths) + "]"


def _export_reader(reader: pa.RecordBatchReader, path: Path, schema: pa.Schema, version: str) -> int:
    expected = [(field.name, field.type) for field in schema]
    observed = [(field.name, field.type) for field in reader.schema]
    require(observed == expected, f"query schema: {path.name}")
    versioned = schema.with_metadata({b"schema_version": version.encode("utf-8")})
    writer = pq.ParquetWriter(
        path,
        versioned,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    rows = 0
    try:
        for batch in reader:
            table = pa.Table.from_batches([batch]).cast(versioned)
            writer.write_table(table, row_group_size=250_000)
            rows += len(table)
    finally:
        writer.close()
    os.chmod(path, 0o600)
    return rows


def _write_table(table: pa.Table, path: Path, schema: pa.Schema, version: str) -> int:
    versioned = schema.with_metadata({b"schema_version": version.encode("utf-8")})
    table = table.cast(versioned)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
        row_group_size=250_000,
    )
    os.chmod(path, 0o600)
    return len(table)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    row = {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        row["rows"] = pq.ParquetFile(path).metadata.num_rows
    return row


def _make_private_directories(staging: Path) -> dict[str, Path]:
    private = staging / "private"
    roots = {
        "private": private,
        "feature": private / "g4_predecision_feature",
        "relation": private / "g4_contributing_peer_source",
        "post": private / "g4_postdecision_target_event_source",
        "byline": private / "g4_postdecision_target_work_byline_source",
    }
    for path in roots.values():
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    return roots


def _build_database(
    *,
    con: duckdb.DuckDBPyConnection,
    membership: Mapping[str, Any],
    taxonomy: pa.Table,
    work_glob: Path,
    event_glob: Path,
    citation_glob: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    con.register("taxonomy_input", taxonomy)
    con.execute("CREATE TABLE taxonomy AS SELECT * FROM taxonomy_input")
    con.execute(f"CREATE TABLE risk AS SELECT * FROM read_parquet({_sql_files(membership['risk_paths'])})")
    con.execute(f"CREATE TABLE guard AS SELECT * FROM read_parquet('{_sql_path(membership['guard_path'])}')")
    con.execute(f"CREATE TABLE episode AS SELECT * FROM read_parquet('{_sql_path(membership['episode_path'])}')")
    con.execute("CREATE INDEX risk_author_year ON risk(author_numeric_id, decision_year)")
    con.execute("CREATE INDEX guard_author ON guard(author_numeric_id)")
    con.execute("CREATE INDEX episode_key ON episode(episode_id)")
    risk_rows = int(con.execute("SELECT count(*) FROM risk").fetchone()[0])
    require(risk_rows == config["feature_contract"]["feature_rows"], "risk row closure")
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
        CREATE TABLE guard_event AS
        SELECT DISTINCT e.author_numeric_id AS author_id, e.work_numeric_id AS work_id,
               e.publication_year, e.declared_authors_count,
               e.primary_field_id AS field_id, w.primary_topic_id AS topic_id,
               tx.subfield_id
        FROM read_parquet('{event}') e
        JOIN guard g ON g.author_numeric_id=e.author_numeric_id
        JOIN read_parquet('{work}') w ON w.work_id_numeric=e.work_numeric_id
        JOIN taxonomy tx ON tx.topic_id=w.primary_topic_id AND tx.field_id=w.primary_field_id
        WHERE e.publication_year<=2025 AND {eligible}
        """
    )
    con.execute("CREATE INDEX guard_event_author ON guard_event(author_id, publication_year)")
    con.execute(
        f"""
        CREATE TABLE recent_event AS
        SELECT DISTINCT e.author_numeric_id AS author_id, e.work_numeric_id AS work_id,
               e.publication_year
        FROM read_parquet('{event}') e
        JOIN read_parquet('{work}') w ON w.work_id_numeric=e.work_numeric_id
        WHERE e.publication_year BETWEEN 2015 AND 2021
          AND e.primary_field_id BETWEEN 11 AND 36 AND {eligible}
        """
    )
    con.execute("CREATE INDEX recent_event_author ON recent_event(author_id, publication_year)")
    con.execute("CREATE INDEX recent_event_work ON recent_event(work_id)")
    con.execute(
        f"""
        CREATE TABLE target_event AS
        SELECT DISTINCT e.author_numeric_id AS author_id, e.work_numeric_id AS work_id,
               e.publication_year, e.declared_authors_count,
               e.primary_field_id AS field_id, w.primary_topic_id AS topic_id
        FROM read_parquet('{event}') e
        JOIN read_parquet('{work}') w ON w.work_id_numeric=e.work_numeric_id
        JOIN (SELECT DISTINCT field_id,topic_id FROM episode) p
          ON p.field_id=e.primary_field_id AND p.topic_id=w.primary_topic_id
        WHERE e.publication_year<=2025 AND {eligible}
        """
    )
    con.execute("CREATE INDEX target_event_topic_author ON target_event(field_id, topic_id, author_id)")
    con.execute("CREATE INDEX target_event_work ON target_event(work_id)")
    con.execute("CREATE TABLE risk_author_year AS SELECT DISTINCT author_numeric_id AS author_id, decision_year FROM risk")
    con.execute("CREATE INDEX risk_author_year_key ON risk_author_year(author_id, decision_year)")
    con.execute(
        """
        CREATE TABLE author_year_feature AS
        SELECT ray.author_id, ray.decision_year,
               count(DISTINCT ge.work_id) FILTER (WHERE ge.publication_year<=ray.decision_year-1) AS pre_entry_count,
               count(DISTINCT ge.work_id) FILTER (WHERE ge.publication_year<=ray.decision_year-1 AND ge.field_id=27) AS pre_entry_focal_count,
               count(DISTINCT ge.work_id) FILTER (WHERE ge.publication_year BETWEEN ray.decision_year-3 AND ray.decision_year-1) AS portfolio_count,
               count(DISTINCT ge.work_id) FILTER (WHERE ge.publication_year BETWEEN ray.decision_year-3 AND ray.decision_year-1 AND ge.field_id=27) AS portfolio_focal_count,
               min(ge.publication_year) FILTER (WHERE ge.publication_year<=ray.decision_year-1) AS first_year
        FROM risk_author_year ray JOIN guard_event ge ON ge.author_id=ray.author_id
        WHERE ge.publication_year<=ray.decision_year-1
        GROUP BY ray.author_id,ray.decision_year
        """
    )
    con.execute("CREATE INDEX author_year_feature_key ON author_year_feature(author_id, decision_year)")
    con.execute(
        """
        CREATE TABLE author_subfield_year AS
        SELECT author_id,publication_year,subfield_id,count(DISTINCT work_id) AS work_count
        FROM guard_event GROUP BY author_id,publication_year,subfield_id
        """
    )
    con.execute(
        """
        CREATE TABLE risk_subfield_feature AS
        SELECT r.episode_id,r.author_numeric_id,
               coalesce(sum(s.work_count),0) AS trailing_subfield_count
        FROM risk r LEFT JOIN author_subfield_year s
          ON s.author_id=r.author_numeric_id AND s.subfield_id=r.subfield_id
         AND s.publication_year BETWEEN r.decision_year-3 AND r.decision_year-1
        GROUP BY r.episode_id,r.author_numeric_id
        """
    )
    con.execute("CREATE INDEX risk_subfield_key ON risk_subfield_feature(episode_id, author_numeric_id)")
    con.execute(
        """
        CREATE TABLE risk_shared_work AS
        SELECT DISTINCT ray.author_id AS focal_author_id, ray.decision_year,
               re.work_id,re.publication_year
        FROM risk_author_year ray JOIN recent_event re ON re.author_id=ray.author_id
        WHERE re.publication_year BETWEEN ray.decision_year-5 AND ray.decision_year-1
        """
    )
    con.execute("CREATE INDEX risk_shared_work_id ON risk_shared_work(work_id)")
    con.execute(
        """
        CREATE TABLE prior_peer_year AS
        SELECT f.focal_author_id,f.decision_year,co.author_id AS peer_author_id,
               min(f.publication_year) AS first_shared_year,
               max(f.publication_year) AS last_shared_year,
               count(DISTINCT f.work_id) AS shared_work_count
        FROM risk_shared_work f JOIN recent_event co ON co.work_id=f.work_id
        WHERE co.author_id<>f.focal_author_id
        GROUP BY f.focal_author_id,f.decision_year,co.author_id
        """
    )
    con.execute("CREATE INDEX prior_peer_focal ON prior_peer_year(focal_author_id, decision_year)")
    con.execute(
        """
        CREATE TABLE network_feature AS
        SELECT focal_author_id,decision_year,count(*) AS network_degree
        FROM prior_peer_year GROUP BY focal_author_id,decision_year
        """
    )
    con.execute("CREATE INDEX network_feature_key ON network_feature(focal_author_id, decision_year)")
    con.execute(
        """
        CREATE TABLE target_first AS
        SELECT ep.episode_id,te.author_id,min(te.publication_year) AS first_target_year
        FROM episode ep JOIN target_event te
          ON te.field_id=ep.field_id AND te.topic_id=ep.topic_id
        WHERE te.publication_year<=ep.decision_year-1
        GROUP BY ep.episode_id,te.author_id
        """
    )
    con.execute("CREATE INDEX target_first_key ON target_first(episode_id, author_id)")
    con.execute(
        """
        CREATE TABLE contributing_peer_base AS
        SELECT r.episode_id,r.author_numeric_id AS focal_author_id,p.peer_author_id,
               p.first_shared_year,p.last_shared_year,p.shared_work_count,
               p.last_shared_year>=r.decision_year-3 AS capacity_bridge
        FROM risk r
        JOIN prior_peer_year p ON p.focal_author_id=r.author_numeric_id AND p.decision_year=r.decision_year
        JOIN target_first tf ON tf.episode_id=r.episode_id AND tf.author_id=p.peer_author_id
        """
    )
    relation_rows = int(con.execute("SELECT count(*) FROM contributing_peer_base").fetchone()[0])
    require(relation_rows == config["feature_contract"]["contributing_peer_relation_rows"], "contributing peer relation closure")
    peer_count_mismatch = int(
        con.execute(
            """
            SELECT count(*) FROM risk r LEFT JOIN (
              SELECT episode_id,focal_author_id,count(*) AS n
              FROM contributing_peer_base GROUP BY episode_id,focal_author_id
            ) p ON p.episode_id=r.episode_id AND p.focal_author_id=r.author_numeric_id
            WHERE r.prior_target_active_peer_count<>coalesce(p.n,0)
            """
        ).fetchone()[0]
    )
    require(peer_count_mismatch == 0, "parent peer count mismatch")
    con.execute(
        f"""
        CREATE TABLE field_work AS
        SELECT w.work_id_numeric AS work_id,w.primary_field_id AS field_id,w.publication_year
        FROM read_parquet('{work}') w
        WHERE w.primary_field_id=27 AND w.publication_year<=2018
          AND w.publication_year IS NOT NULL AND w.temporal_disposition IN (2,3)
          AND w.work_type_code IN (1,2,3,4,7,16,18,21)
          AND NOT w.is_paratext AND coalesce(w.is_retracted,false)=false
        """
    )
    field_work_rows = int(con.execute("SELECT count(*) FROM field_work").fetchone()[0])
    require(field_work_rows == config["public_success_contract"]["expected_reference_work_rows"], "public-success reference work rows")
    con.execute("CREATE INDEX field_work_id ON field_work(work_id)")
    con.execute(
        f"""
        CREATE TABLE field_work_citation AS
        SELECT fw.work_id,fw.field_id,fw.publication_year,
               count(c.citing_work_id) AS citation_count_age_3
        FROM field_work fw LEFT JOIN read_parquet('{citation}') c
          ON c.cited_work_id=fw.work_id
         AND c.available_at<make_date(fw.publication_year+4,1,1)
        GROUP BY fw.work_id,fw.field_id,fw.publication_year
        """
    )
    con.execute(
        """
        CREATE TABLE public_success_work AS
        WITH ranked AS (
          SELECT *,row_number() OVER (
                   PARTITION BY field_id,publication_year
                   ORDER BY citation_count_age_3 DESC,work_id
                 ) AS citation_rank,
                 count(*) OVER (PARTITION BY field_id,publication_year) AS cohort_count
          FROM field_work_citation
        ), cutoffs AS (
          SELECT field_id,publication_year,max(cohort_count) AS cohort_count,
                 ceil(max(cohort_count)*0.10)::UBIGINT AS target_count,
                 max(citation_count_age_3) FILTER (
                   WHERE citation_rank=ceil(cohort_count*0.10)
                 ) AS cutoff
          FROM ranked GROUP BY field_id,publication_year
        )
        SELECT r.work_id,r.field_id,r.publication_year,r.citation_count_age_3,
               c.cohort_count,c.target_count,c.cutoff,
               r.citation_count_age_3>=c.cutoff AS public_success_proxy
        FROM ranked r JOIN cutoffs c USING (field_id,publication_year)
        """
    )
    unresolved_success_groups = int(
        con.execute(
            "SELECT count(*) FROM (SELECT publication_year,count(*) n FROM public_success_work GROUP BY publication_year HAVING n<10)"
        ).fetchone()[0]
    )
    require(unresolved_success_groups == 0, "public-success reference cohort below ten")
    con.execute("CREATE INDEX public_success_work_id ON public_success_work(work_id)")
    con.execute(
        """
        CREATE TABLE peer_success_ranked AS
        SELECT ep.episode_id,te.author_id,te.publication_year+3 AS available_year,
               te.work_id,row_number() OVER (
                 PARTITION BY ep.episode_id,te.author_id
                 ORDER BY te.publication_year+3,te.author_id,te.work_id
               ) AS source_order
        FROM episode ep JOIN target_event te
          ON te.field_id=ep.field_id AND te.topic_id=ep.topic_id
        JOIN public_success_work ps ON ps.work_id=te.work_id AND ps.public_success_proxy
        WHERE te.publication_year+3<=ep.decision_year-1
        """
    )
    con.execute(
        """
        CREATE TABLE peer_success AS
        SELECT episode_id,author_id,min(available_year) AS earliest_available_year,
               max(work_id) FILTER (WHERE source_order=1) AS canonical_source_work_id
        FROM peer_success_ranked GROUP BY episode_id,author_id
        """
    )
    con.execute("CREATE INDEX peer_success_key ON peer_success(episode_id,author_id)")
    con.execute(
        """
        CREATE TABLE contributing_peer AS
        SELECT b.episode_id,b.focal_author_id,b.peer_author_id,b.first_shared_year,
               b.last_shared_year,b.shared_work_count,b.capacity_bridge,
               ps.author_id IS NOT NULL AS peer_visible_success,
               ps.earliest_available_year,ps.canonical_source_work_id
        FROM contributing_peer_base b LEFT JOIN peer_success ps
          ON ps.episode_id=b.episode_id AND ps.author_id=b.peer_author_id
        """
    )
    con.execute("CREATE INDEX contributing_peer_focal ON contributing_peer(episode_id,focal_author_id)")
    con.execute(
        """
        CREATE TABLE peer_feature AS
        SELECT episode_id,focal_author_id,count(*) AS active_count,
               count(*) FILTER (WHERE capacity_bridge) AS capacity_count,
               count(*) FILTER (WHERE peer_visible_success) AS visible_success_count
        FROM contributing_peer GROUP BY episode_id,focal_author_id
        """
    )
    con.execute("CREATE INDEX peer_feature_key ON peer_feature(episode_id,focal_author_id)")
    con.execute(
        """
        CREATE TABLE predecision_feature AS
        SELECT r.episode_id::USMALLINT AS episode_id,
               r.author_numeric_id::UBIGINT AS author_numeric_id,
               r.field_id::UTINYINT AS field_id,r.topic_id::UINTEGER AS topic_id,
               r.subfield_id::USMALLINT AS subfield_id,r.decision_year::USMALLINT AS decision_year,
               r.base_reference_count::UBIGINT AS base_reference_count,
               r.base_recent_count::UBIGINT AS base_recent_count,
               r.excluded_reference_count::UBIGINT AS excluded_reference_count,
               r.excluded_recent_count::UBIGINT AS excluded_recent_count,
               r.network_leaveout_reference_count::UBIGINT AS network_leaveout_reference_count,
               r.network_leaveout_recent_count::UBIGINT AS network_leaveout_recent_count,
               r.network_leaveout_surge_present::BOOLEAN AS network_leaveout_surge_present,
               h.pre_entry_count::UINTEGER AS pre_entry_eligible_work_count,
               h.pre_entry_focal_count::UINTEGER AS pre_entry_focal_field_work_count,
               h.portfolio_count::UINTEGER AS portfolio_3y_eligible_work_count,
               h.portfolio_focal_count::UINTEGER AS portfolio_3y_focal_field_work_count,
               0::UINTEGER AS trailing_3y_target_topic_work_count,
               sf.trailing_subfield_count::UINTEGER AS trailing_3y_target_subfield_work_count,
               h.first_year::USMALLINT AS first_eligible_work_year,
               (r.decision_year-h.first_year)::USMALLINT AS career_age_years,
               coalesce(n.network_degree,0)::UBIGINT AS predecision_network_degree,
               r.prior_target_active_peer_count::UBIGINT AS prior_target_active_peer_count,
               coalesce(p.capacity_count,0)::UBIGINT AS capacity_bridge_peer_count,
               coalesce(p.visible_success_count,0)::UBIGINT AS peer_visible_success_count,
               (r.prior_target_active_peer_count>0)::BOOLEAN AS has_prior_target_active_peer,
               (coalesce(p.capacity_count,0)>0)::BOOLEAN AS has_capacity_bridge,
               (coalesce(p.visible_success_count,0)>0)::BOOLEAN AS has_peer_visible_success,
               true::BOOLEAN AS complete_predecision_feature_row
        FROM risk r
        JOIN author_year_feature h ON h.author_id=r.author_numeric_id AND h.decision_year=r.decision_year
        JOIN risk_subfield_feature sf ON sf.episode_id=r.episode_id AND sf.author_numeric_id=r.author_numeric_id
        LEFT JOIN network_feature n ON n.focal_author_id=r.author_numeric_id AND n.decision_year=r.decision_year
        LEFT JOIN peer_feature p ON p.episode_id=r.episode_id AND p.focal_author_id=r.author_numeric_id
        ORDER BY r.episode_id,r.author_numeric_id
        """
    )
    feature_rows = int(con.execute("SELECT count(*) FROM predecision_feature").fetchone()[0])
    require(feature_rows == risk_rows, "feature row closure")
    bad_feature_rows = int(
        con.execute(
            """
            SELECT count(*) FROM predecision_feature
            WHERE pre_entry_eligible_work_count<portfolio_3y_eligible_work_count
               OR pre_entry_focal_field_work_count<portfolio_3y_focal_field_work_count
               OR portfolio_3y_focal_field_work_count=0
               OR trailing_3y_target_topic_work_count<>0
               OR capacity_bridge_peer_count>prior_target_active_peer_count
               OR peer_visible_success_count>prior_target_active_peer_count
               OR first_eligible_work_year>=decision_year
               OR NOT complete_predecision_feature_row
            """
        ).fetchone()[0]
    )
    require(bad_feature_rows == 0, "feature invariants")
    con.execute(
        """
        CREATE TABLE postdecision_target_event AS
        SELECT r.episode_id,r.author_numeric_id AS focal_author_id,te.work_id,
               te.publication_year,(te.publication_year-r.decision_year)::TINYINT AS event_offset,
               te.declared_authors_count
        FROM risk r JOIN target_event te
          ON te.author_id=r.author_numeric_id AND te.field_id=r.field_id AND te.topic_id=r.topic_id
        WHERE te.publication_year BETWEEN r.decision_year AND r.decision_year+3
        """
    )
    con.execute("CREATE INDEX postdecision_work ON postdecision_target_event(episode_id,work_id)")
    con.execute(
        """
        CREATE TABLE postdecision_target_work_byline AS
        SELECT DISTINCT p.episode_id,p.work_id,p.publication_year,
               te.author_id AS byline_author_id
        FROM (SELECT DISTINCT episode_id,work_id,publication_year FROM postdecision_target_event) p
        JOIN target_event te ON te.work_id=p.work_id AND te.publication_year=p.publication_year
        """
    )
    bad_post_rows = int(
        con.execute(
            """
            SELECT count(*) FROM postdecision_target_event
            WHERE event_offset<0 OR event_offset>3
            """
        ).fetchone()[0]
    )
    require(bad_post_rows == 0, "postdecision offset")
    missing_focal_byline = int(
        con.execute(
            """
            SELECT count(*) FROM postdecision_target_event p
            ANTI JOIN postdecision_target_work_byline b
              ON b.episode_id=p.episode_id AND b.work_id=p.work_id
             AND b.byline_author_id=p.focal_author_id
            """
        ).fetchone()[0]
    )
    require(missing_focal_byline == 0, "postdecision focal byline closure")
    return {
        "risk_rows": risk_rows,
        "guard_event_rows": int(con.execute("SELECT count(*) FROM guard_event").fetchone()[0]),
        "recent_event_rows": int(con.execute("SELECT count(*) FROM recent_event").fetchone()[0]),
        "target_event_rows": int(con.execute("SELECT count(*) FROM target_event").fetchone()[0]),
        "risk_author_year_rows": int(con.execute("SELECT count(*) FROM risk_author_year").fetchone()[0]),
        "prior_peer_year_rows": int(con.execute("SELECT count(*) FROM prior_peer_year").fetchone()[0]),
        "contributing_peer_rows": relation_rows,
        "peer_count_mismatch_rows": peer_count_mismatch,
        "public_success_reference_work_rows": field_work_rows,
        "public_success_work_rows": int(con.execute("SELECT count(*) FROM public_success_work").fetchone()[0]),
        "public_success_positive_work_rows": int(con.execute("SELECT count(*) FROM public_success_work WHERE public_success_proxy").fetchone()[0]),
        "feature_rows": feature_rows,
        "bad_feature_rows": bad_feature_rows,
        "postdecision_target_event_rows": int(con.execute("SELECT count(*) FROM postdecision_target_event").fetchone()[0]),
        "postdecision_target_work_byline_rows": int(con.execute("SELECT count(*) FROM postdecision_target_work_byline").fetchone()[0]),
        "postdecision_offsets_read": [0, 1, 2, 3],
        "labels_materialized": False,
        "analytic_outcome_rows_read": 0,
    }


def _export_package(
    *,
    con: duckdb.DuckDBPyConnection,
    roots: Mapping[str, Path],
    membership: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = config["output_contract"]
    episode_rows = pq.read_table(membership["episode_path"]).to_pylist()
    manifest_rows = []
    exported = []
    for episode in sorted(episode_rows, key=lambda row: int(row["episode_id"])):
        episode_id = int(episode["episode_id"])
        specs = (
            (
                "feature",
                "predecision_feature",
                FEATURE_SCHEMA,
                output["feature_schema_version"],
                "SELECT * FROM predecision_feature WHERE episode_id=? ORDER BY author_numeric_id",
            ),
            (
                "relation",
                "contributing_peer",
                RELATION_SCHEMA,
                output["relation_schema_version"],
                """SELECT episode_id::USMALLINT AS episode_id,
                          focal_author_id::UBIGINT AS focal_author_numeric_id,
                          peer_author_id::UBIGINT AS peer_author_numeric_id,
                          first_shared_year::USMALLINT AS first_shared_work_year,
                          last_shared_year::USMALLINT AS last_shared_work_year,
                          shared_work_count::UINTEGER AS distinct_shared_eligible_work_count,
                          capacity_bridge::BOOLEAN AS capacity_bridge,
                          peer_visible_success::BOOLEAN AS peer_visible_success,
                          earliest_available_year::USMALLINT AS earliest_success_available_year,
                          canonical_source_work_id::UBIGINT AS canonical_success_source_work_numeric_id,
                          true::BOOLEAN AS contributing_source
                   FROM contributing_peer WHERE episode_id=?
                   ORDER BY focal_author_id,peer_author_id""",
            ),
            (
                "post",
                "postdecision_target_event",
                POST_SCHEMA,
                output["postdecision_schema_version"],
                """SELECT episode_id::USMALLINT AS episode_id,
                          focal_author_id::UBIGINT AS focal_author_numeric_id,
                          work_id::UBIGINT AS work_numeric_id,
                          publication_year::USMALLINT AS publication_year,
                          event_offset::TINYINT AS event_offset,
                          declared_authors_count::UINTEGER AS declared_authors_count
                   FROM postdecision_target_event WHERE episode_id=?
                   ORDER BY focal_author_id,publication_year,work_id""",
            ),
            (
                "byline",
                "postdecision_target_work_byline",
                BYLINE_SCHEMA,
                output["byline_schema_version"],
                """SELECT episode_id::USMALLINT AS episode_id,
                          work_id::UBIGINT AS work_numeric_id,
                          publication_year::USMALLINT AS publication_year,
                          byline_author_id::UBIGINT AS byline_author_numeric_id
                   FROM postdecision_target_work_byline WHERE episode_id=?
                   ORDER BY work_id,byline_author_id""",
            ),
        )
        counts = {}
        for root_key, label, schema, version, query in specs:
            path = roots[root_key] / f"episode-{episode_id:04d}.parquet"
            reader = con.execute(query, [episode_id]).fetch_record_batch(1_000_000)
            counts[label] = _export_reader(reader, path, schema, version)
            exported.append({"kind": label, "episode_id": episode_id, "path": path, "rows": counts[label]})
        expected_risk = int(episode["materialized_risk_opportunity_count"])
        expected_peer = int(
            con.execute("SELECT coalesce(sum(prior_target_active_peer_count),0) FROM risk WHERE episode_id=?", [episode_id]).fetchone()[0]
        )
        require(counts["predecision_feature"] == expected_risk, f"episode feature closure: {episode_id}")
        require(counts["contributing_peer"] == expected_peer, f"episode peer closure: {episode_id}")
        manifest_rows.append(
            {
                "episode_id": episode_id,
                "field_id": int(episode["field_id"]),
                "topic_id": int(episode["topic_id"]),
                "subfield_id": int(episode["subfield_id"]),
                "decision_year": int(episode["decision_year"]),
                "feature_rows": counts["predecision_feature"],
                "contributing_peer_rows": counts["contributing_peer"],
                "postdecision_target_event_rows": counts["postdecision_target_event"],
                "postdecision_target_work_byline_rows": counts["postdecision_target_work_byline"],
                "membership_count_closure": True,
                "peer_count_closure": True,
                "labels_materialized": False,
                "operator_only": True,
                "sealed": False,
            }
        )
    success_path = roots["private"] / "g4_public_success_work_source.parquet"
    reader = con.execute(
        """SELECT work_id::UBIGINT AS work_numeric_id,field_id::UTINYINT AS field_id,
                  publication_year::USMALLINT AS publication_year,
                  citation_count_age_3::UBIGINT AS citation_count_age_3,
                  cohort_count::UBIGINT AS reference_cohort_count,
                  target_count::UBIGINT AS target_count,cutoff::UBIGINT AS citation_cutoff_age_3,
                  public_success_proxy::BOOLEAN AS public_success_proxy
           FROM public_success_work ORDER BY publication_year,work_id"""
    ).fetch_record_batch(1_000_000)
    success_rows = _export_reader(reader, success_path, SUCCESS_SCHEMA, output["success_work_schema_version"])
    exported.append({"kind": "public_success_work", "path": success_path, "rows": success_rows})
    guard_path = roots["private"] / "g4_author_cold_guard.parquet"
    shutil.copyfile(membership["guard_path"], guard_path)
    os.chmod(guard_path, 0o600)
    exported.append({"kind": "author_cold_guard", "path": guard_path, "rows": pq.ParquetFile(guard_path).metadata.num_rows})
    manifest_path = roots["private"] / "g4_feature_source_episode_manifest.parquet"
    _write_table(
        pa.Table.from_pylist(manifest_rows, schema=EPISODE_SCHEMA),
        manifest_path,
        EPISODE_SCHEMA,
        output["episode_schema_version"],
    )
    exported.append({"kind": "episode_manifest", "path": manifest_path, "rows": len(manifest_rows)})
    return exported, manifest_rows


def _markdown(report: Mapping[str, Any]) -> str:
    accounting = report["accounting"]
    return "\n".join(
        [
            "# Private G4 feature and postdecision source materialization",
            "",
            f"Status: `{report['status']}`",
            "",
            f"- predecision feature rows: {accounting['feature_rows']:,}",
            f"- contributing peer rows: {accounting['contributing_peer_rows']:,}",
            f"- public-success reference works: {accounting['public_success_work_rows']:,}",
            f"- postdecision target author-work rows: {accounting['postdecision_target_event_rows']:,}",
            f"- postdecision target work-byline rows: {accounting['postdecision_target_work_byline_rows']:,}",
            "- entry or persistence labels: not materialized",
            "- model scores and analytic opening: not performed",
            "",
            "This is a private source package. A separate root/file-list binding",
            "must pass before explicit-list preflight is authorized.",
            "",
        ]
    )


def build(
    *,
    config_path: Path,
    membership_root: Path,
    cutoff_root: Path,
    author_history_root: Path,
    citation_root: Path,
    taxonomy_topics: Path,
    output_root: Path,
    threads: int,
    memory_limit_gb: int,
) -> dict[str, Any]:
    require(1 <= threads <= 64, "threads out of range")
    require(64 <= memory_limit_gb <= 512, "memory limit out of range")
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    repository_bindings = _validate_repository_bindings(config)
    membership = _validate_membership_parent(
        membership_root,
        _mapping(config["verified_private_membership_parent"], "membership parent"),
    )
    inputs = _mapping(config["normalized_inputs"], "normalized inputs")
    output_root = output_root.resolve()
    require(output_root.name == config["output_contract"]["root_basename"], "output basename")
    require(not output_root.exists(), "output already exists")
    require(output_root.parent.is_dir(), "output parent")
    resource_preflight = _resource_preflight(output_root.parent, config)
    work_glob = _validate_partitioned_source(cutoff_root, inputs["cutoff_work_metadata"])
    event_glob = _validate_partitioned_source(author_history_root, inputs["author_history_events"])
    citation_glob = _validate_partitioned_source(citation_root, inputs["citation_chronology"])
    taxonomy, taxonomy_binding = _load_taxonomy(taxonomy_topics, inputs["topic_taxonomy_projection"])
    staging: Path | None = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    os.chmod(staging, 0o700)
    try:
        roots = _make_private_directories(staging)
        scratch = staging / "scratch"
        scratch.mkdir(mode=0o700)
        database = scratch / "feature-source.duckdb"
        spill = scratch / "spill"
        spill.mkdir(mode=0o700)
        con = duckdb.connect(str(database))
        con.execute(f"PRAGMA threads={threads}")
        con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB'")
        con.execute(f"PRAGMA temp_directory='{_sql_path(spill)}'")
        try:
            accounting = _build_database(
                con=con,
                membership=membership,
                taxonomy=taxonomy,
                work_glob=work_glob,
                event_glob=event_glob,
                citation_glob=citation_glob,
                config=config,
            )
            _, episode_manifest = _export_package(
                con=con,
                roots=roots,
                membership=membership,
                config=config,
            )
        finally:
            con.close()
        shutil.rmtree(scratch)
        output_contract = config["output_contract"]
        report = {
            "schema_version": output_contract["report_schema_version"],
            "status": output_contract["passing_status"],
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "execution": {
                "config_relative_path": config_path.relative_to(ROOT).as_posix(),
                "config_sha256": _sha256(config_path),
                "builder_relative_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                "builder_sha256": _sha256(Path(__file__).resolve()),
                "threads": threads,
                "memory_limit_gb": memory_limit_gb,
            },
            "repository_bindings": repository_bindings,
            "parent_bindings": {
                "membership_receipt_sha256": membership["receipt_sha256"],
                "membership_verification_sha256": membership["verification_sha256"],
            },
            "normalized_input_bindings": {
                "cutoff_receipt_sha256": inputs["cutoff_work_metadata"]["receipt_sha256"],
                "author_history_receipt_sha256": inputs["author_history_events"]["receipt_sha256"],
                "citation_receipt_sha256": inputs["citation_chronology"]["receipt_sha256"],
                "taxonomy": taxonomy_binding,
            },
            "resource_preflight": resource_preflight,
            "accounting": accounting,
            "episode_manifest": episode_manifest,
            "completion": {
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
            },
            "authorization": config["authorization_after_pass"],
            "claim_boundary": config["claim_boundary"],
        }
        report_path = staging / "g4_private_feature_source_materialization.json"
        _write_json(report_path, report)
        markdown_path = staging / "G4_PRIVATE_FEATURE_SOURCE_MATERIALIZATION.md"
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        os.chmod(markdown_path, 0o600)
        artifacts = [
            _artifact(path, staging)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        tree = [(row["relative_path"], row["bytes"], row["sha256"]) for row in artifacts]
        receipt = {
            "schema_version": output_contract["receipt_schema_version"],
            "status": output_contract["passing_status"],
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(_canonical_bytes(tree)).hexdigest(),
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
            "receipt_written_last": True,
        }
        receipt_path = staging / "g4_private_feature_source_materialization.outputs.json"
        _write_json(receipt_path, receipt)
        os.replace(staging, output_root)
        staging = None
        return report
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--membership-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--citation-root", type=Path, required=True)
    parser.add_argument("--taxonomy-topics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--memory-limit-gb", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build(
        config_path=args.config,
        membership_root=args.membership_root,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        citation_root=args.citation_root,
        taxonomy_topics=args.taxonomy_topics,
        output_root=args.output_root,
        threads=args.threads,
        memory_limit_gb=args.memory_limit_gb,
    )
    print(report["status"])
    print(json.dumps(report["accounting"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
