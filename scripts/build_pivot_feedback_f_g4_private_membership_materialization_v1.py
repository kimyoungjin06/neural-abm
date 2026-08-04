#!/usr/bin/env python3
"""Materialize private G4 predecision risk rows and the author-cold guard."""

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
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_f_g4_private_membership_materialization_v1.yaml"
CONFIG_SCHEMA = "pivot_feedback_f_g4_private_membership_materialization.v1"

RISK_SCHEMA = pa.schema(
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
        pa.field("prior_target_active_peer_count", pa.uint64(), nullable=False),
        pa.field("network_leaveout_surge_present", pa.bool_(), nullable=False),
    ]
)

GUARD_SCHEMA = pa.schema(
    [
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("first_g4_decision_year", pa.uint16(), nullable=False),
        pa.field("last_g4_decision_year", pa.uint16(), nullable=False),
        pa.field("g4_episode_count", pa.uint16(), nullable=False),
        pa.field("exclude_from_discovery", pa.bool_(), nullable=False),
        pa.field("exclude_from_g1", pa.bool_(), nullable=False),
        pa.field("exclude_from_model_selection", pa.bool_(), nullable=False),
    ]
)

EPISODE_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.uint16(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("subfield_id", pa.uint16(), nullable=False),
        pa.field("decision_year", pa.uint16(), nullable=False),
        pa.field("expected_risk_opportunity_count", pa.uint64(), nullable=False),
        pa.field("materialized_risk_opportunity_count", pa.uint64(), nullable=False),
        pa.field("expected_network_leaveout_surge_count", pa.uint64(), nullable=False),
        pa.field("materialized_network_leaveout_surge_count", pa.uint64(), nullable=False),
        pa.field("risk_count_closure", pa.bool_(), nullable=False),
        pa.field("surge_count_closure", pa.bool_(), nullable=False),
        pa.field("sealed", pa.bool_(), nullable=False),
        pa.field("operator_only", pa.bool_(), nullable=False),
        pa.field("author_cold_guard_bound", pa.bool_(), nullable=False),
    ]
)


class G4PrivateMembershipBuildError(RuntimeError):
    """Fail-closed G4 private membership materialization error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G4PrivateMembershipBuildError(message)


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
        "private materialization not authorized",
    )
    sources = _mapping(config["bound_design_sources"], "bound sources")
    summary_binding = _mapping(sources["atomic_partition_summary"], "atomic summary")
    summary_path = _safe_repo_path(summary_binding["relative_path"])
    require(_sha256(summary_path) == summary_binding["sha256"], "atomic summary SHA")
    summary = _json(summary_path)
    require(summary.get("schema_version") == summary_binding["required_schema_version"], "atomic summary schema")
    require(summary.get("status") == summary_binding["required_status"], "atomic summary status")
    require(summary["fail_closed_state"]["pass_f"] is False, "atomic summary passed F")
    scientific_binding = _mapping(sources["scientific_sections"], "scientific sections")
    scientific_path = _safe_repo_path(scientific_binding["relative_path"])
    require(_sha256(scientific_path) == scientific_binding["sha256"], "scientific sections SHA")
    scientific = _yaml(scientific_path)
    require(scientific.get("schema_version") == scientific_binding["required_schema_version"], "scientific sections schema")
    section = scientific["sections"][scientific_binding["required_section"]]
    require(section["atomicity"]["g4_authors_removed_from_all_earlier_fit_or_selection_rows"] is True, "author-cold rule not frozen")
    return {
        "authorization_relative_path": authorization["relative_path"],
        "authorization_sha256": authorization["sha256"],
        "atomic_summary_relative_path": summary_binding["relative_path"],
        "atomic_summary_sha256": summary_binding["sha256"],
        "scientific_sections_relative_path": scientific_binding["relative_path"],
        "scientific_sections_sha256": scientific_binding["sha256"],
    }


def _load_parent_table(
    root: Path,
    binding: Mapping[str, Any],
    *,
    receipt_authorization_key: str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    root = root.resolve(strict=True)
    require(root.name == binding["expected_root_basename"], "parent basename")
    require(root.is_dir() and not root.is_symlink(), "parent root")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "parent receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "parent receipt schema")
    require(receipt.get("status") == binding["required_status"], "parent receipt status")
    if receipt_authorization_key is not None:
        require(receipt.get(receipt_authorization_key) is True, "parent materialization authorization")
    for key in ("pass_f", "pass_g0", "g4_open", "model_fitted"):
        require(receipt.get(key) is False, f"unsafe parent state: {key}")
    paths = {}
    for prefix in ("episode", "family", "partition", "slot"):
        basename_key = f"{prefix}_basename"
        if basename_key not in binding:
            continue
        path = root / str(binding[basename_key])
        require(_sha256(path) == binding[f"{prefix}_sha256"], f"parent {prefix} SHA")
        require(_schema_version(path) == binding[f"{prefix}_schema_version"], f"parent {prefix} schema")
        require(pq.ParquetFile(path).metadata.num_rows == binding[f"{prefix}_rows"], f"parent {prefix} rows")
        paths[prefix] = path
    if "verification_path" in binding:
        verification_path = Path(str(binding["verification_path"])).resolve(strict=True)
        require(_sha256(verification_path) == binding["verification_sha256"], "parent verification SHA")
        require(_json(verification_path).get("status") == binding["verification_status"], "parent verification status")
    return receipt, paths


def _load_g4_episodes(
    atomic_root: Path,
    final_panel_root: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    atomic_binding = _mapping(config["verified_atomic_partition_parent"], "atomic parent")
    _, atomic_paths = _load_parent_table(
        atomic_root,
        atomic_binding,
        receipt_authorization_key="operator_only_private_holdout_materialization_authorized_next",
    )
    final_binding = _mapping(config["verified_final_panel_parent"], "final-panel parent")
    final_receipt, final_paths = _load_parent_table(final_panel_root, final_binding)
    require(final_receipt.get("final_panel_conformance_complete") is True, "final-panel state")
    atomic_rows = pq.read_table(atomic_paths["episode"]).to_pylist()
    final_rows = {
        int(row["episode_id"]): dict(row)
        for row in pq.read_table(final_paths["episode"]).to_pylist()
    }
    contract = _mapping(config["private_membership_contract"], "membership contract")
    g4_rows = [
        dict(row)
        for row in atomic_rows
        if row["registered_partition_id"] == contract["partition_id"]
    ]
    require(len(g4_rows) == contract["episode_count"] == 37, "G4 episode count")
    require({int(row["field_id"]) for row in g4_rows} == {contract["field_id"]}, "G4 field")
    require({int(row["decision_year"]) for row in g4_rows} <= set(contract["entry_years"]), "G4 years")
    episodes = []
    for row in sorted(g4_rows, key=lambda item: int(item["episode_id"])):
        episode_id = int(row["episode_id"])
        final = final_rows[episode_id]
        require(final["sealed_slot"] is True and final["leaveout_surge_present"] is True, "G4 final-panel row")
        for key in ("field_id", "topic_id", "subfield_id", "decision_year"):
            require(int(row[key]) == int(final[key]), f"G4 parent closure: {episode_id} {key}")
        episodes.append(
            {
                "episode_id": episode_id,
                "field_id": int(row["field_id"]),
                "topic_id": int(row["topic_id"]),
                "subfield_id": int(row["subfield_id"]),
                "decision_year": int(row["decision_year"]),
                "field_year_median": float(final["aligned_field_year_median"]),
                "field_year_mad": float(final["aligned_field_year_mad"]),
                "expected_risk_opportunity_count": int(final["risk_opportunity_count"]),
                "expected_network_leaveout_surge_count": int(final["network_leaveout_surge_opportunity_count"]),
            }
        )
    require(sum(row["expected_risk_opportunity_count"] for row in episodes) == contract["expected_risk_opportunity_rows"], "G4 risk total")
    return episodes, {
        "atomic_receipt_sha256": atomic_binding["receipt_sha256"],
        "atomic_episode_sha256": atomic_binding["episode_sha256"],
        "atomic_verification_sha256": atomic_binding["verification_sha256"],
        "final_panel_receipt_sha256": final_binding["receipt_sha256"],
        "final_panel_episode_sha256": final_binding["episode_sha256"],
    }


def _validate_source_root(root: Path, binding: Mapping[str, Any]) -> Path:
    root = root.resolve(strict=True)
    require(root.name == binding["expected_root_basename"], "source root basename")
    require(root.is_dir() and not root.is_symlink(), "source root")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "source receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "source receipt schema")
    require(receipt.get("status") == binding["required_status"], "source receipt status")
    artifact_root = root / str(binding["artifact_prefix"])
    require(artifact_root.is_dir() and not artifact_root.is_symlink(), "source artifact root")
    files = sorted(artifact_root.glob("*.parquet"))
    require(len(files) == binding["artifact_count"], "source artifact count")
    require(all(path.is_file() and not path.is_symlink() for path in files), "source artifact type")
    return artifact_root / "*.parquet"


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


def _episode_table(episodes: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("episode_id", pa.uint16(), nullable=False),
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("subfield_id", pa.uint16(), nullable=False),
            pa.field("decision_year", pa.uint16(), nullable=False),
            pa.field("field_year_median", pa.float64(), nullable=False),
            pa.field("field_year_mad", pa.float64(), nullable=False),
            pa.field("expected_risk_opportunity_count", pa.uint64(), nullable=False),
            pa.field("expected_network_leaveout_surge_count", pa.uint64(), nullable=False),
        ]
    )
    return pa.Table.from_pylist([dict(row) for row in episodes], schema=schema)


def _export_reader(
    reader: pa.RecordBatchReader,
    path: Path,
    schema: pa.Schema,
    version: str,
) -> int:
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


def _run_materialization(
    *,
    database: Path,
    episodes: Sequence[Mapping[str, Any]],
    work_glob: Path,
    event_glob: Path,
    data_root: Path,
    config: Mapping[str, Any],
    threads: int,
    memory_limit_gb: int,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    con = duckdb.connect(str(database))
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB'")
    spill = database.parent / "duckdb-spill"
    spill.mkdir(mode=0o700)
    con.execute(f"PRAGMA temp_directory='{_sql_path(spill)}'")
    con.register("episode_input", _episode_table(episodes))
    con.execute("CREATE TABLE episode AS SELECT * FROM episode_input")
    work = _sql_path(work_glob)
    event = _sql_path(event_glob)
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
        CREATE TABLE target_event AS
        SELECT DISTINCT e.author_numeric_id AS author_id,
               e.work_numeric_id AS work_id,
               e.publication_year,
               e.declared_authors_count,
               e.primary_field_id AS field_id,
               w.primary_topic_id AS topic_id
        FROM read_parquet('{event}') e
        JOIN read_parquet('{work}') w ON w.work_id_numeric=e.work_numeric_id
        JOIN (SELECT DISTINCT field_id, topic_id FROM episode) p
          ON p.field_id=e.primary_field_id AND p.topic_id=w.primary_topic_id
        WHERE e.publication_year<=2021 AND {eligible}
        """
    )
    con.execute(
        f"""
        CREATE TABLE modern_event AS
        SELECT e.author_numeric_id AS author_id,
               e.work_numeric_id AS work_id,
               e.publication_year,
               e.declared_authors_count,
               e.primary_field_id AS field_id
        FROM read_parquet('{event}') e
        JOIN read_parquet('{work}') w ON w.work_id_numeric=e.work_numeric_id
        WHERE e.publication_year BETWEEN 2015 AND 2021
          AND e.primary_field_id BETWEEN 11 AND 36
          AND {eligible}
        """
    )
    con.execute("CREATE INDEX modern_author_year ON modern_event(author_id, publication_year)")
    con.execute("CREATE INDEX modern_work ON modern_event(work_id)")
    con.execute("CREATE INDEX target_topic_author ON target_event(field_id, topic_id, author_id)")
    con.execute(
        """
        CREATE TABLE target_first AS
        SELECT ep.episode_id, te.author_id, min(te.publication_year) AS first_target_year
        FROM episode ep JOIN target_event te
          ON te.field_id=ep.field_id AND te.topic_id=ep.topic_id
        WHERE te.publication_year<=ep.decision_year-1
        GROUP BY ep.episode_id, te.author_id
        """
    )
    con.execute(
        """
        CREATE TABLE risk_opportunity AS
        WITH active AS (
          SELECT DISTINCT ep.episode_id, me.author_id
          FROM episode ep JOIN modern_event me ON me.field_id=ep.field_id
          WHERE me.publication_year BETWEEN ep.decision_year-3 AND ep.decision_year-1
        )
        SELECT a.episode_id, a.author_id
        FROM active a LEFT JOIN target_first tf
          ON tf.episode_id=a.episode_id AND tf.author_id=a.author_id
        WHERE tf.author_id IS NULL
        """
    )
    con.execute("CREATE INDEX risk_author ON risk_opportunity(author_id, episode_id)")
    con.execute(
        """
        CREATE TABLE target_window_author AS
        SELECT ep.episode_id, te.work_id, te.author_id, te.publication_year,
               CASE WHEN te.publication_year BETWEEN ep.decision_year-6 AND ep.decision_year-4 THEN 0 ELSE 1 END AS window_code
        FROM episode ep JOIN target_event te
          ON te.field_id=ep.field_id AND te.topic_id=ep.topic_id
        WHERE te.publication_year BETWEEN ep.decision_year-6 AND ep.decision_year-1
        """
    )
    con.execute(
        """
        CREATE TABLE attention_work AS
        SELECT twa.episode_id, twa.work_id, min(twa.window_code) AS window_code
        FROM target_window_author twa
        JOIN target_first tf ON tf.episode_id=twa.episode_id AND tf.author_id=twa.author_id
        JOIN episode ep ON ep.episode_id=twa.episode_id
        GROUP BY twa.episode_id, twa.work_id, ep.decision_year
        HAVING count(*)=count(*) FILTER (WHERE tf.first_target_year<ep.decision_year-6)
        """
    )
    focal_incidence = int(
        con.execute(
            """
            SELECT count(*) FROM (
              SELECT DISTINCT r.episode_id, r.author_id, aw.work_id
              FROM risk_opportunity r
              JOIN target_window_author twa ON twa.episode_id=r.episode_id AND twa.author_id=r.author_id
              JOIN attention_work aw ON aw.episode_id=twa.episode_id AND aw.work_id=twa.work_id
            )
            """
        ).fetchone()[0]
    )
    require(focal_incidence == 0, "risk anti-target focal incidence")
    con.execute(
        """
        CREATE TABLE prior_focal_work AS
        SELECT DISTINCT r.episode_id, r.author_id AS focal_author_id, me.work_id
        FROM risk_opportunity r
        JOIN episode ep ON ep.episode_id=r.episode_id
        JOIN modern_event me ON me.author_id=r.author_id
        WHERE me.publication_year BETWEEN ep.decision_year-5 AND ep.decision_year-1
        """
    )
    con.execute(
        """
        CREATE TABLE prior_target_peer AS
        SELECT DISTINCT pf.episode_id, pf.focal_author_id, co.author_id AS peer_author_id
        FROM prior_focal_work pf
        JOIN modern_event co ON co.work_id=pf.work_id
        JOIN target_first tf ON tf.episode_id=pf.episode_id AND tf.author_id=co.author_id
        WHERE co.author_id<>pf.focal_author_id
        """
    )
    con.execute("CREATE INDEX peer_episode_author ON prior_target_peer(episode_id, peer_author_id)")
    con.execute(
        """
        CREATE TABLE excluded_attention_work AS
        SELECT DISTINCT pp.episode_id, pp.focal_author_id, aw.work_id, aw.window_code
        FROM prior_target_peer pp
        JOIN target_window_author twa
          ON twa.episode_id=pp.episode_id AND twa.author_id=pp.peer_author_id
        JOIN attention_work aw
          ON aw.episode_id=twa.episode_id AND aw.work_id=twa.work_id
        """
    )
    con.execute(
        """
        CREATE TABLE risk_measurement AS
        WITH base AS (
          SELECT episode_id,
                 count(*) FILTER (WHERE window_code=0) AS base_reference,
                 count(*) FILTER (WHERE window_code=1) AS base_recent
          FROM attention_work GROUP BY episode_id
        ), excluded AS (
          SELECT episode_id, focal_author_id,
                 count(*) FILTER (WHERE window_code=0) AS excluded_reference,
                 count(*) FILTER (WHERE window_code=1) AS excluded_recent
          FROM excluded_attention_work GROUP BY episode_id, focal_author_id
        ), peers AS (
          SELECT episode_id, focal_author_id, count(*) AS peer_count
          FROM prior_target_peer GROUP BY episode_id, focal_author_id
        )
        SELECT r.episode_id::USMALLINT AS episode_id,
               r.author_id::UBIGINT AS author_numeric_id,
               ep.field_id::UTINYINT AS field_id,
               ep.topic_id::UINTEGER AS topic_id,
               ep.subfield_id::USMALLINT AS subfield_id,
               ep.decision_year::USMALLINT AS decision_year,
               coalesce(b.base_reference,0)::UBIGINT AS base_reference_count,
               coalesce(b.base_recent,0)::UBIGINT AS base_recent_count,
               coalesce(x.excluded_reference,0)::UBIGINT AS excluded_reference_count,
               coalesce(x.excluded_recent,0)::UBIGINT AS excluded_recent_count,
               (coalesce(b.base_reference,0)-coalesce(x.excluded_reference,0))::UBIGINT AS network_leaveout_reference_count,
               (coalesce(b.base_recent,0)-coalesce(x.excluded_recent,0))::UBIGINT AS network_leaveout_recent_count,
               coalesce(p.peer_count,0)::UBIGINT AS prior_target_active_peer_count,
               (
                 (coalesce(b.base_recent,0)-coalesce(x.excluded_recent,0))
                   > (coalesce(b.base_reference,0)-coalesce(x.excluded_reference,0))
                 AND ln(1+(coalesce(b.base_recent,0)-coalesce(x.excluded_recent,0))/3.0)
                   - ln(1+(coalesce(b.base_reference,0)-coalesce(x.excluded_reference,0))/3.0)>0
                 AND 0.6744897501960817*(
                   (ln(1+(coalesce(b.base_recent,0)-coalesce(x.excluded_recent,0))/3.0)
                    -ln(1+(coalesce(b.base_reference,0)-coalesce(x.excluded_reference,0))/3.0))
                   -ep.field_year_median
                 )/ep.field_year_mad>=3.0
               )::BOOLEAN AS network_leaveout_surge_present
        FROM risk_opportunity r
        JOIN episode ep ON ep.episode_id=r.episode_id
        LEFT JOIN base b ON b.episode_id=r.episode_id
        LEFT JOIN excluded x ON x.episode_id=r.episode_id AND x.focal_author_id=r.author_id
        LEFT JOIN peers p ON p.episode_id=r.episode_id AND p.focal_author_id=r.author_id
        ORDER BY r.episode_id, r.author_id
        """
    )
    bad = int(
        con.execute(
            """
            SELECT count(*) FROM risk_measurement
            WHERE network_leaveout_reference_count>base_reference_count
               OR network_leaveout_recent_count>base_recent_count
               OR excluded_reference_count>base_reference_count
               OR excluded_recent_count>base_recent_count
            """
        ).fetchone()[0]
    )
    require(bad == 0, "network leaveout monotonicity")
    counts = {
        int(row[0]): {"risk": int(row[1]), "surge": int(row[2])}
        for row in con.execute(
            """
            SELECT episode_id, count(*)::UBIGINT,
                   count(*) FILTER (WHERE network_leaveout_surge_present)::UBIGINT
            FROM risk_measurement GROUP BY episode_id ORDER BY episode_id
            """
        ).fetchall()
    }
    require(len(counts) == len(episodes), "risk episode closure")

    risk_root = data_root / "g4_risk_measurement"
    risk_root.mkdir(mode=0o700)
    risk_records = []
    for episode in episodes:
        episode_id = int(episode["episode_id"])
        path = risk_root / f"episode-{episode_id:04d}.parquet"
        reader = con.execute(
            "SELECT * FROM risk_measurement WHERE episode_id=? ORDER BY author_numeric_id",
            [episode_id],
        ).fetch_record_batch(1_000_000)
        rows = _export_reader(
            reader,
            path,
            RISK_SCHEMA,
            config["output_contract"]["risk_schema_version"],
        )
        require(rows == counts[episode_id]["risk"], f"risk export closure: {episode_id}")
        risk_records.append({"episode_id": episode_id, "relative_path": path.relative_to(data_root.parent).as_posix(), "rows": rows})

    con.execute(
        """
        CREATE TABLE author_guard AS
        SELECT author_numeric_id,
               min(decision_year)::USMALLINT AS first_g4_decision_year,
               max(decision_year)::USMALLINT AS last_g4_decision_year,
               count(DISTINCT episode_id)::USMALLINT AS g4_episode_count,
               true::BOOLEAN AS exclude_from_discovery,
               true::BOOLEAN AS exclude_from_g1,
               true::BOOLEAN AS exclude_from_model_selection
        FROM risk_measurement
        GROUP BY author_numeric_id
        ORDER BY author_numeric_id
        """
    )
    guard_path = data_root / "g4_author_cold_guard.parquet"
    guard_rows = _export_reader(
        con.execute("SELECT * FROM author_guard ORDER BY author_numeric_id").fetch_record_batch(1_000_000),
        guard_path,
        GUARD_SCHEMA,
        config["output_contract"]["guard_schema_version"],
    )
    require(guard_rows > 0, "empty author-cold guard")
    target_event_rows = int(con.execute("SELECT count(*) FROM target_event").fetchone()[0])
    modern_event_rows = int(con.execute("SELECT count(*) FROM modern_event").fetchone()[0])
    con.close()
    manifests = []
    for episode in episodes:
        episode_id = int(episode["episode_id"])
        observed = counts[episode_id]
        risk_close = observed["risk"] == int(episode["expected_risk_opportunity_count"])
        surge_close = observed["surge"] == int(episode["expected_network_leaveout_surge_count"])
        require(risk_close, f"parent risk count closure: {episode_id}")
        require(surge_close, f"parent surge count closure: {episode_id}")
        manifests.append(
            {
                "episode_id": episode_id,
                "field_id": int(episode["field_id"]),
                "topic_id": int(episode["topic_id"]),
                "subfield_id": int(episode["subfield_id"]),
                "decision_year": int(episode["decision_year"]),
                "expected_risk_opportunity_count": int(episode["expected_risk_opportunity_count"]),
                "materialized_risk_opportunity_count": observed["risk"],
                "expected_network_leaveout_surge_count": int(episode["expected_network_leaveout_surge_count"]),
                "materialized_network_leaveout_surge_count": observed["surge"],
                "risk_count_closure": risk_close,
                "surge_count_closure": surge_close,
                "sealed": True,
                "operator_only": True,
                "author_cold_guard_bound": True,
            }
        )
    accounting = {
        "target_event_rows": target_event_rows,
        "modern_event_rows": modern_event_rows,
        "risk_opportunity_rows": sum(row["risk"] for row in counts.values()),
        "network_leaveout_surge_rows": sum(row["surge"] for row in counts.values()),
        "author_cold_guard_rows": guard_rows,
        "focal_excluded_attention_work_incidences": focal_incidence,
        "risk_file_count": len(risk_records),
        "raw_work_ids_emitted": False,
        "entry_or_persistence_labels_emitted": False,
        "postdecision_target_events_emitted": False,
        "realized_citation_effects_emitted": False,
        "model_scores_emitted": False,
    }
    return manifests, guard_rows, accounting


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema, version: str) -> None:
    versioned = schema.with_metadata({b"schema_version": version.encode("utf-8")})
    pq.write_table(
        pa.Table.from_pylist([dict(row) for row in rows], schema=versioned),
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    os.chmod(path, 0o600)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
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


def build(
    *,
    config_path: Path,
    atomic_root: Path,
    final_panel_root: Path,
    cutoff_root: Path,
    author_history_root: Path,
    output_root: Path,
    threads: int,
    memory_limit_gb: int,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    repository_bindings = _validate_repository_bindings(config)
    episodes, parent_bindings = _load_g4_episodes(atomic_root, final_panel_root, config)
    output_root = output_root.resolve()
    output_contract = _mapping(config["output_contract"], "output contract")
    require(output_root.name == output_contract["root_basename"], "output basename")
    require(not output_root.exists() and output_root.parent.is_dir(), "output directory must be new")
    preflight = _resource_preflight(output_root.parent, config)
    normalized = _mapping(config["normalized_inputs"], "normalized inputs")
    work_glob = _validate_source_root(cutoff_root, normalized["cutoff_work_metadata"])
    event_glob = _validate_source_root(author_history_root, normalized["author_history_events"])

    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    scratch = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.scratch-", dir=output_root.parent))
    os.chmod(staging, 0o700)
    os.chmod(scratch, 0o700)
    data_root = staging / "private"
    data_root.mkdir(mode=0o700)
    manifest_rows, guard_rows, accounting = _run_materialization(
        database=scratch / "g4-private-membership.duckdb",
        episodes=episodes,
        work_glob=work_glob,
        event_glob=event_glob,
        data_root=data_root,
        config=config,
        threads=threads,
        memory_limit_gb=memory_limit_gb,
    )
    episode_path = data_root / "g4_episode_manifest.parquet"
    _write_parquet(
        episode_path,
        manifest_rows,
        EPISODE_SCHEMA,
        output_contract["episode_schema_version"],
    )
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
        "parent_bindings": parent_bindings,
        "normalized_input_bindings": {
            "cutoff_receipt_sha256": normalized["cutoff_work_metadata"]["receipt_sha256"],
            "author_history_receipt_sha256": normalized["author_history_events"]["receipt_sha256"],
        },
        "resource_preflight": preflight,
        "accounting": accounting,
        "episode_manifest": manifest_rows,
        "author_cold_guard": {
            "guard_rows": guard_rows,
            "exact_projection_of_private_risk_rows": True,
            "exclude_from_discovery": True,
            "exclude_from_g1": True,
            "exclude_from_model_selection": True,
            "earlier_row_filter_application_status": "PENDING_FUTURE_ROW_MATERIALIZATION",
        },
        "completion": {
            "g4_private_predecision_membership_complete": True,
            "g4_author_cold_guard_complete": True,
            "earlier_partition_guard_application_complete": False,
            "complete_private_g4_artifact_set": False,
            "entry_or_persistence_labels_materialized": False,
            "postdecision_target_source_materialized": False,
            "holdout_preflight_complete": False,
            "holdout_byte_seal_complete": False,
            "confirmatory_or_holdout_outcome_rows_read": 0,
            "model_fitted": False,
            "g4_open": False,
            "pass_f": False,
            "pass_g0": False,
        },
        "authorization": config["authorization_after_pass"],
        "claim_boundary": config["claim_boundary"],
    }
    report_path = staging / "g4_private_membership_materialization.json"
    _write_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = f"""# G4 private predecision membership materialization v1

Status: `{report['status']}`

The operator-only tree contains 37 field-27 episode files with
{accounting['risk_opportunity_rows']:,} predecision author-token risk rows and
an exact distinct author-cold guard. Every episode closes to the independently
verified final-panel risk and network-leaveout surge counts.

The private rows contain no raw work IDs, entry or persistence labels,
postdecision target events, realized citation effects, or model scores. The
guard must be anti-joined when discovery, G1, or model-selection rows are later
materialized. Those earlier rows do not yet exist, so guard application remains
pending.

This is a real private membership foundation, not the complete G4 scoring
package. Explicit-list preflight and byte sealing remain unauthorized.
`pass_f=false`, `pass_g0=false`; G4 is unopened.
"""
    markdown_path = staging / "G4_PRIVATE_MEMBERSHIP_MATERIALIZATION.md"
    _write_text(markdown_path, markdown)

    artifact_paths = [
        path
        for path in staging.rglob("*")
        if path.is_file() and path.name != "g4_private_membership_materialization.outputs.json"
    ]
    artifacts = [_artifact(path, staging) for path in sorted(artifact_paths)]
    tree = [(row["relative_path"], row["bytes"], row["sha256"]) for row in artifacts]
    receipt = {
        "schema_version": output_contract["receipt_schema_version"],
        "status": output_contract["passing_status"],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "artifact_tree_sha256": hashlib.sha256(_canonical_bytes(tree)).hexdigest(),
        "receipt_written_last": True,
        "g4_private_predecision_membership_complete": True,
        "g4_author_cold_guard_complete": True,
        "earlier_partition_guard_application_complete": False,
        "complete_private_g4_artifact_set": False,
        "entry_or_persistence_labels_materialized": False,
        "postdecision_target_source_materialized": False,
        "holdout_preflight_complete": False,
        "holdout_byte_seal_complete": False,
        "confirmatory_or_holdout_outcome_rows_read": 0,
        "model_score_read": False,
        "model_fitted": False,
        "g4_open": False,
        "pass_f": False,
        "pass_g0": False,
    }
    receipt_path = staging / "g4_private_membership_materialization.outputs.json"
    _write_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    shutil.rmtree(scratch)
    staging.rename(output_root)
    return {
        "status": receipt["status"],
        "output_root": str(output_root),
        "risk_episode_files": output_contract["risk_file_count"],
        "risk_opportunity_rows": accounting["risk_opportunity_rows"],
        "g4_private_predecision_membership_complete": True,
        "g4_author_cold_guard_complete": True,
        "complete_private_g4_artifact_set": False,
        "holdout_preflight_complete": False,
        "holdout_byte_seal_complete": False,
        "confirmatory_or_holdout_outcome_rows_read": 0,
        "pass_f": False,
        "pass_g0": False,
        "g4_open": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--atomic-root", type=Path, required=True)
    parser.add_argument("--final-panel-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--memory-limit-gb", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build(
        config_path=args.config,
        atomic_root=args.atomic_root,
        final_panel_root=args.final_panel_root,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        output_root=args.output_root,
        threads=args.threads,
        memory_limit_gb=args.memory_limit_gb,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
