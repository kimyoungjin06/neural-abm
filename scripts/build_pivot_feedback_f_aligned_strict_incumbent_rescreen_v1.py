#!/usr/bin/env python3
"""Build the outcome-blind strict-incumbent aligned all-topic F rescreen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import build_pivot_feedback_d0_certified_author_opportunity_rebuild_v1 as d0


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_f_aligned_strict_incumbent_rescreen_v1.yaml"
CONFIG_SCHEMA = "pivot_feedback_f_aligned_strict_incumbent_rescreen.v1"

SURFACE_SCHEMA = pa.schema(
    [
        pa.field("slot_id", pa.string(), nullable=False),
        pa.field("temporal_partition_id", pa.string(), nullable=False),
        pa.field("field_attempt_order", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("topic_name", pa.string(), nullable=False),
        pa.field("subfield_id", pa.uint16(), nullable=False),
        pa.field("subfield_name", pa.string(), nullable=False),
        pa.field("decision_year", pa.uint16(), nullable=False),
        pa.field("reference_count", pa.uint64(), nullable=False),
        pa.field("recent_count", pa.uint64(), nullable=False),
        pa.field("acceleration", pa.float64(), nullable=False),
        pa.field("field_year_median", pa.float64(), nullable=False),
        pa.field("field_year_mad", pa.float64(), nullable=False),
        pa.field("modified_z", pa.float64(), nullable=True),
        pa.field("strict_attention_candidate", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": b"pivot_feedback_f_aligned_strict_incumbent_surface.v1"},
)

CANDIDATE_SCHEMA = pa.schema(
    [
        *SURFACE_SCHEMA,
        pa.field("aligned_selected_field", pa.bool_(), nullable=False),
        pa.field("retained_for_aligned_panel", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": b"pivot_feedback_f_aligned_strict_incumbent_candidate.v1"},
)

ATTEMPT_SCHEMA = pa.schema(
    [
        pa.field("slot_id", pa.string(), nullable=False),
        pa.field("temporal_partition_id", pa.string(), nullable=False),
        pa.field("field_attempt_order", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("candidate_episode_count", pa.uint32(), nullable=False),
        pa.field("candidate_topic_count", pa.uint32(), nullable=False),
        pa.field("subfield_family_count", pa.uint16(), nullable=False),
        pa.field("minimum_subfield_families_required", pa.uint16(), nullable=False),
        pa.field("aligned_screen_eligible", pa.bool_(), nullable=False),
        pa.field("selected_for_slot", pa.bool_(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("sealed_slot", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": b"pivot_feedback_f_aligned_strict_incumbent_field_attempt.v1"},
)

SLOT_SCHEMA = pa.schema(
    [
        pa.field("slot_id", pa.string(), nullable=False),
        pa.field("temporal_partition_id", pa.string(), nullable=False),
        pa.field("frozen_field_order", pa.list_(pa.uint8(), 3), nullable=False),
        pa.field("aggregate_selected_field_id", pa.uint8(), nullable=False),
        pa.field("aligned_selected_field_id", pa.uint8(), nullable=True),
        pa.field("aligned_selected_attempt_order", pa.uint8(), nullable=True),
        pa.field("aligned_candidate_episode_count", pa.uint32(), nullable=False),
        pa.field("aligned_subfield_family_count", pa.uint16(), nullable=False),
        pa.field("minimum_subfield_families_required", pa.uint16(), nullable=False),
        pa.field("slot_pass", pa.bool_(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("sealed_slot", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": b"pivot_feedback_f_aligned_strict_incumbent_slot_assignment.v1"},
)

CLOSURE_SCHEMA = pa.schema(
    [
        pa.field("predecessor_episode_id", pa.uint16(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("decision_year", pa.uint16(), nullable=False),
        pa.field("expected_reference_count", pa.uint64(), nullable=False),
        pa.field("observed_reference_count", pa.uint64(), nullable=False),
        pa.field("expected_recent_count", pa.uint64(), nullable=False),
        pa.field("observed_recent_count", pa.uint64(), nullable=False),
        pa.field("exact_match", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": b"pivot_feedback_f_aligned_strict_incumbent_predecessor_closure.v1"},
)


class RescreenBuildError(RuntimeError):
    """Fail-closed aligned-rescreen build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RescreenBuildError(message)


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _safe_repo_path(relative: Any) -> Path:
    require(isinstance(relative, str) and relative, "empty repository path")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts, "unsafe repository path")
    path = (ROOT / pure).resolve(strict=True)
    path.relative_to(ROOT.resolve())
    require(path.is_file() and not path.is_symlink(), "invalid repository binding")
    return path


def _schema_version(path: Path) -> str:
    return (pq.ParquetFile(path).schema_arrow.metadata or {}).get(b"schema_version", b"").decode()


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["relative_path"]): dict(row) for row in receipt.get("artifacts", [])}


def _validate_authorization(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = _mapping(config["authorization_parent"], "authorization parent")
    path = _safe_repo_path(binding["relative_path"])
    require(_sha256(path) == binding["sha256"], "authorization parent SHA")
    document = _yaml(path)
    require(document.get("schema_version") == binding["required_schema_version"], "authorization parent schema")
    authorization = _mapping(document["authorization_after_verified_audit"], "parent authorization")
    require(authorization.get(binding["required_authorization"]) == binding["required_value"], "aligned rescreen not authorized")
    return {"relative_path": binding["relative_path"], "sha256": binding["sha256"]}


def _validate_audit_root(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink() and root.name == binding["expected_root_basename"], "audit root")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "audit receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "audit receipt schema")
    require(receipt.get("status") == binding["required_status"], "audit receipt status")
    verification = Path(str(binding["verification_path"])).resolve(strict=True)
    require(_sha256(verification) == binding["verification_sha256"], "audit verification SHA")
    require(_json(verification).get("status") == binding["verification_status"], "audit verification status")
    return {
        "root_basename": root.name,
        "receipt_sha256": binding["receipt_sha256"],
        "verification_sha256": binding["verification_sha256"],
    }


def _validate_assignment_root(
    root: Path, binding: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink() and root.name == binding["expected_root_basename"], "assignment root")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "assignment receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "assignment receipt schema")
    require(receipt.get("status") == binding["required_status"], "assignment receipt status")
    artifacts = _artifact_map(receipt)
    slot_path = root / str(binding["slot_basename"])
    candidate_path = root / str(binding["candidate_basename"])
    for path, sha, rows, schema in (
        (slot_path, binding["slot_sha256"], binding["slot_rows"], binding["slot_schema_version"]),
        (candidate_path, binding["candidate_sha256"], binding["candidate_rows"], binding["candidate_schema_version"]),
    ):
        require(_sha256(path) == sha, f"assignment artifact SHA: {path.name}")
        require(artifacts[path.name]["sha256"] == sha, f"assignment receipt artifact: {path.name}")
        require(pq.ParquetFile(path).metadata.num_rows == rows, f"assignment artifact rows: {path.name}")
        require(_schema_version(path) == schema, f"assignment artifact schema: {path.name}")
    return pq.read_table(slot_path).to_pylist(), candidate_path, {
        "root_basename": root.name,
        "receipt_sha256": binding["receipt_sha256"],
        "slot_sha256": binding["slot_sha256"],
        "candidate_sha256": binding["candidate_sha256"],
    }


def _validate_external_parquet(path: Path, binding: Mapping[str, Any]) -> tuple[Path, tuple[int, int, int, int, int]]:
    path = path.resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), "invalid external parquet")
    if "expected_basename" in binding:
        require(path.name == binding["expected_basename"], "external parquet basename")
    require(_sha256(path) == binding["sha256"], f"external parquet SHA: {path.name}")
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == binding["rows"], f"external parquet rows: {path.name}")
    if "schema_version" in binding:
        require(_schema_version(path) == binding["schema_version"], f"external parquet schema: {path.name}")
    stat = path.stat()
    return path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _validate_taxonomy(
    taxonomy_path: Path,
    binding: Mapping[str, Any],
    aggregate_candidate_path: Path,
    d0_overlap_path: Path,
) -> tuple[pa.Table, dict[str, Any], tuple[int, int, int, int, int]]:
    taxonomy_path, pre_stat = _validate_external_parquet(taxonomy_path, binding)
    manifest = Path(str(binding["manifest_path"])).resolve(strict=True)
    require(_sha256(manifest) == binding["manifest_sha256"], "taxonomy manifest SHA")
    manifest_json = _json(manifest)
    require(manifest_json.get("schema_version") == binding["schema_version_column"], "taxonomy manifest schema")
    require(manifest_json.get("package_version") == binding["package_version"], "taxonomy package version")
    con = duckdb.connect()
    taxonomy_sql = f"""
        SELECT DISTINCT
          cast(replace(primary_topic_id,'T','') AS UINTEGER) AS topic_id,
          primary_topic_name AS topic_name,
          cast(replace(field_id,'fields/','') AS UTINYINT) AS field_id,
          field_name,
          cast(replace(subfield_id,'subfields/','') AS USMALLINT) AS subfield_id,
          subfield_name
        FROM read_parquet('{_sql_path(taxonomy_path)}')
        WHERE has_primary_topic
    """
    version = con.execute(
        f"""SELECT count(DISTINCT schema_version), min(schema_version),
                    count(DISTINCT package_version), min(package_version),
                    count(DISTINCT taxonomy_source), min(taxonomy_source)
             FROM read_parquet('{_sql_path(taxonomy_path)}')"""
    ).fetchone()
    require(version == (1, binding["schema_version_column"], 1, binding["package_version"], 1, binding["taxonomy_source"]), "taxonomy version columns")
    table = con.execute(taxonomy_sql).fetch_arrow_table()
    require(len(table) == binding["distinct_topic_rows"], "taxonomy topic closure")
    require(len(set(table["topic_id"].to_pylist())) == len(table), "taxonomy topic uniqueness")
    require(len(set(table["field_id"].to_pylist())) == binding["distinct_field_rows"], "taxonomy field closure")
    require(len(set(table["subfield_id"].to_pylist())) == binding["distinct_subfield_rows"], "taxonomy subfield closure")
    con.register("taxonomy_map", table)
    overlap: dict[str, Any] = {}
    for label, path in (("aggregate_assignment", aggregate_candidate_path), ("d0_measurement", d0_overlap_path)):
        comparison = con.execute(
            f"""
            WITH frozen AS (
              SELECT DISTINCT field_id, topic_id, topic_name, subfield_id, subfield_name
              FROM read_parquet('{_sql_path(path)}')
            )
            SELECT count(*) AS mapping_count,
                   count(*) FILTER (WHERE t.topic_id IS NOT NULL) AS matched_count,
                   count(*) FILTER (WHERE t.topic_id IS NOT NULL
                     AND t.field_id=f.field_id AND t.topic_name=f.topic_name
                     AND t.subfield_id=f.subfield_id AND t.subfield_name=f.subfield_name) AS exact_count
            FROM frozen f LEFT JOIN taxonomy_map t USING(topic_id)
            """
        ).fetchone()
        require(comparison[0] == comparison[1] == comparison[2], f"taxonomy frozen overlap mismatch: {label}")
        overlap[label] = {"mapping_count": comparison[0], "exact_count": comparison[2]}
    con.close()
    return table, {
        "basename": taxonomy_path.name,
        "sha256": binding["sha256"],
        "manifest_sha256": binding["manifest_sha256"],
        "topic_count": len(table),
        "field_count": binding["distinct_field_rows"],
        "subfield_count": binding["distinct_subfield_rows"],
        "frozen_overlap": overlap,
    }, pre_stat


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _validate_frozen_slots(
    slots: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_id = {str(row["slot_id"]): dict(row) for row in slots}
    expected = list(contract["expected"])
    require(len(by_id) == len(expected) == 4, "frozen slot count")
    normalized: list[dict[str, Any]] = []
    for row in expected:
        slot_id = str(row["slot_id"])
        require(slot_id in by_id, f"frozen slot missing: {slot_id}")
        source = by_id[slot_id]
        order = [int(source["nominal_primary_field_id"]), *[int(value) for value in source["reserve_field_ids"]]]
        require(order == [int(value) for value in row["field_order"]], f"frozen field order: {slot_id}")
        require(str(source["temporal_partition_id"]) == row["temporal_partition_id"], f"frozen period: {slot_id}")
        normalized.append(
            {
                "slot_id": slot_id,
                "temporal_partition_id": str(source["temporal_partition_id"]),
                "field_order": order,
                "aggregate_selected_field_id": int(source["selected_field_id"]),
                "sealed_slot": bool(source["sealed"]),
            }
        )
    return normalized


def _frozen_field_year_table(slots: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> pa.Table:
    periods = _mapping(contract["periods"], "periods")
    rows = []
    for slot in slots:
        period = _mapping(periods[slot["temporal_partition_id"]], "period")
        for attempt, field_id in enumerate(slot["field_order"], 1):
            for year in range(int(period["decision_year_start"]), int(period["decision_year_end"]) + 1):
                rows.append(
                    {
                        "field_id": field_id,
                        "decision_year": year,
                        "slot_id": slot["slot_id"],
                        "temporal_partition_id": slot["temporal_partition_id"],
                        "field_attempt_order": attempt,
                    }
                )
    return pa.Table.from_pylist(rows)


def _run_rescreen(
    *,
    database: Path,
    spill: Path,
    work_glob: Path,
    event_glob: Path,
    taxonomy: pa.Table,
    frozen_field_year: pa.Table,
    predecessor_path: Path,
    primary: Mapping[str, Any],
    threads: int,
    memory_limit_gb: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    con = duckdb.connect(str(database))
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB'")
    con.execute(f"PRAGMA temp_directory='{_sql_path(spill)}'")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.register("frozen_field_year_input", frozen_field_year)
    con.register("taxonomy_map_input", taxonomy)
    con.execute("CREATE TABLE frozen_field_year AS SELECT * FROM frozen_field_year_input")
    fields = sorted(set(frozen_field_year["field_id"].to_pylist()))
    field_sql = ",".join(str(value) for value in fields)
    types = ",".join(str(int(value)) for value in primary["eligible_work_type_codes"])
    dispositions = ",".join(str(int(value)) for value in primary["allowed_temporal_dispositions"])
    work = _sql_path(work_glob)
    event = _sql_path(event_glob)
    con.execute(
        f"""
        CREATE TABLE eligible_work AS
        SELECT work_id_numeric AS work_id, publication_year, temporal_disposition,
               primary_field_id AS field_id, primary_topic_id AS topic_id
        FROM read_parquet('{work}')
        WHERE publication_year<={int(primary['latest_predecision_year'])}
          AND primary_field_id IN ({field_sql})
          AND primary_topic_id IS NOT NULL
          AND temporal_disposition IN ({dispositions})
          AND work_type_code IN ({types})
          AND NOT is_paratext
          AND coalesce(is_retracted,false)=false
        """
    )
    eligible_work_count = int(con.execute("SELECT count(*) FROM eligible_work").fetchone()[0])
    con.execute(
        f"""
        CREATE TABLE eligible_event AS
        SELECT e.author_numeric_id AS author_id, e.work_numeric_id AS work_id,
               w.publication_year, w.field_id, w.topic_id
        FROM read_parquet('{event}') e
        JOIN eligible_work w ON w.work_id=e.work_numeric_id
        WHERE e.publication_year=w.publication_year
          AND e.primary_field_id=w.field_id
          AND e.temporal_disposition=w.temporal_disposition
        """
    )
    eligible_event_count = int(con.execute("SELECT count(*) FROM eligible_event").fetchone()[0])
    con.execute(
        """
        CREATE TABLE author_topic_first AS
        SELECT field_id, topic_id, author_id,
               min(publication_year)::USMALLINT AS first_target_year
        FROM eligible_event GROUP BY field_id, topic_id, author_id
        """
    )
    author_topic_first_count = int(con.execute("SELECT count(*) FROM author_topic_first").fetchone()[0])
    con.execute(
        f"""
        CREATE TABLE strict_work AS
        SELECT e.work_id, e.publication_year, e.field_id, e.topic_id,
               (max(f.first_target_year)+7)::USMALLINT AS first_eligible_decision_year,
               count(*)::UINTEGER AS linked_author_rows,
               count(DISTINCT e.author_id)::UINTEGER AS linked_author_count
        FROM eligible_event e JOIN author_topic_first f USING(field_id, topic_id, author_id)
        WHERE e.publication_year BETWEEN {int(primary['earliest_window_year'])}
                                     AND {int(primary['latest_predecision_year'])}
        GROUP BY e.work_id, e.publication_year, e.field_id, e.topic_id
        """
    )
    strict_work_count = int(con.execute("SELECT count(*) FROM strict_work").fetchone()[0])
    duplicate_work_rows = int(con.execute("SELECT count(*) FROM strict_work WHERE linked_author_rows<>linked_author_count").fetchone()[0])
    require(duplicate_work_rows == 0, "duplicate linked-author rows")
    con.execute(
        """
        CREATE TABLE topic_field_year_count AS
        SELECT f.field_id, w.topic_id, f.decision_year, f.slot_id,
               f.temporal_partition_id, f.field_attempt_order,
               count(*) FILTER (WHERE w.publication_year BETWEEN f.decision_year-6 AND f.decision_year-4)::UBIGINT AS reference_count,
               count(*) FILTER (WHERE w.publication_year BETWEEN f.decision_year-3 AND f.decision_year-1)::UBIGINT AS recent_count
        FROM frozen_field_year f JOIN strict_work w
          ON w.field_id=f.field_id
         AND w.publication_year BETWEEN f.decision_year-6 AND f.decision_year-1
         AND w.first_eligible_decision_year<=f.decision_year
        GROUP BY f.field_id, w.topic_id, f.decision_year, f.slot_id,
                 f.temporal_partition_id, f.field_attempt_order
        """
    )
    multiplier = float(primary["modified_z_multiplier"])
    threshold = float(primary["modified_z_threshold_inclusive"])
    con.execute(
        f"""
        CREATE TABLE strict_surface_base AS
        WITH scored AS (
          SELECT *, ln(1+recent_count/3.0)-ln(1+reference_count/3.0) AS acceleration
          FROM topic_field_year_count WHERE reference_count>0
        ), centered AS (
          SELECT *, median(acceleration) OVER(PARTITION BY field_id,decision_year) AS field_year_median
          FROM scored
        ), scaled AS (
          SELECT *, median(abs(acceleration-field_year_median)) OVER(PARTITION BY field_id,decision_year) AS field_year_mad
          FROM centered
        ), normalized AS (
          SELECT *, CASE WHEN isfinite(field_year_mad) AND field_year_mad>0
                    THEN {multiplier}*(acceleration-field_year_median)/field_year_mad
                    ELSE NULL END AS modified_z
          FROM scaled
        )
        SELECT *, recent_count>reference_count AND acceleration>0
                  AND modified_z>={threshold} AS strict_attention_candidate
        FROM normalized
        """
    )
    con.execute("CREATE TABLE taxonomy_map AS SELECT * FROM taxonomy_map_input")
    missing_taxonomy = int(
        con.execute(
            "SELECT count(*) FROM strict_surface_base s ANTI JOIN taxonomy_map t USING(field_id,topic_id)"
        ).fetchone()[0]
    )
    require(missing_taxonomy == 0, "strict surface taxonomy gap")
    surface = con.execute(
        """
        SELECT s.slot_id, s.temporal_partition_id, s.field_attempt_order,
               s.field_id, s.topic_id, t.topic_name, t.subfield_id, t.subfield_name,
               s.decision_year, s.reference_count, s.recent_count, s.acceleration,
               s.field_year_median, s.field_year_mad, s.modified_z,
               s.strict_attention_candidate
        FROM strict_surface_base s JOIN taxonomy_map t USING(field_id,topic_id)
        ORDER BY s.slot_id,s.field_attempt_order,s.field_id,s.decision_year,t.subfield_id,s.topic_id
        """
    ).fetch_arrow_table().to_pylist()
    candidates = [dict(row) for row in surface if row["strict_attention_candidate"]]
    closure = con.execute(
        f"""
        WITH counted AS (
          SELECT o.episode_id,
                 count(w.work_id) FILTER (WHERE w.publication_year BETWEEN o.decision_year-6 AND o.decision_year-4)::UBIGINT AS observed_reference_count,
                 count(w.work_id) FILTER (WHERE w.publication_year BETWEEN o.decision_year-3 AND o.decision_year-1)::UBIGINT AS observed_recent_count
          FROM read_parquet('{_sql_path(predecessor_path)}') o
          LEFT JOIN strict_work w
            ON w.field_id=o.field_id AND w.topic_id=o.topic_id
           AND w.publication_year BETWEEN o.decision_year-6 AND o.decision_year-1
           AND w.first_eligible_decision_year<=o.decision_year
          GROUP BY o.episode_id
        )
        SELECT o.episode_id AS predecessor_episode_id,o.field_id,o.topic_id,o.decision_year,
               o.all_incumbent_reference_work_count AS expected_reference_count,
               c.observed_reference_count,
               o.all_incumbent_recent_work_count AS expected_recent_count,
               c.observed_recent_count,
               c.observed_reference_count=o.all_incumbent_reference_work_count
                 AND c.observed_recent_count=o.all_incumbent_recent_work_count AS exact_match
        FROM read_parquet('{_sql_path(predecessor_path)}') o JOIN counted c USING(episode_id)
        ORDER BY o.episode_id
        """
    ).fetch_arrow_table().to_pylist()
    require(len(closure) == 194 and all(row["exact_match"] for row in closure), "predecessor strict-count closure")
    group_stats = con.execute(
        """
        SELECT count(*) AS group_count,
               count(*) FILTER (WHERE NOT isfinite(field_year_mad) OR field_year_mad<=0) AS bad_mad_count
        FROM (SELECT DISTINCT field_id,decision_year,field_year_mad FROM strict_surface_base)
        """
    ).fetchone()
    accounting = {
        "eligible_work_count": eligible_work_count,
        "eligible_event_count": eligible_event_count,
        "author_topic_first_count": author_topic_first_count,
        "strict_work_count": strict_work_count,
        "duplicate_linked_author_work_count": duplicate_work_rows,
        "strict_surface_rows": len(surface),
        "normalization_group_count": int(group_stats[0]),
        "mad_zero_or_nonfinite_group_count": int(group_stats[1]),
        "candidate_rows": len(candidates),
        "candidate_topics": len({int(row["topic_id"]) for row in candidates}),
        "predecessor_episode_rows": len(closure),
        "predecessor_exact_count_rows": sum(bool(row["exact_match"]) for row in closure),
    }
    con.close()
    return surface, candidates, closure, accounting


def _select_slots(
    frozen_slots: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    minimum_families: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    attempts: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    selected_by_slot: dict[str, int] = {}
    for slot in frozen_slots:
        slot_candidates = [row for row in candidates if row["slot_id"] == slot["slot_id"]]
        summaries = []
        selected_field: int | None = None
        selected_order: int | None = None
        for order, field_id in enumerate(slot["field_order"], 1):
            rows = [row for row in slot_candidates if int(row["field_id"]) == field_id]
            family_count = len({int(row["subfield_id"]) for row in rows})
            eligible = family_count >= minimum_families
            if selected_field is None and eligible:
                selected_field, selected_order = field_id, order
            summaries.append((order, field_id, rows, family_count, eligible))
        for order, field_id, rows, family_count, eligible in summaries:
            selected = field_id == selected_field
            attempts.append(
                {
                    "slot_id": slot["slot_id"],
                    "temporal_partition_id": slot["temporal_partition_id"],
                    "field_attempt_order": order,
                    "field_id": field_id,
                    "candidate_episode_count": len(rows),
                    "candidate_topic_count": len({int(row["topic_id"]) for row in rows}),
                    "subfield_family_count": family_count,
                    "minimum_subfield_families_required": minimum_families,
                    "aligned_screen_eligible": eligible,
                    "selected_for_slot": selected,
                    "status": "SELECTED_FIRST_CONFORMING_FROZEN_FIELD" if selected else ("CONFORMING_AFTER_EARLIER_SELECTION_NOT_SELECTED" if eligible else "INSUFFICIENT_SUBFIELD_FAMILIES"),
                    "sealed_slot": slot["sealed_slot"],
                }
            )
        selected_rows = [row for row in slot_candidates if int(row["field_id"]) == selected_field] if selected_field is not None else []
        selected_families = len({int(row["subfield_id"]) for row in selected_rows})
        slot_pass = selected_field is not None
        if slot_pass:
            selected_by_slot[str(slot["slot_id"])] = int(selected_field)
        slot_rows.append(
            {
                "slot_id": slot["slot_id"],
                "temporal_partition_id": slot["temporal_partition_id"],
                "frozen_field_order": slot["field_order"],
                "aggregate_selected_field_id": slot["aggregate_selected_field_id"],
                "aligned_selected_field_id": selected_field,
                "aligned_selected_attempt_order": selected_order,
                "aligned_candidate_episode_count": len(selected_rows),
                "aligned_subfield_family_count": selected_families,
                "minimum_subfield_families_required": minimum_families,
                "slot_pass": slot_pass,
                "status": "PASS_FIRST_FROZEN_FIELD_WITH_SUFFICIENT_FAMILIES" if slot_pass else "BLOCKED_NO_FROZEN_FIELD_WITH_SUFFICIENT_FAMILIES",
                "sealed_slot": slot["sealed_slot"],
            }
        )
    return attempts, slot_rows, selected_by_slot


def _parquet_artifact(table: pa.Table, path: Path) -> dict[str, Any]:
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    os.chmod(path, 0o600)
    return {"relative_path": path.name, "bytes": path.stat().st_size, "rows": len(table), "sha256": _sha256(path)}


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Strict-incumbent aligned all-topic rescreen",
        "",
        f"Status: `{report['status']}`",
        "",
        "The frozen z>=3 threshold, strict closed-team attention estimand, field order, and three-family rule were retained.",
        "Aggregate counts and aggregate field-year median/MAD were not reused.",
        "",
        "## Frozen slot result",
        "",
    ]
    for row in report["slot_assignment"]:
        lines.append(
            f"- `{row['slot_id']}`: order {row['frozen_field_order']}; selected field "
            f"{row['aligned_selected_field_id']} with {row['aligned_candidate_episode_count']} episodes "
            f"across {row['aligned_subfield_family_count']} families."
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def _cleanup(temporary: Path | None, output: Path) -> None:
    if temporary is None:
        return
    require(temporary.is_absolute() and temporary.parent == output.parent, "unsafe temporary path")
    require(temporary.name.startswith(f".{output.name}."), "unsafe temporary prefix")
    if temporary.exists():
        shutil.rmtree(temporary)


def build(
    *,
    config_path: Path,
    audit_root: Path,
    assignment_root: Path,
    cutoff_root: Path,
    author_history_root: Path,
    taxonomy_path: Path,
    output: Path,
    threads: int,
    memory_limit_gb: int,
) -> dict[str, Any]:
    started = time.monotonic()
    require(1 <= threads <= 32, "threads must be in [1,32]")
    require(16 <= memory_limit_gb <= 384, "memory limit must be in [16,384] GB")
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(config.get("status") == "FROZEN_ALIGNED_STRICT_INCUMBENT_RESCREEN_PENDING_PRODUCTION_NO_F_PASS", "config status")
    completion = _mapping(config["completion_accounting"], "completion")
    require(completion["confirmatory_entry_or_persistence_labels_read"] == 0, "confirmatory labels read")
    for key in ("realized_citation_effect_read", "holdout_byte_seal_complete", "model_fitted", "g4_analytic_open", "pass_f", "pass_g0"):
        require(completion[key] is False, f"unsafe completion state: {key}")
    repository_binding = _validate_authorization(config)
    audit_binding = _validate_audit_root(audit_root, _mapping(config["verified_audit_parent"], "audit parent"))
    frozen_slot_source, aggregate_candidate_path, assignment_binding = _validate_assignment_root(
        assignment_root, _mapping(config["frozen_assignment_parent"], "assignment parent")
    )
    frozen_slots = _validate_frozen_slots(frozen_slot_source, _mapping(config["frozen_field_attempts"], "frozen attempts"))
    predecessor_binding = _mapping(config["predecessor_episode_closure"], "predecessor closure")
    predecessor_path, predecessor_stat = _validate_external_parquet(Path(str(predecessor_binding["path"])), predecessor_binding)
    inputs = _mapping(config["normalized_inputs"], "normalized inputs")
    work_paths, work_binding, work_stats = d0._validate_partitioned_input(
        cutoff_root, _mapping(inputs["cutoff_work_metadata"], "work input"), schema=d0.WORK_INPUT_SCHEMA, workers=threads
    )
    event_paths, event_binding, event_stats = d0._validate_partitioned_input(
        author_history_root, _mapping(inputs["author_history_events"], "event input"), schema=d0.EVENT_INPUT_SCHEMA, workers=threads
    )
    d0_overlap_binding = _mapping(inputs["d0_taxonomy_overlap"], "D0 taxonomy overlap")
    d0_overlap_path, d0_overlap_stat = _validate_external_parquet(Path(str(d0_overlap_binding["path"])), d0_overlap_binding)
    taxonomy, taxonomy_binding, taxonomy_stat = _validate_taxonomy(
        taxonomy_path,
        _mapping(inputs["taxonomy_recovery"], "taxonomy recovery"),
        aggregate_candidate_path,
        d0_overlap_path,
    )
    primary = _mapping(config["primary_rescreen"], "primary rescreen")
    require(primary["primary_work_inclusion"] == "EVERY_LINKED_AUTHOR_IS_PRE_SIGNAL_INCUMBENT", "primary estimand changed")
    require(primary["aggregate_count_or_normalization_reuse"] == "FORBIDDEN", "aggregate reuse allowed")
    require(primary["threshold_relaxation"] is False and float(primary["modified_z_threshold_inclusive"]) == 3.0, "threshold changed")
    output = output.resolve()
    require(output.name == config["output_contract"]["root_basename"], "output basename")
    require(not output.exists() and output.parent.is_dir(), "output must be new")
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    try:
        spill = temporary / "duckdb-spill"
        spill.mkdir(mode=0o700)
        database = temporary / "aligned-rescreen.duckdb"
        surface_rows, candidate_rows, closure_rows, accounting = _run_rescreen(
            database=database,
            spill=spill,
            work_glob=cutoff_root.resolve() / str(inputs["cutoff_work_metadata"]["artifact_prefix"]) / "*.parquet",
            event_glob=author_history_root.resolve() / str(inputs["author_history_events"]["artifact_prefix"]) / "*.parquet",
            taxonomy=taxonomy,
            frozen_field_year=_frozen_field_year_table(frozen_slots, _mapping(config["frozen_field_attempts"], "frozen attempts")),
            predecessor_path=predecessor_path,
            primary=primary,
            threads=threads,
            memory_limit_gb=memory_limit_gb,
        )
        minimum_families = int(primary["minimum_distinct_current_snapshot_subfield_families"])
        attempts, slots, selected_by_slot = _select_slots(frozen_slots, candidate_rows, minimum_families)
        panel_candidates = []
        for row in candidate_rows:
            selected = selected_by_slot.get(str(row["slot_id"])) == int(row["field_id"])
            panel_candidates.append({**row, "aligned_selected_field": selected, "retained_for_aligned_panel": selected})
        all_slots_pass = len(selected_by_slot) == len(frozen_slots) and all(row["slot_pass"] for row in slots)
        output_contract = _mapping(config["output_contract"], "output contract")
        status = output_contract["passing_status"] if all_slots_pass else output_contract["blocking_status"]
        shutil.rmtree(spill)
        database.unlink()
        artifacts = [
            _parquet_artifact(pa.Table.from_pylist(surface_rows, schema=SURFACE_SCHEMA), temporary / "aligned_all_topic_surface.parquet"),
            _parquet_artifact(pa.Table.from_pylist(panel_candidates, schema=CANDIDATE_SCHEMA), temporary / "aligned_strict_candidates.parquet"),
            _parquet_artifact(pa.Table.from_pylist(attempts, schema=ATTEMPT_SCHEMA), temporary / "aligned_field_attempts.parquet"),
            _parquet_artifact(pa.Table.from_pylist(slots, schema=SLOT_SCHEMA), temporary / "aligned_slot_assignment.parquet"),
            _parquet_artifact(pa.Table.from_pylist(closure_rows, schema=CLOSURE_SCHEMA), temporary / "predecessor_episode_count_closure.parquet"),
        ]
        d0._validate_unchanged(work_paths, work_stats)
        d0._validate_unchanged(event_paths, event_stats)
        for path, before in ((taxonomy_path.resolve(), taxonomy_stat), (d0_overlap_path, d0_overlap_stat), (predecessor_path, predecessor_stat)):
            now = path.stat()
            require((now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns) == before, f"input changed: {path.name}")
        retained = [row for row in panel_candidates if row["retained_for_aligned_panel"]]
        report = {
            "schema_version": output_contract["report_schema_version"],
            "status": status,
            "repository_bindings": {"authorization_parent": repository_binding},
            "production_parent_bindings": {"verified_audit_parent": audit_binding, "frozen_assignment_parent": assignment_binding},
            "input_bindings": {
                "cutoff_work_metadata": work_binding,
                "author_history_events": event_binding,
                "taxonomy_recovery": taxonomy_binding,
                "d0_taxonomy_overlap": {"sha256": d0_overlap_binding["sha256"], "rows": d0_overlap_binding["rows"]},
                "predecessor_episode_closure": {"sha256": predecessor_binding["sha256"], "rows": predecessor_binding["rows"]},
            },
            "replay_accounting": {
                **accounting,
                "frozen_slot_count": len(slots),
                "frozen_field_attempt_count": len(attempts),
                "selected_candidate_episode_count": len(retained),
                "selected_candidate_topic_count": len({(int(row["field_id"]), int(row["topic_id"])) for row in retained}),
                "all_slots_pass": all_slots_pass,
            },
            "field_attempts": attempts,
            "slot_assignment": slots,
            "normalization_decision": {
                "work_universe": primary["primary_work_inclusion"],
                "candidate_universe": primary["candidate_universe"],
                "normalization_group": primary["normalization_group"],
                "aggregate_reuse": primary["aggregate_count_or_normalization_reuse"],
                "modified_z_threshold": primary["modified_z_threshold_inclusive"],
                "minimum_subfield_families": minimum_families,
                "threshold_relaxation": primary["threshold_relaxation"],
            },
            "authorization": config["authorization_after_pass"] if all_slots_pass else config["authorization_after_block"],
            "scope": {
                "confirmatory_entry_or_persistence_labels_read": 0,
                "realized_citation_effect_read": False,
                "raw_author_or_work_ids_emitted": False,
                "holdout_byte_seal_complete": False,
                "model_fitted": False,
                "g4_open": False,
                "pass_f": False,
                "pass_g0": False,
            },
            "execution": {
                "config_relative_path": str(config_path.relative_to(ROOT)),
                "config_sha256": _sha256(config_path),
                "builder_relative_path": str(Path(__file__).resolve().relative_to(ROOT)),
                "builder_sha256": _sha256(Path(__file__).resolve()),
                "threads": threads,
                "memory_limit_gb": memory_limit_gb,
                "elapsed_seconds": round(time.monotonic() - started, 6),
            },
            "claim_boundary": config["claim_boundary"],
        }
        report_path = temporary / "aligned_strict_incumbent_rescreen.json"
        _write_json(report_path, report)
        artifacts.append({"relative_path": report_path.name, "bytes": report_path.stat().st_size, "sha256": _sha256(report_path)})
        markdown_path = temporary / "ALIGNED_STRICT_INCUMBENT_RESCREEN.md"
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        os.chmod(markdown_path, 0o600)
        artifacts.append({"relative_path": markdown_path.name, "bytes": markdown_path.stat().st_size, "sha256": _sha256(markdown_path)})
        artifacts.sort(key=lambda row: row["relative_path"])
        receipt = {
            "schema_version": output_contract["receipt_schema_version"],
            "status": status,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_bytes([(row["relative_path"], row["bytes"], row["sha256"]) for row in artifacts])
            ).hexdigest(),
            "receipt_written_last": True,
            "independent_verification_complete": False,
            "aligned_all_topic_rescreen_complete": True,
            "all_slots_pass": all_slots_pass,
            "final_panel_leaveout_replay_authorized_next": all_slots_pass,
            "confirmatory_entry_or_persistence_labels_read": 0,
            "raw_author_or_work_ids_emitted": False,
            "holdout_byte_seal_complete": False,
            "model_fitted": False,
            "g4_open": False,
            "pass_f": False,
            "pass_g0": False,
        }
        _write_json(temporary / "aligned_strict_incumbent_rescreen.outputs.json", receipt)
        os.replace(temporary, output)
        temporary = None
        return report
    finally:
        _cleanup(temporary, output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit-gb", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build(
        config_path=args.config,
        audit_root=args.audit_root,
        assignment_root=args.assignment_root,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        taxonomy_path=args.taxonomy,
        output=args.output,
        threads=args.threads,
        memory_limit_gb=args.memory_limit_gb,
    )
    print(json.dumps({"status": report["status"], "replay_accounting": report["replay_accounting"], "slot_assignment": report["slot_assignment"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
