#!/usr/bin/env python3
"""Materialize guard-first G1 response-cohort risk rows and outcome labels."""

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
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_e_g1_guarded_risk_label_source_v1.yaml"
CONFIG_SCHEMA = "pivot_feedback_e_g1_guarded_risk_label_source.v1"

COHORT_SCHEMA = pa.schema([
    pa.field("analysis_partition_id", pa.string(), nullable=False),
    pa.field("primary_field_id", pa.uint8(), nullable=False),
    pa.field("certified_episode_id", pa.uint16(), nullable=False),
    pa.field("topic_id", pa.uint32(), nullable=False),
    pa.field("subfield_id", pa.uint16(), nullable=False),
    pa.field("shock_year", pa.uint16(), nullable=False),
    pa.field("response_cohort_offset", pa.int8(), nullable=False),
    pa.field("decision_year", pa.uint16(), nullable=False),
    pa.field("author_numeric_id", pa.uint64(), nullable=False),
    pa.field("entry_outcome", pa.bool_(), nullable=False),
    pa.field("persistence_outcome", pa.bool_(), nullable=True),
    pa.field("one_off_entry", pa.bool_(), nullable=False),
    pa.field("alternative_pivot_entry", pa.bool_(), nullable=False),
    pa.field("entry_target_work_count", pa.uint32(), nullable=False),
    pa.field("persistence_target_work_count", pa.uint32(), nullable=False),
    pa.field("trailing_target_subfield_work_count", pa.uint32(), nullable=False),
])

MANIFEST_SCHEMA = pa.schema([
    pa.field("certified_episode_id", pa.uint16(), nullable=False),
    pa.field("primary_field_id", pa.uint8(), nullable=False),
    pa.field("topic_id", pa.uint32(), nullable=False),
    pa.field("subfield_id", pa.uint16(), nullable=False),
    pa.field("shock_year", pa.uint16(), nullable=False),
    pa.field("response_cohort_offset", pa.int8(), nullable=False),
    pa.field("decision_year", pa.uint16(), nullable=False),
    pa.field("pre_guard_risk_rows", pa.uint64(), nullable=False),
    pa.field("guard_excluded_rows", pa.uint64(), nullable=False),
    pa.field("guarded_risk_rows", pa.uint64(), nullable=False),
    pa.field("entry_rows", pa.uint64(), nullable=False),
    pa.field("non_entry_rows", pa.uint64(), nullable=False),
    pa.field("persistence_rows", pa.uint64(), nullable=False),
    pa.field("one_off_entry_rows", pa.uint64(), nullable=False),
    pa.field("alternative_pivot_entry_rows", pa.uint64(), nullable=False),
    pa.field("guard_overlap_rows", pa.uint64(), nullable=False),
    pa.field("cohort_file_relative_path", pa.string(), nullable=False),
])


class GuardedRiskLabelBuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GuardedRiskLabelBuildError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _safe_repo_path(relative: str) -> Path:
    pure = PurePosixPath(relative)
    require(relative and not pure.is_absolute() and ".." not in pure.parts, "unsafe repository path")
    path = (ROOT / pure).resolve(strict=True)
    path.relative_to(ROOT.resolve())
    require(path.is_file() and not path.is_symlink(), "invalid repository path")
    return path


def _schema_version(path: Path) -> str:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    return metadata.get(b"schema_version", b"").decode()


def _validate_repo_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _safe_repo_path(str(binding["relative_path"]))
    require(_sha256(path) == binding["sha256"], f"repository hash drift: {path.name}")
    payload = _yaml(path) if path.suffix == ".yaml" else _json(path)
    require(payload.get("schema_version") == binding["required_schema_version"], f"repository schema drift: {path.name}")
    require(payload.get("status") == binding["required_status"], f"repository status drift: {path.name}")
    return payload


def _validate_operator_parent(binding: Mapping[str, Any]) -> Path:
    root = Path(str(binding["root"])).resolve(strict=True)
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "operator receipt hash drift")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "operator receipt schema drift")
    require(receipt.get("status") == binding["required_status"], "operator receipt status drift")
    episode = root / str(binding["episode_basename"])
    require(_sha256(episode) == binding["episode_sha256"], "operator episode hash drift")
    require(_schema_version(episode) == binding["episode_schema_version"], "operator episode schema drift")
    require(pq.ParquetFile(episode).metadata.num_rows == binding["episode_rows"], "operator episode rows drift")
    return episode


def _validate_partitioned_source(binding: Mapping[str, Any]) -> Path:
    root = Path(str(binding["root"])).resolve(strict=True)
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "source receipt hash drift")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "source receipt schema drift")
    require(receipt.get("status") == binding["required_status"], "source receipt status drift")
    directory = root / str(binding["artifact_prefix"])
    files = sorted(directory.glob("*.parquet"))
    require(len(files) == binding["artifact_count"], "source artifact count drift")
    return directory / "*.parquet"


def _load_cohorts(config: Mapping[str, Any], final_path: Path, atomic_path: Path) -> list[dict[str, Any]]:
    final = {int(row["episode_id"]): row for row in pq.read_table(final_path).to_pylist()}
    atomic = pq.read_table(atomic_path).to_pylist()
    scope = config["registered_scope"]
    rows = [row for row in atomic if row["registered_partition_id"] == scope["partition_id"]]
    require(len(rows) == scope["episode_count"] == 28, "G1 episode count drift")
    require({int(row["field_id"]) for row in rows} == set(scope["field_ids_in_registered_slot_order"]), "G1 field drift")
    cohorts = []
    for row in sorted(rows, key=lambda value: int(value["episode_id"])):
        episode_id = int(row["episode_id"])
        source = final[episode_id]
        require(source["sealed_slot"] is False and source["leaveout_surge_present"] is True, "unsafe G1 episode")
        for offset in scope["response_cohort_offsets"]:
            cohorts.append({
                "episode_id": episode_id,
                "field_id": int(row["field_id"]),
                "topic_id": int(row["topic_id"]),
                "subfield_id": int(row["subfield_id"]),
                "shock_year": int(row["decision_year"]),
                "response_cohort_offset": int(offset),
                "decision_year": int(row["decision_year"]) + int(offset),
            })
    require(len(cohorts) == scope["response_cohort_count"] == 84, "response cohort count drift")
    return cohorts


def _resource_preflight(output_parent: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    mem = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        mem[key] = int(value.strip().split()[0]) * 1024
    disk = shutil.disk_usage(output_parent)
    nofile, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    execution = config["execution"]
    require(mem["MemAvailable"] >= execution["minimum_available_memory_bytes"], "insufficient memory")
    require(disk.free >= execution["minimum_available_disk_bytes"], "insufficient disk")
    require(nofile >= 8192, "insufficient nofile limit")
    return {"available_memory_bytes": mem["MemAvailable"], "available_disk_bytes": disk.free, "nofile_soft_limit": nofile}


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _cohort_table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = pa.schema([
        pa.field("episode_id", pa.uint16()), pa.field("field_id", pa.uint8()),
        pa.field("topic_id", pa.uint32()), pa.field("subfield_id", pa.uint16()),
        pa.field("shock_year", pa.uint16()), pa.field("response_cohort_offset", pa.int8()),
        pa.field("decision_year", pa.uint16()),
    ])
    return pa.Table.from_pylist(list(rows), schema=schema)


def _taxonomy_table(path: Path) -> pa.Table:
    source = pq.read_table(path, columns=["id", "field__id", "subfield__id"])
    rows = []
    for row in source.to_pylist():
        rows.append({
            "topic_id": int(str(row["id"]).rsplit("/T", 1)[1]),
            "field_id": int(str(row["field__id"]).rsplit("/fields/", 1)[1]),
            "subfield_id": int(str(row["subfield__id"]).rsplit("/subfields/", 1)[1]),
        })
    schema = pa.schema([
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("subfield_id", pa.uint16(), nullable=False),
    ])
    return pa.Table.from_pylist(rows, schema=schema)


def _write_reader(reader: pa.RecordBatchReader, path: Path, schema: pa.Schema, version: str) -> int:
    observed = [(field.name, field.type) for field in reader.schema]
    expected = [(field.name, field.type) for field in schema]
    require(observed == expected, f"query schema drift: {path.name}: {observed}")
    versioned = schema.with_metadata({b"schema_version": version.encode()})
    writer = pq.ParquetWriter(path, versioned, compression="zstd", compression_level=9, use_dictionary=True, write_statistics=True, version="2.6")
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


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema, version: str) -> None:
    versioned = schema.with_metadata({b"schema_version": version.encode()})
    pq.write_table(pa.Table.from_pylist(list(rows), schema=versioned), path, compression="zstd", compression_level=9, version="2.6")
    os.chmod(path, 0o600)


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    row = {"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
    if path.suffix == ".parquet":
        row["rows"] = pq.ParquetFile(path).metadata.num_rows
    return row


def _run_sql(database: Path, cohorts: Sequence[Mapping[str, Any]], work_glob: Path, event_glob: Path, taxonomy_path: Path, guard_path: Path, staging: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    con = duckdb.connect(str(database))
    con.execute(f"PRAGMA threads={int(config['execution']['threads'])}")
    con.execute(f"PRAGMA memory_limit='{int(config['execution']['memory_limit_gb'])}GB'")
    spill = database.parent / "duckdb-spill"
    spill.mkdir(mode=0o700)
    con.execute(f"PRAGMA temp_directory='{_sql_path(spill)}'")
    con.register("cohort_input", _cohort_table(cohorts))
    con.register("taxonomy_input", _taxonomy_table(taxonomy_path))
    con.execute("CREATE TABLE cohort AS SELECT * FROM cohort_input")
    con.execute("CREATE TABLE taxonomy AS SELECT * FROM taxonomy_input")
    con.execute(f"CREATE TABLE guard AS SELECT author_numeric_id FROM read_parquet('{_sql_path(guard_path)}') WHERE exclude_from_g1")
    con.execute("CREATE UNIQUE INDEX guard_author ON guard(author_numeric_id)")
    work = _sql_path(work_glob)
    event = _sql_path(event_glob)
    con.execute(
        f"""
        CREATE TABLE selected_work AS
        SELECT work_id_numeric AS work_id,publication_year,temporal_disposition,
               declared_authors_count,primary_topic_id AS topic_id,primary_field_id AS field_id
        FROM read_parquet('{work}')
        WHERE primary_field_id IN (16,32,11) AND publication_year<=2021
          AND temporal_disposition IN (2,3) AND primary_topic_id IS NOT NULL
          AND work_type_code IN (1,2,3,4,7,16,18,21)
          AND NOT is_paratext AND coalesce(is_retracted,false)=false
        """
    )
    con.execute("CREATE UNIQUE INDEX selected_work_id ON selected_work(work_id)")
    con.execute(
        f"""
        CREATE TABLE selected_event AS
        SELECT DISTINCT e.author_numeric_id AS author_id,e.work_numeric_id AS work_id,
               e.publication_year,w.field_id,w.topic_id,
               tx.subfield_id::USMALLINT AS subfield_id
        FROM read_parquet('{event}') e
        JOIN selected_work w ON w.work_id=e.work_numeric_id
        JOIN taxonomy tx ON tx.topic_id=w.topic_id AND tx.field_id=w.field_id
        WHERE e.publication_year=w.publication_year
          AND e.primary_field_id=w.field_id
          AND e.temporal_disposition=w.temporal_disposition
        """
    )
    con.execute("CREATE INDEX selected_event_author_year ON selected_event(author_id,publication_year)")
    con.execute("CREATE INDEX selected_event_topic_author ON selected_event(field_id,topic_id,author_id)")
    con.execute(
        """
        CREATE TABLE active AS
        SELECT DISTINCT c.episode_id,c.response_cohort_offset,c.decision_year,e.author_id
        FROM cohort c JOIN selected_event e ON e.field_id=c.field_id
        WHERE e.publication_year BETWEEN c.decision_year-3 AND c.decision_year-1
        """
    )
    con.execute(
        """
        CREATE TABLE prior_target AS
        SELECT DISTINCT c.episode_id,c.response_cohort_offset,e.author_id
        FROM cohort c JOIN selected_event e
          ON e.field_id=c.field_id AND e.topic_id=c.topic_id
        WHERE e.publication_year<=c.decision_year-1
        """
    )
    con.execute("CREATE INDEX prior_target_key ON prior_target(episode_id,response_cohort_offset,author_id)")
    con.execute(
        """
        CREATE TABLE pre_guard_risk AS
        SELECT a.* FROM active a
        ANTI JOIN prior_target p USING (episode_id,response_cohort_offset,author_id)
        """
    )
    con.execute("CREATE INDEX pre_guard_author ON pre_guard_risk(author_id)")
    con.execute(
        """
        CREATE TABLE guarded_risk AS
        SELECT r.* FROM pre_guard_risk r ANTI JOIN guard g ON g.author_numeric_id=r.author_id
        """
    )
    con.execute("CREATE INDEX guarded_risk_key ON guarded_risk(episode_id,response_cohort_offset,author_id)")
    overlap = int(con.execute("SELECT count(*) FROM guarded_risk r JOIN guard g ON g.author_numeric_id=r.author_id").fetchone()[0])
    require(overlap == 0, "guard anti-join overlap")
    con.execute(
        """
        CREATE TABLE label_count AS
        SELECT r.episode_id,r.response_cohort_offset,r.author_id,
               count(DISTINCT e.work_id) FILTER (WHERE e.publication_year=r.decision_year)::UINTEGER AS entry_count,
               count(DISTINCT e.work_id) FILTER (WHERE e.publication_year BETWEEN r.decision_year+1 AND r.decision_year+3)::UINTEGER AS persistence_count
        FROM guarded_risk r
        LEFT JOIN cohort c ON c.episode_id=r.episode_id AND c.response_cohort_offset=r.response_cohort_offset
        LEFT JOIN selected_event e ON e.author_id=r.author_id AND e.field_id=c.field_id AND e.topic_id=c.topic_id
          AND e.publication_year BETWEEN r.decision_year AND r.decision_year+3
        GROUP BY r.episode_id,r.response_cohort_offset,r.author_id
        """
    )
    con.execute("CREATE INDEX label_key ON label_count(episode_id,response_cohort_offset,author_id)")
    con.execute(
        """
        CREATE TABLE subfield_count AS
        SELECT r.episode_id,r.response_cohort_offset,r.author_id,
               count(DISTINCT e.work_id)::UINTEGER AS trailing_subfield_count
        FROM guarded_risk r
        JOIN cohort c ON c.episode_id=r.episode_id AND c.response_cohort_offset=r.response_cohort_offset
        LEFT JOIN selected_event e ON e.author_id=r.author_id AND e.field_id=c.field_id AND e.subfield_id=c.subfield_id
          AND e.publication_year BETWEEN r.decision_year-3 AND r.decision_year-1
        GROUP BY r.episode_id,r.response_cohort_offset,r.author_id
        """
    )
    con.execute("CREATE INDEX subfield_key ON subfield_count(episode_id,response_cohort_offset,author_id)")
    con.execute(
        """
        CREATE TABLE guarded_label AS
        SELECT 'g1_empirical_replication'::VARCHAR AS analysis_partition_id,
               c.field_id::UTINYINT AS primary_field_id,c.episode_id::USMALLINT AS certified_episode_id,
               c.topic_id::UINTEGER AS topic_id,c.subfield_id::USMALLINT AS subfield_id,
               c.shock_year::USMALLINT AS shock_year,c.response_cohort_offset::TINYINT AS response_cohort_offset,
               r.decision_year::USMALLINT AS decision_year,r.author_id::UBIGINT AS author_numeric_id,
               (l.entry_count>0)::BOOLEAN AS entry_outcome,
               CASE WHEN l.entry_count>0 THEN l.persistence_count>0 ELSE NULL END::BOOLEAN AS persistence_outcome,
               (l.entry_count>0 AND l.persistence_count=0)::BOOLEAN AS one_off_entry,
               (l.entry_count>0 AND s.trailing_subfield_count=0)::BOOLEAN AS alternative_pivot_entry,
               l.entry_count::UINTEGER AS entry_target_work_count,
               l.persistence_count::UINTEGER AS persistence_target_work_count,
               s.trailing_subfield_count::UINTEGER AS trailing_target_subfield_work_count
        FROM guarded_risk r
        JOIN cohort c ON c.episode_id=r.episode_id AND c.response_cohort_offset=r.response_cohort_offset
        JOIN label_count l ON l.episode_id=r.episode_id AND l.response_cohort_offset=r.response_cohort_offset AND l.author_id=r.author_id
        JOIN subfield_count s ON s.episode_id=r.episode_id AND s.response_cohort_offset=r.response_cohort_offset AND s.author_id=r.author_id
        """
    )
    counts = {
        (int(row[0]), int(row[1])): {
            "pre_guard": int(row[2]), "guarded": int(row[3]), "entry": int(row[4]),
            "persistence": int(row[5]), "one_off": int(row[6]), "alternative": int(row[7]),
        }
        for row in con.execute(
            """
            SELECT c.episode_id,c.response_cohort_offset,
                   count(DISTINCT p.author_id)::UBIGINT AS pre_guard,
                   count(DISTINCT g.author_id)::UBIGINT AS guarded,
                   count(DISTINCT l.author_numeric_id) FILTER (WHERE l.entry_outcome)::UBIGINT AS entry,
                   count(DISTINCT l.author_numeric_id) FILTER (WHERE l.persistence_outcome)::UBIGINT AS persistence,
                   count(DISTINCT l.author_numeric_id) FILTER (WHERE l.one_off_entry)::UBIGINT AS one_off,
                   count(DISTINCT l.author_numeric_id) FILTER (WHERE l.alternative_pivot_entry)::UBIGINT AS alternative
            FROM cohort c
            LEFT JOIN pre_guard_risk p ON p.episode_id=c.episode_id AND p.response_cohort_offset=c.response_cohort_offset
            LEFT JOIN guarded_risk g ON g.episode_id=c.episode_id AND g.response_cohort_offset=c.response_cohort_offset AND g.author_id=p.author_id
            LEFT JOIN guarded_label l ON l.certified_episode_id=c.episode_id AND l.response_cohort_offset=c.response_cohort_offset AND l.author_numeric_id=p.author_id
            GROUP BY c.episode_id,c.response_cohort_offset ORDER BY c.episode_id,c.response_cohort_offset
            """
        ).fetchall()
    }
    require(len(counts) == 84, "cohort accounting closure")
    private = staging / config["output_contract"]["cohort_directory"]
    private.mkdir(mode=0o700, parents=True)
    manifest = []
    by_key = {(int(row["episode_id"]), int(row["response_cohort_offset"])): row for row in cohorts}
    for key in sorted(by_key):
        source = by_key[key]
        episode_id, offset = key
        path = private / f"episode-{episode_id:04d}-k{offset:+d}.parquet"
        rows = _write_reader(
            con.execute(
                """SELECT * FROM guarded_label WHERE certified_episode_id=? AND response_cohort_offset=? ORDER BY author_numeric_id""",
                [episode_id, offset],
            ).fetch_record_batch(1_000_000),
            path, COHORT_SCHEMA, config["output_contract"]["cohort_schema_version"],
        )
        count = counts[key]
        require(rows == count["guarded"], f"cohort export closure: {key}")
        manifest.append({
            "certified_episode_id": episode_id, "primary_field_id": int(source["field_id"]),
            "topic_id": int(source["topic_id"]), "subfield_id": int(source["subfield_id"]),
            "shock_year": int(source["shock_year"]), "response_cohort_offset": offset,
            "decision_year": int(source["decision_year"]), "pre_guard_risk_rows": count["pre_guard"],
            "guard_excluded_rows": count["pre_guard"] - count["guarded"], "guarded_risk_rows": count["guarded"],
            "entry_rows": count["entry"], "non_entry_rows": count["guarded"] - count["entry"],
            "persistence_rows": count["persistence"], "one_off_entry_rows": count["one_off"],
            "alternative_pivot_entry_rows": count["alternative"], "guard_overlap_rows": 0,
            "cohort_file_relative_path": path.relative_to(staging).as_posix(),
        })
    selected_work_rows = int(con.execute("SELECT count(*) FROM selected_work").fetchone()[0])
    selected_event_rows = int(con.execute("SELECT count(*) FROM selected_event").fetchone()[0])
    con.close()
    return manifest, {"selected_work_rows": selected_work_rows, "selected_event_rows": selected_event_rows, "guard_overlap_rows": overlap}


def build(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    authorization = _validate_repo_binding(config["authorization_parent"])
    frozen = _validate_repo_binding(config["frozen_scope_parent"])
    require(authorization["authorization"]["e_g1_execution_resume"] == "AUTHORIZED_EXACT_PARENT_V1_PLUS_V2_SCOPE", "resume not authorized")
    require(frozen["registered_execution_scope"]["selected_field_ids_in_registered_slot_order"] == [16,32,11], "frozen field scope drift")
    final_path = _validate_operator_parent(config["verified_parents"]["aligned_final_panel"])
    atomic_path = _validate_operator_parent(config["verified_parents"]["atomic_partition"])
    work_glob = _validate_partitioned_source(config["exact_inputs"]["normalized_work_metadata"])
    event_glob = _validate_partitioned_source(config["exact_inputs"]["author_history_events"])
    taxonomy = Path(config["exact_inputs"]["topic_taxonomy"]["path"]).resolve(strict=True)
    guard = Path(config["exact_inputs"]["author_cold_guard"]["path"]).resolve(strict=True)
    require(_sha256(taxonomy) == config["exact_inputs"]["topic_taxonomy"]["sha256"], "taxonomy hash drift")
    require(_sha256(guard) == config["exact_inputs"]["author_cold_guard"]["sha256"], "guard hash drift")
    cohorts = _load_cohorts(config, final_path, atomic_path)
    output = Path(config["execution"]["output_root"]).resolve()
    require(not output.exists() and output.parent.is_dir(), "output root must be new")
    resources = _resource_preflight(output.parent, config)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    scratch = Path(tempfile.mkdtemp(prefix=f".{output.name}.scratch-", dir=output.parent))
    os.chmod(staging, 0o700)
    os.chmod(scratch, 0o700)
    try:
        manifest, accounting = _run_sql(scratch / "g1-guarded-risk-label.duckdb", cohorts, work_glob, event_glob, taxonomy, guard, staging, config)
        manifest_path = staging / config["output_contract"]["manifest_basename"]
        _write_table(manifest_path, manifest, MANIFEST_SCHEMA, config["output_contract"]["manifest_schema_version"])
        episode_support = []
        for episode_id in sorted({int(row["certified_episode_id"]) for row in manifest}):
            rows = [row for row in manifest if int(row["certified_episode_id"]) == episode_id]
            total = sum(int(row["guarded_risk_rows"]) for row in rows)
            entries = sum(int(row["entry_rows"]) for row in rows)
            nonentries = sum(int(row["non_entry_rows"]) for row in rows)
            episode_support.append({"episode_id": episode_id, "total": total, "entries": entries, "nonentries": nonentries, "support_pass": total>=500 and entries>=100 and nonentries>=100})
        accounting.update({
            "response_cohort_count": len(manifest),
            "pre_guard_risk_rows": sum(int(row["pre_guard_risk_rows"]) for row in manifest),
            "guard_excluded_rows": sum(int(row["guard_excluded_rows"]) for row in manifest),
            "guarded_risk_rows": sum(int(row["guarded_risk_rows"]) for row in manifest),
            "entry_rows": sum(int(row["entry_rows"]) for row in manifest),
            "persistence_rows": sum(int(row["persistence_rows"]) for row in manifest),
            "one_off_entry_rows": sum(int(row["one_off_entry_rows"]) for row in manifest),
            "alternative_pivot_entry_rows": sum(int(row["alternative_pivot_entry_rows"]) for row in manifest),
            "episode_support_pass_count": sum(bool(row["support_pass"]) for row in episode_support),
            "episode_support_fail_count": sum(not bool(row["support_pass"]) for row in episode_support),
        })
        report = {
            "schema_version": config["output_contract"]["report_schema_version"],
            "status": config["output_contract"]["passing_status"],
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "execution": {"config_relative_path": config_path.relative_to(ROOT).as_posix(), "config_sha256": _sha256(config_path), "builder_relative_path": Path(__file__).resolve().relative_to(ROOT).as_posix(), "builder_sha256": _sha256(Path(__file__).resolve()), "threads": config["execution"]["threads"], "memory_limit_gb": config["execution"]["memory_limit_gb"]},
            "resource_preflight": resources,
            "scope": config["registered_scope"],
            "guard": {"sha256": config["exact_inputs"]["author_cold_guard"]["sha256"], "rows": config["exact_inputs"]["author_cold_guard"]["rows"], "anti_join_before_label_logic": True, "guard_overlap_rows": 0, "sealed_root_accessed": False, "other_private_g4_artifacts_accessed": False},
            "accounting": accounting,
            "episode_support": episode_support,
            "access_accounting": {"confirmatory_rows_read": accounting["guarded_risk_rows"], "holdout_outcome_rows_read": 0, "guard_rows_read": config["exact_inputs"]["author_cold_guard"]["rows"], "other_private_g4_files_read": 0, "seal_open_events": 0, "g4_open": False, "labels_materialized": accounting["guarded_risk_rows"], "estimators_run": 0, "model_selection_scores_read": 0},
            "authorization": config["authorization_after_pass"],
            "claim_boundary": config["claim_boundary"],
        }
        report_path = staging / config["output_contract"]["report_basename"]
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        os.chmod(report_path, 0o600)
        artifacts = [_artifact(path, staging) for path in sorted(staging.rglob("*")) if path.is_file()]
        receipt = {
            "schema_version": config["output_contract"]["receipt_schema_version"],
            "status": config["output_contract"]["passing_status"],
            "artifact_count": len(artifacts), "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "confirmatory_rows_read": accounting["guarded_risk_rows"], "holdout_outcome_rows_read": 0,
            "guard_rows_read": config["exact_inputs"]["author_cold_guard"]["rows"],
            "other_private_g4_files_read": 0, "seal_open_events": 0, "g4_open": False,
            "estimators_run": 0, "receipt_written_last": True,
        }
        receipt_path = staging / config["output_contract"]["receipt_basename"]
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        os.chmod(receipt_path, 0o600)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    receipt = build(args.config)
    print(json.dumps({"status": receipt["status"], "artifact_count": receipt["artifact_count"], "confirmatory_rows_read": receipt["confirmatory_rows_read"], "g4_open": receipt["g4_open"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
