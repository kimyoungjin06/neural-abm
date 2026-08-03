#!/usr/bin/env python3
"""Independently verify raw-authorship ORCID and institution proxy outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/"
    "pivot_feedback_d0_certified_raw_authorship_identity_affiliation_v1.yaml"
)
CONFIG_SCHEMA = (
    "pivot_feedback_d0_certified_raw_authorship_identity_affiliation_config.v1"
)
REPORT_SCHEMA = (
    "pivot_feedback_d0_certified_raw_authorship_identity_affiliation_report.v1"
)
RECEIPT_SCHEMA = (
    "pivot_feedback_d0_certified_raw_authorship_identity_affiliation_outputs.v1"
)
VERIFY_SCHEMA = (
    "pivot_feedback_d0_certified_raw_authorship_identity_affiliation_verification.v1"
)
PRODUCER_STATUS = (
    "PASS_EXACT_FOUR_RAW_AUTHORSHIP_ORCID_SENSITIVITY_AND_"
    "WORK_TIME_INSTITUTION_PROXY"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_EXACT_FOUR_RAW_AUTHORSHIP_ORCID_SENSITIVITY_AND_"
    "WORK_TIME_INSTITUTION_PROXY"
)
AUTHOR_SCHEMA_VERSION = (
    "pivot_feedback_d0_certified_candidate_identity_affiliation_base.v1"
)
EDGE_SCHEMA_VERSION = (
    "pivot_feedback_d0_certified_prior_peer_affiliation_edge.v1"
)
INSTITUTION_SCHEMA_VERSION = (
    "pivot_feedback_d0_certified_relevant_author_work_institution.v1"
)
CAPS = (10, 50, 100)
EXPECTED_FILES = {
    "RAW_AUTHORSHIP_IDENTITY_AFFILIATION.md",
    "candidate_identity_affiliation_base.parquet",
    "prior_peer_affiliation_edges.parquet",
    "raw_authorship_identity_affiliation.json",
    "raw_authorship_identity_affiliation.outputs.json",
    "relevant_author_work_institutions.parquet",
}


class RawAuthorshipIdentityAffiliationVerificationError(RuntimeError):
    """Fail-closed independent raw-authorship verifier error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RawAuthorshipIdentityAffiliationVerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} must be a mapping")
    return value


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} must be a mapping")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be a mapping")
    return value


def _sql_path(path: Path | str) -> str:
    return str(path).replace("'", "''")


def _scalar(connection: duckdb.DuckDBPyConnection, query: str) -> Any:
    row = connection.execute(query).fetchone()
    require(row is not None and len(row) == 1, "scalar query did not return one cell")
    return row[0]


def _difference_count(
    connection: duckdb.DuckDBPyConnection,
    expected_sql: str,
    observed_sql: str,
) -> int:
    missing = int(
        _scalar(
            connection,
            f"SELECT count(*) FROM (({expected_sql}) EXCEPT ALL ({observed_sql}))",
        )
    )
    extra = int(
        _scalar(
            connection,
            f"SELECT count(*) FROM (({observed_sql}) EXCEPT ALL ({expected_sql}))",
        )
    )
    return missing + extra


def _orcid_checksum_valid(value: str | None) -> bool:
    if value is None or re.fullmatch(
        r"https://orcid[.]org/[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]",
        value,
    ) is None:
        return False
    code = value.removeprefix("https://orcid.org/").replace("-", "")
    total = 0
    for character in code[:15]:
        total = (total + int(character)) * 2
    result = (12 - total % 11) % 11
    return code[-1] == ("X" if result == 10 else str(result))


def _validate_root(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(root.is_absolute() and root.is_dir(), "producer root missing")
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, "producer root mode")
    files = {path.name for path in root.iterdir() if path.is_file()}
    require(files == EXPECTED_FILES, "producer root file closure")
    require(not any(path.is_dir() for path in root.iterdir()), "producer root directory")
    for path in root.iterdir():
        require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"{path.name} mode")
    receipt_path = root / "raw_authorship_identity_affiliation.outputs.json"
    report_path = root / "raw_authorship_identity_affiliation.json"
    receipt = _read_json(receipt_path, "receipt")
    report = _read_json(report_path, "report")
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema")
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema")
    require(receipt.get("status") == PRODUCER_STATUS, "receipt status")
    require(report.get("status") == PRODUCER_STATUS, "report status")
    require(receipt.get("artifact_count") == 5, "receipt artifact count")
    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, list) and len(artifacts) == 5, "receipt artifacts")
    require(
        {str(item["relative_path"]) for item in artifacts}
        == EXPECTED_FILES - {receipt_path.name},
        "receipt artifact closure",
    )
    for item in artifacts:
        path = root / str(item["relative_path"])
        require(path.stat().st_size == int(item["bytes"]), f"{path.name} bytes")
        require(_sha256(path) == item["sha256"], f"{path.name} SHA")
    scope = _mapping(report.get("scope"), "report scope")
    require(scope.get("outcomes_read") is False, "producer outcomes opened")
    require(scope.get("raw_name_read") is False, "producer raw names opened")
    require(scope.get("raw_affiliation_string_read") is False, "raw strings opened")
    require(scope.get("team_cap_selected") is False, "team cap selected")
    require(
        scope.get("independent_verification_complete") is False,
        "producer self-certified independent verification",
    )
    return receipt, report, {
        "receipt_sha256": _sha256(receipt_path),
        "report_sha256": _sha256(report_path),
    }


def _validate_parquet(
    path: Path,
    *,
    schema_version: str,
    rows: int,
) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == rows, f"{path.name} rows")
    schema = parquet.schema_arrow
    require(
        schema.metadata is not None
        and schema.metadata.get(b"schema_version") == schema_version.encode(),
        f"{path.name} schema version",
    )
    require(all(not field.nullable for field in schema), f"{path.name} nullable")
    return {
        "rows": rows,
        "row_groups": parquet.metadata.num_row_groups,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "schema_version": schema_version,
    }


def _validate_bindings(
    config_path: Path,
    config: Mapping[str, Any],
    *,
    normalized_root: Path,
    author_history_root: Path,
    raw_root: Path,
    s0_root: Path,
) -> tuple[dict[str, Any], list[Path], list[dict[str, int]]]:
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    forbidden = _mapping(config.get("forbidden_inputs_and_outputs"), "forbidden")
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden opened")
    gate = _mapping(config["repository_bindings"]["gate_state"], "gate")
    gate_path = (REPO_ROOT / str(gate["relative_path"])).resolve(strict=True)
    require(_sha256(gate_path) == gate["sha256"], "gate SHA")
    gate_doc = _read_yaml(gate_path, "gate")
    require(gate_doc.get("schema_version") == gate["required_schema_version"], "gate schema")
    require(gate_doc.get("status") == gate["required_status"], "gate status")

    parent = _mapping(config["verified_normalized_parent"], "normalized parent")
    require(normalized_root.name == parent["root_basename"], "normalized basename")
    for basename_key, sha_key in (
        ("receipt_basename", "receipt_sha256"),
        ("edge_basename", "edge_sha256"),
        ("author_basename", "author_sha256"),
    ):
        path = normalized_root / str(parent[basename_key])
        require(_sha256(path) == parent[sha_key], f"normalized {basename_key}")
    verification_path = normalized_root.parent / str(parent["verification_basename"])
    require(
        _sha256(verification_path) == parent["verification_sha256"],
        "normalized verification SHA",
    )
    verification = _read_json(verification_path, "normalized verification")
    require(
        verification.get("status") == parent["verification_status"],
        "normalized verification status",
    )

    source = _mapping(config["raw_authorship_source"], "raw source")
    require(raw_root.name == source["expected_root_basename"], "raw basename")
    history = _mapping(
        config["normalized_sources"]["author_history_events"],
        "author history",
    )
    require(author_history_root.name == history["expected_root_basename"], "history basename")
    plan_path = author_history_root / str(history["plan_basename"])
    require(_sha256(plan_path) == history["plan_sha256"], "history plan SHA")
    plan = _read_json(plan_path, "history plan")
    tasks = plan.get(str(source["selected_task_list_key"]))
    require(isinstance(tasks, list) and len(tasks) == source["expected_files"], "raw tasks")
    paths = []
    stats = []
    total_bytes = 0
    total_rows = 0
    for index, raw in enumerate(tasks):
        task = _mapping(raw, f"task {index}")
        path = (raw_root / str(task["relative_path"])).resolve(strict=True)
        observed = path.stat()
        current = {
            "bytes": observed.st_size,
            "ctime_ns": observed.st_ctime_ns,
            "device": observed.st_dev,
            "inode": observed.st_ino,
            "mode": stat.S_IMODE(observed.st_mode),
            "mtime_ns": observed.st_mtime_ns,
        }
        require(current == dict(task["source_stat"]), f"raw task stat {index}")
        total_bytes += observed.st_size
        total_rows += pq.ParquetFile(path).metadata.num_rows
        paths.append(path)
        stats.append(current)
    require(total_bytes == source["expected_bytes"], "raw bytes")
    require(total_rows == source["expected_rows"], "raw rows")

    s0 = _mapping(config["s0_semantic_parent"], "S0")
    require(s0_root.name == s0["root_basename"], "S0 basename")
    require(
        _sha256(s0_root / str(s0["receipt_basename"])) == s0["receipt_sha256"],
        "S0 receipt SHA",
    )
    s0_verification = s0_root.parent / str(s0["verification_basename"])
    require(_sha256(s0_verification) == s0["verification_sha256"], "S0 verification SHA")
    s0_doc = _read_json(s0_verification, "S0 verification")
    require(s0_doc.get("status") == s0["required_verification_status"], "S0 status")
    return {
        "config_relative_path": config_path.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": _sha256(config_path),
        "gate_sha256": gate["sha256"],
        "normalized_receipt_sha256": parent["receipt_sha256"],
        "normalized_verification_sha256": parent["verification_sha256"],
        "raw_task_plan_sha256": history["plan_sha256"],
        "raw_source_files": len(paths),
        "raw_source_bytes": total_bytes,
        "raw_source_rows_from_parquet_metadata": total_rows,
        "s0_receipt_sha256": s0["receipt_sha256"],
        "s0_verification_sha256": s0["verification_sha256"],
    }, paths, stats


def _validate_source_unchanged(
    paths: Sequence[Path],
    stats: Sequence[dict[str, int]],
) -> None:
    for index, (path, expected) in enumerate(zip(paths, stats, strict=True)):
        observed = path.stat()
        current = {
            "bytes": observed.st_size,
            "ctime_ns": observed.st_ctime_ns,
            "device": observed.st_dev,
            "inode": observed.st_ino,
            "mode": stat.S_IMODE(observed.st_mode),
            "mtime_ns": observed.st_mtime_ns,
        }
        require(current == expected, f"raw source changed: {index}")


def _create_views(
    connection: duckdb.DuckDBPyConnection,
    *,
    root: Path,
    normalized_root: Path,
) -> None:
    paths = {
        "author_output": root / "candidate_identity_affiliation_base.parquet",
        "affiliation_edge_output": root / "prior_peer_affiliation_edges.parquet",
        "institution_output": root / "relevant_author_work_institutions.parquet",
        "edges": normalized_root / "prior_target_peer_edges.parquet",
        "author_base": normalized_root / "author_leaveout_prior_peer_base.parquet",
    }
    for name, path in paths.items():
        connection.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{_sql_path(path)}')"
        )


def _normalized_replay(
    connection: duckdb.DuckDBPyConnection,
    *,
    cutoff_root: Path,
    author_history_root: Path,
) -> dict[str, Any]:
    connection.execute(
        """
        CREATE TEMP TABLE candidate_tokens AS
        SELECT DISTINCT
            author_numeric_id,
            'https://openalex.org/A' || author_numeric_id::VARCHAR AS author_url
        FROM author_base
        """
    )
    work_glob = cutoff_root / "work_metadata/*.parquet"
    connection.execute(
        f"""
        CREATE TEMP TABLE prior_work AS
        SELECT
            CASE
                WHEN publication_year BETWEEN 1998 AND 2002 THEN 1
                WHEN publication_year BETWEEN 2007 AND 2011 THEN 2
                WHEN publication_year BETWEEN 1988 AND 1992 THEN 3
                WHEN publication_year BETWEEN 1948 AND 1952 THEN 4
            END::UTINYINT AS certified_episode_index,
            work_id_numeric AS work_numeric_id,
            publication_year,
            primary_field_id,
            declared_authors_count,
            temporal_disposition
        FROM read_parquet('{_sql_path(work_glob)}')
        WHERE
            (
                publication_year BETWEEN 1948 AND 1952
                OR publication_year BETWEEN 1988 AND 1992
                OR publication_year BETWEEN 1998 AND 2002
                OR publication_year BETWEEN 2007 AND 2011
            )
            AND primary_field_id BETWEEN 11 AND 36
            AND primary_topic_id IS NOT NULL
            AND primary_topic_id > 0
            AND temporal_disposition IN (2, 3)
            AND work_type_code IN (1, 2, 3, 4, 7, 16, 18, 21)
            AND is_paratext = false
            AND is_retracted IS DISTINCT FROM true
            AND declared_authors_count <= 100
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE peer_tokens AS
        SELECT DISTINCT certified_episode_index, peer_author_id
        FROM edges
        """
    )
    event_glob = author_history_root / "events/*.parquet"
    connection.execute(
        f"""
        CREATE TEMP TABLE peer_events AS
        SELECT DISTINCT
            w.certified_episode_index,
            w.work_numeric_id,
            e.author_numeric_id AS peer_author_id,
            w.publication_year,
            w.declared_authors_count
        FROM read_parquet('{_sql_path(event_glob)}') e
        INNER JOIN prior_work w
            ON w.work_numeric_id = e.work_numeric_id
            AND w.publication_year = e.publication_year
            AND w.primary_field_id = e.primary_field_id
            AND w.declared_authors_count = e.declared_authors_count
            AND w.temporal_disposition = e.temporal_disposition
        INNER JOIN peer_tokens p
            ON p.certified_episode_index = w.certified_episode_index
            AND p.peer_author_id = e.author_numeric_id
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE peer_work AS
        SELECT DISTINCT
            certified_episode_index, work_numeric_id,
            publication_year, declared_authors_count
        FROM peer_events
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE candidate_events AS
        SELECT DISTINCT
            w.certified_episode_index,
            w.work_numeric_id,
            e.author_numeric_id AS candidate_author_id,
            w.publication_year,
            w.declared_authors_count
        FROM read_parquet('{_sql_path(event_glob)}') e
        INNER JOIN peer_work w
            ON w.work_numeric_id = e.work_numeric_id
            AND w.publication_year = e.publication_year
            AND w.declared_authors_count = e.declared_authors_count
        INNER JOIN (
            SELECT DISTINCT certified_episode_index, candidate_author_id
            FROM edges
        ) c
            ON c.certified_episode_index = w.certified_episode_index
            AND c.candidate_author_id = e.author_numeric_id
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE shared_pair_work AS
        SELECT DISTINCT
            e.certified_episode_index,
            c.work_numeric_id,
            c.publication_year,
            c.declared_authors_count,
            e.candidate_author_id,
            e.peer_author_id
        FROM edges e
        INNER JOIN candidate_events c
            ON c.certified_episode_index = e.certified_episode_index
            AND c.candidate_author_id = e.candidate_author_id
        INNER JOIN peer_events p
            ON p.certified_episode_index = e.certified_episode_index
            AND p.work_numeric_id = c.work_numeric_id
            AND p.peer_author_id = e.peer_author_id
        """
    )
    edge_difference = _difference_count(
        connection,
        """
        SELECT
            certified_episode_index, candidate_author_id, peer_author_id,
            min(declared_authors_count)::USMALLINT,
            min(publication_year)::USMALLINT,
            max(publication_year)::USMALLINT,
            count(*)::UBIGINT
        FROM shared_pair_work GROUP BY 1, 2, 3
        """,
        """
        SELECT
            certified_episode_index, candidate_author_id, peer_author_id,
            minimum_declared_team_size,
            first_prior_collaboration_year,
            last_prior_collaboration_year,
            distinct_shared_work_count
        FROM edges
        """,
    )
    require(edge_difference == 0, "normalized shared pair work differs")
    connection.execute(
        """
        CREATE TEMP TABLE relevant_work_author AS
        SELECT DISTINCT
            certified_episode_index, work_numeric_id, publication_year,
            declared_authors_count, candidate_author_id AS author_numeric_id,
            'https://openalex.org/W' || work_numeric_id::VARCHAR AS work_url,
            'https://openalex.org/A' || candidate_author_id::VARCHAR AS author_url
        FROM shared_pair_work
        UNION
        SELECT DISTINCT
            certified_episode_index, work_numeric_id, publication_year,
            declared_authors_count, peer_author_id AS author_numeric_id,
            'https://openalex.org/W' || work_numeric_id::VARCHAR AS work_url,
            'https://openalex.org/A' || peer_author_id::VARCHAR AS author_url
        FROM shared_pair_work
        """
    )
    return {
        "prior_work_rows": int(_scalar(connection, "SELECT count(*) FROM prior_work")),
        "peer_event_rows": int(_scalar(connection, "SELECT count(*) FROM peer_events")),
        "candidate_event_rows_on_peer_work": int(
            _scalar(connection, "SELECT count(*) FROM candidate_events")
        ),
        "shared_pair_work_rows": int(
            _scalar(connection, "SELECT count(*) FROM shared_pair_work")
        ),
        "relevant_work_author_rows": int(
            _scalar(connection, "SELECT count(*) FROM relevant_work_author")
        ),
        "edge_bidirectional_row_difference": edge_difference,
        "strategy": "DUCKDB_PEER_WORK_FIRST_THEN_CANDIDATE_REVERSE_JOIN",
    }


def _identity_replay(
    connection: duckdb.DuckDBPyConnection,
    *,
    raw_glob: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    connection.execute(
        f"""
        CREATE TEMP TABLE author_orcid_groups AS
        SELECT
            c.author_numeric_id,
            r.authorships__author__orcid AS orcid,
            count(*)::UBIGINT AS authorship_rows
        FROM read_parquet('{_sql_path(raw_glob)}') r
        INNER JOIN candidate_tokens c
            ON c.author_url = r.authorships__author__id
        GROUP BY 1, 2
        """
    )
    connection.create_function(
        "orcid_checksum_valid",
        _orcid_checksum_valid,
        [duckdb.sqltypes.VARCHAR],
        duckdb.sqltypes.BOOLEAN,
        null_handling="special",
    )
    connection.execute(
        """
        CREATE TEMP TABLE expected_identity AS
        SELECT
            author_numeric_id,
            sum(authorship_rows)::UBIGINT AS raw_authorship_row_count,
            sum(authorship_rows) FILTER (orcid IS NOT NULL)::UBIGINT
                AS nonnull_orcid_row_count,
            count(*) FILTER (orcid_checksum_valid(orcid))::USMALLINT
                AS distinct_checksum_valid_orcid_count
        FROM author_orcid_groups GROUP BY 1
        """
    )
    expected_sql = """
        SELECT
            a.certified_episode_index, a.field_id, a.topic_id, a.entry_year,
            a.author_numeric_id,
            coalesce(i.raw_authorship_row_count, 0)::UBIGINT,
            coalesce(i.nonnull_orcid_row_count, 0)::UBIGINT,
            coalesce(i.distinct_checksum_valid_orcid_count, 0)::USMALLINT,
            (i.author_numeric_id IS NOT NULL),
            (coalesce(i.distinct_checksum_valid_orcid_count, 0) = 1)
        FROM author_base a
        LEFT JOIN expected_identity i USING (author_numeric_id)
    """
    observed_sql = """
        SELECT
            certified_episode_index, field_id, topic_id, entry_year,
            author_numeric_id, raw_authorship_row_count,
            nonnull_orcid_row_count, distinct_checksum_valid_orcid_count,
            raw_authorship_observed, orcid_stable_single_valid_token
        FROM author_output
    """
    difference = _difference_count(connection, expected_sql, observed_sql)
    require(difference == 0, "candidate identity output differs")
    multiple = int(
        _scalar(
            connection,
            """
            SELECT count(*) FROM expected_identity
            WHERE distinct_checksum_valid_orcid_count > 1
            """,
        )
    )
    require(multiple == 0, "multiple valid ORCID candidate tokens")
    return {
        "candidate_tokens_with_raw_authorship": int(
            _scalar(connection, "SELECT count(*) FROM expected_identity")
        ),
        "candidate_raw_authorship_rows": int(
            _scalar(connection, "SELECT sum(raw_authorship_row_count) FROM expected_identity")
        ),
        "candidate_nonnull_orcid_rows": int(
            _scalar(connection, "SELECT sum(nonnull_orcid_row_count) FROM expected_identity")
        ),
        "candidate_tokens_with_single_checksum_valid_orcid": int(
            _scalar(
                connection,
                """
                SELECT count(*) FROM expected_identity
                WHERE distinct_checksum_valid_orcid_count = 1
                """,
            )
        ),
        "candidate_tokens_with_multiple_checksum_valid_orcids": multiple,
        "bidirectional_row_difference": difference,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "strategy": "PAIR_FIRST_AUTHOR_ORCID_GROUP_THEN_TOKEN_REDUCTION",
    }


def _institution_replay(
    connection: duckdb.DuckDBPyConnection,
    *,
    raw_glob: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    connection.execute(
        f"""
        CREATE TEMP TABLE expected_institution AS
        SELECT DISTINCT
            k.certified_episode_index,
            k.work_numeric_id,
            k.publication_year,
            k.declared_authors_count,
            k.author_numeric_id,
            try_cast(substr(inst.id, 23) AS UBIGINT) AS institution_numeric_id
        FROM read_parquet('{_sql_path(raw_glob)}') r
        INNER JOIN relevant_work_author k
            ON k.work_url = r.id
            AND k.author_url = r.authorships__author__id
        CROSS JOIN UNNEST(r.authorships__institutions) AS u(inst)
        WHERE regexp_full_match(
            inst.id,
            'https://openalex[.]org/I(0|[1-9][0-9]*)'
        )
        """
    )
    columns = """
        certified_episode_index, work_numeric_id, publication_year,
        declared_authors_count, author_numeric_id, institution_numeric_id
    """
    institution_difference = _difference_count(
        connection,
        f"SELECT {columns} FROM expected_institution",
        f"SELECT {columns} FROM institution_output",
    )
    require(institution_difference == 0, "institution relation differs")
    connection.execute(
        """
        CREATE TEMP TABLE comparable_pair_work AS
        SELECT DISTINCT
            s.certified_episode_index, s.work_numeric_id, s.publication_year,
            s.declared_authors_count, s.candidate_author_id, s.peer_author_id
        FROM shared_pair_work s
        INNER JOIN (
            SELECT DISTINCT
                certified_episode_index, work_numeric_id, author_numeric_id
            FROM expected_institution
        ) c
            ON c.certified_episode_index = s.certified_episode_index
            AND c.work_numeric_id = s.work_numeric_id
            AND c.author_numeric_id = s.candidate_author_id
        INNER JOIN (
            SELECT DISTINCT
                certified_episode_index, work_numeric_id, author_numeric_id
            FROM expected_institution
        ) p
            ON p.certified_episode_index = s.certified_episode_index
            AND p.work_numeric_id = s.work_numeric_id
            AND p.author_numeric_id = s.peer_author_id
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE same_pair_work AS
        SELECT DISTINCT
            s.certified_episode_index, s.work_numeric_id, s.publication_year,
            s.declared_authors_count, s.candidate_author_id, s.peer_author_id
        FROM shared_pair_work s
        INNER JOIN expected_institution c
            ON c.certified_episode_index = s.certified_episode_index
            AND c.work_numeric_id = s.work_numeric_id
            AND c.author_numeric_id = s.candidate_author_id
        INNER JOIN expected_institution p
            ON p.certified_episode_index = s.certified_episode_index
            AND p.work_numeric_id = s.work_numeric_id
            AND p.author_numeric_id = s.peer_author_id
            AND p.institution_numeric_id = c.institution_numeric_id
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE expected_affiliation_edge AS
        WITH comparable AS (
            SELECT
                certified_episode_index, candidate_author_id, peer_author_id,
                count(*)::UINTEGER AS work_count,
                min(declared_authors_count)::USMALLINT AS minimum_team,
                min(publication_year)::USMALLINT AS first_year,
                max(publication_year)::USMALLINT AS last_year
            FROM comparable_pair_work GROUP BY 1, 2, 3
        ),
        same_affiliation AS (
            SELECT
                certified_episode_index, candidate_author_id, peer_author_id,
                count(*)::UINTEGER AS work_count,
                min(declared_authors_count)::USMALLINT AS minimum_team,
                min(publication_year)::USMALLINT AS first_year,
                max(publication_year)::USMALLINT AS last_year
            FROM same_pair_work GROUP BY 1, 2, 3
        )
        SELECT
            e.certified_episode_index,
            e.candidate_author_id,
            e.peer_author_id,
            e.minimum_declared_team_size,
            coalesce(c.work_count, 0)::UINTEGER
                AS institution_comparable_shared_work_count,
            coalesce(c.minimum_team, 0)::USMALLINT
                AS minimum_institution_comparable_team_size,
            coalesce(c.first_year, 0)::USMALLINT
                AS first_institution_comparable_year,
            coalesce(c.last_year, 0)::USMALLINT
                AS last_institution_comparable_year,
            coalesce(s.work_count, 0)::UINTEGER
                AS same_affiliation_shared_work_count,
            coalesce(s.minimum_team, 0)::USMALLINT
                AS minimum_same_affiliation_team_size,
            coalesce(s.first_year, 0)::USMALLINT
                AS first_same_affiliation_year,
            coalesce(s.last_year, 0)::USMALLINT
                AS last_same_affiliation_year,
            (c.work_count IS NOT NULL)
                AS has_current_snapshot_work_time_institution_coverage,
            (s.work_count IS NOT NULL)
                AS has_same_affiliation_shared_prior_work
        FROM edges e
        LEFT JOIN comparable c USING (
            certified_episode_index, candidate_author_id, peer_author_id
        )
        LEFT JOIN same_affiliation s USING (
            certified_episode_index, candidate_author_id, peer_author_id
        )
        """
    )
    edge_difference = _difference_count(
        connection,
        "SELECT * FROM expected_affiliation_edge",
        "SELECT * FROM affiliation_edge_output",
    )
    require(edge_difference == 0, "affiliation edge output differs")
    return {
        "relevant_author_work_institution_rows": int(
            _scalar(connection, "SELECT count(*) FROM expected_institution")
        ),
        "institution_comparable_pair_work_rows": int(
            _scalar(connection, "SELECT count(*) FROM comparable_pair_work")
        ),
        "same_affiliation_pair_work_rows": int(
            _scalar(connection, "SELECT count(*) FROM same_pair_work")
        ),
        "edges_with_institution_comparable_shared_work": int(
            _scalar(
                connection,
                """
                SELECT count(*) FROM expected_affiliation_edge
                WHERE has_current_snapshot_work_time_institution_coverage
                """,
            )
        ),
        "edges_with_same_affiliation_shared_work": int(
            _scalar(
                connection,
                """
                SELECT count(*) FROM expected_affiliation_edge
                WHERE has_same_affiliation_shared_prior_work
                """,
            )
        ),
        "institution_bidirectional_row_difference": institution_difference,
        "edge_bidirectional_row_difference": edge_difference,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def _author_cap_expected_sql(cap: int) -> str:
    return f"""
        WITH metric AS (
            SELECT
                certified_episode_index,
                candidate_author_id,
                count(*) FILTER (
                    minimum_declared_team_size <= {cap}
                )::UINTEGER AS prior_peer_count,
                count(*) FILTER (
                    minimum_institution_comparable_team_size BETWEEN 1 AND {cap}
                )::UINTEGER AS comparable_peer_count,
                count(*) FILTER (
                    minimum_same_affiliation_team_size BETWEEN 1 AND {cap}
                )::UINTEGER AS same_peer_count
            FROM expected_affiliation_edge
            GROUP BY 1, 2
        )
        SELECT
            a.certified_episode_index, a.field_id, a.topic_id,
            a.entry_year, a.author_numeric_id,
            coalesce(m.prior_peer_count, 0)::UINTEGER
                AS prior_target_active_peer_count_cap_{cap},
            coalesce(m.comparable_peer_count, 0)::UINTEGER
                AS institution_comparable_prior_peer_count_cap_{cap},
            coalesce(m.same_peer_count, 0)::UINTEGER
                AS same_affiliation_prior_peer_count_cap_{cap},
            (coalesce(m.comparable_peer_count, 0) > 0)
                AS has_institution_comparable_prior_peer_cap_{cap},
            (coalesce(m.same_peer_count, 0) > 0)
                AS has_same_affiliation_bridge_cap_{cap}
        FROM author_base a
        LEFT JOIN metric m
            ON m.certified_episode_index = a.certified_episode_index
            AND m.candidate_author_id = a.author_numeric_id
    """


def _author_cap_replay(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    result = {}
    for cap in CAPS:
        observed = f"""
            SELECT
                certified_episode_index, field_id, topic_id,
                entry_year, author_numeric_id,
                prior_target_active_peer_count_cap_{cap},
                institution_comparable_prior_peer_count_cap_{cap},
                same_affiliation_prior_peer_count_cap_{cap},
                has_institution_comparable_prior_peer_cap_{cap},
                has_same_affiliation_bridge_cap_{cap}
            FROM author_output
        """
        expected = _author_cap_expected_sql(cap)
        difference = _difference_count(connection, expected, observed)
        require(difference == 0, f"author cap {cap} differs")
        row = connection.execute(
            f"""
            SELECT
                count(*) FILTER (
                    has_institution_comparable_prior_peer_cap_{cap}
                ),
                count(*) FILTER (
                    has_same_affiliation_bridge_cap_{cap}
                ),
                sum(institution_comparable_prior_peer_count_cap_{cap}),
                sum(same_affiliation_prior_peer_count_cap_{cap})
            FROM ({expected})
            """
        ).fetchone()
        require(row is not None, f"author cap {cap} summary")
        result[str(cap)] = {
            "author_rows_with_institution_comparable_peer": int(row[0]),
            "author_rows_with_same_affiliation_bridge": int(row[1]),
            "institution_comparable_prior_peer_edges": int(row[2]),
            "same_affiliation_prior_peer_edges": int(row[3]),
            "bidirectional_row_difference": difference,
        }
    violations = int(
        _scalar(
            connection,
            """
            SELECT count(*) FROM author_output WHERE
                same_affiliation_prior_peer_count_cap_10
                    > institution_comparable_prior_peer_count_cap_10
                OR institution_comparable_prior_peer_count_cap_10
                    > prior_target_active_peer_count_cap_10
                OR same_affiliation_prior_peer_count_cap_50
                    > institution_comparable_prior_peer_count_cap_50
                OR institution_comparable_prior_peer_count_cap_50
                    > prior_target_active_peer_count_cap_50
                OR same_affiliation_prior_peer_count_cap_100
                    > institution_comparable_prior_peer_count_cap_100
                OR institution_comparable_prior_peer_count_cap_100
                    > prior_target_active_peer_count_cap_100
            """,
        )
    )
    require(violations == 0, "author affiliation monotonicity")
    return {
        "author_measurement_rows": int(
            _scalar(connection, "SELECT count(*) FROM author_output")
        ),
        "by_cap": result,
        "affiliation_monotonicity_violations": violations,
    }


def _compare_report(
    report: Mapping[str, Any],
    *,
    normalized: Mapping[str, Any],
    identity: Mapping[str, Any],
    institution: Mapping[str, Any],
    author: Mapping[str, Any],
) -> None:
    normalized_report = _mapping(report["normalized_accounting"], "normalized report")
    for key in (
        "prior_work_rows",
        "shared_pair_work_rows",
        "relevant_work_author_rows",
        "edge_bidirectional_row_difference",
    ):
        require(int(normalized_report[key]) == int(normalized[key]), f"normalized {key}")
    identity_report = _mapping(report["identity_accounting"], "identity report")
    for key in (
        "candidate_tokens_with_raw_authorship",
        "candidate_raw_authorship_rows",
        "candidate_nonnull_orcid_rows",
        "candidate_tokens_with_single_checksum_valid_orcid",
        "candidate_tokens_with_multiple_checksum_valid_orcids",
    ):
        require(int(identity_report[key]) == int(identity[key]), f"identity {key}")
    institution_report = _mapping(
        report["institution_accounting"],
        "institution report",
    )
    for key in (
        "relevant_author_work_institution_rows",
        "institution_comparable_pair_work_rows",
        "same_affiliation_pair_work_rows",
        "edges_with_institution_comparable_shared_work",
        "edges_with_same_affiliation_shared_work",
    ):
        require(int(institution_report[key]) == int(institution[key]), f"institution {key}")
    author_report = _mapping(
        report["author_measurement_accounting"],
        "author report",
    )
    require(int(author_report["rows"]) == int(author["author_measurement_rows"]), "author rows")
    for cap in CAPS:
        observed = _mapping(author_report["by_cap"][str(cap)], f"author cap {cap}")
        expected = author["by_cap"][str(cap)]
        for key in (
            "author_rows_with_institution_comparable_peer",
            "author_rows_with_same_affiliation_bridge",
            "institution_comparable_prior_peer_edges",
            "same_affiliation_prior_peer_edges",
        ):
            require(int(observed[key]) == int(expected[key]), f"author {cap} {key}")


def verify(
    *,
    config_path: Path,
    root: Path,
    normalized_root: Path,
    cutoff_root: Path,
    author_history_root: Path,
    raw_root: Path,
    s0_root: Path,
    output: Path,
    workers: int,
    memory_limit: str,
) -> dict[str, Any]:
    started = time.monotonic()
    require(1 <= workers <= 32, "workers must be in [1, 32]")
    config_path = config_path.resolve(strict=True)
    config = _read_yaml(config_path, "config")
    root = root.resolve(strict=True)
    normalized_root = normalized_root.resolve(strict=True)
    cutoff_root = cutoff_root.resolve(strict=True)
    author_history_root = author_history_root.resolve(strict=True)
    raw_root = raw_root.resolve(strict=True)
    s0_root = s0_root.resolve(strict=True)
    output = output.resolve()
    require(not output.exists(), "verification output exists")
    require(output.parent == root.parent, "verification output parent")
    require(
        output.name == f"{root.name}.verification-v1.json",
        "verification output basename",
    )
    receipt, report, producer_integrity = _validate_root(root)
    bindings, source_paths, source_stats = _validate_bindings(
        config_path,
        config,
        normalized_root=normalized_root,
        author_history_root=author_history_root,
        raw_root=raw_root,
        s0_root=s0_root,
    )
    artifacts = {
        "candidate_identity_affiliation_base": _validate_parquet(
            root / "candidate_identity_affiliation_base.parquet",
            schema_version=AUTHOR_SCHEMA_VERSION,
            rows=3_256_271,
        ),
        "prior_peer_affiliation_edges": _validate_parquet(
            root / "prior_peer_affiliation_edges.parquet",
            schema_version=EDGE_SCHEMA_VERSION,
            rows=563_486,
        ),
        "relevant_author_work_institutions": _validate_parquet(
            root / "relevant_author_work_institutions.parquet",
            schema_version=INSTITUTION_SCHEMA_VERSION,
            rows=int(report["institution_relation_accounting"]["rows"]),
        ),
    }

    scratch: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.verification-duckdb.", dir=root.parent)
    )
    os.chmod(scratch, 0o700)
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(str(scratch / "verify.duckdb"))
        connection.execute(f"SET threads = {workers}")
        connection.execute(f"SET memory_limit = '{memory_limit}'")
        connection.execute(
            f"SET temp_directory = '{_sql_path(scratch / 'spill')}'"
        )
        connection.execute("SET preserve_insertion_order = false")
        _create_views(connection, root=root, normalized_root=normalized_root)
        normalized = _normalized_replay(
            connection,
            cutoff_root=cutoff_root,
            author_history_root=author_history_root,
        )
        raw_glob = (
            raw_root
            / str(config["raw_authorship_source"]["table_directory"])
            / "*.parquet"
        )
        identity = _identity_replay(connection, raw_glob=raw_glob)
        institution = _institution_replay(connection, raw_glob=raw_glob)
        author = _author_cap_replay(connection)
        _compare_report(
            report,
            normalized=normalized,
            identity=identity,
            institution=institution,
            author=author,
        )
        _validate_source_unchanged(source_paths, source_stats)
    except BaseException:
        if connection is not None:
            connection.close()
            connection = None
        if scratch is not None and scratch.exists():
            shutil.rmtree(scratch)
        scratch = None
        raise
    finally:
        if connection is not None:
            connection.close()
    if scratch is not None and scratch.exists():
        shutil.rmtree(scratch)

    verification: dict[str, Any] = {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "scope": {
            "exact_certified_episode_rows": 4,
            "full_raw_authorship_source_replayed": True,
            "raw_source_exact_stat_valid_before_and_after": True,
            "producer_imported": False,
            "producer_algorithm_reused": False,
            "pair_first_orcid_replay": True,
            "peer_work_first_normalized_replay": True,
            "institution_relation_reconstructed": True,
            "affiliation_edges_reconstructed": True,
            "all_author_cap_columns_reconstructed": True,
            "team_cap_profiles": [10, 50, 100],
            "team_cap_selected": False,
            "raw_name_read": False,
            "raw_affiliation_string_read": False,
            "natural_person_identity_established": False,
            "historical_affiliation_truth_established": False,
            "peer_visible_success_built": False,
            "outcomes_read": False,
            "entry_or_persistence_rows_read": False,
            "final_candidate_eligibility_established": False,
            "model_fitting_performed": False,
            "g4_rows_read": False,
            "api_requests_created": 0,
            "gpu_used": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "bindings": bindings,
        "producer_integrity": {
            **producer_integrity,
            "receipt_status": receipt["status"],
            "report_status": report["status"],
        },
        "artifact_integrity": artifacts,
        "normalized_replay": normalized,
        "identity_replay": identity,
        "institution_replay": institution,
        "author_replay": author,
        "authorization": {
            "raw_authorship_identity_affiliation_component": "COMPLETE_INDEPENDENTLY_VERIFIED",
            "production_pivot_classification": "NOT_AUTHORIZED",
            "discovery_outcome_opening": "NOT_AUTHORIZED",
            "final_candidate_eligibility": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "execution": {
            "workload_class": "CPU_IO_SQL_HDD",
            "workers": workers,
            "memory_limit": memory_limit,
            "duckdb_version": duckdb.__version__,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "verifier_relative_path": Path(__file__)
            .resolve()
            .relative_to(REPO_ROOT)
            .as_posix(),
            "verifier_sha256": _sha256(Path(__file__).resolve()),
            "independent_strategy": (
                "DUCKDB_PAIR_FIRST_AUTHOR_ORCID_GROUP_AND_PEER_WORK_FIRST_"
                "NORMALIZED_REPLAY_WITH_RAW_RELATION_BIDIRECTIONAL_COMPARISON"
            ),
        },
        "claim_boundary": (
            "This pass independently establishes only current-snapshot "
            "candidate author-token ORCID sensitivity flags and institution-ID "
            "coverage and same-affiliation proxies on exact predecision shared "
            "prior works. It does not establish natural persons, ORCID "
            "ownership, historical affiliation truth, select a cap, construct "
            "visible success, classify pivots, read outcomes, establish final "
            "eligibility, fit a model, open G4, or establish a scientific effect."
        ),
    }
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, output)
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--s0-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--memory-limit", default="24GB")
    return parser


def main() -> int:
    args = _parser().parse_args()
    verification = verify(
        config_path=args.config,
        root=args.root,
        normalized_root=args.normalized_root,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        raw_root=args.raw_root,
        s0_root=args.s0_root,
        output=args.output,
        workers=args.workers,
        memory_limit=args.memory_limit,
    )
    print(
        json.dumps(
            {
                "status": verification["status"],
                "normalized_replay": verification["normalized_replay"],
                "identity_replay": verification["identity_replay"],
                "institution_replay": verification["institution_replay"],
                "author_replay": verification["author_replay"],
                "execution": verification["execution"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
