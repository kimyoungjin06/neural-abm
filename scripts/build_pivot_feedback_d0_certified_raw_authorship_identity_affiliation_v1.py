#!/usr/bin/env python3
"""Build exact-four raw-authorship ORCID and work-time institution proxies."""

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
import pyarrow as pa
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
AUTHOR_SCHEMA_VERSION = (
    "pivot_feedback_d0_certified_candidate_identity_affiliation_base.v1"
)
EDGE_SCHEMA_VERSION = (
    "pivot_feedback_d0_certified_prior_peer_affiliation_edge.v1"
)
INSTITUTION_SCHEMA_VERSION = (
    "pivot_feedback_d0_certified_relevant_author_work_institution.v1"
)
STATUS = (
    "PASS_EXACT_FOUR_RAW_AUTHORSHIP_ORCID_SENSITIVITY_AND_"
    "WORK_TIME_INSTITUTION_PROXY"
)
CAPS = (10, 50, 100)
EXPECTED_EDGE_ROWS = 563_486
EXPECTED_AUTHOR_ROWS = 3_256_271


def _author_schema() -> pa.Schema:
    fields = [
        pa.field("certified_episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("raw_authorship_row_count", pa.uint64(), nullable=False),
        pa.field("nonnull_orcid_row_count", pa.uint64(), nullable=False),
        pa.field(
            "distinct_checksum_valid_orcid_count",
            pa.uint16(),
            nullable=False,
        ),
        pa.field("raw_authorship_observed", pa.bool_(), nullable=False),
        pa.field(
            "orcid_stable_single_valid_token",
            pa.bool_(),
            nullable=False,
        ),
    ]
    for cap in CAPS:
        fields.extend(
            [
                pa.field(
                    f"prior_target_active_peer_count_cap_{cap}",
                    pa.uint32(),
                    nullable=False,
                ),
                pa.field(
                    f"institution_comparable_prior_peer_count_cap_{cap}",
                    pa.uint32(),
                    nullable=False,
                ),
                pa.field(
                    f"same_affiliation_prior_peer_count_cap_{cap}",
                    pa.uint32(),
                    nullable=False,
                ),
                pa.field(
                    f"has_institution_comparable_prior_peer_cap_{cap}",
                    pa.bool_(),
                    nullable=False,
                ),
                pa.field(
                    f"has_same_affiliation_bridge_cap_{cap}",
                    pa.bool_(),
                    nullable=False,
                ),
            ]
        )
    return pa.schema(
        fields,
        metadata={b"schema_version": AUTHOR_SCHEMA_VERSION.encode()},
    )


AUTHOR_SCHEMA = _author_schema()
EDGE_SCHEMA = pa.schema(
    [
        pa.field("certified_episode_index", pa.uint8(), nullable=False),
        pa.field("candidate_author_id", pa.uint64(), nullable=False),
        pa.field("peer_author_id", pa.uint64(), nullable=False),
        pa.field("minimum_declared_team_size", pa.uint16(), nullable=False),
        pa.field(
            "institution_comparable_shared_work_count",
            pa.uint32(),
            nullable=False,
        ),
        pa.field(
            "minimum_institution_comparable_team_size",
            pa.uint16(),
            nullable=False,
        ),
        pa.field(
            "first_institution_comparable_year",
            pa.uint16(),
            nullable=False,
        ),
        pa.field(
            "last_institution_comparable_year",
            pa.uint16(),
            nullable=False,
        ),
        pa.field(
            "same_affiliation_shared_work_count",
            pa.uint32(),
            nullable=False,
        ),
        pa.field(
            "minimum_same_affiliation_team_size",
            pa.uint16(),
            nullable=False,
        ),
        pa.field("first_same_affiliation_year", pa.uint16(), nullable=False),
        pa.field("last_same_affiliation_year", pa.uint16(), nullable=False),
        pa.field(
            "has_current_snapshot_work_time_institution_coverage",
            pa.bool_(),
            nullable=False,
        ),
        pa.field(
            "has_same_affiliation_shared_prior_work",
            pa.bool_(),
            nullable=False,
        ),
    ],
    metadata={b"schema_version": EDGE_SCHEMA_VERSION.encode()},
)
INSTITUTION_SCHEMA = pa.schema(
    [
        pa.field("certified_episode_index", pa.uint8(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("declared_authors_count", pa.uint16(), nullable=False),
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("institution_numeric_id", pa.uint64(), nullable=False),
    ],
    metadata={b"schema_version": INSTITUTION_SCHEMA_VERSION.encode()},
)


class RawAuthorshipIdentityAffiliationError(RuntimeError):
    """Fail-closed raw-authorship identity/affiliation builder error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RawAuthorshipIdentityAffiliationError(message)


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
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


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
            f"SELECT count(*) FROM ({expected_sql} EXCEPT ALL {observed_sql})",
        )
    )
    extra = int(
        _scalar(
            connection,
            f"SELECT count(*) FROM ({observed_sql} EXCEPT ALL {expected_sql})",
        )
    )
    return missing + extra


def _validate_repository_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(_sha256(path) == binding["sha256"], "repository binding SHA")
    document = _read_yaml(path, path.name)
    require(
        document.get("schema_version") == binding["required_schema_version"],
        "repository binding schema",
    )
    require(document.get("status") == binding["required_status"], "repository status")
    for dotted, expected in _mapping(
        binding.get("required_fields", {}),
        "repository required fields",
    ).items():
        value: Any = document
        for component in str(dotted).split("."):
            value = _mapping(value, f"repository field {dotted}")[component]
        require(value == expected, f"repository required field {dotted}")
    return {
        "relative_path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": binding["sha256"],
        "schema_version": document["schema_version"],
        "status": document["status"],
    }


def _validate_json_binding(
    path: Path,
    *,
    sha256: str,
    schema_version: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    require(_sha256(path) == sha256, f"{path.name} SHA")
    document = _read_json(path, path.name)
    if schema_version is not None:
        require(document.get("schema_version") == schema_version, f"{path.name} schema")
    if status is not None:
        require(document.get("status") == status, f"{path.name} status")
    return document


def _source_preflight(
    raw_root: Path,
    plan: Mapping[str, Any],
    source: Mapping[str, Any],
) -> tuple[list[Path], list[dict[str, int]], dict[str, Any]]:
    require(raw_root.name == source["expected_root_basename"], "raw root basename")
    tasks = plan.get(str(source["selected_task_list_key"]))
    require(isinstance(tasks, list), "raw task list missing")
    require(len(tasks) == source["expected_files"], "raw task count")
    paths = []
    stats = []
    total_bytes = 0
    total_rows = 0
    for index, raw in enumerate(tasks):
        task = _mapping(raw, f"raw task {index}")
        path = (raw_root / str(task["relative_path"])).resolve(strict=True)
        require(path.is_file(), f"raw task file missing: {index}")
        observed = path.stat()
        source_stat = _mapping(task["source_stat"], f"raw task stat {index}")
        current = {
            "bytes": observed.st_size,
            "ctime_ns": observed.st_ctime_ns,
            "device": observed.st_dev,
            "inode": observed.st_ino,
            "mode": stat.S_IMODE(observed.st_mode),
            "mtime_ns": observed.st_mtime_ns,
        }
        require(current == dict(source_stat), f"raw task exact stat differs: {index}")
        require(observed.st_size == task["bytes"], f"raw task bytes: {index}")
        parquet = pq.ParquetFile(path)
        total_rows += parquet.metadata.num_rows
        total_bytes += observed.st_size
        paths.append(path)
        stats.append(current)
    require(total_bytes == source["expected_bytes"], "raw source bytes")
    require(total_rows == source["expected_rows"], "raw source rows")
    return paths, stats, {
        "source_files": len(paths),
        "source_bytes": total_bytes,
        "source_rows_from_parquet_metadata": total_rows,
        "source_exact_stat_valid_before": True,
    }


def _validate_source_unchanged(paths: Sequence[Path], stats: Sequence[dict[str, int]]) -> None:
    require(len(paths) == len(stats), "raw source stat length")
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
        require(current == expected, f"raw source changed during scan: {index}")


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
    result = (12 - (total % 11)) % 11
    expected = "X" if result == 10 else str(result)
    return code[-1] == expected


def _create_normalized_relations(
    connection: duckdb.DuckDBPyConnection,
    *,
    normalized_root: Path,
    cutoff_root: Path,
    author_history_root: Path,
) -> dict[str, Any]:
    edge_path = normalized_root / "prior_target_peer_edges.parquet"
    author_path = normalized_root / "author_leaveout_prior_peer_base.parquet"
    connection.execute(
        f"CREATE VIEW edges AS SELECT * FROM read_parquet('{_sql_path(edge_path)}')"
    )
    connection.execute(
        f"CREATE VIEW author_base AS SELECT * FROM read_parquet('{_sql_path(author_path)}')"
    )
    connection.execute(
        """
        CREATE TEMP TABLE candidate_tokens AS
        SELECT DISTINCT
            author_numeric_id,
            'https://openalex.org/A' || author_numeric_id::VARCHAR AS author_url
        FROM author_base
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE edge_endpoints AS
        SELECT DISTINCT certified_episode_index, candidate_author_id AS author_numeric_id
        FROM edges
        UNION
        SELECT DISTINCT certified_episode_index, peer_author_id AS author_numeric_id
        FROM edges
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
    event_glob = author_history_root / "events/*.parquet"
    connection.execute(
        f"""
        CREATE TEMP TABLE endpoint_events AS
        SELECT DISTINCT
            w.certified_episode_index,
            w.work_numeric_id,
            e.author_numeric_id,
            w.publication_year,
            w.declared_authors_count
        FROM read_parquet('{_sql_path(event_glob)}') e
        INNER JOIN prior_work w
            ON w.work_numeric_id = e.work_numeric_id
            AND w.publication_year = e.publication_year
            AND w.primary_field_id = e.primary_field_id
            AND w.declared_authors_count = e.declared_authors_count
            AND w.temporal_disposition = e.temporal_disposition
        INNER JOIN edge_endpoints n
            ON n.certified_episode_index = w.certified_episode_index
            AND n.author_numeric_id = e.author_numeric_id
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
        INNER JOIN endpoint_events c
            ON c.certified_episode_index = e.certified_episode_index
            AND c.author_numeric_id = e.candidate_author_id
        INNER JOIN endpoint_events p
            ON p.certified_episode_index = e.certified_episode_index
            AND p.work_numeric_id = c.work_numeric_id
            AND p.author_numeric_id = e.peer_author_id
        """
    )
    edge_difference = _difference_count(
        connection,
        """
        SELECT
            certified_episode_index, candidate_author_id, peer_author_id,
            min(declared_authors_count)::USMALLINT AS minimum_declared_team_size,
            min(publication_year)::USMALLINT AS first_prior_collaboration_year,
            max(publication_year)::USMALLINT AS last_prior_collaboration_year,
            count(*)::UBIGINT AS distinct_shared_work_count
        FROM shared_pair_work GROUP BY 1, 2, 3
        """,
        """
        SELECT
            certified_episode_index, candidate_author_id, peer_author_id,
            minimum_declared_team_size, first_prior_collaboration_year,
            last_prior_collaboration_year, distinct_shared_work_count
        FROM edges
        """,
    )
    require(edge_difference == 0, "normalized shared-work replay differs from edges")
    connection.execute(
        """
        CREATE TEMP TABLE relevant_work_author AS
        SELECT DISTINCT
            certified_episode_index,
            work_numeric_id,
            publication_year,
            declared_authors_count,
            candidate_author_id AS author_numeric_id,
            'https://openalex.org/W' || work_numeric_id::VARCHAR AS work_url,
            'https://openalex.org/A' || candidate_author_id::VARCHAR AS author_url
        FROM shared_pair_work
        UNION
        SELECT DISTINCT
            certified_episode_index,
            work_numeric_id,
            publication_year,
            declared_authors_count,
            peer_author_id AS author_numeric_id,
            'https://openalex.org/W' || work_numeric_id::VARCHAR AS work_url,
            'https://openalex.org/A' || peer_author_id::VARCHAR AS author_url
        FROM shared_pair_work
        """
    )
    return {
        "prior_work_rows": int(_scalar(connection, "SELECT count(*) FROM prior_work")),
        "endpoint_event_rows": int(
            _scalar(connection, "SELECT count(*) FROM endpoint_events")
        ),
        "shared_pair_work_rows": int(
            _scalar(connection, "SELECT count(*) FROM shared_pair_work")
        ),
        "relevant_work_author_rows": int(
            _scalar(connection, "SELECT count(*) FROM relevant_work_author")
        ),
        "edge_bidirectional_row_difference": edge_difference,
    }


def _raw_identity_scan(
    connection: duckdb.DuckDBPyConnection,
    *,
    raw_glob: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    connection.execute(
        f"""
        CREATE TEMP TABLE candidate_raw_identity AS
        SELECT
            c.author_numeric_id,
            count(*)::UBIGINT AS raw_authorship_row_count,
            count(r.authorships__author__orcid)::UBIGINT
                AS nonnull_orcid_row_count,
            list_distinct(
                list(r.authorships__author__orcid)
                    FILTER (r.authorships__author__orcid IS NOT NULL)
            ) AS observed_orcids
        FROM read_parquet('{_sql_path(raw_glob)}') r
        INNER JOIN candidate_tokens c
            ON c.author_url = r.authorships__author__id
        GROUP BY 1
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
        CREATE TEMP TABLE candidate_valid_orcid AS
        SELECT DISTINCT
            a.author_numeric_id,
            value AS orcid
        FROM candidate_raw_identity a
        CROSS JOIN UNNEST(a.observed_orcids) AS u(value)
        WHERE orcid_checksum_valid(value)
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE candidate_identity_metric AS
        SELECT
            a.author_numeric_id,
            a.raw_authorship_row_count,
            a.nonnull_orcid_row_count,
            count(v.orcid)::USMALLINT AS distinct_checksum_valid_orcid_count
        FROM candidate_raw_identity a
        LEFT JOIN candidate_valid_orcid v USING (author_numeric_id)
        GROUP BY 1, 2, 3
        """
    )
    return {
        "candidate_tokens_with_raw_authorship": int(
            _scalar(connection, "SELECT count(*) FROM candidate_raw_identity")
        ),
        "candidate_raw_authorship_rows": int(
            _scalar(
                connection,
                "SELECT sum(raw_authorship_row_count) FROM candidate_raw_identity",
            )
        ),
        "candidate_nonnull_orcid_rows": int(
            _scalar(
                connection,
                "SELECT sum(nonnull_orcid_row_count) FROM candidate_raw_identity",
            )
        ),
        "candidate_tokens_with_checksum_valid_orcid": int(
            _scalar(
                connection,
                """
                SELECT count(*) FROM candidate_identity_metric
                WHERE distinct_checksum_valid_orcid_count > 0
                """,
            )
        ),
        "candidate_tokens_with_single_checksum_valid_orcid": int(
            _scalar(
                connection,
                """
                SELECT count(*) FROM candidate_identity_metric
                WHERE distinct_checksum_valid_orcid_count = 1
                """,
            )
        ),
        "candidate_tokens_with_multiple_checksum_valid_orcids": int(
            _scalar(
                connection,
                """
                SELECT count(*) FROM candidate_identity_metric
                WHERE distinct_checksum_valid_orcid_count > 1
                """,
            )
        ),
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def _raw_institution_scan(
    connection: duckdb.DuckDBPyConnection,
    *,
    raw_glob: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    connection.execute(
        f"""
        CREATE TEMP TABLE author_work_institution AS
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
    null_rows = int(
        _scalar(
            connection,
            """
            SELECT count(*) FROM author_work_institution
            WHERE institution_numeric_id IS NULL
            """,
        )
    )
    require(null_rows == 0, "parsed institution ID is null")
    connection.execute(
        """
        CREATE TEMP TABLE institution_covered_pair_work AS
        SELECT DISTINCT
            s.certified_episode_index,
            s.work_numeric_id,
            s.publication_year,
            s.declared_authors_count,
            s.candidate_author_id,
            s.peer_author_id
        FROM shared_pair_work s
        INNER JOIN (
            SELECT DISTINCT
                certified_episode_index, work_numeric_id, author_numeric_id
            FROM author_work_institution
        ) c
            ON c.certified_episode_index = s.certified_episode_index
            AND c.work_numeric_id = s.work_numeric_id
            AND c.author_numeric_id = s.candidate_author_id
        INNER JOIN (
            SELECT DISTINCT
                certified_episode_index, work_numeric_id, author_numeric_id
            FROM author_work_institution
        ) p
            ON p.certified_episode_index = s.certified_episode_index
            AND p.work_numeric_id = s.work_numeric_id
            AND p.author_numeric_id = s.peer_author_id
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE same_affiliation_pair_work AS
        SELECT DISTINCT
            s.certified_episode_index,
            s.work_numeric_id,
            s.publication_year,
            s.declared_authors_count,
            s.candidate_author_id,
            s.peer_author_id
        FROM shared_pair_work s
        INNER JOIN author_work_institution c
            ON c.certified_episode_index = s.certified_episode_index
            AND c.work_numeric_id = s.work_numeric_id
            AND c.author_numeric_id = s.candidate_author_id
        INNER JOIN author_work_institution p
            ON p.certified_episode_index = s.certified_episode_index
            AND p.work_numeric_id = s.work_numeric_id
            AND p.author_numeric_id = s.peer_author_id
            AND p.institution_numeric_id = c.institution_numeric_id
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE edge_affiliation_metric AS
        WITH comparable AS (
            SELECT
                certified_episode_index,
                candidate_author_id,
                peer_author_id,
                count(*)::UINTEGER AS shared_work_count,
                min(declared_authors_count)::USMALLINT AS minimum_team_size,
                min(publication_year)::USMALLINT AS first_year,
                max(publication_year)::USMALLINT AS last_year
            FROM institution_covered_pair_work
            GROUP BY 1, 2, 3
        ),
        same_affiliation AS (
            SELECT
                certified_episode_index,
                candidate_author_id,
                peer_author_id,
                count(*)::UINTEGER AS shared_work_count,
                min(declared_authors_count)::USMALLINT AS minimum_team_size,
                min(publication_year)::USMALLINT AS first_year,
                max(publication_year)::USMALLINT AS last_year
            FROM same_affiliation_pair_work
            GROUP BY 1, 2, 3
        )
        SELECT
            e.certified_episode_index,
            e.candidate_author_id,
            e.peer_author_id,
            e.minimum_declared_team_size,
            coalesce(c.shared_work_count, 0)::UINTEGER
                AS institution_comparable_shared_work_count,
            coalesce(c.minimum_team_size, 0)::USMALLINT
                AS minimum_institution_comparable_team_size,
            coalesce(c.first_year, 0)::USMALLINT
                AS first_institution_comparable_year,
            coalesce(c.last_year, 0)::USMALLINT
                AS last_institution_comparable_year,
            coalesce(s.shared_work_count, 0)::UINTEGER
                AS same_affiliation_shared_work_count,
            coalesce(s.minimum_team_size, 0)::USMALLINT
                AS minimum_same_affiliation_team_size,
            coalesce(s.first_year, 0)::USMALLINT
                AS first_same_affiliation_year,
            coalesce(s.last_year, 0)::USMALLINT
                AS last_same_affiliation_year,
            (c.shared_work_count IS NOT NULL)
                AS has_current_snapshot_work_time_institution_coverage,
            (s.shared_work_count IS NOT NULL)
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
    return {
        "relevant_author_work_institution_rows": int(
            _scalar(connection, "SELECT count(*) FROM author_work_institution")
        ),
        "relevant_author_work_rows_with_institution": int(
            _scalar(
                connection,
                """
                SELECT count(*) FROM (
                    SELECT DISTINCT
                        certified_episode_index, work_numeric_id, author_numeric_id
                    FROM author_work_institution
                )
                """,
            )
        ),
        "institution_comparable_pair_work_rows": int(
            _scalar(
                connection,
                "SELECT count(*) FROM institution_covered_pair_work",
            )
        ),
        "same_affiliation_pair_work_rows": int(
            _scalar(connection, "SELECT count(*) FROM same_affiliation_pair_work")
        ),
        "edges_with_institution_comparable_shared_work": int(
            _scalar(
                connection,
                """
                SELECT count(*) FROM edge_affiliation_metric
                WHERE has_current_snapshot_work_time_institution_coverage
                """,
            )
        ),
        "edges_with_same_affiliation_shared_work": int(
            _scalar(
                connection,
                """
                SELECT count(*) FROM edge_affiliation_metric
                WHERE has_same_affiliation_shared_prior_work
                """,
            )
        ),
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def _author_output_sql() -> str:
    joins = []
    columns = []
    for cap in CAPS:
        alias = f"m{cap}"
        joins.append(
            f"""
            LEFT JOIN (
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
                    )::UINTEGER AS same_affiliation_peer_count
                FROM edge_affiliation_metric
                GROUP BY 1, 2
            ) {alias}
                ON {alias}.certified_episode_index = a.certified_episode_index
                AND {alias}.candidate_author_id = a.author_numeric_id
            """
        )
        columns.extend(
            [
                f"coalesce({alias}.prior_peer_count, 0)::UINTEGER "
                f"AS prior_target_active_peer_count_cap_{cap}",
                f"coalesce({alias}.comparable_peer_count, 0)::UINTEGER "
                f"AS institution_comparable_prior_peer_count_cap_{cap}",
                f"coalesce({alias}.same_affiliation_peer_count, 0)::UINTEGER "
                f"AS same_affiliation_prior_peer_count_cap_{cap}",
                f"(coalesce({alias}.comparable_peer_count, 0) > 0) "
                f"AS has_institution_comparable_prior_peer_cap_{cap}",
                f"(coalesce({alias}.same_affiliation_peer_count, 0) > 0) "
                f"AS has_same_affiliation_bridge_cap_{cap}",
            ]
        )
    return f"""
        SELECT
            a.certified_episode_index,
            a.field_id,
            a.topic_id,
            a.entry_year,
            a.author_numeric_id,
            coalesce(i.raw_authorship_row_count, 0)::UBIGINT
                AS raw_authorship_row_count,
            coalesce(i.nonnull_orcid_row_count, 0)::UBIGINT
                AS nonnull_orcid_row_count,
            coalesce(i.distinct_checksum_valid_orcid_count, 0)::USMALLINT
                AS distinct_checksum_valid_orcid_count,
            (i.author_numeric_id IS NOT NULL) AS raw_authorship_observed,
            (
                coalesce(i.distinct_checksum_valid_orcid_count, 0) = 1
            ) AS orcid_stable_single_valid_token,
            {", ".join(columns)}
        FROM author_base a
        LEFT JOIN candidate_identity_metric i USING (author_numeric_id)
        {" ".join(joins)}
        ORDER BY a.certified_episode_index, a.author_numeric_id
    """


def _write_parquet(
    table: pa.Table,
    path: Path,
    *,
    schema: pa.Schema,
) -> dict[str, Any]:
    table = table.cast(schema).replace_schema_metadata(schema.metadata)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
        row_group_size=250_000,
    )
    os.chmod(path, 0o600)
    parquet = pq.ParquetFile(path)
    return {
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _cleanup(path: Path | None, *, parent: Path, prefix: str) -> None:
    if path is None:
        return
    require(path.parent == parent and path.name.startswith(prefix), "cleanup boundary")
    if path.exists():
        shutil.rmtree(path)


def build(
    *,
    config_path: Path,
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
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status_before_execution") == "AUTHORIZED_BY_GATE_STATE_V39",
        "component not authorized",
    )
    forbidden = _mapping(config.get("forbidden_inputs_and_outputs"), "forbidden")
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden opened")
    gate = _validate_repository_binding(
        _mapping(config["repository_bindings"]["gate_state"], "gate binding")
    )

    normalized_root = normalized_root.resolve(strict=True)
    parent = _mapping(config["verified_normalized_parent"], "normalized parent")
    require(normalized_root.name == parent["root_basename"], "normalized basename")
    receipt_path = normalized_root / str(parent["receipt_basename"])
    _validate_json_binding(
        receipt_path,
        sha256=str(parent["receipt_sha256"]),
        schema_version=str(parent["receipt_schema_version"]),
        status=str(parent["receipt_status"]),
    )
    for basename_key, sha_key in (
        ("edge_basename", "edge_sha256"),
        ("author_basename", "author_sha256"),
    ):
        path = normalized_root / str(parent[basename_key])
        require(_sha256(path) == parent[sha_key], f"normalized {basename_key}")
    normalized_verification = normalized_root.parent / str(
        parent["verification_basename"]
    )
    _validate_json_binding(
        normalized_verification,
        sha256=str(parent["verification_sha256"]),
        schema_version=str(parent["verification_schema_version"]),
        status=str(parent["verification_status"]),
    )

    normalized_sources = _mapping(config["normalized_sources"], "normalized sources")
    for key, root in (
        ("cutoff_work_metadata", cutoff_root.resolve(strict=True)),
        ("author_history_events", author_history_root.resolve(strict=True)),
    ):
        binding = _mapping(normalized_sources[key], key)
        require(root.name == binding["expected_root_basename"], f"{key} basename")
        receipt = root / str(binding["receipt_basename"])
        require(_sha256(receipt) == binding["receipt_sha256"], f"{key} receipt")
        paths = list((root / str(binding["artifact_prefix"])).glob("*.parquet"))
        require(len(paths) == binding["artifact_count"], f"{key} artifact count")
    plan_binding = _mapping(
        normalized_sources["author_history_events"],
        "author plan",
    )
    plan_path = author_history_root / str(plan_binding["plan_basename"])
    plan = _validate_json_binding(
        plan_path,
        sha256=str(plan_binding["plan_sha256"]),
        schema_version=str(plan_binding["plan_schema_version"]),
    )

    s0 = _mapping(config["s0_semantic_parent"], "S0 parent")
    require(s0_root.name == s0["root_basename"], "S0 basename")
    _validate_json_binding(
        s0_root / str(s0["receipt_basename"]),
        sha256=str(s0["receipt_sha256"]),
    )
    s0_verification = s0_root.parent / str(s0["verification_basename"])
    _validate_json_binding(
        s0_verification,
        sha256=str(s0["verification_sha256"]),
        status=str(s0["required_verification_status"]),
    )
    require(s0["global_multiple_valid_orcid_author_tokens"] == 0, "S0 ORCID multiplicity")

    raw_root = raw_root.resolve(strict=True)
    source = _mapping(config["raw_authorship_source"], "raw source")
    source_paths, source_stats, source_accounting = _source_preflight(
        raw_root,
        plan,
        source,
    )
    raw_glob = raw_root / str(source["table_directory"]) / "*.parquet"
    output = output.resolve()
    require(output.name == config["output_contract"]["root_basename"], "output basename")
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir(), "output parent missing")
    for root in (
        normalized_root,
        cutoff_root.resolve(),
        author_history_root.resolve(),
        raw_root,
        s0_root.resolve(),
    ):
        require(output != root and not output.is_relative_to(root), "output inside input")

    scratch: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.duckdb.", dir=output.parent)
    )
    os.chmod(scratch, 0o700)
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(str(scratch / "build.duckdb"))
        connection.execute(f"SET threads = {workers}")
        connection.execute(f"SET memory_limit = '{memory_limit}'")
        connection.execute(
            f"SET temp_directory = '{_sql_path(scratch / 'spill')}'"
        )
        connection.execute("SET preserve_insertion_order = false")
        normalized_accounting = _create_normalized_relations(
            connection,
            normalized_root=normalized_root,
            cutoff_root=cutoff_root,
            author_history_root=author_history_root,
        )
        identity_accounting = _raw_identity_scan(
            connection,
            raw_glob=raw_glob,
        )
        require(
            identity_accounting[
                "candidate_tokens_with_multiple_checksum_valid_orcids"
            ]
            == 0,
            "candidate token has multiple valid ORCIDs",
        )
        institution_accounting = _raw_institution_scan(
            connection,
            raw_glob=raw_glob,
        )
        require(
            int(_scalar(connection, "SELECT count(*) FROM edge_affiliation_metric"))
            == EXPECTED_EDGE_ROWS,
            "affiliation edge row closure",
        )
        author_table = connection.execute(_author_output_sql()).fetch_arrow_table()
        edge_table = connection.execute(
            """
            SELECT * FROM edge_affiliation_metric
            ORDER BY certified_episode_index, candidate_author_id, peer_author_id
            """
        ).fetch_arrow_table()
        institution_table = connection.execute(
            """
            SELECT * FROM author_work_institution
            ORDER BY
                certified_episode_index,
                work_numeric_id,
                author_numeric_id,
                institution_numeric_id
            """
        ).fetch_arrow_table()
        require(len(author_table) == EXPECTED_AUTHOR_ROWS, "author row closure")
        require(len(edge_table) == EXPECTED_EDGE_ROWS, "edge row closure")
        by_cap = {}
        for cap in CAPS:
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
                FROM ({_author_output_sql()})
                """
            ).fetchone()
            require(row is not None, f"cap {cap} accounting absent")
            by_cap[str(cap)] = {
                "author_rows_with_institution_comparable_peer": int(row[0]),
                "author_rows_with_same_affiliation_bridge": int(row[1]),
                "institution_comparable_prior_peer_edges": int(row[2]),
                "same_affiliation_prior_peer_edges": int(row[3]),
            }
        _validate_source_unchanged(source_paths, source_stats)
        source_accounting["source_exact_stat_valid_after"] = True
    except BaseException:
        if connection is not None:
            connection.close()
            connection = None
        _cleanup(
            scratch,
            parent=output.parent,
            prefix=f".{output.name}.duckdb.",
        )
        scratch = None
        raise
    finally:
        if connection is not None:
            connection.close()

    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    os.chmod(temporary, 0o700)
    try:
        author_artifact = _write_parquet(
            author_table,
            temporary / "candidate_identity_affiliation_base.parquet",
            schema=AUTHOR_SCHEMA,
        )
        edge_artifact = _write_parquet(
            edge_table,
            temporary / "prior_peer_affiliation_edges.parquet",
            schema=EDGE_SCHEMA,
        )
        institution_artifact = _write_parquet(
            institution_table,
            temporary / "relevant_author_work_institutions.parquet",
            schema=INSTITUTION_SCHEMA,
        )
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA,
            "label": config["label"],
            "status": STATUS,
            "scope": {
                "exact_certified_episode_rows": 4,
                "full_raw_authorship_source_scanned": True,
                "raw_authorship_scan_passes": 2,
                "raw_name_read": False,
                "raw_affiliation_string_read": False,
                "country_read": False,
                "corresponding_author_read": False,
                "author_position_read": False,
                "candidate_orcid_sensitivity_built": True,
                "relevant_author_work_institution_relation_built": True,
                "same_affiliation_proxy_built": True,
                "graph_team_cap_profiles": [10, 50, 100],
                "team_cap_selected": False,
                "natural_person_identity_established": False,
                "historical_affiliation_truth_established": False,
                "peer_visible_success_built": False,
                "outcomes_read": False,
                "entry_or_persistence_rows_read": False,
                "final_candidate_eligibility_established": False,
                "independent_verification_complete": False,
                "model_fitting_performed": False,
                "g4_rows_read": False,
                "api_requests_created": 0,
                "gpu_used": False,
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "gate_binding": gate,
            "normalized_parent_binding": {
                "root_basename": normalized_root.name,
                "receipt_sha256": parent["receipt_sha256"],
                "verification_sha256": parent["verification_sha256"],
                "edge_sha256": parent["edge_sha256"],
                "author_sha256": parent["author_sha256"],
            },
            "source_binding": {
                "raw_root_basename": raw_root.name,
                "task_plan_sha256": plan_binding["plan_sha256"],
                "s0_receipt_sha256": s0["receipt_sha256"],
                "s0_verification_sha256": s0["verification_sha256"],
            },
            "source_accounting": source_accounting,
            "normalized_accounting": normalized_accounting,
            "identity_accounting": identity_accounting,
            "institution_accounting": institution_accounting,
            "author_measurement_accounting": {
                "rows": len(author_table),
                "by_cap": by_cap,
                "artifact": author_artifact,
            },
            "edge_affiliation_accounting": {
                "rows": len(edge_table),
                "artifact": edge_artifact,
            },
            "institution_relation_accounting": {
                "rows": len(institution_table),
                "artifact": institution_artifact,
            },
            "authorization": {
                "independent_verification": "REQUIRED_NEXT",
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
                "config_relative_path": config_path.relative_to(REPO_ROOT).as_posix(),
                "config_sha256": _sha256(config_path),
                "builder_relative_path": Path(__file__)
                .resolve()
                .relative_to(REPO_ROOT)
                .as_posix(),
                "builder_sha256": _sha256(Path(__file__).resolve()),
                "raw_join_strategy": (
                    "DUCKDB_COLUMN_PRUNED_CANDIDATE_URL_HASH_JOIN_AND_EXACT_"
                    "RELEVANT_WORK_AUTHOR_INSTITUTION_JOIN"
                ),
            },
            "claim_boundary": config["claim_boundary"],
        }
        report_path = temporary / "raw_authorship_identity_affiliation.json"
        _write_json(report_path, report)
        markdown_path = temporary / "RAW_AUTHORSHIP_IDENTITY_AFFILIATION.md"
        markdown_path.write_text(
            "\n".join(
                [
                    "# Certified raw-authorship identity and affiliation proxy",
                    "",
                    f"Status: `{STATUS}`",
                    "",
                    f"Stable candidate author tokens: "
                    f"{identity_accounting['candidate_tokens_with_single_checksum_valid_orcid']:,}",
                    "",
                    "| Cap | Authors with comparable peer | Authors with same-affiliation bridge |",
                    "|---:|---:|---:|",
                    *[
                        f"| {cap} | "
                        f"{by_cap[str(cap)]['author_rows_with_institution_comparable_peer']:,} | "
                        f"{by_cap[str(cap)]['author_rows_with_same_affiliation_bridge']:,} |"
                        for cap in CAPS
                    ],
                    "",
                    "ORCID is a current-snapshot token sensitivity flag. "
                    "Same-affiliation is a secondary work-time institution-ID "
                    "proxy with missing coverage kept explicit. No cap is selected.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        os.chmod(markdown_path, 0o600)
        artifacts = [
            author_artifact,
            edge_artifact,
            institution_artifact,
            {
                "relative_path": report_path.name,
                "bytes": report_path.stat().st_size,
                "sha256": _sha256(report_path),
            },
            {
                "relative_path": markdown_path.name,
                "bytes": markdown_path.stat().st_size,
                "sha256": _sha256(markdown_path),
            },
        ]
        artifacts.sort(key=lambda item: str(item["relative_path"]))
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": STATUS,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_bytes(
                    [
                        (item["relative_path"], item["bytes"], item["sha256"])
                        for item in artifacts
                    ]
                )
            ).hexdigest(),
            "full_raw_authorship_source_scanned": True,
            "candidate_orcid_sensitivity_built": True,
            "relevant_author_work_institution_relation_built": True,
            "same_affiliation_proxy_built": True,
            "team_cap_selected": False,
            "natural_person_identity_established": False,
            "historical_affiliation_truth_established": False,
            "independent_verification_complete": False,
            "production_pivot_classification_authorized": False,
            "discovery_outcome_opening_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "outcomes_read": False,
            "gpu_used": False,
            "receipt_written_last": True,
        }
        _write_json(
            temporary / "raw_authorship_identity_affiliation.outputs.json",
            receipt,
        )
        os.replace(temporary, output)
        temporary = None
        return report
    finally:
        _cleanup(temporary, parent=output.parent, prefix=f".{output.name}.")
        _cleanup(scratch, parent=output.parent, prefix=f".{output.name}.duckdb.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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
    report = build(
        config_path=args.config,
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
                "status": report["status"],
                "source_accounting": report["source_accounting"],
                "normalized_accounting": report["normalized_accounting"],
                "identity_accounting": report["identity_accounting"],
                "institution_accounting": report["institution_accounting"],
                "author_measurement_accounting": report[
                    "author_measurement_accounting"
                ],
                "execution": report["execution"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
