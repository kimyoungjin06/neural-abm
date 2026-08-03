#!/usr/bin/env python3
"""Independently verify exact-four incumbent and prior-peer leave-out outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_certified_leaveout_prior_peer_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_d0_certified_leaveout_prior_peer_config.v1"
REPORT_SCHEMA = "pivot_feedback_d0_certified_leaveout_prior_peer_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0_certified_leaveout_prior_peer_outputs.v1"
VERIFY_SCHEMA = (
    "pivot_feedback_d0_certified_leaveout_prior_peer_verification.v1"
)
PRODUCER_STATUS = (
    "PASS_EXACT_FOUR_OUTCOME_BLIND_INCUMBENT_FOCAL_AND_PRIOR_TARGET_PEER_LEAVEOUT"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_EXACT_FOUR_OUTCOME_BLIND_INCUMBENT_FOCAL_AND_"
    "PRIOR_TARGET_PEER_LEAVEOUT"
)
TARGET_SCHEMA_VERSION = "pivot_feedback_d0_certified_target_work_incumbency.v1"
EDGE_SCHEMA_VERSION = "pivot_feedback_d0_certified_prior_target_peer_edge.v1"
AUTHOR_SCHEMA_VERSION = (
    "pivot_feedback_d0_certified_author_leaveout_prior_peer_base.v1"
)
CAPS = (10, 50, 100)
PARTITIONS = 4096
EXPECTED_ROOT_FILES = {
    "CERTIFIED_LEAVEOUT_PRIOR_PEER.md",
    "author_leaveout_prior_peer_base.parquet",
    "certified_leaveout_prior_peer.json",
    "certified_leaveout_prior_peer.outputs.json",
    "prior_target_peer_edges.parquet",
    "target_work_incumbency.parquet",
}


class CertifiedLeaveoutPriorPeerVerificationError(RuntimeError):
    """Fail-closed independent verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CertifiedLeaveoutPriorPeerVerificationError(message)


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


def _validate_parquet(
    path: Path,
    *,
    schema_version: str,
    expected_rows: int,
) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == expected_rows, f"{path.name} row count")
    schema = parquet.schema_arrow
    require(
        schema.metadata is not None
        and schema.metadata.get(b"schema_version") == schema_version.encode(),
        f"{path.name} schema version",
    )
    require(all(not field.nullable for field in schema), f"{path.name} nullable field")
    return {
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "schema_version": schema_version,
    }


def _validate_root(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(root.is_absolute() and root.is_dir(), "producer root missing")
    require((root.stat().st_mode & 0o777) == 0o700, "producer root mode")
    observed_files = {path.name for path in root.iterdir() if path.is_file()}
    require(observed_files == EXPECTED_ROOT_FILES, "producer root file closure")
    require(
        not any(path.is_dir() for path in root.iterdir()),
        "producer root contains directory",
    )
    for path in root.iterdir():
        require((path.stat().st_mode & 0o777) == 0o600, f"{path.name} mode")

    receipt_path = root / "certified_leaveout_prior_peer.outputs.json"
    report_path = root / "certified_leaveout_prior_peer.json"
    receipt = _read_json(receipt_path, "producer receipt")
    report = _read_json(report_path, "producer report")
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema")
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema")
    require(receipt.get("status") == PRODUCER_STATUS, "receipt status")
    require(report.get("status") == PRODUCER_STATUS, "report status")
    require(receipt.get("artifact_count") == 5, "receipt artifact count")
    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, list) and len(artifacts) == 5, "receipt artifacts")
    expected_artifacts = EXPECTED_ROOT_FILES - {receipt_path.name}
    require(
        {str(item["relative_path"]) for item in artifacts} == expected_artifacts,
        "receipt artifact closure",
    )
    for item in artifacts:
        path = root / str(item["relative_path"])
        require(path.stat().st_size == int(item["bytes"]), f"{path.name} bytes")
        require(_sha256(path) == item["sha256"], f"{path.name} SHA")
    require(receipt.get("receipt_written_last") is True, "receipt write order")
    for document in (receipt, report):
        serialized = json.dumps(document, sort_keys=True)
        require("OPENAI_API_KEY" not in serialized, "OpenAI secret material")
        require("ANTHROPIC_API_KEY" not in serialized, "Anthropic secret material")
    scope = _mapping(report.get("scope"), "report scope")
    require(scope.get("outcomes_read") is False, "producer outcomes opened")
    require(scope.get("raw_authorship_source_read") is False, "raw authorship opened")
    require(scope.get("team_cap_selected") is False, "team cap selected")
    require(
        scope.get("independent_verification_complete") is False,
        "producer self-certified independent verification",
    )
    forbidden = _mapping(config.get("forbidden_inputs_and_outputs"), "forbidden")
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden boundary")
    return receipt, report, {
        "receipt_sha256": _sha256(receipt_path),
        "report_sha256": _sha256(report_path),
    }


def _validate_bindings(
    config_path: Path,
    config: Mapping[str, Any],
    *,
    parent_root: Path,
    cutoff_root: Path,
    author_history_root: Path,
) -> dict[str, Any]:
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    repository = _mapping(config.get("repository_bindings"), "repository bindings")
    repository_result = {}
    for key in ("gate_state", "measurement_abi"):
        binding = _mapping(repository.get(key), f"{key} binding")
        path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
        require(_sha256(path) == binding["sha256"], f"{key} SHA")
        document = _read_yaml(path, key)
        require(
            document.get("schema_version") == binding["required_schema_version"],
            f"{key} schema",
        )
        require(document.get("status") == binding["required_status"], f"{key} status")
        for dotted, expected in _mapping(
            binding.get("required_fields", {}),
            f"{key} required fields",
        ).items():
            value: Any = document
            for component in str(dotted).split("."):
                value = _mapping(value, f"{key} {dotted}")[component]
            require(value == expected, f"{key} required field {dotted}")
        repository_result[key] = {
            "relative_path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": binding["sha256"],
        }

    parent = _mapping(config.get("verified_parent"), "verified parent")
    require(parent_root.name == parent["root_basename"], "parent basename")
    parent_artifacts = {}
    for binding_key, basename_key, sha_key in (
        ("selected_work", "selected_work_basename", "selected_work_sha256"),
        ("author_base", "author_base_basename", "author_base_sha256"),
        ("receipt", "receipt_basename", "receipt_sha256"),
    ):
        path = parent_root / str(parent[basename_key])
        require(_sha256(path) == parent[sha_key], f"parent {binding_key} SHA")
        parent_artifacts[binding_key] = {
            "basename": path.name,
            "sha256": parent[sha_key],
        }
    parent_verification = parent_root.parent / str(parent["verification_basename"])
    require(
        _sha256(parent_verification) == parent["verification_sha256"],
        "parent verification SHA",
    )
    parent_artifacts["verification"] = {
        "basename": parent_verification.name,
        "sha256": parent["verification_sha256"],
    }

    normalized = _mapping(config.get("normalized_inputs"), "normalized inputs")
    source_result = {}
    for key, root in (
        ("cutoff_work_metadata", cutoff_root),
        ("author_history_events", author_history_root),
    ):
        binding = _mapping(normalized.get(key), f"{key} binding")
        require(root.name == binding["expected_root_basename"], f"{key} basename")
        receipt = root / str(binding["receipt_basename"])
        require(_sha256(receipt) == binding["receipt_sha256"], f"{key} receipt SHA")
        source_receipt = _read_json(receipt, f"{key} receipt")
        require(
            source_receipt.get("schema_version") == binding["receipt_schema_version"],
            f"{key} receipt schema",
        )
        require(
            source_receipt.get("status") == binding["required_status"],
            f"{key} receipt status",
        )
        prefix = str(binding["artifact_prefix"]).rstrip("/")
        paths = sorted((root / prefix).glob("*.parquet"))
        require(len(paths) == binding["artifact_count"], f"{key} artifact count")
        source_result[key] = {
            "root_basename": root.name,
            "receipt_sha256": binding["receipt_sha256"],
            "artifact_count": len(paths),
            "bound_rows": binding["rows"],
        }
    return {
        "config_relative_path": config_path.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": _sha256(config_path),
        "repository": repository_result,
        "parent": parent_artifacts,
        "normalized_sources": source_result,
    }


def _create_views(
    connection: duckdb.DuckDBPyConnection,
    *,
    root: Path,
    parent_root: Path,
) -> None:
    paths = {
        "selected": parent_root / "certified_episode_selected_predecision_works.parquet",
        "candidates": parent_root / "certified_pre_entry_author_opportunity_base.parquet",
        "target_output": root / "target_work_incumbency.parquet",
        "edge_output": root / "prior_target_peer_edges.parquet",
        "author_output": root / "author_leaveout_prior_peer_base.parquet",
    }
    for name, path in paths.items():
        connection.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{_sql_path(path)}')"
        )


def _target_replay(
    connection: duckdb.DuckDBPyConnection,
    *,
    event_glob: str,
) -> dict[str, Any]:
    connection.execute(
        f"""
        CREATE TEMP TABLE target_incidence AS
        SELECT
            s.certified_episode_index,
            s.entry_year,
            s.work_numeric_id,
            s.publication_year,
            e.author_numeric_id
        FROM read_parquet('{_sql_path(event_glob)}') e
        INNER JOIN selected s
            ON s.work_numeric_id = e.work_numeric_id
        WHERE
            e.publication_year = s.publication_year
            AND e.primary_field_id = s.field_id
            AND e.declared_authors_count = s.declared_authors_count
            AND e.temporal_disposition IN (2, 3)
        """
    )
    incidence_rows = int(_scalar(connection, "SELECT count(*) FROM target_incidence"))
    require(incidence_rows == 63_262, "target incidence row closure")
    duplicate_rows = int(
        _scalar(
            connection,
            """
            SELECT count(*) FROM (
                SELECT certified_episode_index, work_numeric_id, author_numeric_id
                FROM target_incidence
                GROUP BY ALL HAVING count(*) <> 1
            )
            """,
        )
    )
    require(duplicate_rows == 0, "target incidence duplicates")
    connection.execute(
        """
        CREATE TEMP TABLE incumbent_authors AS
        SELECT DISTINCT certified_episode_index, author_numeric_id
        FROM target_incidence
        WHERE publication_year < entry_year - 6
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE target_authors AS
        SELECT DISTINCT certified_episode_index, author_numeric_id
        FROM target_incidence
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE expected_target AS
        SELECT
            s.certified_episode_index,
            s.field_id,
            s.topic_id,
            s.entry_year,
            s.work_numeric_id,
            s.publication_year,
            s.work_type_code,
            s.declared_authors_count,
            s.window_code,
            count(t.author_numeric_id)::UBIGINT AS valid_author_token_count,
            count(i.author_numeric_id)::UBIGINT
                AS pre_signal_incumbent_author_count,
            (count(i.author_numeric_id) > 0)
                AS has_pre_signal_incumbent_author,
            (count(i.author_numeric_id) = count(t.author_numeric_id))
                AS all_linked_authors_pre_signal_incumbents
        FROM selected s
        INNER JOIN target_incidence t
            ON t.certified_episode_index = s.certified_episode_index
            AND t.work_numeric_id = s.work_numeric_id
        LEFT JOIN incumbent_authors i
            ON i.certified_episode_index = t.certified_episode_index
            AND i.author_numeric_id = t.author_numeric_id
        GROUP BY ALL
        """
    )
    target_columns = """
        certified_episode_index, field_id, topic_id, entry_year,
        work_numeric_id, publication_year, work_type_code,
        declared_authors_count, window_code, valid_author_token_count,
        pre_signal_incumbent_author_count,
        has_pre_signal_incumbent_author,
        all_linked_authors_pre_signal_incumbents
    """
    difference = _difference_count(
        connection,
        f"SELECT {target_columns} FROM expected_target",
        f"SELECT {target_columns} FROM target_output",
    )
    require(difference == 0, "target incumbent output differs")
    summary_rows = connection.execute(
        """
        SELECT
            count(*) AS target_rows,
            count(*) FILTER (has_pre_signal_incumbent_author) AS any_rows,
            count(*) FILTER (all_linked_authors_pre_signal_incumbents) AS all_rows
        FROM expected_target
        """
    ).fetchone()
    require(summary_rows is not None, "target summary absent")
    by_episode = connection.execute(
        """
        SELECT
            a.certified_episode_index,
            count(DISTINCT a.author_numeric_id),
            count(DISTINCT i.author_numeric_id)
        FROM target_authors a
        LEFT JOIN incumbent_authors i USING (
            certified_episode_index, author_numeric_id
        )
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    return {
        "target_author_incidence_rows": incidence_rows,
        "target_work_rows": int(summary_rows[0]),
        "any_incumbent_target_work_rows": int(summary_rows[1]),
        "all_incumbent_target_work_rows": int(summary_rows[2]),
        "by_episode": {
            str(int(row[0])): {
                "target_authors": int(row[1]),
                "pre_signal_incumbent_authors": int(row[2]),
            }
            for row in by_episode
        },
        "bidirectional_row_difference": difference,
    }


def _edge_replay(
    connection: duckdb.DuckDBPyConnection,
    *,
    work_glob: str,
    event_glob: str,
) -> dict[str, Any]:
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
            publication_year BETWEEN 1948 AND 2011
            AND (
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
    duplicate_work = int(
        _scalar(
            connection,
            """
            SELECT count(*) FROM (
                SELECT work_numeric_id FROM prior_work
                GROUP BY 1 HAVING count(*) <> 1
            )
            """,
        )
    )
    require(duplicate_work == 0, "prior eligible work duplicates")
    connection.execute(
        f"""
        CREATE TEMP TABLE peer_events AS
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
        INNER JOIN target_authors t
            ON t.certified_episode_index = w.certified_episode_index
            AND t.author_numeric_id = e.author_numeric_id
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE peer_work AS
        SELECT DISTINCT
            p.certified_episode_index,
            p.work_numeric_id,
            p.publication_year,
            p.declared_authors_count
        FROM peer_events p
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE candidate_events AS
        SELECT DISTINCT
            w.certified_episode_index,
            w.work_numeric_id,
            e.author_numeric_id,
            w.publication_year,
            w.declared_authors_count
        FROM read_parquet('{_sql_path(event_glob)}') e
        INNER JOIN peer_work w
            ON w.work_numeric_id = e.work_numeric_id
            AND w.publication_year = e.publication_year
            AND w.declared_authors_count = e.declared_authors_count
        INNER JOIN candidates c
            ON c.certified_episode_index = w.certified_episode_index
            AND c.author_numeric_id = e.author_numeric_id
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE expected_edges AS
        SELECT
            c.certified_episode_index,
            c.author_numeric_id AS candidate_author_id,
            p.author_numeric_id AS peer_author_id,
            min(c.declared_authors_count)::USMALLINT
                AS minimum_declared_team_size,
            min(c.publication_year)::USMALLINT
                AS first_prior_collaboration_year,
            max(c.publication_year)::USMALLINT
                AS last_prior_collaboration_year,
            count(DISTINCT c.work_numeric_id)::UBIGINT
                AS distinct_shared_work_count,
            (i.author_numeric_id IS NOT NULL) AS peer_pre_signal_incumbent
        FROM candidate_events c
        INNER JOIN peer_events p
            ON p.certified_episode_index = c.certified_episode_index
            AND p.work_numeric_id = c.work_numeric_id
        LEFT JOIN incumbent_authors i
            ON i.certified_episode_index = p.certified_episode_index
            AND i.author_numeric_id = p.author_numeric_id
        WHERE c.author_numeric_id <> p.author_numeric_id
        GROUP BY
            c.certified_episode_index,
            c.author_numeric_id,
            p.author_numeric_id,
            i.author_numeric_id
        """
    )
    edge_columns = """
        certified_episode_index, candidate_author_id, peer_author_id,
        minimum_declared_team_size, first_prior_collaboration_year,
        last_prior_collaboration_year, distinct_shared_work_count,
        peer_pre_signal_incumbent
    """
    difference = _difference_count(
        connection,
        f"SELECT {edge_columns} FROM expected_edges",
        f"SELECT {edge_columns} FROM edge_output",
    )
    require(difference == 0, "prior target-peer edge output differs")
    work_accounting = connection.execute(
        """
        WITH c AS (
            SELECT
                certified_episode_index, work_numeric_id,
                count(*)::UBIGINT AS candidate_count
            FROM candidate_events GROUP BY 1, 2
        ),
        p AS (
            SELECT
                certified_episode_index, work_numeric_id,
                count(*)::UBIGINT AS peer_count
            FROM peer_events GROUP BY 1, 2
        ),
        self_pairs AS (
            SELECT
                c.certified_episode_index, c.work_numeric_id,
                count(*)::UBIGINT AS self_count
            FROM candidate_events c
            INNER JOIN peer_events p USING (
                certified_episode_index, work_numeric_id, author_numeric_id
            )
            GROUP BY 1, 2
        )
        SELECT
            count(*)::UBIGINT,
            sum(c.candidate_count * p.peer_count)::UBIGINT,
            sum(c.candidate_count * p.peer_count - coalesce(s.self_count, 0))::UBIGINT
        FROM c INNER JOIN p USING (certified_episode_index, work_numeric_id)
        LEFT JOIN self_pairs s USING (certified_episode_index, work_numeric_id)
        """
    ).fetchone()
    require(work_accounting is not None, "edge accounting absent")
    prior_rows = connection.execute(
        """
        SELECT certified_episode_index, count(*)
        FROM prior_work GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    return {
        "prior_work_rows": int(_scalar(connection, "SELECT count(*) FROM prior_work")),
        "prior_work_rows_by_episode": {
            str(int(row[0])): int(row[1]) for row in prior_rows
        },
        "peer_event_rows": int(_scalar(connection, "SELECT count(*) FROM peer_events")),
        "candidate_event_rows_on_peer_works": int(
            _scalar(connection, "SELECT count(*) FROM candidate_events")
        ),
        "common_prior_work_rows": int(work_accounting[0]),
        "candidate_target_peer_pair_expansion_before_self_exclusion": int(
            work_accounting[1]
        ),
        "candidate_target_peer_work_incidences": int(work_accounting[2]),
        "distinct_candidate_target_peer_edges": int(
            _scalar(connection, "SELECT count(*) FROM expected_edges")
        ),
        "bidirectional_row_difference": difference,
        "independent_join_strategy": (
            "DUCKDB_TARGET_PEER_WORK_FIRST_SEMIJOIN_THEN_CANDIDATE_REVERSE_JOIN"
        ),
    }


def _create_exclusions(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    connection.execute(
        """
        CREATE TEMP TABLE peer_exclusions AS
        SELECT
            e.certified_episode_index,
            e.candidate_author_id,
            t.work_numeric_id,
            min(e.minimum_declared_team_size)::USMALLINT
                AS minimum_prior_tie_team_size
        FROM expected_edges e
        INNER JOIN target_incidence t
            ON t.certified_episode_index = e.certified_episode_index
            AND t.author_numeric_id = e.peer_author_id
        GROUP BY 1, 2, 3
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE own_exclusions AS
        SELECT DISTINCT
            t.certified_episode_index,
            t.author_numeric_id AS candidate_author_id,
            t.work_numeric_id,
            0::USMALLINT AS minimum_prior_tie_team_size
        FROM target_incidence t
        INNER JOIN candidates c
            ON c.certified_episode_index = t.certified_episode_index
            AND c.author_numeric_id = t.author_numeric_id
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE network_exclusions AS
        SELECT
            certified_episode_index,
            candidate_author_id,
            work_numeric_id,
            min(minimum_prior_tie_team_size)::USMALLINT
                AS minimum_prior_tie_team_size
        FROM (
            SELECT * FROM own_exclusions
            UNION ALL
            SELECT * FROM peer_exclusions
        )
        GROUP BY 1, 2, 3
        """
    )
    peer_incidence_rows = int(
        _scalar(
            connection,
            """
            SELECT count(*)
            FROM expected_edges e
            INNER JOIN target_incidence t
                ON t.certified_episode_index = e.certified_episode_index
                AND t.author_numeric_id = e.peer_author_id
            """,
        )
    )
    return {
        "peer_target_work_exclusion_incidences": peer_incidence_rows,
        "focal_target_work_exclusion_incidences_before_candidate_filter": int(
            _scalar(connection, "SELECT count(*) FROM target_incidence")
        ),
        "focal_candidate_target_work_rows": int(
            _scalar(connection, "SELECT count(*) FROM own_exclusions")
        ),
        "distinct_candidate_peer_target_work_rows": int(
            _scalar(connection, "SELECT count(*) FROM peer_exclusions")
        ),
        "distinct_candidate_network_target_work_rows": int(
            _scalar(connection, "SELECT count(*) FROM network_exclusions")
        ),
    }


def _cap_expected_sql(cap: int) -> str:
    return f"""
        WITH global_metric AS (
            SELECT
                certified_episode_index,
                count(*) FILTER (
                    has_pre_signal_incumbent_author
                    AND declared_authors_count <= {cap}
                )::UBIGINT AS attention_any,
                count(*) FILTER (
                    all_linked_authors_pre_signal_incumbents
                    AND declared_authors_count <= {cap}
                )::UBIGINT AS attention_all
            FROM expected_target GROUP BY 1
        ),
        edge_metric AS (
            SELECT
                certified_episode_index, candidate_author_id,
                count(*)::UBIGINT AS peer_count,
                count(*) FILTER (peer_pre_signal_incumbent)::UBIGINT
                    AS incumbent_peer_count
            FROM expected_edges
            WHERE minimum_declared_team_size <= {cap}
            GROUP BY 1, 2
        ),
        peer_metric AS (
            SELECT
                p.certified_episode_index, p.candidate_author_id,
                count(*)::UBIGINT AS peer_target_count
            FROM peer_exclusions p
            INNER JOIN expected_target t USING (
                certified_episode_index, work_numeric_id
            )
            WHERE
                p.minimum_prior_tie_team_size <= {cap}
                AND t.declared_authors_count <= {cap}
            GROUP BY 1, 2
        ),
        own_metric AS (
            SELECT
                o.certified_episode_index, o.candidate_author_id,
                count(*) FILTER (
                    t.has_pre_signal_incumbent_author
                )::UBIGINT AS own_any,
                count(*) FILTER (
                    t.all_linked_authors_pre_signal_incumbents
                )::UBIGINT AS own_all
            FROM own_exclusions o
            INNER JOIN expected_target t USING (
                certified_episode_index, work_numeric_id
            )
            WHERE t.declared_authors_count <= {cap}
            GROUP BY 1, 2
        ),
        network_metric AS (
            SELECT
                n.certified_episode_index, n.candidate_author_id,
                count(*) FILTER (
                    t.has_pre_signal_incumbent_author
                )::UBIGINT AS network_any,
                count(*) FILTER (
                    t.all_linked_authors_pre_signal_incumbents
                )::UBIGINT AS network_all
            FROM network_exclusions n
            INNER JOIN expected_target t USING (
                certified_episode_index, work_numeric_id
            )
            WHERE
                n.minimum_prior_tie_team_size <= {cap}
                AND t.declared_authors_count <= {cap}
            GROUP BY 1, 2
        )
        SELECT
            c.certified_episode_index,
            c.field_id,
            c.topic_id,
            c.entry_year,
            c.author_numeric_id,
            coalesce(e.peer_count, 0)::UBIGINT
                AS prior_target_active_peer_count_cap_{cap},
            coalesce(e.incumbent_peer_count, 0)::UBIGINT
                AS prior_incumbent_peer_count_cap_{cap},
            coalesce(p.peer_target_count, 0)::UBIGINT
                AS prior_peer_target_work_count_cap_{cap},
            g.attention_any::UBIGINT
                AS any_incumbent_attention_work_count_cap_{cap},
            (g.attention_any - coalesce(o.own_any, 0))::UBIGINT
                AS any_incumbent_focal_leaveout_work_count_cap_{cap},
            (g.attention_any - coalesce(n.network_any, 0))::UBIGINT
                AS any_incumbent_network_leaveout_work_count_cap_{cap},
            g.attention_all::UBIGINT
                AS all_incumbent_attention_work_count_cap_{cap},
            (g.attention_all - coalesce(o.own_all, 0))::UBIGINT
                AS all_incumbent_focal_leaveout_work_count_cap_{cap},
            (g.attention_all - coalesce(n.network_all, 0))::UBIGINT
                AS all_incumbent_network_leaveout_work_count_cap_{cap},
            (coalesce(e.peer_count, 0) > 0)
                AS has_prior_target_active_peer_cap_{cap}
        FROM candidates c
        INNER JOIN global_metric g USING (certified_episode_index)
        LEFT JOIN edge_metric e
            ON e.certified_episode_index = c.certified_episode_index
            AND e.candidate_author_id = c.author_numeric_id
        LEFT JOIN peer_metric p
            ON p.certified_episode_index = c.certified_episode_index
            AND p.candidate_author_id = c.author_numeric_id
        LEFT JOIN own_metric o
            ON o.certified_episode_index = c.certified_episode_index
            AND o.candidate_author_id = c.author_numeric_id
        LEFT JOIN network_metric n
            ON n.certified_episode_index = c.certified_episode_index
            AND n.candidate_author_id = c.author_numeric_id
    """


def _author_replay(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    identity_columns = (
        "certified_episode_index, field_id, topic_id, entry_year, author_numeric_id"
    )
    identity_difference = _difference_count(
        connection,
        f"SELECT {identity_columns} FROM candidates",
        f"SELECT {identity_columns} FROM author_output",
    )
    require(identity_difference == 0, "author output identity differs")
    result: dict[str, Any] = {
        "author_measurement_rows": int(
            _scalar(connection, "SELECT count(*) FROM candidates")
        ),
        "identity_bidirectional_row_difference": identity_difference,
        "by_cap": {},
    }
    for cap in CAPS:
        expected_sql = _cap_expected_sql(cap)
        columns = (
            f"{identity_columns}, "
            f"prior_target_active_peer_count_cap_{cap}, "
            f"prior_incumbent_peer_count_cap_{cap}, "
            f"prior_peer_target_work_count_cap_{cap}, "
            f"any_incumbent_attention_work_count_cap_{cap}, "
            f"any_incumbent_focal_leaveout_work_count_cap_{cap}, "
            f"any_incumbent_network_leaveout_work_count_cap_{cap}, "
            f"all_incumbent_attention_work_count_cap_{cap}, "
            f"all_incumbent_focal_leaveout_work_count_cap_{cap}, "
            f"all_incumbent_network_leaveout_work_count_cap_{cap}, "
            f"has_prior_target_active_peer_cap_{cap}"
        )
        difference = _difference_count(
            connection,
            f"SELECT {columns} FROM ({expected_sql})",
            f"SELECT {columns} FROM author_output",
        )
        require(difference == 0, f"author cap {cap} output differs")
        summary = connection.execute(
            f"""
            SELECT
                count(*) FILTER (has_prior_target_active_peer_cap_{cap}),
                sum(prior_target_active_peer_count_cap_{cap}),
                sum(prior_incumbent_peer_count_cap_{cap}),
                sum(prior_peer_target_work_count_cap_{cap}),
                sum(
                    any_incumbent_attention_work_count_cap_{cap}
                    - any_incumbent_focal_leaveout_work_count_cap_{cap}
                ),
                sum(
                    any_incumbent_attention_work_count_cap_{cap}
                    - any_incumbent_network_leaveout_work_count_cap_{cap}
                )
            FROM ({expected_sql})
            """
        ).fetchone()
        require(summary is not None, f"author cap {cap} summary absent")
        result["by_cap"][str(cap)] = {
            "candidate_rows_with_prior_target_active_peer": int(summary[0]),
            "prior_target_active_peer_edges": int(summary[1]),
            "prior_incumbent_peer_edges": int(summary[2]),
            "distinct_candidate_peer_target_work_rows": int(summary[3]),
            "focal_any_incumbent_target_work_exclusions": int(summary[4]),
            "network_any_incumbent_target_work_exclusions": int(summary[5]),
            "bidirectional_row_difference": difference,
        }
    monotonic_violations = int(
        _scalar(
            connection,
            """
            SELECT count(*) FROM author_output
            WHERE
                any_incumbent_network_leaveout_work_count_cap_10
                    > any_incumbent_focal_leaveout_work_count_cap_10
                OR any_incumbent_focal_leaveout_work_count_cap_10
                    > any_incumbent_attention_work_count_cap_10
                OR any_incumbent_network_leaveout_work_count_cap_50
                    > any_incumbent_focal_leaveout_work_count_cap_50
                OR any_incumbent_focal_leaveout_work_count_cap_50
                    > any_incumbent_attention_work_count_cap_50
                OR any_incumbent_network_leaveout_work_count_cap_100
                    > any_incumbent_focal_leaveout_work_count_cap_100
                OR any_incumbent_focal_leaveout_work_count_cap_100
                    > any_incumbent_attention_work_count_cap_100
                OR all_incumbent_network_leaveout_work_count_cap_10
                    > all_incumbent_focal_leaveout_work_count_cap_10
                OR all_incumbent_focal_leaveout_work_count_cap_10
                    > all_incumbent_attention_work_count_cap_10
                OR all_incumbent_network_leaveout_work_count_cap_50
                    > all_incumbent_focal_leaveout_work_count_cap_50
                OR all_incumbent_focal_leaveout_work_count_cap_50
                    > all_incumbent_attention_work_count_cap_50
                OR all_incumbent_network_leaveout_work_count_cap_100
                    > all_incumbent_focal_leaveout_work_count_cap_100
                OR all_incumbent_focal_leaveout_work_count_cap_100
                    > all_incumbent_attention_work_count_cap_100
            """,
        )
    )
    require(monotonic_violations == 0, "author leaveout monotonicity")
    result["leaveout_monotonicity_violations"] = monotonic_violations
    return result


def _compare_report(
    report: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    edges: Mapping[str, Any],
    exclusions: Mapping[str, Any],
    authors: Mapping[str, Any],
) -> None:
    target_report = _mapping(report["target_work_accounting"], "target report")
    for key in (
        "target_work_rows",
        "target_author_incidence_rows",
        "any_incumbent_target_work_rows",
        "all_incumbent_target_work_rows",
    ):
        require(int(target_report[key]) == int(target[key]), f"report target {key}")
    edge_report = _mapping(
        report["prior_target_peer_edge_accounting"],
        "edge report",
    )
    for key in (
        "common_prior_work_rows",
        "candidate_target_peer_pair_expansion_before_self_exclusion",
        "candidate_target_peer_work_incidences",
        "distinct_candidate_target_peer_edges",
    ):
        require(int(edge_report[key]) == int(edges[key]), f"report edge {key}")
    exclusion_report = _mapping(report["exclusion_accounting"], "exclusion report")
    require(dict(exclusion_report) == dict(exclusions), "report exclusion accounting")
    author_report = _mapping(
        report["author_measurement_accounting"],
        "author report",
    )
    require(
        int(author_report["author_measurement_rows"])
        == int(authors["author_measurement_rows"]),
        "report author rows",
    )
    for cap in CAPS:
        observed = _mapping(author_report["by_cap"][str(cap)], f"author cap {cap}")
        expected = authors["by_cap"][str(cap)]
        for key in (
            "candidate_rows_with_prior_target_active_peer",
            "prior_target_active_peer_edges",
            "prior_incumbent_peer_edges",
            "distinct_candidate_peer_target_work_rows",
            "focal_any_incumbent_target_work_exclusions",
            "network_any_incumbent_target_work_exclusions",
        ):
            require(int(observed[key]) == int(expected[key]), f"report author {cap} {key}")


def verify(
    *,
    config_path: Path,
    root: Path,
    parent_root: Path,
    cutoff_root: Path,
    author_history_root: Path,
    output: Path,
    workers: int,
    memory_limit: str,
) -> dict[str, Any]:
    started = time.monotonic()
    require(1 <= workers <= 32, "workers must be in [1, 32]")
    config_path = config_path.resolve(strict=True)
    config = _read_yaml(config_path, "config")
    root = root.resolve(strict=True)
    parent_root = parent_root.resolve(strict=True)
    cutoff_root = cutoff_root.resolve(strict=True)
    author_history_root = author_history_root.resolve(strict=True)
    output = output.resolve()
    require(not output.exists(), "verification output already exists")
    require(output.parent.is_dir(), "verification output parent missing")
    require(output.parent == root.parent, "verification output parent differs")
    require(
        output.name == f"{root.name}.verification-v1.json",
        "verification output basename",
    )
    receipt, report, producer_integrity = _validate_root(root, config)
    bindings = _validate_bindings(
        config_path,
        config,
        parent_root=parent_root,
        cutoff_root=cutoff_root,
        author_history_root=author_history_root,
    )
    target_artifact = _validate_parquet(
        root / "target_work_incumbency.parquet",
        schema_version=TARGET_SCHEMA_VERSION,
        expected_rows=30_765,
    )
    edge_artifact = _validate_parquet(
        root / "prior_target_peer_edges.parquet",
        schema_version=EDGE_SCHEMA_VERSION,
        expected_rows=int(
            report["prior_target_peer_edge_accounting"][
                "distinct_candidate_target_peer_edges"
            ]
        ),
    )
    author_artifact = _validate_parquet(
        root / "author_leaveout_prior_peer_base.parquet",
        schema_version=AUTHOR_SCHEMA_VERSION,
        expected_rows=3_256_271,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{root.name}.verification-duckdb.",
            dir=root.parent,
        )
    )
    os.chmod(temporary, 0o700)
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        database = temporary / "replay.duckdb"
        connection = duckdb.connect(str(database))
        connection.execute(f"SET threads = {workers}")
        connection.execute(f"SET memory_limit = '{memory_limit}'")
        connection.execute(
            f"SET temp_directory = '{_sql_path(temporary / 'spill')}'"
        )
        connection.execute("SET preserve_insertion_order = false")
        _create_views(connection, root=root, parent_root=parent_root)
        event_glob = str(author_history_root / "events/*.parquet")
        work_glob = str(cutoff_root / "work_metadata/*.parquet")
        target = _target_replay(connection, event_glob=event_glob)
        edges = _edge_replay(
            connection,
            work_glob=work_glob,
            event_glob=event_glob,
        )
        exclusions = _create_exclusions(connection)
        authors = _author_replay(connection)
        _compare_report(
            report,
            target=target,
            edges=edges,
            exclusions=exclusions,
            authors=authors,
        )
    finally:
        if connection is not None:
            connection.close()
        shutil.rmtree(temporary)

    verification: dict[str, Any] = {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "scope": {
            "exact_certified_episode_rows": 4,
            "full_normalized_source_relational_replay": True,
            "producer_imported": False,
            "producer_algorithm_reused": False,
            "target_incumbency_reconstructed": True,
            "prior_target_peer_edges_reconstructed": True,
            "edge_soundness_and_completeness_checked": True,
            "focal_and_network_leaveouts_reconstructed": True,
            "graph_team_cap_profiles": [10, 50, 100],
            "team_cap_selected": False,
            "all_team_graph_materialized": False,
            "raw_authorship_source_read": False,
            "orcid_stable_subset_built": False,
            "same_affiliation_bridge_built": False,
            "peer_visible_success_built": False,
            "outcomes_read": False,
            "entry_or_persistence_rows_read": False,
            "production_pivot_classification_authorized": False,
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
        "artifact_integrity": {
            "target_work_incumbency": target_artifact,
            "prior_target_peer_edges": edge_artifact,
            "author_leaveout_prior_peer_base": author_artifact,
        },
        "target_replay": target,
        "edge_replay": edges,
        "exclusion_replay": exclusions,
        "author_replay": authors,
        "authorization": {
            "normalized_leaveout_prior_peer_component": "COMPLETE_INDEPENDENTLY_VERIFIED",
            "raw_authorship_orcid_affiliation_successor": "ELIGIBLE_FOR_SUCCESSOR_GATE_ONLY",
            "production_pivot_classification": "NOT_AUTHORIZED",
            "discovery_outcome_opening": "NOT_AUTHORIZED",
            "final_candidate_eligibility": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "execution": {
            "workload_class": "CPU_IO_SQL",
            "workers": workers,
            "memory_limit": memory_limit,
            "duckdb_version": duckdb.__version__,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "verifier_relative_path": Path(__file__)
            .resolve()
            .relative_to(REPO_ROOT)
            .as_posix(),
            "verifier_sha256": _sha256(Path(__file__).resolve()),
            "independent_join_strategy": (
                "DUCKDB_TARGET_PEER_WORK_FIRST_SEMIJOIN_THEN_CANDIDATE_"
                "REVERSE_JOIN_AND_SQL_LEAVEOUT_REDUCTION"
            ),
        },
        "claim_boundary": (
            "This pass independently establishes only exact-four, predecision, "
            "outcome-blind incumbent profiles, focal leave-outs, prior "
            "target-active peer edges, and network leave-outs under three "
            "non-ranked team-cap profiles. It does not select a cap, open raw "
            "authorship, build ORCID or affiliation bridges, construct visible "
            "success, classify pivots, read outcomes, establish final "
            "eligibility, fit a model, open G4, or establish a scientific effect."
        ),
    }
    temporary_output = output.parent / f".{output.name}.{os.getpid()}.tmp"
    require(not temporary_output.exists(), "verification temporary output exists")
    temporary_output.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary_output, 0o600)
    os.replace(temporary_output, output)
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--memory-limit", default="24GB")
    return parser


def main() -> int:
    args = _parser().parse_args()
    verification = verify(
        config_path=args.config,
        root=args.root,
        parent_root=args.parent_root,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        output=args.output,
        workers=args.workers,
        memory_limit=args.memory_limit,
    )
    print(
        json.dumps(
            {
                "status": verification["status"],
                "target_replay": verification["target_replay"],
                "edge_replay": verification["edge_replay"],
                "exclusion_replay": verification["exclusion_replay"],
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
