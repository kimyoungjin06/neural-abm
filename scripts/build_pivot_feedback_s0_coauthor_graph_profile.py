#!/usr/bin/env python3
"""Build exact non-ranked S0 coauthor graph profiles."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import yaml

try:
    import duckdb
except ModuleNotFoundError:
    duckdb = None

if duckdb is None:
    class DuckDBRuntimeError(Exception):
        """Placeholder exception when the repository-only engine is absent."""
else:
    DuckDBRuntimeError = duckdb.Error


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "pivot_feedback_s0_coauthor_graph_profile_config.v1"
REPORT_SCHEMA = "pivot_feedback_s0_coauthor_graph_profile_report.v1"
FIELD_YEAR_SCHEMA = "pivot_feedback_s0_coauthor_graph_field_year.v1"
OUTPUT_SCHEMA = "pivot_feedback_s0_coauthor_graph_profile_outputs.v1"
FULL_STATUS = (
    "PASS_S0_EXACT_NONRANKED_COAUTHOR_GRAPH_PROFILES_"
    "WITH_CURRENT_SNAPSHOT_AUTHOR_TOKEN_LIMITATIONS"
)
PILOT_STATUS = "PILOT_S0_COAUTHOR_GRAPH_PROFILE_NONPROMOTABLE"
PARTITION_COUNT = 4096
BUCKET_COUNT = 256
YEAR_MULTIPLIER = 65_536
FIELD_MIN = 11
FIELD_MAX = 36
EVENT_SCHEMA = pa.schema(
    [
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("temporal_disposition", pa.uint8(), nullable=False),
        pa.field("declared_authors_count", pa.uint32(), nullable=False),
        pa.field("primary_field_id", pa.uint8(), nullable=False),
    ]
)
GRAPH_METRICS = (
    "graph_node_tokens",
    "distinct_undirected_edges",
    "degree_sum",
    "degree_min",
    "degree_p25",
    "degree_p50",
    "degree_p75",
    "degree_p90",
    "degree_p95",
    "degree_p99",
    "degree_max",
    "connected_components",
    "largest_component_nodes",
)


class GraphProfileError(RuntimeError):
    """Fail-closed coauthor graph error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GraphProfileError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return list(value)


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        require(isinstance(current, Mapping) and part in current, f"missing {dotted_path}")
        current = current[part]
    return current


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: Any, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} must be a string")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    require(str(pure) == value, f"{label} must be normalized POSIX")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GraphProfileError(f"invalid {label}: {error}") from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise GraphProfileError(f"invalid {label}: {error}") from error


def _schema_version(schema: pa.Schema) -> str | None:
    value = (schema.metadata or {}).get(b"schema_version")
    return None if value is None else value.decode("utf-8")


def _validate_repository_binding(
    binding: Mapping[str, Any],
    *,
    yaml_document: bool,
) -> dict[str, Any]:
    relative = _safe_relative(binding.get("relative_path"), "repository path")
    path = (REPO_ROOT / relative).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes root")
    require(_sha256_file(path) == binding.get("sha256"), "repository SHA mismatch")
    document = _read_yaml(path, relative) if yaml_document else _read_json(path, relative)
    require(
        document.get("schema_version") == binding.get("required_schema_version"),
        "repository schema mismatch",
    )
    require(document.get("status") == binding.get("required_status"), "repository status mismatch")
    for dotted_path, expected in _mapping(
        binding.get("required_fields"), "required repository fields"
    ).items():
        observed = _nested(document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"repository field mismatch: {dotted_path}",
        )
    return {"relative_path": relative, "sha256": binding["sha256"], "status": document["status"]}


def _validate_core(
    root: Path,
    binding: Mapping[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    summary_path = REPO_ROOT / _safe_relative(
        binding.get("summary_relative_path"), "core summary path"
    )
    require(_sha256_file(summary_path) == binding.get("summary_sha256"), "core summary SHA mismatch")
    summary = _read_json(summary_path, "core summary")
    require(summary.get("schema_version") == binding.get("summary_schema_version"), "core summary schema mismatch")
    require(summary.get("status") == binding.get("required_status"), "core summary status mismatch")
    root = root.resolve(strict=True)
    require(root.name == binding.get("expected_root_basename"), "core root basename mismatch")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256_file(receipt_path) == binding.get("receipt_sha256"), "core receipt SHA mismatch")
    receipt = _read_json(receipt_path, "core receipt")
    require(receipt.get("schema_version") == binding.get("receipt_schema_version"), "core receipt schema mismatch")
    require(receipt.get("status") == binding.get("required_status"), "core receipt status mismatch")
    field_path = root / str(binding["field_year_relative_path"])
    require(_sha256_file(field_path) == binding.get("field_year_sha256"), "core field-year SHA mismatch")
    table = pq.read_table(field_path, columns=["field_id", "publication_year"])
    require(_schema_version(table.schema) == binding.get("field_year_schema_version"), "core field-year schema mismatch")
    require(len(table) == binding.get("expected_field_year_rows"), "core field-year row mismatch")
    fields = table["field_id"].to_pylist()
    years = table["publication_year"].to_pylist()
    keys = [
        int(field) * YEAR_MULTIPLIER + (0 if year is None else int(year))
        for field, year in zip(fields, years, strict=True)
    ]
    require(len(keys) == len(set(keys)), "duplicate core field-year key")
    require(set(fields) == set(range(FIELD_MIN, FIELD_MAX + 1)), "core field coverage mismatch")
    require(sum(year is None for year in years) == binding["expected_null_year_cells"], "core null-year mismatch")
    return keys, {
        "summary_sha256": binding["summary_sha256"],
        "receipt_sha256": binding["receipt_sha256"],
        "field_year_sha256": binding["field_year_sha256"],
        "field_year_rows": len(keys),
    }


def _validate_author_events(
    root: Path,
    binding: Mapping[str, Any],
    *,
    maximum_event_files: int | None,
    hash_workers: int,
) -> tuple[list[Path], dict[str, Any]]:
    root = root.resolve(strict=True)
    require(root.name == binding.get("expected_root_basename"), "author root basename mismatch")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256_file(receipt_path) == binding.get("receipt_sha256"), "author receipt SHA mismatch")
    receipt = _read_json(receipt_path, "author receipt")
    require(receipt.get("schema_version") == binding.get("receipt_schema_version"), "author receipt schema mismatch")
    require(receipt.get("status") == binding.get("required_status"), "author receipt status mismatch")
    prefix = str(binding["artifact_prefix"])
    declared = [
        _mapping(item, "author event artifact")
        for item in _list(receipt.get("artifacts"), "author artifacts")
        if str(_mapping(item, "author event artifact").get("relative_path", "")).startswith(prefix)
    ]
    declared.sort(key=lambda item: str(item["relative_path"]))
    require(len(declared) == binding.get("expected_artifact_count"), "author event artifact count mismatch")
    for index, item in enumerate(declared):
        require(item.get("relative_path") == f"{prefix}part-{index:04d}.parquet", "author partition sequence mismatch")
    if maximum_event_files is not None:
        require(1 <= maximum_event_files <= len(declared), "invalid event pilot limit")
        declared = declared[:maximum_event_files]
    paths: list[Path] = []
    pre_stats: list[tuple[int, int, int, int, int]] = []
    for item in declared:
        path = (root / str(item["relative_path"])).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file() and not path.is_symlink(), "invalid author event artifact")
        require(path.stat().st_size == item.get("bytes"), "author event byte mismatch")
        require(pq.ParquetFile(path).schema_arrow == EVENT_SCHEMA, "author event schema mismatch")
        info = path.stat()
        paths.append(path)
        pre_stats.append(
            (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        )

    def rehash(pair: tuple[Path, Mapping[str, Any]]) -> tuple[str, int, str]:
        path, item = pair
        observed = _sha256_file(path)
        require(observed == item.get("sha256"), f"author event SHA mismatch: {item['relative_path']}")
        return str(item["relative_path"]), int(item["bytes"]), observed

    with ThreadPoolExecutor(max_workers=hash_workers) as executor:
        verified = list(executor.map(rehash, zip(paths, declared, strict=True)))
    post_stats = []
    for path in paths:
        info = path.stat()
        post_stats.append(
            (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        )
    require(pre_stats == post_stats, "author event artifact changed during verification")
    tree = hashlib.sha256(_canonical_bytes(verified)).hexdigest()
    selected_bytes = sum(item[1] for item in verified)
    if maximum_event_files is None:
        require(selected_bytes == binding.get("expected_artifact_bytes"), "full author event byte mismatch")
        require(tree == binding.get("expected_artifact_tree_sha256"), "full author event tree mismatch")
    return paths, {
        "receipt_sha256": binding["receipt_sha256"],
        "selected_artifacts_rehashed": len(paths),
        "selected_artifact_bytes": selected_bytes,
        "selected_artifact_tree_sha256": tree,
        "selected_artifact_stats_unchanged": True,
    }


def _available_memory_bytes(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            require(len(parts) == 3 and parts[1].isdigit() and parts[2] == "kB", "invalid MemAvailable")
            return int(parts[1]) * 1024
    raise GraphProfileError("MemAvailable is missing")


def _resource_preflight(output_parent: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    resource = _mapping(config.get("resource_envelope"), "resource envelope")
    memory = _available_memory_bytes()
    disk = shutil.disk_usage(output_parent).free
    require(memory >= resource["minimum_available_memory_bytes"], "insufficient available memory")
    require(disk >= resource["minimum_free_output_bytes"], "insufficient output disk")
    return {
        "available_memory_bytes": memory,
        "free_output_bytes": disk,
        "cpu_count": os.cpu_count(),
        "gpu_used": False,
    }


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _paths_sql(paths: Sequence[Path]) -> str:
    return "[" + ",".join(_sql_string(str(path)) for path in paths) + "]"


def _profile_caps(config: Mapping[str, Any]) -> list[int]:
    graph = _mapping(config.get("graph_contract"), "graph contract")
    caps = [
        int(_mapping(item, "graph profile").get("cap"))
        for item in _list(graph.get("profiles"), "graph profiles")
    ]
    require(caps == [10, 50, 100], "graph cap contract mismatch")
    require(graph.get("profiles_are_ranked_or_selected") is False, "graph caps are selected")
    require(graph.get("all_profiles_must_be_reported") is True, "not all graph caps are required")
    return caps


def _materialize_exact_graph(
    connection: duckdb.DuckDBPyConnection,
    event_paths: Sequence[Path],
    scratch: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    execution = _mapping(config.get("execution"), "execution")
    graph_contract = _mapping(config.get("graph_contract"), "graph contract")
    require(
        graph_contract.get("null_publication_year_cell_encoding")
        == "ZERO_ONLY_FOR_INTERNAL_BUCKETING_OUTPUT_REMAINS_NULL",
        "null publication-year cell contract mismatch",
    )
    temp_path = scratch / "duckdb-temp"
    temp_path.mkdir(mode=0o700)
    connection.execute(f"SET threads = {int(execution['duckdb_threads'])}")
    connection.execute(f"SET memory_limit = {_sql_string(str(execution['duckdb_memory_limit']))}")
    connection.execute(f"SET temp_directory = {_sql_string(str(temp_path))}")
    connection.execute(
        "SET max_temp_directory_size = "
        + _sql_string(str(execution["duckdb_max_temp_directory_size"]))
    )
    connection.execute("SET preserve_insertion_order = false")
    source_sql = _paths_sql(event_paths)
    print("S0_GRAPH stage=source_events_materialize", file=sys.stderr, flush=True)
    connection.execute(
        f"""
        CREATE TABLE source_events AS
        SELECT
          author_numeric_id,
          work_numeric_id,
          publication_year,
          declared_authors_count,
          primary_field_id,
          CAST(regexp_extract(filename, 'part-([0-9]{{4}})[.]parquet$', 1)
               AS USMALLINT) AS source_partition
        FROM read_parquet({source_sql}, filename = true, union_by_name = false)
        """
    )
    source_profile = connection.execute(
        """
        SELECT
          COUNT(*)::UBIGINT AS rows,
          COUNT_IF(author_numeric_id % 4096 != source_partition)::UBIGINT
            AS partition_violations,
          COUNT_IF(primary_field_id BETWEEN 11 AND 36)::UBIGINT
            AS classified_rows,
          COUNT_IF(primary_field_id BETWEEN 11 AND 36
                   AND declared_authors_count <= 10)::UBIGINT AS event_rows_cap_10,
          COUNT_IF(primary_field_id BETWEEN 11 AND 36
                   AND declared_authors_count <= 50)::UBIGINT AS event_rows_cap_50,
          COUNT_IF(primary_field_id BETWEEN 11 AND 36
                   AND declared_authors_count <= 100)::UBIGINT AS event_rows_cap_100
        FROM source_events
        """
    ).fetchone()
    assert source_profile is not None
    require(int(source_profile[1]) == 0, "author event partition formula mismatch")
    connection.execute(
        """
        CREATE TABLE eligible_events AS
        SELECT
          author_numeric_id,
          work_numeric_id,
          publication_year,
          declared_authors_count,
          primary_field_id,
          CASE
            WHEN declared_authors_count <= 10 THEN 10
            WHEN declared_authors_count <= 50 THEN 50
            ELSE 100
          END::UTINYINT AS minimum_cap
        FROM source_events
        WHERE primary_field_id BETWEEN 11 AND 36
          AND declared_authors_count <= 100
        """
    )
    connection.execute("DROP TABLE source_events")
    print("S0_GRAPH stage=work_profile_materialize", file=sys.stderr, flush=True)
    connection.execute(
        """
        CREATE TABLE work_profile AS
        SELECT
          work_numeric_id,
          MIN(publication_year)::USMALLINT AS publication_year,
          MIN(primary_field_id)::UTINYINT AS primary_field_id,
          MIN(declared_authors_count)::UINTEGER AS declared_authors_count,
          COUNT(*)::UINTEGER AS actual_author_tokens,
          COUNT(DISTINCT author_numeric_id)::UINTEGER AS distinct_author_tokens,
          COUNT(DISTINCT publication_year)::UINTEGER AS year_versions,
          COUNT(DISTINCT primary_field_id)::UINTEGER AS field_versions,
          COUNT(DISTINCT declared_authors_count)::UINTEGER AS declared_count_versions
        FROM eligible_events
        GROUP BY work_numeric_id
        """
    )
    violations = connection.execute(
        """
        SELECT
          COUNT_IF(actual_author_tokens != distinct_author_tokens)::UBIGINT,
          COUNT_IF(year_versions != 1 OR field_versions != 1
                   OR declared_count_versions != 1)::UBIGINT,
          COUNT_IF(actual_author_tokens > declared_authors_count)::UBIGINT
        FROM work_profile
        """
    ).fetchone()
    assert violations is not None
    require(tuple(map(int, violations)) == (0, 0, 0), "eligible work invariant violation")
    incidence = connection.execute(
        """
        SELECT
          COUNT(*)::UBIGINT AS work_groups,
          SUM(actual_author_tokens)::HUGEINT AS actual_author_event_rows,
          SUM(CASE WHEN declared_authors_count <= 10
              THEN actual_author_tokens::HUGEINT * (actual_author_tokens - 1) / 2
              ELSE 0 END)::HUGEINT AS actual_pair_incidences_cap_10,
          SUM(CASE WHEN declared_authors_count <= 50
              THEN actual_author_tokens::HUGEINT * (actual_author_tokens - 1) / 2
              ELSE 0 END)::HUGEINT AS actual_pair_incidences_cap_50,
          SUM(actual_author_tokens::HUGEINT * (actual_author_tokens - 1) / 2)
            ::HUGEINT AS actual_pair_incidences_cap_100
        FROM work_profile
        """
    ).fetchone()
    assert incidence is not None
    connection.execute("DROP TABLE work_profile")
    node_root = scratch / "nodes"
    edge_root = scratch / "edges"
    print("S0_GRAPH stage=node_materialize", file=sys.stderr, flush=True)
    connection.execute(
        f"""
        COPY (
          SELECT
            primary_field_id::UTINYINT AS field_id,
            publication_year::USMALLINT AS publication_year,
            author_numeric_id::UBIGINT AS author_numeric_id,
            MIN(minimum_cap)::UTINYINT AS minimum_cap,
            ((primary_field_id::UINTEGER * {YEAR_MULTIPLIER}
              + COALESCE(publication_year::UINTEGER, 0))
              % {BUCKET_COUNT})::UTINYINT
              AS cell_bucket
          FROM eligible_events
          GROUP BY primary_field_id, publication_year, author_numeric_id
        )
        TO {_sql_string(str(node_root))}
        (FORMAT PARQUET, PARTITION_BY (cell_bucket), COMPRESSION ZSTD)
        """
    )
    print("S0_GRAPH stage=edge_materialize", file=sys.stderr, flush=True)
    connection.execute(
        f"""
        COPY (
          SELECT
            a.primary_field_id::UTINYINT AS field_id,
            a.publication_year::USMALLINT AS publication_year,
            a.author_numeric_id::UBIGINT AS author_u,
            b.author_numeric_id::UBIGINT AS author_v,
            MIN(a.minimum_cap)::UTINYINT AS minimum_cap,
            ((a.primary_field_id::UINTEGER * {YEAR_MULTIPLIER}
              + COALESCE(a.publication_year::UINTEGER, 0))
              % {BUCKET_COUNT})::UTINYINT
              AS cell_bucket
          FROM eligible_events AS a
          INNER JOIN eligible_events AS b
            ON a.work_numeric_id = b.work_numeric_id
           AND a.publication_year IS NOT DISTINCT FROM b.publication_year
           AND a.primary_field_id = b.primary_field_id
           AND a.declared_authors_count = b.declared_authors_count
           AND a.author_numeric_id < b.author_numeric_id
          GROUP BY
            a.primary_field_id,
            a.publication_year,
            a.author_numeric_id,
            b.author_numeric_id
        )
        TO {_sql_string(str(edge_root))}
        (FORMAT PARQUET, PARTITION_BY (cell_bucket), COMPRESSION ZSTD)
        """
    )
    node_paths = sorted(node_root.rglob("*.parquet"))
    edge_paths = sorted(edge_root.rglob("*.parquet"))
    require(node_paths, "node materialization is empty")
    node_counts = connection.execute(
        f"""
        SELECT
          COUNT_IF(minimum_cap <= 10)::UBIGINT,
          COUNT_IF(minimum_cap <= 50)::UBIGINT,
          COUNT_IF(minimum_cap <= 100)::UBIGINT
        FROM read_parquet({_paths_sql(node_paths)})
        """
    ).fetchone()
    assert node_counts is not None
    edge_counts = (
        connection.execute(
            f"""
            SELECT
              COUNT_IF(minimum_cap <= 10)::UBIGINT,
              COUNT_IF(minimum_cap <= 50)::UBIGINT,
              COUNT_IF(minimum_cap <= 100)::UBIGINT
            FROM read_parquet({_paths_sql(edge_paths)})
            """
        ).fetchone()
        if edge_paths
        else (0, 0, 0)
    )
    assert edge_counts is not None
    connection.execute("DROP TABLE eligible_events")
    return {
        "source_rows": int(source_profile[0]),
        "partition_violations": int(source_profile[1]),
        "classified_rows": int(source_profile[2]),
        "event_rows_cap_10": int(source_profile[3]),
        "event_rows_cap_50": int(source_profile[4]),
        "event_rows_cap_100": int(source_profile[5]),
        "eligible_work_groups": int(incidence[0]),
        "actual_author_event_rows_cap_100": int(incidence[1]),
        "actual_pair_incidences_cap_10": int(incidence[2]),
        "actual_pair_incidences_cap_50": int(incidence[3]),
        "actual_pair_incidences_cap_100": int(incidence[4]),
        "materialized_node_tokens_cap_10": int(node_counts[0]),
        "materialized_node_tokens_cap_50": int(node_counts[1]),
        "materialized_node_tokens_cap_100": int(node_counts[2]),
        "materialized_distinct_edges_cap_10": int(edge_counts[0]),
        "materialized_distinct_edges_cap_50": int(edge_counts[1]),
        "materialized_distinct_edges_cap_100": int(edge_counts[2]),
        "node_root": str(node_root),
        "edge_root": str(edge_root),
    }


def _order_statistics(
    degrees: np.ndarray,
    quantiles: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    if len(degrees) == 0:
        return {str(item["label"]): 0 for item in quantiles}
    ordered = np.sort(degrees)
    result = {}
    for item in quantiles:
        numerator = int(item["numerator"])
        denominator = int(item["denominator"])
        index = numerator * (len(ordered) - 1) // denominator
        result[str(item["label"])] = int(ordered[index])
    return result


def _cell_profile_metrics(
    *,
    node_authors: np.ndarray,
    node_minimum_caps: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_minimum_caps: np.ndarray,
    cap: int,
    quantiles: Sequence[Mapping[str, Any]],
) -> dict[str, int | float | None]:
    nodes = node_authors[node_minimum_caps <= cap]
    require(len(nodes) == 0 or bool(np.all(nodes[1:] > nodes[:-1])), "node tokens are not unique and sorted")
    selected_edges = edge_minimum_caps <= cap
    u = edge_u[selected_edges]
    v = edge_v[selected_edges]
    require(bool(np.all(u < v)), "edge orientation violation")
    if len(u):
        order_ok = (u[1:] > u[:-1]) | ((u[1:] == u[:-1]) & (v[1:] > v[:-1]))
        require(bool(np.all(order_ok)), "edges are not unique and sorted")
    node_count = len(nodes)
    edge_count = len(u)
    if node_count == 0:
        require(edge_count == 0, "edges exist without nodes")
        result: dict[str, int | float | None] = {
            "graph_node_tokens": 0,
            "distinct_undirected_edges": 0,
            "degree_sum": 0,
            **_order_statistics(np.empty(0, dtype=np.uint64), quantiles),
            "connected_components": 0,
            "largest_component_nodes": 0,
            "largest_component_share": None,
        }
        return result
    source_index = np.searchsorted(nodes, u)
    target_index = np.searchsorted(nodes, v)
    require(
        bool(
            np.all(source_index < node_count)
            and np.all(target_index < node_count)
            and np.all(nodes[source_index] == u)
            and np.all(nodes[target_index] == v)
        ),
        "edge endpoint missing from node universe",
    )
    endpoints = np.concatenate((source_index, target_index))
    degrees = np.bincount(endpoints, minlength=node_count).astype(np.uint64, copy=False)
    require(int(degrees.sum()) == 2 * edge_count, "degree sum does not close")
    if edge_count == 0:
        component_count = node_count
        largest = 1
    else:
        rows = np.concatenate((source_index, target_index))
        columns = np.concatenate((target_index, source_index))
        adjacency = csr_matrix(
            (np.ones(len(rows), dtype=np.bool_), (rows, columns)),
            shape=(node_count, node_count),
        )
        component_count, labels = connected_components(
            adjacency, directed=False, return_labels=True
        )
        largest = int(np.bincount(labels).max())
    require(component_count >= node_count - edge_count, "component lower bound failed")
    result = {
        "graph_node_tokens": node_count,
        "distinct_undirected_edges": edge_count,
        "degree_sum": int(degrees.sum()),
        **_order_statistics(degrees, quantiles),
        "connected_components": int(component_count),
        "largest_component_nodes": largest,
        "largest_component_share": largest / node_count,
    }
    return result


def _slice_map(cells: np.ndarray) -> dict[int, tuple[int, int]]:
    if len(cells) == 0:
        return {}
    unique, first, counts = np.unique(cells, return_index=True, return_counts=True)
    return {
        int(cell): (int(start), int(start + count))
        for cell, start, count in zip(unique, first, counts, strict=True)
    }


def _bucket_table(
    connection: duckdb.DuckDBPyConnection,
    root: Path,
    bucket: int,
    columns: str,
    order: str,
) -> pa.Table:
    directory = root / f"cell_bucket={bucket}"
    if not directory.is_dir():
        return pa.table({})
    paths = sorted(directory.glob("*.parquet"))
    require(paths, f"empty materialization bucket: {directory}")
    return connection.execute(
        f"SELECT {columns} FROM read_parquet({_paths_sql(paths)}) ORDER BY {order}"
    ).to_arrow_table()


def _compute_graph_inventory(
    connection: duckdb.DuckDBPyConnection,
    *,
    core_keys: Sequence[int],
    node_root: Path,
    edge_root: Path,
    config: Mapping[str, Any],
) -> pa.Table:
    caps = _profile_caps(config)
    graph = _mapping(config.get("graph_contract"), "graph contract")
    quantiles = [
        _mapping(item, "degree quantile")
        for item in _list(graph.get("degree_quantiles"), "degree quantiles")
    ]
    rows: list[dict[str, Any]] = []
    keys_by_bucket: dict[int, list[int]] = {}
    for key in core_keys:
        keys_by_bucket.setdefault(key % BUCKET_COUNT, []).append(key)
    for bucket in range(BUCKET_COUNT):
        node_table = _bucket_table(
            connection,
            node_root,
            bucket,
            "field_id, publication_year, author_numeric_id, minimum_cap",
            "field_id, publication_year, author_numeric_id",
        )
        edge_table = _bucket_table(
            connection,
            edge_root,
            bucket,
            "field_id, publication_year, author_u, author_v, minimum_cap",
            "field_id, publication_year, author_u, author_v",
        )
        if len(node_table):
            node_fields = node_table["field_id"].combine_chunks().to_numpy(zero_copy_only=False)
            node_years = (
                pc.fill_null(node_table["publication_year"], 0)
                .combine_chunks()
                .to_numpy(zero_copy_only=False)
            )
            node_cells = node_fields.astype(np.uint64) * YEAR_MULTIPLIER + node_years
            node_authors = node_table["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
            node_caps = node_table["minimum_cap"].combine_chunks().to_numpy(zero_copy_only=False)
        else:
            node_cells = np.empty(0, dtype=np.uint64)
            node_authors = np.empty(0, dtype=np.uint64)
            node_caps = np.empty(0, dtype=np.uint8)
        if len(edge_table):
            edge_fields = edge_table["field_id"].combine_chunks().to_numpy(zero_copy_only=False)
            edge_years = (
                pc.fill_null(edge_table["publication_year"], 0)
                .combine_chunks()
                .to_numpy(zero_copy_only=False)
            )
            edge_cells = edge_fields.astype(np.uint64) * YEAR_MULTIPLIER + edge_years
            edge_u = edge_table["author_u"].combine_chunks().to_numpy(zero_copy_only=False)
            edge_v = edge_table["author_v"].combine_chunks().to_numpy(zero_copy_only=False)
            edge_caps = edge_table["minimum_cap"].combine_chunks().to_numpy(zero_copy_only=False)
        else:
            edge_cells = np.empty(0, dtype=np.uint64)
            edge_u = np.empty(0, dtype=np.uint64)
            edge_v = np.empty(0, dtype=np.uint64)
            edge_caps = np.empty(0, dtype=np.uint8)
        node_slices = _slice_map(node_cells)
        edge_slices = _slice_map(edge_cells)
        require(set(node_slices).issubset(set(core_keys)), "node cell missing from core")
        require(set(edge_slices).issubset(set(core_keys)), "edge cell missing from core")
        for key in keys_by_bucket.get(bucket, []):
            node_start, node_stop = node_slices.get(key, (0, 0))
            edge_start, edge_stop = edge_slices.get(key, (0, 0))
            field_id, year = divmod(key, YEAR_MULTIPLIER)
            prior: dict[str, int] | None = None
            for cap in caps:
                metrics = _cell_profile_metrics(
                    node_authors=node_authors[node_start:node_stop],
                    node_minimum_caps=node_caps[node_start:node_stop],
                    edge_u=edge_u[edge_start:edge_stop],
                    edge_v=edge_v[edge_start:edge_stop],
                    edge_minimum_caps=edge_caps[edge_start:edge_stop],
                    cap=cap,
                    quantiles=quantiles,
                )
                if prior is not None:
                    require(
                        int(metrics["graph_node_tokens"]) >= prior["graph_node_tokens"]
                        and int(metrics["distinct_undirected_edges"])
                        >= prior["distinct_undirected_edges"],
                        "nested graph profile monotonicity failed",
                    )
                prior = {
                    "graph_node_tokens": int(metrics["graph_node_tokens"]),
                    "distinct_undirected_edges": int(metrics["distinct_undirected_edges"]),
                }
                rows.append(
                    {
                        "field_id": field_id,
                        "publication_year": None if year == 0 else year,
                        "declared_team_cap": cap,
                        **metrics,
                    }
                )
        if (bucket + 1) % 16 == 0:
            print(
                f"S0_GRAPH stage=components buckets={bucket + 1}/{BUCKET_COUNT}",
                file=sys.stderr,
                flush=True,
            )
    rows.sort(
        key=lambda row: (
            row["field_id"],
            -1 if row["publication_year"] is None else row["publication_year"],
            row["declared_team_cap"],
        )
    )
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("publication_year", pa.uint16(), nullable=True),
            pa.field("declared_team_cap", pa.uint16(), nullable=False),
            *[
                pa.field(metric, pa.uint64(), nullable=False)
                for metric in GRAPH_METRICS
            ],
            pa.field("largest_component_share", pa.float64(), nullable=True),
        ],
        metadata={b"schema_version": FIELD_YEAR_SCHEMA.encode()},
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    os.chmod(path, 0o600)
    _fsync_file(path)


def _write_parquet(path: Path, table: pa.Table) -> dict[str, Any]:
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
    _fsync_file(path)
    return {
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "rows": len(table),
    }


def build(
    *,
    config_path: Path,
    author_root: Path,
    core_root: Path,
    output: Path,
    maximum_event_files: int | None,
    hash_workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    config_path = config_path.resolve(strict=True)
    config = _read_yaml(config_path, "graph config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(config.get("label") == "pivot_feedback_s0_coauthor_graph_profile_v1", "config label mismatch")
    require(config.get("status_before_execution") == "AUTHORIZED_S0_REMAINING_COMPONENT", "graph component not authorized")
    require(
        duckdb is not None,
        "DuckDB is required; run with --with duckdb==1.5.5",
    )
    require(duckdb.__version__ == config["execution"]["required_duckdb_version"], "DuckDB version mismatch")
    require(1 <= hash_workers <= 64, "invalid hash worker count")
    require(
        config["execution"]["edge_output_partition_count"] == BUCKET_COUNT,
        "edge bucket contract mismatch",
    )
    require(
        config["execution"]["exact_intermediate_node_and_edge_materialization_required"]
        is True
        and config["execution"][
            "exact_intermediate_materialization_removed_before_completion"
        ]
        is True,
        "exact intermediate contract mismatch",
    )
    gate_parent = _validate_repository_binding(
        _mapping(config.get("gate_parent"), "gate parent"), yaml_document=True
    )
    author_root = author_root.resolve(strict=True)
    core_root = core_root.resolve(strict=True)
    output = output.resolve()
    require(not output.exists() and output.parent.is_dir(), "output must be a new directory")
    require(
        not output.is_relative_to(author_root)
        and not output.is_relative_to(core_root),
        "output may not be inside an input root",
    )
    preflight = _resource_preflight(output.parent, config)
    core_keys, core_binding = _validate_core(
        core_root, _mapping(config.get("s0_core_parent"), "core parent")
    )
    event_paths, event_binding = _validate_author_events(
        author_root,
        _mapping(config.get("author_history_parent"), "author parent"),
        maximum_event_files=maximum_event_files,
        hash_workers=hash_workers,
    )
    full_scope = maximum_event_files is None
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    scratch = temporary / "scratch"
    scratch.mkdir(mode=0o700)
    database_path = scratch / "graph.duckdb"
    connection = duckdb.connect(str(database_path))
    try:
        materialization = _materialize_exact_graph(
            connection, event_paths, scratch, config
        )
        author_binding = _mapping(config.get("author_history_parent"), "author parent")
        if full_scope:
            require(materialization["source_rows"] == author_binding["expected_rows"], "full event row closure failed")
            core_summary = _read_json(
                REPO_ROOT / config["s0_core_parent"]["summary_relative_path"],
                "core summary",
            )
            require(
                materialization["classified_rows"]
                == core_summary["author_event_accounting"]["classified_field_rows"],
                "classified event margin mismatch",
            )
        graph_contract = _mapping(config.get("graph_contract"), "graph contract")
        require(
            materialization["actual_author_event_rows_cap_100"]
            == materialization["event_rows_cap_100"],
            "eligible event-to-work margin mismatch",
        )
        require(
            materialization["actual_pair_incidences_cap_10"]
            <= materialization["actual_pair_incidences_cap_50"]
            <= materialization["actual_pair_incidences_cap_100"],
            "actual pair-incidence profiles are not monotone",
        )
        for cap in (10, 50, 100):
            require(
                materialization[f"actual_pair_incidences_cap_{cap}"]
                <= graph_contract[
                    f"cap_{cap}_classified_declared_pair_incidences"
                ],
                f"actual pair incidences exceed declared profile: {cap}",
            )
        graph_table = _compute_graph_inventory(
            connection,
            core_keys=core_keys,
            node_root=Path(materialization.pop("node_root")),
            edge_root=Path(materialization.pop("edge_root")),
            config=config,
        )
        expected_rows = len(core_keys) * len(_profile_caps(config))
        require(len(graph_table) == expected_rows, "graph field-year row closure failed")
        sums = {
            metric: {
                str(cap): int(
                    sum(
                        value
                        for value, observed_cap in zip(
                            graph_table[metric].to_pylist(),
                            graph_table["declared_team_cap"].to_pylist(),
                            strict=True,
                        )
                        if observed_cap == cap
                    )
                )
                for cap in (10, 50, 100)
            }
            for metric in (
                "graph_node_tokens",
                "distinct_undirected_edges",
                "connected_components",
                "largest_component_nodes",
            )
        }
        for cap in (10, 50, 100):
            require(
                sums["distinct_undirected_edges"][str(cap)]
                <= materialization[f"actual_pair_incidences_cap_{cap}"],
                f"distinct edges exceed actual incidences: {cap}",
            )
            require(
                sums["graph_node_tokens"][str(cap)]
                == materialization[f"materialized_node_tokens_cap_{cap}"],
                f"materialized node tokens do not close: {cap}",
            )
            require(
                sums["distinct_undirected_edges"][str(cap)]
                == materialization[f"materialized_distinct_edges_cap_{cap}"],
                f"materialized distinct edges do not close: {cap}",
            )
        graph_artifact = _write_parquet(
            temporary / "field_year_coauthor_graph.parquet", graph_table
        )
        status = FULL_STATUS if full_scope else PILOT_STATUS
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA,
            "status": status,
            "scope": {
                "full_scope": full_scope,
                "component_pass": full_scope,
                "pilot_promotion_allowed": False,
                "s0_gate_pass": False,
                "d0_authorized": False,
                "model_fitting_authorized": False,
                "gpu_used": False,
            },
            "gate_parent": gate_parent,
            "core_binding": core_binding,
            "event_binding": event_binding,
            "preflight": preflight,
            "materialization_accounting": materialization,
            "profiles": {
                "caps": [10, 50, 100],
                "ranked_or_selected": False,
                "all_reported": True,
                "unbounded_all_edge_materialized": False,
            },
            "field_year_inventory": {
                "schema_version": FIELD_YEAR_SCHEMA,
                "rows": len(graph_table),
                "core_cells": len(core_keys),
                "profile_metric_sums": sums,
                "artifact": graph_artifact,
            },
            "gate_status": {
                "g_minus_1a": "PASS_BOUNDED",
                "s0_threshold_free_inventory": (
                    "PARTIAL_GRAPH_PASS_REQUIRES_FULL_ORCID_AFFILIATION_RECEIPT"
                    if full_scope
                    else "PARTIAL_CORE_PASS_GRAPH_PILOT_ONLY"
                ),
                "d0_discovery_pilot": "NOT_AUTHORIZED",
                "g0_measurement_freeze": "PENDING",
                "empirical_g1_signature": "NOT_RUN",
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "provenance": {
                "config_relative_path": str(config_path.relative_to(REPO_ROOT)),
                "config_sha256": _sha256_file(config_path),
                "runner_relative_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                "runner_sha256": _sha256_file(Path(__file__).resolve()),
                "duckdb_version": duckdb.__version__,
                "scipy_connected_components": True,
            },
            "execution": {
                "elapsed_seconds": time.monotonic() - started,
                "exact_intermediate_materialization_removed_before_completion": True,
            },
            "claim_boundary": config["claim_boundary"],
        }
        report["provenance"]["result_fingerprint_sha256"] = hashlib.sha256(
            _canonical_bytes(report)
        ).hexdigest()
        connection.close()
        shutil.rmtree(scratch)
        _write_json(temporary / "coauthor_graph_profile_report.json", report)
        markdown = "\n".join(
            [
                "# S0 exact coauthor graph profiles",
                "",
                f"Status: `{status}`",
                "",
                f"- core field-year cells: {len(core_keys):,}",
                f"- output profile rows: {len(graph_table):,}",
                f"- cap-10 distinct edges: {sums['distinct_undirected_edges']['10']:,}",
                f"- cap-50 distinct edges: {sums['distinct_undirected_edges']['50']:,}",
                f"- cap-100 distinct edges: {sums['distinct_undirected_edges']['100']:,}",
                "",
                "Profiles 10, 50, and 100 are all reported and are not ranked.",
                "The unbounded hyperauthorship profile is not edge-materialized.",
                "These are current-snapshot author-token graphs, not person graphs.",
                "This component alone does not pass S0 or authorize D0.",
                "",
            ]
        )
        md_path = temporary / "coauthor_graph_profile_report.md"
        md_path.write_text(markdown, encoding="utf-8")
        os.chmod(md_path, 0o600)
        _fsync_file(md_path)
        artifacts = [
            {
                "relative_path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(temporary.iterdir())
        ]
        receipt = {
            "schema_version": OUTPUT_SCHEMA,
            "status": status,
            "full_scope": full_scope,
            "component_pass": full_scope,
            "s0_gate_pass": False,
            "d0_authorized": False,
            "completion_contract": (
                "EXACT_INTERMEDIATES_REMOVED_RECEIPT_WRITTEN_LAST_"
                "ATOMIC_DIRECTORY_RENAME"
            ),
            "artifact_count": len(artifacts),
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_bytes(artifacts)
            ).hexdigest(),
            "artifacts": artifacts,
            "result_fingerprint_sha256": report["provenance"][
                "result_fingerprint_sha256"
            ],
        }
        _write_json(temporary / "coauthor_graph_profile_report.outputs.json", receipt)
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(temporary, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return report
    except BaseException:
        try:
            connection.close()
        except Exception:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "experiments/configs/pivot_feedback_s0_coauthor_graph_profile_v1.yaml",
    )
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-event-files", type=int)
    parser.add_argument("--hash-workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build(
            config_path=args.config,
            author_root=args.author_root,
            core_root=args.core_root,
            output=args.output,
            maximum_event_files=args.maximum_event_files,
            hash_workers=args.hash_workers,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    except (
        GraphProfileError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        DuckDBRuntimeError,
        pa.ArrowException,
    ) as error:
        print(json.dumps({"status": "BLOCKED_S0_COAUTHOR_GRAPH_PROFILE", "error": str(error)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
