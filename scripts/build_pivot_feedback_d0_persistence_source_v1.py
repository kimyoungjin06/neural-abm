#!/usr/bin/env python3
"""Build the exact 4,301-row D0 three-year persistence outcome source."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
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
    / "experiments/configs/pivot_feedback_d0_persistence_source_build_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_d0_persistence_source_build_config.v1"
REPORT_SCHEMA = "pivot_feedback_d0_persistence_outcome_source_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0_persistence_outcome_source_outputs.v1"
ARTIFACT_SCHEMA = "pivot_feedback_d0_persistence_outcome_source.v1"
STATUS = "PASS_EXACT_4301_NEW_TOPIC_ENTRY_THREE_YEAR_PERSISTENCE_SOURCE"
PARTITIONS = 4096
CLASSIFICATION_ROWS = 3_256_271
OUTPUT_ROWS = 4_301
KEY_COLUMNS = (
    "certified_episode_index",
    "field_id",
    "topic_id",
    "entry_year",
    "author_numeric_id",
)
OUTPUT_COLUMNS = (
    *KEY_COLUMNS,
    "observable_history_pivot_state",
    "persistence_followup_complete",
    "target_topic_work_count_t_plus_1",
    "target_topic_work_count_t_plus_2",
    "target_topic_work_count_t_plus_3",
    "target_topic_work_count_t_plus_1_through_t_plus_3",
    "three_year_target_topic_persistence_state",
    "three_year_target_topic_persistent",
)
WORK_SCHEMA = pa.schema(
    [
        pa.field("work_id_numeric", pa.uint64(), nullable=False),
        pa.field("publication_day", pa.int32(), nullable=True),
        pa.field("publication_year", pa.uint16(), nullable=True),
        pa.field("temporal_disposition", pa.uint8(), nullable=False),
        pa.field("work_type_code", pa.uint8(), nullable=False),
        pa.field("is_paratext", pa.bool_(), nullable=False),
        pa.field("is_retracted", pa.bool_(), nullable=True),
        pa.field("declared_authors_count", pa.uint16(), nullable=False),
        pa.field("primary_topic_id", pa.uint32(), nullable=True),
        pa.field("primary_field_id", pa.uint8(), nullable=True),
        pa.field("primary_domain_id", pa.uint8(), nullable=True),
    ]
)
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
CLASSIFICATION_SCHEMA = pa.schema(
    [
        pa.field("certified_episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("pre_entry_target_topic_work_count", pa.uint32(), nullable=False),
        pa.field("trailing_3y_target_topic_work_count", pa.uint32(), nullable=False),
        pa.field(
            "focal_confirmed_eligible_target_topic_work_count",
            pa.uint32(),
            nullable=False,
        ),
        pa.field("observable_history_pivot_state", pa.string(), nullable=False),
        pa.field("observable_history_new_topic_entry", pa.bool_(), nullable=False),
    ],
    metadata={
        b"schema_version": b"pivot_feedback_d0_production_pivot_classification.v1"
    },
)
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("certified_episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("observable_history_pivot_state", pa.string(), nullable=False),
        pa.field("persistence_followup_complete", pa.bool_(), nullable=False),
        pa.field("target_topic_work_count_t_plus_1", pa.uint32(), nullable=True),
        pa.field("target_topic_work_count_t_plus_2", pa.uint32(), nullable=True),
        pa.field("target_topic_work_count_t_plus_3", pa.uint32(), nullable=True),
        pa.field(
            "target_topic_work_count_t_plus_1_through_t_plus_3",
            pa.uint32(),
            nullable=True,
        ),
        pa.field(
            "three_year_target_topic_persistence_state",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "three_year_target_topic_persistent",
            pa.bool_(),
            nullable=True,
        ),
    ],
    metadata={b"schema_version": ARTIFACT_SCHEMA.encode()},
)
EXPECTED_FILES = {
    "PERSISTENCE_OUTCOME_SOURCE.md",
    "persistence_outcome_source.json",
    "persistence_outcome_source.outputs.json",
    "persistence_outcome_source.parquet",
}


class PersistenceSourceBuildError(RuntimeError):
    """Fail-closed production persistence-source build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PersistenceSourceBuildError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return list(value)


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
    ).encode()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PersistenceSourceBuildError(f"invalid {label}: {error}") from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise PersistenceSourceBuildError(f"invalid {label}: {error}") from error


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_text(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def _nested(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        require(
            isinstance(current, Mapping) and part in current,
            f"missing bound field {dotted}",
        )
        current = current[part]
    return current


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (REPO_ROOT / pure).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _artifact_items(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    raw = receipt.get("artifacts")
    for item_value in _list(raw, "receipt artifacts"):
        item = _mapping(item_value, "receipt artifact")
        relative = item.get("relative_path")
        require(isinstance(relative, str) and relative, "artifact path")
        require(relative not in result, "duplicate receipt artifact")
        result[relative] = item
    return result


def _stat_tuple(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _validate_repository(config: Mapping[str, Any]) -> dict[str, Any]:
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status_before_execution") == "AUTHORIZED_BY_GATE_STATE_V46",
        "v46 authorization",
    )
    result: dict[str, Any] = {}
    for label, raw in _mapping(
        config.get("repository_bindings"),
        "repository bindings",
    ).items():
        binding = _mapping(raw, f"repository binding {label}")
        path = _safe_repo_path(binding.get("relative_path"), label)
        require(_sha256(path) == binding.get("sha256"), f"{label} SHA-256")
        document = _read_yaml(path, label)
        require(
            document.get("schema_version") == binding.get("required_schema_version"),
            f"{label} schema",
        )
        if "required_status" in binding:
            require(
                document.get("status") == binding.get("required_status"),
                f"{label} status",
            )
        for dotted, expected in _mapping(
            binding.get("required_fields", {}),
            f"{label} fields",
        ).items():
            require(
                _nested(document, str(dotted)) == expected,
                f"{label} field {dotted}",
            )
        result[label] = {
            "relative_path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": binding["sha256"],
            "schema_version": document["schema_version"],
        }
    return result


def _validate_classification(
    *,
    operator_base: Path,
    binding_value: Any,
) -> tuple[Path, tuple[int, int, int, int, int], dict[str, Any]]:
    binding = _mapping(binding_value, "production classification")
    root = (operator_base / str(binding.get("root_basename"))).resolve(strict=True)
    require(root.parent == operator_base, "classification root parent")
    require(root.is_dir() and not root.is_symlink(), "classification root")
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, "classification root mode")
    receipt_path = root / str(binding.get("receipt_basename"))
    require(
        _sha256(receipt_path) == binding.get("receipt_sha256"),
        "classification receipt SHA",
    )
    receipt = _read_json(receipt_path, "classification receipt")
    require(
        receipt.get("schema_version") == binding.get("receipt_schema_version"),
        "classification receipt schema",
    )
    require(
        receipt.get("status") == binding.get("required_status"),
        "classification status",
    )
    require(
        receipt.get("artifact_tree_sha256") == binding.get("artifact_tree_sha256"),
        "classification artifact tree",
    )
    require(receipt.get("classification_rows") == CLASSIFICATION_ROWS, "classification rows")
    require(receipt.get("classification_columns") == 10, "classification columns")
    require(receipt.get("new_topic_entry_rows") == OUTPUT_ROWS, "primary rows")
    artifact_path = root / str(binding.get("artifact_basename"))
    require(_sha256(artifact_path) == binding.get("artifact_sha256"), "classification SHA")
    item = _artifact_items(receipt).get(artifact_path.name)
    require(item is not None, "classification artifact receipt")
    require(item.get("sha256") == binding.get("artifact_sha256"), "classification receipt SHA")
    parquet = pq.ParquetFile(artifact_path)
    require(parquet.metadata.num_rows == CLASSIFICATION_ROWS, "classification Parquet rows")
    require(
        parquet.schema_arrow.equals(CLASSIFICATION_SCHEMA, check_metadata=True),
        "classification schema",
    )
    return artifact_path, _stat_tuple(artifact_path), {
        "root_basename": root.name,
        "receipt_sha256": binding["receipt_sha256"],
        "artifact_sha256": binding["artifact_sha256"],
        "rows": CLASSIFICATION_ROWS,
        "columns": 10,
        "primary_state": "NEW_TOPIC_ENTRY",
        "primary_rows": OUTPUT_ROWS,
    }


def _validate_partitioned_input(
    *,
    operator_base: Path,
    binding_value: Any,
    label: str,
    schema: pa.Schema,
    workers: int,
) -> tuple[list[Path], list[tuple[int, int, int, int, int]], dict[str, Any]]:
    binding = _mapping(binding_value, label)
    root = (operator_base / str(binding.get("root_basename"))).resolve(strict=True)
    require(root.parent == operator_base, f"{label} root parent")
    require(root.is_dir() and not root.is_symlink(), f"{label} root")
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, f"{label} root mode")
    receipt_path = root / str(binding.get("receipt_basename"))
    require(_sha256(receipt_path) == binding.get("receipt_sha256"), f"{label} receipt SHA")
    receipt = _read_json(receipt_path, f"{label} receipt")
    require(
        receipt.get("schema_version") == binding.get("receipt_schema_version"),
        f"{label} receipt schema",
    )
    require(receipt.get("status") == binding.get("required_status"), f"{label} status")
    prefix = str(binding.get("artifact_prefix"))
    artifacts = [
        (relative, item)
        for relative, item in _artifact_items(receipt).items()
        if relative.startswith(prefix)
    ]
    artifacts.sort(key=lambda pair: pair[0])
    require(
        len(artifacts) == binding.get("artifact_count") == PARTITIONS,
        f"{label} artifact count",
    )
    paths: list[Path] = []
    stats: list[tuple[int, int, int, int, int]] = []
    rows = 0
    for index, (relative, item) in enumerate(artifacts):
        require(relative == f"{prefix}part-{index:04d}.parquet", f"{label} sequence")
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root), f"{label} partition escape")
        require(path.is_file() and not path.is_symlink(), f"{label} partition")
        require(path.stat().st_size == item.get("bytes"), f"{label} bytes")
        parquet = pq.ParquetFile(path)
        require(parquet.schema_arrow == schema, f"{label} schema")
        rows += parquet.metadata.num_rows
        paths.append(path)
        stats.append(_stat_tuple(path))
    require(rows == binding.get("rows"), f"{label} row closure")

    def rehash(
        pair: tuple[Path, tuple[str, Mapping[str, Any]]],
    ) -> tuple[str, int, str]:
        path, (relative, item) = pair
        observed = _sha256(path)
        require(observed == item.get("sha256"), f"{label} SHA {path.name}")
        return relative, path.stat().st_size, observed

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        verified = list(executor.map(rehash, zip(paths, artifacts, strict=True)))
    total_bytes = sum(item[1] for item in verified)
    require(total_bytes == binding.get("artifact_bytes"), f"{label} byte closure")
    return paths, stats, {
        "root_basename": root.name,
        "receipt_sha256": binding["receipt_sha256"],
        "artifacts_rehashed": len(verified),
        "artifact_bytes": total_bytes,
        "rows": rows,
        "selected_tree_sha256": hashlib.sha256(
            _canonical_bytes(verified)
        ).hexdigest(),
    }


def _validate_unchanged(
    paths: Sequence[Path],
    before: Sequence[tuple[int, int, int, int, int]],
    label: str,
) -> None:
    require(
        [_stat_tuple(path) for path in paths] == list(before),
        f"{label} changed during build",
    )


def _scalar(connection: duckdb.DuckDBPyConnection, query: str) -> int:
    value = connection.execute(query).fetchone()
    require(value is not None, "scalar query")
    return int(value[0])


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _episode_values(rows: Sequence[Sequence[int]]) -> str:
    return ", ".join(
        f"({int(index)}, {int(field)}, {int(topic)}, {int(year)}, {int(last)})"
        for index, field, topic, year, last in rows
    )


def _write_output(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    path: Path,
) -> None:
    reader = connection.execute(query).arrow()
    table = reader.read_all()
    require(table.num_rows == OUTPUT_ROWS, "output row count before write")
    table = table.cast(OUTPUT_SCHEMA).replace_schema_metadata(OUTPUT_SCHEMA.metadata)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    os.chmod(path, 0o600)
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == OUTPUT_ROWS, "output Parquet rows")
    require(
        parquet.schema_arrow.equals(OUTPUT_SCHEMA, check_metadata=True),
        "output Parquet schema",
    )


def _markdown(report: Mapping[str, Any]) -> str:
    totals = _mapping(report["outcome_accounting"]["totals"], "totals")
    return (
        "# D0 production persistence outcome source v1\n\n"
        f"Status: `{report['status']}`\n\n"
        f"- denominator rows: {totals['rows']:,}\n"
        f"- matured persistence: {totals['matured_persistence_rows']:,}\n"
        f"- matured no persistence: {totals['matured_no_persistence_rows']:,}\n"
        f"- right censored: {totals['right_censored_rows']:,}\n"
        f"- distinct qualifying author-work pairs: "
        f"{totals['qualifying_author_work_pairs']:,}\n\n"
        "These are current-snapshot-annotated source counts for the exact "
        "observable-history NEW_TOPIC_ENTRY denominator. They are not an "
        "association, causal effect, or true researcher-topic persistence "
        "claim. Independent replay is required before interpretation.\n"
    )


def build(
    *,
    config_path: Path,
    operator_base: Path,
    output: Path,
    workers: int,
    threads: int,
) -> dict[str, Any]:
    started = time.monotonic()
    config_path = config_path.resolve(strict=True)
    operator_base = operator_base.resolve(strict=True)
    output = output.resolve()
    require(operator_base.is_absolute() and operator_base.is_dir(), "operator base")
    require(output.parent == operator_base, "output must be operator-base child")
    require(not output.exists(), "output already exists")
    config = _read_yaml(config_path, "config")
    repository = _validate_repository(config)
    output_contract = _mapping(config.get("output_contract"), "output contract")
    require(output.name == output_contract.get("root_basename"), "output basename")
    require(
        tuple(output_contract.get("required_columns", ())) == OUTPUT_COLUMNS,
        "output columns",
    )
    source_contract = _mapping(config.get("source_contract"), "source contract")
    require(source_contract.get("exact_output_rows") == OUTPUT_ROWS, "output rows")
    require(source_contract.get("permitted_post_entry_offsets") == [1, 2, 3], "offsets")
    require(source_contract.get("team_cap") == "NONE", "team cap")
    forbidden = _mapping(
        config.get("forbidden_inputs_and_outputs"),
        "forbidden boundary",
    )
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden boundary")

    inputs = _mapping(config.get("operator_inputs"), "operator inputs")
    classification_path, classification_stat, classification_binding = (
        _validate_classification(
            operator_base=operator_base,
            binding_value=inputs.get("production_classification"),
        )
    )
    work_paths, work_stats, work_binding = _validate_partitioned_input(
        operator_base=operator_base,
        binding_value=inputs.get("cutoff_work_metadata"),
        label="cutoff work metadata",
        schema=WORK_SCHEMA,
        workers=workers,
    )
    event_paths, event_stats, event_binding = _validate_partitioned_input(
        operator_base=operator_base,
        binding_value=inputs.get("author_work_events"),
        label="author work events",
        schema=EVENT_SCHEMA,
        workers=workers,
    )
    scope = _mapping(config.get("exact_episode_scope"), "episode scope")
    episode_rows = _list(scope.get("rows"), "episode rows")
    require(len(episode_rows) == 4 and scope.get("all_matured") is True, "episode maturity")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=operator_base)
    )
    os.chmod(temporary, 0o700)
    scratch = temporary / ".duckdb-tmp"
    scratch.mkdir(mode=0o700)
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(database=":memory:")
        connection.execute(f"PRAGMA threads={max(1, threads)}")
        connection.execute(f"PRAGMA temp_directory='{_sql_path(scratch)}'")
        connection.execute("PRAGMA preserve_insertion_order=false")
        connection.execute(
            """
            CREATE TEMP TABLE episodes(
              certified_episode_index UTINYINT,
              field_id UTINYINT,
              topic_id UINTEGER,
              entry_year USMALLINT,
              last_outcome_year USMALLINT
            )
            """
        )
        connection.execute(f"INSERT INTO episodes VALUES {_episode_values(episode_rows)}")
        classification_sql = _sql_path(classification_path)
        work_glob = _sql_path(work_paths[0].parent / "*.parquet")
        event_glob = _sql_path(event_paths[0].parent / "*.parquet")
        connection.execute(
            f"""
            CREATE TEMP TABLE primary_entries AS
            SELECT
              certified_episode_index,
              field_id,
              topic_id,
              entry_year,
              author_numeric_id,
              observable_history_pivot_state
            FROM read_parquet('{classification_sql}')
            WHERE observable_history_pivot_state = 'NEW_TOPIC_ENTRY'
              AND observable_history_new_topic_entry
            """
        )
        require(
            _scalar(connection, "SELECT count(*) FROM primary_entries") == OUTPUT_ROWS,
            "primary denominator rows",
        )
        require(
            _scalar(
                connection,
                """
                SELECT count(*) - count(
                  DISTINCT row(
                    certified_episode_index,
                    field_id,
                    topic_id,
                    entry_year,
                    author_numeric_id
                  )
                ) FROM primary_entries
                """,
            )
            == 0,
            "primary denominator duplicate key",
        )
        require(
            _scalar(
                connection,
                """
                SELECT count(*) FROM primary_entries AS p
                ANTI JOIN episodes AS e USING (
                  certified_episode_index, field_id, topic_id, entry_year
                )
                """,
            )
            == 0,
            "primary denominator episode scope",
        )

        work_accounting_row = connection.execute(
            f"""
            WITH candidates AS (
              SELECT
                e.certified_episode_index,
                w.*,
                CASE
                  WHEN w.primary_topic_id IS NULL
                    OR w.primary_field_id IS NULL
                    OR w.primary_topic_id != e.topic_id
                    OR w.primary_field_id != e.field_id
                    THEN 'wrong_topic_or_field'
                  WHEN w.temporal_disposition != 2
                    THEN 'temporal_disposition'
                  WHEN w.work_type_code NOT IN (1, 2, 3, 4, 7, 16, 18, 21)
                    OR w.is_paratext
                    THEN 'paratext_or_work_type'
                  WHEN w.is_retracted IS TRUE
                    THEN 'retracted'
                  ELSE 'eligible'
                END AS disposition
              FROM read_parquet('{work_glob}') AS w
              INNER JOIN episodes AS e
                ON w.publication_year
                  BETWEEN e.entry_year + 1 AND e.entry_year + 3
            )
            SELECT
              count(*) AS candidate_window_rows,
              count(*) FILTER (
                WHERE disposition = 'wrong_topic_or_field'
              ) AS excluded_wrong_topic_or_field_rows,
              count(*) FILTER (
                WHERE disposition = 'temporal_disposition'
              ) AS excluded_temporal_disposition_rows,
              count(*) FILTER (
                WHERE disposition = 'paratext_or_work_type'
              ) AS excluded_paratext_or_work_type_rows,
              count(*) FILTER (
                WHERE disposition = 'retracted'
              ) AS excluded_retracted_rows,
              count(*) FILTER (
                WHERE disposition = 'eligible'
              ) AS eligible_prelink_work_rows,
              count(*) FILTER (
                WHERE disposition = 'eligible' AND is_retracted IS NULL
              ) AS retained_null_retraction_rows
            FROM candidates
            """
        ).fetchone()
        require(work_accounting_row is not None, "work accounting")
        work_names = (
            "candidate_window_rows",
            "excluded_wrong_topic_or_field_rows",
            "excluded_temporal_disposition_rows",
            "excluded_paratext_or_work_type_rows",
            "excluded_retracted_rows",
            "eligible_prelink_work_rows",
            "retained_null_retraction_rows",
        )
        work_accounting = dict(
            zip(work_names, map(int, work_accounting_row), strict=True)
        )
        work_accounting["excluded_missing_publication_year_target_rows"] = _scalar(
            connection,
            f"""
            SELECT count(*)
            FROM read_parquet('{work_glob}') AS w
            INNER JOIN episodes AS e
              ON w.primary_topic_id = e.topic_id
             AND w.primary_field_id = e.field_id
            WHERE w.publication_year IS NULL
            """,
        )
        require(
            sum(
                work_accounting[key]
                for key in (
                    "excluded_wrong_topic_or_field_rows",
                    "excluded_temporal_disposition_rows",
                    "excluded_paratext_or_work_type_rows",
                    "excluded_retracted_rows",
                    "eligible_prelink_work_rows",
                )
            )
            == work_accounting["candidate_window_rows"],
            "work accounting closure",
        )

        connection.execute(
            f"""
            CREATE TEMP TABLE eligible_work AS
            SELECT
              e.certified_episode_index,
              e.field_id,
              e.topic_id,
              e.entry_year,
              w.work_id_numeric,
              w.publication_year,
              w.temporal_disposition,
              w.declared_authors_count,
              CAST(w.publication_year - e.entry_year AS UTINYINT) AS outcome_offset
            FROM read_parquet('{work_glob}') AS w
            INNER JOIN episodes AS e
              ON w.publication_year
                BETWEEN e.entry_year + 1 AND e.entry_year + 3
             AND w.primary_topic_id = e.topic_id
             AND w.primary_field_id = e.field_id
            WHERE w.temporal_disposition = 2
              AND w.work_type_code IN (1, 2, 3, 4, 7, 16, 18, 21)
              AND NOT w.is_paratext
              AND w.is_retracted IS DISTINCT FROM TRUE
            """
        )
        require(
            _scalar(connection, "SELECT count(*) FROM eligible_work")
            == work_accounting["eligible_prelink_work_rows"],
            "eligible work closure",
        )
        require(
            _scalar(
                connection,
                "SELECT count(*) - count(DISTINCT work_id_numeric) FROM eligible_work",
            )
            == 0,
            "eligible work duplicate",
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE joined_event AS
            SELECT
              ew.certified_episode_index,
              ew.work_id_numeric,
              ew.publication_year,
              ew.temporal_disposition,
              ew.declared_authors_count,
              ew.field_id,
              ew.outcome_offset,
              ev.author_numeric_id,
              ev.publication_year = ew.publication_year AS publication_year_match,
              ev.temporal_disposition = ew.temporal_disposition
                AS temporal_disposition_match,
              ev.declared_authors_count = ew.declared_authors_count
                AS declared_authors_count_match,
              ev.primary_field_id = ew.field_id AS primary_field_id_match
            FROM read_parquet('{event_glob}') AS ev
            INNER JOIN eligible_work AS ew
              ON ev.work_numeric_id = ew.work_id_numeric
            """
        )
        joined_event_rows = _scalar(connection, "SELECT count(*) FROM joined_event")
        mismatch = {
            "publication_year_mismatch_rows": _scalar(
                connection,
                "SELECT count(*) FROM joined_event WHERE NOT publication_year_match",
            ),
            "temporal_disposition_mismatch_rows": _scalar(
                connection,
                "SELECT count(*) FROM joined_event WHERE NOT temporal_disposition_match",
            ),
            "declared_authors_count_mismatch_rows": _scalar(
                connection,
                "SELECT count(*) FROM joined_event WHERE NOT declared_authors_count_match",
            ),
            "primary_field_id_mismatch_rows": _scalar(
                connection,
                "SELECT count(*) FROM joined_event WHERE NOT primary_field_id_match",
            ),
        }
        require(all(value == 0 for value in mismatch.values()), "event metadata mismatch")
        connection.execute(
            """
            CREATE TEMP TABLE valid_pairs AS
            SELECT DISTINCT
              certified_episode_index,
              author_numeric_id,
              work_id_numeric,
              outcome_offset
            FROM joined_event
            WHERE publication_year_match
              AND temporal_disposition_match
              AND declared_authors_count_match
              AND primary_field_id_match
            """
        )
        distinct_pairs = _scalar(connection, "SELECT count(*) FROM valid_pairs")
        linked_work_rows = _scalar(
            connection,
            """
            SELECT count(DISTINCT row(certified_episode_index, work_id_numeric))
            FROM valid_pairs
            """,
        )
        zero_token_work_rows = _scalar(
            connection,
            """
            SELECT count(*) FROM eligible_work AS ew
            ANTI JOIN (
              SELECT DISTINCT certified_episode_index, work_id_numeric
              FROM valid_pairs
            ) AS vp USING (certified_episode_index, work_id_numeric)
            """,
        )
        require(
            linked_work_rows + zero_token_work_rows
            == work_accounting["eligible_prelink_work_rows"],
            "eligible work linkage closure",
        )
        connection.execute(
            """
            CREATE TEMP TABLE primary_pairs AS
            SELECT vp.*
            FROM valid_pairs AS vp
            INNER JOIN primary_entries AS p
              USING (certified_episode_index, author_numeric_id)
            """
        )
        primary_pairs = _scalar(connection, "SELECT count(*) FROM primary_pairs")
        other_pairs = _scalar(
            connection,
            """
            SELECT count(*) FROM valid_pairs AS vp
            ANTI JOIN primary_entries AS p
              USING (certified_episode_index, author_numeric_id)
            """,
        )
        require(primary_pairs + other_pairs == distinct_pairs, "pair denominator closure")
        connection.execute(
            """
            CREATE TEMP TABLE author_counts AS
            SELECT
              certified_episode_index,
              author_numeric_id,
              CAST(count(*) FILTER (WHERE outcome_offset = 1) AS UINTEGER)
                AS count_t_plus_1,
              CAST(count(*) FILTER (WHERE outcome_offset = 2) AS UINTEGER)
                AS count_t_plus_2,
              CAST(count(*) FILTER (WHERE outcome_offset = 3) AS UINTEGER)
                AS count_t_plus_3,
              CAST(count(*) AS UINTEGER) AS count_total
            FROM primary_pairs
            GROUP BY ALL
            """
        )
        output_query = """
            SELECT
              p.certified_episode_index,
              p.field_id,
              p.topic_id,
              p.entry_year,
              p.author_numeric_id,
              p.observable_history_pivot_state,
              TRUE AS persistence_followup_complete,
              CAST(coalesce(c.count_t_plus_1, 0) AS UINTEGER)
                AS target_topic_work_count_t_plus_1,
              CAST(coalesce(c.count_t_plus_2, 0) AS UINTEGER)
                AS target_topic_work_count_t_plus_2,
              CAST(coalesce(c.count_t_plus_3, 0) AS UINTEGER)
                AS target_topic_work_count_t_plus_3,
              CAST(coalesce(c.count_total, 0) AS UINTEGER)
                AS target_topic_work_count_t_plus_1_through_t_plus_3,
              CASE
                WHEN coalesce(c.count_total, 0) > 0
                  THEN 'MATURED_PERSISTENCE'
                ELSE 'MATURED_NO_PERSISTENCE'
              END AS three_year_target_topic_persistence_state,
              CAST(coalesce(c.count_total, 0) > 0 AS BOOLEAN)
                AS three_year_target_topic_persistent
            FROM primary_entries AS p
            LEFT JOIN author_counts AS c
              USING (certified_episode_index, author_numeric_id)
            ORDER BY
              p.certified_episode_index,
              p.field_id,
              p.topic_id,
              p.entry_year,
              p.author_numeric_id
        """
        artifact_path = temporary / str(output_contract["artifact_basename"])
        _write_output(connection, output_query, artifact_path)
        artifact_sql = _sql_path(artifact_path)
        connection.execute(
            f"CREATE TEMP VIEW output_source AS SELECT * FROM read_parquet('{artifact_sql}')"
        )
        invariant = {
            "output_rows": _scalar(connection, "SELECT count(*) FROM output_source"),
            "duplicate_key_rows": _scalar(
                connection,
                """
                SELECT count(*) - count(
                  DISTINCT row(
                    certified_episode_index,
                    field_id,
                    topic_id,
                    entry_year,
                    author_numeric_id
                  )
                ) FROM output_source
                """,
            ),
            "non_new_topic_entry_rows": _scalar(
                connection,
                """
                SELECT count(*) FROM output_source
                WHERE observable_history_pivot_state != 'NEW_TOPIC_ENTRY'
                """,
            ),
            "incomplete_followup_rows": _scalar(
                connection,
                "SELECT count(*) FROM output_source WHERE NOT persistence_followup_complete",
            ),
            "null_count_or_boolean_rows": _scalar(
                connection,
                """
                SELECT count(*) FROM output_source
                WHERE target_topic_work_count_t_plus_1 IS NULL
                   OR target_topic_work_count_t_plus_2 IS NULL
                   OR target_topic_work_count_t_plus_3 IS NULL
                   OR target_topic_work_count_t_plus_1_through_t_plus_3 IS NULL
                   OR three_year_target_topic_persistent IS NULL
                """,
            ),
            "count_sum_mismatch_rows": _scalar(
                connection,
                """
                SELECT count(*) FROM output_source
                WHERE target_topic_work_count_t_plus_1
                    + target_topic_work_count_t_plus_2
                    + target_topic_work_count_t_plus_3
                  != target_topic_work_count_t_plus_1_through_t_plus_3
                """,
            ),
            "state_boolean_mismatch_rows": _scalar(
                connection,
                """
                SELECT count(*) FROM output_source
                WHERE three_year_target_topic_persistent
                    != (
                      three_year_target_topic_persistence_state
                        = 'MATURED_PERSISTENCE'
                    )
                   OR (
                     target_topic_work_count_t_plus_1_through_t_plus_3 > 0
                   ) != three_year_target_topic_persistent
                """,
            ),
        }
        require(invariant["output_rows"] == OUTPUT_ROWS, "output row closure")
        require(
            all(
                value == 0
                for key, value in invariant.items()
                if key != "output_rows"
            ),
            "output invariant",
        )
        by_episode_raw = connection.execute(
            """
            SELECT
              certified_episode_index,
              field_id,
              topic_id,
              entry_year,
              count(*) AS rows,
              sum(target_topic_work_count_t_plus_1) AS pair_rows_t_plus_1,
              sum(target_topic_work_count_t_plus_2) AS pair_rows_t_plus_2,
              sum(target_topic_work_count_t_plus_3) AS pair_rows_t_plus_3,
              sum(target_topic_work_count_t_plus_1_through_t_plus_3)
                AS qualifying_author_work_pairs,
              count(*) FILTER (
                WHERE three_year_target_topic_persistence_state
                  = 'MATURED_PERSISTENCE'
              ) AS matured_persistence_rows,
              count(*) FILTER (
                WHERE three_year_target_topic_persistence_state
                  = 'MATURED_NO_PERSISTENCE'
              ) AS matured_no_persistence_rows,
              count(*) FILTER (
                WHERE three_year_target_topic_persistence_state
                  = 'RIGHT_CENSORED'
              ) AS right_censored_rows
            FROM output_source
            GROUP BY ALL
            ORDER BY certified_episode_index
            """
        ).fetchall()
        episode_names = (
            "certified_episode_index",
            "field_id",
            "topic_id",
            "entry_year",
            "rows",
            "pair_rows_t_plus_1",
            "pair_rows_t_plus_2",
            "pair_rows_t_plus_3",
            "qualifying_author_work_pairs",
            "matured_persistence_rows",
            "matured_no_persistence_rows",
            "right_censored_rows",
        )
        by_episode = [
            dict(zip(episode_names, map(int, row), strict=True))
            for row in by_episode_raw
        ]
        totals = {
            key: sum(row[key] for row in by_episode)
            for key in (
                "rows",
                "pair_rows_t_plus_1",
                "pair_rows_t_plus_2",
                "pair_rows_t_plus_3",
                "qualifying_author_work_pairs",
                "matured_persistence_rows",
                "matured_no_persistence_rows",
                "right_censored_rows",
            )
        }
        require(totals["rows"] == OUTPUT_ROWS, "episode output row closure")
        require(
            totals["qualifying_author_work_pairs"] == primary_pairs,
            "primary pair output closure",
        )
        require(
            totals["matured_persistence_rows"]
            + totals["matured_no_persistence_rows"]
            == OUTPUT_ROWS,
            "outcome state closure",
        )
        _validate_unchanged(
            [classification_path],
            [classification_stat],
            "classification input",
        )
        _validate_unchanged(work_paths, work_stats, "work input")
        _validate_unchanged(event_paths, event_stats, "event input")

        event_accounting = {
            "source_event_rows": event_binding["rows"],
            "eligible_work_joined_event_rows": joined_event_rows,
            **mismatch,
            "valid_event_rows": joined_event_rows,
            "distinct_author_work_pairs": distinct_pairs,
            "duplicate_author_work_event_excess_rows": joined_event_rows - distinct_pairs,
            "linked_work_rows": linked_work_rows,
            "zero_valid_author_token_work_rows": zero_token_work_rows,
            "registered_primary_distinct_author_work_pairs": primary_pairs,
            "unregistered_or_nonprimary_distinct_author_work_pairs": other_pairs,
        }
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": STATUS,
            "mode": config["mode"],
            "repository_bindings": repository,
            "source_bindings": {
                "production_classification": classification_binding,
                "cutoff_work_metadata": work_binding,
                "author_work_events": event_binding,
            },
            "scope": {
                "exact_certified_episode_rows": 4,
                "source_classification_rows": CLASSIFICATION_ROWS,
                "primary_pivot_state": "NEW_TOPIC_ENTRY",
                "primary_pivot_rows": OUTPUT_ROWS,
                "post_entry_offsets_read": [1, 2, 3],
                "confirmatory_entry_year_rows_read": 0,
                "impact_rows_read": 0,
                "citation_rows_read": 0,
                "peer_rows_read": 0,
                "api_requests_created": 0,
                "association_or_effect_computed": False,
                "model_fitted": False,
                "g4_open": False,
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "work_accounting": {
                "source_work_rows": work_binding["rows"],
                **work_accounting,
            },
            "event_accounting": event_accounting,
            "outcome_accounting": {
                "by_episode": by_episode,
                "totals": totals,
                "invariants": invariant,
            },
            "execution": {
                "workload_class": "CPU_IO",
                "gpu_used": False,
                "workers": max(1, workers),
                "duckdb_threads": max(1, threads),
                "elapsed_seconds": round(time.monotonic() - started, 5),
                "builder_relative_path": (
                    Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
                ),
                "builder_sha256": _sha256(Path(__file__).resolve()),
            },
            "authorization": {
                "independent_all_row_replay": "REQUIRED_NEXT",
                "discovery_outcome_analysis_or_interpretation": "NOT_AUTHORIZED",
                "final_candidate_eligibility": "NOT_AUTHORIZED",
                "g0_measurement_freeze": "NOT_AUTHORIZED",
                "model_fitting": "NOT_AUTHORIZED",
                "g4_open": "NOT_AUTHORIZED",
            },
            "claim_boundary": config["claim_boundary"],
        }
        report_path = temporary / str(output_contract["report_basename"])
        _write_json(report_path, report)
        markdown_path = temporary / str(output_contract["markdown_basename"])
        _write_text(markdown_path, _markdown(report))
        connection.close()
        connection = None
        shutil.rmtree(scratch)

        artifacts: list[dict[str, Any]] = []
        for path in (artifact_path, report_path, markdown_path):
            os.chmod(path, 0o600)
            item: dict[str, Any] = {
                "relative_path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            if path.suffix == ".parquet":
                item["rows"] = pq.ParquetFile(path).metadata.num_rows
            artifacts.append(item)
        artifacts.sort(key=lambda item: str(item["relative_path"]))
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": STATUS,
            "root_basename": output.name,
            "config_sha256": _sha256(config_path),
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
            "source_classification_rows": CLASSIFICATION_ROWS,
            "primary_pivot_rows": OUTPUT_ROWS,
            "persistence_source_rows": OUTPUT_ROWS,
            "persistence_source_columns": len(OUTPUT_COLUMNS),
            "post_entry_offsets_read": [1, 2, 3],
            "qualifying_author_work_pairs": totals[
                "qualifying_author_work_pairs"
            ],
            "matured_persistence_rows": totals["matured_persistence_rows"],
            "matured_no_persistence_rows": totals[
                "matured_no_persistence_rows"
            ],
            "right_censored_rows": totals["right_censored_rows"],
            "denominator_rows_filtered": 0,
            "denominator_rows_mutated": 0,
            "confirmatory_entry_year_rows_read": 0,
            "impact_rows_read": 0,
            "citation_rows_read": 0,
            "peer_rows_read": 0,
            "api_requests_created": 0,
            "association_or_effect_computed": False,
            "independent_verification_complete": False,
            "g0_measurement_freeze_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "receipt_written_last": True,
        }
        receipt_path = temporary / str(output_contract["receipt_basename"])
        _write_json(receipt_path, receipt)
        require(
            {path.name for path in temporary.iterdir()} == EXPECTED_FILES,
            "unexpected output file set",
        )
        os.replace(temporary, output)
        return report
    finally:
        if connection is not None:
            connection.close()
        if temporary.exists():
            require(
                temporary.parent == operator_base
                and temporary.name.startswith(f".{output.name}.partial-"),
                "unsafe temporary cleanup target",
            )
            shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--operator-base", type=Path, default=Path("/ssd"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build(
        config_path=args.config,
        operator_base=args.operator_base,
        output=args.output,
        workers=args.workers,
        threads=args.threads,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
