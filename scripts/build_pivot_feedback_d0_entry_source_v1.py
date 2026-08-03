#!/usr/bin/env python3
"""Build the exact-four three-count pivot feature source without state labels."""

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
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_entry_source_build_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_d0_entry_source_build_config.v1"
REPORT_SCHEMA = "pivot_feedback_d0_entry_source_build_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0_entry_source_build_outputs.v1"
FEATURE_SCHEMA_VERSION = "pivot_feedback_d0_entry_source_features.v1"
STATUS = "PASS_EXACT_FOUR_THREE_COUNT_PRODUCTION_FEATURE_SOURCE_NO_STATE_LABELS"
PARTITIONS = 4096
ROWS = 3_256_271
KEY_COLUMNS = (
    "certified_episode_index",
    "field_id",
    "topic_id",
    "entry_year",
    "author_numeric_id",
)
FEATURE_COLUMNS = (
    *KEY_COLUMNS,
    "pre_entry_target_topic_work_count",
    "trailing_3y_target_topic_work_count",
    "focal_confirmed_eligible_target_topic_work_count",
)
EXPECTED_FILES = {
    "PIVOT_ENTRY_SOURCE_BUILD.md",
    "pivot_entry_source_build.json",
    "pivot_entry_source_build.outputs.json",
    "pivot_entry_source_features.parquet",
}
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
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("certified_episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field(
            "pre_entry_target_topic_work_count",
            pa.uint32(),
            nullable=False,
        ),
        pa.field(
            "trailing_3y_target_topic_work_count",
            pa.uint32(),
            nullable=False,
        ),
        pa.field(
            "focal_confirmed_eligible_target_topic_work_count",
            pa.uint32(),
            nullable=False,
        ),
    ],
    metadata={b"schema_version": FEATURE_SCHEMA_VERSION.encode()},
)


class EntrySourceBuildError(RuntimeError):
    """Fail-closed production entry-source build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EntrySourceBuildError(message)


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
        raise EntrySourceBuildError(f"invalid {label}: {error}") from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise EntrySourceBuildError(f"invalid {label}: {error}") from error


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
    raw = receipt.get("artifacts")
    result: dict[str, dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        for name, item_value in raw.items():
            item = _mapping(item_value, f"artifact {name}")
            relative = item.get("relative_path") or item.get("basename") or name
            require(isinstance(relative, str) and relative, "artifact path")
            result[relative] = item
    else:
        for item_value in _list(raw, "receipt artifacts"):
            item = _mapping(item_value, "receipt artifact")
            relative = item.get("relative_path")
            require(isinstance(relative, str) and relative, "artifact path")
            require(relative not in result, "duplicate receipt artifact")
            result[relative] = item
    return result


def _artifact_metadata(
    receipt: Mapping[str, Any],
    basename: str,
) -> dict[str, Any]:
    items = _artifact_items(receipt)
    if basename in items:
        return items[basename]
    for relative, item in items.items():
        if Path(relative).name == basename:
            return item
    raise EntrySourceBuildError(f"artifact absent from receipt: {basename}")


def _validate_repository(config: Mapping[str, Any]) -> dict[str, Any]:
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status_before_execution") == "AUTHORIZED_BY_GATE_STATE_V43",
        "v43 authorization",
    )
    result: dict[str, Any] = {}
    bindings = _mapping(config.get("repository_bindings"), "repository bindings")
    for label, raw in bindings.items():
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
            f"{label} required fields",
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


def _stat_tuple(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


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
        expected = f"{prefix}part-{index:04d}.parquet"
        require(relative == expected, f"{label} partition sequence")
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root), f"{label} partition escape")
        require(path.is_file() and not path.is_symlink(), f"{label} partition")
        require(path.stat().st_size == item.get("bytes"), f"{label} partition bytes")
        parquet = pq.ParquetFile(path)
        require(parquet.schema_arrow == schema, f"{label} partition schema")
        rows += parquet.metadata.num_rows
        paths.append(path)
        stats.append(_stat_tuple(path))
    require(rows == binding.get("rows"), f"{label} row closure")

    def rehash(pair: tuple[Path, tuple[str, Mapping[str, Any]]]) -> tuple[str, int, str]:
        path, (relative, item) = pair
        observed = _sha256(path)
        require(observed == item.get("sha256"), f"{label} SHA {path.name}")
        return relative, path.stat().st_size, observed

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        verified = list(
            executor.map(rehash, zip(paths, artifacts, strict=True))
        )
    total_bytes = sum(item[1] for item in verified)
    require(total_bytes == binding.get("artifact_bytes"), f"{label} byte closure")
    return paths, stats, {
        "root_basename": root.name,
        "receipt_basename": receipt_path.name,
        "receipt_sha256": binding["receipt_sha256"],
        "artifact_prefix": prefix,
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


def _validate_small_inputs(
    *,
    operator_base: Path,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    selection_spec = _mapping(
        inputs.get("certified_episode_selection"),
        "certified episode selection",
    )
    selection_root = (
        operator_base / str(selection_spec.get("root_basename"))
    ).resolve(strict=True)
    require(
        selection_root.parent == operator_base
        and selection_root.is_dir()
        and not selection_root.is_symlink(),
        "selection root",
    )
    selection_receipt_path = selection_root / str(
        selection_spec.get("receipt_basename")
    )
    require(
        _sha256(selection_receipt_path) == selection_spec.get("receipt_sha256"),
        "selection receipt SHA",
    )
    selection_receipt = _read_json(selection_receipt_path, "selection receipt")
    require(
        selection_receipt.get("schema_version")
        == selection_spec.get("receipt_schema_version"),
        "selection receipt schema",
    )
    require(
        selection_receipt.get("status") == selection_spec.get("required_status"),
        "selection receipt status",
    )
    selection_path = selection_root / str(selection_spec.get("artifact_basename"))
    require(
        _sha256(selection_path) == selection_spec.get("artifact_sha256"),
        "selection artifact SHA",
    )
    require(
        _artifact_metadata(selection_receipt, selection_path.name).get("sha256")
        == selection_spec.get("artifact_sha256"),
        "selection receipt artifact SHA",
    )
    selection = pq.read_table(selection_path)
    require(selection.num_rows == selection_spec.get("rows") == 4, "selection rows")
    metadata = selection.schema.metadata or {}
    require(
        metadata.get(b"schema_version", b"").decode()
        == selection_spec.get("artifact_schema_version"),
        "selection schema",
    )
    rows = [
        [
            int(row["certified_episode_index"]),
            int(row["field_id"]),
            int(row["topic_id"]),
            int(row["entry_year"]),
        ]
        for row in selection.to_pylist()
    ]
    require(rows == selection_spec.get("exact_rows"), "exact selection rows")
    require(
        all(
            value.as_py() == selection_spec.get("required_content_state")
            for value in selection["content_state"]
        ),
        "selection content state",
    )

    certificate_spec = _mapping(
        inputs.get("content_selection_certificate"),
        "content certificate",
    )
    certificate_root = (
        operator_base / str(certificate_spec.get("root_basename"))
    ).resolve(strict=True)
    certificate_receipt_path = certificate_root / str(
        certificate_spec.get("receipt_basename")
    )
    require(
        _sha256(certificate_receipt_path)
        == certificate_spec.get("receipt_sha256"),
        "content certificate receipt SHA",
    )
    certificate_receipt = _read_json(
        certificate_receipt_path,
        "content certificate receipt",
    )
    require(
        certificate_receipt.get("schema_version")
        == certificate_spec.get("receipt_schema_version"),
        "content certificate receipt schema",
    )
    require(
        certificate_receipt.get("status") == certificate_spec.get("required_status"),
        "content certificate receipt status",
    )
    certificate_path = certificate_root / str(
        certificate_spec.get("artifact_basename")
    )
    require(
        _sha256(certificate_path) == certificate_spec.get("artifact_sha256"),
        "content certificate SHA",
    )
    require(
        _artifact_metadata(certificate_receipt, certificate_path.name).get("sha256")
        == certificate_spec.get("artifact_sha256"),
        "content certificate receipt artifact SHA",
    )
    certificate = _read_json(certificate_path, "content certificate")
    require(
        certificate.get("schema_version")
        == certificate_spec.get("artifact_schema_version"),
        "content certificate schema",
    )
    require(
        certificate.get("status") == certificate_spec.get("artifact_status"),
        "content certificate status",
    )
    require(certificate.get("global_selection_certificate_issued") is True, "global certificate")
    require(certificate.get("provider_labels_are_ground_truth") is False, "provider boundary")
    expected_topics = _mapping(
        certificate_spec.get("exact_topic_ids"),
        "certificate topics",
    )
    require(
        certificate.get("computer_science_selected_topic_ids")
        == expected_topics["computer_science"],
        "computer-science certificate topics",
    )
    require(
        certificate.get("medicine_selected_topic_ids")
        == expected_topics["medicine"],
        "medicine certificate topics",
    )

    pre_spec = _mapping(
        inputs.get("predecision_author_measurement"),
        "predecision measurement",
    )
    pre_root = (operator_base / str(pre_spec.get("root_basename"))).resolve(
        strict=True
    )
    pre_receipt_path = pre_root / str(pre_spec.get("receipt_basename"))
    require(
        _sha256(pre_receipt_path) == pre_spec.get("receipt_sha256"),
        "predecision receipt SHA",
    )
    pre_receipt = _read_json(pre_receipt_path, "predecision receipt")
    require(
        pre_receipt.get("schema_version") == pre_spec.get("receipt_schema_version"),
        "predecision receipt schema",
    )
    require(
        pre_receipt.get("status") == pre_spec.get("required_status"),
        "predecision receipt status",
    )
    pre_path = pre_root / str(pre_spec.get("artifact_basename"))
    require(_sha256(pre_path) == pre_spec.get("artifact_sha256"), "predecision SHA")
    require(
        _artifact_metadata(pre_receipt, pre_path.name).get("sha256")
        == pre_spec.get("artifact_sha256"),
        "predecision receipt artifact SHA",
    )
    pre_parquet = pq.ParquetFile(pre_path)
    require(pre_parquet.metadata.num_rows == pre_spec.get("rows") == ROWS, "predecision rows")
    pre_metadata = pre_parquet.schema_arrow.metadata or {}
    require(
        pre_metadata.get(b"schema_version", b"").decode()
        == pre_spec.get("artifact_schema_version"),
        "predecision schema",
    )
    require(
        all(name in pre_parquet.schema_arrow.names for name in (*KEY_COLUMNS, FEATURE_COLUMNS[5])),
        "predecision required columns",
    )
    return {
        "episodes": [
            {
                "certified_episode_index": row[0],
                "field_id": row[1],
                "topic_id": row[2],
                "entry_year": row[3],
            }
            for row in rows
        ],
        "selection_path": selection_path,
        "certificate_path": certificate_path,
        "predecision_path": pre_path,
        "bindings": {
            "certified_episode_selection": {
                "root_basename": selection_root.name,
                "receipt_sha256": selection_spec["receipt_sha256"],
                "artifact_sha256": selection_spec["artifact_sha256"],
                "rows": 4,
            },
            "content_selection_certificate": {
                "root_basename": certificate_root.name,
                "receipt_sha256": certificate_spec["receipt_sha256"],
                "artifact_sha256": certificate_spec["artifact_sha256"],
                "status": certificate["status"],
            },
            "predecision_author_measurement": {
                "root_basename": pre_root.name,
                "receipt_sha256": pre_spec["receipt_sha256"],
                "artifact_sha256": pre_spec["artifact_sha256"],
                "rows": ROWS,
            },
        },
    }


def _sql_path(path: Path | str) -> str:
    return str(path).replace("'", "''")


def _scalar(connection: duckdb.DuckDBPyConnection, query: str) -> int:
    row = connection.execute(query).fetchone()
    require(row is not None and len(row) == 1, "scalar query")
    return int(row[0])


def _episode_values(episodes: Sequence[Mapping[str, int]]) -> str:
    return ",".join(
        "("
        + ",".join(
            str(int(row[key]))
            for key in (
                "certified_episode_index",
                "field_id",
                "topic_id",
                "entry_year",
            )
        )
        + ")"
        for row in episodes
    )


def _write_features(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    path: Path,
) -> int:
    partial = path.with_name(f".{path.name}.partial")
    require(not path.exists() and not partial.exists(), "feature output exists")
    reader = connection.execute(query).fetch_record_batch(rows_per_batch=262_144)
    writer = pq.ParquetWriter(
        partial,
        OUTPUT_SCHEMA,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
    )
    rows = 0
    try:
        for batch in reader:
            cast = batch.cast(OUTPUT_SCHEMA)
            writer.write_batch(cast)
            rows += cast.num_rows
    finally:
        writer.close()
    require(rows == ROWS, "written feature rows")
    os.chmod(partial, 0o600)
    partial.rename(path)
    return rows


def _work_accounting(
    connection: duckdb.DuckDBPyConnection,
    work_glob: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = connection.execute(
        f"""
        SELECT
          e.certified_episode_index,
          e.field_id,
          e.topic_id,
          e.entry_year,
          count(*) AS candidate_window_rows,
          count(*) FILTER (
            WHERE w.primary_topic_id IS NULL OR w.primary_field_id IS NULL
          ) AS missing_primary_topic_or_field_rows,
          count(*) FILTER (
            WHERE w.primary_topic_id IS NOT NULL
              AND w.primary_field_id IS NOT NULL
              AND (
                w.primary_topic_id <> e.topic_id
                OR w.primary_field_id <> e.field_id
              )
          ) AS wrong_primary_topic_or_field_rows,
          count(*) FILTER (
            WHERE w.primary_topic_id = e.topic_id
              AND w.primary_field_id = e.field_id
              AND w.temporal_disposition <> 2
          ) AS invalid_temporal_disposition_rows,
          count(*) FILTER (
            WHERE w.primary_topic_id = e.topic_id
              AND w.primary_field_id = e.field_id
              AND w.temporal_disposition = 2
              AND (
                w.work_type_code NOT IN (1, 2, 3, 4, 7, 16, 18, 21)
                OR w.is_paratext
              )
          ) AS excluded_work_type_or_paratext_rows,
          count(*) FILTER (
            WHERE w.primary_topic_id = e.topic_id
              AND w.primary_field_id = e.field_id
              AND w.temporal_disposition = 2
              AND w.work_type_code IN (1, 2, 3, 4, 7, 16, 18, 21)
              AND NOT w.is_paratext
              AND w.is_retracted
          ) AS retracted_rows,
          count(*) FILTER (
            WHERE w.primary_topic_id = e.topic_id
              AND w.primary_field_id = e.field_id
              AND w.temporal_disposition = 2
              AND w.work_type_code IN (1, 2, 3, 4, 7, 16, 18, 21)
              AND NOT w.is_paratext
              AND w.is_retracted IS DISTINCT FROM TRUE
          ) AS eligible_prelink_rows,
          count(*) FILTER (
            WHERE w.primary_topic_id = e.topic_id
              AND w.primary_field_id = e.field_id
              AND w.temporal_disposition = 2
              AND w.work_type_code IN (1, 2, 3, 4, 7, 16, 18, 21)
              AND NOT w.is_paratext
              AND w.is_retracted IS NULL
          ) AS retained_null_retraction_rows
        FROM read_parquet('{work_glob}') AS w
        INNER JOIN episodes AS e
          ON w.publication_year BETWEEN e.entry_year - 3 AND e.entry_year
        GROUP BY ALL
        ORDER BY e.certified_episode_index
        """
    ).fetchall()
    names = (
        "certified_episode_index",
        "field_id",
        "topic_id",
        "entry_year",
        "candidate_window_rows",
        "missing_primary_topic_or_field_rows",
        "wrong_primary_topic_or_field_rows",
        "invalid_temporal_disposition_rows",
        "excluded_work_type_or_paratext_rows",
        "retracted_rows",
        "eligible_prelink_rows",
        "retained_null_retraction_rows",
    )
    by_episode = [dict(zip(names, map(int, row), strict=True)) for row in rows]
    require(len(by_episode) == 4, "work accounting episodes")
    closure_names = (
        "missing_primary_topic_or_field_rows",
        "wrong_primary_topic_or_field_rows",
        "invalid_temporal_disposition_rows",
        "excluded_work_type_or_paratext_rows",
        "retracted_rows",
        "eligible_prelink_rows",
    )
    for row in by_episode:
        require(
            row["candidate_window_rows"]
            == sum(row[name] for name in closure_names),
            "candidate window accounting closure",
        )
    totals = {
        name: sum(row[name] for row in by_episode)
        for name in names[4:]
    }
    totals["outside_window_rows_scanned"] = 0
    totals["forbidden_post_entry_offset_scan_count"] = 0
    return by_episode, totals


def _markdown(report: Mapping[str, Any]) -> str:
    output = _mapping(report["output_accounting"], "output accounting")
    event = _mapping(report["author_event_accounting"], "event accounting")
    work = _mapping(report["work_accounting"], "work accounting")
    return f"""# Pivot-feedback D0 production entry source

Status: `{report['status']}`

The operator-local feature source retains all {output['rows']:,} registered
author-episode rows across the exact four certified topics. It contains only
the five private key columns and three count features; no classifier state or
pivot label is materialized.

The build found {work['totals']['eligible_prelink_rows']:,} eligible target-topic
works in `t-3..t`, of which {event['linked_work_rows']:,} had at least one valid
author token. After registered-population restriction, the source contains
{event['registered_distinct_author_work_pairs']:,} distinct author-work pairs.

Content confirmation remains an episode-topic certificate plus exact
current-snapshot primary-topic membership. No individual-work API or human
label was requested. No post-entry, persistence, impact, model, or G4 row was
opened. Production classifier states require a successor gate.
"""


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
    config = _read_yaml(config_path, "config")
    repository = _validate_repository(config)
    output_contract = _mapping(config.get("output_contract"), "output contract")
    require(output.name == output_contract.get("root_basename"), "output basename")
    require(not output.exists(), "output already exists")
    feature_contract = _mapping(config.get("feature_contract"), "feature contract")
    require(
        tuple(feature_contract.get("output_columns", ())) == FEATURE_COLUMNS,
        "output columns",
    )
    require(feature_contract.get("post_entry_offsets_allowed") == [], "post-entry offsets")
    require(feature_contract.get("team_cap") == "NONE", "team cap")
    require(
        feature_contract.get("minimum_history_threshold") == "NONE",
        "history threshold",
    )
    forbidden = _mapping(
        config.get("forbidden_inputs_and_outputs"),
        "forbidden inputs and outputs",
    )
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden boundary")

    inputs = _mapping(config.get("operator_inputs"), "operator inputs")
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
    small = _validate_small_inputs(operator_base=operator_base, inputs=inputs)
    episodes = small["episodes"]

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
              entry_year USMALLINT
            )
            """
        )
        connection.execute(
            f"INSERT INTO episodes VALUES {_episode_values(episodes)}"
        )
        work_glob = _sql_path(work_paths[0].parent / "*.parquet")
        event_glob = _sql_path(event_paths[0].parent / "*.parquet")
        predecision_path = _sql_path(small["predecision_path"])
        require(
            _scalar(
                connection,
                f"SELECT count(*) FROM read_parquet('{work_glob}')",
            )
            == work_binding["rows"],
            "work input row closure",
        )
        require(
            _scalar(
                connection,
                f"SELECT count(*) FROM read_parquet('{event_glob}')",
            )
            == event_binding["rows"],
            "event input row closure",
        )
        work_by_episode, work_totals = _work_accounting(connection, work_glob)
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
              CAST(w.publication_year = e.entry_year AS BOOLEAN) AS is_focal
            FROM read_parquet('{work_glob}') AS w
            INNER JOIN episodes AS e
              ON w.publication_year BETWEEN e.entry_year - 3 AND e.entry_year
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
            == work_totals["eligible_prelink_rows"],
            "eligible work accounting",
        )
        require(
            _scalar(
                connection,
                """
                SELECT count(*) - count(DISTINCT work_id_numeric)
                FROM eligible_work
                """,
            )
            == 0,
            "duplicate eligible work",
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE joined_event AS
            SELECT
              ew.certified_episode_index,
              ew.field_id,
              ew.entry_year,
              ew.work_id_numeric,
              ew.publication_year,
              ew.temporal_disposition,
              ew.declared_authors_count,
              ew.is_focal,
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
        event_join_rows = _scalar(connection, "SELECT count(*) FROM joined_event")
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
              is_focal
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
        zero_valid_author = _scalar(
            connection,
            """
            SELECT count(*)
            FROM eligible_work AS ew
            ANTI JOIN (
              SELECT DISTINCT certified_episode_index, work_id_numeric
              FROM valid_pairs
            ) AS vp USING (certified_episode_index, work_id_numeric)
            """,
        )
        require(
            linked_work_rows + zero_valid_author
            == work_totals["eligible_prelink_rows"],
            "eligible work linkage closure",
        )
        connection.execute(
            f"""
            CREATE TEMP VIEW registered AS
            SELECT
              certified_episode_index,
              field_id,
              topic_id,
              entry_year,
              author_numeric_id,
              pre_entry_target_topic_work_count
            FROM read_parquet('{predecision_path}')
            """
        )
        require(
            _scalar(connection, "SELECT count(*) FROM registered") == ROWS,
            "registered rows",
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
                )
                FROM registered
                """,
            )
            == 0,
            "registered duplicate key",
        )
        connection.execute(
            """
            CREATE TEMP TABLE registered_pairs AS
            SELECT vp.*
            FROM valid_pairs AS vp
            INNER JOIN registered AS r
              USING (certified_episode_index, author_numeric_id)
            """
        )
        registered_pairs = _scalar(
            connection,
            "SELECT count(*) FROM registered_pairs",
        )
        unregistered_pairs = _scalar(
            connection,
            """
            SELECT count(*)
            FROM valid_pairs AS vp
            ANTI JOIN registered AS r
              USING (certified_episode_index, author_numeric_id)
            """,
        )
        require(
            registered_pairs + unregistered_pairs == distinct_pairs,
            "registered pair closure",
        )
        connection.execute(
            """
            CREATE TEMP TABLE author_counts AS
            SELECT
              certified_episode_index,
              author_numeric_id,
              CAST(count(*) FILTER (WHERE NOT is_focal) AS UINTEGER)
                AS trailing_3y_target_topic_work_count,
              CAST(count(*) FILTER (WHERE is_focal) AS UINTEGER)
                AS focal_confirmed_eligible_target_topic_work_count
            FROM registered_pairs
            GROUP BY ALL
            """
        )
        feature_path = temporary / str(output_contract["feature_basename"])
        output_query = """
            SELECT
              r.certified_episode_index,
              r.field_id,
              r.topic_id,
              r.entry_year,
              r.author_numeric_id,
              CAST(r.pre_entry_target_topic_work_count AS UINTEGER)
                AS pre_entry_target_topic_work_count,
              CAST(
                coalesce(c.trailing_3y_target_topic_work_count, 0)
                AS UINTEGER
              ) AS trailing_3y_target_topic_work_count,
              CAST(
                coalesce(
                  c.focal_confirmed_eligible_target_topic_work_count,
                  0
                ) AS UINTEGER
              ) AS focal_confirmed_eligible_target_topic_work_count
            FROM registered AS r
            LEFT JOIN author_counts AS c
              USING (certified_episode_index, author_numeric_id)
            ORDER BY
              r.certified_episode_index,
              r.field_id,
              r.topic_id,
              r.entry_year,
              r.author_numeric_id
        """
        _write_features(connection, output_query, feature_path)
        escaped_feature = _sql_path(feature_path)
        connection.execute(
            f"CREATE TEMP VIEW output_features AS SELECT * FROM read_parquet('{escaped_feature}')"
        )
        invariant_violations = {
            "trailing_count_exceeds_pre_entry_count": _scalar(
                connection,
                """
                SELECT count(*) FROM output_features
                WHERE trailing_3y_target_topic_work_count
                    > pre_entry_target_topic_work_count
                """,
            ),
            "zero_pre_entry_positive_trailing": _scalar(
                connection,
                """
                SELECT count(*) FROM output_features
                WHERE pre_entry_target_topic_work_count = 0
                  AND trailing_3y_target_topic_work_count > 0
                """,
            ),
            "negative_or_null_feature_rows": _scalar(
                connection,
                """
                SELECT count(*) FROM output_features
                WHERE pre_entry_target_topic_work_count IS NULL
                   OR trailing_3y_target_topic_work_count IS NULL
                   OR focal_confirmed_eligible_target_topic_work_count IS NULL
                """,
            ),
        }
        require(
            all(value == 0 for value in invariant_violations.values()),
            "feature invariant",
        )
        episode_rows = connection.execute(
            """
            SELECT
              certified_episode_index,
              field_id,
              topic_id,
              entry_year,
              count(*) AS author_rows,
              sum(trailing_3y_target_topic_work_count) AS trailing_pair_rows,
              sum(focal_confirmed_eligible_target_topic_work_count)
                AS focal_pair_rows,
              count(*) FILTER (
                WHERE trailing_3y_target_topic_work_count > 0
              ) AS trailing_positive_author_rows,
              count(*) FILTER (
                WHERE focal_confirmed_eligible_target_topic_work_count > 0
              ) AS focal_positive_author_rows
            FROM output_features
            GROUP BY ALL
            ORDER BY certified_episode_index
            """
        ).fetchall()
        episode_names = (
            "certified_episode_index",
            "field_id",
            "topic_id",
            "entry_year",
            "author_rows",
            "trailing_pair_rows",
            "focal_pair_rows",
            "trailing_positive_author_rows",
            "focal_positive_author_rows",
        )
        output_by_episode = [
            dict(zip(episode_names, map(int, row), strict=True))
            for row in episode_rows
        ]
        require(
            sum(row["author_rows"] for row in output_by_episode) == ROWS,
            "output episode row closure",
        )
        require(
            sum(row["trailing_pair_rows"] for row in output_by_episode)
            == _scalar(
                connection,
                "SELECT count(*) FROM registered_pairs WHERE NOT is_focal",
            ),
            "trailing pair closure",
        )
        require(
            sum(row["focal_pair_rows"] for row in output_by_episode)
            == _scalar(
                connection,
                "SELECT count(*) FROM registered_pairs WHERE is_focal",
            ),
            "focal pair closure",
        )
        _validate_unchanged(work_paths, work_stats, "work input")
        _validate_unchanged(event_paths, event_stats, "event input")
        event_accounting = {
            "source_event_rows": event_binding["rows"],
            "eligible_work_joined_event_rows": event_join_rows,
            **mismatch,
            "valid_event_rows": event_join_rows,
            "distinct_author_work_pairs": distinct_pairs,
            "duplicate_author_work_event_excess_rows": event_join_rows
            - distinct_pairs,
            "linked_work_rows": linked_work_rows,
            "zero_valid_author_token_work_rows": zero_valid_author,
            "registered_distinct_author_work_pairs": registered_pairs,
            "unregistered_distinct_author_work_pairs": unregistered_pairs,
        }
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": STATUS,
            "mode": config["mode"],
            "config_sha256": _sha256(config_path),
            "repository_bindings": repository,
            "operator_input_bindings": {
                "cutoff_work_metadata": work_binding,
                "author_work_events": event_binding,
                **small["bindings"],
            },
            "scope": {
                "exact_certified_episode_rows": 4,
                "registered_author_episode_rows": ROWS,
                "content_confirmation_unit": (
                    "CERTIFIED_EPISODE_TOPIC_NOT_INDIVIDUAL_WORK"
                ),
                "production_classifier_state_materialized": False,
                "pivot_label_materialized": False,
                "post_entry_offsets_read": [],
                "entry_or_persistence_outcomes_read": False,
                "realized_citation_outcomes_read": False,
                "api_requests_created": 0,
                "paid_api_calls_submitted": 0,
                "model_fitted": False,
                "g4_open": False,
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "episodes": episodes,
            "work_accounting": {
                "source_work_rows": work_binding["rows"],
                "by_episode": work_by_episode,
                "totals": work_totals,
            },
            "author_event_accounting": event_accounting,
            "output_accounting": {
                "rows": ROWS,
                "columns": len(FEATURE_COLUMNS),
                "key_columns": list(KEY_COLUMNS),
                "feature_columns": list(FEATURE_COLUMNS[5:]),
                "by_episode": output_by_episode,
                "invariant_violations": invariant_violations,
                "production_classifier_state_columns": 0,
                "pivot_label_columns": 0,
                "forbidden_post_entry_offset_scan_count": 0,
            },
            "authorization": {
                "independent_source_verification": "REQUIRED_NEXT",
                "production_pivot_classification": "NOT_AUTHORIZED",
                "discovery_outcome_opening": "NOT_AUTHORIZED",
                "new_api_request_creation": "NOT_AUTHORIZED",
                "final_candidate_eligibility": "NOT_AUTHORIZED",
                "g0_measurement_freeze": "NOT_AUTHORIZED",
                "model_fitting": "NOT_AUTHORIZED",
                "g4_open": "NOT_AUTHORIZED",
            },
            "execution": {
                "workload_class": "CPU_IO",
                "workers": max(1, workers),
                "duckdb_threads": max(1, threads),
                "gpu_used": False,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "builder_relative_path": Path(__file__)
                .resolve()
                .relative_to(REPO_ROOT)
                .as_posix(),
                "builder_sha256": _sha256(Path(__file__).resolve()),
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
        for path in (feature_path, report_path, markdown_path):
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
            "feature_rows": ROWS,
            "feature_columns": len(FEATURE_COLUMNS),
            "exact_certified_episode_rows": 4,
            "registered_author_episode_rows": ROWS,
            "production_classifier_state_materialized": False,
            "pivot_label_materialized": False,
            "post_entry_rows_read": False,
            "outcomes_read": False,
            "api_requests_created": 0,
            "independent_verification_complete": False,
            "production_pivot_classification_authorized": False,
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
