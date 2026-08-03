#!/usr/bin/env python3
"""Independently replay the exact 4,301-row D0 persistence outcome source."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
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
PRODUCER_STATUS = "PASS_EXACT_4301_NEW_TOPIC_ENTRY_THREE_YEAR_PERSISTENCE_SOURCE"
VERIFY_SCHEMA = "pivot_feedback_d0_persistence_outcome_source_verification.v1"
VERIFY_STATUS = (
    "PASS_INDEPENDENT_EXACT_4301_THREE_YEAR_PERSISTENCE_SOURCE_ALL_ROW_REPLAY"
)
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
WORK_COLUMNS = [
    "work_id_numeric",
    "publication_year",
    "temporal_disposition",
    "work_type_code",
    "is_paratext",
    "is_retracted",
    "declared_authors_count",
    "primary_topic_id",
    "primary_field_id",
]
EVENT_COLUMNS = list(EVENT_SCHEMA.names)
ELIGIBLE_TYPES = np.array([1, 2, 3, 4, 7, 16, 18, 21], dtype=np.uint8)
ELIGIBLE_DTYPE = np.dtype(
    [
        ("work", "<u8"),
        ("episode", "u1"),
        ("field", "u1"),
        ("year", "<u2"),
        ("disposition", "u1"),
        ("declared", "<u4"),
        ("offset", "u1"),
    ]
)
PAIR_DTYPE = np.dtype(
    [
        ("episode", "u1"),
        ("author", "<u8"),
        ("work", "<u8"),
        ("offset", "u1"),
    ]
)
PRIMARY_KEY_DTYPE = np.dtype([("episode", "u1"), ("author", "<u8")])
FULL_KEY_DTYPE = np.dtype(
    [
        ("episode", "u1"),
        ("field", "u1"),
        ("topic", "<u4"),
        ("year", "<u2"),
        ("author", "<u8"),
    ]
)
EXPECTED_FILES = {
    "PERSISTENCE_OUTCOME_SOURCE.md",
    "persistence_outcome_source.json",
    "persistence_outcome_source.outputs.json",
    "persistence_outcome_source.parquet",
}


class PersistenceSourceVerificationError(RuntimeError):
    """Fail-closed independent persistence-source replay error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PersistenceSourceVerificationError(message)


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
        raise PersistenceSourceVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise PersistenceSourceVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


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
    for raw in _list(receipt.get("artifacts"), "receipt artifacts"):
        item = _mapping(raw, "receipt artifact")
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
    spec_value: Any,
) -> tuple[Path, tuple[int, int, int, int, int], dict[str, Any]]:
    spec = _mapping(spec_value, "production classification")
    root = (operator_base / str(spec.get("root_basename"))).resolve(strict=True)
    require(root.parent == operator_base, "classification root parent")
    require(root.is_dir() and not root.is_symlink(), "classification root")
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, "classification root mode")
    receipt_path = root / str(spec.get("receipt_basename"))
    require(_sha256(receipt_path) == spec.get("receipt_sha256"), "classification receipt SHA")
    receipt = _read_json(receipt_path, "classification receipt")
    require(
        receipt.get("schema_version") == spec.get("receipt_schema_version"),
        "classification receipt schema",
    )
    require(receipt.get("status") == spec.get("required_status"), "classification status")
    require(
        receipt.get("artifact_tree_sha256") == spec.get("artifact_tree_sha256"),
        "classification artifact tree",
    )
    path = root / str(spec.get("artifact_basename"))
    require(_sha256(path) == spec.get("artifact_sha256"), "classification artifact SHA")
    item = _artifact_items(receipt).get(path.name)
    require(item is not None and item.get("sha256") == spec.get("artifact_sha256"), "classification receipt artifact")
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == CLASSIFICATION_ROWS, "classification rows")
    require(
        parquet.schema_arrow.equals(CLASSIFICATION_SCHEMA, check_metadata=True),
        "classification schema",
    )
    require(receipt.get("new_topic_entry_rows") == OUTPUT_ROWS, "classification primary rows")
    return path, _stat_tuple(path), {
        "root_basename": root.name,
        "receipt_sha256": spec["receipt_sha256"],
        "artifact_sha256": spec["artifact_sha256"],
        "rows": CLASSIFICATION_ROWS,
        "primary_rows": OUTPUT_ROWS,
    }


def _validate_partitioned_input(
    *,
    operator_base: Path,
    spec_value: Any,
    label: str,
    schema: pa.Schema,
    workers: int,
) -> tuple[list[Path], list[tuple[int, int, int, int, int]], dict[str, Any]]:
    spec = _mapping(spec_value, label)
    root = (operator_base / str(spec.get("root_basename"))).resolve(strict=True)
    require(root.parent == operator_base, f"{label} root parent")
    require(root.is_dir() and not root.is_symlink(), f"{label} root")
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, f"{label} root mode")
    receipt_path = root / str(spec.get("receipt_basename"))
    require(_sha256(receipt_path) == spec.get("receipt_sha256"), f"{label} receipt SHA")
    receipt = _read_json(receipt_path, f"{label} receipt")
    require(
        receipt.get("schema_version") == spec.get("receipt_schema_version"),
        f"{label} receipt schema",
    )
    require(receipt.get("status") == spec.get("required_status"), f"{label} status")
    prefix = str(spec.get("artifact_prefix"))
    artifacts = [
        (relative, item)
        for relative, item in _artifact_items(receipt).items()
        if relative.startswith(prefix)
    ]
    artifacts.sort(key=lambda pair: pair[0])
    require(
        len(artifacts) == spec.get("artifact_count") == PARTITIONS,
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
    require(rows == spec.get("rows"), f"{label} row closure")

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
    require(total_bytes == spec.get("artifact_bytes"), f"{label} byte closure")
    return paths, stats, {
        "root_basename": root.name,
        "receipt_sha256": spec["receipt_sha256"],
        "partitions_rehashed": len(verified),
        "artifact_bytes": total_bytes,
        "rows": rows,
        "selected_tree_sha256": hashlib.sha256(
            _canonical_bytes(verified)
        ).hexdigest(),
    }


def _numpy(
    table: pa.Table,
    name: str,
    dtype: np.dtype[Any],
    *,
    fill: Any | None = None,
) -> np.ndarray[Any, Any]:
    array = table[name].combine_chunks()
    if fill is not None and array.null_count:
        array = pc.fill_null(array, pa.scalar(fill, type=array.type))
    return np.asarray(array.to_numpy(zero_copy_only=False), dtype=dtype)


def _primary_rows(
    classification_path: Path,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    columns = [*KEY_COLUMNS, "observable_history_pivot_state", "observable_history_new_topic_entry"]
    table = pq.read_table(classification_path, columns=columns)
    require(table.num_rows == CLASSIFICATION_ROWS, "classification replay rows")
    state_new = pc.equal(table["observable_history_pivot_state"], "NEW_TOPIC_ENTRY")
    boolean = table["observable_history_new_topic_entry"].combine_chunks()
    mismatch = pc.sum(pc.cast(pc.not_equal(state_new, boolean), pa.int64())).as_py()
    require(int(mismatch or 0) == 0, "classification state/boolean mismatch")
    primary = table.filter(state_new)
    require(primary.num_rows == OUTPUT_ROWS, "classification replay primary rows")

    full_keys = np.empty(OUTPUT_ROWS, dtype=FULL_KEY_DTYPE)
    full_keys["episode"] = _numpy(primary, "certified_episode_index", np.dtype("u1"))
    full_keys["field"] = _numpy(primary, "field_id", np.dtype("u1"))
    full_keys["topic"] = _numpy(primary, "topic_id", np.dtype("<u4"))
    full_keys["year"] = _numpy(primary, "entry_year", np.dtype("<u2"))
    full_keys["author"] = _numpy(primary, "author_numeric_id", np.dtype("<u8"))
    order = np.argsort(
        full_keys,
        order=("episode", "field", "topic", "year", "author"),
        kind="stable",
    )
    full_keys = full_keys[order]
    require(np.unique(full_keys).size == OUTPUT_ROWS, "classification primary duplicate key")
    pair_keys = np.empty(OUTPUT_ROWS, dtype=PRIMARY_KEY_DTYPE)
    pair_keys["episode"] = full_keys["episode"]
    pair_keys["author"] = full_keys["author"]
    pair_order = np.argsort(pair_keys, order=("episode", "author"), kind="stable")
    pair_keys = pair_keys[pair_order]
    require(np.unique(pair_keys).size == OUTPUT_ROWS, "primary pair key duplicate")
    return full_keys, pair_keys


def _scan_work_partition(
    pair: tuple[int, Path],
    episodes: Sequence[Sequence[int]],
) -> dict[str, Any]:
    index, path = pair
    table = pq.read_table(path, columns=WORK_COLUMNS)
    work = _numpy(table, "work_id_numeric", np.dtype("<u8"))
    year_array = table["publication_year"].combine_chunks()
    year_null = np.asarray(
        pc.is_null(year_array).to_numpy(zero_copy_only=False),
        dtype=bool,
    )
    year = _numpy(table, "publication_year", np.dtype("<u2"), fill=0)
    disposition = _numpy(table, "temporal_disposition", np.dtype("u1"))
    work_type = _numpy(table, "work_type_code", np.dtype("u1"))
    paratext = _numpy(table, "is_paratext", np.dtype("?"))
    retract_array = table["is_retracted"].combine_chunks()
    retract_null = np.asarray(
        pc.is_null(retract_array).to_numpy(zero_copy_only=False),
        dtype=bool,
    )
    retracted = _numpy(table, "is_retracted", np.dtype("?"), fill=False)
    declared = _numpy(table, "declared_authors_count", np.dtype("<u4"))
    topic = _numpy(table, "primary_topic_id", np.dtype("<u4"), fill=0)
    field = _numpy(table, "primary_field_id", np.dtype("u1"), fill=0)

    accounting = {
        "candidate_window_rows": 0,
        "excluded_wrong_topic_or_field_rows": 0,
        "excluded_temporal_disposition_rows": 0,
        "excluded_paratext_or_work_type_rows": 0,
        "excluded_retracted_rows": 0,
        "eligible_prelink_work_rows": 0,
        "retained_null_retraction_rows": 0,
        "excluded_missing_publication_year_target_rows": 0,
    }
    chunks: list[np.ndarray[Any, Any]] = []
    for episode, episode_field, episode_topic, entry_year, _ in episodes:
        null_target = (
            year_null & (topic == episode_topic) & (field == episode_field)
        )
        accounting["excluded_missing_publication_year_target_rows"] += int(
            np.count_nonzero(null_target)
        )
        candidate = (
            (~year_null) & (year >= entry_year + 1) & (year <= entry_year + 3)
        )
        accounting["candidate_window_rows"] += int(np.count_nonzero(candidate))
        wrong = candidate & (
            (topic != episode_topic) | (field != episode_field)
        )
        accounting["excluded_wrong_topic_or_field_rows"] += int(
            np.count_nonzero(wrong)
        )
        remaining = candidate & ~wrong
        bad_disposition = remaining & (disposition != 2)
        accounting["excluded_temporal_disposition_rows"] += int(
            np.count_nonzero(bad_disposition)
        )
        remaining &= ~bad_disposition
        bad_type = remaining & (
            (~np.isin(work_type, ELIGIBLE_TYPES)) | paratext
        )
        accounting["excluded_paratext_or_work_type_rows"] += int(
            np.count_nonzero(bad_type)
        )
        remaining &= ~bad_type
        bad_retract = remaining & retracted
        accounting["excluded_retracted_rows"] += int(
            np.count_nonzero(bad_retract)
        )
        eligible = remaining & ~bad_retract
        count = int(np.count_nonzero(eligible))
        accounting["eligible_prelink_work_rows"] += count
        accounting["retained_null_retraction_rows"] += int(
            np.count_nonzero(eligible & retract_null)
        )
        if count:
            chunk = np.empty(count, dtype=ELIGIBLE_DTYPE)
            chunk["work"] = work[eligible]
            chunk["episode"] = episode
            chunk["field"] = episode_field
            chunk["year"] = year[eligible]
            chunk["disposition"] = disposition[eligible]
            chunk["declared"] = declared[eligible]
            chunk["offset"] = year[eligible] - entry_year
            chunks.append(chunk)
    eligible_rows = (
        np.concatenate(chunks)
        if chunks
        else np.empty(0, dtype=ELIGIBLE_DTYPE)
    )
    return {
        "index": index,
        "source_rows": table.num_rows,
        "eligible": eligible_rows,
        "accounting": accounting,
    }


def _scan_event_partition(
    pair: tuple[int, Path],
    eligible_sorted: np.ndarray[Any, Any],
) -> dict[str, Any]:
    index, path = pair
    table = pq.read_table(path, columns=EVENT_COLUMNS)
    work = _numpy(table, "work_numeric_id", np.dtype("<u8"))
    positions = np.searchsorted(eligible_sorted["work"], work)
    matched = positions < eligible_sorted.size
    safe = np.minimum(positions, max(eligible_sorted.size - 1, 0))
    if eligible_sorted.size:
        matched &= eligible_sorted["work"][safe] == work
    else:
        matched[:] = False
    selected = np.flatnonzero(matched)
    if not selected.size:
        return {
            "index": index,
            "source_rows": table.num_rows,
            "joined_rows": 0,
            "mismatches": {
                "publication_year_mismatch_rows": 0,
                "temporal_disposition_mismatch_rows": 0,
                "declared_authors_count_mismatch_rows": 0,
                "primary_field_id_mismatch_rows": 0,
            },
            "pairs": np.empty(0, dtype=PAIR_DTYPE),
        }
    metadata = eligible_sorted[positions[selected]]
    event_year = _numpy(table, "publication_year", np.dtype("<u2"))[selected]
    event_disposition = _numpy(
        table,
        "temporal_disposition",
        np.dtype("u1"),
    )[selected]
    event_declared = _numpy(
        table,
        "declared_authors_count",
        np.dtype("<u4"),
    )[selected]
    event_field = _numpy(table, "primary_field_id", np.dtype("u1"))[selected]
    matches = {
        "publication_year_mismatch_rows": event_year != metadata["year"],
        "temporal_disposition_mismatch_rows": (
            event_disposition != metadata["disposition"]
        ),
        "declared_authors_count_mismatch_rows": (
            event_declared != metadata["declared"]
        ),
        "primary_field_id_mismatch_rows": event_field != metadata["field"],
    }
    valid = np.ones(selected.size, dtype=bool)
    mismatch_counts: dict[str, int] = {}
    for name, mismatch in matches.items():
        mismatch_counts[name] = int(np.count_nonzero(mismatch))
        valid &= ~mismatch
    count = int(np.count_nonzero(valid))
    pairs = np.empty(count, dtype=PAIR_DTYPE)
    if count:
        author = _numpy(table, "author_numeric_id", np.dtype("<u8"))[selected]
        pairs["episode"] = metadata["episode"][valid]
        pairs["author"] = author[valid]
        pairs["work"] = work[selected][valid]
        pairs["offset"] = metadata["offset"][valid]
    return {
        "index": index,
        "source_rows": table.num_rows,
        "joined_rows": selected.size,
        "mismatches": mismatch_counts,
        "pairs": pairs,
    }


def _sum_accounting(
    results: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, int]:
    names = set()
    for result in results:
        names.update(_mapping(result[key], key))
    return {
        name: sum(int(_mapping(result[key], key).get(name, 0)) for result in results)
        for name in sorted(names)
    }


def _is_primary_pair(
    pairs: np.ndarray[Any, Any],
    primary_keys: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    candidate = np.empty(pairs.size, dtype=PRIMARY_KEY_DTYPE)
    candidate["episode"] = pairs["episode"]
    candidate["author"] = pairs["author"]
    positions = np.searchsorted(primary_keys, candidate)
    matched = positions < primary_keys.size
    safe = np.minimum(positions, primary_keys.size - 1)
    matched &= primary_keys[safe] == candidate
    return matched


def _independent_counts(
    full_keys: np.ndarray[Any, Any],
    primary_keys: np.ndarray[Any, Any],
    pairs: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], dict[str, int]]:
    primary_rows = primary_keys.size
    full_rows = full_keys.size
    mask = _is_primary_pair(pairs, primary_keys)
    primary_pairs = pairs[mask]
    other_pairs = pairs[~mask]
    pair_keys = np.empty(primary_pairs.size, dtype=PRIMARY_KEY_DTYPE)
    pair_keys["episode"] = primary_pairs["episode"]
    pair_keys["author"] = primary_pairs["author"]
    positions = np.searchsorted(primary_keys, pair_keys)
    require(
        np.all(positions < primary_keys.size)
        and np.all(primary_keys[positions] == pair_keys),
        "primary pair lookup",
    )
    counts_pair_order = np.zeros((primary_rows, 3), dtype=np.uint32)
    if primary_pairs.size:
        np.add.at(
            counts_pair_order,
            (positions, primary_pairs["offset"].astype(np.int64) - 1),
            1,
        )
    full_pair_keys = np.empty(full_rows, dtype=PRIMARY_KEY_DTYPE)
    full_pair_keys["episode"] = full_keys["episode"]
    full_pair_keys["author"] = full_keys["author"]
    full_positions = np.searchsorted(primary_keys, full_pair_keys)
    require(
        np.all(full_positions < primary_keys.size)
        and np.all(primary_keys[full_positions] == full_pair_keys),
        "full output primary lookup",
    )
    return counts_pair_order[full_positions], {
        "registered_primary_distinct_author_work_pairs": int(primary_pairs.size),
        "unregistered_or_nonprimary_distinct_author_work_pairs": int(other_pairs.size),
    }


def _validate_output(
    *,
    config: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    output_contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == output_contract.get("root_basename"), "output basename")
    require(output_root.is_dir() and not output_root.is_symlink(), "output root")
    require(stat.S_IMODE(output_root.stat().st_mode) == 0o700, "output root mode")
    require(
        {path.name for path in output_root.iterdir()} == EXPECTED_FILES,
        "output file set",
    )
    receipt_path = output_root / str(output_contract.get("receipt_basename"))
    receipt = _read_json(receipt_path, "output receipt")
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "output receipt schema")
    require(receipt.get("status") == PRODUCER_STATUS, "producer status")
    require(receipt.get("receipt_written_last") is True, "receipt written last")
    require(receipt.get("persistence_source_rows") == OUTPUT_ROWS, "receipt rows")
    require(receipt.get("persistence_source_columns") == len(OUTPUT_COLUMNS), "receipt columns")
    require(receipt.get("denominator_rows_filtered") == 0, "filtered denominator")
    require(receipt.get("denominator_rows_mutated") == 0, "mutated denominator")
    require(receipt.get("post_entry_offsets_read") == [1, 2, 3], "receipt offsets")
    require(receipt.get("confirmatory_entry_year_rows_read") == 0, "confirmatory rows")
    require(receipt.get("association_or_effect_computed") is False, "effect boundary")
    require(receipt.get("api_requests_created") == 0, "API boundary")
    items = _artifact_items(receipt)
    tree = hashlib.sha256(
        _canonical_bytes(
            [
                (item["relative_path"], item["bytes"], item["sha256"])
                for item in sorted(items.values(), key=lambda value: value["relative_path"])
            ]
        )
    ).hexdigest()
    require(tree == receipt.get("artifact_tree_sha256"), "output artifact tree")
    for name, item in items.items():
        path = output_root / name
        require(path.is_file() and not path.is_symlink(), f"output artifact {name}")
        require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"output mode {name}")
        require(path.stat().st_size == item.get("bytes"), f"output bytes {name}")
        require(_sha256(path) == item.get("sha256"), f"output SHA {name}")
    report_path = output_root / str(output_contract.get("report_basename"))
    report = _read_json(report_path, "output report")
    require(report.get("schema_version") == REPORT_SCHEMA, "output report schema")
    require(report.get("status") == PRODUCER_STATUS, "output report status")
    artifact_path = output_root / str(output_contract.get("artifact_basename"))
    artifact_item = items.get(artifact_path.name)
    require(artifact_item is not None and artifact_item.get("rows") == OUTPUT_ROWS, "artifact receipt rows")
    parquet = pq.ParquetFile(artifact_path)
    require(parquet.metadata.num_rows == OUTPUT_ROWS, "artifact rows")
    require(
        parquet.schema_arrow.equals(OUTPUT_SCHEMA, check_metadata=True),
        "artifact schema",
    )
    return receipt, report, artifact_path


def _compare_output(
    *,
    artifact_path: Path,
    full_keys: np.ndarray[Any, Any],
    counts: np.ndarray[Any, Any],
) -> dict[str, Any]:
    table = pq.read_table(artifact_path)
    observed_keys = np.empty(OUTPUT_ROWS, dtype=FULL_KEY_DTYPE)
    observed_keys["episode"] = _numpy(table, "certified_episode_index", np.dtype("u1"))
    observed_keys["field"] = _numpy(table, "field_id", np.dtype("u1"))
    observed_keys["topic"] = _numpy(table, "topic_id", np.dtype("<u4"))
    observed_keys["year"] = _numpy(table, "entry_year", np.dtype("<u2"))
    observed_keys["author"] = _numpy(table, "author_numeric_id", np.dtype("<u8"))
    key_mismatches = int(np.count_nonzero(observed_keys != full_keys))
    state = np.asarray(table["observable_history_pivot_state"].to_pylist(), dtype=object)
    followup = _numpy(table, "persistence_followup_complete", np.dtype("?"))
    observed_counts = np.column_stack(
        [
            _numpy(table, "target_topic_work_count_t_plus_1", np.dtype("<u4")),
            _numpy(table, "target_topic_work_count_t_plus_2", np.dtype("<u4")),
            _numpy(table, "target_topic_work_count_t_plus_3", np.dtype("<u4")),
        ]
    )
    total = _numpy(
        table,
        "target_topic_work_count_t_plus_1_through_t_plus_3",
        np.dtype("<u4"),
    )
    expected_total = counts.sum(axis=1, dtype=np.uint32)
    expected_persistent = expected_total > 0
    outcome_state = np.asarray(
        table["three_year_target_topic_persistence_state"].to_pylist(),
        dtype=object,
    )
    persistent = _numpy(
        table,
        "three_year_target_topic_persistent",
        np.dtype("?"),
    )
    expected_state = np.where(
        expected_persistent,
        "MATURED_PERSISTENCE",
        "MATURED_NO_PERSISTENCE",
    )
    mismatches = {
        "key_value_mismatches": key_mismatches,
        "pivot_state_mismatches": int(np.count_nonzero(state != "NEW_TOPIC_ENTRY")),
        "followup_complete_mismatches": int(np.count_nonzero(~followup)),
        "offset_count_value_mismatches": int(np.count_nonzero(observed_counts != counts)),
        "total_count_value_mismatches": int(np.count_nonzero(total != expected_total)),
        "outcome_state_value_mismatches": int(
            np.count_nonzero(outcome_state != expected_state)
        ),
        "persistence_boolean_value_mismatches": int(
            np.count_nonzero(persistent != expected_persistent)
        ),
    }
    require(all(value == 0 for value in mismatches.values()), "output replay mismatch")
    by_episode = []
    for episode in sorted(np.unique(full_keys["episode"])):
        mask = full_keys["episode"] == episode
        row = {
            "certified_episode_index": int(episode),
            "field_id": int(full_keys["field"][mask][0]),
            "topic_id": int(full_keys["topic"][mask][0]),
            "entry_year": int(full_keys["year"][mask][0]),
            "rows": int(np.count_nonzero(mask)),
            "pair_rows_t_plus_1": int(counts[mask, 0].sum()),
            "pair_rows_t_plus_2": int(counts[mask, 1].sum()),
            "pair_rows_t_plus_3": int(counts[mask, 2].sum()),
            "qualifying_author_work_pairs": int(expected_total[mask].sum()),
            "matured_persistence_rows": int(
                np.count_nonzero(expected_persistent[mask])
            ),
            "matured_no_persistence_rows": int(
                np.count_nonzero(~expected_persistent[mask])
            ),
            "right_censored_rows": 0,
        }
        by_episode.append(row)
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
    return {
        "mismatches": mismatches,
        "by_episode": by_episode,
        "totals": totals,
    }


def verify(
    *,
    config_path: Path,
    operator_base: Path,
    output_root: Path,
    verification_output: Path,
    workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    config_path = config_path.resolve(strict=True)
    operator_base = operator_base.resolve(strict=True)
    output_root = output_root.resolve(strict=True)
    verification_output = verification_output.resolve()
    require(verification_output.parent == operator_base, "verification output parent")
    require(not verification_output.exists(), "verification output exists")
    config = _read_yaml(config_path, "config")
    repository = _validate_repository(config)
    inputs = _mapping(config.get("operator_inputs"), "operator inputs")
    classification_path, classification_stat, classification_binding = (
        _validate_classification(
            operator_base=operator_base,
            spec_value=inputs.get("production_classification"),
        )
    )
    work_paths, work_stats, work_binding = _validate_partitioned_input(
        operator_base=operator_base,
        spec_value=inputs.get("cutoff_work_metadata"),
        label="cutoff work metadata",
        schema=WORK_SCHEMA,
        workers=workers,
    )
    event_paths, event_stats, event_binding = _validate_partitioned_input(
        operator_base=operator_base,
        spec_value=inputs.get("author_work_events"),
        label="author work events",
        schema=EVENT_SCHEMA,
        workers=workers,
    )
    receipt, report, artifact_path = _validate_output(
        config=config,
        output_root=output_root,
    )
    full_keys, primary_keys = _primary_rows(classification_path)
    episodes = _list(
        _mapping(config.get("exact_episode_scope"), "episode scope").get("rows"),
        "episode rows",
    )
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        work_results = list(
            executor.map(
                lambda pair: _scan_work_partition(pair, episodes),
                enumerate(work_paths),
            )
        )
    require(
        sum(int(result["source_rows"]) for result in work_results)
        == work_binding["rows"],
        "work scan row closure",
    )
    work_accounting = _sum_accounting(work_results, "accounting")
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
        "independent work accounting closure",
    )
    eligible_chunks = [
        result["eligible"] for result in work_results if result["eligible"].size
    ]
    eligible = (
        np.concatenate(eligible_chunks)
        if eligible_chunks
        else np.empty(0, dtype=ELIGIBLE_DTYPE)
    )
    order = np.argsort(eligible, order=("work",), kind="stable")
    eligible = eligible[order]
    require(np.unique(eligible["work"]).size == eligible.size, "eligible work duplicate")
    require(
        eligible.size == work_accounting["eligible_prelink_work_rows"],
        "eligible work replay closure",
    )
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        event_results = list(
            executor.map(
                lambda pair: _scan_event_partition(pair, eligible),
                enumerate(event_paths),
            )
        )
    require(
        sum(int(result["source_rows"]) for result in event_results)
        == event_binding["rows"],
        "event scan row closure",
    )
    joined_event_rows = sum(int(result["joined_rows"]) for result in event_results)
    mismatch = _sum_accounting(event_results, "mismatches")
    require(all(value == 0 for value in mismatch.values()), "event metadata mismatch")
    pair_chunks = [result["pairs"] for result in event_results if result["pairs"].size]
    raw_pairs = (
        np.concatenate(pair_chunks)
        if pair_chunks
        else np.empty(0, dtype=PAIR_DTYPE)
    )
    pairs = np.unique(raw_pairs)
    linked_dtype = np.dtype([("episode", "u1"), ("work", "<u8")])
    linked = np.empty(pairs.size, dtype=linked_dtype)
    linked["episode"] = pairs["episode"]
    linked["work"] = pairs["work"]
    linked_work_rows = int(np.unique(linked).size)
    zero_token_rows = int(eligible.size - linked_work_rows)
    require(zero_token_rows >= 0, "zero-token work count")
    counts, pair_accounting = _independent_counts(full_keys, primary_keys, pairs)
    comparison = _compare_output(
        artifact_path=artifact_path,
        full_keys=full_keys,
        counts=counts,
    )
    totals = comparison["totals"]
    require(
        totals["qualifying_author_work_pairs"]
        == pair_accounting["registered_primary_distinct_author_work_pairs"],
        "qualifying pair closure",
    )
    report_work = _mapping(report.get("work_accounting"), "report work accounting")
    for key, value in work_accounting.items():
        require(report_work.get(key) == value, f"report work accounting {key}")
    report_event = _mapping(report.get("event_accounting"), "report event accounting")
    event_expected = {
        "eligible_work_joined_event_rows": joined_event_rows,
        **mismatch,
        "valid_event_rows": joined_event_rows,
        "distinct_author_work_pairs": int(pairs.size),
        "duplicate_author_work_event_excess_rows": int(raw_pairs.size - pairs.size),
        "linked_work_rows": linked_work_rows,
        "zero_valid_author_token_work_rows": zero_token_rows,
        **pair_accounting,
    }
    for key, value in event_expected.items():
        require(report_event.get(key) == value, f"report event accounting {key}")
    report_outcome = _mapping(report.get("outcome_accounting"), "report outcome")
    require(report_outcome.get("by_episode") == comparison["by_episode"], "report episode outcome")
    require(report_outcome.get("totals") == totals, "report outcome totals")
    require(receipt.get("qualifying_author_work_pairs") == totals["qualifying_author_work_pairs"], "receipt pair total")
    require(receipt.get("matured_persistence_rows") == totals["matured_persistence_rows"], "receipt persistent rows")
    require(receipt.get("matured_no_persistence_rows") == totals["matured_no_persistence_rows"], "receipt nonpersistent rows")
    require(receipt.get("right_censored_rows") == 0, "receipt censor rows")

    require(_stat_tuple(classification_path) == classification_stat, "classification changed")
    require(
        [_stat_tuple(path) for path in work_paths] == work_stats,
        "work input changed",
    )
    require(
        [_stat_tuple(path) for path in event_paths] == event_stats,
        "event input changed",
    )
    result = {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "config_sha256": _sha256(config_path),
        "output_root_basename": output_root.name,
        "output_receipt_sha256": _sha256(
            output_root / str(config["output_contract"]["receipt_basename"])
        ),
        "output_artifact_sha256": _sha256(artifact_path),
        "repository_bindings": repository,
        "source_bindings": {
            "production_classification": classification_binding,
            "cutoff_work_metadata": work_binding,
            "author_work_events": event_binding,
        },
        "source_replay": {
            "source_classification_rows": CLASSIFICATION_ROWS,
            "primary_pivot_rows": OUTPUT_ROWS,
            "source_work_rows": work_binding["rows"],
            "work_partitions_rehashed": work_binding["partitions_rehashed"],
            "source_author_event_rows": event_binding["rows"],
            "author_event_partitions_rehashed": event_binding[
                "partitions_rehashed"
            ],
            "work_accounting": work_accounting,
            "event_accounting": event_expected,
        },
        "outcome_replay": {
            **comparison,
            "rows": OUTPUT_ROWS,
            "denominator_rows_filtered": 0,
            "denominator_rows_mutated": 0,
            "post_entry_offsets_read": [1, 2, 3],
            "confirmatory_entry_year_rows_read": 0,
        },
        "verifier": {
            "strategy": "PYARROW_NUMPY_FULL_SOURCE_AND_ALL_OUTPUT_ROW_REPLAY",
            "verification_engine": "PYARROW_NUMPY",
            "producer_engine": "DUCKDB",
            "imports_producer_module": False,
            "uses_producer_engine": False,
            "elapsed_seconds": round(time.monotonic() - started, 5),
            "verifier_relative_path": (
                Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
            ),
            "verifier_sha256": _sha256(Path(__file__).resolve()),
        },
        "scope": {
            "production_classification_rows_read": CLASSIFICATION_ROWS,
            "production_post_entry_offsets_read": [1, 2, 3],
            "confirmatory_entry_year_rows_read": 0,
            "impact_rows_read": 0,
            "citation_rows_read": 0,
            "peer_rows_read": 0,
            "api_requests_created": 0,
            "association_or_effect_computed": False,
            "g0_measurement_freeze_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "authorization": {
            "production_persistence_source_independently_verified": True,
            "outcome_analysis_authorized_by_verifier_alone": False,
            "final_candidate_eligibility_authorized": False,
            "g0_measurement_freeze_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "successor_gate_may_decide_next_d0_scope": True,
        },
        "claim_boundary": (
            "This independent PyArrow and NumPy replay rehashes every bound "
            "work and author-event partition, reconstructs the exact 4,301 "
            "observable-history NEW_TOPIC_ENTRY denominator, and recomputes "
            "every t+1 through t+3 current-snapshot-annotated persistence "
            "count, state, and boolean. It establishes no true researcher-"
            "topic persistence, association, causal mechanism, scientific "
            "effect, final eligibility, G0, model, or G4 claim."
        ),
    }
    _write_json_new(verification_output, result)
    os.chmod(verification_output, 0o600)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--operator-base", type=Path, default=Path("/ssd"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(
        config_path=args.config,
        operator_base=args.operator_base,
        output_root=args.output_root,
        verification_output=args.verification_output,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
