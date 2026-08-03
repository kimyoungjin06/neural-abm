#!/usr/bin/env python3
"""Rebuild D0 attention candidates under the frozen author-linked work ABI."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import verify_pivot_feedback_d0_measurement_abi_v1 as abi_verifier


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_measurement_abi_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_d0_measurement_abi.v1"
REPORT_SCHEMA = "pivot_feedback_d0_measurement_abi_reselection_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0_measurement_abi_reselection_outputs.v1"
TOPIC_YEAR_SCHEMA = "pivot_feedback_d0_measurement_abi_topic_year_attention.v1"
CANDIDATE_SCHEMA = "pivot_feedback_d0_measurement_abi_episode_candidates.v1"
PROFILE_SCHEMA = "pivot_feedback_d0_measurement_abi_work_type_year_profile.v1"
SELECTED_WORK_SCHEMA = "pivot_feedback_d0_measurement_abi_selected_episode_work.v1"
STATUS = "PASS_D0_OUTCOME_BLIND_MEASUREMENT_ABI_TWO_FIELD_FOUR_EPISODE_RESELECTION"
PARTITIONS = 4096
MAX_CONTROL_BYTES = 32 * 1024 * 1024
ID_RE = {
    "topic": re.compile(r"^https://openalex[.]org/T([1-9][0-9]*)$"),
    "field": re.compile(r"^https://openalex[.]org/fields/([1-9][0-9]*)$"),
    "subfield": re.compile(r"^https://openalex[.]org/subfields/([1-9][0-9]*)$"),
}
WORK_INPUT_SCHEMA = pa.schema(
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
EVENT_INPUT_SCHEMA = pa.schema(
    [
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("temporal_disposition", pa.uint8(), nullable=False),
        pa.field("declared_authors_count", pa.uint32(), nullable=False),
        pa.field("primary_field_id", pa.uint8(), nullable=False),
    ]
)
EVENT_WORK_DTYPE = np.dtype(
    [
        ("work_numeric_id", "<u8"),
        ("publication_year", "<u2"),
        ("field_id", "u1"),
        ("temporal_disposition", "u1"),
    ],
    align=False,
)
LINKED_WORK_DTYPE = np.dtype(
    [
        ("work_numeric_id", "<u8"),
        ("publication_year", "<u2"),
        ("field_id", "u1"),
        ("topic_id", "<u4"),
        ("work_type_code", "u1"),
        ("declared_authors_count", "<u2"),
        ("is_retracted_null", "u1"),
    ],
    align=False,
)


class MeasurementABIReselectionError(RuntimeError):
    """Fail-closed D0 measurement-ABI reselection error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MeasurementABIReselectionError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return list(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} too large")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MeasurementABIReselectionError(f"invalid {label}: {error}") from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} too large")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise MeasurementABIReselectionError(f"invalid {label}: {error}") from error


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in _list(receipt.get("artifacts"), "receipt artifacts"):
        item = _mapping(raw, "receipt artifact")
        relative = item.get("relative_path")
        require(isinstance(relative, str) and relative, "bad artifact path")
        require(relative not in result, "duplicate artifact path")
        result[relative] = item
    return result


def _stat_key(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _validate_partitioned_input(
    root: Path,
    binding: Mapping[str, Any],
    *,
    schema: pa.Schema,
    workers: int,
) -> tuple[list[Path], dict[str, Any], list[tuple[int, int, int, int, int]]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid partitioned root")
    require(
        root.name == binding.get("expected_root_basename"),
        "input root basename mismatch",
    )
    receipt_path = root / str(binding.get("receipt_basename"))
    require(
        _sha256(receipt_path) == binding.get("receipt_sha256"),
        "input receipt SHA mismatch",
    )
    receipt = _read_json(receipt_path, receipt_path.name)
    require(
        receipt.get("schema_version") == binding.get("receipt_schema_version"),
        "input receipt schema mismatch",
    )
    require(
        receipt.get("status") == binding.get("required_status"), "input status mismatch"
    )
    prefix = str(binding.get("artifact_prefix"))
    artifacts = _artifact_map(receipt)
    declared = [
        artifacts[f"{prefix}part-{index:04d}.parquet"] for index in range(PARTITIONS)
    ]
    require(
        len(declared) == binding.get("artifact_count") == PARTITIONS,
        "partition count mismatch",
    )
    paths = [
        (root / str(item["relative_path"])).resolve(strict=True) for item in declared
    ]
    for path in paths:
        require(path.is_relative_to(root), "partition escapes root")
        require(path.is_file() and not path.is_symlink(), "invalid partition")

    def verify(pair: tuple[int, Path, Mapping[str, Any]]) -> tuple[str, int, int, str]:
        index, path, item = pair
        require(
            item.get("relative_path") == f"{prefix}part-{index:04d}.parquet",
            "partition sequence mismatch",
        )
        require(path.stat().st_size == item.get("bytes"), "partition byte mismatch")
        digest = _sha256(path)
        require(digest == item.get("sha256"), f"partition SHA mismatch: {path.name}")
        parquet = pq.ParquetFile(path)
        require(
            parquet.schema_arrow == schema, f"partition schema mismatch: {path.name}"
        )
        return (
            str(item["relative_path"]),
            int(item["bytes"]),
            int(parquet.metadata.num_rows),
            digest,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        verified = list(
            executor.map(
                verify,
                ((index, path, declared[index]) for index, path in enumerate(paths)),
            )
        )
    observed_rows = sum(item[2] for item in verified)
    require(observed_rows == binding.get("rows"), "partition input row closure failed")
    if "artifact_bytes" in binding:
        require(
            sum(item[1] for item in verified) == binding["artifact_bytes"],
            "partition input byte closure failed",
        )
    return (
        paths,
        {
            "root_basename": root.name,
            "receipt_basename": receipt_path.name,
            "receipt_sha256": binding["receipt_sha256"],
            "artifact_prefix": prefix,
            "artifacts_rehashed": len(verified),
            "artifact_bytes": sum(item[1] for item in verified),
            "rows": observed_rows,
            "selected_tree_sha256": hashlib.sha256(
                _canonical_bytes(verified)
            ).hexdigest(),
        },
        [_stat_key(path) for path in paths],
    )


def _validate_unchanged(
    paths: Sequence[Path],
    before: Sequence[tuple[int, int, int, int, int]],
) -> None:
    require(
        [_stat_key(path) for path in paths] == list(before),
        "input partitions changed during execution",
    )


def _parse_id(value: Any, entity: str) -> int:
    require(isinstance(value, str), f"{entity} ID is not text")
    match = ID_RE[entity].fullmatch(value)
    require(match is not None, f"invalid {entity} ID")
    return int(match.group(1))


def _load_taxonomy(
    path: Path,
    binding: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    path = path.resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), "invalid taxonomy projection")
    require(path.name == binding.get("expected_basename"), "taxonomy basename mismatch")
    require(_sha256(path) == binding.get("sha256"), "taxonomy SHA mismatch")
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == binding.get("rows"), "taxonomy row mismatch")
    columns = _mapping(binding.get("required_columns"), "taxonomy columns")
    for name, expected in columns.items():
        require(name in parquet.schema_arrow.names, f"missing taxonomy column: {name}")
        require(
            str(parquet.schema_arrow.field(name).type) == expected,
            f"taxonomy type mismatch: {name}",
        )
    table = pq.read_table(path, columns=list(columns))
    topics: dict[int, dict[str, Any]] = {}
    for row in table.to_pylist():
        topic_id = _parse_id(row["id"], "topic")
        require(topic_id not in topics, "duplicate taxonomy topic")
        topics[topic_id] = {
            "topic_name": str(row["display_name"]),
            "field_id": _parse_id(row["field__id"], "field"),
            "field_name": str(row["field__display_name"]),
            "subfield_id": _parse_id(row["subfield__id"], "subfield"),
            "subfield_name": str(row["subfield__display_name"]),
        }
    require(len(topics) == binding.get("rows"), "taxonomy unique closure failed")
    return topics, {
        "basename": path.name,
        "sha256": binding["sha256"],
        "rows": len(topics),
        "role": binding["role"],
    }


def _scan_event_partition(
    pair: tuple[int, Path],
    *,
    selected_fields: np.ndarray,
    allowed_dispositions: np.ndarray,
    latest_year: int,
) -> dict[str, Any]:
    index, path = pair
    table = pq.read_table(path)
    author = table["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    work = table["work_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    year = table["publication_year"].combine_chunks().to_numpy(zero_copy_only=False)
    disposition = (
        table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    field = table["primary_field_id"].combine_chunks().to_numpy(zero_copy_only=False)
    require(
        bool(np.all(author % PARTITIONS == index)),
        f"author partition mismatch: {index}",
    )
    selected = (
        np.isin(field, selected_fields)
        & np.isin(disposition, allowed_dispositions)
        & (year > 0)
        & (year <= latest_year)
    )
    author_selected = author[selected]
    work_selected = work[selected]
    if len(work_selected):
        order = np.lexsort((work_selected, author_selected))
        duplicate_pairs = int(
            (
                (author_selected[order][1:] == author_selected[order][:-1])
                & (work_selected[order][1:] == work_selected[order][:-1])
            ).sum()
        )
    else:
        duplicate_pairs = 0
    records = np.empty(int(selected.sum()), dtype=EVENT_WORK_DTYPE)
    records["work_numeric_id"] = work_selected
    records["publication_year"] = year[selected]
    records["field_id"] = field[selected]
    records["temporal_disposition"] = disposition[selected]
    return {
        "rows": len(table),
        "selected_event_rows": len(records),
        "duplicate_author_work_event_rows": duplicate_pairs,
        "work_records": records,
    }


def _event_work_index(
    paths: Sequence[Path],
    *,
    selected_fields: Sequence[int],
    allowed_dispositions: Sequence[int],
    latest_year: int,
    workers: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    selected_array = np.asarray(selected_fields, dtype=np.uint8)
    disposition_array = np.asarray(allowed_dispositions, dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunks = list(
            executor.map(
                lambda pair: _scan_event_partition(
                    pair,
                    selected_fields=selected_array,
                    allowed_dispositions=disposition_array,
                    latest_year=latest_year,
                ),
                enumerate(paths),
            )
        )
    records = np.concatenate(
        [item["work_records"] for item in chunks],
        dtype=EVENT_WORK_DTYPE,
    )
    unique = np.unique(records)
    work_ids = unique["work_numeric_id"]
    if len(work_ids) > 1:
        require(
            not bool(np.any(work_ids[1:] == work_ids[:-1])),
            "author events disagree on work year, field, or disposition",
        )
    accounting = {
        "event_rows_scanned": sum(item["rows"] for item in chunks),
        "selected_field_predecision_event_rows": sum(
            item["selected_event_rows"] for item in chunks
        ),
        "duplicate_author_work_event_rows": sum(
            item["duplicate_author_work_event_rows"] for item in chunks
        ),
        "distinct_valid_author_linked_work_rows": len(unique),
    }
    metadata = {
        "publication_year": unique["publication_year"],
        "field_id": unique["field_id"],
        "temporal_disposition": unique["temporal_disposition"],
    }
    return work_ids, metadata, accounting


def _counter_add(
    target: dict[tuple[int, ...], list[int]],
    key: tuple[int, ...],
    values: Sequence[int],
) -> None:
    cell = target.setdefault(key, [0] * len(values))
    for index, value in enumerate(values):
        cell[index] += int(value)


def _scan_work_partition(
    pair: tuple[int, Path],
    *,
    selected_fields: np.ndarray,
    allowed_dispositions: np.ndarray,
    included_types: np.ndarray,
    latest_year: int,
    valid_work_ids: np.ndarray,
    event_metadata: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    index, path = pair
    table = pq.read_table(path)
    work = table["work_id_numeric"].combine_chunks().to_numpy(zero_copy_only=False)
    require(
        bool(np.all(work % PARTITIONS == index)), f"work partition mismatch: {index}"
    )
    year_array = table["publication_year"].combine_chunks()
    year_valid = year_array.is_valid().to_numpy(zero_copy_only=False)
    year = (
        year_array.fill_null(0)
        .to_numpy(zero_copy_only=False)
        .astype(np.uint16, copy=False)
    )
    disposition = (
        table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    type_code = table["work_type_code"].combine_chunks().to_numpy(zero_copy_only=False)
    is_paratext = table["is_paratext"].combine_chunks().to_numpy(zero_copy_only=False)
    retracted_array = table["is_retracted"].combine_chunks()
    retracted_valid = retracted_array.is_valid().to_numpy(zero_copy_only=False)
    is_retracted = retracted_array.fill_null(False).to_numpy(zero_copy_only=False)
    team = (
        table["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    topic_array = table["primary_topic_id"].combine_chunks()
    topic_valid = topic_array.is_valid().to_numpy(zero_copy_only=False)
    topic = (
        topic_array.fill_null(0)
        .to_numpy(zero_copy_only=False)
        .astype(np.uint32, copy=False)
    )
    field_array = table["primary_field_id"].combine_chunks()
    field_valid = field_array.is_valid().to_numpy(zero_copy_only=False)
    field = (
        field_array.fill_null(0)
        .to_numpy(zero_copy_only=False)
        .astype(np.uint8, copy=False)
    )
    paratext_mismatch = (type_code == 14) != is_paratext
    scope = (
        year_valid
        & topic_valid
        & field_valid
        & np.isin(field, selected_fields)
        & np.isin(disposition, allowed_dispositions)
        & (year > 0)
        & (year <= latest_year)
        & (topic > 0)
    )
    included_type = scope & np.isin(type_code, included_types)
    paratext_true = included_type & is_paratext
    retracted_true = included_type & ~is_paratext & retracted_valid & is_retracted
    eligible = included_type & ~is_paratext & ~retracted_true
    eligible_indices = np.flatnonzero(eligible)
    eligible_work = work[eligible_indices]
    positions = np.searchsorted(valid_work_ids, eligible_work)
    in_range = positions < len(valid_work_ids)
    matched = np.zeros(len(eligible_indices), dtype=bool)
    matched[in_range] = valid_work_ids[positions[in_range]] == eligible_work[in_range]
    if bool(np.any(matched)):
        work_positions = positions[matched]
        source_indices = eligible_indices[matched]
        require(
            bool(
                np.all(
                    event_metadata["publication_year"][work_positions]
                    == year[source_indices]
                )
            ),
            f"event/work year mismatch: {index}",
        )
        require(
            bool(
                np.all(
                    event_metadata["field_id"][work_positions] == field[source_indices]
                )
            ),
            f"event/work field mismatch: {index}",
        )
        require(
            bool(
                np.all(
                    event_metadata["temporal_disposition"][work_positions]
                    == disposition[source_indices]
                )
            ),
            f"event/work disposition mismatch: {index}",
        )
    linked_indices = eligible_indices[matched]
    linked_mask = np.zeros(len(table), dtype=bool)
    linked_mask[linked_indices] = True
    retracted_null = ~retracted_valid

    topic_counts: dict[tuple[int, ...], list[int]] = {}
    if len(eligible_indices):
        keys = (
            (field[eligible_indices].astype(np.uint64) << np.uint64(48))
            | (topic[eligible_indices].astype(np.uint64) << np.uint64(16))
            | year[eligible_indices].astype(np.uint64)
        )
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        eligible_count = np.bincount(inverse)
        linked_count = np.bincount(inverse, weights=matched.astype(np.uint8))
        null_eligible_count = np.bincount(
            inverse,
            weights=retracted_null[eligible_indices].astype(np.uint8),
        )
        null_linked_count = np.bincount(
            inverse,
            weights=(matched & retracted_null[eligible_indices]).astype(np.uint8),
        )
        for cell_index, packed in enumerate(unique_keys.tolist()):
            key = (
                int(packed >> 48),
                int((packed >> 16) & 0xFFFFFFFF),
                int(packed & 0xFFFF),
            )
            _counter_add(
                topic_counts,
                key,
                (
                    int(eligible_count[cell_index]),
                    int(linked_count[cell_index]),
                    int(null_eligible_count[cell_index]),
                    int(null_linked_count[cell_index]),
                ),
            )

    profile: dict[tuple[int, ...], list[int]] = {}
    scope_indices = np.flatnonzero(scope)
    if len(scope_indices):
        profile_keys = (
            (field[scope_indices].astype(np.uint64) << np.uint64(24))
            | (year[scope_indices].astype(np.uint64) << np.uint64(8))
            | type_code[scope_indices].astype(np.uint64)
        )
        unique_profile_keys, inverse = np.unique(
            profile_keys,
            return_inverse=True,
        )
        count_arrays = [
            np.bincount(inverse),
            np.bincount(
                inverse,
                weights=included_type[scope_indices].astype(np.uint8),
            ),
            np.bincount(
                inverse,
                weights=paratext_true[scope_indices].astype(np.uint8),
            ),
            np.bincount(
                inverse,
                weights=retracted_true[scope_indices].astype(np.uint8),
            ),
            np.bincount(
                inverse,
                weights=eligible[scope_indices].astype(np.uint8),
            ),
            np.bincount(
                inverse,
                weights=linked_mask[scope_indices].astype(np.uint8),
            ),
            np.bincount(
                inverse,
                weights=(eligible & ~linked_mask)[scope_indices].astype(np.uint8),
            ),
            np.bincount(
                inverse,
                weights=(eligible & retracted_null)[scope_indices].astype(np.uint8),
            ),
            np.bincount(
                inverse,
                weights=(linked_mask & retracted_null)[scope_indices].astype(np.uint8),
            ),
        ]
        for cell_index, packed in enumerate(unique_profile_keys.tolist()):
            key = (
                int(packed >> 24),
                int((packed >> 8) & 0xFFFF),
                int(packed & 0xFF),
            )
            _counter_add(
                profile,
                key,
                tuple(int(values[cell_index]) for values in count_arrays),
            )

    linked = np.empty(len(linked_indices), dtype=LINKED_WORK_DTYPE)
    linked["work_numeric_id"] = work[linked_indices]
    linked["publication_year"] = year[linked_indices]
    linked["field_id"] = field[linked_indices]
    linked["topic_id"] = topic[linked_indices]
    linked["work_type_code"] = type_code[linked_indices]
    linked["declared_authors_count"] = team[linked_indices]
    linked["is_retracted_null"] = retracted_null[linked_indices].astype(np.uint8)
    return {
        "rows": len(table),
        "scope_rows": int(scope.sum()),
        "included_type_rows": int(included_type.sum()),
        "paratext_true_excluded_rows": int(paratext_true.sum()),
        "retracted_true_rows": int(retracted_true.sum()),
        "paratext_type_boolean_mismatch_rows": int(paratext_mismatch.sum()),
        "scope_paratext_type_boolean_mismatch_rows": int(
            (scope & paratext_mismatch).sum()
        ),
        "scope_type_paratext_boolean_false_rows": int(
            (scope & (type_code == 14) & ~is_paratext).sum()
        ),
        "scope_non_paratext_type_boolean_true_rows": int(
            (scope & (type_code != 14) & is_paratext).sum()
        ),
        "eligible_prelink_rows": len(eligible_indices),
        "linked_rows": len(linked_indices),
        "zero_valid_author_token_rows": len(eligible_indices) - len(linked_indices),
        "is_retracted_null_eligible_rows": int((eligible & retracted_null).sum()),
        "is_retracted_null_linked_rows": int((linked_mask & retracted_null).sum()),
        "topic_counts": topic_counts,
        "profile": profile,
        "linked": linked,
    }


def _merge_counter(
    chunks: Sequence[Mapping[tuple[int, ...], Sequence[int]]],
) -> dict[tuple[int, ...], list[int]]:
    result: dict[tuple[int, ...], list[int]] = {}
    for chunk in chunks:
        for key, values in chunk.items():
            _counter_add(result, key, values)
    return result


def _scan_work_universe(
    paths: Sequence[Path],
    *,
    selected_fields: Sequence[int],
    allowed_dispositions: Sequence[int],
    included_types: Sequence[int],
    latest_year: int,
    valid_work_ids: np.ndarray,
    event_metadata: Mapping[str, np.ndarray],
    workers: int,
) -> tuple[
    np.ndarray,
    dict[tuple[int, ...], list[int]],
    dict[tuple[int, ...], list[int]],
    dict[str, int],
]:
    fields = np.asarray(selected_fields, dtype=np.uint8)
    dispositions = np.asarray(allowed_dispositions, dtype=np.uint8)
    types = np.asarray(included_types, dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunks = list(
            executor.map(
                lambda pair: _scan_work_partition(
                    pair,
                    selected_fields=fields,
                    allowed_dispositions=dispositions,
                    included_types=types,
                    latest_year=latest_year,
                    valid_work_ids=valid_work_ids,
                    event_metadata=event_metadata,
                ),
                enumerate(paths),
            )
        )
    linked = np.concatenate(
        [item["linked"] for item in chunks],
        dtype=LINKED_WORK_DTYPE,
    )
    if len(linked):
        order = np.argsort(linked["work_numeric_id"], kind="stable")
        linked = linked[order]
        require(
            not bool(
                np.any(linked["work_numeric_id"][1:] == linked["work_numeric_id"][:-1])
            ),
            "linked work duplicated across cutoff partitions",
        )
    accounting_names = (
        "rows",
        "scope_rows",
        "included_type_rows",
        "paratext_true_excluded_rows",
        "retracted_true_rows",
        "paratext_type_boolean_mismatch_rows",
        "scope_paratext_type_boolean_mismatch_rows",
        "scope_type_paratext_boolean_false_rows",
        "scope_non_paratext_type_boolean_true_rows",
        "eligible_prelink_rows",
        "linked_rows",
        "zero_valid_author_token_rows",
        "is_retracted_null_eligible_rows",
        "is_retracted_null_linked_rows",
    )
    accounting = {
        name: sum(int(item[name]) for item in chunks) for name in accounting_names
    }
    return (
        linked,
        _merge_counter([item["topic_counts"] for item in chunks]),
        _merge_counter([item["profile"] for item in chunks]),
        accounting,
    )


def _topic_year_table(
    counts: Mapping[tuple[int, ...], Sequence[int]],
    *,
    taxonomy: Mapping[int, Mapping[str, Any]],
) -> pa.Table:
    rows = []
    for (field_id, topic_id, year), values in sorted(counts.items()):
        metadata = taxonomy.get(topic_id)
        require(metadata is not None, "eligible topic missing from taxonomy")
        require(
            metadata["field_id"] == field_id,
            "eligible topic field differs from taxonomy",
        )
        eligible, linked, null_eligible, null_linked = map(int, values)
        rows.append(
            {
                "field_id": field_id,
                "topic_id": topic_id,
                "topic_name": metadata["topic_name"],
                "subfield_id": metadata["subfield_id"],
                "subfield_name": metadata["subfield_name"],
                "publication_year": year,
                "eligible_prelink_works": eligible,
                "author_linked_works": linked,
                "zero_valid_author_token_works": eligible - linked,
                "is_retracted_null_eligible_works": null_eligible,
                "is_retracted_null_author_linked_works": null_linked,
            }
        )
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("topic_name", pa.string(), nullable=False),
            pa.field("subfield_id", pa.uint16(), nullable=False),
            pa.field("subfield_name", pa.string(), nullable=False),
            pa.field("publication_year", pa.uint16(), nullable=False),
            pa.field("eligible_prelink_works", pa.uint64(), nullable=False),
            pa.field("author_linked_works", pa.uint64(), nullable=False),
            pa.field("zero_valid_author_token_works", pa.uint64(), nullable=False),
            pa.field(
                "is_retracted_null_eligible_works",
                pa.uint64(),
                nullable=False,
            ),
            pa.field(
                "is_retracted_null_author_linked_works",
                pa.uint64(),
                nullable=False,
            ),
        ],
        metadata={b"schema_version": TOPIC_YEAR_SCHEMA.encode()},
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _profile_table(
    profile: Mapping[tuple[int, ...], Sequence[int]],
    *,
    type_names: Mapping[int, str],
) -> pa.Table:
    rows = []
    for (field_id, year, type_code), values in sorted(profile.items()):
        (
            scope,
            included,
            paratext,
            retracted,
            eligible,
            linked,
            unlinked,
            null_eligible,
            null_linked,
        ) = map(int, values)
        rows.append(
            {
                "field_id": field_id,
                "publication_year": year,
                "work_type_code": type_code,
                "work_type_name": type_names[type_code],
                "structural_scope_works": scope,
                "included_type_works": included,
                "is_paratext_true_excluded_works": paratext,
                "is_retracted_true_excluded_works": retracted,
                "eligible_prelink_works": eligible,
                "author_linked_works": linked,
                "zero_valid_author_token_works": unlinked,
                "is_retracted_null_eligible_works": null_eligible,
                "is_retracted_null_author_linked_works": null_linked,
            }
        )
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("publication_year", pa.uint16(), nullable=False),
            pa.field("work_type_code", pa.uint8(), nullable=False),
            pa.field("work_type_name", pa.string(), nullable=False),
            pa.field("structural_scope_works", pa.uint64(), nullable=False),
            pa.field("included_type_works", pa.uint64(), nullable=False),
            pa.field(
                "is_paratext_true_excluded_works",
                pa.uint64(),
                nullable=False,
            ),
            pa.field(
                "is_retracted_true_excluded_works",
                pa.uint64(),
                nullable=False,
            ),
            pa.field("eligible_prelink_works", pa.uint64(), nullable=False),
            pa.field("author_linked_works", pa.uint64(), nullable=False),
            pa.field("zero_valid_author_token_works", pa.uint64(), nullable=False),
            pa.field(
                "is_retracted_null_eligible_works",
                pa.uint64(),
                nullable=False,
            ),
            pa.field(
                "is_retracted_null_author_linked_works",
                pa.uint64(),
                nullable=False,
            ),
        ],
        metadata={b"schema_version": PROFILE_SCHEMA.encode()},
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[Fraction, int, int, int, int]:
    return (
        Fraction(
            int(candidate["standardized_gain_numerator"]),
            int(candidate["standardized_gain_denominator"]),
        ),
        int(candidate["minimum_baseline_recent_count"]),
        int(candidate["absolute_gain"]),
        -int(candidate["topic_id"]),
        -int(candidate["entry_year"]),
    )


def _episode_candidates(
    topic_year: pa.Table,
    *,
    selected_fields: Sequence[int],
    contract: Mapping[str, Any],
) -> tuple[pa.Table, list[dict[str, Any]], dict[str, Any]]:
    baseline_offsets = [int(value) for value in contract["baseline_offsets"]]
    recent_offsets = [int(value) for value in contract["recent_offsets"]]
    entry_year_end = int(contract["entry_year_end_inclusive"])
    minimum_separation = int(
        _mapping(contract["pair_selection"], "pair selection")[
            "minimum_entry_year_separation"
        ]
    )
    require(baseline_offsets == [-6, -5, -4], "baseline offsets mismatch")
    require(recent_offsets == [-3, -2, -1], "recent offsets mismatch")
    by_topic: dict[tuple[int, int], dict[str, Any]] = {}
    for row in topic_year.to_pylist():
        if int(row["author_linked_works"]) <= 0:
            continue
        key = (int(row["field_id"]), int(row["topic_id"]))
        item = by_topic.setdefault(
            key,
            {
                "topic_name": row["topic_name"],
                "subfield_id": int(row["subfield_id"]),
                "subfield_name": row["subfield_name"],
                "linked": {},
                "eligible": {},
            },
        )
        year = int(row["publication_year"])
        item["linked"][year] = int(row["author_linked_works"])
        item["eligible"][year] = int(row["eligible_prelink_works"])

    candidates_by_field: dict[int, list[dict[str, Any]]] = {}
    for (field_id, topic_id), item in by_topic.items():
        linked_years = item["linked"]
        eligible_years = item["eligible"]
        earliest_entry = min(linked_years) + 6
        for entry_year in range(earliest_entry, entry_year_end + 1):
            baseline = sum(
                linked_years.get(entry_year + offset, 0) for offset in baseline_offsets
            )
            recent = sum(
                linked_years.get(entry_year + offset, 0) for offset in recent_offsets
            )
            if baseline <= 0 or recent <= baseline:
                continue
            baseline_eligible = sum(
                eligible_years.get(entry_year + offset, 0)
                for offset in baseline_offsets
            )
            recent_eligible = sum(
                eligible_years.get(entry_year + offset, 0) for offset in recent_offsets
            )
            delta = recent - baseline
            numerator = delta * delta
            denominator = baseline + 1
            require(numerator <= np.iinfo(np.uint64).max, "score numerator overflow")
            candidates_by_field.setdefault(field_id, []).append(
                {
                    "field_id": field_id,
                    "topic_id": topic_id,
                    "topic_name": item["topic_name"],
                    "subfield_id": item["subfield_id"],
                    "subfield_name": item["subfield_name"],
                    "entry_year": entry_year,
                    "baseline_start_year": entry_year - 6,
                    "baseline_end_year": entry_year - 4,
                    "recent_start_year": entry_year - 3,
                    "recent_end_year": entry_year - 1,
                    "reserved_response_end_year": entry_year + 2,
                    "baseline_eligible_prelink_works": baseline_eligible,
                    "baseline_author_linked_works": baseline,
                    "recent_eligible_prelink_works": recent_eligible,
                    "recent_author_linked_works": recent,
                    "baseline_link_rate": baseline / baseline_eligible,
                    "recent_link_rate": recent / recent_eligible,
                    "absolute_gain": delta,
                    "minimum_baseline_recent_count": min(baseline, recent),
                    "standardized_gain_numerator": numerator,
                    "standardized_gain_denominator": denominator,
                    "standardized_gain": delta / math.sqrt(denominator),
                    "field_candidate_rank": 0,
                    "episode_selected": False,
                    "episode_order": None,
                }
            )

    selected: list[dict[str, Any]] = []
    for field_id in selected_fields:
        candidates = candidates_by_field.get(field_id, [])
        require(candidates, f"no positive-attention candidate for field {field_id}")
        candidates.sort(key=_candidate_key, reverse=True)
        for rank, candidate in enumerate(candidates, start=1):
            candidate["field_candidate_rank"] = rank
        first = candidates[0]
        compatible = [
            candidate
            for candidate in candidates[1:]
            if int(candidate["subfield_id"]) != int(first["subfield_id"])
            and abs(int(candidate["entry_year"]) - int(first["entry_year"]))
            >= minimum_separation
        ]
        require(compatible, f"no independent episode pair for field {field_id}")
        second = compatible[0]
        for order, candidate in enumerate((first, second), start=1):
            candidate["episode_selected"] = True
            candidate["episode_order"] = order
            selected.append(candidate)

    rows = [
        candidate
        for field_id in selected_fields
        for candidate in candidates_by_field[field_id]
    ]
    rows.sort(
        key=lambda row: (
            selected_fields.index(int(row["field_id"])),
            int(row["field_candidate_rank"]),
        )
    )
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("topic_name", pa.string(), nullable=False),
            pa.field("subfield_id", pa.uint16(), nullable=False),
            pa.field("subfield_name", pa.string(), nullable=False),
            pa.field("entry_year", pa.uint16(), nullable=False),
            pa.field("baseline_start_year", pa.uint16(), nullable=False),
            pa.field("baseline_end_year", pa.uint16(), nullable=False),
            pa.field("recent_start_year", pa.uint16(), nullable=False),
            pa.field("recent_end_year", pa.uint16(), nullable=False),
            pa.field("reserved_response_end_year", pa.uint16(), nullable=False),
            pa.field(
                "baseline_eligible_prelink_works",
                pa.uint64(),
                nullable=False,
            ),
            pa.field(
                "baseline_author_linked_works",
                pa.uint64(),
                nullable=False,
            ),
            pa.field("recent_eligible_prelink_works", pa.uint64(), nullable=False),
            pa.field("recent_author_linked_works", pa.uint64(), nullable=False),
            pa.field("baseline_link_rate", pa.float64(), nullable=False),
            pa.field("recent_link_rate", pa.float64(), nullable=False),
            pa.field("absolute_gain", pa.uint64(), nullable=False),
            pa.field(
                "minimum_baseline_recent_count",
                pa.uint64(),
                nullable=False,
            ),
            pa.field("standardized_gain_numerator", pa.uint64(), nullable=False),
            pa.field("standardized_gain_denominator", pa.uint64(), nullable=False),
            pa.field("standardized_gain", pa.float64(), nullable=False),
            pa.field("field_candidate_rank", pa.uint32(), nullable=False),
            pa.field("episode_selected", pa.bool_(), nullable=False),
            pa.field("episode_order", pa.uint8(), nullable=True),
        ],
        metadata={b"schema_version": CANDIDATE_SCHEMA.encode()},
    )
    selected.sort(
        key=lambda row: (
            selected_fields.index(int(row["field_id"])),
            int(row["episode_order"]),
        )
    )
    return (
        pa.Table.from_pylist(rows, schema=schema),
        selected,
        {
            "candidate_rows": len(rows),
            "candidate_rows_by_field": {
                str(field_id): len(candidates_by_field[field_id])
                for field_id in selected_fields
            },
            "selected_episode_count": len(selected),
            "minimum_entry_year_separation": minimum_separation,
        },
    )


def _selected_work_table(
    linked: np.ndarray,
    selected: Sequence[Mapping[str, Any]],
) -> tuple[pa.Table, list[dict[str, int]]]:
    chunks: list[pa.Table] = []
    accounting: list[dict[str, int]] = []
    for episode_index, episode in enumerate(selected, start=1):
        entry_year = int(episode["entry_year"])
        mask = (
            (linked["field_id"] == int(episode["field_id"]))
            & (linked["topic_id"] == int(episode["topic_id"]))
            & (linked["publication_year"] <= entry_year - 1)
        )
        values = linked[mask]
        year = values["publication_year"]
        window = np.where(
            year <= entry_year - 7,
            0,
            np.where(year <= entry_year - 4, 1, 2),
        ).astype(np.uint8)
        chunks.append(
            pa.Table.from_arrays(
                [
                    pa.array(np.full(len(values), episode_index), type=pa.uint8()),
                    pa.array(values["field_id"], type=pa.uint8()),
                    pa.array(values["topic_id"], type=pa.uint32()),
                    pa.array(np.full(len(values), entry_year), type=pa.uint16()),
                    pa.array(values["work_numeric_id"], type=pa.uint64()),
                    pa.array(values["publication_year"], type=pa.uint16()),
                    pa.array(values["work_type_code"], type=pa.uint8()),
                    pa.array(values["declared_authors_count"], type=pa.uint16()),
                    pa.array(values["is_retracted_null"].astype(bool), type=pa.bool_()),
                    pa.array(window, type=pa.uint8()),
                ],
                names=[
                    "episode_index",
                    "field_id",
                    "topic_id",
                    "entry_year",
                    "work_numeric_id",
                    "publication_year",
                    "work_type_code",
                    "declared_authors_count",
                    "is_retracted_null",
                    "window_code",
                ],
            )
        )
        accounting.append(
            {
                "episode_index": episode_index,
                "field_id": int(episode["field_id"]),
                "topic_id": int(episode["topic_id"]),
                "entry_year": entry_year,
                "pre_signal_work_rows": int((window == 0).sum()),
                "baseline_work_rows": int((window == 1).sum()),
                "signal_work_rows": int((window == 2).sum()),
                "selected_predecision_work_rows": len(values),
            }
        )
    table = pa.concat_tables(chunks)
    table = table.replace_schema_metadata(
        {b"schema_version": SELECTED_WORK_SCHEMA.encode()}
    )
    if len(table):
        episode = table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False)
        work = table["work_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
        order = np.lexsort((work, episode))
        require(
            not bool(
                np.any(
                    (episode[order][1:] == episode[order][:-1])
                    & (work[order][1:] == work[order][:-1])
                )
            ),
            "selected episode-work rows duplicated",
        )
    return table, accounting


def _write_parquet(table: pa.Table, path: Path) -> dict[str, Any]:
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
    return {
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# D0 measurement-ABI episode reselection",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Frozen ABI",
        "",
        f"- ABI SHA-256: `{report['abi']['sha256']}`",
        "- Primary attention requires an eligible scholarly-communication work",
        "  and at least one valid current-snapshot OpenAlex author token.",
        "- Entry, persistence, realized citation, prediction, and G4 outcomes",
        "  were not read.",
        "",
        "## Selected episodes",
        "",
        "| Field | Topic | Entry year | Baseline linked | Signal linked | Rank |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["episode_reselection"]["selected_episodes"]:
        lines.append(
            f"| {row['field_id']} | {row['topic_id']} | {row['entry_year']} | "
            f"{row['baseline_author_linked_works']:,} | "
            f"{row['recent_author_linked_works']:,} | "
            f"{row['field_candidate_rank']:,} |"
        )
    lines.extend(
        [
            "",
            "Content-coherence and indexing-discontinuity review is still",
            "required before leave-out or network construction.",
            "",
        ]
    )
    return "\n".join(lines)


def _cleanup_temporary_directory(
    temporary: Path | None,
    *,
    output: Path,
) -> None:
    if temporary is None:
        return
    require(temporary.is_absolute(), "refusing to remove a relative temporary path")
    require(
        temporary.parent == output.parent,
        "refusing to remove a temporary path outside the output parent",
    )
    require(
        temporary.name.startswith(f".{output.name}."),
        "refusing to remove a path without the expected temporary prefix",
    )
    if temporary.exists():
        shutil.rmtree(temporary)


def build(
    *,
    config_path: Path,
    cutoff_root: Path,
    author_history_root: Path,
    taxonomy_topics: Path,
    output: Path,
    workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    require(workers >= 1, "workers must be positive")
    output = output.resolve()
    require(not output.exists(), "output directory already exists")
    config_path = config_path.resolve(strict=True)
    config = _read_yaml(config_path, config_path.name)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    abi_validation = abi_verifier.validate(config_path)
    require(
        abi_validation["status"] == config["output_contract"]["abi_validation_status"],
        "ABI validation status differs",
    )

    inputs = _mapping(config.get("normalized_inputs"), "normalized inputs")
    work_paths, work_binding, work_stats = _validate_partitioned_input(
        cutoff_root,
        _mapping(inputs["cutoff_work_metadata"], "cutoff binding"),
        schema=WORK_INPUT_SCHEMA,
        workers=workers,
    )
    event_paths, event_binding, event_stats = _validate_partitioned_input(
        author_history_root,
        _mapping(inputs["author_history_events"], "event binding"),
        schema=EVENT_INPUT_SCHEMA,
        workers=workers,
    )
    taxonomy, taxonomy_binding = _load_taxonomy(
        taxonomy_topics,
        _mapping(inputs["topic_taxonomy_projection"], "taxonomy binding"),
    )
    measurement = _mapping(config.get("measurement_abi"), "measurement ABI")
    selection = _mapping(config.get("episode_reselection"), "episode selection")
    selected_fields = [int(value) for value in measurement["selected_field_ids"]]
    allowed_dispositions = [
        int(value) for value in measurement["allowed_temporal_dispositions"]
    ]
    latest_year = int(measurement["latest_attention_year"])
    type_names = {
        int(code): str(name)
        for code, name in _mapping(
            config.get("work_type_dictionary"),
            "work type dictionary",
        ).items()
    }
    name_to_code = {name: code for code, name in type_names.items()}
    included_types = sorted(
        name_to_code[name]
        for name in _mapping(
            config["work_type_eligibility"]["included"],
            "included work types",
        )
    )

    valid_work_ids, event_metadata, event_accounting = _event_work_index(
        event_paths,
        selected_fields=selected_fields,
        allowed_dispositions=allowed_dispositions,
        latest_year=latest_year,
        workers=workers,
    )
    linked, topic_counts, profile_counts, work_accounting = _scan_work_universe(
        work_paths,
        selected_fields=selected_fields,
        allowed_dispositions=allowed_dispositions,
        included_types=included_types,
        latest_year=latest_year,
        valid_work_ids=valid_work_ids,
        event_metadata=event_metadata,
        workers=workers,
    )
    require(
        event_accounting["event_rows_scanned"]
        == inputs["author_history_events"]["rows"],
        "event scan row closure failed",
    )
    require(
        work_accounting["rows"] == inputs["cutoff_work_metadata"]["rows"],
        "work scan row closure failed",
    )
    require(
        work_accounting["linked_rows"] == len(linked),
        "linked work row closure failed",
    )
    crosswalk = _mapping(
        measurement["paratext_type_boolean_crosswalk_audit"],
        "paratext crosswalk audit",
    )
    crosswalk_matches = {
        "full_cutoff_rows_mismatched": work_accounting[
            "paratext_type_boolean_mismatch_rows"
        ],
        "selected_field_predecision_topic_rows_mismatched": work_accounting[
            "scope_paratext_type_boolean_mismatch_rows"
        ],
        "selected_scope_type_paratext_boolean_false_rows": work_accounting[
            "scope_type_paratext_boolean_false_rows"
        ],
        "selected_scope_non_paratext_type_boolean_true_rows": work_accounting[
            "scope_non_paratext_type_boolean_true_rows"
        ],
    }
    for key, observed in crosswalk_matches.items():
        require(observed == crosswalk[key], f"paratext crosswalk differs: {key}")
    topic_year = _topic_year_table(topic_counts, taxonomy=taxonomy)
    profile = _profile_table(profile_counts, type_names=type_names)
    candidate_table, selected, candidate_accounting = _episode_candidates(
        topic_year,
        selected_fields=selected_fields,
        contract=selection,
    )
    require(len(selected) == 4, "two-field/four-episode coverage failed")
    selected_work, selected_work_accounting = _selected_work_table(linked, selected)

    _validate_unchanged(work_paths, work_stats)
    _validate_unchanged(event_paths, event_stats)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    os.chmod(temporary, 0o700)
    try:
        artifacts = [
            _write_parquet(
                topic_year,
                temporary / "author_linked_topic_year_attention.parquet",
            ),
            _write_parquet(
                candidate_table,
                temporary / "episode_candidates.parquet",
            ),
            _write_parquet(
                selected_work,
                temporary / "selected_episode_predecision_works.parquet",
            ),
            _write_parquet(
                profile,
                temporary / "work_type_year_missingness.parquet",
            ),
        ]
        selected_report = []
        for row in selected:
            selected_report.append(
                {
                    key: row[key]
                    for key in (
                        "field_id",
                        "topic_id",
                        "topic_name",
                        "subfield_id",
                        "subfield_name",
                        "entry_year",
                        "baseline_start_year",
                        "baseline_end_year",
                        "recent_start_year",
                        "recent_end_year",
                        "reserved_response_end_year",
                        "baseline_eligible_prelink_works",
                        "baseline_author_linked_works",
                        "recent_eligible_prelink_works",
                        "recent_author_linked_works",
                        "baseline_link_rate",
                        "recent_link_rate",
                        "absolute_gain",
                        "standardized_gain_numerator",
                        "standardized_gain_denominator",
                        "standardized_gain",
                        "field_candidate_rank",
                        "episode_order",
                    )
                }
            )
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA,
            "status": STATUS,
            "scope": {
                "full_cutoff_work_partitions": PARTITIONS,
                "full_author_event_partitions": PARTITIONS,
                "selected_field_ids": selected_fields,
                "latest_attention_year": latest_year,
                "outcomes_read": False,
                "entry_or_persistence_rows_read": False,
                "realized_citation_rows_read": False,
                "historical_episode_ids_used_as_selection_inputs": False,
                "episode_reselection_complete": True,
                "content_and_indexing_audit_complete": False,
                "leaveout_network_bridge_authorized": False,
                "d0_complete": False,
                "g0_authorized": False,
                "model_fitting_performed": False,
                "g4_rows_read": False,
                "gpu_used": False,
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "abi": {
                "relative_path": config_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(config_path),
                "validation_status": abi_validation["status"],
                "included_work_type_codes": included_types,
                "included_work_type_names": [
                    type_names[code] for code in included_types
                ],
                "attention_unit": measurement["attention_unit"],
                "paratext_rule": measurement["paratext_rule"],
                "retraction_rule": measurement["retraction_rule"],
                "is_retracted_null_rule": measurement["is_retracted_null_rule"],
                "valid_author_linkage_rule": measurement["valid_author_linkage_rule"],
                "zero_valid_author_token_rule": measurement[
                    "zero_valid_author_token_rule"
                ],
            },
            "inputs": {
                "cutoff_work_metadata": work_binding,
                "author_history_events": event_binding,
                "topic_taxonomy_projection": taxonomy_binding,
            },
            "author_event_accounting": event_accounting,
            "work_abi_accounting": work_accounting,
            "attention_accounting": {
                "topic_year_rows": len(topic_year),
                "distinct_field_topic_pairs": len(
                    {
                        (int(row["field_id"]), int(row["topic_id"]))
                        for row in topic_year.to_pylist()
                    }
                ),
                "work_type_year_profile_rows": len(profile),
            },
            "episode_reselection": {
                "baseline_offsets": selection["baseline_offsets"],
                "recent_offsets": selection["recent_offsets"],
                "response_offsets_reserved_for_independence_only": selection[
                    "response_offsets_reserved_for_independence_only"
                ],
                "pair_selection": selection["pair_selection"],
                **candidate_accounting,
                "selected_episodes": selected_report,
                "selected_episode_work_accounting": selected_work_accounting,
                "selected_episode_predecision_work_rows": len(selected_work),
            },
            "authorization": {
                "content_and_indexing_audit": "REQUIRED_NEXT",
                "leaveout_network_bridge_build": "NOT_AUTHORIZED",
                "discovery_outcome_opening": "NOT_AUTHORIZED",
                "g0_measurement_freeze": "NOT_AUTHORIZED",
                "model_fitting": "NOT_AUTHORIZED",
                "g4_open": "NOT_AUTHORIZED",
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "runtime": {
                "builder_relative_path": Path(__file__)
                .resolve()
                .relative_to(REPO_ROOT)
                .as_posix(),
                "builder_sha256": _sha256(Path(__file__).resolve()),
                "workers": workers,
                "elapsed_seconds": time.monotonic() - started,
            },
        }
        report_path = temporary / "measurement_abi_reselection.json"
        _write_json(report_path, report)
        artifacts.append(
            {
                "relative_path": report_path.name,
                "bytes": report_path.stat().st_size,
                "sha256": _sha256(report_path),
            }
        )
        markdown_path = temporary / "measurement_abi_reselection.md"
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        os.chmod(markdown_path, 0o600)
        artifacts.append(
            {
                "relative_path": markdown_path.name,
                "bytes": markdown_path.stat().st_size,
                "sha256": _sha256(markdown_path),
            }
        )
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
            "abi_sha256": report["abi"]["sha256"],
            "selected_field_ids": selected_fields,
            "selected_episode_count": len(selected),
            "selected_episode_predecision_work_rows": len(selected_work),
            "outcomes_read": False,
            "content_and_indexing_audit_complete": False,
            "leaveout_network_bridge_authorized": False,
            "d0_complete": False,
            "g0_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "gpu_used": False,
        }
        _write_json(
            temporary / "measurement_abi_reselection.outputs.json",
            receipt,
        )
        os.replace(temporary, output)
        temporary = None
        return report
    finally:
        _cleanup_temporary_directory(temporary, output=output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--taxonomy-topics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build(
        config_path=args.config,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        taxonomy_topics=args.taxonomy_topics,
        output=args.output,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_episodes": [
                    {
                        "field_id": row["field_id"],
                        "topic_id": row["topic_id"],
                        "entry_year": row["entry_year"],
                        "field_candidate_rank": row["field_candidate_rank"],
                    }
                    for row in report["episode_reselection"]["selected_episodes"]
                ],
                "work_abi_accounting": report["work_abi_accounting"],
                "runtime": report["runtime"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
