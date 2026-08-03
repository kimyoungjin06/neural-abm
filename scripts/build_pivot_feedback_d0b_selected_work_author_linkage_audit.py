#!/usr/bin/env python3
"""Audit selected D0-B work linkage to valid author-token events."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "pivot_feedback_d0b_selected_work_author_linkage_audit_config.v1"
REPORT_SCHEMA = "pivot_feedback_d0b_selected_work_author_linkage_audit_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0b_selected_work_author_linkage_audit_outputs.v1"
WORK_LINKAGE_SCHEMA_VERSION = "pivot_feedback_d0b_selected_work_author_linkage.v1"
INCIDENCE_SCHEMA_VERSION = (
    "pivot_feedback_d0b_selected_work_valid_author_incidence.v1"
)
PROFILE_SCHEMA_VERSION = "pivot_feedback_d0b_selected_work_author_linkage_profile.v1"
STATUS = (
    "PASS_D0B_OUTCOME_BLIND_SELECTED_WORK_AUTHOR_LINKAGE_AUDIT_"
    "NO_EPISODE_DECISION"
)
PARTITIONS = 4096
MAX_CONTROL_BYTES = 32 * 1024 * 1024
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
SELECTED_WORK_SCHEMA = pa.schema(
    [
        pa.field("episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("declared_authors_count", pa.uint16(), nullable=False),
        pa.field("incumbency_qualifying_work", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": b"pivot_feedback_d0b_selected_predecision_work.v1"},
)
WORK_LINKAGE_SCHEMA = pa.schema(
    [
        pa.field("episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("work_type_code", pa.uint8(), nullable=False),
        pa.field("work_type_name", pa.string(), nullable=False),
        pa.field("is_paratext", pa.bool_(), nullable=False),
        pa.field("is_retracted", pa.bool_(), nullable=True),
        pa.field("declared_authors_count", pa.uint16(), nullable=False),
        pa.field("valid_distinct_author_token_count", pa.uint32(), nullable=False),
        pa.field("declared_minus_valid_distinct_token_count", pa.uint32(), nullable=False),
        pa.field("has_valid_author_token", pa.bool_(), nullable=False),
        pa.field("pre_signal_incumbency_qualifying_work", pa.bool_(), nullable=False),
        pa.field("incumbent_valid_author_token_count", pa.uint32(), nullable=False),
        pa.field("has_pre_signal_incumbent_author", pa.bool_(), nullable=False),
        pa.field("all_linked_authors_pre_signal_incumbents", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": WORK_LINKAGE_SCHEMA_VERSION.encode()},
)
INCIDENCE_SCHEMA = pa.schema(
    [
        pa.field("episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("declared_authors_count", pa.uint16(), nullable=False),
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("pre_signal_incumbent_author", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": INCIDENCE_SCHEMA_VERSION.encode()},
)
PROFILE_SCHEMA = pa.schema(
    [
        pa.field("episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("aggregation_level", pa.string(), nullable=False),
        pa.field("work_type_code", pa.uint8(), nullable=True),
        pa.field("work_type_name", pa.string(), nullable=False),
        pa.field("team_cap", pa.string(), nullable=False),
        pa.field("calendar_window", pa.string(), nullable=False),
        pa.field("selected_work_rows", pa.uint64(), nullable=False),
        pa.field("linked_work_rows", pa.uint64(), nullable=False),
        pa.field("zero_valid_author_token_work_rows", pa.uint64(), nullable=False),
        pa.field("declared_author_slots", pa.uint64(), nullable=False),
        pa.field("valid_distinct_author_token_rows", pa.uint64(), nullable=False),
        pa.field("declared_minus_valid_distinct_token_rows", pa.uint64(), nullable=False),
        pa.field("any_pre_signal_incumbent_work_rows", pa.uint64(), nullable=False),
        pa.field("all_linked_authors_incumbent_work_rows", pa.uint64(), nullable=False),
    ],
    metadata={b"schema_version": PROFILE_SCHEMA_VERSION.encode()},
)


class SelectedWorkAuthorLinkageError(RuntimeError):
    """Fail-closed D0-B linkage-audit error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectedWorkAuthorLinkageError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return list(value)


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        require(
            isinstance(current, Mapping) and part in current,
            f"missing field {dotted_path}",
        )
        current = current[part]
    return current


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
        raise SelectedWorkAuthorLinkageError(f"invalid {label}: {error}") from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} too large")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise SelectedWorkAuthorLinkageError(f"invalid {label}: {error}") from error


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (REPO_ROOT / value).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _validate_repo_binding(
    binding: Mapping[str, Any],
    *,
    yaml_document: bool,
) -> dict[str, Any]:
    path = _safe_repo_path(binding.get("relative_path"), "repository binding")
    digest = _sha256(path)
    require(digest == binding.get("sha256"), f"binding SHA mismatch: {path.name}")
    document = (
        _read_yaml(path, path.name)
        if yaml_document
        else _read_json(path, path.name)
    )
    require(
        document.get("schema_version") == binding.get("required_schema_version"),
        "binding schema mismatch",
    )
    require(
        document.get("status") == binding.get("required_status"),
        "binding status mismatch",
    )
    for dotted_path, expected in _mapping(
        binding.get("required_fields", {}),
        "binding fields",
    ).items():
        observed = _nested(document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"binding field mismatch: {dotted_path}",
        )
    return {
        "relative_path": binding["relative_path"],
        "sha256": digest,
        "status": document["status"],
    }


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for raw in _list(receipt.get("artifacts"), "receipt artifacts"):
        item = _mapping(raw, "receipt artifact")
        relative = str(item.get("relative_path"))
        require(relative not in result, "duplicate receipt artifact")
        result[relative] = item
    return result


def _validate_foundation_output(
    root: Path,
    binding: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid foundation root")
    require(root.name == binding.get("expected_root_basename"), "foundation basename mismatch")
    receipt_path = root / str(binding["receipt_basename"])
    digest = _sha256(receipt_path)
    require(digest == binding.get("receipt_sha256"), "foundation receipt SHA mismatch")
    receipt = _read_json(receipt_path, receipt_path.name)
    require(
        receipt.get("schema_version") == binding.get("receipt_schema_version"),
        "foundation receipt schema mismatch",
    )
    require(receipt.get("status") == binding.get("required_status"), "foundation status mismatch")
    require(
        receipt.get("artifact_tree_sha256") == binding.get("artifact_tree_sha256"),
        "foundation artifact tree mismatch",
    )
    artifact = _mapping(binding.get("selected_work_artifact"), "selected artifact")
    relative = str(artifact["relative_path"])
    receipt_item = _artifact_map(receipt).get(relative)
    require(receipt_item is not None, "selected artifact absent from receipt")
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), "selected artifact escapes root")
    require(path.is_file() and not path.is_symlink(), "invalid selected artifact")
    require(_sha256(path) == artifact.get("sha256"), "selected artifact SHA mismatch")
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == artifact.get("rows"), "selected row mismatch")
    require(parquet.schema_arrow == SELECTED_WORK_SCHEMA, "selected schema mismatch")
    require(receipt_item.get("sha256") == artifact.get("sha256"), "receipt selected SHA mismatch")
    return path, {
        "root_basename": root.name,
        "receipt_sha256": digest,
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "selected_work_artifact_sha256": artifact["sha256"],
        "selected_work_rows": artifact["rows"],
    }


def _validate_partitioned_input(
    root: Path,
    binding: Mapping[str, Any],
    *,
    workers: int,
) -> tuple[list[Path], dict[str, Any], list[tuple[int, int, int, int, int]]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid input root")
    require(root.name == binding.get("expected_root_basename"), "input basename mismatch")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding.get("receipt_sha256"), "input receipt SHA mismatch")
    receipt = _read_json(receipt_path, receipt_path.name)
    require(
        receipt.get("schema_version") == binding.get("receipt_schema_version"),
        "input receipt schema mismatch",
    )
    require(receipt.get("status") == binding.get("required_status"), "input status mismatch")
    prefix = str(binding["artifact_prefix"])
    items = [
        item
        for relative, item in _artifact_map(receipt).items()
        if relative.startswith(prefix)
    ]
    items.sort(key=lambda item: str(item["relative_path"]))
    require(len(items) == binding.get("artifact_count") == PARTITIONS, "partition count mismatch")
    paths = []
    stats = []
    for index, item in enumerate(items):
        relative = f"{prefix}part-{index:04d}.parquet"
        require(item.get("relative_path") == relative, "partition sequence mismatch")
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root), "partition escapes input root")
        require(path.is_file() and not path.is_symlink(), "invalid partition")
        current = path.stat()
        require(current.st_size == item.get("bytes"), "partition bytes mismatch")
        paths.append(path)
        stats.append(
            (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
        )

    def rehash(pair: tuple[Path, Mapping[str, Any]]) -> tuple[str, int, str]:
        path, item = pair
        digest = _sha256(path)
        require(digest == item.get("sha256"), f"partition SHA mismatch: {path.name}")
        return str(item["relative_path"]), int(item["bytes"]), digest

    with ThreadPoolExecutor(max_workers=workers) as executor:
        verified = list(executor.map(rehash, zip(paths, items, strict=True)))
    expected_bytes = binding.get("artifact_bytes")
    if expected_bytes is not None:
        require(sum(row[1] for row in verified) == expected_bytes, "input byte total mismatch")
    return paths, {
        "root_basename": root.name,
        "receipt_sha256": binding["receipt_sha256"],
        "artifact_prefix": prefix,
        "artifacts_rehashed": len(verified),
        "artifact_bytes": sum(row[1] for row in verified),
        "selected_tree_sha256": hashlib.sha256(_canonical_bytes(verified)).hexdigest(),
    }, stats


def _validate_unchanged(
    paths: Sequence[Path],
    stats: Sequence[tuple[int, int, int, int, int]],
) -> None:
    observed = []
    for path in paths:
        current = path.stat()
        observed.append(
            (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
        )
    require(observed == list(stats), "input changed during execution")


def _episodes(config: Mapping[str, Any]) -> list[dict[str, int]]:
    rows = [
        {key: int(value) for key, value in _mapping(raw, "episode").items()}
        for raw in _list(config.get("frozen_episodes"), "frozen episodes")
    ]
    require(
        [
            (row["episode_index"], row["field_id"], row["topic_id"], row["entry_year"])
            for row in rows
        ]
        == [
            (1, 17, 14430, 2003),
            (2, 17, 10579, 2012),
            (3, 27, 11361, 2007),
            (4, 27, 12505, 1991),
        ],
        "frozen episodes differ",
    )
    return rows


def _selected_arrays(table: pa.Table) -> dict[str, np.ndarray]:
    require(table.schema == SELECTED_WORK_SCHEMA, "selected table schema mismatch")
    values = {
        name: table[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in table.schema.names
    }
    require(
        len(np.unique(values["work_numeric_id"])) == len(table),
        "selected work IDs not globally unique",
    )
    require(
        bool(
            np.all(
                (values["publication_year"] > 0)
                & (values["publication_year"] < values["entry_year"])
            )
        ),
        "selected work crosses decision boundary",
    )
    return values


def _lookup(
    selected: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    order = np.argsort(selected["work_numeric_id"])
    ids = selected["work_numeric_id"][order].astype(np.uint64, copy=False)
    require(
        len(ids) < 2 or bool(np.all(ids[1:] > ids[:-1])),
        "lookup work IDs not strictly sorted",
    )
    return ids, order, {
        key: value[order]
        for key, value in selected.items()
    }


def _positions(values: np.ndarray, sorted_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(sorted_ids, values)
    keep = positions < len(sorted_ids)
    matched = np.zeros(len(values), dtype=bool)
    matched[keep] = sorted_ids[positions[keep]] == values[keep]
    return matched, positions[matched]


def _join_metadata(
    paths: Sequence[Path],
    selected: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int]:
    sorted_ids, order, lookup = _lookup(selected)
    count = len(sorted_ids)
    work_type = np.zeros(count, dtype=np.uint8)
    paratext = np.zeros(count, dtype=bool)
    retracted = np.zeros(count, dtype=bool)
    retracted_null = np.zeros(count, dtype=bool)
    seen = np.zeros(count, dtype=bool)
    scanned = 0
    for index, path in enumerate(paths):
        require(pq.ParquetFile(path).schema_arrow == WORK_INPUT_SCHEMA, f"work schema mismatch: {index}")
        table = pq.read_table(path)
        scanned += len(table)
        work = table["work_id_numeric"].combine_chunks().to_numpy(zero_copy_only=False)
        require(bool(np.all(work % PARTITIONS == index)), f"work partition mismatch: {index}")
        matched, positions = _positions(work, sorted_ids)
        if not len(positions):
            continue
        require(not bool(np.any(seen[positions])), "duplicate selected metadata row")
        seen[positions] = True
        years = table["publication_year"].combine_chunks().fill_null(0).to_numpy(zero_copy_only=False)[matched]
        disposition = table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)[matched]
        teams = table["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)[matched]
        topics = table["primary_topic_id"].combine_chunks().fill_null(0).to_numpy(zero_copy_only=False)[matched]
        fields = table["primary_field_id"].combine_chunks().fill_null(0).to_numpy(zero_copy_only=False)[matched]
        require(bool(np.all(years == lookup["publication_year"][positions])), "selected year mismatch")
        require(bool(np.all(teams == lookup["declared_authors_count"][positions])), "selected team mismatch")
        require(bool(np.all(topics == lookup["topic_id"][positions])), "selected topic mismatch")
        require(bool(np.all(fields == lookup["field_id"][positions])), "selected field mismatch")
        require(bool(np.all(np.isin(disposition, [2, 3]))), "selected disposition mismatch")
        work_type[positions] = table["work_type_code"].combine_chunks().to_numpy(zero_copy_only=False)[matched]
        paratext[positions] = table["is_paratext"].combine_chunks().to_numpy(zero_copy_only=False)[matched]
        retract_column = table["is_retracted"].combine_chunks()
        retracted[positions] = retract_column.fill_null(False).to_numpy(zero_copy_only=False)[matched]
        retracted_null[positions] = retract_column.is_null().to_numpy(zero_copy_only=False)[matched]
    require(bool(np.all(seen)), "selected work missing from cutoff metadata")
    inverse = np.empty(count, dtype=np.int64)
    inverse[order] = np.arange(count)
    return {
        "work_type_code": work_type[inverse],
        "is_paratext": paratext[inverse],
        "is_retracted_value": retracted[inverse],
        "is_retracted_null": retracted_null[inverse],
    }, scanned


def _join_events(
    paths: Sequence[Path],
    selected: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int]:
    sorted_ids, order, lookup = _lookup(selected)
    chunks: dict[str, list[np.ndarray]] = {
        "selected_index": [],
        "author_numeric_id": [],
    }
    scanned = 0
    for index, path in enumerate(paths):
        require(pq.ParquetFile(path).schema_arrow == EVENT_INPUT_SCHEMA, f"event schema mismatch: {index}")
        table = pq.read_table(path)
        scanned += len(table)
        author = table["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
        require(bool(np.all(author % PARTITIONS == index)), f"author partition mismatch: {index}")
        work = table["work_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
        matched, positions = _positions(work, sorted_ids)
        if not len(positions):
            continue
        years = table["publication_year"].combine_chunks().to_numpy(zero_copy_only=False)[matched]
        disposition = table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)[matched]
        teams = table["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)[matched]
        fields = table["primary_field_id"].combine_chunks().to_numpy(zero_copy_only=False)[matched]
        require(bool(np.all(years == lookup["publication_year"][positions])), "event year mismatch")
        require(bool(np.all(teams == lookup["declared_authors_count"][positions])), "event team mismatch")
        require(bool(np.all(fields == lookup["field_id"][positions])), "event field mismatch")
        require(bool(np.all(np.isin(disposition, [2, 3]))), "event disposition mismatch")
        chunks["selected_index"].append(order[positions])
        chunks["author_numeric_id"].append(author[matched].astype(np.uint64, copy=False))
    incidence = {
        key: np.concatenate(value) if value else np.empty(0, dtype=np.uint64)
        for key, value in chunks.items()
    }
    incidence["selected_index"] = incidence["selected_index"].astype(np.int64, copy=False)
    order = np.lexsort(
        (
            incidence["author_numeric_id"],
            selected["work_numeric_id"][incidence["selected_index"]],
            selected["episode_index"][incidence["selected_index"]],
        )
    )
    incidence = {key: value[order] for key, value in incidence.items()}
    if len(order) > 1:
        work = selected["work_numeric_id"][incidence["selected_index"]]
        duplicate = (
            (work[1:] == work[:-1])
            & (
                incidence["author_numeric_id"][1:]
                == incidence["author_numeric_id"][:-1]
            )
        )
        require(not bool(np.any(duplicate)), "duplicate selected author-work event")
    return incidence, scanned


def _build_tables(
    selected_table: pa.Table,
    metadata: Mapping[str, np.ndarray],
    incidence: Mapping[str, np.ndarray],
    *,
    episodes: Sequence[Mapping[str, int]],
    type_names: Mapping[int, str],
) -> tuple[pa.Table, pa.Table, dict[str, Any]]:
    selected = _selected_arrays(selected_table)
    selected_index = incidence["selected_index"]
    valid_count = np.bincount(
        selected_index,
        minlength=len(selected_table),
    ).astype(np.uint32, copy=False)
    declared = selected["declared_authors_count"].astype(np.uint32, copy=False)
    require(bool(np.all(valid_count <= declared)), "valid token count exceeds declared count")
    incumbent_event = np.zeros(len(selected_index), dtype=bool)
    incumbent_authors_by_episode = {}
    for episode in episodes:
        episode_index = episode["episode_index"]
        event_episode = selected["episode_index"][selected_index] == episode_index
        event_year = selected["publication_year"][selected_index]
        incumbents = np.unique(
            incidence["author_numeric_id"][
                event_episode & (event_year < episode["entry_year"] - 6)
            ]
        )
        incumbent_authors_by_episode[str(episode_index)] = len(incumbents)
        incumbent_event[event_episode] = np.isin(
            incidence["author_numeric_id"][event_episode],
            incumbents,
        )
    incumbent_count = np.bincount(
        selected_index,
        weights=incumbent_event.astype(np.uint8),
        minlength=len(selected_table),
    ).astype(np.uint32, copy=False)
    has_valid = valid_count > 0
    any_incumbent = incumbent_count > 0
    all_incumbent = has_valid & (incumbent_count == valid_count)
    unlinked = declared - valid_count
    work_type_code = metadata["work_type_code"].astype(np.uint8, copy=False)
    require(set(np.unique(work_type_code)).issubset(type_names), "unknown work type code")
    retracted_values: list[bool | None] = [
        None if is_null else bool(value)
        for value, is_null in zip(
            metadata["is_retracted_value"],
            metadata["is_retracted_null"],
            strict=True,
        )
    ]
    work_linkage = pa.Table.from_arrays(
        [
            selected_table["episode_index"],
            selected_table["field_id"],
            selected_table["topic_id"],
            selected_table["entry_year"],
            selected_table["work_numeric_id"],
            selected_table["publication_year"],
            pa.array(work_type_code, type=pa.uint8()),
            pa.array([type_names[int(code)] for code in work_type_code], type=pa.string()),
            pa.array(metadata["is_paratext"], type=pa.bool_()),
            pa.array(retracted_values, type=pa.bool_()),
            selected_table["declared_authors_count"],
            pa.array(valid_count, type=pa.uint32()),
            pa.array(unlinked, type=pa.uint32()),
            pa.array(has_valid, type=pa.bool_()),
            selected_table["incumbency_qualifying_work"],
            pa.array(incumbent_count, type=pa.uint32()),
            pa.array(any_incumbent, type=pa.bool_()),
            pa.array(all_incumbent, type=pa.bool_()),
        ],
        schema=WORK_LINKAGE_SCHEMA,
    )
    incidence_table = pa.Table.from_arrays(
        [
            pa.array(selected["episode_index"][selected_index], type=pa.uint8()),
            pa.array(selected["field_id"][selected_index], type=pa.uint8()),
            pa.array(selected["topic_id"][selected_index], type=pa.uint32()),
            pa.array(selected["entry_year"][selected_index], type=pa.uint16()),
            pa.array(selected["work_numeric_id"][selected_index], type=pa.uint64()),
            pa.array(selected["publication_year"][selected_index], type=pa.uint16()),
            pa.array(selected["declared_authors_count"][selected_index], type=pa.uint16()),
            pa.array(incidence["author_numeric_id"], type=pa.uint64()),
            pa.array(incumbent_event, type=pa.bool_()),
        ],
        schema=INCIDENCE_SCHEMA,
    )
    accounting = {
        "selected_work_rows": len(work_linkage),
        "valid_author_incidence_rows": len(incidence_table),
        "linked_work_rows": int(has_valid.sum()),
        "zero_valid_author_token_work_rows": int((~has_valid).sum()),
        "declared_author_slots": int(declared.sum(dtype=np.uint64)),
        "valid_distinct_author_token_rows": int(valid_count.sum(dtype=np.uint64)),
        "declared_minus_valid_distinct_token_rows": int(unlinked.sum(dtype=np.uint64)),
        "incumbent_authors_by_episode": incumbent_authors_by_episode,
    }
    return work_linkage, incidence_table, accounting


def _profile(
    work_linkage: pa.Table,
    *,
    episodes: Sequence[Mapping[str, int]],
    type_names: Mapping[int, str],
) -> tuple[pa.Table, dict[str, Any]]:
    values = {
        name: work_linkage[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in work_linkage.schema.names
        if name not in {"work_type_name", "is_retracted"}
    }
    rows = []
    summary = {}
    for episode in episodes:
        episode_index = episode["episode_index"]
        entry = episode["entry_year"]
        episode_mask = values["episode_index"] == episode_index
        summary[str(episode_index)] = {}
        for level, code, type_name in [
            ("ALL_TYPES", None, "ALL_TYPES"),
            *[
                ("WORK_TYPE", code, name)
                for code, name in sorted(type_names.items())
            ],
        ]:
            type_mask = (
                np.ones(len(work_linkage), dtype=bool)
                if code is None
                else values["work_type_code"] == code
            )
            for cap_label, cap in (("10", 10), ("50", 50), ("100", 100), ("ALL", None)):
                cap_mask = (
                    np.ones(len(work_linkage), dtype=bool)
                    if cap is None
                    else values["declared_authors_count"] <= cap
                )
                for window, low, high in (
                    ("ALL_PREDECISION", None, entry - 1),
                    ("REFERENCE", entry - 6, entry - 4),
                    ("SIGNAL", entry - 3, entry - 1),
                ):
                    window_mask = values["publication_year"] <= high
                    if low is not None:
                        window_mask &= values["publication_year"] >= low
                    mask = episode_mask & type_mask & cap_mask & window_mask
                    selected_rows = int(mask.sum())
                    linked_rows = int((mask & values["has_valid_author_token"]).sum())
                    row = {
                        "episode_index": episode_index,
                        "field_id": episode["field_id"],
                        "topic_id": episode["topic_id"],
                        "entry_year": entry,
                        "aggregation_level": level,
                        "work_type_code": code,
                        "work_type_name": type_name,
                        "team_cap": cap_label,
                        "calendar_window": window,
                        "selected_work_rows": selected_rows,
                        "linked_work_rows": linked_rows,
                        "zero_valid_author_token_work_rows": selected_rows - linked_rows,
                        "declared_author_slots": int(
                            values["declared_authors_count"][mask].sum(dtype=np.uint64)
                        ),
                        "valid_distinct_author_token_rows": int(
                            values["valid_distinct_author_token_count"][mask].sum(dtype=np.uint64)
                        ),
                        "declared_minus_valid_distinct_token_rows": int(
                            values["declared_minus_valid_distinct_token_count"][mask].sum(dtype=np.uint64)
                        ),
                        "any_pre_signal_incumbent_work_rows": int(
                            (mask & values["has_pre_signal_incumbent_author"]).sum()
                        ),
                        "all_linked_authors_incumbent_work_rows": int(
                            (
                                mask
                                & values["all_linked_authors_pre_signal_incumbents"]
                            ).sum()
                        ),
                    }
                    rows.append(row)
                    if level == "ALL_TYPES" and cap_label == "ALL":
                        summary[str(episode_index)][window] = {
                            key: row[key]
                            for key in (
                                "selected_work_rows",
                                "linked_work_rows",
                                "zero_valid_author_token_work_rows",
                                "declared_author_slots",
                                "valid_distinct_author_token_rows",
                                "declared_minus_valid_distinct_token_rows",
                                "any_pre_signal_incumbent_work_rows",
                                "all_linked_authors_incumbent_work_rows",
                            )
                        }
    return pa.Table.from_pylist(rows, schema=PROFILE_SCHEMA), summary


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
    parquet = pq.ParquetFile(path)
    return {
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    os.chmod(path, 0o600)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# D0-B selected-work author linkage audit",
        "",
        f"Status: `{report['status']}`",
        "",
        "| Episode | Selected works | Linked works | Zero-valid-token works | Link rate | Reference any-incumbent | Signal any-incumbent |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for episode_index in range(1, 5):
        row = report["profile_summary"][str(episode_index)]
        all_rows = row["ALL_PREDECISION"]
        rate = all_rows["linked_work_rows"] / all_rows["selected_work_rows"]
        lines.append(
            f"| {episode_index} | {all_rows['selected_work_rows']:,} | "
            f"{all_rows['linked_work_rows']:,} | "
            f"{all_rows['zero_valid_author_token_work_rows']:,} | "
            f"{rate:.4f} | "
            f"{row['REFERENCE']['any_pre_signal_incumbent_work_rows']:,} | "
            f"{row['SIGNAL']['any_pre_signal_incumbent_work_rows']:,} |"
        )
    lines.extend(
        [
            "",
            "This audit makes no episode-exclusion, threshold, eligible-author,",
            "entry, persistence, association, or scientific-effect decision.",
            "",
        ]
    )
    return "\n".join(lines)


def build(
    config_path: Path,
    *,
    foundation_root: Path,
    cutoff_root: Path,
    author_history_root: Path,
    output: Path,
    workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    require(1 <= workers <= 32, "workers must be in [1, 32]")
    config_path = config_path.resolve(strict=True)
    config = _read_yaml(config_path, "linkage audit config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(
        config.get("mode")
        == "OUTCOME_BLIND_SELECTED_WORK_TO_VALID_AUTHOR_TOKEN_LINKAGE_AUDIT",
        "config mode mismatch",
    )
    require(
        config.get("status_before_execution")
        == "AUTHORIZED_BY_D0B_FOUNDATION_GATE_STATE_V5",
        "audit not authorized",
    )
    bindings = _mapping(config.get("repository_bindings"), "repository bindings")
    repository_bindings = {
        "gate_state": _validate_repo_binding(
            _mapping(bindings["gate_state"], "gate binding"),
            yaml_document=True,
        ),
        "author_foundation_summary": _validate_repo_binding(
            _mapping(bindings["author_foundation_summary"], "summary binding"),
            yaml_document=False,
        ),
    }
    forbidden = _mapping(config.get("forbidden_inputs_and_outputs"), "forbidden boundary")
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden boundary opened")
    episodes = _episodes(config)
    selected_path, foundation_binding = _validate_foundation_output(
        foundation_root,
        _mapping(config["author_foundation_output"], "foundation output"),
    )
    inputs = _mapping(config.get("normalized_inputs"), "normalized inputs")
    cutoff_paths, cutoff_binding, cutoff_stats = _validate_partitioned_input(
        cutoff_root,
        _mapping(inputs["cutoff_work_metadata"], "cutoff input"),
        workers=workers,
    )
    event_paths, event_binding, event_stats = _validate_partitioned_input(
        author_history_root,
        _mapping(inputs["author_history_events"], "event input"),
        workers=workers,
    )
    selected_table = pq.read_table(selected_path)
    selected = _selected_arrays(selected_table)
    require(len(selected_table) == 83_565, "selected work count differs")
    require(
        [
            (
                int(selected["field_id"][selected["episode_index"] == index][0]),
                int(selected["topic_id"][selected["episode_index"] == index][0]),
                int(selected["entry_year"][selected["episode_index"] == index][0]),
            )
            for index in range(1, 5)
        ]
        == [(row["field_id"], row["topic_id"], row["entry_year"]) for row in episodes],
        "selected episode scope differs",
    )
    type_names = {
        int(code): str(name)
        for code, name in _mapping(
            config.get("work_type_dictionary"),
            "work type dictionary",
        ).items()
    }
    require(set(type_names) == set(range(1, 25)), "work type dictionary differs")

    output = output.resolve()
    for input_root in (
        foundation_root.resolve(),
        cutoff_root.resolve(),
        author_history_root.resolve(),
    ):
        require(
            output != input_root and not output.is_relative_to(input_root),
            "output lies inside input root",
        )
    require(not output.exists(), "output directory already exists")
    require(output.parent.is_dir(), "output parent absent")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    try:
        metadata, cutoff_rows = _join_metadata(cutoff_paths, selected)
        incidence, event_rows = _join_events(event_paths, selected)
        require(cutoff_rows == inputs["cutoff_work_metadata"]["rows"], "cutoff row closure failed")
        require(event_rows == inputs["author_history_events"]["rows"], "event row closure failed")
        work_linkage, incidence_table, linkage_accounting = _build_tables(
            selected_table,
            metadata,
            incidence,
            episodes=episodes,
            type_names=type_names,
        )
        profile, profile_summary = _profile(
            work_linkage,
            episodes=episodes,
            type_names=type_names,
        )
        artifacts = [
            _write_parquet(
                work_linkage,
                temporary / "selected_work_author_linkage.parquet",
            ),
            _write_parquet(
                incidence_table,
                temporary / "selected_work_valid_author_incidence.parquet",
            ),
            _write_parquet(
                profile,
                temporary / "selected_work_author_linkage_profile.parquet",
            ),
        ]
        _validate_unchanged(cutoff_paths, cutoff_stats)
        _validate_unchanged(event_paths, event_stats)
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA,
            "label": config["label"],
            "status": STATUS,
            "scope": {
                "selected_work_author_linkage_audited": True,
                "episode_exclusion_decision_made": False,
                "linkage_threshold_selected": False,
                "incumbent_attention_definition_validated": False,
                "focal_or_network_leaveout_built": False,
                "d0b_complete": False,
                "d0_complete": False,
                "entry_or_post_entry_target_works_read": False,
                "persistence_rows_read": False,
                "realized_citation_rows_read": False,
                "predictive_scores_computed": False,
                "model_fitting_performed": False,
                "g4_rows_read": False,
                "gpu_used": False,
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "repository_bindings": repository_bindings,
            "foundation_output_binding": foundation_binding,
            "normalized_input_bindings": {
                "cutoff_work_metadata": cutoff_binding,
                "author_history_events": event_binding,
            },
            "episodes": episodes,
            "linkage_accounting": linkage_accounting,
            "profile_summary": profile_summary,
            "nonranked_team_cap_profiles": [10, 50, 100, "ALL"],
            "team_cap_selected": False,
            "artifacts": artifacts,
            "authorization": {
                "measurement_review_required": True,
                "remaining_d0b_outcome_blind_measurements": "AUTHORIZED",
                "episode_exclusion_or_threshold_decision": "NOT_AUTHORIZED_BY_THIS_COMPONENT",
                "discovery_outcome_opening": "NOT_AUTHORIZED",
                "g0_measurement_freeze": "NOT_AUTHORIZED",
                "model_fitting": "NOT_AUTHORIZED",
                "g4_open": "NOT_AUTHORIZED",
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "execution": {
                "workload_class": "CPU_IO",
                "workers": workers,
                "runtime_seconds": round(time.monotonic() - started, 6),
                "config_relative_path": str(config_path.relative_to(REPO_ROOT)),
                "config_sha256": _sha256(config_path),
                "builder_relative_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                "builder_sha256": _sha256(Path(__file__).resolve()),
            },
            "claim_boundary": config["claim_boundary"],
        }
        report_path = temporary / "selected_work_author_linkage_audit.json"
        _write_json(report_path, report)
        artifacts.append(
            {
                "relative_path": report_path.name,
                "bytes": report_path.stat().st_size,
                "sha256": _sha256(report_path),
            }
        )
        markdown_path = temporary / "selected_work_author_linkage_audit.md"
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        os.chmod(markdown_path, 0o600)
        artifacts.append(
            {
                "relative_path": markdown_path.name,
                "bytes": markdown_path.stat().st_size,
                "sha256": _sha256(markdown_path),
            }
        )
        artifacts.sort(key=lambda row: str(row["relative_path"]))
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": STATUS,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_bytes(
                    [
                        (row["relative_path"], row["bytes"], row["sha256"])
                        for row in artifacts
                    ]
                )
            ).hexdigest(),
            "selected_work_author_linkage_audited": True,
            "episode_exclusion_decision_made": False,
            "linkage_threshold_selected": False,
            "d0b_complete": False,
            "d0_complete": False,
            "discovery_outcome_opening_authorized": False,
            "g0_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "gpu_used": False,
            "receipt_written_last": True,
        }
        _write_json(
            temporary / "selected_work_author_linkage_audit.outputs.json",
            receipt,
        )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / "experiments/configs/"
            "pivot_feedback_d0b_selected_work_author_linkage_audit_v1.yaml"
        ),
    )
    parser.add_argument("--foundation-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build(
        args.config,
        foundation_root=args.foundation_root,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        output=args.output,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_work_rows": report["linkage_accounting"]["selected_work_rows"],
                "linked_work_rows": report["linkage_accounting"]["linked_work_rows"],
                "valid_author_incidence_rows": report["linkage_accounting"][
                    "valid_author_incidence_rows"
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
