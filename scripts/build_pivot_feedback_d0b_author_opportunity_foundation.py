#!/usr/bin/env python3
"""Build the outcome-blind D0-B pre-entry author measurement foundation."""

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
CONFIG_SCHEMA = "pivot_feedback_d0b_author_opportunity_foundation_config.v1"
REPORT_SCHEMA = "pivot_feedback_d0b_author_opportunity_foundation_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0b_author_opportunity_foundation_outputs.v1"
WORK_SCHEMA_VERSION = "pivot_feedback_d0b_selected_predecision_work.v1"
AUTHOR_SCHEMA_VERSION = "pivot_feedback_d0b_pre_entry_author_measurement_base.v1"
STATUS = "PASS_D0B_OUTCOME_BLIND_PRE_ENTRY_AUTHOR_OPPORTUNITY_FOUNDATION"
PARTITIONS = 4096
MAX_CONTROL_BYTES = 32 * 1024 * 1024
CAPS = (10, 50, 100)
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
WORK_OUTPUT_SCHEMA = pa.schema(
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
    metadata={b"schema_version": WORK_SCHEMA_VERSION.encode()},
)


def _author_schema() -> pa.Schema:
    fields = [
        pa.field("episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
    ]
    for stem in (
        "pre_entry_all_field_work_count",
        "pre_entry_focal_field_work_count",
        "portfolio_window_all_field_work_count",
        "portfolio_window_focal_field_work_count",
    ):
        for suffix in ("cap_10", "cap_50", "cap_100", "all"):
            fields.append(pa.field(f"{stem}_{suffix}", pa.uint32(), nullable=False))
    fields.extend(
        [
            pa.field(
                "pre_entry_target_topic_work_count",
                pa.uint32(),
                nullable=False,
            ),
            pa.field(
                "pre_signal_target_topic_work_count",
                pa.uint32(),
                nullable=False,
            ),
            pa.field("first_target_topic_year", pa.uint16(), nullable=True),
            pa.field("observed_no_prior_target_topic", pa.bool_(), nullable=False),
            pa.field("incumbent_before_signal_window", pa.bool_(), nullable=False),
        ]
    )
    return pa.schema(
        fields,
        metadata={b"schema_version": AUTHOR_SCHEMA_VERSION.encode()},
    )


AUTHOR_OUTPUT_SCHEMA = _author_schema()


class D0BAuthorFoundationError(RuntimeError):
    """Fail-closed D0-B author-foundation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise D0BAuthorFoundationError(message)


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
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} is too large")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0BAuthorFoundationError(f"invalid {label}: {error}") from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} is too large")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise D0BAuthorFoundationError(f"invalid {label}: {error}") from error


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
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _safe_repo_path(binding.get("relative_path"), "repository binding")
    observed_sha = _sha256(path)
    require(observed_sha == binding.get("sha256"), f"binding SHA mismatch: {path.name}")
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
        "binding required fields",
    ).items():
        observed = _nested(document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"binding field mismatch: {dotted_path}",
        )
    return document, {
        "relative_path": binding["relative_path"],
        "sha256": observed_sha,
        "status": document["status"],
    }


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for raw in _list(receipt.get("artifacts"), "receipt artifacts"):
        item = _mapping(raw, "receipt artifact")
        relative = str(item.get("relative_path"))
        require(relative not in result, "duplicate input artifact")
        result[relative] = item
    return result


def _validate_partitioned_input(
    root: Path,
    binding: Mapping[str, Any],
    *,
    workers: int,
) -> tuple[list[Path], dict[str, Any], list[tuple[int, int, int, int, int]]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid normalized input root")
    require(root.name == binding.get("expected_root_basename"), "input root basename mismatch")
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
    pre_stats = []
    for index, item in enumerate(items):
        expected = f"{prefix}part-{index:04d}.parquet"
        require(item.get("relative_path") == expected, "partition sequence mismatch")
        path = (root / expected).resolve(strict=True)
        require(path.is_relative_to(root), "partition escapes input root")
        require(path.is_file() and not path.is_symlink(), "invalid input partition")
        current = path.stat()
        require(current.st_size == item.get("bytes"), "partition byte mismatch")
        paths.append(path)
        pre_stats.append(
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
        observed = _sha256(path)
        require(observed == item.get("sha256"), f"partition SHA mismatch: {path.name}")
        return str(item["relative_path"]), int(item["bytes"]), observed

    with ThreadPoolExecutor(max_workers=workers) as executor:
        verified = list(executor.map(rehash, zip(paths, items, strict=True)))
    expected_bytes = binding.get("artifact_bytes")
    if expected_bytes is not None:
        require(
            sum(item[1] for item in verified) == expected_bytes,
            "partition byte total mismatch",
        )
    return paths, {
        "root_basename": root.name,
        "receipt_basename": receipt_path.name,
        "receipt_sha256": binding["receipt_sha256"],
        "artifact_prefix": prefix,
        "artifacts_rehashed": len(verified),
        "artifact_bytes": sum(item[1] for item in verified),
        "selected_tree_sha256": hashlib.sha256(
            _canonical_bytes(verified)
        ).hexdigest(),
    }, pre_stats


def _validate_unchanged(
    paths: Sequence[Path],
    pre_stats: Sequence[tuple[int, int, int, int, int]],
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
    require(observed == list(pre_stats), "input changed during execution")


def _episodes(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        _mapping(raw, "frozen episode")
        for raw in _list(config.get("frozen_episodes"), "frozen episodes")
    ]
    require([int(row["episode_index"]) for row in rows] == [1, 2, 3, 4], "episode order mismatch")
    expected = [
        (17, 14430, 2003),
        (17, 10579, 2012),
        (27, 11361, 2007),
        (27, 12505, 1991),
    ]
    require(
        [
            (int(row["field_id"]), int(row["topic_id"]), int(row["entry_year"]))
            for row in rows
        ]
        == expected,
        "frozen episodes differ",
    )
    for row in rows:
        entry = int(row["entry_year"])
        require(row["signal_window"] == [entry - 3, entry - 1], "signal window mismatch")
        require(row["reference_window"] == [entry - 6, entry - 4], "reference window mismatch")
        require(
            row["incumbent_entry_must_precede_year"] == entry - 6,
            "incumbent boundary mismatch",
        )
    return rows


def _selected_works(
    paths: Sequence[Path],
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[pa.Table, dict[int, np.ndarray], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scanned = 0
    for index, path in enumerate(paths):
        table = pq.read_table(path)
        require(table.schema == WORK_INPUT_SCHEMA, f"work schema mismatch: {index}")
        scanned += len(table)
        work_ids = (
            table["work_id_numeric"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.uint64, copy=False)
        )
        require(
            bool(np.all(work_ids % PARTITIONS == index)),
            f"work partition formula mismatch: {index}",
        )
        years = (
            table["publication_year"]
            .combine_chunks()
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.uint16, copy=False)
        )
        dispositions = (
            table["temporal_disposition"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )
        topics = (
            table["primary_topic_id"]
            .combine_chunks()
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
        )
        fields = (
            table["primary_field_id"]
            .combine_chunks()
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
        )
        teams = (
            table["declared_authors_count"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )
        usable = np.isin(dispositions, [2, 3])
        for episode in episodes:
            entry = int(episode["entry_year"])
            keep = (
                usable
                & (fields == int(episode["field_id"]))
                & (topics == int(episode["topic_id"]))
                & (years > 0)
                & (years <= entry - 1)
            )
            for work_id, year, team in zip(
                work_ids[keep].tolist(),
                years[keep].tolist(),
                teams[keep].tolist(),
                strict=True,
            ):
                rows.append(
                    {
                        "episode_index": int(episode["episode_index"]),
                        "field_id": int(episode["field_id"]),
                        "topic_id": int(episode["topic_id"]),
                        "entry_year": entry,
                        "work_numeric_id": int(work_id),
                        "publication_year": int(year),
                        "declared_authors_count": int(team),
                        "incumbency_qualifying_work": int(year) < entry - 6,
                    }
                )
    rows.sort(key=lambda row: (row["episode_index"], row["work_numeric_id"]))
    require(
        len({(row["episode_index"], row["work_numeric_id"]) for row in rows})
        == len(rows),
        "duplicate selected episode work",
    )
    table = pa.Table.from_pylist(rows, schema=WORK_OUTPUT_SCHEMA)
    work_ids = {
        int(episode["episode_index"]): np.asarray(
            [
                row["work_numeric_id"]
                for row in rows
                if row["episode_index"] == int(episode["episode_index"])
            ],
            dtype=np.uint64,
        )
        for episode in episodes
    }
    accounting = {
        "source_rows_scanned": scanned,
        "selected_work_rows": len(rows),
        "by_episode": {},
    }
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        selected = [row for row in rows if row["episode_index"] == episode_index]
        entry = int(episode["entry_year"])
        accounting["by_episode"][str(episode_index)] = {
            "selected_work_rows": len(selected),
            "incumbency_qualifying_work_rows": sum(
                bool(row["incumbency_qualifying_work"]) for row in selected
            ),
            "reference_window_work_rows": sum(
                entry - 6 <= row["publication_year"] <= entry - 4
                for row in selected
            ),
            "signal_window_work_rows": sum(
                entry - 3 <= row["publication_year"] <= entry - 1
                for row in selected
            ),
        }
    return table, work_ids, accounting


def _counts_by_author(
    inverse: np.ndarray,
    mask: np.ndarray,
    size: int,
) -> np.ndarray:
    values = np.bincount(inverse[mask], minlength=size)
    require(bool(np.all(values <= np.iinfo(np.uint32).max)), "author count overflow")
    return values.astype(np.uint32, copy=False)


def _cap_profile_counts(
    inverse: np.ndarray,
    base_mask: np.ndarray,
    teams: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Count one event family at all four non-ranked team-cap profiles."""
    selected_inverse = inverse[base_mask]
    selected_teams = teams[base_mask]
    bucket = np.full(len(selected_teams), 3, dtype=np.uint8)
    bucket[selected_teams <= 100] = 2
    bucket[selected_teams <= 50] = 1
    bucket[selected_teams <= 10] = 0
    flat_index = selected_inverse.astype(np.int64, copy=False) * 4 + bucket
    exact = np.bincount(flat_index, minlength=size * 4).reshape(size, 4)
    require(
        bool(np.all(exact <= np.iinfo(np.uint32).max)),
        "author cap-profile count overflow",
    )
    cumulative = np.cumsum(exact, axis=1, dtype=np.uint64)
    require(
        bool(np.all(cumulative <= np.iinfo(np.uint32).max)),
        "author cap-profile cumulative count overflow",
    )
    cumulative = cumulative.astype(np.uint32, copy=False)
    return (
        cumulative[:, 0],
        cumulative[:, 1],
        cumulative[:, 2],
        cumulative[:, 3],
    )


def _sorted_membership(values: np.ndarray, sorted_candidates: np.ndarray) -> np.ndarray:
    """Return exact membership without repeatedly sorting the candidate IDs."""
    require(
        len(sorted_candidates) < 2
        or bool(np.all(sorted_candidates[1:] > sorted_candidates[:-1])),
        "selected work IDs must be strictly sorted",
    )
    if not len(sorted_candidates):
        return np.zeros(len(values), dtype=bool)
    positions = np.searchsorted(sorted_candidates, values)
    in_range = positions < len(sorted_candidates)
    result = np.zeros(len(values), dtype=bool)
    result[in_range] = (
        sorted_candidates[positions[in_range]] == values[in_range]
    )
    return result


def _author_batch(
    table: pa.Table,
    *,
    partition_index: int,
    episodes: Sequence[Mapping[str, Any]],
    selected_work_ids: Mapping[int, np.ndarray],
) -> tuple[list[pa.Table], dict[int, dict[str, int]]]:
    require(table.schema == EVENT_INPUT_SCHEMA, f"event schema mismatch: {partition_index}")
    authors = (
        table["author_numeric_id"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype(np.uint64, copy=False)
    )
    require(
        bool(np.all(authors % PARTITIONS == partition_index)),
        f"author partition formula mismatch: {partition_index}",
    )
    works = (
        table["work_numeric_id"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype(np.uint64, copy=False)
    )
    years = (
        table["publication_year"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
    )
    dispositions = (
        table["temporal_disposition"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
    )
    teams = (
        table["declared_authors_count"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
    )
    fields = (
        table["primary_field_id"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
    )
    unique_authors, inverse = np.unique(authors, return_inverse=True)
    size = len(unique_authors)
    usable = np.isin(dispositions, [2, 3]) & (years > 0)
    result = []
    accounting = {}
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        field_id = int(episode["field_id"])
        entry_year = int(episode["entry_year"])
        pre_entry = usable & (years <= entry_year - 1)
        portfolio = pre_entry & (years >= entry_year - 3)
        focal = fields == field_id
        metric_arrays: dict[str, np.ndarray] = {}
        for stem, mask in (
            ("pre_entry_all_field_work_count", pre_entry),
            ("pre_entry_focal_field_work_count", pre_entry & focal),
            ("portfolio_window_all_field_work_count", portfolio),
            ("portfolio_window_focal_field_work_count", portfolio & focal),
        ):
            profiles = _cap_profile_counts(inverse, mask, teams, size)
            for suffix, values in zip(
                ("cap_10", "cap_50", "cap_100", "all"),
                profiles,
                strict=True,
            ):
                metric_arrays[f"{stem}_{suffix}"] = values
        target_match = _sorted_membership(
            works,
            selected_work_ids[episode_index],
        )
        require(
            not bool(
                np.any(
                    target_match
                    & (
                        ~usable
                        | (years > entry_year - 1)
                        | ~focal
                    )
                )
            ),
            f"selected target work disagrees with author-history event: {episode_index}",
        )
        target = target_match & usable & pre_entry & focal
        target_counts = _counts_by_author(inverse, target, size)
        pre_signal = target & (years < entry_year - 6)
        pre_signal_counts = _counts_by_author(inverse, pre_signal, size)
        first_target = np.full(size, np.iinfo(np.uint16).max, dtype=np.uint16)
        np.minimum.at(
            first_target,
            inverse[target],
            years[target].astype(np.uint16, copy=False),
        )
        candidate = metric_arrays[
            "portfolio_window_focal_field_work_count_all"
        ] > 0
        candidate_indices = np.flatnonzero(candidate)
        first_values = first_target[candidate_indices]
        first_array = pa.array(
            first_values,
            mask=first_values == np.iinfo(np.uint16).max,
            type=pa.uint16(),
        )
        data: dict[str, pa.Array] = {
            "episode_index": pa.array(
                np.full(len(candidate_indices), episode_index, dtype=np.uint8),
                type=pa.uint8(),
            ),
            "field_id": pa.array(
                np.full(len(candidate_indices), field_id, dtype=np.uint8),
                type=pa.uint8(),
            ),
            "topic_id": pa.array(
                np.full(
                    len(candidate_indices),
                    int(episode["topic_id"]),
                    dtype=np.uint32,
                ),
                type=pa.uint32(),
            ),
            "entry_year": pa.array(
                np.full(len(candidate_indices), entry_year, dtype=np.uint16),
                type=pa.uint16(),
            ),
            "author_numeric_id": pa.array(
                unique_authors[candidate_indices],
                type=pa.uint64(),
            ),
        }
        for name in AUTHOR_OUTPUT_SCHEMA.names[5:21]:
            data[name] = pa.array(
                metric_arrays[name][candidate_indices],
                type=pa.uint32(),
            )
        data["pre_entry_target_topic_work_count"] = pa.array(
            target_counts[candidate_indices],
            type=pa.uint32(),
        )
        data["pre_signal_target_topic_work_count"] = pa.array(
            pre_signal_counts[candidate_indices],
            type=pa.uint32(),
        )
        data["first_target_topic_year"] = first_array
        data["observed_no_prior_target_topic"] = pa.array(
            target_counts[candidate_indices] == 0,
            type=pa.bool_(),
        )
        data["incumbent_before_signal_window"] = pa.array(
            pre_signal_counts[candidate_indices] > 0,
            type=pa.bool_(),
        )
        batch = pa.Table.from_arrays(
            [data[name] for name in AUTHOR_OUTPUT_SCHEMA.names],
            schema=AUTHOR_OUTPUT_SCHEMA,
        )
        result.append(batch)
        accounting[episode_index] = {
            "author_rows": len(batch),
            "observed_no_prior_target_topic_rows": int(
                (target_counts[candidate_indices] == 0).sum()
            ),
            "prior_target_topic_rows": int(
                (target_counts[candidate_indices] > 0).sum()
            ),
            "incumbent_before_signal_window_rows": int(
                (pre_signal_counts[candidate_indices] > 0).sum()
            ),
            "portfolio_focal_work_events_all": int(
                metric_arrays[
                    "portfolio_window_focal_field_work_count_all"
                ][candidate_indices].sum(dtype=np.uint64)
            ),
        }
    return result, accounting


def _write_author_base(
    paths: Sequence[Path],
    output: Path,
    *,
    episodes: Sequence[Mapping[str, Any]],
    selected_work_ids: Mapping[int, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    writer = pq.ParquetWriter(
        output,
        AUTHOR_OUTPUT_SCHEMA,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    totals = {
        int(episode["episode_index"]): {
            "author_rows": 0,
            "observed_no_prior_target_topic_rows": 0,
            "prior_target_topic_rows": 0,
            "incumbent_before_signal_window_rows": 0,
            "portfolio_focal_work_events_all": 0,
        }
        for episode in episodes
    }
    source_rows = 0
    maximum_batch_rows = 0
    try:
        for index, path in enumerate(paths):
            table = pq.read_table(path)
            source_rows += len(table)
            batches, accounting = _author_batch(
                table,
                partition_index=index,
                episodes=episodes,
                selected_work_ids=selected_work_ids,
            )
            for batch in batches:
                maximum_batch_rows = max(maximum_batch_rows, len(batch))
                if len(batch):
                    writer.write_table(batch)
            for episode_index, values in accounting.items():
                for key, value in values.items():
                    totals[episode_index][key] += int(value)
    finally:
        writer.close()
    os.chmod(output, 0o600)
    parquet = pq.ParquetFile(output)
    require(
        parquet.metadata.num_rows
        == sum(values["author_rows"] for values in totals.values()),
        "author output row closure failed",
    )
    return {
        "relative_path": output.name,
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
    }, {
        "source_rows_scanned": source_rows,
        "maximum_partition_episode_batch_rows": maximum_batch_rows,
        "by_episode": {str(key): value for key, value in totals.items()},
    }


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
        "sha256": _sha256(path),
        "rows": len(table),
        "row_groups": pq.ParquetFile(path).metadata.num_row_groups,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    os.chmod(path, 0o600)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# D0-B outcome-blind author-opportunity foundation",
        "",
        f"Status: `{report['status']}`",
        "",
        "| Episode | Selected pre-decision works | Observed field-active authors | No prior target topic | Pre-signal incumbents |",
        "|---:|---:|---:|---:|---:|",
    ]
    for episode_index in range(1, 5):
        work = report["selected_work_accounting"]["by_episode"][
            str(episode_index)
        ]
        author = report["author_measurement_accounting"]["by_episode"][
            str(episode_index)
        ]
        lines.append(
            f"| {episode_index} | {work['selected_work_rows']:,} | "
            f"{author['author_rows']:,} | "
            f"{author['observed_no_prior_target_topic_rows']:,} | "
            f"{author['incumbent_before_signal_window_rows']:,} |"
        )
    lines.extend(
        [
            "",
            "This component reads only information available by t-1 and does not",
            "open entry, persistence, realized-citation, model-selection, or G4",
            "outcomes. It is not a final eligible-author panel and does not complete",
            "D0-B or D0.",
            "",
        ]
    )
    return "\n".join(lines)


def build(
    config_path: Path,
    *,
    cutoff_root: Path,
    author_history_root: Path,
    output: Path,
    workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    require(1 <= workers <= 32, "workers must be in [1, 32]")
    config_path = config_path.resolve(strict=True)
    config = _read_yaml(config_path, "D0-B config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(
        config.get("label") == "pivot_feedback_d0b_author_opportunity_foundation_v1",
        "config label mismatch",
    )
    require(
        config.get("mode") == "OUTCOME_BLIND_PRE_ENTRY_AUTHOR_MEASUREMENT_FOUNDATION",
        "config mode mismatch",
    )
    require(
        config.get("status_before_execution") == "AUTHORIZED_BY_D0A_GATE_STATE_V4",
        "D0-B foundation is not authorized",
    )
    repo_bindings = _mapping(config.get("repository_bindings"), "repository bindings")
    _, gate_binding = _validate_repo_binding(
        _mapping(repo_bindings["gate_state"], "gate binding"),
        yaml_document=True,
    )
    d0a_document, d0a_binding = _validate_repo_binding(
        _mapping(repo_bindings["d0a_freeze"], "D0-A binding"),
        yaml_document=False,
    )
    episodes = _episodes(config)
    require(
        [
            (
                int(row["field_id"]),
                int(row["topic_id"]),
                int(row["entry_year"]),
            )
            for row in d0a_document["selected_pilot_episodes"]
        ]
        == [
            (
                int(row["field_id"]),
                int(row["topic_id"]),
                int(row["entry_year"]),
            )
            for row in episodes
        ],
        "D0-A episode binding differs",
    )
    forbidden = _mapping(
        config.get("forbidden_inputs_and_outputs"),
        "forbidden boundary",
    )
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden boundary opened")
    inputs = _mapping(config.get("normalized_inputs"), "normalized inputs")
    cutoff_paths, cutoff_binding, cutoff_stats = _validate_partitioned_input(
        cutoff_root,
        _mapping(inputs["cutoff_work_metadata"], "cutoff input"),
        workers=workers,
    )
    event_paths, event_binding, event_stats = _validate_partitioned_input(
        author_history_root,
        _mapping(inputs["author_history_events"], "author-history input"),
        workers=workers,
    )

    output = output.resolve()
    for input_root in (cutoff_root.resolve(), author_history_root.resolve()):
        require(
            output != input_root and not output.is_relative_to(input_root),
            "output lies inside input root",
        )
    require(not output.exists(), "output directory already exists")
    require(output.parent.is_dir(), "output parent does not exist")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    os.chmod(temporary, 0o700)
    try:
        selected_work_table, selected_work_ids, work_accounting = _selected_works(
            cutoff_paths,
            episodes,
        )
        require(
            work_accounting["source_rows_scanned"]
            == inputs["cutoff_work_metadata"]["rows"],
            "cutoff row closure failed",
        )
        selected_work_artifact = _write_parquet(
            selected_work_table,
            temporary / "episode_selected_predecision_works.parquet",
        )
        author_artifact, author_accounting = _write_author_base(
            event_paths,
            temporary / "pre_entry_author_measurement_base.parquet",
            episodes=episodes,
            selected_work_ids=selected_work_ids,
        )
        require(
            author_accounting["source_rows_scanned"]
            == inputs["author_history_events"]["rows"],
            "author-history row closure failed",
        )
        _validate_unchanged(cutoff_paths, cutoff_stats)
        _validate_unchanged(event_paths, event_stats)
        artifacts = [selected_work_artifact, author_artifact]
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA,
            "label": "pivot_feedback_d0b_author_opportunity_foundation_v1",
            "status": STATUS,
            "scope": {
                "d0b_author_foundation_built": True,
                "d0b_complete": False,
                "d0_complete": False,
                "entry_or_post_entry_target_works_read": False,
                "persistence_rows_read": False,
                "realized_citation_rows_read": False,
                "predictive_scores_computed": False,
                "threshold_tuning_performed": False,
                "final_candidate_eligibility_established": False,
                "model_fitting_performed": False,
                "g4_rows_read": False,
                "gpu_used": False,
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "repository_bindings": {
                "gate_state": gate_binding,
                "d0a_freeze": d0a_binding,
            },
            "normalized_input_bindings": {
                "cutoff_work_metadata": cutoff_binding,
                "author_history_events": event_binding,
            },
            "episodes": episodes,
            "selected_work_accounting": {
                **work_accounting,
                "artifact": selected_work_artifact,
            },
            "author_measurement_accounting": {
                **author_accounting,
                "artifact": author_artifact,
                "nonranked_team_cap_profiles": [10, 50, 100, "ALL"],
                "team_cap_selected": False,
            },
            "deferred_d0b_components": config["deferred_d0b_components"],
            "authorization": {
                "remaining_d0b_measurement_components": "AUTHORIZED",
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
                "builder_relative_path": str(
                    Path(__file__).resolve().relative_to(REPO_ROOT)
                ),
                "builder_sha256": _sha256(Path(__file__).resolve()),
            },
            "claim_boundary": config["claim_boundary"],
        }
        report_path = temporary / "d0b_author_opportunity_foundation.json"
        _write_json(report_path, report)
        artifacts.append(
            {
                "relative_path": report_path.name,
                "bytes": report_path.stat().st_size,
                "sha256": _sha256(report_path),
            }
        )
        markdown_path = temporary / "d0b_author_opportunity_foundation.md"
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
            "d0b_author_foundation_built": True,
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
            temporary / "d0b_author_opportunity_foundation.outputs.json",
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
        default=REPO_ROOT
        / "experiments/configs/pivot_feedback_d0b_author_opportunity_foundation_v1.yaml",
    )
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build(
        args.config,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        output=args.output,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_work_rows": report["selected_work_accounting"][
                    "selected_work_rows"
                ],
                "author_rows": sum(
                    row["author_rows"]
                    for row in report["author_measurement_accounting"][
                        "by_episode"
                    ].values()
                ),
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
