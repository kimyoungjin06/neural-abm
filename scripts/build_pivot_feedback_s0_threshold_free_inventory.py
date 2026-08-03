#!/usr/bin/env python3
"""Build the exact threshold-free S0 core field-year inventory.

The builder reads only receipt-bound normalized work metadata and distinct
author-work event partitions.  It does not read outcomes, select fields, or
claim that the deferred ORCID, affiliation, and graph-topology components have
run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "pivot_feedback_s0_threshold_free_inventory_config.v1"
REPORT_SCHEMA = "pivot_feedback_s0_threshold_free_core_inventory_report.v1"
FIELD_YEAR_SCHEMA = "pivot_feedback_s0_field_year_core.v1"
ATTENTION_SCHEMA = "pivot_feedback_s0_attention_trajectory_core.v1"
STATUS = (
    "PASS_S0_THRESHOLD_FREE_CORE_FIELD_YEAR_INVENTORY_"
    "WITH_ORCID_AFFILIATION_AND_GRAPH_TOPOLOGY_PENDING"
)
PARTITION_COUNT = 4096
FIELD_MIN = 11
FIELD_MAX = 36
YEAR_MULTIPLIER = 65_536
MAX_CONTROL_BYTES = 16 * 1024 * 1024

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

WORK_METRICS = (
    "total_canonical_works",
    "usable_date_works",
    "primary_topic_relation_works",
    "all_topic_relation_works",
    "declared_author_slots",
    "zero_declared_author_works",
    "single_declared_author_works",
    "multi_declared_author_works",
    "declared_pair_incidences_cap_10",
    "declared_pair_incidences_cap_50",
    "declared_pair_incidences_cap_100",
    "declared_pair_incidences_all",
    "citation_calendar_support_at_least_1y",
    "citation_calendar_support_at_least_3y",
    "citation_calendar_support_at_least_5y",
    "citation_calendar_support_at_least_10y",
)
AUTHOR_METRICS = (
    "distinct_author_work_pairs",
    "history_usable_distinct_author_work_pairs",
    "distinct_author_tokens",
    "history_usable_distinct_author_tokens",
)


class S0InventoryError(RuntimeError):
    """Fail-closed S0 core inventory error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise S0InventoryError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return list(value)


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        require(isinstance(current, Mapping) and part in current, f"missing field {dotted_path}")
        current = current[part]
    return current


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _safe_relative(value: Any, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} must be a non-empty string")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    require(str(pure) == value, f"{label} must be normalized POSIX")
    return value


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} is too large")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise S0InventoryError(f"invalid {label}: {error}") from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} is too large")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S0InventoryError(f"invalid {label}: {error}") from error


def _validate_parent(config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    binding = _mapping(config.get("g_minus_1a_parent"), "G-1A parent")
    relative = _safe_relative(binding.get("relative_path"), "G-1A parent path")
    path = (repo_root / relative).resolve(strict=True)
    require(path.is_relative_to(repo_root.resolve()), "G-1A parent escapes repository")
    require(_sha256_file(path) == binding.get("sha256"), "G-1A parent SHA-256 mismatch")
    document = _read_json(path, "G-1A parent")
    require(
        document.get("schema_version") == binding.get("required_schema_version"),
        "G-1A parent schema mismatch",
    )
    require(document.get("status") == binding.get("required_status"), "G-1A parent status mismatch")
    for dotted_path, expected in _mapping(
        binding.get("required_fields"), "G-1A parent fields"
    ).items():
        observed = _nested(document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"G-1A parent field mismatch: {dotted_path}",
        )
    return {
        "relative_path": relative,
        "sha256": binding["sha256"],
        "status": document["status"],
    }


def _validate_root(
    root: Path,
    binding: Mapping[str, Any],
    *,
    workers: int,
) -> tuple[list[Path], dict[str, Any]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "normalized input root is invalid")
    require(root.name == binding.get("expected_root_basename"), "input root basename mismatch")
    receipt_name = str(binding.get("receipt_basename"))
    receipt_path = root / receipt_name
    require(_sha256_file(receipt_path) == binding.get("receipt_sha256"), "input receipt SHA mismatch")
    receipt = _read_json(receipt_path, f"input receipt {root.name}")
    require(
        receipt.get("schema_version") == binding.get("receipt_schema_version"),
        "input receipt schema mismatch",
    )
    require(receipt.get("status") == binding.get("required_status"), "input receipt status mismatch")
    if "expected_tree_sha256" in binding:
        require(
            receipt.get("work_metadata_tree_sha256")
            == binding.get("expected_tree_sha256"),
            "input work-metadata tree mismatch",
        )
    prefix = str(binding.get("artifact_prefix"))
    declared = [
        _mapping(item, "input artifact")
        for item in _list(receipt.get("artifacts"), "input artifacts")
        if str(_mapping(item, "input artifact").get("relative_path", "")).startswith(prefix)
    ]
    require(len(declared) == binding.get("expected_artifact_count"), "selected artifact count mismatch")
    declared.sort(key=lambda item: str(item["relative_path"]))
    paths: list[Path] = []
    pre_stats: list[tuple[int, int, int, int, int]] = []
    for index, item in enumerate(declared):
        expected_relative = f"{prefix}part-{index:04d}.parquet"
        require(item.get("relative_path") == expected_relative, "partition sequence mismatch")
        path = (root / expected_relative).resolve(strict=True)
        require(path.is_relative_to(root), "input artifact escapes root")
        require(path.is_file() and not path.is_symlink(), "input artifact is not regular")
        stat = path.stat()
        require(stat.st_size == item.get("bytes"), "input artifact byte size mismatch")
        paths.append(path)
        pre_stats.append(
            (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        )

    def rehash(pair: tuple[Path, Mapping[str, Any]]) -> tuple[str, int, str]:
        path, item = pair
        observed = _sha256_file(path)
        require(observed == item.get("sha256"), f"input artifact SHA mismatch: {item['relative_path']}")
        return str(item["relative_path"]), int(item["bytes"]), observed

    with ThreadPoolExecutor(max_workers=workers) as executor:
        verified = list(executor.map(rehash, zip(paths, declared, strict=True)))
    post_stats = []
    for path in paths:
        stat = path.stat()
        post_stats.append(
            (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        )
    require(post_stats == pre_stats, "input artifact stat changed during verification")
    selected_tree = hashlib.sha256(_canonical_bytes(verified)).hexdigest()
    selected_bytes = sum(item[1] for item in verified)
    return paths, {
        "root_basename": root.name,
        "receipt_basename": receipt_name,
        "receipt_sha256": binding["receipt_sha256"],
        "selected_artifact_prefix": prefix,
        "selected_artifacts_rehashed": len(verified),
        "selected_artifact_bytes": selected_bytes,
        "selected_artifact_tree_sha256": selected_tree,
        "selected_artifact_stats_unchanged": True,
    }


def _add_grouped(
    target: dict[int, dict[str, int]],
    keys: np.ndarray,
    metric: str,
    weights: np.ndarray | None = None,
) -> None:
    if len(keys) == 0:
        return
    unique, inverse = np.unique(keys, return_inverse=True)
    if weights is None:
        values = np.bincount(inverse)
    else:
        values = np.bincount(inverse, weights=weights.astype(np.float64))
        require(
            np.all(values <= np.float64(2**53)),
            f"{metric} exceeds exact float64 grouped-sum range",
        )
    for key, value in zip(unique.tolist(), values.tolist(), strict=True):
        cell = target.setdefault(int(key), {})
        cell[metric] = cell.get(metric, 0) + int(value)


def _year_array(column: pa.ChunkedArray) -> np.ndarray:
    return (
        column.combine_chunks()
        .fill_null(0)
        .to_numpy(zero_copy_only=False)
        .astype(np.uint32, copy=False)
    )


def _aggregate_work_files(
    paths: Sequence[Path],
    contract: Mapping[str, Any],
) -> tuple[dict[int, dict[str, int]], dict[str, int]]:
    cells: dict[int, dict[str, int]] = {}
    totals = {
        "rows": 0,
        "classified_field_rows": 0,
        "null_field_rows": 0,
        "null_year_rows": 0,
        "usable_date_rows": 0,
    }
    usable_dispositions = np.array(contract["usable_temporal_dispositions"], dtype=np.uint8)
    last_complete_year = int(contract["last_complete_calendar_year"])
    for index, path in enumerate(paths):
        table = pq.read_table(path)
        require(table.schema == WORK_SCHEMA, f"work schema mismatch: partition {index}")
        work_ids = table["work_id_numeric"].combine_chunks().to_numpy(zero_copy_only=False)
        require(
            bool(np.all(work_ids % PARTITION_COUNT == index)),
            f"work partition formula mismatch: {index}",
        )
        years = _year_array(table["publication_year"])
        fields = (
            table["primary_field_id"]
            .combine_chunks()
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.uint32, copy=False)
        )
        disposition = (
            table["temporal_disposition"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )
        authors = (
            table["declared_authors_count"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.uint64, copy=False)
        )
        topic_present = ~table["primary_topic_id"].combine_chunks().is_null().to_numpy(
            zero_copy_only=False
        )
        valid = (fields >= FIELD_MIN) & (fields <= FIELD_MAX)
        keys = fields[valid].astype(np.uint64) * YEAR_MULTIPLIER + years[valid]
        valid_authors = authors[valid]
        valid_years = years[valid]
        valid_usable = np.isin(disposition[valid], usable_dispositions)

        _add_grouped(cells, keys, "total_canonical_works")
        _add_grouped(cells, keys[valid_usable], "usable_date_works")
        _add_grouped(cells, keys[topic_present[valid]], "primary_topic_relation_works")
        _add_grouped(cells, keys[topic_present[valid]], "all_topic_relation_works")
        _add_grouped(cells, keys, "declared_author_slots", valid_authors)
        _add_grouped(cells, keys[valid_authors == 0], "zero_declared_author_works")
        _add_grouped(cells, keys[valid_authors == 1], "single_declared_author_works")
        _add_grouped(cells, keys[valid_authors >= 2], "multi_declared_author_works")
        pair_counts = valid_authors * (valid_authors - 1) // 2
        for cap in (10, 50, 100):
            mask = valid_authors <= cap
            _add_grouped(
                cells,
                keys[mask],
                f"declared_pair_incidences_cap_{cap}",
                pair_counts[mask],
            )
        _add_grouped(cells, keys, "declared_pair_incidences_all", pair_counts)
        for horizon in (1, 3, 5, 10):
            support = (
                (valid_years > 0)
                & (valid_years <= last_complete_year)
                & ((last_complete_year - valid_years) >= horizon)
            )
            _add_grouped(
                cells,
                keys[support],
                f"citation_calendar_support_at_least_{horizon}y",
            )

        totals["rows"] += len(table)
        totals["classified_field_rows"] += int(valid.sum())
        totals["null_field_rows"] += int((fields == 0).sum())
        totals["null_year_rows"] += int((years == 0).sum())
        totals["usable_date_rows"] += int(np.isin(disposition, usable_dispositions).sum())
    return cells, totals


def _distinct_author_cell_keys(
    authors: np.ndarray,
    cell_keys: np.ndarray,
) -> np.ndarray:
    if len(authors) == 0:
        return np.empty(0, dtype=np.uint64)
    order = np.lexsort((cell_keys, authors))
    sorted_authors = authors[order]
    sorted_keys = cell_keys[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = (sorted_authors[1:] != sorted_authors[:-1]) | (
        sorted_keys[1:] != sorted_keys[:-1]
    )
    return sorted_keys[keep]


def _aggregate_event_files(
    paths: Sequence[Path],
    contract: Mapping[str, Any],
) -> tuple[dict[int, dict[str, int]], dict[str, int]]:
    cells: dict[int, dict[str, int]] = {}
    totals = {
        "rows": 0,
        "history_usable_rows": 0,
        "classified_field_rows": 0,
        "null_field_rows": 0,
    }
    usable_dispositions = np.array(contract["usable_temporal_dispositions"], dtype=np.uint8)
    for index, path in enumerate(paths):
        table = pq.read_table(path)
        require(table.schema == EVENT_SCHEMA, f"event schema mismatch: partition {index}")
        authors = (
            table["author_numeric_id"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.uint64, copy=False)
        )
        require(
            bool(np.all(authors % PARTITION_COUNT == index)),
            f"author partition formula mismatch: {index}",
        )
        fields = (
            table["primary_field_id"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.uint32, copy=False)
        )
        years = (
            table["publication_year"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.uint32, copy=False)
        )
        dispositions = (
            table["temporal_disposition"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )
        valid = (fields >= FIELD_MIN) & (fields <= FIELD_MAX)
        usable = np.isin(dispositions, usable_dispositions)
        keys = fields[valid].astype(np.uint64) * YEAR_MULTIPLIER + years[valid]
        valid_authors = authors[valid]

        _add_grouped(cells, keys, "distinct_author_work_pairs")
        _add_grouped(
            cells,
            keys[usable[valid]],
            "history_usable_distinct_author_work_pairs",
        )
        distinct_keys = _distinct_author_cell_keys(valid_authors, keys)
        _add_grouped(cells, distinct_keys, "distinct_author_tokens")
        usable_valid = usable[valid]
        usable_distinct_keys = _distinct_author_cell_keys(
            valid_authors[usable_valid],
            keys[usable_valid],
        )
        _add_grouped(
            cells,
            usable_distinct_keys,
            "history_usable_distinct_author_tokens",
        )

        totals["rows"] += len(table)
        totals["history_usable_rows"] += int(usable.sum())
        totals["classified_field_rows"] += int(valid.sum())
        totals["null_field_rows"] += int((fields == 0).sum())
    return cells, totals


def _field_year_table(
    work_cells: Mapping[int, Mapping[str, int]],
    author_cells: Mapping[int, Mapping[str, int]],
) -> pa.Table:
    require(set(author_cells) <= set(work_cells), "author field-year cells fall outside work cells")
    rows: list[dict[str, Any]] = []
    for key in sorted(work_cells):
        field_id, year = divmod(key, YEAR_MULTIPLIER)
        work = work_cells[key]
        author = author_cells.get(key, {})
        row: dict[str, Any] = {
            "field_id": field_id,
            "publication_year": None if year == 0 else year,
        }
        for metric in WORK_METRICS:
            row[metric] = int(work.get(metric, 0))
        for metric in AUTHOR_METRICS:
            row[metric] = int(author.get(metric, 0))
        require(
            row["primary_topic_relation_works"] == row["all_topic_relation_works"],
            "primary/all-topic work closure failed",
        )
        require(
            row["usable_date_works"] <= row["total_canonical_works"],
            "usable work count exceeds total",
        )
        require(
            row["history_usable_distinct_author_work_pairs"]
            <= row["distinct_author_work_pairs"],
            "usable author-work pairs exceed total",
        )
        require(
            row["history_usable_distinct_author_tokens"] <= row["distinct_author_tokens"],
            "usable author tokens exceed total",
        )
        rows.append(row)
    fields = [
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=True),
    ]
    fields.extend(pa.field(metric, pa.uint64(), nullable=False) for metric in WORK_METRICS)
    fields.extend(pa.field(metric, pa.uint64(), nullable=False) for metric in AUTHOR_METRICS)
    schema = pa.schema(fields, metadata={b"schema_version": FIELD_YEAR_SCHEMA.encode()})
    return pa.Table.from_pylist(rows, schema=schema)


def _attention_table(field_year: pa.Table) -> pa.Table:
    rows = field_year.to_pylist()
    annual = {
        (int(row["field_id"]), int(row["publication_year"])): int(
            row["total_canonical_works"]
        )
        for row in rows
        if row["publication_year"] is not None
    }
    result = []
    for field_id, year in sorted(annual):
        result.append(
            {
                "field_id": field_id,
                "publication_year": year,
                "annual_canonical_works": annual[(field_id, year)],
                "trailing_3y_canonical_works": sum(
                    annual.get((field_id, candidate), 0)
                    for candidate in range(year - 2, year + 1)
                ),
                "trailing_5y_canonical_works": sum(
                    annual.get((field_id, candidate), 0)
                    for candidate in range(year - 4, year + 1)
                ),
            }
        )
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("publication_year", pa.uint16(), nullable=False),
            pa.field("annual_canonical_works", pa.uint64(), nullable=False),
            pa.field("trailing_3y_canonical_works", pa.uint64(), nullable=False),
            pa.field("trailing_5y_canonical_works", pa.uint64(), nullable=False),
        ],
        metadata={b"schema_version": ATTENTION_SCHEMA.encode()},
    )
    return pa.Table.from_pylist(result, schema=schema)


def _parquet_sha(table: pa.Table, path: Path) -> dict[str, Any]:
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
        "sha256": _sha256_file(path),
        "rows": len(table),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    os.chmod(path, 0o600)


def _markdown(report: Mapping[str, Any]) -> str:
    work = report["work_accounting"]
    author = report["author_event_accounting"]
    return "\n".join(
        [
            "# S0 threshold-free core field-year inventory",
            "",
            f"Status: `{report['status']}`",
            "",
            f"- canonical work rows scanned: {work['rows']:,}",
            f"- classified 26-field work rows: {work['classified_field_rows']:,}",
            f"- distinct author-work event rows scanned: {author['rows']:,}",
            f"- history-usable author-work event rows: {author['history_usable_rows']:,}",
            f"- field-year rows: {report['field_year_inventory']['rows']:,}",
            f"- observed publication-year range: {report['field_year_inventory']['observed_year_min']}–{report['field_year_inventory']['observed_year_max']}",
            "",
            "This component does not complete S0. ORCID/token stability,",
            "author-work affiliation, and distinct coauthor graph topology remain",
            "explicitly not run. D0 and model fitting remain unauthorized.",
            "",
        ]
    )


def build(
    config_path: Path,
    cutoff_root: Path,
    author_root: Path,
    output: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _read_yaml(config_path, "S0 config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "S0 config schema mismatch")
    require(config.get("label") == "pivot_feedback_s0_threshold_free_inventory_v1", "S0 config label mismatch")
    require(
        config.get("mode") == "READ_ONLY_NORMALIZED_SURFACE_STRUCTURAL_AGGREGATION",
        "S0 mode mismatch",
    )
    require(
        config.get("status_before_execution") == "AUTHORIZED_BY_G_MINUS_1A",
        "S0 is not authorized",
    )
    parent = _validate_parent(config, REPO_ROOT)
    inputs = _mapping(config.get("normalized_inputs"), "normalized inputs")
    cutoff_paths, cutoff_validation = _validate_root(
        cutoff_root,
        _mapping(inputs.get("cutoff_primitives"), "cutoff input"),
        workers=workers,
    )
    author_paths, author_validation = _validate_root(
        author_root,
        _mapping(inputs.get("author_history"), "author input"),
        workers=workers,
    )
    output = output.resolve()
    for input_root in (cutoff_root.resolve(), author_root.resolve()):
        require(
            output != input_root and not output.is_relative_to(input_root),
            "output may not be inside an input root",
        )
    require(not output.exists(), "output directory already exists")
    require(output.parent.is_dir(), "output parent does not exist")
    require(workers >= 1 and workers <= 32, "workers must be in [1, 32]")

    structural = _mapping(config.get("structural_contract"), "structural contract")
    require(structural.get("field_ids") == list(range(FIELD_MIN, FIELD_MAX + 1)), "field domain mismatch")
    require(structural.get("field_count") == 26, "field count mismatch")
    require(structural.get("team_cap_profiles_are_ranked_or_selected") is False, "team caps are selected")
    require(structural.get("citation_calendar_support_is_outcome_success") is False, "calendar support is labeled success")
    require(structural.get("attention_resolution_is_ranked_or_selected") is False, "attention resolution is selected")
    forbidden = set(_list(config.get("forbidden_metrics"), "forbidden metrics"))
    require(
        {
            "eligible_author_count",
            "matured_entry_opportunity_count",
            "fixed_horizon_outcome_coverage",
            "independent_attention_shock_count",
            "entry_effect_size",
            "persistence_effect_size",
            "overshoot_timing_gap",
            "predictive_score",
            "selected_best_field",
            "selected_best_threshold",
            "g4_holdout_outcome",
        }
        <= forbidden,
        "forbidden metric boundary is incomplete",
    )
    required = _mapping(config.get("required_core_metrics"), "required core metrics")
    require(tuple(required["work_field_year"]) == WORK_METRICS, "work metric contract mismatch")
    require(tuple(required["author_field_year"]) == AUTHOR_METRICS, "author metric contract mismatch")

    work_cells, work_totals = _aggregate_work_files(cutoff_paths, structural)
    author_cells, author_totals = _aggregate_event_files(author_paths, structural)
    cutoff_binding = _mapping(inputs["cutoff_primitives"], "cutoff input")
    author_binding = _mapping(inputs["author_history"], "author input")
    require(work_totals["rows"] == cutoff_binding["expected_rows"], "work row closure failed")
    require(author_totals["rows"] == author_binding["expected_rows"], "author event row closure failed")
    require(
        author_totals["history_usable_rows"]
        == author_binding["expected_history_usable_rows"],
        "history-usable author event closure failed",
    )
    field_year = _field_year_table(work_cells, author_cells)
    attention = _attention_table(field_year)
    years = [
        int(value)
        for value in field_year["publication_year"].to_pylist()
        if value is not None
    ]
    deferred = _mapping(config.get("deferred_required_metrics"), "deferred metrics")
    require(
        all(str(_mapping(value, key).get("state")).startswith("NOT_RUN") for key, value in deferred.items()),
        "deferred component is improperly promoted",
    )
    output_contract = _mapping(config.get("output_contract"), "output contract")
    require(output_contract.get("passing_component_status") == STATUS, "component status mismatch")
    require(output_contract.get("s0_gate_pass_after_core_component") is False, "core passes S0 gate")
    require(output_contract.get("d0_authorized_after_core_component") is False, "core authorizes D0")
    require(output_contract.get("gpu_used") is False, "GPU boundary differs")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    os.chmod(temporary, 0o700)
    try:
        field_artifact = _parquet_sha(field_year, temporary / "field_year_core.parquet")
        attention_artifact = _parquet_sha(
            attention, temporary / "attention_trajectory_core.parquet"
        )
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA,
            "status": STATUS,
            "scope": {
                "core_component_pass": True,
                "s0_gate_pass": False,
                "d0_authorized": False,
                "model_fitting_authorized": False,
                "gpu_used": False,
                "field_count": 26,
                "source_snapshot_date": structural["source_snapshot_date"],
            },
            "g_minus_1a_parent": parent,
            "input_verification": {
                "cutoff_primitives": cutoff_validation,
                "author_history": author_validation,
                "all_selected_artifacts_rehashed_exactly": True,
                "input_roots_mutated": False,
            },
            "work_accounting": work_totals,
            "author_event_accounting": author_totals,
            "field_year_inventory": {
                "schema_version": FIELD_YEAR_SCHEMA,
                "rows": len(field_year),
                "observed_year_min": min(years),
                "observed_year_max": max(years),
                "null_year_cells": sum(value is None for value in field_year["publication_year"].to_pylist()),
                "artifact": field_artifact,
            },
            "attention_trajectory_inventory": {
                "schema_version": ATTENTION_SCHEMA,
                "rows": len(attention),
                "resolutions_years": structural["raw_attention_trailing_resolutions_years"],
                "ranked_or_selected": False,
                "artifact": attention_artifact,
            },
            "descriptive_profiles": {
                "team_caps": structural["descriptive_team_cap_profiles"],
                "team_caps_ranked_or_selected": False,
                "citation_calendar_support_years": structural[
                    "citation_calendar_support_years"
                ],
                "calendar_support_labeled_outcome_success": False,
            },
            "deferred_required_metrics": deferred,
            "forbidden_metrics_emitted": [],
            "gate_status": {
                "g_minus_1a": "PASS_BOUNDED",
                "s0_threshold_free_inventory": "PARTIAL_CORE_PASS_DEFERRED_COMPONENTS_NOT_RUN",
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
            },
            "claim_boundary": config["claim_boundary"],
        }
        fingerprint = hashlib.sha256(_canonical_bytes(report)).hexdigest()
        report["provenance"]["result_fingerprint_sha256"] = fingerprint
        report_path = temporary / "s0_core_inventory.json"
        _write_json(report_path, report)
        markdown_path = temporary / "s0_core_inventory.md"
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        os.chmod(markdown_path, 0o600)
        artifacts = []
        for path in sorted(temporary.iterdir()):
            if path.name == "s0_core_inventory.outputs.json":
                continue
            artifacts.append(
                {
                    "relative_path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        receipt = {
            "schema_version": "pivot_feedback_s0_threshold_free_core_inventory_outputs.v1",
            "status": STATUS,
            "core_component_pass": True,
            "s0_gate_pass": False,
            "d0_authorized": False,
            "completion_contract": "RECEIPT_WRITTEN_LAST_ATOMIC_DIRECTORY_RENAME",
            "artifact_count": len(artifacts),
            "artifact_tree_sha256": hashlib.sha256(_canonical_bytes(artifacts)).hexdigest(),
            "artifacts": artifacts,
            "result_fingerprint_sha256": fingerprint,
        }
        _write_json(temporary / "s0_core_inventory.outputs.json", receipt)
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
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "experiments/configs/pivot_feedback_s0_threshold_free_inventory_v1.yaml",
    )
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build(
            args.config,
            args.cutoff_root,
            args.author_root,
            args.output,
            workers=args.workers,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    except (S0InventoryError, OSError, ValueError, pa.ArrowException) as error:
        print(json.dumps({"status": "BLOCKED_S0_CORE_INVENTORY", "error": str(error)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
