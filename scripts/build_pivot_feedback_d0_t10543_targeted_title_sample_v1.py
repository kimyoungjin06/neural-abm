#!/usr/bin/env python3
"""Build the exact local 96-title T10543 targeted sample v1."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10543_targeted_title_sample_v1.yaml"
)
DEFAULT_WORK_ROOT = Path("/ssd/pivot-feedback-cutoff-primitives-v0-full-20260723")
DEFAULT_EVENT_ROOT = Path("/ssd/openalex-g1-author-history-v3-full-20260722")
DEFAULT_OPENALEX_ROOT = Path(
    "/home/kimyoungjin06/Desktop/HDD/raid_backup_20260624/data/OpenAlex/"
    "parquet_exports/openalex_works_20260330_repairreplay_20260410_190630"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-t10543-targeted-title-sample-v1-20260729"
)
PARTITIONS = 4096
CONFIG_SCHEMA = "pivot_feedback_d0_t10543_targeted_title_sample_config.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_t10543_targeted_title_sample_outputs.v1"
SAMPLE_SCHEMA = "pivot_feedback_d0_t10543_targeted_title_sample.v1"
STATUS = "PASS_EXACT_T10543_96_TITLE_LOCAL_SAMPLE"


class T10543TargetedSampleBuildError(RuntimeError):
    """Fail-closed T10543 targeted-sample build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise T10543TargetedSampleBuildError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise T10543TargetedSampleBuildError(f"invalid YAML: {error}") from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise T10543TargetedSampleBuildError(f"invalid JSON: {error}") from error
    return _mapping(value, path.name)


def _repo_file(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _dotted(document: Mapping[str, Any], key: str) -> Any:
    value: Any = document
    for part in key.split("."):
        value = _mapping(value, part)[part]
    return value


def _validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config = _read_yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status") == "FROZEN_EXACT_T10543_96_TITLE_LOCAL_SAMPLE_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_path = _repo_file(parent)
    parent_document = _read_yaml(parent_path)
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    target = _mapping(config.get("target"), "target")
    require(
        (target["field_id"], target["field_candidate_rank"], target["topic_id"], target["entry_year"])
        == (27, 2, 10543, 1993),
        "target differs",
    )
    sampling = _mapping(config.get("sampling"), "sampling")
    require(
        (sampling["algorithm"], sampling["rows_per_window"], sampling["total_rows"])
        == ("SHA256_LOWEST_BY_CANDIDATE_AND_WINDOW", 32, 96),
        "sampling contract",
    )
    boundary = _mapping(config.get("execution_boundary"), "execution boundary")
    for key in (
        "general_raw_title_scan",
        "full_candidate_title_copy",
        "network_accessed",
        "credentials_accessed",
        "api_requests_created",
        "api_requests_submitted",
        "content_disposition_created",
        "selection_certificate_issued",
        "deterministic_reselection_executed",
        "outcomes_read",
    ):
        require(boundary[key] is False, f"unsafe boundary: {key}")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "general_raw_title_scan",
        "full_candidate_title_copy",
        "api_request_creation",
        "api_submission",
        "production_content_disposition",
        "selection_certificate",
        "deterministic_reselection",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(authorization[key] == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return config, {"parent_sha256": _sha256(parent_path)}


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw in receipt["artifacts"]:
        item = _mapping(raw, "receipt artifact")
        relative = str(item["relative_path"])
        require(relative not in output, "duplicate receipt artifact")
        output[relative] = item
    return output


def _validate_partition_root(
    root: Path,
    binding: Mapping[str, Any],
    *,
    workers: int,
) -> tuple[list[Path], str]:
    root = root.resolve(strict=True)
    require(root.name == binding["expected_root_basename"], "input root basename")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "input receipt SHA")
    receipt = _read_json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "receipt schema")
    require(receipt.get("status") == binding["required_status"], "receipt status")
    artifacts = _artifact_map(receipt)
    prefix = str(binding["artifact_prefix"])
    declared = [artifacts[f"{prefix}part-{index:04d}.parquet"] for index in range(PARTITIONS)]
    paths = [root / str(item["relative_path"]) for item in declared]

    def verify(pair: tuple[Path, Mapping[str, Any]]) -> tuple[int, str]:
        path, item = pair
        require(path.is_file() and not path.is_symlink(), "input partition")
        require(path.stat().st_size == item["bytes"], f"partition bytes: {path.name}")
        digest = _sha256(path)
        require(digest == item["sha256"], f"partition SHA: {path.name}")
        return int(pq.ParquetFile(path).metadata.num_rows), digest

    with ThreadPoolExecutor(max_workers=workers) as executor:
        verified = list(executor.map(verify, zip(paths, declared, strict=True)))
    require(sum(row_count for row_count, _ in verified) == binding["rows"], "input rows")
    tree = hashlib.sha256(_canonical_json_bytes(verified)).hexdigest()
    return paths, tree


def _window_code(year: int) -> int:
    if year <= 1986:
        return 0
    if year <= 1989:
        return 1
    require(year <= 1992, "year outside target windows")
    return 2


def _scan_work_candidates(
    pair: tuple[int, Path],
    *,
    included_types: np.ndarray,
    allowed_dispositions: np.ndarray,
) -> dict[str, Any]:
    index, work_path = pair
    work = pq.read_table(
        work_path,
        columns=[
            "work_id_numeric",
            "publication_year",
            "temporal_disposition",
            "work_type_code",
            "is_paratext",
            "is_retracted",
            "declared_authors_count",
            "primary_topic_id",
            "primary_field_id",
        ],
    )
    work_id = work["work_id_numeric"].combine_chunks().to_numpy(zero_copy_only=False)
    require(bool(np.all(work_id % PARTITIONS == index)), "work partition")
    year_array = work["publication_year"].combine_chunks()
    year_valid = year_array.is_valid().to_numpy(zero_copy_only=False)
    year = year_array.fill_null(0).to_numpy(zero_copy_only=False).astype(np.uint16, copy=False)
    disposition = work["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
    type_code = work["work_type_code"].combine_chunks().to_numpy(zero_copy_only=False)
    is_paratext = work["is_paratext"].combine_chunks().to_numpy(zero_copy_only=False)
    retracted_array = work["is_retracted"].combine_chunks()
    retracted_valid = retracted_array.is_valid().to_numpy(zero_copy_only=False)
    is_retracted = retracted_array.fill_null(False).to_numpy(zero_copy_only=False)
    team = work["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)
    topic_array = work["primary_topic_id"].combine_chunks()
    topic_valid = topic_array.is_valid().to_numpy(zero_copy_only=False)
    topic = topic_array.fill_null(0).to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
    field_array = work["primary_field_id"].combine_chunks()
    field_valid = field_array.is_valid().to_numpy(zero_copy_only=False)
    field = field_array.fill_null(0).to_numpy(zero_copy_only=False).astype(np.uint8, copy=False)
    target_mask = (
        year_valid
        & topic_valid
        & field_valid
        & (topic == 10543)
        & (field == 27)
        & (year > 0)
        & (year <= 1992)
        & np.isin(disposition, allowed_dispositions)
        & np.isin(type_code, included_types)
        & ~is_paratext
        & ~(retracted_valid & is_retracted)
    )
    selected_indices = np.flatnonzero(target_mask)
    rows = [
        {
            "work_numeric_id": int(work_id[source_index]),
            "publication_year": int(year[source_index]),
            "temporal_disposition": int(disposition[source_index]),
            "work_type_code": int(type_code[source_index]),
            "declared_authors_count": int(team[source_index]),
        }
        for source_index in selected_indices.tolist()
    ]
    return {
        "rows": rows,
        "work_rows": len(work),
        "eligible_prelink_rows": len(selected_indices),
    }


def _scan_event_links(
    pair: tuple[int, Path],
    *,
    candidate_ids: np.ndarray,
    candidate_year: np.ndarray,
    candidate_disposition: np.ndarray,
) -> dict[str, Any]:
    index, event_path = pair
    events = pq.read_table(
        event_path,
        columns=[
            "author_numeric_id",
            "work_numeric_id",
            "publication_year",
            "temporal_disposition",
            "primary_field_id",
        ],
    )
    author = events["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    require(bool(np.all(author % PARTITIONS == index)), "author partition")
    event_work = events["work_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    positions = np.searchsorted(candidate_ids, event_work)
    in_range = positions < len(candidate_ids)
    matched = np.zeros(len(events), dtype=bool)
    matched[in_range] = candidate_ids[positions[in_range]] == event_work[in_range]
    event_year = events["publication_year"].combine_chunks().to_numpy(zero_copy_only=False)
    event_disposition = events["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
    event_field = events["primary_field_id"].combine_chunks().to_numpy(zero_copy_only=False)
    safe_positions = np.minimum(positions, len(candidate_ids) - 1)
    matched &= event_year == candidate_year[safe_positions]
    matched &= event_disposition == candidate_disposition[safe_positions]
    matched &= event_field == 27
    return {
        "matched_work_ids": event_work[matched],
        "event_rows": len(events),
    }


def _select_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: str,
    per_window: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    for raw in rows:
        row = dict(raw)
        window = _window_code(int(row["publication_year"]))
        preimage = (
            f"{seed}\0{27}\0{10543}\0{1993}\0{window}\0"
            f"{int(row['work_numeric_id'])}"
        )
        row["window_code"] = window
        row["selection_sha256"] = hashlib.sha256(preimage.encode()).hexdigest()
        groups[window].append(row)
    selected: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"window_pool_rows": {}, "selected_rows": {}}
    for window in (0, 1, 2):
        group = sorted(
            groups[window],
            key=lambda row: (row["selection_sha256"], row["work_numeric_id"]),
        )
        require(len(group) >= per_window, f"insufficient target window: {window}")
        chosen = group[:per_window]
        selected.extend(chosen)
        summary["window_pool_rows"][str(window)] = len(group)
        summary["selected_rows"][str(window)] = len(chosen)
    require(len(selected) == 96, "selected rows")
    require(len({row["work_numeric_id"] for row in selected}) == 96, "duplicate sample")
    selected.sort(key=lambda row: (row["window_code"], row["selection_sha256"]))
    return selected, summary


def _validate_raw_source(
    config: Mapping[str, Any],
    openalex_root: Path,
) -> tuple[list[Path], list[str], int]:
    source = _mapping(config.get("current_snapshot_source"), "source")
    root = openalex_root.resolve(strict=True)
    require(root.name == source["required_root_id"], "source root ID")
    main = root / str(source["main_table_directory"])
    files = sorted(main.glob("part-*.parquet"), key=lambda path: int(path.stem.split("-")[1]))
    require(len(files) == source["expected_main_file_count"], "main file count")
    footer_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    require(footer_rows == source["expected_main_footer_rows"], "main footer rows")
    columns = [str(value) for value in source["required_columns"]]
    require(set(columns) <= set(pq.ParquetFile(files[0]).schema_arrow.names), "source columns")
    return files, columns, footer_rows


def _lookup_selected(
    files: Sequence[Path],
    columns: Sequence[str],
    selected: Sequence[Mapping[str, Any]],
    *,
    workers: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    by_openalex = {
        f"https://openalex.org/W{int(row['work_numeric_id'])}": dict(row)
        for row in selected
    }
    values = pa.array(sorted(by_openalex), type=pa.string())

    def scan_file(pair: tuple[int, Path]) -> tuple[list[dict[str, Any]], int, int]:
        file_index, path = pair
        parquet = pq.ParquetFile(path)
        found: list[dict[str, Any]] = []
        id_row_groups = 0
        content_row_groups = 0
        for row_group in range(parquet.metadata.num_row_groups):
            id_row_groups += 1
            identifiers = parquet.read_row_group(row_group, columns=["id"])["id"]
            mask = pc.fill_null(pc.is_in(identifiers, value_set=values), False)
            if not pc.any(mask).as_py():
                continue
            content_row_groups += 1
            table = parquet.read_row_group(row_group, columns=list(columns))
            for row in table.filter(mask).to_pylist():
                row["source_file_index"] = file_index
                row["source_row_group"] = row_group
                found.append(row)
        return found, id_row_groups, content_row_groups

    with ThreadPoolExecutor(max_workers=min(workers, 4)) as executor:
        results = list(executor.map(scan_file, enumerate(files)))
    found: dict[int, dict[str, Any]] = {}
    for rows, _, _ in results:
        for row in rows:
            work_id = int(str(row["id"]).rsplit("W", 1)[1])
            require(work_id not in found, f"duplicate selected ID: {work_id}")
            found[work_id] = row
    require(len(found) == 96, "selected IDs not found exactly once")
    return found, {
        "main_files_scanned": len(files),
        "id_row_groups_scanned": sum(item[1] for item in results),
        "content_row_groups_read": sum(item[2] for item in results),
        "selected_ids_found": len(found),
    }


def _sample_table(
    selected: Sequence[Mapping[str, Any]],
    found: Mapping[int, Mapping[str, Any]],
) -> pa.Table:
    type_names = {
        1: "article",
        2: "book",
        3: "book-chapter",
        4: "book-section",
        7: "dissertation",
        16: "preprint",
        18: "report",
        21: "review",
    }
    output: list[dict[str, Any]] = []
    for selected_row in selected:
        work_id = int(selected_row["work_numeric_id"])
        raw = found[work_id]
        raw_year = int(raw["publication_year"])
        require(raw_year == selected_row["publication_year"], "publication year")
        require(raw["primary_topic__id"] == "https://openalex.org/T10543", "topic")
        require(raw["primary_topic__field__id"] == "https://openalex.org/fields/27", "field")
        require(raw["type"] == type_names[int(selected_row["work_type_code"])], "work type")
        output.append(
            {
                "field_id": 27,
                "field_candidate_rank": 2,
                "topic_id": 10543,
                "entry_year": 1993,
                "window_code": int(selected_row["window_code"]),
                "work_numeric_id": work_id,
                "openalex_id": str(raw["id"]),
                "selection_sha256": str(selected_row["selection_sha256"]),
                "publication_year": raw_year,
                "temporal_disposition": int(selected_row["temporal_disposition"]),
                "work_type_code": int(selected_row["work_type_code"]),
                "work_type_name": str(raw["type"]),
                "declared_authors_count": int(selected_row["declared_authors_count"]),
                "matching_author_event_rows": int(selected_row["matching_author_event_rows"]),
                "title": None if raw["title"] is None else str(raw["title"]),
                "language": None if raw["language"] is None else str(raw["language"]),
                "observed_authors_count": None if raw["authors_count"] is None else int(raw["authors_count"]),
                "is_paratext": None if raw["is_paratext"] is None else bool(raw["is_paratext"]),
                "is_retracted": None if raw["is_retracted"] is None else bool(raw["is_retracted"]),
                "primary_topic_name": None
                if raw["primary_topic__display_name"] is None
                else str(raw["primary_topic__display_name"]),
                "primary_topic_score": None
                if raw["primary_topic__score"] is None
                else float(raw["primary_topic__score"]),
                "primary_subfield_id": None
                if raw["primary_topic__subfield__id"] is None
                else str(raw["primary_topic__subfield__id"]),
                "source_id": None
                if raw["primary_location__source__id"] is None
                else str(raw["primary_location__source__id"]),
                "source_name": None
                if raw["primary_location__source__display_name"] is None
                else str(raw["primary_location__source__display_name"]),
                "source_file_index": int(raw["source_file_index"]),
                "source_row_group": int(raw["source_row_group"]),
            }
        )
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("field_candidate_rank", pa.uint32(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("entry_year", pa.uint16(), nullable=False),
            pa.field("window_code", pa.uint8(), nullable=False),
            pa.field("work_numeric_id", pa.uint64(), nullable=False),
            pa.field("openalex_id", pa.string(), nullable=False),
            pa.field("selection_sha256", pa.string(), nullable=False),
            pa.field("publication_year", pa.uint16(), nullable=False),
            pa.field("temporal_disposition", pa.uint8(), nullable=False),
            pa.field("work_type_code", pa.uint8(), nullable=False),
            pa.field("work_type_name", pa.string(), nullable=False),
            pa.field("declared_authors_count", pa.uint16(), nullable=False),
            pa.field("matching_author_event_rows", pa.uint32(), nullable=False),
            pa.field("title", pa.string()),
            pa.field("language", pa.string()),
            pa.field("observed_authors_count", pa.int64()),
            pa.field("is_paratext", pa.bool_()),
            pa.field("is_retracted", pa.bool_()),
            pa.field("primary_topic_name", pa.string()),
            pa.field("primary_topic_score", pa.float64()),
            pa.field("primary_subfield_id", pa.string()),
            pa.field("source_id", pa.string()),
            pa.field("source_name", pa.string()),
            pa.field("source_file_index", pa.uint16(), nullable=False),
            pa.field("source_row_group", pa.uint16(), nullable=False),
        ],
        metadata={b"schema_version": SAMPLE_SCHEMA.encode()},
    )
    return pa.Table.from_pylist(output, schema=schema)


def _write_private(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
    path.chmod(0o600)


def _write_parquet(table: pa.Table, path: Path) -> dict[str, Any]:
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    path.chmod(0o600)
    return {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "sha256": _sha256(path),
    }


def _markdown(pool: Mapping[str, Any], scan: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# T10543 exact targeted title sample",
            "",
            f"Status: `{STATUS}`",
            "",
            f"- linked earlier pool: {pool['window_pool_rows']['0']:,}",
            f"- linked baseline pool: {pool['window_pool_rows']['1']:,}",
            f"- linked signal pool: {pool['window_pool_rows']['2']:,}",
            "- selected rows: 32 + 32 + 32 = 96",
            f"- raw main files scanned for exact IDs: {scan['main_files_scanned']}",
            f"- content row groups read: {scan['content_row_groups_read']}",
            "",
            "No network, credential, API, outcome, disposition, certificate,",
            "or deterministic reselection operation occurred.",
            "",
        ]
    )


def build(
    *,
    config_path: Path = DEFAULT_CONFIG,
    work_root: Path = DEFAULT_WORK_ROOT,
    event_root: Path = DEFAULT_EVENT_ROOT,
    openalex_root: Path = DEFAULT_OPENALEX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    workers: int = 16,
) -> dict[str, Any]:
    require(workers >= 1, "workers")
    config_path = config_path.resolve(strict=True)
    require(not output_root.exists(), "output root exists")
    require(output_root.parent.resolve(strict=True) == Path("/ssd"), "output parent")
    config, bindings = _validate_config(config_path)
    normalized = _mapping(config.get("normalized_inputs"), "normalized inputs")
    work_paths, work_tree = _validate_partition_root(
        work_root,
        _mapping(normalized.get("cutoff_work_metadata"), "work input"),
        workers=workers,
    )
    event_paths, event_tree = _validate_partition_root(
        event_root,
        _mapping(normalized.get("author_history_events"), "event input"),
        workers=workers,
    )
    included_types = np.array(config["target"]["included_work_type_codes"], dtype=np.uint8)
    allowed_dispositions = np.array(
        config["target"]["allowed_temporal_dispositions"],
        dtype=np.uint8,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        work_parts = list(
            executor.map(
                lambda pair: _scan_work_candidates(
                    pair,
                    included_types=included_types,
                    allowed_dispositions=allowed_dispositions,
                ),
                enumerate(work_paths),
            )
        )
    prelink_rows = [row for part in work_parts for row in part["rows"]]
    require(prelink_rows, "no eligible work candidates")
    prelink_rows.sort(key=lambda row: int(row["work_numeric_id"]))
    candidate_ids = np.array(
        [row["work_numeric_id"] for row in prelink_rows],
        dtype=np.uint64,
    )
    require(
        len(candidate_ids) == len(np.unique(candidate_ids)),
        "duplicate eligible work candidate",
    )
    candidate_year = np.array(
        [row["publication_year"] for row in prelink_rows],
        dtype=np.uint16,
    )
    candidate_disposition = np.array(
        [row["temporal_disposition"] for row in prelink_rows],
        dtype=np.uint8,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        event_parts = list(
            executor.map(
                lambda pair: _scan_event_links(
                    pair,
                    candidate_ids=candidate_ids,
                    candidate_year=candidate_year,
                    candidate_disposition=candidate_disposition,
                ),
                enumerate(event_paths),
            )
        )
    matched_arrays = [
        part["matched_work_ids"]
        for part in event_parts
        if len(part["matched_work_ids"]) > 0
    ]
    require(matched_arrays, "no matching author events")
    matched_ids, matching_event_counts = np.unique(
        np.concatenate(matched_arrays),
        return_counts=True,
    )
    prelink_by_id = {
        int(row["work_numeric_id"]): row
        for row in prelink_rows
    }
    pool_rows = [
        {
            **prelink_by_id[int(work_id)],
            "matching_author_event_rows": int(event_count),
        }
        for work_id, event_count in zip(
            matched_ids.tolist(),
            matching_event_counts.tolist(),
            strict=True,
        )
    ]
    selected, pool = _select_sample(
        pool_rows,
        seed=str(config["sampling"]["seed"]),
        per_window=int(config["sampling"]["rows_per_window"]),
    )
    pool.update(
        {
            "schema_version": "pivot_feedback_d0_t10543_candidate_pool_summary.v1",
            "field_id": 27,
            "topic_id": 10543,
            "entry_year": 1993,
            "normalized_work_rows_scanned": sum(
                part["work_rows"] for part in work_parts
            ),
            "normalized_event_rows_scanned": sum(
                part["event_rows"] for part in event_parts
            ),
            "eligible_prelink_rows": sum(
                part["eligible_prelink_rows"] for part in work_parts
            ),
            "author_linked_pool_rows": len(pool_rows),
            "recent_six_year_author_linked_rows": (
                int(pool["window_pool_rows"]["1"])
                + int(pool["window_pool_rows"]["2"])
            ),
        }
    )
    raw_files, columns, footer_rows = _validate_raw_source(config, openalex_root)
    found, raw_scan = _lookup_selected(
        raw_files,
        columns,
        selected,
        workers=workers,
    )
    table = _sample_table(selected, found)
    require(len(table) == 96, "sample table rows")
    pool["selected_title_rows"] = len(table)
    pool["missing_title_rows"] = int(pc.sum(pc.is_null(table["title"])).as_py())
    pool["raw_main_footer_rows"] = footer_rows
    contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == contract["root_basename"], "output basename")

    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(output_root.parent))
    )
    temporary.chmod(0o700)
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        artifacts["sample"] = _write_parquet(
            table,
            temporary / str(contract["sample_basename"]),
        )
        pool_path = temporary / str(contract["pool_summary_basename"])
        _write_private(pool_path, _canonical_json_bytes(pool))
        artifacts["pool_summary"] = {
            "basename": pool_path.name,
            "bytes": pool_path.stat().st_size,
            "sha256": _sha256(pool_path),
        }
        report_path = temporary / str(contract["report_basename"])
        _write_private(report_path, _markdown(pool, raw_scan).encode("utf-8"))
        artifacts["report"] = {
            "basename": report_path.name,
            "bytes": report_path.stat().st_size,
            "sha256": _sha256(report_path),
        }
        receipt = {
            "schema_version": OUTPUT_SCHEMA,
            "status": STATUS,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_relative_path": config_path.relative_to(REPO_ROOT).as_posix(),
            "config_sha256": _sha256(config_path),
            "producer_relative_path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "repository_bindings": bindings,
            "normalized_input_evidence": {
                "work_root_basename": work_root.name,
                "work_selected_tree_sha256": work_tree,
                "event_root_basename": event_root.name,
                "event_selected_tree_sha256": event_tree,
            },
            "pool": pool,
            "raw_scan": raw_scan,
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_json_bytes(artifacts)
            ).hexdigest(),
            "target_topic_ids_read": [10543],
            "selected_title_ids_read": 96,
            "general_raw_title_scan": False,
            "full_candidate_title_copy": False,
            "network_accessed": False,
            "credentials_accessed": False,
            "api_requests_created": 0,
            "api_requests_submitted": 0,
            "api_cost_usd": "0.000000",
            "content_disposition_created": False,
            "selection_certificate_issued": False,
            "deterministic_reselection_executed": False,
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
        }
        receipt_path = temporary / str(contract["receipt_basename"])
        _write_private(receipt_path, _canonical_json_bytes(receipt))
        os.replace(temporary, output_root)
        temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--openalex-root", type=Path, default=DEFAULT_OPENALEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build(
            config_path=args.config,
            work_root=args.work_root,
            event_root=args.event_root,
            openalex_root=args.openalex_root,
            output_root=args.output_root,
            workers=args.workers,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        T10543TargetedSampleBuildError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_D0_T10543_TARGETED_SAMPLE_V1",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
