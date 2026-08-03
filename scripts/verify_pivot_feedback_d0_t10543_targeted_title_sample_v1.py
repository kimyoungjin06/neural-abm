#!/usr/bin/env python3
"""Independently replay and verify the exact T10543 targeted title sample."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
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
DEFAULT_REPORT = Path(
    "/ssd/pivot-feedback-d0-t10543-targeted-title-sample-v1-20260729."
    "verification-v1.json"
)
PARTITIONS = 4096
CONFIG_SCHEMA = "pivot_feedback_d0_t10543_targeted_title_sample_config.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_t10543_targeted_title_sample_outputs.v1"
SAMPLE_SCHEMA = "pivot_feedback_d0_t10543_targeted_title_sample.v1"
VERIFY_SCHEMA = "pivot_feedback_d0_t10543_targeted_title_sample_verification.v1"
VERIFY_STATUS = "PASS_INDEPENDENT_EXACT_T10543_96_TITLE_LOCAL_SAMPLE"


class T10543TargetedSampleVerificationError(RuntimeError):
    """Fail-closed T10543 targeted-sample verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise T10543TargetedSampleVerificationError(message)


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
        raise T10543TargetedSampleVerificationError(
            f"invalid YAML: {error}"
        ) from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise T10543TargetedSampleVerificationError(
            f"invalid JSON: {error}"
        ) from error
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


def _validate_config(config_path: Path) -> dict[str, Any]:
    config = _read_yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status") == "FROZEN_EXACT_T10543_96_TITLE_LOCAL_SAMPLE_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_document = _read_yaml(_repo_file(parent))
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    target = _mapping(config.get("target"), "target")
    require(
        (
            target["field_id"],
            target["field_candidate_rank"],
            target["topic_id"],
            target["entry_year"],
        )
        == (27, 2, 10543, 1993),
        "target",
    )
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
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return config


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw in receipt["artifacts"]:
        artifact = _mapping(raw, "input artifact")
        relative = str(artifact["relative_path"])
        require(relative not in output, "duplicate input artifact")
        output[relative] = artifact
    return output


def _partition_inputs(
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
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "input schema")
    require(receipt.get("status") == binding["required_status"], "input status")
    artifacts = _artifact_map(receipt)
    prefix = str(binding["artifact_prefix"])
    declared = [
        artifacts[f"{prefix}part-{index:04d}.parquet"]
        for index in range(PARTITIONS)
    ]
    paths = [root / str(item["relative_path"]) for item in declared]

    def verify(pair: tuple[Path, Mapping[str, Any]]) -> tuple[int, str]:
        path, item = pair
        require(path.is_file() and not path.is_symlink(), "input partition")
        require(path.stat().st_size == item["bytes"], f"input bytes: {path.name}")
        digest = _sha256(path)
        require(digest == item["sha256"], f"input SHA: {path.name}")
        return int(pq.ParquetFile(path).metadata.num_rows), digest

    with ThreadPoolExecutor(max_workers=workers) as executor:
        verified = list(executor.map(verify, zip(paths, declared, strict=True)))
    require(sum(rows for rows, _ in verified) == binding["rows"], "input row count")
    return paths, hashlib.sha256(_canonical_json_bytes(verified)).hexdigest()


def _verify_output(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    config_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    contract = _mapping(config.get("output_contract"), "output contract")
    output_root = output_root.resolve(strict=True)
    require(output_root.name == contract["root_basename"], "output basename")
    require(output_root.is_dir() and not output_root.is_symlink(), "output root")
    require(output_root.stat().st_mode & 0o777 == 0o700, "output mode")
    paths = {
        "sample": output_root / str(contract["sample_basename"]),
        "pool_summary": output_root / str(contract["pool_summary_basename"]),
        "report": output_root / str(contract["report_basename"]),
    }
    receipt_path = output_root / str(contract["receipt_basename"])
    require(
        {path.name for path in output_root.iterdir()}
        == {path.name for path in paths.values()} | {receipt_path.name},
        "output artifact set",
    )
    receipt = _read_json(receipt_path)
    require(receipt.get("schema_version") == OUTPUT_SCHEMA, "output schema")
    require(
        receipt.get("status") == "PASS_EXACT_T10543_96_TITLE_LOCAL_SAMPLE",
        "output status",
    )
    require(receipt.get("config_sha256") == config_sha256, "config SHA")
    for key, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"artifact: {key}")
        require(path.stat().st_mode & 0o777 == 0o600, f"artifact mode: {key}")
        artifact = _mapping(receipt["artifacts"][key], f"artifact receipt: {key}")
        require(path.stat().st_size == artifact["bytes"], f"artifact bytes: {key}")
        require(_sha256(path) == artifact["sha256"], f"artifact SHA: {key}")
    require(receipt_path.stat().st_mode & 0o777 == 0o600, "receipt mode")
    require(
        hashlib.sha256(_canonical_json_bytes(receipt["artifacts"])).hexdigest()
        == receipt["artifact_tree_sha256"],
        "artifact tree",
    )
    return receipt, paths


def _work_candidates(
    pair: tuple[int, Path],
    *,
    included_types: np.ndarray,
    dispositions: np.ndarray,
) -> tuple[list[dict[str, int]], int]:
    index, path = pair
    table = pq.read_table(
        path,
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
    work_id = table["work_id_numeric"].combine_chunks().to_numpy(zero_copy_only=False)
    require(bool(np.all(work_id % PARTITIONS == index)), "work partition")
    year_column = table["publication_year"].combine_chunks()
    year_valid = year_column.is_valid().to_numpy(zero_copy_only=False)
    year = year_column.fill_null(0).to_numpy(zero_copy_only=False)
    disposition = table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
    type_code = table["work_type_code"].combine_chunks().to_numpy(zero_copy_only=False)
    paratext = table["is_paratext"].combine_chunks().to_numpy(zero_copy_only=False)
    retracted_column = table["is_retracted"].combine_chunks()
    retracted_valid = retracted_column.is_valid().to_numpy(zero_copy_only=False)
    retracted = retracted_column.fill_null(False).to_numpy(zero_copy_only=False)
    team = table["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)
    topic_column = table["primary_topic_id"].combine_chunks()
    topic_valid = topic_column.is_valid().to_numpy(zero_copy_only=False)
    topic = topic_column.fill_null(0).to_numpy(zero_copy_only=False)
    field_column = table["primary_field_id"].combine_chunks()
    field_valid = field_column.is_valid().to_numpy(zero_copy_only=False)
    field = field_column.fill_null(0).to_numpy(zero_copy_only=False)
    mask = (
        year_valid
        & topic_valid
        & field_valid
        & (topic == 10543)
        & (field == 27)
        & (year > 0)
        & (year <= 1992)
        & np.isin(disposition, dispositions)
        & np.isin(type_code, included_types)
        & ~paratext
        & ~(retracted_valid & retracted)
    )
    selected = np.flatnonzero(mask)
    return (
        [
            {
                "work_numeric_id": int(work_id[row]),
                "publication_year": int(year[row]),
                "temporal_disposition": int(disposition[row]),
                "work_type_code": int(type_code[row]),
                "declared_authors_count": int(team[row]),
            }
            for row in selected.tolist()
        ],
        len(table),
    )


def _linked_work_ids(
    pair: tuple[int, Path],
    *,
    candidate_ids: np.ndarray,
    candidate_year: np.ndarray,
    candidate_disposition: np.ndarray,
) -> tuple[np.ndarray, int]:
    index, path = pair
    table = pq.read_table(
        path,
        columns=[
            "author_numeric_id",
            "work_numeric_id",
            "publication_year",
            "temporal_disposition",
            "primary_field_id",
        ],
    )
    author = table["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    require(bool(np.all(author % PARTITIONS == index)), "author partition")
    event_work = table["work_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    positions = np.searchsorted(candidate_ids, event_work)
    in_range = positions < len(candidate_ids)
    matched = np.zeros(len(table), dtype=bool)
    matched[in_range] = candidate_ids[positions[in_range]] == event_work[in_range]
    safe_positions = np.minimum(positions, len(candidate_ids) - 1)
    event_year = table["publication_year"].combine_chunks().to_numpy(zero_copy_only=False)
    event_disposition = (
        table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    event_field = table["primary_field_id"].combine_chunks().to_numpy(zero_copy_only=False)
    matched &= event_year == candidate_year[safe_positions]
    matched &= event_disposition == candidate_disposition[safe_positions]
    matched &= event_field == 27
    return event_work[matched], len(table)


def _window(year: int) -> int:
    if year <= 1986:
        return 0
    if year <= 1989:
        return 1
    require(year <= 1992, "window year")
    return 2


def _replay_normalized(
    config: Mapping[str, Any],
    work_paths: Sequence[Path],
    event_paths: Sequence[Path],
    *,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = _mapping(config.get("target"), "target")
    included_types = np.array(target["included_work_type_codes"], dtype=np.uint8)
    dispositions = np.array(target["allowed_temporal_dispositions"], dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        work_parts = list(
            executor.map(
                lambda pair: _work_candidates(
                    pair,
                    included_types=included_types,
                    dispositions=dispositions,
                ),
                enumerate(work_paths),
            )
        )
    candidates = [row for rows, _ in work_parts for row in rows]
    candidates.sort(key=lambda row: row["work_numeric_id"])
    require(candidates, "candidate pool")
    candidate_ids = np.array(
        [row["work_numeric_id"] for row in candidates],
        dtype=np.uint64,
    )
    require(len(candidate_ids) == len(np.unique(candidate_ids)), "candidate IDs")
    candidate_year = np.array(
        [row["publication_year"] for row in candidates],
        dtype=np.uint16,
    )
    candidate_disposition = np.array(
        [row["temporal_disposition"] for row in candidates],
        dtype=np.uint8,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        event_parts = list(
            executor.map(
                lambda pair: _linked_work_ids(
                    pair,
                    candidate_ids=candidate_ids,
                    candidate_year=candidate_year,
                    candidate_disposition=candidate_disposition,
                ),
                enumerate(event_paths),
            )
        )
    nonempty = [values for values, _ in event_parts if len(values)]
    require(nonempty, "author linkage")
    linked_ids, linked_counts = np.unique(np.concatenate(nonempty), return_counts=True)
    by_id = {row["work_numeric_id"]: row for row in candidates}
    linked = [
        {**by_id[int(work_id)], "matching_author_event_rows": int(count)}
        for work_id, count in zip(
            linked_ids.tolist(),
            linked_counts.tolist(),
            strict=True,
        )
    ]
    seed = str(config["sampling"]["seed"])
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in linked:
        window = _window(row["publication_year"])
        preimage = (
            f"{seed}\0{27}\0{10543}\0{1993}\0{window}\0"
            f"{row['work_numeric_id']}"
        )
        groups[window].append(
            {
                **row,
                "window_code": window,
                "selection_sha256": hashlib.sha256(preimage.encode()).hexdigest(),
            }
        )
    selected: list[dict[str, Any]] = []
    for window in (0, 1, 2):
        ordered = sorted(
            groups[window],
            key=lambda row: (row["selection_sha256"], row["work_numeric_id"]),
        )
        require(len(ordered) >= 32, f"window pool: {window}")
        selected.extend(ordered[:32])
    selected.sort(key=lambda row: (row["window_code"], row["selection_sha256"]))
    return selected, {
        "normalized_work_rows_scanned": sum(rows for _, rows in work_parts),
        "normalized_event_rows_scanned": sum(rows for _, rows in event_parts),
        "eligible_prelink_rows": len(candidates),
        "author_linked_pool_rows": len(linked),
        "recent_six_year_author_linked_rows": len(groups[1]) + len(groups[2]),
        "window_pool_rows": {str(window): len(groups[window]) for window in (0, 1, 2)},
    }


def _verify_raw_rows(
    config: Mapping[str, Any],
    openalex_root: Path,
    sample_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    source = _mapping(config.get("current_snapshot_source"), "source")
    root = openalex_root.resolve(strict=True)
    require(root.name == source["required_root_id"], "raw root")
    main = root / str(source["main_table_directory"])
    files = sorted(main.glob("part-*.parquet"), key=lambda path: int(path.stem.split("-")[1]))
    require(len(files) == source["expected_main_file_count"], "raw file count")
    require(
        sum(pq.ParquetFile(path).metadata.num_rows for path in files)
        == source["expected_main_footer_rows"],
        "raw footer rows",
    )
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[(int(row["source_file_index"]), int(row["source_row_group"]))].append(row)
    columns = [str(value) for value in source["required_columns"]]
    checked = 0
    for (file_index, row_group), expected_rows in grouped.items():
        require(0 <= file_index < len(files), "raw source file index")
        parquet = pq.ParquetFile(files[file_index])
        require(0 <= row_group < parquet.metadata.num_row_groups, "raw row group")
        table = parquet.read_row_group(row_group, columns=columns)
        wanted = pa.array([str(row["openalex_id"]) for row in expected_rows])
        observed = table.filter(pc.fill_null(pc.is_in(table["id"], value_set=wanted), False))
        observed_by_id = {str(row["id"]): row for row in observed.to_pylist()}
        require(len(observed_by_id) == len(expected_rows), "raw selected IDs")
        for expected in expected_rows:
            raw = observed_by_id[str(expected["openalex_id"])]
            require(raw["title"] == expected["title"], "raw title")
            require(raw["language"] == expected["language"], "raw language")
            require(int(raw["publication_year"]) == expected["publication_year"], "raw year")
            require(raw["type"] == expected["work_type_name"], "raw type")
            require(raw["authors_count"] == expected["observed_authors_count"], "raw authors")
            require(raw["is_paratext"] == expected["is_paratext"], "raw paratext")
            require(raw["is_retracted"] == expected["is_retracted"], "raw retracted")
            require(raw["primary_topic__id"] == "https://openalex.org/T10543", "raw topic")
            require(
                raw["primary_topic__field__id"] == "https://openalex.org/fields/27",
                "raw field",
            )
            require(
                raw["primary_topic__display_name"] == expected["primary_topic_name"],
                "raw topic name",
            )
            require(
                raw["primary_topic__score"] == expected["primary_topic_score"],
                "raw topic score",
            )
            require(
                raw["primary_topic__subfield__id"] == expected["primary_subfield_id"],
                "raw subfield",
            )
            require(
                raw["primary_location__source__id"] == expected["source_id"],
                "raw source ID",
            )
            require(
                raw["primary_location__source__display_name"] == expected["source_name"],
                "raw source name",
            )
            checked += 1
    require(checked == 96, "raw rows checked")
    return {
        "main_files_declared": len(files),
        "source_row_groups_read": len(grouped),
        "selected_rows_checked": checked,
    }


def validate(
    config_path: Path = DEFAULT_CONFIG,
    work_root: Path = DEFAULT_WORK_ROOT,
    event_root: Path = DEFAULT_EVENT_ROOT,
    openalex_root: Path = DEFAULT_OPENALEX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    workers: int = 16,
    replay_normalized: bool = True,
) -> dict[str, Any]:
    require(workers >= 1, "workers")
    config_path = config_path.resolve(strict=True)
    config = _validate_config(config_path)
    config_sha256 = _sha256(config_path)
    receipt, paths = _verify_output(
        config,
        output_root,
        config_sha256=config_sha256,
    )
    sample_file = pq.ParquetFile(paths["sample"])
    require(sample_file.metadata.num_rows == 96, "sample rows")
    require(
        sample_file.schema_arrow.metadata == {b"schema_version": SAMPLE_SCHEMA.encode()},
        "sample schema metadata",
    )
    sample_rows = sample_file.read().to_pylist()
    require(len({row["openalex_id"] for row in sample_rows}) == 96, "sample IDs")
    require(
        [row["window_code"] for row in sample_rows].count(0) == 32
        and [row["window_code"] for row in sample_rows].count(1) == 32
        and [row["window_code"] for row in sample_rows].count(2) == 32,
        "sample windows",
    )
    pool_summary = _read_json(paths["pool_summary"])
    require(pool_summary == receipt["pool"], "pool receipt")
    raw_replay = _verify_raw_rows(config, openalex_root, sample_rows)
    normalized_replay: dict[str, Any] | None = None
    work_tree: str | None = None
    event_tree: str | None = None
    if replay_normalized:
        normalized = _mapping(config.get("normalized_inputs"), "normalized")
        work_paths, work_tree = _partition_inputs(
            work_root,
            _mapping(normalized.get("cutoff_work_metadata"), "work input"),
            workers=workers,
        )
        event_paths, event_tree = _partition_inputs(
            event_root,
            _mapping(normalized.get("author_history_events"), "event input"),
            workers=workers,
        )
        expected, normalized_replay = _replay_normalized(
            config,
            work_paths,
            event_paths,
            workers=workers,
        )
        require(
            [row["work_numeric_id"] for row in expected]
            == [row["work_numeric_id"] for row in sample_rows],
            "deterministic selected IDs",
        )
        for expected_row, observed in zip(expected, sample_rows, strict=True):
            for key in (
                "publication_year",
                "temporal_disposition",
                "work_type_code",
                "declared_authors_count",
                "matching_author_event_rows",
                "window_code",
                "selection_sha256",
            ):
                require(observed[key] == expected_row[key], f"sample metadata: {key}")
        for key, expected_value in normalized_replay.items():
            require(pool_summary[key] == expected_value, f"pool replay: {key}")
        require(
            work_tree
            == receipt["normalized_input_evidence"]["work_selected_tree_sha256"],
            "work input tree",
        )
        require(
            event_tree
            == receipt["normalized_input_evidence"]["event_selected_tree_sha256"],
            "event input tree",
        )
    for key in (
        "general_raw_title_scan",
        "full_candidate_title_copy",
        "network_accessed",
        "credentials_accessed",
        "content_disposition_created",
        "selection_certificate_issued",
        "deterministic_reselection_executed",
        "outcomes_read",
        "model_fitting_authorized",
        "g4_open",
    ):
        require(receipt[key] is False, f"unsafe receipt flag: {key}")
    require(receipt["api_requests_created"] == 0, "API requests created")
    require(receipt["api_requests_submitted"] == 0, "API requests submitted")
    require(receipt["api_cost_usd"] == "0.000000", "API cost")
    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "verifier_sha256": _sha256(Path(__file__).resolve()),
        "producer_sha256": receipt["producer_sha256"],
        "receipt_sha256": _sha256(
            output_root / str(config["output_contract"]["receipt_basename"])
        ),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "normalized_replay_executed": replay_normalized,
        "normalized_work_selected_tree_sha256": work_tree,
        "normalized_event_selected_tree_sha256": event_tree,
        "normalized_replay": normalized_replay,
        "raw_selected_row_replay": raw_replay,
        "selected_title_rows": 96,
        "missing_title_rows": pool_summary["missing_title_rows"],
        "recent_six_year_author_linked_rows": pool_summary[
            "recent_six_year_author_linked_rows"
        ],
        "network_accessed": False,
        "credentials_accessed": False,
        "api_requests_submitted": 0,
        "content_disposition_created": False,
        "selection_certificate_issued": False,
        "outcomes_read": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--openalex-root", type=Path, default=DEFAULT_OPENALEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(
            args.config,
            args.work_root,
            args.event_root,
            args.openalex_root,
            args.output_root,
            workers=args.workers,
        )
        report_path = args.report.absolute()
        require(not report_path.exists(), "verification report exists")
        require(report_path.parent == Path("/ssd"), "verification report parent")
        temporary = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
        with temporary.open("xb") as handle:
            handle.write(_canonical_json_bytes(result))
        temporary.chmod(0o600)
        os.replace(temporary, report_path)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        T10543TargetedSampleVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_T10543_TARGETED_TITLE_SAMPLE_VERIFICATION_V1",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
