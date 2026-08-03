#!/usr/bin/env python3
"""Independent read-only verifier for author-history v3 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "experiments/configs/openalex_g1_work_side_author_history_v3.yaml"
PRODUCER_PATH = REPO_ROOT / "scripts/audit_openalex_work_side_author_history_v3.py"
CONFIG_SHA256 = "ef73d41387d6186a50f44eddd31db7c9f10dd5b26cda869fb54dcea3352018f3"
PRODUCER_SHA256 = "fe226e1d1348c32a4a077c8ce4852f44e0064b92fe3aaebfd10a43cd9f41e9b9"
OUTPUTS_SCHEMA = "openalex_g1_work_side_author_history_outputs.v3"
PLAN_SCHEMA = "openalex_g1_work_side_author_history_production_plan.v3"
REPORT_SCHEMA = "openalex_g1_work_side_author_history_report.v3"
AUTHOR_CHECKPOINT_SCHEMA = "openalex_g1_work_side_author_history_author_checkpoint.v3"
WORK_CHECKPOINT_SCHEMA = "openalex_g1_work_side_author_history_work_checkpoint.v3"
SCAN_CHECKPOINT_SCHEMA = "openalex_g1_work_side_author_history_scan_checkpoint.v3"
VERIFY_SCHEMA = "openalex_g1_work_side_author_history_independent_verification.v1"
PASS_STATUS = "PASS_READ_ONLY_INDEPENDENT_AUTHOR_HISTORY_OUTPUT_REDUCTION_VERIFICATION"
COMPLETION_CONTRACT = "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
FULL_SCOPE = "FULL_CURRENT_SNAPSHOT_PRODUCTION"
PILOT_SCOPE = "PRODUCTION_SOURCE_PREFIX_PILOT_NONPROMOTABLE"
FULL_OUTPUT_STATUS = (
    "PASS_CURRENT_SNAPSHOT_WORK_SIDE_AUTHOR_TOKEN_LINKED_RETROSPECTIVE_HISTORY_"
    "ACCOUNTING_AND_SENSITIVITY_COMPONENT_AUTHOR_ENTITY_AND_PAST_OBSERVABILITY_"
    "UNVERIFIED"
)
PILOT_OUTPUT_STATUS = "PILOT_WORK_SIDE_AUTHOR_HISTORY_ONLY"
PARTITION_COUNT = 4096
HISTORY_USABLE = frozenset({2, 3})
TEAM_CAPS = (10, 50, 100)
THRESHOLDS = (50, 100, 250, 500, 1000)
MAIN_STAGE_RECORD_BYTES = 13
AUTHORSHIP_STAGE_RECORD_BYTES = 16
FULL_EXPECTED = {
    "main_raw_rows": 492_293_207,
    "main_retained_rows": 492_293_206,
    "main_quarantined_w_minus_1_rows": 1,
    "authorship_raw_rows": 1_361_551_038,
    "authorship_retained_rows": 1_361_551_037,
    "authorship_quarantined_w_minus_1_rows": 1,
    "authorship_null_author_id_rows": 580_116_419,
    "authorship_grammar_valid_non_null_author_id_rows": 781_434_618,
    "all_distinct_author_token_rows": 113_748_578,
}

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
YEAR_SCHEMA = pa.schema(
    [
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("unique_work_count_all", pa.uint32(), nullable=False),
        pa.field("unique_work_count_cap_10", pa.uint32(), nullable=False),
        pa.field("unique_work_count_cap_50", pa.uint32(), nullable=False),
        pa.field("unique_work_count_cap_100", pa.uint32(), nullable=False),
    ]
)
TOKEN_SCHEMA = pa.schema(
    [
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("first_usable_publication_year", pa.uint16(), nullable=False),
        pa.field("last_usable_publication_year", pa.uint16(), nullable=False),
        pa.field("active_year_count", pa.uint16(), nullable=False),
        pa.field("inclusive_span_years", pa.uint16(), nullable=False),
        pa.field("inactive_gap_years", pa.uint16(), nullable=False),
        pa.field("total_unique_work_count_all", pa.uint32(), nullable=False),
        pa.field("maximum_annual_unique_work_count_all", pa.uint32(), nullable=False),
        pa.field("total_unique_work_count_cap_10", pa.uint32(), nullable=False),
        pa.field("maximum_annual_unique_work_count_cap_10", pa.uint32(), nullable=False),
        pa.field("total_unique_work_count_cap_50", pa.uint32(), nullable=False),
        pa.field("maximum_annual_unique_work_count_cap_50", pa.uint32(), nullable=False),
        pa.field("total_unique_work_count_cap_100", pa.uint32(), nullable=False),
        pa.field("maximum_annual_unique_work_count_cap_100", pa.uint32(), nullable=False),
    ]
)


class VerificationError(RuntimeError):
    """Controlled verification failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 16 << 20) -> str:
    info_before = path.stat(follow_symlinks=False)
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    hasher = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(chunk_bytes):
            hasher.update(chunk)
    info_after = path.stat(follow_symlinks=False)
    require(
        (info_before.st_dev, info_before.st_ino, info_before.st_size, info_before.st_mtime_ns, info_before.st_ctime_ns)
        == (info_after.st_dev, info_after.st_ino, info_after.st_size, info_after.st_mtime_ns, info_after.st_ctime_ns),
        f"file changed while hashing: {path}",
    )
    return hasher.hexdigest()


def read_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid {field}") from exc
    require(isinstance(value, dict), f"{field} must be an object")
    return value


def safe_relative(value: Any, field: str) -> Path:
    require(isinstance(value, str) and value, f"{field} must be a path")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe {field}")
    require(path.as_posix() == value, f"noncanonical {field}")
    return path


def artifact_contract(root: Path, relative_text: str) -> dict[str, Any]:
    relative = safe_relative(relative_text, "artifact path")
    path = root / relative
    return {
        "relative_path": relative_text,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def tree_digest(items: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "relative_path": item["relative_path"],
                    "bytes": item["bytes"],
                    "sha256": item["sha256"],
                }
                for item in sorted(items, key=lambda value: value["relative_path"])
            ]
        )
    )


def semantic_digest(table: pa.Table) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"OPENALEX_G1_AUTHOR_HISTORY_TABLE_V3\0")
    hasher.update(
        canonical_json_bytes(
            [(field.name, str(field.type), field.nullable) for field in table.schema]
        )
    )
    # Parquet readers may expose one physical row group as several Arrow chunks.
    # The producer digests one contiguous in-memory array per column, so remove
    # reader-specific chunk boundaries before applying the same fixed batching.
    for batch in table.combine_chunks().to_batches(max_chunksize=262_144):
        for column in batch.columns:
            require(column.null_count == 0, "null in output table")
            hasher.update(np.asarray(column.to_numpy(zero_copy_only=False)).tobytes(order="C"))
    return hasher.hexdigest()


def report_fingerprint(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    provenance = dict(payload["provenance"])
    provenance.pop("result_fingerprint_sha256", None)
    payload["provenance"] = provenance
    return sha256_bytes(canonical_json_bytes(payload))


def _columns(table: pa.Table) -> dict[str, np.ndarray[Any, Any]]:
    result: dict[str, np.ndarray[Any, Any]] = {}
    for name in table.column_names:
        column = table[name].combine_chunks()
        require(column.null_count == 0, f"null output column {name}")
        result[name] = np.asarray(column.to_numpy(zero_copy_only=False))
    return result


def _independent_reduction(
    events: dict[str, np.ndarray[Any, Any]]
) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, np.ndarray[Any, Any]]]:
    usable = np.isin(
        events["temporal_disposition"], np.array(sorted(HISTORY_USABLE), dtype=np.uint8)
    )
    author = events["author_numeric_id"][usable]
    year_values = events["publication_year"][usable]
    team = events["declared_authors_count"][usable]
    if not len(author):
        year = {
            name: np.empty(0, dtype=dtype)
            for name, dtype in (
                ("author_numeric_id", np.uint64),
                ("publication_year", np.uint16),
                ("unique_work_count_all", np.uint32),
                ("unique_work_count_cap_10", np.uint32),
                ("unique_work_count_cap_50", np.uint32),
                ("unique_work_count_cap_100", np.uint32),
            )
        }
        token = {
            field.name: np.empty(0, dtype=field.type.to_pandas_dtype())
            for field in TOKEN_SCHEMA
        }
        return year, token

    starts = np.flatnonzero(
        np.concatenate(
            (
                np.array([True]),
                (author[1:] != author[:-1]) | (year_values[1:] != year_values[:-1]),
            )
        )
    )
    ends = np.concatenate((starts[1:], np.array([len(author)])))
    year: dict[str, np.ndarray[Any, Any]] = {
        "author_numeric_id": author[starts].astype(np.uint64, copy=False),
        "publication_year": year_values[starts].astype(np.uint16, copy=False),
        "unique_work_count_all": (ends - starts).astype(np.uint32),
    }
    for cap in TEAM_CAPS:
        year[f"unique_work_count_cap_{cap}"] = np.add.reduceat(
            (team <= cap).astype(np.uint32), starts
        ).astype(np.uint32)

    token_starts = np.flatnonzero(
        np.concatenate(
            (
                np.array([True]),
                year["author_numeric_id"][1:] != year["author_numeric_id"][:-1],
            )
        )
    )
    token_ends = np.concatenate(
        (token_starts[1:], np.array([len(year["author_numeric_id"])]))
    )
    first = year["publication_year"][token_starts]
    last = year["publication_year"][token_ends - 1]
    active = (token_ends - token_starts).astype(np.uint16)
    span = (last.astype(np.uint32) - first.astype(np.uint32) + 1).astype(np.uint16)
    token: dict[str, np.ndarray[Any, Any]] = {
        "author_numeric_id": year["author_numeric_id"][token_starts],
        "first_usable_publication_year": first,
        "last_usable_publication_year": last,
        "active_year_count": active,
        "inclusive_span_years": span,
        "inactive_gap_years": (span - active).astype(np.uint16),
    }
    for label in ("all", "cap_10", "cap_50", "cap_100"):
        source = year[f"unique_work_count_{label}"]
        token[f"total_unique_work_count_{label}"] = np.add.reduceat(
            source, token_starts
        ).astype(np.uint32)
        token[f"maximum_annual_unique_work_count_{label}"] = np.maximum.reduceat(
            source, token_starts
        ).astype(np.uint32)
    return year, token


def _sensitivity(
    events: dict[str, np.ndarray[Any, Any]],
    token: dict[str, np.ndarray[Any, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    usable = np.isin(
        events["temporal_disposition"], np.array(sorted(HISTORY_USABLE), dtype=np.uint8)
    )
    team = events["declared_authors_count"][usable]
    pair_counts = {
        "all": int(np.count_nonzero(usable)),
        "cap10": int(np.count_nonzero(team <= 10)),
        "cap50": int(np.count_nonzero(team <= 50)),
        "cap100": int(np.count_nonzero(team <= 100)),
    }
    result: dict[str, int] = {}
    author = token["author_numeric_id"]
    for view, selected in (
        ("base", np.ones(len(author), dtype=bool)),
        ("positive", author > 0),
    ):
        for cap_name, field_name in (
            ("all", "all"),
            ("cap10", "cap_10"),
            ("cap50", "cap_50"),
            ("cap100", "cap_100"),
        ):
            total = token[f"total_unique_work_count_{field_name}"]
            maximum = token[f"maximum_annual_unique_work_count_{field_name}"]
            result[f"{view}_{cap_name}_token_count"] = int(
                np.count_nonzero(selected & (total > 0))
            )
            for threshold in THRESHOLDS:
                result[f"{view}_{cap_name}_gt_{threshold}"] = int(
                    np.count_nonzero(selected & (maximum > threshold))
                )
    return pair_counts, result


def _compare_columns(
    actual: Mapping[str, np.ndarray[Any, Any]],
    expected: Mapping[str, np.ndarray[Any, Any]],
    field: str,
) -> None:
    require(set(actual) == set(expected), f"{field} column set mismatch")
    for name in actual:
        require(np.array_equal(actual[name], expected[name]), f"{field}.{name} mismatch")


def verify_author_partition(root: Path, partition: int, task_fingerprint: str) -> dict[str, Any]:
    checkpoint_path = root / "checkpoints" / "author" / f"part-{partition:04d}.json"
    checkpoint = read_json(checkpoint_path, "author checkpoint")
    require(checkpoint.get("schema_version") == AUTHOR_CHECKPOINT_SCHEMA, "author checkpoint schema mismatch")
    require(checkpoint.get("task_fingerprint_sha256") == task_fingerprint, "author checkpoint task mismatch")
    require(checkpoint.get("partition_id") == partition, "author checkpoint partition mismatch")

    paths = {
        "events": root / "events" / f"part-{partition:04d}.parquet",
        "author_year": root / "author_year_history" / f"part-{partition:04d}.parquet",
        "author_token": root / "author_token_history" / f"part-{partition:04d}.parquet",
    }
    schemas = {"events": EVENT_SCHEMA, "author_year": YEAR_SCHEMA, "author_token": TOKEN_SCHEMA}
    tables: dict[str, pa.Table] = {}
    for name, path in paths.items():
        table = pq.read_table(path)
        require(table.schema == schemas[name], f"{name} schema mismatch in partition {partition}")
        contract = checkpoint["outputs"][name]
        require(contract["relative_path"] == path.relative_to(root).as_posix(), "checkpoint output path mismatch")
        require(contract["bytes"] == path.stat().st_size, "checkpoint output bytes mismatch")
        require(contract["sha256"] == sha256_file(path), "checkpoint output SHA mismatch")
        require(contract["semantic_sha256"] == semantic_digest(table), "checkpoint semantic SHA mismatch")
        tables[name] = table

    events = _columns(tables["events"])
    if len(events["author_numeric_id"]):
        author = events["author_numeric_id"]
        year = events["publication_year"]
        work = events["work_numeric_id"]
        require(bool(np.all(author % PARTITION_COUNT == partition)), "event author partition mismatch")
        order_ok = (author[1:] > author[:-1]) | (
            (author[1:] == author[:-1])
            & ((year[1:] > year[:-1]) | ((year[1:] == year[:-1]) & (work[1:] >= work[:-1])))
        )
        require(bool(np.all(order_ok)), "event ordering mismatch")
        duplicate = (author[1:] == author[:-1]) & (work[1:] == work[:-1])
        require(not bool(np.any(duplicate)), "duplicate author-work event")
        excluded_null_year = events["temporal_disposition"] == 0
        require(bool(np.all(events["publication_year"][excluded_null_year] == 0)), "disposition-zero year mismatch")
    expected_year, expected_token = _independent_reduction(events)
    actual_year = _columns(tables["author_year"])
    actual_token = _columns(tables["author_token"])
    _compare_columns(actual_year, expected_year, "author_year")
    _compare_columns(actual_token, expected_token, "author_token")
    pair_counts, sensitivity = _sensitivity(events, expected_token)

    usable = np.isin(events["temporal_disposition"], np.array(sorted(HISTORY_USABLE), dtype=np.uint8))
    all_author_tokens = (
        int(np.count_nonzero(np.concatenate((np.array([True]), events["author_numeric_id"][1:] != events["author_numeric_id"][:-1]))))
        if len(events["author_numeric_id"])
        else 0
    )
    expected_metrics = {
        "event_rows": len(tables["events"]),
        "history_usable_event_rows": int(np.count_nonzero(usable)),
        "all_distinct_author_token_rows": all_author_tokens,
        "author_year_rows": len(tables["author_year"]),
        "author_token_rows": len(tables["author_token"]),
        "a_zero_event_rows": int(np.count_nonzero(events["author_numeric_id"] == 0)),
        "a_zero_token_rows": int(np.count_nonzero(expected_token["author_numeric_id"] == 0)),
        "pair_counts_by_cap": pair_counts,
        "sensitivity": sensitivity,
    }
    for key, value in expected_metrics.items():
        require(checkpoint.get(key) == value, f"author checkpoint {key} mismatch")
    return expected_metrics


def _sum_nested(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        value = record[key]
        require(isinstance(value, dict), f"{key} must be object")
        for item_key, item_value in value.items():
            require(isinstance(item_value, int) and not isinstance(item_value, bool), f"{key}.{item_key} must be int")
            result[item_key] = result.get(item_key, 0) + item_value
    return result


def _sum_author(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "event_rows",
        "history_usable_event_rows",
        "all_distinct_author_token_rows",
        "author_year_rows",
        "author_token_rows",
        "a_zero_event_rows",
        "a_zero_token_rows",
    )
    result: dict[str, Any] = {key: sum(int(record[key]) for record in records) for key in keys}
    result["pair_counts_by_cap"] = _sum_nested(records, "pair_counts_by_cap")
    result["sensitivity"] = _sum_nested(records, "sensitivity")
    return result


def _sum_work(root: Path, task_fingerprint: str) -> dict[str, Any]:
    integer_keys = (
        "main_rows", "canonical_rows_joined", "raw_authorship_slots_joined",
        "unmatched_authorship_slots", "authors_count_mismatch_works",
        "null_author_slots", "grammar_valid_author_slots",
        "distinct_author_work_pairs", "duplicate_author_work_pair_excess_rows",
        "a_zero_distinct_author_work_pairs", "history_usable_distinct_author_work_pairs",
    )
    result: dict[str, Any] = {key: 0 for key in integer_keys}
    dispositions = {str(index): 0 for index in range(6)}
    for partition in range(PARTITION_COUNT):
        checkpoint = read_json(root / "checkpoints" / "work" / f"part-{partition:04d}.json", "work checkpoint")
        require(checkpoint.get("schema_version") == WORK_CHECKPOINT_SCHEMA, "work checkpoint schema mismatch")
        require(checkpoint.get("task_fingerprint_sha256") == task_fingerprint, "work checkpoint task mismatch")
        require(checkpoint.get("partition_id") == partition, "work checkpoint partition mismatch")
        for key in integer_keys:
            result[key] += int(checkpoint[key])
        for key, value in checkpoint["pair_temporal_disposition_counts"].items():
            dispositions[key] += int(value)
    result["pair_temporal_disposition_counts"] = dispositions
    return result


def _sum_scan(root: Path, plan: Mapping[str, Any], kind: str) -> dict[str, int]:
    result: dict[str, int] = {}
    selected = plan[f"selected_{kind}_tasks"]
    record_bytes = (
        MAIN_STAGE_RECORD_BYTES if kind == "main" else AUTHORSHIP_STAGE_RECORD_BYTES
    )
    projected_schema = (
        {
            "id": "string",
            "authors_count": "int64",
            "primary_topic__field__id": "string",
        }
        if kind == "main"
        else {"id": "string", "authorships__author__id": "string"}
    )
    for task in selected:
        digest = hashlib.sha256(task["relative_path"].encode("utf-8")).hexdigest()
        checkpoint = read_json(root / "checkpoints" / "scan" / kind / f"{digest}.json", "scan checkpoint")
        require(checkpoint.get("schema_version") == SCAN_CHECKPOINT_SCHEMA, "scan checkpoint schema mismatch")
        require(checkpoint.get("task_fingerprint_sha256") == plan["task_fingerprint_sha256"], "scan checkpoint task mismatch")
        require(checkpoint.get("kind") == kind, "scan checkpoint kind mismatch")
        require(checkpoint.get("relative_source_path") == task["relative_path"], "scan source path mismatch")
        require(checkpoint.get("source_identity") == task["source_stat"], "scan source identity mismatch")
        require(checkpoint.get("projected_schema") == projected_schema, "scan projected schema mismatch")
        stage = root / safe_relative(checkpoint["staging_relative_path"], "staging path")
        require(checkpoint["staging_bytes"] == stage.stat().st_size, "staging bytes mismatch")
        require(checkpoint["staging_rows"] == checkpoint["metrics"]["retained_rows"], "staging row count mismatch")
        require(checkpoint["staging_bytes"] == checkpoint["staging_rows"] * record_bytes, "staging record-size mismatch")
        require(checkpoint["staging_sha256"] == sha256_file(stage), "staging SHA mismatch")
        for key, value in checkpoint["metrics"].items():
            result[key] = result.get(key, 0) + int(value)
    return result


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    require(sha256_file(CONFIG_PATH) == CONFIG_SHA256, "repository config SHA mismatch")
    require(sha256_file(PRODUCER_PATH) == PRODUCER_SHA256, "repository producer SHA mismatch")
    receipt_path = root / "author_history_audit.outputs.json"
    receipt_bytes_before = receipt_path.read_bytes()
    receipt = read_json(receipt_path, "receipt")
    require(receipt.get("schema_version") == OUTPUTS_SCHEMA, "receipt schema mismatch")
    require(receipt.get("completion_contract") == COMPLETION_CONTRACT, "completion contract mismatch")
    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, list), "receipt artifacts must be array")
    expected_paths = {
        safe_relative(item["relative_path"], "artifact path").as_posix()
        for item in artifacts
    }
    require(len(expected_paths) == len(artifacts), "duplicate receipt artifact path")
    actual_paths_before = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    require(
        actual_paths_before == expected_paths | {"audit.lock", receipt_path.name},
        "output file-set closure mismatch",
    )
    lock_path = root / "audit.lock"
    require(lock_path.is_file() and not lock_path.is_symlink(), "invalid audit lock")
    require(lock_path.stat().st_size == 0, "audit lock must be empty")
    actual_artifacts = [artifact_contract(root, item["relative_path"]) for item in artifacts]
    require(actual_artifacts == artifacts, "receipt artifact contract mismatch")
    require(receipt.get("artifact_count") == len(artifacts), "receipt artifact count mismatch")
    require(receipt.get("artifact_tree_sha256") == tree_digest(artifacts), "artifact tree mismatch")

    plan = read_json(root / "audit_plan.json", "plan")
    require(plan.get("schema_version") == PLAN_SCHEMA, "plan schema mismatch")
    require(plan.get("config_sha256") == CONFIG_SHA256, "plan config SHA mismatch")
    require(plan.get("tool_sha256") == PRODUCER_SHA256, "plan producer SHA mismatch")
    task_payload = dict(plan)
    task_fingerprint = task_payload.pop("task_fingerprint_sha256")
    require(task_fingerprint == sha256_bytes(canonical_json_bytes(task_payload)), "task fingerprint mismatch")
    require(receipt.get("task_fingerprint_sha256") == task_fingerprint, "receipt task mismatch")

    report = read_json(root / "author_history_audit.json", "report")
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema mismatch")
    require(report.get("status") == receipt.get("status"), "report/receipt status mismatch")
    require(report.get("scope") == receipt.get("scope"), "report/receipt scope mismatch")
    require(
        report.get("production_component_pass")
        == receipt.get("production_component_pass"),
        "report/receipt pass mismatch",
    )
    if receipt.get("production_component_pass") is True:
        require(receipt.get("scope") == FULL_SCOPE, "passing output must be full scope")
        require(receipt.get("status") == FULL_OUTPUT_STATUS, "full output status mismatch")
    else:
        require(receipt.get("scope") == PILOT_SCOPE, "nonpassing output must be pilot scope")
        require(receipt.get("status") == PILOT_OUTPUT_STATUS, "pilot output status mismatch")
    result_fingerprint = report["provenance"]["result_fingerprint_sha256"]
    require(report_fingerprint(report) == result_fingerprint, "report fingerprint mismatch")
    require(receipt.get("result_fingerprint_sha256") == result_fingerprint, "receipt result mismatch")

    author_records: list[dict[str, Any]] = []
    for partition in range(PARTITION_COUNT):
        if partition % 256 == 0:
            print(f"verify author partition {partition}/{PARTITION_COUNT}", file=sys.stderr, flush=True)
        author_records.append(verify_author_partition(root, partition, task_fingerprint))
    author_metrics = _sum_author(author_records)
    require(author_metrics == report["author_history_accounting"], "global author metrics mismatch")
    work_metrics = _sum_work(root, task_fingerprint)
    require(work_metrics == report["work_event_accounting"], "global work metrics mismatch")
    main_scan = _sum_scan(root, plan, "main")
    authorship_scan = _sum_scan(root, plan, "authorship")
    require(main_scan == report["source_accounting"]["main"], "main scan metrics mismatch")
    require(authorship_scan == report["source_accounting"]["authorship"], "authorship scan metrics mismatch")
    require(work_metrics["main_rows"] == main_scan["retained_rows"], "main scan/work closure mismatch")
    require(work_metrics["canonical_rows_joined"] == work_metrics["main_rows"], "canonical/work closure mismatch")
    require(
        work_metrics["raw_authorship_slots_joined"]
        + work_metrics["unmatched_authorship_slots"]
        == authorship_scan["retained_rows"],
        "authorship scan/work closure mismatch",
    )
    require(
        work_metrics["distinct_author_work_pairs"]
        + work_metrics["duplicate_author_work_pair_excess_rows"]
        == work_metrics["grammar_valid_author_slots"],
        "author-pair dedup closure mismatch",
    )
    require(
        sum(work_metrics["pair_temporal_disposition_counts"].values())
        == work_metrics["distinct_author_work_pairs"],
        "pair disposition closure mismatch",
    )
    require(
        author_metrics["event_rows"] == work_metrics["distinct_author_work_pairs"],
        "work/event row closure mismatch",
    )
    require(
        author_metrics["history_usable_event_rows"]
        == work_metrics["history_usable_distinct_author_work_pairs"],
        "history-usable work/event closure mismatch",
    )
    require(
        author_metrics["a_zero_event_rows"]
        == work_metrics["a_zero_distinct_author_work_pairs"],
        "A0 work/event closure mismatch",
    )
    require(
        author_metrics["pair_counts_by_cap"]["all"]
        == author_metrics["history_usable_event_rows"],
        "history-usable cap-all closure mismatch",
    )
    if receipt.get("production_component_pass") is True:
        require(work_metrics["unmatched_authorship_slots"] == 0, "full unmatched authorship slots")
        require(work_metrics["authors_count_mismatch_works"] == 0, "full authors_count mismatch")
        require(authorship_scan["malformed_non_null_author_id_rows"] == 0, "full malformed author IDs")
        require(authorship_scan["null_author_id_rows"] == work_metrics["null_author_slots"], "full null-slot closure mismatch")
        require(
            authorship_scan["grammar_valid_non_null_author_id_rows"]
            == work_metrics["grammar_valid_author_slots"],
            "full grammar-valid-slot closure mismatch",
        )
        observed_full = {
            "main_raw_rows": main_scan["raw_rows"],
            "main_retained_rows": main_scan["retained_rows"],
            "main_quarantined_w_minus_1_rows": main_scan["quarantined_w_minus_1_rows"],
            "authorship_raw_rows": authorship_scan["raw_rows"],
            "authorship_retained_rows": authorship_scan["retained_rows"],
            "authorship_quarantined_w_minus_1_rows": authorship_scan["quarantined_w_minus_1_rows"],
            "authorship_null_author_id_rows": authorship_scan["null_author_id_rows"],
            "authorship_grammar_valid_non_null_author_id_rows": authorship_scan["grammar_valid_non_null_author_id_rows"],
            "all_distinct_author_token_rows": author_metrics["all_distinct_author_token_rows"],
        }
        require(observed_full == FULL_EXPECTED, "full frozen source-count closure mismatch")
    actual_paths_after = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    require(actual_paths_after == actual_paths_before, "output file set changed during verification")
    require(receipt_path.read_bytes() == receipt_bytes_before, "receipt changed during verification")
    return {
        "schema_version": VERIFY_SCHEMA,
        "status": PASS_STATUS,
        "output_status": receipt["status"],
        "scope": receipt["scope"],
        "production_component_pass": receipt["production_component_pass"],
        "artifact_count": len(artifacts),
        "partition_count": PARTITION_COUNT,
        "event_rows": author_metrics["event_rows"],
        "history_usable_event_rows": author_metrics["history_usable_event_rows"],
        "author_token_rows": author_metrics["author_token_rows"],
        "task_fingerprint_sha256": task_fingerprint,
        "result_fingerprint_sha256": result_fingerprint,
        "output_receipt_sha256": sha256_bytes(receipt_bytes_before),
        "output_tree_unchanged": True,
        "source_payloads_opened": 0,
        "producer_imported": False,
        "claim_boundary": (
            "This read-only verifier independently checks the exact output file set, "
            "rehashes every receipt artifact, validates staging record sizes, reopens "
            "every event/year/token Parquet partition, recomputes all published author "
            "reductions and sensitivity cells, and checks aggregate cross-surface "
            "closures. It does not replay the production source projection or "
            "work-partition join from OpenAlex payloads, identify people, or promote "
            "G-1 or G0."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify(args.output_root)
    except (VerificationError, OSError, pa.ArrowException) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
