#!/usr/bin/env python3
"""Receipt-bound production author-token history successor.

This runner is intentionally repository-only.  It treats OpenAlex author IDs as
current-snapshot tokens, never as verified people, and uses the independently
verified canonical work-time spine as its only row-level temporal parent.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import resource
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

import audit_openalex_canonical_work_time_spine_v3 as canonical


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/openalex_g1_work_side_author_history_v3.yaml"
)
CONFIG_SHA256 = "ef73d41387d6186a50f44eddd31db7c9f10dd5b26cda869fb54dcea3352018f3"
CONFIG_SCHEMA = "openalex_g1_work_side_author_history_production_config.v3"
PLAN_SCHEMA = "openalex_g1_work_side_author_history_production_plan.v3"
SCAN_CHECKPOINT_SCHEMA = "openalex_g1_work_side_author_history_scan_checkpoint.v3"
WORK_CHECKPOINT_SCHEMA = "openalex_g1_work_side_author_history_work_checkpoint.v3"
AUTHOR_CHECKPOINT_SCHEMA = "openalex_g1_work_side_author_history_author_checkpoint.v3"
REPORT_SCHEMA = "openalex_g1_work_side_author_history_report.v3"
OUTPUTS_SCHEMA = "openalex_g1_work_side_author_history_outputs.v3"
STATE_SCHEMA = "openalex_g1_work_side_author_history_state.v3"

FULL_SCOPE = "FULL_CURRENT_SNAPSHOT_PRODUCTION"
PILOT_SCOPE = "PRODUCTION_SOURCE_PREFIX_PILOT_NONPROMOTABLE"
FULL_STATUS = (
    "PASS_CURRENT_SNAPSHOT_WORK_SIDE_AUTHOR_TOKEN_LINKED_RETROSPECTIVE_"
    "HISTORY_ACCOUNTING_AND_SENSITIVITY_COMPONENT_AUTHOR_ENTITY_AND_"
    "PAST_OBSERVABILITY_UNVERIFIED"
)
PILOT_STATUS = "PILOT_WORK_SIDE_AUTHOR_HISTORY_ONLY"
BLOCKED_STATUS = "BLOCKED_WORK_SIDE_AUTHOR_HISTORY_COMPONENT"
COMPLETION_CONTRACT = "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"

PARTITION_COUNT = 4096
BATCH_ROWS = 262_144
UINT64_MAX = 2**64 - 1
NULL_AUTHOR_SENTINEL = np.uint64(UINT64_MAX)
W_MINUS_ONE = "https://openalex.org/W-1"
WORK_PATTERN = r"^https://openalex[.]org/W(0|[1-9][0-9]*)$"
AUTHOR_PATTERN = r"^https://openalex[.]org/A(0|[1-9][0-9]*)$"
FIELD_PATTERN = r"^https://openalex[.]org/fields/(1[1-9]|2[0-9]|3[0-6])$"
TEAM_CAPS = (10, 50, 100)
THRESHOLDS = (50, 100, 250, 500, 1000)
HISTORY_USABLE = frozenset({2, 3})

MAIN_DTYPE = np.dtype(
    [("work", "<u8"), ("team", "<u4"), ("field", "u1")], align=False
)
SLOT_DTYPE = np.dtype([("work", "<u8"), ("author", "<u8")], align=False)
EVENT_DTYPE = np.dtype(
    [
        ("author", "<u8"),
        ("work", "<u8"),
        ("year", "<u2"),
        ("disposition", "u1"),
        ("team", "<u4"),
        ("field", "u1"),
    ],
    align=False,
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


class AuthorHistoryError(RuntimeError):
    """Controlled fail-closed production error."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Strict trusted-config loader."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorHistoryError(message)


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


def sha256_file(path: Path) -> str:
    return canonical._sha256_file(path)


def read_json(path: Path, field: str) -> dict[str, Any]:
    value = canonical._read_json(path, field)
    require(isinstance(value, dict), f"{field} must be an object")
    return value


def require_mapping(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field} must be an object")
    return dict(value)


def require_list(value: Any, field: str) -> list[Any]:
    require(isinstance(value, list), f"{field} must be an array")
    return list(value)


def require_int(value: Any, field: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{field} must be int")
    return int(value)


def safe_relative(value: Any, field: str) -> Path:
    require(isinstance(value, str) and value, f"{field} must be a path")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe {field}")
    require(path.as_posix() == value, f"noncanonical {field}")
    return path


def atomic_json(path: Path, value: Any) -> None:
    canonical._atomic_bytes(path, canonical_json_bytes(value))


def atomic_text(path: Path, value: str) -> None:
    canonical._atomic_bytes(path, value.encode("utf-8"))


def private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def runtime_fingerprint() -> dict[str, str]:
    base = canonical._runtime_fingerprint()
    return {
        "numpy_version": base["numpy_version"],
        "parquet_cpp_version": base["parquet_cpp_version"],
        "platform_machine": base["platform_machine"],
        "pyarrow_version": base["pyarrow_version"],
        "python_implementation": base["python_implementation"],
        "python_version": base["python_version"],
    }


@dataclass(frozen=True)
class SourceTask:
    index: int
    relative_path: str
    table_name: str
    rows: int
    source_bytes: int
    source_stat: dict[str, int]


@dataclass
class ParentContext:
    content_root: Path
    temporal_root: Path
    canonical_root: Path
    content_receipt: dict[str, Any]
    temporal_receipt: dict[str, Any]
    canonical_receipt: dict[str, Any]
    content_plan: dict[str, Any]
    canonical_plan: dict[str, Any]
    main_tasks: list[SourceTask]
    authorship_tasks: list[SourceTask]


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve(strict=True)
    require(resolved == DEFAULT_CONFIG.resolve(strict=True), "only the frozen v3 config is accepted")
    payload = canonical._read_regular_file_bytes(resolved, field="config")
    digest = sha256_bytes(payload)
    require(digest == CONFIG_SHA256, "config SHA-256 mismatch")
    try:
        value = yaml.load(payload.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AuthorHistoryError("invalid strict YAML config") from exc
    config = require_mapping(value, "config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    for relative_text, expected in require_mapping(
        config["frozen_predecessors"], "frozen_predecessors"
    )["paths_and_sha256"].items():
        target = REPO_ROOT / safe_relative(relative_text, "frozen predecessor path")
        require(sha256_file(target) == expected, f"frozen predecessor mismatch: {relative_text}")
    return config, digest


def _validate_receipt_artifacts(root: Path, receipt: Mapping[str, Any], field: str) -> None:
    raw_artifacts = receipt.get("artifacts")
    if isinstance(raw_artifacts, dict):
        artifacts = []
        for relative_text, raw in sorted(raw_artifacts.items()):
            item = require_mapping(raw, f"{field}.artifacts[{relative_text!r}]")
            require(item.get("relative_path") == relative_text, f"{field} artifact key/path mismatch")
            artifacts.append(item)
    else:
        artifacts = require_list(raw_artifacts, f"{field}.artifacts")
    seen: set[str] = set()
    for index, raw in enumerate(artifacts):
        item = require_mapping(raw, f"{field}.artifacts[{index}]")
        relative = safe_relative(item.get("relative_path"), f"{field}.relative_path")
        relative_text = relative.as_posix()
        require(relative_text not in seen, f"duplicate {field} artifact")
        seen.add(relative_text)
        actual = canonical._artifact_contract(root, relative_text)
        require(actual == item, f"{field} artifact mismatch: {relative_text}")


def _validate_parent_receipt(
    root: Path,
    basename: str,
    expected: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    receipt_path = root / basename
    require(sha256_file(receipt_path) == expected["receipt_sha256"], f"{field} receipt SHA mismatch")
    receipt = read_json(receipt_path, f"{field} receipt")
    require(receipt.get("schema_version") == expected["receipt_schema_version"], f"{field} schema mismatch")
    require(receipt.get("status") == expected["required_status"], f"{field} status mismatch")
    for key in ("task_fingerprint_sha256", "result_fingerprint_sha256"):
        expected_value = expected.get(key)
        if expected_value is not None:
            require(receipt.get(key) == expected_value, f"{field} {key} mismatch")
    _validate_receipt_artifacts(root, receipt, field)
    return receipt


def _task_from_content(raw: Mapping[str, Any], expected_table: str) -> SourceTask:
    require(raw.get("table_name") == expected_table, "content task table mismatch")
    source_stat = require_mapping(raw.get("source_stat"), "content task source_stat")
    return SourceTask(
        index=require_int(raw.get("index"), "content task index"),
        relative_path=safe_relative(raw.get("relative_path"), "content task path").as_posix(),
        table_name=expected_table,
        rows=0,
        source_bytes=require_int(raw.get("bytes"), "content task bytes"),
        source_stat={key: require_int(value, f"source_stat.{key}") for key, value in source_stat.items()},
    )


def _footer_rows(source_root: Path, task: SourceTask) -> int:
    path = source_root / task.relative_path
    actual_identity = canonical._rich_identity(path, field=f"source {task.relative_path}")
    observed = {
        key: (actual_identity[key] & 0o7777 if key == "mode" else actual_identity[key])
        for key in task.source_stat
    }
    require(observed == task.source_stat, f"source identity changed: {task.relative_path}")
    parquet = pq.ParquetFile(path)
    return int(parquet.metadata.num_rows)


def validate_parents(
    config: Mapping[str, Any],
    *,
    source_root: Path,
    content_receipt_path: Path,
    temporal_receipt_path: Path,
    canonical_root: Path,
) -> ParentContext:
    parents = require_mapping(config["parents"], "parents")
    content_expected = require_mapping(parents["current_content_baseline"], "content parent")
    temporal_expected = require_mapping(parents["main_temporal_v3"], "temporal parent")
    canonical_expected = require_mapping(parents["canonical_work_time_spine"], "canonical parent")

    content_root = content_receipt_path.resolve(strict=True).parent
    temporal_root = temporal_receipt_path.resolve(strict=True).parent
    canonical_root = canonical_root.resolve(strict=True)
    require(content_receipt_path.name == "content_freeze_audit.outputs.json", "content receipt basename mismatch")
    require(temporal_receipt_path.name == "main_temporal_validity_audit.outputs.json", "temporal receipt basename mismatch")

    content = _validate_parent_receipt(
        content_root, content_receipt_path.name, content_expected, "content"
    )
    require(content.get("plan_sha256") == content_expected["plan_sha256"], "content plan SHA mismatch")
    require(content.get("content_tree_sha256") == content_expected["content_tree_sha256"], "content tree mismatch")
    temporal = _validate_parent_receipt(
        temporal_root, temporal_receipt_path.name, temporal_expected, "temporal"
    )
    canonical_receipt = _validate_parent_receipt(
        canonical_root,
        "work_time_spine_audit.outputs.json",
        canonical_expected,
        "canonical",
    )
    require(canonical_receipt.get("production_component_pass") is True, "canonical component is not a production PASS")
    require(canonical_receipt.get("artifact_count") == canonical_expected["artifact_count"], "canonical artifact count mismatch")
    require(canonical_receipt.get("spine_data_tree_sha256") == canonical_expected["spine_data_tree_sha256"], "canonical spine tree mismatch")

    content_plan = read_json(content_root / "audit_plan.json", "content plan")
    require(content_plan.get("plan_sha256") == content_expected["plan_sha256"], "content plan payload mismatch")
    canonical_plan = read_json(canonical_root / "audit_plan.json", "canonical plan")
    canonical_parents = require_mapping(canonical_plan.get("parents"), "canonical plan parents")
    require(canonical_parents["content"]["receipt_sha256"] == content_expected["receipt_sha256"], "canonical/content cross-binding mismatch")
    require(canonical_parents["temporal"]["receipt_sha256"] == temporal_expected["receipt_sha256"], "canonical/temporal cross-binding mismatch")
    require(canonical_plan.get("config_sha256") == canonical_expected["config_sha256"], "canonical config mismatch")
    require(canonical_plan.get("tool_sha256") == canonical_expected["runner_sha256"], "canonical tool mismatch")

    source_cfg = require_mapping(config["source"], "source")
    require(source_root.resolve(strict=True).name == source_cfg["root_id"], "source root ID mismatch")
    raw_tasks = require_list(content_plan.get("tasks"), "content plan tasks")
    main_table = source_cfg["main_table"]
    authorship_table = source_cfg["authorship_table"]
    main_tasks = [
        _task_from_content(require_mapping(item, "content task"), main_table)
        for item in raw_tasks
        if isinstance(item, dict) and item.get("table_name") == main_table
    ]
    authorship_tasks = [
        _task_from_content(require_mapping(item, "content task"), authorship_table)
        for item in raw_tasks
        if isinstance(item, dict) and item.get("table_name") == authorship_table
    ]
    main_tasks.sort(key=lambda item: item.relative_path)
    authorship_tasks.sort(key=lambda item: item.relative_path)
    require(len(main_tasks) == source_cfg["main"]["files"], "main source file count mismatch")
    require(len(authorship_tasks) == source_cfg["authorship"]["files"], "authorship source file count mismatch")
    require(sum(item.source_bytes for item in main_tasks) == source_cfg["main"]["source_bytes"], "main source bytes mismatch")
    require(sum(item.source_bytes for item in authorship_tasks) == source_cfg["authorship"]["source_bytes"], "authorship source bytes mismatch")
    return ParentContext(
        content_root=content_root,
        temporal_root=temporal_root,
        canonical_root=canonical_root,
        content_receipt=content,
        temporal_receipt=temporal,
        canonical_receipt=canonical_receipt,
        content_plan=content_plan,
        canonical_plan=canonical_plan,
        main_tasks=main_tasks,
        authorship_tasks=authorship_tasks,
    )


def _bool_array(value: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return pc.fill_null(value, False).to_numpy(zero_copy_only=False).astype(bool, copy=False)


def _parse_numeric_strings(
    values: pa.Array,
    *,
    active: np.ndarray,
    prefix_length: int,
    field: str,
) -> np.ndarray:
    if not len(active):
        return np.empty(0, dtype="<u8")
    selected = values.take(pa.array(active, type=pa.int64()))
    suffix = pc.utf8_slice_codeunits(selected, start=prefix_length)
    try:
        parsed = pc.cast(suffix, pa.uint64(), safe=True).to_numpy(zero_copy_only=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise AuthorHistoryError(f"{field} does not fit uint64") from exc
    return np.asarray(parsed, dtype="<u8")


def _field_values(values: pa.Array) -> np.ndarray:
    result = np.zeros(len(values), dtype="u1")
    nonnull = _bool_array(pc.invert(pc.is_null(values)))
    indices = np.flatnonzero(nonnull)
    if not len(indices):
        return result
    selected = values.take(pa.array(indices, type=pa.int64()))
    valid = _bool_array(pc.match_substring_regex(selected, pattern=FIELD_PATTERN))
    require(bool(np.all(valid)), "malformed primary field ID")
    suffix = pc.utf8_slice_codeunits(selected, start=len("https://openalex.org/fields/"))
    parsed = pc.cast(suffix, pa.uint8(), safe=True).to_numpy(zero_copy_only=False)
    result[indices] = np.asarray(parsed, dtype="u1")
    return result


def _projected_schema(parquet: pq.ParquetFile, columns: Sequence[str]) -> dict[str, str]:
    schema = parquet.schema_arrow
    result: dict[str, str] = {}
    for column in columns:
        require(column in schema.names, f"missing source column {column}")
        result[column] = str(schema.field(column).type)
    return result


def _write_array_atomic(path: Path, values: np.ndarray[Any, Any]) -> None:
    private_dir(path.parent)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(values.tobytes(order="C"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        canonical._fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _checkpoint_path(output: Path, kind: str, task: SourceTask) -> Path:
    digest = hashlib.sha256(task.relative_path.encode("utf-8")).hexdigest()
    return output / "checkpoints" / "scan" / kind / f"{digest}.json"


def _staging_path(output: Path, kind: str, task: SourceTask) -> Path:
    digest = hashlib.sha256(task.relative_path.encode("utf-8")).hexdigest()
    return output / "staging" / kind / f"{digest}.bin"


def _source_path(source_root: Path, task: SourceTask) -> Path:
    path = source_root / task.relative_path
    actual = canonical._rich_identity(path, field=f"source {task.relative_path}")
    observed = {
        key: (actual[key] & 0o7777 if key == "mode" else actual[key])
        for key in task.source_stat
    }
    require(observed == task.source_stat, f"source identity changed: {task.relative_path}")
    return path


def _scan_main_task(
    source_root: Path, output: Path, task: SourceTask, task_fingerprint: str
) -> dict[str, Any]:
    source = _source_path(source_root, task)
    parquet = pq.ParquetFile(source)
    expected_schema = {"id": "string", "authors_count": "int64", "primary_topic__field__id": "string"}
    require(_projected_schema(parquet, tuple(expected_schema)) == expected_schema, "main projected schema mismatch")
    records: list[np.ndarray[Any, Any]] = []
    metrics = {"raw_rows": 0, "quarantined_w_minus_1_rows": 0, "retained_rows": 0}
    for batch in parquet.iter_batches(batch_size=BATCH_ROWS, columns=list(expected_schema)):
        ids = batch.column(0)
        require(ids.null_count == 0, "null main work ID")
        raw = len(ids)
        minus = _bool_array(pc.equal(ids, W_MINUS_ONE))
        valid = _bool_array(pc.match_substring_regex(ids, pattern=WORK_PATTERN))
        require(bool(np.all(valid | minus)), "malformed main work ID")
        retained = np.flatnonzero(~minus)
        work = _parse_numeric_strings(ids, active=retained, prefix_length=len("https://openalex.org/W"), field="main work ID")
        teams_all = batch.column(1).to_numpy(zero_copy_only=False)
        require(batch.column(1).null_count == 0, "null authors_count")
        teams = np.asarray(teams_all[retained], dtype=np.int64)
        require(bool(np.all((teams >= 0) & (teams <= np.iinfo(np.uint32).max))), "authors_count outside uint32")
        fields = _field_values(batch.column(2))[retained]
        block = np.empty(len(retained), dtype=MAIN_DTYPE)
        block["work"] = work
        block["team"] = teams.astype("<u4", copy=False)
        block["field"] = fields
        records.append(block)
        metrics["raw_rows"] += raw
        metrics["quarantined_w_minus_1_rows"] += int(np.count_nonzero(minus))
        metrics["retained_rows"] += len(block)
    require(metrics["raw_rows"] == parquet.metadata.num_rows, "main footer row mismatch")
    values = np.concatenate(records) if records else np.empty(0, dtype=MAIN_DTYPE)
    stage = _staging_path(output, "main", task)
    _write_array_atomic(stage, values)
    checkpoint = {
        "schema_version": SCAN_CHECKPOINT_SCHEMA,
        "task_fingerprint_sha256": task_fingerprint,
        "kind": "main",
        "relative_source_path": task.relative_path,
        "source_identity": task.source_stat,
        "projected_schema": expected_schema,
        "metrics": metrics,
        "staging_relative_path": stage.relative_to(output).as_posix(),
        "staging_rows": len(values),
        "staging_bytes": stage.stat().st_size,
        "staging_sha256": sha256_file(stage),
    }
    atomic_json(_checkpoint_path(output, "main", task), checkpoint)
    return checkpoint


def _scan_authorship_task(
    source_root: Path, output: Path, task: SourceTask, task_fingerprint: str
) -> dict[str, Any]:
    source = _source_path(source_root, task)
    parquet = pq.ParquetFile(source)
    expected_schema = {"id": "string", "authorships__author__id": "string"}
    require(_projected_schema(parquet, tuple(expected_schema)) == expected_schema, "authorship projected schema mismatch")
    records: list[np.ndarray[Any, Any]] = []
    metrics = {
        "raw_rows": 0,
        "quarantined_w_minus_1_rows": 0,
        "retained_rows": 0,
        "null_author_id_rows": 0,
        "grammar_valid_non_null_author_id_rows": 0,
        "malformed_non_null_author_id_rows": 0,
        "a_zero_author_id_rows": 0,
    }
    for batch in parquet.iter_batches(batch_size=BATCH_ROWS, columns=list(expected_schema)):
        ids = batch.column(0)
        authors = batch.column(1)
        require(ids.null_count == 0, "null authorship work ID")
        minus = _bool_array(pc.equal(ids, W_MINUS_ONE))
        valid_work = _bool_array(pc.match_substring_regex(ids, pattern=WORK_PATTERN))
        require(bool(np.all(valid_work | minus)), "malformed authorship work ID")
        retained = np.flatnonzero(~minus)
        work = _parse_numeric_strings(ids, active=retained, prefix_length=len("https://openalex.org/W"), field="authorship work ID")
        selected_authors = authors.take(pa.array(retained, type=pa.int64()))
        nulls = _bool_array(pc.is_null(selected_authors))
        nonnull_indices = np.flatnonzero(~nulls)
        author_numeric = np.full(len(retained), NULL_AUTHOR_SENTINEL, dtype="<u8")
        if len(nonnull_indices):
            nonnull = selected_authors.take(pa.array(nonnull_indices, type=pa.int64()))
            valid_author = _bool_array(pc.match_substring_regex(nonnull, pattern=AUTHOR_PATTERN))
            malformed = int(np.count_nonzero(~valid_author))
            metrics["malformed_non_null_author_id_rows"] += malformed
            require(malformed == 0, "malformed non-null author ID")
            parsed = _parse_numeric_strings(
                nonnull,
                active=np.arange(len(nonnull), dtype=np.int64),
                prefix_length=len("https://openalex.org/A"),
                field="author ID",
            )
            require(bool(np.all(parsed != NULL_AUTHOR_SENTINEL)), "author ID collides with null sentinel")
            author_numeric[nonnull_indices] = parsed
            metrics["a_zero_author_id_rows"] += int(np.count_nonzero(parsed == 0))
        block = np.empty(len(retained), dtype=SLOT_DTYPE)
        block["work"] = work
        block["author"] = author_numeric
        records.append(block)
        metrics["raw_rows"] += len(ids)
        metrics["quarantined_w_minus_1_rows"] += int(np.count_nonzero(minus))
        metrics["retained_rows"] += len(retained)
        metrics["null_author_id_rows"] += int(np.count_nonzero(nulls))
        metrics["grammar_valid_non_null_author_id_rows"] += len(nonnull_indices)
    require(metrics["raw_rows"] == parquet.metadata.num_rows, "authorship footer row mismatch")
    values = np.concatenate(records) if records else np.empty(0, dtype=SLOT_DTYPE)
    stage = _staging_path(output, "authorship", task)
    _write_array_atomic(stage, values)
    checkpoint = {
        "schema_version": SCAN_CHECKPOINT_SCHEMA,
        "task_fingerprint_sha256": task_fingerprint,
        "kind": "authorship",
        "relative_source_path": task.relative_path,
        "source_identity": task.source_stat,
        "projected_schema": expected_schema,
        "metrics": metrics,
        "staging_relative_path": stage.relative_to(output).as_posix(),
        "staging_rows": len(values),
        "staging_bytes": stage.stat().st_size,
        "staging_sha256": sha256_file(stage),
    }
    atomic_json(_checkpoint_path(output, "authorship", task), checkpoint)
    return checkpoint


class SpoolWriter:
    """Bounded fixed-partition writer with one small buffer per partition."""

    def __init__(self, root: Path, dtype: np.dtype[Any], *, prefix: str) -> None:
        self.root = root
        self.dtype = dtype
        self.prefix = prefix
        private_dir(root)
        self.paths = [root / f"{prefix}-{index:04d}.bin" for index in range(PARTITION_COUNT)]
        self.handles: list[BinaryIO] = []

    def __enter__(self) -> SpoolWriter:
        for path in self.paths:
            raw = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                0o600,
            )
            self.handles.append(os.fdopen(raw, "wb", buffering=65_536))
        return self

    def write(self, values: np.ndarray[Any, Any], partitions: np.ndarray[Any, Any]) -> None:
        require(len(values) == len(partitions), "spool partition length mismatch")
        if not len(values):
            return
        require(bool(np.all(partitions < PARTITION_COUNT)), "spool partition out of range")
        order = np.argsort(partitions, kind="stable")
        sorted_partitions = partitions[order]
        sorted_values = values[order]
        starts = np.flatnonzero(
            np.concatenate((np.array([True]), sorted_partitions[1:] != sorted_partitions[:-1]))
        )
        ends = np.concatenate((starts[1:], np.array([len(sorted_values)])))
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
            partition = int(sorted_partitions[start])
            self.handles[partition].write(sorted_values[start:end].tobytes(order="C"))

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        close_error: BaseException | None = None
        for handle in self.handles:
            try:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            except BaseException as error:  # pragma: no cover - catastrophic I/O
                close_error = close_error or error
        canonical._fsync_directory(self.root)
        if exc is None and close_error is not None:
            raise close_error


def _read_records(path: Path, dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
    size = path.stat().st_size
    require(size % dtype.itemsize == 0, f"misaligned binary record file: {path.name}")
    if size == 0:
        return np.empty(0, dtype=dtype)
    return np.fromfile(path, dtype=dtype)


def build_work_spools(
    output: Path,
    main_checkpoints: Sequence[Mapping[str, Any]],
    authorship_checkpoints: Sequence[Mapping[str, Any]],
) -> tuple[list[Path], list[Path]]:
    work_root = output / ".author_history_tmp" / "work"
    main_root = work_root / "main"
    slot_root = work_root / "slots"
    with SpoolWriter(main_root, MAIN_DTYPE, prefix="main") as writer:
        for checkpoint in main_checkpoints:
            stage = output / checkpoint["staging_relative_path"]
            values = _read_records(stage, MAIN_DTYPE)
            writer.write(values, (values["work"] % PARTITION_COUNT).astype("<u2"))
    with SpoolWriter(slot_root, SLOT_DTYPE, prefix="slots") as writer:
        for checkpoint in authorship_checkpoints:
            stage = output / checkpoint["staging_relative_path"]
            values = _read_records(stage, SLOT_DTYPE)
            writer.write(values, (values["work"] % PARTITION_COUNT).astype("<u2"))
    return (
        [main_root / f"main-{index:04d}.bin" for index in range(PARTITION_COUNT)],
        [slot_root / f"slots-{index:04d}.bin" for index in range(PARTITION_COUNT)],
    )


def _canonical_partition(
    canonical_root: Path, partition: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    path = canonical_root / "spine" / f"part-{partition:04d}.parquet"
    table = pq.read_table(
        path,
        columns=["work_id_numeric", "publication_year", "temporal_disposition"],
    )
    work = table["work_id_numeric"].combine_chunks().to_numpy(zero_copy_only=False).astype("<u8", copy=False)
    disposition = table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False).astype("u1", copy=False)
    years_arrow = table["publication_year"].combine_chunks()
    years = pc.fill_null(years_arrow, 0).to_numpy(zero_copy_only=False).astype("<u2", copy=False)
    require(len(work) == len(disposition) == len(years), "canonical partition length mismatch")
    require(not len(work) or bool(np.all(work[1:] > work[:-1])), "canonical work IDs not strictly increasing")
    return work, years, disposition


def _work_checkpoint_path(output: Path, partition: int) -> Path:
    return output / "checkpoints" / "work" / f"part-{partition:04d}.json"


def process_work_partitions(
    *,
    output: Path,
    canonical_root: Path,
    main_spools: Sequence[Path],
    slot_spools: Sequence[Path],
    task_fingerprint: str,
    full_scope: bool,
) -> tuple[list[dict[str, Any]], list[Path]]:
    event_spool_root = output / ".author_history_tmp" / "author_events"
    checkpoints: list[dict[str, Any]] = []
    with SpoolWriter(event_spool_root, EVENT_DTYPE, prefix="events") as event_writer:
        for partition in range(PARTITION_COUNT):
            main = _read_records(main_spools[partition], MAIN_DTYPE)
            if len(main):
                order = np.argsort(main["work"], kind="stable")
                main = main[order]
                require(bool(np.all(main["work"][1:] > main["work"][:-1])), "duplicate main work ID")
            canonical_work, canonical_year, canonical_disposition = _canonical_partition(
                canonical_root, partition
            )
            if full_scope:
                require(np.array_equal(main["work"], canonical_work), f"main/canonical closure failed in partition {partition}")
                selected_canonical = np.arange(len(canonical_work), dtype=np.int64)
            else:
                selected_canonical = np.searchsorted(canonical_work, main["work"])
                valid = selected_canonical < len(canonical_work)
                require(bool(np.all(valid)), "pilot main work missing from canonical spine")
                require(bool(np.all(canonical_work[selected_canonical] == main["work"])), "pilot main/canonical mismatch")

            years = canonical_year[selected_canonical]
            dispositions = canonical_disposition[selected_canonical]
            slots = _read_records(slot_spools[partition], SLOT_DTYPE)
            if len(slots):
                order = np.lexsort((slots["author"], slots["work"]))
                slots = slots[order]
            slot_main_index = np.searchsorted(main["work"], slots["work"])
            joined = slot_main_index < len(main)
            if len(main) and bool(np.any(joined)):
                joined_indices = np.flatnonzero(joined)
                joined[joined_indices] &= main["work"][slot_main_index[joined_indices]] == slots["work"][joined_indices]
            unmatched_slots = int(np.count_nonzero(~joined))
            require(not full_scope or unmatched_slots == 0, f"authorship/main join failed in partition {partition}")
            slots = slots[joined]
            slot_main_index = slot_main_index[joined]

            observed = np.bincount(slot_main_index, minlength=len(main)).astype(np.uint64, copy=False)
            team_mismatches = int(np.count_nonzero(observed != main["team"].astype(np.uint64)))
            require(not full_scope or team_mismatches == 0, f"authors_count closure failed in partition {partition}")

            nonnull = slots["author"] != NULL_AUTHOR_SENTINEL
            valid_slots = slots[nonnull]
            if len(valid_slots):
                unique = np.ones(len(valid_slots), dtype=bool)
                unique[1:] = (valid_slots["work"][1:] != valid_slots["work"][:-1]) | (
                    valid_slots["author"][1:] != valid_slots["author"][:-1]
                )
                pairs = valid_slots[unique]
            else:
                pairs = np.empty(0, dtype=SLOT_DTYPE)
            pair_main_index = np.searchsorted(main["work"], pairs["work"])
            pair_disposition = dispositions[pair_main_index] if len(pairs) else np.empty(0, dtype="u1")
            pair_year = years[pair_main_index] if len(pairs) else np.empty(0, dtype="<u2")
            usable = np.isin(pair_disposition, np.array(sorted(HISTORY_USABLE), dtype="u1"))
            events = np.empty(len(pairs), dtype=EVENT_DTYPE)
            events["author"] = pairs["author"]
            events["work"] = pairs["work"]
            events["year"] = pair_year
            events["disposition"] = pair_disposition
            events["team"] = main["team"][pair_main_index]
            events["field"] = main["field"][pair_main_index]
            event_writer.write(events, (events["author"] % PARTITION_COUNT).astype("<u2"))

            disposition_counts = {
                str(index): int(np.count_nonzero(pair_disposition == index))
                for index in range(6)
            }
            checkpoint = {
                "schema_version": WORK_CHECKPOINT_SCHEMA,
                "task_fingerprint_sha256": task_fingerprint,
                "partition_id": partition,
                "main_rows": len(main),
                "canonical_rows_joined": len(selected_canonical),
                "raw_authorship_slots_joined": len(slots),
                "unmatched_authorship_slots": unmatched_slots,
                "authors_count_mismatch_works": team_mismatches,
                "null_author_slots": int(np.count_nonzero(~nonnull)),
                "grammar_valid_author_slots": len(valid_slots),
                "distinct_author_work_pairs": len(pairs),
                "duplicate_author_work_pair_excess_rows": len(valid_slots) - len(pairs),
                "a_zero_distinct_author_work_pairs": int(np.count_nonzero(pairs["author"] == 0)),
                "history_usable_distinct_author_work_pairs": int(np.count_nonzero(usable)),
                "pair_temporal_disposition_counts": disposition_counts,
                "main_spool_sha256": sha256_file(main_spools[partition]),
                "slot_spool_sha256": sha256_file(slot_spools[partition]),
            }
            atomic_json(_work_checkpoint_path(output, partition), checkpoint)
            checkpoints.append(checkpoint)
    return checkpoints, [
        event_spool_root / f"events-{index:04d}.bin" for index in range(PARTITION_COUNT)
    ]


def _table_from_events(events: np.ndarray[Any, Any]) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array(events["author"], type=pa.uint64()),
            pa.array(events["work"], type=pa.uint64()),
            pa.array(events["year"], type=pa.uint16()),
            pa.array(events["disposition"], type=pa.uint8()),
            pa.array(events["team"], type=pa.uint32()),
            pa.array(events["field"], type=pa.uint8()),
        ],
        schema=EVENT_SCHEMA,
    )


def _write_parquet_atomic(path: Path, table: pa.Table) -> None:
    private_dir(path.parent)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
        write_statistics=True,
        row_group_size=BATCH_ROWS,
        version="2.6",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    canonical._fsync_directory(path.parent)


def _year_and_token_arrays(
    events: np.ndarray[Any, Any]
) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, np.ndarray[Any, Any]]]:
    if not len(events):
        year = {
            "author": np.empty(0, dtype="<u8"),
            "year": np.empty(0, dtype="<u2"),
            "all": np.empty(0, dtype="<u4"),
            "cap10": np.empty(0, dtype="<u4"),
            "cap50": np.empty(0, dtype="<u4"),
            "cap100": np.empty(0, dtype="<u4"),
        }
        token = {
            key: np.empty(0, dtype=dtype)
            for key, dtype in (
                ("author", "<u8"), ("first", "<u2"), ("last", "<u2"),
                ("active", "<u2"), ("span", "<u2"), ("gap", "<u2"),
                ("total_all", "<u4"), ("max_all", "<u4"),
                ("total_cap10", "<u4"), ("max_cap10", "<u4"),
                ("total_cap50", "<u4"), ("max_cap50", "<u4"),
                ("total_cap100", "<u4"), ("max_cap100", "<u4"),
            )
        }
        return year, token

    year_start = np.flatnonzero(
        np.concatenate(
            (
                np.array([True]),
                (events["author"][1:] != events["author"][:-1])
                | (events["year"][1:] != events["year"][:-1]),
            )
        )
    )
    year_end = np.concatenate((year_start[1:], np.array([len(events)])))
    count_all = (year_end - year_start).astype("<u4")
    counts: dict[int, np.ndarray[Any, Any]] = {}
    for cap in TEAM_CAPS:
        counts[cap] = np.add.reduceat((events["team"] <= cap).astype("<u4"), year_start).astype("<u4")
    year = {
        "author": events["author"][year_start].astype("<u8", copy=False),
        "year": events["year"][year_start].astype("<u2", copy=False),
        "all": count_all,
        "cap10": counts[10],
        "cap50": counts[50],
        "cap100": counts[100],
    }

    token_start = np.flatnonzero(
        np.concatenate((np.array([True]), year["author"][1:] != year["author"][:-1]))
    )
    token_end = np.concatenate((token_start[1:], np.array([len(year["author"])])))
    first = year["year"][token_start]
    last = year["year"][token_end - 1]
    active = (token_end - token_start).astype("<u2")
    span = (last.astype(np.uint32) - first.astype(np.uint32) + 1).astype("<u2")
    require(bool(np.all(active <= span)), "author active-year/span closure failed")
    token: dict[str, np.ndarray[Any, Any]] = {
        "author": year["author"][token_start],
        "first": first,
        "last": last,
        "active": active,
        "span": span,
        "gap": (span - active).astype("<u2"),
    }
    for name in ("all", "cap10", "cap50", "cap100"):
        token[f"total_{name}"] = np.add.reduceat(year[name], token_start).astype("<u4")
        token[f"max_{name}"] = np.maximum.reduceat(year[name], token_start).astype("<u4")
        require(bool(np.all(token[f"max_{name}"] <= token[f"total_{name}"])), f"{name} max/total closure failed")
    return year, token


def _year_table(year: Mapping[str, np.ndarray[Any, Any]]) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array(year["author"], type=pa.uint64()),
            pa.array(year["year"], type=pa.uint16()),
            pa.array(year["all"], type=pa.uint32()),
            pa.array(year["cap10"], type=pa.uint32()),
            pa.array(year["cap50"], type=pa.uint32()),
            pa.array(year["cap100"], type=pa.uint32()),
        ],
        schema=YEAR_SCHEMA,
    )


def _token_table(token: Mapping[str, np.ndarray[Any, Any]]) -> pa.Table:
    names = [field.name for field in TOKEN_SCHEMA]
    keys = [
        "author", "first", "last", "active", "span", "gap",
        "total_all", "max_all", "total_cap10", "max_cap10",
        "total_cap50", "max_cap50", "total_cap100", "max_cap100",
    ]
    return pa.Table.from_arrays(
        [pa.array(token[key], type=TOKEN_SCHEMA.field(name).type) for key, name in zip(keys, names, strict=True)],
        schema=TOKEN_SCHEMA,
    )


def _semantic_digest(table: pa.Table) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"OPENALEX_G1_AUTHOR_HISTORY_TABLE_V3\0")
    hasher.update(canonical_json_bytes([(field.name, str(field.type), field.nullable) for field in table.schema]))
    for batch in table.to_batches(max_chunksize=BATCH_ROWS):
        for column in batch.columns:
            array = column
            require(array.null_count == 0, "unexpected null in output table")
            values = array.to_numpy(zero_copy_only=False)
            hasher.update(np.asarray(values).tobytes(order="C"))
    return hasher.hexdigest()


def _author_checkpoint_path(output: Path, partition: int) -> Path:
    return output / "checkpoints" / "author" / f"part-{partition:04d}.json"


def process_author_partitions(
    *,
    output: Path,
    event_spools: Sequence[Path],
    task_fingerprint: str,
) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    for partition in range(PARTITION_COUNT):
        events = _read_records(event_spools[partition], EVENT_DTYPE)
        if len(events):
            order = np.lexsort((events["work"], events["year"], events["author"]))
            events = events[order]
            require(bool(np.all(events["author"] % PARTITION_COUNT == partition)), "author partition mismatch")
            duplicate = (
                (events["author"][1:] == events["author"][:-1])
                & (events["work"][1:] == events["work"][:-1])
            )
            require(not bool(np.any(duplicate)), "duplicate author-work event after work dedup")
        event_table = _table_from_events(events)
        usable_events = events[
            np.isin(events["disposition"], np.array(sorted(HISTORY_USABLE), dtype="u1"))
        ]
        year, token = _year_and_token_arrays(usable_events)
        year_table = _year_table(year)
        token_table = _token_table(token)

        event_path = output / "events" / f"part-{partition:04d}.parquet"
        year_path = output / "author_year_history" / f"part-{partition:04d}.parquet"
        token_path = output / "author_token_history" / f"part-{partition:04d}.parquet"
        _write_parquet_atomic(event_path, event_table)
        _write_parquet_atomic(year_path, year_table)
        _write_parquet_atomic(token_path, token_table)

        sensitivity: dict[str, int] = {}
        for view, selected in (
            ("base", np.ones(len(token["author"]), dtype=bool)),
            ("positive", token["author"] > 0),
        ):
            for cap_name in ("all", "cap10", "cap50", "cap100"):
                sensitivity[f"{view}_{cap_name}_token_count"] = int(np.count_nonzero(selected & (token[f"total_{cap_name}"] > 0)))
                for threshold in THRESHOLDS:
                    sensitivity[f"{view}_{cap_name}_gt_{threshold}"] = int(
                        np.count_nonzero(selected & (token[f"max_{cap_name}"] > threshold))
                    )
        pair_counts = {
            "all": len(usable_events),
            "cap10": int(np.count_nonzero(usable_events["team"] <= 10)),
            "cap50": int(np.count_nonzero(usable_events["team"] <= 50)),
            "cap100": int(np.count_nonzero(usable_events["team"] <= 100)),
        }
        all_author_tokens = int(
            np.count_nonzero(
                np.concatenate(
                    (
                        np.array([True]),
                        events["author"][1:] != events["author"][:-1],
                    )
                )
            )
        ) if len(events) else 0
        checkpoint = {
            "schema_version": AUTHOR_CHECKPOINT_SCHEMA,
            "task_fingerprint_sha256": task_fingerprint,
            "partition_id": partition,
            "event_rows": len(event_table),
            "history_usable_event_rows": len(usable_events),
            "all_distinct_author_token_rows": all_author_tokens,
            "author_year_rows": len(year_table),
            "author_token_rows": len(token_table),
            "a_zero_event_rows": int(np.count_nonzero(events["author"] == 0)),
            "a_zero_token_rows": int(np.count_nonzero(token["author"] == 0)),
            "pair_counts_by_cap": pair_counts,
            "sensitivity": sensitivity,
            "outputs": {
                "events": {
                    "relative_path": event_path.relative_to(output).as_posix(),
                    "bytes": event_path.stat().st_size,
                    "sha256": sha256_file(event_path),
                    "semantic_sha256": _semantic_digest(event_table),
                },
                "author_year": {
                    "relative_path": year_path.relative_to(output).as_posix(),
                    "bytes": year_path.stat().st_size,
                    "sha256": sha256_file(year_path),
                    "semantic_sha256": _semantic_digest(year_table),
                },
                "author_token": {
                    "relative_path": token_path.relative_to(output).as_posix(),
                    "bytes": token_path.stat().st_size,
                    "sha256": sha256_file(token_path),
                    "semantic_sha256": _semantic_digest(token_table),
                },
            },
        }
        atomic_json(_author_checkpoint_path(output, partition), checkpoint)
        checkpoints.append(checkpoint)
    return checkpoints


def _sum_scan_metrics(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for checkpoint in checkpoints:
        for key, value in require_mapping(checkpoint["metrics"], "scan metrics").items():
            result[key] = result.get(key, 0) + require_int(value, f"scan metric {key}")
    return result


def _sum_work_metrics(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    integer_keys = (
        "main_rows",
        "canonical_rows_joined",
        "raw_authorship_slots_joined",
        "unmatched_authorship_slots",
        "authors_count_mismatch_works",
        "null_author_slots",
        "grammar_valid_author_slots",
        "distinct_author_work_pairs",
        "duplicate_author_work_pair_excess_rows",
        "a_zero_distinct_author_work_pairs",
        "history_usable_distinct_author_work_pairs",
    )
    result: dict[str, Any] = {key: 0 for key in integer_keys}
    dispositions = {str(index): 0 for index in range(6)}
    for checkpoint in checkpoints:
        for key in integer_keys:
            result[key] += require_int(checkpoint[key], f"work metric {key}")
        for key, value in require_mapping(
            checkpoint["pair_temporal_disposition_counts"], "disposition counts"
        ).items():
            dispositions[key] += require_int(value, f"disposition {key}")
    result["pair_temporal_disposition_counts"] = dispositions
    return result


def _sum_author_metrics(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "event_rows": 0,
        "history_usable_event_rows": 0,
        "all_distinct_author_token_rows": 0,
        "author_year_rows": 0,
        "author_token_rows": 0,
        "a_zero_event_rows": 0,
        "a_zero_token_rows": 0,
        "pair_counts_by_cap": {"all": 0, "cap10": 0, "cap50": 0, "cap100": 0},
        "sensitivity": {},
    }
    for checkpoint in checkpoints:
        for key in (
            "event_rows",
            "history_usable_event_rows",
            "all_distinct_author_token_rows",
            "author_year_rows",
            "author_token_rows",
            "a_zero_event_rows",
            "a_zero_token_rows",
        ):
            result[key] += require_int(checkpoint[key], f"author metric {key}")
        for key, value in require_mapping(checkpoint["pair_counts_by_cap"], "pair counts").items():
            result["pair_counts_by_cap"][key] += require_int(value, f"pair count {key}")
        for key, value in require_mapping(checkpoint["sensitivity"], "sensitivity").items():
            result["sensitivity"][key] = result["sensitivity"].get(key, 0) + require_int(value, f"sensitivity {key}")
    caps = result["pair_counts_by_cap"]
    require(caps["cap10"] <= caps["cap50"] <= caps["cap100"] <= caps["all"], "team-cap pair counts not cumulative")
    for view in ("base", "positive"):
        for cap in ("all", "cap10", "cap50", "cap100"):
            values = [result["sensitivity"][f"{view}_{cap}_gt_{threshold}"] for threshold in THRESHOLDS]
            require(values == sorted(values, reverse=True), "threshold sensitivity not monotone")
    return result


def _tree_digest(items: Sequence[Mapping[str, Any]]) -> str:
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


def _checkpoint_contract(output: Path, path: Path) -> dict[str, Any]:
    return canonical._artifact_contract(output, path.relative_to(output).as_posix())


def _build_plan(
    *,
    config_sha256: str,
    parent: ParentContext,
    selected_main: Sequence[SourceTask],
    selected_authorship: Sequence[SourceTask],
    scope: str,
) -> dict[str, Any]:
    tool_path = Path(__file__).resolve(strict=True)
    base = {
        "schema_version": PLAN_SCHEMA,
        "scope": scope,
        "config_relative_path": DEFAULT_CONFIG.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": config_sha256,
        "tool_relative_path": tool_path.relative_to(REPO_ROOT).as_posix(),
        "tool_sha256": sha256_file(tool_path),
        "runtime_fingerprint": runtime_fingerprint(),
        "parents": {
            "content_receipt_sha256": sha256_file(parent.content_root / "content_freeze_audit.outputs.json"),
            "temporal_receipt_sha256": sha256_file(parent.temporal_root / "main_temporal_validity_audit.outputs.json"),
            "canonical_receipt_sha256": sha256_file(parent.canonical_root / "work_time_spine_audit.outputs.json"),
            "canonical_result_fingerprint_sha256": parent.canonical_receipt["result_fingerprint_sha256"],
            "canonical_spine_data_tree_sha256": parent.canonical_receipt["spine_data_tree_sha256"],
        },
        "partition_count": PARTITION_COUNT,
        "closure_strategy": "RECEIPT_BOUND_AUTHOR_WORK_EVENT_ARTIFACT",
        "selected_main_tasks": [
            {"relative_path": item.relative_path, "bytes": item.source_bytes, "source_stat": item.source_stat}
            for item in selected_main
        ],
        "selected_authorship_tasks": [
            {"relative_path": item.relative_path, "bytes": item.source_bytes, "source_stat": item.source_stat}
            for item in selected_authorship
        ],
        "full_scope": scope == FULL_SCOPE,
        "operator_paths_redacted": True,
    }
    base["task_fingerprint_sha256"] = sha256_bytes(canonical_json_bytes(base))
    return base


def _preflight(config: Mapping[str, Any], source_root: Path, output_parent: Path) -> dict[str, Any]:
    resource_cfg = require_mapping(config["resource_envelope"], "resource envelope")
    available_ram = canonical._available_memory_bytes()
    free_disk = shutil.disk_usage(output_parent).free
    nofile_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    source_device = os.stat(source_root).st_dev
    output_device = os.stat(output_parent).st_dev
    evidence = {
        "available_memory_bytes": available_ram,
        "required_available_memory_bytes": resource_cfg["minimum_available_ram_bytes"],
        "free_disk_bytes": free_disk,
        "required_free_disk_bytes": resource_cfg["minimum_free_disk_bytes"],
        "nofile_soft_limit": nofile_soft,
        "required_nofile_soft_limit": PARTITION_COUNT + 128,
        "source_device": source_device,
        "output_device": output_device,
        "distinct_devices": source_device != output_device,
    }
    require(available_ram >= resource_cfg["minimum_available_ram_bytes"], "insufficient available RAM")
    require(free_disk >= resource_cfg["minimum_free_disk_bytes"], "insufficient free disk")
    require(nofile_soft >= PARTITION_COUNT + 128, "insufficient RLIMIT_NOFILE")
    require(source_device != output_device, "source and output must use distinct devices")
    canonical._atomic_rename_probe(output_parent)
    evidence["atomic_rename_fsync_probe"] = "PASS"
    return evidence


def _prepare_output(output: Path) -> None:
    parent = output.parent.resolve(strict=True)
    require(not output.exists(), "fresh output directory required")
    output.mkdir(mode=0o700)
    os.chmod(output, 0o700)
    canonical._fsync_directory(parent)
    for relative in (
        "checkpoints/scan/main",
        "checkpoints/scan/authorship",
        "checkpoints/work",
        "checkpoints/author",
        "staging/main",
        "staging/authorship",
        "events",
        "author_year_history",
        "author_token_history",
    ):
        private_dir(output / relative)


def _artifact_paths(output: Path) -> list[Path]:
    excluded = {"audit.lock", "author_history_audit.outputs.json"}
    result: list[Path] = []
    for path in output.rglob("*"):
        if not path.is_file() or path.name in excluded or ".author_history_tmp" in path.parts:
            continue
        require(not path.is_symlink(), "symlink in output tree")
        result.append(path)
    result.sort(key=lambda value: value.relative_to(output).as_posix())
    return result


def _report_fingerprint(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    provenance = dict(payload["provenance"])
    provenance.pop("result_fingerprint_sha256", None)
    payload["provenance"] = provenance
    return sha256_bytes(canonical_json_bytes(payload))


def _render_markdown(report: Mapping[str, Any]) -> str:
    main = report["source_accounting"]["main"]
    authorship = report["source_accounting"]["authorship"]
    work = report["work_event_accounting"]
    author = report["author_history_accounting"]
    return f"""# OpenAlex Work-Side Author History v3

- Status: `{report['status']}`
- Scope: `{report['scope']}`
- GPU used: `false`
- Main rows: {main['raw_rows']:,} raw; {main['retained_rows']:,} retained
- Authorship rows: {authorship['raw_rows']:,} raw; {authorship['retained_rows']:,} retained
- Grammar-valid non-null author slots: {authorship['grammar_valid_non_null_author_id_rows']:,}
- Distinct author-work pairs: {work['distinct_author_work_pairs']:,}
- History-usable distinct author-work pairs: {work['history_usable_distinct_author_work_pairs']:,}
- Author-year rows: {author['author_year_rows']:,}
- Author-token rows: {author['author_token_rows']:,}

This is current-snapshot author-ID-token accounting, not person identification,
historical token observability, candidate eligibility, G-1, G0, empirical G1,
or a scientific effect.
"""


def _finalize(
    *,
    output: Path,
    plan: Mapping[str, Any],
    scope: str,
    preflight: Mapping[str, Any],
    main_scan: Sequence[Mapping[str, Any]],
    authorship_scan: Sequence[Mapping[str, Any]],
    work_checkpoints: Sequence[Mapping[str, Any]],
    author_checkpoints: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    main_metrics = _sum_scan_metrics(main_scan)
    authorship_metrics = _sum_scan_metrics(authorship_scan)
    work_metrics = _sum_work_metrics(work_checkpoints)
    author_metrics = _sum_author_metrics(author_checkpoints)
    full_scope = scope == FULL_SCOPE
    source_cfg = config["source"]
    if full_scope:
        for key in ("raw_rows", "quarantined_w_minus_1_rows", "retained_rows"):
            require(main_metrics[key] == source_cfg["main"][key], f"full main {key} mismatch")
        for key in (
            "raw_rows",
            "quarantined_w_minus_1_rows",
            "retained_rows",
            "null_author_id_rows",
            "grammar_valid_non_null_author_id_rows",
            "malformed_non_null_author_id_rows",
        ):
            require(authorship_metrics[key] == source_cfg["authorship"][key], f"full authorship {key} mismatch")
        require(work_metrics["main_rows"] == source_cfg["main"]["retained_rows"], "full work closure mismatch")
        require(work_metrics["raw_authorship_slots_joined"] == source_cfg["authorship"]["retained_rows"], "full authorship join closure mismatch")
        require(work_metrics["unmatched_authorship_slots"] == 0, "full unmatched authorship slots")
        require(work_metrics["authors_count_mismatch_works"] == 0, "full authors_count mismatches")
        require(author_metrics["event_rows"] == work_metrics["distinct_author_work_pairs"], "all-event closure mismatch")
        require(author_metrics["history_usable_event_rows"] == work_metrics["history_usable_distinct_author_work_pairs"], "usable-event closure mismatch")
        require(author_metrics["all_distinct_author_token_rows"] == source_cfg["authorship"]["distinct_grammar_valid_non_null_author_tokens"], "distinct author-token closure mismatch")

    status = FULL_STATUS if full_scope else PILOT_STATUS
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": status,
        "scope": scope,
        "production_component_pass": full_scope,
        "gpu_used": False,
        "source_accounting": {"main": main_metrics, "authorship": authorship_metrics},
        "work_event_accounting": work_metrics,
        "author_history_accounting": author_metrics,
        "parent_evidence": plan["parents"],
        "preflight": dict(preflight),
        "gate_state": dict(config["gate_state_after_component_pass"]),
        "claim_boundary": config["claim_boundary"],
        "provenance": {
            "config_sha256": plan["config_sha256"],
            "tool_sha256": plan["tool_sha256"],
            "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        },
    }
    report["provenance"]["result_fingerprint_sha256"] = _report_fingerprint(report)
    atomic_json(output / "author_history_audit.json", report)
    atomic_text(output / "author_history_audit.md", _render_markdown(report))
    state = {
        "schema_version": STATE_SCHEMA,
        "status": "COMPLETE_PENDING_RECEIPT",
        "scope": scope,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "result_fingerprint_sha256": report["provenance"]["result_fingerprint_sha256"],
        "production_component_pass": full_scope,
    }
    atomic_json(output / "run_state.json", state)

    artifacts = [canonical._artifact_contract(output, path.relative_to(output).as_posix()) for path in _artifact_paths(output)]
    receipt = {
        "schema_version": OUTPUTS_SCHEMA,
        "status": status,
        "scope": scope,
        "production_component_pass": full_scope,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "result_fingerprint_sha256": report["provenance"]["result_fingerprint_sha256"],
        "canonical_result_fingerprint_sha256": plan["parents"]["canonical_result_fingerprint_sha256"],
        "canonical_spine_data_tree_sha256": plan["parents"]["canonical_spine_data_tree_sha256"],
        "closure_strategy": "RECEIPT_BOUND_AUTHOR_WORK_EVENT_ARTIFACT",
        "artifact_count": len(artifacts),
        "artifact_tree_sha256": _tree_digest(artifacts),
        "artifacts": artifacts,
        "completion_contract": COMPLETION_CONTRACT,
        "operator_paths_redacted": True,
    }
    atomic_json(output / "author_history_audit.outputs.json", receipt)
    _validate_receipt_artifacts(output, receipt, "author output")
    require(sha256_file(output / "author_history_audit.json") == next(item["sha256"] for item in artifacts if item["relative_path"] == "author_history_audit.json"), "final report hash mismatch")
    return report


def _terminal_reentry(output: Path) -> dict[str, Any]:
    receipt = read_json(output / "author_history_audit.outputs.json", "author output receipt")
    require(receipt.get("schema_version") == OUTPUTS_SCHEMA, "terminal receipt schema mismatch")
    require(receipt.get("completion_contract") == COMPLETION_CONTRACT, "terminal completion contract mismatch")
    _validate_receipt_artifacts(output, receipt, "author output")
    artifacts = require_list(receipt["artifacts"], "terminal artifacts")
    require(receipt["artifact_count"] == len(artifacts), "terminal artifact count mismatch")
    require(receipt["artifact_tree_sha256"] == _tree_digest(artifacts), "terminal artifact tree mismatch")
    report = read_json(output / "author_history_audit.json", "author report")
    require(report["provenance"]["result_fingerprint_sha256"] == receipt["result_fingerprint_sha256"], "terminal result fingerprint mismatch")
    require(_report_fingerprint(report) == receipt["result_fingerprint_sha256"], "terminal report fingerprint invalid")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_sha256 = load_config(args.config)
    source_root = args.source_root.resolve(strict=True)
    output = args.output_dir.absolute()
    output_parent = output.parent.resolve(strict=True)
    require(not str(output).startswith(str(source_root) + os.sep), "output may not be inside source root")
    full_scope = args.max_main_files is None and args.max_authorship_files is None
    scope = FULL_SCOPE if full_scope else PILOT_SCOPE
    require(args.execute, "execution requires --execute")
    if full_scope:
        require(args.authorize_full, "full execution requires --authorize-full")
        require(args.acknowledge_exclusive_source_scan, "full execution requires --acknowledge-exclusive-source-scan")

    parent = validate_parents(
        config,
        source_root=source_root,
        content_receipt_path=args.content_receipt,
        temporal_receipt_path=args.temporal_receipt,
        canonical_root=args.canonical_root,
    )
    selected_main = parent.main_tasks[: args.max_main_files] if args.max_main_files is not None else parent.main_tasks
    selected_authorship = parent.authorship_tasks[: args.max_authorship_files] if args.max_authorship_files is not None else parent.authorship_tasks
    require(selected_main and selected_authorship, "selected source task set is empty")
    if full_scope:
        require(len(selected_main) == len(parent.main_tasks), "full main selection incomplete")
        require(len(selected_authorship) == len(parent.authorship_tasks), "full authorship selection incomplete")
    preflight = _preflight(config, source_root, output_parent)
    plan = _build_plan(
        config_sha256=config_sha256,
        parent=parent,
        selected_main=selected_main,
        selected_authorship=selected_authorship,
        scope=scope,
    )

    if output.exists() and (output / "author_history_audit.outputs.json").is_file():
        return _terminal_reentry(output)
    _prepare_output(output)
    with canonical._output_lock(output):
        atomic_json(output / "audit_plan.json", plan)
        atomic_json(
            output / "run_state.json",
            {
                "schema_version": STATE_SCHEMA,
                "status": "RUNNING",
                "scope": scope,
                "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
                "production_component_pass": False,
            },
        )
        try:
            main_scan: list[dict[str, Any]] = []
            for index, task in enumerate(selected_main, start=1):
                print(f"main scan {index}/{len(selected_main)} {task.relative_path}", file=sys.stderr, flush=True)
                main_scan.append(_scan_main_task(source_root, output, task, plan["task_fingerprint_sha256"]))
            authorship_scan: list[dict[str, Any]] = []
            for index, task in enumerate(selected_authorship, start=1):
                print(f"authorship scan {index}/{len(selected_authorship)} {task.relative_path}", file=sys.stderr, flush=True)
                authorship_scan.append(_scan_authorship_task(source_root, output, task, plan["task_fingerprint_sha256"]))

            print("building 4096 work partitions", file=sys.stderr, flush=True)
            main_spools, slot_spools = build_work_spools(output, main_scan, authorship_scan)
            print("joining work partitions and emitting author events", file=sys.stderr, flush=True)
            work_checkpoints, event_spools = process_work_partitions(
                output=output,
                canonical_root=parent.canonical_root,
                main_spools=main_spools,
                slot_spools=slot_spools,
                task_fingerprint=plan["task_fingerprint_sha256"],
                full_scope=full_scope,
            )
            shutil.rmtree(output / ".author_history_tmp" / "work")
            print("reducing author partitions", file=sys.stderr, flush=True)
            author_checkpoints = process_author_partitions(
                output=output,
                event_spools=event_spools,
                task_fingerprint=plan["task_fingerprint_sha256"],
            )
            shutil.rmtree(output / ".author_history_tmp")
            canonical._fsync_directory(output)
            return _finalize(
                output=output,
                plan=plan,
                scope=scope,
                preflight=preflight,
                main_scan=main_scan,
                authorship_scan=authorship_scan,
                work_checkpoints=work_checkpoints,
                author_checkpoints=author_checkpoints,
                config=config,
            )
        except BaseException:
            with contextlib.suppress(BaseException):
                atomic_json(
                    output / "run_state.json",
                    {
                        "schema_version": STATE_SCHEMA,
                        "status": "FAILED_NO_VALID_RECEIPT",
                        "scope": scope,
                        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
                        "production_component_pass": False,
                    },
                )
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--content-receipt", type=Path, required=True)
    parser.add_argument("--temporal-receipt", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-main-files", type=int)
    parser.add_argument("--max-authorship-files", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-full", action="store_true")
    parser.add_argument("--acknowledge-exclusive-source-scan", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("max_main_files", "max_authorship_files"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            print(f"error: {name} must be positive", file=sys.stderr)
            return 2
    try:
        report = run(args)
    except (AuthorHistoryError, canonical.SpineAuditError, OSError, pa.ArrowException) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
