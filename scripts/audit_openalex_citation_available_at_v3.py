#!/usr/bin/env python3
"""Receipt-bound production citation chronology successor.

The component consumes the validated canonical work-time spine and current
OpenAlex referenced-work rows.  It publishes exact current-snapshot directed
pair accounting plus operator-local chronology-safe proxy edges.  Publication
date is a bibliographic convention; this is not a historical edge-observation
reconstruction.
"""

from __future__ import annotations

import argparse
import contextlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

import audit_openalex_canonical_work_time_spine_v3 as canonical
import audit_openalex_work_side_author_history_v3 as helper


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/openalex_g1_citation_available_at_v2.yaml"
CONFIG_SHA256 = "735239ecd2194ec60542c7c5b6e5e15f4634ee9e8d0449b244daf261df58414b"
CONFIG_SCHEMA = "openalex_g1_citation_edge_chronology_production_config.v3"
PLAN_SCHEMA = "openalex_g1_citation_edge_chronology_production_plan.v3"
SCAN_CHECKPOINT_SCHEMA = "openalex_g1_citation_edge_chronology_scan_checkpoint.v3"
PAIR_CHECKPOINT_SCHEMA = "openalex_g1_citation_edge_chronology_pair_checkpoint.v3"
SAFE_CHECKPOINT_SCHEMA = "openalex_g1_citation_edge_chronology_safe_checkpoint.v3"
REPORT_SCHEMA = "openalex_g1_citation_edge_chronology_report.v3"
OUTPUTS_SCHEMA = "openalex_g1_citation_edge_chronology_outputs.v3"
STATE_SCHEMA = "openalex_g1_citation_edge_chronology_state.v3"
MANIFEST_SCHEMA = "openalex_g1_citation_edge_partition_manifest.v3"

FULL_SCOPE = "FULL_CURRENT_SNAPSHOT_PRODUCTION"
PILOT_SCOPE = "PRODUCTION_SOURCE_PREFIX_PILOT_NONPROMOTABLE"
FULL_STATUS = (
    "PASS_CURRENT_SNAPSHOT_CITATION_EDGE_CHRONOLOGY_ACCOUNTING_WITH_CANONICAL_"
    "TEMPORAL_V3_PARENT_AND_BIBLIOGRAPHIC_EVENT_TIME_CONVENTION_LIMITATION"
)
PILOT_STATUS = "PILOT_CITATION_EDGE_CHRONOLOGY_ONLY"
COMPLETION_CONTRACT = "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"

PARTITION_COUNT = 4096
BATCH_ROWS = 262_144
WORK_PATTERN = r"^https://openalex[.]org/W(0|[1-9][0-9]*)$"
W_MINUS_ONE = "https://openalex.org/W-1"
USABLE = frozenset({2, 3})
DAY_BIAS = 400_000

BUCKET_NAMES = (
    "self_loop",
    "source_temporal_unusable",
    "target_external_open_world",
    "target_temporal_unusable",
    "negative_lag_target_after_source",
    "same_day",
    "positive_lag",
)
BUCKET_CODE = {name: index for index, name in enumerate(BUCKET_NAMES)}

PAIR_DTYPE = np.dtype([("source", "<u8"), ("target", "<u8")], align=False)
SAFE_DTYPE = np.dtype(
    [
        ("source", "<u8"),
        ("target", "<u8"),
        ("source_day", "<i4"),
        ("target_day", "<i4"),
        ("lag", "<i4"),
        ("multiplicity", "<u4"),
    ],
    align=False,
)

DEDUP_SCHEMA = pa.schema(
    [
        pa.field("citing_work_id", pa.uint64(), nullable=False),
        pa.field("cited_work_id", pa.uint64(), nullable=False),
        pa.field("raw_pair_multiplicity", pa.uint32(), nullable=False),
        pa.field("citing_temporal_disposition", pa.uint8(), nullable=False),
        pa.field("target_membership", pa.uint8(), nullable=False),
        pa.field("cited_temporal_disposition", pa.uint8(), nullable=True),
        pa.field("citing_publication_date", pa.date32(), nullable=True),
        pa.field("available_at", pa.date32(), nullable=True),
        pa.field("cited_publication_date", pa.date32(), nullable=True),
        pa.field("signed_lag_days", pa.int32(), nullable=True),
        pa.field("primary_bucket", pa.uint8(), nullable=False),
    ]
)
SAFE_SCHEMA = pa.schema(
    [
        pa.field("cited_work_id", pa.uint64(), nullable=False),
        pa.field("citing_work_id", pa.uint64(), nullable=False),
        pa.field("available_at", pa.date32(), nullable=False),
        pa.field("cited_publication_date", pa.date32(), nullable=False),
        pa.field("signed_lag_days", pa.int32(), nullable=False),
        pa.field("raw_pair_multiplicity", pa.uint32(), nullable=False),
    ]
)

AXIS_CATEGORIES = {
    "primary_bucket_x_source_temporal_disposition": tuple(
        f"disposition_{value}" for value in range(6)
    ),
    "primary_bucket_x_target_membership": ("external_open_world", "in_snapshot"),
    "primary_bucket_x_target_temporal_disposition_or_external_na": (
        "disposition_0",
        "disposition_1",
        "disposition_2",
        "disposition_3",
        "disposition_4",
        "disposition_5",
        "external_NA",
    ),
    "primary_bucket_x_self_loop_flag": ("false", "true"),
    "primary_bucket_x_signed_lag_class_or_na": (
        "negative",
        "same_day",
        "positive",
        "NA",
    ),
    "primary_bucket_x_pair_multiplicity_class": (
        "one",
        "two",
        "three_or_more",
    ),
}

LAG_BIN_EDGES = np.array(
    [np.iinfo(np.int32).min, -365, -30, 0, 1, 31, 366, 1826, np.iinfo(np.int32).max],
    dtype=np.int64,
)
LAG_BIN_LABELS = (
    "lt_minus_365",
    "minus_365_to_minus_31",
    "minus_30_to_minus_1",
    "zero",
    "plus_1_to_30",
    "plus_31_to_365",
    "plus_366_to_1825",
    "gt_1825",
)


class CitationProductionError(RuntimeError):
    """Controlled fail-closed production error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CitationProductionError(message)


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


def atomic_json(path: Path, value: Any) -> None:
    canonical._atomic_bytes(path, canonical_json_bytes(value))


def atomic_text(path: Path, value: str) -> None:
    canonical._atomic_bytes(path, value.encode("utf-8"))


def private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field} must be an object")
    return dict(value)


def _list(value: Any, field: str) -> list[Any]:
    require(isinstance(value, list), f"{field} must be an array")
    return list(value)


def _integer(value: Any, field: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{field} must be int")
    return int(value)


def _safe_relative(value: Any, field: str) -> Path:
    require(isinstance(value, str) and value, f"{field} must be a path")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe {field}")
    require(path.as_posix() == value, f"noncanonical {field}")
    return path


@dataclass
class ParentContext:
    content_root: Path
    temporal_root: Path
    canonical_root: Path
    content_receipt: dict[str, Any]
    temporal_receipt: dict[str, Any]
    canonical_receipt: dict[str, Any]
    content_plan: dict[str, Any]
    reference_tasks: list[helper.SourceTask]


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve(strict=True)
    require(resolved == DEFAULT_CONFIG.resolve(strict=True), "only the frozen v2 production config is accepted")
    payload = canonical._read_regular_file_bytes(resolved, field="config")
    digest = sha256_bytes(payload)
    require(digest == CONFIG_SHA256, "config SHA-256 mismatch")
    try:
        value = yaml.load(payload.decode("utf-8"), Loader=helper._UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CitationProductionError("invalid strict YAML config") from exc
    config = _mapping(value, "config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    predecessor = _mapping(config["frozen_predecessors"], "frozen_predecessors")
    for relative_text, expected in _mapping(
        predecessor["paths_and_sha256"], "frozen predecessor hashes"
    ).items():
        target = REPO_ROOT / _safe_relative(relative_text, "frozen predecessor path")
        require(sha256_file(target) == expected, f"frozen predecessor mismatch: {relative_text}")
    dependencies = _mapping(config["implementation_dependencies"], "implementation dependencies")
    for prefix in ("canonical_helper", "partition_helper"):
        relative = _safe_relative(dependencies[f"{prefix}_path"], f"{prefix} path")
        require(
            sha256_file(REPO_ROOT / relative) == dependencies[f"{prefix}_sha256"],
            f"{prefix} hash mismatch",
        )
    return config, digest


def validate_parents(
    config: Mapping[str, Any],
    *,
    source_root: Path,
    content_receipt_path: Path,
    temporal_receipt_path: Path,
    canonical_root: Path,
) -> ParentContext:
    parents = _mapping(config["parents"], "parents")
    content_expected = _mapping(parents["current_content_baseline"], "content parent")
    temporal_expected = _mapping(parents["main_temporal_v3"], "temporal parent")
    canonical_expected = _mapping(parents["canonical_work_time_spine"], "canonical parent")
    content_root = content_receipt_path.resolve(strict=True).parent
    temporal_root = temporal_receipt_path.resolve(strict=True).parent
    canonical_root = canonical_root.resolve(strict=True)
    require(content_receipt_path.name == "content_freeze_audit.outputs.json", "content receipt basename mismatch")
    require(temporal_receipt_path.name == "main_temporal_validity_audit.outputs.json", "temporal receipt basename mismatch")
    content = helper._validate_parent_receipt(
        content_root, content_receipt_path.name, content_expected, "content"
    )
    require(content.get("plan_sha256") == content_expected["plan_sha256"], "content plan SHA mismatch")
    require(content.get("content_tree_sha256") == content_expected["content_tree_sha256"], "content tree mismatch")
    temporal = helper._validate_parent_receipt(
        temporal_root, temporal_receipt_path.name, temporal_expected, "temporal"
    )
    canonical_receipt = helper._validate_parent_receipt(
        canonical_root,
        "work_time_spine_audit.outputs.json",
        canonical_expected,
        "canonical",
    )
    require(canonical_receipt.get("production_component_pass") is True, "canonical parent is not a production PASS")
    require(canonical_receipt.get("artifact_count") == canonical_expected["artifact_count"], "canonical artifact count mismatch")
    require(canonical_receipt.get("spine_data_tree_sha256") == canonical_expected["spine_data_tree_sha256"], "canonical spine tree mismatch")
    canonical_plan = helper.read_json(canonical_root / "audit_plan.json", "canonical plan")
    canonical_parents = _mapping(canonical_plan.get("parents"), "canonical plan parents")
    require(canonical_parents["content"]["receipt_sha256"] == content_expected["receipt_sha256"], "canonical/content cross-binding mismatch")
    require(canonical_parents["temporal"]["receipt_sha256"] == temporal_expected["receipt_sha256"], "canonical/temporal cross-binding mismatch")
    require(canonical_plan.get("config_sha256") == canonical_expected["config_sha256"], "canonical config mismatch")
    require(canonical_plan.get("tool_sha256") == canonical_expected["runner_sha256"], "canonical runner mismatch")

    content_plan = helper.read_json(content_root / "audit_plan.json", "content plan")
    require(content_plan.get("plan_sha256") == content_expected["plan_sha256"], "content plan payload mismatch")
    source_cfg = _mapping(config["source"], "source")
    require(source_root.resolve(strict=True).name == source_cfg["root_id"], "source root ID mismatch")
    table_name = source_cfg["referenced_work_table"]
    tasks = [
        helper._task_from_content(_mapping(item, "content task"), table_name)
        for item in _list(content_plan.get("tasks"), "content plan tasks")
        if isinstance(item, dict) and item.get("table_name") == table_name
    ]
    tasks.sort(key=lambda item: item.relative_path)
    reference_cfg = _mapping(source_cfg["referenced_works"], "referenced source")
    require(len(tasks) == reference_cfg["files"], "reference source file count mismatch")
    require(sum(item.source_bytes for item in tasks) == reference_cfg["source_bytes"], "reference source bytes mismatch")
    return ParentContext(
        content_root=content_root,
        temporal_root=temporal_root,
        canonical_root=canonical_root,
        content_receipt=content,
        temporal_receipt=temporal,
        canonical_receipt=canonical_receipt,
        content_plan=content_plan,
        reference_tasks=tasks,
    )


def _new_sparse_memmap(path: Path, dtype: str, length: int) -> np.memmap:
    private_dir(path.parent)
    itemsize = np.dtype(dtype).itemsize
    raw = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.ftruncate(raw, itemsize * length)
        os.fsync(raw)
    finally:
        os.close(raw)
    return np.memmap(path, dtype=dtype, mode="r+", shape=(length,))


def _canonical_partition(root: Path, partition: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = root / "spine" / f"part-{partition:04d}.parquet"
    table = pq.read_table(
        path,
        columns=["work_id_numeric", "publication_day", "temporal_disposition"],
    )
    work = table["work_id_numeric"].combine_chunks().to_numpy(zero_copy_only=False).astype("<u8", copy=False)
    day_array = table["publication_day"].combine_chunks()
    day = pc.fill_null(day_array, 0).to_numpy(zero_copy_only=False).astype("<i4", copy=False)
    disposition = table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False).astype("u1", copy=False)
    require(len(work) == len(day) == len(disposition), "canonical partition length mismatch")
    require(not len(work) or bool(np.all(work[1:] > work[:-1])), "canonical work IDs are not strictly increasing")
    require(not len(work) or bool(np.all(work % PARTITION_COUNT == partition)), "canonical partition formula mismatch")
    require(bool(np.all(disposition <= 5)), "unknown canonical temporal disposition")
    require(bool(np.array_equal(pc.is_null(day_array).to_numpy(zero_copy_only=False), disposition == 0)), "canonical date nullability mismatch")
    return work, day, disposition


def build_canonical_lookup(
    output: Path,
    canonical_root: Path,
    *,
    maximum_numeric_id: int,
    expected_rows: int,
) -> tuple[np.memmap, Path]:
    path = output / ".citation_tmp" / "canonical_lookup.u32"
    lookup = _new_sparse_memmap(path, "<u4", maximum_numeric_id + 1)
    observed_rows = 0
    for partition in range(PARTITION_COUNT):
        if partition % 128 == 0:
            print(f"canonical lookup {partition}/{PARTITION_COUNT}", file=sys.stderr, flush=True)
        work, day, disposition = _canonical_partition(canonical_root, partition)
        encoded_day = day.astype(np.int64) + DAY_BIAS
        require(bool(np.all((encoded_day >= 0) & (encoded_day < 2**29))), "canonical publication day outside lookup encoding")
        packed = ((encoded_day.astype(np.uint32) << np.uint32(3)) | (disposition.astype(np.uint32) + np.uint32(1)))
        require(bool(np.all(packed != 0)), "canonical lookup zero sentinel collision")
        lookup[work.astype(np.intp, copy=False)] = packed
        observed_rows += len(work)
    require(observed_rows == expected_rows, "canonical lookup row count mismatch")
    lookup.flush()
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    return lookup, path


def _lookup_values(lookup: np.memmap, ids: np.ndarray) -> np.ndarray:
    result = np.zeros(len(ids), dtype="<u4")
    in_range = ids < len(lookup)
    if bool(np.any(in_range)):
        result[in_range] = lookup[ids[in_range].astype(np.intp, copy=False)]
    return result


def _decode_packed(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    require(bool(np.all(values != 0)), "cannot decode absent canonical lookup value")
    disposition = ((values & np.uint32(7)) - np.uint32(1)).astype("u1")
    day = ((values >> np.uint32(3)).astype(np.int64) - DAY_BIAS).astype("<i4")
    return day, disposition


def _scan_checkpoint_path(output: Path, task: helper.SourceTask) -> Path:
    digest = hashlib.sha256(task.relative_path.encode("utf-8")).hexdigest()
    return output / "checkpoints" / "scan" / f"{digest}.json"


def _scan_reference_task(
    *,
    source_root: Path,
    output: Path,
    task: helper.SourceTask,
    task_fingerprint: str,
    lookup: np.memmap,
    target_seen: np.memmap,
    outside_targets: set[int],
    writer: helper.SpoolWriter,
) -> dict[str, Any]:
    source = helper._source_path(source_root, task)
    parquet = pq.ParquetFile(source)
    expected_schema = {"id": "string", "referenced_works": "string"}
    require(helper._projected_schema(parquet, tuple(expected_schema)) == expected_schema, "reference projected schema mismatch")
    metrics = {
        "raw_pair_rows": 0,
        "raw_in_snapshot_target_rows": 0,
        "raw_external_target_rows": 0,
        "raw_self_loop_rows": 0,
        "raw_pair_rows_with_available_at": 0,
        "null_source_rows": 0,
        "malformed_source_rows": 0,
        "source_w_minus_1_rows": 0,
        "source_w_zero_rows": 0,
        "null_target_rows": 0,
        "malformed_target_rows": 0,
        "target_w_minus_1_rows": 0,
        "target_w_zero_rows": 0,
    }
    payload_hash = hashlib.sha256()
    for batch in parquet.iter_batches(batch_size=BATCH_ROWS, columns=list(expected_schema)):
        source_ids = batch.column(0)
        target_ids = batch.column(1)
        metrics["null_source_rows"] += source_ids.null_count
        metrics["null_target_rows"] += target_ids.null_count
        require(source_ids.null_count == 0 and target_ids.null_count == 0, "null reference work ID")
        source_minus = helper._bool_array(pc.equal(source_ids, W_MINUS_ONE))
        target_minus = helper._bool_array(pc.equal(target_ids, W_MINUS_ONE))
        source_valid = helper._bool_array(pc.match_substring_regex(source_ids, pattern=WORK_PATTERN))
        target_valid = helper._bool_array(pc.match_substring_regex(target_ids, pattern=WORK_PATTERN))
        metrics["source_w_minus_1_rows"] += int(np.count_nonzero(source_minus))
        metrics["target_w_minus_1_rows"] += int(np.count_nonzero(target_minus))
        metrics["malformed_source_rows"] += int(np.count_nonzero(~source_valid & ~source_minus))
        metrics["malformed_target_rows"] += int(np.count_nonzero(~target_valid & ~target_minus))
        metrics["source_w_zero_rows"] += int(np.count_nonzero(helper._bool_array(pc.equal(source_ids, "https://openalex.org/W0"))))
        metrics["target_w_zero_rows"] += int(np.count_nonzero(helper._bool_array(pc.equal(target_ids, "https://openalex.org/W0"))))
        require(bool(np.all(source_valid)) and bool(np.all(target_valid)), "sentinel or malformed reference work ID")
        active = np.arange(len(source_ids), dtype=np.int64)
        source_numeric = helper._parse_numeric_strings(
            source_ids,
            active=active,
            prefix_length=len("https://openalex.org/W"),
            field="citing work ID",
        )
        target_numeric = helper._parse_numeric_strings(
            target_ids,
            active=active,
            prefix_length=len("https://openalex.org/W"),
            field="cited work ID",
        )
        require(bool(np.all(source_numeric < len(lookup))), "citing work outside canonical lookup range")
        source_packed = lookup[source_numeric.astype(np.intp, copy=False)]
        require(bool(np.all(source_packed != 0)), "citing work missing from canonical spine")
        _, source_disposition = _decode_packed(source_packed)
        target_packed = _lookup_values(lookup, target_numeric)
        internal = target_packed != 0
        in_range = target_numeric < len(target_seen)
        if bool(np.any(in_range)):
            target_seen[target_numeric[in_range].astype(np.intp, copy=False)] = np.uint8(1)
        if bool(np.any(~in_range)):
            outside_targets.update(int(value) for value in target_numeric[~in_range].tolist())
        block = np.empty(len(source_numeric), dtype=PAIR_DTYPE)
        block["source"] = source_numeric
        block["target"] = target_numeric
        payload_hash.update(block.tobytes(order="C"))
        writer.write(block, (source_numeric % PARTITION_COUNT).astype("<u2"))
        metrics["raw_pair_rows"] += len(block)
        metrics["raw_in_snapshot_target_rows"] += int(np.count_nonzero(internal))
        metrics["raw_external_target_rows"] += int(np.count_nonzero(~internal))
        metrics["raw_self_loop_rows"] += int(np.count_nonzero(source_numeric == target_numeric))
        metrics["raw_pair_rows_with_available_at"] += int(np.count_nonzero(np.isin(source_disposition, np.array(sorted(USABLE), dtype="u1"))))
    require(metrics["raw_pair_rows"] == parquet.metadata.num_rows, "reference footer row mismatch")
    checkpoint = {
        "schema_version": SCAN_CHECKPOINT_SCHEMA,
        "task_fingerprint_sha256": task_fingerprint,
        "relative_source_path": task.relative_path,
        "source_identity": task.source_stat,
        "projected_schema": expected_schema,
        "projected_pair_payload_sha256": payload_hash.hexdigest(),
        "metrics": metrics,
    }
    atomic_json(_scan_checkpoint_path(output, task), checkpoint)
    return checkpoint


def _write_parquet_atomic(path: Path, table: pa.Table) -> None:
    private_dir(path.parent)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    require(not temporary.exists(), "stale parquet temporary path")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=6,
            row_group_size=BATCH_ROWS,
            use_dictionary=False,
            write_statistics=True,
        )
        os.chmod(temporary, 0o600)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        canonical._fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _dedup_table(
    source: np.ndarray,
    target: np.ndarray,
    multiplicity: np.ndarray,
    source_disposition: np.ndarray,
    internal: np.ndarray,
    target_disposition: np.ndarray,
    source_day: np.ndarray,
    target_day: np.ndarray,
    lag: np.ndarray,
    source_has_day: np.ndarray,
    target_has_day: np.ndarray,
    lag_available: np.ndarray,
    source_usable: np.ndarray,
    bucket: np.ndarray,
) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array(source, type=pa.uint64()),
            pa.array(target, type=pa.uint64()),
            pa.array(multiplicity, type=pa.uint32()),
            pa.array(source_disposition, type=pa.uint8()),
            pa.array(internal.astype("u1"), type=pa.uint8()),
            pa.array(target_disposition, mask=~internal, type=pa.uint8()),
            pa.array(source_day, mask=~source_has_day, type=pa.date32()),
            pa.array(source_day, mask=~source_usable, type=pa.date32()),
            pa.array(target_day, mask=~target_has_day, type=pa.date32()),
            pa.array(lag, mask=~lag_available, type=pa.int32()),
            pa.array(bucket, type=pa.uint8()),
        ],
        schema=DEDUP_SCHEMA,
    )


def _update_cross_tabs(
    totals: dict[str, np.ndarray], bucket: np.ndarray, categories: dict[str, np.ndarray]
) -> None:
    for axis, values in categories.items():
        category_count = len(AXIS_CATEGORIES[axis])
        joint = bucket.astype(np.int64) * category_count + values.astype(np.int64)
        observed = np.bincount(
            joint, minlength=len(BUCKET_NAMES) * category_count
        ).reshape(len(BUCKET_NAMES), category_count)
        totals[axis] += observed.astype(np.uint64)


def _pair_checkpoint_path(output: Path, partition: int) -> Path:
    return output / "checkpoints" / "pair" / f"part-{partition:04d}.json"


def process_pair_partitions(
    *,
    output: Path,
    raw_spools: Sequence[Path],
    lookup: np.memmap,
    task_fingerprint: str,
) -> tuple[list[dict[str, Any]], list[Path], dict[str, Any]]:
    safe_root = output / ".citation_tmp" / "safe_by_target"
    checkpoints: list[dict[str, Any]] = []
    bucket_totals = np.zeros(len(BUCKET_NAMES), dtype=np.uint64)
    cross_totals = {
        axis: np.zeros((len(BUCKET_NAMES), len(categories)), dtype=np.uint64)
        for axis, categories in AXIS_CATEGORIES.items()
    }
    aggregate = Counter()
    multiplicity_histogram: Counter[int] = Counter()
    lag_bins = np.zeros(len(LAG_BIN_LABELS), dtype=np.uint64)
    sensitivity = Counter(
        {"primary_zero_day": 0, "tolerance_30_days": 0, "tolerance_365_days": 0, "annual_only": 0}
    )
    dedup_records: list[dict[str, Any]] = []
    with helper.SpoolWriter(safe_root, SAFE_DTYPE, prefix="safe") as safe_writer:
        for partition in range(PARTITION_COUNT):
            if partition % 64 == 0:
                print(f"pair reduction {partition}/{PARTITION_COUNT}", file=sys.stderr, flush=True)
            raw = helper._read_records(raw_spools[partition], PAIR_DTYPE)
            if len(raw):
                order = np.lexsort((raw["target"], raw["source"]))
                raw = raw[order]
                start = np.ones(len(raw), dtype=bool)
                start[1:] = (raw["source"][1:] != raw["source"][:-1]) | (raw["target"][1:] != raw["target"][:-1])
                starts = np.flatnonzero(start)
                ends = np.concatenate((starts[1:], np.array([len(raw)], dtype=np.int64)))
                multiplicity64 = ends.astype(np.uint64) - starts.astype(np.uint64)
                require(bool(np.all(multiplicity64 <= np.iinfo(np.uint32).max)), "pair multiplicity exceeds uint32")
                multiplicity = multiplicity64.astype("<u4")
                unique = raw[starts]
            else:
                unique = np.empty(0, dtype=PAIR_DTYPE)
                multiplicity = np.empty(0, dtype="<u4")
            source = unique["source"]
            target = unique["target"]
            source_packed = _lookup_values(lookup, source)
            require(bool(np.all(source_packed != 0)), "deduplicated source missing canonical time")
            source_day, source_disposition = _decode_packed(source_packed) if len(source) else (np.empty(0, dtype="<i4"), np.empty(0, dtype="u1"))
            target_packed = _lookup_values(lookup, target)
            internal = target_packed != 0
            target_day = np.zeros(len(target), dtype="<i4")
            target_disposition = np.zeros(len(target), dtype="u1")
            if bool(np.any(internal)):
                decoded_day, decoded_disposition = _decode_packed(target_packed[internal])
                target_day[internal] = decoded_day
                target_disposition[internal] = decoded_disposition
            source_has_day = source_disposition != 0
            target_has_day = internal & (target_disposition != 0)
            lag_available = source_has_day & target_has_day
            lag = np.zeros(len(source), dtype="<i4")
            if bool(np.any(lag_available)):
                lag64 = source_day[lag_available].astype(np.int64) - target_day[lag_available].astype(np.int64)
                require(bool(np.all((lag64 >= np.iinfo(np.int32).min) & (lag64 <= np.iinfo(np.int32).max))), "signed lag outside int32")
                lag[lag_available] = lag64.astype("<i4")
            source_usable = np.isin(source_disposition, np.array(sorted(USABLE), dtype="u1"))
            target_usable = internal & np.isin(target_disposition, np.array(sorted(USABLE), dtype="u1"))
            self_loop = source == target
            bucket = np.full(len(source), np.uint8(255), dtype="u1")
            remaining = np.ones(len(source), dtype=bool)
            for name, condition in (
                ("self_loop", self_loop),
                ("source_temporal_unusable", ~source_usable),
                ("target_external_open_world", ~internal),
                ("target_temporal_unusable", ~target_usable),
                ("negative_lag_target_after_source", lag < 0),
                ("same_day", lag == 0),
                ("positive_lag", lag > 0),
            ):
                selected = remaining & condition
                bucket[selected] = np.uint8(BUCKET_CODE[name])
                remaining[selected] = False
            require(not bool(np.any(remaining)) and not bool(np.any(bucket == 255)), "pair classifier is not total")
            observed_buckets = np.bincount(bucket, minlength=len(BUCKET_NAMES)).astype(np.uint64)
            bucket_totals += observed_buckets

            lag_class = np.full(len(source), 3, dtype="u1")
            lag_class[lag_available & (lag < 0)] = 0
            lag_class[lag_available & (lag == 0)] = 1
            lag_class[lag_available & (lag > 0)] = 2
            multiplicity_class = np.full(len(source), 2, dtype="u1")
            multiplicity_class[multiplicity == 1] = 0
            multiplicity_class[multiplicity == 2] = 1
            target_disposition_category = target_disposition.copy()
            target_disposition_category[~internal] = 6
            _update_cross_tabs(
                cross_totals,
                bucket,
                {
                    "primary_bucket_x_source_temporal_disposition": source_disposition,
                    "primary_bucket_x_target_membership": internal.astype("u1"),
                    "primary_bucket_x_target_temporal_disposition_or_external_na": target_disposition_category,
                    "primary_bucket_x_self_loop_flag": self_loop.astype("u1"),
                    "primary_bucket_x_signed_lag_class_or_na": lag_class,
                    "primary_bucket_x_pair_multiplicity_class": multiplicity_class,
                },
            )
            if len(multiplicity):
                values, counts = np.unique(multiplicity, return_counts=True)
                multiplicity_histogram.update({int(value): int(count) for value, count in zip(values, counts, strict=True)})
            if bool(np.any(lag_available)):
                lag_values = lag[lag_available].astype(np.int64)
                observed, _ = np.histogram(lag_values, bins=LAG_BIN_EDGES)
                lag_bins += observed.astype(np.uint64)
            common = (~self_loop) & source_usable & target_usable
            sensitivity["primary_zero_day"] += int(np.count_nonzero(common & (lag >= 0)))
            sensitivity["tolerance_30_days"] += int(np.count_nonzero(common & (lag >= -30)))
            sensitivity["tolerance_365_days"] += int(np.count_nonzero(common & (lag >= -365)))
            if bool(np.any(common)):
                source_year = source_day[common].astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64)
                target_year = target_day[common].astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64)
                sensitivity["annual_only"] += int(np.count_nonzero(target_year <= source_year))

            safe = np.isin(bucket, np.array([BUCKET_CODE["same_day"], BUCKET_CODE["positive_lag"]], dtype="u1"))
            safe_values = np.empty(int(np.count_nonzero(safe)), dtype=SAFE_DTYPE)
            safe_values["source"] = source[safe]
            safe_values["target"] = target[safe]
            safe_values["source_day"] = source_day[safe]
            safe_values["target_day"] = target_day[safe]
            safe_values["lag"] = lag[safe]
            safe_values["multiplicity"] = multiplicity[safe]
            safe_writer.write(safe_values, (safe_values["target"] % PARTITION_COUNT).astype("<u2"))

            dedup_path = output / "deduplicated_edge_events" / f"part-{partition:04d}.parquet"
            _write_parquet_atomic(
                dedup_path,
                _dedup_table(
                    source,
                    target,
                    multiplicity,
                    source_disposition,
                    internal,
                    target_disposition,
                    source_day,
                    target_day,
                    lag,
                    source_has_day,
                    target_has_day,
                    lag_available,
                    source_usable,
                    bucket,
                ),
            )
            dedup_record = {
                "partition_id": partition,
                "relative_path": dedup_path.relative_to(output).as_posix(),
                "rows": len(source),
                "bytes": dedup_path.stat().st_size,
                "sha256": sha256_file(dedup_path),
            }
            dedup_records.append(dedup_record)
            checkpoint = {
                "schema_version": PAIR_CHECKPOINT_SCHEMA,
                "task_fingerprint_sha256": task_fingerprint,
                "partition_id": partition,
                "raw_pair_rows": len(raw),
                "unique_pair_rows": len(source),
                "duplicate_pair_groups": int(np.count_nonzero(multiplicity > 1)),
                "duplicate_pair_row_excess": len(raw) - len(source),
                "maximum_pair_multiplicity": int(multiplicity.max(initial=0)),
                "raw_self_loop_rows": int(multiplicity[self_loop].sum(dtype=np.uint64)),
                "unique_self_loop_pairs": int(np.count_nonzero(self_loop)),
                "unique_in_snapshot_target_pairs": int(np.count_nonzero(internal)),
                "unique_external_target_pairs": int(np.count_nonzero(~internal)),
                "unique_pairs_with_available_at": int(np.count_nonzero(source_usable)),
                "bucket_counts": {name: int(observed_buckets[index]) for index, name in enumerate(BUCKET_NAMES)},
                "chronology_safe_primary_pairs": len(safe_values),
                "raw_spool_sha256": sha256_file(raw_spools[partition]),
                "deduplicated_edge_artifact": dedup_record,
            }
            atomic_json(_pair_checkpoint_path(output, partition), checkpoint)
            checkpoints.append(checkpoint)
            aggregate["raw_pair_rows"] += len(raw)
            aggregate["unique_pair_rows"] += len(source)
            aggregate["duplicate_pair_groups"] += checkpoint["duplicate_pair_groups"]
            aggregate["duplicate_pair_row_excess"] += checkpoint["duplicate_pair_row_excess"]
            aggregate["raw_self_loop_rows"] += checkpoint["raw_self_loop_rows"]
            aggregate["unique_self_loop_pairs"] += checkpoint["unique_self_loop_pairs"]
            aggregate["unique_in_snapshot_target_pairs"] += checkpoint["unique_in_snapshot_target_pairs"]
            aggregate["unique_external_target_pairs"] += checkpoint["unique_external_target_pairs"]
            aggregate["unique_pairs_with_available_at"] += checkpoint["unique_pairs_with_available_at"]
            aggregate["chronology_safe_primary_pairs"] += len(safe_values)
            aggregate["maximum_pair_multiplicity"] = max(aggregate["maximum_pair_multiplicity"], checkpoint["maximum_pair_multiplicity"])

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "kind": "DEDUPLICATED_DIRECTED_EDGE_EVENTS_BY_CITING_WORK_MODULO",
        "task_fingerprint_sha256": task_fingerprint,
        "partition_count": PARTITION_COUNT,
        "rows": int(aggregate["unique_pair_rows"]),
        "partition_tree_sha256": helper._tree_digest(dedup_records),
        "partitions": dedup_records,
    }
    atomic_json(output / "deduplicated_edge_manifest.json", manifest)
    result = {
        "aggregate": dict(aggregate),
        "bucket_counts": {name: int(bucket_totals[index]) for index, name in enumerate(BUCKET_NAMES)},
        "cross_totals": cross_totals,
        "multiplicity_histogram": multiplicity_histogram,
        "lag_bins": lag_bins,
        "sensitivity": sensitivity,
        "deduplicated_manifest": manifest,
    }
    return checkpoints, [safe_root / f"safe-{index:04d}.bin" for index in range(PARTITION_COUNT)], result


def _safe_checkpoint_path(output: Path, partition: int) -> Path:
    return output / "checkpoints" / "safe" / f"part-{partition:04d}.json"


def process_safe_partitions(
    *, output: Path, safe_spools: Sequence[Path], task_fingerprint: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    total_rows = 0
    for partition in range(PARTITION_COUNT):
        if partition % 64 == 0:
            print(f"safe-edge materialization {partition}/{PARTITION_COUNT}", file=sys.stderr, flush=True)
        values = helper._read_records(safe_spools[partition], SAFE_DTYPE)
        if len(values):
            order = np.lexsort((values["source"], values["target"]))
            values = values[order]
            require(bool(np.all(values["target"] % PARTITION_COUNT == partition)), "safe target partition mismatch")
            require(bool(np.all(values["lag"] >= 0)), "negative lag in chronology-safe edge")
            duplicate = (values["target"][1:] == values["target"][:-1]) & (values["source"][1:] == values["source"][:-1])
            require(not bool(np.any(duplicate)), "duplicate chronology-safe pair")
        table = pa.Table.from_arrays(
            [
                pa.array(values["target"], type=pa.uint64()),
                pa.array(values["source"], type=pa.uint64()),
                pa.array(values["source_day"], type=pa.date32()),
                pa.array(values["target_day"], type=pa.date32()),
                pa.array(values["lag"], type=pa.int32()),
                pa.array(values["multiplicity"], type=pa.uint32()),
            ],
            schema=SAFE_SCHEMA,
        )
        path = output / "chronology_safe_edges_by_cited_work" / f"part-{partition:04d}.parquet"
        _write_parquet_atomic(path, table)
        record = {
            "partition_id": partition,
            "relative_path": path.relative_to(output).as_posix(),
            "rows": len(values),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        checkpoint = {
            "schema_version": SAFE_CHECKPOINT_SCHEMA,
            "task_fingerprint_sha256": task_fingerprint,
            "partition_id": partition,
            "rows": len(values),
            "spool_sha256": sha256_file(safe_spools[partition]),
            "artifact": record,
        }
        atomic_json(_safe_checkpoint_path(output, partition), checkpoint)
        checkpoints.append(checkpoint)
        records.append(record)
        total_rows += len(values)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "kind": "CHRONOLOGY_SAFE_PROXY_EDGES_BY_CITED_WORK_MODULO",
        "task_fingerprint_sha256": task_fingerprint,
        "partition_count": PARTITION_COUNT,
        "rows": total_rows,
        "partition_tree_sha256": helper._tree_digest(records),
        "partitions": records,
    }
    atomic_json(output / "chronology_safe_edge_manifest.json", manifest)
    return checkpoints, manifest


def _distinct_target_counts(
    target_seen: np.memmap, lookup: np.memmap, outside_targets: set[int]
) -> dict[str, int]:
    all_count = len(outside_targets)
    internal_count = 0
    external_count = len(outside_targets)
    chunk = 64 * 1024 * 1024
    for start in range(0, len(target_seen), chunk):
        stop = min(len(target_seen), start + chunk)
        selected = np.asarray(target_seen[start:stop]) != 0
        observed = int(np.count_nonzero(selected))
        if observed:
            parent_values = np.asarray(lookup[start:stop])[selected]
            internal = int(np.count_nonzero(parent_values != 0))
            all_count += observed
            internal_count += internal
            external_count += observed - internal
    return {
        "distinct_valid_target_ids": all_count,
        "distinct_in_snapshot_target_ids": internal_count,
        "distinct_external_target_ids": external_count,
    }


def _sum_scan(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for checkpoint in checkpoints:
        for key, value in _mapping(checkpoint["metrics"], "scan metrics").items():
            total[key] += _integer(value, f"scan metric {key}")
    return dict(total)


def _render_cross_tabs(raw: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    cross: dict[str, Any] = {}
    margins: dict[str, Any] = {}
    for axis, categories in AXIS_CATEGORIES.items():
        values = raw[axis]
        cross[axis] = {
            bucket: {
                category: int(values[bucket_index, category_index])
                for category_index, category in enumerate(categories)
            }
            for bucket_index, bucket in enumerate(BUCKET_NAMES)
        }
        margins[axis] = {
            category: int(values[:, category_index].sum(dtype=np.uint64))
            for category_index, category in enumerate(categories)
        }
    return cross, margins


def _write_metric_table(path: Path, rows: Mapping[str, int]) -> None:
    table = pa.table(
        {
            "metric": pa.array(list(rows), type=pa.string()),
            "value": pa.array([int(rows[key]) for key in rows], type=pa.uint64()),
        }
    )
    _write_parquet_atomic(path, table)


def _write_aggregate_tables(output: Path, result: Mapping[str, Any], target_counts: Mapping[str, int], scan: Mapping[str, int]) -> None:
    multiplicity = result["multiplicity_histogram"]
    _write_parquet_atomic(
        output / "duplicate_multiplicity_aggregates.parquet",
        pa.table(
            {
                "raw_pair_multiplicity": pa.array(sorted(multiplicity), type=pa.uint32()),
                "unique_pair_groups": pa.array([multiplicity[key] for key in sorted(multiplicity)], type=pa.uint64()),
            }
        ),
    )
    _write_parquet_atomic(
        output / "chronology_lag_aggregates.parquet",
        pa.table(
            {
                "lag_bin": pa.array(LAG_BIN_LABELS, type=pa.string()),
                "unique_pair_count": pa.array(result["lag_bins"], type=pa.uint64()),
            }
        ),
    )
    coverage = {
        "raw_in_snapshot_target_rows": scan["raw_in_snapshot_target_rows"],
        "raw_external_target_rows": scan["raw_external_target_rows"],
        "unique_in_snapshot_target_pairs": result["aggregate"]["unique_in_snapshot_target_pairs"],
        "unique_external_target_pairs": result["aggregate"]["unique_external_target_pairs"],
        **target_counts,
    }
    _write_metric_table(output / "open_world_target_coverage.parquet", coverage)
    _write_metric_table(output / "cutoff_and_maturation_coverage.parquet", result["sensitivity"])


def _build_plan(
    *, config_sha256: str, parent: ParentContext, selected: Sequence[helper.SourceTask], scope: str
) -> dict[str, Any]:
    tool_path = Path(__file__).resolve()
    base: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "scope": scope,
        "execution_abi": "CITATION_CANONICAL_SPINE_REUSE_PRODUCTION_V3",
        "config_relative_path": DEFAULT_CONFIG.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": config_sha256,
        "tool_relative_path": tool_path.relative_to(REPO_ROOT).as_posix(),
        "tool_sha256": sha256_file(tool_path),
        "runtime_fingerprint": helper.runtime_fingerprint(),
        "parents": {
            "content_receipt_sha256": sha256_file(parent.content_root / "content_freeze_audit.outputs.json"),
            "temporal_receipt_sha256": sha256_file(parent.temporal_root / "main_temporal_validity_audit.outputs.json"),
            "canonical_receipt_sha256": sha256_file(parent.canonical_root / "work_time_spine_audit.outputs.json"),
            "canonical_result_fingerprint_sha256": parent.canonical_receipt["result_fingerprint_sha256"],
            "canonical_spine_data_tree_sha256": parent.canonical_receipt["spine_data_tree_sha256"],
        },
        "partition_count": PARTITION_COUNT,
        "selected_reference_tasks": [
            {"relative_path": item.relative_path, "bytes": item.source_bytes, "source_stat": item.source_stat}
            for item in selected
        ],
        "full_scope": scope == FULL_SCOPE,
        "operator_paths_redacted": True,
    }
    base["task_fingerprint_sha256"] = sha256_bytes(canonical_json_bytes(base))
    return base


def _preflight(config: Mapping[str, Any], source_root: Path, output_parent: Path) -> dict[str, Any]:
    envelope = _mapping(config["resource_envelope"], "resource envelope")
    available_ram = canonical._available_memory_bytes()
    free_disk = shutil.disk_usage(output_parent).free
    nofile_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    source_device = os.stat(source_root).st_dev
    output_device = os.stat(output_parent).st_dev
    evidence = {
        "available_memory_bytes": available_ram,
        "required_available_memory_bytes": envelope["minimum_available_ram_bytes"],
        "free_disk_bytes": free_disk,
        "required_free_disk_bytes": envelope["minimum_free_disk_bytes"],
        "nofile_soft_limit": nofile_soft,
        "required_nofile_soft_limit": PARTITION_COUNT + 128,
        "source_device": source_device,
        "output_device": output_device,
        "distinct_devices": source_device != output_device,
    }
    require(available_ram >= envelope["minimum_available_ram_bytes"], "insufficient available RAM")
    require(free_disk >= envelope["minimum_free_disk_bytes"], "insufficient free disk")
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
        "checkpoints/scan",
        "checkpoints/pair",
        "checkpoints/safe",
        "deduplicated_edge_events",
        "chronology_safe_edges_by_cited_work",
    ):
        private_dir(output / relative)


def _artifact_paths(output: Path) -> list[Path]:
    excluded = {"audit.lock", "citation_edge_chronology_audit.outputs.json"}
    paths: list[Path] = []
    for path in output.rglob("*"):
        if not path.is_file() or path.name in excluded or ".citation_tmp" in path.parts:
            continue
        require(not path.is_symlink(), "symlink in output tree")
        paths.append(path)
    paths.sort(key=lambda item: item.relative_to(output).as_posix())
    return paths


def _report_fingerprint(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    provenance = dict(payload["provenance"])
    provenance.pop("result_fingerprint_sha256", None)
    payload["provenance"] = provenance
    return sha256_bytes(canonical_json_bytes(payload))


def _render_markdown(report: Mapping[str, Any]) -> str:
    accounting = report["pair_accounting"]
    buckets = report["exclusive_pair_buckets"]
    return f"""# OpenAlex Citation Edge Chronology v3

- Status: `{report['status']}`
- Scope: `{report['scope']}`
- GPU used: `false`
- Raw directed pair rows: {accounting['raw_pair_rows']:,}
- Exact distinct directed pairs: {accounting['unique_pair_rows']:,}
- Duplicate row excess: {accounting['duplicate_pair_row_excess']:,}
- Chronology-safe internal non-self pairs: {report['chronology_safe_primary_pairs']:,}
- External-target distinct pairs: {accounting['unique_external_target_pairs']:,}
- Negative-lag pairs: {buckets['negative_lag_target_after_source']:,}

`available_at` is the citing work publication date under a bibliographic-event
time convention. It is not an OpenAlex first-observation timestamp or proof of
historical researcher visibility. Maturation, cutoff panels, G-1, G0, empirical
G1, NABM fitting, and scientific effects remain outside this component.
"""


def _finalize(
    *,
    output: Path,
    plan: Mapping[str, Any],
    scope: str,
    preflight: Mapping[str, Any],
    scan_checkpoints: Sequence[Mapping[str, Any]],
    pair_checkpoints: Sequence[Mapping[str, Any]],
    safe_checkpoints: Sequence[Mapping[str, Any]],
    pair_result: Mapping[str, Any],
    safe_manifest: Mapping[str, Any],
    target_counts: Mapping[str, int],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    scan = _sum_scan(scan_checkpoints)
    accounting = dict(pair_result["aggregate"])
    accounting["sum_pair_multiplicity"] = accounting["raw_pair_rows"]
    accounting["duplicate_self_loop_row_excess"] = accounting["raw_self_loop_rows"] - accounting["unique_self_loop_pairs"]
    accounting["unique_pairs_without_available_at"] = accounting["unique_pair_rows"] - accounting["unique_pairs_with_available_at"]
    accounting.update(target_counts)
    buckets = dict(pair_result["bucket_counts"])
    full_scope = scope == FULL_SCOPE
    source_anchor = _mapping(_mapping(config["source"], "source")["referenced_works"], "reference anchors")
    if full_scope:
        anchor_map = {
            "raw_pair_rows": "raw_rows",
            "raw_in_snapshot_target_rows": "raw_in_snapshot_target_rows",
            "raw_external_target_rows": "raw_external_target_rows",
            "raw_self_loop_rows": "raw_self_loop_rows",
            "null_source_rows": "null_source_rows",
            "malformed_source_rows": "malformed_source_rows",
            "source_w_minus_1_rows": "source_w_minus_1_rows",
            "source_w_zero_rows": "source_w_zero_rows",
            "null_target_rows": "null_target_rows",
            "malformed_target_rows": "malformed_target_rows",
            "target_w_minus_1_rows": "target_w_minus_1_rows",
            "target_w_zero_rows": "target_w_zero_rows",
        }
        for observed_key, anchor_key in anchor_map.items():
            require(scan[observed_key] == source_anchor[anchor_key], f"full source anchor mismatch: {observed_key}")
        for key in ("distinct_valid_target_ids", "distinct_in_snapshot_target_ids", "distinct_external_target_ids"):
            require(target_counts[key] == source_anchor[key], f"full target anchor mismatch: {key}")
    require(accounting["raw_pair_rows"] == accounting["unique_pair_rows"] + accounting["duplicate_pair_row_excess"], "raw/unique duplicate closure failed")
    require(accounting["raw_pair_rows"] == scan["raw_pair_rows"], "scan/pair raw closure failed")
    require(accounting["raw_self_loop_rows"] == scan["raw_self_loop_rows"], "self-loop scan closure failed")
    require(accounting["unique_pair_rows"] == accounting["unique_in_snapshot_target_pairs"] + accounting["unique_external_target_pairs"], "unique target membership closure failed")
    require(accounting["unique_pair_rows"] == sum(buckets.values()), "seven-bucket conservation failed")
    require(accounting["unique_self_loop_pairs"] == buckets["self_loop"], "self-loop precedence closure failed")
    require(accounting["chronology_safe_primary_pairs"] == buckets["same_day"] + buckets["positive_lag"], "chronology-safe bucket closure failed")
    require(accounting["chronology_safe_primary_pairs"] == safe_manifest["rows"], "safe artifact row closure failed")
    sensitivity = {key: int(value) for key, value in pair_result["sensitivity"].items()}
    require(sensitivity["primary_zero_day"] == accounting["chronology_safe_primary_pairs"], "primary sensitivity closure failed")
    require(sensitivity["primary_zero_day"] <= sensitivity["tolerance_30_days"] <= sensitivity["tolerance_365_days"], "lag sensitivity monotonicity failed")
    cross_tabs, attribute_margins = _render_cross_tabs(pair_result["cross_totals"])
    for axis, values in pair_result["cross_totals"].items():
        require(int(values.sum(dtype=np.uint64)) == accounting["unique_pair_rows"], f"cross-tab total failed: {axis}")
        require(bool(np.array_equal(values.sum(axis=1, dtype=np.uint64), np.array([buckets[name] for name in BUCKET_NAMES], dtype=np.uint64))), f"cross-tab bucket margin failed: {axis}")

    status = FULL_STATUS if full_scope else PILOT_STATUS
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": status,
        "scope": scope,
        "production_component_pass": full_scope,
        "gpu_used": False,
        "source_accounting": scan,
        "pair_accounting": accounting,
        "availability_accounting": {
            "resolution": "BIBLIOGRAPHIC_EVENT_TIME_CONVENTION",
            "edge_available_at_field": "citing_work_publication_date",
            "raw_pair_rows_with_available_at": scan["raw_pair_rows_with_available_at"],
            "raw_pair_rows_without_available_at": scan["raw_pair_rows"] - scan["raw_pair_rows_with_available_at"],
            "unique_pairs_with_available_at": accounting["unique_pairs_with_available_at"],
            "unique_pairs_without_available_at": accounting["unique_pairs_without_available_at"],
            "openalex_observation_time_established": False,
            "historical_openalex_graph_established": False,
        },
        "exclusive_pair_buckets": buckets,
        "chronology_safe_primary_pairs": accounting["chronology_safe_primary_pairs"],
        "chronology_sensitivity_unique_pairs": sensitivity,
        "cross_tabs": cross_tabs,
        "attribute_margins": attribute_margins,
        "partition_products": {
            "deduplicated_edge_manifest": {
                "rows": pair_result["deduplicated_manifest"]["rows"],
                "partition_count": pair_result["deduplicated_manifest"]["partition_count"],
                "partition_tree_sha256": pair_result["deduplicated_manifest"]["partition_tree_sha256"],
            },
            "chronology_safe_edge_manifest": {
                "rows": safe_manifest["rows"],
                "partition_count": safe_manifest["partition_count"],
                "partition_tree_sha256": safe_manifest["partition_tree_sha256"],
            },
        },
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
    atomic_json(output / "citation_edge_chronology_audit.json", report)
    atomic_text(output / "citation_edge_chronology_audit.md", _render_markdown(report))
    atomic_json(
        output / "run_state.json",
        {
            "schema_version": STATE_SCHEMA,
            "status": "COMPLETE_PENDING_RECEIPT",
            "scope": scope,
            "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
            "result_fingerprint_sha256": report["provenance"]["result_fingerprint_sha256"],
            "production_component_pass": full_scope,
        },
    )
    artifacts = [
        canonical._artifact_contract(output, path.relative_to(output).as_posix())
        for path in _artifact_paths(output)
    ]
    receipt = {
        "schema_version": OUTPUTS_SCHEMA,
        "status": status,
        "scope": scope,
        "production_component_pass": full_scope,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "result_fingerprint_sha256": report["provenance"]["result_fingerprint_sha256"],
        "canonical_result_fingerprint_sha256": plan["parents"]["canonical_result_fingerprint_sha256"],
        "canonical_spine_data_tree_sha256": plan["parents"]["canonical_spine_data_tree_sha256"],
        "artifact_count": len(artifacts),
        "artifact_tree_sha256": helper._tree_digest(artifacts),
        "artifacts": artifacts,
        "completion_contract": COMPLETION_CONTRACT,
        "operator_paths_redacted": True,
    }
    atomic_json(output / "citation_edge_chronology_audit.outputs.json", receipt)
    helper._validate_receipt_artifacts(output, receipt, "citation output")
    return report


def _terminal_reentry(output: Path) -> dict[str, Any]:
    receipt = helper.read_json(output / "citation_edge_chronology_audit.outputs.json", "citation output receipt")
    require(receipt.get("schema_version") == OUTPUTS_SCHEMA, "terminal receipt schema mismatch")
    require(receipt.get("completion_contract") == COMPLETION_CONTRACT, "terminal completion contract mismatch")
    helper._validate_receipt_artifacts(output, receipt, "citation output")
    artifacts = _list(receipt["artifacts"], "terminal artifacts")
    require(receipt["artifact_count"] == len(artifacts), "terminal artifact count mismatch")
    require(receipt["artifact_tree_sha256"] == helper._tree_digest(artifacts), "terminal artifact tree mismatch")
    report = helper.read_json(output / "citation_edge_chronology_audit.json", "citation report")
    require(_report_fingerprint(report) == receipt["result_fingerprint_sha256"], "terminal report fingerprint invalid")
    for manifest_name, schema in (
        ("deduplicated_edge_manifest.json", DEDUP_SCHEMA),
        ("chronology_safe_edge_manifest.json", SAFE_SCHEMA),
    ):
        manifest = helper.read_json(output / manifest_name, manifest_name)
        require(manifest.get("partition_count") == PARTITION_COUNT, "terminal manifest partition count mismatch")
        for item in _list(manifest.get("partitions"), "manifest partitions"):
            path = output / _safe_relative(item["relative_path"], "manifest data path")
            parquet = pq.ParquetFile(path)
            require(parquet.schema_arrow == schema, f"terminal parquet schema mismatch: {path.name}")
            require(parquet.metadata.num_rows == item["rows"], f"terminal parquet footer mismatch: {path.name}")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_sha256 = load_config(args.config)
    source_root = args.source_root.resolve(strict=True)
    output = args.output_dir.absolute()
    output_parent = output.parent.resolve(strict=True)
    require(args.execute, "execution requires --execute")
    require(not str(output).startswith(str(source_root) + os.sep), "output may not be inside source root")
    full_scope = args.max_reference_files is None
    scope = FULL_SCOPE if full_scope else PILOT_SCOPE
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
    selected = parent.reference_tasks if full_scope else parent.reference_tasks[: args.max_reference_files]
    require(selected, "selected reference task set is empty")
    preflight = _preflight(config, source_root, output_parent)
    plan = _build_plan(config_sha256=config_sha256, parent=parent, selected=selected, scope=scope)
    if output.exists() and (output / "citation_edge_chronology_audit.outputs.json").is_file():
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
            canonical_cfg = _mapping(_mapping(config["parents"], "parents")["canonical_work_time_spine"], "canonical parent")
            print("building receipt-bound canonical direct lookup", file=sys.stderr, flush=True)
            lookup, lookup_path = build_canonical_lookup(
                output,
                parent.canonical_root,
                maximum_numeric_id=canonical_cfg["maximum_numeric_id"],
                expected_rows=canonical_cfg["retained_rows"],
            )
            target_seen_path = output / ".citation_tmp" / "target_seen.u8"
            target_seen = _new_sparse_memmap(target_seen_path, "u1", len(lookup))
            outside_targets: set[int] = set()
            raw_root = output / ".citation_tmp" / "raw_by_source"
            scan_checkpoints: list[dict[str, Any]] = []
            with helper.SpoolWriter(raw_root, PAIR_DTYPE, prefix="pairs") as writer:
                for index, task in enumerate(selected, start=1):
                    print(f"reference scan {index}/{len(selected)} {task.relative_path}", file=sys.stderr, flush=True)
                    scan_checkpoints.append(
                        _scan_reference_task(
                            source_root=source_root,
                            output=output,
                            task=task,
                            task_fingerprint=plan["task_fingerprint_sha256"],
                            lookup=lookup,
                            target_seen=target_seen,
                            outside_targets=outside_targets,
                            writer=writer,
                        )
                    )
            raw_spools = [raw_root / f"pairs-{index:04d}.bin" for index in range(PARTITION_COUNT)]
            print("reducing exact directed pairs", file=sys.stderr, flush=True)
            pair_checkpoints, safe_spools, pair_result = process_pair_partitions(
                output=output,
                raw_spools=raw_spools,
                lookup=lookup,
                task_fingerprint=plan["task_fingerprint_sha256"],
            )
            shutil.rmtree(raw_root)
            print("materializing chronology-safe target partitions", file=sys.stderr, flush=True)
            safe_checkpoints, safe_manifest = process_safe_partitions(
                output=output,
                safe_spools=safe_spools,
                task_fingerprint=plan["task_fingerprint_sha256"],
            )
            shutil.rmtree(safe_spools[0].parent)
            print("closing distinct-target accounting", file=sys.stderr, flush=True)
            target_counts = _distinct_target_counts(target_seen, lookup, outside_targets)
            scan_metrics = _sum_scan(scan_checkpoints)
            _write_aggregate_tables(output, pair_result, target_counts, scan_metrics)
            del target_seen
            del lookup
            target_seen_path.unlink()
            lookup_path.unlink()
            with contextlib.suppress(OSError):
                (output / ".citation_tmp").rmdir()
            canonical._fsync_directory(output)
            return _finalize(
                output=output,
                plan=plan,
                scope=scope,
                preflight=preflight,
                scan_checkpoints=scan_checkpoints,
                pair_checkpoints=pair_checkpoints,
                safe_checkpoints=safe_checkpoints,
                pair_result=pair_result,
                safe_manifest=safe_manifest,
                target_counts=target_counts,
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
    parser.add_argument("--max-reference-files", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-full", action="store_true")
    parser.add_argument("--acknowledge-exclusive-source-scan", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_reference_files is not None and args.max_reference_files <= 0:
        print("error: max_reference_files must be positive", file=sys.stderr)
        return 2
    try:
        report = run(args)
    except (CitationProductionError, helper.AuthorHistoryError, canonical.SpineAuditError, OSError, pa.ArrowException) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
