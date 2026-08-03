#!/usr/bin/env python3
"""Independently replay cutoff metadata from source and compare every work row.

This verifier is additive to ``verify_pivot_feedback_cutoff_primitives.py``.
It imports no cutoff producer code.  The existing v1 verifier proves output
integrity and canonical-time closure; this v2 verifier fills the remaining
source-projection gap by independently parsing the eight source columns,
partitioning them by work ID, and comparing every retained value and null mask
to the published work-metadata shards.

The production output is strictly read-only.  Temporary fixed-width spools live
under an explicitly supplied scratch root and are deleted after a successful
run.  The verification report must be outside both the production and scratch
trees.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import resource
import shutil
import stat
import sys
import time
from collections import Counter
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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/pivot_feedback_cutoff_primitives_v0.yaml"

PARTITION_COUNT = 4096
BATCH_ROWS = 262_144
OUTPUTS_SCHEMA = "pivot_feedback_cutoff_primitives_outputs.v1"
OUTPUT_REPORT_SCHEMA = "pivot_feedback_cutoff_primitives_report.v1"
OUTPUT_MANIFEST_SCHEMA = "pivot_feedback_cutoff_primitives_manifest.v1"
CONTENT_RECEIPT_SCHEMA = "openalex_g1_content_freeze_outputs.v1"
CONTENT_PLAN_SCHEMA = "openalex_g1_content_freeze_plan.v1"
CONTENT_CHECKPOINT_SCHEMA = "openalex_g1_content_freeze_checkpoint.v1"
CANONICAL_RECEIPT_SCHEMA = "openalex_g1_canonical_work_time_spine_production_outputs.v1"
VERIFICATION_SCHEMA = "pivot_feedback_cutoff_primitives_source_replay_verification.v2"

FULL_SCOPE = "FULL_CURRENT_SNAPSHOT_SOURCE_REPLAY"
PILOT_SCOPE = "SOURCE_PREFIX_REPLAY_PILOT_PERMANENTLY_NONPROMOTABLE"
FULL_STATUS = (
    "PASS_INDEPENDENT_FULL_SOURCE_PROJECTION_REPLAY_AND_CANONICAL_TIME_"
    "VERIFICATION_WITH_CURRENT_SNAPSHOT_PROXY_LIMITATIONS"
)
PILOT_STATUS = "PASS_SOURCE_PROJECTION_REPLAY_PILOT_PERMANENTLY_NONPROMOTABLE"
BLOCKED_STATUS = "BLOCKED_CUTOFF_PRIMITIVES_SOURCE_REPLAY_VERIFICATION"

W_MINUS_ONE = "https://openalex.org/W-1"
WORK_PATTERN = r"^https://openalex[.]org/W(0|[1-9][0-9]*)$"
TOPIC_PATTERN = r"^https://openalex[.]org/T(0|[1-9][0-9]*)$"
FIELD_PATTERN = r"^https://openalex[.]org/fields/(0|[1-9][0-9]*)$"
DOMAIN_PATTERN = r"^https://openalex[.]org/domains/(0|[1-9][0-9]*)$"

SOURCE_RECORD_DTYPE = np.dtype(
    [
        ("work", "<u8"),
        ("type", "u1"),
        ("paratext", "u1"),
        ("retracted", "u1"),  # 0 false, 1 true, 2 null
        ("authors", "<u2"),
        ("topic", "<u4"),  # zero is the null sentinel
        ("field", "u1"),
        ("domain", "u1"),
    ],
    align=False,
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

CANONICAL_SCHEMA = pa.schema(
    [
        pa.field("work_id_numeric", pa.uint64(), nullable=False),
        pa.field("publication_day", pa.int32(), nullable=True),
        pa.field("publication_year", pa.uint16(), nullable=True),
        pa.field("temporal_disposition", pa.uint8(), nullable=False),
    ]
)


class ReplayVerificationError(RuntimeError):
    """Controlled fail-closed verification error."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Strict YAML loader for the repository contract."""


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
        raise ReplayVerificationError(message)


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        while payload := handle.read(chunk_bytes):
            digest.update(payload)
    return digest.hexdigest()


def _strict_json(path: Path, field: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReplayVerificationError(f"duplicate JSON key in {field}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=unique)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReplayVerificationError(f"invalid {field} JSON") from exc
    require(isinstance(value, dict), f"{field} must be an object")
    return dict(value)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field} must be an object")
    return dict(value)


def _list(value: Any, field: str) -> list[Any]:
    require(isinstance(value, list), f"{field} must be an array")
    return list(value)


def _safe_relative(value: Any, field: str) -> Path:
    require(isinstance(value, str) and value, f"{field} must be a nonempty path")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe {field}")
    require(path.as_posix() == value, f"noncanonical {field}")
    return path


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.resolve(strict=True).read_bytes()
    try:
        value = yaml.load(raw, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ReplayVerificationError("invalid verifier config YAML") from exc
    require(isinstance(value, dict), "config must be an object")
    config = dict(value)
    require(
        config.get("schema_version") == "pivot_feedback_cutoff_primitives_config.v1",
        "config schema mismatch",
    )
    projected = _list(_mapping(config.get("source"), "source").get("projected_columns"), "projected columns")
    require(
        projected
        == [
            "id",
            "type",
            "is_paratext",
            "is_retracted",
            "authors_count",
            "primary_topic__id",
            "primary_topic__domain__id",
            "primary_topic__field__id",
        ],
        "source projection differs from the frozen eight-column ABI",
    )
    return config, _sha256_bytes(raw)


def _tree_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(records)))


def _tree_identity(root: Path) -> dict[str, tuple[int, int, int, int, int, int]]:
    identity: dict[str, tuple[int, int, int, int, int, int]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), f"symlink in read-only output: {relative}")
        identity[relative] = (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
    return identity


def _rich_file_identity(path: Path) -> dict[str, int]:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), f"source is not a regular file: {path.name}")
    require(not path.is_symlink(), f"source is a symlink: {path.name}")
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": stat.S_IMODE(info.st_mode),
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class SourceTask:
    index: int
    relative_path: str
    source_bytes: int
    source_stat: dict[str, int]
    content_sha256: str
    checkpoint_sha256: str


def _validate_content_checkpoint(
    content_root: Path,
    raw_task: Mapping[str, Any],
    *,
    task_index: int,
    plan_sha256: str,
    task_fingerprint: str,
) -> SourceTask:
    relative = _safe_relative(raw_task.get("relative_path"), "content task path").as_posix()
    path_digest = _sha256_bytes(relative.encode("utf-8"))
    checkpoint_path = content_root / "checkpoints" / path_digest[:2] / f"{path_digest}.json"
    require(checkpoint_path.is_file() and not checkpoint_path.is_symlink(), "missing content checkpoint")
    checkpoint = _strict_json(checkpoint_path, "content checkpoint")
    require(checkpoint.get("schema_version") == CONTENT_CHECKPOINT_SCHEMA, "content checkpoint schema mismatch")
    require(checkpoint.get("relative_path") == relative, "content checkpoint path mismatch")
    require(checkpoint.get("table_name") == raw_task.get("table_name"), "content checkpoint table mismatch")
    require(checkpoint.get("task_index") == task_index, "content checkpoint task index mismatch")
    require(checkpoint.get("plan_sha256") == plan_sha256, "content checkpoint plan mismatch")
    require(
        checkpoint.get("task_fingerprint_sha256") == task_fingerprint,
        "content checkpoint task fingerprint mismatch",
    )
    checkpoint_digest = checkpoint.get("checkpoint_sha256")
    require(
        isinstance(checkpoint_digest, str) and len(checkpoint_digest) == 64,
        "invalid content checkpoint digest",
    )
    digest_payload = dict(checkpoint)
    digest_payload.pop("checkpoint_sha256")
    require(
        _sha256_bytes(_canonical_json_bytes(digest_payload)) == checkpoint_digest,
        "content checkpoint self-digest mismatch",
    )
    source_stat = _mapping(raw_task.get("source_stat"), "content task source stat")
    require(
        checkpoint.get("source_identity_before") == source_stat
        and checkpoint.get("source_identity_after") == source_stat,
        "content checkpoint source identity mismatch",
    )
    require(
        checkpoint.get("bytes_hashed") == raw_task.get("bytes") == source_stat.get("bytes"),
        "content checkpoint byte count mismatch",
    )
    content_sha = checkpoint.get("content_sha256")
    require(isinstance(content_sha, str) and len(content_sha) == 64, "invalid source content SHA")
    return SourceTask(
        index=task_index,
        relative_path=relative,
        source_bytes=int(raw_task["bytes"]),
        source_stat={key: int(value) for key, value in source_stat.items()},
        content_sha256=content_sha,
        checkpoint_sha256=checkpoint_digest,
    )


def _load_source_tasks(
    config: Mapping[str, Any], content_root: Path
) -> tuple[list[SourceTask], dict[str, Any], str]:
    parents = _mapping(config.get("parents"), "parents")
    expected = _mapping(parents.get("current_content_baseline"), "content parent")
    receipt_path = content_root / "content_freeze_audit.outputs.json"
    require(receipt_path.is_file() and not receipt_path.is_symlink(), "missing content receipt")
    receipt_sha = _sha256_file(receipt_path)
    require(receipt_sha == expected.get("receipt_sha256"), "content receipt SHA mismatch")
    receipt = _strict_json(receipt_path, "content receipt")
    require(receipt.get("schema_version") == CONTENT_RECEIPT_SCHEMA, "content receipt schema mismatch")
    require(receipt.get("status") == expected.get("required_status"), "content receipt status mismatch")
    require(receipt.get("plan_sha256") == expected.get("plan_sha256"), "content receipt plan mismatch")
    require(
        receipt.get("result_fingerprint_sha256") == expected.get("result_fingerprint_sha256"),
        "content receipt result mismatch",
    )
    artifact_map = {
        _safe_relative(item.get("relative_path"), "content artifact path").as_posix(): item
        for item in (
            _mapping(raw, "content artifact")
            for raw in _list(receipt.get("artifacts"), "content artifacts")
        )
    }
    plan_path = content_root / "audit_plan.json"
    plan_artifact = _mapping(artifact_map.get("audit_plan.json"), "content plan artifact")
    require(plan_path.stat().st_size == plan_artifact.get("bytes"), "content plan bytes mismatch")
    require(_sha256_file(plan_path) == plan_artifact.get("sha256"), "content plan file SHA mismatch")
    plan = _strict_json(plan_path, "content plan")
    require(plan.get("schema_version") == CONTENT_PLAN_SCHEMA, "content plan schema mismatch")
    require(plan.get("plan_sha256") == expected.get("plan_sha256"), "content plan payload mismatch")
    require(
        plan.get("task_fingerprint_sha256") == expected.get("task_fingerprint_sha256"),
        "content task fingerprint mismatch",
    )
    source_cfg = _mapping(config.get("source"), "source")
    source_descriptor = _mapping(plan.get("source"), "content source descriptor")
    require(source_descriptor.get("root_id") == source_cfg.get("root_id"), "content source root ID mismatch")
    table_name = source_cfg.get("table")
    raw_tasks = _list(plan.get("tasks"), "content tasks")
    tasks: list[SourceTask] = []
    for index, raw in enumerate(raw_tasks):
        item = _mapping(raw, f"content task {index}")
        if item.get("table_name") != table_name:
            continue
        tasks.append(
            _validate_content_checkpoint(
                content_root,
                item,
                task_index=index,
                plan_sha256=str(plan["plan_sha256"]),
                task_fingerprint=str(plan["task_fingerprint_sha256"]),
            )
        )
    tasks.sort(key=lambda item: item.relative_path)
    require(len(tasks) == source_cfg.get("files"), "main source task count mismatch")
    require(sum(item.source_bytes for item in tasks) == source_cfg.get("source_bytes"), "main source task bytes mismatch")
    return tasks, receipt, receipt_sha


def _verify_output_artifacts(
    output_root: Path, config_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
    receipt_path = output_root / "pivot_feedback_cutoff_panel.outputs.json"
    receipt_sha = _sha256_file(receipt_path)
    receipt = _strict_json(receipt_path, "cutoff output receipt")
    require(receipt.get("schema_version") == OUTPUTS_SCHEMA, "cutoff receipt schema mismatch")
    require(receipt.get("scope") == "FULL_CURRENT_SNAPSHOT_PRODUCTION", "cutoff output is not full production")
    require(receipt.get("production_component_pass") is True, "cutoff production component did not pass")
    raw_artifacts = _list(receipt.get("artifacts"), "cutoff artifacts")
    artifacts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_artifacts):
        item = _mapping(raw, f"cutoff artifact {index}")
        relative = _safe_relative(item.get("relative_path"), "cutoff artifact path").as_posix()
        require(relative not in seen, "duplicate cutoff artifact")
        seen.add(relative)
        path = output_root / relative
        require(path.is_file() and not path.is_symlink(), f"missing cutoff artifact {relative}")
        require(path.stat().st_size == item.get("bytes"), f"cutoff artifact bytes mismatch: {relative}")
        require(_sha256_file(path) == item.get("sha256"), f"cutoff artifact SHA mismatch: {relative}")
        artifacts.append(item)
    published = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    require(
        published == seen | {"pivot_feedback_cutoff_panel.outputs.json"},
        "cutoff output file set differs from receipt",
    )
    require(len(artifacts) == receipt.get("artifact_count"), "cutoff artifact count mismatch")
    require(_tree_digest(artifacts) == receipt.get("artifact_tree_sha256"), "cutoff artifact tree mismatch")
    report = _strict_json(output_root / "pivot_feedback_cutoff_panel.json", "cutoff report")
    require(report.get("schema_version") == OUTPUT_REPORT_SCHEMA, "cutoff report schema mismatch")
    require(report.get("status") == receipt.get("status"), "cutoff report status mismatch")
    require(
        _mapping(report.get("provenance"), "cutoff provenance").get("config_sha256")
        == config_sha256,
        "cutoff report config binding mismatch",
    )
    require(
        report.get("result_fingerprint_sha256") == receipt.get("result_fingerprint_sha256"),
        "cutoff report result mismatch",
    )
    manifest = _strict_json(output_root / "work_metadata_manifest.json", "work manifest")
    require(manifest.get("schema_version") == OUTPUT_MANIFEST_SCHEMA, "work manifest schema mismatch")
    require(manifest.get("partition_count") == PARTITION_COUNT, "work manifest partition count mismatch")
    parts = _list(manifest.get("partitions"), "work manifest partitions")
    require(len(parts) == PARTITION_COUNT, "work manifest partition list mismatch")
    require(
        manifest.get("work_metadata_tree_sha256") == receipt.get("work_metadata_tree_sha256"),
        "work manifest tree binding mismatch",
    )
    return receipt, manifest, artifacts, receipt_sha


def _load_canonical_contract(
    config: Mapping[str, Any], canonical_root: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    expected = _mapping(
        _mapping(config.get("parents"), "parents").get("canonical_work_time_spine"),
        "canonical parent",
    )
    receipt_path = canonical_root / "work_time_spine_audit.outputs.json"
    receipt_sha = _sha256_file(receipt_path)
    require(receipt_sha == expected.get("receipt_sha256"), "canonical receipt SHA mismatch")
    receipt = _strict_json(receipt_path, "canonical receipt")
    require(receipt.get("schema_version") == CANONICAL_RECEIPT_SCHEMA, "canonical receipt schema mismatch")
    require(receipt.get("status") == expected.get("required_status"), "canonical receipt status mismatch")
    require(receipt.get("production_component_pass") is True, "canonical component is not production")
    require(
        receipt.get("spine_data_tree_sha256") == expected.get("spine_data_tree_sha256"),
        "canonical tree binding mismatch",
    )
    artifacts: dict[str, dict[str, Any]] = {}
    raw_artifacts = _mapping(receipt.get("artifacts"), "canonical artifacts")
    require(
        len(raw_artifacts) == receipt.get("artifact_count") == expected.get("artifact_count"),
        "canonical artifact count mismatch",
    )
    for declared_relative, raw in raw_artifacts.items():
        item = _mapping(raw, "canonical artifact")
        relative = _safe_relative(item.get("relative_path"), "canonical artifact path").as_posix()
        require(relative == declared_relative, "canonical artifact key/path mismatch")
        require(relative not in artifacts, "duplicate canonical artifact")
        artifacts[relative] = item
    return receipt, artifacts, receipt_sha


class PartitionSpoolWriter:
    """Independent fixed-record demultiplexer with one handle per partition."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.paths = [root / f"source-{index:04d}.bin" for index in range(PARTITION_COUNT)]
        self.handles: list[BinaryIO] = []

    def __enter__(self) -> PartitionSpoolWriter:
        self.root.mkdir(mode=0o700, parents=False, exist_ok=False)
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        require(soft_limit >= PARTITION_COUNT + 64, "open-file limit is too small for 4096-way spool")
        for path in self.paths:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            self.handles.append(os.fdopen(descriptor, "wb", buffering=65_536))
        return self

    def write(self, values: np.ndarray[Any, Any]) -> None:
        require(values.dtype == SOURCE_RECORD_DTYPE, "source record dtype mismatch")
        if not len(values):
            return
        partitions = (values["work"] % PARTITION_COUNT).astype("<u2", copy=False)
        order = np.argsort(partitions, kind="stable")
        sorted_partitions = partitions[order]
        sorted_values = values[order]
        starts = np.flatnonzero(
            np.concatenate(
                (np.array([True]), sorted_partitions[1:] != sorted_partitions[:-1])
            )
        )
        ends = np.concatenate((starts[1:], np.array([len(sorted_values)])))
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
            partition = int(sorted_partitions[start])
            self.handles[partition].write(sorted_values[start:end].tobytes(order="C"))

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        first_error: BaseException | None = None
        for handle in self.handles:
            try:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            except BaseException as error:  # pragma: no cover - catastrophic I/O
                first_error = first_error or error
        if exc is None and first_error is not None:
            raise first_error


def _bool_values(array: pa.Array | pa.ChunkedArray) -> np.ndarray[Any, Any]:
    return (
        pc.fill_null(array, False)
        .to_numpy(zero_copy_only=False)
        .astype(bool, copy=False)
    )


def _parse_work_ids(values: pa.Array, retained: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    if not len(retained):
        return np.empty(0, dtype="<u8")
    selected = values.take(pa.array(retained, type=pa.int64()))
    suffix = pc.utf8_slice_codeunits(selected, start=len("https://openalex.org/W"))
    try:
        parsed = pc.cast(suffix, pa.uint64(), safe=True).to_numpy(zero_copy_only=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise ReplayVerificationError("work ID does not fit uint64") from exc
    return np.asarray(parsed, dtype="<u8")


def _nullable_source_id(
    values: pa.Array,
    *,
    pattern: str,
    prefix_length: int,
    dtype: str,
    field: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    nulls = pc.is_null(values).to_numpy(zero_copy_only=False).astype(bool, copy=False)
    result = np.zeros(len(values), dtype=dtype)
    indices = np.flatnonzero(~nulls)
    if not len(indices):
        return result, nulls
    selected = values.take(pa.array(indices, type=pa.int64()))
    valid = pc.match_substring_regex(selected, pattern=pattern).to_numpy(
        zero_copy_only=False
    )
    require(bool(np.all(valid)), f"malformed {field}")
    suffix = pc.utf8_slice_codeunits(selected, start=prefix_length)
    arrow_type = {
        "<u4": pa.uint32(),
        "u1": pa.uint8(),
    }[dtype]
    try:
        parsed = pc.cast(suffix, arrow_type, safe=True).to_numpy(zero_copy_only=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise ReplayVerificationError(f"{field} does not fit target type") from exc
    parsed_array = np.asarray(parsed, dtype=dtype)
    require(bool(np.all(parsed_array != 0)), f"{field} zero collides with null sentinel")
    result[indices] = parsed_array
    return result, nulls


def _projected_schema(parquet: pq.ParquetFile, columns: Sequence[str]) -> dict[str, str]:
    schema = parquet.schema_arrow
    result: dict[str, str] = {}
    for column in columns:
        require(column in schema.names, f"source column is missing: {column}")
        result[column] = str(schema.field(column).type)
    return result


def _parse_source_batch(
    batch: pa.RecordBatch,
    *,
    projected_columns: Sequence[str],
    type_code_by_name: Mapping[str, int],
) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    require(batch.schema.names == list(projected_columns), "source batch column order mismatch")
    ids = batch.column(0)
    types = batch.column(1)
    paratext = batch.column(2)
    retracted = batch.column(3)
    authors = batch.column(4)
    topics = batch.column(5)
    domains = batch.column(6)
    fields = batch.column(7)

    require(ids.null_count == 0, "null work ID")
    minus = _bool_values(pc.equal(ids, W_MINUS_ONE))
    valid_work = _bool_values(pc.match_substring_regex(ids, pattern=WORK_PATTERN))
    require(bool(np.all(valid_work | minus)), "malformed work ID")
    retained = np.flatnonzero(~minus)
    work = _parse_work_ids(ids, retained)

    require(types.null_count == 0, "null work type")
    alphabetical_names = sorted(type_code_by_name)
    indices = pc.index_in(
        types, value_set=pa.array(alphabetical_names, type=pa.string())
    )
    require(indices.null_count == 0, "unknown work type")
    type_indices = np.asarray(indices.to_numpy(zero_copy_only=False), dtype=np.int32)
    require(bool(np.all(type_indices >= 0)), "unknown work type")
    alphabetical_codes = np.array(
        [type_code_by_name[name] for name in alphabetical_names], dtype="u1"
    )
    type_codes = alphabetical_codes[type_indices]

    require(paratext.null_count == 0, "null is_paratext")
    paratext_values = np.asarray(
        paratext.to_numpy(zero_copy_only=False), dtype=bool
    ).astype("u1", copy=False)

    retract_null = pc.is_null(retracted).to_numpy(zero_copy_only=False).astype(
        bool, copy=False
    )
    retract_values = _bool_values(retracted).astype("u1", copy=False)
    retract_values[retract_null] = 2

    require(authors.null_count == 0, "null authors_count")
    author_values = np.asarray(authors.to_numpy(zero_copy_only=False), dtype=np.int64)
    require(
        bool(
            np.all(
                (author_values >= 0)
                & (author_values <= np.iinfo(np.uint16).max)
            )
        ),
        "authors_count outside uint16",
    )

    topic_values, topic_null = _nullable_source_id(
        topics,
        pattern=TOPIC_PATTERN,
        prefix_length=len("https://openalex.org/T"),
        dtype="<u4",
        field="primary topic ID",
    )
    domain_values, domain_null = _nullable_source_id(
        domains,
        pattern=DOMAIN_PATTERN,
        prefix_length=len("https://openalex.org/domains/"),
        dtype="u1",
        field="primary domain ID",
    )
    field_values, field_null = _nullable_source_id(
        fields,
        pattern=FIELD_PATTERN,
        prefix_length=len("https://openalex.org/fields/"),
        dtype="u1",
        field="primary field ID",
    )
    require(np.array_equal(topic_null, domain_null), "source topic/domain null-mask mismatch")
    require(np.array_equal(topic_null, field_null), "source topic/field null-mask mismatch")

    records = np.empty(len(retained), dtype=SOURCE_RECORD_DTYPE)
    records["work"] = work
    records["type"] = type_codes[retained]
    records["paratext"] = paratext_values[retained]
    records["retracted"] = retract_values[retained]
    records["authors"] = author_values[retained].astype("<u2", copy=False)
    records["topic"] = topic_values[retained]
    records["field"] = field_values[retained]
    records["domain"] = domain_values[retained]

    metrics = {
        "raw_rows": len(ids),
        "quarantined_w_minus_1_rows": int(np.count_nonzero(minus)),
        "retained_rows": len(retained),
        "null_counts": {
            "type": types.null_count,
            "is_paratext": paratext.null_count,
            "is_retracted": retracted.null_count,
            "authors_count": authors.null_count,
            "primary_topic__id": topics.null_count,
            "primary_topic__domain__id": domains.null_count,
            "primary_topic__field__id": fields.null_count,
        },
        "work_type_codes": Counter(type_codes.tolist()),
        "authors_count_minimum": int(author_values.min()) if len(author_values) else None,
        "authors_count_maximum": int(author_values.max()) if len(author_values) else None,
    }
    return records, metrics


def _new_scan_metrics() -> dict[str, Any]:
    return {
        "files": 0,
        "source_bytes": 0,
        "raw_rows": 0,
        "quarantined_w_minus_1_rows": 0,
        "retained_rows": 0,
        "null_counts": {
            "type": 0,
            "is_paratext": 0,
            "is_retracted": 0,
            "authors_count": 0,
            "primary_topic__id": 0,
            "primary_topic__domain__id": 0,
            "primary_topic__field__id": 0,
        },
        "work_type_codes": Counter(),
        "authors_count_minimum": None,
        "authors_count_maximum": None,
        "source_content_files_rehashed": 0,
        "source_content_bytes_rehashed": 0,
    }


def _merge_scan_metrics(total: dict[str, Any], item: Mapping[str, Any]) -> None:
    for key in ("raw_rows", "quarantined_w_minus_1_rows", "retained_rows"):
        total[key] += int(item[key])
    for key, value in _mapping(item["null_counts"], "batch null counts").items():
        total["null_counts"][key] += int(value)
    total["work_type_codes"].update(item["work_type_codes"])
    local_min = item["authors_count_minimum"]
    local_max = item["authors_count_maximum"]
    if local_min is not None:
        total["authors_count_minimum"] = (
            int(local_min)
            if total["authors_count_minimum"] is None
            else min(total["authors_count_minimum"], int(local_min))
        )
        total["authors_count_maximum"] = (
            int(local_max)
            if total["authors_count_maximum"] is None
            else max(total["authors_count_maximum"], int(local_max))
        )


def _scan_sources(
    config: Mapping[str, Any],
    *,
    source_root: Path,
    tasks: Sequence[SourceTask],
    spool_root: Path,
    rehash_source_content: bool,
) -> dict[str, Any]:
    source_cfg = _mapping(config.get("source"), "source")
    projected_columns = [
        str(value)
        for value in _list(source_cfg.get("projected_columns"), "projected columns")
    ]
    expected_schema = {
        "id": "string",
        "type": "string",
        "is_paratext": "bool",
        "is_retracted": "bool",
        "authors_count": "int64",
        "primary_topic__id": "string",
        "primary_topic__domain__id": "string",
        "primary_topic__field__id": "string",
    }
    dictionary = _mapping(
        _mapping(config.get("input_profile"), "input profile").get(
            "work_type_dictionary"
        ),
        "work type dictionary",
    )
    type_code_by_name = {str(name): int(code) for code, name in dictionary.items()}
    metrics = _new_scan_metrics()
    with PartitionSpoolWriter(spool_root) as writer:
        for task_number, task in enumerate(tasks, start=1):
            if task_number == 1 or task_number % 16 == 0 or task_number == len(tasks):
                print(
                    f"source replay scan {task_number}/{len(tasks)}",
                    file=sys.stderr,
                    flush=True,
                )
            relative = _safe_relative(task.relative_path, "source task path")
            path = source_root / relative
            require(path.resolve(strict=True).is_relative_to(source_root), "source path escapes root")
            before = _rich_file_identity(path)
            require(before == task.source_stat, f"source identity changed: {task.relative_path}")
            if rehash_source_content:
                require(
                    _sha256_file(path) == task.content_sha256,
                    f"source content SHA changed: {task.relative_path}",
                )
                metrics["source_content_files_rehashed"] += 1
                metrics["source_content_bytes_rehashed"] += task.source_bytes
            parquet = pq.ParquetFile(path)
            require(
                _projected_schema(parquet, projected_columns) == expected_schema,
                f"projected source schema mismatch: {task.relative_path}",
            )
            file_rows = 0
            for batch in parquet.iter_batches(
                batch_size=BATCH_ROWS, columns=projected_columns
            ):
                records, batch_metrics = _parse_source_batch(
                    batch,
                    projected_columns=projected_columns,
                    type_code_by_name=type_code_by_name,
                )
                writer.write(records)
                _merge_scan_metrics(metrics, batch_metrics)
                file_rows += len(batch)
            require(
                file_rows == parquet.metadata.num_rows,
                f"source footer row mismatch: {task.relative_path}",
            )
            after = _rich_file_identity(path)
            require(after == before, f"source identity changed during scan: {task.relative_path}")
            metrics["files"] += 1
            metrics["source_bytes"] += task.source_bytes
    metrics["work_type_codes"] = dict(
        sorted((int(code), int(count)) for code, count in metrics["work_type_codes"].items())
    )
    return metrics


def _nullable_numpy(
    array: pa.Array | pa.ChunkedArray, dtype: str
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    nulls = pc.is_null(array).to_numpy(zero_copy_only=False).astype(bool, copy=False)
    values = (
        pc.fill_null(array, 0)
        .to_numpy(zero_copy_only=False)
        .astype(dtype, copy=False)
    )
    return values, nulls


def _output_records(
    table: pa.Table, selected: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    indices = pa.array(selected, type=pa.int64())
    result = np.empty(len(selected), dtype=SOURCE_RECORD_DTYPE)
    result["work"] = table["work_id_numeric"].take(indices).to_numpy(
        zero_copy_only=False
    )
    result["type"] = table["work_type_code"].take(indices).to_numpy(
        zero_copy_only=False
    )
    result["paratext"] = (
        table["is_paratext"]
        .take(indices)
        .to_numpy(zero_copy_only=False)
        .astype("u1", copy=False)
    )
    retracted = table["is_retracted"].take(indices)
    retract_null = pc.is_null(retracted).to_numpy(zero_copy_only=False).astype(
        bool, copy=False
    )
    retract_values = _bool_values(retracted).astype("u1", copy=False)
    retract_values[retract_null] = 2
    result["retracted"] = retract_values
    result["authors"] = table["declared_authors_count"].take(indices).to_numpy(
        zero_copy_only=False
    )
    for source_name, output_name, dtype in (
        ("topic", "primary_topic_id", "<u4"),
        ("field", "primary_field_id", "u1"),
        ("domain", "primary_domain_id", "u1"),
    ):
        values, _ = _nullable_numpy(table[output_name].take(indices), dtype)
        result[source_name] = values
    return result


def _first_mismatch(
    source: np.ndarray[Any, Any],
    expected: np.ndarray[Any, Any],
    *,
    source_field: str,
    output_field: str,
) -> None:
    mismatch = source[source_field] != expected[source_field]
    if not bool(np.any(mismatch)):
        return
    index = int(np.flatnonzero(mismatch)[0])
    raise ReplayVerificationError(
        "source replay mismatch "
        f"field={output_field} work_id={int(source['work'][index])} "
        f"source_value={int(source[source_field][index])} "
        f"output_value={int(expected[source_field][index])}"
    )


def _compare_partition_records(
    source: np.ndarray[Any, Any],
    output_table: pa.Table,
    canonical_table: pa.Table,
    *,
    partition: int,
    full_scope: bool,
) -> tuple[np.ndarray[Any, Any], int]:
    require(source.dtype == SOURCE_RECORD_DTYPE, "partition source dtype mismatch")
    require(output_table.schema == WORK_SCHEMA, f"work schema mismatch in partition {partition}")
    require(canonical_table.schema == CANONICAL_SCHEMA, f"canonical schema mismatch in partition {partition}")
    if len(source):
        order = np.argsort(source["work"], kind="stable")
        source = source[order]
        require(
            bool(np.all(source["work"][1:] > source["work"][:-1])),
            f"duplicate source work ID in partition {partition}",
        )
        require(
            bool(np.all(source["work"] % PARTITION_COUNT == partition)),
            f"source partition formula mismatch in partition {partition}",
        )
    output_work = output_table["work_id_numeric"].to_numpy(zero_copy_only=False)
    canonical_work = canonical_table["work_id_numeric"].to_numpy(zero_copy_only=False)
    require(
        np.array_equal(output_work, canonical_work),
        f"output/canonical work closure mismatch in partition {partition}",
    )
    if full_scope:
        require(
            np.array_equal(source["work"], output_work),
            f"full source/output work closure mismatch in partition {partition}",
        )
        selected = np.arange(len(output_work), dtype=np.int64)
    else:
        selected = np.searchsorted(output_work, source["work"])
        require(
            bool(np.all(selected < len(output_work))),
            f"pilot source work outside output in partition {partition}",
        )
        require(
            np.array_equal(output_work[selected], source["work"]),
            f"pilot source work absent from output in partition {partition}",
        )
    expected = _output_records(output_table, selected)
    for source_field, output_field in (
        ("work", "work_id_numeric"),
        ("type", "work_type_code"),
        ("paratext", "is_paratext"),
        ("retracted", "is_retracted"),
        ("authors", "declared_authors_count"),
        ("topic", "primary_topic_id"),
        ("field", "primary_field_id"),
        ("domain", "primary_domain_id"),
    ):
        _first_mismatch(
            source,
            expected,
            source_field=source_field,
            output_field=output_field,
        )

    canonical_selected = pa.array(selected, type=pa.int64())
    for name in ("publication_day", "publication_year"):
        output_values, output_null = _nullable_numpy(
            output_table[name].take(canonical_selected),
            "<i8",
        )
        canonical_values, canonical_null = _nullable_numpy(
            canonical_table[name].take(canonical_selected),
            "<i8",
        )
        require(
            np.array_equal(output_null, canonical_null),
            f"{name} null-mask mismatch in partition {partition}",
        )
        require(
            np.array_equal(output_values, canonical_values),
            f"{name} value mismatch in partition {partition}",
        )
    require(
        np.array_equal(
            output_table["temporal_disposition"]
            .take(canonical_selected)
            .to_numpy(zero_copy_only=False),
            canonical_table["temporal_disposition"]
            .take(canonical_selected)
            .to_numpy(zero_copy_only=False),
        ),
        f"temporal disposition mismatch in partition {partition}",
    )
    return expected, len(source)


def _read_spool(path: Path) -> np.ndarray[Any, Any]:
    size = path.stat().st_size
    require(size % SOURCE_RECORD_DTYPE.itemsize == 0, f"misaligned spool {path.name}")
    if size == 0:
        return np.empty(0, dtype=SOURCE_RECORD_DTYPE)
    return np.fromfile(path, dtype=SOURCE_RECORD_DTYPE)


def _verify_partitions(
    *,
    output_root: Path,
    canonical_root: Path,
    canonical_artifacts: Mapping[str, Mapping[str, Any]],
    spool_root: Path,
    full_scope: bool,
) -> dict[str, Any]:
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    partition_digest_records: list[dict[str, Any]] = []
    rows = 0
    for partition in range(PARTITION_COUNT):
        if partition % 128 == 0:
            print(
                f"source replay compare {partition}/{PARTITION_COUNT}",
                file=sys.stderr,
                flush=True,
            )
        source = _read_spool(spool_root / f"source-{partition:04d}.bin")
        if len(source):
            source = source[np.argsort(source["work"], kind="stable")]
        output_path = output_root / "work_metadata" / f"part-{partition:04d}.parquet"
        output_table = pq.read_table(output_path).combine_chunks()
        canonical_relative = f"spine/part-{partition:04d}.parquet"
        canonical_path = canonical_root / canonical_relative
        canonical_contract = _mapping(
            canonical_artifacts.get(canonical_relative),
            f"canonical partition artifact {partition}",
        )
        require(
            canonical_path.stat().st_size == canonical_contract.get("bytes"),
            f"canonical partition bytes mismatch {partition}",
        )
        require(
            _sha256_file(canonical_path) == canonical_contract.get("sha256"),
            f"canonical partition SHA mismatch {partition}",
        )
        canonical_table = pq.read_table(canonical_path).combine_chunks()
        expected, count = _compare_partition_records(
            source,
            output_table,
            canonical_table,
            partition=partition,
            full_scope=full_scope,
        )
        source_bytes = source.tobytes(order="C")
        output_bytes = expected.tobytes(order="C")
        source_digest.update(source_bytes)
        output_digest.update(output_bytes)
        partition_digest_records.append(
            {
                "partition_id": partition,
                "rows": count,
                "source_projection_sha256": _sha256_bytes(source_bytes),
                "output_projection_sha256": _sha256_bytes(output_bytes),
            }
        )
        rows += count
    require(
        source_digest.hexdigest() == output_digest.hexdigest(),
        "global source/output projection digest mismatch",
    )
    return {
        "rows": rows,
        "partition_count": PARTITION_COUNT,
        "source_projection_record_sha256": source_digest.hexdigest(),
        "output_projection_record_sha256": output_digest.hexdigest(),
        "partition_comparison_tree_sha256": _tree_digest(partition_digest_records),
        "canonical_spine_artifacts_rehashed": PARTITION_COUNT,
        "canonical_time_join_replayed": True,
        "source_output_field_comparisons": {
            "work_id_numeric": rows,
            "work_type_code": rows,
            "is_paratext": rows,
            "is_retracted_value_and_null_mask": rows,
            "declared_authors_count": rows,
            "primary_topic_id_value_and_null_mask": rows,
            "primary_field_id_value_and_null_mask": rows,
            "primary_domain_id_value_and_null_mask": rows,
        },
    }


def _validate_full_profile(config: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    source = _mapping(config.get("source"), "source")
    profile = _mapping(config.get("input_profile"), "input profile")
    for key in (
        "files",
        "source_bytes",
        "raw_rows",
        "quarantined_w_minus_1_rows",
        "retained_rows",
    ):
        require(metrics.get(key) == source.get(key), f"full source {key} mismatch")
    require(metrics.get("null_counts") == profile.get("null_counts"), "full source null profile mismatch")
    expected_types = _mapping(profile.get("exact_work_type_counts"), "work type counts")
    dictionary = _mapping(profile.get("work_type_dictionary"), "work type dictionary")
    name_by_code = {int(code): str(name) for code, name in dictionary.items()}
    actual_types = {
        name_by_code[int(code)]: int(count)
        for code, count in _mapping(metrics.get("work_type_codes"), "work type codes").items()
    }
    require(actual_types == expected_types, "full source work-type profile mismatch")
    require(
        metrics.get("authors_count_minimum") == profile.get("authors_count_minimum"),
        "authors_count minimum mismatch",
    )
    require(
        metrics.get("authors_count_maximum") == profile.get("authors_count_maximum"),
        "authors_count maximum mismatch",
    )


def _result_fingerprint(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("result_fingerprint_sha256", None)
    payload.pop("execution", None)
    return _sha256_bytes(_canonical_json_bytes(payload))


def _atomic_report(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(_canonical_json_bytes(value, newline=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    started_at = dt.datetime.now(dt.UTC).isoformat()
    config, config_sha = _load_config(args.config)
    source_root = args.source_root.resolve(strict=True)
    content_root = args.content_root.resolve(strict=True)
    canonical_root = args.canonical_root.resolve(strict=True)
    output_root = args.output_root.resolve(strict=True)
    require(source_root.name == _mapping(config.get("source"), "source").get("root_id"), "source root ID mismatch")
    report_path = args.report.resolve(strict=False)
    scratch_path = args.scratch_root.resolve(strict=False)
    require(not _inside(report_path, output_root), "verification report must be outside output root")
    require(not _inside(report_path, scratch_path), "verification report must be outside scratch root")
    require(not _inside(scratch_path, output_root), "scratch root must be outside output root")
    require(not scratch_path.exists(), "scratch root must be fresh")
    require(report_path != scratch_path, "report and scratch paths collide")
    if report_path.exists():
        require(args.replace_report, "verification report already exists")
        require(report_path.is_file() and not report_path.is_symlink(), "existing report is unsafe")

    output_before = _tree_identity(output_root)
    output_receipt, output_manifest, output_artifacts, output_receipt_sha = (
        _verify_output_artifacts(output_root, config_sha)
    )
    source_tasks, content_receipt, content_receipt_sha = _load_source_tasks(
        config, content_root
    )
    canonical_receipt, canonical_artifacts, canonical_receipt_sha = (
        _load_canonical_contract(config, canonical_root)
    )
    full_scope = args.max_source_files is None
    if full_scope:
        require(args.authorize_full, "full replay requires --authorize-full")
        selected_tasks = source_tasks
    else:
        require(
            isinstance(args.max_source_files, int) and args.max_source_files > 0,
            "--max-source-files must be positive",
        )
        require(
            args.max_source_files <= len(source_tasks),
            "--max-source-files exceeds source task count",
        )
        require(not args.authorize_full, "pilot replay cannot use --authorize-full")
        selected_tasks = source_tasks[: args.max_source_files]

    scratch_path.mkdir(mode=0o700, parents=False, exist_ok=False)
    spool_root = scratch_path / "source_spools"
    try:
        scan_metrics = _scan_sources(
            config,
            source_root=source_root,
            tasks=selected_tasks,
            spool_root=spool_root,
            rehash_source_content=bool(args.rehash_source_content),
        )
        if full_scope:
            _validate_full_profile(config, scan_metrics)
        comparison = _verify_partitions(
            output_root=output_root,
            canonical_root=canonical_root,
            canonical_artifacts=canonical_artifacts,
            spool_root=spool_root,
            full_scope=full_scope,
        )
        require(
            comparison["rows"] == scan_metrics["retained_rows"],
            "source scan/partition comparison row closure mismatch",
        )
        if full_scope:
            require(
                comparison["rows"]
                == output_manifest.get("rows")
                == output_receipt.get("work_metadata_rows"),
                "full source/output global row closure mismatch",
            )
        output_after = _tree_identity(output_root)
        require(output_before == output_after, "source replay changed the production output tree")
    except BaseException:
        if not args.keep_scratch_on_failure:
            shutil.rmtree(scratch_path, ignore_errors=True)
        raise
    shutil.rmtree(scratch_path)

    scope = FULL_SCOPE if full_scope else PILOT_SCOPE
    status = FULL_STATUS if full_scope else PILOT_STATUS
    task_payload = {
        "config_sha256": config_sha,
        "scope": scope,
        "source_tasks": [
            {
                "relative_path": task.relative_path,
                "bytes": task.source_bytes,
                "source_stat": task.source_stat,
                "content_sha256": task.content_sha256,
                "checkpoint_sha256": task.checkpoint_sha256,
            }
            for task in selected_tasks
        ],
        "output_receipt_sha256": output_receipt_sha,
        "content_receipt_sha256": content_receipt_sha,
        "canonical_receipt_sha256": canonical_receipt_sha,
        "rehash_source_content": bool(args.rehash_source_content),
    }
    report: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": status,
        "scope": scope,
        "production_source_replay_pass": full_scope,
        "pilot_promotion_allowed": False,
        "gpu_used": False,
        "source_replay": {
            **scan_metrics,
            "content_checkpoint_count_verified": len(selected_tasks),
            "current_source_stat_exact_for_every_file": True,
            "source_content_rehashed": bool(args.rehash_source_content),
            "projected_columns": _mapping(config.get("source"), "source")[
                "projected_columns"
            ],
            "record_abi": [
                ["work", "<u8"],
                ["type", "u1"],
                ["paratext", "u1"],
                ["retracted", "u1"],
                ["authors", "<u2"],
                ["topic", "<u4"],
                ["field", "u1"],
                ["domain", "u1"],
            ],
            "record_bytes": SOURCE_RECORD_DTYPE.itemsize,
        },
        "row_level_verification": comparison,
        "output_verification": {
            "receipt_sha256": output_receipt_sha,
            "artifacts_rehashed": len(output_artifacts),
            "artifact_tree_sha256": output_receipt["artifact_tree_sha256"],
            "work_metadata_tree_sha256": output_receipt[
                "work_metadata_tree_sha256"
            ],
            "output_tree_unchanged": True,
        },
        "parent_bindings": {
            "content_receipt_sha256": content_receipt_sha,
            "content_result_fingerprint_sha256": content_receipt[
                "result_fingerprint_sha256"
            ],
            "canonical_receipt_sha256": canonical_receipt_sha,
            "canonical_result_fingerprint_sha256": canonical_receipt[
                "result_fingerprint_sha256"
            ],
        },
        "gate_state": {
            "cutoff_primitives_source_projection_replay": (
                "PASS" if full_scope else "PILOT_NONPROMOTABLE"
            ),
            "author_entity_validity": "UNRESOLVED_CURRENT_SNAPSHOT_TOKEN_ONLY",
            "historical_topic_observability": "UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT",
            "historical_citation_observation_time": "UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT",
            "final_cutoff_panel": "PENDING_G0_RESOLVED_SUCCESSOR",
            "g_minus_1_data_readiness": "PENDING",
            "g0_measurement_freeze": "PENDING",
            "empirical_g1_signature": "NOT_RUN",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "provenance": {
            "config_relative_path": args.config.resolve(strict=True)
            .relative_to(REPO_ROOT)
            .as_posix(),
            "config_sha256": config_sha,
            "tool_relative_path": Path(__file__).resolve()
            .relative_to(REPO_ROOT)
            .as_posix(),
            "tool_sha256": _sha256_file(Path(__file__).resolve()),
            "task_fingerprint_sha256": _sha256_bytes(
                _canonical_json_bytes(task_payload)
            ),
        },
        "claim_boundary": (
            "A full pass establishes a separately implemented row-level replay "
            "of the eight current-snapshot main-work source columns against every "
            "published cutoff work-metadata row, plus canonical-time closure and "
            "read-only output integrity. Source content bytes are independently "
            + (
                "rehashed against the content baseline. "
                if args.rehash_source_content
                else "bound through exact content checkpoints and unchanged rich "
                "file identities but are not rehashed in this run. "
            )
            + "It does not establish natural-person identity, historical topic or "
            "citation observability, candidate eligibility, a final analytic "
            "panel, G-1, G0, empirical G1, an NABM effect, or a scientific effect."
        ),
        "execution": {
            "started_at_utc": started_at,
            "finished_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "workload_class": "CPU_IO_HDD_SOURCE_TO_SSD_SPOOL_AND_FULL_ROW_COMPARE",
            "scratch_removed_after_success": True,
        },
    }
    report["result_fingerprint_sha256"] = _result_fingerprint(report)
    _atomic_report(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--content-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-source-files", type=int)
    parser.add_argument("--authorize-full", action="store_true")
    parser.add_argument(
        "--rehash-source-content",
        action="store_true",
        help="Also rehash every selected source file against the content baseline.",
    )
    parser.add_argument(
        "--replace-report",
        action="store_true",
        help="Replace an existing external report; never changes production output.",
    )
    parser.add_argument(
        "--keep-scratch-on-failure",
        action="store_true",
        help="Keep private scratch after a failed run for operator diagnosis.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (
        ReplayVerificationError,
        OSError,
        ValueError,
        pa.ArrowException,
        yaml.YAMLError,
    ) as exc:
        print(json.dumps({"status": BLOCKED_STATUS, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
