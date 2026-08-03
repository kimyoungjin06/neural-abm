#!/usr/bin/env python3
"""Run a resumable exact OpenAlex G-1 main-work ID body recount.

The command reads the real Parquet ID data pages, validates canonical OpenAlex
work IDs, materializes their numeric suffixes as uint64, sorts the full keys,
and computes exact distinct and duplicate counts. Hashes are used only for a
diagnostic bucket histogram; equality is always evaluated on the full key.

Passing this command is not PASS_G1. Child joins, author validity, dates,
cutoff-aware features, leakage tests, and a representative pilot remain gated.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/openalex_g1_recount_v0.yaml"
STATE_SCHEMA_VERSION = "openalex_g1_main_id_recount_state.v1"
REPORT_SCHEMA_VERSION = "openalex_g1_main_id_recount_report.v1"
PLAN_SCHEMA_VERSION = "openalex_g1_main_id_recount_plan.v1"
UINT64_BYTES = np.dtype("<u8").itemsize
SPLITMIX64_BUCKETS = 256
SPLITMIX64_SEED = 0


class RecountError(RuntimeError):
    """Raised when a frozen recount contract cannot be honored."""


def lstat_mode(path: Path) -> int | None:
    """Return a path's own mode without following symlinks."""

    try:
        return path.lstat().st_mode
    except FileNotFoundError:
        return None


def validate_output_directory_path(path: Path) -> None:
    mode = lstat_mode(path)
    if mode is None:
        return
    if stat.S_ISLNK(mode):
        raise RecountError(f"recount output directory must not be a symlink: {path}")
    if not stat.S_ISDIR(mode):
        raise RecountError(f"recount output path is not a directory: {path}")


def validate_managed_output_path(path: Path) -> None:
    mode = lstat_mode(path)
    if mode is None:
        return
    if stat.S_ISLNK(mode):
        raise RecountError(f"managed recount path must not be a symlink: {path.name}")
    if not stat.S_ISREG(mode):
        raise RecountError(f"managed recount path must be a regular file: {path.name}")


def lexical_absolute(path: Path) -> Path:
    """Normalize dots without resolving symlink components."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def validate_output_containment(*, root: Path, output_dir: Path) -> None:
    lexical_root = lexical_absolute(root)
    lexical_output = lexical_absolute(output_dir)
    resolved_root = root.expanduser().resolve()
    resolved_output = output_dir.expanduser().resolve()
    if lexical_output.is_relative_to(lexical_root):
        raise RecountError(
            "recount output must not be lexically inside the source root"
        )
    if resolved_output.is_relative_to(resolved_root):
        raise RecountError("recount output must not resolve inside the source root")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    validate_managed_output_path(path)
    validate_managed_output_path(temporary)
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    validate_managed_output_path(path)
    validate_managed_output_path(temporary)
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def git_metadata() -> dict[str, Any]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "status_error": status.stderr.strip() if status.returncode != 0 else None,
    }


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecountError("recount config must be a mapping")
    if value.get("schema_version") != "openalex_g1_recount_config.v1":
        raise RecountError("unsupported recount config schema_version")
    if not isinstance(value.get("main_id_recount"), dict):
        raise RecountError("recount config must define main_id_recount")
    settings = value["main_id_recount"]
    if settings.get("spool_dtype") != "uint64":
        raise RecountError("main_id_recount.spool_dtype must be uint64")
    for field in (
        "expected_files",
        "expected_rows",
        "batch_size",
        "duplicate_scan_chunk_rows",
    ):
        if int(settings.get(field, 0)) < 1:
            raise RecountError(f"main_id_recount.{field} must be positive")
    return value


def resolve_root(config: dict[str, Any], explicit_root: Path | None) -> Path:
    if explicit_root is None:
        env_name = str(config["parent_inventory"]["source_root_env"])
        raw = os.environ.get(env_name)
        if not raw:
            raise RecountError(f"set {env_name} or pass --root")
        root = Path(raw)
    else:
        root = explicit_root
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise RecountError(f"source root is not a directory: {root}")
    if root.name != str(config["parent_inventory"]["source_root_id"]):
        raise RecountError("source root basename does not match the frozen contract")
    return root


def safe_relative_path(raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise RecountError(f"unsafe path in parent inventory: {raw}")
    return relative


def inventory_fingerprint(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["relative_path"])):
        digest.update(
            f"{record['relative_path']}\0{int(record['bytes'])}\n".encode("utf-8")
        )
    return digest.hexdigest()


def load_and_validate_parent(
    *,
    path: Path,
    root: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    parent_sha256 = sha256_file(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    frozen = config["parent_inventory"]
    if manifest.get("schema_version") != frozen["schema_version"]:
        raise RecountError("parent inventory schema_version mismatch")
    if manifest.get("manifest_id") != frozen["manifest_id"]:
        raise RecountError("parent inventory manifest_id mismatch")
    root_record = manifest.get("root", {})
    if root_record.get("id") != frozen["source_root_id"]:
        raise RecountError("parent inventory root ID mismatch")
    expected_fingerprint = str(frozen["inventory_fingerprint_sha256"])
    if root_record.get("inventory_fingerprint_sha256") != expected_fingerprint:
        raise RecountError("parent inventory fingerprint does not match the config")
    if int(manifest.get("summary", {}).get("block_count", 1)) != 0:
        raise RecountError("parent inventory contains blocking checks")

    all_records: list[dict[str, Any]] = []
    main_table_name = str(config["main_id_recount"]["table"])
    main_records: list[dict[str, Any]] | None = None
    main_table_footer_rows: int | None = None
    problems: list[str] = []
    for table in manifest.get("tables", []):
        records = list(table.get("files", []))
        if table.get("name") == main_table_name:
            main_records = records
            main_table_footer_rows = int(table.get("footer_rows", -1))
        for record in records:
            if record.get("error"):
                problems.append(f"footer error: {record.get('relative_path')}")
                continue
            relative = safe_relative_path(str(record["relative_path"]))
            source = root / relative
            if source.is_symlink() or not source.is_file():
                problems.append(f"missing or symlinked input: {relative.as_posix()}")
                continue
            stat = source.stat()
            if int(stat.st_size) != int(record["bytes"]):
                problems.append(f"size drift: {relative.as_posix()}")
            if int(stat.st_mtime_ns) != int(record["mtime_ns"]):
                problems.append(f"mtime drift: {relative.as_posix()}")
            all_records.append(record)
    if problems:
        preview = "; ".join(problems[:10])
        raise RecountError(f"parent inventory no longer matches live input: {preview}")
    if inventory_fingerprint(all_records) != expected_fingerprint:
        raise RecountError("live path-and-size inventory fingerprint mismatch")
    if main_records is None:
        raise RecountError(
            f"main table not found in parent inventory: {main_table_name}"
        )
    expected_files = int(config["main_id_recount"]["expected_files"])
    if len(main_records) != expected_files:
        raise RecountError(
            f"main input file count mismatch: {len(main_records)} != {expected_files}"
        )
    expected_rows = int(config["main_id_recount"]["expected_rows"])
    observed_rows = sum(int(record["footer_rows"]) for record in main_records)
    if observed_rows != expected_rows or main_table_footer_rows != expected_rows:
        raise RecountError(
            "main footer row total does not match the frozen recount contract: "
            f"records={observed_rows}, table={main_table_footer_rows}, "
            f"expected={expected_rows}"
        )
    return (
        manifest,
        parent_sha256,
        sorted(main_records, key=lambda item: str(item["relative_path"])),
    )


def file_record_sha256(record: dict[str, Any]) -> str:
    identity = {
        "relative_path": record["relative_path"],
        "bytes": int(record["bytes"]),
        "mtime_ns": int(record["mtime_ns"]),
        "footer_rows": int(record["footer_rows"]),
        "canonical_schema_fingerprint_sha256": record.get(
            "canonical_schema_fingerprint_sha256"
        ),
        "physical_schema_fingerprint_sha256": record.get(
            "physical_schema_fingerprint_sha256"
        ),
    }
    return sha256_bytes(canonical_json_bytes(identity))


def compressed_column_bytes(parquet: pq.ParquetFile, column: str) -> int:
    metadata = parquet.metadata
    total = 0
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        path_to_index = {
            row_group.column(index).path_in_schema: index
            for index in range(row_group.num_columns)
        }
        if column not in path_to_index:
            raise RecountError(f"column missing from physical schema: {column}")
        total += int(row_group.column(path_to_index[column]).total_compressed_size)
    return total


def _python_parse_ids(
    values: pa.Array,
    *,
    prefix: str,
    sample_limit: int,
) -> tuple[np.ndarray, int, int, list[str]]:
    canonical = re.compile(rf"{re.escape(prefix)}(0|[1-9][0-9]*)")
    maximum = int(np.iinfo(np.uint64).max)
    numeric = np.zeros(len(values), dtype="<u8")
    null_count = 0
    invalid_count = 0
    invalid_samples: list[str] = []
    for index, value in enumerate(values.to_pylist()):
        if value is None:
            null_count += 1
            continue
        match = canonical.fullmatch(value)
        if match is None:
            invalid_count += 1
            if len(invalid_samples) < sample_limit:
                invalid_samples.append(str(value)[:200])
            continue
        parsed = int(match.group(1))
        if parsed > maximum:
            invalid_count += 1
            if len(invalid_samples) < sample_limit:
                invalid_samples.append(str(value)[:200])
            continue
        numeric[index] = parsed
    return numeric, null_count, invalid_count, invalid_samples


def parse_canonical_openalex_ids(
    values: pa.Array,
    *,
    prefix: str,
    sample_limit: int,
) -> tuple[np.ndarray, int, int, list[str]]:
    """Return injective uint64 keys and exact null/noncanonical counts."""

    suffix = pc.utf8_slice_codeunits(values, start=len(prefix))
    try:
        numeric_arrow = pc.cast(suffix, pa.uint64(), safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return _python_parse_ids(values, prefix=prefix, sample_limit=sample_limit)

    prefix_ok = pc.fill_null(pc.starts_with(values, pattern=prefix), False)
    nonempty = pc.fill_null(pc.greater(pc.utf8_length(suffix), 0), False)
    leading_zero = pc.fill_null(
        pc.equal(pc.utf8_slice_codeunits(suffix, 0, 1), "0"), True
    )
    canonical_zero = pc.fill_null(pc.equal(suffix, "0"), False)
    canonical_leading_digit = pc.or_(canonical_zero, pc.invert(leading_zero))
    shape_ok = pc.and_(prefix_ok, pc.and_(nonempty, canonical_leading_digit))
    reconstructed = pc.cast(numeric_arrow, pa.string())
    exact_numeric_text = pc.fill_null(pc.equal(suffix, reconstructed), False)
    canonical_ok = pc.and_(
        shape_ok,
        pc.and_(pc.is_valid(numeric_arrow), exact_numeric_text),
    )
    invalid_mask = pc.and_(pc.is_valid(values), pc.invert(canonical_ok))
    invalid_scalar = pc.sum(pc.cast(invalid_mask, pa.int64())).as_py()
    invalid_count = int(invalid_scalar or 0)
    invalid_samples: list[str] = []
    if invalid_count:
        invalid_values = pc.filter(values, invalid_mask)
        invalid_samples = [
            str(value)[:200]
            for value in invalid_values.slice(0, sample_limit).to_pylist()
        ]
    canonical_numeric = pc.if_else(canonical_ok, numeric_arrow, None)
    filled = pc.fill_null(canonical_numeric, 0)
    numeric = np.asarray(filled.to_numpy(zero_copy_only=False), dtype="<u8")
    return numeric, int(values.null_count), invalid_count, invalid_samples


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Versioned uint64 SplitMix64 transform used only for bucket placement."""

    with np.errstate(over="ignore"):
        mixed = values.astype(np.uint64, copy=True) + np.uint64(
            0x9E3779B97F4A7C15 + SPLITMIX64_SEED
        )
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        mixed ^= mixed >> np.uint64(31)
    return mixed


def splitmix64_bucket_histogram(values: np.ndarray) -> np.ndarray:
    buckets = np.bitwise_and(splitmix64(values), np.uint64(255)).astype(np.uint8)
    return np.bincount(buckets, minlength=SPLITMIX64_BUCKETS).astype(np.uint64)


def available_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def ensure_capacity(output_dir: Path, rows: int) -> dict[str, int | None]:
    spool_bytes = rows * UINT64_BYTES
    required_disk = spool_bytes * 2 + 1024**3
    disk_free = int(shutil.disk_usage(output_dir).free)
    if disk_free < required_disk:
        raise RecountError(
            f"insufficient output disk: {disk_free} < required {required_disk}"
        )
    memory_available = available_memory_bytes()
    required_memory = spool_bytes * 2 + 1024**3
    if memory_available is not None and memory_available < required_memory:
        raise RecountError(
            "insufficient available memory for the exact sorted-copy phase: "
            f"{memory_available} < {required_memory}"
        )
    return {
        "spool_bytes_each": spool_bytes,
        "required_disk_bytes": required_disk,
        "disk_free_bytes_at_start": disk_free,
        "required_memory_bytes": required_memory,
        "memory_available_bytes_at_start": memory_available,
    }


def preallocate(path: Path, size: int) -> None:
    validate_managed_output_path(path)
    with path.open("wb") as handle:
        handle.truncate(size)
        handle.flush()
        os.fsync(handle.fileno())


def fsync_file(path: Path) -> None:
    validate_managed_output_path(path)
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def flush_memmap_range(
    spool: np.memmap,
    *,
    start_row: int,
    end_row: int,
    path: Path,
) -> None:
    start_byte = start_row * UINT64_BYTES
    end_byte = end_row * UINT64_BYTES
    aligned_start = start_byte - (start_byte % mmap.PAGESIZE)
    try:
        spool._mmap.flush(aligned_start, end_byte - aligned_start)  # type: ignore[attr-defined]
    except (AttributeError, BufferError, OSError, ValueError):
        spool.flush()
    fsync_file(path)


def file_scope_fingerprint(records: Iterable[dict[str, Any]]) -> str:
    values = [
        {
            "relative_path": record["relative_path"],
            "bytes": int(record["bytes"]),
            "mtime_ns": int(record["mtime_ns"]),
            "footer_rows": int(record["footer_rows"]),
        }
        for record in records
    ]
    return sha256_bytes(canonical_json_bytes(values))


def plan_payload(
    *,
    config: dict[str, Any],
    config_path: Path,
    parent_sha256: str,
    parent_manifest: dict[str, Any],
    records: list[dict[str, Any]],
    max_files: int | None,
    batch_size: int,
) -> dict[str, Any]:
    full_scope = max_files is None
    selected = records if full_scope else records[:max_files]
    selected_rows = sum(int(record["footer_rows"]) for record in selected)
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "label": str(config["label"]),
        "scope": "full_main_id_recount" if full_scope else "engineering_file_pilot",
        "selected_file_count": len(selected),
        "selected_footer_rows": selected_rows,
        "selected_file_fingerprint_sha256": file_scope_fingerprint(selected),
        "parent_manifest_sha256": parent_sha256,
        "parent_inventory_fingerprint_sha256": parent_manifest["root"][
            "inventory_fingerprint_sha256"
        ],
        "parent_stability_fingerprint_sha256": parent_manifest["root"][
            "stability_fingerprint_sha256"
        ],
        "config_sha256": sha256_file(config_path),
        "tool_sha256": sha256_file(Path(__file__)),
        "table": str(config["main_id_recount"]["table"]),
        "column": str(config["main_id_recount"]["column"]),
        "id_codec": "canonical_openalex_work_id_numeric_suffix_uint64.v1",
        "batch_size": batch_size,
        "bucket_algorithm": {
            "name": "splitmix64-v1",
            "seed": SPLITMIX64_SEED,
            "bucket_count": SPLITMIX64_BUCKETS,
            "equality_key": "full_uint64_not_hash",
            "test_vectors": {
                "0": "e220a8397b1dcdaf",
                "1": "910a2dec89025cc1",
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
        },
    }
    payload["plan_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def validate_state(state: dict[str, Any], plan: dict[str, Any]) -> None:
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise RecountError("unsupported recount state schema")
    for key in (
        "plan_sha256",
        "parent_manifest_sha256",
        "config_sha256",
        "tool_sha256",
        "selected_file_fingerprint_sha256",
    ):
        if state.get(key) != plan.get(key):
            raise RecountError(f"resume refused because {key} changed")


def validate_complete_outputs(
    *,
    state: dict[str, Any],
    report_path: Path,
    markdown_path: Path,
    raw_path: Path,
    sorted_path: Path,
) -> dict[str, Any]:
    artifact_paths = {raw_path.name: raw_path, sorted_path.name: sorted_path}
    artifact_hashes = dict(state.get("artifact_sha256", {}))
    unknown_artifacts = sorted(set(artifact_hashes) - set(artifact_paths))
    if unknown_artifacts:
        raise RecountError(f"unknown artifacts in complete state: {unknown_artifacts}")
    required = (report_path, markdown_path) + tuple(
        artifact_paths[name] for name in artifact_hashes
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RecountError(f"complete recount outputs are missing: {missing}")
    if sha256_file(report_path) != state.get("report_sha256"):
        raise RecountError("complete recount report checksum mismatch")
    if sha256_file(markdown_path) != state.get("markdown_sha256"):
        raise RecountError("complete recount Markdown checksum mismatch")
    validate_artifact_hashes(
        state=state,
        raw_path=raw_path,
        sorted_path=sorted_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("plan_sha256") != state.get("plan_sha256"):
        raise RecountError("complete recount report plan mismatch")
    return report


def validate_artifact_hashes(
    *,
    state: dict[str, Any],
    raw_path: Path,
    sorted_path: Path,
) -> None:
    artifact_paths = {raw_path.name: raw_path, sorted_path.name: sorted_path}
    for name, expected_sha256 in dict(state.get("artifact_sha256", {})).items():
        artifact = artifact_paths.get(name)
        if artifact is None or not artifact.is_file():
            raise RecountError(f"recount artifact is missing or unknown: {name}")
        if sha256_file(artifact) != expected_sha256:
            raise RecountError(f"recount artifact checksum mismatch: {name}")


def unavailable_exact_statistics() -> dict[str, int | None]:
    return {
        "distinct_ids": None,
        "duplicate_row_excess": None,
        "duplicate_id_groups": None,
        "max_multiplicity": None,
        "min_numeric_id": None,
        "max_numeric_id": None,
    }


def checkpoint_id(
    *,
    plan_sha256: str,
    record: dict[str, Any],
    column: str,
) -> str:
    value = {
        "plan_sha256": plan_sha256,
        "stage": "materialize_main_id",
        "table_relative_path": record["relative_path"],
        "frozen_file_record_sha256": file_record_sha256(record),
        "column_projection": [column],
        "id_codec": "canonical_openalex_work_id_numeric_suffix_uint64.v1",
        "bucket_algorithm": "splitmix64-v1/256/seed-0",
    }
    return sha256_bytes(canonical_json_bytes(value))


def segment_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    view = np.ascontiguousarray(array).view(np.uint8)
    digest.update(view)
    return digest.hexdigest()


def validate_completed_segments(
    *,
    spool: np.memmap,
    file_metrics: list[dict[str, Any]],
) -> None:
    for metric in file_metrics:
        start = int(metric["start_row"])
        end = start + int(metric["rows"])
        observed = segment_sha256(spool[start:end])
        if observed != metric["decoded_id_sha256"]:
            raise RecountError(
                "resume refused because a completed spool segment checksum changed: "
                f"{metric['relative_path']}"
            )


def validate_checkpoint_metrics(
    *,
    records: list[dict[str, Any]],
    state: dict[str, Any],
    plan: dict[str, Any],
    column: str,
) -> None:
    metrics = list(state.get("file_metrics", []))
    completed_files = int(state.get("completed_files", -1))
    if completed_files != len(metrics) or completed_files > len(records):
        raise RecountError("checkpoint file count is inconsistent")
    expected_start = 0
    null_ids = 0
    invalid_ids = 0
    compressed_bytes = 0
    histogram = np.zeros(SPLITMIX64_BUCKETS, dtype=np.uint64)
    for index, metric in enumerate(metrics):
        record = records[index]
        expected_checkpoint = checkpoint_id(
            plan_sha256=plan["plan_sha256"],
            record=record,
            column=column,
        )
        conditions = (
            int(metric.get("file_index", -1)) == index,
            metric.get("relative_path") == record["relative_path"],
            metric.get("checkpoint_id") == expected_checkpoint,
            metric.get("frozen_file_record_sha256") == file_record_sha256(record),
            int(metric.get("start_row", -1)) == expected_start,
            int(metric.get("rows", -1)) == int(record["footer_rows"]),
        )
        if not all(conditions):
            raise RecountError(
                f"checkpoint metadata mismatch: {record['relative_path']}"
            )
        file_histogram = list(metric.get("bucket_histogram", []))
        if len(file_histogram) != SPLITMIX64_BUCKETS:
            raise RecountError(
                f"checkpoint bucket histogram missing: {record['relative_path']}"
            )
        histogram += np.asarray(file_histogram, dtype=np.uint64)
        expected_start += int(metric["rows"])
        null_ids += int(metric["null_ids"])
        invalid_ids += int(metric["invalid_ids"])
        compressed_bytes += int(metric["id_column_compressed_bytes"])
    aggregate_conditions = (
        int(state.get("written_rows", -1)) == expected_start,
        int(state.get("null_ids", -1)) == null_ids,
        int(state.get("invalid_ids", -1)) == invalid_ids,
        int(state.get("id_column_compressed_bytes", -1)) == compressed_bytes,
        list(state.get("bucket_histogram", [])) == [int(value) for value in histogram],
    )
    if not all(aggregate_conditions):
        raise RecountError("checkpoint aggregate counters are inconsistent")


def create_state(
    plan: dict[str, Any], capacity: dict[str, int | None]
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "phase": "materialize",
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "parent_manifest_sha256": plan["parent_manifest_sha256"],
        "config_sha256": plan["config_sha256"],
        "tool_sha256": plan["tool_sha256"],
        "selected_file_fingerprint_sha256": plan["selected_file_fingerprint_sha256"],
        "completed_files": 0,
        "written_rows": 0,
        "null_ids": 0,
        "invalid_ids": 0,
        "invalid_samples": [],
        "id_column_compressed_bytes": 0,
        "bucket_histogram": [0] * SPLITMIX64_BUCKETS,
        "file_metrics": [],
        "capacity": capacity,
    }


def update_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = utc_now()
    atomic_write_json(path, state)


def materialize_ids(
    *,
    root: Path,
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    config: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    spool_path: Path,
    pause_after_new_files: int | None,
) -> bool:
    expected_rows = int(plan["selected_footer_rows"])
    spool = np.memmap(spool_path, dtype="<u8", mode="r+", shape=(expected_rows,))
    settings = config["main_id_recount"]
    column = str(settings["column"])
    validate_checkpoint_metrics(
        records=records,
        state=state,
        plan=plan,
        column=column,
    )
    validate_completed_segments(spool=spool, file_metrics=state["file_metrics"])
    offset = int(state["written_rows"])
    completed_at_start = int(state["completed_files"])
    prefix = str(settings["canonical_prefix"])
    batch_size = int(plan["batch_size"])
    sample_limit = int(settings["invalid_sample_limit"])
    histogram = np.asarray(state["bucket_histogram"], dtype=np.uint64)

    for file_index in range(completed_at_start, len(records)):
        record = records[file_index]
        source = root / safe_relative_path(str(record["relative_path"]))
        file_started = time.perf_counter()
        parquet = pq.ParquetFile(source)
        compressed_bytes = compressed_column_bytes(parquet, column)
        file_rows = 0
        file_nulls = 0
        file_invalid = 0
        file_samples: list[str] = []
        file_digest = hashlib.sha256()
        file_histogram = np.zeros(SPLITMIX64_BUCKETS, dtype=np.uint64)
        start_row = offset

        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=[column],
            use_threads=True,
        ):
            values = batch.column(0)
            remaining_samples = max(0, sample_limit - len(file_samples))
            numeric, null_count, invalid_count, samples = parse_canonical_openalex_ids(
                values,
                prefix=prefix,
                sample_limit=remaining_samples,
            )
            if offset + len(numeric) > expected_rows:
                raise RecountError("body scan exceeds the selected footer row total")
            spool[offset : offset + len(numeric)] = numeric
            file_digest.update(numeric.view(np.uint8))
            file_histogram += splitmix64_bucket_histogram(numeric)
            offset += len(numeric)
            file_rows += len(numeric)
            file_nulls += null_count
            file_invalid += invalid_count
            file_samples.extend(samples)

        expected_file_rows = int(record["footer_rows"])
        if file_rows != expected_file_rows:
            raise RecountError(
                f"data-page rows differ from footer rows for {record['relative_path']}: "
                f"{file_rows} != {expected_file_rows}"
            )
        flush_memmap_range(
            spool,
            start_row=start_row,
            end_row=offset,
            path=spool_path,
        )
        histogram += file_histogram
        metric = {
            "file_index": file_index,
            "relative_path": record["relative_path"],
            "checkpoint_id": checkpoint_id(
                plan_sha256=plan["plan_sha256"],
                record=record,
                column=column,
            ),
            "frozen_file_record_sha256": file_record_sha256(record),
            "start_row": start_row,
            "rows": file_rows,
            "null_ids": file_nulls,
            "invalid_ids": file_invalid,
            "invalid_samples": file_samples,
            "id_column_compressed_bytes": compressed_bytes,
            "bucket_histogram": [int(value) for value in file_histogram],
            "decoded_id_sha256": file_digest.hexdigest(),
            "elapsed_seconds": time.perf_counter() - file_started,
        }
        state["file_metrics"].append(metric)
        state["completed_files"] = file_index + 1
        state["written_rows"] = offset
        state["null_ids"] = int(state["null_ids"]) + file_nulls
        state["invalid_ids"] = int(state["invalid_ids"]) + file_invalid
        state["invalid_samples"] = (list(state["invalid_samples"]) + file_samples)[
            :sample_limit
        ]
        state["id_column_compressed_bytes"] = (
            int(state["id_column_compressed_bytes"]) + compressed_bytes
        )
        state["bucket_histogram"] = [int(value) for value in histogram]
        update_state(state_path, state)

        if pause_after_new_files is not None:
            new_completed = int(state["completed_files"]) - completed_at_start
            if new_completed >= pause_after_new_files and file_index + 1 < len(records):
                state["status"] = "PAUSED"
                update_state(state_path, state)
                return False

    if offset != expected_rows:
        raise RecountError(
            f"data-page row total mismatch: {offset} != selected {expected_rows}"
        )
    if int(histogram.sum()) != offset:
        raise RecountError("SplitMix64 bucket histogram does not conserve rows")
    state["phase"] = "sort"
    state["status"] = "RUNNING"
    update_state(state_path, state)
    return True


def exact_duplicate_statistics(
    sorted_ids: np.ndarray,
    *,
    chunk_rows: int,
) -> dict[str, int | None]:
    rows = len(sorted_ids)
    if not rows:
        return {
            "distinct_ids": 0,
            "duplicate_row_excess": 0,
            "duplicate_id_groups": 0,
            "max_multiplicity": 0,
            "min_numeric_id": None,
            "max_numeric_id": None,
        }
    duplicate_excess = 0
    duplicate_groups = 0
    max_multiplicity = 1
    carry_value: int | None = None
    carry_length = 0

    def accumulate_finalized(lengths: np.ndarray) -> tuple[int, int, int]:
        if not len(lengths):
            return 0, 0, 0
        duplicate_mask = lengths > 1
        groups = int(np.count_nonzero(duplicate_mask))
        excess = int(np.sum(lengths[duplicate_mask] - 1, dtype=np.int64))
        maximum = int(lengths.max())
        return excess, groups, maximum

    for start in range(0, rows, chunk_rows):
        block = np.asarray(sorted_ids[start : min(rows, start + chunk_rows)])
        changes = np.flatnonzero(block[1:] != block[:-1]) + 1
        boundaries = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                changes.astype(np.int64, copy=False),
                np.array([len(block)], dtype=np.int64),
            )
        )
        lengths = np.diff(boundaries)
        if carry_value is not None:
            if int(block[0]) == carry_value:
                lengths[0] += carry_length
            else:
                excess, groups, maximum = accumulate_finalized(
                    np.array([carry_length], dtype=np.int64)
                )
                duplicate_excess += excess
                duplicate_groups += groups
                max_multiplicity = max(max_multiplicity, maximum)
        finalized = lengths[:-1]
        excess, groups, maximum = accumulate_finalized(finalized)
        duplicate_excess += excess
        duplicate_groups += groups
        max_multiplicity = max(max_multiplicity, maximum)
        carry_value = int(block[-1])
        carry_length = int(lengths[-1])

    excess, groups, maximum = accumulate_finalized(
        np.array([carry_length], dtype=np.int64)
    )
    duplicate_excess += excess
    duplicate_groups += groups
    max_multiplicity = max(max_multiplicity, maximum)
    return {
        "distinct_ids": rows - duplicate_excess,
        "duplicate_row_excess": duplicate_excess,
        "duplicate_id_groups": duplicate_groups,
        "max_multiplicity": max_multiplicity,
        "min_numeric_id": int(sorted_ids[0]),
        "max_numeric_id": int(sorted_ids[-1]),
    }


def sort_and_count(
    *,
    state: dict[str, Any],
    state_path: Path,
    raw_path: Path,
    sorted_path: Path,
    rows: int,
    chunk_rows: int,
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    column: str,
) -> tuple[dict[str, int | None], dict[str, str], float]:
    started = time.perf_counter()
    raw = np.memmap(raw_path, dtype="<u8", mode="r", shape=(rows,))
    validate_checkpoint_metrics(
        records=records,
        state=state,
        plan=plan,
        column=column,
    )
    validate_completed_segments(spool=raw, file_metrics=state["file_metrics"])
    values = np.array(raw, dtype="<u8", copy=True)
    values.sort(kind="quicksort")
    preallocate(sorted_path, rows * UINT64_BYTES)
    sorted_ids = np.memmap(sorted_path, dtype="<u8", mode="r+", shape=(rows,))
    sorted_ids[:] = values
    sorted_ids.flush()
    fsync_file(sorted_path)
    del values
    del raw
    statistics = exact_duplicate_statistics(sorted_ids, chunk_rows=chunk_rows)
    artifact_hashes = {
        raw_path.name: sha256_file(raw_path),
        sorted_path.name: sha256_file(sorted_path),
    }
    del sorted_ids
    elapsed = time.perf_counter() - started
    state["phase"] = "report"
    state["artifact_sha256"] = artifact_hashes
    state["exact_statistics"] = statistics
    state["sort_count_and_checksum_seconds"] = elapsed
    update_state(state_path, state)
    return statistics, artifact_hashes, elapsed


def evaluate_checks(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
    statistics: dict[str, int | None],
) -> list[dict[str, Any]]:
    settings = config["main_id_recount"]
    full_scope = plan["scope"] == "full_main_id_recount"
    checks: list[dict[str, Any]] = []

    def add(
        name: str, observed: Any, expected: Any, *, full_only: bool = False
    ) -> None:
        if full_only and not full_scope:
            status = "PENDING"
        else:
            status = "PASS" if observed == expected else "BLOCK"
        checks.append(
            {"name": name, "status": status, "observed": observed, "expected": expected}
        )

    add(
        "data_page_rows",
        int(state["written_rows"]),
        (
            int(settings["expected_rows"])
            if full_scope
            else int(plan["selected_footer_rows"])
        ),
    )
    add(
        "null_main_work_ids", int(state["null_ids"]), int(settings["expected_null_ids"])
    )
    add(
        "invalid_main_work_ids",
        int(state["invalid_ids"]),
        int(settings["expected_invalid_ids"]),
    )
    add(
        "exact_distinct_main_work_ids",
        statistics["distinct_ids"],
        int(settings["expected_distinct_ids"]),
        full_only=True,
    )
    add(
        "duplicate_main_work_row_excess",
        statistics["duplicate_row_excess"],
        int(settings["expected_duplicate_row_excess"]),
        full_only=True,
    )
    checks.append(
        {
            "name": "bucket_histogram_conserves_rows",
            "status": (
                "PASS"
                if sum(int(value) for value in state["bucket_histogram"])
                == int(state["written_rows"])
                else "BLOCK"
            ),
            "observed": sum(int(value) for value in state["bucket_histogram"]),
            "expected": int(state["written_rows"]),
        }
    )
    return checks


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# OpenAlex G-1 Main-ID Body Recount",
        "",
        f"- Status: `{report['status']}`",
        f"- Scope: `{report['scope']}`",
        f"- PASS main-ID recount: `{str(report['pass_main_id_recount']).lower()}`",
        f"- PASS G-1: `{str(report['pass_g1']).lower()}`",
        f"- Source root ID: `{report['source_root_id']}`",
        f"- Parent inventory: `{report['parent_inventory_fingerprint_sha256']}`",
        "",
        "## Exact Counts",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Data-page rows scanned | {counts['data_page_rows']} |",
        f"| Null IDs | {counts['null_ids']} |",
        f"| Invalid IDs | {counts['invalid_ids']} |",
        f"| Exact distinct IDs | {counts['distinct_ids']} |",
        f"| Duplicate row excess | {counts['duplicate_row_excess']} |",
        f"| Duplicate ID groups | {counts['duplicate_id_groups']} |",
        f"| Maximum multiplicity | {counts['max_multiplicity']} |",
        "",
        "## Checks",
        "",
        "| Check | Status | Observed | Expected |",
        "| --- | --- | ---: | ---: |",
    ]
    for item in report["checks"]:
        lines.append(
            f"| `{item['name']}` | {item['status']} | {item['observed']} | "
            f"{item['expected']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            report["claim_boundary"],
            "",
            "The remaining relation, author, date, cutoff, and leakage gates are pending.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
    statistics: dict[str, int | None],
    artifact_hashes: dict[str, str],
    sort_seconds: float,
    parent_stable_after: bool,
    invocation_elapsed_seconds: float,
    raw_path: Path,
    sorted_path: Path,
) -> dict[str, Any]:
    checks = evaluate_checks(
        config=config,
        plan=plan,
        state=state,
        statistics=statistics,
    )
    checks.append(
        {
            "name": "parent_inventory_stable_after_recount",
            "status": "PASS" if parent_stable_after else "BLOCK",
            "observed": parent_stable_after,
            "expected": True,
        }
    )
    blocks = [item for item in checks if item["status"] == "BLOCK"]
    full_scope = plan["scope"] == "full_main_id_recount"
    pass_main = full_scope and not blocks
    status = (
        "BLOCKED"
        if blocks
        else "PASS_MAIN_ID_RECOUNT"
        if full_scope
        else "PASS_ENGINEERING_PILOT"
    )
    pilot_bucket = int(plan["parent_inventory_fingerprint_sha256"][:16], 16) % 256
    total_materialize_seconds = sum(
        float(item["elapsed_seconds"]) for item in state["file_metrics"]
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    artifact_bytes = {
        name: int(path.stat().st_size)
        for name, path in {
            raw_path.name: raw_path,
            sorted_path.name: sorted_path,
        }.items()
        if name in artifact_hashes and path.is_file()
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "label": str(config["label"]),
        "status": status,
        "scope": plan["scope"],
        "pass_main_id_recount": pass_main,
        "pass_g1": False,
        "source_root_id": str(config["parent_inventory"]["source_root_id"]),
        "parent_manifest_sha256": plan["parent_manifest_sha256"],
        "parent_inventory_fingerprint_sha256": plan[
            "parent_inventory_fingerprint_sha256"
        ],
        "parent_stability_fingerprint_sha256": plan[
            "parent_stability_fingerprint_sha256"
        ],
        "plan_sha256": plan["plan_sha256"],
        "creator": {**git_metadata(), "tool_sha256": plan["tool_sha256"]},
        "started_at_utc": state["started_at_utc"],
        "finished_at_utc": utc_now(),
        "execution": {
            "selected_files": int(plan["selected_file_count"]),
            "batch_size": int(plan["batch_size"]),
            "materialize_file_seconds": total_materialize_seconds,
            "sort_count_and_checksum_seconds": sort_seconds,
            "current_invocation_elapsed_seconds": invocation_elapsed_seconds,
            "id_column_compressed_bytes": int(state["id_column_compressed_bytes"]),
            "spool_dtype": "uint64-little-endian",
            "artifact_bytes": artifact_bytes,
            "artifact_bytes_total": sum(artifact_bytes.values()),
            "peak_rss_kib": int(usage.ru_maxrss),
            "process_user_cpu_seconds": float(usage.ru_utime),
            "process_system_cpu_seconds": float(usage.ru_stime),
            "process_input_blocks": int(usage.ru_inblock),
            "process_output_blocks": int(usage.ru_oublock),
            "runtime": dict(plan["runtime"]),
            "path_redacted": True,
        },
        "counts": {
            "data_page_rows": int(state["written_rows"]),
            "null_ids": int(state["null_ids"]),
            "invalid_ids": int(state["invalid_ids"]),
            **statistics,
        },
        "diagnostic_buckets": {
            "algorithm": "splitmix64-v1",
            "seed": SPLITMIX64_SEED,
            "bucket_count": SPLITMIX64_BUCKETS,
            "histogram": [int(value) for value in state["bucket_histogram"]],
            "pilot_bucket": pilot_bucket,
            "pilot_bucket_rows": int(state["bucket_histogram"][pilot_bucket]),
            "hash_is_equality_key": False,
        },
        "invalid_samples": list(state["invalid_samples"]),
        "artifact_sha256": artifact_hashes,
        "checks": checks,
        "pending_after_main_id_recount": list(config["pending_after_main_id_recount"]),
        "claim_boundary": str(config["claim_boundary"]),
    }


def run_recount(
    *,
    config_path: Path,
    parent_path: Path,
    root: Path,
    output_dir: Path,
    max_files: int | None,
    batch_size: int | None,
    pause_after_new_files: int | None,
) -> dict[str, Any]:
    invocation_started = time.perf_counter()
    config = load_config(config_path)
    parent, parent_sha256, all_main_records = load_and_validate_parent(
        path=parent_path,
        root=root,
        config=config,
    )
    if max_files is not None and not 1 <= max_files < len(all_main_records):
        raise RecountError("--max-files must select at least one but not all files")
    records = all_main_records if max_files is None else all_main_records[:max_files]
    selected_batch_size = int(batch_size or config["main_id_recount"]["batch_size"])
    if selected_batch_size < 1:
        raise RecountError("batch size must be positive")
    plan = plan_payload(
        config=config,
        config_path=config_path,
        parent_sha256=parent_sha256,
        parent_manifest=parent,
        records=all_main_records,
        max_files=max_files,
        batch_size=selected_batch_size,
    )
    validate_output_directory_path(output_dir)
    validate_output_containment(root=root, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_output_directory_path(output_dir)

    lock_path = output_dir / "recount.lock"
    plan_path = output_dir / "recount_plan.json"
    state_path = output_dir / "recount_state.json"
    raw_path = output_dir / "main_ids.unsorted.u64"
    sorted_path = output_dir / "main_ids.sorted.u64"
    report_path = output_dir / "main_id_recount.json"
    markdown_path = output_dir / "main_id_recount.md"
    selected_rows = int(plan["selected_footer_rows"])
    atomic_targets = (plan_path, state_path, report_path, markdown_path)
    managed_paths = (
        lock_path,
        plan_path,
        state_path,
        raw_path,
        sorted_path,
        report_path,
        markdown_path,
        *(path.with_suffix(path.suffix + ".partial") for path in atomic_targets),
    )
    for path in managed_paths:
        validate_managed_output_path(path)

    validate_managed_output_path(lock_path)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RecountError("another recount process holds the output lock") from exc

        if plan_path.exists():
            frozen_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            claimed_plan_sha256 = frozen_plan.pop("plan_sha256", None)
            observed_plan_sha256 = sha256_bytes(canonical_json_bytes(frozen_plan))
            frozen_plan["plan_sha256"] = claimed_plan_sha256
            if claimed_plan_sha256 != observed_plan_sha256 or frozen_plan != plan:
                raise RecountError("output directory contains a different recount plan")
        else:
            atomic_write_json(plan_path, plan)

        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            validate_state(state, plan)
            if state.get("phase") == "complete" and report_path.exists():
                validate_checkpoint_metrics(
                    records=records,
                    state=state,
                    plan=plan,
                    column=str(config["main_id_recount"]["column"]),
                )
                return validate_complete_outputs(
                    state=state,
                    report_path=report_path,
                    markdown_path=markdown_path,
                    raw_path=raw_path,
                    sorted_path=sorted_path,
                )
            if not raw_path.is_file() or raw_path.stat().st_size != selected_rows * 8:
                raise RecountError("resume spool is missing or has the wrong size")
            state["status"] = "RUNNING"
            update_state(state_path, state)
        else:
            stale_outputs = [
                path.name
                for path in (raw_path, sorted_path, report_path, markdown_path)
                if path.exists()
            ]
            if stale_outputs:
                raise RecountError(
                    "state is missing but recount outputs already exist: "
                    f"{stale_outputs}"
                )
            capacity = ensure_capacity(output_dir, selected_rows)
            preallocate(raw_path, selected_rows * UINT64_BYTES)
            state = create_state(plan, capacity)
            update_state(state_path, state)

        if state["phase"] == "materialize":
            finished = materialize_ids(
                root=root,
                records=records,
                plan=plan,
                config=config,
                state=state,
                state_path=state_path,
                spool_path=raw_path,
                pause_after_new_files=pause_after_new_files,
            )
            if not finished:
                return {
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "status": "PAUSED",
                    "scope": plan["scope"],
                    "pass_main_id_recount": False,
                    "pass_g1": False,
                    "completed_files": int(state["completed_files"]),
                    "selected_files": len(records),
                    "written_rows": int(state["written_rows"]),
                    "claim_boundary": str(config["claim_boundary"]),
                }

        validate_checkpoint_metrics(
            records=records,
            state=state,
            plan=plan,
            column=str(config["main_id_recount"]["column"]),
        )
        if state["phase"] == "sort":
            if int(state["null_ids"]) or int(state["invalid_ids"]):
                statistics = unavailable_exact_statistics()
                artifact_hashes = {raw_path.name: sha256_file(raw_path)}
                sort_seconds = 0.0
                state["phase"] = "report"
                state["artifact_sha256"] = artifact_hashes
                state["exact_statistics"] = statistics
                state["sort_count_and_checksum_seconds"] = sort_seconds
                update_state(state_path, state)
            else:
                statistics, artifact_hashes, sort_seconds = sort_and_count(
                    state=state,
                    state_path=state_path,
                    raw_path=raw_path,
                    sorted_path=sorted_path,
                    rows=selected_rows,
                    chunk_rows=int(
                        config["main_id_recount"]["duplicate_scan_chunk_rows"]
                    ),
                    records=records,
                    plan=plan,
                    column=str(config["main_id_recount"]["column"]),
                )
        elif state["phase"] == "report":
            statistics = dict(state["exact_statistics"])
            artifact_hashes = dict(state["artifact_sha256"])
            sort_seconds = float(state["sort_count_and_checksum_seconds"])
            validate_artifact_hashes(
                state=state,
                raw_path=raw_path,
                sorted_path=sorted_path,
            )
        else:
            raise RecountError(f"unsupported recount phase: {state['phase']}")
        try:
            end_parent, end_parent_sha256, _ = load_and_validate_parent(
                path=parent_path,
                root=root,
                config=config,
            )
            parent_stable_after = (
                end_parent_sha256 == plan["parent_manifest_sha256"]
                and end_parent["root"]["stability_fingerprint_sha256"]
                == plan["parent_stability_fingerprint_sha256"]
            )
        except RecountError:
            parent_stable_after = False
        report = build_report(
            config=config,
            plan=plan,
            state=state,
            statistics=statistics,
            artifact_hashes=artifact_hashes,
            sort_seconds=sort_seconds,
            parent_stable_after=parent_stable_after,
            invocation_elapsed_seconds=time.perf_counter() - invocation_started,
            raw_path=raw_path,
            sorted_path=sorted_path,
        )
        atomic_write_json(report_path, report)
        atomic_write_text(markdown_path, render_markdown(report))
        state["phase"] = "complete"
        state["status"] = report["status"]
        state["report_sha256"] = sha256_file(report_path)
        state["markdown_sha256"] = sha256_file(markdown_path)
        update_state(state_path, state)
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--inventory-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--pause-after-new-files", type=int)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.pause_after_new_files is not None and args.pause_after_new_files < 1:
        parser.error("--pause-after-new-files must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        config_path = args.config.expanduser().resolve()
        config = load_config(config_path)
        root = resolve_root(config, args.root)
        report = run_recount(
            config_path=config_path,
            parent_path=args.inventory_manifest.expanduser().resolve(),
            root=root,
            output_dir=args.output_dir.expanduser(),
            max_files=args.max_files,
            batch_size=args.batch_size,
            pause_after_new_files=args.pause_after_new_files,
        )
    except (OSError, ValueError, RecountError, json.JSONDecodeError) as exc:
        print(f"recount failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "scope": report["scope"],
                "pass_main_id_recount": report["pass_main_id_recount"],
                "pass_g1": report["pass_g1"],
                "rows": report.get("counts", {}).get(
                    "data_page_rows", report.get("written_rows")
                ),
            },
            sort_keys=True,
        )
    )
    if args.require_pass and not report["pass_main_id_recount"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
