#!/usr/bin/env python3
"""Independently verify the bounded downstream stress benchmark read-only.

This verifier imports neither the stress producer, its benchmark helper, nor the
production work-time-spine runner.  It reconstructs the synthetic q generator,
spools, sorted runs, Parquet payloads, checkpoints, parent bindings, and retained
scratch topology using independent NumPy/PyArrow code.  It creates no output file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, NoReturn, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SCHEMA = "openalex_g1_canonical_work_time_spine_downstream_stress_benchmark.v1"
STATUS = (
    "PARTIAL_FULL_LIKE_PARTITION_AND_BALANCED_MULTI_RUN_STRESS_"
    "OPERATOR_LOCAL_NONPROMOTABLE"
)
VERIFICATION_SCHEMA = (
    "openalex_g1_canonical_work_time_spine_downstream_stress_verification.v1"
)
VERIFICATION_STATUS = (
    "PASS_READ_ONLY_INDEPENDENT_BOUNDED_DOWNSTREAM_STRESS_VERIFICATION_"
    "NONPRODUCTION"
)
PRODUCTION_NOT_RUN = "PRODUCTION_NOT_RUN"
STRESS_TOOL_SHA256 = "19bae4881bce030876d884a96500936d3b5a96ef97ccfffe28d7a3d605154b44"
HELPER_SHA256 = "c7809ac74425a837e2b5f58c021f46c297ba0cc7cedd4d60ecf0ad969c6332ff"
RUNNER_SHA256 = "bf9ed534b2e9f9c2d6c9851be7fde298eb372d8a3cf5e17363ea6afe51cf1280"
CONFIG_SHA256 = "6e8b74ddd12cbef46c8d15d3560f0d54f470b5017685fc1bce2389250cde0484"
PILOT_VERIFIER_SHA256 = "2cc47587ecd70bd1ae169f1f987828f024d6b8f7dd217e2e9b7964fabfcd5ede"
FIRST_BENCHMARK_SHA256 = "6fd87a6aec825b690803a36d60ae271dd9fa7a691563fbb1b18b2e5a07649109"
FIRST_BENCHMARK_FINGERPRINT = "5e12129cafbda0bc5918eedeb43c3ec6352357f3e78e67b0d6d6494f7a71e287"
PILOT_RECEIPT_SHA256 = "eef08252fc6147084cb50752a456541b408992d599d6d80d2d5806b48310b099"
PILOT_STAGING_SHA256 = "ccbe4ffe751d148076c1711838aa15b976dfb305da4a0b28ef4937743b2a1165"
EXPECTED_STRESS_RESULT_SHA256 = (
    "f93960649ff2742cc1b4f7efca82fd88b86c85c1ef5066f066d2507f9a766e46"
)
EXPECTED_STRESS_RESULT_FINGERPRINT = (
    "7f6ef339b7a7c64702a9b8dcb31d09d75439d90bc45c576e6cca43603c69960e"
)
PILOT_SCOPE = "PRODUCTION_SOURCE_PREFIX_PILOT_UNPROMOTABLE"
PILOT_STATUS = "PILOT_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_NONPROMOTABLE"

REPEAT_COUNT = 3
PARTITION_COUNT = 4096
PARTITION_ID = 4095
SORT_CHUNK_ROWS = 1_048_576
PILOT_STAGING_ROWS = 1_000_000
SYNTHETIC_NAMESPACE_BASE = 1 << 63
GENERATION_BATCH_ROWS = 262_144
PARQUET_ROW_GROUP_ROWS = 262_144
PARQUET_FORMAT_VERSION = "2.6"
PARQUET_COLUMN_ENCODINGS = ("RLE", "PLAIN")
RECORD_BYTES = 16
NULL_DAY = -(2**31)
RECORD_HASH_DOMAIN = b"openalex-work-time-records-v1"
MAX_JSON_BYTES = 16 << 20
DISPOSITION_NAMES = {
    0: "BOTH_NULL_EXCLUDED",
    1: "VALID_PRE_1900_EXCLUDED",
    2: "USABLE_1900_BEFORE_SNAPSHOT_YEAR",
    3: "USABLE_SNAPSHOT_YEAR_ON_OR_BEFORE_SNAPSHOT",
    4: "VALID_SNAPSHOT_YEAR_AFTER_SNAPSHOT_EXCLUDED",
    5: "VALID_POST_SNAPSHOT_YEAR_EXCLUDED",
}
RECORD_DTYPE = np.dtype(
    [
        ("work_id_numeric", "<u8"),
        ("publication_day", "<i4"),
        ("publication_year", "<u2"),
        ("temporal_disposition", "u1"),
        ("reserved", "u1"),
    ],
    align=False,
)
FRAMED_DTYPE = np.dtype(
    [("encoded_length", "<u8"), ("record", f"V{RECORD_BYTES}")], align=False
)
ARROW_SCHEMA = pa.schema(
    [
        pa.field("work_id_numeric", pa.uint64(), nullable=False),
        pa.field("publication_day", pa.int32(), nullable=True),
        pa.field("publication_year", pa.uint16(), nullable=True),
        pa.field("temporal_disposition", pa.uint8(), nullable=False),
    ],
    metadata=None,
)
SCHEMA_CONTRACT = [
    {"name": field.name, "type": str(field.type), "nullable": field.nullable}
    for field in ARROW_SCHEMA
]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    rows: int
    expected_runs: int
    role: str


SCENARIOS = (
    Scenario("RMEAN", 120_189, 1, "FULL_LIKE_MEAN_PARTITION_ROWS"),
    Scenario("A2_BALANCED", 2_097_152, 2, "BALANCED_TWO_WAY_HEAP_MERGE"),
    Scenario("A8_BALANCED", 8_388_608, 8, "BALANCED_EIGHT_WAY_HEAP_MERGE"),
)

TOP_LEVEL_KEYS = {
    "schema_version", "status", "started_at", "completed_at",
    "implementation_binding", "runtime", "parent_bindings", "preflight",
    "scenario_contracts", "repeats", "aggregate",
    "full_like_per_partition_row_scale_measured",
    "balanced_multi_run_heap_branch_measured", "maximum_measured_fan_in",
    "theoretical_470_run_heap_merge_measured",
    "actual_full_4096_partition_occupancy_profile_measured",
    "full_byte_finalization_surface_measured", "independent_verification_status",
    "cache_condition", "independent_quiet_window_sessions", "source_hdd_opened",
    "production_execution_status", "production_component_pass", "pass_g1",
    "full_authorization_sufficient", "downstream_capacity_gate_passed",
    "programmatic_parent_of_full_receipt", "operator_acceptance_status",
    "strongest_permitted_interpretation", "operator_paths_redacted",
    "result_fingerprint_sha256",
}
SCENARIO_RESULT_KEYS = {
    "scenario_id", "role", "rows", "partition_id", "sort_chunk_rows",
    "expected_runs", "observed_runs", "maximum_measured_fan_in", "preparation",
    "run_validation", "sort", "merge_parquet_and_validation", "output_rows",
    "output_bytes", "output_sha256", "record_sha256", "integrity_status",
    "operator_paths_redacted",
}
CHECKPOINT_KEYS = {
    "schema_version", "task_fingerprint_sha256", "runtime_fingerprint",
    "scan_checkpoint_tree_sha256", "staging_shard_tree_sha256", "partition_id",
    "partition_formula", "output_relative_path", "output_rows",
    "minimum_work_id_numeric", "maximum_work_id_numeric", "output_schema",
    "output_schema_sha256", "pre_parquet_sorted_record_sha256",
    "decoded_parquet_record_sha256", "disposition_counts",
    "w_zero_disposition_counts", "publication_year_extrema",
    "nullability_counts", "output_bytes", "output_sha256",
    "strictly_increasing_ids", "operator_paths_redacted", "checkpoint_sha256",
}
RUNTIME_KEYS = {
    "python_version", "python_implementation", "platform_machine",
    "runner_runtime_fingerprint",
}
RUNNER_RUNTIME_KEYS = {
    "pyarrow_version", "parquet_cpp_version", "numpy_version",
    "python_implementation", "python_version", "platform_machine",
}
PREFLIGHT_KEYS = {
    "required_disk_bytes", "required_available_ram_bytes",
    "required_nofile_soft_limit", "available_disk_bytes",
    "available_ram_bytes", "nofile_soft_limit", "nofile_hard_limit",
    "observed_at", "bounded_preflight_passed", "production_preflight_passed",
    "source_drive_health_not_applicable_no_source_opened",
    "quiet_window_independently_verified", "operator_paths_redacted",
    "stress_kernel_minimum_free_disk_bytes",
    "stress_kernel_minimum_available_ram_bytes",
    "effective_required_free_disk_bytes",
    "effective_required_available_ram_bytes",
    "pinned_helper_preflight_is_enforced", "atomic_rename_fsync_probe",
}
ATOMIC_FSYNC_PROBE_KEYS = {"status", "wall_seconds"}
PREPARATION_KEYS = {
    "rows", "bytes", "wall_seconds", "spool_sha256",
    "synthetic_namespace_base", "partition_id", "temporal_payload_source",
    "excluded_from_kernel_timing",
}
RUN_VALIDATION_KEYS = {
    "run_count", "run_contract_tree_sha256", "runs", "balanced_fan_in",
    "wall_seconds", "excluded_from_kernel_timing",
    "prewarms_external_run_page_cache_before_merge",
}
SORT_PHASE_KEYS = {"wall_rows_per_second", "resources", "coarse_metrics"}
MERGE_PHASE_KEYS = {
    "wall_rows_per_second", "resources", "coarse_metrics", "cache_condition",
    "derived_heap_push_count_not_observed", "derived_heap_pop_count_not_observed",
}
RESOURCE_KEYS = {
    "wall_seconds", "process_usage_delta", "proc_io_delta",
    "sampling_interval_seconds", "peak_rss_kib_sampled",
    "peak_open_file_descriptors_sampled", "peak_hwm_kib_observed",
}
PROCESS_USAGE_KEYS = {
    "user_seconds", "system_seconds", "minor_page_faults", "major_page_faults",
    "voluntary_context_switches", "involuntary_context_switches",
    "block_inputs", "block_outputs",
}
PROC_IO_KEYS = {
    "rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes",
    "cancelled_write_bytes",
}
COARSE_METRIC_KEYS = {
    "numpy_stable_argsort", "fsync", "parquet_writer",
    "arrow_table_conversion", "decoded_validation", "parquet_sha256",
    "merge_and_parquet_before_decoded_validation_seconds",
    "heap_operation_timing_proxy_installed", "run_reader_proxy_installed",
}
ROW_RATE_KEYS = {"calls", "rows", "seconds", "rows_per_second"}
BYTE_RATE_KEYS = {"calls", "bytes", "seconds", "bytes_per_second"}
PARQUET_WRITER_KEYS = {
    "write_calls", "close_calls", "rows", "seconds", "rows_per_second",
}
FSYNC_KEYS = {"calls", "total_seconds", "latency_seconds"}
DISTRIBUTION_KEYS = {"count", "minimum", "median", "maximum"}
PILOT_DIRECTORY_PATHS = {
    ".", "checkpoints", "checkpoints/partition", "checkpoints/scan",
    "checkpoints/scan/main", "spine", "staging", "staging/main",
}


class StressVerificationError(RuntimeError):
    """The independent stress verification failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StressVerificationError(message)


def fail(message: str, cause: BaseException | None = None) -> NoReturn:
    if cause is None:
        raise StressVerificationError(message)
    raise StressVerificationError(message) from cause


def _mapping_with_exact_keys(
    value: Any, expected: set[str], field: str,
) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{field} is not a mapping")
    require(set(value) == expected, f"{field} key set differs")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    require(type(value) is int, f"{field} is not an integer")
    require(value >= minimum, f"{field} is below its minimum")
    return value


def _finite_float(value: Any, field: str, *, positive: bool = False) -> float:
    require(type(value) is float, f"{field} is not a float")
    require(math.isfinite(value), f"{field} is not finite")
    if positive:
        require(value > 0.0, f"{field} is not positive")
    else:
        require(value >= 0.0, f"{field} is negative")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    require(isinstance(value, str) and bool(value), f"{field} is not a string")
    return value


def _iso_timestamp(value: Any, field: str) -> datetime:
    text = _nonempty_string(value, field)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        fail(f"{field} is not ISO-8601", exc)
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None, f"{field} lacks timezone")
    require(parsed.isoformat(timespec="seconds") == text, f"{field} is not canonical to seconds")
    return parsed


def _validate_no_absolute_paths(value: Any, field: str = "stress result") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _validate_no_absolute_paths(nested, f"{field}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_no_absolute_paths(nested, f"{field}[{index}]")
        return
    if not isinstance(value, str):
        return
    windows_absolute = (
        len(value) >= 3 and value[1] == ":" and value[2] in ("/", "\\")
    )
    require(
        not value.startswith(("/", "file://")) and not windows_absolute,
        f"{field} contains an absolute operator path",
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _published_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_mode), int(info.st_nlink),
        int(info.st_uid), int(info.st_gid), int(info.st_size),
        int(info.st_mtime_ns), int(info.st_ctime_ns),
    )


def _assert_no_symlink_components(path: Path, field: str) -> None:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except OSError as exc:
            fail(f"cannot inspect {field}", exc)
        require(not stat.S_ISLNK(info.st_mode), f"{field} contains a symlink")


def _open_readonly(path: Path, field: str) -> tuple[int, tuple[int, ...]]:
    _assert_no_symlink_components(path, field)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        fail(f"cannot open {field} read-only", exc)
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), f"{field} is not regular")
        require(info.st_nlink == 1, f"{field} is hard-linked")
        require(stat.S_IMODE(info.st_mode) == 0o600, f"{field} mode is not 0600")
        require(info.st_uid == os.geteuid(), f"{field} owner differs")
        return descriptor, _identity(info)
    except BaseException:
        os.close(descriptor)
        raise


def _validate_open(path: Path, descriptor: int, identity: tuple[int, ...], field: str) -> None:
    require(_identity(os.fstat(descriptor)) == identity, f"open {field} changed")
    require(_identity(path.lstat()) == identity, f"{field} path identity changed")


def _read_exact(descriptor: int, size: int, field: str) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        block = os.read(descriptor, size - len(payload))
        require(bool(block), f"{field} ended early")
        payload.extend(block)
    return bytes(payload)


def _read_bytes(path: Path, field: str) -> tuple[bytes, str]:
    descriptor, identity = _open_readonly(path, field)
    blocks: list[bytes] = []
    digest = hashlib.sha256()
    try:
        while block := os.read(descriptor, 1 << 20):
            blocks.append(block)
            digest.update(block)
        _validate_open(path, descriptor, identity, field)
    finally:
        os.close(descriptor)
    return b"".join(blocks), digest.hexdigest()


def _hash_file(path: Path, field: str) -> tuple[str, int]:
    descriptor, identity = _open_readonly(path, field)
    digest = hashlib.sha256()
    try:
        while block := os.read(descriptor, 16 << 20):
            digest.update(block)
        _validate_open(path, descriptor, identity, field)
    finally:
        os.close(descriptor)
    return digest.hexdigest(), identity[6]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"JSON repeats key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, field: str) -> tuple[dict[str, Any], str]:
    payload, digest = _read_bytes(path, field)
    require(0 < len(payload) <= MAX_JSON_BYTES, f"{field} size is outside its bound")
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_strict_object,
            parse_constant=lambda token: fail(f"{field} contains {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot parse {field}", exc)
    require(isinstance(value, dict), f"{field} is not a mapping")
    require(payload == _published_json_bytes(value), f"{field} is not canonical JSON")
    return dict(value), digest


def _snapshot_tree(root: Path, field: str) -> dict[str, Any]:
    _assert_no_symlink_components(root, field)
    root_info = root.lstat()
    require(stat.S_ISDIR(root_info.st_mode), f"{field} is not a directory")
    items: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    regular_bytes = 0
    directory_count = 1
    root_item = {"relative_path": ".", "kind": "directory", "identity": list(_identity(root_info))}
    items.append(root_item)
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), f"{field} contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            kind = "directory"
            directory_count += 1
            item = {"relative_path": relative, "kind": kind, "identity": list(_identity(info))}
        else:
            require(stat.S_ISREG(info.st_mode), f"{field} contains a special entry")
            digest, size = _hash_file(path, f"{field} artifact {relative}")
            item = {"relative_path": relative, "kind": "file", "identity": list(_identity(info)), "sha256": digest}
            contracts.append({"relative_path": relative, "bytes": size, "sha256": digest})
            regular_bytes += size
        items.append(item)
    return {
        "tree_identity_and_content_sha256": _sha256_bytes(_canonical_json_bytes(items)),
        "artifact_contract_tree_sha256": _sha256_bytes(_canonical_json_bytes(contracts)),
        "regular_file_count": len(contracts),
        "regular_file_bytes": regular_bytes,
        "directory_count_including_root": directory_count,
    }


def _result_fingerprint(result: Mapping[str, Any]) -> str:
    core = dict(result)
    core.pop("result_fingerprint_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(core))


def _pilot_report_fingerprint(report: Mapping[str, Any]) -> str:
    core = copy.deepcopy(dict(report))
    provenance = core.get("provenance")
    require(isinstance(provenance, dict), "pilot provenance differs")
    provenance.pop("result_fingerprint_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(core))


def _validate_record_array(records: np.ndarray[Any, Any]) -> None:
    require(records.dtype == RECORD_DTYPE, "record dtype differs")
    if not len(records):
        return
    dispositions = records["temporal_disposition"]
    require(bool(np.all(records["reserved"] == 0)), "reserved byte differs")
    require(bool(np.all(dispositions <= max(DISPOSITION_NAMES))), "disposition code differs")
    nulls = dispositions == 0
    require(bool(np.all(records["publication_day"][nulls] == NULL_DAY)), "null-day sentinel differs")
    require(bool(np.all(records["publication_year"][nulls] == 0)), "null-year sentinel differs")
    require(bool(np.all(records["publication_day"][~nulls] != NULL_DAY)), "nonnull day uses sentinel")
    require(bool(np.all(records["publication_year"][~nulls] != 0)), "nonnull year uses sentinel")


def _new_record_digest(rows: int) -> hashlib._Hash:
    digest = hashlib.sha256()
    digest.update(len(RECORD_HASH_DOMAIN).to_bytes(8, "little"))
    digest.update(RECORD_HASH_DOMAIN)
    digest.update(rows.to_bytes(8, "little"))
    return digest


def _update_record_digest(digest: Any, records: np.ndarray[Any, Any]) -> None:
    framed = np.empty(len(records), dtype=FRAMED_DTYPE)
    framed["encoded_length"] = RECORD_BYTES
    framed["record"] = records.view(f"V{RECORD_BYTES}").reshape(-1)
    digest.update(memoryview(framed).cast("B"))


def _validate_temporal_binding(records: np.ndarray[Any, Any], pilot: np.ndarray[Any, Any], q_values: np.ndarray[Any, Any]) -> None:
    _validate_record_array(records)
    expected = pilot[(q_values % len(pilot)).astype(np.int64, copy=False)]
    for field in ("publication_day", "publication_year", "temporal_disposition", "reserved"):
        require(bool(np.array_equal(records[field], expected[field])), f"pilot temporal binding differs: {field}")


def _validate_first_benchmark(path: Path) -> tuple[dict[str, Any], str]:
    result, digest = _read_json(path, "first downstream benchmark")
    require(digest == FIRST_BENCHMARK_SHA256, "first benchmark SHA-256 differs")
    require(result.get("schema_version") == "openalex_g1_canonical_work_time_spine_downstream_benchmark.v1", "first benchmark schema differs")
    require(result.get("result_fingerprint_sha256") == _result_fingerprint(result) == FIRST_BENCHMARK_FINGERPRINT, "first benchmark fingerprint differs")
    for key in ("production_component_pass", "pass_g1", "full_authorization_sufficient", "downstream_capacity_gate_passed", "programmatic_parent_of_full_receipt"):
        require(result.get(key) is False, f"first benchmark claim differs: {key}")
    require(result.get("production_execution_status") == PRODUCTION_NOT_RUN, "first benchmark claims production")
    pilot_binding = result.get("pilot_binding")
    require(isinstance(pilot_binding, dict), "first benchmark pilot binding differs")
    require(pilot_binding.get("receipt_sha256") == PILOT_RECEIPT_SHA256, "first benchmark pilot receipt differs")
    require(pilot_binding.get("staging_sha256") == PILOT_STAGING_SHA256, "first benchmark pilot staging differs")
    return result, digest


def _validate_pilot(root: Path) -> tuple[np.ndarray[Any, Any], str]:
    root_info = root.lstat()
    require(stat.S_ISDIR(root_info.st_mode) and stat.S_IMODE(root_info.st_mode) == 0o700, "pilot root is not private")
    files: set[str] = set()
    directories: set[str] = {"."}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), "pilot tree contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            require(stat.S_IMODE(info.st_mode) == 0o700, "pilot directory mode differs")
            directories.add(relative)
        else:
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and stat.S_IMODE(info.st_mode) == 0o600, "pilot file contract differs")
            files.add(relative)
    require(directories == PILOT_DIRECTORY_PATHS, "pilot directory topology differs")
    report, _ = _read_json(root / "work_time_spine_pilot.json", "pilot report")
    manifest, _ = _read_json(root / "staging_manifest.json", "pilot staging manifest")
    receipt, receipt_sha = _read_json(root / "work_time_spine_pilot.outputs.json", "pilot receipt")
    require(receipt_sha == PILOT_RECEIPT_SHA256, "pilot receipt SHA-256 differs")
    require(report.get("status") == PILOT_STATUS, "pilot status differs")
    scope = report.get("scope")
    require(isinstance(scope, dict) and scope.get("execution_scope") == PILOT_SCOPE, "pilot scope differs")
    require(scope.get("production_status") == PRODUCTION_NOT_RUN and scope.get("retained_rows") == PILOT_STAGING_ROWS, "pilot scope accounting differs")
    provenance = report.get("provenance")
    require(isinstance(provenance, dict), "pilot provenance differs")
    require(provenance.get("tool_sha256") == RUNNER_SHA256 and provenance.get("config_sha256") == CONFIG_SHA256, "pilot implementation binding differs")
    require(provenance.get("result_fingerprint_sha256") == _pilot_report_fingerprint(report), "pilot result fingerprint differs")
    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, dict) and receipt.get("artifact_count") == len(artifacts), "pilot receipt artifacts differ")
    for relative, contract in artifacts.items():
        require(isinstance(relative, str) and not relative.startswith("/") and ".." not in Path(relative).parts, "pilot artifact path differs")
        digest, size = _hash_file(root / relative, f"pilot artifact {relative}")
        require(contract == {"relative_path": relative, "bytes": size, "sha256": digest}, f"pilot artifact differs: {relative}")
    require(files == set(artifacts) | {"audit.lock", "work_time_spine_pilot.outputs.json"}, "pilot file topology differs")
    lock_sha, lock_size = _hash_file(root / "audit.lock", "pilot audit lock")
    require(lock_size == 0 and lock_sha == _sha256_bytes(b""), "pilot audit lock differs")
    shards = manifest.get("shards")
    require(isinstance(shards, list) and len(shards) == 1 and isinstance(shards[0], dict), "pilot staging shard contract differs")
    shard = shards[0]
    relative = shard.get("relative_path")
    require(isinstance(relative, str) and not relative.startswith("/") and ".." not in Path(relative).parts, "pilot staging path differs")
    payload, staging_sha = _read_bytes(root / relative, "pilot staging payload")
    require(staging_sha == shard.get("sha256") == PILOT_STAGING_SHA256, "pilot staging SHA-256 differs")
    require(len(payload) == shard.get("bytes") == PILOT_STAGING_ROWS * RECORD_BYTES, "pilot staging bytes differ")
    require(shard.get("rows") == PILOT_STAGING_ROWS, "pilot staging rows differ")
    records = np.frombuffer(payload, dtype=RECORD_DTYPE).copy()
    _validate_record_array(records)
    return records, str(report.get("task_fingerprint_sha256"))


def _validate_runtime(value: Any) -> None:
    runtime = _mapping_with_exact_keys(value, RUNTIME_KEYS, "runtime")
    runner = _mapping_with_exact_keys(
        runtime["runner_runtime_fingerprint"],
        RUNNER_RUNTIME_KEYS,
        "runner runtime fingerprint",
    )
    for key in ("python_version", "python_implementation", "platform_machine"):
        observed = _nonempty_string(runtime[key], f"runtime.{key}")
        require(observed == runner[key], f"runtime.{key} differs from runner")
    for key in RUNNER_RUNTIME_KEYS:
        _nonempty_string(runner[key], f"runner runtime fingerprint.{key}")


def _validate_preflight(value: Any, *, started_at: datetime) -> None:
    preflight = _mapping_with_exact_keys(value, PREFLIGHT_KEYS, "preflight")
    fixed = {
        "required_disk_bytes": 36_366_736_928,
        "required_available_ram_bytes": 9_685_696_512,
        "required_nofile_soft_limit": 8_192,
        "bounded_preflight_passed": True,
        "production_preflight_passed": False,
        "source_drive_health_not_applicable_no_source_opened": True,
        "quiet_window_independently_verified": False,
        "operator_paths_redacted": True,
        "stress_kernel_minimum_free_disk_bytes": 4_294_967_296,
        "stress_kernel_minimum_available_ram_bytes": 2_147_483_648,
        "effective_required_free_disk_bytes": 36_366_736_928,
        "effective_required_available_ram_bytes": 9_685_696_512,
        "pinned_helper_preflight_is_enforced": True,
    }
    for key, expected in fixed.items():
        if type(expected) is bool:
            require(preflight[key] is expected, f"preflight field differs: {key}")
        else:
            require(preflight[key] == expected, f"preflight field differs: {key}")
    integer_keys = (
        "required_disk_bytes", "required_available_ram_bytes",
        "required_nofile_soft_limit", "available_disk_bytes",
        "available_ram_bytes", "nofile_soft_limit", "nofile_hard_limit",
        "stress_kernel_minimum_free_disk_bytes",
        "stress_kernel_minimum_available_ram_bytes",
        "effective_required_free_disk_bytes",
        "effective_required_available_ram_bytes",
    )
    for key in integer_keys:
        _integer(preflight[key], f"preflight.{key}", minimum=1)
    require(
        preflight["effective_required_free_disk_bytes"]
        == max(
            preflight["required_disk_bytes"],
            preflight["stress_kernel_minimum_free_disk_bytes"],
        ),
        "effective disk requirement differs",
    )
    require(
        preflight["effective_required_available_ram_bytes"]
        == max(
            preflight["required_available_ram_bytes"],
            preflight["stress_kernel_minimum_available_ram_bytes"],
        ),
        "effective RAM requirement differs",
    )
    require(
        preflight["available_disk_bytes"]
        >= preflight["effective_required_free_disk_bytes"],
        "available disk does not satisfy bounded preflight",
    )
    require(
        preflight["available_ram_bytes"]
        >= preflight["effective_required_available_ram_bytes"],
        "available RAM does not satisfy bounded preflight",
    )
    require(
        preflight["nofile_hard_limit"] >= preflight["nofile_soft_limit"]
        >= preflight["required_nofile_soft_limit"],
        "nofile limits do not satisfy bounded preflight",
    )
    observed_at = _iso_timestamp(preflight["observed_at"], "preflight.observed_at")
    require(observed_at <= started_at, "preflight observation follows benchmark start")
    probe = _mapping_with_exact_keys(
        preflight["atomic_rename_fsync_probe"],
        ATOMIC_FSYNC_PROBE_KEYS,
        "preflight atomic fsync probe",
    )
    require(
        probe["status"] == "PASS_BOUNDED_SCRATCH_ATOMIC_RENAME_FSYNC",
        "preflight fsync probe differs",
    )
    _finite_float(probe["wall_seconds"], "preflight fsync probe wall_seconds", positive=True)


def _validate_throughput_metric(
    value: Any,
    *,
    field: str,
    expected_calls: int,
    expected_units: int,
    units_key: str,
    rate_key: str,
) -> float:
    keys = ROW_RATE_KEYS if units_key == "rows" else BYTE_RATE_KEYS
    metric = _mapping_with_exact_keys(value, keys, field)
    require(
        _integer(metric["calls"], f"{field}.calls") == expected_calls,
        f"{field}.calls differs",
    )
    require(
        _integer(metric[units_key], f"{field}.{units_key}") == expected_units,
        f"{field}.{units_key} differs",
    )
    seconds = _finite_float(metric["seconds"], f"{field}.seconds")
    if expected_calls == 0:
        require(
            expected_units == 0 and seconds == 0.0 and metric[rate_key] is None,
            f"{field} zero-work contract differs",
        )
        return seconds
    require(seconds > 0.0, f"{field}.seconds is not positive")
    rate = _finite_float(metric[rate_key], f"{field}.{rate_key}", positive=True)
    implied_seconds = expected_units / rate
    require(
        abs(implied_seconds - seconds) <= 0.500001e-6,
        f"{field} rate does not agree with rounded seconds",
    )
    return seconds


def _validate_parquet_writer_metric(
    value: Any,
    *,
    field: str,
    expected_write_calls: int,
    expected_close_calls: int,
    expected_rows: int,
) -> float:
    metric = _mapping_with_exact_keys(value, PARQUET_WRITER_KEYS, field)
    require(
        _integer(metric["write_calls"], f"{field}.write_calls")
        == expected_write_calls,
        f"{field}.write_calls differs",
    )
    require(
        _integer(metric["close_calls"], f"{field}.close_calls")
        == expected_close_calls,
        f"{field}.close_calls differs",
    )
    require(
        _integer(metric["rows"], f"{field}.rows") == expected_rows,
        f"{field}.rows differs",
    )
    seconds = _finite_float(metric["seconds"], f"{field}.seconds")
    if expected_write_calls == 0:
        require(
            expected_rows == 0 and seconds == 0.0
            and metric["rows_per_second"] is None,
            f"{field} zero-work contract differs",
        )
        return seconds
    require(seconds > 0.0, f"{field}.seconds is not positive")
    rate = _finite_float(
        metric["rows_per_second"], f"{field}.rows_per_second", positive=True,
    )
    require(
        abs(expected_rows / rate - seconds) <= 0.500001e-6,
        f"{field} rate does not agree with rounded seconds",
    )
    return seconds


def _validate_fsync_metric(value: Any, *, field: str, expected_calls: int) -> None:
    metric = _mapping_with_exact_keys(value, FSYNC_KEYS, field)
    calls = _integer(metric["calls"], f"{field}.calls")
    require(calls == expected_calls, f"{field}.calls differs")
    total = _finite_float(metric["total_seconds"], f"{field}.total_seconds")
    distribution = _mapping_with_exact_keys(
        metric["latency_seconds"], DISTRIBUTION_KEYS, f"{field}.latency_seconds",
    )
    require(
        _integer(distribution["count"], f"{field}.latency_seconds.count") == calls,
        f"{field} latency count differs",
    )
    if calls == 0:
        require(
            total == 0.0
            and distribution["minimum"] is None
            and distribution["median"] is None
            and distribution["maximum"] is None,
            f"{field} empty distribution differs",
        )
        return
    minimum = _finite_float(
        distribution["minimum"], f"{field}.latency_seconds.minimum", positive=True,
    )
    median = _finite_float(
        distribution["median"], f"{field}.latency_seconds.median", positive=True,
    )
    maximum = _finite_float(
        distribution["maximum"], f"{field}.latency_seconds.maximum", positive=True,
    )
    require(minimum <= median <= maximum, f"{field} latency ordering differs")
    tolerance = (calls + 1) * 0.500001e-6
    require(
        calls * minimum - tolerance <= total <= calls * maximum + tolerance,
        f"{field} total latency is outside distribution bounds",
    )


def _validate_resources(value: Any, *, field: str) -> float:
    resources = _mapping_with_exact_keys(value, RESOURCE_KEYS, field)
    wall = _finite_float(resources["wall_seconds"], f"{field}.wall_seconds", positive=True)
    _finite_float(
        resources["sampling_interval_seconds"],
        f"{field}.sampling_interval_seconds",
        positive=True,
    )
    rss = _integer(resources["peak_rss_kib_sampled"], f"{field}.peak_rss_kib_sampled", minimum=1)
    hwm = _integer(resources["peak_hwm_kib_observed"], f"{field}.peak_hwm_kib_observed", minimum=1)
    _integer(
        resources["peak_open_file_descriptors_sampled"],
        f"{field}.peak_open_file_descriptors_sampled",
        minimum=1,
    )
    require(hwm >= rss, f"{field} peak HWM is below sampled RSS")
    usage = _mapping_with_exact_keys(
        resources["process_usage_delta"], PROCESS_USAGE_KEYS,
        f"{field}.process_usage_delta",
    )
    for key in ("user_seconds", "system_seconds"):
        _finite_float(usage[key], f"{field}.process_usage_delta.{key}")
    for key in PROCESS_USAGE_KEYS - {"user_seconds", "system_seconds"}:
        _integer(usage[key], f"{field}.process_usage_delta.{key}")
    io = _mapping_with_exact_keys(
        resources["proc_io_delta"], PROC_IO_KEYS, f"{field}.proc_io_delta",
    )
    for key in PROC_IO_KEYS:
        _integer(io[key], f"{field}.proc_io_delta.{key}")
    return wall


def _validate_coarse_metrics(
    value: Any,
    *,
    field: str,
    phase: str,
    scenario: Scenario,
    output_bytes: int,
    phase_wall_seconds: float,
) -> None:
    metrics = _mapping_with_exact_keys(value, COARSE_METRIC_KEYS, field)
    require(
        metrics["heap_operation_timing_proxy_installed"] is False
        and metrics["run_reader_proxy_installed"] is False,
        f"{field} proxy boundary differs",
    )
    row_groups = len(_expected_parquet_row_group_rows(scenario.rows))
    if phase == "sort":
        _validate_throughput_metric(
            metrics["numpy_stable_argsort"], field=f"{field}.numpy_stable_argsort",
            expected_calls=scenario.expected_runs, expected_units=scenario.rows,
            units_key="rows", rate_key="rows_per_second",
        )
        _validate_fsync_metric(
            metrics["fsync"], field=f"{field}.fsync",
            expected_calls=scenario.expected_runs + 1,
        )
        _validate_parquet_writer_metric(
            metrics["parquet_writer"], field=f"{field}.parquet_writer",
            expected_write_calls=0, expected_close_calls=0, expected_rows=0,
        )
        for key in ("arrow_table_conversion", "decoded_validation"):
            _validate_throughput_metric(
                metrics[key], field=f"{field}.{key}", expected_calls=0,
                expected_units=0, units_key="rows", rate_key="rows_per_second",
            )
        _validate_throughput_metric(
            metrics["parquet_sha256"], field=f"{field}.parquet_sha256",
            expected_calls=0, expected_units=0, units_key="bytes",
            rate_key="bytes_per_second",
        )
        require(
            metrics["merge_and_parquet_before_decoded_validation_seconds"] is None,
            f"{field} pre-validation timing differs",
        )
        return
    require(phase == "merge", "unknown coarse-metric phase")
    _validate_throughput_metric(
        metrics["numpy_stable_argsort"], field=f"{field}.numpy_stable_argsort",
        expected_calls=0, expected_units=0, units_key="rows",
        rate_key="rows_per_second",
    )
    _validate_fsync_metric(metrics["fsync"], field=f"{field}.fsync", expected_calls=1)
    _validate_parquet_writer_metric(
        metrics["parquet_writer"], field=f"{field}.parquet_writer",
        expected_write_calls=row_groups, expected_close_calls=1,
        expected_rows=scenario.rows,
    )
    _validate_throughput_metric(
        metrics["arrow_table_conversion"],
        field=f"{field}.arrow_table_conversion", expected_calls=row_groups,
        expected_units=scenario.rows, units_key="rows", rate_key="rows_per_second",
    )
    _validate_throughput_metric(
        metrics["decoded_validation"], field=f"{field}.decoded_validation",
        expected_calls=1, expected_units=scenario.rows, units_key="rows",
        rate_key="rows_per_second",
    )
    _validate_throughput_metric(
        metrics["parquet_sha256"], field=f"{field}.parquet_sha256",
        expected_calls=1, expected_units=output_bytes, units_key="bytes",
        rate_key="bytes_per_second",
    )
    pre_validation = _finite_float(
        metrics["merge_and_parquet_before_decoded_validation_seconds"],
        f"{field}.merge_and_parquet_before_decoded_validation_seconds",
        positive=True,
    )
    require(
        pre_validation <= phase_wall_seconds,
        f"{field} pre-validation timing exceeds phase wall time",
    )


def _validate_kernel_phase(
    value: Any,
    *,
    phase: str,
    scenario: Scenario,
    output_bytes: int,
) -> None:
    expected_keys = SORT_PHASE_KEYS if phase == "sort" else MERGE_PHASE_KEYS
    phase_result = _mapping_with_exact_keys(value, expected_keys, f"{scenario.scenario_id} {phase}")
    wall = _validate_resources(
        phase_result["resources"], field=f"{scenario.scenario_id} {phase}.resources",
    )
    rate = _finite_float(
        phase_result["wall_rows_per_second"],
        f"{scenario.scenario_id} {phase}.wall_rows_per_second",
        positive=True,
    )
    require(
        rate == round(scenario.rows / wall, 6),
        f"{scenario.scenario_id} {phase} rate does not reconstruct",
    )
    _validate_coarse_metrics(
        phase_result["coarse_metrics"],
        field=f"{scenario.scenario_id} {phase}.coarse_metrics", phase=phase,
        scenario=scenario, output_bytes=output_bytes, phase_wall_seconds=wall,
    )


def _validate_result_contract(result: Mapping[str, Any]) -> None:
    require(set(result) == TOP_LEVEL_KEYS, "stress result key set differs")
    require(result.get("schema_version") == SCHEMA and result.get("status") == STATUS, "stress result schema/status differs")
    require(
        result.get("result_fingerprint_sha256")
        == _result_fingerprint(result)
        == EXPECTED_STRESS_RESULT_FINGERPRINT,
        "stress result fingerprint differs",
    )
    _validate_no_absolute_paths(result)
    started_at = _iso_timestamp(result.get("started_at"), "stress result started_at")
    completed_at = _iso_timestamp(result.get("completed_at"), "stress result completed_at")
    require(started_at < completed_at, "stress result timestamps are not ordered")
    _validate_runtime(result.get("runtime"))
    _validate_preflight(result.get("preflight"), started_at=started_at)
    false_claims = (
        "theoretical_470_run_heap_merge_measured", "actual_full_4096_partition_occupancy_profile_measured",
        "full_byte_finalization_surface_measured", "source_hdd_opened", "production_component_pass",
        "pass_g1", "full_authorization_sufficient", "downstream_capacity_gate_passed",
        "programmatic_parent_of_full_receipt",
    )
    for key in false_claims:
        require(result.get(key) is False, f"stress claim boundary differs: {key}")
    require(result.get("full_like_per_partition_row_scale_measured") is True, "full-like evidence boundary differs")
    require(result.get("balanced_multi_run_heap_branch_measured") is True, "multi-run evidence boundary differs")
    require(result.get("maximum_measured_fan_in") == 8, "maximum fan-in differs")
    require(result.get("production_execution_status") == PRODUCTION_NOT_RUN, "stress result claims production")
    require(result.get("independent_verification_status") == "PENDING", "producer verification status differs")
    require(result.get("operator_acceptance_status") == "NOT_RECORDED", "operator acceptance differs")
    require(result.get("cache_condition") == "UNCONTROLLED_WARM_OR_CACHE_ASSISTED", "cache boundary differs")
    require(result.get("independent_quiet_window_sessions") == 0, "quiet-window boundary differs")
    require(result.get("strongest_permitted_interpretation") == "PARTIAL_SYNTHETIC_FULL_LIKE_PARTITION_AND_BALANCED_FAN_IN_KERNEL_EVIDENCE_ONLY", "permitted interpretation differs")
    require(result.get("operator_paths_redacted") is True, "operator paths are not redacted")
    implementation = result.get("implementation_binding")
    require(implementation == {
        "stress_tool_sha256": STRESS_TOOL_SHA256, "helper_tool_sha256": HELPER_SHA256,
        "runner_sha256": RUNNER_SHA256, "config_sha256": CONFIG_SHA256,
        "pilot_verifier_sha256": PILOT_VERIFIER_SHA256,
        "heap_operation_timing_proxy_installed": False, "run_reader_proxy_installed": False,
        "private_downstream_api_used": True, "production_partition_publisher_called": False,
        "production_finalizer_called": False, "production_receipt_builder_called": False,
    }, "stress implementation binding differs")
    expected_contracts = [
        {"scenario_id": item.scenario_id, "rows": item.rows, "expected_runs": item.expected_runs, "role": item.role}
        for item in SCENARIOS
    ]
    require(result.get("scenario_contracts") == expected_contracts, "scenario contracts differ")
    repeats = result.get("repeats")
    require(isinstance(repeats, list) and len(repeats) == REPEAT_COUNT, "repeat count differs")
    require([item.get("repeat_index") for item in repeats if isinstance(item, dict)] == list(range(1, REPEAT_COUNT + 1)), "repeat indices differ")


def _expected_run_rows(scenario: Scenario) -> list[int]:
    if scenario.expected_runs == 1:
        require(scenario.rows <= SORT_CHUNK_ROWS, "single-run scenario exceeds chunk")
        return [scenario.rows]
    require(scenario.rows == scenario.expected_runs * SORT_CHUNK_ROWS, "balanced scenario rows differ")
    return [SORT_CHUNK_ROWS] * scenario.expected_runs


def _spool_q_chunks(scenario: Scenario) -> Iterator[np.ndarray[Any, Any]]:
    for run_index, run_rows in enumerate(_expected_run_rows(scenario)):
        stop = run_rows
        while stop:
            start = max(0, stop - GENERATION_BATCH_ROWS)
            local_j = np.arange(stop - 1, start - 1, -1, dtype=np.uint64)
            yield np.uint64(scenario.expected_runs) * local_j + np.uint64(run_index)
            stop = start


def _expected_ids(q_values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.uint64(SYNTHETIC_NAMESPACE_BASE) + (q_values.astype(np.uint64, copy=False) << np.uint64(12)) + np.uint64(PARTITION_ID)


def _expected_parquet_row_group_rows(rows: int) -> list[int]:
    complete, remainder = divmod(rows, PARQUET_ROW_GROUP_ROWS)
    expected = [PARQUET_ROW_GROUP_ROWS] * complete
    if remainder:
        expected.append(remainder)
    return expected


def _validate_spool(path: Path, scenario: Scenario, pilot: np.ndarray[Any, Any]) -> str:
    descriptor, identity = _open_readonly(path, f"{scenario.scenario_id} spool")
    digest = hashlib.sha256()
    rows = 0
    try:
        for q_values in _spool_q_chunks(scenario):
            payload = _read_exact(descriptor, len(q_values) * RECORD_BYTES, "synthetic spool")
            digest.update(payload)
            records = np.frombuffer(payload, dtype=RECORD_DTYPE)
            require(bool(np.array_equal(records["work_id_numeric"], _expected_ids(q_values))), "spool q generator differs")
            _validate_temporal_binding(records, pilot, q_values)
            rows += len(records)
        require(os.read(descriptor, 1) == b"", "synthetic spool has trailing bytes")
        _validate_open(path, descriptor, identity, f"{scenario.scenario_id} spool")
    finally:
        os.close(descriptor)
    require(rows == scenario.rows, "spool row conservation differs")
    return digest.hexdigest()


def _validate_runs(root: Path, scenario: Scenario, pilot: np.ndarray[Any, Any], recorded: Mapping[str, Any]) -> list[dict[str, Any]]:
    run_rows = _expected_run_rows(scenario)
    expected_contracts: list[dict[str, Any]] = []
    raw_runs = recorded.get("runs")
    require(isinstance(raw_runs, list) and len(raw_runs) == scenario.expected_runs, "recorded run count differs")
    for run_index, rows in enumerate(run_rows):
        path = root / f".work_time_spine_tmp/runs/part-{PARTITION_ID:04d}/run-{run_index:04d}.bin"
        descriptor, identity = _open_readonly(path, f"{scenario.scenario_id} run {run_index}")
        digest = hashlib.sha256()
        observed = 0
        try:
            while observed < rows:
                count = min(GENERATION_BATCH_ROWS, rows - observed)
                q_values = np.uint64(run_index) + np.uint64(scenario.expected_runs) * np.arange(observed, observed + count, dtype=np.uint64)
                payload = _read_exact(descriptor, count * RECORD_BYTES, "sorted run")
                digest.update(payload)
                records = np.frombuffer(payload, dtype=RECORD_DTYPE)
                require(bool(np.array_equal(records["work_id_numeric"], _expected_ids(q_values))), "sorted run q sequence differs")
                _validate_temporal_binding(records, pilot, q_values)
                observed += count
            require(os.read(descriptor, 1) == b"", "sorted run has trailing bytes")
            _validate_open(path, descriptor, identity, f"{scenario.scenario_id} run {run_index}")
        finally:
            os.close(descriptor)
        contract = {
            "run_index": run_index, "rows": rows, "bytes": rows * RECORD_BYTES,
            "sha256": digest.hexdigest(), "q_start": run_index,
            "q_stop": scenario.expected_runs * (rows - 1) + run_index,
            "q_stride": scenario.expected_runs,
        }
        require(raw_runs[run_index] == contract, f"run contract differs: {scenario.scenario_id}:{run_index}")
        expected_contracts.append(contract)
    require(recorded.get("run_count") == scenario.expected_runs, "run validation count differs")
    require(recorded.get("run_contract_tree_sha256") == _sha256_bytes(_canonical_json_bytes(expected_contracts)), "run contract tree differs")
    require(recorded.get("balanced_fan_in") is (scenario.expected_runs > 1), "balanced fan-in flag differs")
    require(recorded.get("excluded_from_kernel_timing") is True and recorded.get("prewarms_external_run_page_cache_before_merge") is True, "run validation timing boundary differs")
    return expected_contracts


def _validate_parquet(path: Path, scenario: Scenario, pilot: np.ndarray[Any, Any]) -> dict[str, Any]:
    descriptor, identity = _open_readonly(path, f"{scenario.scenario_id} Parquet")
    digest = _new_record_digest(scenario.rows)
    disposition_counts = {name: 0 for name in DISPOSITION_NAMES.values()}
    null_rows = 0
    year_minimum: int | None = None
    year_maximum: int | None = None
    rows = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            parquet = pq.ParquetFile(handle)
            require(parquet.schema_arrow.equals(ARROW_SCHEMA, check_metadata=True), "stress Parquet schema differs")
            require((parquet.metadata.metadata or {}) == {}, "stress Parquet metadata differs")
            require(parquet.metadata.num_rows == scenario.rows, "stress Parquet footer rows differ")
            require(parquet.metadata.format_version == PARQUET_FORMAT_VERSION, "stress Parquet format version differs")
            expected_row_groups = _expected_parquet_row_group_rows(scenario.rows)
            require(parquet.metadata.num_row_groups == len(expected_row_groups), "stress Parquet row-group count differs")
            for row_group_id in range(parquet.metadata.num_row_groups):
                row_group = parquet.metadata.row_group(row_group_id)
                require(row_group.num_rows == expected_row_groups[row_group_id], "stress Parquet row-group rows differ")
                require(row_group.num_columns == len(ARROW_SCHEMA), "stress Parquet row-group columns differ")
                require(tuple(row_group.sorting_columns) == (), "stress Parquet sorting columns differ")
                for column_id in range(row_group.num_columns):
                    column = row_group.column(column_id)
                    require(column.compression == "UNCOMPRESSED", "stress Parquet compression differs")
                    require(column.is_stats_set is False, "stress Parquet statistics exist")
                    require(tuple(column.encodings) == PARQUET_COLUMN_ENCODINGS, "stress Parquet encodings differ")
                    require(column.has_dictionary_page is False, "stress Parquet dictionary page exists")
                    require(column.has_column_index is False, "stress Parquet column index exists")
                    require(column.has_offset_index is False, "stress Parquet offset index exists")
                    require(column.total_compressed_size == column.total_uncompressed_size, "stress Parquet encoded sizes differ")
            for batch in parquet.iter_batches(batch_size=GENERATION_BATCH_ROWS, use_threads=False):
                count = batch.num_rows
                q_values = np.arange(rows, rows + count, dtype=np.uint64)
                ids = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.uint64)
                require(bool(np.array_equal(ids, _expected_ids(q_values))), "Parquet q/ID sequence differs")
                require(bool(np.all(ids % PARTITION_COUNT == PARTITION_ID)), "Parquet modulo differs")
                dispositions = np.asarray(batch.column(3).to_numpy(zero_copy_only=False), dtype=np.uint8)
                day_nulls = np.asarray(batch.column(1).is_null().to_numpy(zero_copy_only=False), dtype=np.bool_)
                year_nulls = np.asarray(batch.column(2).is_null().to_numpy(zero_copy_only=False), dtype=np.bool_)
                require(bool(np.array_equal(day_nulls, year_nulls)), "Parquet nullability differs")
                require(bool(np.array_equal(day_nulls, dispositions == 0)), "Parquet disposition/nullability differs")
                days = np.asarray(pc.fill_null(batch.column(1), NULL_DAY).to_numpy(zero_copy_only=False), dtype=np.int32)
                years = np.asarray(pc.fill_null(batch.column(2), 0).to_numpy(zero_copy_only=False), dtype=np.uint16)
                records = np.empty(count, dtype=RECORD_DTYPE)
                records["work_id_numeric"] = ids
                records["publication_day"] = days
                records["publication_year"] = years
                records["temporal_disposition"] = dispositions
                records["reserved"] = 0
                _validate_temporal_binding(records, pilot, q_values)
                if bool(np.any(~day_nulls)):
                    nonnull_years = years[~year_nulls]
                    batch_min = int(np.min(nonnull_years))
                    batch_max = int(np.max(nonnull_years))
                    year_minimum = batch_min if year_minimum is None else min(year_minimum, batch_min)
                    year_maximum = batch_max if year_maximum is None else max(year_maximum, batch_max)
                    dates = pa.array(days, mask=day_nulls, type=pa.date32())
                    derived = np.asarray(pc.fill_null(pc.year(dates), 0).to_numpy(zero_copy_only=False), dtype=np.int64)
                    require(bool(np.all(derived[~day_nulls] == years[~day_nulls])), "Parquet day/year differs")
                counts = np.bincount(dispositions, minlength=len(DISPOSITION_NAMES))
                require(len(counts) == len(DISPOSITION_NAMES), "Parquet disposition differs")
                for code, name in DISPOSITION_NAMES.items():
                    disposition_counts[name] += int(counts[code])
                null_rows += int(np.count_nonzero(day_nulls))
                _update_record_digest(digest, records)
                rows += count
        _validate_open(path, descriptor, identity, f"{scenario.scenario_id} Parquet")
    finally:
        os.close(descriptor)
    require(rows == scenario.rows, "Parquet row conservation differs")
    file_sha, file_bytes = _hash_file(path, f"{scenario.scenario_id} Parquet hash")
    return {
        "rows": rows, "bytes": file_bytes, "sha256": file_sha,
        "record_sha256": digest.hexdigest(),
        "minimum_work_id_numeric": int(SYNTHETIC_NAMESPACE_BASE + PARTITION_ID),
        "maximum_work_id_numeric": int(SYNTHETIC_NAMESPACE_BASE + ((scenario.rows - 1) << 12) + PARTITION_ID),
        "disposition_counts": disposition_counts,
        "w_zero_disposition_counts": {name: 0 for name in DISPOSITION_NAMES.values()},
        "nullability_counts": {"publication_day_null_rows": null_rows, "publication_year_null_rows": null_rows},
        "publication_year_extrema": {"minimum": year_minimum, "maximum": year_maximum},
    }


def _validate_scenario(root: Path, scenario: Scenario, repeat_index: int, recorded: Mapping[str, Any], result: Mapping[str, Any], pilot: np.ndarray[Any, Any]) -> None:
    require(set(recorded) == SCENARIO_RESULT_KEYS, "scenario result key set differs")
    fixed = {
        "scenario_id": scenario.scenario_id, "role": scenario.role, "rows": scenario.rows,
        "partition_id": PARTITION_ID, "sort_chunk_rows": SORT_CHUNK_ROWS,
        "expected_runs": scenario.expected_runs, "observed_runs": scenario.expected_runs,
        "maximum_measured_fan_in": scenario.expected_runs, "output_rows": scenario.rows,
        "integrity_status": "PASS_SYNTHETIC_STRESS_KERNEL_CLOSURE", "operator_paths_redacted": True,
    }
    for key, value in fixed.items():
        require(recorded.get(key) == value, f"scenario field differs: {scenario.scenario_id}:{key}")
    preparation = _mapping_with_exact_keys(
        recorded.get("preparation"), PREPARATION_KEYS,
        f"{scenario.scenario_id} preparation",
    )
    spool_path = root / f".work_time_spine_tmp/spool/part-{PARTITION_ID:04d}.bin"
    spool_sha = _validate_spool(spool_path, scenario, pilot)
    for key, value in {
        "rows": scenario.rows, "bytes": scenario.rows * RECORD_BYTES,
        "spool_sha256": spool_sha, "synthetic_namespace_base": SYNTHETIC_NAMESPACE_BASE,
        "partition_id": PARTITION_ID, "temporal_payload_source": "VERIFIED_PILOT_STAGING_CYCLIC_BY_Q",
        "excluded_from_kernel_timing": True,
    }.items():
        require(preparation.get(key) == value, f"preparation differs: {scenario.scenario_id}:{key}")
    _finite_float(
        preparation["wall_seconds"],
        f"{scenario.scenario_id} preparation.wall_seconds",
        positive=True,
    )
    run_validation = _mapping_with_exact_keys(
        recorded.get("run_validation"), RUN_VALIDATION_KEYS,
        f"{scenario.scenario_id} run validation",
    )
    _validate_runs(root, scenario, pilot, run_validation)
    _finite_float(
        run_validation["wall_seconds"],
        f"{scenario.scenario_id} run validation.wall_seconds",
        positive=True,
    )
    output_relative = f"spine/part-{PARTITION_ID:04d}.parquet"
    parquet = _validate_parquet(root / output_relative, scenario, pilot)
    checkpoint, _ = _read_json(root / "stress_checkpoint.json", f"{scenario.scenario_id} checkpoint")
    require(set(checkpoint) == CHECKPOINT_KEYS, "stress checkpoint key set differs")
    core = dict(checkpoint)
    checkpoint_sha = core.pop("checkpoint_sha256")
    require(checkpoint_sha == _sha256_bytes(_canonical_json_bytes(core)), "stress checkpoint self-hash differs")
    task_fp = _sha256_bytes(_canonical_json_bytes({"schema": SCHEMA, "repeat_index": repeat_index, "scenario": scenario.scenario_id, "first_benchmark_sha256": FIRST_BENCHMARK_SHA256}))
    scan_tree = _sha256_bytes(_canonical_json_bytes({"scenario": scenario.scenario_id, "repeat": repeat_index}))
    expected_checkpoint = {
        "schema_version": "openalex_g1_canonical_work_time_spine_downstream_stress_partition_checkpoint.v1",
        "task_fingerprint_sha256": task_fp,
        "runtime_fingerprint": result["runtime"]["runner_runtime_fingerprint"],
        "scan_checkpoint_tree_sha256": scan_tree, "staging_shard_tree_sha256": spool_sha,
        "partition_id": PARTITION_ID, "partition_formula": "work_id_numeric_mod_4096",
        "output_relative_path": output_relative, "output_rows": scenario.rows,
        "minimum_work_id_numeric": parquet["minimum_work_id_numeric"],
        "maximum_work_id_numeric": parquet["maximum_work_id_numeric"],
        "output_schema": SCHEMA_CONTRACT,
        "output_schema_sha256": _sha256_bytes(_canonical_json_bytes(SCHEMA_CONTRACT)),
        "pre_parquet_sorted_record_sha256": parquet["record_sha256"],
        "decoded_parquet_record_sha256": parquet["record_sha256"],
        "disposition_counts": parquet["disposition_counts"],
        "w_zero_disposition_counts": parquet["w_zero_disposition_counts"],
        "publication_year_extrema": parquet["publication_year_extrema"],
        "nullability_counts": parquet["nullability_counts"],
        "output_bytes": parquet["bytes"], "output_sha256": parquet["sha256"],
        "strictly_increasing_ids": True, "operator_paths_redacted": True,
    }
    for key, value in expected_checkpoint.items():
        require(checkpoint.get(key) == value, f"checkpoint differs: {scenario.scenario_id}:{key}")
    for key, value in {
        "output_bytes": parquet["bytes"], "output_sha256": parquet["sha256"],
        "record_sha256": parquet["record_sha256"],
    }.items():
        require(recorded.get(key) == value, f"scenario output differs: {scenario.scenario_id}:{key}")
    sort = recorded.get("sort")
    merge = recorded.get("merge_parquet_and_validation")
    _validate_kernel_phase(
        sort, phase="sort", scenario=scenario, output_bytes=parquet["bytes"],
    )
    _validate_kernel_phase(
        merge, phase="merge", scenario=scenario, output_bytes=parquet["bytes"],
    )
    require(isinstance(merge, Mapping), "merge phase differs")
    require(merge.get("cache_condition") == "CACHE_ASSISTED_BY_IMMEDIATELY_PRECEDING_FULL_RUN_VALIDATION", "merge cache boundary differs")
    derived = scenario.rows if scenario.expected_runs > 1 else 0
    require(
        _integer(
            merge.get("derived_heap_push_count_not_observed"),
            f"{scenario.scenario_id} derived heap push count",
        )
        == derived
        and _integer(
            merge.get("derived_heap_pop_count_not_observed"),
            f"{scenario.scenario_id} derived heap pop count",
        )
        == derived,
        "derived heap count boundary differs",
    )


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    return {"count": len(ordered), "minimum": round(ordered[0], 6), "median": round(ordered[len(ordered) // 2], 6), "maximum": round(ordered[-1], 6)}


def _validate_aggregate(result: Mapping[str, Any]) -> None:
    by_scenario: dict[str, Any] = {}
    for scenario in SCENARIOS:
        observations = [next(item for item in repeat["scenarios"] if item["scenario_id"] == scenario.scenario_id) for repeat in result["repeats"]]
        by_scenario[scenario.scenario_id] = {
            "repeat_count": REPEAT_COUNT,
            "sort_wall_rows_per_second": _distribution([item["sort"]["wall_rows_per_second"] for item in observations]),
            "merge_wall_rows_per_second": _distribution([item["merge_parquet_and_validation"]["wall_rows_per_second"] for item in observations]),
            "output_bytes": _distribution([float(item["output_bytes"]) for item in observations]),
        }
    expected = {"by_scenario": by_scenario, "one_sided_95_percent_lower_confidence_bound_computed": False, "minimum_observed_rates_are_not_adopted_capacity_floors": True, "independent_quiet_window_sessions": 0}
    require(result.get("aggregate") == expected, "stress aggregate does not reconstruct")


def _expected_scratch_topology() -> tuple[dict[str, int], set[str]]:
    directories: dict[str, int] = {".": 0o700}
    files: set[str] = set()
    for repeat_index in range(1, REPEAT_COUNT + 1):
        repeat = f"repeat-{repeat_index:02d}"
        directories[repeat] = 0o700
        for scenario in SCENARIOS:
            base = f"{repeat}/{scenario.scenario_id}"
            directories[base] = 0o700
            directories[f"{base}/.work_time_spine_tmp"] = 0o775
            for relative in (".work_time_spine_tmp/runs", f".work_time_spine_tmp/runs/part-{PARTITION_ID:04d}", ".work_time_spine_tmp/sorted", ".work_time_spine_tmp/spool", "spine"):
                directories[f"{base}/{relative}"] = 0o700
            files.add(f"{base}/.work_time_spine_tmp/spool/part-{PARTITION_ID:04d}.bin")
            for run_index in range(scenario.expected_runs):
                files.add(f"{base}/.work_time_spine_tmp/runs/part-{PARTITION_ID:04d}/run-{run_index:04d}.bin")
            files.add(f"{base}/spine/part-{PARTITION_ID:04d}.parquet")
            files.add(f"{base}/stress_checkpoint.json")
    return directories, files


def _validate_scratch_topology(root: Path) -> dict[str, Any]:
    expected_directories, expected_files = _expected_scratch_topology()
    observed_directories: dict[str, int] = {".": stat.S_IMODE(root.lstat().st_mode)}
    observed_files: set[str] = set()
    total_bytes = 0
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), "scratch contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            observed_directories[relative] = stat.S_IMODE(info.st_mode)
        else:
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and stat.S_IMODE(info.st_mode) == 0o600, "scratch file contract differs")
            observed_files.add(relative)
            total_bytes += int(info.st_size)
    require(observed_directories == expected_directories, "scratch directory topology/modes differ")
    require(observed_files == expected_files, "scratch file topology differs")
    contained_temp = [relative for relative, mode in observed_directories.items() if mode == 0o775]
    require(len(contained_temp) == REPEAT_COUNT * len(SCENARIOS), "0775 temp-parent count differs")
    for relative in contained_temp:
        scenario_parent = root / relative
        require(stat.S_IMODE(scenario_parent.parent.lstat().st_mode) == 0o700, "0775 temp parent lacks 0700 scenario ancestor")
    return {
        "regular_file_count": len(observed_files), "regular_file_bytes": total_bytes,
        "directory_count_including_root": len(observed_directories),
        "mode_0700_directory_count": sum(mode == 0o700 for mode in observed_directories.values()),
        "contained_mode_0775_temp_parent_count": len(contained_temp),
        "contained_mode_0775_temp_parents_have_mode_0700_scenario_ancestors": True,
    }


def verify_stress(
    *, result_path: Path, scratch_root: Path, pilot_output_root: Path,
    first_benchmark_result: Path,
) -> dict[str, Any]:
    result_file = _absolute(result_path)
    scratch = _absolute(scratch_root)
    pilot = _absolute(pilot_output_root)
    first_file = _absolute(first_benchmark_result)
    require(result_file.name == "openalex_g1_canonical_work_time_spine_downstream_stress_v1.json", "stress result basename differs")
    require(first_file.name == "openalex_g1_canonical_work_time_spine_downstream_benchmark.json", "first benchmark basename differs")
    for root, field in (
        (result_file.parent, "stress result input root"),
        (first_file.parent, "first benchmark input root"),
    ):
        info = root.lstat()
        require(stat.S_ISDIR(info.st_mode), f"{field} is not a directory")
        require(info.st_uid == os.geteuid(), f"{field} owner differs")
        require(stat.S_IMODE(info.st_mode) == 0o700, f"{field} mode is not 0700")
    result_tree_before = _snapshot_tree(result_file.parent, "stress result input tree")
    scratch_before = _snapshot_tree(scratch, "stress scratch input tree")
    pilot_before = _snapshot_tree(pilot, "pilot input tree")
    first_before = _snapshot_tree(first_file.parent, "first benchmark input tree")
    require(result_tree_before["regular_file_count"] == 1, "stress result tree has unexpected files")
    require(result_tree_before["directory_count_including_root"] == 1, "stress result tree has unexpected directories")
    require(first_before["regular_file_count"] == 1, "first benchmark tree has unexpected files")
    require(first_before["directory_count_including_root"] == 1, "first benchmark tree has unexpected directories")
    result, result_sha = _read_json(result_file, "stress benchmark result")
    require(result_sha == EXPECTED_STRESS_RESULT_SHA256, "stress result SHA-256 differs")
    _validate_result_contract(result)
    first_result, first_sha = _validate_first_benchmark(first_file)
    pilot_records, pilot_task_fp = _validate_pilot(pilot)
    parents = result.get("parent_bindings")
    require(parents == {
        "first_benchmark_sha256": first_sha,
        "first_benchmark_fingerprint": first_result["result_fingerprint_sha256"],
        "pilot_receipt_sha256": PILOT_RECEIPT_SHA256,
        "pilot_staging_sha256": PILOT_STAGING_SHA256,
        "pilot_task_fingerprint_sha256": pilot_task_fp,
        "first_benchmark_tree_unchanged": True, "pilot_tree_unchanged": True,
    }, "stress parent bindings differ")
    topology = _validate_scratch_topology(scratch)
    for repeat_index, repeat in enumerate(result["repeats"], start=1):
        require(set(repeat) == {"repeat_index", "scenarios", "status", "operator_paths_redacted"}, "repeat result key set differs")
        require(repeat.get("repeat_index") == repeat_index and repeat.get("status") == "PASS_SYNTHETIC_STRESS_REPEAT" and repeat.get("operator_paths_redacted") is True, "repeat result differs")
        scenarios = repeat.get("scenarios")
        require(isinstance(scenarios, list) and [item.get("scenario_id") for item in scenarios if isinstance(item, dict)] == [item.scenario_id for item in SCENARIOS], "repeat scenario order differs")
        for scenario, recorded in zip(SCENARIOS, scenarios, strict=True):
            _validate_scenario(scratch / f"repeat-{repeat_index:02d}/{scenario.scenario_id}", scenario, repeat_index, recorded, result, pilot_records)
    _validate_aggregate(result)
    result_tree_after = _snapshot_tree(result_file.parent, "stress result input tree readback")
    scratch_after = _snapshot_tree(scratch, "stress scratch input tree readback")
    pilot_after = _snapshot_tree(pilot, "pilot input tree readback")
    first_after = _snapshot_tree(first_file.parent, "first benchmark input tree readback")
    require(result_tree_after == result_tree_before, "stress result input tree changed")
    require(scratch_after == scratch_before, "stress scratch input tree changed")
    require(pilot_after == pilot_before, "pilot input tree changed")
    require(first_after == first_before, "first benchmark input tree changed")
    return {
        "schema_version": VERIFICATION_SCHEMA, "status": VERIFICATION_STATUS,
        "verified_stress_status": STATUS, "stress_result_sha256": result_sha,
        "stress_result_fingerprint_sha256": result["result_fingerprint_sha256"],
        "first_benchmark_sha256": first_sha, "pilot_receipt_sha256": PILOT_RECEIPT_SHA256,
        "pilot_staging_sha256": PILOT_STAGING_SHA256, "repeat_count": REPEAT_COUNT,
        "scenario_ids": [item.scenario_id for item in SCENARIOS],
        "scratch_artifact_contract_tree_sha256": scratch_before["artifact_contract_tree_sha256"],
        **topology,
        "stress_result_input_tree_unchanged": True, "stress_scratch_input_tree_unchanged": True,
        "pilot_input_tree_unchanged": True, "first_benchmark_input_tree_unchanged": True,
        "tree_identity_and_content_sha256_verified_before_and_after": True,
        "production_execution_status": PRODUCTION_NOT_RUN, "production_component_pass": False,
        "pass_g1": False, "downstream_capacity_gate_passed": False,
        "full_authorization_sufficient": False, "programmatic_parent_of_full_receipt": False,
        "operator_paths_redacted": True,
        "independence_boundary": "Independent read-only NumPy/PyArrow verification; no stress producer, benchmark helper, or production runner import.",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--pilot-output-root", type=Path, required=True)
    parser.add_argument("--first-benchmark-result", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        verification = verify_stress(
            result_path=args.result, scratch_root=args.scratch_root,
            pilot_output_root=args.pilot_output_root,
            first_benchmark_result=args.first_benchmark_result,
        )
    except (StressVerificationError, OSError, ValueError, TypeError, pa.ArrowException) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_published_json_bytes(verification))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
