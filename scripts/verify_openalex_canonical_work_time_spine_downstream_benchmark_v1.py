#!/usr/bin/env python3
"""Independently verify the bounded canonical-spine downstream benchmark.

The verifier deliberately imports neither the benchmark harness nor any work-time
spine producer.  Every supplied artifact is opened read-only, the three retained
scratch trees are reconstructed from bytes, and the output remains explicitly
non-production and non-promotable.
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
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


BENCHMARK_SCHEMA = "openalex_g1_canonical_work_time_spine_downstream_benchmark.v1"
BENCHMARK_STATUS = (
    "PARTIAL_FIXED_OVERHEAD_SINGLE_RUN_AND_BOUNDED_H9185_EVIDENCE_"
    "OPERATOR_LOCAL_NONPROMOTABLE"
)
VERIFICATION_SCHEMA = (
    "openalex_g1_canonical_work_time_spine_downstream_benchmark_verification.v1"
)
VERIFICATION_STATUS = (
    "PASS_READ_ONLY_INDEPENDENT_BOUNDED_DOWNSTREAM_BENCHMARK_VERIFICATION_"
    "NONPRODUCTION"
)
PILOT_SCOPE = "PRODUCTION_SOURCE_PREFIX_PILOT_UNPROMOTABLE"
PILOT_STATUS = "PILOT_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_NONPROMOTABLE"
PRODUCTION_NOT_RUN = "PRODUCTION_NOT_RUN"
RUNNER_SHA256 = "bf9ed534b2e9f9c2d6c9851be7fde298eb372d8a3cf5e17363ea6afe51cf1280"
CONFIG_SHA256 = "6e8b74ddd12cbef46c8d15d3560f0d54f470b5017685fc1bce2389250cde0484"
PILOT_VERIFIER_SHA256 = "2cc47587ecd70bd1ae169f1f987828f024d6b8f7dd217e2e9b7964fabfcd5ede"

REPEAT_COUNT = 3
SHARD_COUNT = 493
PARTITION_COUNT = 4096
TOTAL_ROWS = 1_000_000
RECORD_BYTES = 16
CONTROL_ARTIFACT_COUNT = 7
EXACT_ARTIFACT_COUNT = 9_185
MAX_JSON_BYTES = 16 << 20
NULL_DAY = -(2**31)
RECORD_HASH_DOMAIN = b"openalex-work-time-records-v1"
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
CONTROL_PATHS = (
    "benchmark_plan.json",
    "preflight_snapshot.json",
    "run_state.json",
    "staging_manifest.json",
    "spine_manifest.json",
    "benchmark_report.json",
    "benchmark_report.md",
)
TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "started_at",
    "completed_at",
    "implementation_binding",
    "runtime",
    "pilot_binding",
    "preflight",
    "benchmark_scope",
    "repeats",
    "aggregate",
    "pilot_output_tree_unchanged",
    "production_execution_status",
    "production_component_pass",
    "pass_g1",
    "full_authorization_sufficient",
    "downstream_capacity_gate_passed",
    "programmatic_parent_of_full_receipt",
    "operator_acceptance_status",
    "strongest_permitted_interpretation",
    "operator_paths_redacted",
    "result_fingerprint_sha256",
}
PARTITION_CHECKPOINT_KEYS = {
    "schema_version",
    "task_fingerprint_sha256",
    "runtime_fingerprint",
    "scan_checkpoint_tree_sha256",
    "staging_shard_tree_sha256",
    "partition_id",
    "partition_formula",
    "output_relative_path",
    "output_rows",
    "minimum_work_id_numeric",
    "maximum_work_id_numeric",
    "output_schema",
    "output_schema_sha256",
    "pre_parquet_sorted_record_sha256",
    "decoded_parquet_record_sha256",
    "disposition_counts",
    "w_zero_disposition_counts",
    "publication_year_extrema",
    "nullability_counts",
    "output_bytes",
    "output_sha256",
    "strictly_increasing_ids",
    "operator_paths_redacted",
    "checkpoint_sha256",
}


class VerificationError(RuntimeError):
    """The independent benchmark verifier failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def fail(message: str, cause: BaseException | None = None) -> NoReturn:
    if cause is None:
        raise VerificationError(message)
    raise VerificationError(message) from cause


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _published_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_nlink),
        int(info.st_uid),
        int(info.st_gid),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


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


def _read_bytes(path: Path, field: str) -> tuple[bytes, str]:
    descriptor, identity = _open_readonly(path, field)
    blocks: list[bytes] = []
    digest = hashlib.sha256()
    try:
        while payload := os.read(descriptor, 1 << 20):
            blocks.append(payload)
            digest.update(payload)
        _validate_open(path, descriptor, identity, field)
    finally:
        os.close(descriptor)
    return b"".join(blocks), digest.hexdigest()


def _hash_file(path: Path, field: str) -> tuple[str, int]:
    descriptor, identity = _open_readonly(path, field)
    digest = hashlib.sha256()
    try:
        while payload := os.read(descriptor, 16 << 20):
            digest.update(payload)
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
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: fail(f"{field} contains {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot parse {field}", exc)
    require(isinstance(value, dict), f"{field} is not a mapping")
    require(payload == _published_json_bytes(value), f"{field} is not canonical JSON")
    return dict(value), digest


def _directory_snapshot(root: Path) -> dict[str, tuple[int, ...]]:
    snapshot = {".": _identity(root.lstat())}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        snapshot[path.relative_to(root).as_posix()] = _identity(path.lstat())
    return snapshot


def _inspect_private_tree(root: Path, field: str) -> tuple[list[str], list[str]]:
    _assert_no_symlink_components(root, field)
    root_info = root.lstat()
    require(stat.S_ISDIR(root_info.st_mode), f"{field} is not a directory")
    require(stat.S_IMODE(root_info.st_mode) == 0o700, f"{field} mode is not 0700")
    require(root_info.st_uid == os.geteuid(), f"{field} owner differs")
    files: list[str] = []
    directories: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), f"{field} contains a symlink")
        require(info.st_uid == os.geteuid(), f"{field} entry owner differs")
        if stat.S_ISREG(info.st_mode):
            require(info.st_nlink == 1, f"{field} contains a hard link")
            require(stat.S_IMODE(info.st_mode) == 0o600, f"{field} file mode differs")
            files.append(relative)
        elif stat.S_ISDIR(info.st_mode):
            require(stat.S_IMODE(info.st_mode) == 0o700, f"{field} directory mode differs")
            directories.append(relative)
        else:
            fail(f"{field} contains an unsupported entry")
    return sorted(files), sorted(directories)


def _tree_identity_digest(root: Path) -> dict[str, Any]:
    _assert_no_symlink_components(root, "pilot output root")
    require(stat.S_ISDIR(root.lstat().st_mode), "pilot output root is not a directory")
    items: list[dict[str, Any]] = []
    regular_files = 0
    regular_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), "pilot tree contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            kind = "directory"
        else:
            require(stat.S_ISREG(info.st_mode), "pilot tree entry type differs")
            kind = "file"
        item: dict[str, Any] = {
            "relative_path": path.relative_to(root).as_posix(),
            "kind": kind,
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
            "mode": int(info.st_mode),
            "uid": int(info.st_uid),
            "gid": int(info.st_gid),
            "links": int(info.st_nlink),
            "bytes": int(info.st_size),
            "mtime_ns": int(info.st_mtime_ns),
            "ctime_ns": int(info.st_ctime_ns),
        }
        if kind == "file":
            item["sha256"] = _hash_file(path, "pilot output artifact")[0]
            regular_files += 1
            regular_bytes += int(info.st_size)
        items.append(item)
    return {
        "tree_identity_sha256": _sha256_bytes(_canonical_json_bytes(items)),
        "regular_file_count": regular_files,
        "regular_file_bytes": regular_bytes,
    }


def _result_fingerprint(result: Mapping[str, Any]) -> str:
    core = dict(result)
    core.pop("result_fingerprint_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(core))


def _report_fingerprint(report: Mapping[str, Any]) -> str:
    core = copy.deepcopy(dict(report))
    provenance = core.get("provenance")
    require(isinstance(provenance, dict), "pilot report provenance is not a mapping")
    provenance.pop("result_fingerprint_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(core))


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(math.ceil(probability * len(ordered)) - 1, len(ordered) - 1))
    return round(ordered[index], 9)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "minimum": round(min(values), 9),
        "p50": _quantile(values, 0.5),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "maximum": round(max(values), 9),
        "mean": round(sum(values) / len(values), 9),
    }


def _validate_result_contract(result: Mapping[str, Any]) -> None:
    require(set(result) == TOP_LEVEL_KEYS, "benchmark result key set differs")
    require(result.get("schema_version") == BENCHMARK_SCHEMA, "benchmark schema differs")
    require(result.get("status") == BENCHMARK_STATUS, "benchmark status differs")
    require(result.get("result_fingerprint_sha256") == _result_fingerprint(result), "result fingerprint differs")
    require(result.get("production_execution_status") == PRODUCTION_NOT_RUN, "result claims production execution")
    for key in (
        "production_component_pass",
        "pass_g1",
        "full_authorization_sufficient",
        "downstream_capacity_gate_passed",
        "programmatic_parent_of_full_receipt",
    ):
        require(result.get(key) is False, f"claim boundary differs: {key}")
    require(result.get("pilot_output_tree_unchanged") is True, "result says pilot changed")
    require(result.get("operator_acceptance_status") == "NOT_RECORDED", "operator acceptance differs")
    require(
        result.get("strongest_permitted_interpretation")
        == "PARTIAL_BOUNDED_FIXED_OVERHEAD_SINGLE_RUN_AND_EXACT_ENTRY_COUNT_EVIDENCE_ONLY",
        "strongest permitted interpretation differs",
    )
    require(result.get("operator_paths_redacted") is True, "result paths are not redacted")
    implementation = result.get("implementation_binding")
    require(isinstance(implementation, dict), "implementation binding is not a mapping")
    require(
        implementation
        == {
            "runner_sha256": RUNNER_SHA256,
            "config_sha256": CONFIG_SHA256,
            "independent_verifier_sha256": PILOT_VERIFIER_SHA256,
            "benchmark_tool_sha256": implementation.get("benchmark_tool_sha256"),
            "loaded_code_attestation": False,
            "private_downstream_api_used": True,
            "production_finalizer_called": False,
            "production_receipt_builder_called": False,
        },
        "implementation binding differs",
    )
    tool_sha = implementation.get("benchmark_tool_sha256")
    require(isinstance(tool_sha, str) and len(tool_sha) == 64, "benchmark tool SHA differs")
    scope = result.get("benchmark_scope")
    require(isinstance(scope, dict), "benchmark scope is not a mapping")
    require(
        scope
        == {
            "scenarios": ["D4096", "R4096", "H9185_BOUNDED_BYTES"],
            "repeat_count": REPEAT_COUNT,
            "artifact_sweeps_per_repeat": scope.get("artifact_sweeps_per_repeat"),
            "source_hdd_opened": False,
            "cache_condition": "UNCONTROLLED_WARM_OR_CACHE_ASSISTED",
            "full_like_partition_occupancy_measured": False,
            "multi_run_heap_merge_measured": False,
            "full_byte_finalization_surface_measured": False,
            "independent_quiet_window_sessions": 0,
        },
        "benchmark scope differs",
    )
    sweeps = scope.get("artifact_sweeps_per_repeat")
    require(type(sweeps) is int and 1 <= sweeps <= 5, "artifact sweep count differs")
    repeats = result.get("repeats")
    require(isinstance(repeats, list) and len(repeats) == REPEAT_COUNT, "result repeat count differs")
    require([item.get("repeat_index") for item in repeats if isinstance(item, dict)] == list(range(1, REPEAT_COUNT + 1)), "repeat indices differ")


def _validate_preflight(preflight: Mapping[str, Any]) -> None:
    required = {
        "observed_at",
        "available_disk_bytes",
        "available_ram_bytes",
        "nofile_soft_limit",
        "nofile_hard_limit",
        "required_disk_bytes",
        "required_available_ram_bytes",
        "required_nofile_soft_limit",
        "bounded_preflight_passed",
        "production_preflight_passed",
        "source_drive_health_not_applicable_no_source_opened",
        "quiet_window_independently_verified",
        "operator_paths_redacted",
        "atomic_rename_fsync_probe",
    }
    require(set(preflight) == required, "preflight key set differs")
    expected = {
        "required_disk_bytes": 36_366_736_928,
        "required_available_ram_bytes": 9_685_696_512,
        "required_nofile_soft_limit": 8_192,
        "bounded_preflight_passed": True,
        "production_preflight_passed": False,
        "source_drive_health_not_applicable_no_source_opened": True,
        "quiet_window_independently_verified": False,
        "operator_paths_redacted": True,
    }
    for key, value in expected.items():
        require(preflight.get(key) == value, f"preflight field differs: {key}")
    probe = preflight.get("atomic_rename_fsync_probe")
    require(isinstance(probe, dict), "preflight fsync probe differs")
    require(set(probe) == {"status", "wall_seconds"}, "fsync probe keys differ")
    require(probe.get("status") == "PASS_BOUNDED_SCRATCH_ATOMIC_RENAME_FSYNC", "fsync probe status differs")
    require(isinstance(probe.get("wall_seconds"), (int, float)) and probe["wall_seconds"] >= 0, "fsync probe duration differs")


def _validate_pilot(pilot_root: Path, binding: Mapping[str, Any]) -> str:
    files, _ = _inspect_private_tree(pilot_root, "pilot output root")
    report, _ = _read_json(pilot_root / "work_time_spine_pilot.json", "pilot report")
    manifest, _ = _read_json(pilot_root / "staging_manifest.json", "pilot staging manifest")
    receipt, receipt_sha = _read_json(pilot_root / "work_time_spine_pilot.outputs.json", "pilot receipt")
    require(receipt_sha == binding.get("receipt_sha256"), "pilot receipt SHA differs")
    require(report.get("status") == PILOT_STATUS, "pilot report status differs")
    require(report.get("task_fingerprint_sha256") == binding.get("task_fingerprint_sha256"), "pilot task fingerprint differs")
    scope = report.get("scope")
    require(isinstance(scope, dict), "pilot report scope differs")
    require(scope.get("execution_scope") == PILOT_SCOPE, "pilot execution scope differs")
    require(scope.get("production_status") == PRODUCTION_NOT_RUN, "pilot claims production execution")
    require(scope.get("retained_rows") == TOTAL_ROWS, "pilot retained rows differ")
    provenance = report.get("provenance")
    require(isinstance(provenance, dict), "pilot provenance differs")
    require(provenance.get("tool_sha256") == RUNNER_SHA256, "pilot runner binding differs")
    require(provenance.get("config_sha256") == CONFIG_SHA256, "pilot config binding differs")
    result_fp = _report_fingerprint(report)
    require(provenance.get("result_fingerprint_sha256") == result_fp, "pilot result fingerprint differs")
    require(result_fp == binding.get("result_fingerprint_sha256"), "pilot result binding differs")
    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, dict), "pilot receipt artifacts differ")
    require(receipt.get("artifact_count") == len(artifacts), "pilot receipt artifact count differs")
    require(receipt.get("artifact_count") == binding.get("independent_verifier_artifact_count"), "pilot verifier artifact count differs")
    require(receipt.get("scope") == PILOT_SCOPE, "pilot receipt scope differs")
    require(receipt.get("production_status") == PRODUCTION_NOT_RUN, "pilot receipt claims production")
    require(receipt.get("production_component_pass") is False, "pilot receipt claims component pass")
    require(receipt.get("task_fingerprint_sha256") == binding.get("task_fingerprint_sha256"), "pilot receipt task differs")
    require(receipt.get("result_fingerprint_sha256") == result_fp, "pilot receipt result differs")
    for relative, contract in artifacts.items():
        require(isinstance(relative, str) and not relative.startswith("/") and ".." not in Path(relative).parts, "pilot artifact path differs")
        require(isinstance(contract, dict), "pilot artifact contract differs")
        digest, size = _hash_file(pilot_root / relative, f"pilot artifact {relative}")
        require(contract == {"relative_path": relative, "bytes": size, "sha256": digest}, f"pilot artifact contract differs: {relative}")
    require(set(files) == set(artifacts) | {"audit.lock", "work_time_spine_pilot.outputs.json"}, "pilot file set differs")
    lock_sha, lock_bytes = _hash_file(pilot_root / "audit.lock", "pilot audit lock")
    require(lock_bytes == 0 and lock_sha == _sha256_bytes(b""), "pilot audit lock differs")
    shards = manifest.get("shards")
    require(isinstance(shards, list) and len(shards) == 1, "pilot staging shard count differs")
    shard = shards[0]
    require(isinstance(shard, dict), "pilot staging shard differs")
    relative = shard.get("relative_path")
    require(isinstance(relative, str) and not relative.startswith("/"), "pilot staging path differs")
    digest, size = _hash_file(pilot_root / relative, "pilot staging payload")
    require(digest == shard.get("sha256") == binding.get("staging_sha256"), "pilot staging SHA differs")
    require(size == shard.get("bytes") == binding.get("staging_bytes") == TOTAL_ROWS * RECORD_BYTES, "pilot staging bytes differ")
    require(shard.get("rows") == binding.get("staging_rows") == TOTAL_ROWS, "pilot staging rows differ")
    current_tree = _tree_identity_digest(pilot_root)
    require(binding.get("tree_before") == current_tree, "pilot tree differs from pre-benchmark binding")
    require(binding.get("tree_after") == current_tree, "pilot tree differs from post-benchmark binding")
    return digest


def _validate_record_array(records: np.ndarray[Any, Any]) -> None:
    require(records.dtype == RECORD_DTYPE, "staging record dtype differs")
    if not len(records):
        return
    dispositions = records["temporal_disposition"]
    require(bool(np.all(records["reserved"] == 0)), "staging reserved byte differs")
    require(bool(np.all(dispositions <= max(DISPOSITION_NAMES))), "staging disposition differs")
    nulls = dispositions == 0
    require(bool(np.all(records["publication_day"][nulls] == NULL_DAY)), "staging null day differs")
    require(bool(np.all(records["publication_year"][nulls] == 0)), "staging null year differs")
    require(bool(np.all(records["publication_day"][~nulls] != NULL_DAY)), "staging nonnull day differs")
    require(bool(np.all(records["publication_year"][~nulls] != 0)), "staging nonnull year differs")


def _framed_digest(records: np.ndarray[Any, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(len(RECORD_HASH_DOMAIN).to_bytes(8, "little"))
    digest.update(RECORD_HASH_DOMAIN)
    digest.update(len(records).to_bytes(8, "little"))
    framed = np.empty(len(records), dtype=FRAMED_DTYPE)
    framed["encoded_length"] = RECORD_BYTES
    framed["record"] = records.view(f"V{RECORD_BYTES}").reshape(-1)
    digest.update(memoryview(framed).cast("B"))
    return digest.hexdigest()


def _expected_paths() -> list[str]:
    paths = list(CONTROL_PATHS)
    for index in range(SHARD_COUNT):
        token = _sha256_bytes(f"benchmark-shard-{index:04d}".encode("utf-8"))
        paths.extend((f"staging/main/{token}.bin", f"checkpoints/scan/main/{token}.json"))
    for partition_id in range(PARTITION_COUNT):
        paths.extend((f"spine/part-{partition_id:04d}.parquet", f"checkpoints/partition/part-{partition_id:04d}.json"))
    return sorted(paths)


def _artifact_contracts(root: Path, paths: Sequence[str]) -> tuple[list[dict[str, Any]], int]:
    contracts: list[dict[str, Any]] = []
    total = 0
    for relative in paths:
        digest, size = _hash_file(root / relative, f"scratch artifact {relative}")
        contracts.append({"relative_path": relative, "bytes": size, "sha256": digest})
        total += size
    return contracts, total


def _validate_staging(
    root: Path,
) -> tuple[list[bytearray], list[int], str, str]:
    base, remainder = divmod(TOTAL_ROWS, SHARD_COUNT)
    partition_payloads = [bytearray() for _ in range(PARTITION_COUNT)]
    source_digest = hashlib.sha256()
    total_rows = 0
    for index in range(SHARD_COUNT):
        rows = base + (1 if index < remainder else 0)
        logical = f"benchmark-shard-{index:04d}"
        token = _sha256_bytes(logical.encode("utf-8"))
        staging_relative = f"staging/main/{token}.bin"
        checkpoint_relative = f"checkpoints/scan/main/{token}.json"
        payload, raw_sha = _read_bytes(root / staging_relative, f"staging shard {index}")
        require(len(payload) == rows * RECORD_BYTES, "staging shard byte count differs")
        records = np.frombuffer(payload, dtype=RECORD_DTYPE)
        _validate_record_array(records)
        checkpoint, _ = _read_json(root / checkpoint_relative, f"scan checkpoint {index}")
        expected_checkpoint = {
            "schema_version": "openalex_g1_canonical_work_time_spine_downstream_benchmark_scan_shard.v1",
            "benchmark_shard_index": index,
            "relative_path": logical,
            "staging_shard_relative_path": staging_relative,
            "staging_shard_rows": rows,
            "staging_shard_bytes": len(payload),
            "staging_shard_sha256": raw_sha,
            "row_level_disposition_payload_sha256": _framed_digest(records),
            "operator_paths_redacted": True,
        }
        require(checkpoint == expected_checkpoint, f"scan checkpoint differs: {index}")
        source_digest.update(payload)
        total_rows += len(records)
        partition_ids = np.remainder(records["work_id_numeric"], PARTITION_COUNT).astype(np.uint16, copy=False)
        order = np.argsort(partition_ids, kind="stable")
        ordered_ids = partition_ids[order]
        ordered_records = np.ascontiguousarray(records[order])
        if len(records):
            boundaries = np.flatnonzero(ordered_ids[1:] != ordered_ids[:-1]) + 1
            starts = np.concatenate((np.array([0]), boundaries))
            stops = np.concatenate((boundaries, np.array([len(records)])))
            for start, stop in zip(starts, stops, strict=True):
                partition_id = int(ordered_ids[int(start)])
                partition_payloads[partition_id].extend(memoryview(ordered_records[int(start):int(stop)]).cast("B"))
    require(total_rows == TOTAL_ROWS, "staging row conservation differs")
    counts = [len(payload) // RECORD_BYTES for payload in partition_payloads]
    require(sum(counts) == TOTAL_ROWS, "demultiplex row conservation differs")
    spool_contracts = [
        {
            "partition_id": partition_id,
            "relative_path": f".work_time_spine_tmp/spool/part-{partition_id:04d}.bin",
            "rows": counts[partition_id],
            "bytes": len(partition_payloads[partition_id]),
            "sha256": _sha256_bytes(bytes(partition_payloads[partition_id])),
        }
        for partition_id in range(PARTITION_COUNT)
    ]
    spool_tree = _sha256_bytes(_canonical_json_bytes(spool_contracts))
    return partition_payloads, counts, source_digest.hexdigest(), spool_tree


def _empty_stats(expected_rows: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(len(RECORD_HASH_DOMAIN).to_bytes(8, "little"))
    digest.update(RECORD_HASH_DOMAIN)
    digest.update(expected_rows.to_bytes(8, "little"))
    return {
        "digest": digest,
        "rows": 0,
        "minimum": None,
        "maximum": None,
        "year_minimum": None,
        "year_maximum": None,
        "dispositions": {name: 0 for name in DISPOSITION_NAMES.values()},
        "w0_dispositions": {name: 0 for name in DISPOSITION_NAMES.values()},
        "nulls": {"publication_day_null_rows": 0, "publication_year_null_rows": 0},
    }


def _validate_parquet(path: Path, partition_id: int, expected_rows: int) -> dict[str, Any]:
    descriptor, identity = _open_readonly(path, f"partition Parquet {partition_id}")
    stats = _empty_stats(expected_rows)
    previous: int | None = None
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            parquet = pq.ParquetFile(handle)
            require(parquet.schema_arrow.equals(ARROW_SCHEMA, check_metadata=True), "partition Arrow schema differs")
            require((parquet.metadata.metadata or {}) == {}, "partition file metadata differs")
            require(parquet.metadata.num_rows == expected_rows, "partition footer rows differ")
            for row_group_id in range(parquet.metadata.num_row_groups):
                row_group = parquet.metadata.row_group(row_group_id)
                for column_id in range(row_group.num_columns):
                    column = row_group.column(column_id)
                    require(column.compression == "UNCOMPRESSED", "partition compression differs")
                    require(column.is_stats_set is False, "partition statistics exist")
            for batch in parquet.iter_batches(batch_size=262_144, use_threads=False):
                ids = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.uint64)
                dispositions = np.asarray(batch.column(3).to_numpy(zero_copy_only=False), dtype=np.uint8)
                day_nulls = np.asarray(batch.column(1).is_null().to_numpy(zero_copy_only=False), dtype=np.bool_)
                year_nulls = np.asarray(batch.column(2).is_null().to_numpy(zero_copy_only=False), dtype=np.bool_)
                require(bool(np.array_equal(day_nulls, year_nulls)), "Parquet nullability differs")
                require(bool(np.array_equal(day_nulls, dispositions == 0)), "Parquet disposition/nullability differs")
                days = np.asarray(pc.fill_null(batch.column(1), NULL_DAY).to_numpy(zero_copy_only=False), dtype=np.int32)
                years = np.asarray(pc.fill_null(batch.column(2), 0).to_numpy(zero_copy_only=False), dtype=np.uint16)
                require(bool(np.all(ids % PARTITION_COUNT == partition_id)), "partition modulo differs")
                require(bool(np.all(dispositions <= max(DISPOSITION_NAMES))), "Parquet disposition differs")
                if len(ids):
                    require(previous is None or int(ids[0]) > previous, "partition IDs are not strictly increasing")
                    require(bool(np.all(ids[1:] > ids[:-1])), "partition IDs are not strictly increasing")
                    previous = int(ids[-1])
                    if stats["minimum"] is None:
                        stats["minimum"] = int(ids[0])
                    stats["maximum"] = int(ids[-1])
                if bool(np.any(~day_nulls)):
                    nonnull_years = years[~year_nulls]
                    batch_min = int(np.min(nonnull_years))
                    batch_max = int(np.max(nonnull_years))
                    stats["year_minimum"] = batch_min if stats["year_minimum"] is None else min(stats["year_minimum"], batch_min)
                    stats["year_maximum"] = batch_max if stats["year_maximum"] is None else max(stats["year_maximum"], batch_max)
                    dates = pa.array(days, mask=day_nulls, type=pa.date32())
                    derived = np.asarray(pc.fill_null(pc.year(dates), 0).to_numpy(zero_copy_only=False), dtype=np.int64)
                    require(bool(np.all(derived[~day_nulls] == years[~day_nulls])), "Parquet day/year differs")
                counts = np.bincount(dispositions, minlength=len(DISPOSITION_NAMES))
                require(len(counts) == len(DISPOSITION_NAMES), "Parquet disposition code differs")
                for code, name in DISPOSITION_NAMES.items():
                    stats["dispositions"][name] += int(counts[code])
                    stats["w0_dispositions"][name] += int(np.count_nonzero((ids == 0) & (dispositions == code)))
                null_count = int(np.count_nonzero(day_nulls))
                stats["nulls"]["publication_day_null_rows"] += null_count
                stats["nulls"]["publication_year_null_rows"] += null_count
                encoded = np.empty(len(ids), dtype=RECORD_DTYPE)
                encoded["work_id_numeric"] = ids
                encoded["publication_day"] = days
                encoded["publication_year"] = years
                encoded["temporal_disposition"] = dispositions
                encoded["reserved"] = 0
                _validate_record_array(encoded)
                framed = np.empty(len(encoded), dtype=FRAMED_DTYPE)
                framed["encoded_length"] = RECORD_BYTES
                framed["record"] = encoded.view(f"V{RECORD_BYTES}").reshape(-1)
                stats["digest"].update(memoryview(framed).cast("B"))
                stats["rows"] += len(ids)
        _validate_open(path, descriptor, identity, f"partition Parquet {partition_id}")
    finally:
        os.close(descriptor)
    require(stats["rows"] == expected_rows, "decoded partition row count differs")
    return {
        "rows": stats["rows"],
        "minimum_work_id_numeric": stats["minimum"],
        "maximum_work_id_numeric": stats["maximum"],
        "publication_year_extrema": {"minimum": stats["year_minimum"], "maximum": stats["year_maximum"]},
        "record_sha256": stats["digest"].hexdigest(),
        "disposition_counts": stats["dispositions"],
        "w_zero_disposition_counts": stats["w0_dispositions"],
        "nullability_counts": stats["nulls"],
    }


def _sorted_partition_digest(payload: bytearray) -> tuple[str, int | None, int | None]:
    records = np.frombuffer(payload, dtype=RECORD_DTYPE)
    _validate_record_array(records)
    if len(records):
        order = np.argsort(records["work_id_numeric"], kind="stable")
        records = np.ascontiguousarray(records[order])
        identifiers = records["work_id_numeric"]
        require(bool(np.all(identifiers[1:] > identifiers[:-1])), "staging contains duplicate numeric work IDs")
        minimum = int(identifiers[0])
        maximum = int(identifiers[-1])
    else:
        minimum = maximum = None
    return _framed_digest(records), minimum, maximum


def _validate_partition(
    root: Path,
    partition_id: int,
    payload: bytearray,
    expected_rows: int,
    result: Mapping[str, Any],
    repeat_index: int,
    spool_tree: str,
    observations: Mapping[str, tuple[str, int]],
) -> dict[str, Any]:
    checkpoint_path = root / f"checkpoints/partition/part-{partition_id:04d}.json"
    parquet_relative = f"spine/part-{partition_id:04d}.parquet"
    checkpoint, _ = _read_json(checkpoint_path, f"partition checkpoint {partition_id}")
    require(set(checkpoint) == PARTITION_CHECKPOINT_KEYS, "partition checkpoint key set differs")
    core = dict(checkpoint)
    checkpoint_sha = core.pop("checkpoint_sha256")
    require(checkpoint_sha == _sha256_bytes(_canonical_json_bytes(core)), "partition checkpoint self-hash differs")
    binding = result["pilot_binding"]
    expected_task_fp = _sha256_bytes(_canonical_json_bytes({"schema": BENCHMARK_SCHEMA, "repeat_index": repeat_index, "pilot_task_fingerprint_sha256": binding["task_fingerprint_sha256"]}))
    expected_scan_tree = _sha256_bytes(_canonical_json_bytes({"benchmark_spool_partition_tree_sha256": spool_tree, "repeat_index": repeat_index}))
    parquet_sha, parquet_bytes = observations[parquet_relative]
    decoded = _validate_parquet(root / parquet_relative, partition_id, expected_rows)
    pre_digest, staging_minimum, staging_maximum = _sorted_partition_digest(payload)
    expected_values = {
        "schema_version": "openalex_g1_canonical_work_time_spine_downstream_benchmark_partition_checkpoint.v1",
        "task_fingerprint_sha256": expected_task_fp,
        "runtime_fingerprint": result["runtime"]["runner_runtime_fingerprint"],
        "scan_checkpoint_tree_sha256": expected_scan_tree,
        "staging_shard_tree_sha256": spool_tree,
        "partition_id": partition_id,
        "partition_formula": "work_id_numeric_mod_4096",
        "output_relative_path": parquet_relative,
        "output_rows": expected_rows,
        "minimum_work_id_numeric": decoded["minimum_work_id_numeric"],
        "maximum_work_id_numeric": decoded["maximum_work_id_numeric"],
        "output_schema": SCHEMA_CONTRACT,
        "output_schema_sha256": _sha256_bytes(_canonical_json_bytes(SCHEMA_CONTRACT)),
        "pre_parquet_sorted_record_sha256": pre_digest,
        "decoded_parquet_record_sha256": decoded["record_sha256"],
        "disposition_counts": decoded["disposition_counts"],
        "w_zero_disposition_counts": decoded["w_zero_disposition_counts"],
        "publication_year_extrema": decoded["publication_year_extrema"],
        "nullability_counts": decoded["nullability_counts"],
        "output_bytes": parquet_bytes,
        "output_sha256": parquet_sha,
        "strictly_increasing_ids": True,
        "operator_paths_redacted": True,
    }
    for key, value in expected_values.items():
        require(checkpoint.get(key) == value, f"partition checkpoint field differs: {partition_id}:{key}")
    require(pre_digest == decoded["record_sha256"], "pre-/decoded-Parquet record digest differs")
    require(staging_minimum == decoded["minimum_work_id_numeric"] and staging_maximum == decoded["maximum_work_id_numeric"], "staging/Parquet ID extrema differ")
    return {
        "partition_id": partition_id,
        "rows": expected_rows,
        "bytes": parquet_bytes,
        "sha256": parquet_sha,
        "record_sha256": decoded["record_sha256"],
    }


def _validate_controls(
    root: Path,
    repeat_index: int,
    result: Mapping[str, Any],
    spool_tree: str,
    partition_contracts: Sequence[Mapping[str, Any]],
) -> None:
    repeat = result["repeats"][repeat_index - 1]
    preflight = result["preflight"]
    expected_documents = {
        "benchmark_plan.json": {
            "schema_version": BENCHMARK_SCHEMA,
            "status": "BOUNDED_NONPROMOTABLE_PLAN",
            "repeat_index": repeat_index,
            "scenarios": ["D4096", "R4096", "H9185_BOUNDED_BYTES"],
            "production_execution": PRODUCTION_NOT_RUN,
            "operator_paths_redacted": True,
        },
        "preflight_snapshot.json": preflight,
        "run_state.json": {
            "schema_version": f"{BENCHMARK_SCHEMA}.state",
            "status": "BOUNDED_REPEAT_COMPLETE_NO_PRODUCTION_RECEIPT",
            "repeat_index": repeat_index,
            "production_execution": PRODUCTION_NOT_RUN,
            "operator_paths_redacted": True,
        },
        "staging_manifest.json": {
            "schema_version": "openalex_g1_canonical_work_time_spine_downstream_benchmark_staging_manifest.v1",
            "repeat_index": repeat_index,
            "shard_count": SHARD_COUNT,
            "rows": TOTAL_ROWS,
            "bytes": TOTAL_ROWS * RECORD_BYTES,
            "spool_partition_tree_sha256": spool_tree,
            "operator_paths_redacted": True,
        },
        "spine_manifest.json": {
            "schema_version": "openalex_g1_canonical_work_time_spine_downstream_benchmark_spine_manifest.v1",
            "repeat_index": repeat_index,
            "partition_count": PARTITION_COUNT,
            "rows": TOTAL_ROWS,
            "bytes": sum(int(item["bytes"]) for item in partition_contracts),
            "partition_contract_tree_sha256": _sha256_bytes(_canonical_json_bytes(list(partition_contracts))),
            "operator_paths_redacted": True,
        },
        "benchmark_report.json": {
            "schema_version": f"{BENCHMARK_SCHEMA}.repeat_report",
            "status": BENCHMARK_STATUS,
            "repeat_index": repeat_index,
            "demultiplex_status": repeat["demultiplex"]["status"],
            "representative_status": repeat["representative_single_run"]["status"],
            "production_component_pass": False,
            "programmatic_parent_of_full_receipt": False,
            "operator_paths_redacted": True,
        },
    }
    for relative, expected in expected_documents.items():
        observed, _ = _read_json(root / relative, f"repeat control {relative}")
        require(observed == expected, f"repeat control differs: {relative}")
    markdown, _ = _read_bytes(root / "benchmark_report.md", "repeat Markdown report")
    expected_markdown = (
        "# Bounded downstream benchmark repeat\n\n"
        f"Repeat: {repeat_index}\n\n"
        "Production execution: PRODUCTION_NOT_RUN\n\n"
        "This scratch surface is operator-local and nonpromotable.\n"
    ).encode("utf-8")
    require(markdown == expected_markdown, "repeat Markdown report differs")


def _validate_repeat(
    root: Path,
    repeat_index: int,
    result: Mapping[str, Any],
    pilot_staging_sha: str,
) -> None:
    repeat = result["repeats"][repeat_index - 1]
    require(set(repeat) == {"repeat_index", "status", "preparation", "demultiplex", "representative_single_run", "bounded_artifact_surface", "scratch_surface_retained_for_operator_review", "operator_paths_redacted"}, "repeat result keys differ")
    require(repeat.get("repeat_index") == repeat_index, "repeat index differs")
    require(repeat.get("status") == "PASS_BOUNDED_REPEAT_INTEGRITY", "repeat status differs")
    require(repeat.get("scratch_surface_retained_for_operator_review") is True, "scratch retention differs")
    require(repeat.get("operator_paths_redacted") is True, "repeat paths are not redacted")
    expected_paths = _expected_paths()
    require(len(expected_paths) == EXACT_ARTIFACT_COUNT, "frozen artifact cardinality formula differs")
    files, _ = _inspect_private_tree(root, f"scratch repeat {repeat_index}")
    require(files == expected_paths, "scratch repeat topology differs")
    contracts, total_artifact_bytes = _artifact_contracts(root, expected_paths)
    contract_tree = _sha256_bytes(_canonical_json_bytes(contracts))
    observations = {item["relative_path"]: (item["sha256"], item["bytes"]) for item in contracts}
    payloads, counts, staging_sha, spool_tree = _validate_staging(root)
    require(staging_sha == pilot_staging_sha, "repeat staging concatenation SHA differs from pilot")
    preparation = repeat.get("preparation")
    require(isinstance(preparation, dict), "repeat preparation differs")
    for key, value in {
        "shard_count": SHARD_COUNT,
        "rows": TOTAL_ROWS,
        "bytes": TOTAL_ROWS * RECORD_BYTES,
        "source_sha256": pilot_staging_sha,
        "excluded_from_downstream_phase_timings": True,
    }.items():
        require(preparation.get(key) == value, f"preparation field differs: {key}")
    demux = repeat.get("demultiplex")
    require(isinstance(demux, dict), "demultiplex result differs")
    require(demux.get("scenario") == "D4096", "demultiplex scenario differs")
    require(demux.get("status") == "PASS_BOUNDED_4096_WAY_DEMULTIPLEX_INTEGRITY", "demultiplex status differs")
    for key, value in {"rows": TOTAL_ROWS, "logical_input_bytes": TOTAL_ROWS * RECORD_BYTES, "logical_spool_bytes": TOTAL_ROWS * RECORD_BYTES, "full_corpus_sustained_rate_measured": False}.items():
        require(demux.get(key) == value, f"demultiplex field differs: {key}")
    occupancy = demux.get("occupancy")
    require(isinstance(occupancy, dict), "demultiplex occupancy differs")
    nonempty = [count for count in counts if count]
    expected_occupancy = {
        "partition_count": PARTITION_COUNT,
        "nonempty_partition_count": len(nonempty),
        "empty_partition_count": PARTITION_COUNT - len(nonempty),
        "all_partitions": _distribution([float(value) for value in counts]),
        "nonempty_partitions": _distribution([float(value) for value in nonempty]),
        "partition_rows_sha256": _sha256_bytes(_canonical_json_bytes(counts)),
    }
    require(occupancy == expected_occupancy, "demultiplex occupancy differs")
    partition_contracts = [
        _validate_partition(root, partition_id, payloads[partition_id], counts[partition_id], result, repeat_index, spool_tree, observations)
        for partition_id in range(PARTITION_COUNT)
    ]
    require(sum(item["rows"] for item in partition_contracts) == TOTAL_ROWS, "Parquet row conservation differs")
    parquet_bytes = sum(item["bytes"] for item in partition_contracts)
    representative = repeat.get("representative_single_run")
    require(isinstance(representative, dict), "representative result differs")
    for key, value in {
        "scenario": "R4096",
        "status": "PASS_BOUNDED_PILOT_OCCUPANCY_SINGLE_RUN_PARQUET_INTEGRITY",
        "partition_count": PARTITION_COUNT,
        "rows": TOTAL_ROWS,
        "parquet_bytes": parquet_bytes,
        "run_count_total": len(nonempty),
        "run_count_maximum_per_partition": 1 if nonempty else 0,
        "multi_run_partitions": 0,
        "single_run_partitions": len(nonempty),
        "zero_run_partitions": PARTITION_COUNT - len(nonempty),
        "full_like_partition_occupancy_measured": False,
        "python_multi_run_heap_merge_measured": False,
    }.items():
        require(representative.get(key) == value, f"representative field differs: {key}")
    _validate_controls(root, repeat_index, result, spool_tree, partition_contracts)
    surface = repeat.get("bounded_artifact_surface")
    require(isinstance(surface, dict), "bounded artifact result differs")
    expected_sweeps = result["benchmark_scope"]["artifact_sweeps_per_repeat"]
    for key, value in {
        "scenario": "H9185_BOUNDED_BYTES",
        "status": "PASS_EXACT_9185_ENTRY_TOPOLOGY_AND_HASH_SWEEPS_BOUNDED_BYTES",
        "artifact_count": EXACT_ARTIFACT_COUNT,
        "topology_formula": "493_staging+493_scan_metadata+4096_parquet+4096_partition_metadata+7_control",
        "sweep_count": expected_sweeps,
        "large_full_byte_surface_measured": False,
        "production_finalizer_or_receipt_called": False,
        "cache_condition": "UNCONTROLLED_WARM_OR_CACHE_ASSISTED",
    }.items():
        require(surface.get(key) == value, f"H9185 field differs: {key}")
    sweeps = surface.get("sweeps")
    require(isinstance(sweeps, list) and len(sweeps) == expected_sweeps, "H9185 sweep count differs")
    for sweep_index, sweep in enumerate(sweeps, start=1):
        require(isinstance(sweep, dict), "H9185 sweep differs")
        for key, value in {"sweep_index": sweep_index, "artifact_count": EXACT_ARTIFACT_COUNT, "hashed_bytes": total_artifact_bytes, "artifact_contract_tree_sha256": contract_tree}.items():
            require(sweep.get(key) == value, f"H9185 sweep field differs: {key}")


def _validate_aggregate(result: Mapping[str, Any]) -> None:
    repeats = result["repeats"]
    aggregate = result.get("aggregate")
    require(isinstance(aggregate, dict), "aggregate result differs")
    expected = {
        "repeat_count": REPEAT_COUNT,
        "demultiplex_rows_per_second": _distribution([float(item["demultiplex"]["rows_per_second_wall"]) for item in repeats]),
        "representative_rows_per_second": _distribution([float(item["representative_single_run"]["rows_per_second_wall"]) for item in repeats]),
        "h9185_artifacts_per_second": _distribution([float(sweep["artifacts_per_second"]) for item in repeats for sweep in item["bounded_artifact_surface"]["sweeps"]]),
        "one_sided_95_percent_lower_confidence_bound_computed": False,
        "minimum_observed_rates_are_not_adopted_capacity_floors": True,
        "independent_quiet_window_sessions": 0,
    }
    require(aggregate == expected, "aggregate result does not reconstruct")


def verify_benchmark(
    *, benchmark_result: Path, scratch_root: Path, pilot_output_root: Path
) -> dict[str, Any]:
    result_path = _absolute(benchmark_result)
    scratch = _absolute(scratch_root)
    pilot = _absolute(pilot_output_root)
    require(result_path.name == "openalex_g1_canonical_work_time_spine_downstream_benchmark.json", "benchmark result basename differs")
    result_identity_before = _identity(result_path.lstat())
    scratch_before = _directory_snapshot(scratch)
    pilot_before = _directory_snapshot(pilot)
    result, result_sha256 = _read_json(result_path, "benchmark result")
    _validate_result_contract(result)
    preflight = result.get("preflight")
    require(isinstance(preflight, dict), "result preflight differs")
    _validate_preflight(preflight)
    binding = result.get("pilot_binding")
    require(isinstance(binding, dict), "pilot binding differs")
    pilot_staging_sha = _validate_pilot(pilot, binding)
    expected_repeat_dirs = [f"repeat-{index:02d}" for index in range(1, REPEAT_COUNT + 1)]
    entries = sorted(path.name for path in scratch.iterdir())
    require(entries == expected_repeat_dirs, "scratch root does not contain exactly three repeat directories")
    for index in range(1, REPEAT_COUNT + 1):
        _validate_repeat(scratch / f"repeat-{index:02d}", index, result, pilot_staging_sha)
    _validate_aggregate(result)
    require(_identity(result_path.lstat()) == result_identity_before, "benchmark result changed during verification")
    require(_directory_snapshot(scratch) == scratch_before, "scratch tree changed during verification")
    require(_directory_snapshot(pilot) == pilot_before, "pilot tree changed during verification")
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "status": VERIFICATION_STATUS,
        "verified_benchmark_status": BENCHMARK_STATUS,
        "repeat_count": REPEAT_COUNT,
        "artifacts_per_repeat": EXACT_ARTIFACT_COUNT,
        "verified_total_scratch_artifacts": REPEAT_COUNT * EXACT_ARTIFACT_COUNT,
        "verified_rows_per_repeat": TOTAL_ROWS,
        "benchmark_result_sha256": result_sha256,
        "pilot_staging_sha256": pilot_staging_sha,
        "result_fingerprint_sha256": result["result_fingerprint_sha256"],
        "scratch_tree_unchanged": True,
        "pilot_tree_unchanged": True,
        "benchmark_result_unchanged": True,
        "production_execution_status": PRODUCTION_NOT_RUN,
        "production_component_pass": False,
        "pass_g1": False,
        "downstream_capacity_gate_passed": False,
        "full_authorization_sufficient": False,
        "programmatic_parent_of_full_receipt": False,
        "operator_paths_redacted": True,
        "independence_boundary": "Independent read-only verification; no benchmark or production runner import.",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-result", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--pilot-output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        verification = verify_benchmark(
            benchmark_result=args.benchmark_result,
            scratch_root=args.scratch_root,
            pilot_output_root=args.pilot_output_root,
        )
    except (VerificationError, OSError, ValueError, TypeError, pa.ArrowException) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_published_json_bytes(verification))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
