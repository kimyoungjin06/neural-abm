#!/usr/bin/env python3
"""Independent, repository-only verifier for canonical work-time-spine outputs.

This module deliberately does not import the production runner, its predecessor,
or the frozen YAML configuration.  It independently implements the published v1
terminal-output ABI and opens the output tree read-only.  The default verification
streams every Parquet partition and reconstructs the row-record commitments.

The independence is implementation-level.  It is not trusted-launcher evidence,
loaded-code attestation, source-projection replay, or parent-artifact revalidation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


FULL_SCOPE = "FULL_CURRENT_SNAPSHOT_PRODUCTION"
PILOT_SCOPE = "PRODUCTION_SOURCE_PREFIX_PILOT_UNPROMOTABLE"
CLOSED_FIXTURE_SCOPE = "CLOSED_FIXTURE_STREAMING_ENGINE_ONLY"
PRODUCTION_NOT_RUN = "PRODUCTION_NOT_RUN"
FULL_STATUS = (
    "PASS_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_COMPONENT_WITH_TEMPORAL_V3_PARENT"
)
PILOT_STATUS = "PILOT_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_NONPROMOTABLE"
CLOSED_FIXTURE_STATUS = (
    "PASS_TEST_FIXTURE_CANONICAL_WORK_TIME_SPINE_EXTERNAL_MEMORY_CONTRACT_ONLY"
)
COMPLETION_ABI = "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
VERIFIER_SCHEMA = "openalex_g1_canonical_work_time_spine_independent_verifier_result.v1"
VERIFIER_PASS = "PASS_READ_ONLY_INDEPENDENT_OUTPUT_VERIFICATION"
VERIFIER_HASH_ONLY_PASS = (
    "PASS_READ_ONLY_INDEPENDENT_CONTROL_AND_HASH_VERIFICATION_"
    "PARQUET_PAYLOAD_NOT_DECODED"
)
INDEPENDENCE_BOUNDARY = (
    "Implementation-level independent output verification only; this is not "
    "trusted-launcher evidence, loaded-code attestation, source-projection replay, "
    "or current parent-artifact revalidation."
)

# Independently copied, reviewed anchors for the current production contract.
# This verifier must be versioned whenever any of these anchors changes; it does
# not load the producer, its YAML configuration, or parent artifacts at runtime.
FROZEN_SNAPSHOT_YEAR = 2026
FROZEN_PRE_1900_DAY = -25_567
FROZEN_SNAPSHOT_DAY = 20_542  # 2026-03-30 relative to 1970-01-01.
FULL_TASK_COUNT = 493
FULL_RAW_ROWS = 492_293_207
FULL_RETAINED_ROWS = 492_293_206
FULL_SOURCE_BYTES = 150_062_449_219
FULL_PARTITION_COUNT = 4096
FULL_EXPECTED_ARTIFACT_COUNT = 9185
FULL_SOURCE_TREE_SHA256 = (
    "b201376e677853f8dc996e93c56995f59a7b20499c9ac96faf0a71a2c5219985"
)
FULL_TASK_RECORDS_SHA256 = (
    "5565996276b76ae86c186d883b21d9f765a13e6be1c4ea808b00715ca8e0dedc"
)
PILOT_TASK_COUNT = 1
PILOT_RAW_ROWS = 1_000_000
PILOT_SOURCE_BYTES = 312_250_942
PILOT_SOURCE_TREE_SHA256 = (
    "311c140178aa5aae01b4ac11b92e8faebaf6efa539f32a91784050d73ae585e7"
)
PILOT_TASK_RECORDS_SHA256 = (
    "f3223f37503bff2acf345b0f4dfae0bc46135de75da51fc0ad4d1984cb019682"
)
PRODUCTION_BATCH_SIZE = 262_144
PRODUCTION_SORT_CHUNK_ROWS = 1_048_576
PRODUCTION_PROJECTED_SCHEMA = {
    "id": {"nullable": True, "type": "string"},
    "publication_date": {"nullable": True, "type": "string"},
    "publication_year": {"nullable": True, "type": "double"},
}
PRODUCTION_FROZEN_INPUTS = {
    "config": {
        "path": "experiments/configs/openalex_g1_canonical_work_time_spine_v2.yaml",
        "sha256": "6e8b74ddd12cbef46c8d15d3560f0d54f470b5017685fc1bce2389250cde0484",
    },
    "tool": {
        "path": "scripts/audit_openalex_canonical_work_time_spine_v3.py",
        "sha256": "bf9ed534b2e9f9c2d6c9851be7fde298eb372d8a3cf5e17363ea6afe51cf1280",
    },
    "foundation_config": {
        "path": "experiments/configs/openalex_g1_canonical_work_time_spine_v1.yaml",
        "sha256": "2caef77c8b3dc2ffcb0a35920c9e3527096ec3a9f31d20166eec3a3486f4196b",
    },
    "foundation_synthetic_runner": {
        "path": "scripts/audit_openalex_canonical_work_time_spine_v2.py",
        "sha256": "6a2d1d87dd4a3e7d3c97056fabbe951a1528cfd62d8448acc74ae337f344b7b1",
    },
}
PRODUCTION_PARENTS = {
    "content": {
        "receipt_sha256": "082161a29bcd8fd79fd85e0bf6814a5c7950b97b847ea08056e17eebcd369a12",
        "task_fingerprint_sha256": "f915cade4ce6581832c46957a2ad0a37ceabe2296099f0faeaac56ffa035693e",
        "content_tree_sha256": "73bb8b0409c782ce5964d480ac6c6cecb04bab8c09df31999e581000b79ccff6",
    },
    "temporal": {
        "receipt_sha256": "2908087fc6c96f2aadfdfb08774ed8c3cd54dd38bd54d4e12b1831e7d863ddfe",
        "task_fingerprint_sha256": "5d86a797481a14e95d08ecbe333bb207ff476bf7dc49886aec1db7b5b366001d",
        "result_fingerprint_sha256": "a844c624e81567bf89536218b0c488144fe2730f93c53e72193e6cfe4931bfb5",
        "checkpoint_tree_sha256": "725cf22ce96eca153fa1c9542bd09bd628a4ea1687a2fee3e5610a8cded95392",
        "source_tree_sha256": "da3fa570a5fb49a36668dffdee008c2fea4d4b6fd5a3f7f17873df78b603b2a0",
        "config_sha256": "672940908090fedd0b562dd938ad4583f5e8238830fc1ff5b613be9618f033f7",
        "tool_sha256": "76e25ffb2bb45977f70ac089db1011b4219d6215e546f16cbe11ce1efdb2cc1d",
    },
}
FIXTURE_PARENTS = {
    "content": {"scope": "CLOSED_FIXTURE_ONLY"},
    "temporal": {"scope": "CLOSED_FIXTURE_ONLY"},
}
FULL_RESOURCE_REQUIREMENTS = {
    "staging_bytes_upper_bound": 7_876_691_296,
    "partition_spool_bytes_upper_bound": 7_876_691_296,
    "sorted_run_bytes_upper_bound": 7_876_691_296,
    "parquet_bytes_upper_bound": 16_290_253_504,
    "parquet_bytes_per_row_upper_bound": 32,
    "parquet_fixed_bytes_per_partition_upper_bound": 131_072,
    "control_bytes_upper_bound": 2_147_483_648,
    "atomic_publish_headroom_bytes": 15_753_513_664,
    "disk_safety_multiplier_milli": 1500,
    "disk_reserve_bytes": 8_589_934_592,
    "base_required_free_disk_bytes_full": 95_321_921_648,
    "required_free_disk_bytes_full_fresh": 153_143_246_352,
    "required_free_disk_bytes_full_nonterminal_resume": 95_321_921_648,
    "preferred_free_disk_bytes_full": 171_798_691_840,
    "projected_arrow_batch_bytes_upper_bound": 268_435_456,
    "projected_ipc_serialized_bytes_upper_bound": 268_435_456,
    "encoded_record_batch_bytes_upper_bound": 4_194_304,
    "projected_batch_bytes_upper_bound": 541_065_216,
    "sort_working_bytes_preflight_estimate": 134_217_728,
    "spool_demultiplex_working_bytes_preflight_estimate": 134_217_728,
    "spool_buffer_bytes_upper_bound": 536_870_912,
    "parquet_writer_buffer_bytes_preflight_estimate": 16_777_216,
    "merge_heap_bytes_preflight_estimate": 1_048_576,
    "ram_reserve_bytes": 8_589_934_592,
    "calculated_ram_preflight_sizing_bytes": 9_685_696_512,
    "required_available_ram_bytes": 17_179_869_184,
    "minimum_nofile_soft_limit": 8192,
}
PILOT_RESOURCE_REQUIREMENTS = {
    **FULL_RESOURCE_REQUIREMENTS,
    "staging_bytes_upper_bound": 16_000_000,
    "partition_spool_bytes_upper_bound": 16_000_000,
    "sorted_run_bytes_upper_bound": 16_000_000,
    "parquet_bytes_upper_bound": 568_870_912,
    "base_required_free_disk_bytes_full": 36_366_736_928,
    "required_free_disk_bytes_full_fresh": 36_366_736_928,
    "required_free_disk_bytes_full_nonterminal_resume": 36_366_736_928,
    "preferred_free_disk_bytes_full": 36_366_736_928,
    "required_available_ram_bytes": 9_685_696_512,
}
FULL_DISPOSITION_COUNTS = {
    "BOTH_NULL_EXCLUDED": 27_565_532,
    "VALID_PRE_1900_EXCLUDED": 6_692_869,
    "USABLE_1900_BEFORE_SNAPSHOT_YEAR": 446_514_060,
    "USABLE_SNAPSHOT_YEAR_ON_OR_BEFORE_SNAPSHOT": 11_509_999,
    "VALID_SNAPSHOT_YEAR_AFTER_SNAPSHOT_EXCLUDED": 6_736,
    "VALID_POST_SNAPSHOT_YEAR_EXCLUDED": 4_010,
}

SCHEMAS: dict[str, dict[str, str]] = {
    FULL_SCOPE: {
        "plan": "openalex_g1_canonical_work_time_spine_production_plan.v1",
        "preflight_evidence": "openalex_g1_canonical_work_time_spine_production_preflight_evidence.v1",
        "scan_checkpoint": "openalex_g1_canonical_work_time_spine_production_scan_checkpoint.v1",
        "partition_checkpoint": "openalex_g1_canonical_work_time_spine_production_partition_checkpoint.v1",
        "staging_manifest": "openalex_g1_canonical_work_time_spine_production_staging_manifest.v1",
        "spine_manifest": "openalex_g1_canonical_work_time_spine_production_manifest.v1",
        "report": "openalex_g1_canonical_work_time_spine_production_report.v1",
        "outputs": "openalex_g1_canonical_work_time_spine_production_outputs.v1",
        "state": "openalex_g1_canonical_work_time_spine_production_state.v1",
    },
    PILOT_SCOPE: {
        "plan": "openalex_g1_canonical_work_time_spine_pilot_plan.v1",
        "preflight_evidence": "openalex_g1_canonical_work_time_spine_pilot_preflight_evidence.v1",
        "scan_checkpoint": "openalex_g1_canonical_work_time_spine_pilot_scan_checkpoint.v1",
        "staging_manifest": "openalex_g1_canonical_work_time_spine_pilot_staging_manifest.v1",
        "report": "openalex_g1_canonical_work_time_spine_pilot_report.v1",
        "outputs": "openalex_g1_canonical_work_time_spine_pilot_outputs.v1",
        "state": "openalex_g1_canonical_work_time_spine_pilot_state.v1",
    },
    CLOSED_FIXTURE_SCOPE: {
        "plan": "openalex_g1_canonical_work_time_spine_test_fixture_plan.v1",
        "preflight_evidence": "openalex_g1_canonical_work_time_spine_test_fixture_preflight_evidence.v1",
        "scan_checkpoint": "openalex_g1_canonical_work_time_spine_test_fixture_scan_checkpoint.v1",
        "partition_checkpoint": "openalex_g1_canonical_work_time_spine_test_fixture_partition_checkpoint.v1",
        "staging_manifest": "openalex_g1_canonical_work_time_spine_test_fixture_staging_manifest.v1",
        "spine_manifest": "openalex_g1_canonical_work_time_spine_test_fixture_manifest.v1",
        "report": "openalex_g1_canonical_work_time_spine_test_fixture_report.v1",
        "outputs": "openalex_g1_canonical_work_time_spine_test_fixture_outputs.v1",
        "state": "openalex_g1_canonical_work_time_spine_test_fixture_state.v1",
    },
}
STATUSES = {
    FULL_SCOPE: FULL_STATUS,
    PILOT_SCOPE: PILOT_STATUS,
    CLOSED_FIXTURE_SCOPE: CLOSED_FIXTURE_STATUS,
}
BASENAMES = {
    FULL_SCOPE: {
        "report": "work_time_spine_audit.json",
        "markdown": "work_time_spine_audit.md",
        "receipt": "work_time_spine_audit.outputs.json",
    },
    PILOT_SCOPE: {
        "report": "work_time_spine_pilot.json",
        "markdown": "work_time_spine_pilot.md",
        "receipt": "work_time_spine_pilot.outputs.json",
    },
    CLOSED_FIXTURE_SCOPE: {
        "report": "work_time_spine_test_fixture.json",
        "markdown": "work_time_spine_test_fixture.md",
        "receipt": "work_time_spine_test_fixture.outputs.json",
    },
}

FINAL_DIRECTORIES = {
    "checkpoints",
    "checkpoints/scan",
    "checkpoints/scan/main",
    "checkpoints/partition",
    "staging",
    "staging/main",
    "spine",
}
RECORD_BYTES = 16
RECORD_HASH_DOMAIN = b"openalex-work-time-records-v1"
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
FRAMED_RECORD_DTYPE = np.dtype(
    [("encoded_length", "<u8"), ("record", "V16")], align=False
)
EXPECTED_ARROW_SCHEMA = pa.schema(
    [
        pa.field("work_id_numeric", pa.uint64(), nullable=False),
        pa.field("publication_day", pa.int32(), nullable=True),
        pa.field("publication_year", pa.uint16(), nullable=True),
        pa.field("temporal_disposition", pa.uint8(), nullable=False),
    ],
    metadata=None,
)
EXPECTED_SCHEMA_CONTRACT = [
    {"name": field.name, "type": str(field.type), "nullable": field.nullable}
    for field in EXPECTED_ARROW_SCHEMA
]
DISPOSITION_NAMES = {
    0: "BOTH_NULL_EXCLUDED",
    1: "VALID_PRE_1900_EXCLUDED",
    2: "USABLE_1900_BEFORE_SNAPSHOT_YEAR",
    3: "USABLE_SNAPSHOT_YEAR_ON_OR_BEFORE_SNAPSHOT",
    4: "VALID_SNAPSHOT_YEAR_AFTER_SNAPSHOT_EXCLUDED",
    5: "VALID_POST_SNAPSHOT_YEAR_EXCLUDED",
}
SPINE_METRIC_KEYS = {
    "raw_rows",
    "quarantined_w_minus_1_rows",
    "retained_rows",
    "null_id_rows",
    "grammar_valid_id_rows",
    "unexpected_malformed_id_rows",
    "w_zero_rows",
    "publication_both_null_rows",
    "publication_exactly_one_null_rows",
    "publication_both_nonnull_rows",
    "publication_date_parse_error_rows",
    "publication_year_invalid_rows",
    "publication_concordance_checked_rows",
    "publication_date_year_match_rows",
    "publication_date_year_mismatch_rows",
    "publication_pre_1900_rows",
    "publication_1900_before_snapshot_year_rows",
    "publication_snapshot_year_on_or_before_snapshot_rows",
    "publication_snapshot_year_after_snapshot_rows",
    "publication_post_snapshot_year_rows",
    "publication_year_null_rows",
    "publication_year_pre_1900_rows",
    "publication_year_snapshot_year_rows",
    "publication_year_post_snapshot_year_rows",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
PLAN_KEYS = {
    "batch_size",
    "config_relative_path",
    "config_rich_identity",
    "config_sha256",
    "expected_receipt_artifact_count",
    "external_sort_algorithm",
    "external_sort_record_dtype",
    "foundation_config_relative_path",
    "foundation_config_rich_identity",
    "foundation_config_sha256",
    "foundation_synthetic_runner_relative_path",
    "foundation_synthetic_runner_rich_identity",
    "foundation_synthetic_runner_sha256",
    "frozen_file_snapshot_is_loaded_code_attestation",
    "operator_paths_redacted",
    "own_frozen_input_revalidation_required",
    "parents",
    "partition_count",
    "partition_formula",
    "partition_sort_chunk_rows",
    "production_status_before_receipt",
    "resource_requirements",
    "runtime_fingerprint",
    "schema_version",
    "scope",
    "source_tree_sha256",
    "status_if_complete",
    "task_fingerprint_sha256",
    "tasks",
    "tool_relative_path",
    "tool_rich_identity",
    "tool_sha256",
}
TASK_KEYS = {
    "file_index",
    "footer_rows",
    "projected_columns",
    "projected_schema",
    "projected_schema_sha256",
    "relative_path",
    "source_rich_identity",
    "temporal_parent_checkpoint_sha256",
    "temporal_parent_task_sha256",
}
RICH_IDENTITY_KEYS = {
    "device",
    "inode",
    "mode",
    "uid",
    "gid",
    "links",
    "bytes",
    "mtime_ns",
    "ctime_ns",
}
RUNTIME_KEYS = {
    "numpy_version",
    "parquet_cpp_version",
    "platform_machine",
    "pyarrow_version",
    "python_implementation",
    "python_version",
}
RESOURCE_KEYS = {
    "atomic_publish_headroom_bytes",
    "base_required_free_disk_bytes_full",
    "calculated_ram_preflight_sizing_bytes",
    "control_bytes_upper_bound",
    "disk_reserve_bytes",
    "disk_safety_multiplier_milli",
    "encoded_record_batch_bytes_upper_bound",
    "merge_heap_bytes_preflight_estimate",
    "minimum_nofile_soft_limit",
    "parquet_bytes_per_row_upper_bound",
    "parquet_bytes_upper_bound",
    "parquet_fixed_bytes_per_partition_upper_bound",
    "parquet_writer_buffer_bytes_preflight_estimate",
    "partition_spool_bytes_upper_bound",
    "preferred_free_disk_bytes_full",
    "projected_arrow_batch_bytes_upper_bound",
    "projected_batch_bytes_upper_bound",
    "projected_ipc_serialized_bytes_upper_bound",
    "ram_reserve_bytes",
    "required_available_ram_bytes",
    "required_free_disk_bytes_full_fresh",
    "required_free_disk_bytes_full_nonterminal_resume",
    "sort_working_bytes_preflight_estimate",
    "sorted_run_bytes_upper_bound",
    "spool_buffer_bytes_upper_bound",
    "spool_demultiplex_working_bytes_preflight_estimate",
    "staging_bytes_upper_bound",
}
SCAN_CHECKPOINT_KEYS = {
    "checkpoint_sha256",
    "disposition_counts",
    "disposition_metrics",
    "disposition_metrics_sha256",
    "file_index",
    "footer_rows",
    "operator_paths_redacted",
    "projected_payload_sha256",
    "projected_schema_sha256",
    "publication_year_extrema",
    "relative_path",
    "row_level_disposition_payload_sha256",
    "schema_version",
    "source_rich_identity",
    "staging_shard_bytes",
    "staging_shard_relative_path",
    "staging_shard_rows",
    "staging_shard_sha256",
    "task_fingerprint_sha256",
    "temporal_parent_checkpoint_sha256",
    "temporal_parent_task_sha256",
}
PARTITION_CHECKPOINT_KEYS = {
    "checkpoint_sha256",
    "decoded_parquet_record_sha256",
    "disposition_counts",
    "maximum_work_id_numeric",
    "minimum_work_id_numeric",
    "nullability_counts",
    "operator_paths_redacted",
    "output_bytes",
    "output_relative_path",
    "output_rows",
    "output_schema",
    "output_schema_sha256",
    "output_sha256",
    "partition_formula",
    "partition_id",
    "pre_parquet_sorted_record_sha256",
    "publication_year_extrema",
    "runtime_fingerprint",
    "scan_checkpoint_tree_sha256",
    "schema_version",
    "staging_shard_tree_sha256",
    "strictly_increasing_ids",
    "task_fingerprint_sha256",
    "w_zero_disposition_counts",
}
REPORT_KEYS = {
    "claim_boundary",
    "gate_status",
    "identifier_accounting",
    "parents",
    "partition_checkpoint_tree_sha256",
    "pass_g1",
    "production_component_pass",
    "provenance",
    "publication_accounting",
    "runtime_fingerprint",
    "scan_checkpoint_tree_sha256",
    "schema_version",
    "scope",
    "source_tree_sha256",
    "spine_data_tree_sha256",
    "staging_shard_tree_sha256",
    "status",
    "task_fingerprint_sha256",
}


class VerificationError(RuntimeError):
    """The terminal output does not satisfy the independent ABI checks."""


def _fail(message: str, cause: BaseException | None = None) -> NoReturn:
    if cause is None:
        raise VerificationError(message)
    raise VerificationError(message) from cause


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return dict(value)


def _require_list(value: Any, field: str) -> list[Any]:
    _require(isinstance(value, list), f"{field} must be an array")
    return list(value)


def _require_int(value: Any, field: str) -> int:
    _require(type(value) is int, f"{field} must be an integer")
    return int(value)


def _require_sha256(value: Any, field: str) -> str:
    _require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{field} must be a lowercase SHA-256",
    )
    return str(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _published_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tree_digest(items: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(items)))


def _safe_relative(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value != "", f"{field} must be text")
    _require("\\" not in value, f"{field} contains a backslash")
    path = PurePosixPath(value)
    _require(not path.is_absolute(), f"{field} must be relative")
    _require(path.parts and path != PurePosixPath("."), f"{field} is empty")
    _require(".." not in path.parts, f"{field} contains '..'")
    _require(path.as_posix() == value, f"{field} is not normalized POSIX text")
    return value


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


def _assert_no_symlink_components(path: Path, field: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except OSError as exc:
            _fail(f"cannot inspect {field}", exc)
        _require(not stat.S_ISLNK(info.st_mode), f"{field} contains a symlink")


def _open_readonly(path: Path, field: str) -> tuple[int, tuple[int, ...]]:
    _assert_no_symlink_components(path, field)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        _fail(f"cannot open {field} read-only", exc)
    try:
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode), f"{field} is not a regular file")
        _require(info.st_nlink == 1, f"{field} is hard-linked")
        _require(stat.S_IMODE(info.st_mode) == 0o600, f"{field} mode is not 0600")
        _require(info.st_uid == os.geteuid(), f"{field} owner differs")
        return descriptor, _identity(info)
    except BaseException:
        os.close(descriptor)
        raise


def _validate_open_identity(
    path: Path, descriptor: int, expected: tuple[int, ...], field: str
) -> None:
    try:
        descriptor_identity = _identity(os.fstat(descriptor))
        path_identity = _identity(path.lstat())
    except OSError as exc:
        _fail(f"cannot revalidate {field}", exc)
    _require(descriptor_identity == expected, f"open {field} changed during read")
    _require(path_identity == expected, f"{field} path changed during read")


def _read_bytes(path: Path, field: str) -> tuple[bytes, str]:
    descriptor, identity = _open_readonly(path, field)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        _validate_open_identity(path, descriptor, identity, field)
    finally:
        os.close(descriptor)
    return payload, _sha256_bytes(payload)


def _hash_file(path: Path, field: str) -> tuple[str, int]:
    descriptor, identity = _open_readonly(path, field)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while payload := handle.read(16 << 20):
                digest.update(payload)
        _validate_open_identity(path, descriptor, identity, field)
    finally:
        os.close(descriptor)
    return digest.hexdigest(), identity[6]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, field: str) -> tuple[dict[str, Any], str]:
    payload, digest = _read_bytes(path, field)
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: _fail(f"{field} contains {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot decode {field}", exc)
    mapping = _require_dict(value, field)
    _require(
        payload == _published_json_bytes(mapping),
        f"{field} bytes are not canonical pretty JSON plus LF",
    )
    return mapping, digest


def _tree_snapshot(root: Path) -> dict[str, tuple[int, ...]]:
    snapshot = {".": _identity(root.lstat())}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        snapshot[path.relative_to(root).as_posix()] = _identity(path.lstat())
    return snapshot


def _inspect_tree(root: Path) -> tuple[set[str], set[str]]:
    _assert_no_symlink_components(root, "output root")
    try:
        root_info = root.lstat()
    except OSError as exc:
        _fail("cannot inspect output root", exc)
    _require(stat.S_ISDIR(root_info.st_mode), "output root is not a directory")
    _require(not stat.S_ISLNK(root_info.st_mode), "output root is a symlink")
    _require(stat.S_IMODE(root_info.st_mode) == 0o700, "output root mode is not 0700")
    _require(root_info.st_uid == os.geteuid(), "output root owner differs")
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        _require(not stat.S_ISLNK(info.st_mode), "output tree contains a symlink")
        _require(info.st_uid == os.geteuid(), f"output entry owner differs: {relative}")
        if stat.S_ISREG(info.st_mode):
            _require(info.st_nlink == 1, f"output file is hard-linked: {relative}")
            _require(
                stat.S_IMODE(info.st_mode) == 0o600,
                f"output file mode differs: {relative}",
            )
            files.add(relative)
        elif stat.S_ISDIR(info.st_mode):
            _require(
                stat.S_IMODE(info.st_mode) == 0o700,
                f"output directory mode differs: {relative}",
            )
            directories.add(relative)
        else:
            _fail(f"output tree contains a non-file entry: {relative}")
    return files, directories


def _assert_redacted(value: Any, field: str = "control") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_redacted(item, f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_redacted(item, f"{field}[{index}]")
    elif isinstance(value, str):
        _require(not value.startswith("/"), f"absolute operator path leaked at {field}")
        _require(
            re.search(r"(^|[\s=:])/(?!/)(?:[^\s]+)", value) is None,
            f"embedded absolute operator path leaked at {field}",
        )
        _require(
            re.match(r"^[A-Za-z]:[\\/]", value) is None,
            f"absolute operator path leaked at {field}",
        )


def _validate_recorded_identity(
    value: Any, field: str, *, require_nonempty: bool
) -> dict[str, int]:
    identity = _require_dict(value, field)
    _require(set(identity) == RICH_IDENTITY_KEYS, f"{field} keys differ")
    _require(
        all(type(item) is int for item in identity.values()),
        f"{field} value is not an integer",
    )
    for key in ("device", "inode", "uid", "gid", "bytes", "mtime_ns", "ctime_ns"):
        _require(identity[key] >= 0, f"{field}.{key} is negative")
    _require(stat.S_ISREG(identity["mode"]), f"{field} is not regular")
    _require(identity["links"] >= 1, f"{field} has no links")
    if require_nonempty:
        _require(identity["bytes"] > 0, f"{field} is empty")
    return {str(key): int(item) for key, item in identity.items()}


def _validate_parent_contract(scope: str, value: Any) -> dict[str, Any]:
    parents = _require_dict(value, "plan parents")
    expected = FIXTURE_PARENTS if scope == CLOSED_FIXTURE_SCOPE else PRODUCTION_PARENTS
    _require(
        parents == expected,
        "plan parent mapping differs from the frozen verifier anchors",
    )
    if scope != CLOSED_FIXTURE_SCOPE:
        for family, contract in parents.items():
            for key, item in _require_dict(contract, f"plan parent {family}").items():
                _require_sha256(item, f"plan parent {family}.{key}")
    return parents


def _validate_production_frozen_inputs(plan: Mapping[str, Any]) -> None:
    fields = {
        "config": ("config_relative_path", "config_sha256", "config_rich_identity"),
        "tool": ("tool_relative_path", "tool_sha256", "tool_rich_identity"),
        "foundation_config": (
            "foundation_config_relative_path",
            "foundation_config_sha256",
            "foundation_config_rich_identity",
        ),
        "foundation_synthetic_runner": (
            "foundation_synthetic_runner_relative_path",
            "foundation_synthetic_runner_sha256",
            "foundation_synthetic_runner_rich_identity",
        ),
    }
    for name, (path_key, sha_key, identity_key) in fields.items():
        anchor = PRODUCTION_FROZEN_INPUTS[name]
        _require(plan.get(path_key) == anchor["path"], f"plan {name} path differs")
        _require(plan.get(sha_key) == anchor["sha256"], f"plan {name} hash differs")
        _validate_recorded_identity(
            plan.get(identity_key), f"plan {name} identity", require_nonempty=True
        )


def _validate_frozen_scope_plan(
    plan: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]], scope: str
) -> None:
    if scope == CLOSED_FIXTURE_SCOPE:
        return
    _validate_production_frozen_inputs(plan)
    _require(
        plan.get("batch_size") == PRODUCTION_BATCH_SIZE,
        "production/pilot batch size differs",
    )
    _require(
        plan.get("partition_sort_chunk_rows") == PRODUCTION_SORT_CHUNK_ROWS,
        "production/pilot sort chunk differs",
    )
    task_digest = _sha256_bytes(_canonical_json_bytes(list(tasks)))
    resources = _require_dict(plan.get("resource_requirements"), "plan resources")
    if scope == FULL_SCOPE:
        _require(len(tasks) == FULL_TASK_COUNT, "full plan task count differs")
        _require(
            task_digest == FULL_TASK_RECORDS_SHA256,
            "full plan task records differ from the frozen parent-derived anchor",
        )
        _require(
            plan.get("source_tree_sha256") == FULL_SOURCE_TREE_SHA256,
            "full plan source tree differs from the frozen anchor",
        )
        _require(
            sum(int(task["footer_rows"]) for task in tasks) == FULL_RAW_ROWS,
            "full plan footer-row total differs",
        )
        _require(
            sum(int(task["source_rich_identity"]["bytes"]) for task in tasks)
            == FULL_SOURCE_BYTES,
            "full plan source-byte total differs",
        )
        _require(
            resources == FULL_RESOURCE_REQUIREMENTS,
            "full plan resource contract differs from the frozen anchor",
        )
        _require(
            plan.get("expected_receipt_artifact_count") == FULL_EXPECTED_ARTIFACT_COUNT,
            "full plan artifact-count anchor differs",
        )
    else:
        _require(scope == PILOT_SCOPE, "unknown frozen execution scope")
        _require(len(tasks) == PILOT_TASK_COUNT, "pilot plan task count differs")
        _require(
            task_digest == PILOT_TASK_RECORDS_SHA256,
            "pilot plan task differs from the frozen lexical-prefix anchor",
        )
        _require(
            plan.get("source_tree_sha256") == PILOT_SOURCE_TREE_SHA256,
            "pilot plan source tree differs from the frozen anchor",
        )
        _require(
            sum(int(task["footer_rows"]) for task in tasks) == PILOT_RAW_ROWS,
            "pilot plan footer-row total differs",
        )
        _require(
            sum(int(task["source_rich_identity"]["bytes"]) for task in tasks)
            == PILOT_SOURCE_BYTES,
            "pilot plan source-byte total differs",
        )
        _require(
            resources == PILOT_RESOURCE_REQUIREMENTS,
            "pilot plan resource contract differs from the frozen anchor",
        )


def _validate_plan(plan: Mapping[str, Any]) -> tuple[str, int, list[dict[str, Any]]]:
    _require(set(plan) == PLAN_KEYS, "plan key set differs from the v1 ABI")
    scope = str(plan.get("scope"))
    _require(scope in SCHEMAS, "plan scope is unknown")
    schemas = SCHEMAS[scope]
    _require(plan.get("schema_version") == schemas["plan"], "plan schema differs")
    _require(plan.get("status_if_complete") == STATUSES[scope], "plan status differs")
    _require(
        plan.get("production_status_before_receipt") == PRODUCTION_NOT_RUN,
        "plan pre-receipt status differs",
    )
    _require(plan.get("operator_paths_redacted") is True, "plan redaction flag differs")
    _require(
        plan.get("frozen_file_snapshot_is_loaded_code_attestation") is False,
        "plan overclaims loaded-code attestation",
    )
    _validate_parent_contract(scope, plan.get("parents"))
    plan_core = dict(plan)
    fingerprint = plan_core.pop("task_fingerprint_sha256", None)
    _require_sha256(fingerprint, "plan task fingerprint")
    _require(
        fingerprint == _sha256_bytes(_canonical_json_bytes(plan_core)),
        "plan task fingerprint does not reconstruct",
    )
    runtime = _require_dict(plan.get("runtime_fingerprint"), "plan runtime fingerprint")
    _require(set(runtime) == RUNTIME_KEYS, "plan runtime fingerprint key set differs")
    _require(
        all(isinstance(value, str) and value != "" for value in runtime.values()),
        "plan runtime fingerprint value is not nonempty text",
    )
    resources = _require_dict(plan.get("resource_requirements"), "plan resources")
    _require(set(resources) == RESOURCE_KEYS, "plan resource key set differs")
    _require(
        all(type(value) is int and value >= 0 for value in resources.values()),
        "plan resource value is not a nonnegative integer",
    )
    _require(
        plan.get("external_sort_algorithm") == "numpy_argsort_work_id_numeric_stable",
        "plan sort algorithm differs",
    )
    _require(
        plan.get("external_sort_record_dtype")
        == {
            "description": "<u8,<i4,<u2,u1,u1",
            "itemsize": RECORD_BYTES,
            "field_offsets": {
                "work_id_numeric": 0,
                "publication_day": 8,
                "publication_year": 12,
                "temporal_disposition": 14,
                "reserved": 15,
            },
        },
        "plan external-sort record ABI differs",
    )
    _require(
        _require_int(plan.get("batch_size"), "plan batch size") > 0,
        "plan batch size is not positive",
    )
    _require(
        _require_int(plan.get("partition_sort_chunk_rows"), "plan sort chunk") > 0,
        "plan sort chunk is not positive",
    )
    _require_sha256(plan.get("config_sha256"), "plan config hash")
    _require_sha256(plan.get("tool_sha256"), "plan tool hash")
    _require_sha256(plan.get("foundation_config_sha256"), "plan foundation config hash")
    _require_sha256(
        plan.get("foundation_synthetic_runner_sha256"), "plan predecessor hash"
    )
    tasks = [
        _require_dict(item, "plan task")
        for item in _require_list(plan.get("tasks"), "plan tasks")
    ]
    _require(bool(tasks), "plan has no tasks")
    relative_paths: list[str] = []
    for index, task in enumerate(tasks):
        _require(set(task) == TASK_KEYS, "plan task key set differs")
        _require(
            task.get("file_index") == index, "plan task indexes are not contiguous"
        )
        relative = _safe_relative(task.get("relative_path"), "plan task path")
        relative_paths.append(relative)
        footer_rows = _require_int(task.get("footer_rows"), "plan task footer rows")
        _require(footer_rows >= 0, "plan task footer rows are negative")
        _validate_recorded_identity(
            task.get("source_rich_identity"),
            "plan source identity",
            require_nonempty=False,
        )
        _require(
            task.get("projected_columns")
            == ["id", "publication_date", "publication_year"],
            "plan projected columns differ",
        )
        projected = _require_dict(task.get("projected_schema"), "plan projected schema")
        if scope != CLOSED_FIXTURE_SCOPE:
            _require(
                projected == PRODUCTION_PROJECTED_SCHEMA,
                "production/pilot projected schema differs",
            )
        _require(
            task.get("projected_schema_sha256")
            == _sha256_bytes(_canonical_json_bytes(projected)),
            "plan projected schema hash differs",
        )
        _require_sha256(task.get("temporal_parent_task_sha256"), "temporal task hash")
        _require_sha256(
            task.get("temporal_parent_checkpoint_sha256"),
            "temporal checkpoint hash",
        )
    _require(len(set(relative_paths)) == len(relative_paths), "plan task paths repeat")
    _require(
        relative_paths == sorted(relative_paths),
        "plan task paths are not strictly lexical",
    )
    expected_source_tree = _sha256_bytes(
        _canonical_json_bytes(
            [
                {
                    "relative_path": task["relative_path"],
                    "source_rich_identity": task["source_rich_identity"],
                }
                for task in tasks
            ]
        )
    )
    _require(
        plan.get("source_tree_sha256") == expected_source_tree,
        "plan source tree differs",
    )
    partition_count = _require_int(plan.get("partition_count"), "plan partition count")
    if scope == FULL_SCOPE:
        _require(partition_count == 4096, "production partition count is not 4096")
        _require(
            plan.get("partition_formula") == "work_id_numeric_mod_4096",
            "production partition formula differs",
        )
    elif scope == PILOT_SCOPE:
        _require(
            partition_count == 0 and plan.get("partition_formula") is None,
            "pilot partition contract differs",
        )
    else:
        _require(
            0 < partition_count <= 4096,
            "fixture partition count is outside its bounded test range",
        )
        _require(
            plan.get("partition_formula") == "work_id_numeric_mod_4096",
            "fixture partition formula differs",
        )
    if scope == CLOSED_FIXTURE_SCOPE:
        _require(
            plan.get("own_frozen_input_revalidation_required") is False,
            "fixture own-input policy differs",
        )
        _require(
            plan.get("config_relative_path") is None,
            "fixture config path must be absent",
        )
        _require(plan.get("config_sha256") == "0" * 64, "fixture config hash differs")
        _require(
            plan.get("config_rich_identity") == {}, "fixture config identity differs"
        )
    else:
        _require(
            plan.get("own_frozen_input_revalidation_required") is True,
            "production own-input policy differs",
        )
        for key in (
            "config_relative_path",
            "foundation_config_relative_path",
            "foundation_synthetic_runner_relative_path",
            "tool_relative_path",
        ):
            _safe_relative(plan.get(key), f"plan {key}")
        for key in (
            "config_rich_identity",
            "foundation_config_rich_identity",
            "foundation_synthetic_runner_rich_identity",
            "tool_rich_identity",
        ):
            identity = _require_dict(plan.get(key), f"plan {key}")
            _require(set(identity) == RICH_IDENTITY_KEYS, f"plan {key} keys differ")
    expected_count = (
        6 + 2 * len(tasks)
        if scope == PILOT_SCOPE
        else 7 + 2 * len(tasks) + 2 * partition_count
    )
    _require(
        plan.get("expected_receipt_artifact_count") == expected_count,
        "plan receipt artifact count differs",
    )
    _validate_frozen_scope_plan(plan, tasks, scope)
    return scope, partition_count, tasks


def _expected_artifact_paths(
    scope: str, partition_count: int, tasks: Sequence[Mapping[str, Any]]
) -> set[str]:
    names = BASENAMES[scope]
    paths = {
        "audit_plan.json",
        "preflight_evidence.json",
        "run_state.json",
        "staging_manifest.json",
        names["report"],
        names["markdown"],
    }
    for task in tasks:
        token = _sha256_bytes(str(task["relative_path"]).encode("utf-8"))
        paths.add(f"checkpoints/scan/main/{token}.json")
        paths.add(f"staging/main/{token}.bin")
    if scope != PILOT_SCOPE:
        paths.add("work_time_spine_manifest.json")
        for partition_id in range(partition_count):
            paths.add(f"checkpoints/partition/part-{partition_id:04d}.json")
            paths.add(f"spine/part-{partition_id:04d}.parquet")
    return paths


def _validate_preflight(
    evidence: Mapping[str, Any], plan: Mapping[str, Any], root: Path
) -> None:
    scope = str(plan["scope"])
    required_keys = {
        "schema_version",
        "scope",
        "task_fingerprint_sha256",
        "invocation_kind",
        "selected_required_available_disk_bytes",
        "observed_free_disk_bytes",
        "required_available_ram_bytes",
        "observed_available_ram_bytes",
        "required_nofile_soft_limit",
        "observed_nofile_soft_limit",
        "atomic_rename_fsync_readback_probe",
        "source_st_dev",
        "output_st_dev",
        "source_output_device_gate",
        "latest_nonterminal_invocation_only",
        "operator_paths_redacted",
    }
    _require(set(evidence) == required_keys, "preflight key set differs")
    _require(
        evidence.get("schema_version") == SCHEMAS[scope]["preflight_evidence"],
        "preflight schema differs",
    )
    _require(evidence.get("scope") == scope, "preflight scope differs")
    _require(
        evidence.get("task_fingerprint_sha256") == plan["task_fingerprint_sha256"],
        "preflight task binding differs",
    )
    _require(
        evidence.get("invocation_kind") in {"FRESH", "NONTERMINAL_RESUME"},
        "preflight invocation differs",
    )
    requirements = _require_dict(plan.get("resource_requirements"), "plan resources")
    disk_key = (
        "required_free_disk_bytes_full_fresh"
        if evidence["invocation_kind"] == "FRESH"
        else "required_free_disk_bytes_full_nonterminal_resume"
    )
    selected_disk = _require_int(requirements.get(disk_key), f"plan {disk_key}")
    _require(
        evidence.get("selected_required_available_disk_bytes") == selected_disk,
        "preflight selected disk differs",
    )
    _require(
        _require_int(evidence.get("observed_free_disk_bytes"), "preflight disk")
        >= selected_disk,
        "recorded disk gate did not pass",
    )
    required_ram = _require_int(
        requirements.get("required_available_ram_bytes"), "plan required RAM"
    )
    _require(
        evidence.get("required_available_ram_bytes") == required_ram,
        "preflight RAM requirement differs",
    )
    _require(
        _require_int(evidence.get("observed_available_ram_bytes"), "preflight RAM")
        >= required_ram,
        "recorded RAM gate did not pass",
    )
    required_nofile = _require_int(
        requirements.get("minimum_nofile_soft_limit"), "plan nofile"
    )
    _require(
        evidence.get("required_nofile_soft_limit") == required_nofile,
        "preflight nofile requirement differs",
    )
    _require(
        _require_int(evidence.get("observed_nofile_soft_limit"), "preflight nofile")
        >= required_nofile,
        "recorded nofile gate did not pass",
    )
    _require(
        evidence.get("atomic_rename_fsync_readback_probe") == "PASS",
        "preflight atomic probe differs",
    )
    source_device = _require_int(
        evidence.get("source_st_dev"), "preflight source device"
    )
    output_device = _require_int(
        evidence.get("output_st_dev"), "preflight output device"
    )
    _require(source_device >= 0 and output_device >= 0, "preflight device is negative")
    _require(output_device == root.stat().st_dev, "recorded output device differs")
    if scope == CLOSED_FIXTURE_SCOPE:
        _require(
            evidence.get("source_output_device_gate") == "EXEMPT_CLOSED_TEST_FIXTURE",
            "fixture device gate differs",
        )
    else:
        _require(
            evidence.get("source_output_device_gate") == "PASS_DISTINCT_DEVICES",
            "production device gate differs",
        )
        _require(source_device != output_device, "production devices are not distinct")
    _require(
        evidence.get("latest_nonterminal_invocation_only") is True,
        "preflight lifecycle flag differs",
    )
    _require(
        evidence.get("operator_paths_redacted") is True,
        "preflight redaction flag differs",
    )


def _update_framed_digest(digest: Any, records: np.ndarray[Any, Any]) -> None:
    framed = np.empty(len(records), dtype=FRAMED_RECORD_DTYPE)
    framed["encoded_length"] = RECORD_BYTES
    framed["record"] = records.view("V16").reshape(-1)
    digest.update(memoryview(framed).cast("B"))


def _empty_record_stats(rows: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(len(RECORD_HASH_DOMAIN).to_bytes(8, "little"))
    digest.update(RECORD_HASH_DOMAIN)
    digest.update(rows.to_bytes(8, "little"))
    return {
        "digest": digest,
        "rows": 0,
        "minimum": None,
        "maximum": None,
        "publication_year_minimum": None,
        "publication_year_maximum": None,
        "disposition_counts": {name: 0 for name in DISPOSITION_NAMES.values()},
        "w_zero_disposition_counts": {name: 0 for name in DISPOSITION_NAMES.values()},
        "publication_day_null_rows": 0,
        "publication_year_null_rows": 0,
    }


def _consume_records(stats: dict[str, Any], records: np.ndarray[Any, Any]) -> None:
    if not len(records):
        return
    _require(records.dtype == RECORD_DTYPE, "record dtype differs")
    dispositions = records["temporal_disposition"]
    _require(bool(np.all(records["reserved"] == 0)), "record reserved byte is nonzero")
    _require(
        bool(np.all(dispositions <= max(DISPOSITION_NAMES))),
        "record disposition is unknown",
    )
    nulls = dispositions == 0
    _require(
        bool(np.all(records["publication_day"][nulls] == -(2**31))),
        "null record day sentinel differs",
    )
    _require(
        bool(np.all(records["publication_year"][nulls] == 0)),
        "null record year sentinel differs",
    )
    _require(
        bool(np.all(records["publication_day"][~nulls] != -(2**31))),
        "nonnull record uses day sentinel",
    )
    _require(
        bool(np.all(records["publication_year"][~nulls] != 0)),
        "nonnull record uses year sentinel",
    )
    expected_dispositions = np.zeros(len(records), dtype=np.uint8)
    if bool(np.any(~nulls)):
        days = records["publication_day"][~nulls].astype(np.int64)
        years = records["publication_year"][~nulls].astype(np.int64)
        derived_years = (
            records["publication_day"][~nulls]
            .astype("datetime64[D]")
            .astype("datetime64[Y]")
            .astype(np.int64)
            + 1970
        )
        _require(
            bool(np.all((years >= 1) & (years <= 9999))),
            "record publication year is outside 1..9999",
        )
        _require(
            bool(np.array_equal(derived_years, years)),
            "record publication day/year relation differs",
        )
        expected_nonnull = np.full(len(days), 5, dtype=np.uint8)
        expected_nonnull[days < FROZEN_PRE_1900_DAY] = 1
        expected_nonnull[
            (days >= FROZEN_PRE_1900_DAY) & (years < FROZEN_SNAPSHOT_YEAR)
        ] = 2
        expected_nonnull[
            (years == FROZEN_SNAPSHOT_YEAR) & (days <= FROZEN_SNAPSHOT_DAY)
        ] = 3
        expected_nonnull[
            (years == FROZEN_SNAPSHOT_YEAR) & (days > FROZEN_SNAPSHOT_DAY)
        ] = 4
        expected_dispositions[~nulls] = expected_nonnull
    _require(
        bool(np.array_equal(dispositions, expected_dispositions)),
        "record temporal disposition differs from the frozen classifier",
    )
    ids = records["work_id_numeric"]
    low, high = int(np.min(ids)), int(np.max(ids))
    stats["minimum"] = low if stats["minimum"] is None else min(stats["minimum"], low)
    stats["maximum"] = high if stats["maximum"] is None else max(stats["maximum"], high)
    years = records["publication_year"][~nulls]
    if len(years):
        year_low, year_high = int(np.min(years)), int(np.max(years))
        stats["publication_year_minimum"] = (
            year_low
            if stats["publication_year_minimum"] is None
            else min(stats["publication_year_minimum"], year_low)
        )
        stats["publication_year_maximum"] = (
            year_high
            if stats["publication_year_maximum"] is None
            else max(stats["publication_year_maximum"], year_high)
        )
    counts = np.bincount(dispositions, minlength=len(DISPOSITION_NAMES))
    _require(len(counts) == len(DISPOSITION_NAMES), "record disposition width differs")
    for code, name in DISPOSITION_NAMES.items():
        stats["disposition_counts"][name] += int(counts[code])
    w_zero = ids == 0
    if bool(np.any(w_zero)):
        counts = np.bincount(dispositions[w_zero], minlength=len(DISPOSITION_NAMES))
        for code, name in DISPOSITION_NAMES.items():
            stats["w_zero_disposition_counts"][name] += int(counts[code])
    null_count = int(np.count_nonzero(nulls))
    stats["publication_day_null_rows"] += null_count
    stats["publication_year_null_rows"] += null_count
    _update_framed_digest(stats["digest"], records)
    stats["rows"] += len(records)


def _final_record_stats(stats: dict[str, Any], expected_rows: int) -> dict[str, Any]:
    _require(stats["rows"] == expected_rows, "record row count differs")
    return {
        "rows": stats["rows"],
        "minimum_work_id_numeric": stats["minimum"],
        "maximum_work_id_numeric": stats["maximum"],
        "publication_year_extrema": {
            "minimum": stats["publication_year_minimum"],
            "maximum": stats["publication_year_maximum"],
        },
        "record_sha256": stats["digest"].hexdigest(),
        "disposition_counts": stats["disposition_counts"],
        "w_zero_disposition_counts": stats["w_zero_disposition_counts"],
        "nullability_counts": {
            "publication_day_null_rows": stats["publication_day_null_rows"],
            "publication_year_null_rows": stats["publication_year_null_rows"],
        },
    }


def _validate_staging_records(path: Path, expected_rows: int) -> dict[str, Any]:
    stats = _empty_record_stats(expected_rows)
    descriptor, identity = _open_readonly(path, "staging shard")
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while payload := handle.read(RECORD_BYTES * 65_536):
                _require(
                    len(payload) % RECORD_BYTES == 0, "staging record is truncated"
                )
                records = np.frombuffer(payload, dtype=RECORD_DTYPE)
                _consume_records(stats, records)
        _validate_open_identity(path, descriptor, identity, "staging shard")
    finally:
        os.close(descriptor)
    return _final_record_stats(stats, expected_rows)


def _validate_parquet(
    path: Path, *, partition_id: int, partition_count: int, expected_rows: int
) -> dict[str, Any]:
    stats = _empty_record_stats(expected_rows)
    previous: int | None = None
    descriptor, identity = _open_readonly(path, "partition Parquet")
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            parquet = pq.ParquetFile(handle)
            _require(
                parquet.schema_arrow.equals(EXPECTED_ARROW_SCHEMA, check_metadata=True),
                "partition Arrow schema differs",
            )
            _require(
                (parquet.metadata.metadata or {}) == {},
                "partition file metadata differs",
            )
            physical = [parquet.schema.column(index) for index in range(4)]
            _require(
                physical[0].physical_type == "INT64"
                and "bitWidth=64" in str(physical[0].logical_type)
                and "isSigned=false" in str(physical[0].logical_type)
                and physical[0].max_definition_level == 0,
                "partition work-ID physical schema differs",
            )
            _require(
                physical[1].physical_type == "INT32"
                and str(physical[1].logical_type) == "None"
                and physical[1].max_definition_level == 1,
                "partition publication-day physical schema differs",
            )
            _require(
                physical[2].physical_type == "INT32"
                and "bitWidth=16" in str(physical[2].logical_type)
                and "isSigned=false" in str(physical[2].logical_type)
                and physical[2].max_definition_level == 1,
                "partition publication-year physical schema differs",
            )
            _require(
                physical[3].physical_type == "INT32"
                and "bitWidth=8" in str(physical[3].logical_type)
                and "isSigned=false" in str(physical[3].logical_type)
                and physical[3].max_definition_level == 0,
                "partition disposition physical schema differs",
            )
            _require(
                parquet.metadata.num_rows == expected_rows,
                "partition footer row count differs",
            )
            for group_index in range(parquet.metadata.num_row_groups):
                group = parquet.metadata.row_group(group_index)
                _require(
                    0 < group.num_rows <= 262_144,
                    "partition row-group size differs",
                )
                for column_index in range(group.num_columns):
                    column = group.column(column_index)
                    _require(
                        column.compression == "UNCOMPRESSED",
                        "partition compression differs",
                    )
                    _require(
                        column.is_stats_set is False,
                        "partition statistics unexpectedly exist",
                    )
                    _require(
                        tuple(column.encodings) == ("RLE", "PLAIN")
                        and not column.has_dictionary_page,
                        "partition encoding differs",
                    )
            for batch in parquet.iter_batches(batch_size=262_144, use_threads=False):
                ids = np.asarray(
                    batch.column(0).to_numpy(zero_copy_only=False), dtype=np.uint64
                )
                day_nulls = np.asarray(
                    batch.column(1).is_null().to_numpy(zero_copy_only=False),
                    dtype=np.bool_,
                )
                year_nulls = np.asarray(
                    batch.column(2).is_null().to_numpy(zero_copy_only=False),
                    dtype=np.bool_,
                )
                _require(
                    bool(np.array_equal(day_nulls, year_nulls)),
                    "partition day/year nullability differs",
                )
                days = np.asarray(
                    pc.fill_null(batch.column(1), -(2**31)).to_numpy(
                        zero_copy_only=False
                    ),
                    dtype=np.int32,
                )
                years = np.asarray(
                    pc.fill_null(batch.column(2), 0).to_numpy(zero_copy_only=False),
                    dtype=np.uint16,
                )
                dispositions = np.asarray(
                    batch.column(3).to_numpy(zero_copy_only=False), dtype=np.uint8
                )
                _require(
                    bool(np.array_equal(day_nulls, dispositions == 0)),
                    "partition disposition/nullability differs",
                )
                _require(
                    bool(np.all(ids % partition_count == partition_id)),
                    "partition modulo differs",
                )
                if len(ids):
                    _require(
                        previous is None or int(ids[0]) > previous,
                        "partition IDs are not globally increasing",
                    )
                    _require(
                        bool(np.all(ids[1:] > ids[:-1])),
                        "partition IDs are not strictly increasing",
                    )
                    previous = int(ids[-1])
                if bool(np.any(~day_nulls)):
                    date_values = pa.array(days, mask=day_nulls, type=pa.date32())
                    derived_years = np.asarray(
                        pc.fill_null(pc.year(date_values), 0).to_numpy(
                            zero_copy_only=False
                        ),
                        dtype=np.int64,
                    )
                    _require(
                        bool(np.all(derived_years[~day_nulls] == years[~day_nulls])),
                        "partition year/day relation differs",
                    )
                records = np.empty(batch.num_rows, dtype=RECORD_DTYPE)
                records["work_id_numeric"] = ids
                records["publication_day"] = days
                records["publication_year"] = years
                records["temporal_disposition"] = dispositions
                records["reserved"] = 0
                _consume_records(stats, records)
        _validate_open_identity(path, descriptor, identity, "partition Parquet")
    except VerificationError:
        raise
    except (OSError, pa.ArrowException, ValueError, OverflowError) as exc:
        _fail("cannot decode partition Parquet", exc)
    finally:
        os.close(descriptor)
    return _final_record_stats(stats, expected_rows)


def _checkpoint_self_hash(checkpoint: Mapping[str, Any], field: str) -> None:
    core = dict(checkpoint)
    observed = core.pop("checkpoint_sha256", None)
    _require_sha256(observed, f"{field} checkpoint hash")
    _require(
        observed == _sha256_bytes(_canonical_json_bytes(core)),
        f"{field} checkpoint self-hash differs",
    )


def _validate_scan_checkpoints(
    root: Path,
    plan: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    artifact_observations: Mapping[str, tuple[str, int]],
) -> list[dict[str, Any]]:
    scope = str(plan["scope"])
    checkpoints: list[dict[str, Any]] = []
    for task in tasks:
        token = _sha256_bytes(str(task["relative_path"]).encode("utf-8"))
        checkpoint, _ = _read_json(
            root / f"checkpoints/scan/main/{token}.json", "scan checkpoint"
        )
        _require(
            set(checkpoint) == SCAN_CHECKPOINT_KEYS,
            "scan checkpoint key set differs from the v1 ABI",
        )
        _checkpoint_self_hash(checkpoint, "scan")
        expected = {
            "schema_version": SCHEMAS[scope]["scan_checkpoint"],
            "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
            "file_index": task["file_index"],
            "relative_path": task["relative_path"],
            "source_rich_identity": task["source_rich_identity"],
            "footer_rows": task["footer_rows"],
            "temporal_parent_task_sha256": task["temporal_parent_task_sha256"],
            "temporal_parent_checkpoint_sha256": task[
                "temporal_parent_checkpoint_sha256"
            ],
            "projected_schema_sha256": task["projected_schema_sha256"],
            "operator_paths_redacted": True,
        }
        for key, value in expected.items():
            _require(
                checkpoint.get(key) == value, f"scan checkpoint binding differs: {key}"
            )
        metrics = _require_dict(checkpoint.get("disposition_metrics"), "scan metrics")
        _require(set(metrics) == SPINE_METRIC_KEYS, "scan metric key set differs")
        _require(
            all(type(value) is int and value >= 0 for value in metrics.values()),
            "scan metric is not a nonnegative integer",
        )
        _require(
            checkpoint.get("disposition_metrics_sha256")
            == _sha256_bytes(_canonical_json_bytes(metrics)),
            "scan metric hash differs",
        )
        rows = _require_int(checkpoint.get("staging_shard_rows"), "staging rows")
        staging_relative = f"staging/main/{token}.bin"
        _require(
            checkpoint.get("staging_shard_relative_path") == staging_relative,
            "staging path binding differs",
        )
        staging_path = root / staging_relative
        digest, size = artifact_observations[staging_relative]
        _require(
            size == rows * RECORD_BYTES == checkpoint.get("staging_shard_bytes"),
            "staging byte count differs",
        )
        _require(
            digest == checkpoint.get("staging_shard_sha256"), "staging SHA-256 differs"
        )
        decoded = _validate_staging_records(staging_path, rows)
        _require(
            decoded["record_sha256"]
            == checkpoint.get("row_level_disposition_payload_sha256"),
            "staging record commitment differs",
        )
        _require(
            decoded["disposition_counts"] == checkpoint.get("disposition_counts"),
            "staging disposition counts differ",
        )
        _require(
            decoded["publication_year_extrema"]
            == checkpoint.get("publication_year_extrema"),
            "staging publication-year extrema differ",
        )
        counts = decoded["disposition_counts"]
        _require(
            metrics["w_zero_rows"]
            == sum(decoded["w_zero_disposition_counts"].values()),
            "scan W0 accounting differs from staging",
        )
        metric_bindings = {
            "publication_both_null_rows": counts["BOTH_NULL_EXCLUDED"],
            "publication_both_nonnull_rows": rows - counts["BOTH_NULL_EXCLUDED"],
            "publication_concordance_checked_rows": rows - counts["BOTH_NULL_EXCLUDED"],
            "publication_date_year_match_rows": rows - counts["BOTH_NULL_EXCLUDED"],
            "publication_pre_1900_rows": counts["VALID_PRE_1900_EXCLUDED"],
            "publication_1900_before_snapshot_year_rows": counts[
                "USABLE_1900_BEFORE_SNAPSHOT_YEAR"
            ],
            "publication_snapshot_year_on_or_before_snapshot_rows": counts[
                "USABLE_SNAPSHOT_YEAR_ON_OR_BEFORE_SNAPSHOT"
            ],
            "publication_snapshot_year_after_snapshot_rows": counts[
                "VALID_SNAPSHOT_YEAR_AFTER_SNAPSHOT_EXCLUDED"
            ],
            "publication_post_snapshot_year_rows": counts[
                "VALID_POST_SNAPSHOT_YEAR_EXCLUDED"
            ],
            "publication_year_null_rows": counts["BOTH_NULL_EXCLUDED"],
            "publication_year_pre_1900_rows": counts["VALID_PRE_1900_EXCLUDED"],
            "publication_year_snapshot_year_rows": counts[
                "USABLE_SNAPSHOT_YEAR_ON_OR_BEFORE_SNAPSHOT"
            ]
            + counts["VALID_SNAPSHOT_YEAR_AFTER_SNAPSHOT_EXCLUDED"],
            "publication_year_post_snapshot_year_rows": counts[
                "VALID_POST_SNAPSHOT_YEAR_EXCLUDED"
            ],
        }
        for key, value in metric_bindings.items():
            _require(metrics[key] == value, f"scan classifier metric differs: {key}")
        _require(
            metrics["retained_rows"] == rows, "scan retained rows differ from staging"
        )
        _require(
            metrics["raw_rows"] == rows + metrics["quarantined_w_minus_1_rows"],
            "scan raw-row conservation differs",
        )
        _require(
            metrics["grammar_valid_id_rows"] == rows, "scan grammar-valid rows differ"
        )
        for key in (
            "null_id_rows",
            "unexpected_malformed_id_rows",
            "publication_exactly_one_null_rows",
            "publication_date_parse_error_rows",
            "publication_year_invalid_rows",
            "publication_date_year_mismatch_rows",
        ):
            _require(metrics[key] == 0, f"scan failure metric is nonzero: {key}")
        _require(
            metrics["publication_both_null_rows"]
            + metrics["publication_both_nonnull_rows"]
            == rows,
            "scan publication nullability accounting differs",
        )
        _require(
            metrics["publication_year_null_rows"]
            == metrics["publication_both_null_rows"],
            "scan publication-year null accounting differs",
        )
        _require(
            metrics["publication_concordance_checked_rows"]
            == metrics["publication_date_year_match_rows"],
            "scan publication concordance accounting differs",
        )
        _require(
            sum(checkpoint["disposition_counts"].values()) == rows,
            "scan disposition accounting differs",
        )
        checkpoints.append(checkpoint)
    return checkpoints


def _scan_tree(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    commitments = [
        {
            "file_index": item["file_index"],
            "relative_path": item["relative_path"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "projected_payload_sha256": item["projected_payload_sha256"],
            "row_level_disposition_payload_sha256": item[
                "row_level_disposition_payload_sha256"
            ],
        }
        for item in checkpoints
    ]
    return {
        "checkpoint_count": len(commitments),
        "checkpoint_tree_sha256": _tree_digest(commitments),
        "commitments": commitments,
    }


def _expected_staging_manifest(
    plan: Mapping[str, Any], checkpoints: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    shards = [
        {
            "relative_path": item["staging_shard_relative_path"],
            "rows": item["staging_shard_rows"],
            "bytes": item["staging_shard_bytes"],
            "sha256": item["staging_shard_sha256"],
            "record_sha256": item["row_level_disposition_payload_sha256"],
        }
        for item in checkpoints
    ]
    return {
        "schema_version": SCHEMAS[str(plan["scope"])]["staging_manifest"],
        "scope": plan["scope"],
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "scan_checkpoint_tree_sha256": _scan_tree(checkpoints)[
            "checkpoint_tree_sha256"
        ],
        "shard_count": len(shards),
        "rows": sum(int(item["rows"]) for item in shards),
        "bytes": sum(int(item["bytes"]) for item in shards),
        "staging_shard_tree_sha256": _tree_digest(shards),
        "shards": shards,
        "operator_paths_redacted": True,
    }


def _validate_partitions(
    root: Path,
    plan: Mapping[str, Any],
    partition_count: int,
    scan_tree_sha256: str,
    staging_tree_sha256: str,
    *,
    parquet_validation: str,
    artifact_observations: Mapping[str, tuple[str, int]],
) -> list[dict[str, Any]]:
    scope = str(plan["scope"])
    manifest, _ = _read_json(root / "work_time_spine_manifest.json", "spine manifest")
    raw_partitions = _require_list(manifest.get("partitions"), "manifest partitions")
    _require(len(raw_partitions) == partition_count, "manifest partition count differs")
    manifest_rows: dict[int, int] = {}
    for index, raw in enumerate(raw_partitions):
        item = _require_dict(raw, "manifest partition")
        _require(
            item.get("partition_id") == index,
            "manifest partition IDs are not contiguous",
        )
        manifest_rows[index] = _require_int(item.get("rows"), "manifest partition rows")
    checkpoints: list[dict[str, Any]] = []
    resources = _require_dict(plan.get("resource_requirements"), "plan resources")
    per_row_bound = _require_int(
        resources.get("parquet_bytes_per_row_upper_bound"), "Parquet per-row bound"
    )
    fixed_bound = _require_int(
        resources.get("parquet_fixed_bytes_per_partition_upper_bound"),
        "Parquet fixed bound",
    )
    for partition_id in range(partition_count):
        checkpoint, _ = _read_json(
            root / f"checkpoints/partition/part-{partition_id:04d}.json",
            "partition checkpoint",
        )
        _require(
            set(checkpoint) == PARTITION_CHECKPOINT_KEYS,
            "partition checkpoint key set differs from the v1 ABI",
        )
        _checkpoint_self_hash(checkpoint, "partition")
        expected_rows = manifest_rows[partition_id]
        expected = {
            "schema_version": SCHEMAS[scope]["partition_checkpoint"],
            "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
            "runtime_fingerprint": plan["runtime_fingerprint"],
            "scan_checkpoint_tree_sha256": scan_tree_sha256,
            "staging_shard_tree_sha256": staging_tree_sha256,
            "partition_id": partition_id,
            "partition_formula": "work_id_numeric_mod_4096",
            "output_relative_path": f"spine/part-{partition_id:04d}.parquet",
            "output_rows": expected_rows,
            "strictly_increasing_ids": True,
            "operator_paths_redacted": True,
        }
        for key, value in expected.items():
            _require(
                checkpoint.get(key) == value,
                f"partition checkpoint binding differs: {key}",
            )
        output_path = root / str(checkpoint["output_relative_path"])
        digest, size = artifact_observations[str(checkpoint["output_relative_path"])]
        _require(size == checkpoint.get("output_bytes"), "partition byte count differs")
        _require(digest == checkpoint.get("output_sha256"), "partition SHA-256 differs")
        _require(
            size <= expected_rows * per_row_bound + fixed_bound,
            "partition exceeds byte envelope",
        )
        _require(
            checkpoint.get("output_schema") == EXPECTED_SCHEMA_CONTRACT,
            "partition schema contract differs",
        )
        _require(
            checkpoint.get("output_schema_sha256")
            == _sha256_bytes(_canonical_json_bytes(EXPECTED_SCHEMA_CONTRACT)),
            "partition schema hash differs",
        )
        if parquet_validation == "full":
            decoded = _validate_parquet(
                output_path,
                partition_id=partition_id,
                partition_count=partition_count,
                expected_rows=expected_rows,
            )
            decoded_bindings = {
                "minimum_work_id_numeric": decoded["minimum_work_id_numeric"],
                "maximum_work_id_numeric": decoded["maximum_work_id_numeric"],
                "decoded_parquet_record_sha256": decoded["record_sha256"],
                "disposition_counts": decoded["disposition_counts"],
                "w_zero_disposition_counts": decoded["w_zero_disposition_counts"],
                "publication_year_extrema": decoded["publication_year_extrema"],
                "nullability_counts": decoded["nullability_counts"],
            }
            for key, value in decoded_bindings.items():
                _require(
                    checkpoint.get(key) == value,
                    f"decoded partition evidence differs: {key}",
                )
            _require(
                checkpoint.get("pre_parquet_sorted_record_sha256")
                == decoded["record_sha256"],
                "pre/post Parquet record bridge differs",
            )
        checkpoints.append(checkpoint)
    partitions = [
        {
            "partition_id": item["partition_id"],
            "relative_path": item["output_relative_path"],
            "rows": item["output_rows"],
            "bytes": item["output_bytes"],
            "sha256": item["output_sha256"],
            "record_sha256": item["decoded_parquet_record_sha256"],
        }
        for item in checkpoints
    ]
    expected_manifest = {
        "schema_version": SCHEMAS[scope]["spine_manifest"],
        "scope": scope,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "runtime_fingerprint": plan["runtime_fingerprint"],
        "scan_checkpoint_tree_sha256": scan_tree_sha256,
        "staging_shard_tree_sha256": staging_tree_sha256,
        "partition_count": partition_count,
        "rows": sum(int(item["rows"]) for item in partitions),
        "bytes": sum(int(item["bytes"]) for item in partitions),
        "spine_data_tree_sha256": _tree_digest(partitions),
        "partitions": partitions,
        "operator_paths_redacted": True,
    }
    _require(manifest == expected_manifest, "spine manifest does not reconstruct")
    return checkpoints


def _partition_tree(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    commitments = [
        {
            "partition_id": item["partition_id"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "output_sha256": item["output_sha256"],
            "pre_parquet_sorted_record_sha256": item[
                "pre_parquet_sorted_record_sha256"
            ],
            "decoded_parquet_record_sha256": item["decoded_parquet_record_sha256"],
        }
        for item in checkpoints
    ]
    return {
        "checkpoint_count": len(commitments),
        "partition_checkpoint_tree_sha256": _tree_digest(commitments),
        "commitments": commitments,
    }


def _sum_named(checkpoints: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    totals = {name: 0 for name in DISPOSITION_NAMES.values()}
    for checkpoint in checkpoints:
        observed = _require_dict(checkpoint.get(key), key)
        _require(set(observed) == set(totals), f"{key} names differ")
        for name in totals:
            totals[name] += _require_int(observed[name], f"{key}.{name}")
    return totals


def _extrema(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    lows: list[int] = []
    highs: list[int] = []
    for checkpoint in checkpoints:
        value = _require_dict(
            checkpoint.get("publication_year_extrema"), "publication-year extrema"
        )
        _require(
            set(value) == {"minimum", "maximum"}, "publication-year extrema keys differ"
        )
        if value["minimum"] is None:
            _require(
                value["maximum"] is None, "publication-year extrema nullability differs"
            )
        else:
            lows.append(_require_int(value["minimum"], "publication-year minimum"))
            highs.append(_require_int(value["maximum"], "publication-year maximum"))
    return {
        "minimum": min(lows) if lows else None,
        "maximum": max(highs) if highs else None,
    }


def _expected_gates(scope: str) -> dict[str, Any]:
    production = scope == FULL_SCOPE
    fixture = scope == CLOSED_FIXTURE_SCOPE
    return {
        "canonical_work_time_spine_component": "PASS"
        if production
        else "NONPROMOTABLE_VALIDATION_ONLY",
        "current_content_baseline_component": "FIXTURE_ONLY" if fixture else "PASS",
        "main_temporal_validity_component": "FIXTURE_ONLY" if fixture else "PASS",
        "row_level_spine_commitment": "CREATED_CURRENT_SNAPSHOT"
        if production
        else "NONPRODUCTION_ONLY",
        "row_level_temporal_parent_artifact_identity": "NOT_AVAILABLE_TEMPORAL_V3",
        "historical_content_identity": "UNVERIFIED",
        "raw_source_id_validity": "BLOCKED",
        "w_zero_entity_validity": "UNRESOLVED",
        "citation_edge_chronology": "PENDING_NOT_IN_THIS_COMPONENT",
        "openalex_edge_observation_time": "PENDING_UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT",
        "fixed_horizon_outcome_maturation": "PENDING_SEPARATE_G0_POLICY",
        "author_entity_validity": "PENDING",
        "candidate_eligibility": "PENDING",
        "cutoff_aware_t_minus_1_researcher_graph": "PENDING",
        "g_minus_1_data_readiness": "PENDING",
        "g0_measurement_freeze": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }


def _claim_boundary(scope: str) -> str:
    if scope == FULL_SCOPE:
        return (
            "This pass creates a deterministic current-snapshot row-level canonical "
            "work-time spine bound to the exact content and temporal-v3 parents. It "
            "does not establish historical content identity, W0 entity validity, "
            "citation chronology or observation time, author or candidate validity, "
            "G-1, G0, empirical G1, an NABM effect, or any scientific effect."
        )
    if scope == PILOT_SCOPE:
        return (
            "This prefix pilot validates bounded source scanning and staging only. "
            "It is nonpromotable and establishes neither global ID uniqueness nor "
            "full-snapshot accounting, a production component pass, or a scientific "
            "effect."
        )
    return (
        "This closed fixture validates the external-memory engine contract only. "
        "It is Python-API-only, nonproduction, nonpromotable, and establishes no "
        "OpenAlex or scientific claim."
    )


def _result_fingerprint(report: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(report))
    provenance = _require_dict(payload.get("provenance"), "report provenance")
    provenance.pop("result_fingerprint_sha256", None)
    payload["provenance"] = provenance
    return _sha256_bytes(_canonical_json_bytes(payload))


def _render_markdown(report: Mapping[str, Any]) -> str:
    scope = _require_dict(report.get("scope"), "report scope")
    return "\n".join(
        [
            "# OpenAlex Canonical Work-Time Spine Audit",
            "",
            f"- Status: `{report['status']}`",
            f"- Scope: `{scope['execution_scope']}`",
            f"- Source files / retained rows: `{scope['source_files']}` / `{scope['retained_rows']}`",
            f"- Published partitions: `{scope['partition_count']}`",
            f"- Production component pass: `{str(report['production_component_pass']).lower()}`",
            "- Empirical G1 / scientific effect: `NOT_RUN / NOT_ESTABLISHED`",
            "",
            "## Claim boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )


def _validate_report(
    root: Path,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    scans: Sequence[Mapping[str, Any]],
    partitions: Sequence[Mapping[str, Any]],
    staging_manifest: Mapping[str, Any],
) -> str:
    scope = str(plan["scope"])
    production = scope == FULL_SCOPE
    _require(set(report) == REPORT_KEYS, "report key set differs from the v1 ABI")
    _require(
        report.get("schema_version") == SCHEMAS[scope]["report"],
        "report schema differs",
    )
    _require(report.get("status") == STATUSES[scope], "report status differs")
    _require(report.get("parents") == plan["parents"], "report parent binding differs")
    _require(
        report.get("task_fingerprint_sha256") == plan["task_fingerprint_sha256"],
        "report task binding differs",
    )
    _require(
        report.get("source_tree_sha256") == plan["source_tree_sha256"],
        "report source tree differs",
    )
    scan_tree = _scan_tree(scans)
    _require(
        report.get("scan_checkpoint_tree_sha256")
        == scan_tree["checkpoint_tree_sha256"],
        "report scan tree differs",
    )
    _require(
        report.get("staging_shard_tree_sha256")
        == staging_manifest["staging_shard_tree_sha256"],
        "report staging tree differs",
    )
    partition_tree = _partition_tree(partitions)
    expected_partition_tree = (
        None
        if scope == PILOT_SCOPE
        else partition_tree["partition_checkpoint_tree_sha256"]
    )
    _require(
        report.get("partition_checkpoint_tree_sha256") == expected_partition_tree,
        "report partition tree differs",
    )
    if scope == PILOT_SCOPE:
        _require(
            report.get("spine_data_tree_sha256") is None,
            "pilot report has a spine tree",
        )
    else:
        manifest, _ = _read_json(
            root / "work_time_spine_manifest.json", "spine manifest readback"
        )
        _require(
            report.get("spine_data_tree_sha256") == manifest["spine_data_tree_sha256"],
            "report spine tree differs",
        )
    metrics = {key: 0 for key in SPINE_METRIC_KEYS}
    for checkpoint in scans:
        for key, value in _require_dict(
            checkpoint.get("disposition_metrics"), "scan metrics"
        ).items():
            metrics[key] += _require_int(value, f"scan metric {key}")
    scan_dispositions = _sum_named(scans, "disposition_counts")
    dispositions = (
        scan_dispositions
        if scope == PILOT_SCOPE
        else _sum_named(partitions, "disposition_counts")
    )
    w_zero_counts = (
        None
        if scope == PILOT_SCOPE
        else _sum_named(partitions, "w_zero_disposition_counts")
    )
    rows = metrics["retained_rows"]
    minimum = None
    maximum = None
    if scope != PILOT_SCOPE:
        _require(
            sum(
                _require_int(item.get("output_rows"), "partition rows")
                for item in partitions
            )
            == rows,
            "scan/partition retained-row conservation differs",
        )
        _require(
            dispositions == scan_dispositions,
            "scan/partition disposition counts differ",
        )
        _require(
            _extrema(partitions) == _extrema(scans),
            "scan/partition publication-year extrema differ",
        )
        _require(
            sum((w_zero_counts or {}).values()) == metrics["w_zero_rows"],
            "scan/partition W0 accounting differs",
        )
        partition_nulls = {
            key: sum(
                _require_int(
                    _require_dict(
                        item.get("nullability_counts"), "partition nullability"
                    ).get(key),
                    f"partition nullability {key}",
                )
                for item in partitions
            )
            for key in (
                "publication_day_null_rows",
                "publication_year_null_rows",
            )
        }
        _require(
            partition_nulls["publication_day_null_rows"]
            == partition_nulls["publication_year_null_rows"]
            == metrics["publication_both_null_rows"]
            == metrics["publication_year_null_rows"]
            == dispositions["BOTH_NULL_EXCLUDED"],
            "scan/partition nullability conservation differs",
        )
        nonempty = [item for item in partitions if item["output_rows"]]
        if nonempty:
            minimum = min(int(item["minimum_work_id_numeric"]) for item in nonempty)
            maximum = max(int(item["maximum_work_id_numeric"]) for item in nonempty)
    identifier_expected = {
        "raw_rows": metrics["raw_rows"],
        "quarantined_w_minus_1_rows": metrics["quarantined_w_minus_1_rows"],
        "retained_rows": rows,
        "distinct_numeric_ids": rows if scope != PILOT_SCOPE else None,
        "duplicate_numeric_id_groups": 0 if scope != PILOT_SCOPE else None,
        "duplicate_numeric_row_excess": 0 if scope != PILOT_SCOPE else None,
        "minimum_numeric_id": minimum,
        "maximum_numeric_id": maximum,
        "w_zero_rows": metrics["w_zero_rows"],
        "w_zero_disposition_counts": w_zero_counts,
        "w_zero_disposition_category": (
            next(
                (name for name, count in (w_zero_counts or {}).items() if count == 1),
                None,
            )
            if metrics["w_zero_rows"] == 1 and scope != PILOT_SCOPE
            else None
        ),
        "w_zero_disposition_cross_tab_status": "NOT_EVALUATED_IN_PREFIX_PILOT"
        if scope == PILOT_SCOPE
        else "DECODED_FINAL_PARQUET_VERIFIED",
        "w_zero_entity_validity": "UNRESOLVED",
        "global_uniqueness_verified": scope != PILOT_SCOPE,
    }
    if scope == PILOT_SCOPE:
        identifier_expected["global_uniqueness_status"] = (
            "NOT_EVALUATED_IN_PREFIX_PILOT"
        )
    _require(
        report.get("identifier_accounting") == identifier_expected,
        "report identifier accounting differs",
    )
    scan_extrema = _extrema(scans)
    publication_expected = {
        "publication_year_minimum": scan_extrema["minimum"],
        "publication_year_maximum": scan_extrema["maximum"],
        "publication_year_extrema_status": (
            "FULL_FROZEN_ANCHOR_VERIFIED"
            if scope == FULL_SCOPE
            else "OBSERVED_PREFIX_ONLY_NONPROMOTABLE"
            if scope == PILOT_SCOPE
            else "OBSERVED_CLOSED_FIXTURE_ONLY"
        ),
        "w_zero_disposition_counts": w_zero_counts,
        "disposition_counts": dispositions,
        "failure_metrics": {
            key: metrics[key]
            for key in (
                "publication_exactly_one_null_rows",
                "publication_date_parse_error_rows",
                "publication_year_invalid_rows",
                "publication_date_year_mismatch_rows",
            )
        },
    }
    _require(
        report.get("publication_accounting") == publication_expected,
        "report publication accounting differs",
    )
    report_scope = _require_dict(report.get("scope"), "report scope")
    expected_scope = {
        "execution_scope": scope,
        "production_status": STATUSES[scope] if production else PRODUCTION_NOT_RUN,
        "source_files": len(plan["tasks"]),
        "source_bytes": sum(
            _require_int(task["source_rich_identity"].get("bytes"), "source bytes")
            for task in plan["tasks"]
        ),
        "raw_rows": metrics["raw_rows"],
        "retained_rows": rows,
        "partition_count": len(partitions),
        "receipt_artifact_count": plan["expected_receipt_artifact_count"],
    }
    _require(report_scope == expected_scope, "report scope accounting differs")
    _require(
        report.get("gate_status") == _expected_gates(scope),
        "report gate boundary differs",
    )
    _require(
        report.get("claim_boundary") == _claim_boundary(scope),
        "report claim boundary differs",
    )
    _require(
        report.get("runtime_fingerprint") == plan["runtime_fingerprint"],
        "report runtime binding differs",
    )
    _require(
        report.get("production_component_pass") is production,
        "report production pass differs",
    )
    _require(report.get("pass_g1") is False, "report overclaims G1")
    if scope == FULL_SCOPE:
        full_metric_anchors = {
            "raw_rows": FULL_RAW_ROWS,
            "quarantined_w_minus_1_rows": 1,
            "retained_rows": FULL_RETAINED_ROWS,
            "w_zero_rows": 1,
        }
        for key, value in full_metric_anchors.items():
            _require(metrics[key] == value, f"full report metric anchor differs: {key}")
        _require(
            dispositions == FULL_DISPOSITION_COUNTS,
            "full report disposition anchors differ",
        )
        _require(
            scan_extrema == {"minimum": 1000, "maximum": 2050},
            "full report publication-year extrema anchors differ",
        )
        _require(
            identifier_expected["minimum_numeric_id"] == 0
            and identifier_expected["maximum_numeric_id"] == 7_143_276_261,
            "full report numeric-ID extrema anchors differ",
        )
    provenance = _require_dict(report.get("provenance"), "report provenance")
    result = _result_fingerprint(report)
    _require(
        provenance
        == {
            "config_sha256": plan["config_sha256"],
            "tool_sha256": plan["tool_sha256"],
            "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
            "operator_paths_redacted": True,
            "result_fingerprint_sha256": result,
        },
        "report provenance or result fingerprint differs",
    )
    markdown, _ = _read_bytes(root / BASENAMES[scope]["markdown"], "report Markdown")
    _require(
        markdown == _render_markdown(report).encode("utf-8"),
        "report Markdown does not reconstruct",
    )
    return result


def _validate_state(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    result_fingerprint: str,
    partition_count: int,
) -> None:
    scope = str(plan["scope"])
    expected = {
        "schema_version": SCHEMAS[scope]["state"],
        "scope": scope,
        "production_status": PRODUCTION_NOT_RUN,
        "status": "COMPLETE_PENDING_RECEIPT",
        "completed_scan_files": len(plan["tasks"]),
        "total_scan_files": len(plan["tasks"]),
        "completed_partitions": 0 if scope == PILOT_SCOPE else partition_count,
        "total_partitions": 0 if scope == PILOT_SCOPE else partition_count,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "operator_paths_redacted": True,
        "result_fingerprint_sha256": result_fingerprint,
        "completion_receipt_relative_path": BASENAMES[scope]["receipt"],
        "complete_only_if_receipt_exists": True,
    }
    _require(state == expected, "terminal state does not reconstruct")


def _validate_receipt(
    receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    result_fingerprint: str,
    artifact_contracts: Mapping[str, Any],
) -> None:
    scope = str(plan["scope"])
    production = scope == FULL_SCOPE
    expected = {
        "schema_version": SCHEMAS[scope]["outputs"],
        "status": STATUSES[scope],
        "scope": scope,
        "production_status": STATUSES[scope] if production else PRODUCTION_NOT_RUN,
        "production_component_pass": production,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "result_fingerprint_sha256": result_fingerprint,
        "scan_checkpoint_tree_sha256": report["scan_checkpoint_tree_sha256"],
        "staging_shard_tree_sha256": report["staging_shard_tree_sha256"],
        "partition_checkpoint_tree_sha256": report["partition_checkpoint_tree_sha256"],
        "spine_data_tree_sha256": report["spine_data_tree_sha256"],
        "runtime_fingerprint": plan["runtime_fingerprint"],
        "artifacts": dict(artifact_contracts),
        "artifact_count": len(artifact_contracts),
        "completion_contract": COMPLETION_ABI,
        "operator_paths_redacted": True,
    }
    _require(receipt == expected, "terminal receipt does not reconstruct")


def verify_output(
    output_root: Path, *, parquet_validation: str = "full"
) -> dict[str, Any]:
    """Verify a terminal output without opening any path for write."""

    _require(
        parquet_validation in {"full", "hash-only"}, "unknown Parquet validation mode"
    )
    root = Path(os.path.abspath(os.fspath(output_root)))
    files, directories = _inspect_tree(root)
    before = _tree_snapshot(root)
    plan, _ = _read_json(root / "audit_plan.json", "audit plan")
    scope, partition_count, tasks = _validate_plan(plan)
    expected_artifacts = _expected_artifact_paths(scope, partition_count, tasks)
    receipt_path = root / BASENAMES[scope]["receipt"]
    receipt, receipt_sha256 = _read_json(receipt_path, "terminal receipt")
    raw_artifacts = _require_dict(receipt.get("artifacts"), "receipt artifacts")
    _require(set(raw_artifacts) == expected_artifacts, "receipt artifact set differs")
    _require(
        receipt.get("artifact_count") == len(expected_artifacts),
        "receipt artifact count differs",
    )
    _require(
        files == expected_artifacts | {"audit.lock", BASENAMES[scope]["receipt"]},
        "published file set differs",
    )
    _require(directories == FINAL_DIRECTORIES, "published directory set differs")
    lock_sha256, lock_bytes = _hash_file(root / "audit.lock", "audit lock")
    _require(
        lock_bytes == 0 and lock_sha256 == _sha256_bytes(b""),
        "audit lock is not empty",
    )
    artifact_contracts: dict[str, dict[str, Any]] = {}
    artifact_observations: dict[str, tuple[str, int]] = {}
    control_documents: list[Mapping[str, Any]] = [plan, receipt]
    for relative in sorted(expected_artifacts):
        _safe_relative(relative, "receipt artifact path")
        contract = _require_dict(raw_artifacts[relative], "receipt artifact contract")
        _require(
            set(contract) == {"relative_path", "bytes", "sha256"},
            "receipt artifact contract keys differ",
        )
        digest, size = _hash_file(root / relative, f"receipt artifact {relative}")
        expected_contract = {"relative_path": relative, "bytes": size, "sha256": digest}
        _require(contract == expected_contract, f"receipt artifact differs: {relative}")
        artifact_contracts[relative] = expected_contract
        artifact_observations[relative] = (digest, size)
        if relative.endswith(".json"):
            document, _ = _read_json(root / relative, f"control artifact {relative}")
            control_documents.append(document)
    preflight, _ = _read_json(root / "preflight_evidence.json", "preflight evidence")
    _validate_preflight(preflight, plan, root)
    scans = _validate_scan_checkpoints(root, plan, tasks, artifact_observations)
    staging_manifest, _ = _read_json(root / "staging_manifest.json", "staging manifest")
    expected_staging = _expected_staging_manifest(plan, scans)
    _require(
        staging_manifest == expected_staging, "staging manifest does not reconstruct"
    )
    partitions: list[dict[str, Any]] = []
    if scope != PILOT_SCOPE:
        partitions = _validate_partitions(
            root,
            plan,
            partition_count,
            _scan_tree(scans)["checkpoint_tree_sha256"],
            staging_manifest["staging_shard_tree_sha256"],
            parquet_validation=parquet_validation,
            artifact_observations=artifact_observations,
        )
    report, _ = _read_json(root / BASENAMES[scope]["report"], "terminal report")
    result_fingerprint = _validate_report(
        root, plan, report, scans, partitions, staging_manifest
    )
    state, _ = _read_json(root / "run_state.json", "terminal state")
    _validate_state(state, plan, result_fingerprint, partition_count)
    _validate_receipt(
        receipt,
        plan,
        report,
        result_fingerprint,
        artifact_contracts,
    )
    for document in control_documents:
        _assert_redacted(document)
    receipt_readback, receipt_readback_sha = _read_json(
        receipt_path, "terminal receipt readback"
    )
    _require(
        receipt_readback == receipt and receipt_readback_sha == receipt_sha256,
        "terminal receipt changed during verification",
    )
    after = _tree_snapshot(root)
    _require(
        after == before, "output tree identity changed during read-only verification"
    )
    return {
        "schema_version": VERIFIER_SCHEMA,
        "status": VERIFIER_PASS
        if parquet_validation == "full"
        else VERIFIER_HASH_ONLY_PASS,
        "verified_output_scope": scope,
        "verified_output_status": STATUSES[scope],
        "parquet_validation": parquet_validation,
        "artifact_count": len(expected_artifacts),
        "source_file_count": len(tasks),
        "partition_count": partition_count,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "result_fingerprint_sha256": result_fingerprint,
        "receipt_sha256": receipt_sha256,
        "output_tree_unchanged": True,
        "operator_paths_redacted": True,
        "production_component_pass": scope == FULL_SCOPE,
        "pass_g1": False,
        "independence_boundary": INDEPENDENCE_BOUNDARY,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--parquet-validation",
        choices=("full", "hash-only"),
        default="full",
        help="full decodes every partition; hash-only is explicitly weaker",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_output(
            args.output_root, parquet_validation=args.parquet_validation
        )
    except VerificationError as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 2
    print(_published_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
